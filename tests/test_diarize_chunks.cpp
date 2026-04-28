// Verify the C++ chunked VAD inference protocol against NeMo's per-chunk output.
//
// For each 0.63 s sliding window with 0.01 s shift over the tutorial audio,
// run preprocessor + encoder (with masking) + decoder (mean over time +
// Linear + softmax) and compare to tests/diarize/vad_ref/chunk_*.f32.
//
// Usage:
//   ./test_diarize_chunks weights/diarize-v0.1.f32.gguf
//                         tests/diarize/an4_diarize_test.wav
//                         tests/diarize/vad_ref/

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include "diarize.h"
#include "diarize_audio.h"
#include "diarize_vad.h"

constexpr int SR             = 16000;
constexpr int WINDOW_SAMPLES = 10080;     // 0.63 s
constexpr int SHIFT_SAMPLES  = 160;       // 0.01 s
constexpr int MEL_VALID      = 63;        // 10080 / 160
constexpr int MEL_PADDED     = 64;        // pad_to=16 → ceil(63/16)*16
constexpr int N_MELS         = 80;
constexpr int ENC_C          = 128;
constexpr int N_CLASSES      = 2;

static std::vector<float> read_f32(const std::string & p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "open %s failed\n", p.c_str()); std::exit(2); }
    size_t bytes = (size_t)f.tellg();
    f.seekg(0);
    std::vector<float> buf(bytes / sizeof(float));
    f.read(reinterpret_cast<char *>(buf.data()), bytes);
    return buf;
}

// Read 16-bit mono PCM WAV (little-endian, 16 kHz). Minimal parser — assumes
// the standard "RIFF...WAVE...fmt ...data..." layout.
static std::vector<float> read_wav_mono16(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "open %s failed\n", path.c_str()); std::exit(2); }
    char hdr[4];
    f.read(hdr, 4); // "RIFF"
    f.ignore(4);    // chunk size
    f.read(hdr, 4); // "WAVE"

    while (f) {
        f.read(hdr, 4);
        uint32_t sz; f.read(reinterpret_cast<char *>(&sz), 4);
        if (!f) break;
        if (std::memcmp(hdr, "data", 4) == 0) {
            std::vector<int16_t> pcm(sz / 2);
            f.read(reinterpret_cast<char *>(pcm.data()), sz);
            std::vector<float> out(pcm.size());
            for (size_t i = 0; i < pcm.size(); i++) out[i] = pcm[i] / 32768.0f;
            return out;
        } else {
            f.ignore(sz);
        }
    }
    fprintf(stderr, "no data chunk in %s\n", path.c_str());
    std::exit(2);
}

// (n_mels, T) row-major  ->  (n_mels, T) channels-innermost.
static void to_chan_first(const float * in, int C, int T, float * out) {
    for (int c = 0; c < C; c++)
        for (int t = 0; t < T; t++)
            out[(size_t)t * C + c] = in[(size_t)c * T + t];
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <diarize.gguf> <audio.wav> <ref_dir>\n", argv[0]);
        return 1;
    }
    std::string ref_dir = argv[3];
    if (!ref_dir.empty() && ref_dir.back() != '/') ref_dir += "/";

    diarize_model m;
    if (!diarize_model_load(argv[1], m)) return 2;
    vad_weights w;
    if (!vad_weights_resolve(m, m.backend, w)) return 3;

    auto audio = read_wav_mono16(argv[2]);
    const int n_chunks = 1 + ((int)audio.size() - WINDOW_SAMPLES) / SHIFT_SAMPLES;
    fprintf(stdout, "audio: %zu samples, %d chunks (window=%d, shift=%d)\n",
            audio.size(), n_chunks, WINDOW_SAMPLES, SHIFT_SAMPLES);

    // Pull the preprocessor weights out of the gguf.
    const ggml_tensor * fb_t  = diarize_model_get_tensor(m, "vad.preprocessor.featurizer.fb");
    const ggml_tensor * win_t = diarize_model_get_tensor(m, "vad.preprocessor.featurizer.window");
    const float * fb  = static_cast<const float *>(fb_t->data);
    const float * win = static_cast<const float *>(win_t->data);

    diarize_audio_cfg cfg; // VAD defaults

    // Pull the decoder weights into host memory once.
    std::vector<float> dec_w(ENC_C * N_CLASSES);
    std::vector<float> dec_b(N_CLASSES);
    ggml_backend_tensor_get(w.dec_w, dec_w.data(), 0, dec_w.size() * sizeof(float));
    ggml_backend_tensor_get(w.dec_b, dec_b.data(), 0, dec_b.size() * sizeof(float));
    // dec_w storage: ne=(C, n_classes); memory dec_w[c + C*k]  -- channel innermost.

    // Build the encoder graph once for T=MEL_PADDED.
    vad_graph g = vad_graph_build(w, MEL_PADDED, N_MELS);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));

    // Reused scratch.
    std::vector<float> mel_pp;        // preprocessor output (n_mels, T_padded) row-major
    std::vector<float> mel_chan(N_MELS * MEL_PADDED);
    std::vector<float> enc_out_buf(ENC_C * MEL_PADDED);

    std::vector<float> all_logits(n_chunks * N_CLASSES);
    std::vector<float> all_probs(n_chunks * N_CLASSES);
    std::vector<float> all_speech(n_chunks);

    for (int i = 0; i < n_chunks; i++) {
        const float * chunk_audio = audio.data() + (size_t)i * SHIFT_SAMPLES;

        size_t t_valid_pp = 0;
        size_t t_padded_pp = diarize_compute_logmel(chunk_audio, WINDOW_SAMPLES, cfg,
                                                    fb, win, mel_pp, &t_valid_pp);
        if ((int)t_padded_pp != MEL_PADDED || (int)t_valid_pp != MEL_VALID) {
            fprintf(stderr, "chunk %d: unexpected mel shape t_valid=%zu t_padded=%zu\n",
                    i, t_valid_pp, t_padded_pp);
            return 4;
        }

        to_chan_first(mel_pp.data(), N_MELS, MEL_PADDED, mel_chan.data());
        if (!vad_graph_compute(g, m.backend, alloc, mel_chan.data(), MEL_VALID)) return 5;

        // Read encoder output: (T, C) channels-innermost in our layout.
        ggml_backend_tensor_get(g.encoder_out, enc_out_buf.data(), 0,
                                enc_out_buf.size() * sizeof(float));

        // Mean over T (NeMo's AdaptiveAvgPool1d(1) — averages all 64 frames,
        // including the masked-zero frame 63).
        float mean_ch[ENC_C] = {0};
        for (int t = 0; t < MEL_PADDED; t++) {
            const float * row = enc_out_buf.data() + (size_t)t * ENC_C;
            for (int c = 0; c < ENC_C; c++) mean_ch[c] += row[c];
        }
        const float inv_T = 1.0f / (float)MEL_PADDED;
        for (int c = 0; c < ENC_C; c++) mean_ch[c] *= inv_T;

        // Linear: logits[k] = sum_c dec_w[c, k] * mean[c] + dec_b[k]
        float logits[N_CLASSES];
        for (int k = 0; k < N_CLASSES; k++) {
            float s = dec_b[k];
            // dec_w row-major: dec_w[k * ENC_C + c]? Let's check layout.
            // ne=(C, n_classes), C innermost, so memory is dec_w[k*C + c].
            const float * row = dec_w.data() + (size_t)k * ENC_C;
            for (int c = 0; c < ENC_C; c++) s += row[c] * mean_ch[c];
            logits[k] = s;
        }

        // Softmax.
        float mx = std::max(logits[0], logits[1]);
        float e0 = std::exp(logits[0] - mx);
        float e1 = std::exp(logits[1] - mx);
        float Z  = e0 + e1;
        float p0 = e0 / Z, p1 = e1 / Z;

        all_logits[2*i + 0] = logits[0];
        all_logits[2*i + 1] = logits[1];
        all_probs[2*i + 0]  = p0;
        all_probs[2*i + 1]  = p1;
        all_speech[i]       = p1;
    }

    // Dump for inspection.
    std::ofstream("/tmp/cpp_chunk_logits.f32", std::ios::binary)
        .write(reinterpret_cast<char *>(all_logits.data()),
               all_logits.size() * sizeof(float));
    std::ofstream("/tmp/cpp_chunk_probs.f32", std::ios::binary)
        .write(reinterpret_cast<char *>(all_probs.data()),
               all_probs.size() * sizeof(float));
    std::ofstream("/tmp/cpp_chunk_speech.f32", std::ios::binary)
        .write(reinterpret_cast<char *>(all_speech.data()),
               all_speech.size() * sizeof(float));

    // Diff against the NeMo fixtures.
    auto ref_logits = read_f32(ref_dir + "chunk_logits.f32");
    auto ref_probs  = read_f32(ref_dir + "chunk_probs.f32");
    auto ref_speech = read_f32(ref_dir + "chunk_speech.f32");

    auto stats = [&](const char * label, const std::vector<float> & a, const std::vector<float> & b) {
        if (a.size() != b.size()) {
            fprintf(stderr, "%s: size mismatch cpp=%zu ref=%zu\n", label, a.size(), b.size());
            return;
        }
        double max_abs = 0, sum_abs = 0;
        int max_idx = -1;
        for (size_t i = 0; i < a.size(); i++) {
            double d = std::fabs((double)a[i] - (double)b[i]);
            if (d > max_abs) { max_abs = d; max_idx = (int)i; }
            sum_abs += d;
        }
        fprintf(stdout, "%-12s max_abs=%.6f mean_abs=%.6f n=%zu",
                label, max_abs, sum_abs / a.size(), a.size());
        if (max_idx >= 0) {
            fprintf(stdout, "  worst[%d] cpp=%.4f ref=%.4f", max_idx, a[max_idx], b[max_idx]);
        }
        fprintf(stdout, "\n");
    };

    stats("logits", all_logits, ref_logits);
    stats("probs",  all_probs,  ref_probs);
    stats("speech", all_speech, ref_speech);

    ggml_gallocr_free(alloc);
    vad_graph_free(g);
    vad_weights_free(w);
    diarize_model_free(m);
    return 0;
}
