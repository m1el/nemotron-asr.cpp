// Verify the C++ log-mel preprocessor against the PyTorch fixture.
//
// Usage:
//   ./test_diarize_preproc weights/diarize-v0.1.f32.gguf
//                          tests/diarize/vad_ref/input_audio.f32
//                          tests/diarize/vad_ref/mel.f32
//
// Reports max abs error and mean abs error against the fixture.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "diarize.h"
#include "diarize_audio.h"

static std::vector<float> read_f32(const char * path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "open %s failed\n", path); std::exit(2); }
    size_t bytes = (size_t)f.tellg();
    f.seekg(0);
    std::vector<float> buf(bytes / sizeof(float));
    f.read(reinterpret_cast<char *>(buf.data()), bytes);
    return buf;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <diarize.gguf> <audio.f32> <mel_ref.f32>\n", argv[0]);
        return 1;
    }

    diarize_model m;
    if (!diarize_model_load(argv[1], m)) return 2;

    const ggml_tensor * fb_t  = diarize_model_get_tensor(m, "vad.preprocessor.featurizer.fb");
    const ggml_tensor * win_t = diarize_model_get_tensor(m, "vad.preprocessor.featurizer.window");
    if (!fb_t || !win_t) {
        fprintf(stderr, "missing preprocessor tensors\n");
        return 3;
    }

    const float * fb  = static_cast<const float *>(fb_t->data);
    const float * win = static_cast<const float *>(win_t->data);

    diarize_audio_cfg cfg; // defaults match VAD: 80-mel, no normalize, pad_to=16

    auto audio = read_f32(argv[2]);
    auto mel_ref = read_f32(argv[3]);

    std::vector<float> mel_cpp;
    size_t t_valid = 0;
    size_t t_padded = diarize_compute_logmel(audio.data(), audio.size(), cfg,
                                             fb, win, mel_cpp, &t_valid);

    fprintf(stdout, "audio:    %zu samples\n", audio.size());
    fprintf(stdout, "mel cpp:  %zu valid + %zu padding = %zu frames, %d mels\n",
            t_valid, t_padded - t_valid, t_padded, cfg.n_mels);
    fprintf(stdout, "mel ref:  %zu values  (expected %d × %zu = %zu)\n",
            mel_ref.size(), cfg.n_mels, t_padded, (size_t)cfg.n_mels * t_padded);

    if (mel_ref.size() != mel_cpp.size()) {
        fprintf(stderr, "shape mismatch: cpp=%zu ref=%zu\n",
                mel_cpp.size(), mel_ref.size());
        return 4;
    }

    // Diff over the valid window only (NeMo zero-pads the remaining frames; we do too,
    // so they should also agree, but we focus the metric on real data).
    double max_abs = 0, sum_abs = 0;
    int max_idx = -1;
    int n_compared = 0;
    for (int m_idx = 0; m_idx < cfg.n_mels; m_idx++) {
        for (size_t t = 0; t < t_valid; t++) {
            size_t i = (size_t)m_idx * t_padded + t;
            float a = mel_cpp[i];
            float b = mel_ref[i];
            float d = std::fabs(a - b);
            if (d > max_abs) { max_abs = d; max_idx = (int)i; }
            sum_abs += d;
            n_compared++;
        }
    }
    fprintf(stdout, "diff (valid frames):  max_abs=%.6f mean_abs=%.6f  n=%d\n",
            max_abs, sum_abs / n_compared, n_compared);

    // Print sample values around the max-error location.
    if (max_idx >= 0) {
        int m_i = max_idx / (int)t_padded;
        int t_i = max_idx % (int)t_padded;
        fprintf(stdout, "  worst at mel=%d t=%d:  cpp=% .6f  ref=% .6f\n",
                m_i, t_i, mel_cpp[max_idx], mel_ref[max_idx]);
        // dump a 5-frame slice
        fprintf(stdout, "  cpp[mel=%d, t=%d..%d] = ", m_i, std::max(0, t_i-2), t_i+2);
        for (int t = std::max(0, t_i-2); t <= t_i+2; t++) {
            fprintf(stdout, "% .4f ", mel_cpp[(size_t)m_i * t_padded + t]);
        }
        fprintf(stdout, "\n  ref[mel=%d, t=%d..%d] = ", m_i, std::max(0, t_i-2), t_i+2);
        for (int t = std::max(0, t_i-2); t <= t_i+2; t++) {
            fprintf(stdout, "% .4f ", mel_ref[(size_t)m_i * t_padded + t]);
        }
        fprintf(stdout, "\n");
    }

    diarize_model_free(m);
    return (max_abs < 1e-3) ? 0 : 5;
}
