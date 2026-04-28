// End-to-end VAD session test: load gguf, init session, run on tutorial audio,
// extract speech segments, and dump them.
//
// Usage:
//   ./test_diarize_session weights/diarize-v0.1.f32.gguf
//                          tests/diarize/an4_diarize_test.wav

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <vector>

#include "diarize.h"
#include "diarize_vad.h"

static std::vector<float> read_wav_mono16(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "open %s failed\n", path.c_str()); std::exit(2); }
    char hdr[4]; f.read(hdr, 4); f.ignore(4); f.read(hdr, 4);
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
        }
        f.ignore(sz);
    }
    fprintf(stderr, "no data chunk\n");
    std::exit(2);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <diarize.gguf> <audio.wav>\n", argv[0]);
        return 1;
    }
    diarize_model m;
    if (!diarize_model_load(argv[1], m)) return 2;
    vad_weights w;
    if (!vad_weights_resolve(m, m.backend, w)) return 3;

    auto audio = read_wav_mono16(argv[2]);
    fprintf(stdout, "audio: %zu samples (%.2f s)\n",
            audio.size(), audio.size() / (double)kVadSampleRate);

    vad_session * s = vad_session_init(m, w);

    std::vector<float> probs;
    size_t n = vad_session_run_batch(s, audio.data(), audio.size(), probs);
    fprintf(stdout, "VAD probs: %zu  (min=%.4f mean=%.4f max=%.4f)\n",
            n,
            *std::min_element(probs.begin(), probs.end()),
            std::accumulate(probs.begin(), probs.end(), 0.0) / probs.size(),
            *std::max_element(probs.begin(), probs.end()));

    // Default thresholds match NeMo's diarizer asymmetric onset/offset.
    vad_post_cfg cfg;
    cfg.onset            = 0.5f;
    cfg.offset           = 0.3f;
    cfg.min_duration_on  = 0.10f;
    cfg.min_duration_off = 0.20f;
    cfg.pad_onset        = 0.05f;
    cfg.pad_offset       = -0.05f;

    auto segs = vad_extract_segments(probs, cfg);
    fprintf(stdout, "segments: %zu\n", segs.size());
    for (auto & seg : segs) {
        fprintf(stdout, "  %.3f .. %.3f  (%.3f s)\n",
                seg.start_sec, seg.end_sec, seg.end_sec - seg.start_sec);
    }

    vad_session_free(s);
    vad_weights_free(w);
    diarize_model_free(m);
    return 0;
}
