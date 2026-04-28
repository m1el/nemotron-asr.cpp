// End-to-end test of the diarization pipeline on the tutorial audio.
//
// Usage:
//   ./test_diarize_pipeline weights/diarize-v0.1.f32.gguf
//                           tests/diarize/an4_diarize_test.wav

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "diarize_pipeline.h"

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
    auto cfg = diarize_pipeline_default_cfg();
    cfg.diarize_gguf_path = argv[1];
    // For the 5.2 s tutorial clip with single-scale 1.5/0.75 we get only ~5
    // sub-segments. Tighten the shift so NMESC has more samples to look at.
    cfg.sub_shift_sec  = 0.25f;
    cfg.cluster.min_samples_for_nmesc = 4;
    cfg.cluster.oracle_num_speakers   = 2;  // tutorial has 2 speakers

    auto * p = diarize_pipeline_init(cfg);
    if (!p) return 2;

    auto audio = read_wav_mono16(argv[2]);
    fprintf(stdout, "audio: %zu samples (%.2f s)\n",
            audio.size(), audio.size() / 16000.0);

    // Drip the audio in 0.5 s chunks to exercise the streaming path.
    constexpr size_t chunk = 8000;
    for (size_t i = 0; i < audio.size(); i += chunk) {
        size_t n = std::min(chunk, audio.size() - i);
        diarize_pipeline_push_audio(p, audio.data() + i, n);
    }

    fprintf(stdout, "after push: subs=%zu, segments=%zu\n",
            diarize_pipeline_n_embeddings(p),
            diarize_pipeline_n_segments(p));

    // Synthesize some words at made-up times so we can exercise the
    // word-to-speaker alignment.
    diarize_pipeline_push_text(p, "eleven twenty seven fifty seven", 1.5);
    diarize_pipeline_push_text(p, "october twenty four nineteen seventy", 4.0);

    std::string out = diarize_pipeline_finalize(p);
    fprintf(stdout, "subs=%zu words=%zu\n",
            diarize_pipeline_n_embeddings(p),
            diarize_pipeline_n_words(p));
    fprintf(stdout, "speaker-tagged transcript:\n%s\n", out.c_str());

    diarize_pipeline_free(p);
    return 0;
}
