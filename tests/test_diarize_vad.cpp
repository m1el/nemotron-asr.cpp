// Verify the C++ MarbleNet encoder against PyTorch fixtures.
//
// Usage:
//   ./test_diarize_vad weights/diarize-v0.1.f32.gguf
//                      tests/diarize/vad_ref/mel.f32
//                      tests/diarize/vad_ref/   <-- dir holding enc_block_*.f32

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include "diarize.h"
#include "diarize_vad.h"

static std::vector<float> read_f32(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "open %s failed\n", path.c_str()); std::exit(2); }
    size_t bytes = (size_t)f.tellg();
    f.seekg(0);
    std::vector<float> buf(bytes / sizeof(float));
    f.read(reinterpret_cast<char *>(buf.data()), bytes);
    return buf;
}

// Convert (n_mels, T) row-major fixture into (n_mels, T) channels-innermost.
// I.e., transpose from `mel[m * T + t]` to `mel[t * n_mels + m]`.
static std::vector<float> transpose_chan_first(const std::vector<float> & in,
                                               int n_mels, int T) {
    std::vector<float> out((size_t)n_mels * (size_t)T);
    for (int m = 0; m < n_mels; m++) {
        for (int t = 0; t < T; t++) {
            out[(size_t)t * n_mels + m] = in[(size_t)m * T + t];
        }
    }
    return out;
}

// Inverse: (n_mels, T) channels-innermost → row-major (n_mels, T).
static std::vector<float> transpose_time_first(const float * in_data,
                                               int C, int T) {
    std::vector<float> out((size_t)C * (size_t)T);
    for (int t = 0; t < T; t++) {
        for (int c = 0; c < C; c++) {
            out[(size_t)c * T + t] = in_data[(size_t)t * C + c];
        }
    }
    return out;
}

static void diff_stats(const float * a, const float * b, size_t n,
                       const char * label, int t_valid_start, int t_valid_end,
                       int C, int T_total) {
    double max_abs = 0, sum_abs = 0;
    int max_idx = -1;
    int n_compared = 0;
    for (int c = 0; c < C; c++) {
        for (int t = t_valid_start; t < t_valid_end; t++) {
            size_t i = (size_t)c * T_total + t;
            if (i >= n) break;
            float d = std::fabs(a[i] - b[i]);
            if (d > max_abs) { max_abs = d; max_idx = (int)i; }
            sum_abs += d;
            n_compared++;
        }
    }
    fprintf(stdout, "%s: max_abs=%.6f mean_abs=%.6f over %d cells",
            label, max_abs, sum_abs / std::max(1, n_compared), n_compared);
    if (max_idx >= 0) {
        int c_i = max_idx / T_total;
        int t_i = max_idx % T_total;
        fprintf(stdout, "  worst at C=%d t=%d (cpp=%.4f ref=%.4f)\n",
                c_i, t_i, a[max_idx], b[max_idx]);
    } else {
        fprintf(stdout, "\n");
    }
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <diarize.gguf> <mel.f32> <ref_dir>\n", argv[0]);
        return 1;
    }
    const char * gguf_path = argv[1];
    const char * mel_path = argv[2];
    std::string ref_dir = argv[3];
    if (!ref_dir.empty() && ref_dir.back() != '/') ref_dir += "/";

    diarize_model m;
    if (!diarize_model_load(gguf_path, m)) return 2;

    vad_weights w;
    if (!vad_weights_resolve(m, m.backend, w)) return 3;

    // Mel from fixture is (1, n_mels=80, T=528) row-major.
    auto mel_ref = read_f32(mel_path);
    const int n_mels = 80;
    const int T = (int)mel_ref.size() / n_mels;
    fprintf(stdout, "mel: n_mels=%d T=%d (%zu values)\n",
            n_mels, T, mel_ref.size());

    auto mel_in = transpose_chan_first(mel_ref, n_mels, T);

    vad_graph g = vad_graph_build(w, T, n_mels);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    if (!vad_graph_compute(g, m.backend, alloc, mel_in.data())) return 4;

    // Read each block output and compare.
    static const int kBlockChannels[6] = {128, 64, 64, 64, 128, 128};
    // NeMo zero-pads beyond t_valid (= 520 for our test clip).
    const int t_valid = 520;
    for (int b = 0; b < 6; b++) {
        const int C = kBlockChannels[b];
        std::vector<float> cpp_chan_first(ggml_nelements(g.block_out[b]));
        ggml_backend_tensor_get(g.block_out[b], cpp_chan_first.data(), 0,
                                cpp_chan_first.size() * sizeof(float));
        // (T, C) channels-innermost → (C, T) row-major to match fixture.
        auto cpp_row_major = transpose_time_first(cpp_chan_first.data(), C, T);

        auto ref = read_f32(ref_dir + "enc_block_" + std::to_string(b) + ".f32");
        if (ref.size() != cpp_row_major.size()) {
            fprintf(stderr, "block %d: size mismatch cpp=%zu ref=%zu\n",
                    b, cpp_row_major.size(), ref.size());
            continue;
        }
        char label[64];
        std::snprintf(label, sizeof(label), "block_%d (C=%d)", b, C);
        diff_stats(cpp_row_major.data(), ref.data(), ref.size(),
                   label, 0, t_valid, C, T);
    }

    ggml_gallocr_free(alloc);
    vad_graph_free(g);
    vad_weights_free(w);
    diarize_model_free(m);
    return 0;
}
