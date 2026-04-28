// Verify the C++ TitaNet-L speaker encoder + decoder against PyTorch fixtures.
//
// Usage:
//   ./test_diarize_spk weights/diarize-v0.1.f32.gguf
//                      tests/diarize/spk_ref/

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
#include "diarize_spk.h"

static std::vector<float> read_f32(const std::string & p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "open %s failed\n", p.c_str()); std::exit(2); }
    size_t bytes = (size_t)f.tellg();
    f.seekg(0);
    std::vector<float> buf(bytes / sizeof(float));
    f.read(reinterpret_cast<char *>(buf.data()), bytes);
    return buf;
}

// (T, C) channels-innermost  -> (C, T) row-major.
static std::vector<float> to_row_major(const float * in, int C, int T) {
    std::vector<float> out((size_t)C * T);
    for (int t = 0; t < T; t++)
        for (int c = 0; c < C; c++)
            out[(size_t)c * T + t] = in[(size_t)t * C + c];
    return out;
}

// (C, T) row-major  ->  (T, C) channels-innermost.
static std::vector<float> to_chan_first(const float * in, int C, int T) {
    std::vector<float> out((size_t)C * T);
    for (int c = 0; c < C; c++)
        for (int t = 0; t < T; t++)
            out[(size_t)t * C + c] = in[(size_t)c * T + t];
    return out;
}

static void diff_stats(const char * label,
                       const std::vector<float> & cpp,
                       const std::vector<float> & ref,
                       int t_max_compare, int C, int T_total) {
    if (cpp.size() != ref.size()) {
        fprintf(stdout, "%-16s SIZE MISMATCH cpp=%zu ref=%zu\n",
                label, cpp.size(), ref.size());
        return;
    }
    double max_abs = 0, sum_abs = 0;
    int max_idx = -1, n = 0;
    for (int c = 0; c < C; c++) {
        for (int t = 0; t < t_max_compare; t++) {
            size_t i = (size_t)c * T_total + t;
            float d = std::fabs(cpp[i] - ref[i]);
            if (d > max_abs) { max_abs = d; max_idx = (int)i; }
            sum_abs += d;
            n++;
        }
    }
    fprintf(stdout, "%-16s max_abs=%.6f mean_abs=%.6f cells=%d",
            label, max_abs, sum_abs / std::max(1, n), n);
    if (max_idx >= 0) {
        fprintf(stdout, "  worst@%d cpp=%.4f ref=%.4f", max_idx, cpp[max_idx], ref[max_idx]);
    }
    fprintf(stdout, "\n");
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <diarize.gguf> <spk_ref_dir>\n", argv[0]);
        return 1;
    }
    std::string ref_dir = argv[2];
    if (!ref_dir.empty() && ref_dir.back() != '/') ref_dir += "/";

    diarize_model m;
    if (!diarize_model_load(argv[1], m)) return 2;
    spk_weights w;
    if (!spk_weights_resolve(m, m.backend, w)) return 3;

    fprintf(stdout, "spk weights resolved: emb_dim=%d enc_C=%d attn_C=%d\n",
            w.emb_dim, w.enc_out_channels, w.attn_channels);

    auto mel_ref = read_f32(ref_dir + "mel.f32");
    constexpr int n_mels = 80;
    constexpr int T = 160;
    if ((int)mel_ref.size() != n_mels * T) {
        fprintf(stderr, "mel size %zu != %d*%d\n", mel_ref.size(), n_mels, T);
        return 4;
    }
    auto mel_in = to_chan_first(mel_ref.data(), n_mels, T);

    spk_graph g = spk_graph_build(w, T, n_mels);
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));

    const int lens = 150;
    if (!spk_graph_compute(g, m.backend, alloc, mel_in.data(), lens)) return 5;

    // Compare per-block outputs.
    static const int kBlockChannels[5] = {1024, 1024, 1024, 1024, 3072};
    for (int b = 0; b < 5; b++) {
        const int C = kBlockChannels[b];
        std::vector<float> cpp_chan(ggml_nelements(g.block_out[b]));
        ggml_backend_tensor_get(g.block_out[b], cpp_chan.data(), 0,
                                cpp_chan.size() * sizeof(float));
        auto cpp_rm = to_row_major(cpp_chan.data(), C, T);
        auto ref = read_f32(ref_dir + "enc_block_" + std::to_string(b) + ".f32");
        char label[32];
        std::snprintf(label, sizeof(label), "block_%d", b);
        diff_stats(label, cpp_rm, ref, T, C, T);
    }

    // Pool out: (1, 6144, 1) in PyTorch, our shape is (2C, 1) ne=(2C, 1).
    {
        std::vector<float> cpp_pool(ggml_nelements(g.pool_out));
        ggml_backend_tensor_get(g.pool_out, cpp_pool.data(), 0,
                                cpp_pool.size() * sizeof(float));
        auto ref = read_f32(ref_dir + "pool_out.f32");
        diff_stats("pool",
                   cpp_pool, ref,
                   1, (int)g.pool_out->ne[0], (int)g.pool_out->ne[1]);
    }

    // Embedding: (192,) — should match (1, 192) from PyTorch.
    {
        std::vector<float> cpp_emb(kSpkEmbDim);
        ggml_backend_tensor_get(g.embedding, cpp_emb.data(), 0,
                                cpp_emb.size() * sizeof(float));
        auto ref = read_f32(ref_dir + "embedding.f32");
        // Direct elementwise compare for the embedding.
        double max_abs = 0, sum_abs = 0;
        int max_idx = -1;
        for (int i = 0; i < kSpkEmbDim; i++) {
            float d = std::fabs(cpp_emb[i] - ref[i]);
            if (d > max_abs) { max_abs = d; max_idx = i; }
            sum_abs += d;
        }
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < kSpkEmbDim; i++) {
            dot += cpp_emb[i] * ref[i];
            na  += cpp_emb[i] * cpp_emb[i];
            nb  += ref[i]     * ref[i];
        }
        double cos_sim = dot / (std::sqrt(na) * std::sqrt(nb));
        fprintf(stdout, "embedding        max_abs=%.6f mean_abs=%.6f  cos=%.6f  norm cpp=%.4f ref=%.4f\n",
                max_abs, sum_abs / kSpkEmbDim, cos_sim, std::sqrt(na), std::sqrt(nb));
        if (max_idx >= 0) {
            fprintf(stdout, "  worst@%d cpp=%.4f ref=%.4f\n", max_idx, cpp_emb[max_idx], ref[max_idx]);
        }
    }

    ggml_gallocr_free(alloc);
    spk_graph_free(g);
    spk_weights_free(w);
    diarize_model_free(m);
    return 0;
}
