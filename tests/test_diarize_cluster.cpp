// Verify NME-SC clustering against NeMo on the synthetic embeddings fixture.
//
// Usage:
//   ./test_diarize_cluster tests/diarize/cluster_ref/

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <vector>

#include "diarize_cluster.h"

static std::vector<float> read_f32(const std::string & p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "open %s failed\n", p.c_str()); std::exit(2); }
    size_t bytes = (size_t)f.tellg();
    f.seekg(0);
    std::vector<float> buf(bytes / sizeof(float));
    f.read(reinterpret_cast<char *>(buf.data()), bytes);
    return buf;
}

static std::vector<int32_t> read_i32(const std::string & p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "open %s failed\n", p.c_str()); std::exit(2); }
    size_t bytes = (size_t)f.tellg();
    f.seekg(0);
    std::vector<int32_t> buf(bytes / 4);
    f.read(reinterpret_cast<char *>(buf.data()), bytes);
    return buf;
}

// Permutation-invariant accuracy: best alignment of cpp labels to ref labels.
static double match_accuracy(const std::vector<int> & cpp,
                             const std::vector<int32_t> & ref) {
    if (cpp.size() != ref.size() || cpp.empty()) return 0.0;
    int Kc = 0, Kr = 0;
    for (int x : cpp) Kc = std::max(Kc, x + 1);
    for (int x : ref) Kr = std::max(Kr, x + 1);
    int K = std::max(Kc, Kr);
    // Confusion table, then row-greedy assignment.
    std::vector<std::vector<int>> conf(K, std::vector<int>(K, 0));
    for (size_t i = 0; i < cpp.size(); i++) conf[cpp[i]][ref[i]]++;
    // Try all permutations for K<=3 (good enough for our tests).
    std::vector<int> perm(K);
    std::iota(perm.begin(), perm.end(), 0);
    int best = 0;
    do {
        int s = 0;
        for (int i = 0; i < K; i++) s += conf[i][perm[i]];
        if (s > best) best = s;
    } while (std::next_permutation(perm.begin(), perm.end()));
    return (double)best / (double)cpp.size();
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <cluster_ref_dir>\n", argv[0]);
        return 1;
    }
    std::string dir = argv[1];
    if (!dir.empty() && dir.back() != '/') dir += "/";

    auto embs = read_f32(dir + "embeddings.f32");
    auto ref_aff = read_f32(dir + "affinity.f32");
    auto ref_labels = read_i32(dir + "labels.i32");

    // Recover N, D from sizes (D=192 known; N from total).
    constexpr int D = 192;
    const int N = (int)embs.size() / D;
    fprintf(stdout, "N=%d D=%d  ref_labels=%zu  ref_aff=%zu\n",
            N, D, ref_labels.size(), ref_aff.size());

    // 1. Affinity matrix.
    auto aff = nmesc_cosine_affinity(embs.data(), N, D);
    {
        double max_abs = 0;
        int max_idx = -1;
        for (size_t i = 0; i < aff.size(); i++) {
            double d = std::fabs((double)aff[i] - (double)ref_aff[i]);
            if (d > max_abs) { max_abs = d; max_idx = (int)i; }
        }
        fprintf(stdout, "affinity:  max_abs=%.6f  cells=%zu  worst@%d (cpp=%.4f ref=%.4f)\n",
                max_abs, aff.size(), max_idx, aff[max_idx], ref_aff[max_idx]);
    }

    // 2. Full clustering pipeline.
    nmesc_cfg cfg;
    cfg.max_num_speakers = 8;
    cfg.max_rp_threshold = 0.25f;
    cfg.sparse_search_volume = 30;
    cfg.nme_mat_size = 512;
    cfg.kmeans_random_trials = 1;
    cfg.kmeans_seed = 0;

    auto out = nmesc_cluster(embs.data(), N, D, cfg);
    fprintf(stdout, "result:    est_num_spk=%d  p_hat=%d  labels first 10 = ",
            out.est_num_speakers, out.p_hat);
    for (int i = 0; i < std::min<int>(10, (int)out.labels.size()); i++)
        fprintf(stdout, "%d ", out.labels[i]);
    fprintf(stdout, "\n");

    double acc = match_accuracy(out.labels, ref_labels);
    fprintf(stdout, "accuracy:  %.4f (perm-invariant match against NeMo labels)\n", acc);
    return (acc < 0.95) ? 4 : 0;
}
