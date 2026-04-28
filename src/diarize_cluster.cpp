#include "diarize_cluster.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {

constexpr float kAffEps = 3.5e-4f;
constexpr double kEig    = 1e-10;
constexpr int    kMinPVal = 2;

// Min-max scale matrix in place to [0, 1].
void min_max_scale(std::vector<float> & X) {
    if (X.empty()) return;
    float vmin = X[0], vmax = X[0];
    for (float v : X) { if (v < vmin) vmin = v; if (v > vmax) vmax = v; }
    const float range = vmax - vmin;
    if (range <= 0) return;
    const float inv = 1.0f / range;
    for (auto & v : X) v = (v - vmin) * inv;
}

} // namespace

std::vector<float> nmesc_cosine_affinity(const float * embeddings, size_t N, size_t D) {
    if (N <= 1) return std::vector<float>{1.0f};

    // L2 normalize each row with eps in the denominator (NeMo cos_similarity).
    std::vector<float> normed((size_t)N * D);
    for (size_t i = 0; i < N; i++) {
        double s = 0.0;
        for (size_t d = 0; d < D; d++) s += (double)embeddings[i*D+d] * embeddings[i*D+d];
        const float inv = 1.0f / (std::sqrt((float)s) + kAffEps);
        for (size_t d = 0; d < D; d++) normed[i*D+d] = embeddings[i*D+d] * inv;
    }

    // Cosine sim = normed @ normed^T.
    std::vector<float> aff((size_t)N * N, 0.0f);
    for (size_t i = 0; i < N; i++) {
        for (size_t j = i; j < N; j++) {
            double s = 0.0;
            for (size_t d = 0; d < D; d++) s += (double)normed[i*D+d] * normed[j*D+d];
            aff[i*N+j] = (float)s;
            aff[j*N+i] = (float)s;
        }
        aff[i*N+i] = 1.0f; // NeMo fills diagonal to 1 explicitly.
    }
    min_max_scale(aff);
    return aff;
}

// Top-p kNN binarize each row, then symmetrize (X + X^T)/2.
// Diagonal of the input is left as-is (cosine_affinity sets it to 1).
static std::vector<float> affinity_graph_mat(const std::vector<float> & X, int N, int p) {
    std::vector<float> out((size_t)N * N, 0.0f);
    if (p <= 0) return X; // edge case: no binarization.
    std::vector<int> idx((size_t)N);
    for (int i = 0; i < N; i++) {
        std::iota(idx.begin(), idx.end(), 0);
        const float * row = X.data() + (size_t)i * N;
        // Partial sort: pick top-p indices by descending value (stable for ties).
        std::partial_sort(idx.begin(), idx.begin() + std::min(p, N), idx.end(),
            [&](int a, int b) {
                if (row[a] != row[b]) return row[a] > row[b];
                return a < b;
            });
        const int k = std::min(p, N);
        for (int j = 0; j < k; j++) out[(size_t)i * N + idx[j]] = 1.0f;
    }
    // Symmetrize: 0.5 * (out + out^T).
    std::vector<float> sym((size_t)N * N, 0.0f);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            sym[(size_t)i * N + j] = 0.5f * (out[(size_t)i * N + j] + out[(size_t)j * N + i]);
        }
    }
    return sym;
}

// Build the unnormalized Laplacian L = D - A where D = diag(row-sum(|A|)).
// Diagonal of A is forced to 0 first (matches NeMo's getLaplacian).
static MatrixXd laplacian_from_affinity(const float * A, int N) {
    MatrixXd Md(N, N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) Md(i, j) = (i == j) ? 0.0 : (double)A[(size_t)i * N + j];
    }
    VectorXd d = Md.cwiseAbs().rowwise().sum();
    MatrixXd L = -Md;
    for (int i = 0; i < N; i++) L(i, i) = d(i);
    return L;
}

// Eigen-decompose the Laplacian. Returns sorted-ascending eigenvalues.
static std::pair<VectorXd, MatrixXd> eig_lap(const MatrixXd & L) {
    Eigen::SelfAdjointEigenSolver<MatrixXd> es(L);
    return {es.eigenvalues(), es.eigenvectors()};
}

// Lambda gap list = lambdas[1:] - lambdas[:-1].
static VectorXd lambda_gaps(const VectorXd & lambdas) {
    if (lambdas.size() <= 1) return VectorXd::Zero(0);
    VectorXd g(lambdas.size() - 1);
    for (int i = 0; i < g.size(); i++) g(i) = lambdas(i + 1) - lambdas(i);
    return g;
}

// Connectivity check via BFS over a binarized affinity matrix (edges where
// X[i, j] > 0).
static bool fully_connected(const std::vector<float> & X, int N) {
    if (N == 0) return true;
    std::vector<int> seen(N, 0);
    std::vector<int> q;
    q.reserve(N);
    q.push_back(0);
    seen[0] = 1;
    int reached = 1;
    while (!q.empty()) {
        int v = q.back(); q.pop_back();
        const float * row = X.data() + (size_t)v * N;
        for (int j = 0; j < N; j++) {
            if (!seen[j] && row[j] > 0.0f) {
                seen[j] = 1;
                reached++;
                q.push_back(j);
            }
        }
    }
    return reached == N;
}

// Subsample affinity matrix by stride. Returns (subsample_ratio, sub_mat).
static std::pair<int, std::vector<float>> subsample_affinity(
    const std::vector<float> & A, int N, int target_size)
{
    int ratio = std::max(1, (int)std::ceil((double)N / (double)target_size));
    if (ratio == 1) return {1, A};
    std::vector<int> keep;
    for (int i = 0; i < N; i += ratio) keep.push_back(i);
    int M = (int)keep.size();
    std::vector<float> S((size_t)M * M, 0.0f);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            S[(size_t)i * M + j] = A[(size_t)keep[i] * N + keep[j]];
        }
    }
    return {ratio, S};
}

// Generate the candidate p-value list via NeMo's sparse search.
//   max_N = max(2, floor(N * max_rp_threshold))
//   if sparse: linspace(1, max_N, min(max_N, sparse_search_volume))  (int-cast)
//   else:      [1, ..., max_N]
static std::vector<int> p_value_list(int N, float max_rp_threshold,
                                     int sparse_search_volume,
                                     bool sparse_search,
                                     int & max_N_out) {
    int max_N = std::max(kMinPVal, (int)std::floor((double)N * (double)max_rp_threshold));
    max_N_out = max_N;
    std::vector<int> ps;
    if (sparse_search) {
        int steps = std::min(max_N, std::max(2, sparse_search_volume));
        // torch.linspace(start=1, end=max_N, steps=steps).int()
        for (int i = 0; i < steps; i++) {
            // linspace_i = 1 + i*(max_N-1)/(steps-1)
            double v = 1.0 + (double)i * ((double)max_N - 1.0) / std::max(1, steps - 1);
            ps.push_back((int)v); // PyTorch's int cast truncates toward zero
        }
    } else {
        for (int i = 1; i <= max_N; i++) ps.push_back(i);
    }
    // De-dup while preserving order (linspace can repeat after int-cast).
    std::set<int> seen;
    std::vector<int> uniq;
    for (int p : ps) if (seen.insert(p).second) uniq.push_back(p);
    return uniq;
}

// For one p, compute (g_p, est_num_spk).
struct eig_ratio_out {
    double g_p;
    int    est_num_spk;
    VectorXd lambdas;     // eigenvalues sorted ascending
    VectorXd gaps;        // gaps
};

static eig_ratio_out get_eig_ratio(const std::vector<float> & A, int N, int p,
                                   int max_num_speakers) {
    auto Ap = affinity_graph_mat(A, N, p);
    MatrixXd L = laplacian_from_affinity(Ap.data(), N);
    Eigen::SelfAdjointEigenSolver<MatrixXd> es(L, Eigen::EigenvaluesOnly);
    VectorXd lam = es.eigenvalues();
    VectorXd g = lambda_gaps(lam);
    int K = std::min((int)g.size(), max_num_speakers);
    int kbest = 0;
    for (int i = 1; i < K; i++) if (g(i) > g(kbest)) kbest = i;
    int est_num_spk = kbest + 1;
    double max_gap = (K > 0) ? g(kbest) / (lam.maxCoeff() + kEig) : 0.0;
    double g_p = ((double)p / (double)N) / (max_gap + kEig);
    return {g_p, est_num_spk, lam, g};
}

// Lloyd k-means on N×D row-major data with kmeans++ init.
struct kmeans_out {
    std::vector<int> labels;
    double inertia;
};

static kmeans_out kmeans_pp(const std::vector<float> & X, int N, int D,
                            int K, std::mt19937_64 & rng,
                            int max_iter = 300, double tol = 1e-4) {
    if (K <= 1) return {std::vector<int>(N, 0), 0.0};

    auto sq_dist = [&](const float * a, const float * b) -> double {
        double s = 0.0;
        for (int d = 0; d < D; d++) { double t = (double)a[d] - b[d]; s += t * t; }
        return s;
    };

    // ----- kmeans++ init -----
    std::vector<int> ctr_idx;
    ctr_idx.reserve(K);
    std::uniform_int_distribution<int> uid(0, N - 1);
    ctr_idx.push_back(uid(rng));
    std::vector<double> dist2(N, std::numeric_limits<double>::infinity());
    while ((int)ctr_idx.size() < K) {
        const float * last = X.data() + (size_t)ctr_idx.back() * D;
        for (int i = 0; i < N; i++) {
            double d = sq_dist(X.data() + (size_t)i * D, last);
            if (d < dist2[i]) dist2[i] = d;
        }
        double total = 0.0;
        for (double d : dist2) total += d;
        if (total <= 0.0) {
            // All points coincide; fill with random distinct indices.
            int next = uid(rng);
            ctr_idx.push_back(next);
            continue;
        }
        std::uniform_real_distribution<double> urd(0.0, total);
        double pick = urd(rng);
        double cum = 0.0;
        int next = N - 1;
        for (int i = 0; i < N; i++) {
            cum += dist2[i];
            if (cum >= pick) { next = i; break; }
        }
        ctr_idx.push_back(next);
    }
    std::vector<float> centers((size_t)K * D);
    for (int k = 0; k < K; k++) {
        std::copy_n(X.data() + (size_t)ctr_idx[k] * D, D,
                    centers.data() + (size_t)k * D);
    }

    // ----- Lloyd -----
    std::vector<int> labels(N, 0);
    double prev_inertia = std::numeric_limits<double>::infinity();
    std::vector<double> sum((size_t)K * D, 0.0);
    std::vector<int> count(K, 0);
    for (int it = 0; it < max_iter; it++) {
        std::fill(sum.begin(), sum.end(), 0.0);
        std::fill(count.begin(), count.end(), 0);
        double inertia = 0.0;
        for (int i = 0; i < N; i++) {
            const float * x = X.data() + (size_t)i * D;
            int best = 0;
            double bd = std::numeric_limits<double>::infinity();
            for (int k = 0; k < K; k++) {
                double d = sq_dist(x, centers.data() + (size_t)k * D);
                if (d < bd) { bd = d; best = k; }
            }
            labels[i] = best;
            inertia += bd;
            count[best]++;
            for (int d = 0; d < D; d++) sum[(size_t)best * D + d] += x[d];
        }
        for (int k = 0; k < K; k++) {
            if (count[k] > 0) {
                for (int d = 0; d < D; d++) {
                    centers[(size_t)k * D + d] = (float)(sum[(size_t)k * D + d] / count[k]);
                }
            }
        }
        if (std::fabs(prev_inertia - inertia) < tol) break;
        prev_inertia = inertia;
    }
    return {std::move(labels), prev_inertia};
}

// Spectral embedding: take the first n_spks eigenvectors of L corresponding to
// the smallest eigenvalues, reverse the column order (NeMo convention), result
// is N×n_spks row-major.
static std::vector<float> spectral_embedding(const float * affinity, int N, int n_spks) {
    MatrixXd L = laplacian_from_affinity(affinity, N);
    auto [lam, vec] = eig_lap(L);  // sorted ascending
    std::vector<float> emb((size_t)N * n_spks, 0.0f);
    // NeMo:
    //   diffusion_map = vec[:, :n_spks]
    //   inv_idx = arange(n_spks-1, -1, -1)
    //   embedding = diffusion_map.T[inv_idx, :]   -- (n_spks, N) reversed
    //   return embedding[:n_spks].T               -- (N, n_spks)
    // i.e., column i in emb (0..n_spks-1) is vec[:, n_spks - 1 - i].
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < n_spks; k++) {
            emb[(size_t)i * n_spks + k] = (float)vec(i, n_spks - 1 - k);
        }
    }
    return emb;
}

nmesc_result nmesc_cluster(const float * embeddings, size_t N_, size_t D,
                           const nmesc_cfg & cfg) {
    nmesc_result r;
    int N = (int)N_;
    if (N <= 0) return r;

    auto aff = nmesc_cosine_affinity(embeddings, N, D);

    if (N <= cfg.min_samples_for_nmesc) {
        // NeMo: skip NMESC, use full affinity, set num_spk = 1.
        // Since for small N spectral clustering is unreliable, just emit zeros.
        r.est_num_speakers = (cfg.oracle_num_speakers > 0) ? cfg.oracle_num_speakers : 1;
        r.labels.assign(N, 0);
        r.p_hat = N - 1;
        return r;
    }

    // ----- NME analysis on (subsampled) affinity -----
    auto [subsample_ratio, sub_aff] = subsample_affinity(aff, N, cfg.nme_mat_size);
    int Nsub = (int)std::sqrt(sub_aff.size());

    int max_N = 0;
    auto p_list = (cfg.fixed_thres > 0.0f)
        ? std::vector<int>{ std::max(kMinPVal,
              (int)std::floor((double)Nsub * (double)cfg.fixed_thres)) }
        : p_value_list(Nsub, cfg.max_rp_threshold, cfg.sparse_search_volume,
                       /*sparse_search=*/true, max_N);

    // Per-p: find argmin g_p; record est_num_spk per candidate.
    int best_idx = 0;
    double best_g = std::numeric_limits<double>::infinity();
    std::vector<int> est_per_p((size_t)p_list.size(), 1);
    for (size_t i = 0; i < p_list.size(); i++) {
        auto out = get_eig_ratio(sub_aff, Nsub, p_list[i], cfg.max_num_speakers);
        est_per_p[i] = out.est_num_spk;
        if (out.g_p < best_g) { best_g = out.g_p; best_idx = (int)i; }
    }
    int rp_p_value = p_list[best_idx];
    int est_num_spk = est_per_p[best_idx];

    // Final affinity on the ORIGINAL (non-subsampled) matrix, with p scaled.
    int p_hat = subsample_ratio * rp_p_value;
    auto final_aff = affinity_graph_mat(aff, N, p_hat);

    // Connectivity guard.
    if (!fully_connected(final_aff, N)) {
        // Walk up p until the graph is connected, mirroring getMinimumConnection.
        int cur_p = 1;
        for (int p : p_list) {
            cur_p = subsample_ratio * p;
            auto trial = affinity_graph_mat(aff, N, cur_p);
            if (fully_connected(trial, N)) {
                final_aff = std::move(trial);
                break;
            }
            // last iteration falls through with the largest trial.
            final_aff = std::move(trial);
        }
        p_hat = cur_p;
    }

    int n_clusters = (cfg.oracle_num_speakers > 0)
        ? cfg.oracle_num_speakers
        : est_num_spk;
    n_clusters = std::max(1, std::min(n_clusters, cfg.max_num_speakers));

    // Spectral embedding then k-means++.
    if (n_clusters == 1) {
        r.labels.assign(N, 0);
    } else {
        auto emb = spectral_embedding(final_aff.data(), N, n_clusters);
        std::mt19937_64 rng(cfg.kmeans_seed);
        kmeans_out best;
        best.inertia = std::numeric_limits<double>::infinity();
        for (int t = 0; t < std::max(1, cfg.kmeans_random_trials); t++) {
            auto km = kmeans_pp(emb, N, n_clusters, n_clusters, rng);
            if (km.inertia < best.inertia) best = std::move(km);
        }
        r.labels = std::move(best.labels);
    }

    r.est_num_speakers = n_clusters;
    r.p_hat = p_hat;
    return r;
}
