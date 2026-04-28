// NME-SC (Normalized Maximum Eigengap Spectral Clustering) for speaker diarization.
//
// Ported from NeMo's offline_clustering.py (NMESC + SpectralClustering).
// Uses Eigen for the symmetric eigendecomposition.

#ifndef NEMOTRON_DIARIZE_CLUSTER_H
#define NEMOTRON_DIARIZE_CLUSTER_H

#include <cstddef>
#include <cstdint>
#include <vector>

struct nmesc_cfg {
    int   max_num_speakers     = 8;
    float max_rp_threshold     = 0.25f;
    int   sparse_search_volume = 30;
    int   nme_mat_size         = 512;
    int   min_samples_for_nmesc = 6;
    int   oracle_num_speakers  = -1;        // < 0 -> estimate
    float fixed_thres          = -1.0f;     // > 0 -> skip NME analysis
    int   kmeans_random_trials = 1;
    uint64_t kmeans_seed       = 0;
};

struct nmesc_result {
    int est_num_speakers = 1;
    int p_hat = 1;
    std::vector<int> labels; // length N
};

// Compute speaker labels via NME-SC clustering.
//
// Inputs:
//   embeddings  — N×D row-major matrix (embeddings[i*D + d])
//   N, D
//   cfg
//
// Returns labels of length N where labels[i] in [0, est_num_speakers).
nmesc_result nmesc_cluster(
    const float * embeddings, size_t N, size_t D,
    const nmesc_cfg & cfg = {});

// Lower-level building blocks (exposed for testing).

// NeMo getCosAffinityMatrix:
//   - cos similarity with eps=3.5e-4
//   - diagonal forced to 1
//   - min-max scaled to [0, 1]
// Result is N×N row-major.
std::vector<float> nmesc_cosine_affinity(
    const float * embeddings, size_t N, size_t D);

#endif // NEMOTRON_DIARIZE_CLUSTER_H
