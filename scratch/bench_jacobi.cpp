// Cyclic Jacobi rotation eigensolver for dense symmetric matrices.
//
// Self-contained prototype to be promoted into src/diarize_cluster.cpp once
// validated. Targets the diarization affinity / normalized-Laplacian use case:
// real symmetric N x N with N typically 100..2000.
//
// Build:
//   g++ -O3 -std=c++17 -march=native scratch/bench_jacobi.cpp -o scratch/bench_jacobi
// Run:
//   ./scratch/bench_jacobi

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Cyclic Jacobi eigensolver.
//
// Input:  A is row-major NxN, symmetric. Overwritten with diagonalized form.
// Output: eigvals[i]  = i-th eigenvalue (unsorted)
//         V (row-major NxN) accumulates the right eigenvectors as columns:
//         A_orig * V[:, i] = eigvals[i] * V[:, i].
//
// Returns the number of sweeps performed.
//
// Convergence: stop when off-diagonal Frobenius norm < tol * diag Frobenius norm.
// O(N^3) per sweep, typically converges in 5..15 sweeps -> O(N^3 * sweeps).
// ---------------------------------------------------------------------------

static int jacobi_eigen(double *A, int n, double *eigvals, double *V,
                        double tol = 1e-12, int max_sweeps = 80) {
    // Initialize V = I.
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            V[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    auto off_norm_sq = [&]() {
        double s = 0.0;
        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) {
                double a = A[p * n + q];
                s += a * a;
            }
        }
        return s; // upper triangle only; full off-diag = 2*s
    };

    auto diag_norm_sq = [&]() {
        double s = 0.0;
        for (int i = 0; i < n; i++) {
            s += A[i * n + i] * A[i * n + i];
        }
        return s;
    };

    double tol2 = tol * tol;
    int sweep;
    for (sweep = 0; sweep < max_sweeps; sweep++) {
        double off2 = off_norm_sq();
        double diag2 = diag_norm_sq();
        if (off2 <= tol2 * std::max(diag2, 1e-300)) break;

        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) {
                double apq = A[p * n + q];
                if (std::fabs(apq) < 1e-300) continue;

                double app = A[p * n + p];
                double aqq = A[q * n + q];

                // Compute c, s for Givens rotation that zeroes A[p,q].
                double theta = (aqq - app) / (2.0 * apq);
                double t;
                if (std::fabs(theta) > 1e15) {
                    t = 0.5 / theta; // avoid overflow in 1/(|theta|+sqrt(...))
                } else {
                    double sgn = (theta >= 0.0) ? 1.0 : -1.0;
                    t = sgn / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                }
                double c = 1.0 / std::sqrt(1.0 + t * t);
                double s = t * c;

                // Update diagonal (p,p), (q,q).
                A[p * n + p] = app - t * apq;
                A[q * n + q] = aqq + t * apq;
                A[p * n + q] = 0.0;
                A[q * n + p] = 0.0;

                // Update remaining rows/cols of A.
                for (int i = 0; i < n; i++) {
                    if (i == p || i == q) continue;
                    double aip = A[i * n + p];
                    double aiq = A[i * n + q];
                    A[i * n + p] = c * aip - s * aiq;
                    A[i * n + q] = s * aip + c * aiq;
                    A[p * n + i] = A[i * n + p];
                    A[q * n + i] = A[i * n + q];
                }

                // Accumulate rotation into V (eigenvectors are columns).
                for (int i = 0; i < n; i++) {
                    double vip = V[i * n + p];
                    double viq = V[i * n + q];
                    V[i * n + p] = c * vip - s * viq;
                    V[i * n + q] = s * vip + c * viq;
                }
            }
        }
    }

    for (int i = 0; i < n; i++) eigvals[i] = A[i * n + i];
    return sweep;
}

// ---------------------------------------------------------------------------
// Validation + bench harness.
// ---------------------------------------------------------------------------

static void make_random_symmetric(std::vector<double> &A, int n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    A.assign(n * n, 0.0);
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double v = nd(rng);
            A[i * n + j] = v;
            A[j * n + i] = v;
        }
    }
}

// Make a "Laplacian-like" PSD matrix: A = L L^T (with L lower-tri Gaussian).
// More representative of the normalized-Laplacian we'll be eigendecomposing.
static void make_psd(std::vector<double> &A, int n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> L(n * n, 0.0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) L[i * n + j] = nd(rng);
    }
    A.assign(n * n, 0.0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k <= std::min(i, j); k++) {
                s += L[i * n + k] * L[j * n + k];
            }
            A[i * n + j] = s;
        }
    }
}

// Reconstruction error: ||A_orig - V diag(lambda) V^T||_F / ||A_orig||_F.
static double reconstruction_err(const std::vector<double> &A_orig,
                                 const std::vector<double> &eigvals,
                                 const std::vector<double> &V, int n) {
    double err2 = 0.0, norm2 = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double recon = 0.0;
            for (int k = 0; k < n; k++) {
                recon += V[i * n + k] * eigvals[k] * V[j * n + k];
            }
            double d = A_orig[i * n + j] - recon;
            err2 += d * d;
            norm2 += A_orig[i * n + j] * A_orig[i * n + j];
        }
    }
    return std::sqrt(err2 / std::max(norm2, 1e-300));
}

// Orthogonality error: ||V^T V - I||_F.
static double ortho_err(const std::vector<double> &V, int n) {
    double err2 = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double dot = 0.0;
            for (int k = 0; k < n; k++) dot += V[k * n + i] * V[k * n + j];
            double e = dot - (i == j ? 1.0 : 0.0);
            err2 += e * e;
        }
    }
    return std::sqrt(err2);
}

static void bench_one(int n, bool psd, uint64_t seed) {
    std::vector<double> A;
    if (psd) make_psd(A, n, seed);
    else make_random_symmetric(A, n, seed);

    std::vector<double> A_orig = A;
    std::vector<double> eigvals(n);
    std::vector<double> V(n * n);

    auto t0 = std::chrono::steady_clock::now();
    int sweeps = jacobi_eigen(A.data(), n, eigvals.data(), V.data());
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double recon = reconstruction_err(A_orig, eigvals, V, n);
    double ortho = ortho_err(V, n);

    printf("  N=%4d  %-3s  sweeps=%2d  time=%8.2f ms   recon=%8.2e  ortho=%8.2e\n",
           n, psd ? "PSD" : "SYM", sweeps, ms, recon, ortho);
    fflush(stdout);
}

int main() {
    printf("Jacobi eigensolver bench (single-thread, dense symmetric)\n");
    printf("---------------------------------------------------------\n");
    int sizes[] = {64, 128, 256, 512, 1024, 2048};
    uint64_t seed = 0xC0FFEEull;
    for (int n : sizes) {
        bench_one(n, false, seed);
        bench_one(n, true, seed + 1);
    }
    return 0;
}
