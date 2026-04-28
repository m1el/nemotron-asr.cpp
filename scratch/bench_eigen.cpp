// Eigen SelfAdjointEigenSolver bench, same problem sizes as bench_jacobi.cpp.
//
// Build:
//   g++ -O3 -std=c++17 -march=native -DNDEBUG \
//       -I vendor/eigen scratch/bench_eigen.cpp -o scratch/bench_eigen
// Run:
//   ./scratch/bench_eigen

#include <chrono>
#include <cstdio>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

static Eigen::MatrixXd random_symmetric(int n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    Eigen::MatrixXd A(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double v = nd(rng);
            A(i, j) = v;
            A(j, i) = v;
        }
    }
    return A;
}

static Eigen::MatrixXd random_psd(int n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++) L(i, j) = nd(rng);
    return L * L.transpose();
}

static void bench_one(int n, bool psd, uint64_t seed) {
    Eigen::MatrixXd A = psd ? random_psd(n, seed) : random_symmetric(n, seed);
    Eigen::MatrixXd A_orig = A;

    auto t0 = std::chrono::steady_clock::now();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(A);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto evals = es.eigenvalues();
    auto V = es.eigenvectors();

    Eigen::MatrixXd recon = V * evals.asDiagonal() * V.transpose();
    double recon_err = (A_orig - recon).norm() / A_orig.norm();
    double ortho_err = (V.transpose() * V - Eigen::MatrixXd::Identity(n, n)).norm();

    printf("  N=%4d  %-3s  time=%8.2f ms   recon=%8.2e  ortho=%8.2e\n",
           n, psd ? "PSD" : "SYM", ms, recon_err, ortho_err);
    fflush(stdout);
}

int main() {
    printf("Eigen SelfAdjointEigenSolver bench (single-thread, dense symmetric)\n");
    printf("------------------------------------------------------------------\n");
    int sizes[] = {64, 128, 256, 512, 1024, 2048};
    uint64_t seed = 0xC0FFEEull;
    for (int n : sizes) {
        bench_one(n, false, seed);
        bench_one(n, true, seed + 1);
    }
    return 0;
}
