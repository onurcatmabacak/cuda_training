// File: $NAME.cpp
// C++20, optimized MKL GEMM benchmark
// - NUMA-first-touch initialization (OpenMP)
// - Warm-up GEMM
// - Raw MKL allocations (mkl_malloc/mkl_free)
// - Uses omp_get_wtime() for timing
// Build with the provided build_and_run.sh script

#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <mkl.h>
#include <omp.h>

#ifndef MATMUL_N
#define MATMUL_N 4096
#endif
#ifndef MATMUL_RUNS
#define MATMUL_RUNS 100
#endif

constexpr std::size_t N = MATMUL_N;     // matrix dimension
constexpr int RUNS = MATMUL_RUNS;       // measured runs
constexpr int ALIGNMENT = 64;       // mkl_malloc alignment

[[noreturn]] static void die(const char* msg) {
    std::cerr << msg << "\n";
    std::exit(EXIT_FAILURE);
}

static double* alloc_matrix(std::size_t n) {
    double* p = static_cast<double*>(mkl_malloc(n * n * sizeof(double), ALIGNMENT));
    if (!p) die("mkl_malloc failed");
    return p;
}

// NUMA-aware first-touch initialization: each thread initializes its block
static void init_matrices_numa(double* A, double* B, std::size_t n) {
    // Use per-thread RNG with simple LCG (fast)
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        // Split by rows for good locality
        std::size_t rows_per = n / nthreads;
        std::size_t r0 = tid * rows_per;
        std::size_t r1 = (tid == nthreads - 1) ? n : r0 + rows_per;

        // fast LCG seed
        uint64_t seed = 1469598103934665603ull + (uint64_t)tid;
        auto lcg = [&seed]() {
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            return (double)(seed & 0xFFFFFFFFFFFFull) / (double)0xFFFFFFFFFFFFull;
        };

        for (std::size_t i = r0; i < r1; ++i) {
            std::size_t row_off = i * n;
            for (std::size_t j = 0; j < n; ++j) {
                A[row_off + j] = lcg();
                B[row_off + j] = lcg();
            }
        }
    }
}

static inline void zero_matrix(double* M, std::size_t n) {
    // memset is fastest for large blocks
    std::memset(M, 0, n * n * sizeof(double));
}

static inline void gemm_mkl(double* A, double* B, double* C, std::size_t n) {
    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        static_cast<MKL_INT>(n),
        static_cast<MKL_INT>(n),
        static_cast<MKL_INT>(n),
        1.0,
        A, static_cast<MKL_INT>(n),
        B, static_cast<MKL_INT>(n),
        0.0,
        C, static_cast<MKL_INT>(n)
    );
}

int main(int argc, char** argv) {
    // Optional: take thread count from argv[1]
    int threads = 8;
    if (argc > 1) {
        threads = std::atoi(argv[1]);
        if (threads <= 0) threads = 1;
    }

    // Let the runtime/environment control MKL/OMP threads. But set these as safety.
    mkl_set_dynamic(0);
    mkl_set_num_threads(threads);
    omp_set_num_threads(threads);

    std::cout << "MKL/OMP threads: " << threads << "\n";
    std::cout << "Allocating matrices (" << N << "x" << N << ")...\n";

    double* A = alloc_matrix(N);
    double* B = alloc_matrix(N);
    double* C = alloc_matrix(N);

    std::cout << "Initializing matrices with NUMA-first-touch (parallel)...\n";
    init_matrices_numa(A, B, N);

    // Touch C pages to ensure locality
    std::cout << "Zeroing C (page warm-up)...\n";
    zero_matrix(C, N);

    std::cout << "Performing warm-up GEMM...\n";
    gemm_mkl(A, B, C, N);

    double total = 0.0;
    for (int run = 0; run < RUNS; ++run) {
        // zero C to avoid accumulation across runs (and to re-touch pages)
        zero_matrix(C, N);

        double t0 = omp_get_wtime();
        gemm_mkl(A, B, C, N);
        double dt = omp_get_wtime() - t0;
        total += dt;

        // Uncomment to print per-run timing (costly)
        // std::cout << "run " << run << ": " << dt << " s\n";
    }

    double avg = total / RUNS;
    double gflops = 2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N) / (avg * 1e9);

    std::cout << "\nAverage time: " << avg << " s | " << gflops << " GFLOPS | C[0]=" << C[0] << "\n";

    mkl_free(A);
    mkl_free(B);
    mkl_free(C);
    return 0;
}

