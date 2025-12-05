#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mkl.h>
#include <omp.h>

#define N 4096       // Matrix size
#define RUNS 100      // Number of repetitions
#define ALIGNMENT 64 // Memory alignment

_Noreturn void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(EXIT_FAILURE);
}

// Allocate aligned 1D matrix
static inline double *alloc_matrix(size_t n) {
    double *p = (double *)mkl_malloc(n * n * sizeof(double), ALIGNMENT);
    if (!p) die("Memory allocation failed");
    return p;
}

// Fill matrix with random values
static inline void fill_random(double *M, size_t n) {
    for (size_t i = 0; i < n * n; i++)
        M[i] = (double)rand() / RAND_MAX;
}

// Zero matrix
static inline void zero_matrix(double *M, size_t n) {
    for (size_t i = 0; i < n * n; i++)
        M[i] = 0.0;
}

// Multiply matrices using MKL (C = A * B)
static inline void multiply_matrices_mkl(double *A, double *B, double *C, size_t n) {
    cblas_dgemm(
        CblasRowMajor, // Row-major layout
        CblasNoTrans,  // A not transposed
        CblasNoTrans,  // B not transposed
        n,             // rows of A/C
        n,             // columns of B/C
        n,             // columns of A / rows of B
        1.0,           // alpha
        A, n,          // matrix A and leading dimension
        B, n,          // matrix B and leading dimension
        0.0,           // beta
        C, n           // matrix C and leading dimension
    );
}

int main(int argc, char *argv[]) {
    srand((unsigned)time(NULL));

    // Determine number of threads
    int num_threads = 8; // default
    if (argc > 1) num_threads = atoi(argv[1]);
    if (num_threads <= 0) num_threads = 1;

    // Set MKL and OpenMP threads
    mkl_set_dynamic(0);
    mkl_set_num_threads(num_threads);
    omp_set_num_threads(num_threads);

    printf("Using %d thread(s)\n", num_threads);
    printf("Allocating matrices...\n");

    double *A = alloc_matrix(N);
    double *B = alloc_matrix(N);
    double *C = alloc_matrix(N);

    printf("Filling matrices with random values...\n");
    fill_random(A, N);
    fill_random(B, N);

    printf("Running matrix multiplication %d times for %dx%d matrices...\n", RUNS, N, N);

    double total_time = 0.0;
    for (int run = 0; run < RUNS; run++) {
        zero_matrix(C, N);

        double start_time = omp_get_wtime();
        multiply_matrices_mkl(A, B, C, N);
        double elapsed = omp_get_wtime() - start_time;

        total_time += elapsed;
        // printf("Run %2d: %.6f s\n", run + 1, elapsed);
    }

    double avg_time = total_time / RUNS;
    double gflops = 2.0 * N * N * N / (avg_time * 1e9);
    printf("\nAverage time: %.6f s | %.2f GFLOPS | C[0]=%.6f\n", avg_time, gflops, C[0]);

    mkl_free(A);
    mkl_free(B);
    mkl_free(C);
    return 0;
}
