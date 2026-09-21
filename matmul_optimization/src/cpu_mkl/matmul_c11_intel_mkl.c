// matmul_c11_blas_float64.c  (Float64 DGEMM version)
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>
#include <omp.h>

#ifndef MATMUL_N
#define MATMUL_N 4096
#endif
#ifndef MATMUL_RUNS
#define MATMUL_RUNS 100
#endif
/* Define after <cblas.h>/<omp.h>: their prototypes name a parameter 'N'. */
#define N MATMUL_N
#define RUNS MATMUL_RUNS
#define ALIGNMENT 64

static inline void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(EXIT_FAILURE);
}

static inline double *alloc_matrix(size_t n) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, ALIGNMENT, n * n * sizeof(double)) != 0)
        die("aligned alloc failed");
    return (double *)ptr;
}

static inline void fill_random(double *M, size_t n) {
    for (size_t i = 0; i < n*n; i++)
        M[i] = (double)rand() / (double)RAND_MAX;
}

int main() {
    srand((unsigned)time(NULL));

    printf("C11 DGEMM BLAS benchmark (Float64) %dx%d, %d runs\n", N, N, RUNS);

    double *A = alloc_matrix(N);
    double *B = alloc_matrix(N);
    double *C = alloc_matrix(N);

    fill_random(A, N);
    fill_random(B, N);

    // Warm-up run (Julia BLAS also does warm-up)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N,
                1.0, A, N, B, N, 0.0, C, N);

    double total = 0.0;

    for (int r = 0; r < RUNS; r++) {
        double t0 = omp_get_wtime();

        // Equivalent to Julia: mul!(C, A, B)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N,
                    1.0, A, N, B, N, 0.0, C, N);

        double t1 = omp_get_wtime();
        total += (t1 - t0);
    }

    double avg = total / RUNS;
    double gflops = 2.0 * N * N * N / (avg * 1e9);

    printf("\nAverage time per run: %.6f s\n", avg);
    printf("Effective GFLOPS: %.2f\n", gflops);
    printf("C[0] = %.6f\n", C[0]);  // same as Julia prints C[1,1]

    free(A);
    free(B);
    free(C);
    return 0;
}
