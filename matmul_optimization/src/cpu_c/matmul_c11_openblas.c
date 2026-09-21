#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>

#ifndef MATMUL_N
#define MATMUL_N 1024 // Matrix size (N x N)
#endif
#ifndef MATMUL_RUNS
#define MATMUL_RUNS 100
#endif
/* Define after <cblas.h>: its prototypes name a parameter 'N'. */
#define N MATMUL_N
#define RUNS MATMUL_RUNS
#define ALIGNMENT 32 //64-byte alignment (AVX/AVX-512 friendly)

_Noreturn void die(const char *msg){
    fprintf(stderr, "%s\n", msg);
    exit(EXIT_FAILURE);
}

// Allocate a single contigious block (aligned)
static inline double *alloc_matrix(size_t n){
    double *ptr = aligned_alloc(ALIGNMENT, n * n * sizeof(double));
    if (!ptr)
        die("Memory allocation failed.");
    return ptr;
}

// Fill matrix with random numbers in parallel
static inline void fill_random(double *M, size_t n){
    for(size_t i = 0; i < n * n; i++)
        M[i] = (double)rand() / RAND_MAX;
}

// Zero matrix in parallel
static inline void zero_matrix(double *M, size_t n){
    for(size_t i = 0; i < n * n; i++)
        M[i] = 0.0;
}

// Multiply mTRICES USING OpenBLAS (C = A*B)

static inline void multiply_matrices_openblas(double *A, double *B, double *C, size_t n){
    cblas_dgemm(
        CblasRowMajor, // row-major layout
        CblasNoTrans,  // A not tranposed
        CblasNoTrans,  // B not transposed
        n,             // rows of A/C
        n,             // columns of B/c
        n,             // columns of A / rows of B
        1.0,           // alpha
        A, n,          // matrix A and leading dimension
        B, n,          // matrix B and leading dimension
        0.0,           // beta
        C, n          // matric C and leading dimension
    );
}

int main(){

    srand((unsigned)time(NULL));

    printf("Allocating matrices...\n");
    double* A = alloc_matrix(N);
    double* B = alloc_matrix(N);
    double* C = alloc_matrix(N);

    printf("Filling matrices with random values...\n");
    fill_random(A, N);
    fill_random(B, N);
    zero_matrix(C, N);

    printf("Running matrix multiplication %d times *%dx%d()... \n", RUNS, N, N);

    double total_time = 0.0;

    for (int run = 0; run < RUNS; run++){

        //printf("Multiplying matrices (%dx%d)...\n", N, N);
        zero_matrix(C, N); // reset results each run
        double start = (double)clock() / CLOCKS_PER_SEC;
        multiply_matrices_openblas(A, B, C, N);
        double end = (double)clock() / CLOCKS_PER_SEC;
        total_time += (end - start);
    }

    double avrg_time = total_time / RUNS;
    double gflops = (2.0 * N * N * N) / (avrg_time * 1e9);
    printf("\nAverage time over %d runs: %.3f seconds | %.2f GFLOPS\n", RUNS, avrg_time, gflops);

    // Optionally print a small portion of the result
    // printf("Sample output (C[0][0..4]): ");

    // for (int j = 0; j < 5; j++){
    //     printf("\n %8.4f", C[0][j]);
    // }

    // printf("\n");

    free(A);
    free(B);
    free(C);

    return 0;
}