#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <omp.h>

#ifndef N
#define N 1024 // Matrix size (N x N)
#endif
#ifndef RUNS
#define RUNS 100
#endif
#define BLOCK 32 // Block size for cache tiling
#define ALIGNMENT 64 //64-byte alignment (AVX/AVX-512 friendly)

_Noreturn void die(const char *msg){
    fprintf(stderr, "%s\n", msg);
    exit(EXIT_FAILURE);
}

// Allocate a single contigious block (aligned)
static inline double *alloc_matrix(size_t n){
    double *ptr = NULL;
    if (posix_memalign((void **)&ptr, ALIGNMENT, n * n * sizeof(double)) != 0)
        die("Memory allocation failed.");
    return ptr;
}

// Fill matrix with random numbers in parallel
static inline void fill_random(double *M, size_t n){
    #pragma omp parallel for schedule(static)
    for(size_t i = 0; i < n * n; i++)
        M[i] = (double)rand() / RAND_MAX;
}

// Zero matrix in parallel
static inline void zero_matrix(double *M, size_t n){
    #pragma omp parallel for schedule(static)
    for(size_t i = 0; i < n * n; i++)
        M[i] = 0.0;
}

// Blocked + parallel + vectorized matrix multiplication
// Optimized blocked multiply for base case
static inline void matmul_blocked(double *restrict A, double *restrict B, double *restrict C, size_t n, size_t stride){
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t ii = 0; ii < n; ii += BLOCK){
        for (size_t kk = 0; kk < n; kk += BLOCK){
            for (size_t jj = 0; jj < n; jj += BLOCK){

                size_t imax = (ii + BLOCK < n) ? (ii + BLOCK) : n;
                size_t kmax = (kk + BLOCK < n) ? (kk + BLOCK) : n;
                size_t jmax = (jj + BLOCK < n) ? (jj + BLOCK) : n;

                for (size_t i = ii; i < imax; i++){
                    for (size_t k = kk; k < kmax; k++){
                        #pragma omp simd
                        for (size_t j = jj; j < jmax; j++){
                            C[i * stride + j] += A[i * stride + k] * B[k * stride + j];
                        }
                    }
                }
            }
        }
    }
}

// Recursive parallel divide-and-conquer multiply
void matmul_recursive(double *restrict A, double *restrict B, double *restrict C, size_t n, size_t stride){

    if(n <= BLOCK){
        matmul_blocked(A,B,C,n,stride);
        return;
    }

    size_t half = n/2;

    // Pointers to submatrices
    double *A11 = A;
    double *A12 = A + half;
    double *A21 = A + half*stride;
    double *A22 = A + half*stride + half;

    double *B11 = B;
    double *B12 = B + half;
    double *B21 = B + half*stride;
    double *B22 = B + half*stride + half;

    double *C11 = C;
    double *C12 = C + half;
    double *C21 = C + half*stride;
    double *C22 = C + half*stride + half;

    #pragma omp task shared(A11,B11,C11)
    matmul_recursive(A11,B11,C11,half,stride);
    #pragma omp task shared(A12,B21,C11)
    matmul_recursive(A12,B21,C11,half,stride);

    #pragma omp task shared(A11,B12,C12)
    matmul_recursive(A11,B12,C12,half,stride);
    #pragma omp task shared(A12,B22,C12)
    matmul_recursive(A12,B22,C12,half,stride);

    #pragma omp task shared(A21,B11,C21)
    matmul_recursive(A21,B11,C21,half,stride);
    #pragma omp task shared(A22,B21,C21)
    matmul_recursive(A22,B21,C21,half,stride);

    #pragma omp task shared(A21,B12,C22)
    matmul_recursive(A21,B12,C22,half,stride);
    #pragma omp task shared(A22,B22,C22)
    matmul_recursive(A22,B22,C22,half,stride);

    #pragma omp taskwait
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
        double start = omp_get_wtime();

        #pragma omp parallel
        {
            #pragma omp single 
            matmul_recursive(A,B,C,N,N);
        }
        
        double end = omp_get_wtime();
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