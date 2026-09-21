// matmul_c_cublas_double.c
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifndef N
#define N 4096
#endif
#ifndef RUNS
#define RUNS 100
#endif

inline void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) { 
        printf("CUDA Error %s: %s\n", msg, cudaGetErrorString(e)); 
        exit(EXIT_FAILURE); 
    }
}

inline void checkCublas(cublasStatus_t s, const char* msg) {
    if (s != CUBLAS_STATUS_SUCCESS) { 
        printf("cuBLAS Error %s\n", msg); 
        exit(EXIT_FAILURE); 
    }
}

int main() {
    size_t bytes = N * N * sizeof(double);

    double *h_A = (double*)malloc(bytes);
    double *h_B = (double*)malloc(bytes);
    double *h_C = (double*)malloc(bytes);

    for (int i = 0; i < N*N; i++) {
        h_A[i] = (double)((i % 17 + 1) * 1e-3 + 1.0);
        h_B[i] = (double)((i % 13 + 1) * 1e-3 + 2.0);
        h_C[i] = 0.0;
    }

    double *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc(&d_A, bytes), "d_A");
    checkCuda(cudaMalloc(&d_B, bytes), "d_B");
    checkCuda(cudaMalloc(&d_C, bytes), "d_C");

    checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "H2D A");
    checkCuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "H2D B");

    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "create handle");

    const double alpha = 1.0, beta = 0.0;

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "start event");
    checkCuda(cudaEventCreate(&stop), "stop event");

    // Warmup
    checkCublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N), "warmup");
    checkCuda(cudaDeviceSynchronize(), "sync warmup");

    float total_ms = 0.0f;
    for (int r = 0; r < RUNS; r++) {
        checkCuda(cudaEventRecord(start), "start");
        checkCublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N), "dgemm");
        checkCuda(cudaEventRecord(stop), "stop");
        checkCuda(cudaEventSynchronize(stop), "sync");
        float ms = 0.0f;
        checkCuda(cudaEventElapsedTime(&ms, start, stop), "elapsed");
        total_ms += ms;
    }

    double avg_time_s = (total_ms / RUNS) / 1000.0;
    double gflops = 2.0 * N * N * N / (avg_time_s * 1e9);

    checkCuda(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost), "D2H C");

    printf("Average time per run: %f s\n", avg_time_s);
    printf("Effective GFLOPS: %f\n", gflops);
    printf("C[0,0] = %f\n", h_C[0]);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    cublasDestroy(handle);

    return 0;
}
