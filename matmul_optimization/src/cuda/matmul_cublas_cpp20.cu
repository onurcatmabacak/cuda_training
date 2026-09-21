// matmul_cublas_cpp20_double.cpp
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>

#ifndef MATMUL_N
#define MATMUL_N 4096
#endif
#ifndef MATMUL_RUNS
#define MATMUL_RUNS 100
#endif

constexpr int N = MATMUL_N;
constexpr int RUNS = MATMUL_RUNS;

inline void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA Error " << msg << ": " << cudaGetErrorString(e) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

inline void checkCublas(cublasStatus_t s, const char* msg) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error " << msg << ": " << s << "\n";
        std::exit(EXIT_FAILURE);
    }
}

int main() {
    const size_t bytes = N * N * sizeof(double);

    // Allocate host pinned memory
    double *h_A{}, *h_B{}, *h_C{};
    checkCuda(cudaMallocHost(&h_A, bytes), "cudaMallocHost A");
    checkCuda(cudaMallocHost(&h_B, bytes), "cudaMallocHost B");
    checkCuda(cudaMallocHost(&h_C, bytes), "cudaMallocHost C");

    // Initialize matrices
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<double>((i % 17 + 1) * 1e-3 + 1.0);
        h_B[i] = static_cast<double>((i % 13 + 1) * 1e-3 + 2.0);
        h_C[i] = 0.0;
    }

    // Allocate device memory
    double *d_A{}, *d_B{}, *d_C{};
    checkCuda(cudaMalloc(&d_A, bytes), "d_A");
    checkCuda(cudaMalloc(&d_B, bytes), "d_B");
    checkCuda(cudaMalloc(&d_C, bytes), "d_C");

    // Copy to device
    checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "H2D A");
    checkCuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "H2D B");

    // Create cuBLAS handle
    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "create handle");

    double alpha = 1.0, beta = 0.0;

    // Warmup
    checkCublas(cublasDgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, N, N,
        &alpha,
        d_A, N,
        d_B, N,
        &beta,
        d_C, N
    ), "warmup");

    checkCuda(cudaDeviceSynchronize(), "sync warmup");

    // Timing
    double ms_total = 0.0;
    for (int r = 0; r < RUNS; ++r) {
        auto start = std::chrono::high_resolution_clock::now();

        checkCublas(cublasDgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N,
            &alpha,
            d_A, N,
            d_B, N,
            &beta,
            d_C, N
        ), "dgemm");

        checkCuda(cudaDeviceSynchronize(), "sync");

        auto end = std::chrono::high_resolution_clock::now();
        ms_total += std::chrono::duration<double, std::milli>(end - start).count();
    }

    double ms_avg_s = ms_total / RUNS / 1000.0; // seconds
    double gflops = 2.0 * N * N * N / (ms_avg_s * 1e9);

    // Copy result back to host
    checkCuda(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost), "D2H C");

    std::cout << "GPU cuBLAS " << N << "x" << N << " matrix multiplication\n";
    std::cout << "Average time per run: " << ms_avg_s << " s\n";
    std::cout << "Effective GFLOPS: " << gflops << "\n";
    std::cout << "C[1,1] = " << h_C[0] << "\n";

    // Cleanup
    checkCublas(cublasDestroy(handle), "destroy handle");
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C);

    return 0;
}
