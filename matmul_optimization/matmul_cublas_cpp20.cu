// matmul_cublas_cpp20.cpp
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>

constexpr int N = 4096;
constexpr int RUNS = 100;

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
    const size_t bytes = N * N * sizeof(float);

    // Allocate host pinned memory
    float *h_A{}, *h_B{}, *h_C{};
    checkCuda(cudaMallocHost(&h_A, bytes), "cudaMallocHost A");
    checkCuda(cudaMallocHost(&h_B, bytes), "cudaMallocHost B");
    checkCuda(cudaMallocHost(&h_C, bytes), "cudaMallocHost C");

    // Initialize matrices
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>((i % 17 + 1) * 1e-3f + 1.0f);
        h_B[i] = static_cast<float>((i % 13 + 1) * 1e-3f + 2.0f);
        h_C[i] = 0.0f;
    }

    // Allocate device memory
    float *d_A{}, *d_B{}, *d_C{};
    checkCuda(cudaMalloc(&d_A, bytes), "d_A");
    checkCuda(cudaMalloc(&d_B, bytes), "d_B");
    checkCuda(cudaMalloc(&d_C, bytes), "d_C");

    // Copy to device
    checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "H2D A");
    checkCuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "H2D B");

    // Create cuBLAS handle
    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "create handle");

    float alpha = 1.0f;
    float beta = 0.0f;

    // Warmup
    checkCublas(cublasSgemm(
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
    float ms_total = 0.0f;
    for (int r = 0; r < RUNS; ++r) {
        auto start = std::chrono::high_resolution_clock::now();

        checkCublas(cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N,
            &alpha,
            d_A, N,
            d_B, N,
            &beta,
            d_C, N
        ), "sgemm");

        checkCuda(cudaDeviceSynchronize(), "sync");

        auto end = std::chrono::high_resolution_clock::now();
        ms_total += std::chrono::duration<float, std::milli>(end - start).count();
    }

    float ms_avg = ms_total / RUNS;

    // Copy result back to host
    checkCuda(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost), "D2H C");

    // Print result sample and performance
    std::cout << "GPU cuBLAS " << N << "x" << N << " matrix multiplication\n";
    std::cout << "Average time per run: " << ms_avg / 1000.0f << " s\n";
    double gflops = 2.0 * N * N * N / (ms_avg * 1e6);
    std::cout << "Effective GFLOPS: " << gflops << "\n";
    std::cout << "C[1,1] = " << h_C[0] << "\n";

    // Cleanup
    checkCublas(cublasDestroy(handle), "destroy handle");
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C);

    return 0;
}
