// matmul_cuda_cpp20_960m_fast.cu
// C++20 CUDA matrix multiplication for GeForce 960M
// Features:
// - 1 thread per output element (best for Maxwell)
// - Shared-memory tiling (32x32)
// - Pinned host memory
// - Loop unrolling in inner tile
// - Modern C++20 host code

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

constexpr int N = 4096;
constexpr int TILE = 32;
constexpr int RUNS = 100;

inline void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA Error " << msg << ": " << cudaGetErrorString(e) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

__global__ void matmul_tiled(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int n) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;

    for (int m = 0; m < n; m += TILE) {
        sA[ty][tx] = A[row * n + (m + tx)];
        sB[ty][tx] = B[(m + ty) * n + col];

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += sA[ty][k] * sB[k][tx];
        }
        __syncthreads();
    }

    C[row * n + col] = acc;
}

void matmul_cpu(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
}

int main() {
    const size_t bytes = N * N * sizeof(float);

    float *h_A{}, *h_B{}, *h_C{}, *h_C_ref{};
    checkCuda(cudaMallocHost(&h_A, bytes), "cudaMallocHost A");
    checkCuda(cudaMallocHost(&h_B, bytes), "cudaMallocHost B");
    checkCuda(cudaMallocHost(&h_C, bytes), "cudaMallocHost C");
    checkCuda(cudaMallocHost(&h_C_ref, bytes), "cudaMallocHost C_ref");

    for (int i = 0; i < N*N; ++i) {
        h_A[i] = static_cast<float>((i % 17 + 1) * 1e-3f + 1.0f);
        h_B[i] = static_cast<float>((i % 13 + 1) * 1e-3f + 2.0f);
        h_C[i] = 0.0f;
        h_C_ref[i] = 0.0f;
    }

    float *d_A{}, *d_B{}, *d_C{};
    checkCuda(cudaMalloc(&d_A, bytes), "d_A");
    checkCuda(cudaMalloc(&d_B, bytes), "d_B");
    checkCuda(cudaMalloc(&d_C, bytes), "d_C");

    checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "H2D A");
    checkCuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "H2D B");

    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, N / TILE);

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "start event");
    checkCuda(cudaEventCreate(&stop), "stop event");

    // Warmup
    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
    checkCuda(cudaGetLastError(), "warmup");
    checkCuda(cudaDeviceSynchronize(), "sync warmup");

    float ms_total = 0.0f;
    for (int r = 0; r < RUNS; ++r) {
        checkCuda(cudaEventRecord(start), "start");
        matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
        checkCuda(cudaGetLastError(), "kernel launch");
        checkCuda(cudaEventRecord(stop), "stop");
        checkCuda(cudaEventSynchronize(stop), "sync");
        float ms;
        checkCuda(cudaEventElapsedTime(&ms, start, stop), "elapsed");
        ms_total += ms;
    }
    float ms_avg = ms_total / RUNS;

    checkCuda(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost), "D2H C");

    std::vector<float> vec_A(h_A, h_A + N*N);
    std::vector<float> vec_B(h_B, h_B + N*N);
    std::vector<float> vec_C_ref(N*N);
    matmul_cpu(vec_A, vec_B, vec_C_ref, N);

    double maxRelErr = 0.0;
    for (int i = 0; i < N*N; ++i) {
        double rel = std::abs(vec_C_ref[i] - h_C[i]) / (std::abs(vec_C_ref[i]) + 1e-12);
        if (rel > maxRelErr) maxRelErr = rel;
    }

    double gflops = 2.0 * N * N * N / (ms_avg * 1e6);
    std::cout << "Matrix mul " << N << "x" << N << " using CUDA 960M kernel\n";
    std::cout << "Average kernel time: " << ms_avg << " ms\n";
    std::cout << "Effective GFLOPS: " << gflops << "\n";
    std::cout << "Max relative error vs CPU: " << maxRelErr << "\n";
    std::cout << (maxRelErr < 1e-4 ? "Result: OK\n" : "Result: MISMATCH\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C); cudaFreeHost(h_C_ref);

    return 0;
}
