// matmul_cuda_cpp20_faster.cu
// C++20 Float64 DGEMM, tiled shared-memory kernel, vectorised host code.
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#ifndef N
#define N 4096
#endif
#define TILE 32
#ifndef RUNS
#define RUNS 100
#endif

inline void checkCuda(cudaError_t e, const char *msg) {
  if (e != cudaSuccess) {
    std::cerr << "CUDA Error " << msg << ": " << cudaGetErrorString(e) << "\n";
    std::exit(EXIT_FAILURE);
  }
}

__global__ void matmul_tiled(const double *__restrict__ A,
                             const double *__restrict__ B,
                             double *__restrict__ C, int n) {
  __shared__ double sA[TILE][TILE];
  __shared__ double sB[TILE][TILE];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = blockIdx.y * TILE + ty;
  const int col = blockIdx.x * TILE + tx;

  double acc = 0.0;

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

// CPU reference: cache-friendly (i,k,j) and parallelised. C zero-initialised.
void matmul_cpu(const std::vector<double> &A, const std::vector<double> &B,
                std::vector<double> &C, int n) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    double *Ci = C.data() + (size_t)i * n;
    for (int k = 0; k < n; ++k) {
      const double a = A[(size_t)i * n + k];
      const double *Bk = B.data() + (size_t)k * n;
      for (int j = 0; j < n; ++j)
        Ci[j] += a * Bk[j];
    }
  }
}

int main() {
  const size_t ne = (size_t)N * N;
  const size_t bytes = ne * sizeof(double);

  std::vector<double> h_A(ne), h_B(ne), h_C(ne), h_C_ref(ne, 0.0);
  for (size_t i = 0; i < ne; ++i) {
    h_A[i] = static_cast<double>((i % 17) + 1) * 1e-3 + 1.0;
    h_B[i] = static_cast<double>((i % 13) + 1) * 1e-3 + 2.0;
  }

  double *d_A{}, *d_B{}, *d_C{};
  checkCuda(cudaMalloc(&d_A, bytes), "d_A");
  checkCuda(cudaMalloc(&d_B, bytes), "d_B");
  checkCuda(cudaMalloc(&d_C, bytes), "d_C");

  checkCuda(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice), "H2D A");
  checkCuda(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice), "H2D B");

  dim3 block(TILE, TILE);
  dim3 grid(N / TILE, N / TILE);

  cudaEvent_t start, stop;
  checkCuda(cudaEventCreate(&start), "start event");
  checkCuda(cudaEventCreate(&stop), "stop event");

  // Warm up until the GPU reaches its boost clock.
  {
    auto warm_start = std::chrono::steady_clock::now();
    double warm_s = 0.0;
    do {
      matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
      checkCuda(cudaGetLastError(), "warmup");
      checkCuda(cudaDeviceSynchronize(), "sync warmup");
      warm_s = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             warm_start)
                   .count();
    } while (warm_s < 2.0);
  }

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

  checkCuda(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost), "D2H C");

  std::cout << "Computing CPU reference (parallel i-k-j)...\n";
  matmul_cpu(h_A, h_B, h_C_ref, N);

  double maxRelErr = 0.0;
  for (size_t i = 0; i < ne; ++i) {
    double rel = std::abs(h_C_ref[i] - h_C[i]) / (std::abs(h_C_ref[i]) + 1e-12);
    if (rel > maxRelErr) maxRelErr = rel;
  }

  double gflops = 2.0 * (double)N * N * N / (ms_avg * 1e6);
  std::cout << "Matrix mul " << N << "x" << N << " using CUDA tiled DGEMM kernel\n";
  std::cout << "Average kernel time over " << RUNS << " runs: " << ms_avg
            << " ms\n";
  std::cout << "Effective GFLOPS: " << gflops << "\n";
  std::cout << "Max relative error vs CPU: " << maxRelErr << "\n";
  std::cout << (maxRelErr < 1e-4 ? "Result: OK\n" : "Result: MISMATCH\n");

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
