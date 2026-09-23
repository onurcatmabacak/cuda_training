// matmul_cuda_faster.cu -- shared-memory tiled Float64 DGEMM (one output/thread).
// Includes a parallel CPU reference for verification.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#ifndef N
#define N 4096
#endif
#define TILE 32
#ifndef RUNS
#define RUNS 100
#endif
#define BLOCK_DIM_X TILE
#define BLOCK_DIM_Y TILE

static inline void checkCuda(cudaError_t e, const char *msg) {
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA Error %s: %s\n", msg, cudaGetErrorString(e));
    exit(EXIT_FAILURE);
  }
}

// Tiled shared-memory kernel: C = A * B  (Float64)
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

// CPU reference: cache-friendly (i,k,j) and parallelised with OpenMP.
// C must be zero-initialised before calling.
void matmul_cpu(const double *A, const double *B, double *C, int n) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    double *Ci = C + (size_t)i * n;
    for (int k = 0; k < n; ++k) {
      const double a = A[(size_t)i * n + k];
      const double *Bk = B + (size_t)k * n;
      for (int j = 0; j < n; ++j)
        Ci[j] += a * Bk[j];
    }
  }
}

int main(void) {
  const int bytes = N * N * sizeof(double);

  double *h_A = NULL, *h_B = NULL, *h_C = NULL, *h_C_ref = NULL;
  checkCuda(cudaMallocHost((void **)&h_A, bytes), "cudaMallocHost A");
  checkCuda(cudaMallocHost((void **)&h_B, bytes), "cudaMallocHost B");
  checkCuda(cudaMallocHost((void **)&h_C, bytes), "cudaMallocHost C");
  checkCuda(cudaMallocHost((void **)&h_C_ref, bytes), "cudaMallocHost C_ref");

  for (int i = 0; i < N * N; ++i) {
    h_A[i] = (double)((i % 17) + 1) * 1e-3 + 1.0;
    h_B[i] = (double)((i % 13) + 1) * 1e-3 + 2.0;
    h_C[i] = 0.0;
    h_C_ref[i] = 0.0;
  }

  double *d_A = NULL, *d_B = NULL, *d_C = NULL;
  checkCuda(cudaMalloc((void **)&d_A, bytes), "cudaMalloc d_A");
  checkCuda(cudaMalloc((void **)&d_B, bytes), "cudaMalloc d_B");
  checkCuda(cudaMalloc((void **)&d_C, bytes), "cudaMalloc d_C");

  checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "H2D A");
  checkCuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "H2D B");

  dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 gridDim(N / TILE, N / TILE);

  cudaEvent_t start, stop;
  checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
  checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop");

  // Warm up until the GPU reaches its boost clock.
  {
    cudaEvent_t warm_start, warm_stop;
    checkCuda(cudaEventCreate(&warm_start), "warmup event");
    checkCuda(cudaEventCreate(&warm_stop), "warmup event");
    checkCuda(cudaEventRecord(warm_start), "warmup record");
    float warm_ms = 0.0f;
    do {
      matmul_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
      checkCuda(cudaGetLastError(), "kernel launch warmup");
      checkCuda(cudaEventRecord(warm_stop), "warmup record stop");
      checkCuda(cudaEventSynchronize(warm_stop), "warmup sync");
      checkCuda(cudaEventElapsedTime(&warm_ms, warm_start, warm_stop),
                "warmup elapsed");
    } while (warm_ms < 2000.0f);
    checkCuda(cudaEventDestroy(warm_start), "warmup destroy");
    checkCuda(cudaEventDestroy(warm_stop), "warmup destroy");
  }

  const int runs = RUNS;
  float ms_total = 0.0f;
  for (int r = 0; r < runs; ++r) {
    checkCuda(cudaEventRecord(start), "cudaEventRecord start");
    matmul_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaEventRecord(stop), "cudaEventRecord stop");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    ms_total += ms;
  }
  float ms_avg = ms_total / runs;

  checkCuda(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost), "D2H C");

  printf("Computing CPU reference (parallel i-k-j)...\n");
  matmul_cpu(h_A, h_B, h_C_ref, N);

  double maxRelErr = 0.0;
  double eps = 1e-4;
  for (int i = 0; i < N * N; ++i) {
    double ref = h_C_ref[i];
    double got = h_C[i];
    double rel = fabs(ref - got) / (fabs(ref) + 1e-12);
    if (rel > maxRelErr) maxRelErr = rel;
    if (rel > eps) {
      fprintf(stderr, "Result mismatch at index %d: ref=%f got=%f rel=%e\n", i,
              ref, got, rel);
      break;
    }
  }

  double flops = 2.0 * (double)N * (double)N * (double)N;
  double gflops = (flops / 1e9) / (ms_avg / 1000.0);
  printf("Matrix mul %dx%d using GPU tiled DGEMM kernel\n", N, N);
  printf("Average kernel time over %d runs: %.3f ms\n", runs, ms_avg);
  printf("Effective GFLOPS: %.2f\n", gflops);
  printf("Max relative error vs CPU: %.3e\n", maxRelErr);
  if (maxRelErr <= eps) {
    printf("Result: OK\n");
  } else {
    printf("Result: MISMATCH\n");
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFreeHost(h_A);
  cudaFreeHost(h_B);
  cudaFreeHost(h_C);
  cudaFreeHost(h_C_ref);

  return 0;
}
