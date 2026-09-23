// matmul_cuda.cu -- naive Float64 DGEMM: one thread per output, no shared memory.
// Compared against a single-threaded CPU reference.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#ifndef M
#define M 1024
#endif
#ifndef K
#define K 1024
#endif
#ifndef N
#define N 1024
#endif
#ifndef RUNS
#define RUNS 20
#endif
#ifndef WARMUP
#define WARMUP 3
#endif
#define BLOCK_SIZE 32

// CPU matrix multiplication (Float64)
void matmul_cpu(const double *A, const double *B, double *C, int m, int k, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int l = 0; l < k; l++) {
        sum += A[i * k + l] * B[l * n + j];
      }
      C[i * n + j] = sum;
    }
  }
}

// CUDA kernel: one thread per output element, reads straight from global memory
__global__ void matmul_gpu(const double *A, const double *B, double *C, int m,
                           int k, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < n) {
    double sum = 0.0;
    for (int l = 0; l < k; l++) {
      sum += A[row * k + l] * B[l * n + col];
    }
    C[row * n + col] = sum;
  }
}

void init_matrix(double *mat, int rows, int cols) {
  for (int i = 0; i < rows * cols; i++) {
    mat[i] = (double)rand() / RAND_MAX;
  }
}

double get_time(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
  double *h_A, *h_B, *h_C_cpu, *h_C_gpu;
  double *d_A, *d_B, *d_C;
  int size_A = M * K * sizeof(double);
  int size_B = K * N * sizeof(double);
  int size_C = M * N * sizeof(double);

  h_A = (double *)malloc(size_A);
  h_B = (double *)malloc(size_B);
  h_C_cpu = (double *)malloc(size_C);
  h_C_gpu = (double *)malloc(size_C);

  srand(time(NULL));
  init_matrix(h_A, M, K);
  init_matrix(h_B, K, N);

  cudaMalloc(&d_A, size_A);
  cudaMalloc(&d_B, size_B);
  cudaMalloc(&d_C, size_C);

  cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
               (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

  printf("Matrix size: %dx%d (Float64)\n", M, N);

  // Warm-up runs
  printf("Performing warm-up runs...\n");
  for (int i = 0; i < WARMUP; i++) {
    matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
    matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();
  }

  // Benchmark CPU implementation
  printf("Benchmarking CPU implementation...\n");
  double cpu_total_time = 0.0;
  for (int i = 0; i < RUNS; i++) {
    double start_time = get_time();
    matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
    double end_time = get_time();
    cpu_total_time += end_time - start_time;
  }
  double cpu_avg_time = cpu_total_time / RUNS;

  // Warm the GPU until it boosts its clocks.
  {
    double warm_start = get_time();
    do {
      matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
      cudaDeviceSynchronize();
    } while (get_time() - warm_start < 2.0);
  }

  // Benchmark GPU implementation
  printf("Benchmarking GPU implementation...\n");
  double gpu_total_time = 0.0;
  for (int i = 0; i < RUNS; i++) {
    double start_time = get_time();
    matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();
    double end_time = get_time();
    gpu_total_time += end_time - start_time;
  }
  double gpu_avg_time = gpu_total_time / RUNS;

  // Print results
  printf("CPU average time: %f microseconds\n", (cpu_avg_time * 1e6));
  printf("GPU average time: %f microseconds\n", (gpu_avg_time * 1e6));
  printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

  free(h_A);
  free(h_B);
  free(h_C_cpu);
  free(h_C_gpu);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
