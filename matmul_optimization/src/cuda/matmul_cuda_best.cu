// matmul_cuda_best.cu
// Optimised SGEMM: a hand-written register-tiled CUDA kernel vs cuBLAS.
//
//   Micro-tile per thread : TM x TN = 8 x 8  (64 FP32 accumulators)
//   Block tile            : BM x BN x BK = 128 x 128 x 16, 256 threads
//   Shared memory         : As[BM][BK] + Bs[BK][BN]
//   Bank-conflict-free    : the A fragment is a warp broadcast
//                           (As[ty*TM+i][k]); the B fragment is stride-1
//                           (Bs[k][tx + j*16]) instead of a strided pattern.
//
// Usage: matmul_cuda_best [cuda|cublas|both]
//   cuda   -> time the hand-written kernel
//   cublas -> time cuBLAS SGEMM
//   both   -> time both and check the custom kernel against cuBLAS

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifndef N
#define N 4096
#endif
#ifndef RUNS
#define RUNS 10
#endif

#ifndef BM
#define BM 128
#endif
#ifndef BN
#define BN 128
#endif
#ifndef BK
#define BK 32
#endif
#ifndef TM
#define TM 8
#endif
#ifndef TN
#define TN 8
#endif
#ifndef THREADS
#define THREADS 256
#endif

#define CUDA_CHECK(...)                                                          \
  do {                                                                           \
    cudaError_t e = (__VA_ARGS__);                                               \
    if (e != cudaSuccess) {                                                      \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,              \
              cudaGetErrorString(e));                                            \
      exit(1);                                                                   \
    }                                                                            \
  } while (0)

#define CUBLAS_CHECK(...)                                                        \
  do {                                                                           \
    cublasStatus_t s = (__VA_ARGS__);                                            \
    if (s != CUBLAS_STATUS_SUCCESS) {                                            \
      fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)s);   \
      exit(1);                                                                   \
    }                                                                            \
  } while (0)

__global__ void __launch_bounds__(THREADS, 2)
sgemm_regtile(const float *__restrict__ A, const float *__restrict__ B,
              float *__restrict__ C, int M, int Nn, int K) {
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  const int tx = threadIdx.x; // 0..15
  const int ty = threadIdx.y; // 0..15
  const int tid = ty * blockDim.x + tx;
  const int row0 = blockIdx.y * BM;
  const int col0 = blockIdx.x * BN;

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
      acc[i][j] = 0.0f;

  for (int k0 = 0; k0 < K; k0 += BK) {
    // A tile: BM*BK elements, contiguous in K -> coalesced loads, no smem conflict
#pragma unroll
    for (int idx = tid; idx < BM * BK; idx += THREADS) {
      const int m = idx / BK, k = idx % BK;
      const int gm = row0 + m, gk = k0 + k;
      As[m][k] = (gm < M && gk < K) ? A[(size_t)gm * K + gk] : 0.0f;
    }
    // B tile: BK*BN elements, contiguous in N
#pragma unroll
    for (int idx = tid; idx < BK * BN; idx += THREADS) {
      const int k = idx / BN, n = idx % BN;
      const int gk = k0 + k, gn = col0 + n;
      Bs[k][n] = (gk < K && gn < Nn) ? B[(size_t)gk * Nn + gn] : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      float a[TM], b[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i)
        a[i] = As[ty * TM + i][k]; // broadcast within the warp
#pragma unroll
      for (int j = 0; j < TN; ++j)
        b[j] = Bs[k][tx + j * 16]; // stride-1 across tx -> no bank conflict
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] += a[i] * b[j];
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int gm = row0 + ty * TM + i;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int gn = col0 + tx + j * 16;
      if (gm < M && gn < Nn)
        C[(size_t)gm * Nn + gn] = acc[i][j];
    }
  }
}

static void fill_mat(float *p, size_t n, int mod, float base) {
  for (size_t i = 0; i < n; ++i)
    p[i] = (float)((i % mod) + 1) * 1e-3f + base;
}

int main(int argc, char **argv) {
  const char *which = (argc > 1) ? argv[1] : "both";
  const bool do_cuda = !strcmp(which, "cuda") || !strcmp(which, "both");
  const bool do_blas = !strcmp(which, "cublas") || !strcmp(which, "both");
  const int M = N, Nn = N, K = N;
  const size_t ne = (size_t)N * N, bytes = ne * sizeof(float);
  const double flop = 2.0 * (double)N * N * N;

  float *hA = (float *)malloc(bytes), *hB = (float *)malloc(bytes),
        *hC = (float *)malloc(bytes);
  fill_mat(hA, ne, 17, 1.0f);
  fill_mat(hB, ne, 13, 2.0f);
  memset(hC, 0, bytes);

  float *dA, *dB, *dC, *dCref;
  CUDA_CHECK(cudaMalloc(&dA, bytes));
  CUDA_CHECK(cudaMalloc(&dB, bytes));
  CUDA_CHECK(cudaMalloc(&dC, bytes));
  CUDA_CHECK(cudaMalloc(&dCref, bytes));
  CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

  cublasHandle_t h;
  CUBLAS_CHECK(cublasCreate(&h));
  const float alpha = 1.0f, beta = 0.0f;

  // cuBLAS reference (also used to verify the custom kernel)
  CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, Nn, Nn, Nn, &alpha, dA,
                           Nn, dB, Nn, &beta, dCref, Nn));
  CUDA_CHECK(cudaDeviceSynchronize());

  dim3 block(16, 16);
  dim3 grid((Nn + BN - 1) / BN, (M + BM - 1) / BM);

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  // ---- warm up the GPU to boost clocks (2 s of sustained work) -------------
  if (do_cuda || do_blas) {
    cudaEvent_t w0, w1;
    CUDA_CHECK(cudaEventCreate(&w0));
    CUDA_CHECK(cudaEventCreate(&w1));
    CUDA_CHECK(cudaEventRecord(w0));
    float wms = 0.0f;
    do {
      if (do_cuda)
        sgemm_regtile<<<grid, block>>>(dA, dB, dC, M, Nn, K);
      else
        CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, Nn, Nn, Nn,
                                 &alpha, dA, Nn, dB, Nn, &beta, dC, Nn));
      CUDA_CHECK(cudaEventRecord(w1));
      CUDA_CHECK(cudaEventSynchronize(w1));
      CUDA_CHECK(cudaEventElapsedTime(&wms, w0, w1));
    } while (wms < 2000.0f);
    CUDA_CHECK(cudaEventDestroy(w0));
    CUDA_CHECK(cudaEventDestroy(w1));
  }

  double cuda_ms = 0.0, blas_ms = 0.0;

  if (do_cuda) {
    CUDA_CHECK(cudaMemset(dC, 0, bytes));
    float total = 0.0f;
    for (int r = 0; r < RUNS; ++r) {
      CUDA_CHECK(cudaEventRecord(ev_start));
      sgemm_regtile<<<grid, block>>>(dA, dB, dC, M, Nn, K);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaEventRecord(ev_stop));
      CUDA_CHECK(cudaEventSynchronize(ev_stop));
      float ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
      total += ms;
    }
    cuda_ms = total / RUNS;
  }

  if (do_blas) {
    float total = 0.0f;
    for (int r = 0; r < RUNS; ++r) {
      CUDA_CHECK(cudaEventRecord(ev_start));
      CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, Nn, Nn, Nn, &alpha,
                               dA, Nn, dB, Nn, &beta, dC, Nn));
      CUDA_CHECK(cudaEventRecord(ev_stop));
      CUDA_CHECK(cudaEventSynchronize(ev_stop));
      float ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
      total += ms;
    }
    blas_ms = total / RUNS;
  }

  // ---- verification of the custom kernel against cuBLAS --------------------
  double maxrel = 0.0;
  if (do_cuda) {
    float *hK = (float *)malloc(bytes), *hR = (float *)malloc(bytes);
    CUDA_CHECK(cudaMemcpy(hK, dC, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hR, dCref, bytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < ne; ++i) {
      double ref = hR[i], got = hK[i];
      double rel = fabs(ref - got) / (fabs(ref) + 1e-12);
      if (rel > maxrel)
        maxrel = rel;
    }
    free(hK);
    free(hR);
  }

  // ---- report (single result per invocation, standard format) --------------
  if (do_cuda) {
    printf("Custom register-tiled SGEMM N=%d (TM=TN=%d, BM=BN=%d, BK=%d)\n", N,
           TM, BM, BK);
    printf("Average kernel time: %.3f ms\n", cuda_ms);
    printf("Effective GFLOPS: %.1f\n", flop / (cuda_ms * 1e6));
    printf("Max relative error vs cuBLAS: %.3e\n", maxrel);
    printf("Result: %s\n", maxrel < 1e-4 ? "OK" : "MISMATCH");
  }
  if (do_blas) {
    printf("cuBLAS SGEMM N=%d\n", N);
    printf("Average kernel time: %.3f ms\n", blas_ms);
    printf("Effective GFLOPS: %.1f\n", flop / (blas_ms * 1e6));
  }

  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  cublasDestroy(h);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dCref);
  free(hA);
  free(hB);
  free(hC);
  return 0;
}
