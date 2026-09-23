// matmul_cpp26_cuda.cpp
// C++26 host code driving Float64 matrix multiplication on the GPU.
// Because nvcc 12.0 caps at C++20, the host is compiled by g++-14 -std=c++26
// and links the CUDA runtime + cuBLAS libraries directly (no device code here).
//
// Modes (argv[1]):
//   cublas    -> cublasDgemm
//   cublaslt  -> cublasLtMatmul
//   graph     -> cublasDgemm captured in a CUDA graph and replayed
//
// All modes use N x N Float64, RUNS timed iterations, 2 s boost warm-up.

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <print>
#include <string>
#include <vector>

#ifndef N
#define N 1024
#endif
#ifndef RUNS
#define RUNS 10
#endif

#define CK(x)                                                                    \
  do {                                                                           \
    cudaError_t e = (x);                                                         \
    if (e != cudaSuccess) {                                                      \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,         \
                   cudaGetErrorString(e));                                       \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

#define CB(x)                                                                    \
  do {                                                                           \
    cublasStatus_t s = (x);                                                      \
    if (s != CUBLAS_STATUS_SUCCESS) {                                            \
      std::fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__,       \
                   static_cast<int>(s));                                         \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

int main(int argc, char **argv) {
  const std::string mode = (argc > 1) ? argv[1] : "cublas";
  const std::size_t ne = static_cast<std::size_t>(N) * N;
  const std::size_t bytes = ne * sizeof(double);
  const double alpha = 1.0, beta = 0.0;
  const double flop = 2.0 * static_cast<double>(N) * N * N;

  std::vector<double> hA(ne), hB(ne);
  for (std::size_t i = 0; i < ne; ++i) {
    hA[i] = static_cast<double>((i % 17) + 1) * 1e-3 + 1.0;
    hB[i] = static_cast<double>((i % 13) + 1) * 1e-3 + 2.0;
  }

  double *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CK(cudaMalloc(&dA, bytes));
  CK(cudaMalloc(&dB, bytes));
  CK(cudaMalloc(&dC, bytes));
  CK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CK(cudaStreamCreate(&stream));

  cublasHandle_t h;
  CB(cublasCreate(&h));
  CB(cublasSetStream(h, stream));

  // cuBLASLt objects (only used in cublaslt mode)
  cublasLtHandle_t lt = nullptr;
  cublasLtMatmulDesc_t opDesc = nullptr;
  cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
  if (mode == "cublaslt") {
    CB(cublasLtCreate(&lt));
    CB(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_64F, CUDA_R_64F));
    cublasOperation_t opN = CUBLAS_OP_N;
    CB(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                      &opN, sizeof(opN)));
    CB(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                      &opN, sizeof(opN)));
    CB(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_64F, N, N, N));
    CB(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_64F, N, N, N));
    CB(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_64F, N, N, N));
  }

  // CUDA graph (only used in graph mode)
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graphExec = nullptr;
  if (mode == "graph") {
    CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    CB(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dA, N, dB, N,
                   &beta, dC, N));
    CK(cudaStreamEndCapture(stream, &graph));
    CK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  }

  auto launch = [&]() {
    if (mode == "cublaslt") {
      CB(cublasLtMatmul(lt, opDesc, &alpha, dA, Adesc, dB, Bdesc, &beta, dC,
                        Cdesc, dC, Cdesc, nullptr, nullptr, 0, stream));
    } else if (mode == "graph") {
      CK(cudaGraphLaunch(graphExec, stream));
    } else {
      CB(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dA, N, dB, N,
                     &beta, dC, N));
    }
  };

  // Warm up until the GPU boosts its clocks (~2 s).
  {
    cudaEvent_t w0, w1;
    CK(cudaEventCreate(&w0));
    CK(cudaEventCreate(&w1));
    CK(cudaEventRecord(w0, stream));
    float wms = 0.0f;
    do {
      launch();
      CK(cudaEventRecord(w1, stream));
      CK(cudaEventSynchronize(w1));
      CK(cudaEventElapsedTime(&wms, w0, w1));
    } while (wms < 2000.0f);
    CK(cudaEventDestroy(w0));
    CK(cudaEventDestroy(w1));
  }

  cudaEvent_t start, stop;
  CK(cudaEventCreate(&start));
  CK(cudaEventCreate(&stop));

  float total = 0.0f;
  for (int r = 0; r < RUNS; ++r) {
    CK(cudaEventRecord(start, stream));
    launch();
    CK(cudaEventRecord(stop, stream));
    CK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CK(cudaEventElapsedTime(&ms, start, stop));
    total += ms;
  }
  const double avg_ms = total / RUNS;

  const char *label = (mode == "cublaslt") ? "cuBLASLt DGEMM"
                      : (mode == "graph")  ? "CUDA-graph cublasDgemm"
                                           : "cuBLAS DGEMM";
  std::println("C++26 + CUDA {} N={}", label, N);
  std::printf("Average kernel time: %.3f ms\n", avg_ms);
  std::printf("Effective GFLOPS: %.1f\n", flop / (avg_ms * 1e6));

  if (mode == "graph") {
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
  }
  if (mode == "cublaslt") {
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(opDesc);
    cublasLtDestroy(lt);
  }
  cublasDestroy(h);
  cudaStreamDestroy(stream);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  return 0;
}
