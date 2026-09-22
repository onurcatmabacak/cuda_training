// Compile: nvcc -O3 -arch=sm_50 -std=c11 -o matmul_tiled matmul_tiled.cu
// Run: ./matmul_tiled

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#ifndef N
#define N 4096                     // matrix dimension (N x N)
#endif
#define TILE 32                    // tile size (must divide N)
#ifndef RUNS
#define RUNS 100                   // timed runs
#endif
#define BLOCK_DIM_X TILE
#define BLOCK_DIM_Y TILE

static inline void checkCuda(cudaError_t e, const char *msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA Error %s: %s\n", msg, cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }
}

// Tiled shared-memory matrix multiplication kernel
// C = A * B
// Each block computes a TILE x TILE submatrix of C.
__global__ void matmul_tiled(const float *__restrict__ A,
                             const float *__restrict__ B,
                             float *__restrict__ C,
                             int n) {
    // Shared tiles
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    // Thread indices inside block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Global row and column this thread computes
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    // Accumulator in register
    float acc = 0.0f;

    // Loop over tiles of A and B
    // Each iteration loads a TILExTILE sub-block of A and B into shared memory
    for (int m = 0; m < n; m += TILE) {
        // Load A tile: row, (m + tx)
        sA[ty][tx] = A[row * n + (m + tx)];
        // Load B tile: (m + ty), col
        sB[ty][tx] = B[(m + ty) * n + col];

        __syncthreads();

        // Compute partial products for this tile
        // Unroll the inner k-loop for performance
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    // Write result
    C[row * n + col] = acc;
}

// CPU reference for verification. Cache-friendly (i,k,j) order and parallelised
// with OpenMP: at N=4096 the naive single-threaded version takes ~25 minutes.
// NOTE: C must be zero-initialised before calling.
void matmul_cpu(const float *A, const float *B, float *C, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        float *Ci = C + (size_t)i * n;
        for (int k = 0; k < n; ++k) {
            const float a = A[(size_t)i * n + k];
            const float *Bk = B + (size_t)k * n;
            for (int j = 0; j < n; ++j)
                Ci[j] += a * Bk[j];
        }
    }
}

int main(void) {
    const int bytes = N * N * sizeof(float);

    // Allocate pinned (page-locked) host memory for faster H2D/D2H transfers
    float *h_A = NULL, *h_B = NULL, *h_C = NULL, *h_C_ref = NULL;
    checkCuda(cudaMallocHost((void**)&h_A, bytes), "cudaMallocHost A");
    checkCuda(cudaMallocHost((void**)&h_B, bytes), "cudaMallocHost B");
    checkCuda(cudaMallocHost((void**)&h_C, bytes), "cudaMallocHost C");
    checkCuda(cudaMallocHost((void**)&h_C_ref, bytes), "cudaMallocHost C_ref");

    // Initialize matrices (example: random or deterministic)
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (float)( (i % 17) + 1 ) * 1e-3f + 1.0f; // small deterministic values
        h_B[i] = (float)( (i % 13) + 1 ) * 1e-3f + 2.0f;
        h_C[i] = 0.0f;
        h_C_ref[i] = 0.0f;
    }

    // Device allocations
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    checkCuda(cudaMalloc((void**)&d_A, bytes), "cudaMalloc d_A");
    checkCuda(cudaMalloc((void**)&d_B, bytes), "cudaMalloc d_B");
    checkCuda(cudaMalloc((void**)&d_C, bytes), "cudaMalloc d_C");

    // Copy H->D (synchronous is fine for single-stream)
    checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D A");
    checkCuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D B");

    // Launch kernel
    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridDim(N / TILE, N / TILE);

    // Create events for timing
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    // Warm up until the GPU reaches its boost clock. Short kernels timed right
    // after startup otherwise run at the idle clock (P8 ~135 MHz vs boosted
    // P0 ~1200 MHz on this laptop), reading ~8x too slow.
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
            checkCuda(cudaEventElapsedTime(&warm_ms, warm_start, warm_stop), "warmup elapsed");
        } while (warm_ms < 2000.0f);
        checkCuda(cudaEventDestroy(warm_start), "warmup destroy");
        checkCuda(cudaEventDestroy(warm_stop), "warmup destroy");
    }

    // Timed runs
    float ms_total = 0.0f;
    for (int r = 0; r < RUNS; ++r) {
        checkCuda(cudaEventRecord(start), "cudaEventRecord start");
        matmul_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        checkCuda(cudaGetLastError(), "kernel launch");
        checkCuda(cudaEventRecord(stop), "cudaEventRecord stop");
        checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
        float ms = 0.0f;
        checkCuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
        ms_total += ms;
    }
    float ms_avg = ms_total / RUNS;

    // Copy result back
    checkCuda(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H C");

    // Compute reference on CPU (might take time)
    printf("Computing CPU reference (parallel i-k-j)...\n");
    matmul_cpu(h_A, h_B, h_C_ref, N);

    // Verify results (relative error)
    double maxRelErr = 0.0;
    double eps = 1e-4;
    for (int i = 0; i < N * N; ++i) {
        double ref = (double)h_C_ref[i];
        double got = (double)h_C[i];
        double rel = fabs(ref - got) / (fabs(ref) + 1e-12);
        if (rel > maxRelErr) maxRelErr = rel;
        if (rel > eps) {
            fprintf(stderr, "Result mismatch at index %d: ref=%f got=%f rel=%e\n", i, (float)ref, (float)got, rel);
            break;
        }
    }

    // Report
    double flops = 2.0 * (double)N * (double)N * (double)N; // 2*N^3 FLOPs
    double gflops = (flops / 1e9) / (ms_avg / 1000.0);
    printf("Matrix mul %dx%d using GPU tiled kernel\n", N, N);
    printf("Average kernel time over %d runs: %.3f ms\n", RUNS, ms_avg);
    printf("Effective GFLOPS: %.2f\n", gflops);
    printf("Max relative error vs CPU: %.3e\n", maxRelErr);
    if (maxRelErr <= eps) {
        printf("Result: OK\n");
    } else {
        printf("Result: MISMATCH\n");
    }

    // Cleanup
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
