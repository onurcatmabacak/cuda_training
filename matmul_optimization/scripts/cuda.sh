#!/usr/bin/env bash
# cuda.sh -- CUDA / cuBLAS GPU benchmarks (nvcc).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
ensure_dirs

MM_CAT="cuda"

if ! have nvcc; then
  warn "nvcc not found -- skipping [$MM_CAT]"
  exit 0
fi
if ! have_gpu; then
  warn "no NVIDIA GPU detected (nvidia-smi) -- skipping [$MM_CAT]"
  exit 0
fi

MM_NVCC="${MATMUL_NVCC:-nvcc}"
MM_ARCH="${MATMUL_CUDA_ARCH:-sm_50}"

# plain N/RUNS flags (naive kernel and the C tiled kernels)
SIZE=()
[[ -n "$MATMUL_N" ]] && SIZE+=(-DN="$MATMUL_N")
[[ -n "$MATMUL_RUNS" ]] && SIZE+=(-DRUNS="$MATMUL_RUNS")

# the naive kernel also runs a slow single-threaded CPU reference, so cap its
# timed-run count independently of the fast GPU benchmarks
MM_NAIVE_RUNS=""
if [[ -n "$MATMUL_RUNS" ]]; then
  MM_NAIVE_RUNS="$MATMUL_RUNS"
  (( MM_NAIVE_RUNS > 20 )) && MM_NAIVE_RUNS=20
fi
NAIVE_SIZE=()
[[ -n "$MATMUL_N" ]] && NAIVE_SIZE+=(-DM="$MATMUL_N" -DK="$MATMUL_N" -DN="$MATMUL_N")
[[ -n "$MM_NAIVE_RUNS" ]] && NAIVE_SIZE+=(-DRUNS="$MM_NAIVE_RUNS")

# MATMUL_N/MATMUL_RUNS flags for the C++/CUDA sources that declare constexpr N
CPP_SIZE=()
[[ -n "$MATMUL_N" ]] && CPP_SIZE+=(-DMATMUL_N="$MATMUL_N")
[[ -n "$MATMUL_RUNS" ]] && CPP_SIZE+=(-DMATMUL_RUNS="$MATMUL_RUNS")

# 1) Naive one-thread-per-output kernel vs. CPU reference.
name="${MM_CAT}__matmul_cuda"
bench_build_run "$name" "$MM_CAT" ::: \
  "$MM_NVCC" -O3 -arch="$MM_ARCH" "${NAIVE_SIZE[@]}" -DWARMUP=1 \
    "$MM_SRC/cuda/matmul_cuda.cu" -o "$MM_BIN/matmul_cuda" ::: \
  "$MM_BIN/matmul_cuda"

# 2) Shared-memory tiled kernel (C).
name="${MM_CAT}__matmul_cuda_faster"
bench_build_run "$name" "$MM_CAT" ::: \
  "$MM_NVCC" -O3 -arch="$MM_ARCH" -Xcompiler -fopenmp "${SIZE[@]}" \
    "$MM_SRC/cuda/matmul_cuda_faster.cu" -o "$MM_BIN/matmul_cuda_faster" ::: \
  "$MM_BIN/matmul_cuda_faster"

# 3) Shared-memory tiled kernel (C++20 host code).
name="${MM_CAT}__matmul_cuda_cpp20_faster"
bench_build_run "$name" "$MM_CAT" ::: \
  "$MM_NVCC" -O3 -arch="$MM_ARCH" -std=c++20 -Xcompiler -fopenmp "${CPP_SIZE[@]}" \
    "$MM_SRC/cuda/matmul_cuda_cpp20_faster.cu" -o "$MM_BIN/matmul_cuda_cpp20_faster" ::: \
  "$MM_BIN/matmul_cuda_cpp20_faster"

# 4) cuBLAS DGEMM (C11 host code, Float64).
name="${MM_CAT}__matmul_cublas_c11"
bench_build_run "$name" "$MM_CAT" ::: \
  "$MM_NVCC" -O3 -arch="$MM_ARCH" "${SIZE[@]}" -lcublas \
    "$MM_SRC/cuda/matmul_cublas_c11.cu" -o "$MM_BIN/matmul_cublas_c11" ::: \
  "$MM_BIN/matmul_cublas_c11"

# 5) cuBLAS DGEMM (C++20 host code, Float64).
name="${MM_CAT}__matmul_cublas_cpp20"
bench_build_run "$name" "$MM_CAT" ::: \
  "$MM_NVCC" -O3 -arch="$MM_ARCH" -std=c++20 "${CPP_SIZE[@]}" -lcublas \
    "$MM_SRC/cuda/matmul_cublas_cpp20.cu" -o "$MM_BIN/matmul_cublas_cpp20" ::: \
  "$MM_BIN/matmul_cublas_cpp20"

# 6) Best Float32 SGEMM: hand-written register-tiled kernel vs cuBLAS.
#    One binary, two timed backends (selected by argv[1]).
BEST_BIN="$MM_BIN/matmul_cuda_best"
if build "${MM_CAT}__matmul_cuda_best" \
    "$MM_NVCC" -O3 -arch="$MM_ARCH" "${SIZE[@]}" -lcublas \
      "$MM_SRC/cuda/matmul_cuda_best.cu" -o "$BEST_BIN"; then
  run_bench "${MM_CAT}__matmul_cuda_optimized" "$MM_CAT" "$BEST_BIN" cuda
  run_bench "${MM_CAT}__matmul_cublas_sgemm"   "$MM_CAT" "$BEST_BIN" cublas
else
  manifest_add "${MM_CAT}__matmul_cuda_optimized" "$MM_CAT" "BUILD_FAIL" "1" "0" \
    "$MM_BUILD/${MM_CAT}__matmul_cuda_best.build.log" "build failed"
  manifest_add "${MM_CAT}__matmul_cublas_sgemm" "$MM_CAT" "BUILD_FAIL" "1" "0" \
    "$MM_BUILD/${MM_CAT}__matmul_cuda_best.build.log" "build failed"
fi
