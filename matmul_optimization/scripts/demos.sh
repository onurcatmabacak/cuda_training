#!/usr/bin/env bash
# demos.sh -- small CUDA teaching programs (not matrix-multiply benchmarks,
# but part of the original folder). Kept so one run reproduces everything.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
ensure_dirs

MM_CAT="demos"

if ! have nvcc; then
  warn "nvcc not found -- skipping [$MM_CAT]"
  exit 0
fi
if ! have_gpu; then
  warn "no NVIDIA GPU detected -- skipping [$MM_CAT]"
  exit 0
fi

MM_NVCC="${MATMUL_NVCC:-nvcc}"
MM_ARCH="${MATMUL_CUDA_ARCH:-sm_50}"

name="${MM_CAT}__whoami"
bench_build_run "$name" "$MM_CAT" ::: \
  "$MM_NVCC" -O3 -arch="$MM_ARCH" "$MM_SRC/demos/whoami_cuda.cu" -o "$MM_BIN/whoami" ::: \
  "$MM_BIN/whoami"

name="${MM_CAT}__vector_add_v1"
bench_build_run "$name" "$MM_CAT" ::: \
  "$MM_NVCC" -O3 -arch="$MM_ARCH" "$MM_SRC/demos/vector_add_v1.cu" -o "$MM_BIN/vector_add_v1" ::: \
  "$MM_BIN/vector_add_v1"

name="${MM_CAT}__vector_add_v2"
bench_build_run "$name" "$MM_CAT" ::: \
  "$MM_NVCC" -O3 -arch="$MM_ARCH" "$MM_SRC/demos/vector_add_v2.cu" -o "$MM_BIN/vector_add_v2" ::: \
  "$MM_BIN/vector_add_v2"
