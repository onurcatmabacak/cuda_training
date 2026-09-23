#!/usr/bin/env bash
# cpp26.sh -- C++26 host code + CUDA (cuBLAS / cuBLASLt / CUDA graph).
#
# nvcc 12.0 caps at C++20, so C++26 needs a newer host compiler.  This uses a
# user-local g++-14 (see README); override with CXX26 / CXX26_LIBDIR.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
ensure_dirs

MM_CAT="cpp26"
CXX26="${CXX26:-$HOME/opt/cxx26/usr/bin/g++-14}"
CXX26_LIBDIR="${CXX26_LIBDIR:-$HOME/opt/cxx26/usr/lib/x86_64-linux-gnu}"

if [[ ! -x "$CXX26" ]]; then
  warn "g++-14 with C++26 support not found at $CXX26 -- skipping [$MM_CAT]"
  record_skip "${MM_CAT}__matmul_cpp26_cublas" "$MM_CAT" "no C++26 compiler"
  exit 0
fi

SIZE=()
[[ -n "${MATMUL_N:-}" ]] && SIZE+=(-DN="$MATMUL_N")
[[ -n "${MATMUL_RUNS:-}" ]] && SIZE+=(-DRUNS="$MATMUL_RUNS")

CPP26_BIN="$MM_BIN/matmul_cpp26_cuda"
if build "${MM_CAT}__matmul_cpp26_build" \
    "$CXX26" -std=c++26 -O3 "${SIZE[@]}" \
      "$MM_SRC/cpp26/matmul_cpp26_cuda.cpp" -o "$CPP26_BIN" \
      -lcudart -lcublas -lcublasLt; then
  RUN_ENV=(env "LD_LIBRARY_PATH=$CXX26_LIBDIR:${LD_LIBRARY_PATH:-}")
  run_bench "${MM_CAT}__matmul_cpp26_cublas"   "$MM_CAT" "${RUN_ENV[@]}" "$CPP26_BIN" cublas
  run_bench "${MM_CAT}__matmul_cpp26_cublaslt" "$MM_CAT" "${RUN_ENV[@]}" "$CPP26_BIN" cublaslt
  run_bench "${MM_CAT}__matmul_cpp26_graph"    "$MM_CAT" "${RUN_ENV[@]}" "$CPP26_BIN" graph
else
  manifest_add "${MM_CAT}__matmul_cpp26_cublas" "$MM_CAT" "BUILD_FAIL" "1" "0" \
    "$MM_BUILD/${MM_CAT}__matmul_cpp26_build.build.log" "build failed"
fi
