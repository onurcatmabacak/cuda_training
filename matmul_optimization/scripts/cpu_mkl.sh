#!/usr/bin/env bash
# cpu_mkl.sh -- Intel oneAPI / MKL CPU benchmarks (icx + icpx).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
ensure_dirs

MM_CAT="cpu_mkl"

if ! source_oneapi; then
  warn "Intel oneAPI (icx/icpx) not available -- skipping [$MM_CAT]"
  exit 0
fi

MKL_INC=(-I"$MKLROOT/include")
MKL_LIB=(-L"$MKLROOT/lib/intel64"
         -Wl,--start-group -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -Wl,--end-group
         -liomp5 -lpthread -lm -ldl)
MKL_ENV=(env MKL_NUM_THREADS="$MATMUL_THREADS" OMP_NUM_THREADS="$MATMUL_THREADS" MKL_DYNAMIC=FALSE)

SIZE=()
[[ -n "$MATMUL_N" ]] && SIZE+=(-DMATMUL_N="$MATMUL_N")
[[ -n "$MATMUL_RUNS" ]] && SIZE+=(-DMATMUL_RUNS="$MATMUL_RUNS")

# 1) C11 DGEMM: the four icx flag sets from the original benchmark.sh.
i=0
for flags in \
    "-O3 -xHost -qopenmp -fp-model fast=2" \
    "-O3 -xHost -qopenmp -fp-model fast=2 -ipo" \
    "-O3 -xHost -qopenmp -fp-model fast=2 -ipo -qopt-report=5" \
    "-O3 -march=native -qopenmp -fp-model fast=2"
do
  i=$((i + 1))
  name="${MM_CAT}__matmul_c11_intel_mkl_flags${i}"
  # $flags is intentionally word-split into separate compiler arguments.
  # shellcheck disable=SC2086
  bench_build_run "$name" "$MM_CAT" ::: \
    icx $flags -std=c11 "${SIZE[@]}" \
      "$MM_SRC/cpu_mkl/matmul_c11_intel_mkl.c" "${MKL_INC[@]}" "${MKL_LIB[@]}" \
      -o "$MM_BIN/matmul_c11_intel_mkl_flags${i}" ::: \
    "${MKL_ENV[@]}" "$MM_BIN/matmul_c11_intel_mkl_flags${i}"
done

# 2) C++20 MKL with NUMA-first-touch initialisation.
name="${MM_CAT}__matmul_cpp20_intel_mkl"
bench_build_run "$name" "$MM_CAT" ::: \
  icpx -O3 -march=native -ffast-math -funroll-loops -fprefetch-loop-arrays -qopenmp -std=c++20 \
    "${SIZE[@]}" \
    "$MM_SRC/cpu_mkl/matmul_cpp20_intel_mkl.cpp" "${MKL_INC[@]}" "${MKL_LIB[@]}" \
    -o "$MM_BIN/matmul_cpp20_intel_mkl" ::: \
  "${MKL_ENV[@]}" "$MM_BIN/matmul_cpp20_intel_mkl" "$MATMUL_THREADS"
