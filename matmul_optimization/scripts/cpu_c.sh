#!/usr/bin/env bash
# cpu_c.sh -- GCC-based CPU benchmarks: pure C triple loop + OpenBLAS.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
ensure_dirs

MM_CAT="cpu_c"

if ! have gcc; then
  warn "gcc not found -- skipping [$MM_CAT]"
  exit 0
fi

MM_CC="${MATMUL_CC:-gcc}"

# Only override the source defaults when --size/--runs/--quick was given.
GEN=(-march=native -ffast-math)
[[ -n "$MATMUL_N" ]] && GEN+=(-DN="$MATMUL_N")
[[ -n "$MATMUL_RUNS" ]] && GEN+=(-DRUNS="$MATMUL_RUNS")

# 1) Pure C triple loop: language standard x optimisation level sweep.
#    (The original matmul.sh used `-o1/-o2/-o3` by mistake, so those runs were
#     in fact unoptimised; here they are real -O levels.)
for spec in "matmul_c99|-std=c99" "matmul_c11|-std=c11" "matmul_c23|-std=c2x"; do
  src="${spec%%|*}"
  std="${spec##*|}"
  for opt in O1 O2 O3; do
    name="${MM_CAT}__${src}_${opt}"
    bench_build_run "$name" "$MM_CAT" ::: \
      "$MM_CC" "$std" "-$opt" "${GEN[@]}" -o "$MM_BIN/${src}_${opt}" "$MM_SRC/cpu_c/${src}.c" ::: \
      "$MM_BIN/${src}_${opt}"
  done
done

# 2) C11 with the cache-friendly (i,k,j) index order.
name="${MM_CAT}__matmul_c11_index_order_O3"
bench_build_run "$name" "$MM_CAT" ::: \
  "$MM_CC" -std=c11 -O3 "${GEN[@]}" -o "$MM_BIN/matmul_c11_index_order" "$MM_SRC/cpu_c/matmul_c11_index_order.c" ::: \
  "$MM_BIN/matmul_c11_index_order"

# 3) OpenMP blocked + recursive divide & conquer.
name="${MM_CAT}__matmul_c11_parallel_loops_O3"
bench_build_run "$name" "$MM_CAT" ::: \
  "$MM_CC" -std=c11 -O3 -fopenmp -funroll-loops -ftree-vectorize "${GEN[@]}" \
    -o "$MM_BIN/matmul_c11_parallel_loops" "$MM_SRC/cpu_c/matmul_c11_parallel_loops.c" ::: \
  env OMP_NUM_THREADS="$MATMUL_THREADS" "$MM_BIN/matmul_c11_parallel_loops"

# 4) OpenBLAS DGEMM.  NB: cblas.h names a parameter `N`, so the size is passed
#    as MATMUL_N and mapped to N inside the source after the include.
BLAS_FLAGS=(-std=c11 -O3 -march=native -ffast-math)
[[ -n "$MATMUL_N" ]] && BLAS_FLAGS+=(-DMATMUL_N="$MATMUL_N")
[[ -n "$MATMUL_RUNS" ]] && BLAS_FLAGS+=(-DMATMUL_RUNS="$MATMUL_RUNS")
name="${MM_CAT}__matmul_c11_openblas_O3"
bench_build_run "$name" "$MM_CAT" ::: \
  "$MM_CC" "${BLAS_FLAGS[@]}" \
    -o "$MM_BIN/matmul_c11_openblas" "$MM_SRC/cpu_c/matmul_c11_openblas.c" -lopenblas -lm ::: \
  env OPENBLAS_NUM_THREADS="$MATMUL_THREADS" "$MM_BIN/matmul_c11_openblas"
