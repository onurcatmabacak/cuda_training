#!/bin/bash
NAME=matmul_cpp20_intel_mkl

set -euo pipefail

# Load Intel oneAPI environment (adjust path if needed)
# --- disable nounset before sourcing ---
set +u
source /home/onur/intel/oneapi/setvars.sh
set -u
# --- re-enable nounset if you want ---

# Tunable settings
NUM_THREADS=8   # choose number of threads (physical cores recommended)
export MKL_NUM_THREADS=${NUM_THREADS}
export OMP_NUM_THREADS=${NUM_THREADS}
export MKL_DYNAMIC=FALSE

# Affinity: compact placement is often best for GEMM
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export KMP_AFFINITY=granularity=fine,compact
# Optionally disable SMT (1T) if you want one thread per physical core
# export KMP_HW_SUBSET=1T

# Prefer huge pages (requires sudo). If not available, the program still runs.
export MKL_ENABLE_HUGEPAGES=1

# Try to set MKL instruction set (optional)
export MKL_ENABLE_INSTRUCTIONS=AVX512

echo "Compiling with icx..."
icpx -O3 -march=native -ffast-math -funroll-loops -fprefetch-loop-arrays -qopenmp -std=c++20 \
      $NAME.cpp \
      -I${MKLROOT}/include \
      -L${MKLROOT}/lib/intel64 \
      -Wl,--start-group -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -Wl,--end-group \
      -lpthread -lm -ldl \
      -o $NAME

echo "Running with ${NUM_THREADS} threads..."
./$NAME ${NUM_THREADS} | tee $NAME.txt

echo "Done. Output saved to ${NAME}.txt"

