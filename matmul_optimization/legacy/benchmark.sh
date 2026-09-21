#!/bin/bash
# Example: ./benchmark_flags.sh

source /home/onur/intel/oneapi/setvars.sh 

SRC=matmul_c11_intel_mkl.c
EXE=matmul_c11_intel_mkl
THREADS=8

# Array of compiler flag combinations
FLAGS=(
  "-O3 -xHost -qopenmp -fp-model fast=2"
  "-O3 -xHost -qopenmp -fp-model fast=2 -ipo"
  "-O3 -xHost -qopenmp -fp-model fast=2 -ipo -qopt-report=5"
  "-O3 -march=native -qopenmp -fp-model fast=2"
)

for f in "${FLAGS[@]}"; do
    echo "-----------------------------------------"
    echo "Compiling with flags: $f"
    icx $f $SRC \
        -I${MKLROOT}/include \
        -L${MKLROOT}/lib/intel64 \
        -Wl,--start-group -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -Wl,--end-group \
        -lpthread -lm -ldl -o $EXE

    echo "Running benchmark with $THREADS threads..."
    export MKL_NUM_THREADS=$THREADS
    export OMP_NUM_THREADS=$THREADS
    export KMP_AFFINITY=granularity=fine,compact,1,0

    ./$EXE $THREADS
done
