#!/bin/bash
clear

export MKL_NUM_THREADS=8      # Threads used by MKL
export OMP_NUM_THREADS=8      # Threads used by OpenMP (if used outside MKL)
export KMP_AFFINITY=granularity=fine,compact,1,0   # Bind threads to cores for cache efficiency
# export KMP_AFFINITY=compact,granularity=fine

source /home/onur/intel/oneapi/setvars.sh 

icx -O3 -xHost -qopenmp -fp-model fast=2 -ipo matmul_c11_intel_mkl.c \
    -I${MKLROOT}/include \
    -L${MKLROOT}/lib/intel64 \
    -Wl,--start-group -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -Wl,--end-group \
    -lpthread -lm -ldl \
    -o matmul_c11_intel_mkl

./matmul_c11_intel_mkl
