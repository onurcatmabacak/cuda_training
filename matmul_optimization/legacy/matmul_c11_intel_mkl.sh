#!/bin/bash
clear

set -e
source ~/intel/oneapi/setvars.sh --force || true

NAME=matmul_c11_intel_mkl
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

icx $NAME.c \
    -std=c11 -O3 -march=native -ffast-math \
    -DMKL_ILP64 \
    -fiopenmp \
    -I$MKLROOT/include \
    -L$MKLROOT/lib/intel64 \
    -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core \
    -liomp5 -lpthread -lm -ldl \
    -o $NAME

./$NAME > $NAME.txt
