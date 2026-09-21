#!/bin/bash
clear

# Single point name for source, binary, and output
NAME=matmul_cuda_cpp20_faster

# Compile with CUDA for Maxwell (sm_50), C++20, max optimization
nvcc -O3 -arch=sm_50 -std=c++20 $NAME.cu -o $NAME

# Run and redirect output
./$NAME > $NAME.txt