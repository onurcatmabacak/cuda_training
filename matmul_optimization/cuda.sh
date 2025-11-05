#!/bin/bash
clear

# nvcc -arch=sm_50 -o whoami whoami_cuda.cu && ./whoami > whoami.txt
# nvcc -arch=sm_50 -o vector_add_v1 vector_add_v1.cu && ./vector_add_v1 > vector_add_v1.txt
# nvcc -arch=sm_50 -o vector_add_v2 vector_add_v2.cu && ./vector_add_v2 > vector_add_v2.txt
# nvcc -arch=sm_50 -o matmul_cuda matmul_cuda.cu && ./matmul_cuda > matmul_cuda.txt
nvcc -arch=sm_50 -o matmul_cuda_faster matmul_cuda_faster.cu && ./matmul_cuda_faster > matmul_cuda_faster.txt
