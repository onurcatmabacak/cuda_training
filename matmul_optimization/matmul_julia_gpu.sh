#!/bin/bash
clear 

NAME=matmul_julia_gpu

# Run Julia GPU (CUDA) matrix multiplication and redirect output
julia $NAME.jl > $NAME.txt