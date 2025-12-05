#!/bin/bash
clear 

NAME=matmul_julia_cpu

# Run Julia CPU-only matrix multiplication and redirect output
julia $NAME.jl > $NAME.txt