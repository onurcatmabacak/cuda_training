#!/bin/bash
clear

NAME=matmul_cublas_c11
nvcc -O3 -arch=sm_50 -lcublas -o $NAME $NAME.cu && ./$NAME > $NAME.txt
