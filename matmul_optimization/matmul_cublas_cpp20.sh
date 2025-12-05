#!/bin/bash
clear

NAME=matmul_cublas_cpp20
nvcc -O3 -arch=sm_50 -lcublas -o $NAME $NAME.cu && ./$NAME > $NAME.txt
