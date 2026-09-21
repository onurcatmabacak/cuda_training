#!/bin/bash
clear

gcc -o1 matmul_c99.c -o matmul_c99 && ./matmul_c99
gcc -o2 matmul_c99.c -o matmul_c99 && ./matmul_c99
gcc -o3 matmul_c99.c -o matmul_c99 && ./matmul_c99
gcc -std=c11 -o1 matmul_c11.c -o matmul_c11 && ./matmul_c11
gcc -std=c11 -o2 matmul_c11.c -o matmul_c11 && ./matmul_c11
gcc -std=c11 -o3 matmul_c11.c -o matmul_c11 && ./matmul_c11
gcc -std=c2x -o1 matmul_c23.c -o matmul_c23 && ./matmul_c23
gcc -std=c2x -o2 matmul_c23.c -o matmul_c23 && ./matmul_c23
gcc -std=c2x -o3 matmul_c23.c -o matmul_c23 && ./matmul_c23