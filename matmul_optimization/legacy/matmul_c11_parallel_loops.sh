#!/bin/bash
clear

gcc -fopenmp -o1 -march=native -funroll-loops -ftree-vectorize -ffast-math matmul_c11_parallel_loops.c -o matmul_c11_parallel_loops && ./matmul_c11_parallel_loops
gcc -fopenmp -o2 -march=native -funroll-loops -ftree-vectorize -ffast-math matmul_c11_parallel_loops.c -o matmul_c11_parallel_loops && ./matmul_c11_parallel_loops
gcc -fopenmp -o3 -march=native -funroll-loops -ftree-vectorize -ffast-math matmul_c11_parallel_loops.c -o matmul_c11_parallel_loops && ./matmul_c11_parallel_loops
gcc -std=c11 -fopenmp -o1 -march=native -funroll-loops -ftree-vectorize -ffast-math matmul_c11_parallel_loops.c -o matmul_c11_parallel_loops && ./matmul_c11_parallel_loops
gcc -std=c11 -fopenmp -o2 -march=native -funroll-loops -ftree-vectorize -ffast-math matmul_c11_parallel_loops.c -o matmul_c11_parallel_loops && ./matmul_c11_parallel_loops
gcc -std=c11 -fopenmp -o3 -march=native -funroll-loops -ftree-vectorize -ffast-math matmul_c11_parallel_loops.c -o matmul_c11_parallel_loops && ./matmul_c11_parallel_loops
gcc -std=c2x -fopenmp -o1 -march=native -funroll-loops -ftree-vectorize -ffast-math matmul_c11_parallel_loops.c -o matmul_c11_parallel_loops && ./matmul_c11_parallel_loops
gcc -std=c2x -fopenmp -o2 -march=native -funroll-loops -ftree-vectorize -ffast-math matmul_c11_parallel_loops.c -o matmul_c11_parallel_loops && ./matmul_c11_parallel_loops
gcc -std=c2x -fopenmp -o3 -march=native -funroll-loops -ftree-vectorize -ffast-math matmul_c11_parallel_loops.c -o matmul_c11_parallel_loops && ./matmul_c11_parallel_loops
