#!/bin/bash
clear

export OMP_NUM_THREADS=8
gcc -o1 -march=native -ffast-math -Wall -o matmul_c11_openblas matmul_c11_openblas.c -lopenblas -lm && ./matmul_c11_openblas
gcc -o2 -march=native -ffast-math -Wall -o matmul_c11_openblas matmul_c11_openblas.c -lopenblas -lm && ./matmul_c11_openblas
gcc -o3 -march=native -ffast-math -Wall -o matmul_c11_openblas matmul_c11_openblas.c -lopenblas -lm && ./matmul_c11_openblas
gcc -std=c11 -o1 -march=native -ffast-math -Wall -o matmul_c11_openblas matmul_c11_openblas.c -lopenblas -lm && ./matmul_c11_openblas
gcc -std=c11 -o2 -march=native -ffast-math -Wall -o matmul_c11_openblas matmul_c11_openblas.c -lopenblas -lm && ./matmul_c11_openblas
gcc -std=c11 -o3 -march=native -ffast-math -Wall -o matmul_c11_openblas matmul_c11_openblas.c -lopenblas -lm && ./matmul_c11_openblas
gcc -std=c2x -o1 -march=native -ffast-math -Wall -o matmul_c11_openblas matmul_c11_openblas.c -lopenblas -lm && ./matmul_c11_openblas
gcc -std=c2x -o2 -march=native -ffast-math -Wall -o matmul_c11_openblas matmul_c11_openblas.c -lopenblas -lm && ./matmul_c11_openblas
gcc -std=c2x -o3 -march=native -ffast-math -Wall -o matmul_c11_openblas matmul_c11_openblas.c -lopenblas -lm && ./matmul_c11_openblas
