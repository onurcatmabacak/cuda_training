#!/bin/bash
clear

export OPENBLAS_NUM_THREADS=8

# gcc -O1 -march=native -ffast-math -Wall -o matmul_c11_openblas matmul_c11_openblas.c -lopenblas -lm && ./matmul_c11_openblas
# gcc -O2 -march=native -ffast-math -Wall -o matmul_c11_openblas matmul_c11_openblas.c -lopenblas -lm && ./matmul_c11_openblas
# gcc -O3 -march=native -ffast-math -Wall -o matmul_c11_openblas matmul_c11_openblas.c -lopenblas -lm && ./matmul_c11_openblas
# gcc -std=c11 -O1 -march=native -ffast-math -Wall -o matmul_c11_openblas matmul_c11_openblas.c -lopenblas -lm && ./matmul_c11_openblas
# gcc -std=c11 -O2 -march=native -ffast-math -Wall -o matmul_c11_openblas matmul_c11_openblas.c -lopenblas -lm && ./matmul_c11_openblas
# gcc -std=c11 -O3 -march=native -ffast-math -Wall -o matmul_c11_openblas matmul_c11_openblas.c -lopenblas -lm && ./matmul_c11_openblas
# gcc -std=c2x -O1 -march=native -ffast-math -Wall -o matmul_c11_openblas matmul_c11_openblas.c -lopenblas -lm && ./matmul_c11_openblas
# gcc -std=c2x -O2 -march=native -ffast-math -Wall -o matmul_c11_openblas matmul_c11_openblas.c -lopenblas -lm && ./matmul_c11_openblas
# gcc -std=c2x -O3 -march=native -ffast-math -Wall -o matmul_c11_openblas matmul_c11_openblas.c -lopenblas -lm && ./matmul_c11_openblas

gcc -O3 -march=native matmul_c11_openblas.c -lopenblas -o matmul_c11_openblas && ./matmul_c11_openblas