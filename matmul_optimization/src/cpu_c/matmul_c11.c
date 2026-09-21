#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

#ifndef N
#define N 1024 // Matrix size (N x N)
#endif
#ifndef RUNS
#define RUNS 100
#endif

_Noreturn void die(const char *msg){
    fprintf(stderr, "%s\n", msg);
    exit(EXIT_FAILURE);
}

double **allocateMatrix(int n){
    double **mat = malloc(n * sizeof *mat);

    if (mat == NULL)
        die("Memory allocation failed for matrix row. \n");

    for(int i = 0; i < n; i++){

        mat[i] = malloc(n * sizeof *mat[i]);
        if (mat[i] == NULL)
            die("Memory allocation failed for matrix columns. \n");
    }

    return mat;
}

void freeMatrix(double **mat, int n){
    for (int i = 0; i < n; i++)
        free(mat[i]);
    free(mat);
}

void fillRandom(double **mat, int n){
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)

            mat[i][j] = (double) rand() / RAND_MAX;
}

void multiplyMatrices(double **A, double **B, double **C, int n){

    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            double sum = 0.0;

            for (int k = 0; k < n; k++)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    }
        
}

int main(){

    srand((unsigned int) time(NULL));

    printf("Allocating matrices...\n");
    double** A = allocateMatrix(N);
    double** B = allocateMatrix(N);
    double** C = allocateMatrix(N);

    printf("Filling matrices with random values...\n");
    fillRandom(A, N);
    fillRandom(B, N);

    printf("Running matrix multip-lication %d times *%dx%d()... \n", RUNS, N, N);

    double total_time = 0.0;

    for (int run = 0; run < RUNS; run++){

        //printf("Multiplying matrices (%dx%d)...\n", N, N);
        clock_t start = clock();
        multiplyMatrices(A, B, C, N);
        clock_t end = clock();
        double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

        total_time += elapsed;

        // printf("Run %3d: %.3f seconds \n", run + 1, elapsed);

    }

    double avrg_time = total_time / RUNS;
    printf("\nAverage time over %d runs: %.3f seconds\n", RUNS, avrg_time);

    // Optionally print a small portion of the result
    // printf("Sample output (C[0][0..4]): ");

    // for (int j = 0; j < 5; j++){
    //     printf("\n %8.4f", C[0][j]);
    // }

    // printf("\n");

    freeMatrix(A, N);
    freeMatrix(B, N);
    freeMatrix(C, N);

    return 0;
}