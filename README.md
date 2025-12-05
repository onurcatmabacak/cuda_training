# cuda_training
cuda training 

Average of 100 runs for 4096x4096 matrix multiplication

CUBLAS + C11

Average time per run: 0.136673 s
Effective GFLOPS: 1005.607761
C[0,0] = 8294.673828

CUBLAS + C++20

GPU cuBLAS 4096x4096 matrix multiplication
Average time per run: 0.13688 s
Effective GFLOPS: 1004.09
C[1,1] = 8294.67

CUBLAS + Julia 1.12.2

GPU Julia 4096x4096 matrix multiplication (cuBLAS)
Average time per run: 0.154373 s
Effective GFLOPS: 890.31
C[1,1] = 1035.1787
