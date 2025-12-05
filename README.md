# cuda_training
cuda training 

Average of 100 runs for 4096x4096 matrix multiplication FLOAT32

CUBLAS + C11

Average time per run: 0.136673 s
Effective GFLOPS: 1005.607761
C[0,0] = 8294.673828

CUBLAS + C++20

GPU cuBLAS
Average time per run: 0.13688 s
Effective GFLOPS: 1004.09
C[1,1] = 8294.67

CUBLAS + Julia 1.12.2

GPU Julia (cuBLAS)
Average time per run: 0.154373 s
Effective GFLOPS: 890.31
C[1,1] = 1035.1787

Average of 100 runs for 4096x4096 matrix multiplication FLOAT64

C11 DGEMM BLAS benchmark (Float64) 4096x4096, 100 runs

Average time per run: 5.658008 s 
Effective GFLOPS: 24.291048 
C[0,0] = 8294.665937

C++20

GPU cuBLAS 4096x4096 matrix multiplication
Average time per run: 8.18943 s
Effective GFLOPS: 16.7825
C[1,1] = 8294.67

GPU Julia 4096x4096 matrix multiplication (cuBLAS)
Average time per run: 8.914042 s
Effective GFLOPS: 15.42
C[1,1] = 1038.9856662159339
