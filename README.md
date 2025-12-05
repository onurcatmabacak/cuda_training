# cuda_training
CUDA training for matrix multiplication benchmarks.

---

## FLOAT32 (32-bit floating point)

Average of 100 runs for 4096×4096 matrix multiplication.

| Implementation       | Average Time (s) | Effective GFLOPS | C[0,0] / C[1,1] |
|----------------------|----------------|-----------------|----------------|
| CUBLAS + C11         | 0.136673       | 1005.61         | 8294.673828    |
| CUBLAS + C++20       | 0.13688        | 1004.09         | 8294.67        |
| CUBLAS + Julia 1.12.2| 0.154373       | 890.31          | 1035.1787      |

---

## FLOAT64 (64-bit floating point)

Average of 100 runs for 4096×4096 matrix multiplication.

| Implementation       | Average Time (s) | Effective GFLOPS | C[0,0] / C[1,1] |
|----------------------|----------------|-----------------|----------------|
| C11 cuBLAS           | 5.658008       | 24.29           | 8294.665937    |
| C++20 cuBLAS         | 8.18943        | 16.78           | 8294.67        |
| GPU Julia cuBLAS      | 8.914042       | 15.42           | 1038.985666    |

---

### Notes

- All results are averaged over 100 runs.
- Float32 arrays are used for single-precision benchmarks.
- Float64 arrays are used for double-precision benchmarks.
- `C[0,0]` in C/C++ corresponds to `C[1,1]` in Julia (1-based indexing).

