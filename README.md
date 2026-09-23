# cuda_training

CUDA training for matrix multiplication benchmarks.

The full benchmark suite — CPU C (gcc + OpenBLAS), Intel MKL, CUDA/cuBLAS and
Julia — lives in [`matmul_optimization/`](matmul_optimization/README.md).
One command builds and runs every benchmark and regenerates the table below:

```bash
cd matmul_optimization
./run_all.sh            # full suite, each benchmark's own size/runs
./run_all.sh --quick    # fast smoke test
```

Raw per-benchmark output and machine-readable results are written to
`matmul_optimization/results/latest/` (`summary.csv`, `FULL_REPORT.md`, …).

## Results

_Measured on an Acer laptop (Intel i7-6700HQ 4c/8t, NVIDIA GTX 960M / sm_50). GPU rows
warm up for 2 s so they measure at boost clocks. CPU rows depend on the machine's power
state — this unit currently caps the CPU near 900 MHz, so CPU GFLOPS sit well below the
chip's peak._

<!-- RESULTS:START -->

_20260923_061531 — size **1024**, **10** runs, 8 threads, CUDA arch `sm_50`. GFLOPS is as reported, or computed as `2·N³/t` when only a time is printed._

| Category | Benchmark | Status | Avg time | GFLOPS | Notes |
|---|---|---|---|---|---|
| cuda | `cuda__matmul_cuda[gpu]` | OK | 48.455 ms | 44.32 |  |
| cuda | `cuda__matmul_cublas_c11` | OK | 48.770 ms | 44.03 | C=2073.633457 |
| cuda | `cuda__matmul_cublas_cpp20` | OK | 48.775 ms | 44.03 | C=2073.63 |
| julia | `julia__matmul_cublas_julia` | OK | 48.796 ms | 44.01 | C=251.84999342373902 |
| cpp26 | `cpp26__matmul_cpp26_cublas` | OK | 48.792 ms | 44.00 |  |
| cpp26 | `cpp26__matmul_cpp26_graph` | OK | 48.792 ms | 44.00 |  |
| cuda | `cuda__matmul_cublas_dgemm` | OK | 48.794 ms | 44.00 |  |
| julia | `julia__matmul_julia_gpu` | OK | 49.204 ms | 43.64 | C=259.04201903275134 |
| cpp26 | `cpp26__matmul_cpp26_cublaslt` | OK | 49.205 ms | 43.60 |  |
| rust | `rust__matmul_rust_cublas` | OK | 49.620 ms | 43.30 |  |
| rust | `rust__matmul_rust_graph` | OK | 49.641 ms | 43.30 |  |
| cuda | `cuda__matmul_cuda_optimized` | OK | 52.503 ms | 40.90 | max rel err 4.463e-05; OK |
| cuda | `cuda__matmul_cuda_cpp20_faster` | OK | 67.710 ms | 31.72 | max rel err 4.38591e-16; OK |
| cuda | `cuda__matmul_cuda_faster` | OK | 67.713 ms | 31.71 | max rel err 4.386e-16; OK |
| cpu_mkl | `cpu_mkl__matmul_c11_intel_mkl_flags4` | OK | 68.235 ms | 31.47 | C=255.081692 |
| cpu_mkl | `cpu_mkl__matmul_c11_intel_mkl_flags2` | OK | 70.729 ms | 30.36 | C=252.683208 |
| cpu_mkl | `cpu_mkl__matmul_c11_intel_mkl_flags3` | OK | 71.538 ms | 30.02 | C=267.236012 |
| cpu_mkl | `cpu_mkl__matmul_cpp20_intel_mkl` | OK | 72.768 ms | 29.51 | C=254.863 |
| cpu_mkl | `cpu_mkl__matmul_c11_intel_mkl_flags1` | OK | 73.558 ms | 29.19 | C=242.667056 |
| julia | `julia__matmul_julia_cpu` | OK | 82.524 ms | 26.02 | C=246.74955049655716 |
| julia | `julia__matmul_julia_gpu_faster` | OK | 84.019 ms | 25.56 | max rel err 0.0 |
| cpu_c | `cpu_c__matmul_c11_parallel_loops_O3` | OK | 360.000 ms | 5.96 |  |
| cpu_c | `cpu_c__matmul_c11_openblas_O3` | OK | 638.000 ms | 3.36 |  |
| cpu_c | `cpu_c__matmul_c11_index_order_O3` | OK | 1.2200 s | 1.76 |  |
| cpu_c | `cpu_c__matmul_c23_O2` | OK | 4.3160 s | 0.50 |  |
| cpu_c | `cpu_c__matmul_c99_O3` | OK | 4.3160 s | 0.50 |  |
| cpu_c | `cpu_c__matmul_c99_O2` | OK | 4.3280 s | 0.50 |  |
| cpu_c | `cpu_c__matmul_c23_O3` | OK | 4.3300 s | 0.50 |  |
| cpu_c | `cpu_c__matmul_c11_O3` | OK | 4.3750 s | 0.49 |  |
| cpu_c | `cpu_c__matmul_c11_O2` | OK | 4.3910 s | 0.49 |  |
| cuda | `cuda__matmul_cuda[cpu]` | OK | 5.8187 s | 0.37 | speedup 120.1x |
| cpu_c | `cpu_c__matmul_c11_O1` | OK | 5.9710 s | 0.36 |  |
| cpu_c | `cpu_c__matmul_c23_O1` | OK | 5.9870 s | 0.36 |  |
| cpu_c | `cpu_c__matmul_c99_O1` | OK | 6.0530 s | 0.35 |  |
| demos | `demos__vector_add_v1` | OK |  |  |  |
| demos | `demos__vector_add_v2` | OK |  |  |  |
| demos | `demos__whoami` | OK |  |  |  |

<!-- RESULTS:END -->

The suite runs every benchmark with the same size, run count and precision
(**Float64, N=1024, 10 runs**), so the numbers are directly comparable.  See
[`matmul_optimization/README.md`](matmul_optimization/README.md) for the
cross-language DGEMM comparison: cuBLAS reaches ~44 GFLOPS whether called from C,
C++26, Rust or Julia, while a hand-written register-tiled CUDA kernel reaches
~41.
