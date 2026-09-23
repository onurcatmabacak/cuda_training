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

_20260923_045119 — size **1024**, **10** runs, 8 threads, CUDA arch `sm_50`. GFLOPS is as reported, or computed as `2·N³/t` when only a time is printed._

| Category | Benchmark | Status | Avg time | GFLOPS | Notes |
|---|---|---|---|---|---|
| cuda | `cuda__matmul_cublas_cpp20` | OK | 48.770 ms | 44.03 | C=2073.63 |
| cuda | `cuda__matmul_cublas_c11` | OK | 48.772 ms | 44.03 | C=2073.633457 |
| julia | `julia__matmul_cublas_julia` | OK | 48.782 ms | 44.02 | C=261.10895616210365 |
| julia | `julia__matmul_julia_gpu` | OK | 49.120 ms | 43.72 | C=261.9018577353593 |
| cuda | `cuda__matmul_cublas_dgemm` | OK | 49.607 ms | 43.30 |  |
| cuda | `cuda__matmul_cuda[gpu]` | OK | 50.539 ms | 42.49 |  |
| cuda | `cuda__matmul_cuda_optimized` | OK | 52.511 ms | 40.90 | max rel err 4.463e-05; OK |
| cuda | `cuda__matmul_cuda_faster` | OK | 67.748 ms | 31.70 | max rel err 4.386e-16; OK |
| cuda | `cuda__matmul_cuda_cpp20_faster` | OK | 67.757 ms | 31.69 | max rel err 4.38591e-16; OK |
| cpu_mkl | `cpu_mkl__matmul_c11_intel_mkl_flags3` | OK | 69.489 ms | 30.90 | C=252.622460 |
| cpu_mkl | `cpu_mkl__matmul_c11_intel_mkl_flags4` | OK | 69.637 ms | 30.84 | C=255.434551 |
| cpu_mkl | `cpu_mkl__matmul_cpp20_intel_mkl` | OK | 70.610 ms | 30.41 | C=254.863 |
| cpu_mkl | `cpu_mkl__matmul_c11_intel_mkl_flags2` | OK | 71.256 ms | 30.14 | C=258.282140 |
| cpu_mkl | `cpu_mkl__matmul_c11_intel_mkl_flags1` | OK | 73.656 ms | 29.16 | C=252.466792 |
| julia | `julia__matmul_julia_cpu` | OK | 77.994 ms | 27.53 | C=268.5329402090787 |
| julia | `julia__matmul_julia_gpu_faster` | OK | 83.993 ms | 25.57 | max rel err 2.2160732406049341e-16 |
| cpu_c | `cpu_c__matmul_c11_parallel_loops_O3` | OK | 352.000 ms | 6.10 |  |
| cpu_c | `cpu_c__matmul_c11_openblas_O3` | OK | 656.000 ms | 3.27 |  |
| cpu_c | `cpu_c__matmul_c11_index_order_O3` | OK | 1.2060 s | 1.78 |  |
| cpu_c | `cpu_c__matmul_c23_O3` | OK | 4.2990 s | 0.50 |  |
| cpu_c | `cpu_c__matmul_c99_O3` | OK | 4.3010 s | 0.50 |  |
| cpu_c | `cpu_c__matmul_c11_O3` | OK | 4.3520 s | 0.49 |  |
| cpu_c | `cpu_c__matmul_c23_O2` | OK | 4.3690 s | 0.49 |  |
| cpu_c | `cpu_c__matmul_c11_O2` | OK | 4.4640 s | 0.48 |  |
| cpu_c | `cpu_c__matmul_c99_O2` | OK | 4.4650 s | 0.48 |  |
| cuda | `cuda__matmul_cuda[cpu]` | OK | 5.0441 s | 0.43 | speedup 99.8x |
| cpu_c | `cpu_c__matmul_c11_O1` | OK | 6.0070 s | 0.36 |  |
| cpu_c | `cpu_c__matmul_c23_O1` | OK | 6.0320 s | 0.36 |  |
| cpu_c | `cpu_c__matmul_c99_O1` | OK | 6.3090 s | 0.34 |  |
| demos | `demos__vector_add_v1` | OK |  |  |  |
| demos | `demos__vector_add_v2` | OK |  |  |  |
| demos | `demos__whoami` | OK |  |  |  |

<!-- RESULTS:END -->

The suite runs every benchmark with the same size, run count and precision
(**Float64, N=1024, 10 runs**), so the numbers are directly comparable.  See
[`matmul_optimization/README.md`](matmul_optimization/README.md) for the
C++ vs Julia DGEMM comparison (~44 GFLOPS with cuBLAS, ~41 for the hand-written
register-tiled C++ kernel).
