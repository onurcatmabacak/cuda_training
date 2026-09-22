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

_20260922_195159 — size **default**, **10** runs, 8 threads, CUDA arch `sm_50`. GFLOPS is as reported, or computed as `2·N³/t` when only a time is printed._

| Category | Benchmark | Status | Avg time | GFLOPS | Notes |
|---|---|---|---|---|---|
| cpu_c | `cpu_c__matmul_c11_O1` | OK | 5.9570 s | 0.36 |  |
| cpu_c | `cpu_c__matmul_c11_O2` | OK | 4.3240 s | 0.50 |  |
| cpu_c | `cpu_c__matmul_c11_O3` | OK | 4.2910 s | 0.50 |  |
| cpu_c | `cpu_c__matmul_c11_index_order_O3` | OK | 1.2010 s | 1.79 |  |
| cpu_c | `cpu_c__matmul_c11_openblas_O3` | OK | 633.000 ms | 3.39 |  |
| cpu_c | `cpu_c__matmul_c11_parallel_loops_O3` | OK | 367.000 ms | 5.86 |  |
| cpu_c | `cpu_c__matmul_c23_O1` | OK | 5.9230 s | 0.36 |  |
| cpu_c | `cpu_c__matmul_c23_O2` | OK | 4.2630 s | 0.50 |  |
| cpu_c | `cpu_c__matmul_c23_O3` | OK | 4.3060 s | 0.50 |  |
| cpu_c | `cpu_c__matmul_c99_O1` | OK | 5.9150 s | 0.36 |  |
| cpu_c | `cpu_c__matmul_c99_O2` | OK | 4.2370 s | 0.51 |  |
| cpu_c | `cpu_c__matmul_c99_O3` | OK | 4.2440 s | 0.51 |  |
| cpu_mkl | `cpu_mkl__matmul_c11_intel_mkl_flags1` | OK | 3.1014 s | 44.32 | C=1036.157623 |
| cpu_mkl | `cpu_mkl__matmul_c11_intel_mkl_flags2` | OK | 3.0865 s | 44.53 | C=1015.716854 |
| cpu_mkl | `cpu_mkl__matmul_c11_intel_mkl_flags3` | OK | 3.0994 s | 44.34 | C=1029.048248 |
| cpu_mkl | `cpu_mkl__matmul_c11_intel_mkl_flags4` | OK | 3.1072 s | 44.23 | C=1034.242207 |
| cpu_mkl | `cpu_mkl__matmul_cpp20_intel_mkl` | OK | 3.2011 s | 42.93 | C=1034.18 |
| cuda | `cuda__matmul_cublas_c11` | OK | 3.2313 s | 42.53 | C=8294.665937 |
| cuda | `cuda__matmul_cublas_cpp20` | OK | 3.3701 s | 40.78 | C=8294.67 |
| cuda | `cuda__matmul_cublas_sgemm` | OK | 143.010 ms | 961.0 |  |
| cuda | `cuda__matmul_cuda[cpu]` | OK | 1.9975 s | 1.08 | speedup 80.8x |
| cuda | `cuda__matmul_cuda[gpu]` | OK | 24.707 ms | 86.92 |  |
| cuda | `cuda__matmul_cuda_cpp20_faster` | OK | 717.605 ms | 191.5 | max rel err 1.17734e-07; OK |
| cuda | `cuda__matmul_cuda_faster` | OK | 690.881 ms | 198.9 | max rel err 1.177e-07; OK |
| cuda | `cuda__matmul_cuda_optimized` | OK | 233.592 ms | 588.4 | max rel err 5.769e-06; OK |
| julia | `julia__matmul_cublas_julia` | OK | 3.1614 s | 43.47 | C=1009.0117242372455 |
| julia | `julia__matmul_julia_cpu` | OK | 1.9259 s | 71.36 | C=1008.1797 |
| julia | `julia__matmul_julia_gpu` | OK | 192.778 ms | 712.9 | C=1027.0885 |
| julia | `julia__matmul_julia_gpu_faster` | OK | 1.7562 s | 78.26 | max rel err 6.041840835315907e-8 |
| demos | `demos__vector_add_v1` | OK |  |  |  |
| demos | `demos__vector_add_v2` | OK |  |  |  |
| demos | `demos__whoami` | OK |  |  |  |

<!-- RESULTS:END -->

See [`matmul_optimization/README.md`](matmul_optimization/README.md) for the
best-of C++ vs Julia CUDA SGEMM comparison (~1200 GFLOPS with cuBLAS in both,
~600 for a hand-written register-tiled C++ kernel).
