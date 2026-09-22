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

_20260922_040540 — size **default**, **10** runs, 8 threads, CUDA arch `sm_50`. GFLOPS is as reported, or computed as `2·N³/t` when only a time is printed._

| Category | Benchmark | Status | Avg time | GFLOPS | Notes |
|---|---|---|---|---|---|
| cpu_c | `cpu_c__matmul_c11_O1` | OK | 5.8860 s | 0.36 |  |
| cpu_c | `cpu_c__matmul_c11_O2` | OK | 4.2830 s | 0.50 |  |
| cpu_c | `cpu_c__matmul_c11_O3` | OK | 4.3080 s | 0.50 |  |
| cpu_c | `cpu_c__matmul_c11_index_order_O3` | OK | 1.2240 s | 1.75 |  |
| cpu_c | `cpu_c__matmul_c11_openblas_O3` | OK | 620.000 ms | 3.46 |  |
| cpu_c | `cpu_c__matmul_c11_parallel_loops_O3` | OK | 362.000 ms | 5.94 |  |
| cpu_c | `cpu_c__matmul_c23_O1` | OK | 5.9460 s | 0.36 |  |
| cpu_c | `cpu_c__matmul_c23_O2` | OK | 4.2640 s | 0.50 |  |
| cpu_c | `cpu_c__matmul_c23_O3` | OK | 4.2780 s | 0.50 |  |
| cpu_c | `cpu_c__matmul_c99_O1` | OK | 5.9350 s | 0.36 |  |
| cpu_c | `cpu_c__matmul_c99_O2` | OK | 4.2910 s | 0.50 |  |
| cpu_c | `cpu_c__matmul_c99_O3` | OK | 4.3010 s | 0.50 |  |
| cpu_mkl | `cpu_mkl__matmul_c11_intel_mkl_flags1` | OK | 3.0647 s | 44.85 | C=1032.600174 |
| cpu_mkl | `cpu_mkl__matmul_c11_intel_mkl_flags2` | OK | 3.0657 s | 44.83 | C=1014.253074 |
| cpu_mkl | `cpu_mkl__matmul_c11_intel_mkl_flags3` | OK | 3.0843 s | 44.56 | C=997.627889 |
| cpu_mkl | `cpu_mkl__matmul_c11_intel_mkl_flags4` | OK | 3.0552 s | 44.99 | C=1028.618455 |
| cpu_mkl | `cpu_mkl__matmul_cpp20_intel_mkl` | OK | 3.1481 s | 43.66 | C=1034.18 |
| cuda | `cuda__matmul_cublas_c11` | OK | 3.2475 s | 42.32 | C=8294.665937 |
| cuda | `cuda__matmul_cublas_cpp20` | OK | 3.3820 s | 40.64 | C=8294.67 |
| cuda | `cuda__matmul_cuda[cpu]` | OK | 2.0420 s | 1.05 | speedup 81.5x |
| cuda | `cuda__matmul_cuda[gpu]` | OK | 25.066 ms | 85.67 |  |
| cuda | `cuda__matmul_cuda_cpp20_faster` | OK | 723.575 ms | 189.9 | max rel err 1.17734e-07; OK |
| cuda | `cuda__matmul_cuda_faster` | OK | 697.541 ms | 197.0 | max rel err 1.177e-07; OK |
| julia | `julia__matmul_cublas_julia` | OK | 3.1572 s | 43.53 | C=1027.6428833405673 |
| julia | `julia__matmul_julia_cpu` | OK | 1.9397 s | 70.86 | C=1016.8564 |
| julia | `julia__matmul_julia_gpu` | OK | 188.860 ms | 727.7 | C=1037.6733 |
| julia | `julia__matmul_julia_gpu_faster` | OK | 1.7574 s | 78.21 | max rel err 6.04588670818457e-8 |
| demos | `demos__vector_add_v1` | OK |  |  |  |
| demos | `demos__vector_add_v2` | OK |  |  |  |
| demos | `demos__whoami` | OK |  |  |  |

<!-- RESULTS:END -->
