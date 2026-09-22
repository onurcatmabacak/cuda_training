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

_20260922_052253 — size **default**, **10** runs, 8 threads, CUDA arch `sm_50`. GFLOPS is as reported, or computed as `2·N³/t` when only a time is printed._

| Category | Benchmark | Status | Avg time | GFLOPS | Notes |
|---|---|---|---|---|---|
| cpu_c | `cpu_c__matmul_c11_O1` | OK | 6.2220 s | 0.35 |  |
| cpu_c | `cpu_c__matmul_c11_O2` | OK | 4.6030 s | 0.47 |  |
| cpu_c | `cpu_c__matmul_c11_O3` | OK | 4.5810 s | 0.47 |  |
| cpu_c | `cpu_c__matmul_c11_index_order_O3` | OK | 1.3320 s | 1.61 |  |
| cpu_c | `cpu_c__matmul_c11_openblas_O3` | OK | 669.000 ms | 3.21 |  |
| cpu_c | `cpu_c__matmul_c11_parallel_loops_O3` | OK | 467.000 ms | 4.59 |  |
| cpu_c | `cpu_c__matmul_c23_O1` | OK | 6.2170 s | 0.35 |  |
| cpu_c | `cpu_c__matmul_c23_O2` | OK | 4.5620 s | 0.47 |  |
| cpu_c | `cpu_c__matmul_c23_O3` | OK | 4.5370 s | 0.47 |  |
| cpu_c | `cpu_c__matmul_c99_O1` | OK | 6.2170 s | 0.35 |  |
| cpu_c | `cpu_c__matmul_c99_O2` | OK | 4.5060 s | 0.48 |  |
| cpu_c | `cpu_c__matmul_c99_O3` | OK | 4.5570 s | 0.47 |  |
| cpu_mkl | `cpu_mkl__matmul_c11_intel_mkl_flags1` | OK | 4.7757 s | 28.78 | C=1031.629657 |
| cpu_mkl | `cpu_mkl__matmul_c11_intel_mkl_flags2` | OK | 5.0079 s | 27.44 | C=1032.935925 |
| cpu_mkl | `cpu_mkl__matmul_c11_intel_mkl_flags3` | OK | 6.6394 s | 20.70 | C=1025.207612 |
| cpu_mkl | `cpu_mkl__matmul_c11_intel_mkl_flags4` | OK | 8.5409 s | 16.09 | C=1032.482294 |
| cpu_mkl | `cpu_mkl__matmul_cpp20_intel_mkl` | OK | 4.4035 s | 31.21 | C=1034.18 |
| cuda | `cuda__matmul_cublas_c11` | OK | 3.6063 s | 38.11 | C=8294.665937 |
| cuda | `cuda__matmul_cublas_cpp20` | OK | 3.6979 s | 37.17 | C=8294.67 |
| cuda | `cuda__matmul_cublas_sgemm` | OK | 180.575 ms | 761.1 |  |
| cuda | `cuda__matmul_cuda[cpu]` | OK | 1.9185 s | 1.12 | speedup 46.4x |
| cuda | `cuda__matmul_cuda[gpu]` | OK | 41.363 ms | 51.92 |  |
| cuda | `cuda__matmul_cuda_cpp20_faster` | OK | 861.454 ms | 159.5 | max rel err 1.17734e-07; OK |
| cuda | `cuda__matmul_cuda_faster` | OK | 866.898 ms | 158.5 | max rel err 1.177e-07; OK |
| cuda | `cuda__matmul_cuda_optimized` | OK | 273.858 ms | 501.9 | max rel err 5.769e-06; OK |
| julia | `julia__matmul_cublas_julia` | OK | 3.2145 s | 42.76 | C=1008.8971024030221 |
| julia | `julia__matmul_julia_cpu` | OK | 5.8040 s | 23.68 | C=1035.1074 |
| julia | `julia__matmul_julia_gpu` | OK | 142.041 ms | 967.6 | C=1005.6342 |
| julia | `julia__matmul_julia_gpu_faster` | OK | 1.7704 s | 77.63 | max rel err 0.0 |
| demos | `demos__vector_add_v1` | OK |  |  |  |
| demos | `demos__vector_add_v2` | OK |  |  |  |
| demos | `demos__whoami` | OK |  |  |  |

<!-- RESULTS:END -->

See [`matmul_optimization/README.md`](matmul_optimization/README.md) for the
best-of C++ vs Julia CUDA SGEMM comparison (~1200 GFLOPS with cuBLAS in both,
~600 for a hand-written register-tiled C++ kernel).
