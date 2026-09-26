# cuda_training

CUDA training for matrix multiplication benchmarks.

The full benchmark suite — CPU C (gcc + OpenBLAS), Intel MKL, CUDA/cuBLAS,
C++26, Rust, Kokkos and Julia (including a vendor-agnostic GPU kernel) — lives in
[`matmul_optimization/`](matmul_optimization/README.md).
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

_20260926_190739 — size **1024**, **10** runs, 8 threads, CUDA arch `sm_50`. GFLOPS is as reported, or computed as `2·N³/t` when only a time is printed._

| Category | Benchmark | Status | Avg time | GFLOPS | Notes |
|---|---|---|---|---|---|
| cuda | `matmul_cuda.cu` | OK | 48.335 ms | 44.43 |  |
| cpp26 | `matmul_cpp26_cuda.cpp` | OK | 48.739 ms | 44.10 |  |
| cpp26 | `matmul_cpp26_cuda.cpp` | OK | 48.737 ms | 44.10 |  |
| cuda | `matmul_cuda_best.cu` | OK | 48.731 ms | 44.10 |  |
| rust | `matmul_rust_cublas.rs` | OK | 48.750 ms | 44.10 |  |
| rust | `matmul_rust_cublas.rs` | OK | 48.739 ms | 44.10 |  |
| cuda | `matmul_cublas_c11.cu` | OK | 48.711 ms | 44.09 | C=2073.633457 |
| cuda | `matmul_cublas_cpp20.cu` | OK | 48.721 ms | 44.08 | C=2073.63 |
| julia | `matmul_cublas_julia.jl` | OK | 48.744 ms | 44.06 | C=255.18000596727404 |
| julia | `matmul_julia_gpu.jl` | OK | 49.011 ms | 43.82 | C=260.4584074539862 |
| cpp26 | `matmul_cpp26_cuda.cpp` | OK | 49.125 ms | 43.70 |  |
| cuda | `matmul_cuda_best.cu` | OK | 52.449 ms | 40.90 | max rel err 4.463e-05; OK |
| cpu_mkl | `matmul_c11_intel_mkl.c` | OK | 63.899 ms | 33.61 | C=252.684312 |
| cpu_mkl | `matmul_c11_intel_mkl.c` | OK | 65.136 ms | 32.97 | C=263.388071 |
| cpu_mkl | `matmul_c11_intel_mkl.c` | OK | 66.279 ms | 32.40 | C=257.149500 |
| cpu_mkl | `matmul_c11_intel_mkl.c` | OK | 67.614 ms | 31.76 | C=244.827859 |
| cuda | `matmul_cuda_faster.cu` | OK | 67.647 ms | 31.75 | max rel err 4.386e-16; OK |
| cuda | `matmul_cuda_cpp20_faster.cu` | OK | 67.667 ms | 31.74 | max rel err 4.38591e-16; OK |
| cpu_mkl | `matmul_cpp20_intel_mkl.cpp` | OK | 68.120 ms | 31.52 | C=254.863 |
| julia | `matmul_julia_cpu.jl` | OK | 76.554 ms | 28.05 | C=238.532167827893 |
| julia | `matmul_julia_gpu_vendor_agnostic.jl` | OK | 80.140 ms | 26.80 | max rel err 7.974838814153552e-16; OK; backend CUDA |
| julia | `matmul_julia_gpu_faster.jl` | OK | 83.932 ms | 25.59 | max rel err 1.1704803540383514e-16 |
| cpu_c | `matmul_c11_parallel_loops.c` | OK | 371.000 ms | 5.78 |  |
| cpu_c | `matmul_c11_openblas.c` | OK | 637.000 ms | 3.37 |  |
| cpu_c | `matmul_c11_index_order.c` | OK | 1.1840 s | 1.81 |  |
| cpu_c | `matmul_c99.c` | OK | 4.2290 s | 0.51 |  |
| cpu_c | `matmul_c23.c` | OK | 4.2440 s | 0.51 |  |
| cpu_c | `matmul_c99.c` | OK | 4.2530 s | 0.50 |  |
| cpu_c | `matmul_c11.c` | OK | 4.2730 s | 0.50 |  |
| cpu_c | `matmul_c11.c` | OK | 4.2750 s | 0.50 |  |
| cpu_c | `matmul_c23.c` | OK | 4.3210 s | 0.50 |  |
| kokkos | `matmul_kokkos.cpp` | OK | 4.7571 s | 0.45 | max rel err 0.000e+00; OK; Kokkos OpenMP |
| cpu_c | `matmul_c99.c` | OK | 5.8550 s | 0.37 |  |
| cpu_c | `matmul_c23.c` | OK | 5.9170 s | 0.36 |  |
| cpu_c | `matmul_c11.c` | OK | 5.9340 s | 0.36 |  |
| cuda | `matmul_cuda.cu` | OK | 6.4988 s | 0.33 | speedup 134.5x |
| demos | `vector_add_v1.cu` | OK |  |  |  |
| demos | `vector_add_v2.cu` | OK |  |  |  |
| demos | `whoami_cuda.cu` | OK |  |  |  |

<!-- RESULTS:END -->

The suite runs every benchmark with the same size, run count and precision
(**Float64, N=1024, 10 runs**), so the numbers are directly comparable.  See
[`matmul_optimization/README.md`](matmul_optimization/README.md) for the
cross-language DGEMM comparison: cuBLAS reaches ~44 GFLOPS whether called from C,
C++26, Rust or Julia, while a hand-written register-tiled CUDA kernel reaches
~41.  The suite also includes a C++ Kokkos DGEMM (vendor-neutral, skipped when
Kokkos is not installed) and a Julia kernel written with KernelAbstractions.jl
that runs unchanged on any GPU vendor's backend.
