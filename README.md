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

_20260923_061531 — size **1024**, **10** runs, 8 threads, CUDA arch `sm_50`. GFLOPS is as reported, or computed as `2·N³/t` when only a time is printed._

| Category | Benchmark | Status | Avg time | GFLOPS | Notes |
|---|---|---|---|---|---|
| cuda | `matmul_cuda.cu` | OK | 48.455 ms | 44.32 |  |
| cuda | `matmul_cublas_c11.cu` | OK | 48.770 ms | 44.03 | C=2073.633457 |
| cuda | `matmul_cublas_cpp20.cu` | OK | 48.775 ms | 44.03 | C=2073.63 |
| julia | `matmul_cublas_julia.jl` | OK | 48.796 ms | 44.01 | C=251.84999342373902 |
| cpp26 | `matmul_cpp26_cuda.cpp` | OK | 48.792 ms | 44.00 |  |
| cpp26 | `matmul_cpp26_cuda.cpp` | OK | 48.792 ms | 44.00 |  |
| cuda | `matmul_cuda_best.cu` | OK | 48.794 ms | 44.00 |  |
| julia | `matmul_julia_gpu.jl` | OK | 49.204 ms | 43.64 | C=259.04201903275134 |
| cpp26 | `matmul_cpp26_cuda.cpp` | OK | 49.205 ms | 43.60 |  |
| rust | `matmul_rust_cublas.rs` | OK | 49.620 ms | 43.30 |  |
| rust | `matmul_rust_cublas.rs` | OK | 49.641 ms | 43.30 |  |
| cuda | `matmul_cuda_best.cu` | OK | 52.503 ms | 40.90 | max rel err 4.463e-05; OK |
| cuda | `matmul_cuda_cpp20_faster.cu` | OK | 67.710 ms | 31.72 | max rel err 4.38591e-16; OK |
| cuda | `matmul_cuda_faster.cu` | OK | 67.713 ms | 31.71 | max rel err 4.386e-16; OK |
| cpu_mkl | `matmul_c11_intel_mkl.c` | OK | 68.235 ms | 31.47 | C=255.081692 |
| cpu_mkl | `matmul_c11_intel_mkl.c` | OK | 70.729 ms | 30.36 | C=252.683208 |
| cpu_mkl | `matmul_c11_intel_mkl.c` | OK | 71.538 ms | 30.02 | C=267.236012 |
| cpu_mkl | `matmul_cpp20_intel_mkl.cpp` | OK | 72.768 ms | 29.51 | C=254.863 |
| cpu_mkl | `matmul_c11_intel_mkl.c` | OK | 73.558 ms | 29.19 | C=242.667056 |
| julia | `matmul_julia_cpu.jl` | OK | 82.524 ms | 26.02 | C=246.74955049655716 |
| julia | `matmul_julia_gpu_faster.jl` | OK | 84.019 ms | 25.56 | max rel err 0.0 |
| cpu_c | `matmul_c11_parallel_loops.c` | OK | 360.000 ms | 5.96 |  |
| cpu_c | `matmul_c11_openblas.c` | OK | 638.000 ms | 3.36 |  |
| cpu_c | `matmul_c11_index_order.c` | OK | 1.2200 s | 1.76 |  |
| cpu_c | `matmul_c23.c` | OK | 4.3160 s | 0.50 |  |
| cpu_c | `matmul_c99.c` | OK | 4.3160 s | 0.50 |  |
| cpu_c | `matmul_c99.c` | OK | 4.3280 s | 0.50 |  |
| cpu_c | `matmul_c23.c` | OK | 4.3300 s | 0.50 |  |
| cpu_c | `matmul_c11.c` | OK | 4.3750 s | 0.49 |  |
| cpu_c | `matmul_c11.c` | OK | 4.3910 s | 0.49 |  |
| cuda | `matmul_cuda.cu` | OK | 5.8187 s | 0.37 | speedup 120.1x |
| cpu_c | `matmul_c11.c` | OK | 5.9710 s | 0.36 |  |
| cpu_c | `matmul_c23.c` | OK | 5.9870 s | 0.36 |  |
| cpu_c | `matmul_c99.c` | OK | 6.0530 s | 0.35 |  |
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
