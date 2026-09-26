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

_20260926_194058 — size **1024**, **10** runs, 8 threads, CUDA arch `sm_50`. GFLOPS is as reported, or computed as `2·N³/t` when only a time is printed._

| Category | Benchmark | Status | Avg time | GFLOPS | Notes |
|---|---|---|---|---|---|
| cuda | `matmul_cuda.cu` | OK | 48.391 ms | 44.38 |  |
| cuda | `matmul_cuda_best.cu` | OK | 48.751 ms | 44.10 |  |
| cuda | `matmul_cublas_c11.cu` | OK | 48.730 ms | 44.07 | C=2073.633457 |
| cuda | `matmul_cublas_cpp20.cu` | OK | 48.732 ms | 44.07 | C=2073.63 |
| julia | `matmul_cublas_julia.jl` | OK | 48.751 ms | 44.05 | C=254.62961121619713 |
| cpp26 | `matmul_cpp26_cuda.cpp` | OK | 48.755 ms | 44.00 |  |
| julia | `matmul_julia_gpu.jl` | OK | 49.053 ms | 43.78 | C=243.85194879366693 |
| cpp26 | `matmul_cpp26_cuda.cpp` | OK | 49.579 ms | 43.30 |  |
| rust | `matmul_rust_cublas.rs` | OK | 49.598 ms | 43.30 |  |
| cpp26 | `matmul_cpp26_cuda.cpp` | OK | 49.855 ms | 43.10 |  |
| rust | `matmul_rust_cublas.rs` | OK | 50.219 ms | 42.80 |  |
| cuda | `matmul_cuda_best.cu` | OK | 52.463 ms | 40.90 | max rel err 4.463e-05; OK |
| cuda | `matmul_cuda_cpp20_faster.cu` | OK | 67.667 ms | 31.74 | max rel err 4.38591e-16; OK |
| cuda | `matmul_cuda_faster.cu` | OK | 67.670 ms | 31.73 | max rel err 4.386e-16; OK |
| cpu_mkl | `matmul_c11_intel_mkl.c` | OK | 69.251 ms | 31.01 | C=251.536653 |
| cpu_mkl | `matmul_c11_intel_mkl.c` | OK | 70.236 ms | 30.58 | C=250.902362 |
| cpu_mkl | `matmul_c11_intel_mkl.c` | OK | 70.634 ms | 30.40 | C=257.630088 |
| cpu_mkl | `matmul_cpp20_intel_mkl.cpp` | OK | 72.403 ms | 29.66 | C=254.863 |
| cpu_mkl | `matmul_c11_intel_mkl.c` | OK | 72.687 ms | 29.54 | C=257.502524 |
| julia | `matmul_julia_gpu_vendor_agnostic.jl` | OK | 80.141 ms | 26.80 | max rel err 1.310268260007699e-15; OK; backend CUDA |
| julia | `matmul_julia_cpu.jl` | OK | 82.195 ms | 26.13 | C=256.53110531394475 |
| kokkos | `matmul_kokkos.cpp` | OK | 82.287 ms | 26.10 | max rel err 1.185e-16; OK; Kokkos Cuda |
| julia | `matmul_julia_gpu_faster.jl` | OK | 83.973 ms | 25.57 | max rel err 0.0 |
| cpu_c | `matmul_c11_parallel_loops.c` | OK | 353.000 ms | 6.09 |  |
| cpu_c | `matmul_c11_openblas.c` | OK | 637.000 ms | 3.37 |  |
| cpu_c | `matmul_c11_index_order.c` | OK | 1.1790 s | 1.82 |  |
| cpu_c | `matmul_c11.c` | OK | 4.2270 s | 0.51 |  |
| cpu_c | `matmul_c99.c` | OK | 4.2640 s | 0.50 |  |
| cpu_c | `matmul_c99.c` | OK | 4.2800 s | 0.50 |  |
| cpu_c | `matmul_c11.c` | OK | 4.2810 s | 0.50 |  |
| cpu_c | `matmul_c23.c` | OK | 4.3180 s | 0.50 |  |
| cpu_c | `matmul_c23.c` | OK | 4.3480 s | 0.49 |  |
| cuda | `matmul_cuda.cu` | OK | 5.1732 s | 0.42 | speedup 106.9x |
| cpu_c | `matmul_c23.c` | OK | 5.8410 s | 0.37 |  |
| cpu_c | `matmul_c99.c` | OK | 5.8610 s | 0.37 |  |
| cpu_c | `matmul_c11.c` | OK | 5.9100 s | 0.36 |  |
| demos | `vector_add_v1.cu` | OK |  |  |  |
| demos | `vector_add_v2.cu` | OK |  |  |  |
| demos | `whoami_cuda.cu` | OK |  |  |  |

<!-- RESULTS:END -->

The suite runs every benchmark with the same size, run count and precision
(**Float64, N=1024, 10 runs**), so the numbers are directly comparable.  See
[`matmul_optimization/README.md`](matmul_optimization/README.md) for the
cross-language DGEMM comparison: cuBLAS reaches ~44 GFLOPS whether called from C,
C++26, Rust or Julia, while a hand-written register-tiled CUDA kernel reaches
~41.  The suite also includes a C++ Kokkos DGEMM (the same source runs on the
CUDA backend here — 26.1 GFLOPS on this GPU — or on OpenMP/Serial/HIP/SYCL
builds) and a Julia kernel written with KernelAbstractions.jl that runs
unchanged on any GPU vendor's backend.
