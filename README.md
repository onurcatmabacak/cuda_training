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

_Measured on an Acer laptop (Intel i7-6700HQ 4c/8t, NVIDIA GTX 960M / sm_50). This run was
made from a bare TTY3 session with no desktop (gdm stopped), so the CPU is no longer
power-capped and no compositor shares the GPU. GPU rows warm up for 2 s so they measure at
boost clocks; both CPU and GPU results still vary with the machine's thermal state._

<!-- RESULTS:START -->

_20261002_164512 — size **1024**, **10** runs, 8 threads, CUDA arch `sm_50`. GFLOPS is as reported, or computed as `2·N³/t` when only a time is printed._

| Category | Benchmark | Status | Avg time | GFLOPS | Notes |
|---|---|---|---|---|---|
| cpu_mkl | `matmul_c11_intel_mkl.c` | OK | 15.691 ms | 136.9 | C=269.860369 |
| cpu_mkl | `matmul_c11_intel_mkl.c` | OK | 15.763 ms | 136.2 | C=251.335104 |
| cpu_mkl | `matmul_c11_intel_mkl.c` | OK | 15.792 ms | 136.0 | C=254.954425 |
| cpu_mkl | `matmul_c11_intel_mkl.c` | OK | 15.797 ms | 135.9 | C=254.279666 |
| cpu_mkl | `matmul_cpp20_intel_mkl.cpp` | OK | 15.861 ms | 135.4 | C=254.863 |
| julia | `matmul_julia_cpu.jl` | OK | 18.716 ms | 114.7 | C=254.68423977943078 |
| cuda | `matmul_cuda.cu` | OK | 48.379 ms | 44.39 |  |
| cuda | `matmul_cublas_c11.cu` | OK | 48.726 ms | 44.07 | C=2073.633457 |
| cuda | `matmul_cublas_cpp20.cu` | OK | 48.737 ms | 44.06 | C=2073.63 |
| cpp26 | `matmul_cpp26_cuda.cpp` | OK | 48.772 ms | 44.00 |  |
| cuda | `matmul_cuda_best.cu` | OK | 48.770 ms | 44.00 |  |
| julia | `matmul_cublas_julia.jl` | OK | 49.610 ms | 43.29 | C=254.8073663688729 |
| cpp26 | `matmul_cpp26_cuda.cpp` | OK | 50.678 ms | 42.40 |  |
| cpp26 | `matmul_cpp26_cuda.cpp` | OK | 50.738 ms | 42.30 |  |
| rust | `matmul_rust_cublas.rs` | OK | 50.735 ms | 42.30 |  |
| rust | `matmul_rust_cublas.rs` | OK | 51.260 ms | 41.90 |  |
| cuda | `matmul_cuda_best.cu` | OK | 52.483 ms | 40.90 | max rel err 4.463e-05; OK |
| julia | `matmul_julia_gpu.jl` | OK | 52.715 ms | 40.74 | C=257.53690624459995 |
| kokkos | `matmul_kokkos.cpp` | OK | 58.857 ms | 36.49 | max rel err 1.185e-16; OK; Kokkos Cuda |
| cuda | `matmul_cuda_faster.cu` | OK | 67.670 ms | 31.73 | max rel err 4.386e-16; OK |
| cuda | `matmul_cuda_cpp20_faster.cu` | OK | 67.691 ms | 31.72 | max rel err 4.38591e-16; OK |
| julia | `matmul_julia_gpu_vendor_agnostic.jl` | OK | 103.685 ms | 20.71 | max rel err 1.1972545220710027e-16; OK; backend CUDA |
| cpu_c | `matmul_c11_parallel_loops.c` | OK | 105.000 ms | 20.38 |  |
| julia | `matmul_julia_gpu_faster.jl` | OK | 132.054 ms | 16.26 | max rel err 2.200557425881903e-16 |
| cpu_c | `matmul_c11_openblas.c` | OK | 132.000 ms | 16.24 |  |
| cpu_c | `matmul_c11_index_order.c` | OK | 458.000 ms | 4.69 |  |
| cpu_c | `matmul_c23.c` | OK | 1.8770 s | 1.14 |  |
| cpu_c | `matmul_c11.c` | OK | 1.8860 s | 1.14 |  |
| cpu_c | `matmul_c11.c` | OK | 1.8990 s | 1.13 |  |
| cpu_c | `matmul_c99.c` | OK | 1.9020 s | 1.13 |  |
| cpu_c | `matmul_c23.c` | OK | 1.9330 s | 1.11 |  |
| cpu_c | `matmul_c99.c` | OK | 1.9470 s | 1.10 |  |
| cuda | `matmul_cuda.cu` | OK | 2.4080 s | 0.89 | speedup 49.8x |
| cpu_c | `matmul_c11.c` | OK | 2.4280 s | 0.88 |  |
| cpu_c | `matmul_c23.c` | OK | 2.5020 s | 0.86 |  |
| cpu_c | `matmul_c99.c` | OK | 2.6060 s | 0.82 |  |
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
CUDA backend here — 36.5 GFLOPS on this GPU — or on OpenMP/Serial/HIP/SYCL
builds) and a Julia kernel written with KernelAbstractions.jl that runs
unchanged on any GPU vendor's backend (register-blocked, 33.5 GFLOPS).
