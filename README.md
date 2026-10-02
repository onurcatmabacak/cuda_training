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
boost clocks. BLAS/DGEMM rows use the 4 physical cores (SMT adds no FP64 throughput) and
the suite pauses 20 s between benchmarks to limit thermal throttling; CPU and GPU results
still vary with the machine's thermal state._

<!-- RESULTS:START -->

_20261002_174119 — size **1024**, **10** runs, 8 threads (BLAS/DGEMM 4), 20 s cooldown between runs, CUDA arch `sm_50`. Theoretical FP64 peak: CPU 198.4, GPU 48.1 GFLOPS; `% peak` is measured/theoretical. GFLOPS is as reported, or computed as `2·N³/t` when only a time is printed._

| Category | Benchmark | Status | Avg time | GFLOPS | % peak |
|---|---|---|---|---|---|
| cpu_mkl | `matmul_cpp20_intel_mkl.cpp` | OK | 14.462 ms | 148.5 | 75% |
| cpu_mkl | `matmul_c11_intel_mkl.c` | OK | 14.480 ms | 148.3 | 75% |
| cpu_mkl | `matmul_c11_intel_mkl.c` | OK | 14.609 ms | 147.0 | 74% |
| cpu_mkl | `matmul_c11_intel_mkl.c` | OK | 14.630 ms | 146.8 | 74% |
| cpu_mkl | `matmul_c11_intel_mkl.c` | OK | 14.675 ms | 146.3 | 74% |
| julia | `matmul_julia_cpu.jl` | OK | 17.399 ms | 123.4 | 62% |
| cuda | `matmul_cuda.cu` | OK | 48.419 ms | 44.35 | 92% |
| cuda | `matmul_cublas_c11.cu` | OK | 48.747 ms | 44.05 | 92% |
| julia | `matmul_cublas_julia.jl` | OK | 48.765 ms | 44.04 | 92% |
| cpp26 | `matmul_cpp26_cuda.cpp` | OK | 48.789 ms | 44.00 | 91% |
| cpp26 | `matmul_cpp26_cuda.cpp` | OK | 48.785 ms | 44.00 | 91% |
| cuda | `matmul_cuda_best.cu` | OK | 48.777 ms | 44.00 | 91% |
| rust | `matmul_rust_cublas.rs` | OK | 48.772 ms | 44.00 | 91% |
| rust | `matmul_rust_cublas.rs` | OK | 48.782 ms | 44.00 | 91% |
| cpp26 | `matmul_cpp26_cuda.cpp` | OK | 49.167 ms | 43.70 | 91% |
| julia | `matmul_julia_gpu.jl` | OK | 49.480 ms | 43.40 | 90% |
| cuda | `matmul_cublas_cpp20.cu` | OK | 49.692 ms | 43.22 | 90% |
| cuda | `matmul_cuda_best.cu` | OK | 52.501 ms | 40.90 | 85% |
| kokkos | `matmul_kokkos.cpp` | OK | 58.862 ms | 36.48 | 76% |
| cpu_c | `matmul_c11_openblas.c` | OK | 60.000 ms | 35.54 | 18% |
| julia | `matmul_julia_gpu_vendor_agnostic.jl` | OK | 65.643 ms | 32.71 | 68% |
| cuda | `matmul_cuda_cpp20_faster.cu` | OK | 67.692 ms | 31.72 | 66% |
| cuda | `matmul_cuda_faster.cu` | OK | 67.732 ms | 31.71 | 66% |
| julia | `matmul_julia_gpu_faster.jl` | OK | 85.922 ms | 24.99 | 52% |
| cpu_c | `matmul_c11_parallel_loops.c` | OK | 99.000 ms | 21.63 | 11% |
| cpu_c | `matmul_c11_index_order.c` | OK | 459.000 ms | 4.68 | 2% |
| cpu_c | `matmul_c23.c` | OK | 1.8940 s | 1.13 | 1% |
| cpu_c | `matmul_c11.c` | OK | 1.9130 s | 1.12 | 1% |
| cpu_c | `matmul_c99.c` | OK | 1.9480 s | 1.10 | 1% |
| cpu_c | `matmul_c11.c` | OK | 1.9550 s | 1.10 | 1% |
| cpu_c | `matmul_c99.c` | OK | 1.9650 s | 1.09 | 1% |
| cpu_c | `matmul_c23.c` | OK | 1.9790 s | 1.09 | 1% |
| cpu_c | `matmul_c23.c` | OK | 2.4940 s | 0.86 | 0% |
| cpu_c | `matmul_c11.c` | OK | 2.5140 s | 0.85 | 0% |
| cpu_c | `matmul_c99.c` | OK | 2.5690 s | 0.84 | 0% |
| cuda | `matmul_cuda.cu` | OK | 2.8548 s | 0.75 | 0% |
| demos | `vector_add_v1.cu` | OK |  |  |  |
| demos | `vector_add_v2.cu` | OK |  |  |  |
| demos | `whoami_cuda.cu` | OK |  |  |  |

<!-- RESULTS:END -->

The suite runs every benchmark with the same size, run count and precision
(**Float64, N=1024, 10 runs**), so the numbers are directly comparable.  See
[`matmul_optimization/README.md`](matmul_optimization/README.md) for the
cross-language DGEMM comparison: on this FP64 workload Intel MKL (148.5 GFLOPS,
75 % of the CPU's FP64 peak) and Julia's CPU BLAS (123.4) lead, while cuBLAS
reaches ~44 GFLOPS (92 % of the GPU's FP64 peak) whether called from C, C++26,
Rust or Julia.  The suite also includes a C++ Kokkos DGEMM (the same source runs
on the CUDA backend here — 36.5 GFLOPS on this GPU — or on OpenMP/Serial/HIP/SYCL
builds) and a Julia kernel written with KernelAbstractions.jl that runs
unchanged on any GPU vendor's backend (register-blocked, 32.7 GFLOPS).  The GPU's
much lower FP64 ceiling is a Maxwell property (FP64 = 1/32 of FP32), not a kernel
problem.
