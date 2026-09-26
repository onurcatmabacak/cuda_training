# matmul_optimization

A self-contained matrix-multiplication benchmark suite (CPU C, Intel MKL,
CUDA/cuBLAS, C++26, Rust, Kokkos, Julia).  One script builds and runs everything
and produces a combined report.

## Quick start

```bash
./run_all.sh                 # full suite, every benchmark's own default size/runs
./run_all.sh --quick         # smoke test: force 512x512, 3 runs, ~1-3 min
./run_all.sh --only cuda     # GPU benchmarks only
./run_all.sh --only cpu_c,cpu_mkl
./run_all.sh --skip julia,demos
./run_all.sh --size 2048 --runs 20 --threads 8   # override every benchmark
./run_all.sh --build-only    # compile everything, run nothing
```

Each run writes to `results/run_<timestamp>/` and points `results/latest` at it:

```
results/run_20250921_120000/
├── manifest.tsv        # name, category, status, rc, wall time, output path
├── config.txt          # size / runs / threads / timeout used
├── FULL_REPORT.md      # EVERYTHING in one file: table + raw output of every benchmark
├── summary.md          # combined Markdown report (all-in-one table + detail)
├── all_results.md      # just the single all-results table
├── summary.csv         # same data, one row per benchmark (spreadsheet-friendly)
├── cpu_c__matmul_c11_O3.txt
├── cuda__matmul_cublas_c11.txt
└── ...
```

**One file with everything:** `RESULTS.md` in this directory is rewritten after every
`run_all.sh` run (a copy of `results/latest/FULL_REPORT.md`). It contains the run
configuration, the single all-results table, the per-category detail tables, and the raw
stdout of every benchmark. For a spreadsheet, use `results/latest/summary.csv`.

## Latest results

_Measured on an Acer laptop (Intel i7-6700HQ 4c/8t, NVIDIA GTX 960M / sm_50). GPU
rows warm up for 2 s so they measure at boost clocks. CPU rows depend on the machine's
power state — this unit currently caps the CPU near 900 MHz, so CPU GFLOPS sit well
below the chip's peak._

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
| kokkos | `matmul_kokkos.cpp` | OK | 58.882 ms | 36.47 | max rel err 1.185e-16; OK; Kokkos Cuda |
| julia | `matmul_julia_gpu_vendor_agnostic.jl` | OK | 64.186 ms | 33.46 | max rel err 0.0; OK; backend CUDA |
| cuda | `matmul_cuda_cpp20_faster.cu` | OK | 67.667 ms | 31.74 | max rel err 4.38591e-16; OK |
| cuda | `matmul_cuda_faster.cu` | OK | 67.670 ms | 31.73 | max rel err 4.386e-16; OK |
| cpu_mkl | `matmul_c11_intel_mkl.c` | OK | 69.251 ms | 31.01 | C=251.536653 |
| cpu_mkl | `matmul_c11_intel_mkl.c` | OK | 70.236 ms | 30.58 | C=250.902362 |
| cpu_mkl | `matmul_c11_intel_mkl.c` | OK | 70.634 ms | 30.40 | C=257.630088 |
| cpu_mkl | `matmul_cpp20_intel_mkl.cpp` | OK | 72.403 ms | 29.66 | C=254.863 |
| cpu_mkl | `matmul_c11_intel_mkl.c` | OK | 72.687 ms | 29.54 | C=257.502524 |
| julia | `matmul_julia_cpu.jl` | OK | 82.195 ms | 26.13 | C=256.53110531394475 |
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

## Sizes, runs and precision

Every benchmark uses the **same matrix size, run count and precision** so the
numbers are directly comparable: **`N = 1024`, 10 runs, Float64 throughout**.
Override with `--size N` and `--runs R` (the C/CUDA sources read
`-DN`/`-DRUNS`, Julia reads `MATMUL_N`/`MATMUL_RUNS`); `--quick` forces
`N=512`, 3 runs.  A full run takes about 20 minutes here, most of it the
pure-C `-O1/-O2/-O3` sweep and the Julia JIT.

## Options

| Option | Meaning |
|--------|---------|
| `--only CATS` | comma-separated categories to run |
| `--skip CATS` | comma-separated categories to skip |
| `--quick` | force `N=512`, 3 runs, 300 s timeout |
| `--size N` | matrix dimension for every benchmark (default 1024) |
| `--runs R` | timed runs for every benchmark (default 10) |
| `--threads T` | BLAS/OpenMP threads (default `nproc`) |
| `--timeout SEC` | per-benchmark timeout, `0` disables (default 1800) |
| `--cuda-arch ARCH` | CUDA gencode target (default `sm_50`, the GTX 960M) |
| `--build-only` | compile but do not run |

Categories: `cpu_c`, `cpu_mkl`, `cuda`, `cpp26`, `rust`, `kokkos`, `julia`, `demos`.

## Layout

```
matmul_optimization/
├── run_all.sh              single entry point
├── src/
│   ├── cpu_c/              gcc: pure C + OpenBLAS
│   ├── cpu_mkl/            icx/icpx: Intel MKL DGEMM
│   ├── cuda/               nvcc: naive, tiled, cuBLAS, register-tiled
│   ├── cpp26/              C++26 host + CUDA (cuBLAS / cuBLASLt / graph)
│   ├── rust/               Rust + CUDA (FFI to cudart / cuBLAS)
│   ├── kokkos/             C++ Kokkos DGEMM (vendor-neutral: OpenMP/CUDA/HIP/SYCL)
│   ├── julia/              Julia CPU BLAS + CUDA + vendor-agnostic (KernelAbstractions)
│   └── demos/              whoami / vector_add teaching programs
├── scripts/                per-category runners + summarizer
├── bin/                    compiled binaries (generated)
├── build/                  compiler logs + .optrpt reports (generated)
├── results/                per-run results (generated) + results/legacy/
└── legacy/                 the original scattered one-off scripts
```

## What runs in each category

| Category | Benchmarks |
|----------|------------|
| `cpu_c` | pure C triple loop `c99/c11/c23` at `-O1/-O2/-O3`; C11 `(i,k,j)` index order; OpenMP blocked/recursive; OpenBLAS DGEMM |
| `cpu_mkl` | MKL C11 DGEMM under the four original `icx` flag sets; MKL C++20 DGEMM with NUMA-first-touch |
| `cuda` | naive CUDA kernel vs CPU; tiled shared-memory kernel (C and C++20); cuBLAS DGEMM (C11 and C++20); hand-written register-tiled DGEMM |
| `cpp26` | C++26 host + CUDA: cuBLAS, cuBLASLt and CUDA-graph DGEMM |
| `rust` | Rust + CUDA via direct FFI: cuBLAS and CUDA-graph DGEMM |
| `kokkos` | C++ Kokkos DGEMM (naive + shared-memory tiled + register-blocked); builds against whatever backend Kokkos ships with |
| `julia` | CPU BLAS; cuBLAS DGEMM (Float64); hand-written tiled CUDA kernel (Float64); vendor-agnostic register-blocked kernel via KernelAbstractions.jl |
| `demos` | `whoami`, `vector_add_v1`, `vector_add_v2` |

## Best DGEMM across languages (Float64, N=1024)

Every entry computes the same Float64 product at N=1024 over 10 runs, so the
numbers are directly comparable.  Values from the latest run; the laptop GPU
throttles, so repeat runs vary.

| Implementation | Language | Technique | GFLOPS |
|---|---|---|---|
| naive kernel (1 thread/output) | CUDA C++ | hand-written | 44.4 |
| `cublasDgemm` | C / C++ | NVIDIA cuBLAS | 44.1 |
| `mul!` (cuBLAS DGEMM) | Julia | NVIDIA cuBLAS | 44.1 |
| `cublasDgemm` | **C++26** | NVIDIA cuBLAS | 44.0 |
| CUDA graph of `cublasDgemm` | **C++26** | CUDA graph | 43.3 |
| `cublasDgemm` | **Rust** (FFI) | NVIDIA cuBLAS | 43.3 |
| `cublasLtMatmul` | **C++26** | NVIDIA cuBLASLt | 43.1 |
| CUDA graph of `cublasDgemm` | **Rust** (FFI) | CUDA graph | 42.8 |
| register-tiled kernel (4×4/thread) | CUDA C++ | hand-written | 40.9 |
| register-tiled kernel (16×4/thread) | **C++** (Kokkos) | hand-written (CUDA backend) | 36.5 |
| register-tiled kernel (4×4/work-item) | **Julia** (KernelAbstractions) | hand-written, vendor-agnostic | 33.5 |
| tiled kernel (1 output/thread) | CUDA C++ | hand-written | 31.7 |
| Intel MKL DGEMM | C (icx) | Intel MKL | 31.0 |
| CPU BLAS | Julia | OpenBLAS | 26.1 |
| tiled kernel (1 output/thread) | Julia (CUDA.jl) | hand-written | 25.6 |

- At N=1024 in Float64 the GPU work is small, so it is launch/bandwidth-bound
  and everything clusters in the 26–44 GFLOPS band: **cuBLAS ≈ naive ≈
  register-tiled**.
- **C++26 and Rust both reach cuBLAS speed** — when the work is delegated to
  the vendor library the host language does not matter (cuBLAS performs
  identically whether called from C, C++26, Rust or Julia).
- **Register blocking moves the portable kernels into the same band.**  The
  Kokkos kernel now uses a 128×128 tile with a 16×4 micro-tile (36.5 GFLOPS)
  and the vendor-agnostic KernelAbstractions kernel a 64×64 tile with a 4×4
  micro-tile (33.5 GFLOPS) — both far above their one-output-per-thread
  versions (26.1 and 26.8).
- **The vendor-agnostic Julia kernel beats the CUDA.jl one** (33.5 vs 25.6)
  with no CUDA-specific syntax; the same source runs on AMDGPU / oneAPI / Metal.
  In KernelAbstractions the micro-tile must be written as **16 scalar
  accumulators**: a `@private` array of the same size spills to local memory
  and runs ~5x slower.
- **The Kokkos DGEMM runs on the CUDA backend** (Kokkos 4.2.1, `MAXWELL50`).
  The same source falls back to a naive kernel when Kokkos is built for a CPU
  backend (or the OpenMP team is too small to tile).
- CUDA graphs and cuBLASLt give no measurable gain at this size; the kernel
  dominates the ~49 ms per call.
- Everything uses the same size, run count and precision, so the comparison is
  apples-to-apples.

## Environment and current caveats

* **CPU C** needs `gcc` (and `libopenblas-dev` for the OpenBLAS benchmark).
* **`cpu_mkl`** needs Intel oneAPI. The runner sources
  `~/intel/oneapi/setvars.sh` automatically (override with `$ONEAPI_SETVARS`).
  MKL is linked **LP64** (`-lmkl_intel_lp64`) consistently.
  *Note:* oneAPI's `setvars.sh` exports generic variables such as `BIN_DIR`;
  the suite namespaces its own variables (`MM_*`) to stay safe against that.
* **`cuda`/`demos`** need `nvcc` and a GPU; default gencode is `sm_50`
  (GTX 960M). The system toolkit here is CUDA 12.0, which still supports sm_50.
  Categories are skipped gracefully if the tool or GPU is missing.
* **`cpp26`** needs a C++26 compiler (nvcc 12.0 only goes up to C++20).  This
  box uses a user-local `g++-14` at `~/opt/cxx26/usr/bin/g++-14` (override with
  `$CXX26`); it compiles the host as C++26 and links the CUDA runtime + cuBLAS
  directly.  Skipped gracefully if absent.
* **`rust`** needs `rustc` at `~/opt/rust/bin/rustc` (override with `$RUSTC`).
  No crates are used — the program binds `cudart` and `cuBLAS` by FFI, so it
  works without crates.io access (only the `_v2` cuBLAS symbols are referenced).
* **`kokkos`** needs a Kokkos installation.  It is optional: if no Kokkos is
  found the category is skipped.  Preferred build path: when CMake and a
  Kokkos CMake package are available the runner uses `find_package(Kokkos)`,
  so the backend and its compiler (e.g. `nvcc_wrapper` for CUDA) are selected
  automatically.  Point it at a build with `KOKKOS_ROOT=/path/to/kokkos` or
  `Kokkos_DIR=/path/to/kokkos/lib/cmake/Kokkos`; if `cmake` is not on `PATH`
  set `MATMUL_KOKKOS_CMAKE=/path/to/cmake`.  A direct-compile fallback covers
  CMake-less setups via `KOKKOS_CXXFLAGS`/`KOKKOS_LDFLAGS`/`KOKKOS_LIBS`.
  This box uses a user-local **Kokkos 4.2.1 CUDA build** (`MAXWELL50`, sm_50)
  at `~/opt/kokkos-cuda`, plus `~/opt/cmake` and `g++-12` as the nvcc host
  compiler.  Because the source is vendor-neutral, the *same* file runs on a
  Kokkos built for Serial, OpenMP, CUDA, HIP or SYCL.  The default kernel is
  register-blocked (128×128 tile, 16×4 micro-tile) when the backend can schedule
  a 256-thread team — any GPU; it falls back to the naive kernel on a small CPU
  OpenMP pool.  Note: Kokkos 4.2 defaults to
  `cudaMallocAsync`, which Maxwell does not support, so this build passes
  `-DKokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=OFF`.
* **`julia`** needs Julia with `CUDA` and `LinearAlgebra`; the vendor-agnostic
  GPU kernel additionally needs `KernelAbstractions`
  (`julia -e 'import Pkg; Pkg.add("KernelAbstractions")'`).  The generic kernel
  is skipped with a clear message if that package is missing.
  *Maxwell caveat:* the host driver is 580 / CUDA 13, which dropped sm_50,
  so CUDA.jl would otherwise download the CUDA 13 runtime and fail on the
  GTX 960M.  CUDA.jl is therefore pinned to the last Maxwell-capable runtime
  in the default Julia environment:
  `julia -e 'using CUDA; CUDA.set_runtime_version!(v"12.9")'` (writes
  `~/.julia/environments/v1.12/LocalPreferences.toml`; a new Julia process is
  required).  That selects the 12.9 runtime **and** a matching 12.9 compiler
  while the driver stays at 580, so all four Julia benchmarks run.  Undo with
  `CUDA.reset_runtime_version!()`.  `scripts/julia.sh` still runs a GPU
  pre-flight and falls back to `SKIPPED` if a setup cannot run CUDA.
  Note: Ubuntu's `nvidia-driver-570`/`575` packages are dummies that install
  `nvidia-driver-580`; a real 575 is only in NVIDIA's own repo.

## Parameterisation

`N` and the run count are overridable without editing sources:

* C/CUDA sources use `#ifndef N`/`#ifndef RUNS` guards, so they keep their own
  default unless `-DN=... -DRUNS=...` is passed.
* The sources that declare `N` as a `constexpr` (`matmul_cpp20_intel_mkl.cpp`,
  `matmul_cublas_cpp20.cu`, `matmul_cuda_cpp20_faster.cu`) use
  `-DMATMUL_N=... -DMATMUL_RUNS=...`.
* `matmul_c11_openblas.c` and `matmul_c11_intel_mkl.c` define `N` *after*
  including `<cblas.h>`, because that header names a `cblas_dgemm` parameter
  `N`; passing `-DN` directly would break the prototype.
* Julia scripts read `MATMUL_N`, `MATMUL_RUNS`, `MATMUL_THREADS`.
* The Kokkos source uses `-DMATMUL_N=... -DMATMUL_RUNS=...` (the macro is not
  named `N` because Kokkos headers use `N` as a template parameter).

## Notes on the original scripts (now in `legacy/`)

* `matmul.sh` and `matmul_c11_parallel_loops.sh` used `-o1/-o2/-o3` (lowercase
  `o`), which GCC treats as an output-file name, so those runs were actually
  **unoptimised**. The suite uses the intended `-O1/-O2/-O3`.
* Output destinations are centralised under `results/run_<stamp>/` instead of
  scattering `*.txt` files next to the sources.
* The old results are preserved under `results/legacy/` and the old one-off
  scripts under `legacy/`.
