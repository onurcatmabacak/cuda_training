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

_Measured on an Acer laptop (Intel i7-6700HQ 4c/8t, NVIDIA GTX 960M / sm_50). This run
was made from a bare TTY3 session with no desktop (gdm stopped), so the CPU is no longer
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
numbers are directly comparable.  Values are the best observed for each
implementation; the laptop GPU throttles, so a full-suite run can read lower for
the GPU benchmarks that run late.  `% peak` compares each measurement with the
theoretical FP64 peak of the engine it ran on (CPU **198.4**, GPU **48.1**
GFLOPS — see *Why the CPU beats the GPU in FP64* below).

| Implementation | Language | Technique | GFLOPS | % peak |
|---|---|---|---|---|
| Intel MKL DGEMM | C / C++ (oneAPI) | Intel MKL | 148.5 | 75% |
| CPU BLAS | Julia | OpenBLAS | 123.4 | 62% |
| naive kernel (1 thread/output) | CUDA C++ | hand-written | 44.4 | 92% |
| `cublasDgemm` | C / C++ | NVIDIA cuBLAS | 44.1 | 92% |
| `mul!` (cuBLAS DGEMM) | Julia | NVIDIA cuBLAS | 44.0 | 92% |
| `cublasDgemm` | **C++26** | NVIDIA cuBLAS | 44.0 | 91% |
| CUDA graph of `cublasDgemm` | **C++26** | CUDA graph | 44.0 | 91% |
| `cublasDgemm` | **Rust** (FFI) | NVIDIA cuBLAS | 44.0 | 91% |
| CUDA graph of `cublasDgemm` | **Rust** (FFI) | CUDA graph | 44.0 | 91% |
| `cublasLtMatmul` | **C++26** | NVIDIA cuBLASLt | 43.7 | 91% |
| register-tiled kernel (4×4/thread) | CUDA C++ | hand-written | 40.9 | 85% |
| register-tiled kernel (16×4/thread) | **C++** (Kokkos) | hand-written (CUDA backend) | 36.5 | 76% |
| register-tiled kernel (4×4/work-item) | **Julia** (KernelAbstractions) | hand-written, vendor-agnostic | 32.7 | 68% |
| tiled kernel (1 output/thread) | CUDA C++ | hand-written | 31.7 | 66% |
| tiled kernel (1 output/thread) | Julia (CUDA.jl) | hand-written | 25.0 | 52% |

- At N=1024 in Float64 the GPU work is small, so it is launch/bandwidth-bound
  and the GPU rows cluster in the 25–44 GFLOPS band: **cuBLAS ≈ naive ≈
  register-tiled**.  With the desktop gone the CPU is no longer power-capped,
  so MKL (148.5, 75 % of its FP64 peak) and Julia's CPU BLAS (123.4, 62 %)
  now sit well above every GPU row — see *Why the CPU beats the GPU in FP64*.
- **C++26 and Rust both reach cuBLAS speed** — when the work is delegated to
  the vendor library the host language does not matter (cuBLAS performs
  identically whether called from C, C++26, Rust or Julia).
- **Register blocking moves the portable kernels into the same band.**  The
  Kokkos kernel now uses a 128×128 tile with a 16×4 micro-tile (36.5 GFLOPS)
  and the vendor-agnostic KernelAbstractions kernel a 64×64 tile with a 4×4
  micro-tile (32.7 GFLOPS) — both far above their one-output-per-thread
  versions (26.1 and 26.8).
- **The vendor-agnostic Julia kernel beats the CUDA.jl one** (32.7 vs 25.0)
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

### Why the CPU beats the GPU in FP64

The two FP64 (Double) peaks on this laptop are wildly asymmetric:

| Engine | FP64 peak | How it is derived |
|---|---|---|
| Intel i7-6700HQ (4c/8t) | **198.4 GFLOPS** | 4 cores × 2 AVX2 FMA × 4 doubles × 2 FLOP/cycle × 3.1 GHz all-core turbo |
| NVIDIA GTX 960M (GM107, sm_50) | **48.1 GFLOPS** | 640 cores × 2 FLOP × 1.202 GHz × **1/32** (Maxwell runs FP64 at 1/32 of FP32) |

The GPU is not underperforming: cuBLAS DGEMM reaches 44.1 GFLOPS, i.e. **92 % of
the card's hardware FP64 peak**.  The CPU simply has ~4× the FP64 throughput of
this laptop GPU, because NVIDIA de-rates FP64 to 1/32 on consumer Maxwell while
Skylake issues 16 DP FLOPs per core per cycle through AVX2+FMA.  For scale, the
same GPU measured **961–1220 GFLOPS** in FP32 SGEMM with the old pre-DGEMM suite
— ~25× its FP64 number, exactly the 1/32 ratio.  FP64 is the one regime where a
2015 gaming-laptop CPU wins; in FP32/FP16 the GPU is several times faster.

The CPU peak is also thermal-state dependent: 3.1 GHz all-core turbo gives the
198.4 GFLOPS roofline, but sustained all-core load on this chassis settles at
~2.5–2.8 GHz (~160–180 GFLOPS).  The suite measures short bursts with a 20 s
cooldown before each benchmark, which keeps measurements near the turbo peak.

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
