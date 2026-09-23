# matmul_optimization

A self-contained matrix-multiplication benchmark suite (CPU C, Intel MKL,
CUDA/cuBLAS, Julia).  One script builds and runs everything and produces a
combined report.

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

_20260923_061531 — size **1024**, **10** runs, 8 threads, CUDA arch `sm_50`. GFLOPS is as reported, or computed as `2·N³/t` when only a time is printed._

| Category | Benchmark | Status | Avg time | GFLOPS | Notes |
|---|---|---|---|---|---|
| cuda | `cuda__matmul_cuda[gpu]` | OK | 48.455 ms | 44.32 |  |
| cuda | `cuda__matmul_cublas_c11` | OK | 48.770 ms | 44.03 | C=2073.633457 |
| cuda | `cuda__matmul_cublas_cpp20` | OK | 48.775 ms | 44.03 | C=2073.63 |
| julia | `julia__matmul_cublas_julia` | OK | 48.796 ms | 44.01 | C=251.84999342373902 |
| cpp26 | `cpp26__matmul_cpp26_cublas` | OK | 48.792 ms | 44.00 |  |
| cpp26 | `cpp26__matmul_cpp26_graph` | OK | 48.792 ms | 44.00 |  |
| cuda | `cuda__matmul_cublas_dgemm` | OK | 48.794 ms | 44.00 |  |
| julia | `julia__matmul_julia_gpu` | OK | 49.204 ms | 43.64 | C=259.04201903275134 |
| cpp26 | `cpp26__matmul_cpp26_cublaslt` | OK | 49.205 ms | 43.60 |  |
| rust | `rust__matmul_rust_cublas` | OK | 49.620 ms | 43.30 |  |
| rust | `rust__matmul_rust_graph` | OK | 49.641 ms | 43.30 |  |
| cuda | `cuda__matmul_cuda_optimized` | OK | 52.503 ms | 40.90 | max rel err 4.463e-05; OK |
| cuda | `cuda__matmul_cuda_cpp20_faster` | OK | 67.710 ms | 31.72 | max rel err 4.38591e-16; OK |
| cuda | `cuda__matmul_cuda_faster` | OK | 67.713 ms | 31.71 | max rel err 4.386e-16; OK |
| cpu_mkl | `cpu_mkl__matmul_c11_intel_mkl_flags4` | OK | 68.235 ms | 31.47 | C=255.081692 |
| cpu_mkl | `cpu_mkl__matmul_c11_intel_mkl_flags2` | OK | 70.729 ms | 30.36 | C=252.683208 |
| cpu_mkl | `cpu_mkl__matmul_c11_intel_mkl_flags3` | OK | 71.538 ms | 30.02 | C=267.236012 |
| cpu_mkl | `cpu_mkl__matmul_cpp20_intel_mkl` | OK | 72.768 ms | 29.51 | C=254.863 |
| cpu_mkl | `cpu_mkl__matmul_c11_intel_mkl_flags1` | OK | 73.558 ms | 29.19 | C=242.667056 |
| julia | `julia__matmul_julia_cpu` | OK | 82.524 ms | 26.02 | C=246.74955049655716 |
| julia | `julia__matmul_julia_gpu_faster` | OK | 84.019 ms | 25.56 | max rel err 0.0 |
| cpu_c | `cpu_c__matmul_c11_parallel_loops_O3` | OK | 360.000 ms | 5.96 |  |
| cpu_c | `cpu_c__matmul_c11_openblas_O3` | OK | 638.000 ms | 3.36 |  |
| cpu_c | `cpu_c__matmul_c11_index_order_O3` | OK | 1.2200 s | 1.76 |  |
| cpu_c | `cpu_c__matmul_c23_O2` | OK | 4.3160 s | 0.50 |  |
| cpu_c | `cpu_c__matmul_c99_O3` | OK | 4.3160 s | 0.50 |  |
| cpu_c | `cpu_c__matmul_c99_O2` | OK | 4.3280 s | 0.50 |  |
| cpu_c | `cpu_c__matmul_c23_O3` | OK | 4.3300 s | 0.50 |  |
| cpu_c | `cpu_c__matmul_c11_O3` | OK | 4.3750 s | 0.49 |  |
| cpu_c | `cpu_c__matmul_c11_O2` | OK | 4.3910 s | 0.49 |  |
| cuda | `cuda__matmul_cuda[cpu]` | OK | 5.8187 s | 0.37 | speedup 120.1x |
| cpu_c | `cpu_c__matmul_c11_O1` | OK | 5.9710 s | 0.36 |  |
| cpu_c | `cpu_c__matmul_c23_O1` | OK | 5.9870 s | 0.36 |  |
| cpu_c | `cpu_c__matmul_c99_O1` | OK | 6.0530 s | 0.35 |  |
| demos | `demos__vector_add_v1` | OK |  |  |  |
| demos | `demos__vector_add_v2` | OK |  |  |  |
| demos | `demos__whoami` | OK |  |  |  |

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

Categories: `cpu_c`, `cpu_mkl`, `cuda`, `cpp26`, `rust`, `julia`, `demos`.

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
│   ├── julia/              Julia CPU BLAS + CUDA
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
| `julia` | CPU BLAS; cuBLAS DGEMM (Float64); hand-written tiled CUDA kernel (Float64) |
| `demos` | `whoami`, `vector_add_v1`, `vector_add_v2` |

## Best DGEMM across languages (Float64, N=1024)

Every entry computes the same Float64 product at N=1024 over 10 runs, so the
numbers are directly comparable.  Values from the latest run; the laptop GPU
throttles, so repeat runs vary.

| Implementation | Language | Technique | GFLOPS |
|---|---|---|---|
| naive kernel (1 thread/output) | CUDA C++ | hand-written | 44.3 |
| `cublasDgemm` | C / C++ | NVIDIA cuBLAS | 44.0 |
| `cublasDgemm` | **C++26** | NVIDIA cuBLAS | 44.0 |
| CUDA graph of `cublasDgemm` | **C++26** | CUDA graph | 44.0 |
| `mul!` (cuBLAS DGEMM) | Julia | NVIDIA cuBLAS | 44.0 |
| `cublasLtMatmul` | **C++26** | NVIDIA cuBLASLt | 43.6 |
| `cublasDgemm` | **Rust** (FFI) | NVIDIA cuBLAS | 43.3 |
| CUDA graph of `cublasDgemm` | **Rust** (FFI) | CUDA graph | 43.3 |
| register-tiled kernel (4×4/thread) | CUDA C++ | hand-written | 40.9 |
| tiled kernel (1 output/thread) | CUDA C++ | hand-written | 31.7 |
| Intel MKL DGEMM | C (icx) | Intel MKL | 31.5 |
| CPU BLAS | Julia | OpenBLAS | 26.0 |
| tiled kernel (1 output/thread) | Julia (CUDA.jl) | hand-written | 25.6 |

- At N=1024 in Float64 the GPU work is small, so it is launch/bandwidth-bound
  and everything clusters in the 25–44 GFLOPS band: **cuBLAS ≈ naive ≈
  register-tiled**.
- **C++26 and Rust both reach cuBLAS speed** — when the work is delegated to
  the vendor library the host language does not matter (cuBLAS 12.0 performs
  identically whether called from C, C++26, Rust or Julia).
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
* **`julia`** needs Julia with `CUDA` and `LinearAlgebra`.
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

## Notes on the original scripts (now in `legacy/`)

* `matmul.sh` and `matmul_c11_parallel_loops.sh` used `-o1/-o2/-o3` (lowercase
  `o`), which GCC treats as an output-file name, so those runs were actually
  **unoptimised**. The suite uses the intended `-O1/-O2/-O3`.
* Output destinations are centralised under `results/run_<stamp>/` instead of
  scattering `*.txt` files next to the sources.
* The old results are preserved under `results/legacy/` and the old one-off
  scripts under `legacy/`.
