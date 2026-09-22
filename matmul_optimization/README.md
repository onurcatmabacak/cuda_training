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

## Sizes and run counts

By default **each benchmark keeps its own built-in size and run count** — the
pure-C teaching kernels use `1024`/`100`, while MKL, CUDA and Julia use
`4096`/`100`.  `--size N` and `--runs R` override *everything* uniformly (the
C/C++/CUDA sources read `-DN`/`-DRUNS`, Julia reads `MATMUL_N`/`MATMUL_RUNS`).

The full run is long (~1 h) because several 4096×4096 benchmarks average over
100 runs.  Use `--quick`, `--only`, or lower `--runs` while iterating.

## Options

| Option | Meaning |
|--------|---------|
| `--only CATS` | comma-separated categories to run |
| `--skip CATS` | comma-separated categories to skip |
| `--quick` | force `N=512`, 3 runs, 300 s timeout |
| `--size N` | matrix dimension for every benchmark (default: per-benchmark) |
| `--runs R` | timed runs for every benchmark (default: per-benchmark) |
| `--threads T` | BLAS/OpenMP threads (default `nproc`) |
| `--timeout SEC` | per-benchmark timeout, `0` disables (default 1800) |
| `--cuda-arch ARCH` | CUDA gencode target (default `sm_50`, the GTX 960M) |
| `--build-only` | compile but do not run |

Categories: `cpu_c`, `cpu_mkl`, `cuda`, `julia`, `demos`.

## Layout

```
matmul_optimization/
├── run_all.sh              single entry point
├── src/
│   ├── cpu_c/              gcc: pure C + OpenBLAS
│   ├── cpu_mkl/            icx/icpx: Intel MKL DGEMM
│   ├── cuda/               nvcc: naive, tiled, cuBLAS
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
| `cuda` | naive CUDA kernel vs CPU; tiled shared-memory kernel (C and C++20); cuBLAS DGEMM (C11 and C++20) |
| `julia` | CPU BLAS; cuBLAS Float64; cuBLAS Float32; hand-written tiled CUDA kernel |
| `demos` | `whoami`, `vector_add_v1`, `vector_add_v2` |

## Best CUDA SGEMM: C++ vs Julia (Float32, N=4096)

Peak values observed on this GTX 960M. The GPU thermally throttles, so repeat
runs can read 30–50 % lower; treat these as ceilings, not averages.

| Implementation | Language | Technique | GFLOPS |
|---|---|---|---|
| `cublasSgemm` | C++ | NVIDIA cuBLAS | ~1220 |
| `mul!` (cuBLAS) | Julia | NVIDIA cuBLAS | ~970–1230 |
| register-tiled kernel (8×8/thread) | CUDA C++ | hand-written | ~600 |
| tiled kernel (1 output/thread) | CUDA C++ | hand-written | ~197 |
| tiled kernel (1 output/thread) | Julia (CUDA.jl) | hand-written | ~78 |
| register-tiled kernel (8×8/thread) | Julia (CUDA.jl) | hand-written | ~41 |
| naive kernel | CUDA C++ | hand-written | ~85 (N=1024) |

- **Best C++ and best Julia are the same number**, because Julia's fastest GPU
  path *is* cuBLAS: `mul!(C, A, B)` vs C's `cublasSgemm`, both ~1200 GFLOPS.
- A tuned hand-written C++ kernel reaches about half of cuBLAS; the naive one
  about 7 %.
- The same register-tiled algorithm in CUDA.jl runs ~15x slower — its compiler
  spills the 64-accumulator tile to local memory — so for Julia the vendor
  library is the only competitive route. That kernel is in
  `src/julia/matmul_julia_gpu_best.jl` (not in the default suite: ~4 min JIT).

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
