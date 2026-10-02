---
name: "matmul-optimization-suite"
description: "Run or extend the matmul_optimization benchmark suite under cuda_training."
version: 3
created: "2026-09-21"
updated: "2026-09-21"
---
## When to Use
Use when asked to run, extend, or debug the CPU/GPU matrix-multiplication benchmarks in cuda_training/matmul_optimization (pure C, Intel MKL, CUDA/cuBLAS, Julia).

## Procedure
1. From matmul_optimization/ run ./run_all.sh for the full suite (each benchmark uses its own default size/runs) or ./run_all.sh --quick for a 512x512x3-run smoke test. Select categories with --only/--skip (cpu_c, cpu_mkl, cuda, julia, demos); use --build-only to compile only and --size/--runs to override every benchmark.
2. Read results/latest/summary.md (Markdown tables per category; also summary.csv and manifest.tsv). Raw per-benchmark stdout is in results/latest/<category>__<name>.txt.
3. Add a benchmark by dropping the source in src/<category>/ and adding one bench_build_run call in scripts/<category>.sh: bench_build_run "<cat>__<name>" "<cat>" ::: <compile args> ::: <run command>. A build failure is recorded as BUILD_FAIL and does not abort the suite.
4. Parameterise sizes with #ifndef N/#ifndef RUNS guards (built via -DN/-DRUNS), or -DMATMUL_N/-DMATMUL_RUNS for sources that declare N as constexpr. Julia scripts read MATMUL_N/MATMUL_RUNS/MATMUL_THREADS from the environment.
5. For sources that #include <cblas.h> (OpenBLAS/MKL), define N from a differently-named macro AFTER the include; never pass -DN on the command line.

## Pitfalls
- oneAPI setvars.sh exports BIN_DIR=bin64, clobbering generic build-path variables; keep the suite's internal variables namespaced (MM_*).
- cblas.h declares a cblas_dgemm parameter named N, so -DN=<n> breaks compilation; use -DMATMUL_N and map it after the include.
- CUDA work is asynchronous: any GPU timing must synchronize before stopping the clock, or it measures only kernel-launch overhead. Julia's `C .= A*B` needs an explicit `synchronize()`; prefer `mul!`.
- The GTX 960M only boosts clocks under sustained load (idle P8/135 MHz vs P0/1202 MHz), so short GPU kernels with few runs read ~8x slow. Warm up ~1-2 s of GPU work or lock clocks before timing, and don't compare GPU rows across different run counts.
- The pure-C naive kernels are only feasible near N=1024; leave N/RUNS unset so each source keeps its own default unless the user overrides.
- GPU driver 580 is CUDA 13 which dropped Maxwell (sm_50). CUDA.jl is pinned to v12.9 via CUDA.set_runtime_version! in the default Julia env so the GTX 960M works; scripts/julia.sh preflights and records SKIPPED if a setup cannot run CUDA. Ubuntu's nvidia-driver-470/535/550/570/575 are dummy packages that install 580.
- A full default run takes roughly an hour (and the two tiled CUDA programs each spend ~25 min in a single-threaded N=4096 CPU reference); iterate with --quick/--only/--runs and --timeout.
## Verification
1. ./run_all.sh --quick exits 0 and results/latest/summary.md shows OK for cpu_c/cpu_mkl/cuda/demos and SKIPPED for the three Julia GPU benchmarks.
2. ./run_all.sh --help prints the documented option list.
3. bash -n passes on run_all.sh and scripts/*.sh, and python3 -m py_compile scripts/summarize.py succeeds.