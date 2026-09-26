#!/usr/bin/env bash
# julia.sh -- Julia LinearAlgebra / CUDA benchmarks.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
ensure_dirs

MM_CAT="julia"

if ! have julia; then
  warn "julia not found -- skipping [$MM_CAT]"
  exit 0
fi

JL=(julia --startup-file=no)
JULIA_ENV=(env MATMUL_THREADS="$MATMUL_THREADS")
[[ -n "$MATMUL_N" ]]    && JULIA_ENV+=(MATMUL_N="$MATMUL_N")
[[ -n "$MATMUL_RUNS" ]] && JULIA_ENV+=(MATMUL_RUNS="$MATMUL_RUNS")

# 1) CPU BLAS (Float32, multi-threaded).
name="${MM_CAT}__matmul_julia_cpu"
run_bench "$name" "$MM_CAT" "${JULIA_ENV[@]}" "${JL[@]}" "$MM_SRC/julia/matmul_julia_cpu.jl"

# GPU benchmarks only make sense with a CUDA-capable device.
MM_JL_GPU_OK=0
if have_gpu; then
  # Pre-flight: CUDA.jl's bundled runtime must actually support this GPU.
  # (CUDA 13 dropped Maxwell/sm_50, so a GTX 960M fails at runtime even though
  #  CUDA.functional() still returns true.)
  preflight_timeout="${MATMUL_JULIA_PREFLIGHT_TIMEOUT:-600}"
  if timeout "$preflight_timeout" julia --startup-file=no -e \
      'using CUDA; CUDA.functional() || exit(3); x = CUDA.rand(Float32, 4); CUDA.synchronize(); exit(0)' \
      >"$MM_BUILD/julia_gpu_preflight.log" 2>&1; then
    MM_JL_GPU_OK=1
  else
    warn "Julia CUDA pre-flight failed -- skipping Julia GPU benchmarks"
    sed -n '1,8p' "$MM_BUILD/julia_gpu_preflight.log" >&2
  fi
fi

if [[ "$MM_JL_GPU_OK" == "1" ]]; then
  # 2) cuBLAS DGEMM (Float64) via Julia's `A * B`.
  name="${MM_CAT}__matmul_cublas_julia"
  run_bench "$name" "$MM_CAT" "${JULIA_ENV[@]}" "${JL[@]}" "$MM_SRC/julia/matmul_cublas_julia.jl"

  # 3) cuBLAS SGEMM (Float32) via Julia's `A * B`.
  name="${MM_CAT}__matmul_julia_gpu"
  run_bench "$name" "$MM_CAT" "${JULIA_ENV[@]}" "${JL[@]}" "$MM_SRC/julia/matmul_julia_gpu.jl"

  # 4) Hand-written tiled CUDA kernel (Float32).
  name="${MM_CAT}__matmul_julia_gpu_faster"
  run_bench "$name" "$MM_CAT" "${JULIA_ENV[@]}" "${JL[@]}" "$MM_SRC/julia/matmul_julia_gpu_faster.jl"

  # 5) Vendor-agnostic GPU kernel (KernelAbstractions.jl): the same @kernel
  #    runs on any GPU backend (CUDA / AMDGPU / oneAPI / Metal).
  if timeout "$preflight_timeout" julia --startup-file=no -e 'using KernelAbstractions' \
      >/dev/null 2>&1; then
    name="${MM_CAT}__matmul_julia_gpu_vendor_agnostic"
    run_bench "$name" "$MM_CAT" "${JULIA_ENV[@]}" "${JL[@]}" \
      "$MM_SRC/julia/matmul_julia_gpu_vendor_agnostic.jl"
  else
    record_skip "julia__matmul_julia_gpu_vendor_agnostic" "$MM_CAT" \
      "KernelAbstractions.jl not installed (julia -e 'import Pkg; Pkg.add(\"KernelAbstractions\")')"
  fi
else
  if have_gpu; then
    reason="CUDA runtime does not support this GPU (see build/julia_gpu_preflight.log)"
  else
    reason="no GPU"
  fi
  record_skip "julia__matmul_cublas_julia"     "$MM_CAT" "$reason"
  record_skip "julia__matmul_julia_gpu"        "$MM_CAT" "$reason"
  record_skip "julia__matmul_julia_gpu_faster" "$MM_CAT" "$reason"
  record_skip "julia__matmul_julia_gpu_vendor_agnostic" "$MM_CAT" "$reason"
fi
