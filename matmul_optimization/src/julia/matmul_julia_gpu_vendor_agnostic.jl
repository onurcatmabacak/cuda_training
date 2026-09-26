# matmul_julia_gpu_vendor_agnostic.jl -- vendor-agnostic GPU DGEMM (Float64).
#
# The kernel is written once with KernelAbstractions.jl and runs unchanged on
# any GPU backend (CUDA, AMDGPU, oneAPI, Metal, ...).  The script discovers
# whichever backend package is installed and uses its `*Backend()` object; the
# algorithmic code below is identical for all of them.
#
#   * @index(Group, i) / @index(Local, i)  -- portable block/thread indices
#   * @localmem                             -- portable shared memory
#   * @synchronize                          -- portable barrier
#
# The workgroup is TILE x TILE (32 x 32 = 1024 work-items), one team per output
# tile, staging A and B through shared memory exactly like the hand-written CUDA
# kernel -- but with no CUDA-specific syntax.
#
# Run:  MATMUL_N=1024 MATMUL_RUNS=10 julia matmul_julia_gpu_vendor_agnostic.jl

using KernelAbstractions
using LinearAlgebra

const TILE = 32
const N = parse(Int, get(ENV, "MATMUL_N", "1024"))
const RUNS = parse(Int, get(ENV, "MATMUL_RUNS", "10"))

@kernel function tiled_matmul_kernel!(
        C, @Const(A), @Const(B), ::Val{BANK} = Val(0)) where {BANK}
    # Portable block/thread indices, shared memory, private accumulator and
    # barrier -- all vendor-neutral KernelAbstractions constructs.
    gi, gj = @index(Group, NTuple)
    i, j = @index(Local, NTuple)
    T = @uniform @groupsize()[1]

    # +1 padding avoids shared-memory bank conflicts (BANK = 0 by default).
    t1 = @localmem eltype(C) (T + BANK, T)
    t2 = @localmem eltype(C) (T + BANK, T)
    acc = @private eltype(C) 1
    @inbounds acc[1] = zero(eltype(C))

    @uniform n = size(C, 1)
    @uniform NT = div(n + T - 1, T)
    for t in 0:(NT - 1)
        I = (gi - 1) * T + i
        J = (gj - 1) * T + j
        if I <= n && t * T + j <= n
            @inbounds t1[i, j] = A[I, t * T + j]
        else
            @inbounds t1[i, j] = zero(eltype(C))
        end
        if t * T + i <= n && J <= n
            @inbounds t2[i, j] = B[t * T + i, J]
        else
            @inbounds t2[i, j] = zero(eltype(C))
        end
        @synchronize

        I = (gi - 1) * T + i
        J = (gj - 1) * T + j
        out = zero(eltype(C))
        @simd for k in 1:T
            @inbounds out += t1[i, k] * t2[k, j]
        end
        acc[1] += out
        @synchronize
    end

    I = (gi - 1) * T + i
    J = (gj - 1) * T + j
    if I <= n && J <= n
        @inbounds C[I, J] = acc[1]
    end
end

# Pick the first installed GPU backend.  The imports happen at top level (as
# each `const` initialiser is evaluated) so the backend methods are visible in
# `main` without world-age surprises; only the vendor-neutral kernel body below
# is shared.  Pairs are (package, backend constructor).
function _try_import(pkg::Symbol)
    try
        Core.eval(Main, :(import $pkg))
        return true
    catch
        return false
    end
end

const HAVE_CUDA   = _try_import(:CUDA)
const HAVE_AMDGPU = _try_import(:AMDGPU)
const HAVE_ONEAPI = _try_import(:oneAPI)
const HAVE_METAL  = _try_import(:Metal)

function pick_backend()
    if HAVE_CUDA
        return CUDA.CUDABackend(), "CUDA"
    elseif HAVE_AMDGPU
        return AMDGPU.ROCBackend(), "AMDGPU"
    elseif HAVE_ONEAPI
        return oneAPI.oneAPIBackend(), "oneAPI"
    elseif HAVE_METAL
        return Metal.MetalBackend(), "Metal"
    end
    return nothing, nothing
end

function main()
    backend, vendor = pick_backend()
    if backend === nothing
        println("No vendor-agnostic GPU backend available (tried CUDA, ",
                "AMDGPU, oneAPI, Metal)")
        println("Result: SKIPPED")
        return
    end
    println("Matrix size: $(N)x$(N) (Float64)")
    println("Vendor-agnostic backend: $vendor via KernelAbstractions")

    A = KernelAbstractions.allocate(backend, Float64, N, N)
    B = KernelAbstractions.allocate(backend, Float64, N, N)
    C = KernelAbstractions.allocate(backend, Float64, N, N)
    copyto!(A, rand(Float64, N, N))
    copyto!(B, rand(Float64, N, N))
    fill!(C, 0.0)

    kernel! = tiled_matmul_kernel!(backend, (TILE, TILE))
    launch = () -> kernel!(C, A, B, ndrange = size(C))

    launch()
    KernelAbstractions.synchronize(backend)

    # Keep the GPU busy until it boosts its clocks.
    t0 = time()
    while time() - t0 < 2.0
        launch()
        KernelAbstractions.synchronize(backend)
    end

    total = 0.0
    for _ in 1:RUNS
        t = @elapsed begin
            launch()
            KernelAbstractions.synchronize(backend)
        end
        total += t
    end
    avg_ms = total / RUNS * 1000.0
    gflops = 2.0 * N^3 / (avg_ms * 1e6)

    # Sample correctness check against a host reference.
    Ah = Array(A)
    Bh = Array(B)
    Ch = Array(C)
    maxrel = 0.0
    for (i, j) in ((1, 1), (2, 3), (N ÷ 2, N ÷ 3), (N, N))
        ref = 0.0
        @inbounds for k in 1:N
            ref += Ah[i, k] * Bh[k, j]
        end
        maxrel = max(maxrel, abs(ref - Ch[i, j]) / (abs(ref) + 1e-12))
    end

    println("Average kernel time (ms): $avg_ms")
    println("Effective GFLOPS: $gflops")
    println("Max relative error vs CPU: $maxrel")
    println("Result: ", maxrel < 1e-9 ? "OK" : "MISMATCH")
end

main()
