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
# The kernel is register-blocked: one team per 64 x 64 output tile, 16 x 16 =
# 256 work-items per team, and each work-item accumulates a 4 x 4 micro-tile in
# 16 scalar accumulators (scalars stay in registers; a KernelAbstractions
# `@private` array of this size spills to local memory and is ~5x slower).
# A and B are staged through shared memory in a layout that makes the staging
# stores conflict-free: As[BK, BM] (k fastest) and Bs[BN, BK] (n fastest).
#
# Run:  MATMUL_N=1024 MATMUL_RUNS=10 julia matmul_julia_gpu_vendor_agnostic.jl

using KernelAbstractions
using LinearAlgebra

const BM = 64    # block tile rows
const BN = 64    # block tile cols
const BK = 8     # block tile depth
const NTX = 16   # work-items along n (each owns 4 columns)
const NTY = 16   # work-items along m (each owns 4 rows)

const N = parse(Int, get(ENV, "MATMUL_N", "1024"))
const RUNS = parse(Int, get(ENV, "MATMUL_RUNS", "10"))

@kernel function tiled_matmul_kernel!(C, @Const(A), @Const(B))
    tx1, ty1 = @index(Local, NTuple)
    bm1, bn1 = @index(Group, NTuple)
    tx = tx1 - 1
    ty = ty1 - 1
    bm = bm1 - 1
    bn = bn1 - 1
    m0 = bm * 64
    n0 = bn * 64
    tid = ty * 16 + tx

    As = @localmem eltype(C) (8, 64)   # [k, m]
    Bs = @localmem eltype(C) (64, 8)   # [n, k]

    c00 = zero(eltype(C)); c01 = zero(eltype(C)); c02 = zero(eltype(C)); c03 = zero(eltype(C))
    c10 = zero(eltype(C)); c11 = zero(eltype(C)); c12 = zero(eltype(C)); c13 = zero(eltype(C))
    c20 = zero(eltype(C)); c21 = zero(eltype(C)); c22 = zero(eltype(C)); c23 = zero(eltype(C))
    c30 = zero(eltype(C)); c31 = zero(eltype(C)); c32 = zero(eltype(C)); c33 = zero(eltype(C))

    n = size(C, 1)
    NK = div(n + 7, 8)
    for kk in 0:(NK - 1)
        # Stage A (conflict-free: consecutive work-items write consecutive k).
        for l in 0:1
            idx = tid + l * 256
            mm = div(idx, 8)
            k = idx % 8
            gr = m0 + mm + 1
            gc = kk * 8 + k + 1
            As[k + 1, mm + 1] = (gr <= n && gc <= n) ? A[gr, gc] : zero(eltype(C))
        end
        # Stage B (conflict-free: consecutive work-items write consecutive n).
        for l in 0:1
            idx = tid + l * 256
            k = div(idx, 64)
            cc = idx % 64
            gr = kk * 8 + k + 1
            gc = n0 + cc + 1
            Bs[cc + 1, k + 1] = (gr <= n && gc <= n) ? B[gr, gc] : zero(eltype(C))
        end
        @synchronize

        for k in 1:8
            a0 = As[k, ty * 4 + 1]; a1 = As[k, ty * 4 + 2]
            a2 = As[k, ty * 4 + 3]; a3 = As[k, ty * 4 + 4]
            b0 = Bs[tx * 4 + 1, k]; b1 = Bs[tx * 4 + 2, k]
            b2 = Bs[tx * 4 + 3, k]; b3 = Bs[tx * 4 + 4, k]
            c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3
            c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3
            c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3
            c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3
        end
        @synchronize
    end

    r = m0 + ty * 4
    c = n0 + tx * 4
    if r + 1 <= n && c + 1 <= n; C[r + 1, c + 1] = c00; end
    if r + 1 <= n && c + 2 <= n; C[r + 1, c + 2] = c01; end
    if r + 1 <= n && c + 3 <= n; C[r + 1, c + 3] = c02; end
    if r + 1 <= n && c + 4 <= n; C[r + 1, c + 4] = c03; end
    if r + 2 <= n && c + 1 <= n; C[r + 2, c + 1] = c10; end
    if r + 2 <= n && c + 2 <= n; C[r + 2, c + 2] = c11; end
    if r + 2 <= n && c + 3 <= n; C[r + 2, c + 3] = c12; end
    if r + 2 <= n && c + 4 <= n; C[r + 2, c + 4] = c13; end
    if r + 3 <= n && c + 1 <= n; C[r + 3, c + 1] = c20; end
    if r + 3 <= n && c + 2 <= n; C[r + 3, c + 2] = c21; end
    if r + 3 <= n && c + 3 <= n; C[r + 3, c + 3] = c22; end
    if r + 3 <= n && c + 4 <= n; C[r + 3, c + 4] = c23; end
    if r + 4 <= n && c + 1 <= n; C[r + 4, c + 1] = c30; end
    if r + 4 <= n && c + 2 <= n; C[r + 4, c + 2] = c31; end
    if r + 4 <= n && c + 3 <= n; C[r + 4, c + 3] = c32; end
    if r + 4 <= n && c + 4 <= n; C[r + 4, c + 4] = c33; end
end

# Pick the first installed GPU backend.  The imports happen at top level (as
# each `const` initialiser is evaluated) so the backend methods are visible in
# `main` without world-age surprises; only the vendor-neutral kernel body above
# is shared.
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

    kernel! = tiled_matmul_kernel!(backend, (NTX, NTY))
    ngm = cld(N, BM)
    ngn = cld(N, BN)
    launch = () -> kernel!(C, A, B, ndrange = (ngm * NTX, ngn * NTY))

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
