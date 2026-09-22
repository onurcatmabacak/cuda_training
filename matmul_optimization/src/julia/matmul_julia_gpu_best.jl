# matmul_julia_gpu_best.jl
# Hand-written register-tiled SGEMM in Julia (CUDA.jl), port of
# matmul_cuda_best.cu:  BM=BN=128, BK=32, TM=TN=8 per thread, 256 threads.
# Used as the "best custom CUDA Julia" counterpart to the C++ kernel.

using CUDA
using LinearAlgebra

const BM = 128
const BN = 128
const BK = 32
const TM = 8
const TN = 8
const NTHREADS = 256

function sgemm_regtile!(A::CuDeviceMatrix{Float32}, B::CuDeviceMatrix{Float32},
                        C::CuDeviceMatrix{Float32}, M::Int, N::Int, K::Int)
    tx = threadIdx().x - 1
    ty = threadIdx().y - 1
    tid = ty * blockDim().x + tx

    row0 = (blockIdx().y - 1) * BM
    col0 = (blockIdx().x - 1) * BN

    As = @cuStaticSharedMem(Float32, (BM, BK))
    Bs = @cuStaticSharedMem(Float32, (BK, BN))

    acc = zeros(Float32, TM, TN)

    k0 = 0
    while k0 < K
        # ---- load A tile (BM x BK), contiguous in K -> coalesced ----
        idx = tid
        while idx < BM * BK
            m = idx ÷ BK
            k = idx % BK
            gm = row0 + m + 1
            gk = k0 + k + 1
            @inbounds As[m + 1, k + 1] = (gm <= M && gk <= K) ? A[gm, gk] : 0.0f0
            idx += NTHREADS
        end
        # ---- load B tile (BK x BN), contiguous in N ----
        idx = tid
        while idx < BK * BN
            k = idx ÷ BN
            n = idx % BN
            gk = k0 + k + 1
            gn = col0 + n + 1
            @inbounds Bs[k + 1, n + 1] = (gk <= K && gn <= N) ? B[gk, gn] : 0.0f0
            idx += NTHREADS
        end
        sync_threads()

        @inbounds for k in 0:(BK - 1)
            a = zeros(Float32, TM)
            b = zeros(Float32, TN)
            for i in 0:(TM - 1)
                a[i + 1] = As[ty * TM + i + 1, k + 1]   # warp broadcast
            end
            for j in 0:(TN - 1)
                b[j + 1] = Bs[k + 1, tx + j * 16 + 1]   # stride-1, no conflict
            end
            for i in 0:(TM - 1)
                @simd for j in 0:(TN - 1)
                    acc[i + 1, j + 1] += a[i + 1] * b[j + 1]
                end
            end
        end
        sync_threads()
        k0 += BK
    end

    @inbounds for i in 0:(TM - 1)
        gm = row0 + ty * TM + i + 1
        for j in 0:(TN - 1)
            gn = col0 + tx + j * 16 + 1
            if gm <= M && gn <= N
                C[gm, gn] = acc[i + 1, j + 1]
            end
        end
    end
    return
end

function main()
    N = parse(Int, get(ENV, "MATMUL_N", "4096"))
    RUNS = parse(Int, get(ENV, "MATMUL_RUNS", "100"))

    A = CUDA.rand(Float32, N, N)
    B = CUDA.rand(Float32, N, N)
    C = CUDA.zeros(Float32, N, N)
    Cref = CUDA.zeros(Float32, N, N)

    mul!(Cref, A, B)   # cuBLAS reference
    synchronize()

    threads = (16, 16)
    blocks = (cld(N, BN), cld(N, BM))

    # Warm up until the GPU boosts its clocks.
    let t0 = time()
        while time() - t0 < 2.0
            @cuda threads = threads blocks = blocks sgemm_regtile!(A, B, C, N, N, N)
            synchronize()
        end
    end

    total = 0.0
    for _ in 1:RUNS
        t = CUDA.@elapsed begin
            @cuda threads = threads blocks = blocks sgemm_regtile!(A, B, C, N, N, N)
            synchronize()
        end
        total += t
    end
    avg = total / RUNS
    gflops = 2.0 * N^3 / (avg * 1e9)

    hK = Array(C)
    hR = Array(Cref)
    maxrel = 0.0
    @inbounds for i in eachindex(hK)
        rel = abs(hR[i] - hK[i]) / (abs(hR[i]) + 1.0f-12)
        rel > maxrel && (maxrel = rel)
    end

    println("Custom register-tiled SGEMM Julia N=$N (TM=TN=$TM, BM=BN=$BM, BK=$BK)")
    println("Average kernel time: $(round(avg * 1000, digits = 3)) ms")
    println("Effective GFLOPS: $(round(gflops, digits = 1))")
    println("Max relative error vs cuBLAS: $(Float64(maxrel))")
    println("Result: ", maxrel < 1.0f-4 ? "OK" : "MISMATCH")
end

main()
