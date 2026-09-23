# matmul_julia_gpu_faster.jl -- hand-written tiled CUDA kernel in Julia (Float64)
using CUDA
using LinearAlgebra

const N = parse(Int, get(ENV, "MATMUL_N", "1024"))
const TILE = 32
const RUNS = parse(Int, get(ENV, "MATMUL_RUNS", "10"))

function kernel_tiled(A, B, C, n)
    tx = threadIdx().x - 1
    ty = threadIdx().y - 1
    row = (blockIdx().y - 1) * TILE + ty
    col = (blockIdx().x - 1) * TILE + tx

    sh = @cuDynamicSharedMem(Float64, 2 * TILE * TILE)
    offsetA = 0
    offsetB = TILE * TILE

    acc = 0.0
    m = 0
    while m < n
        if row < n && (m + tx) < n
            sh[offsetA + ty * TILE + tx + 1] = A[row+1, m+tx+1]
        else
            sh[offsetA + ty * TILE + tx + 1] = 0.0
        end
        if (m + ty) < n && col < n
            sh[offsetB + ty * TILE + tx + 1] = B[m+ty+1, col+1]
        else
            sh[offsetB + ty * TILE + tx + 1] = 0.0
        end
        sync_threads()

        for k in 0:(TILE-1)
            a = sh[offsetA + ty * TILE + k + 1]
            b = sh[offsetB + k * TILE + tx + 1]
            acc += a * b
        end

        sync_threads()
        m += TILE
    end

    if row < n && col < n
        C[row+1, col+1] = acc
    end
    return nothing
end

function main()
    dev = CUDA.device()
    println("Using CUDA device: ", CUDA.name(dev))
    println("Compute capability: ", CUDA.capability(dev))

    A = CUDA.rand(Float64, N, N)
    B = CUDA.rand(Float64, N, N)
    C = CUDA.zeros(Float64, N, N)

    threads = (TILE, TILE)
    blocks = (cld(N, TILE), cld(N, TILE))
    shbytes = 2 * TILE * TILE * sizeof(Float64)

    println("Launching kernel with threads=$threads, blocks=$blocks")

    @cuda threads = threads blocks = blocks shmem = shbytes kernel_tiled(A, B, C, N)
    CUDA.synchronize()

    # Keep the GPU busy until it boosts its clocks.
    let t0 = time()
        while time() - t0 < 2.0
            @cuda threads = threads blocks = blocks shmem = shbytes kernel_tiled(A, B, C, N)
            CUDA.synchronize()
        end
    end

    println("\n=== Method 1: Using @elapsed ===")
    total_time = 0.0
    for r in 1:RUNS
        elapsed = @elapsed begin
            @cuda threads = threads blocks = blocks shmem = shbytes kernel_tiled(A, B, C, N)
            CUDA.synchronize()
        end
        total_time += elapsed
    end

    avg_ms_method1 = (total_time / RUNS) * 1000
    gflops_method1 = 2.0 * N^3 / (avg_ms_method1 * 1e6)

    println("Average kernel time over $RUNS runs: $(round(avg_ms_method1, digits=3)) ms")
    println("Effective GFLOPS: $(round(gflops_method1, digits=2))")

    # Optional correctness check (small sample)
    println("\n=== Correctness Check ===")
    C_host = Array(C)
    A_host = Array(A)
    B_host = Array(B)
    maxrel = 0.0
    inds = [(1, 1), (2, 3), (min(100, N), min(200, N)), (N, N)]
    for (i, j) in inds
        sref = 0.0
        @inbounds for k in 1:N
            sref += A_host[i, k] * B_host[k, j]
        end
        rel = abs(sref - C_host[i, j]) / (abs(sref) + 1e-12)
        maxrel = max(maxrel, rel)
    end
    println("Sample max relative error: $maxrel")
end

main()
