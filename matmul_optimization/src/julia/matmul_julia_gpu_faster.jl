using CUDA  
using LinearAlgebra  
  
const N = parse(Int, get(ENV, "MATMUL_N", "4096"))  # matrix size  
const TILE = 32         # tile size  
const RUNS = parse(Int, get(ENV, "MATMUL_RUNS", "100"))         # number of timed runs  
  
# CUDA kernel: shared-memory tiled GEMM  
function kernel_tiled(A, B, C, n)  
    # Thread/block indices (0-based)  
    tx = threadIdx().x - 1  
    ty = threadIdx().y - 1  
    row = (blockIdx().y - 1) * TILE + ty  
    col = (blockIdx().x - 1) * TILE + tx  
  
    # Shared memory for tiles of A and B
    sh = @cuDynamicSharedMem(Float32, 2 * TILE * TILE)
  
    offsetA = 0  
    offsetB = TILE * TILE  
  
    acc = 0.0f0  
  
    m = 0  
    while m < n  
        # Load A tile  
        if row < n && (m + tx) < n  
            sh[offsetA + ty * TILE + tx + 1] = A[row+1, m+tx+1]  
        else  
            sh[offsetA + ty * TILE + tx + 1] = 0.0f0  
        end  
  
        # Load B tile  
        if (m + ty) < n && col < n  
            sh[offsetB + ty * TILE + tx + 1] = B[m+ty+1, col+1]  
        else  
            sh[offsetB + ty * TILE + tx + 1] = 0.0f0  
        end  
  
        sync_threads()  
  
        # Compute partial product for this tile  
        for k in 0:(TILE-1)  
            a = sh[offsetA + ty * TILE + k + 1]  
            b = sh[offsetB + k * TILE + tx + 1]  
            acc += a * b  
        end  
  
        sync_threads()  
        m += TILE  
    end  
  
    # Write output  
    if row < n && col < n  
        C[row+1, col+1] = acc  
    end  
  
    return nothing  
end  
  
function main()  
    dev = CUDA.device()  
    println("Using CUDA device: ", CUDA.name(dev))  
    println("Compute capability: ", CUDA.capability(dev))  
  
    # Allocate random matrices  
    A = CUDA.rand(Float32, N, N)  
    B = CUDA.rand(Float32, N, N)  
    C = CUDA.zeros(Float32, N, N)  
  
    threads = (TILE, TILE)  
    blocks = (cld(N, TILE), cld(N, TILE))  
    shbytes = 2 * TILE * TILE * sizeof(Float32)  # shared memory for A+B tiles  
  
    println("Launching kernel with threads=$threads, blocks=$blocks")  
  
    # Warmup  
    @cuda threads=threads blocks=blocks shmem=shbytes kernel_tiled(A, B, C, N)  
    CUDA.synchronize()  
  
    # Keep the GPU busy until it boosts its clocks (a short kernel timed right
    # after startup otherwise measures at the idle clock).
    let t0 = time()
        while time() - t0 < 2.0
            @cuda threads=threads blocks=blocks shmem=shbytes kernel_tiled(A, B, C, N)
            CUDA.synchronize()
        end
    end

    # Method 1: Using @elapsed (simpler approach)
    println("\n=== Method 1: Using @elapsed ===")
    total_time = 0.0
    for r in 1:RUNS  
        elapsed = @elapsed begin
            @cuda threads=threads blocks=blocks shmem=shbytes kernel_tiled(A, B, C, N)  
            CUDA.synchronize()
        end
        total_time += elapsed
    end  
  
    avg_ms_method1 = (total_time / RUNS) * 1000  # convert to milliseconds
    gflops_method1 = 2.0 * N^3 / (avg_ms_method1 * 1e6)  
  
    println("Average kernel time over $RUNS runs: $(round(avg_ms_method1,digits=3)) ms")  
    println("Effective GFLOPS: $(round(gflops_method1,digits=2))")
  
    # Optional correctness check (small sample)  
    println("\n=== Correctness Check ===")
    C_host = Array(C)  
    A_host = Array(A)  
    B_host = Array(B)  
    maxrel = 0.0  
    inds = [(1,1),(2,3),(100,200),(N,N)]  
    for (i,j) in inds  
        sref = 0.0f0  
        @inbounds for k in 1:N  
            sref += A_host[i,k] * B_host[k,j]  
        end  
        rel = abs(sref - C_host[i,j]) / (abs(sref)+1e-12)  
        maxrel = max(maxrel, rel)  
    end  
    println("Sample max relative error: $maxrel")  
end  
  
main()