using CUDA
using LinearAlgebra

function benchmark_cublas(N, RUNS)
    A = CUDA.rand(Float32, N, N)
    B = CUDA.rand(Float32, N, N)
    C = CUDA.zeros(Float32, N, N)

    # Warmup
    mul!(C, A, B)
    synchronize()  # wait for GPU to finish

    # Keep the GPU busy until it boosts its clocks: a short kernel timed right
    # after startup otherwise measures at the idle clock (P8 ~135 MHz vs
    # boosted ~1200 MHz on this laptop).
    let t0 = time()
        while time() - t0 < 2.0
            mul!(C, A, B)
            synchronize()
        end
    end

    total_ms = 0.0
    for _ in 1:RUNS
        ms = CUDA.@elapsed begin
            mul!(C, A, B)
            synchronize()   # ensure kernel finishes
        end
        total_ms += ms * 1000.0  # seconds -> ms
    end

    avg_ms = total_ms / RUNS
    gflops = 2.0 * N^3 / (avg_ms * 1e6)  # N^3 flops, ms -> s
    return avg_ms, gflops, Array(C)[1,1]
end

avg_ms, gflops, c11 = benchmark_cublas(parse(Int, get(ENV, "MATMUL_N", "4096")), parse(Int, get(ENV, "MATMUL_RUNS", "100")))
println("Average kernel time (ms): $avg_ms")
println("Effective GFLOPS: $gflops")
println("C[1,1] = $c11")
