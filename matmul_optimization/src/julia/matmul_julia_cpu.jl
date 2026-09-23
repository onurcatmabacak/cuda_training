# matmul_julia_cpu.jl -- CPU BLAS DGEMM (Float64)
using LinearAlgebra

BLAS.set_num_threads(parse(Int, get(ENV, "MATMUL_THREADS", "8")))

const N = parse(Int, get(ENV, "MATMUL_N", "1024"))
const RUNS = parse(Int, get(ENV, "MATMUL_RUNS", "10"))

A = rand(Float64, N, N)
B = rand(Float64, N, N)
C = zeros(Float64, N, N)

println("CPU-only Julia $(N)x$(N) matrix multiplication (Float64)")

mul!(C, A, B)   # warmup

total_time = @elapsed begin
    for _ in 1:RUNS
        mul!(C, A, B)
    end
end

avg_time = total_time / RUNS
gflops = 2.0 * N * N * N / (avg_time * 1e9)

println("Average time per run: $(round(avg_time, digits=6)) s")
println("Effective GFLOPS: $(round(gflops, digits=2))")
println("C[1,1]=", C[1, 1])
