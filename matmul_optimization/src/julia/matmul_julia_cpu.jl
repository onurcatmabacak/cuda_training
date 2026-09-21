# matmul_cpu.jl
using LinearAlgebra
using BenchmarkTools

# Set number of BLAS threads (choose number of physical cores)
BLAS.set_num_threads(parse(Int, get(ENV, "MATMUL_THREADS", "8")))  # adjust to your CPU cores

const N = parse(Int, get(ENV, "MATMUL_N", "4096"))
const RUNS = parse(Int, get(ENV, "MATMUL_RUNS", "100"))  # reduce for benchmarking

# Allocate and initialize matrices
A = rand(Float32, N, N)
B = rand(Float32, N, N)
C = zeros(Float32, N, N)

println("CPU-only Julia 4096x4096 matrix multiplication")

total_time = @elapsed begin
    for _ in 1:RUNS
        C .= A * B
    end
end

avg_time = total_time / RUNS
gflops = 2.0 * N * N * N / (avg_time * 1e9)


println("Average time per run: $(round(avg_time, digits=6)) s")
println("Effective GFLOPS: $(round(gflops, digits=2))")

# Optional: print first element for correctness check
println("C[1,1]=", C[1,1])
