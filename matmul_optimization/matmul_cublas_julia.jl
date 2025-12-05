# matmul_cublas_julia.jl
using CUDA
using BenchmarkTools

# Matrix size and number of runs
const N = 4096
const RUNS = 100

# Initialize random Float32 matrices directly on the GPU
A_d = CUDA.rand(Float32, N, N)
B_d = CUDA.rand(Float32, N, N)
C_d = CUDA.zeros(Float32, N, N)

# Warmup to initialize cuBLAS and GPU
C_d .= A_d * B_d
synchronize()

println("GPU Julia 4096x4096 matrix multiplication (cuBLAS)")

# Measure average time over RUNS
total_time = @elapsed begin
    for _ in 1:RUNS
        C_d .= A_d * B_d
    end
end

avg_time = total_time / RUNS  # average time per run in seconds
gflops = 2.0 * N * N * N / (avg_time * 1e9)  # GFLOPS

# Copy a sample value back to CPU for verification
C_sample = Array(C_d)[1,1]

println("Average time per run: $(round(avg_time, digits=6)) s")
println("Effective GFLOPS: $(round(gflops, digits=2))")
println("C[1,1] = $C_sample")
