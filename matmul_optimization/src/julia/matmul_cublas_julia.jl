# matmul_cublas_julia.jl
using CUDA
using LinearAlgebra

# Matrix size and number of runs
const N = parse(Int, get(ENV, "MATMUL_N", "4096"))
const RUNS = parse(Int, get(ENV, "MATMUL_RUNS", "100"))

# Initialize random Float64 matrices directly on the GPU
A_d = CUDA.rand(Float64, N, N)
B_d = CUDA.rand(Float64, N, N)
C_d = CUDA.zeros(Float64, N, N)

# Warmup to initialize cuBLAS and GPU
mul!(C_d, A_d, B_d)
synchronize()

# Keep the GPU busy until it boosts its clocks (short kernels otherwise measure
# at the idle clock).
let t0 = time()
    while time() - t0 < 2.0
        mul!(C_d, A_d, B_d)
        synchronize()
    end
end

println("GPU Julia $(N)x$(N) matrix multiplication (cuBLAS DGEMM, Float64)")

# Measure average time over RUNS.
# The kernels are asynchronous, so we MUST synchronize before stopping the
# clock -- otherwise @elapsed only measures the time to enqueue the launches.
total_time = @elapsed begin
    for _ in 1:RUNS
        mul!(C_d, A_d, B_d)
    end
    synchronize()
end

avg_time = total_time / RUNS  # average time per run in seconds
gflops = 2.0 * N * N * N / (avg_time * 1e9)  # GFLOPS

# Copy a sample value back to CPU for verification
C_sample = Array(C_d)[1,1]

println("Average time per run: $(round(avg_time, digits=6)) s")
println("Effective GFLOPS: $(round(gflops, digits=2))")
println("C[1,1] = $C_sample")
