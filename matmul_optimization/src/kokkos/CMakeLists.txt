cmake_minimum_required(VERSION 3.16)
project(matmul_kokkos LANGUAGES CXX)

# `find_package(CUDAToolkit)` must come first so the CUDA::cudart* imported
# targets (including CUDA::cudart_static_deps referenced by Kokkos's installed
# config) exist before Kokkos defines its own aliases.
find_package(CUDAToolkit QUIET)

# Vendor-neutral Kokkos DGEMM.  find_package(Kokkos) selects whichever backend
# the installation was built for and supplies the correct compiler flags
# (CUDA/HIP/SYCL/OpenMP/Serial).
find_package(Kokkos REQUIRED)

add_executable(matmul_kokkos matmul_kokkos.cpp)
target_link_libraries(matmul_kokkos PRIVATE Kokkos::kokkoscore)

if(DEFINED MATMUL_N)
  target_compile_definitions(matmul_kokkos PRIVATE MATMUL_N=${MATMUL_N})
endif()
if(DEFINED MATMUL_RUNS)
  target_compile_definitions(matmul_kokkos PRIVATE MATMUL_RUNS=${MATMUL_RUNS})
endif()

# Keep the binary at <build>/matmul_kokkos for the runner to copy to bin/.
set_target_properties(matmul_kokkos PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
