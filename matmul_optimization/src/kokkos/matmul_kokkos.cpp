// matmul_kokkos.cpp -- C++ Kokkos DGEMM (Float64 matrix multiplication).
//
// Kokkos is a vendor-neutral performance-portability layer: this single source
// builds unchanged against whichever backend the Kokkos installation was
// configured with (Serial, OpenMP, CUDA, HIP, SYCL, ...).  Only the backend's
// device/kernel-launch machinery differs; the algorithm is the same.
//
// Two kernels are provided, selected by argv[1]:
//   naive : one work item per C(i,j), reads A and B straight from global memory
//           (representative of the hand-written naive CUDA kernel)
//   tiled : one team per TILE x TILE output tile, staging A and B through a
//           shared-memory scratch pad (TeamPolicy + team_scratch).  TILE is
//           chosen at runtime from the backend's maximum team size, so the same
//           code runs on a GPU (TILE=32) and on a CPU OpenMP backend (smaller).
//
// Usage:   matmul_kokkos [naive|tiled]        (default: tiled)
//
// Build against an installed Kokkos, for example:
//   g++ -O3 -std=c++17 matmul_kokkos.cpp -o matmul_kokkos \
//       -I"$KOKKOS_ROOT/include" -L"$KOKKOS_ROOT/lib" \
//       -lkokkoscore -lkokkoscontainers -fopenmp
//
// Sizes: -DMATMUL_N=<dim> (default 1024), -DMATMUL_RUNS=<runs> (default 10).
// (N is deliberately *not* used as a macro name: Kokkos headers have a template
//  parameter named N, so a global -DN would corrupt them.)

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#ifndef MATMUL_N
#define MATMUL_N 1024
#endif
#ifndef MATMUL_RUNS
#define MATMUL_RUNS 10
#endif

constexpr int N = MATMUL_N;
constexpr int RUNS = MATMUL_RUNS;

using exec_space = Kokkos::DefaultExecutionSpace;
using view_t = Kokkos::View<double **, Kokkos::LayoutRight>;
using scratch_t = Kokkos::View<double *, exec_space::scratch_memory_space,
                               Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

// ---------------------------------------------------------------------------
// naive: one work item per output element (global-memory reads only)
// ---------------------------------------------------------------------------
struct NaiveDGEMM {
  view_t A, B, C;
  int n;
  NaiveDGEMM(const view_t &a, const view_t &b, const view_t &c, int n_)
      : A(a), B(b), C(c), n(n_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, const int j) const {
    double sum = 0.0;
    for (int k = 0; k < n; ++k) {
      sum += A(i, k) * B(k, j);
    }
    C(i, j) = sum;
  }
};

// ---------------------------------------------------------------------------
// tiled: one team per tile x tile output tile, shared-memory staging
// ---------------------------------------------------------------------------
struct TiledDGEMM {
  view_t A, B, C;
  int n, tile, ntiles;

  TiledDGEMM(const view_t &a, const view_t &b, const view_t &c, int n_, int t,
             int nt)
      : A(a), B(b), C(c), n(n_), tile(t), ntiles(nt) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Kokkos::TeamPolicy<exec_space>::member_type &team) const {
    const int t = tile;
    const int id = team.league_rank();
    const int bi = id / ntiles;
    const int bj = id % ntiles;
    const int i0 = bi * t;
    const int j0 = bj * t;
    const int tx = team.team_rank() % t;
    const int ty = team.team_rank() / t;

    // Two t x t panels in one contiguous scratch buffer: As then Bs.
    scratch_t sh(team.team_scratch(0), 2 * t * t);
    const int As = 0;
    const int Bs = t * t;

    double acc = 0.0;
    for (int k0 = 0; k0 < n; k0 += t) {
      const int i = i0 + ty;
      const int j = j0 + tx;
      sh(As + ty * t + tx) = (i < n && (k0 + tx) < n) ? A(i, k0 + tx) : 0.0;
      sh(Bs + ty * t + tx) = ((k0 + ty) < n && j < n) ? B(k0 + ty, j) : 0.0;
      team.team_barrier();
      for (int k = 0; k < t; ++k) {
        acc += sh(As + ty * t + k) * sh(Bs + k * t + tx);
      }
      team.team_barrier();
    }
    const int gi = i0 + ty;
    const int gj = j0 + tx;
    if (gi < n && gj < n) {
      C(gi, gj) = acc;
    }
  }
};

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    const std::string requested = (argc > 1) ? argv[1] : "";
    const double flop = 2.0 * static_cast<double>(N) * N * N;

    view_t A("A", N, N);
    view_t B("B", N, N);
    view_t C("C", N, N);

    Kokkos::parallel_for(
        "init", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, N}),
        KOKKOS_LAMBDA(const int i, const int j) {
          A(i, j) = ((i * 131 + j * 17) % 1009) * 1e-3 + 1.0;
          B(i, j) = ((i * 29 + j * 197) % 997) * 1e-3 + 2.0;
        });
    Kokkos::fence();

    // Largest team the backend can schedule.  On a GPU this is 1024; on an
    // 8-thread OpenMP backend it is 8.  The tiled kernel picks its tile size
    // from it, and the default mode avoids tiling when the team is too small
    // to be useful (a tiny CPU tile is correct but pathologically slow).
    Kokkos::TeamPolicy<exec_space> probe(1, Kokkos::AUTO);
    const int max_team = probe.team_size_max(
        TiledDGEMM(A, B, C, N, 1, 1), Kokkos::ParallelForTag());

    std::string mode = requested;
    if (mode != "naive" && mode != "tiled") {
      mode = (max_team >= 256) ? "tiled" : "naive";
    }

    int tile = 1;
    if (mode == "tiled") {
      tile = static_cast<int>(std::sqrt(static_cast<double>(std::max(1, max_team))));
      tile = std::max(1, std::min(tile, 32));
    }
    const int ntiles = (N + tile - 1) / tile;

    auto launch = [&]() {
      if (mode == "naive") {
        Kokkos::parallel_for(
            "dgemm_naive", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, N}),
            NaiveDGEMM(A, B, C, N));
      } else {
        const int scratch = 2 * tile * tile * static_cast<int>(sizeof(double));
        Kokkos::parallel_for(
            "dgemm_tiled",
            Kokkos::TeamPolicy<exec_space>(ntiles * ntiles, tile * tile)
                .set_scratch_size(0, Kokkos::PerTeam(scratch)),
            TiledDGEMM(A, B, C, N, tile, ntiles));
      }
      Kokkos::fence();
    };

    launch();  // compile + first touch

    // Keep every core busy for ~2 s so the device reaches boost clocks.
    {
      Kokkos::Timer boost;
      while (boost.seconds() < 2.0) {
        launch();
      }
    }

    double total = 0.0;
    for (int r = 0; r < RUNS; ++r) {
      Kokkos::Timer t;
      launch();
      total += t.seconds();
    }
    const double avg_ms = total / RUNS * 1e3;

    // Sample correctness check against a host reference.
    auto hA = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A);
    auto hB = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), B);
    auto hC = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), C);
    const int checks[][2] = {{0, 0},         {1, 2},        {N / 2, N / 3},
                             {N - 1, N - 1}, {0, N - 1},    {N - 1, 0}};
    double maxrel = 0.0;
    for (const auto &p : checks) {
      const int i = p[0], j = p[1];
      double ref = 0.0;
      for (int k = 0; k < N; ++k) {
        ref += hA(i, k) * hB(k, j);
      }
      maxrel = std::max(maxrel, std::fabs(ref - hC(i, j)) / (std::fabs(ref) + 1e-12));
    }

    std::printf("Matrix size: %dx%d (Float64)\n", N, N);
    std::printf("Kokkos backend: %s, mode: %s (tile=%d)\n", exec_space::name(),
                mode.c_str(), tile);
    std::printf("Average kernel time (ms): %.6f\n", avg_ms);
    std::printf("Effective GFLOPS: %.2f\n", flop / (avg_ms * 1e-3) / 1e9);
    std::printf("Max relative error vs CPU: %.3e\n", maxrel);
    std::printf("Result: %s\n", (maxrel < 1e-9) ? "OK" : "MISMATCH");
  }
  Kokkos::finalize();
  return 0;
}
