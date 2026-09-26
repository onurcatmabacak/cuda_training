#!/usr/bin/env bash
# kokkos.sh -- C++ Kokkos DGEMM.  Kokkos is vendor-neutral, so this source
# builds against whatever backend the installed Kokkos was configured with
# (Serial / OpenMP / CUDA / HIP / SYCL).  Kokkos is optional: if it cannot be
# found the category is skipped rather than failing the suite.
#
# Preferred path (CMake): if CMake and a Kokkos CMake package are available,
# `find_package(Kokkos)` is used, which automatically selects the backend and
# compiler (e.g. the CUDA backend via nvcc_wrapper).  Locate one with:
#   KOKKOS_ROOT=/path/to/kokkos
#   Kokkos_DIR=/path/to/kokkos/lib/cmake/Kokkos
#   MATMUL_KOKKOS_CMAKE=/path/to/cmake             (if cmake is not on PATH)
#
# Fallback path (direct compile) when CMake is unavailable:
#   KOKKOS_CXXFLAGS=... KOKKOS_LDFLAGS=... KOKKOS_LIBS=...
# or pkg-config / a system Kokkos install.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
ensure_dirs

MM_CAT="kokkos"
name="${MM_CAT}__matmul_kokkos"
KO_BIN="$MM_BIN/matmul_kokkos"
KOKKOS_SRC="$MM_SRC/kokkos"

SIZE=()
[[ -n "$MATMUL_N" ]]    && SIZE+=(-DMATMUL_N="$MATMUL_N")
[[ -n "$MATMUL_RUNS" ]] && SIZE+=(-DMATMUL_RUNS="$MATMUL_RUNS")

# --- locate CMake -----------------------------------------------------------
KO_CMAKE=""
if have cmake; then KO_CMAKE="$(command -v cmake)"; fi
if [[ -n "${MATMUL_KOKKOS_CMAKE:-}" && -x "$MATMUL_KOKKOS_CMAKE" ]]; then
  KO_CMAKE="$MATMUL_KOKKOS_CMAKE"
fi
if [[ -z "$KO_CMAKE" && -x "$HOME/opt/cmake/bin/cmake" ]]; then
  KO_CMAKE="$HOME/opt/cmake/bin/cmake"
fi

# --- select the build strategy ----------------------------------------------
dir_has_kokkos() { [[ -n "$1" && -f "$1/KokkosConfig.cmake" ]]; }

KO_CMAKE_DIR=""
if [[ -z "${KOKKOS_CXXFLAGS:-}" && -z "${KOKKOS_LIBS:-}" && -n "$KO_CMAKE" ]]; then
  if dir_has_kokkos "${Kokkos_DIR:-}"; then
    KO_CMAKE_DIR="$Kokkos_DIR"
  elif [[ -n "${KOKKOS_ROOT:-}" ]] && dir_has_kokkos "$KOKKOS_ROOT/lib/cmake/Kokkos"; then
    KO_CMAKE_DIR="$KOKKOS_ROOT/lib/cmake/Kokkos"
  elif [[ -n "${KOKKOS_ROOT:-}" ]] && dir_has_kokkos "$KOKKOS_ROOT/lib64/cmake/Kokkos"; then
    KO_CMAKE_DIR="$KOKKOS_ROOT/lib64/cmake/Kokkos"
  else
    for d in \
        "$HOME/opt/kokkos-cuda/lib/cmake/Kokkos" \
        "$HOME/opt/kokkos/lib/cmake/Kokkos" \
        "$HOME/opt/kokkos/lib64/cmake/Kokkos" \
        /usr/lib/cmake/Kokkos /usr/lib64/cmake/Kokkos \
        /usr/local/lib/cmake/Kokkos /opt/kokkos/lib/cmake/Kokkos; do
      dir_has_kokkos "$d" && { KO_CMAKE_DIR="$d"; break; }
    done
  fi
fi

if [[ -n "$KO_CMAKE_DIR" ]]; then
  KO_PREFIX="$(cd "$KO_CMAKE_DIR/../../.." && pwd)"
  # A CUDA/HIP/SYCL Kokkos is compiled with its bundled wrapper; host compiler
  # for nvcc 12.0 must be gcc <= 12.
  KO_CXX=""
  [[ -x "$KO_PREFIX/bin/nvcc_wrapper" ]] && KO_CXX="$KO_PREFIX/bin/nvcc_wrapper"
  KO_HOSTCC="${MATMUL_KOKKOS_HOSTCC:-}"
  if [[ -z "$KO_HOSTCC" ]]; then
    for c in g++-12 g++-11 g++ clang++; do
      have "$c" && { KO_HOSTCC="$(command -v "$c")"; break; }
    done
  fi
  export NVCC_WRAPPER_DEFAULT_COMPILER="${KO_HOSTCC:-c++}"

  KO_BUILD="$MM_BUILD/kokkos_cmake"
  KO_CMAKE_ARGS=(-DKokkos_DIR="$KO_CMAKE_DIR" -DCMAKE_BUILD_TYPE=Release)
  [[ -n "$KO_CXX" ]] && KO_CMAKE_ARGS+=(-DCMAKE_CXX_COMPILER="$KO_CXX")
  [[ -n "$MATMUL_N" ]]    && KO_CMAKE_ARGS+=(-DMATMUL_N="$MATMUL_N")
  [[ -n "$MATMUL_RUNS" ]] && KO_CMAKE_ARGS+=(-DMATMUL_RUNS="$MATMUL_RUNS")

  kokkos_cmake_compile() {
    rm -rf "$KO_BUILD"
    "$KO_CMAKE" -S "$KOKKOS_SRC" -B "$KO_BUILD" "${KO_CMAKE_ARGS[@]}" || return 1
    "$KO_CMAKE" --build "$KO_BUILD" -j"$(nproc 2>/dev/null || echo 4)" || return 1
    cp -f "$KO_BUILD/matmul_kokkos" "$KO_BIN"
  }

  bench_build_run "$name" "$MM_CAT" ::: kokkos_cmake_compile ::: \
    env OMP_NUM_THREADS="$MATMUL_THREADS" "$KO_BIN"
  exit 0
fi

# --- fallback: direct compile against an installed Kokkos -------------------
CXX="${MATMUL_KOKKOS_CXX:-${CXX:-g++}}"
if ! have "$CXX"; then
  warn "C++ compiler $CXX not found -- skipping [$MM_CAT]"
  record_skip "$name" "$MM_CAT" "no C++ compiler"
  exit 0
fi

K_INC=(); K_LIB=(); K_EXTRA=()
K_LIBS=(-lkokkoscore -lkokkoscontainers)

if [[ -n "${KOKKOS_CXXFLAGS:-}" || -n "${KOKKOS_LIBS:-}" ]]; then
  # shellcheck disable=SC2206
  K_INC=(${KOKKOS_CXXFLAGS:-})
  # shellcheck disable=SC2206
  K_LIB=(${KOKKOS_LDFLAGS:-})
  # shellcheck disable=SC2206
  K_LIBS=(${KOKKOS_LIBS:--lkokkoscore -lkokkoscontainers})
  K_FOUND=1
elif [[ -n "${KOKKOS_ROOT:-}" && -d "$KOKKOS_ROOT" ]]; then
  K_INC=(-I"$KOKKOS_ROOT/include")
  if [[ -d "$KOKKOS_ROOT/lib64" ]]; then k_libdir="$KOKKOS_ROOT/lib64"; else k_libdir="$KOKKOS_ROOT/lib"; fi
  K_LIB=(-L"$k_libdir" -Wl,-rpath,"$k_libdir")
  K_EXTRA=(-fopenmp)
  K_FOUND=1
elif have pkg-config && pkg-config --exists kokkoscore 2>/dev/null; then
  # shellcheck disable=SC2206
  K_INC=($(pkg-config --cflags kokkoscore))
  # shellcheck disable=SC2206
  K_LIB=($(pkg-config --libs-only-L kokkoscore))
  K_LIBS=($(pkg-config --libs-only-l kokkoscore kokkoscontainers))
  K_FOUND=1
else
  K_FOUND=0
  for prefix in /usr /usr/local /opt/kokkos "$HOME/opt/kokkos" "$HOME/opt/kokkos-cuda"; do
    if [[ -f "$prefix/include/KokkosCore_config.h" ]]; then
      K_INC=(-I"$prefix/include")
      if [[ -d "$prefix/lib64" ]]; then k_libdir="$prefix/lib64"; else k_libdir="$prefix/lib"; fi
      K_LIB=(-L"$k_libdir" -Wl,-rpath,"$k_libdir")
      K_EXTRA=(-fopenmp)
      K_FOUND=1
      break
    fi
  done
fi

if [[ "${K_FOUND:-0}" != "1" ]]; then
  warn "Kokkos not found (set KOKKOS_ROOT or install CMake + a Kokkos package) -- skipping [$MM_CAT]"
  record_skip "$name" "$MM_CAT" "Kokkos not installed"
  exit 0
fi

bench_build_run "$name" "$MM_CAT" ::: \
  "$CXX" -O3 -std=c++17 "${SIZE[@]}" "${K_INC[@]}" "${K_EXTRA[@]}" \
    "$KOKKOS_SRC/matmul_kokkos.cpp" -o "$KO_BIN" \
    "${K_LIB[@]}" "${K_LIBS[@]}" ::: \
  env OMP_NUM_THREADS="$MATMUL_THREADS" "$KO_BIN"
