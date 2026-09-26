#!/usr/bin/env bash
# kokkos.sh -- C++ Kokkos DGEMM.  Kokkos is vendor-neutral, so this source
# builds against whatever backend the installed Kokkos was configured with
# (Serial / OpenMP / CUDA / HIP / SYCL).  Kokkos is optional: if it cannot be
# found the category is skipped rather than failing the suite.
#
# Point the script at a Kokkos build in one of these ways:
#   KOKKOS_ROOT=/path/to/kokkos                          (include/ + lib/)
#   KOKKOS_CXXFLAGS=... KOKKOS_LDFLAGS=... KOKKOS_LIBS=...   (explicit flags)
# or rely on pkg-config / a system install.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
ensure_dirs

MM_CAT="kokkos"
CXX="${MATMUL_KOKKOS_CXX:-${CXX:-g++}}"

if ! have "$CXX"; then
  warn "C++ compiler $CXX not found -- skipping [$MM_CAT]"
  record_skip "${MM_CAT}__matmul_kokkos" "$MM_CAT" "no C++ compiler"
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
  for prefix in /usr /usr/local /opt/kokkos "$HOME/opt/kokkos"; do
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
  warn "Kokkos not found (set KOKKOS_ROOT or KOKKOS_CXXFLAGS/KOKKOS_LIBS) -- skipping [$MM_CAT]"
  record_skip "${MM_CAT}__matmul_kokkos" "$MM_CAT" "Kokkos not installed"
  exit 0
fi

SIZE=()
[[ -n "$MATMUL_N" ]]    && SIZE+=(-DMATMUL_N="$MATMUL_N")
[[ -n "$MATMUL_RUNS" ]] && SIZE+=(-DMATMUL_RUNS="$MATMUL_RUNS")

KOKKOS_BIN="$MM_BIN/matmul_kokkos"
name="${MM_CAT}__matmul_kokkos"
bench_build_run "$name" "$MM_CAT" ::: \
  "$CXX" -O3 -std=c++17 "${SIZE[@]}" "${K_INC[@]}" "${K_EXTRA[@]}" \
    "$MM_SRC/kokkos/matmul_kokkos.cpp" -o "$KOKKOS_BIN" \
    "${K_LIB[@]}" "${K_LIBS[@]}" ::: \
  env OMP_NUM_THREADS="$MATMUL_THREADS" "$KOKKOS_BIN"
