#!/usr/bin/env bash
# common.sh -- shared helpers for the matmul_optimization benchmark suite.
#
# Sourced by run_all.sh and by every scripts/<category>.sh.
#
# All internal variables are namespaced (MM_*) and all configuration uses the
# MATMUL_* prefix.  This matters: oneAPI's setvars.sh exports generic names
# such as BIN_DIR=bin64, which would otherwise clobber the build paths.
#
# Configuration (set by run_all.sh; defaults here):
#   MATMUL_RESULTS_DIR   outputs + manifest directory (default results/latest)
#   MATMUL_N             matrix dimension (empty = use each source's default)
#   MATMUL_RUNS          number of timed runs (empty = each source's default)
#   MATMUL_THREADS       CPU thread count
#   MATMUL_TIMEOUT       per-benchmark timeout in seconds (0 disables)
#   MATMUL_CUDA_ARCH     CUDA gencode target (default sm_50 for the GTX 960M)
#   MATMUL_BUILD_ONLY    when 1, compile but do not run

# --- paths -------------------------------------------------------------------
MM_COMMON_SH="${BASH_SOURCE[0]}"
MM_SCRIPTS="$(cd "$(dirname "$MM_COMMON_SH")" && pwd)"
MM_ROOT="$(cd "$MM_SCRIPTS/.." && pwd)"
MM_SRC="$MM_ROOT/src"
MM_BIN="$MM_ROOT/bin"
MM_BUILD="$MM_ROOT/build"
MM_RESULTS="${MATMUL_RESULTS_DIR:-$MM_ROOT/results/latest}"
MM_MANIFEST="$MM_RESULTS/manifest.tsv"

# --- configuration -----------------------------------------------------------
# N/RUNS default to empty = keep each source file's own default (the pure-C
# teaching kernels use 1024, MKL/CUDA/Julia use 4096).  --size/--runs/--quick
# fill them in, which then overrides every benchmark.  Empty values are
# unset (not exported empty) so runtimes such as Julia fall back to defaults.
if [[ -n "${MATMUL_N:-}" ]]; then export MATMUL_N; else unset MATMUL_N; fi
if [[ -n "${MATMUL_RUNS:-}" ]]; then export MATMUL_RUNS; else unset MATMUL_RUNS; fi
export MATMUL_THREADS="${MATMUL_THREADS:-$(nproc 2>/dev/null || echo 8)}"
export MATMUL_TIMEOUT="${MATMUL_TIMEOUT:-0}"
export MATMUL_CUDA_ARCH="${MATMUL_CUDA_ARCH:-sm_50}"
export MATMUL_BUILD_ONLY="${MATMUL_BUILD_ONLY:-0}"
export MATMUL_RESULTS_DIR="$MM_RESULTS"

# --- pretty output -----------------------------------------------------------
if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'
  C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'; C_BLUE=$'\033[36m'
else
  C_RESET=''; C_BOLD=''; C_DIM=''; C_GREEN=''; C_YELLOW=''; C_RED=''; C_BLUE=''
fi

log()  { printf '%s\n' "$*"; }
info() { printf '%s%s%s\n' "$C_BLUE" "$*" "$C_RESET"; }
ok()   { printf '%s%s%s\n' "$C_GREEN" "$*" "$C_RESET"; }
warn() { printf '%s%s%s\n' "$C_YELLOW" "$*" "$C_RESET" >&2; }
err()  { printf '%s%s%s\n' "$C_RED" "$*" "$C_RESET" >&2; }

have() { command -v "$1" >/dev/null 2>&1; }

ensure_dirs() { mkdir -p "$MM_BIN" "$MM_BUILD" "$MM_RESULTS"; }

# --- environment helpers -----------------------------------------------------
# Make Intel oneAPI tools available (icx/icpx + MKL) if they are not already.
source_oneapi() {
  if have icx && have icpx; then return 0; fi
  local setvars="${ONEAPI_SETVARS:-$HOME/intel/oneapi/setvars.sh}"
  [[ -f "$setvars" ]] || { warn "oneAPI setvars.sh not found ($setvars)"; return 1; }
  # setvars.sh is not compatible with `set -u`
  set +u
  # shellcheck disable=SC1090
  source "$setvars" >/dev/null 2>&1 || true
  have icx && have icpx
}

have_gpu() {
  have nvidia-smi && nvidia-smi -L >/dev/null 2>&1
}

# --- manifest ----------------------------------------------------------------
manifest_add() {
  # name category status rc wall_s output note
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$1" "$2" "$3" "$4" "$5" "$6" "${7:-}" >>"$MM_MANIFEST"
}

record_skip() {
  local name="$1" category="$2" reason="$3"
  warn "  skip  [$category] $name -- $reason"
  manifest_add "$name" "$category" "SKIPPED" "-" "0" "" "$reason"
}

# --- build + run -------------------------------------------------------------
# build <name> <command...>   -- compile, capturing the log under build/
build() {
  local name="$1"; shift
  local logf="$MM_BUILD/$name.build.log"
  (
    cd "$MM_BUILD" || exit 1
    printf '$ %s\n\n' "$*"
    "$@"
  ) >"$logf" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    err "  build failed: $name (log: $logf)"
    tail -n 15 "$logf" >&2
    return $rc
  fi
  return 0
}

# run_bench <name> <category> <command...>   -- run, capture, record status
run_bench() {
  local name="$1" category="$2"; shift 2
  local out="$MM_RESULTS/$name.txt"

  printf '\n%s────────────────────────────────────────────────────────%s\n' "$C_DIM" "$C_RESET"
  info "▶ [$category] $name"
  printf '%s  $ %s%s\n' "$C_DIM" "$*" "$C_RESET"

  if [[ "$MATMUL_BUILD_ONLY" == "1" ]]; then
    manifest_add "$name" "$category" "BUILT" "-" "0" "" "build-only"
    return 0
  fi

  local rc start end wall status
  start=$(date +%s.%N)
  if [[ -n "$MATMUL_TIMEOUT" && "$MATMUL_TIMEOUT" != "0" ]]; then
    timeout --kill-after=30 "$MATMUL_TIMEOUT" "$@" >"$out" 2>&1
    rc=$?
  else
    "$@" >"$out" 2>&1
    rc=$?
  fi
  end=$(date +%s.%N)
  wall=$(awk -v a="$start" -v b="$end" 'BEGIN{printf "%.2f", b-a}')

  case "$rc" in
    0)   status="OK" ;;
    124) status="TIMEOUT" ;;
    *)   status="FAIL" ;;
  esac

  # Bounded preview: huge outputs (e.g. the whoami demo) stay readable.
  local lines
  lines=$(wc -l <"$out" 2>/dev/null || echo 0)
  if [[ "$lines" -le 30 ]]; then
    cat "$out"
  else
    head -n 15 "$out"
    printf '%s  ... (%s lines total, full output: %s) ...%s\n' "$C_DIM" "$lines" "$out" "$C_RESET"
    tail -n 5 "$out"
  fi

  case "$status" in
    OK)      ok   "  ✓ $name  (${wall}s)" ;;
    TIMEOUT) warn "  ⏱ $name timed out after ${MATMUL_TIMEOUT}s" ;;
    *)       err  "  ✗ $name failed (rc=$rc, ${wall}s) -- see $out" ;;
  esac

  manifest_add "$name" "$category" "$status" "$rc" "$wall" "$out" ""
  return 0   # a single failing benchmark must never abort the suite
}

# bench_build_run <name> <category> ::: <compile...> ::: <run...>
bench_build_run() {
  local name="$1" category="$2"; shift 2
  [[ "$1" == ":::" ]] && shift

  local compile_cmd=() run_cmd=()
  local mode="compile" a
  for a in "$@"; do
    if [[ "$a" == ":::" ]]; then mode="run"; continue; fi
    if [[ "$mode" == "compile" ]]; then compile_cmd+=("$a"); else run_cmd+=("$a"); fi
  done

  if ! build "$name" "${compile_cmd[@]}"; then
    manifest_add "$name" "$category" "BUILD_FAIL" "1" "0" "$MM_BUILD/$name.build.log" "build failed"
    return 0
  fi
  run_bench "$name" "$category" "${run_cmd[@]}"
}
