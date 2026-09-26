#!/usr/bin/env bash
# run_all.sh -- build and run every matmul benchmark in this folder, then
# write a combined summary.  This is the single command for the whole suite.
#
#   ./run_all.sh                 # full run: N=4096, 100 runs, all categories
#   ./run_all.sh --quick         # smoke test: N=1024, 5 runs
#   ./run_all.sh --only cuda     # only one category
#   ./run_all.sh --skip julia,demos --runs 20
#   ./run_all.sh --build-only    # compile everything, run nothing
#
# Results land in results/run_<timestamp>/ (with results/latest symlink),
# containing one .txt per benchmark, manifest.tsv, config.txt, summary.md
# and summary.csv.
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$HERE/scripts"
ALL_CATS=(cpu_c cpu_mkl cuda cpp26 rust kokkos julia demos)

usage() {
  cat <<'EOF'
Usage: ./run_all.sh [options]

Options:
  --only CATS       comma-separated categories to run  (default: all)
  --skip CATS       comma-separated categories to skip
  --quick           quick smoke test: --size 512 --runs 3 --timeout 300
  --size N          matrix dimension (default 1024, same for every benchmark)
  --runs R          timed runs per benchmark (default 10, same for every benchmark)
  --threads T       CPU threads for BLAS/OpenMP (default: nproc)
  --timeout SEC     per-benchmark timeout, 0 = none (default 1800)
  --cuda-arch ARCH  CUDA gencode target (default sm_50, the GTX 960M)
  --build-only      compile everything but run no benchmarks
  -h, --help        show this help

Categories: cpu_c  cpu_mkl  cuda  cpp26  rust  kokkos  julia  demos
EOF
}

MM_ONLY=""
MM_SKIP=""
# Uniform configuration: every benchmark uses the same matrix size and run count
# so the results are directly comparable.  All benchmarks are Float64.
MATMUL_N=1024
MATMUL_RUNS=10
MATMUL_THREADS="$(nproc 2>/dev/null || echo 8)"
MATMUL_TIMEOUT=1800
MATMUL_CUDA_ARCH="${MATMUL_CUDA_ARCH:-sm_50}"
MATMUL_BUILD_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)       MM_ONLY="$2"; shift 2 ;;
    --skip)       MM_SKIP="$2"; shift 2 ;;
    --quick)      MATMUL_N=512; MATMUL_RUNS=3; MATMUL_TIMEOUT=300; shift ;;
    --size)       MATMUL_N="$2"; shift 2 ;;
    --runs)       MATMUL_RUNS="$2"; shift 2 ;;
    --threads)    MATMUL_THREADS="$2"; shift 2 ;;
    --timeout)    MATMUL_TIMEOUT="$2"; shift 2 ;;
    --cuda-arch)  MATMUL_CUDA_ARCH="$2"; shift 2 ;;
    --build-only) MATMUL_BUILD_ONLY=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

STAMP="$(date +%Y%m%d_%H%M%S)"
MM_RUN_DIR="$HERE/results/run_$STAMP"
mkdir -p "$MM_RUN_DIR"

export MATMUL_N MATMUL_RUNS MATMUL_THREADS MATMUL_TIMEOUT
export MATMUL_CUDA_ARCH MATMUL_BUILD_ONLY
export MATMUL_RESULTS_DIR="$MM_RUN_DIR"

# shellcheck source=scripts/common.sh
source "$SCRIPTS/common.sh"
ensure_dirs

printf 'name\tcategory\tstatus\trc\twall_s\toutput\tnote\n' >"$MM_MANIFEST"

{
  echo "date=$STAMP"
  echo "matrix_size=${MATMUL_N:-default}"
  echo "runs=${MATMUL_RUNS:-default}"
  echo "threads=$MATMUL_THREADS"
  echo "timeout_s=$MATMUL_TIMEOUT"
  echo "cuda_arch=$MATMUL_CUDA_ARCH"
  echo "build_only=$MATMUL_BUILD_ONLY"
  echo "only=$MM_ONLY"
  echo "skip=$MM_SKIP"
} >"$MM_RESULTS/config.txt"

should_run() {
  local c="$1"
  if [[ -n "$MM_ONLY" && ",$MM_ONLY," != *",$c,"* ]]; then return 1; fi
  if [[ -n "$MM_SKIP" && ",$MM_SKIP," == *",$c,"* ]]; then return 1; fi
  return 0
}

echo "${C_BOLD}matmul_optimization benchmark suite${C_RESET}"
echo "  results dir : $MM_RESULTS"
echo "  matrix size : ${MATMUL_N:-per-benchmark default}"
echo "  runs        : ${MATMUL_RUNS:-per-benchmark default}"
echo "  threads     : $MATMUL_THREADS"
echo "  timeout     : ${MATMUL_TIMEOUT}s per benchmark"
echo "  categories  : ${ALL_CATS[*]}"
[[ -n "$MM_ONLY" ]] && echo "  only        : $MM_ONLY"
[[ -n "$MM_SKIP" ]] && echo "  skip        : $MM_SKIP"

for cat in "${ALL_CATS[@]}"; do
  should_run "$cat" || { echo "  (skipping category: $cat)"; continue; }
  echo
  echo "${C_BOLD}══ category: $cat ══${C_RESET}"
  bash "$SCRIPTS/$cat.sh"
done

# Point results/latest at this run.
ln -sfn "run_$STAMP" "$HERE/results/latest"

echo
echo "${C_BOLD}══ summary ══${C_RESET}"
if have python3; then
  python3 "$SCRIPTS/summarize.py" "$MM_RESULTS" >"$MM_RESULTS/summary.md"
  cat "$MM_RESULTS/summary.md"
  # Stable, easy-to-find single file with every result + raw outputs.
  cp -f "$MM_RESULTS/FULL_REPORT.md" "$HERE/RESULTS.md"
  # Refresh the results table embedded in the READMEs (visible on GitHub).
  python3 "$SCRIPTS/update_readme.py" "$MM_RESULTS" "$HERE/README.md" "$HERE/../README.md" || true
else
  warn "python3 not found -- skipping summary generation"
fi

echo
ok "Done. Results: $MM_RESULTS"
ok "       Latest: $HERE/results/latest"
ok "   Single file: $HERE/RESULTS.md"
