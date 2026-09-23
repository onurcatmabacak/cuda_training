#!/usr/bin/env bash
# rust.sh -- Rust + CUDA (direct FFI to cudart/cuBLAS, no crates).
#
# Uses a user-local Rust toolchain (see README); override with RUSTC.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
ensure_dirs

MM_CAT="rust"
RUSTC="${RUSTC:-$HOME/opt/rust/bin/rustc}"

if [[ ! -x "$RUSTC" ]]; then
  warn "rustc not found at $RUSTC -- skipping [$MM_CAT]"
  record_skip "${MM_CAT}__matmul_rust_cublas" "$MM_CAT" "no Rust toolchain"
  exit 0
fi

RUST_BIN="$MM_BIN/matmul_rust_cublas"
if build "${MM_CAT}__matmul_rust_build" \
    "$RUSTC" --edition 2021 -O -L /usr/lib/x86_64-linux-gnu -l cudart -l cublas \
      "$MM_SRC/rust/matmul_rust_cublas.rs" -o "$RUST_BIN"; then
  RS_ENV=(env "MATMUL_N=${MATMUL_N:-1024}" "MATMUL_RUNS=${MATMUL_RUNS:-10}")
  run_bench "${MM_CAT}__matmul_rust_cublas" "$MM_CAT" "${RS_ENV[@]}" "$RUST_BIN" cublas
  run_bench "${MM_CAT}__matmul_rust_graph"  "$MM_CAT" "${RS_ENV[@]}" "$RUST_BIN" graph
else
  manifest_add "${MM_CAT}__matmul_rust_cublas" "$MM_CAT" "BUILD_FAIL" "1" "0" \
    "$MM_BUILD/${MM_CAT}__matmul_rust_build.build.log" "build failed"
fi
