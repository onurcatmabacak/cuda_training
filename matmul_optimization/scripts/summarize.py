#!/usr/bin/env python3
"""Summarise matmul benchmark outputs into Markdown + CSV.

Reads <results_dir>/manifest.tsv (written by scripts/common.sh) and parses each
benchmark's captured output.  The output formats differ across languages, so the
parser tries a list of known patterns and falls back to computing GFLOPS from
the matrix dimension when the program does not print it itself.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

DIM_RE = re.compile(r"\b(\d{3,5})\s*[x×]\s*(\d{3,5})\b")

TIME_PATTERNS = [
    (re.compile(r"Average time per run:\s*([\d.eE+-]+)\s*s\b"), 1.0),
    (re.compile(r"Average time over \d+ runs:\s*([\d.eE+-]+)\s*seconds"), 1.0),
    (re.compile(r"Average time:\s*([\d.eE+-]+)\s*s\b"), 1.0),
    (re.compile(r"Average kernel time(?: over \d+ runs)?:\s*([\d.eE+-]+)\s*ms"), 1e-3),
    (re.compile(r"Average kernel time \(ms\):\s*([\d.eE+-]+)"), 1e-3),
]

GFLOPS_PATTERNS = [
    re.compile(r"Effective GFLOPS:\s*([\d.eE+-]+)"),
    re.compile(r"\|\s*([\d.]+)\s*GFLOPS"),
    re.compile(r"GFLOPS[:\s]+([\d.]+)"),
]


def find_dim(text: str) -> int | None:
    for m in DIM_RE.finditer(text):
        if m.group(1) == m.group(2):
            return int(m.group(1))
    return None


def first_float(patterns, text):
    for pat in patterns:
        m = re.search(pat, text) if isinstance(pat, str) else pat.search(text)
        if m:
            return float(m.group(1))
    return None


def find_time(text):
    """Return (seconds, source_unit) for the first matching time pattern."""
    for pat, scale in TIME_PATTERNS:
        m = pat.search(text)
        if m:
            raw = float(m.group(1))
            return raw * scale, ("ms" if scale < 1.0 else "s")
    return None, None


def correctness_notes(text: str) -> list[str]:
    notes = []
    m = re.search(r"Max relative error vs \w+:\s*([\d.eE+-]+)", text)
    if not m:
        m = re.search(r"Sample max relative error:\s*([\d.eE+-]+)", text)
    if m:
        notes.append(f"max rel err {m.group(1)}")
    if re.search(r"Result:\s*OK", text):
        notes.append("OK")
    elif re.search(r"Result:\s*MISMATCH", text):
        notes.append("MISMATCH")
    m = re.search(r"C\[[0-9,\s]+\]\s*=\s*([\d.eE+-]+)", text)
    if m:
        notes.append(f"C={m.group(1)}")
    return notes


def parse_output(text: str) -> list[dict]:
    """Return one result dict per logical measurement in the output."""
    dim = find_dim(text)
    gflops = first_float(GFLOPS_PATTERNS, text)
    notes = correctness_notes(text)

    # The naive CUDA demo reports both a slow CPU reference and the GPU kernel.
    cpu_us = first_float([re.compile(r"CPU average time:\s*([\d.eE+-]+)\s*microseconds")], text)
    gpu_us = first_float([re.compile(r"GPU average time:\s*([\d.eE+-]+)\s*microseconds")], text)
    if cpu_us is not None or gpu_us is not None:
        speedup = first_float([re.compile(r"Speedup:\s*([\d.eE+-]+)x")], text)
        out = []
        if gpu_us is not None:
            sec = gpu_us / 1e6
            g = gflops if gflops is not None else compute_gflops(dim, sec)
            out.append({"suffix": "[gpu]", "seconds": sec, "gflops": g, "notes": list(notes)})
        if cpu_us is not None:
            sec = cpu_us / 1e6
            g = compute_gflops(dim, sec)
            n = list(notes)
            if speedup is not None:
                n.append(f"speedup {speedup:.1f}x")
            out.append({"suffix": "[cpu]", "seconds": sec, "gflops": g, "notes": n})
        return out

    sec, _unit = find_time(text)
    if sec is None:
        return []
    g = gflops if gflops is not None else compute_gflops(dim, sec)
    return [{"suffix": "", "seconds": sec, "gflops": g, "notes": notes}]


def compute_gflops(dim, seconds):
    if dim and seconds and seconds > 0:
        return 2.0 * dim**3 / seconds / 1e9
    return None


def fmt_time(seconds):
    if seconds is None:
        return ""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.3f} ms"
    return f"{seconds:.4f} s"


def fmt_gflops(g):
    if g is None:
        return ""
    if g >= 100:
        return f"{g:.1f}"
    return f"{g:.2f}"


STATUS_ORDER = {"OK": 0, "BUILT": 1, "SKIPPED": 2, "BUILD_FAIL": 3, "TIMEOUT": 4, "FAIL": 5}
CATEGORY_ORDER = ["cpu_c", "cpu_mkl", "cuda", "julia", "demos"]

# Exact source file (with extension) each benchmark is built from.  The report's
# "Benchmark" column shows this file name instead of the internal manifest name.
# A `source` column in the manifest (if present) takes precedence, so new
# benchmarks can declare their own source without touching this table.
SOURCE_MAP = {
    # cpu_c
    "cpu_c__matmul_c99_O1": "matmul_c99.c",
    "cpu_c__matmul_c99_O2": "matmul_c99.c",
    "cpu_c__matmul_c99_O3": "matmul_c99.c",
    "cpu_c__matmul_c11_O1": "matmul_c11.c",
    "cpu_c__matmul_c11_O2": "matmul_c11.c",
    "cpu_c__matmul_c11_O3": "matmul_c11.c",
    "cpu_c__matmul_c23_O1": "matmul_c23.c",
    "cpu_c__matmul_c23_O2": "matmul_c23.c",
    "cpu_c__matmul_c23_O3": "matmul_c23.c",
    "cpu_c__matmul_c11_index_order_O3": "matmul_c11_index_order.c",
    "cpu_c__matmul_c11_parallel_loops_O3": "matmul_c11_parallel_loops.c",
    "cpu_c__matmul_c11_openblas_O3": "matmul_c11_openblas.c",
    # cpu_mkl
    "cpu_mkl__matmul_c11_intel_mkl_flags1": "matmul_c11_intel_mkl.c",
    "cpu_mkl__matmul_c11_intel_mkl_flags2": "matmul_c11_intel_mkl.c",
    "cpu_mkl__matmul_c11_intel_mkl_flags3": "matmul_c11_intel_mkl.c",
    "cpu_mkl__matmul_c11_intel_mkl_flags4": "matmul_c11_intel_mkl.c",
    "cpu_mkl__matmul_cpp20_intel_mkl": "matmul_cpp20_intel_mkl.cpp",
    # cuda
    "cuda__matmul_cuda": "matmul_cuda.cu",
    "cuda__matmul_cuda_faster": "matmul_cuda_faster.cu",
    "cuda__matmul_cuda_cpp20_faster": "matmul_cuda_cpp20_faster.cu",
    "cuda__matmul_cublas_c11": "matmul_cublas_c11.cu",
    "cuda__matmul_cublas_cpp20": "matmul_cublas_cpp20.cu",
    "cuda__matmul_cuda_optimized": "matmul_cuda_best.cu",
    "cuda__matmul_cublas_dgemm": "matmul_cuda_best.cu",
    # cpp26
    "cpp26__matmul_cpp26_cublas": "matmul_cpp26_cuda.cpp",
    "cpp26__matmul_cpp26_cublaslt": "matmul_cpp26_cuda.cpp",
    "cpp26__matmul_cpp26_graph": "matmul_cpp26_cuda.cpp",
    # rust
    "rust__matmul_rust_cublas": "matmul_rust_cublas.rs",
    "rust__matmul_rust_graph": "matmul_rust_cublas.rs",
    # kokkos
    "kokkos__matmul_kokkos": "matmul_kokkos.cpp",
    # julia
    "julia__matmul_julia_cpu": "matmul_julia_cpu.jl",
    "julia__matmul_cublas_julia": "matmul_cublas_julia.jl",
    "julia__matmul_julia_gpu": "matmul_julia_gpu.jl",
    "julia__matmul_julia_gpu_faster": "matmul_julia_gpu_faster.jl",
    "julia__matmul_julia_gpu_vendor_agnostic": "matmul_julia_gpu_vendor_agnostic.jl",
    # demos
    "demos__whoami": "whoami_cuda.cu",
    "demos__vector_add_v1": "vector_add_v1.cu",
    "demos__vector_add_v2": "vector_add_v2.cu",
}


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def main() -> int:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/latest")
    manifest = load_manifest(results_dir / "manifest.tsv")

    config = {}
    cfg_path = results_dir / "config.txt"
    if cfg_path.exists():
        for line in cfg_path.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                config[k.strip()] = v.strip()

    rows = []
    for entry in manifest:
        name = entry.get("name", "")
        source = entry.get("source") or SOURCE_MAP.get(name, "")
        category = entry.get("category", "")
        status = entry.get("status", "")
        outfile = entry.get("output") or ""
        wall = entry.get("wall_s", "")
        rc = entry.get("rc", "")
        note = entry.get("note", "")
        text = ""
        if outfile and Path(outfile).exists():
            text = Path(outfile).read_text(errors="replace")

        parsed = parse_output(text) if status in ("OK", "BUILT") else []
        if not parsed:
            rows.append({
                "category": category, "name": name, "status": status,
                "source": source,
                "seconds": None, "gflops": None, "notes": note,
                "wall_s": wall, "rc": rc,
            })
            continue
        for p in parsed:
            notes = list(p["notes"])
            rows.append({
                "category": category,
                "name": name + p["suffix"],
                "source": source,
                "status": status,
                "seconds": p["seconds"],
                "gflops": p["gflops"],
                "notes": "; ".join(n for n in notes if n),
                "wall_s": wall,
                "rc": rc,
            })

    # ---- CSV ---------------------------------------------------------------
    csv_path = results_dir / "summary.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "category", "name", "source", "status", "seconds", "avg_time",
            "gflops", "notes", "wall_s", "rc",
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "category": r["category"],
                "name": r["name"],
                "source": r["source"],
                "status": r["status"],
                "seconds": f"{r['seconds']:.9f}" if r["seconds"] is not None else "",
                "avg_time": fmt_time(r["seconds"]),
                "gflops": fmt_gflops(r["gflops"]),
                "notes": r["notes"],
                "wall_s": r["wall_s"],
                "rc": r["rc"],
            })

    # ---- Markdown ----------------------------------------------------------
    categories = []
    for r in rows:
        if r["category"] not in categories:
            categories.append(r["category"])

    lines = ["# matmul_optimization benchmark summary", ""]
    if config:
        lines.append("**Run configuration**")
        lines.append("")
        for k in ("date", "matrix_size", "runs", "threads", "cuda_arch", "timeout_s", "only", "skip"):
            if k in config:
                lines.append(f"- `{k}` = `{config[k]}`")
        lines.append("")
    ok = sum(1 for r in rows if r["status"] == "OK")
    other = len(rows) - ok
    lines.append(f"**{ok} benchmark(s) OK, {other} other/skipped.** "
                 "GFLOPS is as reported by the program, or computed as `2·N³/t` when the "
                 "program only prints a time.")
    lines.append("")

    # ---- one consolidated table containing every benchmark ----------------
    # Sorted by GFLOPS, highest first; rows without a GFLOPS value go last.
    def sort_key(r):
        g = r["gflops"]
        return (g is None, -(g or 0.0), r["name"])

    all_table = ["| Category | Benchmark | Status | Avg time | GFLOPS | Notes |",
                 "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=sort_key):
        all_table.append("| {} | `{}` | {} | {} | {} | {} |".format(
            r["category"], r["source"] or r["name"], r["status"],
            fmt_time(r["seconds"]),
            fmt_gflops(r["gflops"]), r["notes"].replace("|", "\\|"),
        ))

    lines.append("## All results")
    lines.append("")
    lines.extend(all_table)
    lines.append("")
    (results_dir / "all_results.md").write_text(
        "# All benchmark results (one table)\n\n" + "\n".join(all_table) + "\n")

    lines.append("## Detail by category")
    lines.append("")

    for cat in categories:
        cat_rows = [r for r in rows if r["category"] == cat]
        cat_rows.sort(key=lambda r: (r["gflops"] is None, -(r["gflops"] or 0.0), r["name"]))
        lines.append(f"### {cat}")
        lines.append("")
        lines.append("| Benchmark | Status | Avg time | GFLOPS | Notes |")
        lines.append("|---|---|---|---|---|")
        for r in cat_rows:
            lines.append("| `{}` | {} | {} | {} | {} |".format(
                r["source"] or r["name"], r["status"], fmt_time(r["seconds"]),
                fmt_gflops(r["gflops"]), r["notes"].replace("|", "\\|"),
            ))
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"Raw outputs and `manifest.tsv` are in `{results_dir}`.")
    lines.append("")

    md = "\n".join(lines)
    (results_dir / "summary.md").write_text(md)

    # ---- single self-contained file: report + raw output of every benchmark --
    full = list(lines)
    full.append("---")
    full.append("")
    full.append("## Raw output of every benchmark")
    full.append("")
    for entry in manifest:
        out = entry.get("output") or ""
        if not out or not Path(out).exists():
            continue
        text = Path(out).read_text(errors="replace").rstrip()
        tlines = text.splitlines()
        status = entry.get("status", "")
        wall = entry.get("wall_s", "")
        full.append(f"### `{entry.get('name', '')}` -- {status}, {wall} s")
        full.append("")
        full.append("```")
        if len(tlines) > 60:
            full.extend(tlines[:45])
            full.append(f"... ({len(tlines)} lines total; full file: {out}) ...")
            full.extend(tlines[-10:])
        else:
            full.extend(tlines)
        full.append("```")
        full.append("")
    (results_dir / "FULL_REPORT.md").write_text("\n".join(full) + "\n")

    sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
