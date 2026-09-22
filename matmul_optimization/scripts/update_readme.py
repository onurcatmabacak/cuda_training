#!/usr/bin/env python3
"""Insert the latest all-results table into one or more README files.

    update_readme.py <results_dir> <readme> [<readme> ...]

Everything between the markers

    <!-- RESULTS:START -->
    <!-- RESULTS:END -->

is replaced with a short header (size/runs/threads/date) plus the single
all-results table.  Files without the markers are skipped.
"""
from __future__ import annotations

import sys
from pathlib import Path

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


def build_block(results_dir: Path) -> str:
    cfg = {}
    cfg_path = results_dir / "config.txt"
    if cfg_path.exists():
        for line in cfg_path.read_text().splitlines():
            if "=" in line:
                key, val = line.split("=", 1)
                cfg[key.strip()] = val.strip()

    table_path = results_dir / "all_results.md"
    table = table_path.read_text().rstrip() if table_path.exists() else ""
    lines = table.splitlines()
    if lines and lines[0].startswith("#"):      # drop the file's own title
        lines = lines[1:]
    body = "\n".join(lines).strip()

    header = (
        f"_{cfg.get('date', '?')} — size **{cfg.get('matrix_size', 'default')}**, "
        f"**{cfg.get('runs', 'default')}** runs, {cfg.get('threads', '?')} threads, "
        f"CUDA arch `{cfg.get('cuda_arch', '?')}`. GFLOPS is as reported, or computed "
        f"as `2·N³/t` when only a time is printed._"
    )
    return f"{START}\n\n{header}\n\n{body}\n\n{END}"


def update(readme: Path, new_block: str) -> bool:
    if not readme.exists():
        return False
    text = readme.read_text()
    if START not in text or END not in text:
        return False
    pre = text.split(START, 1)[0]
    post = text.split(END, 1)[1]
    readme.write_text(pre + new_block + post)
    return True


def main() -> int:
    results_dir = Path(sys.argv[1])
    new_block = build_block(results_dir)
    rc = 0
    for arg in sys.argv[2:]:
        path = Path(arg)
        if update(path, new_block):
            print(f"update_readme: updated {path}")
        else:
            print(f"update_readme: skipped {path} (missing markers)")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
