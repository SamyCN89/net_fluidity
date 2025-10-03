#!/usr/bin/env python3
"""
Plot dFC speed bootstrap figures from existing CSVs (plot-only).

Thin wrapper that calls scripts/bootstrap_speed_groups_cli.py with --plot-only
so no compute/bootstrapping is performed. Useful for fast iteration.

Examples
  # Diff(A-B) vs window per ROI and per pair, plus a grid per ROI
  python scripts/plot_speed_bootstrap.py \
    --tr 500 --subset regions500 \
    --plot-diffs-by-win --plot-diffs-bywin-grid --bywin-grid-cols 2

Optional toggles
- --no-quantiles-win --no-diffs-win to suppress per-window panels if you later
  extend this wrapper to include them. Currently this wrapper focuses on the
  summary by-window plots and grids.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot dFC speed bootstrap figures from existing CSVs (plot-only).")
    ap.add_argument("--tr", type=int, default=500)
    ap.add_argument("--subset", type=str, default=None)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--plot-format", type=str, default="png", choices=["png", "pdf", "svg"])
    ap.add_argument("--plot-diffs-by-win", action="store_true")
    ap.add_argument("--plot-diffs-bywin-grid", action="store_true")
    ap.add_argument("--bywin-grid-cols", type=int, default=2)
    ap.add_argument("--append-subset-to-outdir", action="store_true")
    ap.add_argument("--reuse-group-boots", action="store_true", help="Pass through to CLI for symmetry; plot-only mode ignores it.")
    args = ap.parse_args()

    here = Path(__file__).resolve()
    cli = here.with_name("bootstrap_speed_groups_cli.py")
    if not cli.exists():
        print(f"Underlying CLI not found: {cli}")
        return 2

    cmd = [
        sys.executable,
        str(cli),
        "--tr",
        str(args.tr),
        "--plot-only",
        "--plot",
        "--plot-format",
        str(args.plot_format),
    ]
    if args.subset:
        cmd += ["--subset", args.subset]
    if args.outdir:
        cmd += ["--outdir", args.outdir]
    if args.append_subset_to_outdir:
        cmd += ["--append-subset-to-outdir"]
    if args.plot_diffs_by_win:
        cmd += ["--plot-diffs-by-win"]
    if args.plot_diffs_bywin_grid:
        cmd += ["--plot-diffs-bywin-grid", "--bywin-grid-cols", str(args.bywin_grid_cols)]
    if args.reuse_group_boots:
        cmd += ["--reuse-group-boots"]

    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
