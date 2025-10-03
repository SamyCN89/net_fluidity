#!/usr/bin/env python3
"""
Compute dFC speed bootstrap tables (CSV only), no plotting.

Thin wrapper around scripts/bootstrap_speed_groups_cli.py invoking it in
compute-only mode (no plots). Use this when you want to generate or refresh
the CSVs quickly and plot later with plot_speed_bootstrap.py.

Examples
  python scripts/compute_speed_bootstrap.py \
    --tr 500 --subset regions500 --tau-index 0 \
    --pool-threshold median --pool-all \
    --n-boot 2000 --jobs 8

CSV outputs
- paths['speed']/<outdir>/speed_bootstrap_quantiles.csv
- paths['speed']/<outdir>/speed_bootstrap_diffs.csv
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute dFC speed bootstrap tables (no plotting).")
    ap.add_argument("--tr", type=int, default=500)
    ap.add_argument("--subset", type=str, default=None)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--tau-index", type=int, default=0)
    ap.add_argument("--q", type=str, default="1,5,50,95,99")
    ap.add_argument("--pairs", type=str, default="(WT,VEH)-(WT,LCTB92);(Dp1Yey,VEH)-(Dp1Yey,LCTB92);(WT,VEH)-(Dp1Yey,VEH);(WT,LCTB92)-(Dp1Yey,LCTB92)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ci", type=float, default=95.0)
    ap.add_argument("--pool-threshold", type=str, default=None)
    ap.add_argument("--pool-all", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--parallel-scope", type=str, default="windows")
    ap.add_argument("--append-subset-to-outdir", action="store_true")
    ap.add_argument("--load-cache", action="store_true")
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
        "--tau-index",
        str(args.tau_index),
        "--q",
        str(args.q),
        "--pairs",
        str(args.pairs),
        "--n-boot",
        str(args.n_boot),
        "--seed",
        str(args.seed),
        "--ci",
        str(args.ci),
        "--jobs",
        str(args.jobs),
        "--parallel-scope",
        str(args.parallel_scope),
    ]
    if args.subset:
        cmd += ["--subset", args.subset]
    if args.outdir:
        cmd += ["--outdir", args.outdir]
    if args.pool_threshold is not None:
        cmd += ["--pool-threshold", args.pool_threshold]
    if args.pool_all:
        cmd += ["--pool-all"]
    if args.append_subset_to_outdir:
        cmd += ["--append-subset-to-outdir"]
    if args.load_cache:
        cmd += ["--load-cache"]

    # Do not pass any plotting flags — compute only
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

