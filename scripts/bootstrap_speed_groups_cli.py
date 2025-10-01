#!/usr/bin/env python3
"""
Bootstrap speed percentiles per group, per region and window, with comparisons.

Features
- Loads dFC speed per-window NPZs (as written by julien_data/src/speed_compute.py).
- For each region subfolder (regions-<label>) and each window file:
  - Pools per-animal values (optionally select a tau) and bootstraps percentiles
    q1, q5, q50, q95, q99 for each group.
  - Computes pairwise percentile-difference CIs for target group pairs.
- Optionally aggregates windows into short/long pools and repeats the above.
- Writes tidy CSV tables to reports/ with per-group quantiles and pairwise diffs.

Usage
  python scripts/bootstrap_speed_groups_cli.py \
    --tr 500 --subset shared --tau-index 0 \
    --group-cols genotype,treatment \
    --pairs "(WT,VEH)-(WT,LCTB92);(Dp1Yey,VEH)-(Dp1Yey,LCTB92);(WT,VEH)-(Dp1Yey,VEH);(WT,LCTB92)-(Dp1Yey,LCTB92)" \
    --pool-threshold median

Notes
- Expects per-region outputs under results/<dataset>/speed/regions-<label>/.
  If not present, falls back to a single "all" folder.
- Output CSVs:
  - reports/speed_bootstrap_quantiles.csv
  - reports/speed_bootstrap_diffs.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable, Tuple, List, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe headless plotting
import matplotlib.pyplot as plt

# Robust imports whether run from repo root or from scripts/ directory
try:
    from scripts.speed_bootstrap_nb import (
        get_context,
        load_per_animal_from_npz,
        build_groups_from_columns,
        bootstrap_quantiles_by_group,
        bootstrap_quantile_diffs_by_keys,
    )
except ModuleNotFoundError:
    try:
        # When executed from within scripts/ as CWD
        from speed_bootstrap_nb import (
            get_context,
            load_per_animal_from_npz,
            build_groups_from_columns,
            bootstrap_quantiles_by_group,
            bootstrap_quantile_diffs_by_keys,
        )
    except ModuleNotFoundError:
        # Last resort: append repo root to path and retry
        import sys as _sys
        here = Path(__file__).resolve()
        repo_root = here.parents[1]
        if str(repo_root) not in _sys.path:
            _sys.path.insert(0, str(repo_root))
        from scripts.speed_bootstrap_nb import (
            get_context,
            load_per_animal_from_npz,
            build_groups_from_columns,
            bootstrap_quantiles_by_group,
            bootstrap_quantile_diffs_by_keys,
        )


def _parse_pairs(pairs_arg: str) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
    """Parse pairs string like "(WT,VEH)-(WT,LCTB92);(Dp1Yey,VEH)-(Dp1Yey,LCTB92)"."""
    out: List[Tuple[Tuple[str, str], Tuple[str, str]]] = []
    if not pairs_arg:
        return out
    for chunk in pairs_arg.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            left, right = chunk.split("-", 1)
            l = left.strip().strip("()"),
            r = right.strip().strip("()")
            la, lb = [s.strip() for s in l[0].split(",", 1)]
            ra, rb = [s.strip() for s in r.split(",", 1)]
            out.append(((la, lb), (ra, rb)))
        except Exception as e:
            raise ValueError(f"Invalid pair spec: {chunk}") from e
    return out


def _find_region_folders(speed_root: Path) -> List[Path]:
    """Return list of region subfolders; fallback to ['all'] if none found."""
    cands = sorted([p for p in speed_root.iterdir() if p.is_dir() and p.name.startswith("regions-")])
    if cands:
        return cands
    # Fallback to a single 'all' folder or the root itself
    all_dir = speed_root / "all"
    return [all_dir] if all_dir.exists() else [speed_root]


_WIN_RE = re.compile(r"speed_win(\d+)_.*\.npz$")


def _list_window_files(region_dir: Path) -> List[Tuple[int, Path]]:
    files = []
    for p in sorted(region_dir.glob("speed_win*_*.npz")):
        m = _WIN_RE.search(p.name)
        if m:
            files.append((int(m.group(1)), p))
    return files


def _pool_windows_indices(windows: List[int], threshold: str | int | None) -> Dict[str, List[int]]:
    if threshold is None:
        return {}
    vals = sorted(windows)
    if isinstance(threshold, str) and threshold.lower() == "median":
        cut = int(np.median(vals))
    else:
        cut = int(threshold)
    return {
        "short": [w for w in vals if w <= cut],
        "long": [w for w in vals if w > cut],
    }


def _concat_per_animal(per_animals: List[List[np.ndarray]]) -> List[np.ndarray]:
    """Concatenate multiple per_animal lists along sample axis per animal index."""
    if not per_animals:
        return []
    n = max(len(x) for x in per_animals)
    out: List[np.ndarray] = []
    for i in range(n):
        parts = []
        for lst in per_animals:
            if i < len(lst) and lst[i].size > 0:
                parts.append(lst[i])
        out.append(np.concatenate(parts) if parts else np.array([], float))
    return out


def _plot_pairs_grid(region_label: str, win_label: str,
                     pairs_qd: Dict[Tuple[Tuple[str, str], Tuple[str, str]], dict],
                     pairs_order: List[Tuple[Tuple[str, str], Tuple[str, str]]],
                     fig_path: Path, cols: int = 2) -> None:
    try:
        from scripts.speed_bootstrap_nb import plot_quantile_diffs
    except ModuleNotFoundError:
        from speed_bootstrap_nb import plot_quantile_diffs  # type: ignore
    n = len(pairs_order)
    cols = max(1, int(cols))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.5 * rows), squeeze=False)
    for k, pair in enumerate(pairs_order):
        r, c = divmod(k, cols)
        qd = pairs_qd.get(pair)
        if qd is None:
            axes[r, c].axis('off')
            continue
        A, B = pair
        plot_quantile_diffs(qd, title=f"{A}-{B}", ax=axes[r, c])
    # Hide unused axes
    for k in range(n, rows * cols):
        r, c = divmod(k, cols)
        axes[r, c].axis('off')
    fig.suptitle(f"{region_label} | {win_label}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Bootstrap speed percentiles per group/region/window.")
    ap.add_argument("--tr", type=int, default=500, help="Select metadata by total_tr (e.g., 500).")
    ap.add_argument("--subset", type=str, default=None, help="Subset subfolder under speed/ (e.g., 'shared').")
    ap.add_argument("--tau-index", type=int, default=0, help="Tau index to select (None = pool all taus).")
    ap.add_argument("--group-cols", type=str, default="genotype,treatment", help="Grouping columns, comma-separated.")
    ap.add_argument("--q", type=str, default="1,5,50,95,99", help="Percentiles to bootstrap, comma-separated.")
    ap.add_argument("--pairs", type=str, default="(WT,VEH)-(WT,LCTB92);(Dp1Yey,VEH)-(Dp1Yey,LCTB92);(WT,VEH)-(Dp1Yey,VEH);(WT,LCTB92)-(Dp1Yey,LCTB92)", help="Pairs A-B; semicolon-separated.")
    ap.add_argument("--n-boot", type=int, default=2000, help="Bootstrap resamples.")
    ap.add_argument("--ci", type=float, default=95.0, help="Confidence level (percent).")
    ap.add_argument("--pool-threshold", type=str, default=None, help="Pool windows into short/long by 'median' or integer cutoff.")
    ap.add_argument("--pool-all", action="store_true", help="Also add an 'all' pool combining all windows.")
    ap.add_argument("--outdir", type=str, default="reports", help="Output directory for CSV tables.")
    ap.add_argument("--plot", action="store_true", help="Save plots for per-group quantiles and pairwise diffs.")
    ap.add_argument("--plot-format", type=str, default="png", choices=["png", "pdf", "svg"], help="Image format.")
    ap.add_argument("--grid", action="store_true", help="Also save a grid figure aggregating all pairwise diffs per window/pool.")
    ap.add_argument("--grid-cols", type=int, default=2, help="Max columns in grid layout for pairwise diffs.")
    args = ap.parse_args()

    q = [float(s) for s in args.q.split(",") if s.strip()]
    groups_to_compare = _parse_pairs(args.pairs)
    group_cols = [s.strip() for s in args.group_cols.split(",") if s.strip()]

    # Load context/data
    data = get_context(tr=args.tr)
    speed_root = Path(data.paths["speed"])  # type: ignore[index]
    if args.subset:
        speed_root = speed_root / args.subset
    region_dirs = _find_region_folders(speed_root)

    # Build grouping mapping from cognitive data
    groups_map = build_groups_from_columns(data.cog_data_filtered, group_cols)

    # Prepare outputs
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Figure output root (prefer dataset figures path)
    fig_root = Path(data.paths.get("f_speed", outdir / "figs"))  # type: ignore[attr-defined]
    if args.subset:
        fig_root = fig_root / str(args.subset)
    fig_root.mkdir(parents=True, exist_ok=True)
    quantiles_rows: List[Dict[str, object]] = []
    diffs_rows: List[Dict[str, object]] = []

    def _sanitize(s: str) -> str:
        return (
            str(s)
            .replace("/", "-")
            .replace(" ", "_")
            .replace(",", "-")
            .replace("|", "-")
            .replace("(", "").replace(")", "")
        )

    for region_dir in region_dirs:
        region_label = region_dir.name.replace("regions-", "") if region_dir.name.startswith("regions-") else region_dir.name
        win_files = _list_window_files(region_dir)
        if not win_files:
            continue
        windows = [w for (w, _) in win_files]
        pools = _pool_windows_indices(windows, args.pool_threshold)

        # Per-window processing
        for win, npz in win_files:
            per_animal = load_per_animal_from_npz(npz, tau_index=None if args.tau_index < 0 else args.tau_index)
            # Per-group quantiles
            qa = bootstrap_quantiles_by_group(per_animal, groups_map, q=q, n_boot=args.n_boot, ci=args.ci)
            for gk, res in qa.items():
                for qi, pt, lo, hi in zip(res["q"], res["point"], res["lo"], res["hi"], strict=False):
                    quantiles_rows.append({
                        "region": region_label,
                        "window": int(win),
                        "group": gk,
                        "q": float(qi),
                        "point": float(pt),
                        "lo": float(lo),
                        "hi": float(hi),
                        "n": int(res["n"]),
                    })
            if args.plot:
                # Plot per-group quantiles for this window
                try:
                    from scripts.speed_bootstrap_nb import plot_group_quantiles
                except ModuleNotFoundError:
                    from speed_bootstrap_nb import plot_group_quantiles  # type: ignore
                ax = plot_group_quantiles(qa, title=f"{region_label} | win={win}")
                fig_path = fig_root / f"quantiles_{_sanitize(region_label)}_win{win}.{args.plot_format}"
                ax.figure.savefig(fig_path, dpi=150, bbox_inches="tight")
                plt.close(ax.figure)
            # Pairwise diffs
            window_pairs_qd: Dict[Tuple[Tuple[str, str], Tuple[str, str]], dict] = {}
            for (A, B) in groups_to_compare:
                if A not in groups_map or B not in groups_map:
                    continue
                qd = bootstrap_quantile_diffs_by_keys(per_animal, groups_map, A, B, q=q, n_boot=args.n_boot, ci=args.ci)
                window_pairs_qd[(A, B)] = qd
                for qi, pt, lo, hi, sig in zip(qd["q"], qd["point"], qd["lo"], qd["hi"], qd["sig"], strict=False):
                    diffs_rows.append({
                        "region": region_label,
                        "window": int(win),
                        "A": A,
                        "B": B,
                        "q": float(qi),
                        "diff": float(pt),
                        "lo": float(lo),
                        "hi": float(hi),
                        "significant": bool(sig),
                        "n_a": int(qd.get("n_x", 0)),
                        "n_b": int(qd.get("n_y", 0)),
                    })
                if args.plot:
                    try:
                        from scripts.speed_bootstrap_nb import plot_quantile_diffs
                    except ModuleNotFoundError:
                        from speed_bootstrap_nb import plot_quantile_diffs  # type: ignore
                    ax = plot_quantile_diffs(qd, title=f"{region_label} | win={win} | {A}-{B}")
                    a_str = _sanitize(A)
                    b_str = _sanitize(B)
                    fig_path = fig_root / f"diffs_{_sanitize(region_label)}_win{win}_{a_str}_vs_{b_str}.{args.plot_format}"
                    ax.figure.savefig(fig_path, dpi=150, bbox_inches="tight")
                    plt.close(ax.figure)
            # Save grid per window
            if args.plot and args.grid and window_pairs_qd:
                grid_path = fig_root / f"grid_{_sanitize(region_label)}_win{win}.{args.plot_format}"
                _plot_pairs_grid(region_label, f"win={win}", window_pairs_qd, groups_to_compare, grid_path, cols=args.grid_cols)

        # Window pools (short/long)
        if pools:
            by_win = {w: p for (w, p) in win_files}
            # Optionally add an 'all' pool across all windows
            pool_items = dict(pools)
            if args.pool_all:
                pool_items["all"] = windows
            for pool_name, pool_windows in pool_items.items():
                if not pool_windows:
                    continue
                per_animals = [load_per_animal_from_npz(by_win[w], tau_index=None if args.tau_index < 0 else args.tau_index) for w in pool_windows if w in by_win]
                pooled = _concat_per_animal(per_animals)
                qa = bootstrap_quantiles_by_group(pooled, groups_map, q=q, n_boot=args.n_boot, ci=args.ci)
                for gk, res in qa.items():
                    for qi, pt, lo, hi in zip(res["q"], res["point"], res["lo"], res["hi"], strict=False):
                        quantiles_rows.append({
                            "region": region_label,
                            "window": pool_name,
                            "group": gk,
                            "q": float(qi),
                            "point": float(pt),
                            "lo": float(lo),
                            "hi": float(hi),
                            "n": int(res["n"]),
                        })
                if args.plot:
                    try:
                        from scripts.speed_bootstrap_nb import plot_group_quantiles
                    except ModuleNotFoundError:
                        from speed_bootstrap_nb import plot_group_quantiles  # type: ignore
                    ax = plot_group_quantiles(qa, title=f"{region_label} | pool={pool_name}")
                    fig_path = fig_root / f"quantiles_{_sanitize(region_label)}_pool-{pool_name}.{args.plot_format}"
                    ax.figure.savefig(fig_path, dpi=150, bbox_inches="tight")
                    plt.close(ax.figure)
                pool_pairs_qd: Dict[Tuple[Tuple[str, str], Tuple[str, str]], dict] = {}
                for (A, B) in groups_to_compare:
                    if A not in groups_map or B not in groups_map:
                        continue
                    qd = bootstrap_quantile_diffs_by_keys(pooled, groups_map, A, B, q=q, n_boot=args.n_boot, ci=args.ci)
                    pool_pairs_qd[(A, B)] = qd
                    for qi, pt, lo, hi, sig in zip(qd["q"], qd["point"], qd["lo"], qd["hi"], qd["sig"], strict=False):
                        diffs_rows.append({
                            "region": region_label,
                            "window": pool_name,
                            "A": A,
                            "B": B,
                            "q": float(qi),
                            "diff": float(pt),
                            "lo": float(lo),
                            "hi": float(hi),
                            "significant": bool(sig),
                            "n_a": int(qd.get("n_x", 0)),
                            "n_b": int(qd.get("n_y", 0)),
                        })
                    if args.plot:
                        try:
                            from scripts.speed_bootstrap_nb import plot_quantile_diffs
                        except ModuleNotFoundError:
                            from speed_bootstrap_nb import plot_quantile_diffs  # type: ignore
                        ax = plot_quantile_diffs(qd, title=f"{region_label} | pool={pool_name} | {A}-{B}")
                        a_str = _sanitize(A)
                        b_str = _sanitize(B)
                        fig_path = fig_root / f"diffs_{_sanitize(region_label)}_pool-{pool_name}_{a_str}_vs_{b_str}.{args.plot_format}"
                        ax.figure.savefig(fig_path, dpi=150, bbox_inches="tight")
                        plt.close(ax.figure)
                if args.plot and args.grid and pool_pairs_qd:
                    grid_path = fig_root / f"grid_{_sanitize(region_label)}_pool-{pool_name}.{args.plot_format}"
                    _plot_pairs_grid(region_label, f"pool={pool_name}", pool_pairs_qd, groups_to_compare, grid_path, cols=args.grid_cols)

    # Write CSVs
    q_path = outdir / "speed_bootstrap_quantiles.csv"
    d_path = outdir / "speed_bootstrap_diffs.csv"
    if quantiles_rows:
        q_cols = ["region", "window", "group", "q", "point", "lo", "hi", "n"]
        with q_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=q_cols)
            w.writeheader()
            w.writerows(quantiles_rows)
    if diffs_rows:
        d_cols = ["region", "window", "A", "B", "q", "diff", "lo", "hi", "significant", "n_a", "n_b"]
        with d_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=d_cols)
            w.writeheader()
            w.writerows(diffs_rows)

    print(f"Wrote: {q_path}")
    print(f"Wrote: {d_path}")


if __name__ == "__main__":
    main()
