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
from joblib import Parallel, delayed
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # fallback if tqdm is not installed

# Robust imports whether run from repo root or from scripts/ directory
try:
    from scripts.speed_bootstrap_nb import (
        get_context,
        load_per_animal_from_npz,
        build_groups_from_columns,
        bootstrap_quantiles_by_group,
        bootstrap_quantile_diffs_by_keys,
        plot_group_quantiles,
        plot_quantile_diffs,
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
            plot_group_quantiles,
            plot_quantile_diffs,
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
            plot_group_quantiles,
            plot_quantile_diffs,
        )
        # Optional centralized kernels for reuse optimization
        try:
            from shared_code.shared_code.fun_bootstrap import (
                bootstrap_groups_boots as _central_bootstrap_groups_boots,
                ci_from_boots as _central_ci_from_boots,
                pool_per_animal as _central_pool_per_animal,
            )
        except Exception:
            _central_bootstrap_groups_boots = None
            _central_ci_from_boots = None
            _central_pool_per_animal = None


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
    """Return list of region subfolders; fallback to ['all'] or base if none found.

    Detection rules (in order):
    - Any subfolder whose name starts with 'regions-'
    - Any subfolder that contains at least one per-window NPZ (speed_win*_*.npz)
    - Else, fallback to 'all' or the root itself
    """
    # 1) Prefixed region folders
    prefixed = sorted([p for p in speed_root.iterdir() if p.is_dir() and p.name.startswith("regions-")])
    if prefixed:
        return prefixed
    # 2) Any subfolder with per-window NPZs
    generic = []
    for p in sorted([x for x in speed_root.iterdir() if x.is_dir()]):
        if list(p.glob("speed_win*_*.npz")):
            generic.append(p)
    if generic:
        return generic
    # 3) Fallback to a single 'all' folder or the root itself
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


_SUBSET_TAG_RE = re.compile(
    r"_subset_mode-[^_]*_(?:region-\d+-(?P<region>[^_]+)|lab-(?P<lab>[^_]+))"
)


def _infer_region_from_filename(name: str) -> str | None:
    """Try to extract a region label from NPZ filename subset tag.

    Supports patterns like:
      ..._subset_mode-touching_region-3-ACC_...
      ..._subset_mode-touching_lab-ACC_...
    Returns the inferred label or None if not found.
    """
    m = _SUBSET_TAG_RE.search(name)
    if not m:
        return None
    if m.group("region"):
        return m.group("region")
    if m.group("lab"):
        # When multiple labels present, return the full joined string
        return m.group("lab")
    return None


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


def _process_window_and_return(
    region_label: str,
    win: int,
    npz: Path,
    tau_index: int,
    groups_map: Dict,
    groups_to_compare: List[Tuple[Tuple[str, str], Tuple[str, str]]],
    q: List[float],
    n_boot: int,
    ci: float,
    seed: int,
    early_stop: float,
    chunk: int,
    boots_float32: bool,
    reuse_group_boots: bool,
    plot: bool,
    plot_format: str,
    no_quantiles_win: bool,
    no_diffs_win: bool,
    fig_root: Path,
    load_cache: bool,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[Tuple[Tuple[str, str], Tuple[str, str]], dict]]:
    def _sanitize(s: str) -> str:
        return (
            str(s)
            .replace("/", "-")
            .replace(" ", "_")
            .replace(",", "-")
            .replace("|", "-")
            .replace("(", "").replace(")", "")
        )

    quant_rows: List[Dict[str, object]] = []
    diff_rows: List[Dict[str, object]] = []
    window_pairs_qd: Dict[Tuple[Tuple[str, str], Tuple[str, str]], dict] = {}
    # Infer display region from filename if present (fallback to provided label)
    try:
        region_disp = _infer_region_from_filename(npz.name) or region_label
    except Exception:
        region_disp = region_label

    # Load data and bootstrap
    per_animal = load_per_animal_from_npz(npz, tau_index=None if tau_index < 0 else tau_index)
    if reuse_group_boots and _central_bootstrap_groups_boots is not None and _central_ci_from_boots is not None:
        boots_map = _central_bootstrap_groups_boots(
            per_animal, groups_map, q=q, n_boot=n_boot, seed=seed, chunk=chunk, dtype=(np.float32 if boots_float32 else float)
        )
        q_arr = boots_map.get('__q__', np.asarray(q, float))
        qa = {}
        for gk, idxs in groups_map.items():
            if gk == '__q__':
                continue
            pooled_g = _central_pool_per_animal(per_animal, idxs) if _central_pool_per_animal is not None else np.concatenate([per_animal[i] for i in idxs if i < len(per_animal)])
            point = np.percentile(pooled_g, q_arr)
            lo, hi = _central_ci_from_boots(boots_map[gk], ci=ci)
            qa[gk] = {"q": q_arr, "point": point, "lo": lo, "hi": hi, "n": int(pooled_g.size)}
            for qi, pt, lo_i, hi_i in zip(q_arr, point, lo, hi, strict=False):
                quant_rows.append({
                    "region": region_disp,
                    "roi": region_disp,
                    "window": int(win),
                    "group": gk,
                    "q": float(qi),
                    "point": float(pt),
                    "lo": float(lo_i),
                    "hi": float(hi_i),
                    "n": int(pooled_g.size),
                })
    else:
        qa = bootstrap_quantiles_by_group(
            per_animal,
            groups_map,
            q=q,
            n_boot=n_boot,
            ci=ci,
            seed=seed,
            early_stop=early_stop,
            _chunk=chunk,
            boots_float32=boots_float32,
        )
        for gk, res in qa.items():
            for qi, pt, lo, hi in zip(res["q"], res["point"], res["lo"], res["hi"], strict=False):
                quant_rows.append({
                    "region": region_disp,
                    "roi": region_disp,
                    "window": int(win),
                    "group": gk,
                    "q": float(qi),
                    "point": float(pt),
                    "lo": float(lo),
                    "hi": float(hi),
                    "n": int(res["n"]),
                })
    # Plot per-group quantiles for this window if requested
    if plot and not no_quantiles_win:
        ax = plot_group_quantiles(qa, title=f"{region_disp} | win={win}")
        fig_path = fig_root / f"quantiles_{_sanitize(region_disp)}_win{win}.{plot_format}"
        if not (load_cache and fig_path.exists()):
            ax.figure.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(ax.figure)

    # Pairwise diffs
    for (A, B) in groups_to_compare:
        if A not in groups_map or B not in groups_map:
            continue
        if reuse_group_boots and _central_bootstrap_groups_boots is not None and _central_ci_from_boots is not None:
            q_arr = boots_map.get('__q__', np.asarray(q, float))
            boots_A = boots_map.get(A); boots_B = boots_map.get(B)
            if boots_A is None or boots_B is None:
                continue
            n_used = min(boots_A.shape[0], boots_B.shape[0])
            diff_boots = boots_A[:n_used, :] - boots_B[:n_used, :]
            lo, hi = _central_ci_from_boots(diff_boots, ci=ci)
            pooled_A = _central_pool_per_animal(per_animal, groups_map[A]) if _central_pool_per_animal is not None else np.concatenate([per_animal[i] for i in groups_map[A] if i < len(per_animal)])
            pooled_B = _central_pool_per_animal(per_animal, groups_map[B]) if _central_pool_per_animal is not None else np.concatenate([per_animal[i] for i in groups_map[B] if i < len(per_animal)])
            point = np.percentile(pooled_A, q_arr) - np.percentile(pooled_B, q_arr)
            sig = (lo > 0) | (hi < 0)
            qd = {"q": q_arr, "point": point, "lo": lo, "hi": hi, "sig": sig, "n_x": int(pooled_A.size), "n_y": int(pooled_B.size)}
            window_pairs_qd[(A, B)] = qd
            for qi, pt, lo_i, hi_i, s in zip(q_arr, point, lo, hi, sig, strict=False):
                diff_rows.append({
                    "region": region_disp,
                    "roi": region_disp,
                    "window": int(win),
                    "A": A,
                    "B": B,
                    "q": float(qi),
                    "diff": float(pt),
                    "lo": float(lo_i),
                    "hi": float(hi_i),
                    "significant": bool(s),
                    "n_a": int(pooled_A.size),
                    "n_b": int(pooled_B.size),
                })
            if plot and not no_diffs_win:
                ax = plot_quantile_diffs(qd, title=f"{region_disp} | win={win} | {A}-{B}")
                a_str = _sanitize(A)
                b_str = _sanitize(B)
                fig_path = fig_root / f"diffs_{_sanitize(region_disp)}_win{win}_{a_str}_vs_{b_str}.{plot_format}"
                if not (load_cache and fig_path.exists()):
                    ax.figure.savefig(fig_path, dpi=150, bbox_inches="tight")
                plt.close(ax.figure)
        else:
            qd = bootstrap_quantile_diffs_by_keys(
                per_animal,
                groups_map,
                A,
                B,
                q=q,
                n_boot=n_boot,
                ci=ci,
                seed=seed,
                early_stop=early_stop,
                _chunk=chunk,
                boots_float32=boots_float32,
            )
            window_pairs_qd[(A, B)] = qd
            for qi, pt, lo, hi, sig in zip(qd["q"], qd["point"], qd["lo"], qd["hi"], qd["sig"], strict=False):
                diff_rows.append({
                    "region": region_disp,
                    "roi": region_disp,
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
            if plot and not no_diffs_win:
                ax = plot_quantile_diffs(qd, title=f"{region_disp} | win={win} | {A}-{B}")
                a_str = _sanitize(A)
                b_str = _sanitize(B)
                fig_path = fig_root / f"diffs_{_sanitize(region_disp)}_win{win}_{a_str}_vs_{b_str}.{plot_format}"
                if not (load_cache and fig_path.exists()):
                    ax.figure.savefig(fig_path, dpi=150, bbox_inches="tight")
                plt.close(ax.figure)

    return quant_rows, diff_rows, window_pairs_qd


def main():
    ap = argparse.ArgumentParser(description="Bootstrap speed percentiles per group/region/window.")
    ap.add_argument("--tr", type=int, default=500, help="Select metadata by total_tr (e.g., 500).")
    ap.add_argument("--subset", type=str, default=None, help="Subset subfolder under speed/ (e.g., 'shared').")
    ap.add_argument("--tau-index", type=int, default=0, help="Tau index to select (None = pool all taus).")
    ap.add_argument("--group-cols", type=str, default="genotype,treatment", help="Grouping columns, comma-separated.")
    ap.add_argument("--q", type=str, default="1,5,50,95,99", help="Percentiles to bootstrap, comma-separated.")
    ap.add_argument("--pairs", type=str, default="(WT,VEH)-(WT,LCTB92);(Dp1Yey,VEH)-(Dp1Yey,LCTB92);(WT,VEH)-(Dp1Yey,VEH);(WT,LCTB92)-(Dp1Yey,LCTB92)", help="Pairs A-B; semicolon-separated.")
    ap.add_argument("--n-boot", type=int, default=2000, help="Bootstrap resamples. Default 2000.")
    ap.add_argument("--reuse-group-boots", action="store_true", help="Reuse per-group bootstrap replicates to compute all pairs (faster for many pairs).")
    ap.add_argument("--early-stop", type=float, default=0.0, help="Adaptive CI tolerance (fraction). 0 disables.")
    ap.add_argument("--seed", type=int, default=0, help="Base random seed for bootstrapping.")
    ap.add_argument("--ci", type=float, default=95.0, help="Confidence level (percent).")
    ap.add_argument("--chunk", type=int, default=128, help="Bootstrap chunk size for vectorized resampling.")
    ap.add_argument("--boots-float32", action="store_true", help="Store bootstrap arrays in float32 to reduce memory.")
    ap.add_argument("--pool-threshold", type=str, default=None, help="Pool windows into short/long by 'median' or integer cutoff.")
    ap.add_argument("--pool-all", action="store_true", help="Also add an 'all' pool combining all windows.")
    ap.add_argument("--outdir", type=str, default=None, help="Output folder name under dataset paths; defaults to --subset or 'bootstrap'.")
    ap.add_argument("--plot", action="store_true", help="Save plots for per-group quantiles and pairwise diffs.")
    ap.add_argument("--plot-format", type=str, default="png", choices=["png", "pdf", "svg"], help="Image format.")
    ap.add_argument("--grid", action="store_true", help="Also save a grid figure aggregating all pairwise diffs per window/pool.")
    ap.add_argument("--grid-cols", type=int, default=2, help="Max columns in grid layout for pairwise diffs.")
    ap.add_argument("--no-diffs-win", action="store_true", help="Disable saving per-window pairwise diffs figures (diffs_*_win*.png).")
    ap.add_argument("--no-quantiles-win", action="store_true", help="Disable saving per-window group quantiles figures (quantiles_*_win*.png).")
    ap.add_argument("--plot-diffs-by-win", action="store_true", help="Save a summary plot per ROI and pair: diff(A-B) vs window with percentiles as legends (filled=significant, open=ns).")
    ap.add_argument("--plot-diffs-bywin-grid", action="store_true", help="Save a grid figure per ROI aggregating all by-window diff(A-B) vs window panels.")
    ap.add_argument("--bywin-grid-cols", type=int, default=2, help="Columns for bywin grid layout.")
    # Correlation with cognitive score (NOR)
    ap.add_argument("--correlate-nor", action="store_true", help="Compute correlation between per-animal speed percentiles and NOR cognitive score.")
    ap.add_argument("--nor-col", type=str, default=None, help="Column name for NOR score in cognitive data; auto-detect if omitted.")
    ap.add_argument("--plot-only", action="store_true", help="Generate figures from existing CSVs in outdir; skip all bootstrap and NPZ processing.")
    ap.add_argument("--show-tau", action="store_true", help="Print tau range from metadata and exit.")
    ap.add_argument("--progress", action="store_true", help="Show progress bars for regions/windows/pools (requires tqdm).")
    ap.add_argument("--load-cache", action="store_true", help="Reuse existing outputs if present; skip recomputation/plots where possible.")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel jobs for per-window processing (1 = serial).")
    ap.add_argument(
        "--parallel-scope",
        type=str,
        default="windows",
        choices=["windows"],
        help="Parallelization scope; currently supports 'windows' (within each region).",
    )
    ap.add_argument(
        "--append-subset-to-outdir",
        action="store_true",
        help="If set and --subset is provided, append '__subset-<subset>' to the outdir name for CSVs and figures.",
    )
    ap.add_argument("--print-args", action="store_true", help="Print effective arguments (including defaults applied) and continue.")
    ap.add_argument("--show-defaults", action="store_true", help="Print all default values and exit.")
    ap.add_argument("--dry-run", action="store_true", help="List planned regions/windows, pools, and output paths; then exit without computing.")
    ap.add_argument("--list-inputs", action="store_true", help="Print the full list of NPZ files that will be read.")

    # Capture defaults before parsing CLI args
    _defaults = ap.parse_args([])
    args = ap.parse_args()
    if args.show_defaults:
        print("Default arguments:")
        for k, v in sorted(vars(_defaults).items()):
            print(f"  --{k.replace('_','-')}: {v}")
        return
    if args.print_args:
        print("Effective arguments:")
        for k, v in sorted(vars(args).items()):
            print(f"  {k} = {v}")

    # q as list of floats
    q = [float(s) for s in args.q.split(",") if s.strip()]
    groups_to_compare = _parse_pairs(args.pairs)
    group_cols = [s.strip() for s in args.group_cols.split(",") if s.strip()]

    # Load context/data
    data = get_context(tr=args.tr)
    speed_root = Path(data.paths["speed"])  # type: ignore[index]
    # Validate tau index against metadata (tau_count = tau + 1)
    tau_count = int(getattr(data, 'tau', 0)) + 1
    if args.show_tau:
        print(f"tau (from metadata) = {getattr(data, 'tau', 0)}")
        print(f"tau_count (valid indices) = {tau_count} -> indices 0..{tau_count-1}")
        print("Use --tau-index -1 to pool all taus.")
        return
    if args.tau_index >= 0 and args.tau_index >= tau_count:
        raise ValueError(f"tau-index {args.tau_index} is out of range for tau_count={tau_count}. "
                         f"Use a value in [0, {tau_count-1}] or -1 to pool all taus.")
    if args.subset:
        speed_root = speed_root / args.subset
    region_dirs = _find_region_folders(speed_root)

    # Build grouping mapping from cognitive data
    groups_map = build_groups_from_columns(data.cog_data_filtered, group_cols)

    # Prepare outputs under dataset paths
    # CSV tables under paths['speed']/<outdir>
    outputs_root = Path(data.paths["speed"])  # type: ignore[index]
    # Choose base outdir name: prefer explicit --outdir; else fallback to --subset; else 'bootstrap'
    outdir_name = args.outdir if args.outdir else (args.subset if args.subset else "bootstrap")
    # Optionally append subset to an explicit outdir to avoid collisions
    if args.append_subset_to_outdir and args.subset and args.outdir:
        def _san(s: str) -> str:
            return (
                str(s)
                .replace("/", "-")
                .replace(" ", "_")
                .replace(",", "-")
                .replace("|", "-")
                .replace("(", "").replace(")", "")
            )
        outdir_name = f"{outdir_name}__subset-{_san(args.subset)}"
    outdir = outputs_root / outdir_name
    outdir.mkdir(parents=True, exist_ok=True)

    # Figures under paths['f_speed']/<outdir> (only if plotting is requested)
    fig_root = None
    if args.plot or args.grid or args.plot_diffs_by_win or args.plot_diffs_bywin_grid:
        fig_base = Path(data.paths.get("f_speed", outputs_root))  # type: ignore[attr-defined]
        fig_root = fig_base / outdir_name
        fig_root.mkdir(parents=True, exist_ok=True)

    # CSV paths
    q_path = outdir / "speed_bootstrap_quantiles.csv"
    d_path = outdir / "speed_bootstrap_diffs.csv"
    # Announce output locations upfront
    print(f"Output CSV dir: {outdir}")
    if fig_root is not None:
        print(f"Output figure dir: {fig_root}")
    if args.load_cache:
        print(f"Input data dir (NPZs): {speed_root}")
    if args.dry_run:
        # Show planned work and exit
        print("[dry-run] Would write CSV tables:")
        print(f"  {q_path}")
        print(f"  {d_path}")
        total_windows = 0
        print(f"[dry-run] Regions to process: {len(region_dirs)}")
        for region_dir in region_dirs:
            region_label = region_dir.name.replace("regions-", "") if region_dir.name.startswith("regions-") else region_dir.name
            win_files = _list_window_files(region_dir)
            windows = [w for (w, _) in win_files]
            total_windows += len(windows)
            pools = _pool_windows_indices(windows, args.pool_threshold)
            if windows:
                preview = windows[:10]
                extra = '...' if len(windows) > 10 else ''
                print(f"  - {region_label}: windows={len(windows)} ({preview}{extra})")
            else:
                print(f"  - {region_label}: windows=0")
            # Optionally list exact inputs
            if args.list_inputs and win_files:
                for w, npz in win_files:
                    print(f"      win={w}: {npz}")
            if pools:
                for pname, plist in pools.items():
                    print(f"      pool '{pname}': {len(plist)} windows")
            else:
                print("      pools: none")
            if args.plot:
                print(f"      figures dir: {fig_root}")
        print(f"[dry-run] Total window files: {total_windows}")
        return
    if args.load_cache and q_path.exists() and d_path.exists() and not args.plot:
        print(f"[cache] Found existing outputs: {q_path} and {d_path}. Skipping.")
        return


    quantiles_rows: List[Dict[str, object]] = []
    diffs_rows: List[Dict[str, object]] = []
    corr_rows: List[Dict[str, object]] = []
    # Plot-only: load CSVs and render figures without recompute
    if args.plot_only:
        if not q_path.exists() or not d_path.exists():
            raise FileNotFoundError(f"plot-only requires existing CSVs: {q_path} and {d_path}")
        with q_path.open("r", newline="") as f:
            quantiles_rows = list(csv.DictReader(f))
        with d_path.open("r", newline="") as f:
            diffs_rows = list(csv.DictReader(f))
        # Render bywin and grids if requested
        if fig_root is not None:
            def _san(s: str) -> str:
                return str(s).replace('/', '-').replace(' ', '_').replace(',', '-').replace('|', '-').replace('(', '').replace(')', '')
            # Build mapping: (roi, (A,B)) -> { q -> [(win, diff, sig), ...] }
            by_roi_pair_q: Dict[Tuple[str, Tuple[str, str]], Dict[float, List[Tuple[int, float, bool]]]] = {}
            for r in diffs_rows:
                roi = str(r.get("roi", r.get("region", "")))
                A = r.get("A"); B = r.get("B")
                if A is None or B is None:
                    continue
                key = (roi, (str(A), str(B)))
                # Skip pooled rows
                win_raw = r.get("window", -1)
                try:
                    win = int(win_raw)
                except Exception:
                    continue
                try:
                    qv = float(r.get("q", 0.0)); diffv = float(r.get("diff", 0.0))
                except Exception:
                    continue
                sig = str(r.get("significant", "False")).lower() in ("1","true","yes")
                by_roi_pair_q.setdefault(key, {}).setdefault(qv, []).append((win, diffv, sig))
            import matplotlib.pyplot as _plt
            # Individual bywin plots
            if args.plot_diffs_by_win and by_roi_pair_q:
                for (roi, (A, B)), qmap in by_roi_pair_q.items():
                    fig, ax = _plt.subplots(figsize=(8,4))
                    qs_sorted = sorted(qmap.keys())
                    palette = _plt.rcParams.get('axes.prop_cycle', None)
                    base_colors = (palette.by_key()['color'] if palette is not None else ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'])
                    color_for_q = {float(qi): base_colors[i % len(base_colors)] for i, qi in enumerate(qs_sorted)}
                    for qi in qs_sorted:
                        triples = sorted(qmap[qi], key=lambda t: t[0])
                        wins = [t[0] for t in triples]
                        diffs = [t[1] for t in triples]
                        sigs = [t[2] for t in triples]
                        color = color_for_q[float(qi)]
                        ax.plot(wins, diffs, label=f"q{int(qi)}", color=color)
                        for xw, yw, s in zip(wins, diffs, sigs):
                            ax.plot(xw, yw, 'o', color=color, mfc=(color if s else 'none'), mec=color)
                    ax.axhline(0.0, color='k', linewidth=1, alpha=0.5)
                    ax.set_xlabel('Window size'); ax.set_ylabel('Difference (A - B)')
                    ax.set_title(f"{roi} | {A}-{B}"); ax.legend(loc='best', title='Percentile')
                    fig.tight_layout()
                    a_str = _san(A); b_str = _san(B)
                    out_path = fig_root / f"bywin_{_san(roi)}_{a_str}_vs_{b_str}.{args.plot_format}"
                    fig.savefig(out_path, dpi=150, bbox_inches='tight'); _plt.close(fig)
            # Bywin grids per ROI
            if args.plot_diffs_bywin_grid and by_roi_pair_q:
                roi_to_pairs: Dict[str, List[Tuple[Tuple[str, str], Dict[float, List[Tuple[int, float, bool]]]]]] = {}
                for (roi, pair), qmap in by_roi_pair_q.items():
                    roi_to_pairs.setdefault(roi, []).append((pair, qmap))
                for roi, items in roi_to_pairs.items():
                    n = len(items); cols = max(1, int(args.bywin_grid_cols)); rows = int(np.ceil(n/cols))
                    fig, axes = _plt.subplots(rows, cols, figsize=(6*cols, 3.5*rows), squeeze=False)
                    all_qs = sorted({qi for _, qmap in items for qi in qmap.keys()})
                    palette = _plt.rcParams.get('axes.prop_cycle', None)
                    base_colors = (palette.by_key()['color'] if palette is not None else ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'])
                    color_for_q = {float(qi): base_colors[i % len(base_colors)] for i, qi in enumerate(all_qs)}
                    for idx, (pair, qmap) in enumerate(items):
                        r, c = divmod(idx, cols); ax = axes[r, c]
                        for qi in sorted(qmap.keys()):
                            triples = sorted(qmap[qi], key=lambda t: t[0])
                            wins = [t[0] for t in triples]; diffs = [t[1] for t in triples]; sigs = [t[2] for t in triples]
                            color = color_for_q[float(qi)]
                            ax.plot(wins, diffs, label=f"q{int(qi)}", color=color)
                            for xw, yw, s in zip(wins, diffs, sigs):
                                ax.plot(xw, yw, 'o', color=color, mfc=(color if s else 'none'), mec=color)
                        A, B = pair; ax.set_title(f"{A}-{B}"); ax.axhline(0.0, color='k', linewidth=1, alpha=0.5)
                        if r == rows-1: ax.set_xlabel('Window size')
                        if c == 0: ax.set_ylabel('Diff (A - B)')
                    for k in range(n, rows*cols): r, c = divmod(k, cols); axes[r, c].axis('off')
                    handles, labels = axes[0,0].get_legend_handles_labels();
                    if handles: fig.legend(handles, labels, loc='upper right', title='Percentile')
                    fig.suptitle(f"{roi} | diff(A-B) vs window"); fig.tight_layout(rect=[0,0,0.98,0.95])
                    grid_out = fig_root / f"bywin_grid_{_san(roi)}.{args.plot_format}"; fig.savefig(grid_out, dpi=150, bbox_inches='tight'); _plt.close(fig)
        return

    def _sanitize(s: str) -> str:
        return (
            str(s)
            .replace("/", "-")
            .replace(" ", "_")
            .replace(",", "-")
            .replace("|", "-")
            .replace("(", "").replace(")", "")
        )

    reg_iter = region_dirs
    if args.progress and tqdm is not None:
        reg_iter = tqdm(region_dirs, desc="Regions", unit="region")
    for region_dir in reg_iter:
        region_label = region_dir.name.replace("regions-", "") if region_dir.name.startswith("regions-") else region_dir.name
        win_files = _list_window_files(region_dir)
        if not win_files:
            continue
        windows = [w for (w, _) in win_files]
        pools = _pool_windows_indices(windows, args.pool_threshold)

        # Per-window processing
        if args.jobs > 1 and args.parallel_scope == "windows":
            tasks = win_files
            results = Parallel(n_jobs=args.jobs, prefer="processes")(
                delayed(_process_window_and_return)(
                    region_label, w, p, args.tau_index, groups_map, groups_to_compare, q,
                    args.n_boot, args.ci, args.seed, float(args.early_stop or 0.0), int(args.chunk), bool(args.boots_float32), bool(args.reuse_group_boots), args.plot, args.plot_format, args.no_quantiles_win, args.no_diffs_win, fig_root, args.load_cache
                )
                for (w, p) in tasks
            )
            for idx, (qr, dr, window_pairs_qd) in enumerate(results):
                quantiles_rows.extend(qr)
                diffs_rows.extend(dr)
                if args.plot and args.grid and window_pairs_qd:
                    w = tasks[idx][0]
                    try:
                        roi_name = _infer_region_from_filename(tasks[idx][1].name) or region_label
                    except Exception:
                        roi_name = region_label
                    grid_path = fig_root / f"grid_{_sanitize(roi_name)}_win{w}.{args.plot_format}"
                    if not (args.load_cache and grid_path.exists()):
                        _plot_pairs_grid(roi_name, f"win={w}", window_pairs_qd, groups_to_compare, grid_path, cols=args.grid_cols)
                    else:
                        print(f"[cache] Using cached grid figure: {grid_path}")
        else:
            win_iter = win_files
            if args.progress and tqdm is not None:
                win_iter = tqdm(win_files, desc=f"{region_label} windows", unit="win", leave=False)
            for win, npz in win_iter:
                qr, dr, window_pairs_qd = _process_window_and_return(
                    region_label, win, npz, args.tau_index, groups_map, groups_to_compare, q,
                    args.n_boot, args.ci, args.seed, float(args.early_stop or 0.0), int(args.chunk), bool(args.boots_float32), bool(args.reuse_group_boots), args.plot, args.plot_format, args.no_quantiles_win, args.no_diffs_win, fig_root, args.load_cache
                )
                quantiles_rows.extend(qr)
                diffs_rows.extend(dr)
                if args.plot and args.grid and window_pairs_qd:
                    try:
                        region_name = _infer_region_from_filename(npz.name) or region_label
                    except Exception:
                        region_name = region_label
                    grid_path = fig_root / f"grid_{_sanitize(region_name)}_win{win}.{args.plot_format}"
                    if not (args.load_cache and grid_path.exists()):
                        _plot_pairs_grid(region_name, f"win={win}", window_pairs_qd, groups_to_compare, grid_path, cols=args.grid_cols)
                    else:
                        print(f"[cache] Using cached grid figure: {grid_path}")

        # Window pools (short/long) — enforce per-ROI pooling
        # Build mapping from ROI -> list[(win, path)]
        roi_map: Dict[str, List[Tuple[int, Path]]] = {}
        for w, p in win_files:
            try:
                roi = _infer_region_from_filename(p.name) or region_label
            except Exception:
                roi = region_label
            roi_map.setdefault(roi, []).append((w, p))
        for roi_name, files in roi_map.items():
            roi_windows = [w for (w, _) in files]
            pools_roi = _pool_windows_indices(roi_windows, args.pool_threshold)
            if not pools_roi:
                continue
            by_win = {w: p for (w, p) in files}
            # Optionally add an 'all' pool across all ROI windows
            pool_items = dict(pools_roi)
            if args.pool_all:
                pool_items["all"] = roi_windows
            pool_iter = list(pool_items.items())
            if args.progress and tqdm is not None:
                pool_iter = tqdm(pool_iter, desc=f"{roi_name} pools", unit="pool", leave=False)
            for pool_name, pool_windows in pool_iter:
                if not pool_windows:
                    continue
                per_animals = [load_per_animal_from_npz(by_win[w], tau_index=None if args.tau_index < 0 else args.tau_index) for w in pool_windows if w in by_win]
                pooled = _concat_per_animal(per_animals)
                if args.reuse_group_boots and _central_bootstrap_groups_boots is not None and _central_ci_from_boots is not None:
                    boots_map = _central_bootstrap_groups_boots(pooled, groups_map, q=q, n_boot=args.n_boot, seed=args.seed)
                    q_arr = boots_map.get('__q__', np.asarray(q, float))
                    qa = {}
                    for gk, idxs in groups_map.items():
                        if gk == '__q__':
                            continue
                        pooled_g = _central_pool_per_animal(pooled, idxs) if _central_pool_per_animal is not None else np.concatenate([pooled[i] for i in idxs if i < len(pooled)])
                        lo, hi = _central_ci_from_boots(boots_map[gk], ci=args.ci)
                        point = np.percentile(pooled_g, q_arr)
                        qa[gk] = {"q": q_arr, "point": point, "lo": lo, "hi": hi, "n": int(pooled_g.size)}
                        for qi, pt, lo_i, hi_i in zip(q_arr, point, lo, hi, strict=False):
                            quantiles_rows.append({
                                "region": roi_name,
                                "roi": roi_name,
                                "window": pool_name,
                                "group": gk,
                                "q": float(qi),
                                "point": float(pt),
                                "lo": float(lo_i),
                                "hi": float(hi_i),
                                "n": int(pooled_g.size),
                            })
                else:
                    qa = bootstrap_quantiles_by_group(
                        pooled,
                        groups_map,
                        q=q,
                        n_boot=args.n_boot,
                        ci=args.ci,
                        seed=args.seed,
                        early_stop=float(args.early_stop or 0.0),
                        _chunk=int(args.chunk),
                        boots_float32=bool(args.boots_float32),
                    )
                    for gk, res in qa.items():
                        for qi, pt, lo, hi in zip(res["q"], res["point"], res["lo"], res["hi"], strict=False):
                            quantiles_rows.append({
                                "region": roi_name,
                                "roi": roi_name,
                                "window": pool_name,
                                "group": gk,
                                "q": float(qi),
                                "point": float(pt),
                                "lo": float(lo),
                                "hi": float(hi),
                                "n": int(res["n"]),
                            })
                # Correlation with NOR per percentile for this ROI/pool
                if args.correlate_nor:
                    # Auto-detect NOR column if needed
                    nor_col = args.nor_col
                    if nor_col is None:
                        cand_cols = [c for c in data.cog_data_filtered.columns if 'nor' in str(c).lower()]
                        if len(cand_cols) == 1:
                            nor_col = cand_cols[0]
                        elif len(cand_cols) > 1:
                            # pick first by default; could be refined
                            nor_col = cand_cols[0]
                        else:
                            nor_col = None
                    if nor_col is not None and nor_col in data.cog_data_filtered.columns:
                        import numpy as _np
                        # Build per-animal percentile vectors for requested q
                        q_list = [float(x) for x in q]
                        # assemble per-animal values into matrix (n_animals x len(q))
                        n_anim = len(pooled)
                        mat = _np.full((n_anim, len(q_list)), _np.nan, float)
                        for i in range(n_anim):
                            vals = _np.asarray(pooled[i], float)
                            if vals.size:
                                mat[i, :] = _np.percentile(vals, q_list)
                        nor_vals = _np.asarray(data.cog_data_filtered[nor_col], float)
                        # Compute correlation per percentile (Pearson + Spearman)
                        def _pearson(a, b):
                            m = _np.isfinite(a) & _np.isfinite(b)
                            if m.sum() < 3:
                                return _np.nan
                            return float(_np.corrcoef(a[m], b[m])[0, 1])
                        def _spearman(a, b):
                            m = _np.isfinite(a) & _np.isfinite(b)
                            if m.sum() < 3:
                                return _np.nan
                            aa = a[m]; bb = b[m]
                            # simple rank (ties handled arbitrarily)
                            ra = _np.argsort(_np.argsort(aa))
                            rb = _np.argsort(_np.argsort(bb))
                            return float(_np.corrcoef(ra, rb)[0, 1])
                        for j, qi in enumerate(q_list):
                            r = _pearson(mat[:, j], nor_vals)
                            rho = _spearman(mat[:, j], nor_vals)
                            n_used = int(_np.count_nonzero(_np.isfinite(mat[:, j]) & _np.isfinite(nor_vals)))
                            corr_rows.append({
                                "region": roi_name,
                                "roi": roi_name,
                                "window": pool_name,
                                "q": float(qi),
                                "pearson_r": r,
                                "spearman_rho": rho,
                                "n": n_used,
                                "nor_col": nor_col,
                            })
                if args.plot:
                    try:
                        from scripts.speed_bootstrap_nb import plot_group_quantiles
                    except ModuleNotFoundError:
                        from speed_bootstrap_nb import plot_group_quantiles  # type: ignore
                    ax = plot_group_quantiles(qa, title=f"{roi_name} | pool={pool_name}")
                    fig_path = fig_root / f"quantiles_{_sanitize(roi_name)}_pool-{pool_name}.{args.plot_format}"
                    if not (args.load_cache and fig_path.exists()):
                        ax.figure.savefig(fig_path, dpi=150, bbox_inches="tight")
                    else:
                        print(f"[cache] Using cached quantiles figure: {fig_path}")
                    plt.close(ax.figure)
                pool_pairs_qd: Dict[Tuple[Tuple[str, str], Tuple[str, str]], dict] = {}
                for (A, B) in groups_to_compare:
                    if A not in groups_map or B not in groups_map:
                        continue
                    if args.reuse_group_boots and _central_bootstrap_groups_boots is not None and _central_ci_from_boots is not None:
                        q_arr = boots_map.get('__q__', np.asarray(q, float))
                        boots_A = boots_map.get(A); boots_B = boots_map.get(B)
                        if boots_A is None or boots_B is None:
                            continue
                        n_used = min(boots_A.shape[0], boots_B.shape[0])
                        diff_boots = boots_A[:n_used, :] - boots_B[:n_used, :]
                        lo, hi = _central_ci_from_boots(diff_boots, ci=args.ci)
                        pooled_A = _central_pool_per_animal(pooled, groups_map[A]) if _central_pool_per_animal is not None else np.concatenate([pooled[i] for i in groups_map[A] if i < len(pooled)])
                        pooled_B = _central_pool_per_animal(pooled, groups_map[B]) if _central_pool_per_animal is not None else np.concatenate([pooled[i] for i in groups_map[B] if i < len(pooled)])
                        point = np.percentile(pooled_A, q_arr) - np.percentile(pooled_B, q_arr)
                        sig = (lo > 0) | (hi < 0)
                        qd = {"q": q_arr, "point": point, "lo": lo, "hi": hi, "sig": sig, "n_x": int(pooled_A.size), "n_y": int(pooled_B.size)}
                        pool_pairs_qd[(A, B)] = qd
                        for qi, pt, lo_i, hi_i, s in zip(q_arr, point, lo, hi, sig, strict=False):
                            diffs_rows.append({
                                "region": roi_name,
                                "roi": roi_name,
                                "window": pool_name,
                                "A": A,
                                "B": B,
                                "q": float(qi),
                                "diff": float(pt),
                                "lo": float(lo_i),
                                "hi": float(hi_i),
                                "significant": bool(s),
                                "n_a": int(pooled_A.size),
                                "n_b": int(pooled_B.size),
                            })
                    else:
                        qd = bootstrap_quantile_diffs_by_keys(
                            pooled,
                            groups_map,
                            A,
                            B,
                            q=q,
                            n_boot=args.n_boot,
                            ci=args.ci,
                            seed=args.seed,
                            early_stop=float(args.early_stop or 0.0),
                            _chunk=int(args.chunk),
                            boots_float32=bool(args.boots_float32),
                        )
                        pool_pairs_qd[(A, B)] = qd
                        for qi, pt, lo, hi, sig in zip(qd["q"], qd["point"], qd["lo"], qd["hi"], qd["sig"], strict=False):
                            diffs_rows.append({
                                "region": roi_name,
                                "roi": roi_name,
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
                if args.plot and not args.no_diffs_win and args.grid and pool_pairs_qd:
                    grid_path = fig_root / f"grid_{_sanitize(roi_name)}_pool-{pool_name}.{args.plot_format}"
                    if not (args.load_cache and grid_path.exists()):
                        _plot_pairs_grid(roi_name, f"pool={pool_name}", pool_pairs_qd, groups_to_compare, grid_path, cols=args.grid_cols)
                    else:
                        print(f"[cache] Using cached grid figure: {grid_path}")

    # Write CSVs
    if quantiles_rows:
        q_cols = ["region", "roi", "window", "group", "q", "point", "lo", "hi", "n"]
        with q_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=q_cols)
            w.writeheader()
            w.writerows(quantiles_rows)
    if diffs_rows:
        d_cols = ["region", "roi", "window", "A", "B", "q", "diff", "lo", "hi", "significant", "n_a", "n_b"]
        with d_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=d_cols)
            w.writeheader()
            w.writerows(diffs_rows)

    print(f"Wrote: {q_path}")
    print(f"Wrote: {d_path}")
    # Write correlations CSV if requested
    if corr_rows:
        corr_path = outdir / "speed_nor_correlations.csv"
        c_cols = ["region", "roi", "window", "q", "pearson_r", "spearman_rho", "n", "nor_col"]
        with corr_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=c_cols)
            w.writeheader()
            w.writerows(corr_rows)
        print(f"Wrote: {corr_path}")

    # Optional: plot diff(A-B) vs window by ROI with percentiles as legends
    if fig_root is not None and args.plot and args.plot_diffs_by_win and diffs_rows:
        # Build mapping: (roi, (A,B)) -> { q -> [(win, diff, sig), ...] }
        by_roi_pair_q: Dict[Tuple[str, Tuple[str, str]], Dict[float, List[Tuple[int, float, bool]]]] = {}
        for r in diffs_rows:
            roi = str(r.get("roi", r.get("region", "")))
            A = r.get("A"); B = r.get("B")
            if A is None or B is None:
                continue
            key = (roi, (str(A), str(B)))
            # Skip pooled entries (short/long/all) — only plot true windows
            win_raw = r.get("window", -1)
            try:
                win = int(win_raw)
            except Exception:
                continue
            qv = float(r.get("q", 0.0))
            by_roi_pair_q.setdefault(key, {}).setdefault(qv, []).append((win, float(r.get("diff", 0.0)), bool(r.get("significant", False))))
        for (roi, (A, B)), qmap in by_roi_pair_q.items():
            fig, ax = plt.subplots(figsize=(8, 4))
            # Stable color mapping per percentile
            qs_sorted = sorted(qmap.keys())
            palette = plt.rcParams.get('axes.prop_cycle', None)
            base_colors = (palette.by_key()['color'] if palette is not None else [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ])
            color_for_q = {float(qi): base_colors[i % len(base_colors)] for i, qi in enumerate(qs_sorted)}

            for qi in qs_sorted:
                triples = sorted(qmap[qi], key=lambda t: t[0])
                wins = [t[0] for t in triples]
                diffs = [t[1] for t in triples]
                sigs = [t[2] for t in triples]
                color = color_for_q[float(qi)]
                # Line per percentile
                ax.plot(wins, diffs, label=f"q{int(qi)}", color=color)
                # Markers: filled if significant, open (no fill) if ns
                for x, y, s in zip(wins, diffs, sigs):
                    if s:
                        ax.plot(x, y, 'o', color=color)
                    else:
                        ax.plot(x, y, 'o', mfc='none', mec=color, color=color)
            ax.axhline(0.0, color='k', linewidth=1, alpha=0.5)
            ax.set_xlabel('Window size')
            ax.set_ylabel('Difference (A - B)')
            ax.set_title(f"{roi} | {A}-{B}")
            ax.legend(loc='best', title='Percentile')
            fig.tight_layout()
            a_str = _sanitize(A); b_str = _sanitize(B)
            out_path = fig_root / f"bywin_{_sanitize(roi)}_{a_str}_vs_{b_str}.{args.plot_format}"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        # Optional grid aggregating all pairs per ROI
        if args.plot_diffs_bywin_grid and by_roi_pair_q:
            # Group back by ROI
            roi_to_pairs: Dict[str, List[Tuple[Tuple[str, str], Dict[float, List[Tuple[int, float, bool]]]]]] = {}
            for (roi, pair), qmap in by_roi_pair_q.items():
                roi_to_pairs.setdefault(roi, []).append((pair, qmap))
            for roi, items in roi_to_pairs.items():
                n = len(items)
                cols = max(1, int(args.bywin_grid_cols))
                rows = int(np.ceil(n / cols))
                fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 3.5*rows), squeeze=False)
                # Build stable colors
                all_qs = sorted({qi for _, qmap in items for qi in qmap.keys()})
                palette = plt.rcParams.get('axes.prop_cycle', None)
                base_colors = (palette.by_key()['color'] if palette is not None else [
                    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
                ])
                color_for_q = {float(qi): base_colors[i % len(base_colors)] for i, qi in enumerate(all_qs)}
                for idx, (pair, qmap) in enumerate(items):
                    r, c = divmod(idx, cols)
                    ax = axes[r, c]
                    for qi in sorted(qmap.keys()):
                        triples = sorted(qmap[qi], key=lambda t: t[0])
                        wins = [t[0] for t in triples]
                        diffs = [t[1] for t in triples]
                        sigs = [t[2] for t in triples]
                        color = color_for_q[float(qi)]
                        ax.plot(wins, diffs, label=f"q{int(qi)}", color=color)
                        for x, y, s in zip(wins, diffs, sigs):
                            if s:
                                ax.plot(x, y, 'o', color=color)
                            else:
                                ax.plot(x, y, 'o', mfc='none', mec=color, color=color)
                    ax.axhline(0.0, color='k', linewidth=1, alpha=0.5)
                    A, B = pair
                    ax.set_title(f"{A}-{B}")
                    if r == rows-1:
                        ax.set_xlabel('Window size')
                    if c == 0:
                        ax.set_ylabel('Diff (A - B)')
                # Hide unused axes
                for k in range(n, rows*cols):
                    r, c = divmod(k, cols)
                    axes[r, c].axis('off')
                # Single legend
                handles, labels = axes[0,0].get_legend_handles_labels()
                if handles:
                    fig.legend(handles, labels, loc='upper right', title='Percentile')
                fig.suptitle(f"{roi} | diff(A-B) vs window")
                fig.tight_layout(rect=[0, 0, 0.98, 0.95])
                grid_out = fig_root / f"bywin_grid_{_sanitize(roi)}.{args.plot_format}"
                fig.savefig(grid_out, dpi=150, bbox_inches='tight')
                plt.close(fig)


if __name__ == "__main__":
    main()
