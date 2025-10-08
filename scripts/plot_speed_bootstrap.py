#!/usr/bin/env python3
"""
Plot dFC speed bootstrap figures from existing CSVs (plot-only, self-contained).

Reads speed_bootstrap_diffs.csv (and optionally quantiles) and writes figures:
- bywin_<roi>_(G1,T1)_vs_(G2,T2).<fmt>
- bywin_grid_<roi>.<fmt>
under paths['f_speed']/<outdir>/ (falls back to paths['speed']/<outdir>/).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_context(tr: int | None = None):
    """Return a dataset context with paths and metadata loaded (robust import)."""
    DFC = None
    try:
        from net_fluidity_julien.context import DFCAnalysis as DFC  # type: ignore
    except ModuleNotFoundError:
        try:
            import sys
            here = Path(__file__).resolve()
            repo_root = here.parents[1]
            src_path = repo_root / "src"
            if src_path.exists() and str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            from net_fluidity_julien.context import DFCAnalysis as DFC  # type: ignore
        except Exception:
            DFC = None
    if DFC is None:
        try:
            from class_dataanalysis_julien import DFCAnalysis as DFC  # type: ignore
        except ModuleNotFoundError:
            import sys
            here = Path(__file__).resolve()
            repo_root = here.parents[1]
            jd_path = repo_root / "julien_data"
            if jd_path.exists() and str(jd_path) not in sys.path:
                sys.path.insert(0, str(jd_path))
            from class_dataanalysis_julien import DFCAnalysis as DFC  # type: ignore
    data = DFC()
    if tr is None:
        data.get_metadata()
    else:
        preproc = Path(data.paths["preprocessed"])  # type: ignore[index]
        cands = sorted(preproc.glob(f"metadata_animals_*_tr_{int(tr)}.pkl"))
        if not cands:
            raise FileNotFoundError(f"No metadata file for tr={tr} under {preproc}")
        data.get_metadata(meta_filename=cands[0].name)
    data.get_ts_preprocessed()
    data.get_cogdata_preprocessed()
    data.get_temporal_parameters()
    return data


def build_outdir_name(outdir: str | None, subset: str | None) -> str:
    return outdir if outdir else (subset if subset else "bootstrap")


def resolve_outdirs(tr: int, subset: str | None, outdir: str | None):
    data = get_context(tr=tr)
    speed_root = Path(data.paths["speed"])  # type: ignore[index]
    outdir_name = build_outdir_name(outdir, subset)
    csv_root = speed_root / outdir_name
    fig_base = Path(data.paths.get("f_speed", speed_root))  # type: ignore[attr-defined]
    fig_root = fig_base / outdir_name
    fig_root.mkdir(parents=True, exist_ok=True)
    return csv_root, fig_root


def load_csv_rows(path: Path) -> List[dict]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def sanitize_name(s: str) -> str:
    return (
        str(s).replace("/", "-").replace(" ", "_").replace(",", "-")
        .replace("|", "-").replace("(", "").replace(")", "")
    )


def build_bywin_diffs_index(
    diffs_rows: List[dict],
) -> Dict[Tuple[str, Tuple[str, str]], Dict[float, List[Tuple[int, float, bool]]]]:
    by_roi_pair_q: Dict[Tuple[str, Tuple[str, str]], Dict[float, List[Tuple[int, float, bool]]]] = {}
    for r in diffs_rows:
        roi = str(r.get("roi", r.get("region", "")))
        A = r.get("A"); B = r.get("B")
        if A is None or B is None:
            continue
        # Only per-window rows: window must be an integer
        win_raw = r.get("window", -1)
        try:
            win = int(win_raw)
        except Exception:
            continue
        try:
            qv = float(r.get("q", 0.0))
            diffv = float(r.get("diff", 0.0))
        except Exception:
            continue
        sig = str(r.get("significant", "False")).lower() in ("1", "true", "yes")
        key = (roi, (str(A), str(B)))
        by_roi_pair_q.setdefault(key, {}).setdefault(qv, []).append((win, diffv, sig))
    return by_roi_pair_q


def palette_for_quantiles(qs: List[float]) -> Dict[float, str]:
    prop_cycle = plt.rcParams.get('axes.prop_cycle', None)
    base = prop_cycle.by_key()['color'] if prop_cycle is not None else [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    return {float(qi): base[i % len(base)] for i, qi in enumerate(sorted(qs))}


def plot_bywin(ax, triples_by_q: Dict[float, List[Tuple[int, float, bool]]], title: str | None = None):
    if title:
        ax.set_title(title)
    all_qs = sorted(triples_by_q.keys())
    color_for_q = palette_for_quantiles(all_qs)
    for qi in all_qs:
        triples = sorted(triples_by_q[qi], key=lambda t: t[0])
        wins = [t[0] for t in triples]
        diffs = [t[1] for t in triples]
        sigs = [t[2] for t in triples]
        color = color_for_q[float(qi)]
        ax.plot(wins, diffs, label=f"q{int(qi)}", color=color)
        for xw, yw, s in zip(wins, diffs, sigs):
            ax.plot(xw, yw, 'o', color=color, mfc=(color if s else 'none'), mec=color)
    ax.axhline(0.0, color='k', linewidth=1, alpha=0.5)
    ax.set_xlabel('Window size')
    ax.set_ylabel('Difference (A - B)')
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(title='Percentile', loc='best')


def save_fig(fig, path: Path, cache: bool):
    try:
        if cache and path.exists():
            return
        fig.savefig(path, dpi=150, bbox_inches='tight')
    finally:
        # Always close to avoid accumulating open figures
        try:
            plt.close(fig)
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot dFC speed bootstrap figures from existing CSVs (plot-only).")
    ap.add_argument("--tr", type=int, default=500)
    ap.add_argument("--subset", type=str, default=None)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--plot-format", type=str, default="png", choices=["png", "pdf", "svg"])
    ap.add_argument("--plot-diffs-by-win", action="store_true")
    ap.add_argument("--plot-diffs-bywin-grid", action="store_true")
    ap.add_argument("--bywin-grid-cols", type=int, default=2)
    ap.add_argument("--plot-pooled-diffs", action="store_true", help="Plot pooled (short/long/all) pairwise diffs per ROI and pair.")
    ap.add_argument("--plot-pooled-quantiles", action="store_true", help="Plot pooled (short/long/all) per-group quantiles per ROI.")
    ap.add_argument("--load-cache", action="store_true")
    ap.add_argument("--progress", action="store_true", help="Show progress bars if tqdm is available.")
    args = ap.parse_args()

    # Resolve inputs/outputs
    csv_root, fig_root = resolve_outdirs(args.tr, args.subset, args.outdir)
    diffs_path = csv_root / "speed_bootstrap_diffs.csv"
    if not diffs_path.exists():
        raise FileNotFoundError(f"Missing required diffs CSV: {diffs_path}")

    diffs_rows = load_csv_rows(diffs_path)
    by_roi_pair_q = build_bywin_diffs_index(diffs_rows)

    def maybe_tqdm(progress: bool, it, desc: str):
        if not progress:
            return it
        try:
            from tqdm import tqdm  # type: ignore
            return tqdm(it, desc=desc)
        except Exception:
            return it

    # By-window per-pair figures
    if args.plot_diffs_by_win and by_roi_pair_q:
        it = maybe_tqdm(args.progress, by_roi_pair_q.items(), desc="By-window pairs")
        for (roi, (A, B)), triples_by_q in it:
            fig, ax = plt.subplots(figsize=(8, 4))
            plot_bywin(ax, triples_by_q, title=f"{roi} | diff(A-B) vs window\n{A} vs {B}")
            a_str = sanitize_name(A); b_str = sanitize_name(B)
            out_path = fig_root / f"bywin_{sanitize_name(roi)}_{a_str}_vs_{b_str}.{args.plot_format}"
            save_fig(fig, out_path, cache=args.load_cache)

    # By-window grids per ROI
    if args.plot_diffs_bywin_grid and by_roi_pair_q:
        # Group by ROI
        roi_to_pairs: Dict[str, List[Tuple[Tuple[str, str], Dict[float, List[Tuple[int, float, bool]]]]]] = {}
        for (roi, pair), qmap in by_roi_pair_q.items():
            roi_to_pairs.setdefault(roi, []).append((pair, qmap))
        it = maybe_tqdm(args.progress, roi_to_pairs.items(), desc="By-window grids")
        for roi, items in it:
            n = len(items)
            cols = max(1, int(args.bywin_grid_cols))
            rows = int(np.ceil(n / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.5 * rows), squeeze=False)
            for idx, (pair, qmap) in enumerate(items):
                r, c = divmod(idx, cols)
                ax = axes[r, c]
                A, B = pair
                plot_bywin(ax, qmap, title=f"{A} vs {B}")
            for k in range(n, rows * cols):
                r, c = divmod(k, cols)
                axes[r, c].axis('off')
            fig.suptitle(f"{roi} | diff(A-B) vs window")
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            grid_out = fig_root / f"bywin_grid_{sanitize_name(roi)}.{args.plot_format}"
            save_fig(fig, grid_out, cache=args.load_cache)

    # Pooled diffs (short/long/all)
    def _build_pooled_diffs_index(rows: List[dict]):
        out: Dict[Tuple[str, str, Tuple[str, str]], Dict[float, Tuple[float, float, float, bool, float]]] = {}
        for r in rows:
            roi = str(r.get("roi", r.get("region", "")))
            A = r.get("A"); B = r.get("B")
            if A is None or B is None:
                continue
            w = str(r.get("window", ""))
            if w not in ("short", "long", "all"):
                continue
            try:
                qv = float(r.get("q", 0.0))
                pt = float(r.get("diff", 0.0))
                lo = float(r.get("lo", np.nan))
                hi = float(r.get("hi", np.nan))
            except Exception:
                continue
            sig = str(r.get("significant", "False")).lower() in ("1", "true", "yes")
            try:
                pval = float(r.get("p", np.nan))
            except Exception:
                pval = float("nan")
            key = (roi, w, (str(A), str(B)))
            out.setdefault(key, {})[qv] = (pt, lo, hi, sig, pval)
        return out

    def _plot_pooled_diffs(ax, qmap: Dict[float, Tuple[float, float, float, bool, float]], title: str | None = None):
        if title:
            ax.set_title(title)
        qs = sorted(qmap.keys())
        pts = [qmap[q][0] for q in qs]
        los = [qmap[q][1] for q in qs]
        his = [qmap[q][2] for q in qs]
        sigs = [qmap[q][3] for q in qs]
        pvals = [qmap[q][4] for q in qs]
        errs = [
            [max(0.0, pts[i] - los[i]) if np.isfinite(los[i]) else 0.0,
             max(0.0, his[i] - pts[i]) if np.isfinite(his[i]) else 0.0]
            for i in range(len(qs))
        ]
        # errorbar expects 2xN
        err_arr = np.array(errs).T if len(errs) else None
        main_color = '#1f77b4'
        ax.errorbar(qs, pts, yerr=err_arr, fmt='o-', color=main_color, ecolor=main_color, capsize=3, label=None)
        # overlay filled markers for significant
        for x, y, s in zip(qs, pts, sigs):
            if s:
                ax.plot(x, y, 'o', color=main_color, mfc=main_color)
        ax.axhline(0.0, color='k', linewidth=1, alpha=0.5)
        ax.set_xlabel('Percentile (q)')
        ax.set_ylabel('Difference (A - B)')
        # Legend with per-quantile p-values (filled marker indicates significant)
        try:
            from matplotlib.lines import Line2D
            legend_handles = []
            for qi, s, pv in zip(qs, sigs, pvals):
                label = f"q{int(qi)} (p={pv:.3g})" if np.isfinite(pv) else f"q{int(qi)} (p=NA)"
                handle = Line2D([0], [0], marker='o', linestyle='None', markerfacecolor=(main_color if s else 'none'), markeredgecolor=main_color, color=main_color, label=label)
                legend_handles.append(handle)
            if legend_handles:
                ax.legend(handles=legend_handles, title='Percentile (p-values)', loc='best')
        except Exception:
            pass

    if args.plot_pooled_diffs:
        pooled = _build_pooled_diffs_index(diffs_rows)
        it = maybe_tqdm(args.progress, pooled.items(), desc="Pooled diffs")
        for (roi, pool, (A, B)), qmap in it:
            fig, ax = plt.subplots(figsize=(6, 4))
            _plot_pooled_diffs(ax, qmap, title=f"{roi} | pool={pool}\n{A} vs {B}")
            a_str = sanitize_name(A); b_str = sanitize_name(B)
            out_path = fig_root / f"pooled_diffs_{sanitize_name(roi)}_pool-{pool}_{a_str}_vs_{b_str}.{args.plot_format}"
            save_fig(fig, out_path, cache=args.load_cache)

    # Pooled quantiles (short/long/all)
    def _build_pooled_quants_index(path: Path):
        if not path.exists():
            return {}
        rows = load_csv_rows(path)
        out: Dict[Tuple[str, str, str], Dict[float, Tuple[float, float, float]]] = {}
        for r in rows:
            roi = str(r.get("roi", r.get("region", "")))
            group = str(r.get("group", ""))
            w = str(r.get("window", ""))
            if w not in ("short", "long", "all"):
                continue
            try:
                qv = float(r.get("q", 0.0))
                pt = float(r.get("point", 0.0))
                lo = float(r.get("lo", np.nan))
                hi = float(r.get("hi", np.nan))
            except Exception:
                continue
            out.setdefault((roi, w, group), {})[qv] = (pt, lo, hi)
        return out

    def _plot_pooled_quants(ax, qmap_by_group: Dict[str, Dict[float, Tuple[float, float, float]]], title: str | None = None):
        if title:
            ax.set_title(title)
        groups = sorted(qmap_by_group.keys())
        # Stable colors per group
        prop_cycle = plt.rcParams.get('axes.prop_cycle', None)
        base = prop_cycle.by_key()['color'] if prop_cycle is not None else [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        for gi, g in enumerate(groups):
            qmap = qmap_by_group[g]
            qs = sorted(qmap.keys())
            pts = [qmap[q][0] for q in qs]
            los = [qmap[q][1] for q in qs]
            his = [qmap[q][2] for q in qs]
            err = np.array([
                [max(0.0, pts[i] - los[i]) if np.isfinite(los[i]) else 0.0,
                 max(0.0, his[i] - pts[i]) if np.isfinite(his[i]) else 0.0]
                for i in range(len(qs))
            ]).T if qs else None
            color = base[gi % len(base)]
            ax.errorbar(qs, pts, yerr=err, fmt='o-', label=str(g), color=color, ecolor=color, capsize=3)
        ax.set_xlabel('Percentile (q)')
        ax.set_ylabel('Value')
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(title='Group', loc='best')

    if args.plot_pooled_quantiles:
        pooled_q = _build_pooled_quants_index(csv_root / "speed_bootstrap_quantiles.csv")
        # Group into per-ROI, per-pool collections
        roi_pool_to_group: Dict[Tuple[str, str], Dict[str, Dict[float, Tuple[float, float, float]]]] = {}
        for (roi, pool, group), qmap in pooled_q.items():
            roi_pool_to_group.setdefault((roi, pool), {})[group] = qmap
        it = maybe_tqdm(args.progress, roi_pool_to_group.items(), desc="Pooled quantiles")
        for (roi, pool), group_maps in it:
            fig, ax = plt.subplots(figsize=(7, 4))
            _plot_pooled_quants(ax, group_maps, title=f"{roi} | pool={pool}")
            out_path = fig_root / f"pooled_quantiles_{sanitize_name(roi)}_pool-{pool}.{args.plot_format}"
            save_fig(fig, out_path, cache=args.load_cache)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
