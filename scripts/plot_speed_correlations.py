#!/usr/bin/env python3
"""
Plot correlations (Pearson/Spearman) vs window from speed_nor_correlations.csv, with pooled summaries.

Figures under paths['f_speed']/<outdir>/:
- cor_bywin_<roi>_<metric>.<fmt>
- cor_pooled_<roi>_<metric>.<fmt>
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


def maybe_tqdm(progress: bool, it, desc: str):
    if not progress:
        return it
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm(it, desc=desc)
    except Exception:
        return it


def plot_bywin_cor(ax, triples_by_q, metric: str, alpha: float, title: str):
    ax.set_title(title)
    all_qs = sorted(triples_by_q.keys())
    prop_cycle = plt.rcParams.get('axes.prop_cycle', None)
    base = prop_cycle.by_key()['color'] if prop_cycle is not None else [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    color_for_q = {float(qi): base[i % len(base)] for i, qi in enumerate(all_qs)}
    for qi in all_qs:
        triples = sorted(triples_by_q[qi], key=lambda t: t[0])
        wins = [t[0] for t in triples]
        corrs = [t[1] for t in triples]
        pvals = [t[2] for t in triples]
        sigs = [np.isfinite(p) and p <= alpha for p in pvals]
        color = color_for_q[float(qi)]
        ax.plot(wins, corrs, label=f"q{int(qi)}", color=color)
        for xw, yw, s in zip(wins, corrs, sigs):
            ax.plot(xw, yw, 'o', color=color, mfc=(color if s else 'none'), mec=color)
    ax.set_xlabel('Window size')
    ax.set_ylabel(f'{metric.capitalize()} correlation')
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(title='Percentile', loc='best')


def plot_pooled_cor(ax, qmap_by_pool, metric: str, alpha: float, title: str):
    ax.set_title(title)
    pools_order = ['short', 'long', 'all']
    pools = [p for p in pools_order if p in qmap_by_pool]
    all_qs = sorted({qi for p in pools for qi in qmap_by_pool[p].keys()})
    prop_cycle = plt.rcParams.get('axes.prop_cycle', None)
    base = prop_cycle.by_key()['color'] if prop_cycle is not None else [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    color_for_q = {float(qi): base[i % len(base)] for i, qi in enumerate(all_qs)}
    x = np.arange(len(pools))
    for qi in all_qs:
        y = []
        sigs = []
        for p in pools:
            if qi in qmap_by_pool[p]:
                corr, pval = qmap_by_pool[p][qi]
            else:
                corr, pval = (np.nan, np.nan)
            y.append(corr)
            sigs.append(np.isfinite(pval) and pval <= alpha)
        color = color_for_q[float(qi)]
        ax.plot(x, y, 'o-', color=color, label=f"q{int(qi)}")
        for xi, yi, s in zip(x, y, sigs):
            if s:
                ax.plot(xi, yi, 'o', color=color, mfc=color)
    ax.set_xticks(x, pools)
    ax.set_xlabel('Pool')
    ax.set_ylabel(f'{metric.capitalize()} correlation')
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(title='Percentile', loc='best')


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot correlations vs window and pooled from correlation CSV.")
    ap.add_argument("--tr", type=int, default=500)
    ap.add_argument("--subset", type=str, default=None)
    ap.add_argument("--outdir", type=str, default=None)
    # append-subset option removed; outdir derives from --outdir or --subset
    ap.add_argument("--plot-format", type=str, default="png", choices=["png", "pdf", "svg"]) 
    ap.add_argument("--metric", type=str, default="spearman", choices=["spearman", "pearson", "both"]) 
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--fdr", type=str, default="bh", choices=["none", "bh"], help="Multiple-testing correction per ROI before marking significance.")
    ap.add_argument("--plot-by-win", action="store_true")
    ap.add_argument("--plot-pooled", action="store_true")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--bywin-group-grid", action="store_true", help="Plot a grid per ROI (subplots by quantile), with one line per group vs window.")
    ap.add_argument("--grid-cols", type=int, default=3)
    args = ap.parse_args()

    csv_root, fig_root = resolve_outdirs(args.tr, args.subset, args.outdir)
    corr_path = csv_root / "speed_nor_correlations.csv"
    if not corr_path.exists():
        raise FileNotFoundError(f"Missing correlation CSV: {corr_path}")
    rows = load_csv_rows(corr_path)

    # Build by-window index: (roi -> q -> [(win, corr, p)]) for integer windows
    # We parse both metrics up-front so we can render both if requested
    metrics = ['spearman', 'pearson'] if args.metric == 'both' else [args.metric]
    by_roi_q_all: Dict[str, Dict[str, Dict[float, List[Tuple[int, float, float]]]]] = {}
    pooled_by_roi_all: Dict[str, Dict[str, Dict[str, Dict[float, Tuple[float, float]]]]] = {}
    # Grouped by-window: metric -> roi -> q -> group -> [(win, corr, p)]
    by_roi_q_group_all: Dict[str, Dict[str, Dict[float, Dict[str, List[Tuple[int, float, float]]]]]] = {}
    for r in rows:
        roi = str(r.get('roi', r.get('region', '')))
        w = r.get('window')
        try:
            qv = float(r.get('q', 0.0))
        except Exception:
            continue
        # Collect for both metrics
        vals_per_metric: Dict[str, Tuple[float, float]] = {}
        for m in metrics:
            key_corr = 'spearman_rho' if m == 'spearman' else 'pearson_r'
            key_p = 'spearman_p' if m == 'spearman' else 'pearson_p'
            try:
                corr = float(r.get(key_corr, np.nan))
                pval = float(r.get(key_p, np.nan))
            except Exception:
                corr, pval = (np.nan, np.nan)
            vals_per_metric[m] = (corr, pval)
        # window is integer => by-window
        try:
            win = int(w)
            for m in metrics:
                corr, pval = vals_per_metric[m]
                by_roi_q_all.setdefault(m, {}).setdefault(roi, {}).setdefault(qv, []).append((win, corr, pval))
                # Grouped by 'group' if present
                grp = str(r.get('group', '__ALL__'))
                by_roi_q_group_all.setdefault(m, {}).setdefault(roi, {}).setdefault(qv, {}).setdefault(grp, []).append((win, corr, pval))
        except Exception:
            # pooled rows short/long/all
            ws = str(w)
            if ws in ('short', 'long', 'all'):
                for m in metrics:
                    corr, pval = vals_per_metric[m]
                    pooled_by_roi_all.setdefault(m, {}).setdefault(roi, {}).setdefault(ws, {})[qv] = (corr, pval)

    def fdr_bh(pvals: List[float]) -> List[float]:
        arr = np.asarray(pvals, float)
        n = arr.size
        idx = np.where(np.isfinite(arr))[0]
        if idx.size == 0:
            return arr.tolist()
        pv = arr[idx]
        order = np.argsort(pv)
        pv_sorted = pv[order]
        m = float(pv_sorted.size)
        adj_sorted = np.empty_like(pv_sorted)
        cmin = 1.0
        for i in range(pv_sorted.size - 1, -1, -1):
            rank = i + 1.0
            adj = pv_sorted[i] * m / rank
            cmin = min(cmin, adj)
            adj_sorted[i] = cmin
        adj = np.full(n, np.nan, float)
        adj_vals = np.empty_like(pv)
        adj_vals[order] = adj_sorted
        adj[idx] = adj_vals
        return adj.tolist()

    # By-window plots per ROI
    if args.plot_by_win and by_roi_q_all:
        for m in metrics:
            by_roi_q = by_roi_q_all.get(m, {})
            it = maybe_tqdm(args.progress, by_roi_q.items(), desc=f"Cor by-window ({m})")
            for roi, qmap in it:
                # FDR-correct p-values across all (q,window) in this ROI
                qmap_use = qmap
                if args.fdr == 'bh':
                    flat_p = []
                    idx_map = []  # (qi, k_in_sorted)
                    # ensure sorted order by window
                    sorted_triples_per_q = {}
                    for qi, triples in qmap.items():
                        triples_sorted = sorted(triples, key=lambda t: t[0])
                        sorted_triples_per_q[qi] = triples_sorted
                        for k, (_, _, p) in enumerate(triples_sorted):
                            flat_p.append(p)
                            idx_map.append((qi, k))
                    p_adj_flat = fdr_bh(flat_p)
                    # rebuild qmap with adjusted p
                    qmap_adj = {}
                    # prepare arrays sized per q
                    per_q_counts = {qi: len(sorted_triples_per_q[qi]) for qi in sorted_triples_per_q}
                    per_q_padj = {qi: [np.nan] * per_q_counts[qi] for qi in sorted_triples_per_q}
                    for (qi, k), padj in zip(idx_map, p_adj_flat):
                        per_q_padj[qi][k] = padj
                    for qi, triples_sorted in sorted_triples_per_q.items():
                        repl = []
                        for k, (w, c, p) in enumerate(triples_sorted):
                            repl.append((w, c, per_q_padj[qi][k]))
                        qmap_adj[qi] = repl
                    qmap_use = qmap_adj
                fig, ax = plt.subplots(figsize=(8, 4))
                plot_bywin_cor(ax, qmap_use, m, args.alpha, title=f"{roi} | {m.capitalize()} vs window")
                out = fig_root / f"cor_bywin_{sanitize_name(roi)}_{m}.{args.plot_format}"
                fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)

    # Pooled plots per ROI
    if args.plot_pooled and pooled_by_roi_all:
        for m in metrics:
            pooled_by_roi = pooled_by_roi_all.get(m, {})
            it = maybe_tqdm(args.progress, pooled_by_roi.items(), desc=f"Cor pooled ({m})")
            for roi, pmap in it:
                pmap_use = pmap
                if args.fdr == 'bh':
                    # Flatten p-values over pools and qs
                    flat_p = []
                    idx_map = []  # (pool, qi)
                    for pool, qmap in pmap.items():
                        for qi, (_, p) in qmap.items():
                            flat_p.append(p)
                            idx_map.append((pool, qi))
                    p_adj_flat = fdr_bh(flat_p)
                    # rebuild pmap with adjusted p
                    pmap_adj = {}
                    for (pool, qi), padj in zip(idx_map, p_adj_flat):
                        pmap_adj.setdefault(pool, {})
                        corr = pmap[pool][qi][0]
                        pmap_adj[pool][qi] = (corr, padj)
                    pmap_use = pmap_adj
                fig, ax = plt.subplots(figsize=(7, 4))
                plot_pooled_cor(ax, pmap_use, m, args.alpha, title=f"{roi} | {m.capitalize()} (pooled)")
                out = fig_root / f"cor_pooled_{sanitize_name(roi)}_{m}.{args.plot_format}"
                fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)

    # Grid per ROI with subplots by quantile, lines per group vs window
    if args.bywin_group_grid and by_roi_q_group_all:
        for m in metrics:
            by_roi_q_group = by_roi_q_group_all.get(m, {})
            it = maybe_tqdm(args.progress, by_roi_q_group.items(), desc=f"Cor bywin grid ({m})")
            for roi, qmap in it:
                qs = sorted(qmap.keys())
                cols = max(1, int(args.grid_cols))
                rows = int(np.ceil(len(qs) / cols))
                fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.8 * rows), squeeze=False)
                # Prepare colors per group
                # Collect all groups
                all_groups = sorted({g for q in qs for g in qmap[q].keys() if g != '__ALL__'})
                if not all_groups:
                    all_groups = sorted({g for q in qs for g in qmap[q].keys()})
                prop_cycle = plt.rcParams.get('axes.prop_cycle', None)
                base = prop_cycle.by_key()['color'] if prop_cycle is not None else [
                    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
                ]
                color_for_group = {str(g): base[i % len(base)] for i, g in enumerate(all_groups)}
                for idx, qi in enumerate(qs):
                    r, c = divmod(idx, cols)
                    ax = axes[r, c]
                    group_map = qmap[qi]
                    # FDR across all groups*windows for this quantile
                    p_adj_map = {}
                    if args.fdr == 'bh':
                        flat_p = []
                        idx_map = []  # (group, k)
                        sorted_triples_per_group = {}
                        for g, triples in group_map.items():
                            triples_sorted = sorted(triples, key=lambda t: t[0])
                            sorted_triples_per_group[g] = triples_sorted
                            for k, (_, _, p) in enumerate(triples_sorted):
                                flat_p.append(p)
                                idx_map.append((g, k))
                        p_adj_flat = fdr_bh(flat_p)
                        per_g_counts = {g: len(sorted_triples_per_group[g]) for g in sorted_triples_per_group}
                        per_g_padj = {g: [np.nan] * per_g_counts[g] for g in sorted_triples_per_group}
                        for (g, k), padj in zip(idx_map, p_adj_flat):
                            per_g_padj[g][k] = padj
                        p_adj_map = per_g_padj
                    # Plot lines per group
                    for g, triples in group_map.items():
                        triples_sorted = sorted(triples, key=lambda t: t[0])
                        wins = [t[0] for t in triples_sorted]
                        corrs = [t[1] for t in triples_sorted]
                        pvals = [t[2] for t in triples_sorted]
                        if p_adj_map:
                            p_use = p_adj_map.get(g, pvals)
                        else:
                            p_use = pvals
                        color = color_for_group.get(str(g), '#1f77b4')
                        ax.plot(wins, corrs, label=str(g), color=color)
                        for xw, yw, pv in zip(wins, corrs, p_use):
                            ax.plot(xw, yw, 'o', color=color, mfc=(color if (np.isfinite(pv) and pv <= args.alpha) else 'none'), mec=color)
                    ax.set_title(f"q{int(qi)}")
                    ax.set_xlabel('Window size')
                    ax.set_ylabel(f'{m.capitalize()}')
                # Hide extras
                for k in range(len(qs), rows * cols):
                    r, c = divmod(k, cols)
                    axes[r, c].axis('off')
                # Combined legend
                handles, labels = axes[0, 0].get_legend_handles_labels()
                if handles:
                    fig.legend(handles, labels, loc='upper right', title='Group')
                fig.suptitle(f"{roi} | {m.capitalize()} vs window (by group)")
                fig.tight_layout(rect=[0, 0, 0.98, 0.95])
                out = fig_root / f"cor_bywin_group_{sanitize_name(roi)}_{m}.{args.plot_format}"
                fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
