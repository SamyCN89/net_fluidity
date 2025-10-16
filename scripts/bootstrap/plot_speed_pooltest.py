#!/usr/bin/env python3
"""
Plot pool-test results (target vs pooled supergroup) from speed_bootstrap_pooltest*.csv.

Figures under paths['f_speed']/<outdir>/:
- pooltest_bywin_<roi>_<group_sanitized>.<fmt>
- pooltest_pooled_<roi>_<group_sanitized>.<fmt>

Usage examples
  # Plot by-window and pooled (all quantiles) for a subset
  python scripts/plot_speed_pooltest.py --tr 500 --subset dmn_within --bywin --pooled --show-all-q --progress

  # Only q50 (median)
  python scripts/plot_speed_pooltest.py --tr 500 --subset dmn_within --bywin --q 50
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
    data.get_ts_preprocessed(); data.get_cogdata_preprocessed(); data.get_temporal_parameters()
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


def load_csv_rows(paths: List[Path]) -> List[dict]:
    rows: List[dict] = []
    for p in paths:
        if p.exists():
            with p.open("r", newline="") as f:
                rows.extend(list(csv.DictReader(f)))
    return rows


def sanitize(s: str) -> str:
    return str(s).replace('/', '-').replace(' ', '_').replace(',', '-').replace('|', '-').replace('(', '').replace(')', '')


def maybe_tqdm(progress: bool, it, desc: str):
    if not progress:
        return it
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm(it, desc=desc)
    except Exception:
        return it


def plot_bywin(ax, triples_by_q: Dict[float, List[Tuple[int, float, float, float, float]]], title: str):
    ax.set_title(title)
    qs = sorted(triples_by_q.keys())
    prop_cycle = plt.rcParams.get('axes.prop_cycle', None)
    base = prop_cycle.by_key()['color'] if prop_cycle is not None else [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    color_for_q = {float(qi): base[i % len(base)] for i, qi in enumerate(qs)}
    for qi in qs:
        items = sorted(triples_by_q[qi], key=lambda t: t[0])
        wins = [t[0] for t in items]
        tgt = [t[1] for t in items]
        lo = [t[2] for t in items]
        hi = [t[3] for t in items]
        inside = [t[4] for t in items]
        color = color_for_q[float(qi)]
        ax.plot(wins, tgt, label=f"q{int(qi)}", color=color)
        for xw, y, l, h, ins in zip(wins, tgt, lo, hi, inside):
            ax.plot([xw, xw], [l, h], color=color, alpha=0.5)
            ax.plot(xw, y, 'o', color=color, mfc=(color if not ins else 'none'), mec=color)
    ax.set_xlabel('Window size'); ax.set_ylabel('Target percentile')
    if qs:
        ax.legend(title='Percentile', loc='best')


def plot_pooled(ax, triples_by_q: Dict[float, List[Tuple[str, float, float, float, float]]], title: str):
    ax.set_title(title)
    qs = sorted(triples_by_q.keys())
    pools_order = ['short', 'long', 'all']
    x = np.arange(len(pools_order))
    prop_cycle = plt.rcParams.get('axes.prop_cycle', None)
    base = prop_cycle.by_key()['color'] if prop_cycle is not None else [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    color_for_q = {float(qi): base[i % len(base)] for i, qi in enumerate(qs)}
    for qi in qs:
        items = triples_by_q[qi]
        by_pool = {p: (tgt, lo, hi, ins) for (p, tgt, lo, hi, ins) in items}
        y = []; los = []; his = []; ins_flags = []
        for p in pools_order:
            tgt, lo, hi, ins = by_pool.get(p, (np.nan, np.nan, np.nan, True))
            y.append(tgt); los.append(lo); his.append(hi); ins_flags.append(ins)
        color = color_for_q[float(qi)]
        ax.plot(x, y, 'o-', color=color, label=f"q{int(qi)}")
        for xi, yi, l, h, ins in zip(x, y, los, his, ins_flags):
            ax.plot([xi, xi], [l, h], color=color, alpha=0.5)
            if not ins:
                ax.plot(xi, yi, 'o', color=color, mfc=color)
    ax.set_xticks(x, pools_order); ax.set_xlabel('Pool'); ax.set_ylabel('Target percentile')
    if qs:
        ax.legend(title='Percentile', loc='best')


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot pool-test results (target vs pooled CI)")
    ap.add_argument("--tr", type=int, default=500)
    ap.add_argument("--subset", type=str, default=None)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--plot-format", type=str, default="png", choices=["png", "pdf", "svg"])
    ap.add_argument("--bywin", action="store_true", help="Plot by-window figures (x=window)")
    ap.add_argument("--pooled", action="store_true", help="Plot pooled short/long/all figures")
    ap.add_argument("--show-all-q", action="store_true", help="Plot all quantiles")
    ap.add_argument("--q", type=str, default=None, help="Comma-separated q list to plot (e.g., '50' or '5,50,95')")
    ap.add_argument("--progress", action="store_true")
    # Title hints can be auto-read from CSV; flags below are kept as fallback
    ap.add_argument("--group-cols", type=str, default="genotype,treatment", help="Columns used to build group keys (order matters)")
    ap.add_argument("--bootstrap-pool-cols", type=str, default=None, help="Subset of --group-cols used for pooling (fallback if CSV lacks metadata)")
    ap.add_argument("--pool-exclude-self", action="store_true", help="If set in compute, show note in title (fallback if CSV lacks metadata)")
    args = ap.parse_args()

    csv_root, fig_root = resolve_outdirs(args.tr, args.subset, args.outdir)
    # Prefer suffixed files if present; else base file
    import re
    cands: list[Path] = []
    # Gather all nboot-suffixed files
    for p in sorted(csv_root.glob("speed_bootstrap_pooltest_explicit_nboot-*.csv")):
        cands.append(p)
    for p in sorted(csv_root.glob("speed_bootstrap_pooltest_nboot-*.csv")):
        cands.append(p)
    # Fallback to base file
    # Try explicit first, then standard
    cands.append(csv_root / "speed_bootstrap_pooltest_explicit.csv")
    cands.append(csv_root / "speed_bootstrap_pooltest.csv")
    rows = load_csv_rows(cands)
    if not rows:
        raise FileNotFoundError("No pool-test CSV found under outdir")

    # Filter quantiles
    qs_filter = None
    if not args.show_all_q and args.q:
        try:
            qs_filter = {float(x) for x in args.q.split(',') if x.strip()}
        except Exception:
            qs_filter = None

    # Group by ROI and group (prefer human-friendly group_label if present)
    # By-window: ROI -> group -> q -> list[(win, tgt, lo, hi, inside)]
    bywin: Dict[str, Dict[str, Dict[float, List[Tuple[int, float, float, float, bool]]]]] = {}
    pooled: Dict[str, Dict[str, Dict[float, List[Tuple[str, float, float, float, bool]]]]] = {}
    # Note per ROI/group for title
    notes: Dict[str, Dict[str, str]] = {}
    for r in rows:
        roi = str(r.get('roi', r.get('region', '')))
        grp = str(r.get('group_label') or r.get('group'))
        try:
            qv = float(r.get('q', 0.0))
        except Exception:
            continue
        if qs_filter and qv not in qs_filter:
            continue
        try:
            tgt = float(r.get('target_point', np.nan))
            lo = float(r.get('lo', np.nan))
            hi = float(r.get('hi', np.nan))
            ins = str(r.get('inside', 'True')).lower() in ('1','true','yes')
        except Exception:
            continue
        # Build title note from CSV metadata if present
        pool_by = (r.get('pool_by') or '').strip()
        pool_match = (r.get('pool_match') or '').strip()
        pool_excl = str(r.get('pool_exclude_self', 'False')).lower() in ('1','true','yes')
        if pool_by or pool_match:
            base = pool_match if pool_match else pool_by
            note = f"\nvs pool by {base}{'; exclude self' if pool_excl else ''}"
            notes.setdefault(roi, {})[grp] = note

        w = r.get('window')
        # integer windows → bywin
        try:
            win = int(w)
            bywin.setdefault(roi, {}).setdefault(grp, {}).setdefault(qv, []).append((win, tgt, lo, hi, ins))
        except Exception:
            ws = str(w)
            if ws in ('short','long','all'):
                pooled.setdefault(roi, {}).setdefault(grp, {}).setdefault(qv, []).append((ws, tgt, lo, hi, ins))

    # Helper for title: describe pooled supergroup
    def parse_group_vals(grp: str) -> list[str]:
        s = str(grp).strip()
        if s.startswith('(') and s.endswith(')'):
            s = s[1:-1]
        # remove quotes and spaces around tokens
        parts = [t.strip().strip("'\"") for t in s.split(',')]
        return [p for p in parts if p != '']

    group_cols = [c.strip() for c in str(args.group_cols).split(',') if c.strip()]
    pool_cols = [c.strip() for c in str(args.bootstrap_pool_cols).split(',') if c and c.strip()] if args.bootstrap_pool_cols else []
    pos = {c: i for i, c in enumerate(group_cols)}

    def pool_note(grp: str) -> str:
        if not pool_cols:
            return ""
        vals = parse_group_vals(grp)
        kv = []
        for c in pool_cols:
            i = pos.get(c, None)
            if i is None or i >= len(vals):
                continue
            kv.append(f"{c}={vals[i]}")
        base = ", ".join(kv) if kv else "pooled supergroup"
        excl = "; exclude self" if args.pool_exclude_self else ""
        return f"\nvs pool by {base}{excl}"

    # Plot by-window per ROI/group
    if args.bywin and bywin:
        it = maybe_tqdm(args.progress, bywin.items(), desc='pooltest bywin')
        for roi, gmap in it:
            for grp, qmap in gmap.items():
                fig, ax = plt.subplots(figsize=(8,4))
                note = notes.get(roi, {}).get(grp)
                if note is None:
                    note = pool_note(grp)
                plot_bywin(ax, qmap, title=f"{roi} | {grp}\npool-test (by window){note or ''}")
                out = fig_root / f"pooltest_bywin_{sanitize(roi)}_{sanitize(grp)}.{args.plot_format}"
                fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    elif args.bywin and not bywin:
        print("[warn] --bywin requested but no by-window rows found in CSV (did compute run?)")

    # Plot pooled per ROI/group
    if args.pooled and pooled:
        it = maybe_tqdm(args.progress, pooled.items(), desc='pooltest pooled')
        for roi, gmap in it:
            for grp, qmap in gmap.items():
                fig, ax = plt.subplots(figsize=(7,4))
                note = notes.get(roi, {}).get(grp)
                if note is None:
                    note = pool_note(grp)
                plot_pooled(ax, qmap, title=f"{roi} | {grp}\npool-test (pooled){note or ''}")
                out = fig_root / f"pooltest_pooled_{sanitize(roi)}_{sanitize(grp)}.{args.plot_format}"
                fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    elif args.pooled and not pooled:
        print("[warn] --pooled requested but no pooled rows found in CSV. Compute pooled results with --pool-threshold (and optionally --pool-all).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
