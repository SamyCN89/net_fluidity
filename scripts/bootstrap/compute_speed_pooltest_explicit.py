#!/usr/bin/env python3
"""
Compute explicit pool-test bootstrap for selected target groups against a custom pool.

This script lets you test a group against an explicit pooled supergroup that you
define as a list of groups. It mirrors the per-window and pooled behaviors from
compute_speed_bootstrap.py but bypasses --bootstrap-pool-cols.

Example (your requested test):
  python scripts/compute_speed_pooltest_explicit.py \
    --tr 500 --subset dmn_within --tau-index 0 \
    --group-cols genotype,treatment \
    --targets "(Dp1Yey,LCTB92);(WT,VEH)" \
    --pool "(WT,VEH);(Dp1Yey,LCTB92)" \
    --n-boot 20 --jobs 8 --n-animals 48 --progress \
    --pool-threshold median --pool-all

Outputs a CSV under paths['speed']/<outdir or subset>/ named:
  speed_bootstrap_pooltest_explicit(.csv and _nboot-<N>.csv)
with the same schema as the standard pool-test, plus metadata fields.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
from dataclasses import dataclass, field
from collections.abc import Iterable

import numpy as np
import pandas as pd

from shared_code.fun_bootstrap import (
    bootstrap_group_from_pool,
    pool_per_animal,
)


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


WIN_RE = re.compile(r"speed_win(\d+)_.*\.npz$")


def _list_window_files(region_dir: Path) -> list[tuple[int, Path]]:
    files: list[tuple[int, Path]] = []
    for p in sorted(region_dir.glob("speed_win*_*.npz")):
        m = WIN_RE.search(p.name)
        if m:
            files.append((int(m.group(1)), p))
    return files


def _find_region_folders(speed_root: Path) -> list[Path]:
    prefixed = sorted(
        [p for p in speed_root.iterdir() if p.is_dir() and p.name.startswith("regions-")]
    )
    if prefixed:
        return prefixed
    generic = []
    for p in sorted([x for x in speed_root.iterdir() if x.is_dir()]):
        if list(p.glob("speed_win*_*.npz")):
            generic.append(p)
    if generic:
        return generic
    all_dir = speed_root / "all"
    return [all_dir] if all_dir.exists() else [speed_root]


def _pool_windows_indices(windows: list[int], threshold: str | int | None) -> dict[str, list[int]]:
    if threshold is None:
        return {}
    vals = sorted(windows)
    if isinstance(threshold, str) and threshold.lower() == "median":
        cut = int(np.median(vals))
    else:
        cut = int(threshold)
    return {"short": [w for w in vals if w <= cut], "long": [w for w in vals if w > cut]}


def _concat_per_animal(per_animals: list[list[np.ndarray]]) -> list[np.ndarray]:
    if not per_animals:
        return []
    n = max(len(x) for x in per_animals)
    out: list[np.ndarray] = []
    for i in range(n):
        parts = []
        for lst in per_animals:
            if i < len(lst) and lst[i].size > 0:
                parts.append(lst[i])
        out.append(np.concatenate(parts) if parts else np.array([], float))
    return out


def maybe_tqdm(progress: bool, it, desc: str):
    if progress:
        try:
            from tqdm import tqdm
            return tqdm(it, desc=desc)
        except Exception:
            return it
    return it


def load_per_animal_from_npz(npz_path: Path, tau_index: int | None = None, n_animals: int | None = None) -> list[np.ndarray]:
    z = np.load(npz_path, allow_pickle=True)
    speeds = z["speeds"]
    out: list[np.ndarray] = []
    count = len(speeds) if (n_animals is None or n_animals <= 0) else min(int(n_animals), len(speeds))
    for a in range(count):
        arr = np.asarray(speeds[a], float)
        if arr.ndim != 2:
            out.append(np.array([], float)); continue
        if tau_index is None or int(tau_index) < 0:
            vals = arr[~np.isnan(arr)]
        else:
            if tau_index < 0 or tau_index >= arr.shape[0]:
                vals = np.array([], float)
            else:
                vals = arr[tau_index][~np.isnan(arr[tau_index])]
        out.append(vals)
    return out


def build_groups_from_columns(cog_df: pd.DataFrame, columns: list[str]) -> dict:
    cols = [c.strip() for c in columns if c.strip()]
    tmp = cog_df.reset_index(drop=True)
    grp = tmp.groupby(cols).groups
    out: dict = {}
    for k, idx in grp.items():
        out[k if isinstance(k, tuple) else (k,)] = sorted(int(i) for i in idx)
    return out


def parse_tuple_list(arg: str) -> list[tuple]:
    """Parse "(A,B);(C,D)" into [(A,B),(C,D)]."""
    parts = [s.strip() for s in arg.split(";") if s.strip()]
    out = []
    for p in parts:
        t = p.strip()
        if t.startswith("(") and t.endswith(")"):
            t = t[1:-1]
        fields = [x.strip().strip("'\"") for x in t.split(",")]
        out.append(tuple(fields))
    return out


def format_group_label(cols: list[str], key: tuple) -> str:
    return ",".join(f"{c}={key[i]}" for i, c in enumerate(cols) if i < len(key))


@dataclass
class Config:
    tr: int = 500
    subset: str | None = None
    outdir: str | None = None
    tau_index: int = 0
    group_cols: str = "genotype,treatment"
    targets: str = "(Dp1Yey,LCTB92);(WT,VEH)"
    pool: str = "(WT,VEH);(Dp1Yey,LCTB92)"
    n_boot: int = 2000
    seed: int = 0
    ci: float = 95.0
    chunk: int = 128
    jobs: int = 1
    progress: bool = False
    n_animals: int = 48
    boots_float32: bool = True
    values_float32: bool = False
    index_int32: bool = True
    pool_threshold: str | None = None
    pool_all: bool = False
    q: str = "1,5,50,95,99"
    exclude_self: bool = False

    # derived
    q_list: list[float] = field(default_factory=list)


def write_pooltest_csv(rows: list[dict[str, object]], outdir: Path, n_boot: int):
    if not rows:
        return
    p_cols = [
        "region", "roi", "window", "group", "group_label", "group_cols",
        "pool_by", "pool_match", "pool_exclude_self",
        "q", "target_point", "pool_point", "lo", "hi", "inside", "p", "n_target", "n_pool",
    ]
    ptest_path = outdir / "speed_bootstrap_pooltest_explicit.csv"
    with ptest_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=p_cols)
        w.writeheader(); w.writerows(rows)
    print(f"Wrote: {ptest_path}")
    try:
        ptest_sfx = outdir / f"speed_bootstrap_pooltest_explicit_nboot-{int(n_boot)}.csv"
        with ptest_sfx.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=p_cols)
            w.writeheader(); w.writerows(rows)
        print(f"Wrote: {ptest_sfx}")
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Explicit pool-test bootstrap for selected targets vs a custom pool")
    ap.add_argument("--tr", type=int, default=500)
    ap.add_argument("--subset", type=str, default=None)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--tau-index", type=int, default=0)
    ap.add_argument("--group-cols", type=str, default="genotype,treatment")
    ap.add_argument("--targets", type=str, default="(Dp1Yey,LCTB92);(WT,VEH)")
    ap.add_argument("--pool", type=str, default="(WT,VEH);(Dp1Yey,LCTB92)")
    ap.add_argument("--exclude-self", action="store_true")
    ap.add_argument("--q", type=str, default="1,5,50,95,99")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ci", type=float, default=95.0)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--n-animals", type=int, default=48)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--pool-threshold", type=str, default=None)
    ap.add_argument("--pool-all", action="store_true")
    ap.add_argument("--no-boots-float32", action="store_true")
    ap.add_argument("--no-index-int32", action="store_true")
    args = ap.parse_args()

    cfg = Config(
        tr=args.tr, subset=args.subset, outdir=args.outdir, tau_index=args.tau_index,
        group_cols=args.group_cols, targets=args.targets, pool=args.pool,
        n_boot=args.n_boot, seed=args.seed, ci=args.ci, chunk=args.chunk,
        jobs=args.jobs, progress=args.progress, n_animals=args.n_animals,
        pool_threshold=args.pool_threshold, pool_all=bool(args.pool_all),
        exclude_self=bool(args.exclude_self),
    )
    cfg.q_list = [float(s) for s in str(args.q).split(',') if s.strip()]
    boots_dtype = np.float32 if not args.no_boots_float32 else float
    index_dtype = np.dtype(np.int32) if not args.no_index_int32 else None

    data = get_context(tr=cfg.tr)
    speed_root = Path(data.paths["speed"])  # type: ignore[index]
    if cfg.subset:
        speed_root = speed_root / cfg.subset
    region_dirs = _find_region_folders(speed_root)
    outputs_root = Path(data.paths["speed"])  # type: ignore[index]
    outdir = outputs_root / (cfg.outdir if cfg.outdir else (cfg.subset if cfg.subset else "bootstrap"))
    outdir.mkdir(parents=True, exist_ok=True)

    # Build groups
    group_cols = [c.strip() for c in cfg.group_cols.split(',') if c.strip()]
    groups_map = build_groups_from_columns(data.cog_data_filtered, group_cols)
    targets = parse_tuple_list(cfg.targets)
    include = parse_tuple_list(cfg.pool)
    include_label = "+".join(["("+",".join(t)+")" for t in include])

    # Validate groups
    missing = [t for t in targets if t not in groups_map]
    if missing:
        print(f"[warn] Some targets not found in groups: {missing}")
    missing_inc = [t for t in include if t not in groups_map]
    if missing_inc:
        print(f"[warn] Some include groups not found in groups: {missing_inc}")

    rows: list[dict[str, object]] = []

    for region_dir in maybe_tqdm(cfg.progress, region_dirs, desc='Regions'):
        folder_label = (
            region_dir.name.replace("regions-", "")
            if region_dir.name.startswith("regions-")
            else region_dir.name
        )
        # Per-window
        bywin = _list_window_files(region_dir)
        for win, npz in maybe_tqdm(cfg.progress, bywin, desc=f'{folder_label} windows'):
            per_animal = load_per_animal_from_npz(
                npz,
                tau_index=None if cfg.tau_index < 0 else cfg.tau_index,
                n_animals=cfg.n_animals,
            )
            for tgt in targets:
                if tgt not in groups_map:
                    continue
                tgt_idxs = groups_map[tgt]
                target_vals = pool_per_animal(per_animal, tgt_idxs)
                pool_idxs: list[int] = []
                for g in include:
                    pool_idxs.extend(groups_map.get(g, []))
                if cfg.exclude_self:
                    s_self = set(int(i) for i in tgt_idxs)
                    pool_idxs = [int(i) for i in pool_idxs if int(i) not in s_self]
                pool_vals = pool_per_animal(per_animal, sorted(set(pool_idxs)))
                res = bootstrap_group_from_pool(
                    target_vals, pool_vals, q=cfg.q_list, n_boot=cfg.n_boot, ci=cfg.ci,
                    seed=cfg.seed, chunk=cfg.chunk, dtype=np.dtype(boots_dtype),
                    val_dtype=None, index_dtype=index_dtype,
                )
                p_arr = np.asarray(res.get("p", np.full(len(res.get("q", [])), np.nan)), float).ravel()
                for qi, tpt, ppt, lo_i, hi_i, inside_i, p_i in zip(
                    res["q"], res["target_point"], res["pool_point"], res["lo"], res["hi"], res["inside"], p_arr, strict=False
                ):
                    rows.append(
                        {
                            "region": folder_label,
                            "roi": folder_label,
                            "window": int(win),
                            "group": tgt,
                            "group_label": format_group_label(group_cols, tgt),
                            "group_cols": ",".join(group_cols),
                            "pool_by": "explicit",
                            "pool_match": include_label,
                            "pool_exclude_self": bool(cfg.exclude_self),
                            "q": float(qi),
                            "target_point": float(tpt),
                            "pool_point": float(ppt),
                            "lo": float(lo_i),
                            "hi": float(hi_i),
                            "inside": bool(inside_i),
                            "p": float(p_i),
                            "n_target": int(res.get("n_target", 0)),
                            "n_pool": int(res.get("n_pool", 0)),
                        }
                    )

        # Pooled (short/long/all)
        windows = [w for (w, _) in bywin]
        pools = _pool_windows_indices(windows, cfg.pool_threshold)
        if cfg.pool_all and windows:
            pools["all"] = windows
        if pools:
            bywin_map = {w: p for (w, p) in bywin}
            for pool_name, pool_windows in pools.items():
                if not pool_windows:
                    continue
                per_animals = [
                    load_per_animal_from_npz(
                        bywin_map[w],
                        tau_index=None if cfg.tau_index < 0 else cfg.tau_index,
                        n_animals=cfg.n_animals,
                    )
                    for w in pool_windows
                    if w in bywin_map
                ]
                pooled = _concat_per_animal(per_animals)
                for tgt in targets:
                    if tgt not in groups_map:
                        continue
                    tgt_idxs = groups_map[tgt]
                    target_vals = pool_per_animal(pooled, tgt_idxs)
                    pool_idxs: list[int] = []
                    for g in include:
                        pool_idxs.extend(groups_map.get(g, []))
                    if cfg.exclude_self:
                        s_self = set(int(i) for i in tgt_idxs)
                        pool_idxs = [int(i) for i in pool_idxs if int(i) not in s_self]
                    pool_vals = pool_per_animal(pooled, sorted(set(pool_idxs)))
                    res = bootstrap_group_from_pool(
                        target_vals, pool_vals, q=cfg.q_list, n_boot=cfg.n_boot, ci=cfg.ci,
                        seed=cfg.seed, chunk=cfg.chunk, dtype=np.dtype(boots_dtype),
                        val_dtype=None, index_dtype=index_dtype,
                    )
                    p_arr = np.asarray(res.get("p", np.full(len(res.get("q", [])), np.nan)), float).ravel()
                    for qi, tpt, ppt, lo_i, hi_i, inside_i, p_i in zip(
                        res["q"], res["target_point"], res["pool_point"], res["lo"], res["hi"], res["inside"], p_arr, strict=False
                    ):
                        rows.append(
                            {
                                "region": folder_label,
                                "roi": folder_label,
                                "window": pool_name,
                                "group": tgt,
                                "group_label": format_group_label(group_cols, tgt),
                                "group_cols": ",".join(group_cols),
                                "pool_by": "explicit",
                                "pool_match": include_label,
                                "pool_exclude_self": bool(cfg.exclude_self),
                                "q": float(qi),
                                "target_point": float(tpt),
                                "pool_point": float(ppt),
                                "lo": float(lo_i),
                                "hi": float(hi_i),
                                "inside": bool(inside_i),
                                "p": float(p_i),
                                "n_target": int(res.get("n_target", 0)),
                                "n_pool": int(res.get("n_pool", 0)),
                            }
                        )

    write_pooltest_csv(rows, outdir, cfg.n_boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

