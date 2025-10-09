#!/usr/bin/env python3
"""
Compute dFC speed bootstrap tables (CSV only).

Reads per-window NPZ speed files, bootstraps per-group percentiles and per-pair
percentile differences (with empirical two-sided p-values), optionally pools
windows and correlates pooled per-animal percentiles with a NOR score.
"""
#%%
from __future__ import annotations

import argparse
from collections.abc import Iterable
import csv
from dataclasses import dataclass, field
import os
from pathlib import Path
import re

#%%
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Centralized kernels (require shared_code package to be installed)
from shared_code.fun_bootstrap import (
    bootstrap_percentiles,
    bootstrap_diff_percentiles,
    bootstrap_groups_percentiles,
    bootstrap_groups_boots,
    ci_from_boots,
    pool_per_animal,
    bootstrap_group_from_pool,
)
#%%

# ---------------- Context and IO helpers ---------------- #


def get_context(tr: int | None = None):
    """Return a dataset context with paths and metadata loaded.

    Prefers the packaged context under src/net_fluidity_julien; if missing,
    falls back to julien_data/class_dataanalysis_julien.py by adding the
    julien_data folder to sys.path.
    """
    DFC = None
    # Try package import; if it fails, add src/ to sys.path and retry
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
    # Fallback to julien_data module file
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


def load_per_animal_from_npz(
    npz_path: Path, tau_index: int | None = None, n_animals: int | None = None
) -> list[np.ndarray]:
    """Load per-animal speed arrays from NPZ, optionally selecting a tau and limiting animals.

    - Each entry is a 2D array (n_tau, T_w). If tau_index is provided (>=0),
      select that row; else flatten across taus.
    - If n_animals is provided, only the first n_animals entries are considered.
    """
    z = np.load(npz_path, allow_pickle=True)
    if "speeds" not in z:
        raise KeyError(f"NPZ file missing 'speeds' key: {npz_path}")
    speeds = z["speeds"]  # object array len=n_animals; each is 2D (n_tau, T_w)
    out: list[np.ndarray] = []
    count = len(speeds) if (n_animals is None or n_animals <= 0) else min(int(n_animals), len(speeds))
    for a in range(count):
        arr = np.asarray(speeds[a], float)
        if arr.ndim != 2:
            out.append(np.array([], float))
            continue
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
    if not isinstance(cog_df, pd.DataFrame):
        raise TypeError("cog_df must be a pandas DataFrame")
    cols = [c.strip() for c in columns if c.strip()]
    tmp = cog_df.reset_index(drop=True)
    grp = tmp.groupby(cols).groups
    out: dict = {}
    for k, idx in grp.items():
        out[k if isinstance(k, tuple) else (k,)] = sorted(int(i) for i in idx)
    return out


"""All bootstrapping is delegated to shared_code.fun_bootstrap (no local impls)."""


WIN_RE = re.compile(r"speed_win(\d+)_.*\.npz$")
SUBSET_TAG_RE = re.compile(
    r"_subset_mode-[^_]*_(?:region-\d+-(?P<region>[^_]+)|lab-(?P<lab>[^_]+))"
)


# ---------------- Configuration dataclass ---------------- #


@dataclass
class BootstrapConfig:
    tr: int = 500
    subset: str | None = None
    outdir: str | None = None
    tau_index: int = 0
    q: str = "1,5,50,95,99"
    pairs: str = (
        "(WT,VEH)-(WT,LCTB92);(Dp1Yey,VEH)-(Dp1Yey,LCTB92);"
        "(WT,VEH)-(Dp1Yey,VEH);(WT,LCTB92)-(Dp1Yey,LCTB92)"
    )
    n_boot: int = 2000
    seed: int = 0
    ci: float = 95.0
    chunk: int = 128
    boots_float32: bool = True
    values_float32: bool = False
    index_int32: bool = True
    pool_threshold: str | None = None
    pool_all: bool = False
    jobs: int = 1
    region_jobs: int = 1
    blas_threads: int | None = None
    parallel_scope: str = "windows"
    load_cache: bool = False
    progress: bool = False
    group_cols: str = "genotype,treatment"
    reuse_group_boots: bool = False
    correlate_nor: bool = False
    nor_col: str | None = None
    correlate_nor_by_groups: bool = False
    n_animals: int = 48
    bootstrap_pool_cols: str | None = None
    pool_exclude_self: bool = False

    # Derived
    q_list: list[float] = field(default_factory=list)
    pairs_list: list = field(default_factory=list)
    boots_dtype: type = float
    values_dtype: type = float
    index_dtype: np.dtype | None = None

    @classmethod
    def from_args(cls, args) -> "BootstrapConfig":
        scope = _normalize_parallel_scope(args.parallel_scope)
        region_jobs = getattr(args, "region_jobs", None)
        if region_jobs is None or int(region_jobs) <= 0:
            region_jobs = args.jobs if scope == "regions" else 1
        cfg = cls(
            tr=args.tr,
            subset=args.subset,
            outdir=args.outdir,
            tau_index=args.tau_index,
            q=args.q,
            pairs=args.pairs,
            n_boot=args.n_boot,
            seed=args.seed,
            ci=args.ci,
            chunk=args.chunk,
            boots_float32=True,
            values_float32=bool(args.values_float32),
            index_int32=True,
            pool_threshold=args.pool_threshold,
            pool_all=bool(args.pool_all),
            jobs=args.jobs,
            region_jobs=int(region_jobs),
            blas_threads=args.blas_threads,
            parallel_scope=scope,
            load_cache=bool(args.load_cache),
            progress=bool(args.progress),
            group_cols=args.group_cols,
            reuse_group_boots=bool(args.reuse_group_boots),
            correlate_nor=bool(args.correlate_nor),
            nor_col=args.nor_col,
            correlate_nor_by_groups=bool(args.correlate_nor_by_groups),
            n_animals=int(args.n_animals),
            bootstrap_pool_cols=args.bootstrap_pool_cols,
            pool_exclude_self=bool(getattr(args, 'pool_exclude_self', False)),
        )
        # Flip defaults to memory-friendly unless user opts out
        if hasattr(args, "no_boots_float32") and bool(args.no_boots_float32):
            cfg.boots_float32 = False
        if hasattr(args, "no_index_int32") and bool(args.no_index_int32):
            cfg.index_int32 = False
        # Backward compat: if legacy flags are provided, honor them (force on)
        if hasattr(args, "boots_float32") and bool(args.boots_float32):
            cfg.boots_float32 = True
        if hasattr(args, "index_int32") and bool(args.index_int32):
            cfg.index_int32 = True
        if ";" in str(cfg.q):
            cfg.q_list = [
                float(s) for s in str(cfg.q).split(";") for s in s.split(",") if s.strip()
            ]
        else:
            cfg.q_list = [float(s) for s in str(cfg.q).split(",") if s.strip()]
        cfg.pairs_list = _parse_pairs(cfg.pairs)
        cfg.boots_dtype = np.float32 if cfg.boots_float32 else float
        cfg.values_dtype = np.float32 if cfg.values_float32 else float
        cfg.index_dtype = np.dtype(np.int32) if cfg.index_int32 else None
        return cfg

    @staticmethod
    def build_outdir_name(outdir: str | None, subset: str | None) -> str:
        """Resolve output directory name under paths['speed'].

        If --outdir is provided, use it; else fallback to --subset; else 'bootstrap'.
        """
        return outdir if outdir else (subset if subset else "bootstrap")


def _normalize_parallel_scope(scope: str | None) -> str:
    value = str(scope).strip().lower() if scope is not None else "windows"
    if value not in {"windows", "regions", "both"}:
        raise ValueError(
            f"Unknown parallel scope '{scope}'. Use 'windows', 'regions', or 'both'."
        )
    return value


# ---------------- Utility helpers ---------------- #


def _infer_roi_from_filename(name: str) -> str | None:
    m = SUBSET_TAG_RE.search(name)
    if not m:
        return None
    if m.group("region"):
        return m.group("region")
    if m.group("lab"):
        return m.group("lab")
    return None


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


def _pool_windows_indices(
    windows: list[int], threshold: str | int | None
) -> dict[str, list[int]]:
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


def _parse_pairs(pairs_arg: str) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    out: list[tuple[tuple[str, str], tuple[str, str]]] = []
    if not pairs_arg:
        return out
    for chunk in pairs_arg.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        left, right = chunk.split("-", 1)
        la, lb = [s.strip() for s in left.strip().strip("()").split(",", 1)]
        ra, rb = [s.strip() for s in right.strip().strip("()").split(",", 1)]
        out.append(((la, lb), (ra, rb)))
    return out


def _detect_nor_column(cog_df: pd.DataFrame, nor_col_opt: str | None) -> str | None:
    if not isinstance(cog_df, pd.DataFrame):
        return None
    if nor_col_opt:
        return nor_col_opt if nor_col_opt in cog_df.columns else None
    cands = [c for c in cog_df.columns if "nor" in str(c).lower()]
    return cands[0] if len(cands) == 1 else None


def _format_group_label(group_cols: list[str], gk: tuple) -> str:
    try:
        vals = list(gk)
        return ",".join(f"{c}={vals[i]}" for i, c in enumerate(group_cols) if i < len(vals))
    except Exception:
        return str(gk)


def _format_pool_match(pool_cols: list[str], pos: dict[str, int], gk: tuple) -> str:
    items = []
    for c in pool_cols:
        i = pos.get(c)
        if i is None:
            continue
        try:
            items.append(f"{c}={gk[i]}")
        except Exception:
            pass
    return ",".join(items)


def _compute_nor_correlation_rows(
    pooled: list[np.ndarray],
    q_list: list[float],
    cog_df: pd.DataFrame,
    nor_col: str,
    region_label: str,
    pool_name: str,
    mat: np.ndarray | None = None,
    indices: list[int] | None = None,
    group_label: str | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not isinstance(cog_df, pd.DataFrame) or nor_col not in cog_df.columns:
        return rows
    qs = [float(x) for x in q_list]
    if mat is None:
        n_anim = len(pooled)
        mat = np.full((n_anim, len(qs)), np.nan, float)
        for i in range(n_anim):
            vals = np.asarray(pooled[i], float)
            if vals.size:
                mat[i, :] = np.percentile(vals, qs)
    nor_vals = np.asarray(cog_df[nor_col], float)

    # Subset by indices if provided (per-group correlation)
    if indices is not None:
        idx = np.asarray(indices, int)
        idx = idx[(idx >= 0) & (idx < mat.shape[0])]
        mat = mat[idx, :]
        nor_vals = nor_vals[idx]

    # Optional SciPy for correlation p-values (falls back to NaN when unavailable)
    try:
        from scipy.stats import pearsonr as _pearsonr, spearmanr as _spearmanr  # type: ignore
    except Exception:  # pragma: no cover
        _pearsonr = None  # type: ignore
        _spearmanr = None  # type: ignore

    # Align lengths if mat rows != nor_vals length
    n_mat = int(mat.shape[0])
    n_nor = int(nor_vals.shape[0])
    k = min(n_mat, n_nor)
    mat_use = mat[:k, :]
    nor_use = nor_vals[:k]

    for j, qi in enumerate(qs):
        a = mat_use[:, j]
        m = np.isfinite(a) & np.isfinite(nor_use)
        n_used = int(np.count_nonzero(m))
        pearson_r = float("nan")
        pearson_p = float("nan")
        spearman_rho = float("nan")
        spearman_p = float("nan")
        if n_used >= 3:
            aa, bb = a[m], nor_use[m]
            if _pearsonr is not None:
                try:
                    r_val, p_val = _pearsonr(aa, bb)
                    pearson_r = float(r_val)
                    pearson_p = float(p_val)
                except Exception:
                    pearson_r = float(np.corrcoef(aa, bb)[0, 1])
                    pearson_p = float("nan")
            else:
                pearson_r = float(np.corrcoef(aa, bb)[0, 1])
            if _spearmanr is not None:
                try:
                    rho_val, p_val2 = _spearmanr(aa, bb)
                    spearman_rho = float(rho_val)
                    spearman_p = float(p_val2)
                except Exception:
                    ra = np.argsort(np.argsort(aa))
                    rb = np.argsort(np.argsort(bb))
                    spearman_rho = float(np.corrcoef(ra, rb)[0, 1])
                    spearman_p = float("nan")
            else:
                ra = np.argsort(np.argsort(aa))
                rb = np.argsort(np.argsort(bb))
                spearman_rho = float(np.corrcoef(ra, rb)[0, 1])
        rows.append(
            {
                "region": region_label,
                "roi": region_label,
                "window": pool_name,
                "q": float(qi),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
                "n": n_used,
                "nor_col": nor_col,
                "group": (group_label if group_label is not None else "__ALL__"),
            }
        )
    return rows


def limit_blas_threads(n: int | None):
    if n is None:
        return
    try:
        n = int(n)
    except Exception:
        return
    for k in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        if os.environ.get(k) is None:
            os.environ[k] = str(n)
    try:
        from threadpoolctl import threadpool_limits  # type: ignore

        threadpool_limits(limits=n)
    except Exception:
        pass


def resolve_paths_and_groups(cfg: BootstrapConfig, data):
    speed_root = Path(data.paths["speed"])  # type: ignore[index]
    if cfg.subset:
        speed_root = speed_root / cfg.subset
    region_dirs = _find_region_folders(speed_root)
    groups_map = build_groups_from_columns(
        data.cog_data_filtered,
        [s.strip() for s in cfg.group_cols.split(",") if s.strip()],
    )
    outputs_root = Path(data.paths["speed"])  # type: ignore[index]
    outdir_name = BootstrapConfig.build_outdir_name(cfg.outdir, cfg.subset)
    outdir = outputs_root / outdir_name
    outdir.mkdir(parents=True, exist_ok=True)
    q_path = outdir / "speed_bootstrap_quantiles.csv"
    d_path = outdir / "speed_bootstrap_diffs.csv"
    c_path = outdir / "speed_nor_correlations.csv"
    return region_dirs, groups_map, outdir, q_path, d_path, c_path


def _build_pool_groups(groups_map: dict, group_cols: list[str], pool_cols_opt: str | None) -> dict | None:
    """Build mapping group_key -> pooled supergroup indices from a subset of columns.

    - When pool_cols_opt is None, returns None (no pooling for bootstrap-from-pool tests).
    - Otherwise, aggregates animal indices for all groups that share identical values
      on the given pool columns.
    """
    if not pool_cols_opt:
        return None
    pool_cols = [c.strip() for c in str(pool_cols_opt).split(",") if c.strip()]
    pos = {c: i for i, c in enumerate(group_cols)}
    use_pos = [pos[c] for c in pool_cols if c in pos]
    if not use_pos:
        return None
    # pool_key -> combined indices
    pool_to_idxs: dict[tuple, list[int]] = {}
    for gk, idxs in groups_map.items():
        key = tuple(gk[i] for i in use_pos)
        pool_to_idxs.setdefault(key, []).extend(int(i) for i in idxs)
    # group_key -> combined pool indices
    gk_to_pool: dict = {}
    for gk, _ in groups_map.items():
        key = tuple(gk[i] for i in use_pos)
        gk_to_pool[gk] = sorted(set(pool_to_idxs.get(key, [])))
    return gk_to_pool


def maybe_tqdm(progress: bool, it, desc: str):
    if progress:
        try:
            from tqdm import tqdm

            return tqdm(it, desc=desc)
        except Exception:
            return it
    return it


def write_outputs(
    q_rows: list[dict[str, object]],
    d_rows: list[dict[str, object]],
    c_rows: list[dict[str, object]],
    outdir: Path,
    n_boot: int,
    p_rows: list[dict[str, object]] | None = None,
):
    q_path = outdir / "speed_bootstrap_quantiles.csv"
    d_path = outdir / "speed_bootstrap_diffs.csv"
    if q_rows:
        q_cols = ["region", "roi", "window", "group", "q", "point", "lo", "hi", "n"]
        with q_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=q_cols)
            w.writeheader()
            w.writerows(q_rows)
        print(f"Wrote: {q_path}")
        try:
            q_path_sfx = outdir / f"speed_bootstrap_quantiles_nboot-{int(n_boot)}.csv"
            with q_path_sfx.open("w", newline="") as f2:
                w2 = csv.DictWriter(f2, fieldnames=q_cols)
                w2.writeheader()
                w2.writerows(q_rows)
            print(f"Wrote: {q_path_sfx}")
        except Exception:
            pass
    if d_rows:
        d_cols = [
            "region",
            "roi",
            "window",
            "A",
            "B",
            "q",
            "diff",
            "lo",
            "hi",
            "p",
            "p_method",
            "significant",
            "n_a",
            "n_b",
        ]
        with d_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=d_cols)
            w.writeheader()
            w.writerows(d_rows)
        print(f"Wrote: {d_path}")
        try:
            d_path_sfx = outdir / f"speed_bootstrap_diffs_nboot-{int(n_boot)}.csv"
            with d_path_sfx.open("w", newline="") as f2:
                w2 = csv.DictWriter(f2, fieldnames=d_cols)
                w2.writeheader()
                w2.writerows(d_rows)
            print(f"Wrote: {d_path_sfx}")
        except Exception:
            pass
    if c_rows:
        corr_path = outdir / "speed_nor_correlations.csv"
        c_cols = [
            "region",
            "roi",
            "window",
            "q",
            "pearson_r",
            "pearson_p",
            "spearman_rho",
            "spearman_p",
            "n",
            "nor_col",
            "group",
        ]
        with corr_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=c_cols)
            w.writeheader()
            w.writerows(c_rows)
        print(f"Wrote: {corr_path}")
        try:
            corr_path_sfx = outdir / f"speed_nor_correlations_nboot-{int(n_boot)}.csv"
            with corr_path_sfx.open("w", newline="") as f2:
                w2 = csv.DictWriter(f2, fieldnames=c_cols)
                w2.writeheader()
                w2.writerows(c_rows)
            print(f"Wrote: {corr_path_sfx}")
        except Exception:
            pass
    # Pooltest CSV (bootstrap from pooled supergroups)
    if p_rows:
        ptest_path = outdir / "speed_bootstrap_pooltest.csv"
        p_cols = [
            "region",
            "roi",
            "window",
            "group",
            "group_label",
            "group_cols",
            "pool_by",
            "pool_match",
            "pool_exclude_self",
            "q",
            "target_point",
            "pool_point",
            "lo",
            "hi",
            "inside",
            "p",
            "n_target",
            "n_pool",
        ]
        with ptest_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=p_cols)
            w.writeheader()
            w.writerows(p_rows)
        print(f"Wrote: {ptest_path}")
        try:
            ptest_path_sfx = outdir / f"speed_bootstrap_pooltest_nboot-{int(n_boot)}.csv"
            with ptest_path_sfx.open("w", newline="") as f2:
                w2 = csv.DictWriter(f2, fieldnames=p_cols)
                w2.writeheader()
                w2.writerows(p_rows)
            print(f"Wrote: {ptest_path_sfx}")
        except Exception:
            pass
    # done


# ---------------- Processing ---------------- #


P_METHOD_DESC = (
    "empirical two-sided bootstrap on percentile differences with +1/(B+1) smoothing"
)

#%%
def process_region_dir(
    region_dir: Path,
    folder_label: str,
    cfg: BootstrapConfig,
    groups_map: dict,
    q_list: list[float],
    boots_dtype: type | np.dtype,
    values_dtype: type | np.dtype,
    index_dtype: np.dtype | None,
    data,
    pairs: list,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Process one region directory (one ROI label).

    Reads all per-window NPZ files, pools per-animal values, and computes:
    - Per-group bootstrap percentiles with CIs (quantiles_rows)
    - Per-pair bootstrap percentile differences with CIs and p-values (diffs_rows)
    - Optional correlations of per-animal percentiles vs NOR (per-window and pooled) (corr_rows)

    Returns four lists of dict rows ready to be written as CSVs.
    """
    quantiles_rows: list[dict[str, object]] = []
    diffs_rows: list[dict[str, object]] = []
    corr_rows: list[dict[str, object]] = []
    pooltest_rows: list[dict[str, object]] = []
    pooltest_rows: list[dict[str, object]] = []

    win_files = _list_window_files(region_dir)
    if not win_files:
        return quantiles_rows, diffs_rows, corr_rows, pooltest_rows

    # Build pooled-supergroup mapping if requested
    group_cols_list = [s.strip() for s in str(cfg.group_cols).split(",") if s.strip()]
    pool_cols_list = [s.strip() for s in str(cfg.bootstrap_pool_cols).split(",") if s.strip()] if cfg.bootstrap_pool_cols else []
    pool_groups_map = _build_pool_groups(groups_map, group_cols_list, cfg.bootstrap_pool_cols)
    pos_map = {c: i for i, c in enumerate(group_cols_list)}

    # ------ Per-window processing ------
    def _process_win(
        win: int,
        npz: Path,
        folder_label: str = folder_label,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:

        # Initialize per-window result rows
        rows_q: list[dict[str, object]] = []
        rows_d: list[dict[str, object]] = []
        rows_c: list[dict[str, object]] = []
        rows_p: list[dict[str, object]] = []

        # Infer ROI label from filename if possible
        try:
            roi = _infer_roi_from_filename(npz.name) or folder_label
        except Exception:
            roi = folder_label

        # Load per-animal values from NPZ (A, Speed values for one window)
        per_animal = load_per_animal_from_npz(
            npz,
            tau_index=None if cfg.tau_index < 0 else cfg.tau_index,
            n_animals=cfg.n_animals,
        )
        # Optionally pool all windows together
        if cfg.reuse_group_boots:
            boots_map = bootstrap_groups_boots(
                per_animal,
                groups_map,
                q=q_list,
                n_boot=cfg.n_boot,
                seed=cfg.seed,
                chunk=cfg.chunk,
                dtype=np.dtype(boots_dtype),
                val_dtype=(None if values_dtype is None else np.dtype(values_dtype)),
                index_dtype=(None if index_dtype is None else np.dtype(index_dtype)),
            )
            q_arr = boots_map.get("__q__", np.asarray(q_list, float))
            for gk, idxs in groups_map.items():
                if gk == "__q__":
                    continue
                pooled_g = pool_per_animal(per_animal, idxs)
                if pooled_g.size:
                    point = np.percentile(pooled_g, q_arr)
                else:
                    point = np.full_like(q_arr, np.nan, dtype=float)
                boots = boots_map[gk]
                lo, hi = ci_from_boots(boots, ci=cfg.ci)
                for qi, pt, lo_i, hi_i in zip(q_arr, point, lo, hi, strict=False):
                    rows_q.append(
                        {
                            "region": roi,
                            "roi": roi,
                            "window": int(win),
                            "group": gk,
                            "q": float(qi),
                            "point": float(pt),
                            "lo": float(lo_i),
                            "hi": float(hi_i),
                            "n": int(pooled_g.size),
                        }
                    )
            # Per-pair differences from stored boots
            for A, B in pairs:
                if A not in groups_map or B not in groups_map:
                    continue
                boots_A = boots_map.get(A)
                boots_B = boots_map.get(B)
                if boots_A is None or boots_B is None:
                    continue
                n_used = min(boots_A.shape[0], boots_B.shape[0])
                diff_boots = boots_A[:n_used, :] - boots_B[:n_used, :]
                lo, hi = ci_from_boots(diff_boots, ci=cfg.ci)
                pooled_A = pool_per_animal(per_animal, groups_map[A])
                pooled_B = pool_per_animal(per_animal, groups_map[B])
                if pooled_A.size and pooled_B.size:
                    point = np.percentile(pooled_A, q_arr) - np.percentile(pooled_B, q_arr)
                else:
                    point = np.full_like(q_arr, np.nan, dtype=float)
                p_vals = [
                    float(
                        (np.count_nonzero(np.abs(diff_boots[:, j]) >= abs(point[j])) + 1.0)
                        / (diff_boots.shape[0] + 1.0)
                    )
                    for j in range(diff_boots.shape[1])
                ]
                sig = (lo > 0) | (hi < 0)
                for qi, pt, lo_i, hi_i, p_i, s in zip(
                    q_arr, point, lo, hi, p_vals, sig, strict=False
                ):
                    rows_d.append(
                        {
                            "region": roi,
                            "roi": roi,
                            "window": int(win),
                            "A": A,
                            "B": B,
                            "q": float(qi),
                            "diff": float(pt),
                            "lo": float(lo_i),
                            "hi": float(hi_i),
                            "p": float(p_i),
                            "p_method": P_METHOD_DESC,
                            "significant": bool(s),
                            "n_a": int(pooled_A.size),
                            "n_b": int(pooled_B.size),
                        }
                    )
        # No reuse of group boots; compute everything from scratch
        else:
            # ------ Per-group percentiles Bootstrap with CIs
            qa = bootstrap_groups_percentiles(
                per_animal,
                groups_map,
                q=q_list,
                n_boot=cfg.n_boot,
                ci=cfg.ci,
                seed=cfg.seed,
                chunk=cfg.chunk,
                dtype=np.dtype(boots_dtype),
                val_dtype=(None if values_dtype is None else np.dtype(values_dtype)),
                index_dtype=(None if index_dtype is None else np.dtype(index_dtype)),
            )
            # Store per-group results
            for gk, res in qa.items():
                for qi, pt, lo, hi in zip(
                    res["q"], res["point"], res["lo"], res["hi"], strict=False
                ):
                    rows_q.append(
                        {
                            "region": roi,
                            "roi": roi,
                            "window": int(win),
                            "group": gk,
                            "q": float(qi),
                            "point": float(pt),
                            "lo": float(lo),
                            "hi": float(hi),
                            "n": int(res["n"]),
                        }
                    )
            # Per-window correlations with NOR, if requested
            if cfg.correlate_nor and isinstance(data.cog_data_filtered, pd.DataFrame):
                nor_col = _detect_nor_column(data.cog_data_filtered, cfg.nor_col)
                if nor_col:
                    # Overall (all animals)
                    rows_c.extend(
                        _compute_nor_correlation_rows(
                            per_animal,
                            q_list,
                            data.cog_data_filtered,
                            nor_col,
                            roi,
                            int(win),
                            mat=None,
                            indices=None,
                            group_label="__ALL__",
                        )
                    )
                    # Per-group correlations
                    if cfg.correlate_nor_by_groups:
                        for gk, idxs in groups_map.items():
                            if gk == "__q__":
                                continue
                            rows_c.extend(
                                _compute_nor_correlation_rows(
                                    per_animal,
                                    q_list,
                                    data.cog_data_filtered,
                                    nor_col,
                                    roi,
                                    int(win),
                                    mat=None,
                                    indices=[int(i) for i in idxs],
                                    group_label=str(gk),
                                )
                            )
            # ------ Per-pair percentile differences Bootstrap with CIs and p-values
            for A, B in pairs:
                if A not in groups_map or B not in groups_map:
                    continue
                xa = pool_per_animal(per_animal, groups_map[A])
                xb = pool_per_animal(per_animal, groups_map[B])
                qd = bootstrap_diff_percentiles(
                    xa,
                    xb,
                    q=q_list,
                    n_boot=cfg.n_boot,
                    ci=cfg.ci,
                    seed=cfg.seed,
                    chunk=cfg.chunk,
                    dtype=np.dtype(boots_dtype),
                    val_dtype=(None if values_dtype is None else np.dtype(values_dtype)),
                    index_dtype=(None if index_dtype is None else np.dtype(index_dtype)),
                )
                p_arr = np.asarray(qd.get("p", np.full(len(qd.get("q", [])), np.nan)), float).ravel()
                for qi, pt, lo, hi, sig, p_i in zip(
                    qd["q"], qd["point"], qd["lo"], qd["hi"], qd["sig"], p_arr, strict=False
                ):
                    rows_d.append(
                        {
                            "region": roi,
                            "roi": roi,
                            "window": int(win),
                            "A": A,
                            "B": B,
                            "q": float(qi),
                            "diff": float(pt),
                            "lo": float(lo),
                            "hi": float(hi),
                            "p": float(p_i),
                            "p_method": P_METHOD_DESC,
                            "significant": bool(sig),
                            "n_a": int(qd.get("n_x", 0)),
                            "n_b": int(qd.get("n_y", 0)),
                        }
                    )
        # Pool-test rows (per-window) if pooling supergroups are defined
        if pool_groups_map:
            for gk, idxs in groups_map.items():
                if gk == "__q__":
                    continue
                target_vals = pool_per_animal(per_animal, idxs)
                pool_idxs = list(pool_groups_map.get(gk, []))
                if cfg.pool_exclude_self:
                    s_self = set(int(i) for i in idxs)
                    pool_idxs = [int(i) for i in pool_idxs if int(i) not in s_self]
                pool_vals = pool_per_animal(per_animal, pool_idxs)
                res = bootstrap_group_from_pool(
                    target_vals,
                    pool_vals,
                    q=q_list,
                    n_boot=cfg.n_boot,
                    ci=cfg.ci,
                    seed=cfg.seed,
                    chunk=cfg.chunk,
                    dtype=np.dtype(boots_dtype),
                    val_dtype=(None if values_dtype is None else np.dtype(values_dtype)),
                    index_dtype=(None if index_dtype is None else np.dtype(index_dtype)),
                )
                p_arr = np.asarray(res.get("p", np.full(len(res.get("q", [])), np.nan)), float).ravel()
                for qi, tpt, ppt, lo_i, hi_i, inside_i, p_i in zip(
                    res["q"],
                    res["target_point"],
                    res["pool_point"],
                    res["lo"],
                    res["hi"],
                    res["inside"],
                    p_arr,
                    strict=False,
                ):
                    rows_p.append(
                        {
                            "region": roi,
                            "roi": roi,
                            "window": int(win),
                            "group": gk,
                            "group_label": _format_group_label(group_cols_list, gk),
                            "group_cols": ",".join(group_cols_list),
                            "pool_by": ",".join(pool_cols_list),
                            "pool_match": _format_pool_match(pool_cols_list, pos_map, gk),
                            "pool_exclude_self": bool(cfg.pool_exclude_self),
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

        return rows_q, rows_d, rows_c, rows_p

    # ------ Parallel processing per window ------
    if cfg.jobs and cfg.jobs > 1 and cfg.parallel_scope in {"windows", "both"}:
        print(f"Processing {len(win_files)} windows in parallel with {cfg.jobs} jobs...")
        results = Parallel(n_jobs=cfg.jobs, prefer="processes")(
            delayed(_process_win)(w, p) for (w, p) in win_files
        )
        if results:
            for rq, rd, rc, rp in results:
                quantiles_rows.extend(rq)
                diffs_rows.extend(rd)
                corr_rows.extend(rc)
                pooltest_rows.extend(rp)
    #------ Sequential processing per window ------
    else:
        print(f"Processing {len(win_files)} windows sequentially...")
        for w, p in maybe_tqdm(cfg.progress, win_files, f"{folder_label} windows"):
            rq, rd, rc, rp = _process_win(w, p)
            quantiles_rows.extend(rq)
            diffs_rows.extend(rd)
            corr_rows.extend(rc)
            pooltest_rows.extend(rp)

    # Pools of windows (e.g. short vs long)
    windows = [w for (w, _) in win_files]
    pools = _pool_windows_indices(windows, cfg.pool_threshold)
    if cfg.pool_all and windows:
        pools["all"] = windows
    if pools:
        by_win = {w: p for (w, p) in win_files}
        for pool_name, pool_windows in pools.items():
            if not pool_windows:
                continue
            per_animals = [
            load_per_animal_from_npz(
                by_win[w],
                tau_index=None if cfg.tau_index < 0 else cfg.tau_index,
                n_animals=cfg.n_animals,
            )
            for w in pool_windows
            if w in by_win
        ]
            pooled = _concat_per_animal(per_animals)
            # Precompute per-animal percentiles matrix for correlations
            mat = None
            if cfg.correlate_nor:
                qs = [float(x) for x in q_list]
                mat = np.full((len(pooled), len(qs)), np.nan, float)
                for i in range(len(pooled)):
                    vals = np.asarray(pooled[i], float)
                    if vals.size:
                        mat[i, :] = np.percentile(vals, qs)
            if cfg.reuse_group_boots:
                boots_map = bootstrap_groups_boots(
                    pooled,
                    groups_map,
                    q=q_list,
                    n_boot=cfg.n_boot,
                    seed=cfg.seed,
                    chunk=cfg.chunk,
                    dtype=np.dtype(boots_dtype),
                    val_dtype=(None if values_dtype is None else np.dtype(values_dtype)),
                    index_dtype=(None if index_dtype is None else np.dtype(index_dtype)),
                )
                q_arr = boots_map.get("__q__", np.asarray(q_list, float))
                for gk, idxs in groups_map.items():
                    if gk == "__q__":
                        continue
                    pooled_g = pool_per_animal(pooled, idxs)
                    boots = boots_map[gk]
                    lo, hi = ci_from_boots(boots, ci=cfg.ci)
                    if pooled_g.size:
                        point = np.percentile(pooled_g, q_arr)
                    else:
                        point = np.full_like(q_arr, np.nan, dtype=float)
                    for qi, pt, lo_i, hi_i in zip(q_arr, point, lo, hi, strict=False):
                        quantiles_rows.append(
                            {
                                "region": folder_label,
                                "roi": folder_label,
                                "window": pool_name,
                                "group": gk,
                                "q": float(qi),
                                "point": float(pt),
                                "lo": float(lo_i),
                                "hi": float(hi_i),
                                "n": int(pooled_g.size),
                            }
                        )
                for A, B in pairs:
                    if A not in groups_map or B not in groups_map:
                        continue
                    boots_A = boots_map.get(A)
                    boots_B = boots_map.get(B)
                    if boots_A is None or boots_B is None:
                        continue
                    n_used = min(boots_A.shape[0], boots_B.shape[0])
                    diff_boots = boots_A[:n_used, :] - boots_B[:n_used, :]
                    lo, hi = ci_from_boots(diff_boots, ci=cfg.ci)
                    pooled_A = pool_per_animal(pooled, groups_map[A])
                    pooled_B = pool_per_animal(pooled, groups_map[B])
                    point = np.percentile(pooled_A, q_arr) - np.percentile(pooled_B, q_arr)
                    p_vals = [
                        float(
                            (np.count_nonzero(np.abs(diff_boots[:, j]) >= abs(point[j])) + 1.0)
                            / (diff_boots.shape[0] + 1.0)
                        )
                        for j in range(diff_boots.shape[1])
                    ]
                    sig = (lo > 0) | (hi < 0)
                    for qi, pt, lo_i, hi_i, p_i, s in zip(
                        q_arr, point, lo, hi, p_vals, sig, strict=False
                    ):
                        diffs_rows.append(
                            {
                                "region": folder_label,
                                "roi": folder_label,
                                "window": pool_name,
                                "A": A,
                                "B": B,
                                "q": float(qi),
                                "diff": float(pt),
                                "lo": float(lo_i),
                                "hi": float(hi_i),
                                "p": float(p_i),
                                "p_method": P_METHOD_DESC,
                                "significant": bool(s),
                                "n_a": int(pooled_A.size),
                                "n_b": int(pooled_B.size),
                            }
                        )
                # Pool-test against pooled supergroups (pooled windows), reuse boots not needed
            if pool_groups_map:
                for gk, idxs in groups_map.items():
                    if gk == "__q__":
                        continue
                    target_vals = pool_per_animal(pooled, idxs)
                    pool_idxs = list(pool_groups_map.get(gk, []))
                    if cfg.pool_exclude_self:
                        s_self = set(int(i) for i in idxs)
                        pool_idxs = [int(i) for i in pool_idxs if int(i) not in s_self]
                    pool_vals = pool_per_animal(pooled, pool_idxs)
                    res = bootstrap_group_from_pool(
                        target_vals,
                        pool_vals,
                        q=q_list,
                        n_boot=cfg.n_boot,
                        ci=cfg.ci,
                        seed=cfg.seed,
                        chunk=cfg.chunk,
                        dtype=np.dtype(boots_dtype),
                        val_dtype=(None if values_dtype is None else np.dtype(values_dtype)),
                        index_dtype=(None if index_dtype is None else np.dtype(index_dtype)),
                    )
                    p_arr = np.asarray(res.get("p", np.full(len(res.get("q", [])), np.nan)), float).ravel()
                    for qi, tpt, ppt, lo_i, hi_i, inside_i, p_i in zip(
                        res["q"],
                        res["target_point"],
                        res["pool_point"],
                        res["lo"],
                        res["hi"],
                        res["inside"],
                        p_arr,
                        strict=False,
                    ):
                        pooltest_rows.append(
                                {
                                    "region": folder_label,
                                    "roi": folder_label,
                                    "window": pool_name,
                                    "group": gk,
                                    "group_label": _format_group_label(group_cols_list, gk),
                                    "group_cols": ",".join(group_cols_list),
                                    "pool_by": ",".join(pool_cols_list),
                                    "pool_match": _format_pool_match(pool_cols_list, pos_map, gk),
                                    "pool_exclude_self": bool(cfg.pool_exclude_self),
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
            else:
                qa = bootstrap_groups_percentiles(
                    pooled,
                    groups_map,
                    q=q_list,
                    n_boot=cfg.n_boot,
                    ci=cfg.ci,
                    seed=cfg.seed,
                    early_stop=0.0,
                    chunk=cfg.chunk,
                    dtype=np.dtype(boots_dtype),
                    val_dtype=(None if values_dtype is None else np.dtype(values_dtype)),
                    index_dtype=(None if index_dtype is None else np.dtype(index_dtype)),
                )
                for gk, res in qa.items():
                    for qi, pt, lo, hi in zip(
                        res["q"], res["point"], res["lo"], res["hi"], strict=False
                    ):
                        quantiles_rows.append(
                            {
                                "region": folder_label,
                                "roi": folder_label,
                                "window": pool_name,
                                "group": gk,
                                "q": float(qi),
                                "point": float(pt),
                                "lo": float(lo),
                                "hi": float(hi),
                                "n": int(res["n"]),
                            }
                        )
                if cfg.correlate_nor and isinstance(
                    data.cog_data_filtered, pd.DataFrame
                ):
                    nor_col = _detect_nor_column(data.cog_data_filtered, cfg.nor_col)
                    if nor_col:
                        # Overall
                        corr_rows.extend(
                            _compute_nor_correlation_rows(
                                pooled,
                                q_list,
                                data.cog_data_filtered,
                                nor_col,
                                folder_label,
                                pool_name,
                                mat=mat,
                                indices=None,
                                group_label="__ALL__",
                            )
                        )
                        # Per-group
                        if cfg.correlate_nor_by_groups:
                            for gk, idxs in groups_map.items():
                                if gk == "__q__":
                                    continue
                                corr_rows.extend(
                                    _compute_nor_correlation_rows(
                                        pooled,
                                        q_list,
                                        data.cog_data_filtered,
                                        nor_col,
                                        folder_label,
                                        pool_name,
                                        mat=mat,
                                        indices=[int(i) for i in idxs],
                                        group_label=str(gk),
                                    )
                                )
                for A, B in pairs:
                    if A not in groups_map or B not in groups_map:
                        continue
                    pooled_A = pool_per_animal(pooled, groups_map[A])
                    pooled_B = pool_per_animal(pooled, groups_map[B])
                    qd = bootstrap_diff_percentiles(
                        pooled_A,
                        pooled_B,
                        q=q_list,
                        n_boot=cfg.n_boot,
                        ci=cfg.ci,
                        seed=cfg.seed,
                        chunk=cfg.chunk,
                        dtype=np.dtype(boots_dtype),
                        val_dtype=(None if values_dtype is None else np.dtype(values_dtype)),
                        index_dtype=(None if index_dtype is None else np.dtype(index_dtype)),
                    )
                    p_arr = np.asarray(qd.get("p", np.full(len(qd.get("q", [])), np.nan)), float).ravel()
                    for qi, pt, lo, hi, sig, p_i in zip(
                        qd["q"], qd["point"], qd["lo"], qd["hi"], qd["sig"], p_arr, strict=False
                    ):
                        diffs_rows.append(
                            {
                                "region": folder_label,
                                "roi": folder_label,
                                "window": pool_name,
                                "A": A,
                                "B": B,
                                "q": float(qi),
                                "diff": float(pt),
                                "lo": float(lo),
                                "hi": float(hi),
                                "p": float(p_i),
                                "p_method": P_METHOD_DESC,
                                "significant": bool(sig),
                                "n_a": int(qd.get("n_x", 0)),
                                "n_b": int(qd.get("n_y", 0)),
                            }
                        )
                # Pool-test against pooled supergroups (pooled windows)
                if pool_groups_map:
                    for gk, idxs in groups_map.items():
                        if gk == "__q__":
                            continue
                        target_vals = pool_per_animal(pooled, idxs)
                        pool_idxs = list(pool_groups_map.get(gk, []))
                        if cfg.pool_exclude_self:
                            s_self = set(int(i) for i in idxs)
                            pool_idxs = [int(i) for i in pool_idxs if int(i) not in s_self]
                        pool_vals = pool_per_animal(pooled, pool_idxs)
                        res = bootstrap_group_from_pool(
                            target_vals,
                            pool_vals,
                            q=q_list,
                            n_boot=cfg.n_boot,
                            ci=cfg.ci,
                            seed=cfg.seed,
                            chunk=cfg.chunk,
                            dtype=np.dtype(boots_dtype),
                            val_dtype=(None if values_dtype is None else np.dtype(values_dtype)),
                            index_dtype=(None if index_dtype is None else np.dtype(index_dtype)),
                        )
                        p_arr = np.asarray(res.get("p", np.full(len(res.get("q", [])), np.nan)), float).ravel()
                        for qi, tpt, ppt, lo_i, hi_i, inside_i, p_i in zip(
                            res["q"],
                            res["target_point"],
                            res["pool_point"],
                            res["lo"],
                            res["hi"],
                            res["inside"],
                            p_arr,
                            strict=False,
                        ):
                            pooltest_rows.append(
                                {
                                    "region": folder_label,
                                    "roi": folder_label,
                                    "window": pool_name,
                                    "group": gk,
                                    "group_label": _format_group_label(group_cols_list, gk),
                                    "group_cols": ",".join(group_cols_list),
                                    "pool_by": ",".join(pool_cols_list),
                                    "pool_match": _format_pool_match(pool_cols_list, pos_map, gk),
                                    "pool_exclude_self": bool(cfg.pool_exclude_self),
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
    return quantiles_rows, diffs_rows, corr_rows, pooltest_rows


# ---------------- CLI ---------------- #
#%%

def main() -> int:
    class HelpFormatter(
        argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter
    ):
        pass

    ap = argparse.ArgumentParser(
        description=(
            "Compute dFC speed bootstrap tables (CSV only):\n"
            "- Reads per-window NPZ speed files under paths['speed']/[--subset]\n"
            "- Writes CSVs to [--outdir]"
        ),
        formatter_class=HelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/compute_speed_bootstrap.py \\\n+              --tr 400 --subset regions400 --tau-index 0 --n-boot 500 \\\n+              --reuse-group-boots --boots-float32 --chunk 256 --progress\n"
        ),
    )
    ap.add_argument("--tr", type=int, default=500, help="Total TR used to select metadata.")
    ap.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Subset under paths['speed'] (e.g., 'regions400').",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output folder name under paths['speed']; defaults to --subset or 'bootstrap'.",
    )
    ap.add_argument(
        "--tau-index",
        type=int,
        default=0,
        help="Tau index (0-based). Use -1 to pool across taus.",
    )
    ap.add_argument(
        "--q",
        type=str,
        default="1,5,50,95,99",
        help="Comma-separated percentiles (e.g., '5,50,95').",
    )
    ap.add_argument(
        "--pairs",
        type=str,
        default=(
            "(WT,VEH)-(WT,LCTB92);(Dp1Yey,VEH)-(Dp1Yey,LCTB92);"
            "(WT,VEH)-(Dp1Yey,VEH);(WT,LCTB92)-(Dp1Yey,LCTB92)"
        ),
        help=(
            "Pairs to compare as A-B; semicolon-separated. Format: '(a1,a2,...)-(b1,b2,...)'. "
            "Arity must match --group-cols."
        ),
    )
    ap.add_argument("--n-boot", type=int, default=2000, help="Bootstrap replicates per unit.")
    ap.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    ap.add_argument("--ci", type=float, default=95.0, help="Confidence level for CIs (percent).")
    # Memory-friendly defaults; opt out with --no-* flags
    ap.add_argument("--boots-float32", action="store_true", help="(Deprecated) Use float32 boots (now default).")
    ap.add_argument("--values-float32", action="store_true", help="Cast values to float32 before resampling.")
    ap.add_argument("--index-int32", action="store_true", help="(Deprecated) Use int32 indices (now default).")
    ap.add_argument("--no-boots-float32", action="store_true", help="Store boots in float64 (opt out of default float32).")
    ap.add_argument("--no-index-int32", action="store_true", help="Use platform default index dtype (opt out of int32).")
    ap.add_argument(
        "--pool-threshold",
        type=str,
        default=None,
        help="Pool windows into short/long by 'median' or integer cutoff.",
    )
    ap.add_argument("--pool-all", action="store_true", help="Also add an 'all' pool combining all windows.")
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel jobs for per-window work (used when scope includes windows).",
    )
    ap.add_argument(
        "--region-jobs",
        type=int,
        default=None,
        help="Parallel jobs for per-region work (defaults to --jobs when scope includes regions).",
    )
    ap.add_argument(
        "--blas-threads",
        type=int,
        default=None,
        help="Limit BLAS threads per process (default 1 when --jobs > 1).",
    )
    ap.add_argument(
        "--parallel-scope",
        type=str,
        default="windows",
        help="Parallelization scope: 'windows', 'regions', or 'both'.",
    )
    ap.add_argument("--load-cache", action="store_true", help="Reuse existing CSVs if present.")
    ap.add_argument("--progress", action="store_true", help="Show progress bars (requires tqdm).")
    ap.add_argument(
        "--group-cols",
        type=str,
        default="genotype,treatment",
        help="Comma-separated grouping columns.",
    )
    ap.add_argument(
        "--reuse-group-boots",
        action="store_true",
        help="Reuse per-group bootstrap replicates to compute all pairs.",
    )
    ap.add_argument(
        "--correlate-nor",
        action="store_true",
        help=(
            "On pooled windows (short/long/all), correlate per-animal speed "
            "percentiles with a NOR score column."
        ),
    )
    ap.add_argument(
        "--nor-col",
        type=str,
        default=None,
        help=(
            "Exact column name for NOR score; if omitted, auto-detects a single "
            "column containing 'nor' (case-insensitive)."
        ),
    )
    ap.add_argument(
        "--correlate-nor-by-groups",
        action="store_true",
        help="Also compute correlations within each group (per --group-cols).",
    )
    ap.add_argument("--n-animals", type=int, default=48, help="Limit number of animals loaded per file (0 = all).")
    ap.add_argument(
        "--bootstrap-pool-cols",
        type=str,
        default=None,
        help="Comma-separated subset of --group-cols to form pooling supergroups (e.g., 'genotype').",
    )
    ap.add_argument(
        "--pool-exclude-self",
        action="store_true",
        help="When pool-testing a group, exclude its own indices from the pooled supergroup.",
    )
    # Advanced perf knob: chunk controls vectorized resampling batch size
    ap.add_argument("--chunk", type=int, default=128, help="Bootstrap chunk size (perf/memory knob).")

    # args = ap.parse_args()
    args, _ = ap.parse_known_args()
    cfg = BootstrapConfig.from_args(args)

    multi_worker = False
    if cfg.parallel_scope in {"windows", "both"} and cfg.jobs and cfg.jobs > 1:
        multi_worker = True
    if cfg.parallel_scope in {"regions", "both"} and cfg.region_jobs and cfg.region_jobs > 1:
        multi_worker = True
    _blas_threads = cfg.blas_threads if cfg.blas_threads is not None else (1 if multi_worker else None)
    limit_blas_threads(_blas_threads)

    # Load quantiles and pairs
    q_list = cfg.q_list
    pairs = cfg.pairs_list

    # Load context/data
    data = get_context(tr=cfg.tr)
    region_dirs, groups_map, outdir, q_path, d_path, c_path = resolve_paths_and_groups(
        cfg, data
    ) # region_dirs: list of Path, groups_map: dict, outdir: Path, q_path: Path, d_path: diffs_path, c_path: correaltion Path

    # --------- Load-cache behavior --------- #
    # Load-cache behavior: skip recomputation if outputs exist
    if cfg.load_cache and q_path.exists() and d_path.exists() and (
        not cfg.correlate_nor or c_path.exists()
    ):
        print(
            "Found existing outputs (load-cache enabled):\n"
            f"  {q_path}\n  {d_path}"
            + (f"\n  {c_path}" if cfg.correlate_nor else "")
        )
        return 0

    # Stage 2: per-region processing
    boots_dtype = cfg.boots_dtype
    values_dtype = cfg.values_dtype
    index_dtype = cfg.index_dtype
    quantiles_rows: list[dict[str, object]] = []
    diffs_rows: list[dict[str, object]] = []
    corr_rows: list[dict[str, object]] = []

    #
    pooltest_rows: list[dict[str, object]] = []

    def _process_region_dir(region_dir: Path) -> tuple[
        list[dict[str, object]],
        list[dict[str, object]],
        list[dict[str, object]],
        list[dict[str, object]],
    ]:
        print(region_dir)
        folder_label = (
            region_dir.name.replace("regions-", "")
            if region_dir.name.startswith("regions-")
            else region_dir.name
        )
        return process_region_dir(
            region_dir,
            folder_label,
            cfg,
            groups_map,
            q_list,
            boots_dtype,
            values_dtype,
            index_dtype,
            data,
            pairs,
        )

    if (
        cfg.parallel_scope in {"regions", "both"}
        and cfg.region_jobs
        and cfg.region_jobs > 1
    ):
        print(
            f"Processing {len(region_dirs)} regions in parallel with {cfg.region_jobs} jobs..."
        )
        results = Parallel(n_jobs=cfg.region_jobs, prefer="processes")(
            delayed(_process_region_dir)(region_dir) for region_dir in region_dirs
        )
        for rq, rd, rc, rp in results:
            quantiles_rows.extend(rq)
            diffs_rows.extend(rd)
            corr_rows.extend(rc)
            pooltest_rows.extend(rp)
    else:
        for region_dir in maybe_tqdm(cfg.progress, region_dirs, "Regions"):
            rq, rd, rc, rp = _process_region_dir(region_dir)
            quantiles_rows.extend(rq)
            diffs_rows.extend(rd)
            corr_rows.extend(rc)
            pooltest_rows.extend(rp)

    # Write CSVs consistently (also write suffixed copies with n_boot)
    write_outputs(quantiles_rows, diffs_rows, corr_rows, outdir, cfg.n_boot, p_rows=pooltest_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# %%
