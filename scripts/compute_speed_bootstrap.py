#!/usr/bin/env python3
"""
Compute dFC speed bootstrap tables (CSV only).

Reads per-window NPZ speed files, bootstraps per-group percentiles and per-pair
percentile differences (with empirical two-sided p-values), optionally pools
windows and correlates pooled per-animal percentiles with a NOR score.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable
import csv
from dataclasses import dataclass, field
import os
from pathlib import Path
import re


import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Centralized kernels (prefer shared_code.*; fallback to shared_code.shared_code.*)
try:  # preferred: shared_code.*
    from shared_code.fun_bootstrap import (
        bootstrap_percentiles as _central_bootstrap_percentiles,
        bootstrap_diff_percentiles as _central_bootstrap_diff_percentiles,
        bootstrap_groups_percentiles as _central_bootstrap_groups_percentiles,
        bootstrap_groups_boots as _central_bootstrap_groups_boots,
        ci_from_boots as _central_ci_from_boots,
        pool_per_animal as _central_pool_per_animal,
    )
except Exception:  # pragma: no cover
    try:  # fallback: nested package path
        from shared_code.shared_code.fun_bootstrap import (
            bootstrap_percentiles as _central_bootstrap_percentiles,
            bootstrap_diff_percentiles as _central_bootstrap_diff_percentiles,
            bootstrap_groups_percentiles as _central_bootstrap_groups_percentiles,
            bootstrap_groups_boots as _central_bootstrap_groups_boots,
            ci_from_boots as _central_ci_from_boots,
            pool_per_animal as _central_pool_per_animal,
        )
    except Exception:  # final: disable central kernels
        _central_bootstrap_percentiles = None
        _central_bootstrap_diff_percentiles = None
        _central_bootstrap_groups_percentiles = None
        _central_bootstrap_groups_boots = None
        _central_ci_from_boots = None
        _central_pool_per_animal = None


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


def load_per_animal_from_npz(npz_path: Path, tau_index: int | None = None) -> list[np.ndarray]:
    z = np.load(npz_path, allow_pickle=True)
    if "speeds" not in z:
        raise KeyError(f"NPZ file missing 'speeds' key: {npz_path}")
    speeds = z["speeds"]  # object array len=n_animals; each is 2D (n_tau, T_w)
    out: list[np.ndarray] = []
    for a in range(len(speeds)):
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


# ---------------- Local bootstrap implementations (fallbacks) ---------------- #


def bootstrap_quantiles_1d(
    x: np.ndarray,
    q: Iterable[float],
    n_boot: int,
    ci: float,
    seed: int,
    chunk: int = 128,
    early_stop: float = 0.0,
    dtype: type | np.dtype = float,
    val_dtype: type | np.dtype | None = None,
    index_dtype: type | np.dtype | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if _central_bootstrap_percentiles is not None:
        return _central_bootstrap_percentiles(
            x,
            q=q,
            n_boot=n_boot,
            ci=ci,
            seed=seed,
            chunk=chunk,
            early_stop=early_stop,
            dtype=np.dtype(dtype),
            val_dtype=(None if val_dtype is None else np.dtype(val_dtype)),
            index_dtype=(None if index_dtype is None else np.dtype(index_dtype)),
        )
    x = np.asarray(x, val_dtype or float)
    x = x[~np.isnan(x)]
    q_arr = np.asarray(list(q), float)
    if x.size == 0:
        nan = np.full_like(q_arr, np.nan, float)
        return nan, nan, nan
    point = np.percentile(x, q_arr)
    rng = np.random.default_rng(seed)
    n = x.size
    boots = np.empty((n_boot, q_arr.size), dtype)
    done = 0
    chunk = max(1, int(chunk))
    check_every = max(1, int(0.1 * n_boot))
    last_lo = None
    last_hi = None
    while done < n_boot:
        m = min(chunk, n_boot - done)
        if index_dtype is not None:
            idx = rng.integers(0, n, size=(m, n), endpoint=False, dtype=index_dtype)
        else:
            idx = rng.integers(0, n, size=(m, n), endpoint=False)
        xb = x[idx]
        boots[done : done + m, :] = np.percentile(xb, q_arr, axis=1).T
        done += m
        if early_stop and (done % check_every == 0 or done == n_boot):
            alpha_tmp = (100.0 - float(ci)) / 2.0
            lo_t = np.percentile(boots[:done], alpha_tmp, axis=0)
            hi_t = np.percentile(boots[:done], 100.0 - alpha_tmp, axis=0)
            if (
                last_lo is not None
                and last_hi is not None
                and not (np.any(np.isnan(last_lo)) or np.any(np.isnan(last_hi)))
            ):
                rel_lo = np.max(np.abs(lo_t - last_lo) / (np.abs(last_lo) + 1e-12))
                rel_hi = np.max(np.abs(hi_t - last_hi) / (np.abs(last_hi) + 1e-12))
                if rel_lo <= early_stop and rel_hi <= early_stop:
                    return point, lo_t, hi_t
            last_lo = lo_t
            last_hi = hi_t
    alpha = (100.0 - float(ci)) / 2.0
    lo = np.percentile(boots, alpha, axis=0)
    hi = np.percentile(boots, 100.0 - alpha, axis=0)
    return point, lo, hi


def bootstrap_quantiles_by_group(
    per_animal: list[np.ndarray],
    groups: dict,
    q: Iterable[float],
    n_boot: int,
    ci: float,
    seed: int,
    early_stop: float = 0.0,
    chunk: int = 128,
    dtype: type | np.dtype = float,
    val_dtype: type | np.dtype | None = None,
    index_dtype: type | np.dtype | None = None,
) -> dict:
    if _central_bootstrap_groups_percentiles is not None:
        return _central_bootstrap_groups_percentiles(
            per_animal,
            groups,
            q=q,
            n_boot=n_boot,
            ci=ci,
            seed=seed,
            early_stop=early_stop,
            chunk=chunk,
            dtype=np.dtype(dtype),
            val_dtype=(None if val_dtype is None else np.dtype(val_dtype)),
            index_dtype=(None if index_dtype is None else np.dtype(index_dtype)),
        )
    out: dict = {}
    for g, idxs in groups.items():
        vals = [per_animal[int(i)] for i in idxs if int(i) < len(per_animal)]
        pooled = (
            np.concatenate([v for v in vals if getattr(v, "size", 0) > 0])
            if vals
            else np.array([])
        )
        point, lo, hi = bootstrap_quantiles_1d(
            pooled,
            q=q,
            n_boot=n_boot,
            ci=ci,
            seed=seed,
            early_stop=early_stop,
            chunk=chunk,
            dtype=dtype,
            val_dtype=val_dtype,
            index_dtype=index_dtype,
        )
        out[g] = {
            "q": np.asarray(list(q), float),
            "point": point,
            "lo": lo,
            "hi": hi,
            "n": int(pooled.size),
        }
    return out


def pooled_from_indices(
    per_animal: list[np.ndarray], idxs: Iterable[int]
) -> np.ndarray:
    if _central_pool_per_animal is not None:
        return _central_pool_per_animal(per_animal, idxs)
    vals = [per_animal[int(i)] for i in idxs if int(i) < len(per_animal)]
    nonempty = [v for v in vals if getattr(v, "size", 0) > 0]
    return np.concatenate(nonempty) if nonempty else np.array([])


def bootstrap_quantile_diffs(
    x: np.ndarray,
    y: np.ndarray,
    q: Iterable[float],
    n_boot: int,
    ci: float,
    seed: int,
    chunk: int = 128,
    early_stop: float = 0.0,
    dtype: type | np.dtype = float,
    val_dtype: type | np.dtype | None = None,
    index_dtype: type | np.dtype | None = None,
) -> dict:
    if _central_bootstrap_diff_percentiles is not None:
        return _central_bootstrap_diff_percentiles(
            x,
            y,
            q=q,
            n_boot=n_boot,
            ci=ci,
            seed=seed,
            chunk=chunk,
            early_stop=early_stop,
            dtype=np.dtype(dtype),
            val_dtype=(None if val_dtype is None else np.dtype(val_dtype)),
            index_dtype=(None if index_dtype is None else np.dtype(index_dtype)),
        )
    x = np.asarray(x, val_dtype or float)
    y = np.asarray(y, val_dtype or float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    q_arr = np.asarray(list(q), float)
    if x.size == 0 or y.size == 0:
        m = q_arr.size
        nan = np.full(m, np.nan)
        return {
            "q": q_arr,
            "point": nan,
            "lo": nan,
            "hi": nan,
            "sig": np.zeros(m, bool),
            "n_x": int(x.size),
            "n_y": int(y.size),
        }
    point = np.percentile(x, q_arr) - np.percentile(y, q_arr)
    rng = np.random.default_rng(seed)
    nx, ny = x.size, y.size
    boots = np.empty((n_boot, q_arr.size), dtype)
    done = 0
    chunk = max(1, int(chunk))
    check_every = max(1, int(0.1 * n_boot))
    last_lo = None
    last_hi = None
    while done < n_boot:
        m = min(chunk, n_boot - done)
        if index_dtype is not None:
            idx_x = rng.integers(0, nx, size=(m, nx), endpoint=False, dtype=index_dtype)
            idx_y = rng.integers(0, ny, size=(m, ny), endpoint=False, dtype=index_dtype)
        else:
            idx_x = rng.integers(0, nx, size=(m, nx), endpoint=False)
            idx_y = rng.integers(0, ny, size=(m, ny), endpoint=False)
        xb = x[idx_x]
        yb = y[idx_y]
        boots[done : done + m, :] = (
            np.percentile(xb, q_arr, axis=1) - np.percentile(yb, q_arr, axis=1)
        ).T
        done += m
        if early_stop and (done % check_every == 0 or done == n_boot):
            alpha_tmp = (100.0 - float(ci)) / 2.0
            lo_t = np.percentile(boots[:done], alpha_tmp, axis=0)
            hi_t = np.percentile(boots[:done], 100.0 - alpha_tmp, axis=0)
            if (
                last_lo is not None
                and last_hi is not None
                and not (np.any(np.isnan(last_lo)) or np.any(np.isnan(last_hi)))
            ):
                rel_lo = np.max(np.abs(lo_t - last_lo) / (np.abs(last_lo) + 1e-12))
                rel_hi = np.max(np.abs(hi_t - last_hi) / (np.abs(last_hi) + 1e-12))
                if rel_lo <= early_stop and rel_hi <= early_stop:
                    sig_t = (lo_t > 0) | (hi_t < 0)
                    return {
                        "q": q_arr,
                        "point": point,
                        "lo": lo_t,
                        "hi": hi_t,
                        "sig": sig_t,
                        "n_x": int(nx),
                        "n_y": int(ny),
                    }
            last_lo = lo_t
            last_hi = hi_t
    alpha = (100.0 - float(ci)) / 2.0
    lo = np.percentile(boots, alpha, axis=0)
    hi = np.percentile(boots, 100.0 - alpha, axis=0)
    sig = (lo > 0) | (hi < 0)
    # Empirical two-sided p-values per quantile
    p = np.empty_like(q_arr, dtype=float)
    for j in range(q_arr.size):
        p[j] = (np.count_nonzero(np.abs(boots[:, j]) >= abs(point[j])) + 1.0) / (
            boots.shape[0] + 1.0
        )
    return {
        "q": q_arr,
        "point": point,
        "lo": lo,
        "hi": hi,
        "sig": sig,
        "p": p,
        "n_x": int(nx),
        "n_y": int(ny),
    }


def _bootstrap_diff_p_only_local(
    x: np.ndarray,
    y: np.ndarray,
    q: Iterable[float],
    n_boot: int,
    seed: int,
    chunk: int = 128,
    val_dtype: type | np.dtype | None = None,
    index_dtype: type | np.dtype | None = None,
) -> np.ndarray:
    """Compute empirical two-sided p-values for percentile diffs via local bootstrap."""
    x = np.asarray(x, val_dtype or float)
    y = np.asarray(y, val_dtype or float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    q_arr = np.asarray(list(q), float)
    if x.size == 0 or y.size == 0 or q_arr.size == 0:
        return np.full(q_arr.shape, np.nan, float)
    point = np.percentile(x, q_arr) - np.percentile(y, q_arr)
    rng = np.random.default_rng(seed)
    nx, ny = x.size, y.size
    boots = np.empty((n_boot, q_arr.size), float)
    done = 0
    chunk = max(1, int(chunk))
    while done < n_boot:
        m = min(chunk, n_boot - done)
        if index_dtype is not None:
            idx_x = rng.integers(0, nx, size=(m, nx), endpoint=False, dtype=index_dtype)
            idx_y = rng.integers(0, ny, size=(m, ny), endpoint=False, dtype=index_dtype)
        else:
            idx_x = rng.integers(0, nx, size=(m, nx), endpoint=False)
            idx_y = rng.integers(0, ny, size=(m, ny), endpoint=False)
        xb = x[idx_x]
        yb = y[idx_y]
        boots[done : done + m, :] = (
            np.percentile(xb, q_arr, axis=1) - np.percentile(yb, q_arr, axis=1)
        ).T
        done += m
    p = np.empty_like(q_arr, dtype=float)
    for j in range(q_arr.size):
        p[j] = (np.count_nonzero(np.abs(boots[:, j]) >= abs(point[j])) + 1.0) / (
            boots.shape[0] + 1.0
        )
    return p


def bootstrap_quantile_diffs_by_keys(
    per_animal: list[np.ndarray],
    groups: dict,
    key_a,
    key_b,
    q: Iterable[float],
    n_boot: int,
    ci: float,
    seed: int,
    early_stop: float = 0.0,
    chunk: int = 128,
    dtype: type | np.dtype = float,
    val_dtype: type | np.dtype | None = None,
    index_dtype: type | np.dtype | None = None,
) -> dict:
    xa = pooled_from_indices(per_animal, groups[key_a])
    xb = pooled_from_indices(per_animal, groups[key_b])
    return bootstrap_quantile_diffs(
        xa,
        xb,
        q=q,
        n_boot=n_boot,
        ci=ci,
        seed=seed,
        early_stop=early_stop,
        chunk=chunk,
        dtype=dtype,
        val_dtype=val_dtype,
        index_dtype=index_dtype,
    )


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
    early_stop: float = 0.0
    seed: int = 0
    ci: float = 95.0
    chunk: int = 128
    boots_float32: bool = False
    values_float32: bool = False
    index_int32: bool = False
    pool_threshold: str | None = None
    pool_all: bool = False
    jobs: int = 1
    blas_threads: int | None = None
    parallel_scope: str = "windows"
    append_subset_to_outdir: bool = False
    load_cache: bool = False
    progress: bool = False
    group_cols: str = "genotype,treatment"
    reuse_group_boots: bool = False
    correlate_nor: bool = False
    nor_col: str | None = None
    correlate_nor_by_groups: bool = False

    # Derived
    q_list: list[float] = field(default_factory=list)
    pairs_list: list = field(default_factory=list)
    boots_dtype: type = float
    values_dtype: type = float
    index_dtype: np.dtype | None = None

    @classmethod
    def from_args(cls, args) -> "BootstrapConfig":
        cfg = cls(
            tr=args.tr,
            subset=args.subset,
            outdir=args.outdir,
            tau_index=args.tau_index,
            q=args.q,
            pairs=args.pairs,
            n_boot=args.n_boot,
            early_stop=float(args.early_stop or 0.0),
            seed=args.seed,
            ci=args.ci,
            chunk=args.chunk,
            boots_float32=bool(args.boots_float32),
            values_float32=bool(args.values_float32),
            index_int32=bool(args.index_int32),
            pool_threshold=args.pool_threshold,
            pool_all=bool(args.pool_all),
            jobs=args.jobs,
            blas_threads=args.blas_threads,
            parallel_scope=args.parallel_scope,
            append_subset_to_outdir=bool(args.append_subset_to_outdir),
            load_cache=bool(args.load_cache),
            progress=bool(args.progress),
            group_cols=args.group_cols,
            reuse_group_boots=bool(args.reuse_group_boots),
            correlate_nor=bool(args.correlate_nor),
            nor_col=args.nor_col,
            correlate_nor_by_groups=bool(args.correlate_nor_by_groups),
        )
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
    def build_outdir_name(
        outdir: str | None, subset: str | None, append_subset: bool
    ) -> str:
        name = outdir if outdir else (subset if subset else "bootstrap")
        if append_subset and subset and outdir:
            def _san(s: str) -> str:
                return (
                    str(s)
                    .replace("/", "-")
                    .replace(" ", "_")
                    .replace(",", "-")
                    .replace("|", "-")
                    .replace("(", "")
                    .replace(")", "")
                )
            name = f"{name}__subset-{_san(subset)}"
        return name


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
    outdir_name = BootstrapConfig.build_outdir_name(
        cfg.outdir, cfg.subset, cfg.append_subset_to_outdir
    )
    outdir = outputs_root / outdir_name
    outdir.mkdir(parents=True, exist_ok=True)
    q_path = outdir / "speed_bootstrap_quantiles.csv"
    d_path = outdir / "speed_bootstrap_diffs.csv"
    c_path = outdir / "speed_nor_correlations.csv"
    return region_dirs, groups_map, outdir, q_path, d_path, c_path


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


# ---------------- Processing ---------------- #


P_METHOD_DESC = (
    "empirical two-sided bootstrap on percentile differences with +1/(B+1) smoothing"
)


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
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    quantiles_rows: list[dict[str, object]] = []
    diffs_rows: list[dict[str, object]] = []
    corr_rows: list[dict[str, object]] = []

    win_files = _list_window_files(region_dir)
    if not win_files:
        return quantiles_rows, diffs_rows, corr_rows

    def _process_win(
        win: int,
        npz: Path,
        folder_label: str = folder_label,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
        rows_q: list[dict[str, object]] = []
        rows_d: list[dict[str, object]] = []
        rows_c: list[dict[str, object]] = []
        try:
            roi = _infer_roi_from_filename(npz.name) or folder_label
        except Exception:
            roi = folder_label
        per_animal = load_per_animal_from_npz(
            npz, tau_index=None if cfg.tau_index < 0 else cfg.tau_index
        )
        if (
            cfg.reuse_group_boots
            and _central_bootstrap_groups_boots is not None
            and _central_ci_from_boots is not None
        ):
            boots_map = _central_bootstrap_groups_boots(
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
                pooled_g = pooled_from_indices(per_animal, idxs)
                if pooled_g.size:
                    point = np.percentile(pooled_g, q_arr)
                else:
                    point = np.full_like(q_arr, np.nan, dtype=float)
                boots = boots_map[gk]
                lo, hi = _central_ci_from_boots(boots, ci=cfg.ci)
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
            for A, B in pairs:
                if A not in groups_map or B not in groups_map:
                    continue
                boots_A = boots_map.get(A)
                boots_B = boots_map.get(B)
                if boots_A is None or boots_B is None:
                    continue
                n_used = min(boots_A.shape[0], boots_B.shape[0])
                diff_boots = boots_A[:n_used, :] - boots_B[:n_used, :]
                lo, hi = _central_ci_from_boots(diff_boots, ci=cfg.ci)
                pooled_A = pooled_from_indices(per_animal, groups_map[A])
                pooled_B = pooled_from_indices(per_animal, groups_map[B])
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
        else:
            qa = bootstrap_quantiles_by_group(
                per_animal,
                groups_map,
                q=q_list,
                n_boot=cfg.n_boot,
                ci=cfg.ci,
                seed=cfg.seed,
                early_stop=cfg.early_stop,
                chunk=cfg.chunk,
                dtype=np.dtype(boots_dtype),
                val_dtype=(None if values_dtype is None else np.dtype(values_dtype)),
                index_dtype=(None if index_dtype is None else np.dtype(index_dtype)),
            )
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
            for A, B in pairs:
                if A not in groups_map or B not in groups_map:
                    continue
                qd = bootstrap_quantile_diffs_by_keys(
                    per_animal,
                    groups_map,
                    A,
                    B,
                    q=q_list,
                    n_boot=cfg.n_boot,
                    ci=cfg.ci,
                    seed=cfg.seed,
                    early_stop=cfg.early_stop,
                    chunk=cfg.chunk,
                    dtype=np.dtype(boots_dtype),
                    val_dtype=(None if values_dtype is None else np.dtype(values_dtype)),
                    index_dtype=(None if index_dtype is None else np.dtype(index_dtype)),
                )
                p_arr = qd.get("p")
                if p_arr is None or (isinstance(p_arr, float) and np.isnan(p_arr)):
                    xa = pooled_from_indices(per_animal, groups_map[A])
                    xb = pooled_from_indices(per_animal, groups_map[B])
                    p_arr = _bootstrap_diff_p_only_local(
                        xa,
                        xb,
                        q_list,
                        cfg.n_boot,
                        cfg.seed,
                        chunk=cfg.chunk,
                        val_dtype=values_dtype,
                        index_dtype=index_dtype,
                    )
                p_arr = np.asarray(p_arr, float).ravel()
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
        return rows_q, rows_d, rows_c

    # Per-window
    if cfg.jobs and cfg.jobs > 1 and cfg.parallel_scope == "windows":
        results = Parallel(n_jobs=cfg.jobs, prefer="processes")(
            delayed(_process_win)(w, p) for (w, p) in win_files
        )
        if results:
            for rq, rd, rc in results:
                quantiles_rows.extend(rq)
                diffs_rows.extend(rd)
                corr_rows.extend(rc)
    else:
        for w, p in maybe_tqdm(cfg.progress, win_files, f"{folder_label} windows"):
            rq, rd, rc = _process_win(w, p)
            quantiles_rows.extend(rq)
            diffs_rows.extend(rd)
            corr_rows.extend(rc)

    # Pools
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
            if (
                cfg.reuse_group_boots
                and _central_bootstrap_groups_boots is not None
                and _central_ci_from_boots is not None
            ):
                boots_map = _central_bootstrap_groups_boots(
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
                    pooled_g = pooled_from_indices(pooled, idxs)
                    boots = boots_map[gk]
                    lo, hi = _central_ci_from_boots(boots, ci=cfg.ci)
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
                    lo, hi = _central_ci_from_boots(diff_boots, ci=cfg.ci)
                    pooled_A = pooled_from_indices(pooled, groups_map[A])
                    pooled_B = pooled_from_indices(pooled, groups_map[B])
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
            else:
                qa = bootstrap_quantiles_by_group(
                    pooled,
                    groups_map,
                    q=q_list,
                    n_boot=cfg.n_boot,
                    ci=cfg.ci,
                    seed=cfg.seed,
                    early_stop=cfg.early_stop,
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
                    qd = bootstrap_quantile_diffs_by_keys(
                        pooled,
                        groups_map,
                        A,
                        B,
                        q=q_list,
                        n_boot=cfg.n_boot,
                        ci=cfg.ci,
                        seed=cfg.seed,
                        early_stop=cfg.early_stop,
                        chunk=cfg.chunk,
                        dtype=np.dtype(boots_dtype),
                        val_dtype=(None if values_dtype is None else np.dtype(values_dtype)),
                        index_dtype=(None if index_dtype is None else np.dtype(index_dtype)),
                    )
                    p_arr = qd.get("p")
                    if p_arr is None or (isinstance(p_arr, float) and np.isnan(p_arr)):
                        xa = pooled_from_indices(pooled, groups_map[A])
                        xb = pooled_from_indices(pooled, groups_map[B])
                        p_arr = _bootstrap_diff_p_only_local(
                            xa,
                            xb,
                            q_list,
                            cfg.n_boot,
                            cfg.seed,
                            chunk=cfg.chunk,
                            val_dtype=values_dtype,
                            index_dtype=index_dtype,
                        )
                    p_arr = np.asarray(p_arr, float).ravel()
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
    return quantiles_rows, diffs_rows, corr_rows


# ---------------- CLI ---------------- #


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
    ap.add_argument(
        "--early-stop",
        type=float,
        default=0.0,
        help="Relative tol for adaptive CI stability (0 disables).",
    )
    ap.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    ap.add_argument("--ci", type=float, default=95.0, help="Confidence level for CIs (percent).")
    ap.add_argument("--chunk", type=int, default=128, help="Bootstrap chunk size.")
    ap.add_argument("--boots-float32", action="store_true", help="Use float32 boots.")
    ap.add_argument("--values-float32", action="store_true", help="Cast values to float32 before resampling.")
    ap.add_argument("--index-int32", action="store_true", help="Use int32 index arrays for resampling.")
    ap.add_argument(
        "--pool-threshold",
        type=str,
        default=None,
        help="Pool windows into short/long by 'median' or integer cutoff.",
    )
    ap.add_argument("--pool-all", action="store_true", help="Also add an 'all' pool combining all windows.")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel jobs for per-window processing (1 = serial).")
    ap.add_argument(
        "--blas-threads",
        type=int,
        default=None,
        help="Limit BLAS threads per process (default 1 when --jobs > 1).",
    )
    ap.add_argument("--parallel-scope", type=str, default="windows", help="Parallelization scope.")
    ap.add_argument(
        "--append-subset-to-outdir",
        action="store_true",
        help="Append '__subset-<subset>' suffix to outdir name.",
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

    args = ap.parse_args()
    cfg = BootstrapConfig.from_args(args)

    _blas_threads = (
        cfg.blas_threads
        if cfg.blas_threads is not None
        else (1 if (cfg.jobs and cfg.jobs > 1) else None)
    )
    limit_blas_threads(_blas_threads)

    q_list = cfg.q_list
    pairs = cfg.pairs_list

    # Load context/data
    data = get_context(tr=cfg.tr)
    region_dirs, groups_map, outdir, q_path, d_path, c_path = resolve_paths_and_groups(
        cfg, data
    )

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

    for region_dir in maybe_tqdm(cfg.progress, region_dirs, "Regions"):
        folder_label = (
            region_dir.name.replace("regions-", "")
            if region_dir.name.startswith("regions-")
            else region_dir.name
        )
        rq, rd, rc = process_region_dir(
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
        quantiles_rows.extend(rq)
        diffs_rows.extend(rd)
        corr_rows.extend(rc)

    # Write CSVs consistently
    write_outputs(quantiles_rows, diffs_rows, corr_rows, outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
