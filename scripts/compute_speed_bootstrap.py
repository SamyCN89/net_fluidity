#!/usr/bin/env python3
"""
Compute dFC speed bootstrap tables (CSV only), no plotting.

Standalone implementation that reads per-window NPZ speed files written by the
speed compute step, bootstraps per-group percentiles and per-pair percentile
differences, and writes two CSVs under paths['speed']/<outdir>/.

Outputs
- speed_bootstrap_quantiles.csv: region, roi, window, group, q, point, lo, hi, n
- speed_bootstrap_diffs.csv:     region, roi, window, A, B, q, diff, lo, hi, significant, n_a, n_b
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable
import csv
from pathlib import Path
import re
import sys
import os

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

# Ensure local packages are importable when running from repo without installation.
# - Prefer `src/net_fluidity_julien` (official context)
# - Fallback to `julien_data` module path for legacy DFCAnalysis
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[1]
_SRC_DIR = _REPO_ROOT / "src"
_JULIEN_DIR = _REPO_ROOT / "julien_data"
_SHARED_DIR = _REPO_ROOT / "shared_code"
for _p in (str(_SRC_DIR), str(_SHARED_DIR), str(_JULIEN_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------- Central kernels (preferred) ---------------- #
try:
    from shared_code.shared_code.fun_bootstrap import (
        bootstrap_percentiles as _central_bootstrap_percentiles,
        bootstrap_diff_percentiles as _central_bootstrap_diff_percentiles,
        bootstrap_groups_percentiles as _central_bootstrap_groups_percentiles,
        pool_per_animal as _central_pool_per_animal,
        bootstrap_groups_boots as _central_bootstrap_groups_boots,
        ci_from_boots as _central_ci_from_boots,
    )
except Exception:
    _central_bootstrap_percentiles = None
    _central_bootstrap_diff_percentiles = None
    _central_bootstrap_groups_percentiles = None
    _central_pool_per_animal = None
    _central_bootstrap_groups_boots = None
    _central_ci_from_boots = None

# ---------------- Standalone helpers (no import from scripts.speed_bootstrap_nb) ---------------- #


def get_context(tr: int | None = None):
    """Return a dataset context with paths and metadata loaded.

    Uses net_fluidity_julien.context.DFCAnalysis (preferred), with legacy fallback.
    """
    try:
        from net_fluidity_julien.context import DFCAnalysis
    except ModuleNotFoundError:
        try:
            # legacy (within julien_data)
            from julien_data.class_dataanalysis_julien import (
                DFCAnalysis,  # type: ignore
            )
        except ModuleNotFoundError:
            from class_dataanalysis_julien import DFCAnalysis  # type: ignore

    data = DFCAnalysis()
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
    npz_path: Path, tau_index: int | None = None
) -> list[np.ndarray]:
    """Load per-animal speed arrays from a per-window NPZ (key: 'speeds').
    Returns list of 1D arrays (per animal), pooled over taus if tau_index=None or -1.
    """
    z = np.load(npz_path, allow_pickle=True)
    if "speeds" not in z:
        raise KeyError(f"NPZ file missing 'speeds' key: {npz_path}")
    speeds = z["speeds"]  # object array length n_animals; each entry 2D (n_tau, T_w)
    per_animal: list[np.ndarray] = []
    for a in range(len(speeds)):
        arr = np.asarray(speeds[a], float)
        if arr.ndim != 2:
            per_animal.append(np.array([], float))
            continue
        if tau_index is None or int(tau_index) < 0:
            vals = arr[~np.isnan(arr)]
        else:
            if tau_index < 0 or tau_index >= arr.shape[0]:
                vals = np.array([], float)
            else:
                vals = arr[tau_index][~np.isnan(arr[tau_index])]
        per_animal.append(vals)
    return per_animal


def build_groups_from_columns(cog_df: pd.DataFrame, columns: list[str]) -> dict:
    if not isinstance(cog_df, pd.DataFrame):
        raise TypeError("cog_df must be a pandas DataFrame")
    cols = list(columns)
    tmp = cog_df.reset_index(drop=True)
    grp = tmp.groupby(cols).groups
    out: dict = {}
    for k, idx in grp.items():
        out[k if isinstance(k, tuple) else k] = sorted(int(i) for i in idx)
    return out


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
            x, q=q, n_boot=n_boot, ci=ci, seed=seed, chunk=chunk, early_stop=early_stop, dtype=dtype,
            val_dtype=val_dtype, index_dtype=index_dtype
        )
    x = np.asarray(x, val_dtype or float)
    x = x[~np.isnan(x)]
    q_arr = np.asarray(list(q), dtype=float)
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
            if last_lo is not None and last_hi is not None and not (np.any(np.isnan(last_lo)) or np.any(np.isnan(last_hi))):
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
            dtype=dtype,
            val_dtype=val_dtype,
            index_dtype=index_dtype,
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
            x, y, q=q, n_boot=n_boot, ci=ci, seed=seed, chunk=chunk, early_stop=early_stop, dtype=dtype,
            val_dtype=val_dtype, index_dtype=index_dtype
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
            if last_lo is not None and last_hi is not None and not (np.any(np.isnan(last_lo)) or np.any(np.isnan(last_hi))):
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
    return {
        "q": q_arr,
        "point": point,
        "lo": lo,
        "hi": hi,
        "sig": sig,
        "n_x": int(nx),
        "n_y": int(ny),
    }


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
    # 1) 'regions-' prefix
    prefixed = sorted(
        [
            p
            for p in speed_root.iterdir()
            if p.is_dir() and p.name.startswith("regions-")
        ]
    )
    if prefixed:
        return prefixed
    # 2) any subfolder with NPZs
    generic = []
    for p in sorted([x for x in speed_root.iterdir() if x.is_dir()]):
        if list(p.glob("speed_win*_*.npz")):
            generic.append(p)
    if generic:
        return generic
    # 3) fallback to 'all' or root
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
    return {
        "short": [w for w in vals if w <= cut],
        "long": [w for w in vals if w > cut],
    }


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


def main() -> int:
    class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        pass

    ap = argparse.ArgumentParser(
        description=(
            "Compute dFC speed bootstrap tables (CSV only):\n"
            "- Reads per-window NPZ speed files under paths['speed']/[--subset]\n"
            "- Writes speed_bootstrap_quantiles.csv and speed_bootstrap_diffs.csv to [--outdir]\n"
        ),
        formatter_class=HelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Quick exploratory run with reuse + float32 boots\n"
            "  python scripts/compute_speed_bootstrap.py \\\n+              --tr 400 --subset regions400 --tau-index 0 \\\n+              --n-boot 500 --reuse-group-boots --boots-float32 --chunk 256 --progress\n\n"
            "  # Final run, pooled windows (short/long by median)\n"
            "  python scripts/compute_speed_bootstrap.py \\\n+              --tr 400 --subset regions400 --tau-index 0 \\\n+              --n-boot 2000 --pool-threshold median --pool-all --reuse-group-boots --jobs 8 --progress\n\n"
            "  # Single-column groups\n"
            "  python scripts/compute_speed_bootstrap.py --group-cols treatment --pairs '(VEH)-(LCTB92)'\n"
        ),
    )
    ap.add_argument("--tr", type=int, default=500, help="Total TR used to select metadata (e.g., 400 or 500).")
    ap.add_argument("--subset", type=str, default=None, help="Subset subfolder under paths['speed'] (e.g., 'regions400', 'shared').")
    ap.add_argument("--outdir", type=str, default=None, help="Output folder name under paths['speed']; defaults to --subset or 'bootstrap'.")
    ap.add_argument("--tau-index", type=int, default=0, help="Tau index to select; use a negative value to pool across all taus.")
    ap.add_argument("--q", type=str, default="1,5,50,95,99", help="Comma-separated percentiles to report (e.g., '5,50,95').")
    ap.add_argument(
        "--pairs",
        type=str,
        default="(WT,VEH)-(WT,LCTB92);(Dp1Yey,VEH)-(Dp1Yey,LCTB92);(WT,VEH)-(Dp1Yey,VEH);(WT,LCTB92)-(Dp1Yey,LCTB92)",
        help=(
            "Pairs to compare as A-B; semicolon-separated. Format per pair: '(a1,a2,...)-(b1,b2,...)'\n"
            "The tuple arity must match --group-cols (e.g., 2 entries for 'genotype,treatment')."
        ),
    )
    ap.add_argument("--n-boot", type=int, default=2000, help="Bootstrap replicates per unit (per-group and per-pair).")
    ap.add_argument(
        "--early-stop",
        type=float,
        default=0.0,
        help=(
            "Relative tolerance for adaptive CI stability (0.0 disables). "
            "Every ~10%% of draws, compares CI bounds; stops when both low/high bounds "
            "across all quantiles change <= tolerance."
        ),
    )
    ap.add_argument("--seed", type=int, default=0, help="Base RNG seed for reproducible bootstrap resampling.")
    ap.add_argument("--ci", type=float, default=95.0, help="Confidence level for CIs (percent).")
    ap.add_argument("--chunk", type=int, default=128, help="Bootstrap chunk size for vectorized resampling.")
    ap.add_argument("--boots-float32", action="store_true", help="Store bootstrap arrays in float32 to reduce memory/IO.")
    ap.add_argument("--values-float32", action="store_true", help="Cast pooled values to float32 before resampling (reduces batch memory).")
    ap.add_argument("--index-int32", action="store_true", help="Use int32 index arrays for resampling (reduces index memory).")
    ap.add_argument("--pool-threshold", type=str, default=None, help="Pool windows into short/long by 'median' or integer cutoff.")
    ap.add_argument("--pool-all", action="store_true", help="Also add an 'all' pool combining all windows.")
    ap.add_argument("--jobs", type=int, default=1, help="Parallel jobs for per-window processing (1 = serial).")
    ap.add_argument(
        "--blas-threads",
        type=int,
        default=None,
        help=(
            "Limit BLAS threads per process (sets OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS, etc.). "
            "Default: 1 when --jobs > 1; otherwise unchanged."
        ),
    )
    ap.add_argument("--parallel-scope", type=str, default="windows", help="Parallelization scope; currently only 'windows'.")
    ap.add_argument("--append-subset-to-outdir", action="store_true", help="Append '__subset-<subset>' suffix to outdir name.")
    ap.add_argument("--load-cache", action="store_true", help="Reuse existing CSVs if present; skip recompute.")
    ap.add_argument("--progress", action="store_true", help="Show progress bars for regions/windows/pools (requires tqdm).")
    ap.add_argument("--group-cols", type=str, default="genotype,treatment", help="Comma-separated grouping columns (must exist in cognitive data).")
    ap.add_argument("--reuse-group-boots", action="store_true", help="Reuse per-group bootstrap replicates to compute all pairs (fast for many pairs). Disables early-stop for diffs to keep shapes consistent.")
    args = ap.parse_args()

    # Configure BLAS thread limits for multi-process to avoid oversubscription
    def _limit_blas_threads(n: int | None):
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
            # Only set if not already defined by the user
            if os.environ.get(k) is None:
                os.environ[k] = str(n)
        try:
            from threadpoolctl import threadpool_limits  # type: ignore

            threadpool_limits(limits=n)
        except Exception:
            pass

    _blas_threads = (
        args.blas_threads if args.blas_threads is not None else (1 if (args.jobs and args.jobs > 1) else None)
    )
    _limit_blas_threads(_blas_threads)

    q_list = [float(s) for s in args.q.split(",") if s.strip()]
    pairs = _parse_pairs(args.pairs)

    # Load context/data
    data = get_context(tr=args.tr)
    speed_root = Path(data.paths["speed"])  # type: ignore[index]
    if args.subset:
        speed_root = speed_root / args.subset
    region_dirs = _find_region_folders(speed_root)
    groups_map = build_groups_from_columns(
        data.cog_data_filtered,
        [s.strip() for s in args.group_cols.split(",") if s.strip()],
    )

    # Prepare outputs
    outputs_root = Path(data.paths["speed"])  # type: ignore[index]
    outdir_name = (
        args.outdir if args.outdir else (args.subset if args.subset else "bootstrap")
    )
    if args.append_subset_to_outdir and args.subset and args.outdir:

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

        outdir_name = f"{outdir_name}__subset-{_san(args.subset)}"
    outdir = outputs_root / outdir_name
    outdir.mkdir(parents=True, exist_ok=True)
    q_path = outdir / "speed_bootstrap_quantiles.csv"
    d_path = outdir / "speed_bootstrap_diffs.csv"

    if args.load_cache and q_path.exists() and d_path.exists():
        print(f"[cache] Found existing outputs: {q_path} and {d_path}. Skipping.")
        return 0

    quantiles_rows: list[dict[str, object]] = []
    diffs_rows: list[dict[str, object]] = []

    # Progress helper
    def _tqdm(it, desc):
        if args.progress:
            try:
                from tqdm import tqdm

                return tqdm(it, desc=desc)
            except Exception:
                return it
        return it

    # Process per region folder
    boots_dtype = np.float32 if args.boots_float32 else float
    values_dtype = np.float32 if args.values_float32 else float
    index_dtype = np.int32 if args.index_int32 else None
    for region_dir in _tqdm(region_dirs, desc="Regions"):
        folder_label = (
            region_dir.name.replace("regions-", "")
            if region_dir.name.startswith("regions-")
            else region_dir.name
        )
        win_files = _list_window_files(region_dir)
        if not win_files:
            continue

        # Per-window
        def _process_win(
            win: int, npz: Path
        ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
            rows_q: list[dict[str, object]] = []
            rows_d: list[dict[str, object]] = []
            try:
                roi = _infer_roi_from_filename(npz.name) or folder_label
            except Exception:
                roi = folder_label
            per_animal = load_per_animal_from_npz(
                npz, tau_index=None if args.tau_index < 0 else args.tau_index
            )
            if args.reuse_group_boots and _central_bootstrap_groups_boots is not None and _central_ci_from_boots is not None:
                # Reuse group boots for both quantiles and diffs
                boots_map = _central_bootstrap_groups_boots(
                    per_animal, groups_map, q=q_list, n_boot=args.n_boot, seed=args.seed,
                    chunk=args.chunk, dtype=boots_dtype, val_dtype=values_dtype, index_dtype=index_dtype
                )
                q_arr = boots_map.get('__q__', np.asarray(q_list, float))
                # Per-group point and CI
                for gk, idxs in groups_map.items():
                    if gk == '__q__':
                        continue
                    pooled_g = pooled_from_indices(per_animal, idxs)
                    point = np.percentile(pooled_g, q_arr)
                    boots = boots_map[gk]
                    lo, hi = _central_ci_from_boots(boots, ci=args.ci)
                    for qi, pt, lo_i, hi_i in zip(q_arr, point, lo, hi, strict=False):
                        rows_q.append({
                            "region": roi,
                            "roi": roi,
                            "window": int(win),
                            "group": gk,
                            "q": float(qi),
                            "point": float(pt),
                            "lo": float(lo_i),
                            "hi": float(hi_i),
                            "n": int(pooled_g.size),
                        })
                # Pairwise diffs from reused boots
                for A, B in pairs:
                    if A not in groups_map or B not in groups_map:
                        continue
                    boots_A = boots_map.get(A)
                    boots_B = boots_map.get(B)
                    if boots_A is None or boots_B is None:
                        continue
                    n_used = min(boots_A.shape[0], boots_B.shape[0])
                    diff_boots = boots_A[:n_used, :] - boots_B[:n_used, :]
                    lo, hi = _central_ci_from_boots(diff_boots, ci=args.ci)
                    # Points and sizes
                    pooled_A = pooled_from_indices(per_animal, groups_map[A])
                    pooled_B = pooled_from_indices(per_animal, groups_map[B])
                    point = np.percentile(pooled_A, q_arr) - np.percentile(pooled_B, q_arr)
                    sig = (lo > 0) | (hi < 0)
                    for qi, pt, lo_i, hi_i, s in zip(q_arr, point, lo, hi, sig, strict=False):
                        rows_d.append({
                            "region": roi,
                            "roi": roi,
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
            else:
                qa = bootstrap_quantiles_by_group(
                    per_animal,
                    groups_map,
                    q=q_list,
                    n_boot=args.n_boot,
                    ci=args.ci,
                    seed=args.seed,
                    early_stop=float(args.early_stop or 0.0),
                    chunk=args.chunk,
                    dtype=boots_dtype,
                    val_dtype=values_dtype,
                    index_dtype=index_dtype,
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
                for A, B in pairs:
                    if A not in groups_map or B not in groups_map:
                        continue
                    qd = bootstrap_quantile_diffs_by_keys(
                        per_animal,
                        groups_map,
                        A,
                        B,
                        q=q_list,
                        n_boot=args.n_boot,
                        ci=args.ci,
                        seed=args.seed,
                        early_stop=float(args.early_stop or 0.0),
                        chunk=args.chunk,
                        dtype=boots_dtype,
                        val_dtype=values_dtype,
                        index_dtype=index_dtype,
                    )
                    for qi, pt, lo, hi, sig in zip(
                        qd["q"], qd["point"], qd["lo"], qd["hi"], qd["sig"], strict=False
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
                                "significant": bool(sig),
                                "n_a": int(qd.get("n_x", 0)),
                                "n_b": int(qd.get("n_y", 0)),
                            }
                        )
            return rows_q, rows_d

        if args.jobs and args.jobs > 1 and args.parallel_scope == "windows":
            results = Parallel(n_jobs=args.jobs, prefer="processes")(
                delayed(_process_win)(w, p) for (w, p) in win_files
            )
            for rq, rd in results:
                quantiles_rows.extend(rq)
                diffs_rows.extend(rd)
        else:
            for w, p in _tqdm(win_files, desc=f"{folder_label} windows"):
                rq, rd = _process_win(w, p)
                quantiles_rows.extend(rq)
                diffs_rows.extend(rd)

        # Per-ROI pools
        windows = [w for (w, _) in win_files]
        pools = _pool_windows_indices(windows, args.pool_threshold)
        if args.pool_all and windows:
            pools["all"] = windows
        if pools:
            by_win = {w: p for (w, p) in win_files}
            for pool_name, pool_windows in pools.items():
                if not pool_windows:
                    continue
                per_animals = [
                    load_per_animal_from_npz(
                        by_win[w],
                        tau_index=None if args.tau_index < 0 else args.tau_index,
                    )
                    for w in pool_windows
                    if w in by_win
                ]
                pooled = _concat_per_animal(per_animals)
                if args.reuse_group_boots and _central_bootstrap_groups_boots is not None and _central_ci_from_boots is not None:
                    boots_map = _central_bootstrap_groups_boots(
                        pooled, groups_map, q=q_list, n_boot=args.n_boot, seed=args.seed,
                        chunk=args.chunk, dtype=boots_dtype, val_dtype=values_dtype, index_dtype=index_dtype
                    )
                    q_arr = boots_map.get('__q__', np.asarray(q_list, float))
                    for gk, idxs in groups_map.items():
                        if gk == '__q__':
                            continue
                        pooled_g = pooled_from_indices(pooled, idxs)
                        boots = boots_map[gk]
                        lo, hi = _central_ci_from_boots(boots, ci=args.ci)
                        point = np.percentile(pooled_g, q_arr)
                        for qi, pt, lo_i, hi_i in zip(q_arr, point, lo, hi, strict=False):
                            quantiles_rows.append({
                                "region": folder_label,
                                "roi": folder_label,
                                "window": pool_name,
                                "group": gk,
                                "q": float(qi),
                                "point": float(pt),
                                "lo": float(lo_i),
                                "hi": float(hi_i),
                                "n": int(pooled_g.size),
                            })
                    for A, B in pairs:
                        if A not in groups_map or B not in groups_map:
                            continue
                        boots_A = boots_map.get(A)
                        boots_B = boots_map.get(B)
                        if boots_A is None or boots_B is None:
                            continue
                        n_used = min(boots_A.shape[0], boots_B.shape[0])
                        diff_boots = boots_A[:n_used, :] - boots_B[:n_used, :]
                        lo, hi = _central_ci_from_boots(diff_boots, ci=args.ci)
                        pooled_A = pooled_from_indices(pooled, groups_map[A])
                        pooled_B = pooled_from_indices(pooled, groups_map[B])
                        point = np.percentile(pooled_A, q_arr) - np.percentile(pooled_B, q_arr)
                        sig = (lo > 0) | (hi < 0)
                        for qi, pt, lo_i, hi_i, s in zip(q_arr, point, lo, hi, sig, strict=False):
                            diffs_rows.append({
                                "region": folder_label,
                                "roi": folder_label,
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
                    qa = bootstrap_quantiles_by_group(
                        pooled,
                        groups_map,
                        q=q_list,
                        n_boot=args.n_boot,
                        ci=args.ci,
                        seed=args.seed,
                        early_stop=float(args.early_stop or 0.0),
                        chunk=args.chunk,
                        dtype=boots_dtype,
                        val_dtype=values_dtype,
                        index_dtype=index_dtype,
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
                    for A, B in pairs:
                        if A not in groups_map or B not in groups_map:
                            continue
                        qd = bootstrap_quantile_diffs_by_keys(
                            pooled,
                            groups_map,
                            A,
                            B,
                            q=q_list,
                            n_boot=args.n_boot,
                            ci=args.ci,
                            seed=args.seed,
                            early_stop=float(args.early_stop or 0.0),
                            chunk=args.chunk,
                            dtype=boots_dtype,
                            val_dtype=values_dtype,
                            index_dtype=index_dtype,
                        )
                        for qi, pt, lo, hi, sig in zip(
                            qd["q"],
                            qd["point"],
                            qd["lo"],
                            qd["hi"],
                            qd["sig"],
                            strict=False,
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
                                    "significant": bool(sig),
                                    "n_a": int(qd.get("n_x", 0)),
                                    "n_b": int(qd.get("n_y", 0)),
                                }
                            )

    # Write CSVs
    if quantiles_rows:
        q_cols = ["region", "roi", "window", "group", "q", "point", "lo", "hi", "n"]
        with q_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=q_cols)
            w.writeheader()
            w.writerows(quantiles_rows)
    if diffs_rows:
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
            "significant",
            "n_a",
            "n_b",
        ]
        with d_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=d_cols)
            w.writeheader()
            w.writerows(diffs_rows)

    print(f"Wrote: {q_path}")
    print(f"Wrote: {d_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
