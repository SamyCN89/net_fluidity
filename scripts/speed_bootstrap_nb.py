#!/usr/bin/env python3
"""
Notebook‑friendly helpers to play with dFC speed and bootstrap CIs.

This module avoids argparse and provides simple functions to:
- load per‑animal speed arrays from a per‑window NPZ file
- bootstrap per‑animal and overall confidence intervals
- bootstrap per‑group CIs using the dataset groups
- locate the latest per‑window NPZ from a subset folder

Quick start (in a notebook):
  import sys; sys.path.insert(0, 'src')  # so net_fluidity_julien is importable
  from scripts.speed_bootstrap_nb import (
      get_context, find_speed_npz, load_per_animal_from_npz,
      bootstrap_ci_1d, bootstrap_per_animal, bootstrap_overall,
      bootstrap_by_group, bootstrap_from_subset, plot_group_cis,
  )

  ctx = get_context(tr=400)
  res = bootstrap_from_subset(tr=400, subset_name='shared', tau_index=0, n_boot=2000)
  res['overall'], list(res['per_animal'])[:3]
  plot_group_cis(res['by_group'])
"""
#%%
from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable, Callable

import numpy as np
import matplotlib.pyplot as plt

#%%
def get_context(tr: int | None = None):
    """Return a DFCAnalysis context (package preferred; falls back to legacy)."""
    try:
        from net_fluidity_julien.context import DFCAnalysis
    except ModuleNotFoundError:
        try:
            from julien_data.class_dataanalysis_julien import DFCAnalysis
        except ModuleNotFoundError:
            # Last resort: import from local julien_data directory
            import sys as _sys
            here = Path(__file__).resolve()
            julien_dir = here.parents[1] / 'julien_data'
            if str(julien_dir) not in _sys.path:
                _sys.path.insert(0, str(julien_dir))
            from class_dataanalysis_julien import DFCAnalysis  # type: ignore

    data = DFCAnalysis()
    if tr is None:
        data.get_metadata()
    else:
        preproc = Path(data.paths['preprocessed'])  # type: ignore[index]
        cands = sorted(preproc.glob(f"metadata_animals_*_tr_{int(tr)}.pkl"))
        if not cands:
            raise FileNotFoundError(f"No metadata file for tr={tr} under {preproc}")
        data.get_metadata(meta_filename=cands[0].name)
    data.get_ts_preprocessed()
    data.get_cogdata_preprocessed()
    data.get_temporal_parameters()
    return data


def find_speed_npz(save_root: Path, subset_name: str | None, window: int | str = 'last',
                   tau_count: int | None = None, n_animals: int | None = None, regions: int | None = None) -> Path:
    """
    Find a per-window NPZ speeds file. If window='last', pick the last window from context.
    If tau_count/n_animals/regions are provided, they filter the filename pattern.
    """
    base = save_root / subset_name if subset_name else save_root
    if not base.exists():
        raise FileNotFoundError(f"Subset folder not found: {base}")
    # Generic pattern; we'll refine on demand
    glob = "speed_win*_*.npz"
    cands = sorted(base.glob(glob))
    if not cands:
        raise FileNotFoundError(f"No per-window speeds NPZ under {base}")
    if window == 'last':
        return cands[-1]
    else:
        # try to match speed_win{window}_*
        cands2 = sorted(base.glob(f"speed_win{int(window)}_*.npz"))
        if not cands2:
            raise FileNotFoundError(f"No NPZ for window={window} under {base}")
        return cands2[-1]


def load_per_animal_from_npz(npz_path: Path, tau_index: int | None = None) -> list[np.ndarray]:
    """
    Load per-animal speed arrays from a per-window NPZ (key: 'speeds').
    Returns list of 1D arrays (per animal), pooled over taus if tau_index=None.
    """
    z = np.load(npz_path, allow_pickle=True)
    if 'speeds' not in z:
        raise KeyError(f"NPZ file missing 'speeds' key: {npz_path}")
    speeds = z['speeds']  # object array length n_animals; each entry 2D (n_tau, T_w)
    per_animal: list[np.ndarray] = []
    for a in range(len(speeds)):
        arr = np.asarray(speeds[a], float)
        if arr.ndim != 2:
            per_animal.append(np.array([], float))
            continue
        if tau_index is None:
            vals = arr[~np.isnan(arr)]
        else:
            if tau_index < 0 or tau_index >= arr.shape[0]:
                vals = np.array([], float)
            else:
                vals = arr[tau_index][~np.isnan(arr[tau_index])]
        per_animal.append(vals)
    return per_animal


def bootstrap_ci_1d(x: np.ndarray, n_boot: int = 2000, stat: str = 'median', ci: float = 95.0,
                    random_state: int | None = 0) -> tuple[float, float, float]:
    """Basic bootstrap CI for a 1D array x (ignoring NaNs). Returns (est, lo, hi)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    if stat == 'median':
        stat_fn: Callable[[np.ndarray], float] = lambda a: float(np.median(a))
    elif stat == 'mean':
        stat_fn = lambda a: float(np.mean(a))
    elif stat.startswith('q'):
        q = float(stat[1:]) / 100.0
        stat_fn = lambda a: float(np.quantile(a, q))
    else:
        raise ValueError("stat must be 'median', 'mean' or 'qXX'")

    est = stat_fn(x)
    rng = np.random.default_rng(random_state)
    boots = np.empty(n_boot, float)
    for i in range(n_boot):
        idx = rng.choice(x.size, size=x.size, replace=True)
        boots[i] = stat_fn(x[idx])
    alpha = (100.0 - float(ci)) / 2.0
    lo = float(np.percentile(boots, alpha))
    hi = float(np.percentile(boots, 100.0 - alpha))
    return est, lo, hi


def bootstrap_quantiles_1d(x: np.ndarray, q: Iterable[float] = (1, 5, 50, 95, 99),
                           n_boot: int = 2000, ci: float = 95.0,
                           random_state: int | None = 0) -> dict[str, np.ndarray | int]:
    """Bootstrap CIs for multiple percentiles of a 1D array.

    Returns a dict with keys: 'q' (np.ndarray of percentiles), 'point', 'lo', 'hi', and 'n'.
    """
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        q_arr = np.asarray(list(q), dtype=float)
        nan = np.full_like(q_arr, np.nan, dtype=float)
        return {"q": q_arr, "point": nan, "lo": nan, "hi": nan, "n": 0}

    q_arr = np.asarray(list(q), dtype=float)
    point = np.percentile(x, q_arr)
    rng = np.random.default_rng(random_state)
    boots = np.empty((n_boot, q_arr.size), float)
    n = x.size
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boots[i, :] = np.percentile(x[idx], q_arr)
    alpha = (100.0 - float(ci)) / 2.0
    lo = np.percentile(boots, alpha, axis=0)
    hi = np.percentile(boots, 100.0 - alpha, axis=0)
    return {"q": q_arr, "point": point, "lo": lo, "hi": hi, "n": int(n)}


def bootstrap_per_animal(per_animal: list[np.ndarray], n_boot: int = 2000, stat: str = 'median',
                         ci: float = 95.0, seed: int = 0) -> list[tuple[int, float, float, float, int]]:
    """Return a list of (animal_idx, est, lo, hi, n) for each animal."""
    rows = []
    for i, arr in enumerate(per_animal):
        est, lo, hi = bootstrap_ci_1d(arr, n_boot=n_boot, stat=stat, ci=ci, random_state=seed + i)
        rows.append((i, est, lo, hi, int(arr.size)))
    return rows


def bootstrap_overall(per_animal: list[np.ndarray], n_boot: int = 2000, stat: str = 'median',
                      ci: float = 95.0, seed: int = 0) -> tuple[float, float, float, int]:
    """Bootstrap pooled (sample‑weighted) CI across all animals; returns (est, lo, hi, n)."""
    nonempty = [a for a in per_animal if getattr(a, 'size', 0) > 0]
    pooled = np.concatenate(nonempty) if nonempty else np.array([])
    if pooled.size == 0:
        return (np.nan, np.nan, np.nan, 0)
    est, lo, hi = bootstrap_ci_1d(pooled, n_boot=n_boot, stat=stat, ci=ci, random_state=seed + 12345)
    return est, lo, hi, int(pooled.size)


def bootstrap_by_group(per_animal: list[np.ndarray], groups: dict, n_boot: int = 2000, stat: str = 'median',
                       ci: float = 95.0, seed: int = 0) -> dict:
    """Return {group_key: (est, lo, hi, n)} for pooled group values.

    `group_key` can be a scalar (single column) or a tuple (multi-column).
    """
    out = {}
    for g, idxs in groups.items():
        vals = [per_animal[int(i)] for i in idxs if int(i) < len(per_animal)]
        nonempty = [v for v in vals if getattr(v, 'size', 0) > 0]
        pooled = np.concatenate(nonempty) if nonempty else np.array([])
        if pooled.size == 0:
            out[g] = (np.nan, np.nan, np.nan, 0)
        else:
            est, lo, hi = bootstrap_ci_1d(
                pooled, n_boot=n_boot, stat=stat, ci=ci, random_state=seed + (hash(g) % 9973)
            )
            out[g] = (est, lo, hi, int(pooled.size))
    return out


def bootstrap_quantiles_by_group(per_animal: list[np.ndarray], groups: dict,
                                 q: Iterable[float] = (1, 5, 50, 95, 99),
                                 n_boot: int = 2000, ci: float = 95.0, seed: int = 0) -> dict:
    """Return {group_key: {q, point, lo, hi, n}} for pooled group percentiles.

    Each value is a dict matching the output of `bootstrap_quantiles_1d`.
    """
    out: dict = {}
    for g, idxs in groups.items():
        vals = [per_animal[int(i)] for i in idxs if int(i) < len(per_animal)]
        nonempty = [v for v in vals if getattr(v, 'size', 0) > 0]
        pooled = np.concatenate(nonempty) if nonempty else np.array([])
        res = bootstrap_quantiles_1d(
            pooled, q=q, n_boot=n_boot, ci=ci, random_state=seed + (hash(g) % 9973)
        )
        out[g] = res
    return out


def pooled_from_indices(per_animal: list[np.ndarray], idxs: Iterable[int]) -> np.ndarray:
    """Concatenate values for a set of animal indices, skipping empties."""
    vals = [per_animal[int(i)] for i in idxs if int(i) < len(per_animal)]
    nonempty = [v for v in vals if getattr(v, 'size', 0) > 0]
    return np.concatenate(nonempty) if nonempty else np.array([])


def bootstrap_stat_diff(x: np.ndarray, y: np.ndarray, stat: str = 'median', n_boot: int = 2000,
                        ci: float = 95.0, random_state: int | None = 0) -> tuple[float, float, float]:
    """Bootstrap CI for the difference stat(x) - stat(y)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    if stat == 'median':
        f: Callable[[np.ndarray], float] = lambda a: float(np.median(a))
    elif stat == 'mean':
        f = lambda a: float(np.mean(a))
    elif stat.startswith('q'):
        q = float(stat[1:])
        f = lambda a: float(np.percentile(a, q))
    else:
        raise ValueError("stat must be 'median', 'mean' or 'qXX'")

    if x.size == 0 or y.size == 0:
        return (np.nan, np.nan, np.nan)

    est = f(x) - f(y)
    rng = np.random.default_rng(random_state)
    xb = np.empty(n_boot, float); yb = np.empty(n_boot, float)
    nx, ny = x.size, y.size
    for i in range(n_boot):
        xb[i] = f(x[rng.integers(0, nx, nx)])
        yb[i] = f(y[rng.integers(0, ny, ny)])
    diff = xb - yb
    alpha = (100.0 - float(ci)) / 2.0
    lo, hi = np.percentile(diff, [alpha, 100.0 - alpha])
    return float(est), float(lo), float(hi)


def bootstrap_diff_by_keys(per_animal: list[np.ndarray], groups: dict, key_a, key_b,
                           stat: str = 'median', n_boot: int = 2000, ci: float = 95.0,
                           seed: int = 0) -> dict:
    """Compute bootstrap CI for stat difference between two groups (A - B).

    Returns dict: {'est': est, 'lo': lo, 'hi': hi, 'n_a': n_a, 'n_b': n_b}
    """
    idx_a = groups[key_a]
    idx_b = groups[key_b]
    xa = pooled_from_indices(per_animal, idx_a)
    xb = pooled_from_indices(per_animal, idx_b)
    est, lo, hi = bootstrap_stat_diff(xa, xb, stat=stat, n_boot=n_boot, ci=ci, random_state=seed)
    return {'est': est, 'lo': lo, 'hi': hi, 'n_a': int(xa.size), 'n_b': int(xb.size)}


def bootstrap_diff_of_diffs(per_animal: list[np.ndarray], groups: dict,
                            key_a1, key_a0, key_b1, key_b0,
                            stat: str = 'median', n_boot: int = 2000, ci: float = 95.0,
                            seed: int = 0) -> dict:
    """Bootstrap CI for interaction: (A1 - A0) - (B1 - B0).

    Keys refer to entries in `groups` (e.g., (group, treatment) tuples).
    Returns dict with est, lo, hi and component sizes.
    """
    xa1 = pooled_from_indices(per_animal, groups[key_a1]); xa0 = pooled_from_indices(per_animal, groups[key_a0])
    xb1 = pooled_from_indices(per_animal, groups[key_b1]); xb0 = pooled_from_indices(per_animal, groups[key_b0])

    def stat_fn(a: np.ndarray) -> float:
        if stat == 'median':
            return float(np.median(a))
        if stat == 'mean':
            return float(np.mean(a))
        if stat.startswith('q'):
            return float(np.percentile(a, float(stat[1:])))
        raise ValueError("stat must be 'median', 'mean' or 'qXX'")

    if min(xa1.size, xa0.size, xb1.size, xb0.size) == 0:
        return {'est': np.nan, 'lo': np.nan, 'hi': np.nan,
                'n_a1': int(xa1.size), 'n_a0': int(xa0.size), 'n_b1': int(xb1.size), 'n_b0': int(xb0.size)}

    est = (stat_fn(xa1) - stat_fn(xa0)) - (stat_fn(xb1) - stat_fn(xb0))
    rng = np.random.default_rng(seed)
    na1, na0, nb1, nb0 = xa1.size, xa0.size, xb1.size, xb0.size
    diffs = np.empty(n_boot, float)
    for i in range(n_boot):
        a1b = stat_fn(xa1[rng.integers(0, na1, na1)])
        a0b = stat_fn(xa0[rng.integers(0, na0, na0)])
        b1b = stat_fn(xb1[rng.integers(0, nb1, nb1)])
        b0b = stat_fn(xb0[rng.integers(0, nb0, nb0)])
        diffs[i] = (a1b - a0b) - (b1b - b0b)
    alpha = (100.0 - float(ci)) / 2.0
    lo, hi = np.percentile(diffs, [alpha, 100.0 - alpha])
    return {'est': float(est), 'lo': float(lo), 'hi': float(hi),
            'n_a1': int(na1), 'n_a0': int(na0), 'n_b1': int(nb1), 'n_b0': int(nb0)}


def bootstrap_quantile_diffs(x: np.ndarray, y: np.ndarray,
                             q: Iterable[float] = (1, 5, 50, 95, 99),
                             n_boot: int = 2000, ci: float = 95.0,
                             random_state: int | None = 0) -> dict[str, np.ndarray | int]:
    """Bootstrap CI for percentile differences between two samples.

    Returns dict with keys: 'q', 'point', 'lo', 'hi', 'sig', 'n_x', 'n_y'.
    'point' is the difference of sample percentiles: pct(x) - pct(y).
    'sig' is a boolean array where the CI excludes 0.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    q_arr = np.asarray(list(q), dtype=float)
    if x.size == 0 or y.size == 0:
        m = q_arr.size
        nan = np.full(m, np.nan)
        return {"q": q_arr, "point": nan, "lo": nan, "hi": nan, "sig": np.zeros(m, dtype=bool), "n_x": int(x.size), "n_y": int(y.size)}

    px = np.percentile(x, q_arr)
    py = np.percentile(y, q_arr)
    point = px - py

    rng = np.random.default_rng(random_state)
    nx, ny = x.size, y.size
    boots = np.empty((n_boot, q_arr.size), float)
    for i in range(n_boot):
        xb = x[rng.integers(0, nx, nx)]
        yb = y[rng.integers(0, ny, ny)]
        boots[i, :] = np.percentile(xb, q_arr) - np.percentile(yb, q_arr)
    alpha = (100.0 - float(ci)) / 2.0
    lo = np.percentile(boots, alpha, axis=0)
    hi = np.percentile(boots, 100.0 - alpha, axis=0)
    sig = (lo > 0) | (hi < 0)
    return {"q": q_arr, "point": point, "lo": lo, "hi": hi, "sig": sig, "n_x": int(nx), "n_y": int(ny)}


def bootstrap_quantile_diffs_by_keys(per_animal: list[np.ndarray], groups: dict,
                                     key_a, key_b,
                                     q: Iterable[float] = (1, 5, 50, 95, 99),
                                     n_boot: int = 2000, ci: float = 95.0, seed: int = 0) -> dict:
    """Wrapper: percentile difference CIs between two group keys (A - B)."""
    xa = pooled_from_indices(per_animal, groups[key_a])
    xb = pooled_from_indices(per_animal, groups[key_b])
    return bootstrap_quantile_diffs(xa, xb, q=q, n_boot=n_boot, ci=ci, random_state=seed)


def summarize_significant_quantiles(qdiff_res: dict, min_effect: float | None = None) -> list[dict]:
    """Return a list of significant quantiles with direction and effect size.

    Each item: {'q': float, 'diff': float, 'lo': float, 'hi': float, 'direction': 'A>B'|'A<B'}.
    If `min_effect` is provided, also require abs(diff) >= min_effect.
    """
    q = np.asarray(qdiff_res["q"], dtype=float)
    point = np.asarray(qdiff_res["point"], dtype=float)
    lo = np.asarray(qdiff_res["lo"], dtype=float)
    hi = np.asarray(qdiff_res["hi"], dtype=float)
    sig = np.asarray(qdiff_res.get("sig", (lo > 0) | (hi < 0)))
    out = []
    for qi, d, l, h, s in zip(q, point, lo, hi, sig):
        if not bool(s):
            continue
        if min_effect is not None and not (abs(float(d)) >= float(min_effect)):
            continue
        direction = 'A>B' if float(d) > 0 else 'A<B'
        out.append({'q': float(qi), 'diff': float(d), 'lo': float(l), 'hi': float(h), 'direction': direction})
    return out


def significant_quantiles_by_keys(per_animal: list[np.ndarray], groups: dict, key_a, key_b,
                                  q: Iterable[float] = (1, 5, 50, 95, 99),
                                  n_boot: int = 2000, ci: float = 95.0, seed: int = 0,
                                  min_effect: float | None = None) -> list[dict]:
    """Convenience: compute and summarize significant quantile differences for A vs B."""
    res = bootstrap_quantile_diffs_by_keys(per_animal, groups, key_a, key_b, q=q, n_boot=n_boot, ci=ci, seed=seed)
    return summarize_significant_quantiles(res, min_effect=min_effect)


def build_groups_from_columns(cog_df, columns: Iterable[str]) -> dict:
    """Build a group mapping from `cog_df` using the given columns.

    Returns a dict mapping group key (scalar or tuple) to a list of 0-based
    animal indices corresponding to `per_animal` ordering.
    """
    import pandas as pd  # local import to keep notebook friendliness

    if not isinstance(cog_df, pd.DataFrame):  # defensive fallback
        raise TypeError("cog_df must be a pandas DataFrame with group columns")
    cols = list(columns)
    tmp = cog_df.reset_index(drop=True)
    grp = tmp.groupby(cols).groups
    # Ensure indices are plain Python ints and sorted
    out: dict = {}
    for k, idx in grp.items():
        key = k if isinstance(k, tuple) else k
        out[key] = sorted(int(i) for i in idx)
    return out


def bootstrap_from_subset(tr: int, subset_name: str, window: int | str = 'last', tau_index: int | None = None,
                          n_boot: int = 2000, stat: str = 'median', ci: float = 95.0,
                          seed: int = 0, group_cols: Iterable[str] | None = None):
    """
    High‑level: pick a subset + window, load per‑animal values, and bootstrap CIs.
    Returns dict {overall, per_animal, by_group, npz_path}
    """
    data = get_context(tr=tr)
    save_root = Path(data.paths['speed'])  # type: ignore[index]
    tau_count = int(data.tau + 1)
    npz_path = find_speed_npz(save_root, subset_name=subset_name, window=window,
                              tau_count=tau_count, n_animals=data.n_animals, regions=data.regions)
    per_animal = load_per_animal_from_npz(npz_path, tau_index=tau_index)
    rows = bootstrap_per_animal(per_animal, n_boot=n_boot, stat=stat, ci=ci, seed=seed)
    overall = bootstrap_overall(per_animal, n_boot=n_boot, stat=stat, ci=ci, seed=seed)
    groups = data.groups if group_cols is None else build_groups_from_columns(data.cog_data_filtered, group_cols)
    by_group = bootstrap_by_group(per_animal, groups, n_boot=n_boot, stat=stat, ci=ci, seed=seed)
    return {
        'npz_path': npz_path,
        'per_animal': rows,
        'overall': overall,
        'by_group': by_group,
        'groups': groups,
    }


def _format_group_label(g) -> str:
    if isinstance(g, tuple):
        return "-".join(map(str, g))
    return str(g)


def plot_group_cis(group_cis: dict, title: str | None = None):
    """Simple errorbar plot of per‑group CIs from bootstrap_by_group()."""
    labels = [_format_group_label(g) for g in group_cis.keys()]
    vals = np.array([v[0] for v in group_cis.values()], float)
    los = np.array([v[1] for v in group_cis.values()], float)
    his = np.array([v[2] for v in group_cis.values()], float)
    ns  = np.array([v[3] for v in group_cis.values()], int)
    err_lo = vals - los
    err_hi = his - vals
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(x, vals, yerr=[err_lo, err_hi], fmt='o', capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Speed (bootstrap CI)')
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return ax


def _find_percentile_indices(q_arr: np.ndarray, wants: Iterable[float]) -> dict[float, int | None]:
    wants = list(wants)
    idx_map: dict[float, int | None] = {}
    for w in wants:
        # Find exact or closest index
        diffs = np.abs(q_arr - float(w))
        j = int(np.argmin(diffs))
        if np.isclose(q_arr[j], float(w), rtol=0, atol=1e-9):
            idx_map[float(w)] = j
        else:
            # Not present; mark missing
            idx_map[float(w)] = None
    return idx_map


def plot_group_quantiles(group_quants: dict,
                         title: str | None = None,
                         inner: tuple[float, float] = (5.0, 95.0),
                         outer: tuple[float, float] = (1.0, 99.0),
                         show_median_ci: bool = True):
    """Plot per-group percentile spread and median CI.

    - Whiskers show the percentile spread between `inner` (default 5–95).
    - Thin caps show `outer` percentiles (default 1–99) if available.
    - A point marks the median; optional errorbar shows its bootstrap CI.
    """
    groups = list(group_quants.keys())
    labels = [_format_group_label(g) for g in groups]
    x = np.arange(len(groups), dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, g in enumerate(groups):
        res = group_quants[g]
        q = np.asarray(res["q"], dtype=float)
        point = np.asarray(res["point"], dtype=float)
        lo = np.asarray(res["lo"], dtype=float)
        hi = np.asarray(res["hi"], dtype=float)

        idxs = _find_percentile_indices(q, [50.0, inner[0], inner[1], outer[0], outer[1]])
        i50 = idxs[50.0]
        i_lo, i_hi = idxs[inner[0]], idxs[inner[1]]
        o_lo, o_hi = idxs[outer[0]], idxs[outer[1]]

        # Median point
        if i50 is not None:
            y50 = float(point[i50])
            ax.plot([x[i]], [y50], marker='o', color='C0')
            if show_median_ci:
                y50_lo, y50_hi = float(lo[i50]), float(hi[i50])
                err_lo, err_hi = y50 - y50_lo, y50_hi - y50
                if np.isfinite(err_lo) and np.isfinite(err_hi):
                    ax.errorbar([x[i]], [y50], yerr=[[err_lo], [err_hi]], fmt='none', ecolor='C0', capsize=3)

        # Inner whiskers (5–95 by default) using point percentiles
        if i_lo is not None and i_hi is not None:
            y1, y2 = float(point[i_lo]), float(point[i_hi])
            ax.vlines(x[i], y1, y2, colors='C0', linewidth=3)

        # Outer caps (1–99 by default) using point percentiles
        if o_lo is not None and o_hi is not None:
            y1, y2 = float(point[o_lo]), float(point[o_hi])
            ax.vlines(x[i], y1, y2, colors='C0', linewidth=1, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Speed')
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return ax


def plot_quantile_diffs(qdiff_res: dict,
                        title: str | None = None,
                        color: str = 'C1',
                        ax=None):
    """Plot percentile differences with bootstrap CIs, highlighting significance.

    Accepts the dict returned by `bootstrap_quantile_diffs` or
    `bootstrap_quantile_diffs_by_keys`.
    """
    q = np.asarray(qdiff_res["q"], dtype=float)
    point = np.asarray(qdiff_res["point"], dtype=float)
    lo = np.asarray(qdiff_res["lo"], dtype=float)
    hi = np.asarray(qdiff_res["hi"], dtype=float)
    sig = np.asarray(qdiff_res.get("sig", np.zeros_like(point, dtype=bool)))

    yerr_lo = point - lo
    yerr_hi = hi - point

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        created_fig = True
    else:
        fig = ax.figure
    ax.axhline(0.0, color='k', linewidth=1, alpha=0.5)
    # Error bars for all points
    ax.errorbar(q, point, yerr=[yerr_lo, yerr_hi], fmt='none', ecolor=color, alpha=0.7, capsize=3)
    # Overlay markers: filled if significant, open if not
    if sig.any():
        ax.scatter(q[sig], point[sig], s=40, color=color, edgecolor='none', label='significant')
    if (~sig).any():
        ax.scatter(q[~sig], point[~sig], s=40, facecolors='none', edgecolors=color, label='ns')

    ax.set_xlabel('Percentile')
    ax.set_ylabel('Difference (A - B)')
    if title:
        ax.set_title(title)
    ax.legend(loc='best')
    if created_fig:
        fig.tight_layout()
    return ax


def plot_quantiles_for_factors(per_animal: list[np.ndarray], cog_df,
                               factors: tuple[str, str] = ("genotype", "treatment"),
                               q: Iterable[float] = (1, 5, 50, 95, 99),
                               n_boot: int = 2000, ci: float = 95.0,
                               show: bool = True) -> dict[str, any]:
    """Plot percentile summaries for factor A, factor B, and all A×B combinations.

    - factors: tuple (A, B), typically (genotype, treatment).
    - Returns dict with keys 'A', 'B', 'AB' mapping to axes.
    """
    fa, fb = factors
    groups_a = build_groups_from_columns(cog_df, [fa])
    groups_b = build_groups_from_columns(cog_df, [fb])
    groups_ab = build_groups_from_columns(cog_df, [fa, fb])

    qa = bootstrap_quantiles_by_group(per_animal, groups_a, q=q, n_boot=n_boot, ci=ci)
    qb = bootstrap_quantiles_by_group(per_animal, groups_b, q=q, n_boot=n_boot, ci=ci)
    qab = bootstrap_quantiles_by_group(per_animal, groups_ab, q=q, n_boot=n_boot, ci=ci)

    ax_a = plot_group_quantiles(qa, title=f"Percentiles by {fa}")
    if show:
        plt.show()
    ax_b = plot_group_quantiles(qb, title=f"Percentiles by {fb}")
    if show:
        plt.show()
    ax_ab = plot_group_quantiles(qab, title=f"Percentiles by {fa}–{fb}")
    if show:
        plt.show()
    return {"A": ax_a, "B": ax_b, "AB": ax_ab}


def _get_levels(cog_df, column: str) -> list:
    vals = (
        cog_df[column].dropna().unique().tolist()
        if hasattr(cog_df[column], "unique")
        else sorted({v for v in cog_df[column] if v is not None})
    )
    try:
        return sorted(vals)
    except Exception:
        return list(vals)


def plot_pairwise_quantile_diffs_for_factor(per_animal: list[np.ndarray], cog_df,
                                            factor: str,
                                            q: Iterable[float] = (1, 5, 50, 95, 99),
                                            n_boot: int = 2000, ci: float = 95.0,
                                            max_cols: int = 3, figsize=(12, 4)) -> dict:
    """Grid of significance plots for all pairwise comparisons within a factor.

    Returns dict with keys: 'fig', 'axes', 'results' mapping (A, B) -> qdiff dict.
    """
    groups = build_groups_from_columns(cog_df, [factor])
    levels = _get_levels(cog_df, factor)
    pairs = [(levels[i], levels[j]) for i in range(len(levels)) for j in range(i + 1, len(levels))]
    if not pairs:
        raise ValueError(f"Not enough levels in {factor} for pairwise comparisons")

    n = len(pairs)
    cols = min(max_cols, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(figsize[0], figsize[1] * rows), squeeze=False)
    results = {}
    for k, (A, B) in enumerate(pairs):
        r, c = divmod(k, cols)
        qdiff = bootstrap_quantile_diffs_by_keys(per_animal, groups, A, B, q=q, n_boot=n_boot, ci=ci)
        results[(A, B)] = qdiff
        plot_quantile_diffs(qdiff, title=f"{factor}: {A} − {B}", ax=axes[r, c])
    # Hide any unused axes
    for k in range(n, rows * cols):
        r, c = divmod(k, cols)
        axes[r, c].axis('off')
    fig.tight_layout()
    return {"fig": fig, "axes": axes, "results": results}


def plot_within_contrast_quantile_diffs(per_animal: list[np.ndarray], cog_df,
                                        factors: tuple[str, str] = ("genotype", "treatment"),
                                        contrast: str = "treatment",
                                        q: Iterable[float] = (1, 5, 50, 95, 99),
                                        n_boot: int = 2000, ci: float = 95.0,
                                        figsize=(8, 3)) -> dict:
    """Within each level of the other factor, plot contrast A-B quantile diffs.

    Requires exactly two levels for the contrast factor.
    Returns dict with 'fig', 'axes', and 'results' mapping other_level -> qdiff dict.
    """
    fa, fb = factors
    other = fb if contrast == fa else fa
    groups_ab = build_groups_from_columns(cog_df, [fa, fb])
    contrast_levels = _get_levels(cog_df, contrast)
    if len(contrast_levels) != 2:
        raise ValueError(f"Contrast factor '{contrast}' must have exactly two levels")
    A, B = contrast_levels[0], contrast_levels[1]
    other_levels = _get_levels(cog_df, other)

    fig, axes = plt.subplots(len(other_levels), 1, figsize=(figsize[0], figsize[1] * len(other_levels)), squeeze=False)
    results = {}
    for i, ol in enumerate(other_levels):
        if contrast == fa:
            key_a, key_b = (A, ol), (B, ol)
        else:
            key_a, key_b = (ol, A), (ol, B)
        qdiff = bootstrap_quantile_diffs_by_keys(per_animal, groups_ab, key_a, key_b, q=q, n_boot=n_boot, ci=ci)
        results[ol] = qdiff
        plot_quantile_diffs(qdiff, title=f"{contrast}: {A} − {B} | {other}={ol}", ax=axes[i, 0])
    fig.tight_layout()
    return {"fig": fig, "axes": axes, "results": results}


# Notebook-friendly loader for all windows
_WIN_RE = re.compile(r"speed_win(\d+)_.*\.npz$")


def _list_speed_window_files(base: Path) -> list[tuple[int, Path]]:
    files: list[tuple[int, Path]] = []
    for p in sorted(base.glob("speed_win*_*.npz")):
        m = _WIN_RE.search(p.name)
        if m:
            files.append((int(m.group(1)), p))
    return files


def _resolve_region_dir(save_root: Path, subset_name: str | None, region_label: str | None) -> Path:
    base = save_root / subset_name if subset_name else save_root
    if region_label:
        cand = base / f"regions-{region_label}"
        if not cand.exists():
            raise FileNotFoundError(f"Region folder not found: {cand}")
        return cand
    # Fallbacks: 'all' or base itself
    all_dir = base / "all"
    return all_dir if all_dir.exists() else base


def load_all_speeds_nb(
    tr: int = 500,
    subset_name: str | None = None,
    region_label: str | None = None,
    tau_index: int | None = 0,
    group_cols: Iterable[str] = ("genotype", "treatment"),
):
    """
    Notebook-friendly: load per-animal speeds for all windows into a dict payload.

    Returns a dict with keys:
      - windows: list[int]
      - per_animal_by_window: list[list[np.ndarray]]
      - groups: dict (group_key -> list[animal_idx])
      - group_cols: list[str]
      - region_dir: str; subset: str | None; tr: int; tau_index: int | None
      - cog_df: pandas DataFrame with at least the group columns
    """
    data = get_context(tr=tr)
    save_root = Path(data.paths["speed"])  # type: ignore[index]
    region_dir = _resolve_region_dir(save_root, subset_name, region_label)
    win_files = _list_speed_window_files(region_dir)
    if not win_files:
        raise FileNotFoundError(f"No per-window NPZ under {region_dir}")
    ti = None if (tau_index is not None and tau_index < 0) else tau_index
    windows: list[int] = []
    per_animal_by_window: list[list[np.ndarray]] = []
    for w, npz in win_files:
        per_animal = load_per_animal_from_npz(npz, tau_index=ti)
        windows.append(int(w))
        per_animal_by_window.append(per_animal)
    cols = list(group_cols)
    groups = build_groups_from_columns(data.cog_data_filtered, cols)
    cog_df = data.cog_data_filtered.reset_index(drop=True)[cols]
    return {
        "tr": int(tr),
        "subset": subset_name,
        "tau_index": ti,
        "region_dir": str(region_dir),
        "windows": windows,
        "per_animal_by_window": per_animal_by_window,
        "groups": groups,
        "group_cols": cols,
        "cog_df": cog_df,
    }


def _list_region_dirs(base: Path) -> list[Path]:
    return sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("regions-")])


def load_all_speeds_by_region_nb(
    tr: int = 500,
    subset_name: str | None = None,
    tau_index: int | None = 0,
    group_cols: Iterable[str] = ("genotype", "treatment"),
    include_regions: Iterable[str] | None = None,
):
    """
    Notebook-friendly: load per-animal speeds for ALL regions and all windows.

    Returns a dict:
      - tr, subset, tau_index, group_cols
      - groups: dict (group_key -> list[animal_idx])
      - regions: list[str]
      - by_region: { region_label: { windows, per_animal_by_window } }
    """
    data = get_context(tr=tr)
    save_root = Path(data.paths["speed"])  # type: ignore[index]
    base = save_root / subset_name if subset_name else save_root
    region_dirs = _list_region_dirs(base)
    if not region_dirs:
        raise FileNotFoundError(f"No region folders found under {base}; run per-region speed compute.")

    want = set(include_regions) if include_regions else None
    ti = None if (tau_index is not None and tau_index < 0) else tau_index
    cols = list(group_cols)
    groups = build_groups_from_columns(data.cog_data_filtered, cols)

    by_region: dict[str, dict[str, object]] = {}
    region_labels: list[str] = []
    for rdir in region_dirs:
        label = rdir.name.replace("regions-", "")
        if want and label not in want:
            continue
        win_files = _list_speed_window_files(rdir)
        if not win_files:
            continue
        windows: list[int] = []
        per_animal_by_window: list[list[np.ndarray]] = []
        for w, npz in win_files:
            per_animal = load_per_animal_from_npz(npz, tau_index=ti)
            windows.append(int(w))
            per_animal_by_window.append(per_animal)
        by_region[label] = {
            "windows": windows,
            "per_animal_by_window": per_animal_by_window,
        }
        region_labels.append(label)

    return {
        "tr": int(tr),
        "subset": subset_name,
        "tau_index": ti,
        "group_cols": cols,
        "groups": groups,
        "regions": region_labels,
        "by_region": by_region,
    }


# --- Notebook-friendly pooling helpers ---
def pool_windows_indices_nb(windows: Iterable[int], threshold: int | str = "median") -> tuple[list[int], list[int], int]:
    """Return indices for short/long window pools and the cutoff used.

    - threshold: 'median' to split by median, or an integer cutoff.
    Returns (short_idx, long_idx, cutoff).
    """
    ws = list(int(w) for w in windows)
    if not ws:
        return [], [], 0
    if isinstance(threshold, str) and threshold.lower() == "median":
        cut = int(np.median(ws))
    else:
        cut = int(threshold)
    short_idx = [i for i, w in enumerate(ws) if w <= cut]
    long_idx = [i for i, w in enumerate(ws) if w > cut]
    return short_idx, long_idx, cut


def concat_per_animal_nb(per_animal_by_window: list[list[np.ndarray]], idxs: Iterable[int]) -> list[np.ndarray]:
    """Concatenate per-animal arrays across the given window indices.

    per_animal_by_window: list (per window) of list (per animal) of 1D arrays
    """
    idxs = [int(i) for i in idxs]
    if not idxs:
        return []
    n = max((len(per_animal_by_window[i]) for i in idxs), default=0)
    out: list[np.ndarray] = []
    for a in range(n):
        parts = []
        for i in idxs:
            lst = per_animal_by_window[i]
            if a < len(lst) and lst[a].size:
                parts.append(lst[a])
        out.append(np.concatenate(parts) if parts else np.array([], float))
    return out


def pool_short_long_nb(per_animal_by_window: list[list[np.ndarray]], windows: Iterable[int], threshold: int | str = "median") -> dict:
    """Create short/long pooled per-animal lists with the given threshold.

    Returns {'short': list[np.ndarray], 'long': list[np.ndarray], 'cut': int,
             'short_idx': list[int], 'long_idx': list[int]}.
    """
    short_idx, long_idx, cut = pool_windows_indices_nb(windows, threshold)
    pooled_short = concat_per_animal_nb(per_animal_by_window, short_idx)
    pooled_long = concat_per_animal_nb(per_animal_by_window, long_idx)
    return {
        "short": pooled_short,
        "long": pooled_long,
        "cut": int(cut),
        "short_idx": short_idx,
        "long_idx": long_idx,
    }


def compute_pairs_diffs_nb(
    per_animal: list[np.ndarray],
    groups: dict,
    pairs: Iterable[tuple[tuple, tuple]],
    q: Iterable[float] = (1, 5, 50, 95, 99),
    n_boot: int = 2000,
    ci: float = 95.0,
) -> dict:
    """Compute quantile-difference CIs for a list of (A,B) pairs.

    Returns mapping { (A,B): qdiff_dict } using bootstrap_quantile_diffs_by_keys.
    """
    out: dict = {}
    for A, B in pairs:
        if A not in groups or B not in groups:
            continue
        out[(A, B)] = bootstrap_quantile_diffs_by_keys(per_animal, groups, A, B, q=q, n_boot=n_boot, ci=ci)
    return out


def plot_pairs_grid_nb(
    pairs_qd: dict,
    pairs_order: Iterable[tuple[tuple, tuple]],
    title: str | None = None,
    cols: int = 2,
):
    """Plot a grid of quantile-difference panels for the given pairs.

    Uses plot_quantile_diffs under the hood. Returns (fig, axes).
    """
    pairs_order = list(pairs_order)
    n = len(pairs_order)
    cols = max(1, int(cols))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.5 * rows), squeeze=False)
    for k, pair in enumerate(pairs_order):
        r, c = divmod(k, cols)
        qd = pairs_qd.get(pair)
        if qd is None:
            axes[r, c].axis("off")
            continue
        A, B = pair
        plot_quantile_diffs(qd, title=f"{A}-{B}", ax=axes[r, c])
    for k in range(n, rows * cols):
        r, c = divmod(k, cols)
        axes[r, c].axis("off")
    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()
    return fig, axes


def _match_value(candidates: list, pattern: str):
    """Best-effort matching: exact (case-insensitive) > startswith > contains.

    Returns the matched candidate or raises ValueError if ambiguous or not found.
    """
    patt = str(pattern).lower()
    by_exact = [c for c in candidates if str(c).lower() == patt]
    if len(by_exact) == 1:
        return by_exact[0]
    if len(by_exact) > 1:
        raise ValueError(f"Ambiguous match for pattern '{pattern}': {by_exact}")
    by_sw = [c for c in candidates if str(c).lower().startswith(patt)]
    if len(by_sw) == 1:
        return by_sw[0]
    if len(by_sw) > 1:
        raise ValueError(f"Ambiguous startswith match for pattern '{pattern}': {by_sw}")
    by_contains = [c for c in candidates if patt in str(c).lower()]
    if len(by_contains) == 1:
        return by_contains[0]
    if len(by_contains) > 1:
        raise ValueError(f"Ambiguous contains match for pattern '{pattern}': {by_contains}")
    raise ValueError(f"No match for pattern '{pattern}' in {candidates}")


def find_combo_key(groups_ab: dict, geno_like: str, treat_like: str):
    """Find a (genotype, treatment) key in groups by fuzzy patterns.

    Prefers exact (case-insensitive), then startswith, then contains.
    """
    # Collect unique genotype and treatment levels from keys
    genos = sorted({g for (g, t) in groups_ab.keys()})
    treats = sorted({t for (g, t) in groups_ab.keys()})
    g = _match_value(list(genos), geno_like)
    t = _match_value(list(treats), treat_like)
    key = (g, t)
    if key not in groups_ab:
        raise KeyError(f"Matched ({g}, {t}) not in available combo groups")
    return key


def compare_combos_significance(per_animal: list[np.ndarray], cog_df,
                                geno_a: str, treat_a: str,
                                geno_b: str, treat_b: str,
                                q: Iterable[float] = (1, 5, 50, 95, 99),
                                n_boot: int = 2000, ci: float = 95.0,
                                min_effect: float | None = None,
                                show: bool = True) -> dict:
    """Compare two (genotype, treatment) combos: quantile diffs, significance, and plot.

    Returns dict with keys: 'A_key', 'B_key', 'qdiff', 'significant', and 'ax' (if plotted).
    """
    groups_ab = build_groups_from_columns(cog_df, ["genotype", "treatment"])
    A_key = find_combo_key(groups_ab, geno_a, treat_a)
    B_key = find_combo_key(groups_ab, geno_b, treat_b)
    qdiff = bootstrap_quantile_diffs_by_keys(per_animal, groups_ab, A_key, B_key, q=q, n_boot=n_boot, ci=ci)
    significant = summarize_significant_quantiles(qdiff, min_effect=min_effect)
    ax = None
    if show:
        ax = plot_quantile_diffs(qdiff, title=f"Quantile diffs ({A_key} - {B_key})")
        plt.show()
    return {"A_key": A_key, "B_key": B_key, "qdiff": qdiff, "significant": significant, "ax": ax}


if __name__ == "__main__":
    # Lightweight demo guarded to avoid execution on import
    ctx = get_context(tr=500)
    save_root = Path(ctx.paths["speed"])  # type: ignore[index]
    # Example: pick a window file and compute per‑group CIs for genotype+treatment
    npz_path = find_speed_npz(save_root, subset_name="shared", window="last")
    per_animal = load_per_animal_from_npz(npz_path, tau_index=0)
    res = bootstrap_by_group(
        per_animal,
        build_groups_from_columns(ctx.cog_data_filtered, ["genotype", "treatment"]),
        n_boot=2000,
    )
    ax = plot_group_cis(res, title="Speed by group (genotype-treatment)")
    plt.show()

    # Quantile plot demo
    q_res = bootstrap_quantiles_by_group(
        per_animal,
        build_groups_from_columns(ctx.cog_data_filtered, ["genotype", "treatment"]),
        q=[1, 5, 50, 95, 99],
        n_boot=1000,
    )
    ax2 = plot_group_quantiles(q_res, title="Speed percentiles by group")
    plt.show()

    # Pairwise differences: treatment and genotype (if at least two levels exist)
    try:
        import pandas as _pd  # noqa: F401
        groups_treat = build_groups_from_columns(ctx.cog_data_filtered, ["treatment"])
        groups_geno = build_groups_from_columns(ctx.cog_data_filtered, ["genotype"])
        treats = list(sorted({str(k) if not isinstance(k, tuple) else str(k[0]) for k in groups_treat.keys()}))
        genos = list(sorted({str(k) if not isinstance(k, tuple) else str(k[0]) for k in groups_geno.keys()}))
        if len(treats) >= 2:
            key_a, key_b = treats[0], treats[1]
            diff_treat = bootstrap_diff_by_keys(per_animal, groups_treat, key_a, key_b, stat='median')
            print(f"Median difference (treatment {key_a} - {key_b}): {diff_treat}")
            # Quantile differences with significance flags
            qdiff_treat = bootstrap_quantile_diffs_by_keys(
                per_animal, groups_treat, key_a, key_b, q=[1, 5, 50, 95, 99], n_boot=1000
            )
            print("Quantile diffs (A-B) by treatment:")
            for qi, pd, lo, hi, sig in zip(qdiff_treat["q"], qdiff_treat["point"], qdiff_treat["lo"], qdiff_treat["hi"], qdiff_treat["sig"]):
                print(f"  q{qi:.0f}: {pd:.4g}  CI[{lo:.4g}, {hi:.4g}]  sig={bool(sig)}")
            ax3 = plot_quantile_diffs(qdiff_treat, title=f"Quantile diffs (treatment {key_a} - {key_b})")
            plt.show()
            # Summarized significant ones with direction
            sig_list = summarize_significant_quantiles(qdiff_treat)
            if sig_list:
                print("Significant quantiles (A vs B):")
                for row in sig_list:
                    print(f"  q{row['q']:.0f}: diff={row['diff']:.4g}  CI[{row['lo']:.4g}, {row['hi']:.4g}]  {row['direction']}")

        # Plots for all combinations across treatment and genotype
        try:
            plot_quantiles_for_factors(per_animal, ctx.cog_data_filtered, factors=("genotype", "treatment"))
        except Exception as _e:
            print("Combo plots skipped:", _e)
        if len(genos) >= 2:
            key_a, key_b = genos[0], genos[1]
            diff_geno = bootstrap_diff_by_keys(per_animal, groups_geno, key_a, key_b, stat='median')
            print(f"Median difference (genotype {key_a} - {key_b}): {diff_geno}")

        # Significance plots across all pairwise combinations for treatment and genotype
        try:
            _ = plot_pairwise_quantile_diffs_for_factor(per_animal, ctx.cog_data_filtered, factor="treatment")
            plt.show()
        except Exception as _e:
            print("Pairwise treatment significance grid skipped:", _e)
        try:
            _ = plot_pairwise_quantile_diffs_for_factor(per_animal, ctx.cog_data_filtered, factor="genotype")
            plt.show()
        except Exception as _e:
            print("Pairwise genotype significance grid skipped:", _e)

        # Within-genotype treatment contrasts (A-B) across all genotypes, if exactly two treatments
        try:
            _ = plot_within_contrast_quantile_diffs(
                per_animal, ctx.cog_data_filtered, factors=("genotype", "treatment"), contrast="treatment"
            )
            plt.show()
        except Exception as _e:
            print("Within-genotype treatment contrast grid skipped:", _e)

        # Interaction: choose first two genotypes and two treatments if available
        groups_combo = build_groups_from_columns(ctx.cog_data_filtered, ["genotype", "treatment"])
        geno_vals = sorted(ctx.cog_data_filtered["genotype"].dropna().unique().tolist())
        treat_vals = sorted(ctx.cog_data_filtered["treatment"].dropna().unique().tolist())
        if len(geno_vals) >= 2 and len(treat_vals) >= 2:
            gA, gB = geno_vals[0], geno_vals[1]
            t0, t1 = treat_vals[0], treat_vals[1]
            dod = bootstrap_diff_of_diffs(
                per_animal, groups_combo,
                (gA, t1), (gA, t0), (gB, t1), (gB, t0), stat='median'
            )
            print(f"Interaction DoD [({gA}, {t1})-({gA}, {t0})] - [({gB}, {t1})-({gB}, {t0})]: {dod}")

        # Specific combo comparison example: ('Dp1Yey','LCTB92') vs ('WT','VEH')
        try:
            comp = compare_combos_significance(
                per_animal, ctx.cog_data_filtered,
                geno_a="Dp1Yey", treat_a="LCTB92",
                geno_b="WT", treat_b="VEH",
                q=[1, 5, 50, 95, 99], n_boot=1000, ci=95,
            )
            print("Significant quantiles for", comp["A_key"], "vs", comp["B_key"], ":")
            for row in comp["significant"]:
                print(f"  q{row['q']:.0f}: diff={row['diff']:.4g} CI[{row['lo']:.4g}, {row['hi']:.4g}] {row['direction']}")
        except Exception as _e:
            print("Specific combo comparison skipped:", _e)
    except Exception as e:  # only for demo robustness
        print("Pairwise/interaction demo skipped:", e)

# %%
