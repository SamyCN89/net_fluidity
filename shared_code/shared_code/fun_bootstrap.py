"""
Bootstrap kernels and helpers for dFC speed analyses.

Centralized, vectorized implementations to be reused by CLIs and notebooks.
"""
from __future__ import annotations

from typing import Iterable
import numpy as np


def bootstrap_percentiles(
    x: np.ndarray,
    q: Iterable[float] = (1, 5, 50, 95, 99),
    n_boot: int = 2000,
    ci: float = 95.0,
    seed: int | None = 0,
    chunk: int = 128,
    early_stop: float = 0.0,
    dtype: np.dtype = float,
    val_dtype: np.dtype | None = None,
    index_dtype: np.dtype | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap CI for percentiles of a 1D array.

    Returns (point, lo, hi) with shape (len(q),) each.
    NaNs in x are ignored.
    """
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


def bootstrap_diff_percentiles(
    x: np.ndarray,
    y: np.ndarray,
    q: Iterable[float] = (1, 5, 50, 95, 99),
    n_boot: int = 2000,
    ci: float = 95.0,
    seed: int | None = 0,
    chunk: int = 128,
    early_stop: float = 0.0,
    dtype: np.dtype = float,
    val_dtype: np.dtype | None = None,
    index_dtype: np.dtype | None = None,
) -> dict:
    """Bootstrap CI for percentile differences pct(x) - pct(y).

    Returns dict with keys: 'q', 'point', 'lo', 'hi', 'sig', 'n_x', 'n_y'.
    NaNs in x and y are ignored.
    """
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
                    return {"q": q_arr, "point": point, "lo": lo_t, "hi": hi_t, "sig": sig_t, "n_x": int(nx), "n_y": int(ny)}
            last_lo = lo_t
            last_hi = hi_t
    alpha = (100.0 - float(ci)) / 2.0
    lo = np.percentile(boots, alpha, axis=0)
    hi = np.percentile(boots, 100.0 - alpha, axis=0)
    sig = (lo > 0) | (hi < 0)
    return {"q": q_arr, "point": point, "lo": lo, "hi": hi, "sig": sig, "n_x": int(nx), "n_y": int(ny)}


def pool_per_animal(per_animal: list[np.ndarray], idxs: Iterable[int]) -> np.ndarray:
    vals = [per_animal[int(i)] for i in idxs if int(i) < len(per_animal)]
    nonempty = [v for v in vals if getattr(v, "size", 0) > 0]
    return np.concatenate(nonempty) if nonempty else np.array([])


def bootstrap_groups_percentiles(
    per_animal: list[np.ndarray],
    groups: dict,
    q: Iterable[float],
    n_boot: int,
    ci: float,
    seed: int,
    early_stop: float = 0.0,
    chunk: int = 128,
    dtype: np.dtype = float,
    val_dtype: np.dtype | None = None,
    index_dtype: np.dtype | None = None,
) -> dict:
    out: dict = {}
    q_arr = np.asarray(list(q), float)
    for g, idxs in groups.items():
        pooled = pool_per_animal(per_animal, idxs)
        point, lo, hi = bootstrap_percentiles(
            pooled, q=q_arr, n_boot=n_boot, ci=ci, seed=seed,
            early_stop=early_stop, chunk=chunk, dtype=dtype,
            val_dtype=val_dtype, index_dtype=index_dtype,
        )
        out[g] = {"q": q_arr, "point": point, "lo": lo, "hi": hi, "n": int(pooled.size)}
    return out


def bootstrap_groups_boots(
    per_animal: list[np.ndarray],
    groups: dict,
    q: Iterable[float] = (1, 5, 50, 95, 99),
    n_boot: int = 2000,
    seed: int | None = 0,
    chunk: int = 128,
    dtype: np.dtype = float,
    val_dtype: np.dtype | None = None,
    index_dtype: np.dtype | None = None,
) -> dict:
    """Return per-group bootstrap replicates of percentiles.

    - Returns { group_key: boots } where boots has shape (n_boot, len(q)).
    - Designed for reuse across many pairs: compute once per group, then derive
      pairwise diffs as boots[A] - boots[B] followed by CI on axis=0.
    - Ignores early stopping to ensure consistent shapes across groups.
    """
    q_arr = np.asarray(list(q), float)
    out: dict = {}
    for g, idxs in groups.items():
        x = pool_per_animal(per_animal, idxs)
        x = np.asarray(x, val_dtype or float)
        x = x[~np.isnan(x)]
        n = x.size
        boots = np.full((n_boot, q_arr.size), np.nan, dtype)
        if n == 0:
            out[g] = boots
            continue
        rng = np.random.default_rng(seed if seed is not None else 0)
        done = 0
        c = max(1, int(chunk))
        while done < n_boot:
            m = min(c, n_boot - done)
            if index_dtype is not None:
                idx = rng.integers(0, n, size=(m, n), endpoint=False, dtype=index_dtype)
            else:
                idx = rng.integers(0, n, size=(m, n), endpoint=False)
            xb = x[idx]
            boots[done : done + m, :] = np.percentile(xb, q_arr, axis=1).T
            done += m
        out[g] = boots
    out['__q__'] = q_arr  # attach once for convenience
    return out


def ci_from_boots(boots: np.ndarray, ci: float = 95.0) -> tuple[np.ndarray, np.ndarray]:
    """Compute CI bounds along axis=0 from bootstrap replicates array."""
    alpha = (100.0 - float(ci)) / 2.0
    lo = np.percentile(boots, alpha, axis=0)
    hi = np.percentile(boots, 100.0 - alpha, axis=0)
    return lo, hi
