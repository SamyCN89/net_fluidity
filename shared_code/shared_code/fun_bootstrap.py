"""
Bootstrap kernels and helpers for dFC speed analyses.

Centralized, vectorized implementations to be reused by CLIs and notebooks.
"""

from __future__ import annotations

from collections.abc import Iterable

from numba import njit, prange
import numpy as np


@njit(parallel=True, fastmath=True)
def _bootstrap_diff_inner(x, y, q_arr, n_boot, ci, seed):
    """Low-level parallel bootstrap of percentile differences."""
    nx, ny = x.size, y.size
    nq = q_arr.size
    boots = np.empty((n_boot, nq), np.float32)
    rng = np.random.default_rng(seed)

    for b in prange(n_boot):
        idx_x = rng.integers(0, nx, nx)
        idx_y = rng.integers(0, ny, ny)
        xb = x[idx_x]
        yb = y[idx_y]
        for i in range(nq):
            q = q_arr[i]
            px = np.percentile(xb, q)
            py = np.percentile(yb, q)
            boots[b, i] = px - py

    alpha = (100.0 - ci) / 2.0
    lo = np.percentile(boots, alpha, axis=0)
    hi = np.percentile(boots, 100.0 - alpha, axis=0)
    sig = (lo > 0) | (hi < 0)
    return lo, hi, sig


def bootstrap_percentiles(
    x: np.ndarray,
    q: Iterable[float] = (1, 5, 50, 95, 99),
    n_boot: int = 2000,
    ci: float = 95.0,
    seed: int | None = 0,
    chunk: int = 128,
    dtype: np.dtype = float,
    val_dtype: np.dtype | None = None,
    index_dtype: np.dtype | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap CI for percentiles of a 1D sample.

    - Resamples x with replacement n_boot times, computes percentiles q on each
      bootstrap replicate, and derives CI bounds by percentile-of-bootstrap.
    - Returns (point, lo, hi) with shape (len(q),) each.
    - NaNs in x are ignored.
    - chunk controls vectorized resampling batch size; dtype controls boots array
      type to trade precision vs memory; index_dtype controls index array type.
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
        boots[done : done + m, :] = np.percentile(
            xb, q_arr, axis=1, overwrite_input=True
        ).T
        done += m

    alpha = (100.0 - float(ci)) / 2.0
    lo = np.percentile(boots, alpha, axis=0)
    hi = np.percentile(boots, 100.0 - alpha, axis=0)
    return point, lo, hi


# def bootstrap_diff_percentiles(
#     x: np.ndarray,
#     y: np.ndarray,
#     q: Iterable[float] = (1, 5, 50, 95, 99),
#     n_boot: int = 2000,
#     ci: float = 95.0,
#     seed: int | None = 0,
#     chunk: int = 128,
#     dtype: np.dtype = float,
#     val_dtype: np.dtype | None = None,
#     index_dtype: np.dtype | None = None,
# ) -> dict:
#     """Bootstrap CI for percentile differences pct(x) - pct(y).

#     - Resamples x and y independently, computes pct(x_b) - pct(y_b) per replicate,
#       and derives CI bounds by percentile-of-bootstrap.
#     - Returns dict with keys: 'q', 'point', 'lo', 'hi', 'sig', 'n_x', 'n_y'.
#       'sig' marks whether the CI excludes 0.
#     - NaNs in x and y are ignored.
#     """
#     x = np.asarray(x, val_dtype or float)
#     y = np.asarray(y, val_dtype or float)
#     x = x[~np.isnan(x)]
#     y = y[~np.isnan(y)]
#     q_arr = np.asarray(list(q), float)
#     if x.size == 0 or y.size == 0:
#         m = q_arr.size
#         nan = np.full(m, np.nan)
#         return {
#             "q": q_arr,
#             "point": nan,
#             "lo": nan,
#             "hi": nan,
#             "sig": np.zeros(m, bool),
#             "n_x": int(x.size),
#             "n_y": int(y.size),
#         }
#     point = np.percentile(x, q_arr) - np.percentile(y, q_arr)
#     rng = np.random.default_rng(seed)
#     nx, ny = x.size, y.size
#     boots = np.empty((n_boot, q_arr.size), dtype)
#     done = 0
#     chunk = max(1, int(chunk))
#     while done < n_boot:
#         m = min(chunk, n_boot - done)
#         if index_dtype is not None:
#             idx_x = rng.integers(0, nx, size=(m, nx), endpoint=False, dtype=index_dtype)
#             idx_y = rng.integers(0, ny, size=(m, ny), endpoint=False, dtype=index_dtype)
#         else:
#             idx_x = rng.integers(0, nx, size=(m, nx), endpoint=False)
#             idx_y = rng.integers(0, ny, size=(m, ny), endpoint=False)
#         xb = x[idx_x]
#         yb = y[idx_y]
#         pct_x = np.percentile(xb, q_arr, axis=1, overwrite_input=True)
#         pct_y = np.percentile(yb, q_arr, axis=1, overwrite_input=True)
#         boots[done : done + m, :] = (pct_x - pct_y).T
#         done += m
#     alpha = (100.0 - float(ci)) / 2.0
#     lo = np.percentile(boots, alpha, axis=0)
#     hi = np.percentile(boots, 100.0 - alpha, axis=0)
#     sig = (lo > 0) | (hi < 0)
#     return {"q": q_arr, "point": point, "lo": lo, "hi": hi, "sig": sig, "n_x": int(nx), "n_y": int(ny)}


def bootstrap_diff_percentiles(
    x: np.ndarray,
    y: np.ndarray,
    q: Iterable[float] = (1, 5, 50, 95, 99),
    n_boot: int = 2000,
    ci: float = 95.0,
    seed: int | None = 0,
    chunk: int = 128,
    dtype: np.dtype = float,
    val_dtype: np.dtype | None = None,
    index_dtype: np.dtype | None = None,
) -> dict:
    """Bootstrap CI for percentile differences pct(x) - pct(y), parallelized via Numba."""
    x = np.asarray(x, val_dtype or float)
    y = np.asarray(y, val_dtype or float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    q_arr = np.asarray(list(q), float)
    nx, ny = x.size, y.size

    if nx == 0 or ny == 0:
        m = q_arr.size
        nan = np.full(m, np.nan)
        return {
            "q": q_arr,
            "point": nan,
            "lo": nan,
            "hi": nan,
            "sig": np.zeros(m, bool),
            "n_x": int(nx),
            "n_y": int(ny),
        }

    # Observed percentile difference
    point = np.percentile(x, q_arr) - np.percentile(y, q_arr)

    # Parallel bootstrap using numba-compiled kernel
    lo, hi, sig = _bootstrap_diff_inner(x, y, q_arr, n_boot, ci, seed)

    return {
        "q": q_arr,
        "point": point,
        "lo": lo.astype(dtype),
        "hi": hi.astype(dtype),
        "sig": sig,
        "n_x": int(nx),
        "n_y": int(ny),
    }


def bootstrap_group_from_pool(
    target: np.ndarray,
    pool: np.ndarray,
    q: Iterable[float] = (1, 5, 50, 95, 99),
    n_boot: int = 2000,
    ci: float = 95.0,
    seed: int | None = 0,
    chunk: int = 128,
    dtype: np.dtype = float,
    val_dtype: np.dtype | None = None,
    index_dtype: np.dtype | None = None,
) -> dict:
    """Bootstrap target-group percentiles against a pooled supergroup.

    - Forms a null by resampling WITH replacement from `pool` with sample size
      equal to len(target). Computes percentiles for each replicate.
    - Returns dict with keys:
      'q', 'target_point', 'pool_point', 'lo', 'hi', 'inside', 'p', 'n_target', 'n_pool'
      where 'inside' marks whether target_point lies inside [lo, hi], and 'p' is
      an empirical two-sided p-value based on deviation from pool_point.
    """
    target = np.asarray(target, val_dtype or float)
    pool = np.asarray(pool, val_dtype or float)
    target = target[~np.isnan(target)]
    pool = pool[~np.isnan(pool)]
    q_arr = np.asarray(list(q), float)
    nt = int(target.size)
    npool = int(pool.size)
    if nt == 0 or npool == 0:
        m = q_arr.size
        nan = np.full(m, np.nan)
        return {
            "q": q_arr,
            "target_point": nan,
            "pool_point": nan,
            "lo": nan,
            "hi": nan,
            "inside": np.zeros(m, bool),
            "p": nan,
            "n_target": nt,
            "n_pool": npool,
        }
    # Observed
    target_point = np.percentile(target, q_arr)
    pool_point = np.percentile(pool, q_arr)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, q_arr.size), dtype)
    done = 0
    c = max(1, int(chunk))
    while done < n_boot:
        m = min(c, n_boot - done)
        if index_dtype is not None:
            idx = rng.integers(
                0, npool, size=(m, nt), endpoint=False, dtype=index_dtype
            )
        else:
            idx = rng.integers(0, npool, size=(m, nt), endpoint=False)
        xb = pool[idx]
        boots[done : done + m, :] = np.percentile(
            xb, q_arr, axis=1, overwrite_input=True
        ).T
        done += m
    alpha = (100.0 - float(ci)) / 2.0
    lo = np.percentile(boots, alpha, axis=0)
    hi = np.percentile(boots, 100.0 - alpha, axis=0)
    inside = (target_point >= lo) & (target_point <= hi)
    # Two-sided p-value relative to pool_point
    p = np.empty_like(q_arr, dtype=float)
    for j in range(q_arr.size):
        dev = abs(target_point[j] - pool_point[j])
        dev_boot = np.abs(boots[:, j] - pool_point[j])
        p[j] = (np.count_nonzero(dev_boot >= dev) + 1.0) / (boots.shape[0] + 1.0)
    return {
        "q": q_arr,
        "target_point": target_point,
        "pool_point": pool_point,
        "lo": lo,
        "hi": hi,
        "inside": inside,
        "p": p,
        "n_target": nt,
        "n_pool": npool,
    }


def pool_per_animal(per_animal: list[np.ndarray], idxs: Iterable[int]) -> np.ndarray:
    """Pool selected per-animal arrays into a single 1D array.

    - Filters out empty arrays, concatenates along the sample axis.
    """
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
    chunk: int = 128,
    dtype: np.dtype = float,
    val_dtype: np.dtype | None = None,
    index_dtype: np.dtype | None = None,
) -> dict:
    """Bootstrap percentiles per group using pooled per-animal values.

    - For each group key -> list of animal indices, pools values and calls
      bootstrap_percentiles; returns a dict keyed by group.
    """
    out: dict = {}
    q_arr = np.asarray(list(q), float)
    for g, idxs in groups.items():
        pooled = pool_per_animal(per_animal, idxs)  # pool values for this group
        point, lo, hi = bootstrap_percentiles(
            pooled,
            q=q_arr,
            n_boot=n_boot,
            ci=ci,
            seed=seed,
            chunk=chunk,
            dtype=dtype,
            val_dtype=val_dtype,
            index_dtype=index_dtype,
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
    """Return per-group bootstrap replicates of percentiles (for reuse).

    - Returns { group_key: boots } where boots has shape (n_boot, len(q)).
    - Designed to compute all groups once, then derive pairwise diffs as
      boots[A] - boots[B] followed by CI via ci_from_boots.
    """
    q_arr = np.asarray(list(q), float)
    out: dict = {}

    # Iterate groups, pool values, and compute bootstrap replicates
    for g, idxs in groups.items():
        x = pool_per_animal(per_animal, idxs)
        x = np.asarray(x, val_dtype or float)
        x = x[~np.isnan(x)]
        n = x.size
        boots = np.full((n_boot, q_arr.size), np.nan, dtype)
        if n == 0:
            out[g] = boots
            continue
        rng = np.random.default_rng(seed)
        done = 0
        c = max(1, int(chunk))
        while done < n_boot:
            m = min(c, n_boot - done)
            if index_dtype is not None:
                idx = rng.integers(0, n, size=(m, n), endpoint=False, dtype=index_dtype)
            else:
                idx = rng.integers(0, n, size=(m, n), endpoint=False)
            xb = x[idx]
            boots[done : done + m, :] = np.percentile(
                xb, q_arr, axis=1, overwrite_input=True
            ).T
            done += m
        out[g] = boots
    out["__q__"] = q_arr  # attach once for convenience
    return out


def ci_from_boots(boots: np.ndarray, ci: float = 95.0) -> tuple[np.ndarray, np.ndarray]:
    """Compute CI bounds along axis=0 from bootstrap replicates array."""
    alpha = (100.0 - float(ci)) / 2.0
    lo = np.percentile(boots, alpha, axis=0)
    hi = np.percentile(boots, 100.0 - alpha, axis=0)
    return lo, hi
