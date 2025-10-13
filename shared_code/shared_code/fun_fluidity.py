#!/usr/bin/env python3
"""Fluidity and manifold dimension utilities extracted from legacy scripts."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import genpareto


def extremal_sueveges(samples: np.ndarray, quantile: float) -> float:
    """Estimate Suveges' extremal index for a 1D sample."""
    u = np.quantile(samples, quantile)
    q = 1.0 - quantile
    exceed_idx = np.where(samples > u)[0]
    if exceed_idx.size <= 1:
        return np.nan

    ti = np.diff(exceed_idx)
    si = ti - 1
    nc = np.sum(si > 0)
    n = len(ti)
    sum_qsi = np.sum(q * si)
    if sum_qsi == 0:
        return np.nan

    numerator = sum_qsi + n + nc - np.sqrt((sum_qsi + n + nc) ** 2 - 8 * nc * sum_qsi)
    return numerator / (2 * sum_qsi)


def manifold_fluidity(ts: np.ndarray, quantile: float = 0.98, step: int = 1):
    """Compute fluidity and manifold dimension for a trajectory matrix."""
    n_time = ts.shape[0]
    dimension = np.zeros(n_time, dtype=float)
    fluidity = np.zeros(n_time, dtype=float)

    for t in range(0, n_time, step):
        idx_others = np.setdiff1d(np.arange(0, n_time, step), [t])
        if idx_others.size == 0:
            continue

        distance = cdist(ts[t : t + 1], ts[idx_others])[0]
        logdist = -np.log(distance + np.finfo(float).eps)
        fluidity[t] = extremal_sueveges(logdist, quantile)

        thresh = np.quantile(logdist, quantile)
        above = np.sort(logdist[logdist > thresh])
        if above.size == 0:
            dimension[t] = np.nan
            continue

        try:
            _, _, scale = genpareto.fit(above - thresh, floc=0)
            dimension[t] = 1.0 / (scale + np.finfo(float).eps)
        except Exception:
            dimension[t] = np.nan

    return fluidity, dimension

