#!/usr/bin/env python3
# %%
"""
FP6b (POOLED-iid) — Bootstrap uncertainty of the *pooled empirical distribution*
(i.i.d. value bootstrap). Observed summaries are deterministic (exact pooled).

Consumes
--------
results/<dataset>/mc/mc_dist/fp6a_mc_intra_inter_trimer_tetramer_per_animal.npz

Produces
--------
results/<dataset>/mc/mc_dist/fp6b_bootstrap_mc_fp6_conditions_POOLED_IID.npz

Design
------
- Observed ("obs"): computed from the FULL pooled vector once (exact concat).
- Bootstrap ("CI"): resample values from the pooled vector with replacement.
- This quantifies uncertainty of the pooled empirical distribution under i.i.d. values.
  (NOT between-animal uncertainty.)

Notes
-----
- Pool concatenation happens ONCE per condition. Can be memory-heavy for obs_all/null_all.
- Bootstrap draws use BOOT_DRAWS (EDGE_SUBSAMPLE) for speed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from joblib import Parallel, delayed

from shared_code.fun_paths import get_paths

# =========================
# CONFIG
# =========================
DATASET = "ines_abdallah"
TIMECOURSE_FOLDER = "Timecourses_updated_03052024"
COGNITIVE_FILE = "ROIs.xlsx"
ANAT_LABELS_FILE = "41_Allen.txt"

OUT_SUBDIR = "mc_dist"
FP6A_NAME = "fp6a_mc_intra_inter_trimer_tetramer_per_animal.npz"
OUT_NAME = "fp6b_bootstrap_mc_fp6_conditions_POOLED_IID.npz"
OVERWRITE = True

# Bootstrap
N_BOOT = 2000
SEED = 0

# Bootstrap draws per replicate (values)
# (keep the same name to stay compatible with your pipeline)
EDGE_SUBSAMPLE = 300_000

# Quantiles to save (include tails)
P_GRID = np.linspace(0.0, 1.0, 101).astype(np.float32)

# Histogram resolution
BINS_MIN = -0.8
BINS_MAX = 0.8
NBINS = 401  # 401 => 400 bins

# Parallelism over conditions
N_JOBS = -1
JOBLIB_BACKEND = "loky"

# Optional: run only a subset to save time (set to None to run all)
CONDITIONS_TO_RUN: Optional[List[str]] = None


# =========================
# Helpers
# =========================
def _as_float_1d(x) -> np.ndarray:
    """Convert a per-animal vector to 1D float32, finite-only."""
    arr = np.asarray(x)
    if arr.size == 0:
        return np.array([], dtype=np.float32)
    arr = arr.astype(np.float32, copy=False).ravel()
    arr = arr[np.isfinite(arr)]
    return arr


def _concat_per_animal(a_list, b_list) -> np.ndarray:
    """Per-animal concatenation -> object array length A."""
    A = len(a_list)
    out = np.empty(A, dtype=object)
    for i in range(A):
        a = _as_float_1d(a_list[i])
        b = _as_float_1d(b_list[i])
        if a.size and b.size:
            out[i] = np.concatenate([a, b], axis=0)
        elif a.size:
            out[i] = a
        elif b.size:
            out[i] = b
        else:
            out[i] = np.array([], dtype=np.float32)
    return out


def _pdf_density_from_counts(counts: np.ndarray, bins: np.ndarray, n: int) -> np.ndarray:
    """Convert histogram counts into density."""
    if n <= 0:
        return np.full_like(counts, np.nan, dtype=np.float32)
    widths = np.diff(bins).astype(np.float32)
    return (counts.astype(np.float32) / (n * widths)).astype(np.float32)


def _summaries_from_sample(
    x: np.ndarray, bins: np.ndarray, p_grid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (pdf_density, quantiles). x is 1D finite float32."""
    if x.size == 0:
        pdf = np.full(bins.size - 1, np.nan, dtype=np.float32)
        q = np.full(p_grid.size, np.nan, dtype=np.float32)
        return pdf, q
    counts, _ = np.histogram(x, bins=bins, density=False)
    pdf = _pdf_density_from_counts(counts, bins, x.size)
    q = np.quantile(x, p_grid).astype(np.float32)
    return pdf, q


def _pool_concat(values_per_animal_obj: np.ndarray) -> np.ndarray:
    """
    Exact pooled vector across animals (finite-only), float32.
    WARNING: can be large (done once per condition).
    """
    parts = []
    for v in values_per_animal_obj:
        x = _as_float_1d(v)
        if x.size:
            parts.append(x)
    if not parts:
        return np.array([], dtype=np.float32)
    return np.concatenate(parts, axis=0)


def _bootstrap_sample_from_pool(x_pool: np.ndarray, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """i.i.d. bootstrap sample from pooled values."""
    n = x_pool.size
    if n == 0:
        return np.array([], dtype=np.float32)
    if n_draws is None or n_draws <= 0:
        n_draws = n
    ii = rng.integers(0, n, size=n_draws, dtype=np.int64)
    return x_pool[ii]


def summarize_condition_pooled_iid(
    key: str,
    values_per_animal_obj: np.ndarray,
    n_boot: int,
    seed: int,
    bins: np.ndarray,
    p_grid: np.ndarray,
    boot_draws: int,
) -> Dict[str, np.ndarray]:
    """
    Observed: exact pooled concat.
    Bootstrap: i.i.d. resampling from pooled vector.
    """
    x_pool = _pool_concat(values_per_animal_obj)

    # Deterministic observed summaries (no EDGE_SUBSAMPLE randomness)
    pdf_obs, q_obs = _summaries_from_sample(x_pool, bins, p_grid)

    nb = bins.size - 1
    nq = p_grid.size
    pdf_boot = np.empty((n_boot, nb), dtype=np.float32)
    q_boot = np.empty((n_boot, nq), dtype=np.float32)

    rng = np.random.default_rng(seed)
    for b in range(n_boot):
        x = _bootstrap_sample_from_pool(x_pool, boot_draws, rng)
        pdf_boot[b], q_boot[b] = _summaries_from_sample(x, bins, p_grid)

    pdf_lo = np.quantile(pdf_boot, 0.025, axis=0).astype(np.float32)
    pdf_hi = np.quantile(pdf_boot, 0.975, axis=0).astype(np.float32)
    q_lo = np.quantile(q_boot, 0.025, axis=0).astype(np.float32)
    q_hi = np.quantile(q_boot, 0.975, axis=0).astype(np.float32)

    return dict(
        key=str(key),
        n_animals=np.int32(len(values_per_animal_obj)),
        n_pool=np.int64(x_pool.size),
        n_boot_draws=np.int64(boot_draws),
        pdf_obs=pdf_obs,
        pdf_ci_lo=pdf_lo,
        pdf_ci_hi=pdf_hi,
        q_obs=q_obs,
        q_ci_lo=q_lo,
        q_ci_hi=q_hi,
    )


# =========================
# MAIN
# =========================
paths = get_paths(
    dataset_name=DATASET,
    timecourse_folder=TIMECOURSE_FOLDER,
    cognitive_data_file=COGNITIVE_FILE,
    anat_labels_file=ANAT_LABELS_FILE,
)

mc_dir = Path(paths["mc"])
fp6a_path = mc_dir / OUT_SUBDIR / FP6A_NAME
if not fp6a_path.exists():
    raise FileNotFoundError(f"Missing FP6a: {fp6a_path}")

d = np.load(fp6a_path, allow_pickle=True)

# Atomic conditions from FP6a
base = {
    "obs_intra_trimer": d["obs_intra_trimer"],
    "obs_inter_trimer": d["obs_inter_trimer"],
    "obs_intra_tetramer": d["obs_intra_tetramer"],
    "obs_inter_tetramer": d["obs_inter_tetramer"],
    "null_intra_trimer": d["null_intra_trimer"],
    "null_inter_trimer": d["null_inter_trimer"],
    "null_intra_tetramer": d["null_intra_tetramer"],
    "null_inter_tetramer": d["null_inter_tetramer"],
}

# Derived marginals
derived = {
    "obs_all": _concat_per_animal(
        _concat_per_animal(base["obs_intra_trimer"], base["obs_inter_trimer"]),
        _concat_per_animal(base["obs_intra_tetramer"], base["obs_inter_tetramer"]),
    ),
    "null_all": _concat_per_animal(
        _concat_per_animal(base["null_intra_trimer"], base["null_inter_trimer"]),
        _concat_per_animal(base["null_intra_tetramer"], base["null_inter_tetramer"]),
    ),
    "obs_intra": _concat_per_animal(base["obs_intra_trimer"], base["obs_intra_tetramer"]),
    "obs_inter": _concat_per_animal(base["obs_inter_trimer"], base["obs_inter_tetramer"]),
    "obs_trimer": _concat_per_animal(base["obs_intra_trimer"], base["obs_inter_trimer"]),
    "obs_tetramer": _concat_per_animal(base["obs_intra_tetramer"], base["obs_inter_tetramer"]),
    "null_intra": _concat_per_animal(base["null_intra_trimer"], base["null_intra_tetramer"]),
    "null_inter": _concat_per_animal(base["null_inter_trimer"], base["null_inter_tetramer"]),
    "null_trimer": _concat_per_animal(base["null_intra_trimer"], base["null_inter_trimer"]),
    "null_tetramer": _concat_per_animal(base["null_intra_tetramer"], base["null_inter_tetramer"]),
}

all_conditions = {**base, **derived}

# Optional subset
if CONDITIONS_TO_RUN is not None:
    missing = [k for k in CONDITIONS_TO_RUN if k not in all_conditions]
    if missing:
        raise KeyError(f"Unknown CONDITIONS_TO_RUN keys: {missing}")
    all_conditions = {k: all_conditions[k] for k in CONDITIONS_TO_RUN}

cond_items = list(all_conditions.items())

# =========================
# BINS (MANUAL)
# =========================
if not (np.isfinite(BINS_MIN) and np.isfinite(BINS_MAX) and BINS_MAX > BINS_MIN):
    raise ValueError(f"Bad manual bins: BINS_MIN={BINS_MIN}, BINS_MAX={BINS_MAX}")

bins = np.linspace(BINS_MIN, BINS_MAX, NBINS, dtype=np.float32)
bin_centers = ((bins[:-1] + bins[1:]) * 0.5).astype(np.float32)

print(
    f"[FP6b POOLED-iid] Conditions={len(cond_items)} | "
    f"N_BOOT={N_BOOT} | BOOT_DRAWS={EDGE_SUBSAMPLE} | "
    f"BINS=manual | range=({BINS_MIN},{BINS_MAX}) | NBINS={NBINS}"
)

results = Parallel(n_jobs=N_JOBS, backend=JOBLIB_BACKEND)(
    delayed(summarize_condition_pooled_iid)(
        key=k,
        values_per_animal_obj=v,
        n_boot=N_BOOT,
        seed=SEED + 1000 * i,
        bins=bins,
        p_grid=P_GRID,
        boot_draws=EDGE_SUBSAMPLE,
    )
    for i, (k, v) in enumerate(cond_items)
)

# Save
out_dir = mc_dir / OUT_SUBDIR
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / OUT_NAME

if out_path.exists() and not OVERWRITE:
    raise FileExistsError(out_path)

save = {
    "bins": bins.astype(np.float32),
    "bin_centers": bin_centers,
    "p_grid": P_GRID.astype(np.float32),
}

for res in results:
    key = res["key"]
    for field in [
        "pdf_obs", "pdf_ci_lo", "pdf_ci_hi",
        "q_obs", "q_ci_lo", "q_ci_hi",
        "n_animals", "n_pool", "n_boot_draws",
    ]:
        save[f"{key}__{field}"] = res[field]

params = dict(
    dataset=DATASET,
    fp6a_path=str(fp6a_path),
    out_path=str(out_path),
    n_boot=int(N_BOOT),
    seed=int(SEED),
    bootstrap_unit="pooled_values_iid",
    obs_estimator="exact_pool_concat",
    boot_draws=int(EDGE_SUBSAMPLE),
    nbins=int(NBINS),
    bins_min=float(BINS_MIN),
    bins_max=float(BINS_MAX),
    p_grid=[float(x) for x in P_GRID.tolist()],
    conditions=list(all_conditions.keys()),
    parallel_over="conditions",
)

save["params_json"] = json.dumps(params, sort_keys=True)

np.savez_compressed(out_path, **save)

print("[OK] Saved FP6b POOLED-iid:", out_path)
