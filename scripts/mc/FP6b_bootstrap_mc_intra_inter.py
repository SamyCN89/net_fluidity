#!/usr/bin/env python3
# %%
"""
FP6b (FAST) — Bootstrap MC distribution summaries for FP6 conditions,
parallel over conditions, and *no giant concatenations* per bootstrap replicate.

What it does
------------
From FP6a per-animal pooled vectors, for each condition it computes:
  - Observed pooled distribution summaries (PDF + quantiles on p-grid)
  - Bootstrap CIs by resampling animals (A draws with replacement)

Speed strategy
--------------
- Avoid concatenating all sampled animals each replicate.
- For each bootstrap replicate, draw a fixed EDGE_SUBSAMPLE of MC values by:
    1) sampling animals with replacement (bootstrap)
    2) allocating per-animal draws ~ proportional to that animal's available values
    3) sampling values within each selected animal
- Parallelize over conditions with joblib (no nested parallelism).
- Stores only summaries + CIs (not raw bootstrap vectors).

Consumes
--------
results/<dataset>/mc/mc_dist/fp6a_mc_topology_modularity_per_animal.npz

Produces
--------
results/<dataset>/mc/mc_dist/fp6b_bootstrap_mc_fp6_conditions_FAST.npz
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

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

FP6A_NAME = "fp6a_mc_intra_inter_trimer_tetramer_per_animal.npz"
OUT_SUBDIR = "mc_dist"
OVERWRITE = True

# Bootstrap controls
N_BOOT = 2000
SEED = 0

# Performance / accuracy knob:
#   This is the *number of MC values* used to estimate PDF + quantiles per replicate.
#   200k–500k is usually a good compromise.
EDGE_SUBSAMPLE = 300_000

# Summaries
P_GRID = np.linspace(0, 1, 101)          # includes tails (0..1)
BINS = np.linspace(-1, 1, 401)       # wide support for tails

# Parallelism
N_JOBS = -1  # parallel over conditions


# =========================
# Helpers
# =========================
def _as_float_1d(x) -> np.ndarray:
    """Convert a per-animal vector to 1D float32, finite only."""
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


def _summaries_from_sample(x: np.ndarray, bins: np.ndarray, p_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (pdf_density, quantiles). x is 1D finite float."""
    if x.size == 0:
        pdf = np.full(bins.size - 1, np.nan, dtype=np.float32)
        q = np.full(p_grid.size, np.nan, dtype=np.float32)
        return pdf, q

    counts, _ = np.histogram(x, bins=bins, density=False)
    pdf = _pdf_density_from_counts(counts, bins, x.size)
    q = np.quantile(x, p_grid).astype(np.float32)
    return pdf, q


def _sample_from_animals(
    values: List[np.ndarray],
    pick_animals: np.ndarray,        # indices length A (with replacement)
    n_draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw n_draws values from the *bootstrap-resampled animal pool* without concatenation.

    values: list of length A, each is 1D finite float array
    pick_animals: length A bootstrap sample of animal indices
    """
    A = len(values)
    if A == 0:
        return np.array([], dtype=np.float32)

    # lengths of the selected animals
    lens = np.fromiter((values[i].size for i in pick_animals), dtype=np.int64, count=pick_animals.size)
    tot = int(lens.sum())
    if tot == 0:
        return np.array([], dtype=np.float32)

    # If the pool is smaller than n_draws, just take all by concatenating once (rare).
    if tot <= n_draws:
        # still avoid giant concat most of the time; but tot is small here
        return np.concatenate([values[i] for i in pick_animals if values[i].size], axis=0).astype(np.float32, copy=False)

    # Allocate per-animal number of draws proportional to its available values
    probs = lens / tot
    draws = rng.multinomial(n_draws, probs)  # length = pick_animals.size

    out = np.empty(n_draws, dtype=np.float32)
    pos = 0
    for j, n_j in enumerate(draws):
        if n_j == 0:
            continue
        a_idx = int(pick_animals[j])
        v = values[a_idx]
        if v.size == 0:
            continue
        # sample indices within that animal
        ii = rng.integers(0, v.size, size=n_j)
        out[pos : pos + n_j] = v[ii]
        pos += n_j

    # In extremely rare cases (empty animals eaten draws), pos < n_draws; trim.
    return out[:pos]


def summarize_condition_fast(
    key: str,
    values_per_animal_obj: np.ndarray,
    n_boot: int,
    seed: int,
    bins: np.ndarray,
    p_grid: np.ndarray,
    edge_subsample: int,
) -> Dict[str, np.ndarray]:
    """
    Return obs summaries + bootstrap CI bands for one condition.
    """
    # Preconvert once: list of per-animal finite float arrays
    values = [_as_float_1d(v) for v in list(values_per_animal_obj)]
    A = len(values)

    # Observed: sample from ALL animals pooled by drawing edge_subsample points
    rng_obs = np.random.default_rng(seed + 999)
    pick_all = np.arange(A, dtype=np.int64)
    # To mimic “pooled all animals”, we sample animals *with replacement* or not?
    # Here: not needed. We want the pooled distribution over the dataset -> sample proportional to lengths.
    # We implement it by "bootstrap sample equals identity" + multinomial weighting:
    x_obs = _sample_from_animals(values, pick_all, edge_subsample, rng_obs)
    pdf_obs, q_obs = _summaries_from_sample(x_obs, bins, p_grid)

    # Bootstrap arrays
    nb = bins.size - 1
    nq = p_grid.size
    pdf_boot = np.empty((n_boot, nb), dtype=np.float32)
    q_boot = np.empty((n_boot, nq), dtype=np.float32)

    rng = np.random.default_rng(seed)
    for b in range(n_boot):
        pick = rng.integers(0, A, size=A)  # bootstrap animals
        x = _sample_from_animals(values, pick, edge_subsample, rng)
        pdf_boot[b], q_boot[b] = _summaries_from_sample(x, bins, p_grid)

    # CI bands
    pdf_lo = np.quantile(pdf_boot, 0.025, axis=0).astype(np.float32)
    pdf_hi = np.quantile(pdf_boot, 0.975, axis=0).astype(np.float32)
    q_lo = np.quantile(q_boot, 0.025, axis=0).astype(np.float32)
    q_hi = np.quantile(q_boot, 0.975, axis=0).astype(np.float32)

    return dict(
        key=str(key),
        n_animals=np.int32(A),
        n_obs_used=np.int32(x_obs.size),
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

# Derived marginals (your request)
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
cond_items = list(all_conditions.items())

print(f"[FP6b FAST] Conditions: {len(cond_items)} | N_BOOT={N_BOOT} | EDGE_SUBSAMPLE={EDGE_SUBSAMPLE} | N_JOBS={N_JOBS}")

# Parallel over conditions (each condition independent)
results = Parallel(n_jobs=N_JOBS, backend="loky")(
    delayed(summarize_condition_fast)(
        key=k,
        values_per_animal_obj=v,
        n_boot=N_BOOT,
        seed=SEED + 1000 * i,
        bins=BINS,
        p_grid=P_GRID,
        edge_subsample=EDGE_SUBSAMPLE,
    )
    for i, (k, v) in enumerate(cond_items)
)

# Save
out_dir = mc_dir / OUT_SUBDIR
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / "fp6b_bootstrap_mc_fp6_conditions_FAST.npz"
if out_path.exists() and not OVERWRITE:
    raise FileExistsError(out_path)

save = {
    "bins": BINS.astype(np.float32),
    "p_grid": P_GRID.astype(np.float32),
    "bin_centers": ((BINS[:-1] + BINS[1:]) * 0.5).astype(np.float32),
}


for res in results:
    key = res["key"]
    for field in ["pdf_obs", "pdf_ci_lo", "pdf_ci_hi", "q_obs", "q_ci_lo", "q_ci_hi", "n_animals", "n_obs_used"]:
        save[f"{key}__{field}"] = res[field]


params = dict(
    dataset=DATASET,
    fp6a_path=str(fp6a_path),
    n_boot=int(N_BOOT),
    seed=int(SEED),
    edge_subsample=int(EDGE_SUBSAMPLE),
    bins=[float(x) for x in BINS.tolist()],
    p_grid=[float(x) for x in P_GRID.tolist()],
    conditions=list(all_conditions.keys()),
    parallel_over="conditions",
)

save["params_json"] = json.dumps(params, sort_keys=True)

np.savez_compressed(out_path, **save)

print("[OK] Saved FP6b FAST:", out_path)
