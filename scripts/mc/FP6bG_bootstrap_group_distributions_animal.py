#!/usr/bin/env python3
# %%
"""
FP6bG — Animal-bootstrap uncertainty of group-level pooled distributions.

Consumes:
  mc_dist/fp6a_groups_mc_by_topology_per_animal.npz

Produces:
  mc_dist/fp6b_groups_bootstrap_mc_conditions_ANIMALBOOT.npz

For each group g and each condition key:
  - obs summaries are deterministic from exact pooled concat (recommended)
  - bootstrap: resample animals (within group) with replacement, then pool all values
    (optionally subsample values per replicate for speed)
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
from joblib import Parallel, delayed

from shared_code.fun_paths import get_paths

# -------------------------
# CONFIG
# -------------------------
DATASET = "ines_abdallah"
TIMECOURSE_FOLDER = "Timecourses_updated_03052024"
COGNITIVE_FILE = "ROIs.xlsx"
ANAT_LABELS_FILE = "41_Allen.txt"

MC_DIST_DIRNAME = "mc_dist"
FP6AG_NAME = "fp6a_groups_mc_by_topology_per_animal.npz"
OUT_NAME = "fp6b_groups_bootstrap_mc_conditions_ANIMALBOOT.npz"
OVERWRITE = True

N_BOOT = 2000
SEED = 0

# per bootstrap replicate, optionally cap number of pooled draws for speed
BOOT_DRAWS: Optional[int] = 300_000  # set None for full pooled (can be huge)

# quantiles
P_GRID = np.linspace(0.0, 1.0, 101).astype(np.float32)

# histogram bins (manual, stable)
BINS_MIN = -0.8
BINS_MAX = 0.8
NBINS = 401

N_JOBS = -1
JOBLIB_BACKEND = "loky"

# -------------------------
# Helpers
# -------------------------
def _as_float_1d(x) -> np.ndarray:
    x = np.asarray(x)
    if x.size == 0:
        return np.array([], dtype=np.float32)
    x = x.astype(np.float32, copy=False).ravel()
    x = x[np.isfinite(x)]
    return x

def _pool_concat(obj_arr: np.ndarray) -> np.ndarray:
    parts = []
    for v in obj_arr:
        x = _as_float_1d(v)
        if x.size:
            parts.append(x)
    return np.concatenate(parts, axis=0) if parts else np.array([], dtype=np.float32)

def _summaries(x: np.ndarray, bins: np.ndarray, p_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        return (np.full(bins.size-1, np.nan, np.float32),
                np.full(p_grid.size, np.nan, np.float32))
    counts, _ = np.histogram(x, bins=bins, density=False)
    widths = np.diff(bins).astype(np.float32)
    pdf = (counts.astype(np.float32) / (x.size * widths)).astype(np.float32)
    q = np.quantile(x, p_grid).astype(np.float32)
    return pdf, q

def _resample_animals_pool(obj_arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    A = obj_arr.size
    pick = rng.integers(0, A, size=A)  # bootstrap animals
    parts = []
    for i in pick:
        x = _as_float_1d(obj_arr[int(i)])
        if x.size:
            parts.append(x)
    if not parts:
        return np.array([], dtype=np.float32)
    x_pool = np.concatenate(parts, axis=0)
    if BOOT_DRAWS is not None and x_pool.size > BOOT_DRAWS:
        jj = rng.integers(0, x_pool.size, size=BOOT_DRAWS)
        x_pool = x_pool[jj]
    return x_pool

def summarize_group_condition(g: str, key: str, obj_arr: np.ndarray, seed: int, bins: np.ndarray, p_grid: np.ndarray) -> Dict[str, np.ndarray]:
    # deterministic observed (exact pooled)
    x_obs = _pool_concat(obj_arr)
    pdf_obs, q_obs = _summaries(x_obs, bins, p_grid)

    nb = bins.size - 1
    nq = p_grid.size
    pdf_boot = np.empty((N_BOOT, nb), dtype=np.float32)
    q_boot = np.empty((N_BOOT, nq), dtype=np.float32)

    rng = np.random.default_rng(seed)
    for b in range(N_BOOT):
        xb = _resample_animals_pool(obj_arr, rng)
        pdf_boot[b], q_boot[b] = _summaries(xb, bins, p_grid)

    return dict(
        g=g, key=key,
        n_animals=np.int32(obj_arr.size),
        n_pool=np.int64(x_obs.size),
        pdf_obs=pdf_obs,
        pdf_ci_lo=np.quantile(pdf_boot, 0.025, axis=0).astype(np.float32),
        pdf_ci_hi=np.quantile(pdf_boot, 0.975, axis=0).astype(np.float32),
        q_obs=q_obs,
        q_ci_lo=np.quantile(q_boot, 0.025, axis=0).astype(np.float32),
        q_ci_hi=np.quantile(q_boot, 0.975, axis=0).astype(np.float32),
    )

# -------------------------
# MAIN
# -------------------------
paths = get_paths(DATASET, TIMECOURSE_FOLDER, COGNITIVE_FILE, ANAT_LABELS_FILE)
mc_dir = Path(paths["mc"])
dist_dir = mc_dir / MC_DIST_DIRNAME

fp6ag_path = dist_dir / FP6AG_NAME
if not fp6ag_path.exists():
    raise FileNotFoundError(fp6ag_path)
z = np.load(fp6ag_path, allow_pickle=True)

groups = list(z["groups"])
bins = np.linspace(BINS_MIN, BINS_MAX, NBINS, dtype=np.float32)
bin_centers = ((bins[:-1] + bins[1:]) * 0.5).astype(np.float32)

# conditions to compute
CONDS = [
  "obs_intra_trimer","obs_inter_trimer","obs_intra_tetramer","obs_inter_tetramer",
  "null_intra_trimer","null_inter_trimer","null_intra_tetramer","null_inter_tetramer",
  # derived marginals you’ll likely want:
  "obs_all","null_all","obs_intra","obs_inter","obs_trimer","obs_tetramer","null_intra","null_inter","null_trimer","null_tetramer"
]

def get_group_cond_obj(g: str, key: str) -> np.ndarray:
    # atomic keys are stored directly; derived we build on the fly from atomic
    if key in ("obs_all","null_all","obs_intra","obs_inter","obs_trimer","obs_tetramer","null_intra","null_inter","null_trimer","null_tetramer"):
        # load atomics
        o_it = z[f"{g}__obs_intra_trimer"]; o_et = z[f"{g}__obs_inter_trimer"]
        o_ia = z[f"{g}__obs_intra_tetramer"]; o_ea = z[f"{g}__obs_inter_tetramer"]
        n_it = z[f"{g}__null_intra_trimer"]; n_et = z[f"{g}__null_inter_trimer"]
        n_ia = z[f"{g}__null_intra_tetramer"]; n_ea = z[f"{g}__null_inter_tetramer"]

        # helper: per-animal concat
        def cat(a, b):
            out = np.empty(a.size, dtype=object)
            for i in range(a.size):
                xa = _as_float_1d(a[i]); xb = _as_float_1d(b[i])
                if xa.size and xb.size: out[i] = np.concatenate([xa, xb])
                elif xa.size: out[i] = xa
                elif xb.size: out[i] = xb
                else: out[i] = np.array([], dtype=np.float32)
            return out

        if key == "obs_all":   return cat(cat(o_it, o_et), cat(o_ia, o_ea))
        if key == "null_all":  return cat(cat(n_it, n_et), cat(n_ia, n_ea))
        if key == "obs_intra": return cat(o_it, o_ia)
        if key == "obs_inter": return cat(o_et, o_ea)
        if key == "obs_trimer":return cat(o_it, o_et)
        if key == "obs_tetramer":return cat(o_ia, o_ea)
        if key == "null_intra":return cat(n_it, n_ia)
        if key == "null_inter":return cat(n_et, n_ea)
        if key == "null_trimer":return cat(n_it, n_et)
        if key == "null_tetramer":return cat(n_ia, n_ea)

    return z[f"{g}__{key}"]

jobs = []
for gi, g in enumerate(groups):
    for ci, key in enumerate(CONDS):
        obj_arr = get_group_cond_obj(g, key)
        jobs.append((g, key, obj_arr, SEED + 10_000*gi + 100*ci))

results = Parallel(n_jobs=N_JOBS, backend=JOBLIB_BACKEND)(
    delayed(summarize_group_condition)(g, key, obj_arr, seed, bins, P_GRID)
    for (g, key, obj_arr, seed) in jobs
)

out = {
    "bins": bins,
    "bin_centers": bin_centers,
    "p_grid": P_GRID.astype(np.float32),
    "groups": np.array(groups, dtype=object),
}

for r in results:
    g = r["g"]; key = r["key"]
    for f in ["pdf_obs","pdf_ci_lo","pdf_ci_hi","q_obs","q_ci_lo","q_ci_hi","n_animals","n_pool"]:
        out[f"{g}__{key}__{f}"] = r[f]

params = dict(
    dataset=DATASET,
    fp6ag_path=str(fp6ag_path),
    n_boot=int(N_BOOT),
    seed=int(SEED),
    boot_unit="animal_within_group",
    boot_draws=None if BOOT_DRAWS is None else int(BOOT_DRAWS),
    bins=[float(BINS_MIN), float(BINS_MAX), int(NBINS)],
    conditions=CONDS,
    groups=groups,
)
out["params_json"] = json.dumps(params, sort_keys=True)

out_path = dist_dir / OUT_NAME
if out_path.exists() and not OVERWRITE:
    raise FileExistsError(out_path)
np.savez_compressed(out_path, **out)
print("[OK] Saved FP6bG:", out_path)
