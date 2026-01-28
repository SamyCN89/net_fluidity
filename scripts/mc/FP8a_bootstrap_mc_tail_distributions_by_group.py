#!/usr/bin/env python3
# %%
"""
FP8a — Group-conditioned bootstrap summaries of MC distributions (PDF + quantiles)
using FP6a pooled per-animal vectors and FP6b FAST logic.

Consumes:
  - results/<dataset>/mc/mc_dist/fp6a_mc_intra_inter_trimer_tetramer_per_animal.npz
  - preprocessed/ts_and_meta_<dataset>.npz  (mouse_ids_ts, age_ts)
  - preprocessed/cog_data_filtered_*.csv    (per-mouse genotype/sex)

Produces:
  - results/<dataset>/mc/mc_dist/fp8a_bootstrap_mc_tail_distributions_by_group.npz

Notes:
  - No MC recomputation.
  - No null recomputation.
  - Bootstraps by resampling ANIMALS *within each group*.
  - Keeps event-level distributions; does NOT touch FP7 attribution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
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

# Bootstrap
N_BOOT = 2000
SEED = 0
EDGE_SUBSAMPLE = 300_000

# Summaries
P_GRID = np.linspace(0, 1, 101).astype(np.float32)
BINS = np.linspace(-1, 1, 401).astype(np.float32)

# Parallelism: parallel over GROUP×CONDITION jobs
N_JOBS = -1

# Group inclusion switches
INCLUDE_GENOTYPE_X_SEX = True
INCLUDE_AGE_X_SEX_X_GENOTYPE = True  # NOT recommended unless you know groups are big enough

# Safety threshold (skip tiny groups)
MIN_ANIMALS_PER_GROUP = 3


# =========================
# Helpers (same core idea as FP6b FAST)
# =========================
def find_latest(folder: Path, pattern: str) -> Path:
    hits = sorted(folder.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No files matching {pattern} in {folder}")
    return hits[-1]


def _as_float_1d(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.size == 0:
        return np.array([], dtype=np.float32)
    arr = arr.astype(np.float32, copy=False).ravel()
    arr = arr[np.isfinite(arr)]
    return arr


def _pdf_density_from_counts(counts: np.ndarray, bins: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return np.full_like(counts, np.nan, dtype=np.float32)
    widths = np.diff(bins).astype(np.float32)
    return (counts.astype(np.float32) / (n * widths)).astype(np.float32)


def _summaries_from_sample(x: np.ndarray, bins: np.ndarray, p_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        pdf = np.full(bins.size - 1, np.nan, dtype=np.float32)
        q = np.full(p_grid.size, np.nan, dtype=np.float32)
        return pdf, q
    counts, _ = np.histogram(x, bins=bins, density=False)
    pdf = _pdf_density_from_counts(counts, bins, x.size)
    q = np.quantile(x, p_grid).astype(np.float32)
    return pdf, q


def _sample_from_animals(values: List[np.ndarray], pick_animals: np.ndarray, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    A = len(values)
    if A == 0:
        return np.array([], dtype=np.float32)

    lens = np.fromiter((values[i].size for i in pick_animals), dtype=np.int64, count=pick_animals.size)
    tot = int(lens.sum())
    if tot == 0:
        return np.array([], dtype=np.float32)

    if tot <= n_draws:
        return np.concatenate([values[i] for i in pick_animals if values[i].size], axis=0).astype(np.float32, copy=False)

    probs = lens / tot
    draws = rng.multinomial(n_draws, probs)

    out = np.empty(n_draws, dtype=np.float32)
    pos = 0
    for j, n_j in enumerate(draws):
        if n_j == 0:
            continue
        a_idx = int(pick_animals[j])
        v = values[a_idx]
        if v.size == 0:
            continue
        ii = rng.integers(0, v.size, size=n_j)
        out[pos : pos + n_j] = v[ii]
        pos += n_j

    return out[:pos]


def summarize_condition_group_fast(
    key: str,
    values_per_animal_obj: np.ndarray,
    animal_mask: np.ndarray,   # length A_total, True for animals in group
    n_boot: int,
    seed: int,
    bins: np.ndarray,
    p_grid: np.ndarray,
    edge_subsample: int,
) -> Dict[str, np.ndarray]:
    """
    Same as FP6b FAST, but restricted to a subset of animals via animal_mask.
    """
    # Filter animals to group
    idx = np.where(animal_mask)[0]
    if idx.size == 0:
        raise RuntimeError("Empty group mask.")

    values_all = list(values_per_animal_obj)
    values = [_as_float_1d(values_all[i]) for i in idx]
    A = len(values)

    rng_obs = np.random.default_rng(seed + 999)
    pick_all = np.arange(A, dtype=np.int64)
    x_obs = _sample_from_animals(values, pick_all, edge_subsample, rng_obs)
    pdf_obs, q_obs = _summaries_from_sample(x_obs, bins, p_grid)

    nb = bins.size - 1
    nq = p_grid.size
    pdf_boot = np.empty((n_boot, nb), dtype=np.float32)
    q_boot = np.empty((n_boot, nq), dtype=np.float32)

    rng = np.random.default_rng(seed)
    for b in range(n_boot):
        pick = rng.integers(0, A, size=A)
        x = _sample_from_animals(values, pick, edge_subsample, rng)
        pdf_boot[b], q_boot[b] = _summaries_from_sample(x, bins, p_grid)

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
# Group specs
# =========================
@dataclass(frozen=True)
class GroupSpec:
    name: str
    mask: np.ndarray  # boolean mask over animals (A_total,)


def build_groups(
    mouse_ids_ts: np.ndarray,
    age_ts: np.ndarray,
    cog: pd.DataFrame,
    include_gxsex: bool,
    include_3way: bool,
) -> List[GroupSpec]:
    """
    Build group masks over *sessions/animals in ts stack* (A_total rows).
    Uses:
      - age_ts per session
      - genotype/sex per mouse from cog (per unique Name)
    """
    # map mouse -> genotype/sex
    cog = cog.copy()
    cog["Name"] = cog["Name"].astype(str)
    g_map = dict(zip(cog["Name"].astype(str), cog["Genotype"].astype(str)))
    s_map = dict(zip(cog["Name"].astype(str), cog["Sexe"].astype(str)))

    genotype_ts = np.array([g_map.get(mid, "NA") for mid in mouse_ids_ts], dtype=str)
    sex_ts = np.array([s_map.get(mid, "NA") for mid in mouse_ids_ts], dtype=str)

    groups: List[GroupSpec] = []

    # Main effects
    for age in ["2m", "4m"]:
        m = (age_ts == age)
        groups.append(GroupSpec(name=f"age={age}", mask=m))

    for g in ["wt", "dKI"]:
        m = (genotype_ts == g)
        groups.append(GroupSpec(name=f"genotype={g}", mask=m))

    for s in ["F", "M"]:
        m = (sex_ts == s)
        groups.append(GroupSpec(name=f"sex={s}", mask=m))

    # Interactions
    for age in ["2m", "4m"]:
        for g in ["wt", "dKI"]:
            m = (age_ts == age) & (genotype_ts == g)
            groups.append(GroupSpec(name=f"age={age}&genotype={g}", mask=m))

    for age in ["2m", "4m"]:
        for s in ["F", "M"]:
            m = (age_ts == age) & (sex_ts == s)
            groups.append(GroupSpec(name=f"age={age}&sex={s}", mask=m))

    if include_gxsex:
        for s in ["F", "M"]:
            for g in ["wt", "dKI"]:
                m = (sex_ts == s) & (genotype_ts == g)
                groups.append(GroupSpec(name=f"sex={s}&genotype={g}", mask=m))

    if include_3way:
        for age in ["2m", "4m"]:
            for s in ["F", "M"]:
                for g in ["wt", "dKI"]:
                    m = (age_ts == age) & (sex_ts == s) & (genotype_ts == g)
                    groups.append(GroupSpec(name=f"age={age}&sex={s}&genotype={g}", mask=m))

    return groups


# =========================
# MAIN
# =========================
paths = get_paths(
    dataset_name=DATASET,
    timecourse_folder=TIMECOURSE_FOLDER,
    cognitive_data_file=COGNITIVE_FILE,
    anat_labels_file=ANAT_LABELS_FILE,
)
preproc_dir = Path(paths["preprocessed"])
mc_dir = Path(paths["mc"])

# Load FP6a
fp6a_path = mc_dir / OUT_SUBDIR / FP6A_NAME
if not fp6a_path.exists():
    raise FileNotFoundError(fp6a_path)

d6a = np.load(fp6a_path, allow_pickle=True)

conditions = [
    "obs_intra_trimer",
    "obs_inter_trimer",
    "obs_intra_tetramer",
    "obs_inter_tetramer",
    "null_intra_trimer",
    "null_inter_trimer",
    "null_intra_tetramer",
    "null_inter_tetramer",
]

# Derived marginals (optional but useful for FP8b later)
def _concat_per_animal(a_list, b_list) -> np.ndarray:
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

base = {k: d6a[k] for k in conditions}
derived = {
    "obs_all": _concat_per_animal(_concat_per_animal(base["obs_intra_trimer"], base["obs_inter_trimer"]),
                                 _concat_per_animal(base["obs_intra_tetramer"], base["obs_inter_tetramer"])),
    "null_all": _concat_per_animal(_concat_per_animal(base["null_intra_trimer"], base["null_inter_trimer"]),
                                  _concat_per_animal(base["null_intra_tetramer"], base["null_inter_tetramer"])),
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

# Determine A_total from FP6a arrays
A_total = len(next(iter(all_conditions.values())))
print("[FP8a] Loaded FP6a:", fp6a_path.name, "| A_total:", A_total, "| conditions:", len(all_conditions))

# Load canonical session metadata
canon_npz = preproc_dir / f"ts_and_meta_{DATASET}.npz"
if not canon_npz.exists():
    # fallback: latest ts_and_meta_*.npz
    canon_npz = find_latest(preproc_dir, "ts_and_meta_*.npz")
d0 = np.load(canon_npz, allow_pickle=True)
mouse_ids_ts = d0["mouse_ids_ts"].astype(str)
age_ts = d0["age_ts"].astype(str)
if mouse_ids_ts.shape[0] != A_total:
    raise RuntimeError(f"mouse_ids_ts length {mouse_ids_ts.shape[0]} != FP6a A_total {A_total}")

# Load cognitive table
cog_csv = find_latest(preproc_dir, "cog_data_filtered_*.csv")
cog = pd.read_csv(cog_csv)
cog["Name"] = cog["Name"].astype(str)

# Build groups
groups = build_groups(
    mouse_ids_ts=mouse_ids_ts,
    age_ts=age_ts,
    cog=cog,
    include_gxsex=INCLUDE_GENOTYPE_X_SEX,
    include_3way=INCLUDE_AGE_X_SEX_X_GENOTYPE,
)

# Filter tiny groups
kept = []
for g in groups:
    n = int(g.mask.sum())
    if n >= MIN_ANIMALS_PER_GROUP:
        kept.append(g)
    else:
        print(f"[SKIP] {g.name} (n_animals={n} < {MIN_ANIMALS_PER_GROUP})")
groups = kept

print("[FP8a] Groups kept:", len(groups))

# Prepare jobs: (group, condition)
jobs = []
for gi, g in enumerate(groups):
    for ci, (cond_key, cond_vals) in enumerate(all_conditions.items()):
        jobs.append((gi, g, cond_key, cond_vals, ci))

print("[FP8a] Jobs:", len(jobs), f"(groups={len(groups)} × conditions={len(all_conditions)})")

def _run_job(gi, gspec: GroupSpec, cond_key: str, cond_vals: np.ndarray, ci: int):
    seed = SEED + 10_000 * gi + 100 * ci
    res = summarize_condition_group_fast(
        key=f"{gspec.name}::{cond_key}",
        values_per_animal_obj=cond_vals,
        animal_mask=gspec.mask,
        n_boot=N_BOOT,
        seed=seed,
        bins=BINS,
        p_grid=P_GRID,
        edge_subsample=EDGE_SUBSAMPLE,
    )
    res["group"] = gspec.name
    res["condition"] = cond_key
    return res

results = Parallel(n_jobs=N_JOBS, backend="loky")(
    delayed(_run_job)(gi, g, cond_key, cond_vals, ci)
    for (gi, g, cond_key, cond_vals, ci) in jobs
)

# Save: store per (group,condition) series as keyed arrays
out_dir = mc_dir / OUT_SUBDIR
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "fp8a_bootstrap_mc_tail_distributions_by_group.npz"
if out_path.exists() and not OVERWRITE:
    raise FileExistsError(out_path)

save = {
    "bins": BINS.astype(np.float32),
    "p_grid": P_GRID.astype(np.float32),
    "bin_centers": ((BINS[:-1] + BINS[1:]) * 0.5).astype(np.float32),
    "groups": np.array([g.name for g in groups], dtype=str),
    "conditions": np.array(list(all_conditions.keys()), dtype=str),
}

for r in results:
    key = r["key"]
    for field in ["pdf_obs", "pdf_ci_lo", "pdf_ci_hi", "q_obs", "q_ci_lo", "q_ci_hi", "n_animals", "n_obs_used"]:
        save[f"{key}__{field}"] = r[field]

params = dict(
    dataset=DATASET,
    fp6a_path=str(fp6a_path),
    canon_npz=str(canon_npz),
    cog_csv=str(cog_csv),
    n_boot=int(N_BOOT),
    seed=int(SEED),
    edge_subsample=int(EDGE_SUBSAMPLE),
    bins=[float(x) for x in BINS.tolist()],
    p_grid=[float(x) for x in P_GRID.tolist()],
    include_genotype_x_sex=bool(INCLUDE_GENOTYPE_X_SEX),
    include_age_x_sex_x_genotype=bool(INCLUDE_AGE_X_SEX_X_GENOTYPE),
    min_animals_per_group=int(MIN_ANIMALS_PER_GROUP),
    groups=[g.name for g in groups],
    conditions=list(all_conditions.keys()),
)

save["params_json"] = json.dumps(params, sort_keys=True)

np.savez_compressed(out_path, **save)
print("[OK] Saved FP8a:", out_path)
