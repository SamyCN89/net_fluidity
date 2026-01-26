#!/usr/bin/env python3
# %%
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

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

# FP3 indexed input (your FP3 output containing mc_val_tril etc.)
FP3_PATH = "/media/samy/Elements2/Proyectos/LauraHarsan/results/ines_abdallah/mc/mc_indexed/mc_indexed_ref=wt_2m_animals=126_E=820.npz"

# Bootstrap
N_BOOT = 2000
SEED = 0
N_JOBS = -1          # uses all cores
CHUNK = 100          # bootstrap replicates per worker task
EDGE_SUBSAMPLE = 200_000   # try 200k–500k

# Histogram support
BINS = np.linspace(-0.8, 1.0, 120)  # adjust if needed

# Feature subset
SUBSET = "all"       # "all" | "module" | "trimer"
MODULE_ID = 1
TRIMER_ONLY = True

# Output
OUT_SUBDIR = "mc_uncertainty_single_group_allcombos"
OVERWRITE = True

#%%
# =========================
# Metrics
# =========================
METRIC_NAMES = ["q01","q05","q50","q95","q99","width50","width_extreme","tail_imbalance"]


def _tail_metrics_vec(x: np.ndarray) -> np.ndarray:
    """Return vector in METRIC_NAMES order."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.full(len(METRIC_NAMES), np.nan, dtype=np.float32)
    q01, q05, q50, q95, q99 = np.quantile(x, [0.01, 0.05, 0.5, 0.95, 0.99])
    width50 = q95 - q05
    width_extreme = q99 - q01
    asym = (q95 - q50) - (q50 - q05)
    return np.array([q01, q05, q50, q95, q99, width50, width_extreme, asym], dtype=np.float32)

def _pdf(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
    x = x[np.isfinite(x)]
    h, _ = np.histogram(x, bins=bins, density=True)
    return h.astype(np.float32)

#%%
# =========================
# Group specs
# =========================
@dataclass(frozen=True)
class GroupSpec:
    age: str
    name: str
    selector: Callable[[pd.DataFrame], np.ndarray]  # returns mouse_ids (names) in group


def _mk_specs_all(age: str) -> list[GroupSpec]:
    """All requested group families, within this age."""
    specs: list[GroupSpec] = []

    # Genotype
    for g in ["wt", "dKI"]:
        specs.append(GroupSpec(
            age=age,
            name=f"genotype={g}",
            selector=lambda df, g=g: df.loc[df["Genotype"].astype(str) == g, "Name"].astype(str).to_numpy(),
        ))

    # Sex
    for s in ["F", "M"]:
        specs.append(GroupSpec(
            age=age,
            name=f"sex={s}",
            selector=lambda df, s=s: df.loc[df["Sexe"].astype(str) == s, "Name"].astype(str).to_numpy(),
        ))

    # Phenotype OiP
    for p in ["good", "impaired", "learners", "bad"]:
        specs.append(GroupSpec(
            age=age,
            name=f"phen_oip={p}",
            selector=lambda df, p=p: df.loc[df["Phenotype_OiP"].astype(str) == p, "Name"].astype(str).to_numpy(),
        ))

    # Phenotype RO24h
    for p in ["good", "impaired", "learners", "bad"]:
        specs.append(GroupSpec(
            age=age,
            name=f"phen_ro24h={p}",
            selector=lambda df, p=p: df.loc[df["Phenotype_RO24h"].astype(str) == p, "Name"].astype(str).to_numpy(),
        ))

    # Sex × Genotype
    for s in ["F", "M"]:
        for g in ["wt", "dKI"]:
            specs.append(GroupSpec(
                age=age,
                name=f"sex={s}&genotype={g}",
                selector=lambda df, s=s, g=g: df.loc[
                    (df["Sexe"].astype(str) == s) & (df["Genotype"].astype(str) == g),
                    "Name"
                ].astype(str).to_numpy(),
            ))

    # Sex × Phenotype OiP
    for s in ["F", "M"]:
        for p in ["good", "impaired", "learners", "bad"]:
            specs.append(GroupSpec(
                age=age,
                name=f"sex={s}&phen_oip={p}",
                selector=lambda df, s=s, p=p: df.loc[
                    (df["Sexe"].astype(str) == s) & (df["Phenotype_OiP"].astype(str) == p),
                    "Name"
                ].astype(str).to_numpy(),
            ))

    # Sex × Phenotype RO24h
    for s in ["F", "M"]:
        for p in ["good", "impaired", "learners", "bad"]:
            specs.append(GroupSpec(
                age=age,
                name=f"sex={s}&phen_ro24h={p}",
                selector=lambda df, s=s, p=p: df.loc[
                    (df["Sexe"].astype(str) == s) & (df["Phenotype_RO24h"].astype(str) == p),
                    "Name"
                ].astype(str).to_numpy(),
            ))

    return specs


# =========================
# Feature mask
# =========================
def _feature_mask(mc_mod_idx: np.ndarray, mc_nplets_index: np.ndarray) -> np.ndarray:
    K = mc_mod_idx.shape[0]
    m = np.ones(K, dtype=bool)
    if SUBSET == "module":
        m &= (mc_mod_idx == MODULE_ID)
    elif SUBSET == "trimer":
        m &= (mc_nplets_index > 0) if TRIMER_ONLY else (mc_nplets_index >= 0)
    return m


# =========================
# Bootstrap engine (parallel)
# =========================
def _bootstrap_chunk(X: np.ndarray, bins: np.ndarray, n_rep: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    X: (n_sessions, n_feat)
    Returns:
      pdfs: (n_rep, n_bins-1)
      tails: (n_rep, n_metrics)
    """
    rng = np.random.default_rng(seed)
    n_sessions = X.shape[0]
    nb = bins.size - 1

    pdfs = np.empty((n_rep, nb), dtype=np.float32)
    tails = np.empty((n_rep, len(METRIC_NAMES)), dtype=np.float32)

    n_feat = X.shape[1]
    x = np.empty(n_sessions * n_feat, dtype=X.dtype)

    for i in range(n_rep):
        idx = rng.integers(0, n_sessions, size=n_sessions)
        x[:] = X[idx].reshape(-1)        # reuses the same buffer
        if EDGE_SUBSAMPLE is not None and EDGE_SUBSAMPLE < x.size:
            jj = rng.integers(0, x.size, size=EDGE_SUBSAMPLE)
            x_use = x[jj]
        else:
            x_use = x

        pdfs[i] = _pdf(x_use, bins)
        tails[i] = _tail_metrics_vec(x_use)

    return pdfs, tails


def one_sample_uncertainty_session_bootstrap(
    mc_val: np.ndarray,               # (A, K)
    session_mask: np.ndarray,         # (A,)
    feat_mask: np.ndarray,            # (K,)
    bins: np.ndarray,
    n_boot: int,
    seed: int,
    n_jobs: int,
    chunk: int,
) -> dict:
    X = mc_val[session_mask][:, feat_mask]  # (n_sessions, n_feat)
    n_sessions = int(X.shape[0])
    if n_sessions == 0:
        raise RuntimeError("Empty group (no sessions).")

    # Observed
    x_obs = X.ravel()
    pdf_obs = _pdf(x_obs, bins)
    tail_obs = _tail_metrics_vec(x_obs)

    # Parallel bootstrap in chunks
    n_chunks = int(np.ceil(n_boot / chunk))
    reps = [chunk] * n_chunks
    reps[-1] = n_boot - chunk * (n_chunks - 1)

    # Different seed per chunk
    chunk_seeds = [seed + 10_000 * j for j in range(n_chunks)]

    if n_jobs == 1:
        out = [_bootstrap_chunk(X, bins, reps[j], chunk_seeds[j]) for j in range(n_chunks)]
    else:
        out = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_bootstrap_chunk)(X, bins, reps[j], chunk_seeds[j])
            for j in range(n_chunks)
        )

    pdf_boot = np.concatenate([p for (p, _) in out], axis=0)
    tail_boot = np.concatenate([t for (_, t) in out], axis=0)

    # CIs
    pdf_lo = np.quantile(pdf_boot, 0.025, axis=0).astype(np.float32)
    pdf_hi = np.quantile(pdf_boot, 0.975, axis=0).astype(np.float32)

    tail_lo = np.quantile(tail_boot, 0.025, axis=0).astype(np.float32)
    tail_hi = np.quantile(tail_boot, 0.975, axis=0).astype(np.float32)

    return dict(
        n_sessions=n_sessions,
        pdf_obs=pdf_obs,
        pdf_ci_lo=pdf_lo,
        pdf_ci_hi=pdf_hi,
        tail_obs=tail_obs,
        tail_ci_lo=tail_lo,
        tail_ci_hi=tail_hi,
    )

def run_one_group(spec: GroupSpec):
    mice = spec.selector(cog)
    session_mask = (age_ts == spec.age) & np.isin(mouse_ids_ts, mice)
    n_sessions = int(session_mask.sum())
    key = f"{spec.age}::{spec.name}"
    print(f"FP4 running {key}  n_sessions={n_sessions}")

    if n_sessions == 0:
        return None

    res = one_sample_uncertainty_session_bootstrap(
        mc_val=mc_val,
        session_mask=session_mask,
        feat_mask=feat_mask,
        bins=BINS,
        n_boot=N_BOOT,
        seed=SEED,
        n_jobs=1,          # IMPORTANT: avoid nested parallel
        chunk=CHUNK,
    )
    res["key"] = key
    res["age"] = spec.age
    res["name"] = spec.name
    return res


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

# Load FP3
d3 = np.load(FP3_PATH, allow_pickle=True)
mc_val = d3["mc_val_tril"]                 # (A, K)
mc_mod_idx = d3["mc_mod_idx"]              # (K,)
mc_nplets_index = d3["mc_nplets_index"]    # (K,)

feat_mask = _feature_mask(mc_mod_idx, mc_nplets_index)

# Load session metadata from canonical FP0 bundle
canon_npz = preproc_dir / "ts_and_meta_ines_abdallah.npz"
d0 = np.load(canon_npz, allow_pickle=True)
mouse_ids_ts = d0["mouse_ids_ts"].astype(str)   # (A,)
age_ts = d0["age_ts"].astype(str)               # (A,)

# Load cog table (per mouse)
cog_csv = sorted(preproc_dir.glob("cog_data_filtered_*.csv"))[-1]
cog = pd.read_csv(cog_csv)
cog["Name"] = cog["Name"].astype(str)

# Precompute a fast mouse->row lookup? We just use selectors on the df (small), OK.

# Build group specs for both ages
specs: list[GroupSpec] = []
for age in ["2m", "4m"]:
    specs.extend(_mk_specs_all(age))

import time

t0 = time.time()

# Run all groups in parallel (no nested parallelism inside each group)
results = Parallel(n_jobs=N_JOBS, backend="loky")(
    delayed(run_one_group)(spec) for spec in specs
)
results = [r for r in results if r is not None]

t1 = time.time()
print(f"[TIMER] FP4 group-parallel done in {t1 - t0:.1f}s | kept {len(results)}/{len(specs)} groups")

# Rebuild arrays for saving
group_keys = [r["key"] for r in results]
group_age  = [r["age"] for r in results]
group_name = [r["name"] for r in results]
n_sessions_list = [int(r["n_sessions"]) for r in results]

pdf_obs_list = [r["pdf_obs"] for r in results]
pdf_lo_list  = [r["pdf_ci_lo"] for r in results]
pdf_hi_list  = [r["pdf_ci_hi"] for r in results]

tail_obs_list = [r["tail_obs"] for r in results]
tail_lo_list  = [r["tail_ci_lo"] for r in results]
tail_hi_list  = [r["tail_ci_hi"] for r in results]
# =========================

# Save one artifact
out_dir = mc_dir / OUT_SUBDIR
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / f"fp4_uncertainty_allcombos_subset={SUBSET}_boot={N_BOOT}.npz"
if out_path.exists() and not OVERWRITE:
    raise FileExistsError(f"{out_path} exists. Set OVERWRITE=True to replace.")

params = dict(
    dataset=DATASET,
    fp3_path=str(FP3_PATH),
    canon_npz=str(canon_npz),
    cog_csv=str(cog_csv),
    subset=SUBSET,
    module_id=int(MODULE_ID),
    trimer_only=bool(TRIMER_ONLY),
    bins=BINS.tolist(),
    n_boot=int(N_BOOT),
    seed=int(SEED),
    n_jobs=int(N_JOBS),
    chunk=int(CHUNK),
    metric_names=METRIC_NAMES,
)

np.savez_compressed(
    out_path,
    bins=BINS,
    metric_names=np.array(METRIC_NAMES, dtype=str),
    params_json=json.dumps(params, sort_keys=True),

    group_keys=np.array(group_keys, dtype=str),
    group_age=np.array(group_age, dtype=str),
    group_name=np.array(group_name, dtype=str),
    n_sessions=np.array(n_sessions_list, dtype=int),

    pdf_obs=np.stack(pdf_obs_list, axis=0),
    pdf_ci_lo=np.stack(pdf_lo_list, axis=0),
    pdf_ci_hi=np.stack(pdf_hi_list, axis=0),

    tail_obs=np.stack(tail_obs_list, axis=0),
    tail_ci_lo=np.stack(tail_lo_list, axis=0),
    tail_ci_hi=np.stack(tail_hi_list, axis=0),
)

print("[OK] Saved FP4:", out_path)
# %%
