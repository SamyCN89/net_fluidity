#!/usr/bin/env python3
# %%
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from shared_code.fun_paths import get_paths

# =========================
# CONFIG
# =========================
DATASET = "ines_abdallah"
TIMECOURSE_FOLDER = "Timecourses_updated_03052024"
COGNITIVE_FILE = "ROIs.xlsx"
ANAT_LABELS_FILE = "41_Allen.txt"

# FP3 input (set explicitly)
FP3_PATH = "/media/samy/Elements2/Proyectos/LauraHarsan/results/ines_abdallah/mc/mc_indexed/mc_indexed_ref=wt_2m_animals=126_E=820.npz"

# One-sample bootstrap
N_BOOT = 20
SEED = 0
BINS = np.linspace(-0.8, 1.0, 120)

# Compare within each age, genotype first
AGES = ["2m", "4m"]
GENOTYPES = ["wt", "dKI"]

# Feature subset
SUBSET = "all"      # "all" | "module" | "trimer"
MODULE_ID = 1
TRIMER_ONLY = True

# Output
OUT_SUBDIR = "mc_uncertainty_single_group"
OVERWRITE = False

# =========================
# Helpers
# =========================
def _tail_metrics(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0)
    q01, q05, q50, q95, q99 = np.quantile(x, [0.01, 0.05, 0.5, 0.95, 0.99])
    return dict(
        n=int(x.size),
        q01=float(q01), q05=float(q05), q50=float(q50), q95=float(q95), q99=float(q99),
        width50=float(q95 - q05),
        width_extreme=float(q99 - q01),
        asymmetry=float((q95 - q50) - (q50 - q05)),
    )

def _pdf(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
    x = x[np.isfinite(x)]
    h, _ = np.histogram(x, bins=bins, density=True)
    return h.astype(np.float32)

def _feature_mask(mc_mod_idx: np.ndarray, mc_nplets_index: np.ndarray) -> np.ndarray:
    K = mc_mod_idx.shape[0]
    m = np.ones(K, dtype=bool)
    if SUBSET == "module":
        m &= (mc_mod_idx == MODULE_ID)
    elif SUBSET == "trimer":
        m &= (mc_nplets_index > 0) if TRIMER_ONLY else (mc_nplets_index >= 0)
    return m

def _one_sample_bootstrap_sessionwise(
    mc_val: np.ndarray,         # (A, K)
    session_mask: np.ndarray,   # (A,)
    feat_mask: np.ndarray,      # (K,)
    bins: np.ndarray,
    n_boot: int,
    seed: int,
):
    rng = np.random.default_rng(seed)

    X = mc_val[session_mask][:, feat_mask]   # (n_sessions, n_feat)
    n_sessions = X.shape[0]
    if n_sessions == 0:
        raise RuntimeError("Empty group (no sessions).")
    # observed
    x_obs = X.ravel()
    pdf_obs = _pdf(x_obs, bins)
    tail_obs = _tail_metrics(x_obs)

    # bootstrap replicates
    pdf_boot = np.empty((n_boot, pdf_obs.size), dtype=np.float32)
    tails_boot = {k: np.empty(n_boot, dtype=np.float32) for k in tail_obs.keys() if k != "n"}

    for b in range(n_boot):
        idx = rng.integers(0, n_sessions, size=n_sessions)  # resample sessions
        x_b = X[idx].ravel()
        pdf_boot[b] = _pdf(x_b, bins)
        t_b = _tail_metrics(x_b)
        for k in tails_boot:
            tails_boot[k][b] = t_b[k]

    # CIs
    pdf_lo = np.quantile(pdf_boot, 0.025, axis=0)
    pdf_hi = np.quantile(pdf_boot, 0.975, axis=0)

    tails_ci = {}
    for k, arr in tails_boot.items():
        lo, hi = np.quantile(arr, [0.025, 0.975])
        tails_ci[k] = dict(lo=float(lo), hi=float(hi), mean=float(np.mean(arr)))

    return dict(
        n_sessions=int(n_sessions),
        pdf_obs=pdf_obs,
        pdf_ci_lo=pdf_lo.astype(np.float32),
        pdf_ci_hi=pdf_hi.astype(np.float32),
        tail_obs=tail_obs,
        tail_ci=tails_ci,
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
preproc_dir = Path(paths["preprocessed"])
mc_dir = Path(paths["mc"])

# Load FP3
d3 = np.load(FP3_PATH, allow_pickle=True)
mc_val = d3["mc_val_tril"]                 # (A, K)
mc_mod_idx = d3["mc_mod_idx"]              # (K,)
mc_nplets_index = d3["mc_nplets_index"]    # (K,)

feat_mask = _feature_mask(mc_mod_idx, mc_nplets_index)

# Load session labels (from canonical FP0 bundle)
canon_npz = preproc_dir / "ts_and_meta_ines_abdallah.npz"
d0 = np.load(canon_npz, allow_pickle=True)
mouse_ids_ts = d0["mouse_ids_ts"].astype(str)   # (A,)
age_ts = d0["age_ts"].astype(str)               # (A,)

# Load per-mouse genotype map
cog_csv = sorted(preproc_dir.glob("cog_data_filtered_*.csv"))[-1]
cog = pd.read_csv(cog_csv)
geno_map = dict(zip(cog["Name"].astype(str), cog["Genotype"].astype(str)))
genotype_ts = np.array([geno_map.get(mid, "NA") for mid in mouse_ids_ts], dtype=str)

# Run all groups
out_dir = mc_dir / OUT_SUBDIR
out_dir.mkdir(parents=True, exist_ok=True)

results = {}
for age in AGES:
    for gen in GENOTYPES:
        session_mask = (age_ts == age) & (genotype_ts == gen)
        key = f"{gen}_{age}"
        print("Running:", key, "n_sessions=", int(session_mask.sum()))

        res = _one_sample_bootstrap_sessionwise(
            mc_val=mc_val,
            session_mask=session_mask,
            feat_mask=feat_mask,
            bins=BINS,
            n_boot=N_BOOT,
            seed=SEED,
        )
        results[key] = res

# Save one artifact containing all groups
out_path = out_dir / f"fp4_uncertainty_genotype_by_age_subset={SUBSET}_boot={N_BOOT}.npz"
if out_path.exists() and not OVERWRITE:
    raise FileExistsError(f"{out_path} exists. Set OVERWRITE=True to replace.")

payload = {
    "dataset": DATASET,
    "fp3_path": str(FP3_PATH),
    "canon_npz": str(canon_npz),
    "cog_csv": str(cog_csv),
    "subset": SUBSET,
    "module_id": int(MODULE_ID),
    "trimer_only": bool(TRIMER_ONLY),
    "n_boot": int(N_BOOT),
    "seed": int(SEED),
    "bins": BINS.tolist(),
}

np.savez_compressed(
    out_path,
    bins=BINS,
    params_json=json.dumps(payload, sort_keys=True),

    # store per-group arrays
    group_keys=np.array(list(results.keys()), dtype=str),
    n_sessions=np.array([results[k]["n_sessions"] for k in results], dtype=int),

    pdf_obs=np.stack([results[k]["pdf_obs"] for k in results], axis=0),
    pdf_ci_lo=np.stack([results[k]["pdf_ci_lo"] for k in results], axis=0),
    pdf_ci_hi=np.stack([results[k]["pdf_ci_hi"] for k in results], axis=0),

    tail_obs_json=np.array([json.dumps(results[k]["tail_obs"], sort_keys=True) for k in results], dtype=object),
    tail_ci_json=np.array([json.dumps(results[k]["tail_ci"], sort_keys=True) for k in results], dtype=object),
)

print("[OK] Saved FP4:", out_path)
# %%
