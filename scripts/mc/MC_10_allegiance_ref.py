#!/usr/bin/env python3
# %%
"""
FP2 — Reference allegiance (no CLI)

Consumes:
  - results/<dataset>/mc_raw/mc_raw_*.npz    (FP1)
  - preprocessed_data/cog_data_filtered_*.csv (per-mouse metadata)

Produces:
  - results/<dataset>/allegiance_ref/allegiance_ref_<ref>_gamma=..._runs=....npz

Does NOT compute MC. Does NOT build modules/trimers. Only allegiance reference.
"""
#%%
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from shared_code.fun_paths import get_paths
from shared_code.fun_metaconnectivity import fun_allegiance_communities
#%%
# =========================
# CONFIG (edit only this)
# =========================
DATASET = "ines_abdallah"
TIMECOURSE_FOLDER = "Timecourses_updated_03052024"
COGNITIVE_FILE = "ROIs.xlsx"
ANAT_LABELS_FILE = "41_Allen.txt"

# Reference definition (simple + explicit)
REF_GENOTYPE = "wt"   # "wt" or "dKI"
REF_AGE = "2m"        # "2m" or "4m"

# Allegiance params
N_RUNS = 1000
GAMMA_PT = 100
N_JOBS = -1

# IO
OUT_SUBDIR = "allegiance_ref"
OVERWRITE = False

# =========================
# Helpers
# =========================
def find_latest(folder: Path, pattern: str) -> Path:
    hits = sorted(folder.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No files matching {pattern} in {folder}")
    return hits[-1]

def blockiness_score(M: np.ndarray, communities: np.ndarray) -> tuple[float, float]:
    """Mean |M| intra vs inter (diag ignored)."""
    M = np.asarray(M).copy()
    np.fill_diagonal(M, np.nan)
    c = np.asarray(communities)
    same = c[:, None] == c[None, :]
    eye = np.eye(M.shape[0], dtype=bool)
    intra = np.abs(M[same & ~eye])
    inter = np.abs(M[~same])
    return float(np.nanmean(intra)), float(np.nanmean(inter))
#%%
# =========================
# MAIN
# =========================
paths = get_paths(
    dataset_name=DATASET,
    timecourse_folder=TIMECOURSE_FOLDER,
    cognitive_data_file=COGNITIVE_FILE,
    anat_labels_file=ANAT_LABELS_FILE,
)
results_dir = Path(paths["mc"])
preproc_dir = Path(paths["preprocessed"])

# --- Load FP1 mc_raw ---
mc_raw_path = find_latest(results_dir / "mc_raw", "mc_raw_*.npz")
d = np.load(mc_raw_path, allow_pickle=True)

mc = d["mc"]                              # (A, E, E)
mouse_ids_ts = d["mouse_ids_ts"].astype(str)  # (A,)
age_ts = d["age_ts"].astype(str)              # (A,)

# --- Load per-mouse cognitive table ---
cog_csv_path = find_latest(preproc_dir, "cog_data_filtered_*.csv")
cog = pd.read_csv(cog_csv_path)
cog["Name"] = cog["Name"].astype(str)

# --- Build reference mask in MC rows (sessions) ---
ref_mice = cog.loc[cog["Genotype"].astype(str) == REF_GENOTYPE, "Name"].to_numpy(dtype=str)
ind_ref = np.isin(mouse_ids_ts, ref_mice) & (age_ts == REF_AGE)
n_ref = int(ind_ref.sum())
if n_ref == 0:
    raise RuntimeError(
        f"Reference empty: genotype={REF_GENOTYPE} age={REF_AGE}. "
        f"Check cog table Genotype values and FP0 mouse_ids_ts/age_ts."
    )

mc_ref = np.nanmean(mc[ind_ref], axis=0)
#%%
# --- Compute allegiance (no side effects) ---
ref_label = f"{REF_GENOTYPE}_{REF_AGE}"
t0 = time.time()
communities, sort_idx, contingency = fun_allegiance_communities(
    mc_ref,
    n_runs=N_RUNS,
    gamma_pt=GAMMA_PT,
    save_path=None,
    ref_name=ref_label,
    n_jobs=N_JOBS,
)
dt = time.time() - t0

communities = np.asarray(communities)
sort_idx = np.asarray(sort_idx)

# --- QC: blockiness should improve after sorting (usually) ---
intra0, inter0 = blockiness_score(mc_ref, communities)

mc_ref_sorted = mc_ref[sort_idx][:, sort_idx]
comm_sorted = communities[sort_idx]
intra1, inter1 = blockiness_score(mc_ref_sorted, comm_sorted)

print("FP2 allegiance QC")
print("  ref:", ref_label, "n_ref_sessions:", n_ref)
print(f"  blockiness unsorted: intra={intra0:.4f} inter={inter0:.4f} (intra-inter={intra0-inter0:.4f})")
print(f"  blockiness sorted  : intra={intra1:.4f} inter={inter1:.4f} (intra-inter={intra1-inter1:.4f})")
print(f"  delta(intra-inter): {(intra1-inter1)-(intra0-inter0):.4f}")
print(f"  time: {dt:.2f}s")

# --- Save ONE FP2 artifact ---
out_dir = results_dir / OUT_SUBDIR
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / f"allegiance_ref_{ref_label}_gamma={GAMMA_PT}_runs={N_RUNS}.npz"
if out_path.exists() and not OVERWRITE:
    raise FileExistsError(f"{out_path} exists. Set OVERWRITE=True to replace.")

params = dict(
    dataset=DATASET,
    ref_label=ref_label,
    ref_genotype=REF_GENOTYPE,
    ref_age=REF_AGE,
    n_runs=N_RUNS,
    gamma_pt=GAMMA_PT,
    n_jobs=N_JOBS,
    mc_raw_path=str(mc_raw_path),
    cog_csv_path=str(cog_csv_path),
)

qc = dict(
    seconds=dt,
    n_ref_sessions=n_ref,
    blockiness=dict(
        unsorted=dict(intra=intra0, inter=inter0, intra_minus_inter=intra0 - inter0),
        sorted=dict(intra=intra1, inter=inter1, intra_minus_inter=intra1 - inter1),
        delta=(intra1 - inter1) - (intra0 - inter0),
    ),
)

np.savez_compressed(
    out_path,
    communities=communities,
    sort_idx=sort_idx,
    contingency=contingency if contingency is not None else np.array([], dtype=float),
    mc_ref=mc_ref,
    ind_ref=ind_ref,
    mouse_ids_ts=mouse_ids_ts,
    age_ts=age_ts,
    params_json=json.dumps(params, sort_keys=True),
    qc_json=json.dumps(qc, sort_keys=True),
)

print("[OK] Saved FP2 artifact:", out_path)
# %%
