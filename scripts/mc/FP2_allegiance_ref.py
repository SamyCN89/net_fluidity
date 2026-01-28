#!/usr/bin/env python3
# %%
"""
FP2 — Reference allegiance (V2, no CLI)

Consumes:
  - results/<dataset>/mc_raw/mc_raw_*.npz          (FP1)
  - preprocessed_data/cog_data_filtered_*.csv      (FP0)

Produces:
  - results/<dataset>/allegiance_ref/allegiance_ref_<ref>_g=..._ng=..._runs=..._gcons=....npz

Does NOT compute MC. Does NOT build modules/trimers. Only allegiance reference.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from shared_code.fun_paths import get_paths
from shared_code.fun_allegiance_v2 import (
    v2_prep_undirected_matrix,
    v2_contingency_matrix,
    v2_consensus_from_contingency,
)

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

# --- New protocol params (from your gamma sweep plot) ---
N_RUNS = 1000           # diagnostic / light; bump to 200-500 later, 1000 final if needed
N_GAMMA = 10          # number of gammas in sweep
GMIN = 0.7
GMAX = 1.1
GAMMA_CONSENSUS = 1.2
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

mc = d["mc"]                                   # (A, E, E)
mouse_ids_ts = d["mouse_ids_ts"].astype(str)   # (A,)
age_ts = d["age_ts"].astype(str)               # (A,)

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

mc_ref_raw = np.nanmean(mc[ind_ref], axis=0)

# --- Mandatory hygiene before Louvain (NaN/sym/diag) ---
mc_ref = v2_prep_undirected_matrix(mc_ref_raw)

# =========================
# V2 allegiance
# =========================
ref_label = f"{REF_GENOTYPE}_{REF_AGE}"

t0 = time.time()
contingency, gamma_q, gamma_agree = v2_contingency_matrix(
    mc_data=mc_ref,
    n_runs=N_RUNS,
    n_gamma=N_GAMMA,
    gmin=GMIN,
    gmax=GMAX,
    n_jobs=N_JOBS,
    cache_path=None,   # no side effects here
    ref_name=ref_label,
    return_runs=False,
)
communities, sort_idx, Q_cons, contingency2 = v2_consensus_from_contingency(
    contingency,
    gamma_consensus=GAMMA_CONSENSUS,
)
dt = time.time() - t0

communities = np.asarray(communities)
sort_idx = np.asarray(sort_idx)

# --- QC: blockiness on contingency (more appropriate than on mc_ref) ---
intra0, inter0 = blockiness_score(contingency2, communities)
C_sorted = contingency2[sort_idx][:, sort_idx]
comm_sorted = communities[sort_idx]
intra1, inter1 = blockiness_score(C_sorted, comm_sorted)

print("FP2 allegiance QC (V2)")
print("  ref:", ref_label, "n_ref_sessions:", n_ref)
print(f"  C in [0,1]: min={float(contingency2.min()):.3f} max={float(contingency2.max()):.3f}")
print(f"  consensus: n_modules={len(np.unique(communities))}  Q={Q_cons:.4f}  gamma_consensus={GAMMA_CONSENSUS}")
print(f"  blockiness (C) unsorted: intra={intra0:.4f} inter={inter0:.4f} (intra-inter={intra0-inter0:.4f})")
print(f"  blockiness (C) sorted  : intra={intra1:.4f} inter={inter1:.4f} (intra-inter={intra1-inter1:.4f})")
print(f"  delta(intra-inter): {(intra1-inter1)-(intra0-inter0):.4f}")
print(f"  time: {dt:.2f}s")

# =========================
# SAVE ONE FP2 artifact
# =========================
out_dir = results_dir / OUT_SUBDIR
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / (
    f"allegiance_ref_{ref_label}"
    f"_g={GMIN:.2f}-{GMAX:.2f}_ng={N_GAMMA}"
    f"_runs={N_RUNS}_gcons={GAMMA_CONSENSUS:.2f}.npz"
)
if out_path.exists() and not OVERWRITE:
    raise FileExistsError(f"{out_path} exists. Set OVERWRITE=True to replace.")

params = dict(
    dataset=DATASET,
    ref_label=ref_label,
    ref_genotype=REF_GENOTYPE,
    ref_age=REF_AGE,
    n_runs=N_RUNS,
    n_gamma=N_GAMMA,
    gmin=GMIN,
    gmax=GMAX,
    gamma_consensus=GAMMA_CONSENSUS,
    n_jobs=N_JOBS,
    mc_raw_path=str(mc_raw_path),
    cog_csv_path=str(cog_csv_path),
)

qc = dict(
    seconds=dt,
    n_ref_sessions=n_ref,
    consensus=dict(
        n_modules=int(len(np.unique(communities))),
        Q=float(Q_cons),
    ),
    contingency=dict(
        min=float(contingency2.min()),
        max=float(contingency2.max()),
        sym_err=float(np.max(np.abs(contingency2 - contingency2.T))),
        diag_max=float(np.max(np.abs(np.diag(contingency2)))),
    ),
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
    contingency=contingency2,
    gamma_q=gamma_q,                 # keep: helps justify range later
    # NOTE: gamma_agree is big; keep it only if you want it:
    # gamma_agree=gamma_agree,
    mc_ref=mc_ref,                   # prepped version (safe)
    mc_ref_raw=mc_ref_raw,           # raw mean (for debugging)
    ind_ref=ind_ref,
    mouse_ids_ts=mouse_ids_ts,
    age_ts=age_ts,
    params_json=json.dumps(params, sort_keys=True),
    qc_json=json.dumps(qc, sort_keys=True),
)

print("[OK] Saved FP2 artifact:", out_path)
# %%
