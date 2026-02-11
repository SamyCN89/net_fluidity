#!/usr/bin/env python3
"""
FP7a1 — Root-grouped trimer strength per animal

T[a, i] = sum of MC values of all trimers rooted at ROI i

Consumes:
  FP3 mc_indexed_*.npz

Produces:
  fp7a1_trimer_root_strength_per_animal.npz
"""

import json
from pathlib import Path
import numpy as np
from shared_code.fun_paths import get_paths

# =====================
# CONFIG
# =====================

DATASET = "ines_abdallah"
TIMECOURSE_FOLDER = "Timecourses_updated_03052024"
COGNITIVE_FILE = "ROIs.xlsx"
ANAT_LABELS_FILE = "41_Allen.txt"


# =====================
# Helpers
# =====================




def extract_trimer_root(k4):
    """
    k4 = (r1,r2,r3,r4)

    Find common ROI = root
    """

    # count occurrences
    uniq, cnt = np.unique(k4, return_counts=True)

    # root appears twice
    idx = np.where(cnt == 2)[0]

    if idx.size != 1:
        raise ValueError(f"Bad trimer identity: {k4}")

    return int(uniq[idx[0]])


# =====================
# MAIN
# =====================

paths = get_paths(
    dataset_name=DATASET,
    timecourse_folder=TIMECOURSE_FOLDER,
    cognitive_data_file=COGNITIVE_FILE,
    anat_labels_file=ANAT_LABELS_FILE,
)
#%%
FP3_PATH = paths["mc"] / "mc_indexed" / "mc_indexed_ref=wt_2m_animals=126_E=820.npz"
z = np.load(FP3_PATH, allow_pickle=True)
print("[LOAD]", FP3_PATH.name)
fp3 = FP3_PATH
print("[LOAD]", fp3.name)

mc_idx_dir = paths["mc"] / "mc_dist"
#%%

# Load FP3
z = np.load(fp3, allow_pickle=True)

# Extract data
mc = z["mc_val_tril"]  # (A,K)
k4 = z["fc_k4"]  # (K,4)
is_trimer = z["mc_nplets_index"] == 1
mc_mod_idx = z["mc_mod_idx"].astype(np.int8)  # (K,) 1=intra, 0=inter

params = json.loads(z["params_json"].item())
n_regions = int(params.get("regions", params.get("n_regions")))

A, K = mc.shape

print("Animals:", A)
print("Regions:", n_regions)
print("K:", K)
print("Trimers:", is_trimer.sum())


# --- Precompute root per trimer index ---
roots = np.full(K, -1, dtype=np.int16)

for k in np.where(is_trimer)[0]:
    roots[k] = extract_trimer_root(k4[k])


# --- Accumulate per animal × region ---
T_all   = np.zeros((A, n_regions), dtype=np.float32)
T_intra = np.zeros((A, n_regions), dtype=np.float32)
T_inter = np.zeros((A, n_regions), dtype=np.float32)

trimer_idx = np.where(is_trimer)[0]  # precompute once

for a in range(A):
    v = mc[a]

    for k in trimer_idx:
        r = roots[k]
        x = v[k]

        if not np.isfinite(x):
            continue

        T_all[a, r] += x

        if mc_mod_idx[k] == 1:
            T_intra[a, r] += x
        else:
            T_inter[a, r] += x


# --- Save ---
out = dict(
    T_all=T_all,
    T_intra=T_intra,
    T_inter=T_inter,
    roots=roots,
    is_trimer=is_trimer,
    mc_mod_idx=mc_mod_idx,
)

out_path = mc_idx_dir / "fp7a1_trimer_root_strength_per_animal__all_intra_inter.npz"

np.savez(out_path, **out)

print("T_all:",   T_all.shape,   np.isfinite(T_all).mean())
print("T_intra:", T_intra.shape, np.isfinite(T_intra).mean())
print("T_inter:", T_inter.shape, np.isfinite(T_inter).mean())
print("Check (all ~= intra+inter):", np.allclose(T_all, T_intra + T_inter, atol=1e-6))

print("[DONE]", out_path)
