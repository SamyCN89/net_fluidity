#!/usr/bin/env python3
# %%
"""
FP3 — Index MC using FP2 allegiance scaffold (no CLI)

Consumes:
  - results/<dataset>/mc_raw/mc_raw_*.npz               (FP1)
  - results/<dataset>/mc/allegiance_ref/allegiance_ref_*.npz (FP2)

Produces:
  - results/<dataset>/mc_indexed/mc_indexed_<ref>_*.npz

Does NOT compute MC.
Does NOT recompute allegiance.
Builds:
  - mc_val_tril (animals, K)  where K=E*(E-1)/2 or your chosen indexing
  - mc_idx_tril (K,2)
  - mc_mod_idx  (K,)   module id per MC edge-pair
  - mc_nplets_index (K,) trimer/tetramer flag per MC edge-pair
"""
#%%
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from shared_code.fun_paths import get_paths
from shared_code.fun_metaconnectivity import (
    get_fc_mc_indices,
    get_mc_region_identities,
    intramodule_indices_mask,
    build_trimer_mask,
    compute_trimers_identity,
)

# =========================
# CONFIG
# =========================
DATASET = "ines_abdallah"
TIMECOURSE_FOLDER = "Timecourses_updated_03052024"
COGNITIVE_FILE = "ROIs.xlsx"
ANAT_LABELS_FILE = "41_Allen.txt"

# Pick exact FP2 file (recommended: explicit)
REF_LABEL = "wt_2m"
FP2_FILE = "allegiance_ref_wt_2m_g=0.70-1.10_ng=10_runs=1000_gcons=1.20.npz"

OVERWRITE = False

# =========================
# Helpers
# =========================
def find_latest(folder: Path, pattern: str) -> Path:
    hits = sorted(folder.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No files matching {pattern} in {folder}")
    return hits[-1]

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
mc_dir = Path(paths["mc"])

# --- Load FP1 mc_raw ---
mc_raw_path = find_latest(mc_dir / "mc_raw", "mc_raw_*.npz")
d1 = np.load(mc_raw_path, allow_pickle=True)
mc = d1["mc"]  # (A, E, E)

n_animals, E, E2 = mc.shape
assert E == E2

# --- Load FP2 allegiance scaffold ---
fp2_path = mc_dir / "allegiance_ref" / FP2_FILE
d2 = np.load(fp2_path, allow_pickle=True)

communities = d2["communities"].astype(int)   # (E,)
sort_idx = d2["sort_idx"].astype(int)         # (E,)

# Safety
assert communities.shape == (E,)
assert sort_idx.shape == (E,)

# --- Apply allegiance sorting to MC ---
mc_sorted = mc[:, sort_idx][:, :, sort_idx]
idx_diag = np.arange(E)
mc_sorted[..., idx_diag, idx_diag] = np.nan

# --- Build module mask (E x E) based on communities ---
_, _, mc_modules_mask = intramodule_indices_mask(communities)
mc_modules_mask = mc_modules_mask[sort_idx][:, sort_idx]  # align with sorted MC

#%%
# --- Build MC triangular indexing ---
# fc_idx: (E,2) mapping FC edges to region pairs (r1,r2)
# mc_idx: (K,2) mapping MC upper-tri edge pairs (e1,e2)
# NOTE: this depends on your current shared implementation; keep it consistent.
regions = int(d1["params_json"].item() and json.loads(d1["params_json"].item())["n_regions"]) if "params_json" in d1 else None
if regions is None:
    raise RuntimeError("Need n_regions. Add it to FP1 params_json or load from elsewhere.")
fc_idx, mc_idx = get_fc_mc_indices(regions, allegiance_sort=sort_idx)

# --- Region identities for interpretability (optional but useful) ---
mc_reg_idx, fc_reg_idx = get_mc_region_identities(fc_idx, mc_idx)

# --- Extract vectorized values ---
mc_val = mc_sorted[:, mc_idx[:, 0], mc_idx[:, 1]]              # (A, K)
mc_mod_idx = mc_modules_mask[mc_idx[:, 0], mc_idx[:, 1]].astype(int)  # (K,)

# --- Trimer/tetramer mask ---
trimer_index, trimer_reg_id, trimer_apex = compute_trimers_identity(regions)
mc_nplets_mask = build_trimer_mask(trimer_index, trimer_apex, E)
mc_nplets_mask = mc_nplets_mask[sort_idx][:, sort_idx]
mc_nplets_index = mc_nplets_mask[mc_idx[:, 0], mc_idx[:, 1]]   # (K,)

print("FP3 indexed MC")
print("  mc:", mc.shape, "E:", E, "K:", mc_idx.shape[0])
print("  mc_val:", mc_val.shape)
print("  mc_mod_idx:", mc_mod_idx.shape, "unique modules:", len(np.unique(mc_mod_idx)))
print("  mc_nplets_index:", mc_nplets_index.shape)
#%%
# --- Save FP3 artifact ---
out_dir = mc_dir / "mc_indexed"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"mc_indexed_ref={REF_LABEL}_animals={n_animals}_E={E}.npz"

if out_path.exists() and not OVERWRITE:
    raise FileExistsError(f"{out_path} exists. Set OVERWRITE=True to replace.")

params = dict(
    dataset=DATASET,
    ref_label=REF_LABEL,
    mc_raw_path=str(mc_raw_path),
    fp2_path=str(fp2_path),
    n_animals=int(n_animals),
    E=int(E),
    regions=int(regions),
)

np.savez_compressed(
    out_path,
    mc_val_tril=mc_val,
    mc_idx_tril=mc_idx,
    fc_idx_tril=fc_idx,
    mc_mod_idx=mc_mod_idx,
    mc_reg_idx=mc_reg_idx,
    mc_nplets_index=mc_nplets_index,
    allegiance_sort=sort_idx,
    communities=communities,
    params_json=json.dumps(params, sort_keys=True),
)

print("[OK] Saved FP3 artifact:", out_path)
# %%
