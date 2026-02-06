#!/usr/bin/env python3
# %%
"""
FP3 — Canonical indexing of metaconnectivity (MC)

Consumes
--------
FP1:
  results/<dataset>/mc_raw/mc_raw_*.npz
    - mc : (A, E, E)
    - params_json must contain: n_regions

FP2:
  results/<dataset>/mc/allegiance_ref/allegiance_ref_*.npz
    - communities : (E,)
    - sort_idx    : (E,)

Produces
--------
results/<dataset>/mc_indexed/mc_indexed_ref=<REF>_animals=<A>_E=<E>.npz

Guaranteed outputs
------------------
- mc_val_tril        : (A, K)  observed MC values, K=E*(E-1)//2
- mc_idx_tril        : (K, 2)  canonical np.tril(E, k=-1) ordering
- mc_mod_idx         : (K,)    1=intra-module, 0=inter-module
- mc_nplets_index    : (K,)    1=trimer, 0=tetramer
- fc_edge_idx : (E,2) ROI-pair identity for each FC edge e
- fc_k4 : (K,4) ROI identities for both FC edges in each MC entry k (optional

This file defines the *only* valid MC vector ordering for downstream FP4+.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from shared_code.fun_paths import get_paths
from shared_code.fun_metaconnectivity import (
    get_fc_mc_indices,
    compute_trimers_identity,
    build_trimer_mask,
)

# =========================
# CONFIG
# =========================
DATASET = "ines_abdallah"
TIMECOURSE_FOLDER = "Timecourses_updated_03052024"
COGNITIVE_FILE = "ROIs.xlsx"
ANAT_LABELS_FILE = "41_Allen.txt"

REF_LABEL = "wt_2m"
FP2_FILE = "allegiance_ref_wt_2m_g=0.70-1.10_ng=10_runs=1000_gcons=1.20.npz"

OVERWRITE = True

# =========================
# Helpers
# =========================
def find_latest(folder: Path, pattern: str) -> Path:
    hits = sorted(folder.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No files matching {pattern} in {folder}")
    return hits[-1]


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

# ---------- Load FP1 (mc_raw) ----------
mc_raw_path = find_latest(mc_dir / "mc_raw", "mc_raw_*.npz")
d1 = np.load(mc_raw_path, allow_pickle=True)
mc = d1["mc"]  # (A, E, E)

A, E, E2 = mc.shape
if E != E2:
    raise RuntimeError("MC matrices are not square")

params_fp1 = json.loads(d1["params_json"].item())
regions = int(params_fp1["n_regions"])

E_expected = regions * (regions - 1) // 2
if E != E_expected:
    raise RuntimeError(
        f"E mismatch: mc_raw E={E} but regions={regions} implies E={E_expected}"
    )

print("[FP1] mc_raw loaded:", mc.shape, "| regions:", regions)

# ---------- Load FP2 (allegiance scaffold) ----------
fp2_path = mc_dir / "allegiance_ref" / FP2_FILE
d2 = np.load(fp2_path, allow_pickle=True)

communities = d2["communities"].astype(int)   # (E,)
sort_idx = d2["sort_idx"].astype(int)          # (E,)

if communities.shape != (E,) or sort_idx.shape != (E,):
    raise RuntimeError("FP2 scaffold shape mismatch → abort")

print("[FP2] allegiance scaffold loaded:", fp2_path.name)

# ---------- Sort MC by FP2 scaffold ----------
mc_sorted = mc[:, sort_idx][:, :, sort_idx]

# ---------- Canonical MC indexing (THIS IS THE LAW) ----------
tri = np.tril_indices(E, k=-1)
mc_idx_tril = np.stack([tri[0], tri[1]], axis=1).astype(np.int64)
K = mc_idx_tril.shape[0]

# ---------- FC-edge identities ----------
fc_edge_idx, _ = get_fc_mc_indices(regions)   # expect (E,2)
fc_edge_idx = np.asarray(fc_edge_idx, dtype=np.int64)
if fc_edge_idx.shape != (E, 2):
    raise RuntimeError(f"Expected fc_edge_idx shape (E,2)={(E,2)}, got {fc_edge_idx.shape}")

fc_edge_idx_sorted = fc_edge_idx[sort_idx]  # IMPORTANT

e1 = mc_idx_tril[:, 0]
e2 = mc_idx_tril[:, 1]
fc_k4 = np.concatenate([fc_edge_idx_sorted[e1], fc_edge_idx_sorted[e2]], axis=1).astype(np.int64)

# ---------- Extract observed MC vectors ----------
mc_val_tril = mc_sorted[:, mc_idx_tril[:, 0], mc_idx_tril[:, 1]]

# ---------- Intra / inter (EXPLICIT, UNAMBIGUOUS) ----------
comm_sorted = communities[sort_idx]
mc_mod_idx = (
    comm_sorted[mc_idx_tril[:, 0]] == comm_sorted[mc_idx_tril[:, 1]]
).astype(np.int8)  # 1=intra, 0=inter

# ---------- Trimer / tetramer ----------
trimer_index, _, trimer_apex = compute_trimers_identity(regions)
mc_nplets_mask = build_trimer_mask(trimer_index, trimer_apex, E)
mc_nplets_mask = mc_nplets_mask[sort_idx][:, sort_idx]
mc_nplets_index = mc_nplets_mask[
    mc_idx_tril[:, 0], mc_idx_tril[:, 1]
].astype(np.int8)  # 1=trimer, 0=tetramer

# ---------- Sanity ----------
assert mc_val_tril.shape == (A, K)
assert mc_mod_idx.shape == (K,)
assert mc_nplets_index.shape == (K,)

print("[FP3] indexed MC")
print("  A:", A, "E:", E, "K:", K)
print("  intra frac:", mc_mod_idx.mean())
print("  trimer frac:", mc_nplets_index.mean())

# ---------- Save FP3 ----------
out_dir = mc_dir / "mc_indexed"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"mc_indexed_ref={REF_LABEL}_animals={A}_E={E}.npz"

if out_path.exists() and not OVERWRITE:
    raise FileExistsError(out_path)

params = dict(
    dataset=DATASET,
    ref_label=REF_LABEL,
    mc_raw_path=str(mc_raw_path),
    fp2_path=str(fp2_path),
    n_animals=A,
    E=E,
    regions=regions,
    mc_ordering="canonical tril(E,k=-1)",
    intra_definition="communities[e1]==communities[e2]",
    nplet_definition="1=trimer, 0=tetramer",
)

np.savez_compressed(
    out_path,
    mc_val_tril=mc_val_tril,
    mc_idx_tril=mc_idx_tril,
    fc_edge_idx=fc_edge_idx_sorted,          # (E,2)
    fc_k4=fc_k4,                      # (K,4) optional
    mc_mod_idx=mc_mod_idx,
    mc_nplets_index=mc_nplets_index,
    allegiance_sort=sort_idx,
    communities=communities,
    params_json=json.dumps(params, sort_keys=True),
)


print("[OK] Saved FP3:", out_path)
# %%
