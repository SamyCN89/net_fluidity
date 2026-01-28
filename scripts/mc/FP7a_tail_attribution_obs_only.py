#!/usr/bin/env python3
# %%
"""
FP7a — Tail attribution bookkeeping (OBSERVED only, identity-preserving)

Consumes:
  - FP3 indexed MC: mc_val_tril, mc_idx_tril, fc_idx_tril, mc_mod_idx, mc_nplets_index
  - FP2 scaffold: communities (on FC edges, length E)
  - FP6b summaries: quantiles to define tail thresholds per category

Produces:
  - fp7a_tail_attribution_obs_only.npz
    Counts of tail participation:
      - per MC-pair k (length K)
      - per module-pair (n_mod x n_mod)
      - per region (R)
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from shared_code.fun_paths import get_paths

# =========================
# CONFIG
# =========================
DATASET = "ines_abdallah"
TIMECOURSE_FOLDER = "Timecourses_updated_03052024"
COGNITIVE_FILE = "ROIs.xlsx"
ANAT_LABELS_FILE = "41_Allen.txt"

# Inputs
FP3_PATH = "/media/samy/Elements2/Proyectos/LauraHarsan/results/ines_abdallah/mc/mc_indexed/mc_indexed_ref=wt_2m_animals=126_E=820.npz"
FP2_PATTERN = "allegiance_ref_wt_2m_*.npz"     # used only to load communities scaffold
FP6B_NAME = "fp6b_bootstrap_mc_fp6_conditions_FAST.npz"
MC_DIST_DIRNAME = "mc_dist"

# Tail definition
P_LOW  = 0.05
P_HIGH = 0.95
# (use 0.01/0.99 later if you want stricter tails)

OUT_NAME = "fp7a_tail_attribution_obs_only.npz"
OVERWRITE = True

# =========================
# Helpers
# =========================
def find_latest(folder: Path, pattern: str) -> Path:
    hits = sorted(folder.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No files matching {pattern} in {folder}")
    return hits[-1]

def _get_q_at_p(q_obs: np.ndarray, p_grid: np.ndarray, p: float) -> float:
    j = int(np.argmin(np.abs(p_grid - p)))
    return float(q_obs[j])

def _accum_region_counts(counts_r: np.ndarray, fc_idx: np.ndarray, mc_idx: np.ndarray, hit_k: np.ndarray):
    """
    counts_r: (R,)
    fc_idx: (E,2) region pairs (r1,r2)
    mc_idx: (K,2) FC edge indices (e1,e2)
    hit_k: boolean mask (K,) for tail members
    """
    kk = np.where(hit_k)[0]
    if kk.size == 0:
        return
    e1 = mc_idx[kk, 0]
    e2 = mc_idx[kk, 1]
    r = np.concatenate([fc_idx[e1].ravel(), fc_idx[e2].ravel()], axis=0).astype(np.int64)
    np.add.at(counts_r, r, 1)

def _accum_modulepair_counts(counts_mm: np.ndarray, comm_edges: np.ndarray, mc_idx: np.ndarray, hit_k: np.ndarray):
    """
    counts_mm: (M,M)
    comm_edges: (E,) module id for each FC edge (after FP2)
    """
    kk = np.where(hit_k)[0]
    if kk.size == 0:
        return
    e1 = mc_idx[kk, 0]
    e2 = mc_idx[kk, 1]
    m1 = comm_edges[e1].astype(np.int64)
    m2 = comm_edges[e2].astype(np.int64)
    np.add.at(counts_mm, (m1, m2), 1)
    np.add.at(counts_mm, (m2, m1), 1)  # symmetrize

def _accum_region_participation(counts_r: np.ndarray, fc_idx: np.ndarray, mc_idx: np.ndarray, hit_k: np.ndarray):
    """
    counts_r: (R,) increments by 1 PER ROI if that ROI appears at least once in hit_k.
    """
    kk = np.where(hit_k)[0]
    if kk.size == 0:
        return
    e1 = mc_idx[kk, 0]
    e2 = mc_idx[kk, 1]
    r = np.concatenate([fc_idx[e1].ravel(), fc_idx[e2].ravel()], axis=0).astype(np.int64)
    r_unique = np.unique(r)
    counts_r[r_unique] += 1


def _accum_modulepair_participation(counts_mm: np.ndarray, comm_edges: np.ndarray, mc_idx: np.ndarray, hit_k: np.ndarray):
    """
    counts_mm: (M,M) increments by 1 PER module-pair if that bin appears at least once in hit_k.
    Symmetrized like event version.
    """
    kk = np.where(hit_k)[0]
    if kk.size == 0:
        return
    e1 = mc_idx[kk, 0]
    e2 = mc_idx[kk, 1]
    m1 = comm_edges[e1].astype(np.int64)
    m2 = comm_edges[e2].astype(np.int64)

    # unique pairs
    pairs = np.stack([m1, m2], axis=1)
    pairs_unique = np.unique(pairs, axis=0)

    counts_mm[pairs_unique[:, 0], pairs_unique[:, 1]] += 1
    counts_mm[pairs_unique[:, 1], pairs_unique[:, 0]] += 1

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
dist_dir = mc_dir / MC_DIST_DIRNAME

# --- Load FP3 (identity-preserving MC values) ---
d3 = np.load(FP3_PATH, allow_pickle=True)
mc_val = d3["mc_val_tril"].astype(np.float32)        # (A,K)
mc_idx = d3["mc_idx_tril"].astype(np.int64)          # (K,2)
fc_idx = d3["fc_idx_tril"].astype(np.int64)          # (E,2)
# mc_mod_idx = d3["mc_mod_idx"].astype(np.int64)       # (K,)
mc_nplets = d3["mc_nplets_index"].astype(np.int64)   # (K,)

A, K = mc_val.shape
E = fc_idx.shape[0]

# Infer R from fc_idx
R = int(fc_idx.max() + 1)

# --- Load FP2 communities (module id per FC edge) ---
fp2_path = find_latest(mc_dir / "allegiance_ref", FP2_PATTERN)
d2 = np.load(fp2_path, allow_pickle=True)
comm_edges = d2["communities"].astype(np.int64)      # (E,)
assert comm_edges.shape == (E,)

# Normalize module labels to 0..M-1 (safer for indexing)
uniq = np.unique(comm_edges)
remap = {int(u): i for i, u in enumerate(uniq)}
comm_edges = np.array([remap[int(x)] for x in comm_edges], dtype=np.int64)
M = int(comm_edges.max() + 1)

# Intra/inter based on FP2 communities of the TWO FC edges defining each MC entry
e1 = mc_idx[:, 0]
e2 = mc_idx[:, 1]
is_intra = (comm_edges[e1] == comm_edges[e2])
is_inter = ~is_intra

print("[INFO] is_intra fraction:", float(is_intra.mean()))

# --- Load FP6b thresholds (category-specific) ---
fp6b_path = dist_dir / FP6B_NAME
d6 = np.load(fp6b_path, allow_pickle=True)
p_grid = d6["p_grid"].astype(np.float32)

def load_q(key: str) -> np.ndarray:
    return d6[f"{key}__q_obs"].astype(np.float32)

# Categories we attribute (OBS only)
CATS = {
    "intra_trimer":   dict(obs_key="obs_intra_trimer",   intra=True,  trimer=True),
    "inter_trimer":   dict(obs_key="obs_inter_trimer",   intra=False, trimer=True),
    "intra_tetramer": dict(obs_key="obs_intra_tetramer", intra=True,  trimer=False),
    "inter_tetramer": dict(obs_key="obs_inter_tetramer", intra=False, trimer=False),
}

# Build category masks on K (topology masks)
# NOTE: depending on how mc_mod_idx is encoded in your codebase.
# Here we assume: intra == 1, inter == 0 (common in your pipeline).
# u_mod = np.unique(mc_mod_idx)
# if not set(u_mod).issubset({0, 1}):
#     print("[WARN] mc_mod_idx not binary; treating intra as (mc_mod_idx>0). unique:", u_mod)

# is_intra = (mc_mod_idx == 1) if set(u_mod).issubset({0, 1}) else (mc_mod_idx > 0)
is_trimer = (mc_nplets > 0)          # your convention: >0 = trimer, 0 = tetramer

# Storage
out = {
    "params_json": json.dumps(
        dict(
            dataset=DATASET,
            fp3_path=str(FP3_PATH),
            fp2_path=str(fp2_path),
            fp6b_path=str(fp6b_path),
            p_low=float(P_LOW),
            p_high=float(P_HIGH),
            A=int(A),
            E=int(E),
            K=int(K),
            R=int(R),
            n_modules=int(M),
            categories=list(CATS.keys()),
        ),
        sort_keys=True,
    )
}

for cat, meta in CATS.items():
    # thresholds from FP6b (category-specific)
    q = load_q(meta["obs_key"])
    thr_lo = _get_q_at_p(q, p_grid, P_LOW)
    thr_hi = _get_q_at_p(q, p_grid, P_HIGH)

    # topology mask (K,)
    topo = (is_intra if meta["intra"] else is_inter) & (is_trimer if meta["trimer"] else ~is_trimer)


    # -------------------------------
    # Animal participation counts (0..A)
    # -------------------------------
    cnt_hi_k_anim = np.zeros(K, dtype=np.int32)
    cnt_lo_k_anim = np.zeros(K, dtype=np.int32)

    cnt_hi_mm_anim = np.zeros((M, M), dtype=np.int32)
    cnt_lo_mm_anim = np.zeros((M, M), dtype=np.int32)
    cnt_hi_r_anim  = np.zeros(R, dtype=np.int32)
    cnt_lo_r_anim  = np.zeros(R, dtype=np.int32)

    # -------------------------------
    # Raw event counts (can exceed A)
    # -------------------------------
    cnt_hi_k_evt = np.zeros(K, dtype=np.int64)
    cnt_lo_k_evt = np.zeros(K, dtype=np.int64)

    cnt_hi_mm_evt = np.zeros((M, M), dtype=np.int64)
    cnt_lo_mm_evt = np.zeros((M, M), dtype=np.int64)
    cnt_hi_r_evt  = np.zeros(R, dtype=np.int64)
    cnt_lo_r_evt  = np.zeros(R, dtype=np.int64)



    for a in range(A):
        x = mc_val[a]  # (K,)
        finite = np.isfinite(x)
        m = topo & finite

        hit_hi = m & (x >= thr_hi)
        hit_lo = m & (x <= thr_lo)

        # -------------------------------
        # Animal participation: +1 if k is hit by this animal
        # -------------------------------
        cnt_hi_k_anim[hit_hi] += 1
        cnt_lo_k_anim[hit_lo] += 1

        _accum_modulepair_counts(cnt_hi_mm_anim, comm_edges, mc_idx, hit_hi)
        _accum_modulepair_counts(cnt_lo_mm_anim, comm_edges, mc_idx, hit_lo)
        _accum_region_counts(cnt_hi_r_anim, fc_idx, mc_idx, hit_hi)
        _accum_region_counts(cnt_lo_r_anim, fc_idx, mc_idx, hit_lo)

        # -------------------------------
        # Raw event counts: add number of hits from this animal
        # (each k contributes 1 event for this animal if hit_hi/lo true)
        # -------------------------------
        cnt_hi_k_evt += hit_hi.astype(np.int64)
        cnt_lo_k_evt += hit_lo.astype(np.int64)

        _accum_modulepair_counts(cnt_hi_mm_evt, comm_edges, mc_idx, hit_hi)
        _accum_modulepair_counts(cnt_lo_mm_evt, comm_edges, mc_idx, hit_lo)
        _accum_region_counts(cnt_hi_r_evt, fc_idx, mc_idx, hit_hi)
        _accum_region_counts(cnt_lo_r_evt, fc_idx, mc_idx, hit_lo)



    out[f"{cat}__thr_lo"] = np.float32(thr_lo)
    out[f"{cat}__thr_hi"] = np.float32(thr_hi)
    # Animal-participation (0..A)
    out[f"{cat}__count_hi_k_anim"]  = cnt_hi_k_anim
    out[f"{cat}__count_lo_k_anim"]  = cnt_lo_k_anim
    out[f"{cat}__count_hi_mm_anim"] = cnt_hi_mm_anim
    out[f"{cat}__count_lo_mm_anim"] = cnt_lo_mm_anim
    out[f"{cat}__count_hi_r_anim"]  = cnt_hi_r_anim
    out[f"{cat}__count_lo_r_anim"]  = cnt_lo_r_anim

    # Raw events
    out[f"{cat}__count_hi_k"]  = cnt_hi_k_evt
    out[f"{cat}__count_lo_k"]  = cnt_lo_k_evt
    out[f"{cat}__count_hi_mm"] = cnt_hi_mm_evt
    out[f"{cat}__count_lo_mm"] = cnt_lo_mm_evt
    out[f"{cat}__count_hi_r"]  = cnt_hi_r_evt
    out[f"{cat}__count_lo_r"]  = cnt_lo_r_evt



# Save
out_path = dist_dir / OUT_NAME
if out_path.exists() and not OVERWRITE:
    raise FileExistsError(out_path)

np.savez_compressed(out_path, **out)
print("[OK] Saved FP7a:", out_path)

