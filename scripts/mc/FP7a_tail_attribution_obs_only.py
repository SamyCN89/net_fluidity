#!/usr/bin/env python3
# %%
"""
FP7a — Tail attribution bookkeeping (OBSERVED only, identity-preserving)
      with EVENT-NORMALIZED outputs (Option 1) + event-share support (Option 2)

Consumes
--------
FP3:
  - mc_val_tril      : (A,K)
  - mc_idx_tril      : (K,2)  FC-edge indices in SORTED edge space
  - fc_edge_idx      : (E,2)  ROI pairs per FC edge (SORTED)
  - mc_nplets_index  : (K,)
  - allegiance_sort  : (E,)

FP2:
  - communities      : (E,)   module id per FC edge (UNSORTED, will be reordered using FP3 sort_idx)

FP6b:
  - category-specific quantiles defining tail thresholds (upper/lower tails)

Produces
--------
results/<dataset>/mc/mc_dist/fp7a_tail_attribution_obs_only_evt_norm.npz

For each category (intra/inter × trimer/tetramer), and for each tail side (hi/lo):
  Numerators (EVENTS):
    - {cat}__hit_hi_r_evt    : (R,)  ROI-endpoint tail events
    - {cat}__hit_lo_r_evt    : (R,)
    - {cat}__hit_hi_mm_evt   : (M,M) module-pair tail events (symmetrized)
    - {cat}__hit_lo_mm_evt   : (M,M)
    - {cat}__hit_hi_k_evt    : (K,)  tail-hit events per MC entry k (sums over animals)
    - {cat}__hit_lo_k_evt    : (K,)

  Denominators (OPPORTUNITIES / ELIGIBLE EVENTS, Option 1):
    - {cat}__opp_r_evt       : (R,)  ROI-endpoint eligible events (topology mask + finite)
    - {cat}__opp_mm_evt      : (M,M) module-pair eligible events (symmetrized)
    - {cat}__opp_k_evt       : (K,)  eligible events per MC entry k (sums over animals)

  Participation (still useful, but not the primary quantity here):
    - {cat}__hit_hi_r_anim   : (R,)  #animals where ROI appears at least once in upper tail (endpoint-based)
    - {cat}__hit_lo_r_anim   : (R,)
    - {cat}__hit_hi_mm_anim  : (M,M) #animals where modulepair appears at least once (symmetrized)
    - {cat}__hit_lo_mm_anim  : (M,M)
    - {cat}__hit_hi_k_anim   : (K,)  #animals where k is hit
    - {cat}__hit_lo_k_anim   : (K,)

Scalars per category (for Option 2 in FP7b):
    - {cat}__sum_hit_hi_r_evt
    - {cat}__sum_hit_lo_r_evt
    - {cat}__sum_hit_hi_mm_evt
    - {cat}__sum_hit_lo_mm_evt

Notes
-----
- “Events” here are ROI-endpoints and modulepair-bins counted PER tail-hit MC entry PER animal.
- Option 1 normalization is: hit_evt / opp_evt.
- Option 2 “share” is computed downstream as: hit_evt / sum(hit_evt).
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

FP3_PATH = "/media/samy/Elements2/Proyectos/LauraHarsan/results/ines_abdallah/mc/mc_indexed/mc_indexed_ref=wt_2m_animals=126_E=820.npz"
FP2_PATTERN = "allegiance_ref_wt_2m_*.npz"
FP6B_NAME = "fp6b_bootstrap_mc_fp6_conditions_POOLED_IID.npz"
MC_DIST_DIRNAME = "mc_dist"

# tails: category-specific thresholds read from FP6b q_obs at these p's
P_LOW = 0.10
P_HIGH = 0.90

OUT_NAME = "fp7a_tail_attribution_obs_only_evt_norm.npz"
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


def _accum_region_events(counts_r: np.ndarray, fc_edge_idx: np.ndarray, mc_idx: np.ndarray, mask_k: np.ndarray) -> None:
    """
    counts_r: (R,) int
    Adds +1 per ROI endpoint occurrence for each k where mask_k is True.
    """
    kk = np.where(mask_k)[0]
    if kk.size == 0:
        return
    e1 = mc_idx[kk, 0]
    e2 = mc_idx[kk, 1]
    r = np.concatenate([fc_edge_idx[e1].ravel(), fc_edge_idx[e2].ravel()], axis=0).astype(np.int64)
    np.add.at(counts_r, r, 1)


def _accum_modulepair_events(counts_mm: np.ndarray, comm_edges: np.ndarray, mc_idx: np.ndarray, mask_k: np.ndarray) -> None:
    """
    counts_mm: (M,M) int
    Adds +1 for each (m1,m2) and (m2,m1) for each k where mask_k is True.
    """
    kk = np.where(mask_k)[0]
    if kk.size == 0:
        return
    e1 = mc_idx[kk, 0]
    e2 = mc_idx[kk, 1]
    m1 = comm_edges[e1].astype(np.int64)
    m2 = comm_edges[e2].astype(np.int64)
    np.add.at(counts_mm, (m1, m2), 1)
    np.add.at(counts_mm, (m2, m1), 1)


def _accum_region_participation(counts_r_anim: np.ndarray, fc_edge_idx: np.ndarray, mc_idx: np.ndarray, mask_k: np.ndarray) -> None:
    """
    counts_r_anim: (R,) int
    Adds +1 per ROI if that ROI appears at least once in any selected k for THIS animal.
    """
    kk = np.where(mask_k)[0]
    if kk.size == 0:
        return
    e1 = mc_idx[kk, 0]
    e2 = mc_idx[kk, 1]
    r = np.concatenate([fc_edge_idx[e1].ravel(), fc_edge_idx[e2].ravel()], axis=0).astype(np.int64)
    r_unique = np.unique(r)
    counts_r_anim[r_unique] += 1


def _accum_modulepair_participation(counts_mm_anim: np.ndarray, comm_edges: np.ndarray, mc_idx: np.ndarray, mask_k: np.ndarray) -> None:
    """
    counts_mm_anim: (M,M) int
    Adds +1 per module-pair bin if it appears at least once for THIS animal (symmetrized).
    """
    kk = np.where(mask_k)[0]
    if kk.size == 0:
        return
    e1 = mc_idx[kk, 0]
    e2 = mc_idx[kk, 1]
    m1 = comm_edges[e1].astype(np.int64)
    m2 = comm_edges[e2].astype(np.int64)
    pairs = np.stack([m1, m2], axis=1)
    pairs_unique = np.unique(pairs, axis=0)
    counts_mm_anim[pairs_unique[:, 0], pairs_unique[:, 1]] += 1
    counts_mm_anim[pairs_unique[:, 1], pairs_unique[:, 0]] += 1


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

# ---------- Load FP3 ----------
d3 = np.load(FP3_PATH, allow_pickle=True)

mc_val = d3["mc_val_tril"].astype(np.float32)        # (A,K)
mc_idx = d3["mc_idx_tril"].astype(np.int64)          # (K,2) SORTED
fc_edge_idx = d3["fc_edge_idx"].astype(np.int64)     # (E,2) SORTED
mc_nplets = d3["mc_nplets_index"].astype(np.int8)    # (K,)
sort_idx = d3["allegiance_sort"].astype(np.int64)    # (E,)

A, K = mc_val.shape
E = fc_edge_idx.shape[0]
R = int(fc_edge_idx.max() + 1)

# ---------- Load FP2 communities and SORT THEM ----------
fp2_path = find_latest(mc_dir / "allegiance_ref", FP2_PATTERN)
d2 = np.load(fp2_path, allow_pickle=True)

comm_edges = d2["communities"].astype(np.int64)  # UNSORTED
comm_edges = comm_edges[sort_idx]                # SORTED (matches mc_idx!)
assert comm_edges.shape == (E,)

# Normalize module labels to 0..M-1
uniq = np.unique(comm_edges)
remap = {int(u): i for i, u in enumerate(uniq)}
comm_edges = np.array([remap[int(x)] for x in comm_edges], dtype=np.int64)
M = int(comm_edges.max() + 1)

# ---------- Intra / inter masks (K,) ----------
e1 = mc_idx[:, 0]
e2 = mc_idx[:, 1]
is_intra = comm_edges[e1] == comm_edges[e2]
is_inter = ~is_intra

# ---------- Trimer / tetramer masks (K,) ----------
is_trimer = (mc_nplets > 0)
is_tetramer = ~is_trimer

print("[FP7a] A:", A, "E:", E, "K:", K, "R:", R, "M:", M)
print("[FP7a] intra frac:", float(is_intra.mean()))
print("[FP7a] trimer frac:", float(is_trimer.mean()))

# ---------- Load FP6b thresholds ----------
fp6b_path = dist_dir / FP6B_NAME
d6 = np.load(fp6b_path, allow_pickle=True)
p_grid = d6["p_grid"].astype(np.float32)

def load_q(key: str) -> np.ndarray:
    return d6[f"{key}__q_obs"].astype(np.float32)

CATS = {
    "intra_trimer":   dict(obs_key="obs_intra_trimer",   topo=(is_intra & is_trimer)),
    "inter_trimer":   dict(obs_key="obs_inter_trimer",   topo=(is_inter & is_trimer)),
    "intra_tetramer": dict(obs_key="obs_intra_tetramer", topo=(is_intra & is_tetramer)),
    "inter_tetramer": dict(obs_key="obs_inter_tetramer", topo=(is_inter & is_tetramer)),
}

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
            event_definition="ROI-endpoints + modulepair bins, per k per animal",
            option1="event_rate = hit_evt / opp_evt",
            option2="event_share = hit_evt / sum(hit_evt)",
        ),
        sort_keys=True,
    )
}

# =========================
# Attribution
# =========================
for cat, meta in CATS.items():
    q = load_q(meta["obs_key"])
    thr_lo = _get_q_at_p(q, p_grid, P_LOW)
    thr_hi = _get_q_at_p(q, p_grid, P_HIGH)

    topo = meta["topo"]  # (K,) boolean

    # ---------- Numerators (tail-hit events) ----------
    hit_hi_k_evt  = np.zeros(K, dtype=np.int64)
    hit_lo_k_evt  = np.zeros(K, dtype=np.int64)
    hit_hi_mm_evt = np.zeros((M, M), dtype=np.int64)
    hit_lo_mm_evt = np.zeros((M, M), dtype=np.int64)
    hit_hi_r_evt  = np.zeros(R, dtype=np.int64)
    hit_lo_r_evt  = np.zeros(R, dtype=np.int64)

    # ---------- Denominators (eligible opportunities; Option 1) ----------
    opp_k_evt  = np.zeros(K, dtype=np.int64)
    opp_mm_evt = np.zeros((M, M), dtype=np.int64)
    opp_r_evt  = np.zeros(R, dtype=np.int64)

    # ---------- Participation (animals; optional reporting) ----------
    hit_hi_k_anim  = np.zeros(K, dtype=np.int32)
    hit_lo_k_anim  = np.zeros(K, dtype=np.int32)
    hit_hi_mm_anim = np.zeros((M, M), dtype=np.int32)
    hit_lo_mm_anim = np.zeros((M, M), dtype=np.int32)
    hit_hi_r_anim  = np.zeros(R, dtype=np.int32)
    hit_lo_r_anim  = np.zeros(R, dtype=np.int32)

    for a in range(A):
        x = mc_val[a]                 # (K,)
        finite = np.isfinite(x)
        eligible = topo & finite      # (K,) category-eligible for this animal

        # ----- Option 1 denominators: ALL eligible events -----
        opp_k_evt += eligible.astype(np.int64)
        _accum_modulepair_events(opp_mm_evt, comm_edges, mc_idx, eligible)
        _accum_region_events(opp_r_evt, fc_edge_idx, mc_idx, eligible)

        # ----- Tail hits -----
        hit_hi = eligible & (x >= thr_hi)
        hit_lo = eligible & (x <= thr_lo)

        # numerators: events
        hit_hi_k_evt += hit_hi.astype(np.int64)
        hit_lo_k_evt += hit_lo.astype(np.int64)
        _accum_modulepair_events(hit_hi_mm_evt, comm_edges, mc_idx, hit_hi)
        _accum_modulepair_events(hit_lo_mm_evt, comm_edges, mc_idx, hit_lo)
        _accum_region_events(hit_hi_r_evt, fc_edge_idx, mc_idx, hit_hi)
        _accum_region_events(hit_lo_r_evt, fc_edge_idx, mc_idx, hit_lo)

        # participation (animals)
        hit_hi_k_anim[hit_hi] += 1
        hit_lo_k_anim[hit_lo] += 1
        _accum_modulepair_participation(hit_hi_mm_anim, comm_edges, mc_idx, hit_hi)
        _accum_modulepair_participation(hit_lo_mm_anim, comm_edges, mc_idx, hit_lo)
        _accum_region_participation(hit_hi_r_anim, fc_edge_idx, mc_idx, hit_hi)
        _accum_region_participation(hit_lo_r_anim, fc_edge_idx, mc_idx, hit_lo)

    # scalars for Option 2 convenience (shares computed downstream)
    sum_hit_hi_r_evt  = int(hit_hi_r_evt.sum())
    sum_hit_lo_r_evt  = int(hit_lo_r_evt.sum())
    sum_hit_hi_mm_evt = int(hit_hi_mm_evt.sum())
    sum_hit_lo_mm_evt = int(hit_lo_mm_evt.sum())

    # store
    out[f"{cat}__thr_lo"] = np.float32(thr_lo)
    out[f"{cat}__thr_hi"] = np.float32(thr_hi)

    out[f"{cat}__hit_hi_k_evt"]  = hit_hi_k_evt
    out[f"{cat}__hit_lo_k_evt"]  = hit_lo_k_evt
    out[f"{cat}__hit_hi_mm_evt"] = hit_hi_mm_evt
    out[f"{cat}__hit_lo_mm_evt"] = hit_lo_mm_evt
    out[f"{cat}__hit_hi_r_evt"]  = hit_hi_r_evt
    out[f"{cat}__hit_lo_r_evt"]  = hit_lo_r_evt

    out[f"{cat}__opp_k_evt"]  = opp_k_evt
    out[f"{cat}__opp_mm_evt"] = opp_mm_evt
    out[f"{cat}__opp_r_evt"]  = opp_r_evt

    out[f"{cat}__hit_hi_k_anim"]  = hit_hi_k_anim
    out[f"{cat}__hit_lo_k_anim"]  = hit_lo_k_anim
    out[f"{cat}__hit_hi_mm_anim"] = hit_hi_mm_anim
    out[f"{cat}__hit_lo_mm_anim"] = hit_lo_mm_anim
    out[f"{cat}__hit_hi_r_anim"]  = hit_hi_r_anim
    out[f"{cat}__hit_lo_r_anim"]  = hit_lo_r_anim

    out[f"{cat}__sum_hit_hi_r_evt"]  = np.int64(sum_hit_hi_r_evt)
    out[f"{cat}__sum_hit_lo_r_evt"]  = np.int64(sum_hit_lo_r_evt)
    out[f"{cat}__sum_hit_hi_mm_evt"] = np.int64(sum_hit_hi_mm_evt)
    out[f"{cat}__sum_hit_lo_mm_evt"] = np.int64(sum_hit_lo_mm_evt)

# =========================
# Save
# =========================
out_path = dist_dir / OUT_NAME
if out_path.exists() and not OVERWRITE:
    raise FileExistsError(out_path)

np.savez_compressed(out_path, **out)
print("[OK] Saved FP7a:", out_path)
# %%
