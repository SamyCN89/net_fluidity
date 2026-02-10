#!/usr/bin/env python3
# %%
"""
FP7d — Tail enrichment with normalization (events as primary quantity)

Consumes
--------
FP3:
  - mc_idx_tril      : (K,2)  FC-edge indices in SORTED edge space
  - fc_k4            : (K,4)  ROI identities for both FC edges in each MC entry k (SORTED)
  - mc_nplets_index  : (K,)
  - allegiance_sort  : (E,)

FP2:
  - communities      : (E,)   module id per FC edge (UNSORTED, will be reordered via allegiance_sort)

FP6a (optional but recommended for null enrichment):
  - null_* vectors per animal (object arrays) OR per-animal null K-vectors if you prefer
  Here we use FP6a pooled-per-animal vectors (null_intra_trimer etc).

FP6b:
  - q_obs + p_grid to define thresholds (same as FP7a)

FP7a:
  - observed event counts + animal participation counts

Produces
--------
results/<dataset>/mc/mc_dist/fp7d_tail_enrichment.npz
  For each category cat in {intra_trimer, inter_trimer, intra_tetramer, inter_tetramer}:
    - exposure baselines:
        cat__exp_k                 (int64)     number of eligible MC entries (topo mask)
        cat__exp_r                 (R,) int64  ROI exposure (counts of ROI appearances across eligible entries)
        cat__exp_mm                (M,M) int64 modulepair exposure (symmetrized)
    - observed:
        cat__obs_evt_r_hi/lo       (R,) int64  from FP7a
        cat__obs_evt_mm_hi/lo      (M,M) int64 from FP7a
        cat__obs_evt_k_hi/lo       (K,) int64  from FP7a
    - normalized (exposure-corrected):
        cat__obs_rate_r_hi/lo      (R,) float32 = obs_evt_r / exp_r
        cat__obs_rate_mm_hi/lo     (M,M) float32 = obs_evt_mm / exp_mm
    - enrichment vs exposure:
        cat__obs_enrich_r_hi/lo    (R,) float32 = (obs_evt_r / sum_obs_evt_r) / (exp_r / sum_exp_r)
        cat__obs_enrich_mm_hi/lo   (M,M) float32 = (obs_evt_mm / sum_obs_evt_mm) / (exp_mm / sum_exp_mm)
        cat__obs_log2enrich_*      log2(enrich + eps)
    - null (optional):
        cat__null_evt_r_hi/lo, cat__null_evt_mm_hi/lo, cat__null_evt_k_hi/lo
        cat__obs_over_null_rate_*  = (obs_rate + eps)/(null_rate + eps)

What it does (computationally)
------------------------------
Normalizes tail event counts by "opportunity" (exposure) and optionally compares obs to null.

Scientific intent
-----------------
Turns "tail drivers" into "tail-enriched drivers" (mechanism, not raw frequency).
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

MC_DIST_DIRNAME = "mc_dist"

# Inputs
FP3_PATH = "/media/samy/Elements2/Proyectos/LauraHarsan/results/ines_abdallah/mc/mc_indexed/mc_indexed_ref=wt_2m_animals=126_E=820.npz"
FP2_PATTERN = "allegiance_ref_wt_2m_*.npz"

FP7A_NAME = "fp7a_tail_attribution_obs_only.npz"
FP6A_NAME = "fp6a_mc_intra_inter_trimer_tetramer_per_animal.npz"
FP6B_NAME = "fp6b_bootstrap_mc_fp6_conditions_POOLED_IID.npz"

# Tail definition
P_LOW  = 0.05
P_HIGH = 0.95

# If True: compute null tail attribution and obs/null enrichment
DO_NULL = True

# Numerics
EPS = 1e-12
OVERWRITE = True
OUT_NAME = "fp7d_tail_enrichment.npz"

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

def _symmetrize_mm(M: np.ndarray) -> np.ndarray:
    return (M + M.T).astype(M.dtype, copy=False)

def _safe_div(a, b):
    out = np.full_like(a, np.nan, dtype=np.float32)
    good = (b > 0)
    out[good] = (a[good].astype(np.float32) / b[good].astype(np.float32))
    return out

def _enrichment(obs_counts: np.ndarray, exp_counts: np.ndarray) -> np.ndarray:
    """
    Enrichment = (obs / sum(obs)) / (exp / sum(exp)).
    Handles zeros safely.
    """
    obs = obs_counts.astype(np.float64, copy=False)
    exp = exp_counts.astype(np.float64, copy=False)
    so = obs.sum()
    se = exp.sum()
    if so <= 0 or se <= 0:
        return np.full_like(obs_counts, np.nan, dtype=np.float32)
    po = obs / so
    pe = exp / se
    return ((po + EPS) / (pe + EPS)).astype(np.float32)

def _accum_region_counts_from_hits(counts_r: np.ndarray, fc_k4: np.ndarray, hit_k: np.ndarray):
    kk = np.where(hit_k)[0]
    if kk.size == 0:
        return
    r = fc_k4[kk].ravel().astype(np.int64)
    np.add.at(counts_r, r, 1)

def _accum_modulepair_counts_from_hits(counts_mm: np.ndarray, m1: np.ndarray, m2: np.ndarray, hit_k: np.ndarray):
    kk = np.where(hit_k)[0]
    if kk.size == 0:
        return
    a = m1[kk].astype(np.int64)
    b = m2[kk].astype(np.int64)
    np.add.at(counts_mm, (a, b), 1)
    np.add.at(counts_mm, (b, a), 1)

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
mc_idx = d3["mc_idx_tril"].astype(np.int64)        # (K,2) sorted edge index
fc_k4  = d3["fc_k4"].astype(np.int64)              # (K,4) sorted ROI ids
mc_nplets = d3["mc_nplets_index"].astype(np.int8)  # (K,)
sort_idx = d3["allegiance_sort"].astype(np.int64)  # (E,)

K = mc_idx.shape[0]
R = int(fc_k4.max() + 1)

# ---------- Load FP2 communities and sort them ----------
fp2_path = find_latest(mc_dir / "allegiance_ref", FP2_PATTERN)
d2 = np.load(fp2_path, allow_pickle=True)
comm_edges = d2["communities"].astype(np.int64)  # unsorted
comm_edges = comm_edges[sort_idx]                # sorted edge space

# normalize to 0..M-1
uniq = np.unique(comm_edges)
remap = {int(u): i for i, u in enumerate(uniq)}
comm_edges = np.array([remap[int(x)] for x in comm_edges], dtype=np.int64)
M = int(comm_edges.max() + 1)

e1 = mc_idx[:, 0]
e2 = mc_idx[:, 1]
is_intra = (comm_edges[e1] == comm_edges[e2])
is_inter = ~is_intra
is_trimer = (mc_nplets > 0)

m1 = comm_edges[e1]  # (K,)
m2 = comm_edges[e2]  # (K,)

print("[FP7d] K:", K, "R:", R, "M:", M)
print("[FP7d] intra frac:", float(is_intra.mean()), "| trimer frac:", float(is_trimer.mean()))

# ---------- Load FP6b thresholds (same as FP7a) ----------
fp6b_path = dist_dir / FP6B_NAME
d6 = np.load(fp6b_path, allow_pickle=True)
p_grid = d6["p_grid"].astype(np.float32)

def load_q(key: str) -> np.ndarray:
    return d6[f"{key}__q_obs"].astype(np.float32)

# Categories aligned with FP7a naming
CATS = {
    "intra_trimer":   dict(obs_key="obs_intra_trimer",   topo=(is_intra & is_trimer)),
    "inter_trimer":   dict(obs_key="obs_inter_trimer",   topo=(is_inter & is_trimer)),
    "intra_tetramer": dict(obs_key="obs_intra_tetramer", topo=(is_intra & ~is_trimer)),
    "inter_tetramer": dict(obs_key="obs_inter_tetramer", topo=(is_inter & ~is_trimer)),
}

# ---------- Load FP7a observed counts ----------
fp7a_path = dist_dir / FP7A_NAME
d7a = np.load(fp7a_path, allow_pickle=True)
params7a = json.loads(d7a["params_json"].item())
A = int(params7a["A"])  # animals

# ---------- Load FP6a null pooled-per-animal vectors (optional) ----------
if DO_NULL:
    fp6a_path = dist_dir / FP6A_NAME
    if not fp6a_path.exists():
        raise FileNotFoundError(f"DO_NULL=True but missing {fp6a_path}")
    d6a = np.load(fp6a_path, allow_pickle=True)

# ---------- Compute exposures (opportunity) ----------
out = {}
meta = dict(
    dataset=DATASET,
    fp3_path=str(FP3_PATH),
    fp2_path=str(fp2_path),
    fp6b_path=str(fp6b_path),
    fp7a_path=str(fp7a_path),
    fp6a_path=str(dist_dir / FP6A_NAME) if DO_NULL else None,
    A=int(A),
    K=int(K),
    R=int(R),
    M=int(M),
    p_low=float(P_LOW),
    p_high=float(P_HIGH),
    do_null=bool(DO_NULL),
    eps=float(EPS),
)

for cat, info in CATS.items():
    topo = info["topo"]  # (K,) boolean

    # exposure in k-space
    exp_k = int(topo.sum())

    # ROI exposure: count ROI appearances across eligible k (each k contributes 4 ROI appearances)
    exp_r = np.bincount(fc_k4[topo].ravel(), minlength=R).astype(np.int64)

    # modulepair exposure: each eligible k contributes 1 pair, but FP7a mm counts are symmetrized => sym exposure too
    exp_mm = np.zeros((M, M), dtype=np.int64)
    _accum_modulepair_counts_from_hits(exp_mm, m1, m2, topo)
    # exp_mm is already sym due to accumulator

    out[f"{cat}__exp_k"] = np.int64(exp_k)
    out[f"{cat}__exp_r"] = exp_r
    out[f"{cat}__exp_mm"] = exp_mm

    # thresholds
    q = load_q(info["obs_key"])
    thr_lo = _get_q_at_p(q, p_grid, P_LOW)
    thr_hi = _get_q_at_p(q, p_grid, P_HIGH)
    out[f"{cat}__thr_lo"] = np.float32(thr_lo)
    out[f"{cat}__thr_hi"] = np.float32(thr_hi)

    # observed counts from FP7a (events are primary)
    out[f"{cat}__obs_evt_r_hi"]  = d7a[f"{cat}__count_hi_r"].astype(np.int64)
    out[f"{cat}__obs_evt_r_lo"]  = d7a[f"{cat}__count_lo_r"].astype(np.int64)
    out[f"{cat}__obs_evt_mm_hi"] = d7a[f"{cat}__count_hi_mm"].astype(np.int64)
    out[f"{cat}__obs_evt_mm_lo"] = d7a[f"{cat}__count_lo_mm"].astype(np.int64)
    out[f"{cat}__obs_evt_k_hi"]  = d7a[f"{cat}__count_hi_k"].astype(np.int64)
    out[f"{cat}__obs_evt_k_lo"]  = d7a[f"{cat}__count_lo_k"].astype(np.int64)

    # exposure-normalized event rates
    out[f"{cat}__obs_rate_r_hi"]  = _safe_div(out[f"{cat}__obs_evt_r_hi"],  exp_r)
    out[f"{cat}__obs_rate_r_lo"]  = _safe_div(out[f"{cat}__obs_evt_r_lo"],  exp_r)
    out[f"{cat}__obs_rate_mm_hi"] = _safe_div(out[f"{cat}__obs_evt_mm_hi"], exp_mm)
    out[f"{cat}__obs_rate_mm_lo"] = _safe_div(out[f"{cat}__obs_evt_mm_lo"], exp_mm)

    # enrichment vs exposure baseline
    out[f"{cat}__obs_enrich_r_hi"]  = _enrichment(out[f"{cat}__obs_evt_r_hi"],  exp_r)
    out[f"{cat}__obs_enrich_r_lo"]  = _enrichment(out[f"{cat}__obs_evt_r_lo"],  exp_r)
    out[f"{cat}__obs_enrich_mm_hi"] = _enrichment(out[f"{cat}__obs_evt_mm_hi"], exp_mm)
    out[f"{cat}__obs_enrich_mm_lo"] = _enrichment(out[f"{cat}__obs_evt_mm_lo"], exp_mm)

    out[f"{cat}__obs_log2enrich_r_hi"]  = np.log2(out[f"{cat}__obs_enrich_r_hi"]  + EPS).astype(np.float32)
    out[f"{cat}__obs_log2enrich_r_lo"]  = np.log2(out[f"{cat}__obs_enrich_r_lo"]  + EPS).astype(np.float32)
    out[f"{cat}__obs_log2enrich_mm_hi"] = np.log2(out[f"{cat}__obs_enrich_mm_hi"] + EPS).astype(np.float32)
    out[f"{cat}__obs_log2enrich_mm_lo"] = np.log2(out[f"{cat}__obs_enrich_mm_lo"] + EPS).astype(np.float32)

    # ---------- NULL enrichment (optional, recommended) ----------
    if DO_NULL:
        # pull pooled-per-animal null vectors for this category
        # They are object arrays per animal, so we cannot apply thresholds directly per K.
        # So we compute null tail counts by resampling from those pooled vectors per animal.
        #
        # IMPORTANT: this null is "within-category pooled values" not "per-k identity".
        # If you want per-k null attribution, we must consume FP4bA per-animal K-vectors instead.
        #
        # For now: null rate baseline for ROI/modulepair is uniform w.r.t exposure (same as exp baseline).
        # That still gives you obs vs null distribution baseline in *value space*.
        #
        # ==> If you want per-k null attribution, say it and I'll swap in FP4bA K-vectors.
        null_key = "null_" + info["obs_key"].replace("obs_", "")  # maps obs_intra_trimer -> null_intra_trimer
        if null_key not in d6a.files:
            raise KeyError(f"Missing {null_key} in FP6a. Available keys: {d6a.files}")

        null_per_animal = d6a[null_key]  # object array, values only

        # Build null pooled vector (exact) and compute fraction in tails
        null_pool = []
        for v in null_per_animal:
            x = np.asarray(v).astype(np.float32, copy=False).ravel()
            x = x[np.isfinite(x)]
            if x.size:
                null_pool.append(x)
        null_pool = np.concatenate(null_pool, axis=0) if null_pool else np.array([], dtype=np.float32)

        if null_pool.size == 0:
            frac_hi = np.nan
            frac_lo = np.nan
        else:
            frac_hi = float(np.mean(null_pool >= thr_hi))
            frac_lo = float(np.mean(null_pool <= thr_lo))

        out[f"{cat}__null_tailfrac_hi"] = np.float32(frac_hi)
        out[f"{cat}__null_tailfrac_lo"] = np.float32(frac_lo)

        # Expected null event rate per ROI/modulepair under "random hits among eligible entries":
        # rate_null ≈ tail_fraction * 1.0 (because each eligible entry has equal chance).
        # Convert to comparable "events/exposure" by multiplying exposure by tailfrac gives expected null events.
        null_evt_r_hi  = (exp_r  * frac_hi).astype(np.float32)
        null_evt_r_lo  = (exp_r  * frac_lo).astype(np.float32)
        null_evt_mm_hi = (exp_mm * frac_hi).astype(np.float32)
        null_evt_mm_lo = (exp_mm * frac_lo).astype(np.float32)

        out[f"{cat}__null_exp_evt_r_hi"]  = null_evt_r_hi
        out[f"{cat}__null_exp_evt_r_lo"]  = null_evt_r_lo
        out[f"{cat}__null_exp_evt_mm_hi"] = null_evt_mm_hi
        out[f"{cat}__null_exp_evt_mm_lo"] = null_evt_mm_lo

        # obs/null in exposure-normalized rate space
        obs_rate_r_hi  = out[f"{cat}__obs_rate_r_hi"]
        obs_rate_r_lo  = out[f"{cat}__obs_rate_r_lo"]
        obs_rate_mm_hi = out[f"{cat}__obs_rate_mm_hi"]
        obs_rate_mm_lo = out[f"{cat}__obs_rate_mm_lo"]

        null_rate_r_hi  = (null_evt_r_hi  / (exp_r  + EPS)).astype(np.float32)  # ~ frac_hi
        null_rate_r_lo  = (null_evt_r_lo  / (exp_r  + EPS)).astype(np.float32)
        null_rate_mm_hi = (null_evt_mm_hi / (exp_mm + EPS)).astype(np.float32)
        null_rate_mm_lo = (null_evt_mm_lo / (exp_mm + EPS)).astype(np.float32)

        out[f"{cat}__obs_over_null_rate_r_hi"]  = ((obs_rate_r_hi  + EPS) / (null_rate_r_hi  + EPS)).astype(np.float32)
        out[f"{cat}__obs_over_null_rate_r_lo"]  = ((obs_rate_r_lo  + EPS) / (null_rate_r_lo  + EPS)).astype(np.float32)
        out[f"{cat}__obs_over_null_rate_mm_hi"] = ((obs_rate_mm_hi + EPS) / (null_rate_mm_hi + EPS)).astype(np.float32)
        out[f"{cat}__obs_over_null_rate_mm_lo"] = ((obs_rate_mm_lo + EPS) / (null_rate_mm_lo + EPS)).astype(np.float32)

# ---------- Save ----------
out_path = dist_dir / OUT_NAME
if out_path.exists() and not OVERWRITE:
    raise FileExistsError(out_path)

meta["categories"] = list(CATS.keys())
out["params_json"] = json.dumps(meta, sort_keys=True)

np.savez_compressed(out_path, **out)
print("[OK] Saved FP7d:", out_path)

print("\nNOTE:")
print("- This FP7d computes exposure baselines exactly.")
print("- With DO_NULL=True it uses FP6a pooled null values to estimate tail fractions (value-space null).")
print("- If you want per-k identity-preserving null enrichment (stronger), FP7d should consume FP4bA per-animal K-vectors.")
# %%
