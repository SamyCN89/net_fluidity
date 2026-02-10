#!/usr/bin/env python3
# %%
"""
FP7bG — Group-level tail DRIVER ENRICHMENT (events normalized by exposure)

What it does (scientific intent)
-------------------------------
Turns FP7aG *raw tail-event counts* into *comparable group-level rates* by normalizing
each ROI / module-pair count by the number of eligible opportunities ("exposure")
in that topology class (intra/inter × trimer/tetramer). This is the clean way to
compare groups when group sizes differ.

Key idea
--------
For each group g and category c:
  rate_roi[g,c,r] = count_roi[g,c,r] / exposure_roi[g,c,r]
  rate_mm[g,c,m1,m2] = count_mm[g,c,m1,m2] / exposure_mm[g,c,m1,m2]

Then we also compute a simple across-groups enrichment:
  log2_enrich = log2(rate / mean_rate_across_groups)

Important assumption
--------------------
Exposure is computed from FP3 topology ONLY (same per animal), then multiplied by n_animals in group.
This assumes missing/NaN MC entries are negligible or roughly uniform across animals.
(If you later want to be strict: recompute exposure per animal using finite masks from FP3 mc_val_tril.)

Consumes
--------
- FP7aG:
    results/<dataset>/mc/mc_dist/fp7aG_tail_attribution_obs_only.npz
    keys like: "<group>__<cat>__count_r", "__count_mm", "__n_animals"
- FP3:
    results/<dataset>/mc/mc_indexed/mc_indexed_*.npz
    needs: mc_idx_tril, fc_edge_idx, mc_nplets_index, allegiance_sort
- FP2:
    results/<dataset>/mc/allegiance_ref/allegiance_ref_*.npz
    needs: communities (UNSORTED; will be sorted by FP3 allegiance_sort)

Produces
--------
- results/<dataset>/mc/mc_dist/fp7bG_tail_driver_enrichment.npz
    For each group and cat:
      - rate_r, rate_mm
      - log2_enrich_r, log2_enrich_mm (vs mean across groups)
      - exposure_r_per_animal, exposure_mm_per_animal (stored once per cat)
- results/<dataset>/mc/mc_dist/FP7bG_top_rois_<group>_<cat>.csv
- results/<dataset>/mc/mc_dist/FP7bG_top_modulepairs_<group>_<cat>.csv
- fig/<dataset>/mc/FP7G/ (optional plots)

What it does NOT do
-------------------
- Does NOT do statistical testing (that’s FP7cG/FP7dG).
- Does NOT change thresholds (still anchored by FP6b WT2m in FP7aG).
- Does NOT use nulls.

"""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
FP7AG_NAME = "fp7aG_tail_attribution_obs_only.npz"
FP3_PATTERN = "mc_indexed_*.npz"           # latest
FP2_PATTERN = "allegiance_ref_*.npz"       # latest (only for communities)

# Reporting
TOPK = 15
HEATMAP_VMAX_Q = 0.995

# Plots
MAKE_PLOTS = True
SAVE_PNG = True
SAVE_PDF = True
DPI = 200

OVERWRITE = True

# =========================
# Helpers
# =========================
def find_latest(folder: Path, pattern: str) -> Path:
    hits = sorted(folder.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No files matching {pattern} in {folder}")
    return hits[-1]

def sanitize(s: str) -> str:
    # safe for filenames
    s = s.strip()
    s = re.sub(r"[^\w\s\-\.]", "_", s)  # kill weird chars incl | / :
    s = re.sub(r"\s+", "_", s)
    return s

def robust_vmax(M: np.ndarray, q: float = 0.995) -> float:
    x = M[np.isfinite(M)]
    if x.size == 0:
        return 1.0
    return float(np.quantile(x, q))

def load_roi_labels(paths: dict, R: int) -> np.ndarray:
    # prefer bundle labels if present
    bundle = Path(paths["preprocessed"]) / "ts_and_meta_2m4m.npz"
    if bundle.exists():
        z = np.load(bundle, allow_pickle=True)
        if "anat_labels" in z:
            labels = np.asarray(z["anat_labels"]).astype(str)
            if labels.size >= R:
                return labels[:R]
    return np.array([f"ROI_{i}" for i in range(R)], dtype=object)

def topk_table(values: np.ndarray, labels: np.ndarray, k: int) -> pd.DataFrame:
    order = np.argsort(values)[::-1]
    order = order[np.isfinite(values[order])]
    idx = order[:k]
    return pd.DataFrame({"label": labels[idx], "value": values[idx], "idx": idx})

def topk_pairs_table(M: np.ndarray, k: int) -> pd.DataFrame:
    iu = np.triu_indices(M.shape[0], k=0)
    vals = M[iu]
    order = np.argsort(vals)[::-1]
    order = order[np.isfinite(vals[order])]
    order = order[:k]
    i = iu[0][order]
    j = iu[1][order]
    return pd.DataFrame({"m1": i, "m2": j, "value": vals[order]})

def save_fig(fig, stem: Path):
    fig.tight_layout()
    if SAVE_PNG:
        fig.savefig(stem.with_suffix(".png"), dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    if SAVE_PDF:
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

# =========================
# Exposure computation (per animal, per category)
# =========================
def compute_exposures_per_animal(
    mc_idx: np.ndarray,         # (K,2) FC edge indices (sorted)
    fc_edge_idx: np.ndarray,    # (E,2) ROI pairs per FC edge (sorted)
    comm_edges: np.ndarray,     # (E,) module id per FC edge (sorted, remapped 0..M-1)
    mc_nplets: np.ndarray,      # (K,) >0 trimer
) -> dict:
    """
    Returns dict(cat -> (expo_r, expo_mm)) where:
      expo_r: (R,) counts of ROI appearances across all eligible MC entries k in that cat
      expo_mm: (M,M) counts of module-pair appearances (symmetrized) across eligible k
    """
    E = fc_edge_idx.shape[0]
    R = int(fc_edge_idx.max() + 1)
    M = int(comm_edges.max() + 1)

    e1 = mc_idx[:, 0]
    e2 = mc_idx[:, 1]
    is_intra = comm_edges[e1] == comm_edges[e2]
    is_inter = ~is_intra
    is_trimer = (mc_nplets > 0)
    is_tetramer = ~is_trimer

    cats = {
        "intra_trimer":   is_intra & is_trimer,
        "inter_trimer":   is_inter & is_trimer,
        "intra_tetramer": is_intra & is_tetramer,
        "inter_tetramer": is_inter & is_tetramer,
    }

    out = {}
    for cat, mask in cats.items():
        kk = np.where(mask)[0]
        if kk.size == 0:
            out[cat] = (np.zeros(R, dtype=np.float64), np.zeros((M, M), dtype=np.float64))
            continue

        ee1 = e1[kk]
        ee2 = e2[kk]

        # ROI exposure: each k contributes 4 ROI appearances (2 from each FC edge)
        rois = np.concatenate([fc_edge_idx[ee1], fc_edge_idx[ee2]], axis=1).astype(np.int64)  # (n,4)
        roi_counts = np.bincount(rois.ravel(), minlength=R).astype(np.float64)

        # module-pair exposure: symmetrized
        m1 = comm_edges[ee1].astype(np.int64)
        m2 = comm_edges[ee2].astype(np.int64)
        flat12 = m1 * M + m2
        flat21 = m2 * M + m1
        mm_counts = np.bincount(
            np.concatenate([flat12, flat21], axis=0),
            minlength=M * M
        ).reshape(M, M).astype(np.float64)

        out[cat] = (roi_counts, mm_counts)

    return out

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
fp7ag_path = dist_dir / FP7AG_NAME
if not fp7ag_path.exists():
    raise FileNotFoundError(fp7ag_path)

# ---------- Load FP7aG ----------
d7 = np.load(fp7ag_path, allow_pickle=True)
meta7 = json.loads(d7["params_json"].item())
groups = np.array(meta7["groups"], dtype=object)
cats = ["intra_trimer", "inter_trimer", "intra_tetramer", "inter_tetramer"]

print("[FP7bG] Loaded:", fp7ag_path.name)
print("[FP7bG] Groups:", groups.tolist())

# ---------- Load FP3 ----------
fp3_path = find_latest(mc_dir / "mc_indexed", FP3_PATTERN)
d3 = np.load(fp3_path, allow_pickle=True)

mc_idx = d3["mc_idx_tril"].astype(np.int64)          # (K,2)
fc_edge_idx = d3["fc_edge_idx"].astype(np.int64)     # (E,2) SORTED
mc_nplets = d3["mc_nplets_index"].astype(np.int8)    # (K,)
sort_idx = d3["allegiance_sort"].astype(np.int64)    # (E,)

E = fc_edge_idx.shape[0]
R = int(fc_edge_idx.max() + 1)
roi_labels = load_roi_labels(paths, R)

# ---------- Load FP2 communities and sort like FP3 ----------
fp2_path = find_latest(mc_dir / "allegiance_ref", FP2_PATTERN)
d2 = np.load(fp2_path, allow_pickle=True)
comm_edges = d2["communities"].astype(np.int64)      # UNSORTED (FP2)
comm_edges = comm_edges[sort_idx]                    # SORTED to match FP3 edge space

# Remap module ids to 0..M-1
uniq = np.unique(comm_edges)
remap = {int(u): i for i, u in enumerate(uniq)}
comm_edges = np.array([remap[int(x)] for x in comm_edges], dtype=np.int64)
M = int(comm_edges.max() + 1)

print(f"[FP7bG] FP3: E={E} R={R} | modules M={M}")

# ---------- Compute exposures per animal (topology only) ----------
expo = compute_exposures_per_animal(
    mc_idx=mc_idx,
    fc_edge_idx=fc_edge_idx,
    comm_edges=comm_edges,
    mc_nplets=mc_nplets,
)

# ---------- Build rates and enrichments ----------
# Store arrays: (G,R) per cat and (G,M,M) per cat
G = groups.size
rate_r = {cat: np.full((G, R), np.nan, dtype=np.float64) for cat in cats}
rate_mm = {cat: np.full((G, M, M), np.nan, dtype=np.float64) for cat in cats}

n_animals_g = np.zeros(G, dtype=np.int64)

for gi, g in enumerate(groups):
    for cat in cats:
        k_r = f"{g}__{cat}__count_r"
        k_mm = f"{g}__{cat}__count_mm"
        k_n = f"{g}__{cat}__n_animals"
        if k_r not in d7.files or k_mm not in d7.files or k_n not in d7.files:
            raise KeyError(f"Missing keys for group/cat: {g} / {cat}")

    # group size is repeated per cat; just take one
    n_animals = int(d7[f"{g}__{cats[0]}__n_animals"])
    n_animals_g[gi] = n_animals

    for cat in cats:
        counts_r = d7[f"{g}__{cat}__count_r"].astype(np.float64)
        counts_mm = d7[f"{g}__{cat}__count_mm"].astype(np.float64)

        expo_r_per_animal, expo_mm_per_animal = expo[cat]
        expo_r = expo_r_per_animal * float(n_animals)
        expo_mm = expo_mm_per_animal * float(n_animals)

        # avoid divide by zero
        rr = np.full(R, np.nan, dtype=np.float64)
        okr = expo_r > 0
        rr[okr] = counts_r[okr] / expo_r[okr]

        mm = np.full((M, M), np.nan, dtype=np.float64)
        okm = expo_mm > 0
        mm[okm] = counts_mm[okm] / expo_mm[okm]

        rate_r[cat][gi] = rr
        rate_mm[cat][gi] = mm

print("[FP7bG] Built exposure-normalized rates.")

# ---------- Enrichment vs mean across groups ----------
log2_enrich_r = {}
log2_enrich_mm = {}

for cat in cats:
    mean_r = np.nanmean(rate_r[cat], axis=0)  # (R,)
    mean_mm = np.nanmean(rate_mm[cat], axis=0)  # (M,M)

    # safe log2
    eps = 1e-30
    log2_enrich_r[cat] = np.log2((rate_r[cat] + eps) / (mean_r[None, :] + eps))
    log2_enrich_mm[cat] = np.log2((rate_mm[cat] + eps) / (mean_mm[None, :, :] + eps))

# ---------- Save NPZ ----------
out_npz = dist_dir / "fp7bG_tail_driver_enrichment.npz"
if out_npz.exists() and not OVERWRITE:
    raise FileExistsError(out_npz)

save = {
    "groups": groups.astype(object),
    "cats": np.array(cats, dtype=object),
    "n_animals_g": n_animals_g.astype(np.int64),
    "roi_labels": roi_labels.astype(object),
}

# store exposures once per cat (per animal)
for cat in cats:
    expo_r_pa, expo_mm_pa = expo[cat]
    save[f"{cat}__exposure_r_per_animal"] = expo_r_pa.astype(np.float64)
    save[f"{cat}__exposure_mm_per_animal"] = expo_mm_pa.astype(np.float64)

# store rates/enrichment per group
for cat in cats:
    save[f"{cat}__rate_r"] = rate_r[cat].astype(np.float64)              # (G,R)
    save[f"{cat}__rate_mm"] = rate_mm[cat].astype(np.float64)            # (G,M,M)
    save[f"{cat}__log2_enrich_r"] = log2_enrich_r[cat].astype(np.float64)
    save[f"{cat}__log2_enrich_mm"] = log2_enrich_mm[cat].astype(np.float64)

meta = dict(
    dataset=DATASET,
    fp7ag_path=str(fp7ag_path),
    fp3_path=str(fp3_path),
    fp2_path=str(fp2_path),
    intent="Compare groups by tail-event RATE per eligible opportunity (exposure-normalized).",
    exposure_assumption="Exposure computed from topology only, identical per animal; multiplied by n_animals in group.",
    enrichment_reference="mean_across_groups",
    categories=cats,
    groups=[str(g) for g in groups.tolist()],
)
save["params_json"] = json.dumps(meta, sort_keys=True)

np.savez_compressed(out_npz, **save)
print("[OK] Saved FP7bG NPZ:", out_npz)

# ---------- Save top tables + optional plots ----------
table_dir = dist_dir
fig_dir = Path(paths["f_mod"]) / "FP7G"
fig_dir.mkdir(parents=True, exist_ok=True)

for gi, g in enumerate(groups):
    gsafe = sanitize(str(g))

    for cat in cats:
        # ROI enrichment
        enr_r = log2_enrich_r[cat][gi].astype(np.float64)  # (R,)
        df_rois = topk_table(enr_r, roi_labels, TOPK)
        df_rois.to_csv(table_dir / f"FP7bG_top_rois_{gsafe}_{cat}.csv", index=False)

        # module-pair enrichment
        enr_mm = log2_enrich_mm[cat][gi].astype(np.float64)  # (M,M)
        df_pairs = topk_pairs_table(enr_mm, TOPK)
        df_pairs.to_csv(table_dir / f"FP7bG_top_modulepairs_{gsafe}_{cat}.csv", index=False)

        if not MAKE_PLOTS:
            continue

        # heatmap
        vmax = robust_vmax(enr_mm, HEATMAP_VMAX_Q)
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        im = ax.imshow(enr_mm, vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{g} — {cat} log2 enrichment (module-pairs)")
        ax.set_xlabel("module")
        ax.set_ylabel("module")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        save_fig(fig, fig_dir / f"FP7bG_mm_enrich_{gsafe}_{cat}")

        # ROI barplot (topK)
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        dfp = df_rois.iloc[::-1]
        ax.barh(dfp["label"], dfp["value"])
        ax.set_title(f"{g} — {cat} log2 enrichment (ROIs, top {TOPK})")
        ax.set_xlabel("log2(rate / mean_rate_across_groups)")
        save_fig(fig, fig_dir / f"FP7bG_roi_enrich_{gsafe}_{cat}")

print("[DONE] FP7bG")
print("  NPZ:", out_npz)
print("  tables:", table_dir)
print("  figs:", fig_dir)
# %%
