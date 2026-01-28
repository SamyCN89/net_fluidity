#!/usr/bin/env python3
# %%
"""
FP7b — Interpret FP7a tail attribution counts (rankings + plots)

Consumes:
  results/<dataset>/mc/mc_dist/fp7a_tail_attribution_obs_only.npz
  + (optional) Allen ROI labels file for names (paths["anat_labels_file"])

Produces:
  fig/<dataset>/mc/FP7/
    - FP7b_modulepair_heatmaps_upper_lower_(cat).(png|pdf)
    - FP7b_top_rois_upper_lower_(cat).(png|pdf)
  results/<dataset>/mc/mc_dist/
    - FP7b_top_modulepairs_(cat).csv
    - FP7b_top_rois_(cat).csv
"""

from __future__ import annotations

import json
from pathlib import Path
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
FP7A_NAME = "fp7a_tail_attribution_obs_only.npz"

SAVE_PNG = True
SAVE_PDF = True
DPI = 200

TOPK = 15                 # top ROIs / modulepairs to report
HEATMAP_VMAX_Q = 0.995    # robust cap for heatmaps

# =========================
# Helpers
# =========================
def _save(fig, stem: Path):
    fig.tight_layout()
    if SAVE_PNG:
        fig.savefig(stem.with_suffix(".png"), dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    if SAVE_PDF:
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def robust_vmax(M, q=0.995):
    x = M[np.isfinite(M)]
    if x.size == 0:
        return 1.0
    return float(np.quantile(x, q))

def load_roi_labels(paths, R: int):
    d = Path(paths["preprocessed"]) / "ts_and_meta_2m4m.npz"
    z = np.load(d, allow_pickle=True)

    if "anat_labels" in z:
        labels = np.asarray(z["anat_labels"]).astype(str)
        if labels.size >= R:
            return labels[:R]

    # fallback
    return np.array([f"ROI_{i}" for i in range(R)], dtype=object)


def topk_table(values: np.ndarray, labels: np.ndarray, k: int):
    idx = np.argsort(values)[::-1][:k]
    return pd.DataFrame({"label": labels[idx], "value": values[idx], "idx": idx})

def topk_pairs_table(M: np.ndarray, k: int):
    # upper triangle including diagonal
    iu = np.triu_indices(M.shape[0], k=0)
    vals = M[iu]
    order = np.argsort(vals)[::-1][:k]
    i = iu[0][order]
    j = iu[1][order]
    return pd.DataFrame({"m1": i, "m2": j, "value": vals[order]})

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
dist_dir = mc_dir / MC_DIST_DIRNAME
fp7a_path = dist_dir / FP7A_NAME
if not fp7a_path.exists():
    raise FileNotFoundError(fp7a_path)

d = np.load(fp7a_path, allow_pickle=True)
params = json.loads(d["params_json"].item())

A = int(params["A"])
R = int(params["R"])
M = int(params["n_modules"])
cats = list(params["categories"])

roi_labels = load_roi_labels(paths, R)

# output folders
fig_dir = Path(paths["f_mod"]) / "mod_attributions"
fig_dir.mkdir(parents=True, exist_ok=True)

# also save tables next to mc_dist
table_dir = dist_dir
table_dir.mkdir(parents=True, exist_ok=True)

print("[FP7b] Loaded:", fp7a_path.name, "| A=", A, "R=", R, "M=", M, "| cats:", cats)

for cat in cats:
    # counts
    hi_mm = d[f"{cat}__count_hi_mm"].astype(np.float32)  # (M,M)
    lo_mm = d[f"{cat}__count_lo_mm"].astype(np.float32)
    hi_r  = d[f"{cat}__count_hi_r"].astype(np.float32)   # (R,)
    lo_r  = d[f"{cat}__count_lo_r"].astype(np.float32)

    # normalize: fraction of animals (makes cats comparable)
    hi_mm_frac = hi_mm / float(A)
    lo_mm_frac = lo_mm / float(A)
    hi_r_frac  = hi_r / float(A)
    lo_r_frac  = lo_r / float(A)

    # -------------------------
    # Tables (topK)
    # -------------------------
    top_pairs_hi = topk_pairs_table(hi_mm_frac, TOPK)
    top_pairs_lo = topk_pairs_table(lo_mm_frac, TOPK)
    top_rois_hi  = topk_table(hi_r_frac, roi_labels, TOPK)
    top_rois_lo  = topk_table(lo_r_frac, roi_labels, TOPK)

    top_pairs_hi.to_csv(table_dir / f"FP7b_top_modulepairs_upper_{cat}.csv", index=False)
    top_pairs_lo.to_csv(table_dir / f"FP7b_top_modulepairs_lower_{cat}.csv", index=False)
    top_rois_hi.to_csv(table_dir / f"FP7b_top_rois_upper_{cat}.csv", index=False)
    top_rois_lo.to_csv(table_dir / f"FP7b_top_rois_lower_{cat}.csv", index=False)

    # -------------------------
    # Figure 1: module-pair heatmaps (upper + lower)
    # -------------------------
    vmax_hi = robust_vmax(hi_mm_frac, HEATMAP_VMAX_Q)
    vmax_lo = robust_vmax(lo_mm_frac, HEATMAP_VMAX_Q)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    ax = axes[0]
    im = ax.imshow(hi_mm_frac, vmin=0, vmax=vmax_hi, interpolation="nearest")
    ax.set_title(f"{cat} — upper tail (frac animals)")
    ax.set_xlabel("module")
    ax.set_ylabel("module")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    im = ax.imshow(lo_mm_frac, vmin=0, vmax=vmax_lo, interpolation="nearest")
    ax.set_title(f"{cat} — lower tail (frac animals)")
    ax.set_xlabel("module")
    ax.set_ylabel("module")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _save(fig, fig_dir / f"FP7b_modulepair_heatmaps_upper_lower_{cat}")

    # -------------------------
    # Figure 2: top ROI barplots (upper + lower)
    # -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    ax = axes[0]
    df = top_rois_hi.iloc[::-1]  # plot bottom-to-top
    ax.barh(df["label"], df["value"])
    ax.set_title(f"{cat} — top ROIs in upper tail")
    ax.set_xlabel("fraction of animals (counts/ A)")

    ax = axes[1]
    df = top_rois_lo.iloc[::-1]
    ax.barh(df["label"], df["value"])
    ax.set_title(f"{cat} — top ROIs in lower tail")
    ax.set_xlabel("fraction of animals (counts/ A)")

    _save(fig, fig_dir / f"FP7b_top_rois_upper_lower_{cat}")

print("[DONE] FP7b outputs:")
print("  figs:", fig_dir)
print("  tables:", table_dir)
