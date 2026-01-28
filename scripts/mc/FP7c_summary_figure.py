#!/usr/bin/env python3
# %%
"""
FP7c — Paper-style summary figures from FP7a tail attribution.

Consumes:
  results/<dataset>/mc/mc_dist/fp7a_tail_attribution_obs_only.npz
  + preprocessed/ts_and_meta_2m4m.npz (anat_labels)

Produces:
  fig/<dataset>/mc/FP7/
    - FP7c_ROI_tail_drivers_grid.(png|pdf)
    - FP7c_modulepair_tail_heatmaps_summed.(png|pdf)
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
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

TOPN_ROI = 10

SAVE_PNG = True
SAVE_PDF = True
DPI = 200

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

def load_roi_labels(paths, R: int):
    d = Path(paths["preprocessed"]) / "ts_and_meta_2m4m.npz"
    z = np.load(d, allow_pickle=True)
    if "anat_labels" in z:
        labels = np.asarray(z["anat_labels"]).astype(str)
        if labels.size >= R:
            return labels[:R]
    return np.array([f"ROI_{i}" for i in range(R)], dtype=object)

def top_idx(values: np.ndarray, n: int):
    idx = np.argsort(values)[::-1][:n]
    return idx

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
d = np.load(fp7a_path, allow_pickle=True)
params = json.loads(d["params_json"].item())

A = int(params["A"])
R = int(params["R"])
M = int(params["n_modules"])
cats = list(params["categories"])

roi_labels = load_roi_labels(paths, R)

fig_dir = Path(paths["f_mod"]) / 'mod_resume'
fig_dir.mkdir(parents=True, exist_ok=True)

print("[FP7c] Loaded:", fp7a_path.name, "| A=", A, "R=", R, "M=", M, "| cats:", cats)

# -----------------------------------------------------------------------------
# Figure A: ROI tail drivers grid (4 rows × 2 cols)
# -----------------------------------------------------------------------------
rows = ["intra_trimer", "inter_trimer", "intra_tetramer", "inter_tetramer"]
# enforce consistent ordering even if params differ
rows = [r for r in rows if r in cats]

fig, axes = plt.subplots(len(rows), 2, figsize=(12.5, 2.7 * len(rows)), sharex=False)
if len(rows) == 1:
    axes = np.array([axes])

for i, cat in enumerate(rows):
    hi = d[f"{cat}__count_hi_r_anim"].astype(np.float32) / float(A)
    lo = d[f"{cat}__count_lo_r_anim"].astype(np.float32) / float(A)

    # pick top ROIs separately for upper and lower
    idx_hi = top_idx(hi, TOPN_ROI)
    idx_lo = top_idx(lo, TOPN_ROI)

    ax = axes[i, 0]
    ax.barh(roi_labels[idx_hi][::-1], hi[idx_hi][::-1])
    ax.set_title(f"{cat} — upper tail drivers")
    ax.set_xlabel("fraction of animals")
    ax.set_xlim(0, max(hi[idx_hi].max(), 1e-6) * 1.05)

    ax = axes[i, 1]
    ax.barh(roi_labels[idx_lo][::-1], lo[idx_lo][::-1])
    ax.set_title(f"{cat} — lower tail drivers")
    ax.set_xlabel("fraction of animals")
    ax.set_xlim(0, max(lo[idx_lo].max(), 1e-6) * 1.05)

_save(fig, fig_dir / "FP7c_ROI_tail_drivers_grid")
print("[OK] Saved FP7c_ROI_tail_drivers_grid")

# -----------------------------------------------------------------------------
# Figure B: modulepair heatmaps summed across categories (upper vs lower)
# -----------------------------------------------------------------------------
sum_hi = np.zeros((M, M), dtype=np.float32)
sum_lo = np.zeros((M, M), dtype=np.float32)

for cat in rows:
    sum_hi += d[f"{cat}__count_hi_mm_anim"].astype(np.float32) / float(A)
    sum_lo += d[f"{cat}__count_lo_mm_anim"].astype(np.float32) / float(A)

# robust cap for visibility
vmax_hi = float(np.quantile(sum_hi[np.isfinite(sum_hi)], 0.995)) if np.isfinite(sum_hi).any() else 1.0
vmax_lo = float(np.quantile(sum_lo[np.isfinite(sum_lo)], 0.995)) if np.isfinite(sum_lo).any() else 1.0

fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7))

ax = axes[0]
im = ax.imshow(sum_hi, vmin=0, vmax=vmax_hi, interpolation="nearest")
ax.set_title("Upper tail module-pairs (summed)")
ax.set_xlabel("module")
ax.set_ylabel("module")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax = axes[1]
im = ax.imshow(sum_lo, vmin=0, vmax=vmax_lo, interpolation="nearest")
ax.set_title("Lower tail module-pairs (summed)")
ax.set_xlabel("module")
ax.set_ylabel("module")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

_save(fig, fig_dir / "FP7c_modulepair_tail_heatmaps_summed")
print("[OK] Saved FP7c_modulepair_tail_heatmaps_summed")

print("[DONE] FP7c outputs in:", fig_dir)
