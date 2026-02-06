#!/usr/bin/env python3
# %%
"""
FP7c — Paper-style summary figures from FP7a (EVENTS PRIMARY)

Uses:
  Option 1 (recommended): EVENT-RATE = hit_evt / opp_evt
  Option 2 (secondary):   EVENT-SHARE = hit_evt / sum(hit_evt)

Consumes
--------
results/<dataset>/mc/mc_dist/fp7a_tail_attribution_obs_only_evt_norm.npz
+ preprocessed/ts_and_meta_2m4m.npz (anat_labels, optional)

Produces
--------
fig/<dataset>/mc/FP7/
  - FP7c_ROI_tail_drivers_grid_EVENTRATE.(png|pdf)
  - FP7c_ROI_tail_drivers_grid_EVENTSHARE.(png|pdf)        (optional)
  - FP7c_modulepair_heatmaps_summed_EVENTRATE.(png|pdf)
  - FP7c_modulepair_heatmaps_summed_EVENTSHARE.(png|pdf)   (optional)

What you get (science)
----------------------
- EVENT-RATE: “how tail-biased is this ROI/modulepair after controlling for opportunities?”
- EVENT-SHARE: “who contributes most of the tail mass?” (confounded by opportunities)
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
FP7A_NAME = "fp7a_tail_attribution_obs_only_evt_norm.npz"

TOPN_ROI = 10

SAVE_PNG = True
SAVE_PDF = True
DPI = 200

# Toggle Option-2 figures (shares)
MAKE_SHARE_FIGS = True

# robust caps for heatmaps
HEATMAP_VMAX_Q = 0.995

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
    # prefer labels in your bundle if present
    d = Path(paths["preprocessed"]) / "ts_and_meta_2m4m.npz"
    if d.exists():
        z = np.load(d, allow_pickle=True)
        if "anat_labels" in z:
            labels = np.asarray(z["anat_labels"]).astype(str)
            if labels.size >= R:
                return labels[:R]
    return np.array([f"ROI_{i}" for i in range(R)], dtype=object)

def top_idx(values: np.ndarray, n: int):
    # values may contain NaN; treat as -inf
    v = np.asarray(values, dtype=float)
    v = np.where(np.isfinite(v), v, -np.inf)
    return np.argsort(v)[::-1][:n]

def safe_div(num: np.ndarray, den: np.ndarray):
    return num / np.maximum(den, 1)

def event_share(x: np.ndarray):
    s = float(np.nansum(x))
    if s <= 0:
        return np.zeros_like(x, dtype=np.float32)
    return (x / s).astype(np.float32)

def robust_vmax(M, q=0.995):
    x = M[np.isfinite(M)]
    if x.size == 0:
        return 1.0
    return float(np.quantile(x, q))

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

fig_dir = Path(paths["f_mod"]) / "FP7"
fig_dir.mkdir(parents=True, exist_ok=True)

print("[FP7c] Loaded:", fp7a_path.name, "| A=", A, "R=", R, "M=", M, "| cats:", cats)

rows = ["intra_trimer", "inter_trimer", "intra_tetramer", "inter_tetramer"]
rows = [r for r in rows if r in cats]
if not rows:
    raise RuntimeError("No expected categories found in FP7a params['categories'].")

# -----------------------------------------------------------------------------
# Figure A: ROI tail drivers grid — EVENT RATE (Option 1)
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(len(rows), 2, figsize=(13.0, 2.8 * len(rows)), sharex=False)
if len(rows) == 1:
    axes = np.array([axes])

for i, cat in enumerate(rows):
    # numerators + denominators
    hi_evt = d[f"{cat}__hit_hi_r_evt"].astype(np.float32)   # (R,)
    lo_evt = d[f"{cat}__hit_lo_r_evt"].astype(np.float32)
    opp    = d[f"{cat}__opp_r_evt"].astype(np.float32)

    hi_rate = safe_div(hi_evt, opp)
    lo_rate = safe_div(lo_evt, opp)

    idx_hi = top_idx(hi_rate, TOPN_ROI)
    idx_lo = top_idx(lo_rate, TOPN_ROI)

    ax = axes[i, 0]
    ax.barh(roi_labels[idx_hi][::-1], hi_rate[idx_hi][::-1])
    ax.set_title(f"{cat} — upper tail EVENT-RATE")
    ax.set_xlabel("hit_evt / opp_evt")
    ax.set_xlim(0, max(float(np.nanmax(hi_rate[idx_hi])), 1e-9) * 1.05)

    ax = axes[i, 1]
    ax.barh(roi_labels[idx_lo][::-1], lo_rate[idx_lo][::-1])
    ax.set_title(f"{cat} — lower tail EVENT-RATE")
    ax.set_xlabel("hit_evt / opp_evt")
    ax.set_xlim(0, max(float(np.nanmax(lo_rate[idx_lo])), 1e-9) * 1.05)

_save(fig, fig_dir / "FP7c_ROI_tail_drivers_grid_EVENTRATE")
print("[OK] Saved FP7c_ROI_tail_drivers_grid_EVENTRATE")

# -----------------------------------------------------------------------------
# Figure B: modulepair heatmaps summed across categories — EVENT RATE (Option 1)
# -----------------------------------------------------------------------------
sum_hi_evt = np.zeros((M, M), dtype=np.float32)
sum_lo_evt = np.zeros((M, M), dtype=np.float32)
sum_opp_evt = np.zeros((M, M), dtype=np.float32)

for cat in rows:
    sum_hi_evt += d[f"{cat}__hit_hi_mm_evt"].astype(np.float32)
    sum_lo_evt += d[f"{cat}__hit_lo_mm_evt"].astype(np.float32)
    sum_opp_evt += d[f"{cat}__opp_mm_evt"].astype(np.float32)

sum_hi_rate = safe_div(sum_hi_evt, sum_opp_evt)
sum_lo_rate = safe_div(sum_lo_evt, sum_opp_evt)

vmax_hi = robust_vmax(sum_hi_rate, HEATMAP_VMAX_Q)
vmax_lo = robust_vmax(sum_lo_rate, HEATMAP_VMAX_Q)

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))

ax = axes[0]
im = ax.imshow(sum_hi_rate, vmin=0, vmax=vmax_hi, interpolation="nearest")
ax.set_title("Upper tail module-pairs — EVENT-RATE (summed cats)")
ax.set_xlabel("module")
ax.set_ylabel("module")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax = axes[1]
im = ax.imshow(sum_lo_rate, vmin=0, vmax=vmax_lo, interpolation="nearest")
ax.set_title("Lower tail module-pairs — EVENT-RATE (summed cats)")
ax.set_xlabel("module")
ax.set_ylabel("module")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

_save(fig, fig_dir / "FP7c_modulepair_heatmaps_summed_EVENTRATE")
print("[OK] Saved FP7c_modulepair_heatmaps_summed_EVENTRATE")

# -----------------------------------------------------------------------------
# Optional: Option 2 SHARE figures (good for narrative, not “drivers”)
# -----------------------------------------------------------------------------
if MAKE_SHARE_FIGS:
    # ROI SHARE grid
    fig, axes = plt.subplots(len(rows), 2, figsize=(13.0, 2.8 * len(rows)), sharex=False)
    if len(rows) == 1:
        axes = np.array([axes])

    for i, cat in enumerate(rows):
        hi_evt = d[f"{cat}__hit_hi_r_evt"].astype(np.float32)
        lo_evt = d[f"{cat}__hit_lo_r_evt"].astype(np.float32)

        hi_share = event_share(hi_evt)
        lo_share = event_share(lo_evt)

        idx_hi = top_idx(hi_share, TOPN_ROI)
        idx_lo = top_idx(lo_share, TOPN_ROI)

        ax = axes[i, 0]
        ax.barh(roi_labels[idx_hi][::-1], hi_share[idx_hi][::-1])
        ax.set_title(f"{cat} — upper tail EVENT-SHARE")
        ax.set_xlabel("hit_evt / sum(hit_evt)")
        ax.set_xlim(0, max(float(np.nanmax(hi_share[idx_hi])), 1e-12) * 1.05)

        ax = axes[i, 1]
        ax.barh(roi_labels[idx_lo][::-1], lo_share[idx_lo][::-1])
        ax.set_title(f"{cat} — lower tail EVENT-SHARE")
        ax.set_xlabel("hit_evt / sum(hit_evt)")
        ax.set_xlim(0, max(float(np.nanmax(lo_share[idx_lo])), 1e-12) * 1.05)

    _save(fig, fig_dir / "FP7c_ROI_tail_drivers_grid_EVENTSHARE")
    print("[OK] Saved FP7c_ROI_tail_drivers_grid_EVENTSHARE")

    # modulepair SHARE heatmaps (summed cats)
    sum_hi_share = event_share(sum_hi_evt)
    sum_lo_share = event_share(sum_lo_evt)

    vmax_hi = robust_vmax(sum_hi_share, HEATMAP_VMAX_Q)
    vmax_lo = robust_vmax(sum_lo_share, HEATMAP_VMAX_Q)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))

    ax = axes[0]
    im = ax.imshow(sum_hi_share, vmin=0, vmax=vmax_hi, interpolation="nearest")
    ax.set_title("Upper tail module-pairs — EVENT-SHARE (summed cats)")
    ax.set_xlabel("module")
    ax.set_ylabel("module")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    im = ax.imshow(sum_lo_share, vmin=0, vmax=vmax_lo, interpolation="nearest")
    ax.set_title("Lower tail module-pairs — EVENT-SHARE (summed cats)")
    ax.set_xlabel("module")
    ax.set_ylabel("module")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _save(fig, fig_dir / "FP7c_modulepair_heatmaps_summed_EVENTSHARE")
    print("[OK] Saved FP7c_modulepair_heatmaps_summed_EVENTSHARE")

print("[DONE] FP7c outputs in:", fig_dir)
# %%
