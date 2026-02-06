#!/usr/bin/env python3
# %%
"""
FP7b — Interpret FP7a tail attribution (EVENTS primary)
      - Option 1: event-rate = hit_evt / opp_evt
      - Option 2: event-share = hit_evt / sum(hit_evt)

Consumes
--------
results/<dataset>/mc/mc_dist/fp7a_tail_attribution_obs_only_evt_norm.npz

Produces
--------
Figures:
  fig/<dataset>/mc/FP7/
    - FP7b_eventrate_modulepair_upper_lower_{cat}.(png|pdf)
    - FP7b_eventshare_modulepair_upper_lower_{cat}.(png|pdf)
    - FP7b_eventrate_top_rois_upper_lower_{cat}.(png|pdf)
    - FP7b_eventshare_top_rois_upper_lower_{cat}.(png|pdf)

Tables (csv):
  results/<dataset>/mc/mc_dist/
    - FP7b_eventrate_top_modulepairs_upper_{cat}.csv
    - FP7b_eventrate_top_modulepairs_lower_{cat}.csv
    - FP7b_eventshare_top_modulepairs_upper_{cat}.csv
    - FP7b_eventshare_top_modulepairs_lower_{cat}.csv
    - FP7b_eventrate_top_rois_upper_{cat}.csv
    - FP7b_eventrate_top_rois_lower_{cat}.csv
    - FP7b_eventshare_top_rois_upper_{cat}.csv
    - FP7b_eventshare_top_rois_lower_{cat}.csv

Notes
-----
- Event-rate controls for “opportunities” (eligible events), so it’s the bias metric.
- Event-share is “who dominates the tail”, useful for narrative but confounded by opportunity size.
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
FP7A_NAME = "fp7a_tail_attribution_obs_only_evt_norm.npz"

SAVE_PNG = True
SAVE_PDF = True
DPI = 200

TOPK = 15
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

def robust_vmax(M, q=0.995):
    x = M[np.isfinite(M)]
    if x.size == 0:
        return 1.0
    return float(np.quantile(x, q))

def load_roi_labels(paths, R: int):
    # prefer labels stored in your canonical bundle if present
    d = Path(paths["preprocessed"]) / "ts_and_meta_2m4m.npz"
    if d.exists():
        z = np.load(d, allow_pickle=True)
        if "anat_labels" in z:
            labels = np.asarray(z["anat_labels"]).astype(str)
            if labels.size >= R:
                return labels[:R]
    return np.array([f"ROI_{i}" for i in range(R)], dtype=object)

def topk_table(values: np.ndarray, labels: np.ndarray, k: int):
    idx = np.argsort(values)[::-1][:k]
    return pd.DataFrame({"label": labels[idx], "value": values[idx], "idx": idx})

def topk_pairs_table(M: np.ndarray, k: int):
    iu = np.triu_indices(M.shape[0], k=0)
    vals = M[iu]
    order = np.argsort(vals)[::-1][:k]
    i = iu[0][order]
    j = iu[1][order]
    return pd.DataFrame({"m1": i, "m2": j, "value": vals[order]})

def safe_div(num: np.ndarray, den: np.ndarray):
    return num / np.maximum(den, 1)

def event_share(x: np.ndarray):
    s = float(np.nansum(x))
    if s <= 0:
        return np.zeros_like(x, dtype=np.float32)
    return (x / s).astype(np.float32)

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
table_dir = dist_dir
table_dir.mkdir(parents=True, exist_ok=True)

print("[FP7b] Loaded:", fp7a_path.name, "| A=", A, "R=", R, "M=", M, "| cats:", cats)

for cat in cats:
    # ---- LOAD NUM / DEN ----
    hit_hi_mm = d[f"{cat}__hit_hi_mm_evt"].astype(np.float32)
    hit_lo_mm = d[f"{cat}__hit_lo_mm_evt"].astype(np.float32)
    hit_hi_r  = d[f"{cat}__hit_hi_r_evt"].astype(np.float32)
    hit_lo_r  = d[f"{cat}__hit_lo_r_evt"].astype(np.float32)

    opp_mm = d[f"{cat}__opp_mm_evt"].astype(np.float32)
    opp_r  = d[f"{cat}__opp_r_evt"].astype(np.float32)

    # ---- OPTION 1: EVENT RATE ----
    rate_hi_mm = safe_div(hit_hi_mm, opp_mm)
    rate_lo_mm = safe_div(hit_lo_mm, opp_mm)
    rate_hi_r  = safe_div(hit_hi_r,  opp_r)
    rate_lo_r  = safe_div(hit_lo_r,  opp_r)

    # ---- OPTION 2: EVENT SHARE ----
    share_hi_mm = event_share(hit_hi_mm)
    share_lo_mm = event_share(hit_lo_mm)
    share_hi_r  = event_share(hit_hi_r)
    share_lo_r  = event_share(hit_lo_r)

    # -------------------------
    # Tables (topK)
    # -------------------------
    top_pairs_rate_hi = topk_pairs_table(rate_hi_mm, TOPK)
    top_pairs_rate_lo = topk_pairs_table(rate_lo_mm, TOPK)
    top_pairs_share_hi = topk_pairs_table(share_hi_mm, TOPK)
    top_pairs_share_lo = topk_pairs_table(share_lo_mm, TOPK)

    top_rois_rate_hi = topk_table(rate_hi_r, roi_labels, TOPK)
    top_rois_rate_lo = topk_table(rate_lo_r, roi_labels, TOPK)
    top_rois_share_hi = topk_table(share_hi_r, roi_labels, TOPK)
    top_rois_share_lo = topk_table(share_lo_r, roi_labels, TOPK)

    top_pairs_rate_hi.to_csv(table_dir / f"FP7b_eventrate_top_modulepairs_upper_{cat}.csv", index=False)
    top_pairs_rate_lo.to_csv(table_dir / f"FP7b_eventrate_top_modulepairs_lower_{cat}.csv", index=False)
    top_pairs_share_hi.to_csv(table_dir / f"FP7b_eventshare_top_modulepairs_upper_{cat}.csv", index=False)
    top_pairs_share_lo.to_csv(table_dir / f"FP7b_eventshare_top_modulepairs_lower_{cat}.csv", index=False)

    top_rois_rate_hi.to_csv(table_dir / f"FP7b_eventrate_top_rois_upper_{cat}.csv", index=False)
    top_rois_rate_lo.to_csv(table_dir / f"FP7b_eventrate_top_rois_lower_{cat}.csv", index=False)
    top_rois_share_hi.to_csv(table_dir / f"FP7b_eventshare_top_rois_upper_{cat}.csv", index=False)
    top_rois_share_lo.to_csv(table_dir / f"FP7b_eventshare_top_rois_lower_{cat}.csv", index=False)

    # -------------------------
    # Figure 1: modulepair heatmaps — EVENT RATE
    # -------------------------
    vmax_hi = robust_vmax(rate_hi_mm, HEATMAP_VMAX_Q)
    vmax_lo = robust_vmax(rate_lo_mm, HEATMAP_VMAX_Q)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7))
    ax = axes[0]
    im = ax.imshow(rate_hi_mm, vmin=0, vmax=vmax_hi, interpolation="nearest")
    ax.set_title(f"{cat} — upper tail event-rate")
    ax.set_xlabel("module")
    ax.set_ylabel("module")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    im = ax.imshow(rate_lo_mm, vmin=0, vmax=vmax_lo, interpolation="nearest")
    ax.set_title(f"{cat} — lower tail event-rate")
    ax.set_xlabel("module")
    ax.set_ylabel("module")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _save(fig, fig_dir / f"FP7b_eventrate_modulepair_upper_lower_{cat}")

    # -------------------------
    # Figure 2: modulepair heatmaps — EVENT SHARE
    # -------------------------
    vmax_hi = robust_vmax(share_hi_mm, HEATMAP_VMAX_Q)
    vmax_lo = robust_vmax(share_lo_mm, HEATMAP_VMAX_Q)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7))
    ax = axes[0]
    im = ax.imshow(share_hi_mm, vmin=0, vmax=vmax_hi, interpolation="nearest")
    ax.set_title(f"{cat} — upper tail event-share")
    ax.set_xlabel("module")
    ax.set_ylabel("module")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    im = ax.imshow(share_lo_mm, vmin=0, vmax=vmax_lo, interpolation="nearest")
    ax.set_title(f"{cat} — lower tail event-share")
    ax.set_xlabel("module")
    ax.set_ylabel("module")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _save(fig, fig_dir / f"FP7b_eventshare_modulepair_upper_lower_{cat}")

    # -------------------------
    # Figure 3: top ROI barplots — EVENT RATE
    # -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

    ax = axes[0]
    df = top_rois_rate_hi.iloc[::-1]
    ax.barh(df["label"], df["value"])
    ax.set_title(f"{cat} — top ROIs (upper) event-rate")
    ax.set_xlabel("hit_evt / opp_evt")

    ax = axes[1]
    df = top_rois_rate_lo.iloc[::-1]
    ax.barh(df["label"], df["value"])
    ax.set_title(f"{cat} — top ROIs (lower) event-rate")
    ax.set_xlabel("hit_evt / opp_evt")

    _save(fig, fig_dir / f"FP7b_eventrate_top_rois_upper_lower_{cat}")

    # -------------------------
    # Figure 4: top ROI barplots — EVENT SHARE
    # -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

    ax = axes[0]
    df = top_rois_share_hi.iloc[::-1]
    ax.barh(df["label"], df["value"])
    ax.set_title(f"{cat} — top ROIs (upper) event-share")
    ax.set_xlabel("hit_evt / sum(hit_evt)")

    ax = axes[1]
    df = top_rois_share_lo.iloc[::-1]
    ax.barh(df["label"], df["value"])
    ax.set_title(f"{cat} — top ROIs (lower) event-share")
    ax.set_xlabel("hit_evt / sum(hit_evt)")

    _save(fig, fig_dir / f"FP7b_eventshare_top_rois_upper_lower_{cat}")

print("[DONE] FP7b outputs:")
print("  figs:", fig_dir)
print("  tables:", table_dir)
# %%
