#!/usr/bin/env python3
# %%
"""
FP7e — Enrichment rankings + plots (from FP7d normalization)

Consumes
--------
FP7d:
  results/<dataset>/mc/mc_dist/fp7d_tail_enrichment.npz
    For each category cat in:
      intra_trimer, inter_trimer, intra_tetramer, inter_tetramer
    provides:
      - obs_log2enrich_r_hi/lo      (R,)
      - obs_log2enrich_mm_hi/lo     (M,M)
      - obs_over_null_rate_*        (optional, if DO_NULL=True in FP7d)
      - exp_r, exp_mm, obs_evt_*    (optional for QC)

Optional labels:
  preprocessed/ts_and_meta_2m4m.npz  (anat_labels)

Produces
--------
fig/<dataset>/mc/FP7E/
  - FP7e_modulepair_log2enrich_upper_lower_(cat).(png|pdf)
  - FP7e_top_rois_log2enrich_upper_lower_(cat).(png|pdf)
  - FP7e_top_modulepairs_log2enrich_upper_lower_(cat).(png|pdf)
  - (optional) FP7e_modulepair_obs_over_null_upper_lower_(cat).(png|pdf)
  - (optional) FP7e_top_rois_obs_over_null_upper_lower_(cat).(png|pdf)

results/<dataset>/mc/mc_dist/
  - FP7e_top_rois_upper_(cat).csv / FP7e_top_rois_lower_(cat).csv
  - FP7e_top_modulepairs_upper_(cat).csv / FP7e_top_modulepairs_lower_(cat).csv

What it does
------------
Turns FP7d enrichment arrays into interpretable rankings and figures.

Scientific intent
-----------------
Report “who is enriched in tails after correcting for opportunity (exposure)”.
This is the mechanistic version of FP7b (not biased by ROI/module size).
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
FP7D_NAME = "fp7d_tail_enrichment.npz"

SAVE_PNG = True
SAVE_PDF = True
DPI = 200

TOPK = 15
HEATMAP_VMAX_Q = 0.995   # robust cap for heatmaps
CLIP_LOG2 = 6.0          # clip heatmaps to [-CLIP_LOG2, +CLIP_LOG2] for readability

# plot also obs/null rate ratio if present
PLOT_OBS_OVER_NULL = True

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

def robust_vlim(M: np.ndarray, q=0.995):
    x = M[np.isfinite(M)]
    if x.size == 0:
        return 1.0
    return float(np.quantile(np.abs(x), q))

def load_roi_labels(paths: dict, R: int) -> np.ndarray:
    # prefer the same bundle you already rely on elsewhere
    p = Path(paths["preprocessed"]) / "ts_and_meta_2m4m.npz"
    if p.exists():
        z = np.load(p, allow_pickle=True)
        if "anat_labels" in z.files:
            labels = np.asarray(z["anat_labels"]).astype(str)
            if labels.size >= R:
                return labels[:R]
    return np.array([f"ROI_{i}" for i in range(R)], dtype=object)

def topk_table(values: np.ndarray, labels: np.ndarray, k: int):
    # sort descending; ignore NaNs
    v = values.copy()
    v[~np.isfinite(v)] = -np.inf
    idx = np.argsort(v)[::-1][:k]
    return pd.DataFrame({"label": labels[idx], "value": values[idx], "idx": idx})

def topk_pairs_table(M: np.ndarray, k: int):
    # upper triangle incl diagonal; ignore NaNs
    iu = np.triu_indices(M.shape[0], k=0)
    vals = M[iu].astype(np.float64)
    vals[~np.isfinite(vals)] = -np.inf
    order = np.argsort(vals)[::-1][:k]
    i = iu[0][order]
    j = iu[1][order]
    v = M[i, j]
    return pd.DataFrame({"m1": i, "m2": j, "value": v})

def clip_log2(A: np.ndarray, clip: float) -> np.ndarray:
    out = A.astype(np.float32, copy=True)
    out = np.clip(out, -clip, clip)
    return out

def has_keys(d, keys):
    return all(k in d.files for k in keys)

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
fp7d_path = dist_dir / FP7D_NAME
if not fp7d_path.exists():
    raise FileNotFoundError(fp7d_path)

d = np.load(fp7d_path, allow_pickle=True)
params = json.loads(d["params_json"].item())

cats = list(params["categories"])
R = int(params["R"])
M = int(params["M"])
do_null = bool(params.get("do_null", False))

roi_labels = load_roi_labels(paths, R)

# output folders
fig_dir = Path(paths["f_mod"]) / "FP7E"
fig_dir.mkdir(parents=True, exist_ok=True)
table_dir = dist_dir

print("[FP7e] Loaded:", fp7d_path.name, "| cats:", cats, "| R:", R, "| M:", M, "| do_null:", do_null)
print("[FP7e] figs:", fig_dir)
print("[FP7e] tables:", table_dir)

for cat in cats:
    # -------------------------
    # Load enrichment (log2)
    # -------------------------
    k_hi_r  = f"{cat}__obs_log2enrich_r_hi"
    k_lo_r  = f"{cat}__obs_log2enrich_r_lo"
    k_hi_mm = f"{cat}__obs_log2enrich_mm_hi"
    k_lo_mm = f"{cat}__obs_log2enrich_mm_lo"

    if not has_keys(d, [k_hi_r, k_lo_r, k_hi_mm, k_lo_mm]):
        raise KeyError(f"Missing FP7d keys for cat={cat}. Need {k_hi_r},{k_lo_r},{k_hi_mm},{k_lo_mm}")

    hi_r  = d[k_hi_r].astype(np.float32)    # (R,)
    lo_r  = d[k_lo_r].astype(np.float32)
    hi_mm = d[k_hi_mm].astype(np.float32)   # (M,M)
    lo_mm = d[k_lo_mm].astype(np.float32)

    # clip for heatmaps (don’t distort rankings; only visuals)
    hi_mm_clip = clip_log2(hi_mm, CLIP_LOG2)
    lo_mm_clip = clip_log2(lo_mm, CLIP_LOG2)

    # -------------------------
    # Tables (topK)
    # -------------------------
    top_rois_hi = topk_table(hi_r, roi_labels, TOPK)
    top_rois_lo = topk_table(lo_r, roi_labels, TOPK)
    top_pairs_hi = topk_pairs_table(hi_mm, TOPK)
    top_pairs_lo = topk_pairs_table(lo_mm, TOPK)

    top_rois_hi.to_csv(table_dir / f"FP7e_top_rois_upper_{cat}.csv", index=False)
    top_rois_lo.to_csv(table_dir / f"FP7e_top_rois_lower_{cat}.csv", index=False)
    top_pairs_hi.to_csv(table_dir / f"FP7e_top_modulepairs_upper_{cat}.csv", index=False)
    top_pairs_lo.to_csv(table_dir / f"FP7e_top_modulepairs_lower_{cat}.csv", index=False)

    # -------------------------
    # Figure 1: module-pair heatmaps (log2 enrichment)
    # -------------------------
    vlim = robust_vlim(np.concatenate([hi_mm_clip.ravel(), lo_mm_clip.ravel()]), HEATMAP_VMAX_Q)
    vlim = min(vlim, CLIP_LOG2)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7))
    ax = axes[0]
    im = ax.imshow(hi_mm_clip, vmin=-vlim, vmax=vlim, interpolation="nearest")
    ax.set_title(f"{cat} — upper tail log2 enrichment")
    ax.set_xlabel("module")
    ax.set_ylabel("module")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    im = ax.imshow(lo_mm_clip, vmin=-vlim, vmax=vlim, interpolation="nearest")
    ax.set_title(f"{cat} — lower tail log2 enrichment")
    ax.set_xlabel("module")
    ax.set_ylabel("module")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _save(fig, fig_dir / f"FP7e_modulepair_log2enrich_upper_lower_{cat}")

    # -------------------------
    # Figure 2: top ROI barplots (log2 enrichment)
    # -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

    ax = axes[0]
    df = top_rois_hi.iloc[::-1]
    ax.barh(df["label"], df["value"])
    ax.set_title(f"{cat} — top ROIs (upper tail, log2 enrich)")
    ax.set_xlabel("log2 enrichment (vs exposure)")

    ax = axes[1]
    df = top_rois_lo.iloc[::-1]
    ax.barh(df["label"], df["value"])
    ax.set_title(f"{cat} — top ROIs (lower tail, log2 enrich)")
    ax.set_xlabel("log2 enrichment (vs exposure)")

    _save(fig, fig_dir / f"FP7e_top_rois_log2enrich_upper_lower_{cat}")

    # -------------------------
    # Figure 3: top modulepairs barplots (log2 enrichment)
    # -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

    ax = axes[0]
    df = top_pairs_hi.iloc[::-1].copy()
    df["pair"] = [f"{int(a)}-{int(b)}" for a, b in zip(df["m1"], df["m2"])]
    ax.barh(df["pair"], df["value"])
    ax.set_title(f"{cat} — top modulepairs (upper tail, log2 enrich)")
    ax.set_xlabel("log2 enrichment (vs exposure)")

    ax = axes[1]
    df = top_pairs_lo.iloc[::-1].copy()
    df["pair"] = [f"{int(a)}-{int(b)}" for a, b in zip(df["m1"], df["m2"])]
    ax.barh(df["pair"], df["value"])
    ax.set_title(f"{cat} — top modulepairs (lower tail, log2 enrich)")
    ax.set_xlabel("log2 enrichment (vs exposure)")

    _save(fig, fig_dir / f"FP7e_top_modulepairs_log2enrich_upper_lower_{cat}")

    # -------------------------
    # Optional: obs/null rate ratios
    # -------------------------
    if PLOT_OBS_OVER_NULL and do_null:
        k_hi_r = f"{cat}__obs_over_null_rate_r_hi"
        k_lo_r = f"{cat}__obs_over_null_rate_r_lo"
        k_hi_mm = f"{cat}__obs_over_null_rate_mm_hi"
        k_lo_mm = f"{cat}__obs_over_null_rate_mm_lo"

        if has_keys(d, [k_hi_r, k_lo_r, k_hi_mm, k_lo_mm]):
            rr_hi = d[k_hi_r].astype(np.float32)  # (R,)
            rr_lo = d[k_lo_r].astype(np.float32)
            mm_hi = d[k_hi_mm].astype(np.float32)
            mm_lo = d[k_lo_mm].astype(np.float32)

            # modulepair heatmaps: log2 ratio for symmetry with enrichment
            mm_hi_log2 = np.log2(mm_hi + 1e-12)
            mm_lo_log2 = np.log2(mm_lo + 1e-12)
            mm_hi_log2 = clip_log2(mm_hi_log2, CLIP_LOG2)
            mm_lo_log2 = clip_log2(mm_lo_log2, CLIP_LOG2)

            vlim = robust_vlim(np.concatenate([mm_hi_log2.ravel(), mm_lo_log2.ravel()]), HEATMAP_VMAX_Q)
            vlim = min(vlim, CLIP_LOG2)

            fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7))
            ax = axes[0]
            im = ax.imshow(mm_hi_log2, vmin=-vlim, vmax=vlim, interpolation="nearest")
            ax.set_title(f"{cat} — upper tail log2(obs/null rate)")
            ax.set_xlabel("module")
            ax.set_ylabel("module")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax = axes[1]
            im = ax.imshow(mm_lo_log2, vmin=-vlim, vmax=vlim, interpolation="nearest")
            ax.set_title(f"{cat} — lower tail log2(obs/null rate)")
            ax.set_xlabel("module")
            ax.set_ylabel("module")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            _save(fig, fig_dir / f"FP7e_modulepair_obs_over_null_upper_lower_{cat}")

            # top ROIs by obs/null ratio (NOT log2)
            top_hi = topk_table(rr_hi, roi_labels, TOPK)
            top_lo = topk_table(rr_lo, roi_labels, TOPK)

            fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
            ax = axes[0]
            df = top_hi.iloc[::-1]
            ax.barh(df["label"], df["value"])
            ax.set_title(f"{cat} — top ROIs (upper tail, obs/null rate)")
            ax.set_xlabel("obs rate / null rate")

            ax = axes[1]
            df = top_lo.iloc[::-1]
            ax.barh(df["label"], df["value"])
            ax.set_title(f"{cat} — top ROIs (lower tail, obs/null rate)")
            ax.set_xlabel("obs rate / null rate")

            _save(fig, fig_dir / f"FP7e_top_rois_obs_over_null_upper_lower_{cat}")
        else:
            print(f"[FP7e] do_null=True but obs_over_null keys missing for {cat} (fine).")

print("[DONE] FP7e outputs:")
print("  figs:", fig_dir)
print("  tables:", table_dir)
# %%

