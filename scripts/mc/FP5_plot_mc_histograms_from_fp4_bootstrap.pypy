#!/usr/bin/env python3
# %%
"""
FP5 — Plot observed vs null MC histograms USING FP4 bootstrap artifacts.

Expects:
  results/<dataset>/mc_dist/fp4a_mc_hist_boot.npz
  results/<dataset>/mc_dist/fp4b_mc_hist_null_boot.npz

Each NPZ must contain:
  bins
  h_obs, h_ci_lo, h_ci_hi

Outputs:
  fig/<dataset>/dist/mc_hist_obs_vs_null_fp4boot_linear.(png/pdf)
  fig/<dataset>/dist/mc_hist_obs_vs_null_fp4boot_log.(png/pdf)
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from shared_code.fun_paths import get_paths

# =========================
# CONFIG
# =========================
DATASET = "ines_abdallah"
TIMECOURSE_FOLDER = "Timecourses_updated_03052024"
COGNITIVE_FILE = "ROIs.xlsx"
ANAT_LABELS_FILE = "41_Allen.txt"

OBS_FP4_HIST  = "fp4a_mc_dist_all_boot.npz"
NULL_FP4_HIST = "fp4b_mc_dist_null_boot.npz"

DPI = 200
ALPHA_UNDER = 0.10   # under-curve fill
ALPHA_CI = 0.22      # CI band
LW = 1

EPS_LOG = 1e-8

# =========================
# LOAD
# =========================
paths = get_paths(
    dataset_name=DATASET,
    timecourse_folder=TIMECOURSE_FOLDER,
    cognitive_data_file=COGNITIVE_FILE,
    anat_labels_file=ANAT_LABELS_FILE,
)

dist_dir = Path(paths["mc"]) / "mc_dist"
d_obs = np.load(dist_dir / OBS_FP4_HIST, allow_pickle=True)
d_null = np.load(dist_dir / NULL_FP4_HIST, allow_pickle=True)

bins = d_obs["bins"]
centers = 0.5 * (bins[:-1] + bins[1:])

h_obs = d_obs["h_obs"]
h_obs_lo = d_obs["h_ci_lo"]
h_obs_hi = d_obs["h_ci_hi"]

h_null = d_null["h_obs"]
h_null_lo = d_null["h_ci_lo"]
h_null_hi = d_null["h_ci_hi"]

# sanity
assert np.allclose(bins, d_null["bins"]), "Observed and null bins differ. Use same bins in FP4."

out_dir = Path(paths["f_mod"]) / "dist"
out_dir.mkdir(parents=True, exist_ok=True)

def plot_one(ylog: bool, outname: str):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    # # --- Under-curve fills (visual mass)
    # ax.fill_between(centers, 0, h_obs, color="C0", alpha=ALPHA_UNDER)
    # ax.fill_between(centers, 0, h_null, color="C1", alpha=ALPHA_UNDER)

    # --- CI bands
    ax.fill_between(centers, h_obs_lo, h_obs_hi, color="C0", alpha=ALPHA_CI)
    ax.fill_between(centers, h_null_lo, h_null_hi, color="C1", alpha=ALPHA_CI)

    # --- Main curves
    ax.plot(centers, h_obs, lw=LW, color="C0", label="MC")
    ax.plot(centers, h_null, lw=LW, color="C1", label="Null time-shift")

    ax.set_xlabel("MC value")
    ax.set_ylabel("Probability density")
    ax.set_title("MC vs Null distribution (bootstrap CI fill)")
    ax.legend(frameon=False)
    # Remove top and right spines (classic publication style)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    if ylog:
        # clamp to avoid log spikes
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-6)

    fig.tight_layout()
    fig.savefig(out_dir / f"{outname}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(out_dir / f"{outname}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[OK] Saved", out_dir / f"{outname}.png")

plot_one(False, "mc_hist_obs_vs_null_fp4boot_linear")
plot_one(True,  "mc_hist_obs_vs_null_fp4boot_log")
