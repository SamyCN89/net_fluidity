#!/usr/bin/env python3
# %%
"""
FP6cTopology — 2×2 topology grid comparing groups on obs-only PDFs with CI bands.

Panels:
  - obs_intra
  - obs_inter
  - obs_trimer
  - obs_tetramer

Consumes:
  mc_dist/fp6b_groups__<scheme>.npz

Produces:
  fig/<dataset>/mc/FP6Compare/<scheme>/topology_grid/
    - compare_topology_grid_pdf.(png|pdf)
"""

from __future__ import annotations
import json
from math import e
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from shared_code.fun_paths import get_paths

# -------------------------
# GLOBAL STYLE
# -------------------------
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False

# =========================
# CONFIG
# =========================
DATASET = "ines_abdallah"
TIMECOURSE_FOLDER = "Timecourses_updated_03052024"
COGNITIVE_FILE = "ROIs.xlsx"
ANAT_LABELS_FILE = "41_Allen.txt"

MC_DIST_DIRNAME = "mc_dist"

IN_NAME = "fp6b_groups__by_age_geno.npz"  # <--- change to any fp6b_groups__*.npz

# groups to include (None = all)
INCLUDE = None  # e.g. ["age=2m|geno=wt","age=2m|geno=dKI","age=4m|geno=wt","age=4m|geno=dKI"]

# plot
X_LIM = (-0.8, 0.8)
Y_LOG = True
Y_LOG_FLOOR = 1e-6

LW_MAIN = 1
CI_ALPHA = 0.18

SAVE_PNG = True
SAVE_PDF = True
DPI = 200

# the 4 marginals we want
PANELS = [
    ("obs_intra", "Intra-module"),
    ("obs_inter", "Inter-module"),
    ("obs_trimer", "Trimer"),
    ("obs_tetramer", "Tetramer"),
]

# =========================
# Helpers
# =========================
def _bin_centers(bins: np.ndarray) -> np.ndarray:
    return (bins[:-1] + bins[1:]) * 0.5

def _load_pdf_series(d, g: str, key: str):
    need = [
        f"{g}__{key}__pdf_obs",
        f"{g}__{key}__pdf_ci_lo",
        f"{g}__{key}__pdf_ci_hi",
        f"{g}__{key}__n_animals",
    ]
    missing = [k for k in need if k not in d.files]
    if missing:
        raise KeyError(f"Missing keys for group '{g}' condition '{key}': {missing}")
    return dict(
        pdf_obs=d[need[0]].astype(np.float32),
        pdf_lo=d[need[1]].astype(np.float32),
        pdf_hi=d[need[2]].astype(np.float32),
        n_animals=int(d[need[3]]),
    )

def _save(fig, outbase: Path):
    if SAVE_PNG:
        fig.savefig(outbase.with_suffix(".png"), dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    if SAVE_PDF:
        fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def _apply_axes(ax):
    ax.set_xlim(*X_LIM)
    if Y_LOG:
        ax.set_yscale("log")
        ax.set_ylim(bottom=Y_LOG_FLOOR)

# =========================
# MAIN
# =========================
paths = get_paths(DATASET, TIMECOURSE_FOLDER, COGNITIVE_FILE, ANAT_LABELS_FILE)
mc_dir = Path(paths["mc"])
dist_dir = mc_dir / MC_DIST_DIRNAME

in_path = dist_dir / IN_NAME
if not in_path.exists():
    raise FileNotFoundError(in_path)

d = np.load(in_path, allow_pickle=True)
bins = d["bins"].astype(np.float32)
x = _bin_centers(bins)

params = json.loads(d["params_json"].item())
scheme = params.get("scheme", in_path.stem.replace("fp6b_groups__", ""))

groups = [str(g) for g in d["groups"]]
if INCLUDE is not None:
    keep = set(INCLUDE)
    groups = [g for g in groups if g in keep]

print("[FP6cTopology] Loaded:", in_path.name)
print("  scheme:", scheme)
print("  n_groups:", len(groups))
print("  groups:", groups)

# output folder
out_dir = Path(paths["f_mod"]) / "FP6Compare" / scheme / "topology_grid"
out_dir.mkdir(parents=True, exist_ok=True)

# --- Create stable color mapping per group (matplotlib default cycle) ---
# We create the mapping once, so the same group gets same color in all panels.
prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
if not prop_cycle:
    prop_cycle = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

color_map = {g: prop_cycle[i % len(prop_cycle)] for i, g in enumerate(groups)}

# --- 2×2 grid ---
fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), sharex=True, sharey=True)
fig.subplots_adjust(bottom=0.22, top=0.92, wspace=0.12, hspace=0.18)
axes = axes.ravel()

# reserve more bottom margin for legend

for i, (ax, (cond, title)) in enumerate(zip(axes, PANELS, strict=False)):
    if i in (0,2):
        ax.set_ylabel("Density")
    else:
        ax.set_ylabel("")
    if i in (2,3):
        ax.set_xlabel("MC value")
    else:
        ax.set_xlabel("")
    for g in groups:
        s = _load_pdf_series(d, g, cond)
        c = color_map[g]
        label = f"{g} (A={s['n_animals']})"

        ax.plot(x, s["pdf_obs"], lw=LW_MAIN, color=c, label=label)
        floor = Y_LOG_FLOOR if Y_LOG else 0.0
        lo = np.maximum(s["pdf_lo"], floor)
        hi = np.maximum(s["pdf_hi"], floor)
        ax.fill_between(x, lo, hi, color=c, alpha=CI_ALPHA, linewidth=0)


    ax.set_title(title)
    _apply_axes(ax)

# one shared legend (avoid 4 duplicated legends)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9,     bbox_to_anchor=(0.5, 0.06))

fig.suptitle(f"{scheme} MC distribution (PDFs with CI)", y=0.98)

_save(fig, out_dir / "compare_mcdist_grid_pdf")
print("[OK] Saved to:", out_dir)
