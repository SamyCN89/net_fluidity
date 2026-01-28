
#!/usr/bin/env python3


# %%
from __future__ import annotations

import json
from pathlib import Path
from tkinter import font
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shared_code.fun_paths import get_paths
from shared_code.fun_allegiance_v2 import v2_prep_undirected_matrix

# Optional quick Louvain check (on-the-fly)
DO_LOUVAIN_QUICKCHECK = False
GAMMA_QUICK = 1.2
LOUVAIN_SEED = 0
try:
    from brainconn.modularity import modularity_louvain_und_sign
except Exception:
    modularity_louvain_und_sign = None

# =========================
# CONFIG
# =========================
DATASET = "ines_abdallah"
TIMECOURSE_FOLDER = "Timecourses_updated_03052024"
COGNITIVE_FILE = "ROIs.xlsx"
ANAT_LABELS_FILE = "41_Allen.txt"

# Group to average (mc_ref)
REF_GENOTYPE = "wt"
REF_AGE = "2m"

# FP2 scaffold (use latest match by default)
FP2_DIRNAME = "allegiance_ref"
FP2_PATTERN = "allegiance_ref_wt_2m_*.npz"  # change if you want explicit file

# Plot options
DIAG_BLANK = True
COLOR_MODE = "fixed"   # "robust" or "fixed"

ROBUST_Q = 0.995        # percentile for |MC|, e.g. 0.99–0.999
FIXED_VMAX = 0.20       # used if COLOR_MODE=="fixed"


CMAP = "RdBu_r"

DPI = 200

# =========================
# Helpers
# =========================
def find_latest(folder: Path, pattern: str) -> Path:
    hits = sorted(folder.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No files matching {pattern} in {folder}")
    return hits[-1]

def robust_sym_vmax(M: np.ndarray, q: float = 0.995) -> float:
    x = M[np.isfinite(M)]
    if x.size == 0:
        return 1.0
    return float(np.quantile(np.abs(x), q))

def blockiness_score(M: np.ndarray, communities: np.ndarray) -> tuple[float, float]:
    """Mean |M| intra vs inter (diag ignored)."""
    M = np.asarray(M).copy()
    np.fill_diagonal(M, np.nan)
    c = np.asarray(communities)
    same = c[:, None] == c[None, :]
    eye = np.eye(M.shape[0], dtype=bool)
    intra = np.abs(M[same & ~eye])
    inter = np.abs(M[~same])
    return float(np.nanmean(intra)), float(np.nanmean(inter))

# %%
# =========================
# MAIN
# =========================
paths = get_paths(
    dataset_name=DATASET,
    timecourse_folder=TIMECOURSE_FOLDER,
    cognitive_data_file=COGNITIVE_FILE,
    anat_labels_file=ANAT_LABELS_FILE,
)

preproc_dir = Path(paths["preprocessed"])
mc_dir = Path(paths["mc"])

# --- Load latest FP1 mc_raw ---
mc_raw_path = find_latest(mc_dir / "mc_raw", "mc_raw_*.npz")
d1 = np.load(mc_raw_path, allow_pickle=True)
mc = d1["mc"]  # (A, E, E)
mouse_ids_ts = d1["mouse_ids_ts"].astype(str)
age_ts = d1["age_ts"].astype(str)

A, E, E2 = mc.shape
assert E == E2

print("[FP1] Loaded", mc_raw_path.name, "| mc:", mc.shape)

# --- Load latest cog table (per-mouse) ---
cog_csv_path = find_latest(preproc_dir, "cog_data_filtered_*.csv")
cog = pd.read_csv(cog_csv_path)
cog["Name"] = cog["Name"].astype(str)

# --- Build WT 2m session mask ---
ref_mice = cog.loc[cog["Genotype"].astype(str) == REF_GENOTYPE, "Name"].to_numpy(dtype=str)
session_mask = np.isin(mouse_ids_ts, ref_mice) & (age_ts == REF_AGE)
n_ref = int(session_mask.sum())
if n_ref == 0:
    raise RuntimeError(f"Empty reference group: genotype={REF_GENOTYPE} age={REF_AGE}")

print(f"[REF] {REF_GENOTYPE}_{REF_AGE} sessions:", n_ref)

# --- Compute mc_ref ---
mc_ref = np.nanmean(mc[session_mask], axis=0)  # (E, E)

# Optional: blank diag
if DIAG_BLANK:
    np.fill_diagonal(mc_ref, np.nan)

# --- Load FP2 scaffold (sort_idx + communities) ---
fp2_path = find_latest(mc_dir / FP2_DIRNAME, FP2_PATTERN)
d2 = np.load(fp2_path, allow_pickle=True)
sort_idx_fp2 = d2["sort_idx"].astype(int)
communities_fp2 = d2["communities"].astype(int)

if sort_idx_fp2.shape != (E,):
    raise ValueError(f"FP2 sort_idx shape {sort_idx_fp2.shape} != (E,) where E={E}")
if communities_fp2.shape != (E,):
    raise ValueError(f"FP2 communities shape {communities_fp2.shape} != (E,) where E={E}")

print("[FP2] Loaded scaffold:", fp2_path.name, "| n_modules:", len(np.unique(communities_fp2)))

# --- Sort MC by FP2 scaffold ---
mc_sorted = mc_ref[sort_idx_fp2][:, sort_idx_fp2]

# --- Compute module boundaries in sorted order ---
comm_sorted = communities_fp2[sort_idx_fp2]

# Find change points between modules
boundaries = np.where(np.diff(comm_sorted) != 0)[0] + 1

# --- Module sizes ---
module_ids, module_counts = np.unique(comm_sorted, return_counts=True)
# Build a 1D module strip (length E) where each entry = module index
module_strip = np.zeros(E, dtype=int)
start = 0
for i, cnt in enumerate(module_counts):
    module_strip[start:start+cnt] = i
    start += cnt

# print("[INFO] Module sizes (FP2 scaffold):")
# for mid, mcount in zip(module_ids, module_counts):
#     print(f"  Module {mid}: {mcount} links")

# =========================
# Quick-check: Louvain on the fly (optional)
# =========================
if DO_LOUVAIN_QUICKCHECK:
    if modularity_louvain_und_sign is None:
        raise RuntimeError("brainconn not available; cannot run Louvain quickcheck.")
    W = v2_prep_undirected_matrix(mc_ref)  # safe undirected version
    # (brainconn Louvain is stochastic; seed control depends on implementation)
    Ci, Q = modularity_louvain_und_sign(W, gamma=GAMMA_QUICK)
    Ci = np.asarray(Ci).astype(int)
    # crude “sort by community id then within-id index”
    sort_idx_louvain = np.argsort(Ci, kind="stable")
    mc_sorted_louvain = mc_ref[sort_idx_louvain][:, sort_idx_louvain]
    intra0, inter0 = blockiness_score(mc_ref, Ci)
    intra1, inter1 = blockiness_score(mc_sorted_louvain, Ci[sort_idx_louvain])
    print(f"[Louvain quick] gamma={GAMMA_QUICK} Q={float(Q):.4f} n_mod={len(np.unique(Ci))}")
    print(f"[Louvain quick] blockiness Δ(intra-inter): {(intra1-inter1)-(intra0-inter0):.4f}")

# =========================
# Plot + save
# =========================
# color limits
if COLOR_MODE == "fixed":
    vmax = float(FIXED_VMAX)
else:
    vmax = robust_sym_vmax(mc_sorted, q=ROBUST_Q)


vmin = -vmax

out_dir = Path(paths["f_mod"]) / "matrix"
out_dir.mkdir(parents=True, exist_ok=True)

tag = f"mc_ref_sorted_scaffold=FP2_group={REF_GENOTYPE}_{REF_AGE}_v={COLOR_MODE}"
png_path = out_dir / f"{tag}.png"
pdf_path = out_dir / f"{tag}.pdf"

from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, ax = plt.subplots(figsize=(7.5, 6.5))
divider = make_axes_locatable(ax)

# Right-side strip (vertical)
ax_strip = divider.append_axes("right", size="1.5%", pad=0.02)


im = ax.imshow(mc_sorted, vmin=vmin, vmax=vmax, cmap=CMAP, interpolation="nearest")

# --- Overlay module boundaries ---
for b in boundaries:
    ax.axhline(b - 0.5, color="k", lw=0.6, alpha=0.5)
    ax.axvline(b - 0.5, color="k", lw=0.6, alpha=0.5)
for b in boundaries:
    ax_strip.axhline(b - 0.5, color="k", lw=0.6, alpha=0.8)

# --- Plot module stacked strip (aligned) ---
strip_img = ax_strip.imshow(
    module_strip[:, None],
    aspect="auto",
    cmap="tab20",
    interpolation="nearest",
)

ax_strip.set_xticks([])
ax_strip.set_yticks([])


# --- Figure formatting ---
ax.set_title(f"MC mean  ({REF_GENOTYPE} {REF_AGE} ref)", fontsize=14)
ax.set_xlabel("Inter-regional links", fontsize=12)
ax.set_ylabel("Inter-regional links", fontsize=12)

E = mc_sorted.shape[0]
ax.set_xticks((25, E-50), labels=["1 ...", r"... $N^2-N$"], fontsize=12)
ax.set_yticks(
    (10, 70, 710, E-30),
    labels=["1", " .\n.\n.", " .\n.\n.", r"$N^2-N$"],
    fontsize=12,
)
ax.set_xticks([], minor=True)
ax.set_yticks([], minor=True)
ax.tick_params(axis="both", which="both", length=0)


cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.08)
cbar.set_ticks([vmin, 0.0, vmax])
cbar.set_ticklabels([f"{vmin:.1f}", "0", f"{vmax:.1f}"], fontsize=14)
label_mc_formula = r"MC$_{[ij, kl]} = CC[FC_{ij}(t), FC_{kl}(t)]$"
cbar.set_label(label_mc_formula, fontsize=12)



fig.tight_layout()
fig.savefig(png_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

print("[OK] Saved:")
print(" ", png_path)
print(" ", pdf_path)

