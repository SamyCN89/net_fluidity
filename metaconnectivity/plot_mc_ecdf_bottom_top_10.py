#!/usr/bin/env python3
"""
MC trimer analysis – cleaned & stable version

Author: Samy
"""

# =============================================================================
# Imports
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np

from shared_code.fun_loaddata import load_timeseries_bundle
from shared_code.fun_paths import get_paths
from shared_code.fun_utils import set_figure_params

# =============================================================================
# Global plotting config
# =============================================================================
plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.labelsize": 13,
        "axes.titlesize": 12,
    }
)

save_fig = set_figure_params(True)

# =============================================================================
# Paths & data loading
# =============================================================================
paths = get_paths(
    dataset_name="ines_abdallah",
    timecourse_folder="Timecourses_updated_03052024",
    cognitive_data_file="ROIs.xlsx",
    anat_labels_file="41_Allen.txt",
)

bundle = load_timeseries_bundle(
    paths["preprocessed"] / "ts_and_meta_2m4m.npz",
    paths["preprocessed"] / "grouping_data_oip.pkl",
)

mc_mod_dir = paths["mc_mod"]
trimers_dir = paths["trimers"]
fig_dir = paths["f_motif"]
fig_dir.mkdir(exist_ok=True, parents=True)

mask_groups = bundle.mask_groups
label_variables = bundle.label_variables
n_animals = bundle.n_animals
regions = bundle.n_regions

# =============================================================================
# Load MC + trimer data
# =============================================================================
label_ref = "wt2m"
mc_file = mc_mod_dir / (
    f"mc_allegiance_ref(runs={label_ref}_gammaval=1000)=100_"
    f"lag=1_windowsize=7_animals={n_animals}_regions={regions}.npz"
)
mc = np.load(mc_file, allow_pickle=True)


mc_val = mc["mc_val_tril"]  # (animals, edges)
mc_mod_idx = mc["mc_mod_idx"].squeeze()  # (edges,)
mc_idx = mc["mc_idx_tril"]

trimers = np.load(
    trimers_dir
    / (
        f"trimers_allegiance_ref(runs={label_ref}_gammaval=1000)=100_"
        f"lag=1_windowsize=7_animals={n_animals}_regions={regions}.npz"
    )
)
mc_nplets_index = trimers["nplets_index"]  # 0 = 4-plets, >0 = trimers

# =============================================================================
# Masks (centralized – DO NOT INLINE THESE)
# =============================================================================
EDGE = {
    "intra": (mc_mod_idx > 0)[None, :],
    "inter": (mc_mod_idx == 0)[None, :],
    "trimer": (mc_nplets_index > 0)[None, :],
    "tetra": (mc_nplets_index == 0)[None, :],
}


def tail_mask(mc_val, q):
    """Per-animal percentile tail mask"""
    thr = np.nanpercentile(mc_val, q=q, axis=1)
    return mc_val <= thr[:, None] if q < 50 else mc_val >= thr[:, None]


TOP10 = tail_mask(mc_val, 90)
BOT10 = tail_mask(mc_val, 10)


# =============================================================================
# Safe extractor (NO broadcasting bugs)
# =============================================================================
def extract(mc_val, *masks):
    mask = np.ones_like(mc_val, dtype=bool)
    for m in masks:
        mask &= m
    return mc_val[mask]


# =============================================================================
# ECDF
# =============================================================================
def ecdf(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    x = np.sort(x)
    y = np.arange(1, len(x) + 1) / (len(x) + 1)
    return x, y


# =============================================================================
# Shift function
# =============================================================================
def shift(x_ref, x_cmp, pcts):
    return np.percentile(x_cmp, pcts) - np.percentile(x_ref, pcts)


PCTS = np.linspace(5, 95, 19)

# =============================================================================
# ---- TOP 10% ANALYSIS ----
# =============================================================================
x_base = extract(mc_val, TOP10)

x_sets = {
    "3-intra": extract(mc_val, TOP10, EDGE["trimer"], EDGE["intra"]),
    "4-intra": extract(mc_val, TOP10, EDGE["tetra"], EDGE["intra"]),
    "3-inter": extract(mc_val, TOP10, EDGE["trimer"], EDGE["inter"]),
    "4-inter": extract(mc_val, TOP10, EDGE["tetra"], EDGE["inter"]),
}


def true_indices_nested(mask_groups):
    return [
        [np.flatnonzero(arr) for arr in group]
        for group in mask_groups
    ]

indices_mask = true_indices_nested(mask_groups)


# --- Histogram ---
plt.figure(figsize=(7, 5))
for k, x in x_sets.items():
    plt.hist(x, bins=70, density=True, histtype="step", label=k)
plt.yscale("log")
plt.xlabel(r"MC$_{[ij,kl]}$")
plt.ylabel("Density")
plt.title("Top-10% MC tail decomposition")
plt.legend()
plt.tight_layout()
if save_fig:
    plt.savefig(fig_dir / "top10_hist.png")

#%%
#plot mc_val histogram


ref_bins=np.linspace(-0.8,1,100)

count, bins = np.histogram(mc_val.ravel(), bins=ref_bins)


plt.figure(figsize=(7, 10))

plt.subplot(2,1,1)
plt.plot(bins[:-1], count/count.sum(), '-', label='MC values histogram')
for idx, xx in enumerate(indices_mask[2]):  # 3-intra
    mc_subset = mc_val[xx,:].ravel()
    count2, bins = np.histogram(mc_subset, bins=ref_bins)
    plt.plot(bins[:-1], count2/count2.sum(), '-', label=f'{label_variables[2][idx]} MC values histogram')


# count, bins = np.histogram(mc_val[].ravel(), bins=ref_bins)
plt.xlabel("MC values")
plt.ylabel("Count")
plt.yscale("log")
plt.title("Histogram of MC values")
plt.legend()

plt.subplot(2,1,2)

for idx, xx in enumerate(indices_mask[2]):  # 3-intra
    mc_subset = mc_val[xx,:].ravel()
    count2, bins = np.histogram(mc_subset, bins=ref_bins)

    p_ref = count/count.sum()
    p_2 = count2/count2.sum()

    dp = (p_ref-p_2)/(p_ref+p_2)
    dp = np.nan_to_num(dp, nan=0.0)


    plt.plot(bins[:-1], dp, '-',
            alpha=0.5,
            label=f'{label_variables[2][idx]}')

plt.legend()
plt.xlabel("MC values")
plt.ylabel("ΔP")
plt.title(f"Histogram of MC values")
plt.tight_layout()
if save_fig:
    plt.savefig(fig_dir / "mc_val_hist.png")

#%%

# # --- ECDF ---
# plt.figure(figsize=(6, 4))
# for k, x in x_sets.items():
#     xs, ys = ecdf(x)
#     plt.plot(xs, ys, label=k)
# plt.xlabel(r"MC$_{[ij,kl]}$")
# plt.ylabel("ECDF")
# plt.title("Top-10% ECDF")
# plt.legend()
# plt.grid(alpha=0.3)
# plt.tight_layout()
# if save_fig:
#     plt.savefig(fig_dir / "top10_ecdf.png")

# # --- Shift functions ---
# plt.figure(figsize=(6, 4))
# plt.axhline(0, color="gray", ls="--")
# for k, x in x_sets.items():
#     plt.plot(PCTS, shift(x_base, x, PCTS), marker="o", label=k)
# plt.xlabel("Percentile (within top-10%)")
# plt.ylabel("Δ MC vs baseline")
# plt.title("Top-10% shift functions")
# plt.legend()
# plt.grid(alpha=0.3)
# plt.tight_layout()
# if save_fig:
#     plt.savefig(fig_dir / "top10_shift.png")

# # =============================================================================
# # ---- BOTTOM 10% ANALYSIS ----
# # =============================================================================
# x_base = extract(mc_val, BOT10)

# x_sets = {
#     "3-intra": extract(mc_val, BOT10, EDGE["trimer"], EDGE["intra"]),
#     "4-intra": extract(mc_val, BOT10, EDGE["tetra"], EDGE["intra"]),
#     "3-inter": extract(mc_val, BOT10, EDGE["trimer"], EDGE["inter"]),
#     "4-inter": extract(mc_val, BOT10, EDGE["tetra"], EDGE["inter"]),
# }

# plt.figure(figsize=(6, 4))
# plt.axhline(0, color="gray", ls="--")
# for k, x in x_sets.items():
#     plt.plot(PCTS, shift(x_base, x, PCTS), marker="o", label=k)
# plt.xlabel("Percentile (within bottom-10%)")
# plt.ylabel("Δ MC vs baseline")
# plt.title("Bottom-10% shift functions")
# plt.legend()
# plt.grid(alpha=0.3)
# plt.tight_layout()
# if save_fig:
#     plt.savefig(fig_dir / "bottom10_shift.png")

# print("✔ MC trimer analysis completed cleanly.")
