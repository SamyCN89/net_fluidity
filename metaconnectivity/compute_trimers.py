#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:26:30 2024

@author: samy
"""

# %%
import numpy as np
import time

# from functions_analysis import *
from pathlib import Path

from fun_loaddata import *
from fun_dfcspeed import *

from fun_metaconnectivity import (
    compute_mc_nplets_mask_and_index,
    compute_metaconnectivity,
    intramodule_indices_mask,
    get_fc_mc_indices,
    get_mc_region_identities,
    fun_allegiance_communities,
    compute_trimers_identity,
    build_trimer_mask,
    trimers_leaves_fc,
    trimers_root_fc,
    compute_mc_nplets_mask_and_index,
)

from fun_utils import (
    set_figure_params,
    get_paths,
    load_cognitive_data,
    load_timeseries_data,
    load_grouping_data,
)

# =============================================================================
# This code compute
# Load the dataz
# Intersect the 2 and 4 months to have data that have the two datapoints
# ========================== Figure parameters ================================
save_fig = set_figure_params(False)

# =================== Paths and folders =======================================
timeseries_folder = "Timecourses_updated_03052024"
external_disk = True
if external_disk == True:
    root = Path("/media/samy/Elements1/Proyectos/LauraHarsan/script_mc/")
else:
    root = Path("/home/samy/Bureau/Proyect/LauraHarsan/Ines/")

paths = get_paths(
    external_disk=True, external_path=root, timecourse_folder=timeseries_folder
)

# ========================== Load data =========================
cog_data_filtered = load_cognitive_data(paths["sorted"] / "cog_data_sorted_2m4m.csv")
mask_groups, label_variables = load_grouping_data(
    paths["results"] / "grouping_data_oip.pkl"
)

data_ts = load_timeseries_data(paths["sorted"] / "ts_and_meta_2m4m.npz")
ts = data_ts["ts"]
n_animals = data_ts["n_animals"]
regions = data_ts["regions"]
anat_labels = data_ts["anat_labels"]
# %%
# ======================== Metaconnectivity ==========================================
# Parameters speed
PROCESSORS = -1

lag = 1
tau = 5
window_size = 7
window_parameter = (5, 100, 1)

# Parameters allegiance analysis
n_runs_allegiance = 1000
gamma_pt_allegiance = 100

tau_array = np.append(np.arange(0, tau), tau)
lentau = len(tau_array)

time_window_min, time_window_max, time_window_step = window_parameter
time_window_range = np.arange(time_window_min, time_window_max + 1, time_window_step)


# %%Reference metaconnectivity for allegiance matrix
# # ========================Set Reference parameters ==========================================

label_ref = label_variables[1][0]  # The label of the reference matrix
ind_ref = mask_groups[1][0]  # the mask of the reference matrix
# %% 
# =========================Load metaconnectivity modularity =========================
# Load the metaconnectivity modularity dataset
save_filename = paths[
    "mc_mod"
] / f"mc_allegiance_ref(runs={label_ref}_gammaval={n_runs_allegiance})={gamma_pt_allegiance}_lag={lag}_windowsize={window_size}_animals={n_animals}_regions={regions}.npz".replace(
    " ", ""
)

data_analysis = np.load(save_filename, allow_pickle=True)
sort_allegiance = data_analysis["sort_allegiance"]
mc_modules_mask = data_analysis["mc_modules_mask"]
mc_idx = data_analysis["mc_idx_tril"]
fc_idx = data_analysis["fc_idx_tril"]
mc_val = data_analysis["mc_val_tril"]
mc_mod_idx = data_analysis["mc_mod_idx"]
mc_reg_idx = data_analysis["mc_reg_idx"]
fc_reg_idx = data_analysis["fc_reg_idx"]

#%%
# Compute FC for the animals
fc = np.array(
    [
        ts2fc(ts[animal], format_data="2D", method="pearson")
        for animal in range(n_animals)
    ]
)
# Extract the sorted FC values
fc_values = fc[:, fc_idx[:, 0], fc_idx[:, 1]]

# Compute the dFC stream matrix
dfc_stream = np.array(
    [
        ts2dfc_stream(
            ts[animal], window_size, lag=lag, format_data="3D", method="pearson"
        )
        for animal in range(n_animals)
    ]
)

# %%
# ========================Trimers==========================================
# Compute the mask and index
mc_nplets_mask, mc_nplets_index = compute_mc_nplets_mask_and_index(
    regions, allegiance_sort=sort_allegiance
)
#%%
# # ===============================================
# # Genuine trimers MC_{ir,jr}>FC_{ij}
# #Threshold for the FC_{ij}
# # =============================================================================
# Threshold for the FC_{ij}
# Mask to identify valid trimers entries in MC (root+1 Marked)
trimer_mask = mc_nplets_index>0

#Get region index pairs of the trimers
trimers_idx = fc_reg_idx[trimer_mask] # trimers index

# Get the leaves and root of the trimers
trimer_leaves = np.array(
    [trimers_leaves_fc(tri) for tri in trimers_idx]
)  # (n_trimers, 2) trimers leaves region number 
trimers_root = np.squeeze(
    [trimers_root_fc(tri) for tri in trimers_idx]
)  # (n_trimers,) trimers root region number

# Indexing (0-based)
leaf1_idx = trimer_leaves[:, 0] - 1
leaf2_idx = trimer_leaves[:, 1] - 1
root_idx = trimers_root - 1

# Retrieve FC values between leaves (i, j) and root (r)
fc_leaves_values = fc[:, leaf1_idx, leaf2_idx]  # (i, j)leaves values
fc_root_leaf1 = fc[:, root_idx, leaf1_idx] # (r, i) root-leaf1 values
fc_root_leaf2 = fc[:, root_idx, leaf2_idx] # (r, j) root-leaf2 values

fc_leaves_sign = np.sign(fc_leaves_values)  # sign of the FC values between leaves
fc_root_leaf1_sign = np.sign(fc_root_leaf1)  # sign of the FC values between root and leaf1
fc_root_leaf2_sign = np.sign(fc_root_leaf2)  # sign of the FC values between root and leaf2


# %%
# =============================================================================
# For FC_{ir} > FC_{i,j} or FC_{jr} > FC_{i,j}
# =============================================================================
# Get the FC values between root and leaves
fc_root_min = np.minimum(np.abs(fc_root_leaf1), np.abs(fc_root_leaf2))

trimers_genuine_fc_root_leaves = (fc_root_min) > np.abs(fc_leaves_values)


# %%
# =============================================================================
# For MC_{ir,jr} > dFC_{i,j} and given time windows
# =============================================================================

dfc_leaves_values = dfc_stream[
    :, leaf1_idx, leaf2_idx
]
dfc_leaves_values_mean = np.mean(dfc_leaves_values, axis=-1)
# trimers_leaves_fc(dfc_stream)
# %%
# trimers_genuine_mc_root_dfc_leaves = (mc_val[:, trimer_mask]) > (
#     dfc_leaves_values_mean
# )
# Compare metaconnectivity to FC values
# trimers_genuine_mc_root_fc_leaves = (mc_val[:, trimer_mask]) > (
#     fc_leaves_values
# )  # genuine trimers by MC_{ir,jr} > FC_{i,j}

def compute_trimers_genuine(mc_val, dfc_leaves_values_mean, trimer_mask):
    return np.abs(mc_val[:, trimer_mask]) > np.abs(dfc_leaves_values_mean)

trimers_genuine_mc_root_fc_leaves = compute_trimers_genuine(mc_val, fc_leaves_values, trimer_mask)
trimers_genuine_mc_root_dfc_leaves = compute_trimers_genuine(mc_val, dfc_leaves_values_mean, trimer_mask)


# Global trimer mask with animal consistency applied
genuine_trimers_dfc = np.zeros((n_animals, len(trimer_mask)))
genuine_trimers_dfc[:, trimer_mask] = trimers_genuine_mc_root_dfc_leaves

# %%

#%% Save trimers
# Save the computed data
save_filename = (
    paths['trimers'] / 
    f"trimers_allegiance_ref(runs={label_ref}_gammaval={n_runs_allegiance})={gamma_pt_allegiance}_lag={lag}_windowsize={window_size}_animals={n_animals}_regions={regions}.npz".replace(' ','')
)

# Ensure the directory exists before saving
save_filename.parent.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    save_filename,
    nplets_index                      = mc_nplets_index,
    nplets_mask                      = mc_nplets_mask,
    genuine_trimers_dfc             = genuine_trimers_dfc,
)

#%%
from matplotlib import pyplot as plt

label_fc_root_fc_leaves = r"$min(FC_{i,r}, FC_{j,r}) > FC_{i,j}$"
label_mc_root_fc_leaves = r"$MC_{ir,jr} > FC_{i,j}$"
label_mc_root_dfc_leaves = r"$MC_{ir,jr} > mean(dFC_{i,j})$"


plt.figure(1)
plt.clf()
plt.subplot(311)
plt.scatter(
    np.sum(trimers_genuine_fc_root_leaves, axis=0) / n_animals,
    np.sum(trimers_genuine_mc_root_fc_leaves, axis=0) / n_animals,
    alpha=0.4,
    s=3,
    # label =label_fc_root_fc_leaves + ' vs ' + label_mc_root_fc_leaves
)

plt.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=1)
plt.xlabel(label_fc_root_fc_leaves)
plt.ylabel(label_mc_root_fc_leaves)


plt.subplot(312)
plt.scatter(
    np.sum(trimers_genuine_fc_root_leaves, axis=0) / n_animals,
    np.sum(trimers_genuine_mc_root_dfc_leaves, axis=0) / n_animals,
    alpha=0.4,
    s=3,
    c="C1",
    # label =label_fc_root_fc_leaves + ' vs ' + label_mc_root_fc_leaves
)

plt.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=1)
plt.xlabel(label_fc_root_fc_leaves)
plt.ylabel(label_mc_root_dfc_leaves)

plt.subplot(313)
plt.scatter(
    np.sum(trimers_genuine_mc_root_fc_leaves, axis=0) / n_animals,
    np.sum(trimers_genuine_mc_root_dfc_leaves, axis=0) / n_animals,
    alpha=0.4,
    s=3,
    c="C2",
    # label =label_fc_root_fc_leaves + ' vs ' + label_mc_root_fc_leaves
)

plt.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=1)
plt.xlabel(label_mc_root_fc_leaves)
plt.ylabel(label_mc_root_dfc_leaves)
plt.tight_layout()
# , markersize=1)
# plt.subplot(311)
# plt.plot(np.sum(trimers_genuine_fc_root_leaves, axis=0),'.')
# plt.subplot(312)
# plt.plot(np.sum(trimers_genuine_mc_root_fc_leaves, axis=0),'.')
# plt.subplot(313)
# plt.plot(np.sum(trimers_genuine_mc_root_dfc_leaves, axis=0),'.')
# plt.imshow(fc[:,fc_idx[:,0],fc_idx[:,1]].T,
#            interpolation='none',
#            aspect='auto',
#            cmap = 'coolwarm',
#            )
# plt.colorbar()
# plt.clim(-0.6,0.6)
# %%
plt.figure(2)
plt.clf()
setb = mask_groups[2][3]  # setb = mask_groups[0][0]
plt.scatter(
    np.sum(trimers_genuine_mc_root_fc_leaves[setb], axis=0) / np.sum(setb),
    np.sum(trimers_genuine_mc_root_dfc_leaves[setb], axis=0) / np.sum(setb),
    s=3,
    c="C2",
    # label =label_fc_root_fc_leaves + ' vs ' + label_mc_root_fc_leaves
)
plt.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=1)
plt.xlabel(label_mc_root_fc_leaves)
plt.ylabel(label_mc_root_dfc_leaves)
plt.tight_layout()



# %%
