#!/usr/bin/env python3
"""
Created on Mon Sep 23 13:26:30 2024

@author: samy
"""

# %%
from pathlib import Path

# from functions_analysis import *
import time

import numpy as np

from shared_code.fun_loaddata import load_timeseries_bundle
from shared_code.fun_metaconnectivity import (
    compute_metaconnectivity,
    fun_allegiance_communities,
    get_fc_mc_indices,
    get_mc_region_identities,
    intramodule_indices_mask,
)
from shared_code.fun_paths import get_paths
from shared_code.fun_utils import set_figure_params

# ===============================================================================
# This code compute metaconnectivity and modularity
# ========================== Figure parameters ================================
save_fig = set_figure_params(False)

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
ts = bundle.ts
n_animals = bundle.n_animals
regions = bundle.n_regions
mask_groups = bundle.mask_groups
label_variables = bundle.label_variables

if mask_groups is None or label_variables is None:
    raise ValueError(
        "Grouping data is required; expected masks and labels in the preprocessed bundle."
    )

dataset_name = paths["results"].name
report_root = Path("reports/metaconnectivity") / dataset_name
mc_dir = report_root / "mc"
allegiance_dir = report_root / "allegiance"
mc_mod_dir = report_root / "mc_mod"
for directory in (mc_dir, allegiance_dir, mc_mod_dir):
    directory.mkdir(parents=True, exist_ok=True)

# %%
# ======================== Metaconnectivigty ==========================================
# Parameters speed

PROCESSORS = -1

lag = 1
tau = 5
window_size = 9
window_parameter = (5, 100, 1)

# Parameters allegiance analysis
n_runs_allegiance = 1000
gamma_pt_allegiance = 9

tau_array = np.append(np.arange(0, tau), tau)
lentau = len(tau_array)

time_window_min, time_window_max, time_window_step = window_parameter
time_window_range = np.arange(time_window_min, time_window_max + 1, time_window_step)


# %%compute metaconnectivity
start = time.time()
mc = compute_metaconnectivity(
    ts,
    window_size=window_size,
    lag=lag,
    n_jobs=PROCESSORS,
    save_path=mc_dir,
)
stop = time.time()
print(f"Metaconnectivity time {stop-start}")

# %% Modularity analysis
# # Choose reference condition
# # label_ref = 'good2M_recurrecy' #The label of the reference matrix
# # label_ref = 'wt2M_recurrecy' #The label of the reference matrix
# # =============================================================================
# # Community structered - allegiance matrix
# # Save intramodules_idx, intramodule_indices, mc_modules_mask
# # =============================================================================

# # ========================Communities ==========================================
# #Set reference
label_ref = label_variables[2][0]  # The label of the reference matrix
ind_ref = mask_groups[2][0]  # the mask of the reference matrix
mc_ref = np.mean(mc[ind_ref], axis=0)
# %% Compute allegiance
mc_ref_allegiance_communities, sort_allegiance, contingency_matrix = (
    fun_allegiance_communities(
        mc_ref,
        n_runs=n_runs_allegiance,
        gamma_pt=gamma_pt_allegiance,
        save_path=allegiance_dir,
        ref_name=label_ref,
        n_jobs=PROCESSORS,
    )
)

# %%
# sorted initial mc by communities
mc_allegiance = mc[:, sort_allegiance][:, :, sort_allegiance]
# Optional -fill with 0 the diagonal
idx = np.arange(int(regions * (regions - 1) / 2))
mc_allegiance[..., idx, idx] = (
    np.nan
)  # Zero the diagonal across the last two dimensions

# %% Compute Modules
# ========================Modules==========================================

intramodules_idx, intramodule_indices, mc_modules_mask = intramodule_indices_mask(
    mc_ref_allegiance_communities
)
mc_modules_mask = mc_modules_mask[sort_allegiance][:, sort_allegiance]

# Build basic indices
fc_idx, mc_idx = get_fc_mc_indices(regions, allegiance_sort=sort_allegiance)

# Get the indices of the regions in the functional connectivity matrix
mc_reg_idx, fc_reg_idx = get_mc_region_identities(fc_idx, mc_idx)  # , sort_allegiance)

# Get the indices of the regions in the metaconnectivity matrix
mc_val = mc_allegiance[:, mc_idx[:, 0], mc_idx[:, 1]]
mc_mod_idx = mc_modules_mask[mc_idx[:, 0], mc_idx[:, 1]].astype(int)
# %% Save modularity
save_filename = paths[
    "results"
] / f"mc_allegiance_ref(runs={label_ref}_gammaval={n_runs_allegiance})={gamma_pt_allegiance}_lag={lag}_windowsize={window_size}_animals={n_animals}_regions={regions}.npz".replace(
    " ", ""
)

save_filename = mc_mod_dir / save_filename.name
save_filename.parent.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    save_filename,
    mc=mc_allegiance,
    mc_ref_allegiance_communities=mc_ref_allegiance_communities,
    sort_allegiance=sort_allegiance,
    mc_val_tril=mc_val,
    mc_idx_tril=mc_idx,
    fc_idx_tril=fc_idx,
    mc_modules_mask=mc_modules_mask,
    fc_reg_idx=fc_reg_idx,
    mc_reg_idx=mc_reg_idx,
    mc_mod_idx=mc_mod_idx,
)


# %%
