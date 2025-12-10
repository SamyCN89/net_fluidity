#!/usr/bin/env python3
"""
Created on Mon Sep 23 13:26:30 2024

Legacy pipeline retained for reference. Updated to rely on shared helpers so it
respects repository paths and bundled metadata.
"""

from pathlib import Path
import time

from fun_dfcspeed import *
from fun_metaconnectivity import (
    compute_metaconnectivity,
    fun_allegiance_communities,
    get_fc_mc_indices,
    get_mc_region_identities,
    intramodule_indices_mask,
)
import numpy as np

from shared_code.fun_loaddata import load_timeseries_bundle
from shared_code.fun_paths import get_paths
from shared_code.fun_utils import set_figure_params

# =============================================================================
# This code compute
# Load the data
# Intersect the 2 and 4 months to have data that have the two datapoints
# ========================== Figure parameters ================================
save_fig = set_figure_params(False)

# =================== Paths and folders =======================================
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
        "Grouping data missing from the preprocessed bundle; expected masks and labels."
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
window_size = 7
window_parameter = (5, 100, 1)

# Parameters allegiance analysis
n_runs_allegiance = 1000
gamma_pt_allegiance = 100

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
mc_ref_allegiance_communities, mc_ref_allegiance_sort, contingency_matrix = (
    fun_allegiance_communities(
        mc_ref,
        n_runs=n_runs_allegiance,
        gamma_pt=gamma_pt_allegiance,
        save_path=allegiance_dir,
        ref_name=label_ref,
        n_jobs=PROCESSORS,
    )
)

# sorted initial mc by communities
mc_allegiance = mc[:, mc_ref_allegiance_sort][:, :, mc_ref_allegiance_sort]
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
mc_modules_mask = mc_modules_mask[mc_ref_allegiance_sort][:, mc_ref_allegiance_sort]

# Build basic indices
fc_indx, mc_idx = get_fc_mc_indices(regions)

# mc_idx = mc_idx[mc_ref_allegiance_sort]
mc_reg_idx, fc_reg_idx = get_mc_region_identities(
    fc_indx, mc_idx, mc_ref_allegiance_sort
)
mc_val = mc_allegiance[:, mc_idx[:, 0], mc_idx[:, 1]]

mc_mod_idx = mc_modules_mask[mc_idx[:, 0], mc_idx[:, 1]].astype(int)
# %% Save modularity
save_filename = (
    mc_mod_dir
    / f"mc_allegiance_ref(runs={label_ref}_gammaval={n_runs_allegiance})={gamma_pt_allegiance}_lag={lag}_windowsize={window_size}_animals={n_animals}_regions={regions}.npz".replace(
        " ", ""
    )
)

np.savez_compressed(
    save_filename,
    mc=mc_allegiance,
    mc_val_tril=mc_val,
    mc_ref_allegiance_communities=mc_ref_allegiance_communities,
    mc_ref_allegiance_sort=mc_ref_allegiance_sort,
    mc_idx_tril=mc_idx,
    fc_reg_idx=fc_reg_idx,
    mc_reg_idx=mc_reg_idx,
    mc_mod_idx=mc_mod_idx,
    mc_modules_mask=mc_modules_mask,
)


# #%%
# # ========================Trimers==========================================
# # Compute trimers
# trimer_index, trimer_reg_id, trimer_apex = compute_trimers_identity(regions)

# # Build trimer mask
# n_fc_edges = int(regions * (regions - 1) / 2)
# mc_nplets_mask = build_trimer_mask(trimer_index, trimer_apex, n_fc_edges)
# mc_nplets_mask = mc_nplets_mask[mc_ref_allegiance_sort][:, mc_ref_allegiance_sort]
# mc_nplets_index = mc_nplets_mask[mc_idx[:, 0], mc_idx[:, 1]]

# stop = time.time()
# print(f"Trimer processing time: {stop - start:.3f} seconds")
# #%%
# # =============================================================================
# # Genuine trimers MC_{ir,jr}>FC_{ij}
# #Threshold for the FC_{ij}
# # =============================================================================

# #Compute FC
# def ts2fc(timeseries, format_data = '2D', method='pearson'):
#     """
#     Calculate functional connectivity from time series data.

#     Parameters:
#     timeseries (array): Time series data of shape (timepoints, nodes).
#     format_data (str): Output format, '2D' for full matrix or '1D' for lower-triangular vector.

#     Returns:
#     fc (array): Functional connectivity matrix ('2D') or vector ('1D').

#     Adapted from Lucas Arbabyazd et al 2020. Methods X, doi: 10.1016/j.neuroimage.2020.117156
#     """
#     # Calculate correlation coefficient matrix
#     if method=='pearson':
#         fc = fast_corrcoef(timeseries)

#         # fc = np.corrcoef(timeseries.T)
#     elif method=='plv':
#         fc = compute_plv_matrix_vectorized(timeseries.T)

#     # Optionally zero out the diagonal for '2D' format
#     if format_data=='2D':
#         np.fill_diagonal(fc,0)#fill the diagonal with 0
#         return fc
#     elif format_data=='1D':
#         # Return the lower-triangular part excluding the diagonal
#         return fc[np.tril_indices_from(fc, k=-1)]
# # animal=0
# fc = np.array([ts2fc(ts[animal], format_data = '2D', method='pearson')
#                for animal in range(n_animals)
#                ])

# fc_values = fc[:,fc_indx[:,0], fc_indx[:,1]]
# fc_values_median = np.median(fc_values,axis=0)

# trimers_leaves_idx = fc_reg_idx[mc_nplets_index>0]
# fc_trimers_leaves_bool = np.alltrue((fc_reg_idx* (mc_nplets_index>0)[:,None,None])>0,axis=(1,2))
# # mc_trimers_leaves_bool = np.alltrue( (mc_reg_idx.T * (mc_nplets_index>0)[:,None]),axis=(1))

# def trimers_leaves_fc(arr):
#     flat = arr.flatten()
#     unique, counts = np.unique(flat, return_counts=True)
#     non_repeated = unique[counts == 1]
#     repeated = unique[counts == 2]
#     return non_repeated
# def trimers_root_fc(arr):
#     flat = arr.flatten()
#     unique, counts = np.unique(flat, return_counts=True)
#     # non_repeated = unique[counts == 1]
#     repeated = unique[counts == 2]
#     return repeated

# # =============================================================================
# # For MC_{ir,jr} > FC_{i,j}
# # =============================================================================
# fc_trimers_leaves_idx = np.array([trimers_leaves_fc(tri_idx) for tri_idx in trimers_leaves_idx]) #trimers leaves region number
# fc_leaves_values = fc[:, fc_trimers_leaves_idx[:,0]-1, fc_trimers_leaves_idx[:,1]-1] # trimer leaves values
# trimers_genuine_mc_root_fc_leaves = ((mc_val[:,(mc_nplets_index>0) ]) > (fc_leaves_values)) # genuine trimers by MC_{ir,jr} > FC_{i,j}

# #%%
# # =============================================================================
# # For FC_{ir} > FC_{i,j} or FC_{jr} > FC_{i,j}
# # =============================================================================
# fc_trimers_root_idx = np.squeeze([trimers_root_fc(tri_idx) for tri_idx in trimers_leaves_idx])
# fc_root_values1 = fc[:, fc_trimers_root_idx-1, fc_trimers_leaves_idx[:,0]-1]
# fc_root_values2 = fc[:, fc_trimers_root_idx-1, fc_trimers_leaves_idx[:,1]-1]
# fc_root_min = np.minimum(np.abs(fc_root_values1), np.abs(fc_root_values2))

# trimers_genuine_fc_root_leaves = ((fc_root_min) > (fc_leaves_values))

# #%%
# # =============================================================================
# # For MC_{ir,jr} > dFC_{i,j} and given time windows
# # =============================================================================
# def ts2dfc_stream(ts, windows_size, lag=None, format_data='2D', method='pearson'):
#     """
#     Calculate dynamic functional connectivity stream (dfc_stream) from time series data.

#     Parameters:
#     ts (array): Time series data of shape (t, n), where t is timepoints, n is regions.
#     windows_size (int): Window size to slide over the ts.
#     lag (int): Shift value for the window. Defaults to W if not specified.
#     format (str): Output format. '2D' for a (l, F) shape, '3D' for a (n, n, F) shape.

#     Returns:
#     dFCstream (array): Dynamic functional connectivity stream.
#     """

#     t_total, n = np.shape(ts)
#     #Not overlap
#     if lag is None:
#         lag = windows_size

#     n_pairs               = n * (n-1)//2 #number of pairwise correlations
#     # Calculate the number of frames/windows
#     frames = (t_total - windows_size)//lag + 1

#     if format_data=='2D':
#         dfc_stream = np.empty((n_pairs, frames))
#     elif format_data=='3D':
#         dfc_stream = np.empty((n, n, frames))


#     for k in range(frames):
#         wstart = k * lag
#         wstop = wstart + windows_size
#         if format_data =='2D':
#             dfc_stream[:, k]    = ts2fc(ts[wstart:wstop, :], '1D', method=method)  # Assuming TS2FC returns a vector
#         elif format_data == '3D':
#             dfc_stream[:, :, k] = ts2fc(ts[wstart:wstop, :], '2D',method=method)  # Assuming TS2FC returns a matrix

#     return dfc_stream

# dfc_stream = np.array([
#                 ts2dfc_stream(ts[animal], window_size, lag=lag, format_data='3D', method='pearson')
#                 for animal in range(n_animals)
#                 ])

# dfc_leaves_values = dfc_stream[:,fc_trimers_leaves_idx[:,0]-1, fc_trimers_leaves_idx[:,1]-1]
# dfc_leaves_values_mean = np.mean(dfc_leaves_values, axis=-1)
# # trimers_leaves_fc(dfc_stream)
# #%%
# trimers_genuine_mc_root_dfc_leaves = ((mc_val[:,(mc_nplets_index>0) ]) > (dfc_leaves_values_mean))


# #%%

# label_fc_root_fc_leaves =r'$min(FC_{i,r}, FC_{j,r}) > FC_{i,j}$'
# label_mc_root_fc_leaves =  r'$MC_{ir,jr} > FC_{i,j}$'
# label_mc_root_dfc_leaves = r'$MC_{ir,jr} > mean(dFC_{i,j})$'


# plt.figure(1)
# plt.clf()
# plt.subplot(311)
# plt.scatter(np.sum(trimers_genuine_fc_root_leaves, axis=0)/n_animals, np.sum(trimers_genuine_mc_root_fc_leaves, axis=0)/n_animals,
#             alpha=0.4,
#             s=3,
#             # label =label_fc_root_fc_leaves + ' vs ' + label_mc_root_fc_leaves
#             )

# plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=1)
# plt.xlabel(label_fc_root_fc_leaves)
# plt.ylabel(label_mc_root_fc_leaves)


# plt.subplot(312)
# plt.scatter(np.sum(trimers_genuine_fc_root_leaves, axis=0)/n_animals, np.sum(trimers_genuine_mc_root_dfc_leaves, axis=0)/n_animals,
#             alpha=0.4,
#             s=3,
#             c='C1',
#             # label =label_fc_root_fc_leaves + ' vs ' + label_mc_root_fc_leaves
#             )

# plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=1)
# plt.xlabel(label_fc_root_fc_leaves)
# plt.ylabel(label_mc_root_dfc_leaves)

# plt.subplot(313)
# plt.scatter(np.sum(trimers_genuine_mc_root_fc_leaves, axis=0)/n_animals, np.sum(trimers_genuine_mc_root_dfc_leaves, axis=0)/n_animals,
#             alpha=0.4,
#             s=3,
#             c='C2'
#             # label =label_fc_root_fc_leaves + ' vs ' + label_mc_root_fc_leaves
#             )

# plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=1)
# plt.xlabel(label_mc_root_fc_leaves)
# plt.ylabel(label_mc_root_dfc_leaves)
# plt.tight_layout()
# #, markersize=1)
# # plt.subplot(311)
# # plt.plot(np.sum(trimers_genuine_fc_root_leaves, axis=0),'.')
# # plt.subplot(312)
# # plt.plot(np.sum(trimers_genuine_mc_root_fc_leaves, axis=0),'.')
# # plt.subplot(313)
# # plt.plot(np.sum(trimers_genuine_mc_root_dfc_leaves, axis=0),'.')
# # plt.imshow(fc[:,fc_indx[:,0],fc_indx[:,1]].T,
# #            interpolation='none',
# #            aspect='auto',
# #            cmap = 'coolwarm',
# #            )
# # plt.colorbar()
# # plt.clim(-0.6,0.6)
# #%%

# plt.figure(2,figsize=(12, 8))
# plt.clf()
# offset = 0.07  # vertical offset between time series
# # for i, ts1 in enumerate(ts[0].T):
#     # plt.plot(ts1 + i * offset, label=f"TS {i+1}")
# # plt.ylim(-0.1,0.75)
# plt.title("MC(i,j)")
# plt.ylabel(r"$MC_{(ij, (kl)^{N2 (N2-1)/2)})}$")
# plt.xlabel("Time")
# plt.tight_layout()
# plt.show()
# #%%Save metaconnectivity, modularity and trimers
# # save_filename = os.path.join(path_results, 'mc/mc_allegiance_ref(runs=%s_gammaval=%s)=%s_lag=%s_windowsize=%s_.npz'%(label_ref, n_runs_allegiance, gamma_pt_allegiance, lag, window_size))

# # save_filename = path_mc_mod / f"mc_allegiance_ref(runs={label_ref}_gammaval={n_runs_allegiance})={gamma_pt_allegiance}_lag={lag}_windowsize={window_size}_animals={n_animals}_regions={regions}.npz".replace(' ','')

# # np.savez_compressed(
# #     save_filename,
# #     mc                              = mc_allegiance,
# #     mc_val_tril            = mc_val,

# #     mc_ref_allegiance_communities   = mc_ref_allegiance_communities,
# #     mc_ref_allegiance_sort          = mc_ref_allegiance_sort,

# #     mc_idx_tril             = mc_idx,
# #     fc_reg_idx             = fc_reg_idx,
# #     mc_reg_idx             = mc_reg_idx,
# #     mc_mod_idx             = mc_mod_idx,
# #     mc_modules_mask                 = mc_modules_mask,

# #     mc_nplets_mask         = mc_nplets_mask,
# #     mc_nplets_idx                  = mc_nplets_index,

# # )

# # #Save allegiance sorted anat labels

# # # np.savetxt(path_sorted / 'anat_labels_sorted.txt')
# # #%% Load data metaconnectivity, modularity and trimers
# # # =============================================================================
# # # Load data
# # # =============================================================================
# # # data_analysis = np.load(os.path.join(path_results, 'mc/mc_allegiance_ref=%s_lag=%s_windowsize=%s_.npz'%(lag, window_size)), allow_pickle=True)
# # # data_analysis = np.load(os.path.join(path_results, 'mc/mc_analysis_data_lag=%s_windowsize=%s_.npz'%(lag, window_size)), allow_pickle=True)

# # data_analysis = np.load(save_filename, allow_pickle=True)
# # mc_allegiance = data_analysis['mc']
# # mc_ref_allegiance_communities           = data_analysis['mc_ref_allegiance_communities']
# # mc_ref_allegiance_sort   = data_analysis['mc_ref_allegiance_sort']

# # mc_modules_mask                 = data_analysis['mc_modules_mask']
# # mc_nplets_mask                  = data_analysis['mc_nplets_mask']
# # mc_idx = data_analysis['mc_idx_tril']

# # mc_val = data_analysis['mc_val_tril']
# # mc_reg_idx             = data_analysis['mc_reg_idx']
# # mc_mod_idx             = data_analysis['mc_mod_idx']
# mc_nplets_index = data_analysis['mc_nplets_idx']
