#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:26:30 2024

@author: samy
"""

#%%
import numpy as np
import time
# from functions_analysis import *
from pathlib import Path

from fun_loaddata import *
from fun_dfcspeed import *

from shared_code.fun_metaconnectivity import *

from fun_utils import (set_figure_params, 
                       get_paths, 
                       load_cognitive_data,
                       load_timeseries_data,
                       load_grouping_data,
                       )
# =============================================================================
# This code compute 
# Load the data
# Intersect the 2 and 4 months to have data that have the two datapoints
# ========================== Figure parameters ================================
save_fig = set_figure_params(False)

# =================== Paths and folders =======================================
timeseries_folder = 'Timecourses_updated_03052024'
external_disk = True
if external_disk==True:
    root = Path('/media/samy/Elements1/Proyectos/LauraHarsan/script_mc/')
else:    
    root = Path('/home/samy/Bureau/Proyect/LauraHarsan/Ines/')

paths = get_paths(external_disk=True,
                  external_path=root,
                  timecourse_folder=timeseries_folder)

# ========================== Load data =========================
cog_data_filtered = load_cognitive_data(paths['sorted'] / 'cog_data_sorted_2m4m.csv')
data_ts = load_timeseries_data(paths['sorted'] / 'ts_and_meta_2m4m.npz')
mask_groups, label_variables = load_grouping_data(paths['results'] / "grouping_data_oip.pkl")


# ========================== Indices ==========================================
ts=data_ts['ts']
n_animals = data_ts['n_animals']
regions = data_ts['regions']
anat_labels = data_ts['anat_labels']


#%%
# ======================== Metaconnectivigty ==========================================
#Parameters speed

PROCESSORS =-1

lag=1
tau=5
window_size = 100
window_parameter = (5,100,1)

#Parameters allegiance analysis
n_runs_allegiance = 1000
gamma_pt_allegiance = 9

tau_array       = np.append(np.arange(0,tau), tau ) 
lentau          = len(tau_array)

time_window_min, time_window_max, time_window_step = window_parameter
time_window_range = np.arange(time_window_min,
                              time_window_max+1,
                              time_window_step)


#%%compute metaconnectivity
start = time.time()
mc = compute_metaconnectivity(ts, 
                              window_size=window_size, 
                              lag=lag, 
                              n_jobs =PROCESSORS,
                              save_path = paths['mc'],
                              )
stop = time.time()
print(f'Metaconnectivity time {stop-start}')

#%% Modularity analysis
# # Choose reference condition
# # label_ref = 'good2M_recurrecy' #The label of the reference matrix
# # label_ref = 'wt2M_recurrecy' #The label of the reference matrix
# # =============================================================================
# # Community structered - allegiance matrix
# # Save intramodules_idx, intramodule_indices, mc_modules_mask
# # =============================================================================

# # ========================Communities ==========================================
# #Set reference
label_ref = label_variables[1][0] #The label of the reference matrix
ind_ref = mask_groups[1][0] # the mask of the reference matrix
mc_ref = np.mean(mc[ind_ref],axis=0)
#%% Compute allegiance
mc_ref_allegiance_communities, sort_allegiance, contingency_matrix = fun_allegiance_communities(mc_ref, 
                                                                                                       n_runs = n_runs_allegiance, 
                                                                                                       gamma_pt = gamma_pt_allegiance, 
                                                                                                       save_path=paths['allegiance'],
                                                                                                       ref_name=label_ref, 
                                                                                                       n_jobs=PROCESSORS,
                                                                                                       )

#sorted initial mc by communities
mc_allegiance = mc[:, sort_allegiance][:, :, sort_allegiance]
#Optional -fill with 0 the diagonal
idx = np.arange(int(regions*(regions-1)/2))
mc_allegiance[..., idx, idx] = np.nan # Zero the diagonal across the last two dimensions

#%% Compute Modules
# ========================Modules==========================================

intramodules_idx, intramodule_indices, mc_modules_mask = intramodule_indices_mask(mc_ref_allegiance_communities)
mc_modules_mask = mc_modules_mask[sort_allegiance][:, sort_allegiance]

# Build basic indices
fc_idx, mc_idx = get_fc_mc_indices(regions, allegiance_sort=sort_allegiance)

# Get the indices of the regions in the functional connectivity matrix
mc_reg_idx, fc_reg_idx = get_mc_region_identities(fc_idx, mc_idx)#, sort_allegiance)

# Get the indices of the regions in the metaconnectivity matrix
mc_val = mc_allegiance[:, mc_idx[:, 0], mc_idx[:, 1]]
mc_mod_idx = mc_modules_mask[mc_idx[:, 0], mc_idx[:, 1]].astype(int)
#%% Save modularity
save_filename = (
    paths['mc_mod'] / 
    f"mc_allegiance_ref(runs={label_ref}_gammaval={n_runs_allegiance})={gamma_pt_allegiance}_lag={lag}_windowsize={window_size}_animals={n_animals}_regions={regions}.npz".replace(' ','')
)

# Ensure the directory exists before saving
save_filename.parent.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    save_filename,
    mc                              = mc_allegiance,
    mc_ref_allegiance_communities   = mc_ref_allegiance_communities,
    sort_allegiance          = sort_allegiance,
    mc_val_tril                     = mc_val,

    mc_idx_tril                     = mc_idx,
    fc_idx_tril                     = fc_idx,
    mc_modules_mask                 = mc_modules_mask,

    fc_reg_idx                      = fc_reg_idx,
    mc_reg_idx                      = mc_reg_idx,
    mc_mod_idx                      = mc_mod_idx,
)



# %%
