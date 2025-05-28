#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 23:13:49 2025

@author: samy
"""

import numpy as np
import os
import pandas as pd
import pickle
from pathlib import Path
from fun_utils import (filename_sort_mat, 
                       split_groups_by_age, 
                       extract_hash_numbers, 
                       load_matdata, 
                       classify_phenotypes, 
                       set_figure_params,
                       load_cogdata_sorted,
                       get_paths,
                       
                       )
import matplotlib.pyplot as plt
import time

#%% Figure parameters
#========================== Figure parameters ================================
# Set figure parameters globally

# save_fig = set_figure_params(True)
save_fig = set_figure_params(False)
bins_parameter=200

# =============================================================================
#cognitive_data_ts_sorted.py
# =============================================================================
# =============================================================================
# 1. Rearange the dataset in which
# cognitive_data_ts_sorted.py 
#
#%%Gets paths
paths = get_paths()
#%% Load sorted data
# data_ts, cog_data_filtered, mask_groups, label_variables = load_cogdata_sorted(paths)
    
# ts = data_ts['ts']
# n_animals = int(data_ts['n_animals'])
# regions = data_ts['regions']
# total_tp = data_ts['total_tp']
# is_2month_old = data_ts['is_2month_old']
# anat_labels= data_ts['anat_labels']

# ========================== Parameters ================================
PROCESSORS =-1
# processors = -1

lag=1
tau=5
window_size = 7
window_parameter = (5,100,1)

#Parameters allegiance analysis
n_runs_allegiance = 1000
gamma_pt_allegiance = 100

tau_array       = np.append(np.arange(0,tau), tau ) 
lentau          = len(tau_array)

time_window_min, time_window_max, time_window_step = window_parameter
time_window_range = np.arange(time_window_min,
                              time_window_max+1,
                              time_window_step)

#%%

# =================== Paths and folders =======================================
external_disk = True
if external_disk==True:
    root = Path('/media/samy/Elements1/Proyectos/LauraHarsan/script_mc/')
else:    
    root = Path('/home/samy/Bureau/Proyect/LauraHarsan/Ines/')

folders = {'2mois': 'TC_2months', '4mois': 'TC_4months'}
# folders = {'2mois': 'Lot3_2mois', '4mois': 'Lot3_4mois'}
# paths['timeseries'] = root + 'Timecourses_updated/' # Old data
# paths['cog_data']   = os.path.join(paths['timeseries'], 'Behaviour_exclusions_ROIs_female.xlsx')

#Path results
path_results = root / 'results'
path_mc_mod = path_results / 'mc_mod/'
paths['sorted'] = path_results / 'sorted_data/'
paths['timeseries'] = path_results / 'Timecourses_updated_03052024'
paths['cog_data']   = paths['timeseries'] / 'ROIs.xlsx'

#Path figures
path_figures = root / 'fig'
path_allegiance = path_figures / 'allegiance'
path_modularity = path_figures / 'modularity'

def get_paths(external_disk=True):
    root = Path('/media/samy/Elements1/Proyectos/LauraHarsan/script_mc/' if external_disk 
                else '/home/samy/Bureau/Proyect/LauraHarsan/Ines/')
    return {
        'root': root,
        'results': root / 'results',
        'timeseries': root / 'results/Timecourses_updated_03052024',
        'cog_data': root / 'results/Timecourses_updated_03052024/ROIs.xlsx',
        'sorted': root / 'results/sorted_data/',
        'mc_mod': root / 'results/mc_mod/',
        'allegiance': root / 'results/allegiance/',
        'figures': root / 'fig',
    }

#%%

#Specific hash/filename
grouping_per_sex_hash = paths['sorted'] / "grouping_data_per_sex(gen_phen).pkl"
grouping_hash = paths['sorted'] / "grouping_data_oip.pkl"
cog_data_filtered_filename = paths['sorted'] / 'cog_data_sorted_2m4m.csv'
ts_data_filename = paths['sorted'] / 'ts_and_meta_2m4m.npz'


# =================== Load data snippets ======================================
#Cog data
cog_data_filtered = pd.read_csv(cog_data_filtered_filename)

#Time series
data_ts = np.load(ts_data_filename)

ts=data_ts['ts']
n_animals = data_ts['n_animals']
total_tp = data_ts['total_tp']
regions = data_ts['regions']
is_2month_old = data_ts['is_2month_old']
anat_labels= data_ts['anat_labels']

#Grouping
with open(grouping_per_sex_hash, "rb") as f:
    mask_groups_per_sex, label_variables_per_sex = pickle.load(f)

with open(grouping_hash, "rb") as f:
    mask_groups, label_variables = pickle.load(f)
# =============================================================================
# Compute metaconnectivity - compute_metaconnectivity_modularity.py
# =============================================================================

label_ref = label_variables[0][0] #The label of the reference matrix
ind_ref = mask_groups[0][0] # the mask of the reference matrix

mc_data_filename = f"mc_allegiance_ref(runs={label_ref}_gammaval={n_runs_allegiance})={gamma_pt_allegiance}_lag={lag}_windowsize={window_size}_animals={n_animals}_regions={regions}.npz".replace(' ','')

data_mc_mod_filename = path_mc_mod /mc_data_filename 
# data_mc_mod = np.load(os.path.join(path_results, 'mc/mc_allegiance_ref=%s_lag=%s_windowsize=%s_.npz'%(lag, window_size)), allow_pickle=True)
data_mc_mod = np.load(data_mc_mod_filename, allow_pickle=True)
# data_mc_mod = np.load(os.path.join(path_results, 'mc/mc_analysis_data_lag=%s_windowsize=%s_.npz'%(lag, window_size)), allow_pickle=True)

#MC sorted
mc_allegiance = data_mc_mod['mc']
mc_val                 = data_mc_mod['mc_val_tril']
mc_modules_mask                 = data_mc_mod['mc_modules_mask']
mc_idx = data_mc_mod['mc_idx_tril']

#Community values
mc_ref_allegiance_communities           = data_mc_mod['mc_ref_allegiance_communities']
sort_allegiance   = data_mc_mod['sort_allegiance']

#Indices MC, FC, regions
fc_reg_idx             = data_mc_mod['fc_reg_idx']
mc_reg_idx             = data_mc_mod['mc_reg_idx']
mc_mod_idx             = data_mc_mod['mc_mod_idx']

#%%
#MC motifs
mc_nplets_mask                  = data_mc_mod['mc_nplets_mask']
mc_nplets_index        = data_mc_mod['mc_nplets_idx']

#mc_mod_idx = np.squeeze(mc_mod_idx)
#%%
# =============================================================================
# Plot_metaconnectivity_modularity.py
# =============================================================================

# =============================================================================
# This code uses
#   cog_data_filtered, data_ts, grouping_data, and data_mc_mod

# Plot
    # PLOT MC using allegiance reference
    # Plot MC for each individual
    # Plot MC intra/inter modular per group
    # Plot MC intra/inter for each individual
    # Plot MC modules for each individual
    # plot MC modules per group
# =============================================================================

label_mclinks = r'Inter-regional links'
label_mc_formula = r'MC$_{[ij, kl]} = CC[FC_{ij}(t), FC_{kl}(t)]$'
label_yhist = 'Probability density'

label_fc_root_fc_leaves =r'$min(FC_{i,r}, FC_{j,r}) > FC_{i,j}$'
label_mc_root_fc_leaves =  r'$MC_{ir,jr} > FC_{i,j}$'
label_mc_root_dfc_leaves = r'$MC_{ir,jr} > mean(dFC_{i,j})$'

label_trimer = r'Trimer = MC$_{[ir, jr]}$'
#%% Plot_metaconnectivity_modularity.py
# =============================================================================
# Plot_metaconnectivity_modularity.py
# =============================================================================

# =============================================================================
# This code uses
#   cog_data_filtered, data_ts, grouping_data, and data_mc_mod

# Plot
    # Plot MC trimer intra/inter per group
    
    
    # Plot MC for each individual
    # Plot MC intra/inter modular per group
    # Plot MC intra/inter for each individual
    # Plot MC modules for each individual
    # plot MC modules per group
# =============================================================================




















