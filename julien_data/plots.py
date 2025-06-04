#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 14:42:38 2023

@author: samy
"""

#%%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import brainconn as bct
import os
import time

from joblib import Parallel, delayed, parallel_backend
import pandas as pd
# from functions_analysis import *
from scipy.io import loadmat, savemat
from scipy.special import erfc
from scipy.stats import pearsonr, spearmanr

from shared_code.fun_loaddata import *  # Import only needed functions
from shared_code.fun_dfcspeed import parallel_dfc_speed_oversampled_series
from shared_code.fun_utils import set_figure_params
from shared_code.fun_paths import get_paths
from tqdm import tqdm

from shared_code.shared_code.fun_dfcspeed import get_tenet4window_range



#%% Define paths, folders and hash
# ------------------------ Configuration ------------------------

paths = get_paths(dataset_name='julien_caillette', 
                  timecourse_folder='time_courses',
                  cognitive_data_file='mice_groups_comp_index.xlsx',
                  anat_labels_file='all_ROI_coimagine.txt')

#%%
SAVE_DATA = True

#%%
# ------------------------ Load Data ------------------------

data_ts_pre = load_npz_dict(paths['preprocessed'] / Path('ts_filtered_unstacked.npz'))
ts = data_ts_pre['ts']
n_animals = data_ts_pre['n_animals']
total_tp = data_ts_pre['total_tp']
regions = data_ts_pre['regions']
anat_labels = data_ts_pre['anat_labels']
# Load cognitive data
cog_data = pd.read_csv(paths['preprocessed'] / Path("cog_data_filtered.csv"))


print(f"Loaded {len(ts)} time series")
print(f"Loaded cognitive data for {len(cog_data)} animals")

assert len(ts) == len(cog_data), "Mismatch between time series and cognitive data entries."
#%%
processors = -1

lag=1
tau=5
window_size = 12
window_parameter = (5,100,1)

HASH_TAG = f"lag={lag}_tau={tau}_wmax={window_parameter[1]}_wmin={window_parameter[0]}"

# time_window_min, time_window_max, time_window_step = window_parameter
time_window_range = np.arange(window_parameter[0],
                              window_parameter[1]+1,
                              window_parameter[2])

#%%
from shared_code.fun_loaddata import make_file_path, load_from_cache
# Plot FCD of the first animal

prefix= 'dfc'  # 'dfc' for dynamic functional connectivity, 'mc' for meta-connectivity
file_path = make_file_path(paths['dfc'], prefix, window_size, lag, n_animals, regions)

    #try loading from cache
key = prefix 
label = prefix 
if file_path is not None and file_path.exists():
    dfc_stream = load_from_cache(file_path, key=key, label=label)

# %%
from shared_code.fun_utils import dfc_stream2fcd

fcd = [dfc_stream2fcd(dfc_stream[n]) for n in range(n_animals)]

#%%
# Plot the FCD of the all animals
plt.figure(figsize=(20, 20))
for i in range(n_animals):
    plt.subplot(int(np.sqrt(n_animals)), int(np.ceil(n_animals / np.sqrt(n_animals))), i + 1)
    plt.imshow(fcd[i], cmap='viridis', aspect='auto')
    # plt.colorbar(label=r'FCD = CC($FC_{i,...,w}, FC_{w,...,2w}$)')
    # plt.title(f'FCD for Animal {i} (Window Size: {window_size})')
    plt.xlabel('Time Points')
    plt.ylabel('Time Points')
    plt.clim(-0.02, 0.29)  # Set color limits for better visibility
# plt.tight_layout()
# %%

# ==================== Groups ========================
wt_index          = cog_data['WT']
mut_index          = cog_data['Dp1Yey']

veh_index        = cog_data['VEH']
treat_index        = cog_data['LCTB92']

wt_veh_index = cog_data['WT'] & cog_data['VEH']
wt_treat_index = cog_data['WT'] & cog_data['LCTB92']
mut_veh_index = cog_data['Dp1Yey'] & cog_data['VEH']
mut_treat_index = cog_data['Dp1Yey'] & cog_data['LCTB92']

labels = (
    'WT-VEH',
    'WT-TREAT',
    'Dp1Yey-VEH',
    'Dp1Yey-TREAT'
)

group_gen_treat = (wt_veh_index, wt_treat_index, mut_veh_index, mut_treat_index)

#%%
#Plot one FCD per group
plt.figure(figsize=(9, 7))
for i, group in enumerate(group_gen_treat):
    plt.subplot(2, 2, i + 1)
    print(np.sum(group==True), "animals in group", labels[i])
    # Select the FCDs for the current group
    fcd_group = [fcd[idx] for idx, val in enumerate(group.values) if val]
    # fcd_group = fcd[group]
    if fcd_group:
        # fcd_mean = np.nanmean(fcd_group, axis=0)
        # a randon int number of len(fcd_group) to choose one fcd
        rn_aux = int(np.random.randint(0, len(fcd_group), size=1))
        # fcd_mean = fcd_group, axis=0)
        plt.imshow(fcd_group[rn_aux], cmap='viridis', aspect='auto')
        plt.title(labels[i])
        plt.xlabel('Time Points')
        plt.ylabel('Time Points')
        plt.clim(0.05, 0.6)  # Set color limits for better visibility
        plt.colorbar(label=r'FCD = CC($FC_{i,...,w}, FC_{w,...,2w}$)')
    else:
        plt.title(f'No data for {labels[i]}')
plt.tight_layout()
# %%

#Plot the speed through time windows


