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
from shared_code.fun_utils import set_figure_params, get_paths, load_npz_dict
from tqdm import tqdm

from shared_code.shared_code.fun_dfcspeed import get_tenet4window_range



#%% Define paths, folders and hash
# ------------------------ Configuration ------------------------

paths = get_paths(dataset_name='julien_caillette', 
                  timecourse_folder='time_courses',
                  cognitive_data_file='mice_groups_comp_index.xlsx')

#%%
# USE_EXTERNAL_DISK = True
# ROOT = Path('/media/samy/Elements1/Proyectos/LauraHarsan/dataset/julien_caillette/') if USE_EXTERNAL_DISK \
#         else Path('/home/samy/Bureau/Proyect/LauraHarsan/dataset/julien_caillette/')
# RESULTS_DIR = ROOT / Path('results')
# paths['speed'] = paths['results'] / 'speed'
# paths['speed'].mkdir(parents=True, exist_ok=True)

# TS_FILE = paths['sorted'] / Path("ts_filtered_unstacked.npz")
# COG_FILE = paths['sorted'] / Path("cog_data_filtered.csv")

SAVE_DATA = True


#%%
# ------------------------ Load Data ------------------------

data_ts_pre = load_npz_dict(paths['sorted'] / Path('ts_filtered_unstacked.npz'))
ts = data_ts_pre['ts']
n_animals = data_ts_pre['n_animals']
total_tp = data_ts_pre['total_tp']
regions = data_ts_pre['regions']
anat_labels = data_ts_pre['anat_labels']
# Load cognitive data
cog_data = pd.read_csv(paths['sorted'] / Path("cog_data_filtered.csv"))


print(f"Loaded {len(ts)} time series")
print(f"Loaded cognitive data for {len(cog_data)} animals")

assert len(ts) == len(cog_data), "Mismatch between time series and cognitive data entries."
#%%
processors = -1

lag=1
tau=5
window_size = 9
window_parameter = (5,100,1)

HASH_TAG = f"lag={lag}_tau={tau}_wmax={window_parameter[1]}_wmin={window_parameter[0]}"

# time_window_min, time_window_max, time_window_step = window_parameter
time_window_range = np.arange(window_parameter[0],
                              window_parameter[1]+1,
                              window_parameter[2])


#%%
from shared_code.fun_dfcspeed import dfc_speed
from shared_code.fun_loaddata import make_file_path, load_from_cache


prefix= 'dfc'  # or 'meta'
for window_size in time_window_range:
    # Define file path for caching
    file_path = make_file_path(paths['dfc'], prefix, window_size, lag, n_animals, regions)

    #try loading from cache
    key = 'dfc_stream' if prefix == 'dfc' else prefix
    label = "dfc-stream" if prefix == "dfc" else "meta-connectivity"
    if file_path is not None and file_path.exists():
        dfc_stream = load_from_cache(file_path, key=key, label=label)

    # Precompute the arrays for median, speed, fc
    median = []
    speed = []
    fc = []

    for n in np.arange(n_animals):
        if dfc_stream[n] is None:
            print(f"Warning: dfc_stream[{n}] is None, skipping...")
            continue
        if dfc_stream[n].shape[0] < 2:
            print(f"Warning: dfc_stream[{n}] has less than 2 timepoints, skipping...")
            continue
        aux_speed = dfc_speed(dfc_stream[n], vstep=1, method='pearson', return_fc2=True)
        median.append(aux_speed[0])
        speed.append(aux_speed[1])
        fc.append(aux_speed[2])


# call the function get_tenet4window_range for speed
prefix = 'speed'  # or 'meta'


get_tenet4window_range(window_parameter,
                       lag=lag, 
                       tau=tau, 
                       n_animals=n_animals, 
                       regions=regions, 
                       processors=processors,
                       prefix=prefix,
                       paths=paths['dfc'],
                       HASH_TAG=HASH_TAG,
                       save_data=SAVE_DATA)

#%% # Compute speed dFC
# =============================================================================
# Speed analysis
# Compute the dfc speed distribution using wondow oversampling method for each animal. Also retrieve median speed for each tau, in multiple W, for each animal
# =============================================================================

#%% 
# ------------------------ Compute Speed ------------------------

vel_list = []
speed_medians = []

print("Starting dFC speed computation...")
start_time = time.time()

for ts_aux in tqdm(ts, desc="Computing dFC speed"):
    median_speeds, speed_distribution = parallel_dfc_speed_oversampled_series(
        ts_aux, window_parameter, lag=lag, tau=tau, min_tau_zero=True, get_speed_dist=True
    )
    vel_list.append(speed_distribution)
    speed_medians.append(median_speeds)

elapsed_time = time.time() - start_time
print(f"Speed computation completed in {elapsed_time:.2f} seconds")

#%%
# ------------------------ Save Results ------------------------

if SAVE_DATA:
    vel_array = np.array(vel_list, dtype=object)
    medians_array = np.array(speed_medians)

    np.savez(
        paths['speed'] / f"speed_dfc_{HASH_TAG}.npz",
        vel=vel_array,
        speed_median=medians_array,
    )
    print(f"Saved speed data to: {paths['speed']}")



# %%

parallel_dfc_speed_oversampled_series(ts, 
                            window_parameter, 
                            lag=1, 
                            tau=5, 
                            min_tau_zero=False, 
                            get_speed_dist=True, 
                            method='pearson', 
                            n_jobs=-1,
                            path=paths['dfc'], 
                            prefix='dfc')
# %%
