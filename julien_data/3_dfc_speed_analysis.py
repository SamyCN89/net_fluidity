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

import pandas as pd
# from functions_analysis import *
from scipy.io import loadmat, savemat
from scipy.special import erfc
from scipy.stats import pearsonr, spearmanr

from shared_code.fun_loaddata import *  # Import only needed functions
from shared_code.fun_dfcspeed import parallel_dfc_speed_oversampled_series
from shared_code.fun_utils import set_figure_params, get_paths
from tqdm import tqdm



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
paths['speed'] = paths['results'] / 'speed'
# paths['speed'].mkdir(parents=True, exist_ok=True)

TS_FILE = paths['sorted'] / Path("ts_filtered_unstacked.npz")
COG_FILE = paths['sorted'] / Path("cog_data_filtered.csv")

SAVE_DATA = True

WINDOW_PARAM = (5,100,1)
LAG=1
TAU=5

HASH_TAG = f"lag={LAG}_tau={TAU}_wmax={WINDOW_PARAM[1]}_wmin={WINDOW_PARAM[0]}"

#%%
# ------------------------ Load Data ------------------------

ts_data = np.load(TS_FILE, allow_pickle=True)['ts']
cog_data = pd.read_csv(COG_FILE)


print(f"Loaded {len(ts_data)} time series")
print(f"Loaded cognitive data for {len(cog_data)} animals")

assert len(ts_data) == len(cog_data), "Mismatch between time series and cognitive data entries."
#%%



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

for ts in tqdm(ts_data, desc="Computing dFC speed"):
    median_speeds, speed_distribution = parallel_dfc_speed_oversampled_series(
        ts, WINDOW_PARAM, lag=LAG, tau=TAU, min_tau_zero=True, get_speed_dist=True
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
