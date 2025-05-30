#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 14:42:38 2023

@author: samy
"""


#%%
from pathlib import Path
import numpy as np
import pandas as pd
from shared_code.fun_loaddata import *
from shared_code.fun_dfcspeed import get_tenet4window_range
from shared_code.fun_utils import set_figure_params, get_paths, load_npz_dict
from tqdm import tqdm

#%% Define paths
timeseries_folder = 'time_courses'
paths = get_paths(dataset_name='julien_caillette', 
                  timecourse_folder=timeseries_folder,
                  cognitive_data_file='mice_groups_comp_index.xlsx')

#%% Load data
print("Loading data...")
data = load_npz_dict(paths['sorted'] / Path('ts_filtered_unstacked.npz'))
data_ts = data['ts']  # This is a 1D object array with 50 elements
n_animals = data['n_animals']
total_tp = data['total_tp']
regions = data['regions']
anat_labels = data['anat_labels']

# Load cognitive data
cog_data = pd.read_csv(paths['sorted'] / Path('cog_data_filtered.csv'))
filtered_cog_data = cog_data[cog_data["n_timepoints"] >= 500]
remaining_cog_data = cog_data[cog_data["n_timepoints"] < 500]

#%% Convert data to 3D format
print("\nConverting data to 3D format...")

# Separate time series by length
ts_500 = [ts for ts in data_ts if ts.shape[0] == 500]  # 48 animals
ts_400 = [ts for ts in data_ts if ts.shape[0] == 400]  # 2 animals

print(f"Found {len(ts_500)} animals with 500 timepoints")
print(f"Found {len(ts_400)} animals with 400 timepoints")

# Convert to 3D arrays - no padding needed within each group
ts_500_3d = np.array(ts_500)  # Shape: (48, 500, 37)
ts_400_3d = np.array(ts_400)  # Shape: (2, 400, 37)

# If you need all animals together, pad the shorter sequences
max_timepoints = 500
all_animals_3d = np.zeros((50, max_timepoints, 37), dtype=np.float32)

# Fill in the data
idx = 0
for ts in data_ts:
    n_tp = ts.shape[0]
    all_animals_3d[idx, :n_tp, :] = ts
    idx += 1

print(f"\nData shapes after conversion:")
print(f"All animals (padded): {all_animals_3d.shape}")
print(f"500-timepoint animals: {ts_500_3d.shape}")
print(f"400-timepoint animals: {ts_400_3d.shape}")

#%% Parameters for dFC analysis
processors = -1
lag = 1
tau = 5
window_size = 9
window_parameter = (5, 100, 1)
time_window_range = np.arange(window_parameter[0], window_parameter[1] + 1, window_parameter[2])

print(f"\ndFC parameters: lag={lag}, tau={tau}, window_range={time_window_range[0]}-{time_window_range[-1]}")

#%% Compute dFC
# Option 1: Process all animals together (with padding)
print(f"\nProcessing all {all_animals_3d.shape[0]} animals together...")
get_tenet4window_range(all_animals_3d, time_window_range, prefix='dfc', 
                      paths=paths, lag=lag, n_animals=all_animals_3d.shape[0], 
                      regions=regions, processors=processors)

# Option 2: Process groups separately (uncomment if you prefer this approach)
# print(f"\nProcessing {ts_500_3d.shape[0]} animals with 500 timepoints...")
# get_tenet4window_range(ts_500_3d, time_window_range, prefix='dfc', 
#                       paths=paths, lag=lag, n_animals=ts_500_3d.shape[0], 
#                       regions=regions, processors=processors)
# 
# print(f"\nProcessing {ts_400_3d.shape[0]} animals with 400 timepoints...")
# get_tenet4window_range(ts_400_3d, time_window_range, prefix='dfc', 
#                       paths=paths, lag=lag, n_animals=ts_400_3d.shape[0], 
#                       regions=regions, processors=processors)

print("\nDFC computation completed!")
# %%
