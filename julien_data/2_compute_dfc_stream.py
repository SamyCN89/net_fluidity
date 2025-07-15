#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 14:42:38 2023

@author: samy
"""


#%%
from os import path
from pathlib import Path
import comm
import numpy as np
import pandas as pd
from shared_code.fun_loaddata import *
from shared_code.fun_dfcspeed import get_tenet4window_range
from shared_code.fun_paths import get_paths
from tqdm import tqdm
import pickle
#%% Define paths

from class_dataanalysis_julien import DFCAnalysis
data = DFCAnalysis()

#%% Load data
#Preprocessed data
processors = -1
meta_file = 'metadata_animals_50_regions_37_tr_500.pkl'

print("Loading data...")
data.get_metadata(meta_filename=meta_file)
data.get_ts_preprocessed()
data.get_cogdata_preprocessed()
data.get_temporal_parameters()

# Get paths from the DFCAnalysis class
paths = data.paths  # Use the paths from the DFCAnalysis class

# Print metadata for debugging
print(f"Loaded metadata: {data.metadata.keys()}")
n_animals = data.metadata.get('n_animals', None)
regions = data.metadata.get('regions', None)
total_tr = data.metadata.get('total_tr', None)

# Load cognitive data
cog_data = data.cog_data_filtered
filtered_cog_data = cog_data[cog_data["n_timepoints"] >= data.total_tr]
remaining_cog_data = cog_data[cog_data["n_timepoints"] < data.total_tr]

#%% Convert data to 3D format
print("\nConverting data to 3D format...")

# Separate time series by length
ts_500 = [ts for ts in data.ts if ts.shape[0] == data.total_tr]  # 48 animals
ts_400 = [ts for ts in data.ts if ts.shape[0] == 400]  # 2 animals

print(f"Found {len(ts_500)} animals with data.total_tr timepoints")
print(f"Found {len(ts_400)} animals with 400 timepoints")

# Convert to 3D arrays - no padding needed within each group
ts_500_3d = np.array(ts_500)  # Shape: (48, 500, 37)
ts_400_3d = np.array(ts_400)  # Shape: (2, 400, 37)

# If you need all animals together, pad the shorter sequences
max_timepoints = data.total_tr
all_animals_3d = np.zeros((50, max_timepoints, 37), dtype=np.float32)

# Fill in the data
idx = 0
for ts in data.ts:
    n_tp = ts.shape[0]
    all_animals_3d[idx, :n_tp, :] = ts
    idx += 1

print(f"\nData shapes after conversion:")
print(f"All animals (padded): {all_animals_3d.shape}")
print(f"500-timepoint animals: {ts_500_3d.shape}")
print(f"400-timepoint animals: {ts_400_3d.shape}")

#%% Parameters for dFC analysis

print(f"\ndFC parameters: lag={data.lag}, tau={data.tau}, window_range={data.time_window_range[0]}-{data.time_window_range[-1]}")

#%% Compute dFC

load_cache_bool = False  # Set to True if you want to load cached results

# Option 1: Process all animals together (with padding)
print(f"\nProcessing all {all_animals_3d.shape[0]} animals together...")
get_tenet4window_range(all_animals_3d, data.time_window_range, prefix='dfc', 
                      paths=paths, lag=data.lag, n_animals=all_animals_3d.shape[0], 
                      regions=regions, processors=processors, load_cache=load_cache_bool)

# Option 2: Process groups separately (uncomment if you prefer this approach)
print(f"\nProcessing {ts_500_3d.shape[0]} animals with 500 timepoints...")
get_tenet4window_range(ts_500_3d, data.time_window_range, prefix='dfc', 
                      paths=paths, lag=data.lag, n_animals=ts_500_3d.shape[0], 
                      regions=regions, processors=processors, load_cache=load_cache_bool)
# 
print(f"\nProcessing {ts_400_3d.shape[0]} animals with 400 timepoints...")
get_tenet4window_range(ts_400_3d, data.time_window_range, prefix='dfc', 
                      paths=paths, lag=data.lag, n_animals=ts_400_3d.shape[0], 
                      regions=regions, processors=processors, load_cache=load_cache_bool)

print("\nDFC computation completed!")

# %%
load_cache_bool = False  # Set to True if you want to load cached results
processors = -1  # Use all available processors
with open(Path(data.paths['allegiance']) / 'communities_wt_veh.pkl', 'rb') as f:
    communities = pickle.load(f)

for i, c in enumerate(np.unique(communities)):
    print(f"Animal {i}: Community {c}")
    ts_500_3d_mod1 = ts_500_3d[:,:,communities==c]
    regions_mod1 = np.sum(communities==c)


    # Option 2: Process groups separately (uncomment if you prefer this approach)
    print(f"\nProcessing {ts_500_3d_mod1.shape[0]} animals with 500 timepoints...")
    get_tenet4window_range(ts_500_3d_mod1, data.time_window_range, prefix='dfc', 
                        paths=paths, lag=data.lag, n_animals=ts_500_3d_mod1.shape[0], 
                        regions=regions_mod1, processors=processors, load_cache=load_cache_bool)
# %%
