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
from fun_loaddata import make_file_path, load_from_cache
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

fcd = dfc_stream2fcd(dfc_stream[0])
# %%
