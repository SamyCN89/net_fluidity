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

from metaconnectivity.fun_metaconnectivity import compute_metaconnectivity
from shared_code.fun_loaddata import *  # Import only needed functions
from shared_code.fun_dfcspeed import parallel_dfc_speed_oversampled_series
from shared_code.fun_utils import set_figure_params, get_paths, load_npz_dict
from tqdm import tqdm

from shared_code.fun_dfcspeed import get_tenet4window_range



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

#%%

# Modify the handlrer_get_tenet function to include the new parameters
import logging

def _handle_dfc_speed_analysis(ts_data, window_size, lag, save_path, n_animals, nodes, **kwargs):
    """
    Handle DFC speed analysis within the unified handler.
    
    Parameters:
        ts_data (np.ndarray): 3D array (n_animals, n_regions, n_timepoints).
        window_size (int): Sliding window size.
        lag (int): Step size for the sliding window.
        save_path (str): Directory to save results.
        n_animals (int): Number of animals.
        nodes (int): Number of regions.
        **kwargs: DFC speed specific parameters.
        
    Returns:
        np.ndarray: DFC speed results.
    """
    logger = logging.getLogger(__name__)
    
    # Extract DFC speed specific parameters
    tau = kwargs.get('tau', 3)
    min_tau_zero = kwargs.get('min_tau_zero', False)
    method = kwargs.get('method', 'pearson')
    
    # Create custom file path for DFC speed (uses different naming convention)
    if save_path:
        file_path = make_file_path(save_path, prefix, window_size, lag, n_animals, nodes)

        # Try to load from cache
        if file_path.exists():
            try:
                return load_from_cache(file_path, key=prefix, label=prefix)
                logger.info(f"Loading DFC speed from cache: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to load cached DFC speed (reason: {e}). Recomputing...")
    else:
        file_path = None
    
    logger.info(f"Computing DFC speed (window_size={window_size}, lag={lag}, tau={tau}, method={method})")
    
    # First, load or warning about DFC streams
    dfc_file_path = make_file_path(save_path, 'dfc', window_size, lag, n_animals, nodes)
    
    dfc_stream = None
    if dfc_file_path and dfc_file_path.exists():
        try:
            logger.info(f"Loading DFC stream from cache: {dfc_file_path}")
            dfc_stream = load_from_cache(dfc_file_path, key='dfc_stream', label='dfc-stream')
        except Exception as e:
            logger.warning(f"Failed to load cached DFC stream (reason: {e}). Computing...")



#%%
    # Compute speed for each animal
    speed_results = []
    for i in range(n_animals):
        if dfc_stream[i] is None or dfc_stream[i].shape[1] < 2:
            logger.warning(f"Skipping animal {i}: insufficient data")
            speed_results.append(np.nan)
            continue
            
        # Compute speed using the unified dfc_speed function
        vstep = max(1, window_size // lag)  # Ensure vstep is at least 1
        try:
            median_speed, _ = dfc_speed(
                dfc_stream[i],
                vstep=vstep,
                method=method,
                return_fc2=False
            )
            speed_results.append(median_speed)
        except Exception as e:
            logger.warning(f"Failed to compute speed for animal {i}: {e}")
            speed_results.append(np.nan)
    
    speed_medians = np.array(speed_results)
    
    # Save results if path provided
    if file_path:
        try:
            np.savez_compressed(
                file_path,
                speed_medians=speed_medians,
                window_size=window_size,
                lag=lag,
                tau=tau,
                min_tau_zero=min_tau_zero,
                method=method
            )
            logger.info(f"Saved DFC speed results to: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save DFC speed results: {e}")
    
    return speed_medians