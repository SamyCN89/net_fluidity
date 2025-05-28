#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 14:42:38 2023

@author: samy
"""

#%%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as pltµ
import brainconn as bct
import os
import time
from joblib import Parallel, delayed, parallel_backend
import pandas as pd
# from functions_analysis import *
from scipy.io import loadmat, savemat
from scipy.special import erfc
from scipy.stats import pearsonr, spearmanr
from sklearn.manifold import TSNE

from shared_code.fun_loaddata import *  # Import only needed functions
from shared_code.fun_dfcspeed import *
from shared_code.fun_utils import set_figure_params, get_paths, load_npz_dict
from tqdm import tqdm



#%% Define paths, folders and hash
# ------------------------ Configuration ------------------------
timeseries_folder = 'time_courses'
paths = get_paths(dataset_name='julien_caillette', 
                  timecourse_folder=timeseries_folder,
                  cognitive_data_file='mice_groups_comp_index.xlsx')

#%%
# Load time series data
# data_ts, n_animals, total_tp, regions, anat_labels = load_npz_dict(paths['sorted'] / Path('ts_filtered_unstacked.npz'))
data = load_npz_dict(paths['sorted'] / Path('ts_filtered_unstacked.npz'))
data_ts = data['ts']
n_animals = data['n_animals']
total_tp = data['total_tp']
regions = data['regions']
anat_labels = data['anat_labels']

cog_data = pd.read_csv(paths['sorted'] / Path('cog_data_filtered.csv'))
filtered_cog_data = cog_data[cog_data["n_timepoints"] >= 500]
remaining_cog_data = cog_data[cog_data["n_timepoints"] < 500]
# Assume cog_data is your DataFrame

#%%

# Print loaded data information
print(f"Loaded time series data with shape: {data_ts.shape}")

ts_data_filt = np.array([w for w in data_ts if w.shape[0] != 400])
ts_remaining = np.array([w for w in data_ts if w.shape[0] == 400])

#%%
# Parameters for dfc analysis
processors = -1

lag=1
tau=5
window_size = 9
window_parameter = (5,100,1)

# time_window_min, time_window_max, time_window_step = window_parameter
time_window_range = np.arange(window_parameter[0],
                              window_parameter[1]+1,
                              window_parameter[2])

#%% # Compute speed dFC
# =============================================================================
# Speed analysis
# Compute the dfc speed distribution using wondow oversampling method for each animal. Also retrieve median speed for each tau, in multiple W, for each animal
# =============================================================================

#%% 
# ------------------------ Compute DFC ------------------------
import logging
import joblib 
logging.basicConfig(level=logging.INFO)


    
def handler_get_tenet(ts_data, prefix, window_size, lag, format_data='3D', save_path=None):
    """
    Generate temporal networks (dfc_stream, meta-connectivity) for time-series data.

    Parameters:
        ts_data (np.ndarray): 3D array (n_animals, n_regions, n_timepoints).
        window_size (int): Sliding window size.
        lag (int): Step size for the sliding window.
        format_data (str): '2D' for vectorized, '3D' for matrices.
        save_path (str): Directory to save results.

    Returns:
        np.ndarray: 4D array of DFC streams (n_animals, time_windows, roi, roi)
    """

    logger = logging.getLogger(__name__)
    n_animals, _, nodes = ts_data.shape

    # Define the full save path based on parameters and save_path folder
    file_path = make_file_path(save_path, prefix, window_size, lag, n_animals, nodes)
    logger.info(f'file path: {file_path}')

    #try loading from cache
    key = 'dfc_stream' if prefix == 'dfc' else prefix
    label = "dfc-stream" if prefix == "dfc" else "meta-connectivity"
    if file_path is not None and file_path.exists():
        logger.info(f"Loading from cache: {file_path}")
        try:
            return load_from_cache(file_path, key=key, label=label)
        except Exception as e:
            logger.error(f"Failed to load {label} (reason: {e}). Recomputing...")

    # Compute in parallel
    logger.info(f"Computing {prefix} (window_size={window_size}, lag={lag})...")
    results = np.array([ts2dfc_stream(
        ts_data[i], window_size, lag, format_data) 
        for i in tqdm(range(n_animals), desc=f'Computing {label}')])

    results = results.astype(np.float32)  # Convert to float32 for memory efficiency
    #Save results
    try:
        save2disk(file_path, prefix, key=results)
        logger.info(f'Saved results to {file_path}')
    except Exception as e:
        logger.error(f'Failed to save results: {e}')
    return results

def compute_for_1_window(ws, ts, prefix, lag, save_path):
    """
    Compute the analysis for a single window size.
    """ 
    logger = logging.getLogger(__name__)
    try:
        logger.info(f"Starting {prefix} computation for window_size={ws}")
        start = time.time()
        handler_get_tenet(
            ts,
            prefix=prefix,
            window_size=ws,
            lag=lag,
            save_path=save_path,
        )
        logger.info(f"Finished window_size={ws} in {time.time()-start:.2f} seconds")
    except Exception as e:
        logger.error(f"Error during {prefix} computation for window_size={ws}: {e}")
        raise

def get_tenet4window_range(ts, time_window_range, prefix, paths, lag, n_animals, regions, processors=-1):
    """
    Get the range of window sizes for tenet files. 'DC AND 'MC' are the two prefixes implemented.
    Args:
        ts (roi, timepoints): Time series data.
        time_window_range (list): List of time window sizes.
        prefix (str): Prefix for the tenet files. 'dfc' for dynamic functional connectivity.
                   'mc' for meta-connectivity analysis.        
        lag (int): Lag value for the analysis.
        n_animals (int): Number of animals in the dataset.
        regions (list): List of regions in the dataset.
        processors (int): joblib. Number of processors to use for parallel computation.
    Returns:
        None
    """
    try:
        save_path = paths.get(prefix)
        if not save_path:
            raise ValueError(f"Invalid prefix '{prefix}'. Save path not found in paths dictionary.")
        # Run parallel dfc stream over window sizes

        #set the processors
        processors = min(processors, joblib.cpu_count())
        logging.info(f'Starting analysis for {prefix}, n_jobs={processors}')

        start = time.time()
        Parallel(n_jobs=min(processors, len(time_window_range)))(
            delayed(compute_for_1_window)(ws, ts, prefix, lag, save_path) 
            for ws in tqdm(time_window_range, desc=f'Window sizes')
        )
        logging.info(f'{prefix} computation time {time.time()-start:.2f} seconds')

        # Handle missing files and rerun if necessary
        missing_files = check_and_rerun_missing_files(
            save_path, prefix, time_window_range, lag, n_animals, regions
        )
        if missing_files:
            logging.warning(f"Missing files detected for {prefix}: {missing_files}")
            time_window_range = np.array(missing_files)
            # Rerun for missing files
            Parallel(n_jobs=min(processors, len(time_window_range)))(
                delayed(compute_for_1_window)(ws, ts, prefix, lag, save_path) for ws in time_window_range
            )
    except Exception as e:
        logger.error(f"Error occurred during {prefix} computation: {e}")
        raise
get_tenet4window_range(ts_data_filt, time_window_range, prefix='dfc', paths=paths, lag=lag, n_animals=len(ts_data_filt), regions=regions, processors=processors)
get_tenet4window_range(ts_remaining, time_window_range, prefix='dfc', paths=paths, lag=lag, n_animals=len(ts_remaining), regions=regions, processors=processors)
# %%
