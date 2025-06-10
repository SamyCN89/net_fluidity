#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 14:42:38 2023

@author: samy
"""

#%%
from pathlib import Path
import numpy as np
import pickle
import logging
from tqdm import tqdm
import gc

from joblib import Parallel, delayed
from typing import Dict, List, Tuple, Optional, Union

from class_dataanalysis_julien import DFCAnalysis
from shared_code.fun_loaddata import save_pickle
from shared_code.fun_utils import set_figure_params
# logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

#%%

def dfc_speed_split(dfc_stream, 
            vstep=1, 
            tau_range=0,
            method='pearson', 
            return_fc2=False,
            tril_indices=None,
            time_offset=0,
            ):
    """
    Unified function to calculate the speed of variation in dynamic functional connectivity (dFC).
    
    ----------
    dfc_stream : numpy.ndarray
        Dynamic functional connectivity stream. Can be either: 2D array (n_pairs, n_frames) 3D array (n_rois, n_rois, n_frames): Full FC matrices over time
    vstep : int, optional
        Time step for computing FC speed (default=1). Must be positive and < n_frames.
    method : str, optional
        Correlation method to use for speed computation (default='pearson').
        Supported methods:
        - 'pearson': Pearson correlation coefficient
        - 'spearman': Spearman rank correlation 
        - 'cosine': Cosine similarity
    tril_indices : tuple, optional
        Pre-computed triangular indices for 3D input (default=None).
        If None, will be computed automatically for 3D input.
    return_fc2 : bool, optional
        If True, also return the second FC matrix for each time step (default=False).
        
    Returns
    -------
    speed_median : float
        Median of the computed speed distribution.
    speeds : numpy.ndarray
        Time series of computed speeds with shape (n_frames - vstep,).
    fc2_stream : numpy.ndarray, optional
        Second FC matrix for each time step. Only returned if return_fc2=True.
        Shape: (n_pairs, n_frames - vstep) for vectorized output.

    References
    ----------
    Dynamic Functional Connectivity as a complex random walk: Definitions and the dFCwalk toolbox
    Lucas Arbabyazd, Diego Lombardo, Olivier Blin, Mira Didic, Demian Battaglia, Viktor Jirsa
    MethodsX 2020, doi: 10.1016/j.mex.2020.101168
    """
    from shared_code.fun_optimization import (
        pearson_speed_vectorized,
        spearman_speed,
        cosine_speed_vectorized
    )
    
    # Input validation
    if not isinstance(dfc_stream, np.ndarray):
        raise TypeError("dfc_stream must be a numpy array")
    
    if dfc_stream.ndim not in [2, 3]:
        raise ValueError("dfc_stream must be 2D (n_pairs, frames) or 3D (roi, roi, frames)")
    
    if not isinstance(vstep, int) or vstep <= 0:
        raise TypeError("vstep must be a positive integer")
        
    if method not in ['pearson', 'spearman', 'cosine']:
        raise ValueError(f"Unsupported method '{method}'. Use 'pearson', 'spearman', or 'cosine'")
    
    # Handle input format conversion
    # 3D input: (n_rois, n_rois, n_frames)
    if dfc_stream.ndim == 3:
        n_rois = dfc_stream.shape[0]
        n_frames = dfc_stream.shape[2]
        
        # Generate triangular indices if not provided
        if tril_indices is None:
            tril_indices = np.tril_indices(n_rois, k=-1)
        
        # Extract lower triangular values efficiently
        fc_stream = dfc_stream[tril_indices[0], tril_indices[1], :]
    else:
        # 2D input: (n_pairs, n_frames)
        fc_stream = dfc_stream
        n_frames = fc_stream.shape[1]
    
    # Validate frame count vs vstep
    if vstep >= n_frames:
        raise ValueError(f"vstep ({vstep}) must be less than number of frames ({n_frames})")

    fc1_indices = []
    fc2_indices = []    

    indices_max = n_frames - (vstep + np.max(tau) + time_offset) 
    indices = np.arange(0, indices_max, vstep)
    if np.size(tau_range) > 1:
        for tau_aux in tau_range:
            fc1_indices.append(indices[:-1])  # Indices for the first FC matrix
            fc2_indices.append(indices[1:]+tau_aux+time_offset)   # Indices for the second FC matrix
    else:
        tau_aux = tau_range
        fc1_indices.append(indices[:-1])
        fc2_indices.append(indices[1:]+tau_aux+time_offset)   # Indices for the second FC matrix

    n_speeds = (len(indices)-1) * np.size(tau_range)
    n_pairs = fc_stream.shape[0]
    
    # Pre-allocate output arrays for efficiency
    speeds = np.empty((n_speeds, np.size(tau_range)), dtype=np.float32)
    fc2_stream = None
    
    # Extract FC matrices for vectorized computation
    fc1_matrices = fc_stream[:, np.array(fc1_indices).T.flatten()]  # Shape: (n_pairs, n_speeds)
    fc2_matrices = fc_stream[:, np.array(fc2_indices).T.flatten()]  # Shape: (n_pairs, n_speeds)
    if return_fc2:
        fc2_stream_indices = np.empty(n_speeds, dtype=int)  # Pre-allocate for second FC matrix indices
        # fc2_stream[:, :] = fc2_matrices
        fc2_stream_indices[:] = (np.array(fc2_indices).T.flatten()).astype(int)
        return fc2_stream_indices

    # Use optimized speed computation functions for maximum performance
    if method == 'pearson':
        speeds = pearson_speed_vectorized(fc1_matrices, fc2_matrices)
    elif method == 'spearman':
        speeds = spearman_speed(fc1_matrices, fc2_matrices)
    elif method == 'cosine':
        speeds = cosine_speed_vectorized(fc1_matrices, fc2_matrices)
    else:
        raise ValueError(f"Unsupported method '{method}'. Use 'pearson', 'spearman', or 'cosine'")

    # Ensure speeds are within valid range [-1, 2] for numerical stability
    speeds = np.clip(speeds, -1.0, 2.0)
    speeds_mat = speeds.reshape(len(tau_range), -1)  # Reshape to (n_pairs, n_speeds)

    return speeds_mat

#%%
def run_dfc_speed_analysis(data, time_window_range, tau_range, lag, save_path, n_animals, nodes, load_cache=False, processors=1, **kwargs):
    """
    DFC speed analysis handler: saves results per animal per tau.
    """
    
    # Parameter extraction & checks (same as before)
    min_tau_zero = kwargs.get('min_tau_zero', True)
    method = kwargs.get('method', 'pearson')
    return_fc2 = kwargs.get('return_fc2', False)
    prefix = kwargs.get('prefix', 'speed')  # Prefix for the DFC speed results


    # for ws_idx, window_size in tqdm(enumerate(time_window_range), desc=f"Processing animals for window_size "):
    def process_window(window_size):
        # Loads one window size of DFC data
        dfc_stream= data.load_dfc_1_window(lag=lag, window=window_size)
        # dfc_stream = data.dfc_stream
        logging.getLogger(__name__).info(f"Loaded DFC stream shape: {dfc_stream.shape}")

        #Generate the file name for the current window size data
        window_file = save_path / (
            f"{prefix}_{'fc_' if return_fc2 else ''}win{window_size}_tau{np.size(tau_range)}_animals_{n_animals}.npz"
        )

        # Initialize lists to store results for each animal
        results = []
        for animal_idx in range(n_animals):
            logging.getLogger(__name__).info(f"Processing animal {animal_idx + 1}/{n_animals} for window size {window_size}")
            if load_cache and window_file.exists():
                logging.getLogger(__name__).info(f"Loading cached results for window {window_file}")
                return
            try:
                logging.getLogger(__name__).info(f"Computing for window {window_file} and animal {animal_idx + 1}/{n_animals}")
                if return_fc2:
                    fc2 = dfc_speed_split(
                        dfc_stream[animal_idx], vstep=int(window_size), tau_range=0, method=method, return_fc2=return_fc2
                    )
                    results.append(fc2)
                    logging.getLogger(__name__).debug(f"Animal {animal_idx} window {window_size}: computed FC2")
                else:
                    speeds = dfc_speed_split(
                        dfc_stream[animal_idx], vstep=1, tau_range=tau_range, method=method, return_fc2=return_fc2, time_offset=window_size
                    )
                    results.append(speeds)
                    logging.getLogger(__name__).debug(f"Animal {animal_idx} window {window_size}: computed speeds")
            except Exception as e:
                logging.getLogger(__name__).error(f"Error for window {window_file}: {e}")

        # Save the fc2 results if return_fc2 else the speed results
        if return_fc2:
            np.savez_compressed(
                window_file,
                fc2=np.array(results, dtype=object),  # Use object dtype for variable-length arrays
                window_size=window_size
            )
            logging.getLogger(__name__).info(f"✓ Saved fc2 for window {window_file}")
        else:
            np.savez_compressed(
                window_file,
                speeds=np.array(results, dtype=object),  # Use object dtype for variable-length arrays
                window_size=window_size
            )
            logging.getLogger(__name__).info(f"✓ Saved speeds for window {window_file}")

    # Use Parallel to handle multiple windows in parallel
    Parallel(n_jobs=processors, verbose=1)(
        delayed(process_window)(ws)
    for ws in tqdm(time_window_range, desc=f"Processing windows for ...")
    )

    logging.getLogger(__name__).info("All windows processed successfully.")

#%%
processors = -1  # Use all available processors

data = DFCAnalysis()

#Preprocessed data
data.get_metadata()
data.get_ts_preprocessed()
data.get_cogdata_preprocessed()
data.get_temporal_parameters()


#%%
tau = data.tau
lag = data.lag
n_animals = data.n_animals
nodes = data.regions
save_path = data.paths['speed']

min_tau_zero = True
tau_range = np.arange(0, tau + 1) if min_tau_zero else np.arange(-tau, tau + 1)

time_window_range = data.time_window_range

prefix = 'speed'  # Prefix for the DFC speed results
data.load_dfc_1_window(lag=data.lag, window=data.time_window_range[0])
# dfc_stream = data.dfc_stream
# Example usage for fc2
# if return_fc2:

analysis_kwargs = {
    'method': 'pearson',
    'prefix': prefix,  # Prefix for the DFC speed results
    'return_fc2': True,  # Set to True if you want to return the second FC matrix
    'preprocessors': processors,  # Number of parallel processors to use    
}

#%%
run_dfc_speed_analysis(data,
                        time_window_range,
                        tau_range,
                        lag,
                        save_path,
                        n_animals,
                        nodes,
                        load_cache=False,
                        processors=processors,
                        **analysis_kwargs)
#%%
analysis_kwargs = {
    'method': 'pearson',
    'prefix': prefix,  # Prefix for the DFC speed results
    'return_fc2': False,  # Set to True if you want to return the second FC matrix
}
run_dfc_speed_analysis(data, 
                        time_window_range, 
                        tau_range, 
                        lag, 
                        save_path, 
                        n_animals, 
                        nodes, 
                        load_cache=False, 
                        processors=processors,
                        **analysis_kwargs)


# %%
#Load results per window speed
save_speed = []
# Load the speed results for each window size
for idx, window_size in enumerate(time_window_range):
    window_file = save_path / f"{prefix}_win{window_size}_tau{np.size(tau_range)}_animals_{n_animals}.npz"
    with np.load(window_file, allow_pickle=True) as arr:
        if 'speeds' in arr:
            load_speed = arr['speeds']
            logger.info("Keys in file:", arr.files)
        else:
            logger.warning(f"Warning: 'speeds' not found in {window_file}, available: {arr.files}")
            continue
    save_speed.append(load_speed)
    
# Save the total speed results across all windows, tau and animals    
window_file_total = save_path / f"{prefix}_windows{len(time_window_range)}_tau{np.size(tau_range)}_animals_{n_animals}.pkl"
save_pickle(save_speed, window_file_total)
# with open(window_file_total, 'wb') as f:
#     pickle.dump(save_speed, f)

# %%
# Load fc2 results per window
total_fc2 = []

for window_size in time_window_range:
    # Load the fc2 results
    window_file = save_path / f"{prefix}_fc_win{window_size}_tau{np.size(tau_range)}_animals_{n_animals}.npz"
    load_fc2 = np.load(window_file, allow_pickle=True)['fc2']
    total_fc2.append(load_fc2[0])
    logger.info(f"Shape of fc2 for window {window_size}: {len(load_fc2[1])}")

# Save the total fc2 results across all windows, tau and animals
window_file_total = save_path / f"{prefix}_fc_windows{len(time_window_range)}_tau{np.size(tau_range)}_animals_{n_animals}.npz"
np.savez(window_file_total, fc2=np.array(total_fc2, dtype=object), allow_pickle=True)
# %%
