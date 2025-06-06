#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 14:42:38 2023

@author: samy
"""

#%%
from curses import window
from re import L
from tkinter import N
import numpy as np
import logging
import gc
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import Dict, List, Tuple, Optional, Union
import psutil
import warnings

from class_dataanalysis_julien import DFCAnalysis

#%%

def dfc_speed(dfc_stream, 
            vstep=1, 
            tau_range=0,
            method='pearson', 
            return_fc2=False,
            tril_indices=None
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

    indices_max = n_frames - (vstep + np.max(tau))
    
    indices = np.arange(0, indices_max, vstep)
    if np.size(tau_range) > 1:
        for tau_aux in tau_range:
            fc1_indices.append(indices[:-1])  # Indices for the first FC matrix
            fc2_indices.append(indices[1:]+tau_aux)   # Indices for the second FC matrix
    else:
        tau_aux = tau_range
        fc1_indices.append(indices[:-1])
        fc2_indices.append(indices[1:]+tau_aux)   # Indices for the second FC matrix

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


# data = DFCAnalysis()

# #Preprocessed data
# data.get_metadata()
# data.get_ts_preprocessed()
# data.get_cogdata_preprocessed()
# data.get_temporal_parameters()


# tau = data.tau

# save_path = data.paths['speed']
# min_tau_zero = True
# tau_range = np.arange(0, tau + 1) if min_tau_zero else np.arange(-tau, tau + 1)

# time_window_range = data.time_window_range

# processors = -1  # Use all available processors

# analysis_kwargs = {
#     'method': 'pearson',
#     'overlap_mode': 'points',
#     'overlap_value': 1,
#     'return_fc2': True,
#     'min_tau_zero': min_tau_zero,
# }

# data.load_dfc_1_window(lag=data.lag, window=data.time_window_range[0])
# dfc_stream = data.dfc_stream
# # Example usage for fc2
# # if return_fc2:
# processors = -1
# save_path = data.paths['speed']  # Directory to save results
# prefix= 'speed'  # Prefix for the DFC speed results

# result_0 = dfc_speed(
#     dfc_stream[0], 
#     vstep=int(time_window_range[0]),    
#     method='pearson', 
#     return_fc2=True
# )
# # Example usage for speeds
# result_0_speed = dfc_speed(
#     dfc_stream[0], 
#     vstep=1,    
#     method='pearson',
#     tau_range=tau_range, 
#     return_fc2=False
# )
#%%
def _handle_dfc_speed_analysis_per_animal(data, time_window_range, tau_range, lag, save_path, n_animals, nodes, load_cache=False, **kwargs):
    """
    DFC speed analysis handler: saves results per animal per tau.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Parameter extraction & checks (same as before)
    min_tau_zero = kwargs.get('min_tau_zero', True)
    method = kwargs.get('method', 'pearson')
    return_fc2 = kwargs.get('return_fc2', False)
    prefix = kwargs.get('prefix', 'speed')  # Prefix for the DFC speed results
    
    
    # Organize results per animal per tau: save immediately
    summary = {
        'metadata': {
            'window_size': time_window_range,
            'lag': lag,
            'tau_range': tau_range,
            'min_tau_zero': min_tau_zero,
            'method': method,
            'n_animals': n_animals,
            'nodes': nodes
        }
    }



    # for ws_idx, window_size in tqdm(enumerate(time_window_range), desc=f"Processing animals for window_size "):
    def parallel_helper(window_size):
        # window_size = 7
        # Load 1 DFC stream
        data.load_dfc_1_window(lag=lag, window=window_size)
        dfc_stream = data.dfc_stream
        logger.info(f"Loaded DFC stream shape: {dfc_stream.shape}")

        #Generate the file name for the current window size data
        if return_fc2:
            window_file = save_path / f"{prefix}_fc_win{window_size}_tau{np.size(tau_range)}_animals_{n_animals}.npz"
        else:
            window_file = save_path / f"{prefix}_win{window_size}_tau{np.size(tau_range)}_animals_{n_animals}.npz"

        # Iterate for a given window across all animals
        logger.info(f"Processing window size {window_size} with {n_animals} animals")
        
        fc2_animals = []
        speed_animals = []

        for animal_idx in range(n_animals):
            print(f"Processing animal {animal_idx + 1}/{n_animals} for window size {window_size}")

            # return
            # Check if results already exist
            if load_cache and window_file.exists():
                logger.info(f"Loading cached results for window {window_file}")
                return
            else:
                logger.info(f"Computing for window {window_file} and animal {animal_idx + 1}/{n_animals}")
                try:
                    if return_fc2:
                        print(return_fc2)
                        fc2 = dfc_speed(
                            dfc_stream[animal_idx], vstep=int(window_size), tau_range=0, method=method, return_fc2=True
                        )
                        fc2_animals.append(fc2)
                        # print(np.shape(fc2_animals))
                    else:
                        speeds = dfc_speed(
                            dfc_stream[animal_idx], vstep=1, tau_range=tau_range, method=method, return_fc2=False
                        )
                        speed_animals.append(speeds)
                        print(np.shape(speed_animals))

                except Exception as e:
                    print(f"Error for window {window_file}: {e}")
                # Free memory before next iteration
                if return_fc2:
                    del fc2
                else:
                    del speeds
                gc.collect()
                print(f"✓ Saved all animals and tau for windows={window_size} in {window_file}")

        # Save results for the current window size
        if return_fc2:
            fc2 = np.stack(fc2_animals)
            print(f"Shape of fc2: {np.shape(fc2)}")
            # Save the fc2 results
            np.savez_compressed(
                window_file,
                fc2=fc2,
                window_size=window_size
            )
            print(f"✓ Saved fc2 for window {window_file}")
        else:
            speeds = np.stack(speed_animals)
            # # Save the speeds results
            # if np.size(speeds) == 0:
            #     speeds = np.array([np.nan])
            # elif np.size(speeds) == 1:
            #     speeds = np.array([speeds])
            # else:
            #     speeds = speeds.reshape(-1, np.size(tau_range))
            np.savez_compressed(
                window_file,
                speeds=speeds,
                window_size=window_size
            )
            print(f"✓ Saved speeds for window {window_file}")
            print(f"✓ Saved fc2 for window {window_file}")

    # Use Parallel to handle multiple windows in parallel
    results = Parallel(n_jobs=processors, verbose=10)(
        delayed(parallel_helper)(window_size)
    for window_size in tqdm(time_window_range, desc=f"Processing animals for ...")
    )

    logger.info("All windows processed successfully.")

#%%
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

processors = -1  # Use all available processors

prefix = 'speed'  # Prefix for the DFC speed results

data.load_dfc_1_window(lag=data.lag, window=data.time_window_range[0])
dfc_stream = data.dfc_stream
# Example usage for fc2
# if return_fc2:
processors = -1
save_path = data.paths['speed']  # Directory to save results
window_size = time_window_range

analysis_kwargs = {
    'method': 'pearson',
    'min_tau_zero': min_tau_zero,
    'prefix': prefix,  # Prefix for the DFC speed results
    'return_fc2': True,  # Set to True if you want to return the second FC matrix
}
_handle_dfc_speed_analysis_per_animal(data, 
                                      time_window_range, 
                                      tau_range, 
                                      lag, 
                                      save_path, 
                                      n_animals, 
                                      nodes, 
                                      load_cache=False, 
                                      **analysis_kwargs)
#%%
analysis_kwargs = {
    'method': 'pearson',
    'min_tau_zero': min_tau_zero,
    'prefix': prefix,  # Prefix for the DFC speed results
    'return_fc2': False,  # Set to True if you want to return the second FC matrix
}
_handle_dfc_speed_analysis_per_animal(data, 
                                      time_window_range, 
                                      tau_range, 
                                      lag, 
                                      save_path, 
                                      n_animals, 
                                      nodes, 
                                      load_cache=True, 
                                      **analysis_kwargs)


# %%
#Load results
total_speed = []

for window_size in time_window_range:
    window_file = save_path / f"{prefix}_win{window_size}_tau{np.size(tau_range)}_animals_{n_animals}.npz"
    load_speed = np.load(window_file, allow_pickle=True)['speeds']
    total_speed.append(load_speed)
    print(np.shape(load_speed))
    
    
window_file_total = save_path / f"{prefix}_windows{len(time_window_range)}_tau{np.size(tau_range)}_animals_{n_animals}.npz"
np.savez_compressed(window_file_total, speeds=total_speed, allow_pickle=True)
# %%
# Load fc2 results
total_fc2 = []

for window_size in time_window_range:
    # Load the fc2 results
    window_file = save_path / f"{prefix}_fc_win{window_size}_tau{np.size(tau_range)}_animals_{n_animals}.npz"
    load_fc2 = np.load(window_file, allow_pickle=True)['fc2']
    total_fc2.append(load_fc2[0])
    print(f"Shape of fc2 for window {window_size}: {len(load_fc2[1])}")
    # print((total_fc2))

window_file_total = save_path / f"{prefix}_fc_windows{len(time_window_range)}_tau{np.size(tau_range)}_animals_{n_animals}.npz"
np.savez(window_file_total, fc2=np.array(total_fc2, dtype=object), allow_pickle=True)
# %%
