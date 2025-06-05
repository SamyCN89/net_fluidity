#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 14:42:38 2023

@author: samy
"""

#%%
from pathlib import Path
from unittest import result
import numpy as np
import matplotlib.pyplot as plt
import brainconn as bct
import os
import time
import logging

from joblib import Parallel, delayed, parallel_backend
import pandas as pd
# from functions_analysis import *
from scipy.io import loadmat, savemat
from scipy.special import erfc
from scipy.stats import pearsonr, spearmanr
from sklearn import base
from tqdm import tqdm

from shared_code.fun_metaconnectivity import compute_metaconnectivity
from shared_code.fun_dfcspeed import parallel_dfc_speed_oversampled_series, dfc_speed
from shared_code.fun_loaddata import *  # Import only needed functions
from shared_code.fun_utils import set_figure_params
from shared_code.fun_paths import get_paths

from class_dataanalysis_julien import DFCAnalysis

data = DFCAnalysis()

#Preprocessed data
data.get_metadata()
data.get_ts_preprocessed()
data.get_cogdata_preprocessed()
data.get_temporal_parameters()


processors = -1
save_path = data.paths['results']  # Directory to save results

SAVE_DATA = True


#%%

window_size = 9
prefix= 'speed'  # Prefix for the DFC speed results
kwargs = {'tau': data.tau, 'min_tau_zero': True, 'method': 'pearson'}

def _handle_dfc_speed_analysis(window_size, lag, save_path, n_animals, nodes, load_cache=True, **kwargs):
    """
    Improved DFC speed analysis handler with better organization and functionality.
    
    Parameters:
        window_size (int): Sliding window size for DFC computation.
        lag (int): Step size between windows.
        save_path (Path): Directory to save results.
        n_animals (int): Number of animals in the dataset.
        nodes (int): Number of brain regions/nodes.
        load_cache (bool): Whether to load from cache if available.
        **kwargs: Additional parameters:
            - tau (int): Maximum temporal shift for oversampling (default: 3)
            - min_tau_zero (bool): Whether tau starts at 0 (default: True)
            - method (str): Correlation method ('pearson', 'spearman', 'cosine')
            - overlap_mode (str): 'points' or 'percentage' (default: 'points')
            - overlap_value (float): Overlap amount (default: 1 for points, 0.5 for percentage)
            - return_fc2 (bool): Whether to return FC2 arrays (default: True)
        
    Returns:
        dict: Results containing median speeds, speed arrays, and FC2 arrays organized by tau.
    """
    logger = logging.getLogger(__name__)
    
    # =================== PARAMETER VALIDATION AND SETUP ===================
    # Extract and validate parameters
    tau = kwargs.get('tau', 3)
    min_tau_zero = kwargs.get('min_tau_zero', True) 
    method = kwargs.get('method', 'pearson')
    overlap_mode = kwargs.get('overlap_mode', 'points')
    overlap_value = kwargs.get('overlap_value', 1 if overlap_mode == 'points' else 0.5)
    return_fc2 = kwargs.get('return_fc2', True)
    
    # Validate parameters
    if not isinstance(tau, int) or tau < 0:
        raise ValueError(f"tau must be a non-negative integer, got {tau}")
    if method not in ['pearson', 'spearman', 'cosine']:
        raise ValueError(f"Unsupported method '{method}'")
    if overlap_mode not in ['points', 'percentage']:
        raise ValueError(f"overlap_mode must be 'points' or 'percentage', got '{overlap_mode}'")
    
    # Create tau range
    tau_range = np.arange(0, tau + 1) if min_tau_zero else np.arange(-tau, tau + 1)
    logger.info(f"Using tau range: {tau_range} (min_tau_zero={min_tau_zero})")
    
    # =================== FILE PATH MANAGEMENT ===================
    prefix = 'speed'
    file_path = None
    if save_path:
        file_path = make_file_path(save_path / prefix, prefix, window_size, tau, n_animals, nodes)
        
        # Try loading from cache
        if load_cache and file_path and file_path.exists():
            try:
                cached_data = load_npz_dict(file_path)
                logger.info(f"Successfully loaded DFC speed from cache: {file_path}")
                return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cached DFC speed: {e}. Recomputing...")
    
    # =================== DFC STREAM LOADING ===================
    logger.info(f"Computing DFC speed (window_size={window_size}, lag={lag}, tau={tau}, method={method})")
    
    # Load DFC stream
    data.load_dfc_1_window(lag=lag,window=window_size)
    # dfc_file_path = make_file_path(save_path / 'dfc', 'dfc', window_size, lag, n_animals, nodes)
    # dfc_stream = None
    logger.info(f"Loaded DFC stream shape: {data.dfc_stream.shape}")
    
    # =================== OVERLAP CALCULATION ===================
    def calculate_vstep(window_size, lag, overlap_mode, overlap_value):
        """Calculate vstep based on overlap mode and value."""
        if overlap_mode == 'points':
            return max(1, int(overlap_value))
        elif overlap_mode == 'percentage':
            if not 0 <= overlap_value <= 1:
                raise ValueError(f"Overlap percentage must be between 0 and 1, got {overlap_value}")
            step_size = int(window_size * (1 - overlap_value))
            return max(1, step_size)
        else:
            raise ValueError(f"Unknown overlap_mode: {overlap_mode}")
    
    base_vstep = calculate_vstep(window_size, lag, overlap_mode, overlap_value)
    logger.info(f"Base vstep: {base_vstep} (overlap_mode={overlap_mode}, overlap_value={overlap_value})")
    
    # =================== SPEED COMPUTATION ===================
    # Organize results by tau and animal
    results_by_tau = {}
    
    # print(tau_range, base_vstep)
    
    for tau_val in tau_range:
        logger.info(f"Computing speeds for tau={tau_val}")
        
        # Calculate effective vstep for this tau
        effective_vstep = int(base_vstep + tau_val)
        if effective_vstep <= 0:
            logger.warning(f"Skipping tau={tau_val}: effective_vstep={effective_vstep} <= 0")
            continue
        
        # Compute speeds for all animals with this tau
        animal_results = []
        for i in tqdm(range(n_animals), desc=f'Animal loop (tau={tau_val})'):
            try:
                if return_fc2:
                    # print(f"Effective vstep for tau={tau_val}: {effective_vstep}")
                    # print("Effective vstep is not an integer" if not isinstance(effective_vstep, int) else "Effective vstep is an integer")
                    median_speed, speeds, fc2 = dfc_speed(
                        data.dfc_stream[i], vstep=effective_vstep, method=method, return_fc2=True
                    )
                    animal_results.append({
                        'median_speed': median_speed,
                        'speeds': speeds,
                        'fc2': fc2,
                        'animal_id': i
                    })
                else:
                    median_speed, speeds = dfc_speed(
                        data.dfc_stream[i], vstep=effective_vstep, method=method, return_fc2=False
                    )
                    animal_results.append({
                        'median_speed': median_speed,
                        'speeds': speeds,
                        'animal_id': i
                    })
            except Exception as e:
                logger.error(f"Error computing speed for animal {i}, tau={tau_val}: {e}")
                # Add NaN results to maintain structure
                animal_results.append({
                    'median_speed': np.nan,
                    'speeds': np.array([np.nan]),
                    'fc2': np.array([np.nan]) if return_fc2 else None,
                    'animal_id': i
                })
        
        results_by_tau[tau_val] = animal_results
    
    # =================== RESULTS ORGANIZATION ===================
    # Organize results into structured format
    organized_results = {
        'metadata': {
            'window_size': window_size,
            'lag': lag,
            'tau': tau,
            'tau_range': tau_range,
            'min_tau_zero': min_tau_zero,
            'method': method,
            'overlap_mode': overlap_mode,
            'overlap_value': overlap_value,
            'base_vstep': base_vstep,
            'n_animals': n_animals,
            'nodes': nodes
        }
    }
    
    # Extract arrays organized by tau
    for tau_val in tau_range:
        if tau_val not in results_by_tau:
            continue
        
        tau_results = results_by_tau[tau_val]
        
        # Extract median speeds
        # median_speeds = np.array([r['median_speed'] for r in tau_results])
        
        # Extract speed arrays (keep as object array due to potentially different lengths)
        speed_arrays = np.array([r['speeds'] for r in tau_results], dtype=object)
        
        # Extract FC2 arrays if available
        if return_fc2:
            fc2_arrays = np.array([r['fc2'] for r in tau_results], dtype=object)
            organized_results[f'fc2_tau_{tau_val}'] = fc2_arrays
        
        # organized_results[f'median_speeds_tau_{tau_val}'] = median_speeds
        organized_results[f'speed_arrays_tau_{tau_val}'] = speed_arrays
    
    # =================== SAVE RESULTS ===================
    if file_path:
        try:
            # Use numpy savez_compressed for better organization
            np.savez_compressed(file_path, **organized_results)
            logger.info(f'Successfully saved DFC speed results to: {file_path}')
            logger.info(f'Saved keys: {list(organized_results.keys())}')
        except Exception as e:
            logger.error(f'Failed to save results to {file_path}: {e}')
    
    logger.info(f"DFC speed analysis completed for window_size={window_size}")
    return organized_results
#%%

# ---------------------------- TESTING THE IMPROVED FUNCTION ------------------------

#%%
# Test the improved function with just the first window size
try:
    test_window_size = data.time_window_range[0]
    results = _handle_dfc_speed_analysis(test_window_size, 
                                            data.lag, 
                                            save_path, 
                                            data.n_animals, 
                                            data.regions, load_cache=True, **kwargs)
    print(f"Testing improved DFC speed analysis with window_size={test_window_size}")
    print(f"✓ Success! Results keys: {list(results.keys())}")
    
    # If successful, run for all window sizes
    print("Running for all window sizes...")
    all_results = {}
    for window_size in data.time_window_range:
        print(f"Processing window_size={window_size}")
        results = _handle_dfc_speed_analysis(window_size, data.lag, save_path, data.n_animals, data.regions, load_cache=True, **kwargs)
        all_results[window_size] = results
    
    print(f"✓ Completed processing {len(all_results)} window sizes")
    
except Exception as e:
    print(f"✗ Error in improved DFC speed analysis: {e}")
    import traceback
    traceback.print_exc()

# %%

# Load the computed results
speed = []
for window_size in data.time_window_range:
    prefix = 'speed'
    file_path = make_file_path(save_path / prefix, prefix, window_size, data.tau, data.n_animals, data.regions)
    
    if file_path.exists():
        print(f"Loading results for window_size={window_size}")
        print(f"Available keys:", np.load(file_path, allow_pickle=True).files)
        
        # Check available keys
        results = load_npz_dict(file_path)
        
        # Look for the median speeds for the main tau value (tau=3)
        # The key should be 'median_speeds_tau_3' based on our improved function
        speed_key = f'median_speeds_tau_{data.tau}'
        
        if speed_key in results:
            speed.append(results[speed_key])
            print(f"✓ Loaded {speed_key} with shape: {results[speed_key].shape}")
        else:
            # Fallback: try to find any median speeds key
            available_speed_keys = [key for key in results.keys() if 'median_speeds_tau' in key]
            if available_speed_keys:
                # Use the first available tau
                fallback_key = available_speed_keys[0]
                speed.append(results[fallback_key])
                print(f"⚠ Using fallback key {fallback_key} instead of {speed_key}")
            else:
                print(f"✗ No speed data found for window_size={window_size}")
                print(f"Available keys: {list(results.keys())}")
                speed.append(np.array([]))  # Add empty array to maintain structure
    else:
        print(f"✗ File not found: {file_path}")
        speed.append(np.array([]))  # Add empty array to maintain structure

print(f"Loaded speed data for {len(speed)} window sizes")
# speed = np.array(speed, dtype=object)  # Convert to numpy array if needed
# %%

a = [speed[ws] for ws in np.arange(len(data.time_window_range))]
# %%
