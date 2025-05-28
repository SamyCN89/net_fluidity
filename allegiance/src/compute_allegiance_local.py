#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:26:30 2024

@author: samy
"""

#%%
from calendar import c
from matplotlib import pyplot as plt
import numpy as np
import time
# from functions_analysis import *
from pathlib import Path
import sys
sys.path.append('../../shared_code')

# from sphinx import ret

from shared_code.fun_loaddata import *
from shared_code.fun_dfcspeed import *

from shared_code.fun_metaconnectivity import (compute_metaconnectivity, 
                                  intramodule_indices_mask, 
                                  get_fc_mc_indices, 
                                  get_mc_region_identities, 
                                  fun_allegiance_communities,
                                  compute_trimers_identity,
                                    build_trimer_mask,
                                  )

from shared_code.fun_utils import (set_figure_params, 
                       get_paths, 
                       load_cognitive_data,
                       load_timeseries_data,
                       load_grouping_data,
                       )
# =============================================================================
# This code compute 
# Load the data
# Intersect the 2 and 4 months to have data that have the two datapoints
# ========================== Figure parameters ================================
save_fig = set_figure_params(False)

# =================== Paths and folders =======================================
timeseries_folder = 'Timecourses_updated_03052024'
# Define the timeseries directory
timeseries_folder = 'Timecourses_updated_03052024'
# Will prioritize PROJECT_DATA_ROOT if set
paths = get_paths(timecourse_folder=timeseries_folder)

# external_disk = True
# if external_disk==True:
#     root = Path('/media/samy/Elements1/Proyectos/LauraHarsan/script_mc/')
# else:    
#     root = Path('/home/samy/Bureau/Proyect/LauraHarsan/Ines/')

# paths = get_paths(external_disk=True,
#                   external_path=root,
#                   timecourse_folder=timeseries_folder)

# ========================== Load data =========================
cog_data_filtered = load_cognitive_data(paths['sorted'] / 'cog_data_sorted_2m4m.csv')
data_ts = load_timeseries_data(paths['sorted'] / 'ts_and_meta_2m4m.npz')
mask_groups, label_variables = load_grouping_data(paths['results'] / "grouping_data_oip.pkl")


# ========================== Indices ==========================================
ts=data_ts['ts']
n_animals = data_ts['n_animals']
regions = data_ts['regions']
anat_labels = data_ts['anat_labels']

#Binarize the ts of one animal
ts_bin = np.array([np.where(ts[i] > np.std(ts[i])/10, 1, 0) for i in range(n_animals)])


# Plot the ts of one animal in a matrix
plt.figure(figsize=(10, 5))
plt.imshow(ts_bin[0].T, aspect='auto', interpolation='none', cmap='Greys')
plt.colorbar()
plt.title('Time series of one animal')
plt.xlabel('Time')
plt.ylabel('Regions')
if save_fig:
    plt.savefig(paths['figures'] / 'ts_bin_animal_0.png', dpi=300, bbox_inches='tight')
plt.show()
# plt.figure(figsize=(10, 5))
# plt.imshow(ts_bin[0], aspect='auto')
# plt.colorbar()
# plt.title('Time series of one animal')
# %%
#Plot the ts_bin of one animal in plot
plt.figure(figsize=(10, 5))
plt.clf()
plt.plot(ts_bin[0], color='grey', alpha=0.1)
plt.title('Time series of one animal')
plt.xlabel('Time')
plt.ylabel('Regions')
# plt.xlim(0, 1000)
plt.ylim(0, 1)
plt.show()

#%%Metaconnectivity
from joblib import Parallel, delayed, parallel_backend
def compute_dfc_stream(ts_data, window_size=7, lag=1, format_data='3D',save_path=None, n_jobs=-1):
    """
    This function calculates dynamic functional connectivity (DFC) streams from time-series data using 
    a sliding window approach. It supports parallel computation and caching of results 
    to optimize performance.

    -----------
    ts_data : np.ndarray
        A 3D array of shape (n_animals, n_regions, n_timepoints) representing the 
        time-series data for multiple animals and brain regions.
    window_size : int, optional
        The size of the sliding window used for dynamic functional connectivity (DFC) 
        computation. Default is 7.
    lag : int, optional
        The lag parameter for time-series analysis. Default is 1.
    return_dfc : bool, optional
        If True, the function also returns the DFC stream. Default is False.
    save_path : str or None, optional
        The directory path where the computed meta-connectivity and DFC stream will 
        be saved. If None, results are not saved. Default is None.
    n_jobs : int, optional
        The number of parallel jobs to use for computation. Use -1 to utilize all 
        available CPU cores. Default is -1.

    --------
    mc : np.ndarray
        A 3D array of meta-connectivity matrices for each animal.
    dfc_stream : np.ndarray, optional
        A 4D array of DFC streams for each animal, returned only if `return_dfc` is True.

    Notes:
    ------
    - If a `save_path` is provided and a cached result exists, the function will load 
      the cached data instead of recomputing it.
    - The function uses joblib for parallel computation, with the "loky" backend.
    - The meta-connectivity matrices are computed by correlating the DFC streams.

    Examples:
    ---------
    # Example usage:
    mc = compute_metaconnectivity(ts_data, window_size=10, lag=2, save_path="./cache")
    mc, dfc_stream = compute_metaconnectivity(ts_data, return_dfc=True, n_jobs=4)
    """

    n_animals, tr_points, nodes  = ts_data.shape
    dfc_stream  = None
    mc          = None

    # File path setup
    save_path = Path(save_path) if save_path else None
    file_path = (
        save_path / f"dfc_window_size={window_size}_lag={lag}_animals={n_animals}_regions={nodes}.npz"
        if save_path else None
    )
    if file_path:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # file_path = os.path.join(save_path, f'mc_window_size={window_size}_lag={lag}_animals={n_animals}_regions={nodes}.npz')
        # os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Load from cache
    if file_path and file_path.exists():
        print(f"Loading dFC stream from: {file_path}")
        data = np.load(file_path, allow_pickle=True)
        dfc_stream = data['dfc_stream'] 
    else:
        print(f"Computing dFC stream in parallel (window_size={window_size}, lag={lag})...")

        # Parallel DFC stream computation per animal
        with parallel_backend("loky", n_jobs=n_jobs):
            dfc_stream_list = Parallel()(
                delayed(ts2dfc_stream)(ts_data[i], window_size, lag, format_data=format_data)
                # for i in tqdm(range(n_animals), desc="DFC Streams")
                for i in range(n_animals)
            )
        dfc_stream = np.stack(dfc_stream_list)

    # Save results if path is provided
    if file_path:
        print(f"Saving dFC stream to: {file_path}")
        np.savez_compressed(file_path, dfc_stream=dfc_stream)
    return dfc_stream

#%% Compute the DFC stream
#Parameters speed

PROCESSORS =-1

lag=1
tau=5
window_size = 10
window_parameter = (5,100,1)

#Parameters allegiance analysis
n_runs_allegiance = 1000
gamma_pt_allegiance = 100

tau_array       = np.append(np.arange(0,tau), tau ) 
lentau          = len(tau_array)

time_window_min, time_window_max, time_window_step = window_parameter
time_window_range = np.arange(time_window_min,
                              time_window_max+1,
                              time_window_step)

#%%compute dfc stream
# Compute the DFC stream
start = time.time()
dfc_stream = compute_dfc_stream(ts, 
                              window_size=window_size, 
                              lag=lag, 
                              n_jobs =PROCESSORS,
                              save_path = paths['mc'],
                              )
stop = time.time()
print(f'DFC stream computation time {stop-start}')

dfc_stream = np.transpose(dfc_stream, (0, 3, 2, 1)) # (n_animals, n_windows, n_regions, n_regions)

#%%
subject= dfc_stream[0]

# Plot the dfc stream of one animal in a matrix
plt.figure(figsize=(10, 5))
plt.imshow(abs(subject)>0.4, aspect='auto', interpolation='none', cmap='Greys')
plt.colorbar()
plt.title('DFC stream of one animal')
plt.xlabel('Time')
plt.ylabel('Regions')


# %%
dfc_communities = np.empty((n_animals, dfc_stream.shape[1], regions))
sort_allegiance = np.empty((n_animals, dfc_stream.shape[1], regions))
contingency_matrix = np.empty((n_animals, dfc_stream.shape[1], regions, regions))

# Compute the allegiance communities for each time window
for i in range(dfc_stream.shape[1]):
    # Compute the allegiance communities for each time window
    dfc_communities[i], sort_allegiance[i], contingency_matrix[i] = fun_allegiance_communities(dfc_stream[0,i], 
                                                                                                       n_runs = n_runs_allegiance, 
                                                                                                       gamma_pt = gamma_pt_allegiance, 
                                                                                                       save_path=paths['allegiance'],
                                                                                                       ref_name='test_dfc', 
                                                                                                       n_jobs=PROCESSORS,
                                                                                                       )


# %%

# Plot the contingency matrix
plt.figure(figsize=(10, 5))
plt.imshow(contingency_matrix, aspect='auto', interpolation='none', cmap='Greys')
plt.colorbar()
plt.title('Contingency matrix')
plt.xlabel('Time')
plt.ylabel('Regions')
plt.xticks(np.arange(len(anat_labels)), anat_labels, rotation=90)
# plt.yticks(np.arange(len(label_variables)), label_variables)
plt.tight_layout()
# if save_fig:
#     plt.savefig(paths['figures'] / 'contingency_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

#%%
# Ines relevant sites
#RSP, ENT, HIP, thalamic nuclei, amygdala, and substantia nigra, reuniens, sensorimotor
