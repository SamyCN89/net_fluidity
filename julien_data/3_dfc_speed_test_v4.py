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
import scipy as sp
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

import gc
import numpy as np
from tqdm import tqdm

#%%
from class_dataanalysis_julien import DFCAnalysis
data = DFCAnalysis()

#Preprocessed data
data.get_metadata()
data.get_ts_preprocessed()
data.get_cogdata_preprocessed()
data.get_temporal_parameters()


#%%
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
import scipy as sp
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

import gc
import numpy as np
from tqdm import tqdm

#%%
from class_dataanalysis_julien import DFCAnalysis
data = DFCAnalysis()

#Preprocessed data
data.get_metadata()
data.get_ts_preprocessed()
data.get_cogdata_preprocessed()
data.get_temporal_parameters()


#%%

def _handle_dfc_speed_analysis_per_animal(window_size, lag, save_path, n_animals, nodes, load_cache=True, **kwargs):
    """
    DFC speed analysis handler: saves results per animal per tau.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Parameter extraction & checks (same as before)
    tau = kwargs.get('tau', 3)
    min_tau_zero = kwargs.get('min_tau_zero', True)
    method = kwargs.get('method', 'pearson')
    overlap_mode = kwargs.get('overlap_mode', 'points')
    overlap_value = kwargs.get('overlap_value', 1 if overlap_mode == 'points' else 0.5)
    return_fc2 = kwargs.get('return_fc2', False)
    
    
    # Compute vstep based on overlap_mode and overlap_value
    def calculate_vstep(window_size, lag, overlap_mode, overlap_value):
        if overlap_mode == 'points':
            return max(1, int(overlap_value))
        elif overlap_mode == 'percentage':
            if not 0 <= overlap_value <= 1:
                raise ValueError(f"Overlap percentage must be between 0 and 1, got {overlap_value}")
            return max(1, int(window_size * (1 - overlap_value)))
        else:
            raise ValueError(f"Unknown overlap_mode: {overlap_mode}")

    # Load DFC stream once
    data.load_dfc_1_window(lag=lag, window=window_size)
    logger.info(f"Loaded DFC stream shape: {data.dfc_stream.shape}")
    
    # Calculate vstep based on overlap_mode and overlap_value
    base_vstep = calculate_vstep(window_size, lag, overlap_mode, overlap_value)
    effective_vstep = int(base_vstep + tau)
    logger.info(f"Base vstep: {base_vstep} (overlap_mode={overlap_mode}, overlap_value={overlap_value})")

    # Organize results per animal per tau: save immediately
    summary = {
        'metadata': {
            'window_size': window_size,
            'lag': lag,
            'tau': tau,
            'min_tau_zero': min_tau_zero,
            'method': method,
            'overlap_mode': overlap_mode,
            'overlap_value': overlap_value,
            'base_vstep': base_vstep,
            'n_animals': n_animals,
            'regions': nodes
        }
    }

    # Use subdirectory for this window_size (optional but tidy)
    out_dir = save_path / f"speed_win{window_size}"
    out_dir.mkdir(exist_ok=True, parents=True)


    tau_dir = out_dir 


    for i in tqdm(range(n_animals), desc=f"Window {window_size} (tau={tau})"):
    # Effective vstep calculation
        if effective_vstep <= 0:
            logger.warning(f"Skipping tau={tau}: effective_vstep={effective_vstep} <= 0")
            continue
        animal_file = tau_dir / f"speed_window={window_size}_tau={tau}_animals={n_animals}_regions={nodes}.npz"
        # Check if results already exist
        if load_cache and animal_file.exists():

            logger.info(f"Loading cached results for animals {n_animals} tau={tau} from {animal_file}")
            continue
        try:
            if return_fc2:
                median_speed, speeds, fc2 = dfc_speed(
                    data.dfc_stream[i], vstep=effective_vstep, method=method, return_fc2=True
                )
                np.savez_compressed(
                    animal_file,
                    median_speed=median_speed,
                    speeds=speeds,
                    fc2=fc2
                )
            else:
                median_speed, speeds = dfc_speed(
                    data.dfc_stream[i], vstep=effective_vstep, method=method, return_fc2=False
                )
                np.savez_compressed(
                    animal_file,
                    median_speed=median_speed,
                    speeds=speeds
                )
        except Exception as e:
            logger.error(f"Error for animal {i} tau={tau_val}: {e}")
            # Save NaNs so you know you have a result file
            np.savez_compressed(
                animal_file,
                median_speed=np.nan,
                #np.array of shape of speeds with naN
                speeds=np.array([np.nan]),
                fc2=np.array([np.nan]) if return_fc2 else None
            )
        # Free memory before next iteration
        del median_speed, speeds
        if return_fc2:
            del fc2
        gc.collect()
    logger.info(f"✓ Saved all animals for tau={tau_val} in {tau_dir}")

    # return summary  # Only summary metadata in memory
# %%
#Run the code with the refactored function

processors = -1
save_path = data.paths['speed']  # Directory to save results
prefix= 'speed'  # Prefix for the DFC speed results


min_tau_zero =True
tau_range = np.arange(0, data.tau + 1) if min_tau_zero else np.arange(-data.tau, data.tau + 1)

for tau_val in tau_range:

    # Load DFC stream once
    data.load_dfc_1_window(lag=lag, window=window_size)
    logger.info(f"Loaded DFC stream shape: {data.dfc_stream.shape}")

	kwargs = {
	    'tau': tau_val,
	    'min_tau_zero': min_tau_zero,
	    'method': 'pearson',
	    'overlap_mode': 'points',
	    'overlap_value': 1,    # Or use 'percentage' and e.g. 0.5
	    'return_fc2': True,   # Save memory: only save FC2 if needed!
	}

	Parallel(n_jobs=processors)(delayed(_handle_dfc_speed_analysis_per_animal)(
	    window_size=window_size,
	    lag=data.lag,
	    save_path=save_path,
	    n_animals=data.n_animals,
	    nodes=data.regions,
	    load_cache=True,   # Always recompute (True if you want to check for files)
	    **kwargs
	) for window_size in data.time_window_range)







# def _handle_dfc_speed_analysis_per_animal(window_size, lag, save_path, n_animals, nodes, load_cache=True, **kwargs):
#     """
#     DFC speed analysis handler: saves results per animal per tau.
#     """
#     import logging
#     logger = logging.getLogger(__name__)
    
#     # Parameter extraction & checks (same as before)
#     tau = kwargs.get('tau', 3)
#     min_tau_zero = kwargs.get('min_tau_zero', True)
#     method = kwargs.get('method', 'pearson')
#     overlap_mode = kwargs.get('overlap_mode', 'points')
#     overlap_value = kwargs.get('overlap_value', 1 if overlap_mode == 'points' else 0.5)
#     return_fc2 = kwargs.get('return_fc2', False)
    
#     tau_range = np.arange(0, tau + 1) if min_tau_zero else np.arange(-tau, tau + 1)

#     # Load DFC stream once
#     data.load_dfc_1_window(lag=lag, window=window_size)
#     logger.info(f"Loaded DFC stream shape: {data.dfc_stream.shape}")
    
#     def calculate_vstep(window_size, lag, overlap_mode, overlap_value):
#         if overlap_mode == 'points':
#             return max(1, int(overlap_value))
#         elif overlap_mode == 'percentage':
#             if not 0 <= overlap_value <= 1:
#                 raise ValueError(f"Overlap percentage must be between 0 and 1, got {overlap_value}")
#             return max(1, int(window_size * (1 - overlap_value)))
#         else:
#             raise ValueError(f"Unknown overlap_mode: {overlap_mode}")
    
#     base_vstep = calculate_vstep(window_size, lag, overlap_mode, overlap_value)
#     logger.info(f"Base vstep: {base_vstep} (overlap_mode={overlap_mode}, overlap_value={overlap_value})")

#     # Organize results per animal per tau: save immediately
#     summary = {
#         'metadata': {
#             'window_size': window_size,
#             'lag': lag,
#             'tau': tau,
#             'tau_range': tau_range,
#             'min_tau_zero': min_tau_zero,
#             'method': method,
#             'overlap_mode': overlap_mode,
#             'overlap_value': overlap_value,
#             'base_vstep': base_vstep,
#             'n_animals': n_animals,
#             'nodes': nodes
#         }
#     }

#     # Use subdirectory for this window_size (optional but tidy)
#     out_dir = save_path / f"speed_win{window_size}"
#     out_dir.mkdir(exist_ok=True, parents=True)

#     for tau_val in tau_range:
#         logger.info(f"Processing tau={tau_val}")
#         tau_dir = out_dir / f"tau_{tau_val}"
#         tau_dir.mkdir(exist_ok=True, parents=True)
#         effective_vstep = int(base_vstep + tau_val)
#         if effective_vstep <= 0:
#             logger.warning(f"Skipping tau={tau_val}: effective_vstep={effective_vstep} <= 0")
#             continue
#         for i in tqdm(range(n_animals), desc=f"Animal (tau={tau_val})"):
#             animal_file = tau_dir / f"animal_{i}.npz"
#             # Check if results already exist
#             if load_cache and animal_file.exists():
#                 logger.info(f"Loading cached results for animal {i} tau={tau_val} from {animal_file}")
#                 continue
#             try:
#                 if return_fc2:
#                     median_speed, speeds, fc2 = dfc_speed(
#                         data.dfc_stream[i], vstep=effective_vstep, method=method, return_fc2=True
#                     )
#                     np.savez_compressed(
#                         tau_dir / f"animal_{i}.npz",
#                         median_speed=median_speed,
#                         speeds=speeds,
#                         fc2=fc2
#                     )
#                 else:
#                     median_speed, speeds = dfc_speed(
#                         data.dfc_stream[i], vstep=effective_vstep, method=method, return_fc2=False
#                     )
#                     np.savez_compressed(
#                         tau_dir / f"animal_{i}.npz",
#                         median_speed=median_speed,
#                         speeds=speeds
#                     )
#             except Exception as e:
#                 logger.error(f"Error for animal {i} tau={tau_val}: {e}")
#                 # Save NaNs so you know you have a result file
#                 np.savez_compressed(
#                     tau_dir / f"animal_{i}.npz",
#                     median_speed=np.nan,
#                     speeds=np.array([np.nan]),
#                     fc2=np.array([np.nan]) if return_fc2 else None
#                 )
#             # Free memory before next iteration
#             del median_speed, speeds
#             if return_fc2:
#                 del fc2
#             gc.collect()
#         logger.info(f"✓ Saved all animals for tau={tau_val} in {tau_dir}")

#     logger.info(f"✓ DFC speed analysis complete for window_size={window_size}. Results saved per animal per tau.")
#     # return summary  # Only summary metadata in memory
# # %%
# #Run the code with the refactored function

# processors = -1
# save_path = data.paths['speed']  # Directory to save results
# prefix= 'speed'  # Prefix for the DFC speed results

# # --- Ensure the refactored function is in your script or imported ---
# # Parameters for DFC speed analysis
# # window_size = 9
# # Example kwargs (customize as needed)
# kwargs = {
#     'tau': data.tau,
#     'min_tau_zero': True,
#     'method': 'pearson',
#     'overlap_mode': 'points',
#     'overlap_value': 1,    # Or use 'percentage' and e.g. 0.5
#     'return_fc2': True,   # Save memory: only save FC2 if needed!
# }

# # save_path = Path(data.paths['results'])   # Or your own desired path

# # all_summaries = {}

# # for window_size in data.time_window_range:
# #     print(f"\n=== Processing window_size={window_size} ===")
# #     summary = _handle_dfc_speed_analysis_per_animal(
# #         window_size=window_size,
# #         lag=data.lag,
# #         save_path=save_path,
# #         n_animals=data.n_animals,
# #         nodes=data.regions,
# #         load_cache=True,   # Always recompute (True if you want to check for files)
# #         **kwargs
# #     )

# # for window_size in data.time_window_range:
# # print(f"\n=== Processing window_size={window_size} ===")
# Parallel(n_jobs=processors)(delayed(_handle_dfc_speed_analysis_per_animal)(
#     window_size=window_size,
#     lag=data.lag,
#     save_path=save_path,
#     n_animals=data.n_animals,
#     nodes=data.regions,
#     load_cache=True,   # Always recompute (True if you want to check for files)
#     **kwargs
# ) for window_size in data.time_window_range)

# # all_summaries[window_size] = summary
# # print(f"✓ Done window_size={window_size}")

# # print("\n=== All window sizes processed! ===")


# #%%





# # Merge data from all animals and all windows into a single structure
# # This will load all speed and fc2 arrays saved per animal, per tau, per window.


# #V2


# # def load_speed_fc2_all(save_path, time_window_range, tau_range, n_animals):
# #     """
# #     Loads all speeds and fc2 arrays saved per animal, per tau, per window.

# #     Returns:
# #         results[window_size][tau_val]['speeds']: list of speed arrays, shape (n_animals, variable_length)
# #         results[window_size][tau_val]['fc2']: list of fc2 arrays, shape (n_animals, variable_shape)
# #     """
# #     results = {}
# #     for window_size in tqdm(time_window_range):
# #         results[window_size] = {}
# #         for tau_val in tau_range:
# #             speeds = []
# #             fc2s = []
# #             for i in range(n_animals):
# #                 file_path = Path(save_path) / f"speed_win{window_size}/tau_{tau_val}/animal_{i}.npz"
# #                 if file_path.exists():
# #                     with np.load(file_path, allow_pickle=True) as npzfile:
# #                         # Get speeds and fc2 if they exist
# #                         speeds.append(npzfile['speeds'])
# #                         fc2s.append(npzfile['fc2'] if 'fc2' in npzfile.files else None)
# #                 else:
# #                     speeds.append(np.array([np.nan]))
# #                     fc2s.append(None)
# #             results[window_size][tau_val] = {
# #                 'speeds': speeds,   # list of arrays (length=n_animals)
# #                 'fc2': fc2s        # list of arrays (length=n_animals)
# #             }
# #     return results

# # # ---- Example usage ----

# # time_window_range = data.time_window_range
# # tau_range = np.arange(0, data.tau + 1) if True else np.arange(-data.tau, data.tau + 1)  # True if min_tau_zero
# # n_animals = data.n_animals

# # results = load_speed_fc2_all(save_path, time_window_range, tau_range, n_animals)

# # # Now you can, for example:
# # # results[window_size][tau_val]['speeds'][animal_index]  # gives you that animal's speed array for that window/tau
# # # results[window_size][tau_val]['fc2'][animal_index]     # gives you fc2 array

# # print("Shape example (first window, tau=3):")
# # print("Speeds list:", len(results[time_window_range[0]][3]['speeds']))
# # print("FC2 list:", len(results[time_window_range[0]][3]['fc2']))

# # # Convert to array if all lengths match (optional):
# # import numpy as np
# # speeds_matrix = np.stack(results[time_window_range[0]][3]['speeds'])
# # print("Speeds matrix shape:", speeds_matrix.shape)

# # %%

# def aggregate_and_save_speeds_fc2(save_path, window_size, tau_val, n_animals):
#     """
#     Aggregates per-animal speeds and fc2 for the given window_size and tau_val,
#     saves as a single .npz file: speeds_all, fc2_all (object arrays, length=n_animals).
#     """
#     speeds_all = []
#     fc2_all = []
#     out_path_speed = Path(save_path) / f"speed_win{window_size}/speed_all_animals_tau{tau_val}.npz"
#     out_path_fc2 = Path(save_path) / f"speed_win{window_size}/speed_fc_all_animals_tau{tau_val}.npz"
#     # Check if already computed and saved

#     if out_path_fc2.exists():
#         with np.load(out_path_fc2, allow_pickle=True) as npzfile:
#             fc2_all = npzfile['fc2_all']
#     if out_path_speed.exists():
#         with np.load(out_path_speed, allow_pickle=True) as npzfile:
#             speeds_all = npzfile['speeds_all']
#             return
#             # return speeds_all, fc2_all  # Already aggregated, no need to recompute

#     for i in tqdm(range(n_animals)):
#         file_path = Path(save_path) / f"speed_win{window_size}/tau_{tau_val}/animal_{i}.npz"
#         if file_path.exists():
#             with np.load(file_path, allow_pickle=True) as npzfile:
#                 speeds_all.append(npzfile['speeds'])
#                 fc2_all.append(npzfile['fc2'])
#         else:
#             speeds_all.append(np.array([np.nan]))
#             fc2_all.append(np.array([np.nan]))
#     # Save both as object arrays in a single file
#     np.savez_compressed(out_path_speed, speeds_all=np.array(speeds_all, dtype=object))
#     np.savez_compressed(out_path_fc2, fc2_all=np.array(fc2_all, dtype=object))
#     print(f"✓ Saved all speeds and fc2 for window={window_size} tau={tau_val} to {out_path_speed} and {out_path_fc2}")
#     # return speeds_all, fc2_all

# # ---- Example usage ----
# [aggregate_and_save_speeds_fc2(
#     save_path=data.paths['speed'],
#     window_size=ws,
#     tau_val=tau_val,
#     n_animals=data.n_animals
# ) 
#  for tau_val in np.arange(0, data.tau + 1) if True
#  for ws in tqdm(data.time_window_range, desc=f"Time windows {tau_val}") 
#  ]  # True if min_tau_zero

# # %%














# # Agregate and save all speeds and fc2 for all windows and all animals, sparate tau values
# def aggregate_all_speeds_animals(save_path, time_window_range, tau_val, n_animals):
#     """
#     Aggregates all speeds and fc2 for all windows and specific tau values,
#     saves as a single .npz file for each window size.
#     """
#     speeds_all = []
#     speed_filename = f"speed_all_animals_{n_animals}_all_windows_{len(time_window_range)}_tau_{tau_val}.npz"
#     out_path_speed = Path(save_path) / speed_filename
#     # Check if already computed and saved

#     if out_path_speed.exists():
#         print(f"File {out_path_speed} already exists. Loading and returning.")
#         with np.load(out_path_speed, allow_pickle=True) as npzfile:
#             speeds_all = npzfile['speeds_all']
#             return
#             # return speeds_all, fc2_all  # Already aggregated, no need to recompute

#     for ws in tqdm(time_window_range, desc="Loading speeds for all animals"):
#         path_speed = Path(save_path) / f"speed_win{ws}/speed_all_animals_tau{tau_val}.npz"
#         if not path_speed.exists():
#             print(f"WARNING: {path_speed} missing, filling with NaN.")
#             speeds_all.append(np.full((n_animals, 1), np.nan))
#             continue

#         with np.load(path_speed, allow_pickle=True) as npzfile:
#             arr = npzfile['speeds_all']
#             speeds_all.append(arr)

#     # Now, this will work!
#     speeds_all = np.array(speeds_all, dtype=object)

#     # # Save both as object arrays in a single file
#     # np.savez_compressed(out_path_speed, speeds_all=speeds_all)
#     # print(f"✓ Saved all speeds and fc2 for tau={tau_val} to {out_path_speed}")

# # %%        Visualization
# # ---- Example usage ----
# aggregate_all_speeds_animals(
#     save_path=data.paths['speed'],
#     time_window_range=data.time_window_range,
#     tau_val=0, # True if min_tau_zero
#     n_animals=data.n_animals
# )
# #%%
# # #%%# Agregate and save all speeds and fc2 for all windows and all animals, sparate tau values
# # def aggregate_all_fc_animals(save_path, time_window_range, tau_val, n_animals):
# #     """
# #     Aggregates all speeds and fc2 for all windows and specific tau values,
# #     saves as a single .npz file for each window size.
# #     """
    
# #     fc2_all = []
# #     fc2_filename = f"speed_fc2_all_animals_{n_animals}_all_windows_{len(time_window_range)}_tau_{tau_val}.npz"
# #     out_path_fc2 = Path(save_path) / fc2_filename
# #     # Check if already computed and saved

# #     if out_path_fc2.exists():
# #         with np.load(out_path_fc2, allow_pickle=True) as npzfile:
# #             fc2_all = npzfile['fc2_all']
# #             return

# #     for ws in tqdm(time_window_range, desc="Loading speeds and fc2 for all animals"):
# #         # Speed
# #         # file_path = Path(save_path) / f"speed_win{ws}/animal_{i}.npz"
# #         path_fc2 = Path(save_path) / f"speed_win{ws}/speed_fc_all_animals_tau{tau_val}.npz"       
# #         if path_fc2.exists():
# #             with np.load(path_fc2, allow_pickle=True) as npzfile:
# #                 fc2_all.append(npzfile['fc2_all'])
# #         else:
# #             fc2_all.append(np.array([np.nan]))
# #     # Save both as object arrays in a single file
# #     np.savez_compressed(out_path_fc2, fc2_all=np.array(fc2_all, dtype=object))
# #     print(f"✓ Saved all speeds and fc2 for tau={tau_val} to {out_path_fc2}")
# # # %%


# # %%

# %%
