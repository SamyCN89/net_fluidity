#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 15:56:50 2024

@author: samy
"""

# =============================================================================
#  Functions to load data of Laura Harsan and Ines a
# Samy Castro March 2024
# =============================================================================

from pathlib import Path
import numpy as np
import os
from scipy.io import loadmat
from joblib import Parallel, delayed, parallel_backend
import re
import logging
from typing import Any, Union
import pickle
# from .fun_dfcspeed import compute4window_new

#%%
# -------- Utility functions for loading and saving raw data --------

def load_mat_timeseries(folder: Path, verbose: bool = False) -> tuple:
    """
    Load all time series from .mat files in a given folder, regardless of shape consistency.

    Parameters
    ----------
    folder : Path
        Path to the folder containing .mat files. Each file is expected to contain a variable 'tc' representing the time series.

    ts_list : list of np.ndarray
        List of loaded time series arrays, each typically of shape [regions, timepoints].
    shapes : list of tuple
        List of shapes corresponding to each loaded array.
    names : list of str
        List of filenames corresponding to each loaded time series.

    Notes
    -----
    Files are loaded in reverse alphabetical order. If a file cannot be loaded or does not contain 'tc', an error message is printed and the file is skipped.
    """
    mat_files = sorted([f.name for f in folder.iterdir() if f.is_file()], reverse=True)
    ts_list, shapes, names = [], [], []
    for fname in mat_files:
        try:
            data = loadmat(folder / fname)['tc']
            ts_list.append(data)
            shapes.append(data.shape)
            names.append(fname)
            if verbose:
                # Print the shape of each loaded time series
                print(f"Loaded {fname}: shape {data.shape}")
        except Exception as e:
            print(f"Error loading {fname}: {e}")
    return ts_list, shapes, names
# Load julien time series data from .mat files

def extract_mouse_ids(filenames: list) -> list:
    """Extract mouse IDs from filenames.

    Args:
        filenames (list): List of filename strings.

    Returns:
        list: List of extracted mouse IDs.
    """
    cleaned = []
    for name in filenames:
        match = re.match(r"tc_Coimagine_(.+?)_(\d+)_\d+_seeds\.mat", name)
        if match:
            cleaned.append(f"{match.group(1)}_{match.group(2)}")
        else:
            print(f"Warning: No match for {name}")
    return cleaned

# =============================================================================
# Save data functions
# =============================================================================
def save_pickle(obj: Any, path: Union[str, Path]) -> None:
    """Save a Python object to a file using pickle."""
    path = Path(path)  # always use Path object
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    logging.getLogger(__name__).info(f"Saved pickle: {path}")

def load_pickle(path):
    """Load a Python object from a pickle file."""
    with open(path, 'rb') as f:
        logging.getLogger(__name__).info(f"Loaded pickle: {path}")
        return pickle.load(f)

import numpy as np

def load_fc2_npz(path: Union[str, Path]) -> Any:
    """Load fc2 results from a .npz file."""
    path = Path(path)
    try:
        with np.load(path, allow_pickle=True) as arr:
            if 'fc2' in arr:
                data = arr['fc2']
                logging.getLogger(__name__).info(f"Loaded fc2 results from: {path}")
                return data
            else:
                logging.getLogger(__name__).warning(f"'fc2' key not found in {path}. Available keys: {arr.files}")
                return None
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to load fc2 results from {path}: {e}")
        return None

#Load preprocessed data from .npz files
def load_npz_dict(path_to_npz: Path) -> dict:
    """
    Load all arrays (and scalars) from an .npz file into a Python dict.

    Parameters
    ----------
    path_to_npz : Path
        Path to the .npz file.

    Returns
    -------
    dict
        A mapping from each key in the .npz to its value. 0-dim arrays
        are converted to native Python scalars via .item().
    """
    data = np.load(path_to_npz, allow_pickle=True)
    out = {}
    for key in data.files:
        # print(f"Loading key: {key}")
        # Get the array for the current key')
        arr = data[key]
        # Convert 0-dim arrays to scalars
        if isinstance(arr, np.ndarray) and arr.shape == ():
            out[key] = arr.item()
        else:
            out[key] = arr
    data.close()
    return out
#%%
def make_file_path(save_path, prefix, window_size, lag, n_animals, nodes):
    """
    Generate a consistent save path for cached files.

    Args:
        save_path (str or Path): Directory to save the file.
        prefix (str): File type prefix, e.g., 'mc' or 'dfc'.
        window_size (int): Window size parameter.
        lag (int): Lag parameter.
        n_animals (int): Number of animals.
        nodes (int): Number of regions/nodes.

    Returns:
        Path or None: Full file path for saving, or None if save_path not given.
    """
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        if prefix== 'speed':
            return save_path / f"{prefix}_window_size={window_size}_tau={lag}_animals={n_animals}_regions={nodes}.npz"
        else:
            return save_path / f"{prefix}_window_size={window_size}_lag={lag}_animals={n_animals}_regions={nodes}.npz"
    return None

def load_from_cache(file_path, key, logger=None, label=None):
    """
    Load a value from an npz cache file by key.

    Args:
        file_path (Path or str): Path to the .npz file.
        key (str): Key to extract from the npz file (e.g., 'mc', 'dfc_stream').
        logger (logging.Logger, optional): Logger for info messages.
        label (str, optional): Label to use in printed/logged messages.

    Returns:
        The value from the npz file for the specified key, or None if not found.
    """
    if file_path and Path(file_path).exists():
        msg = f"Loading {label or key} from: {file_path}"
        if logger:
            logger.info(msg)
        print(msg)
        try:
            data = np.load(file_path, allow_pickle=True)
            if key in data:
                return data[key]
            else:
                print(f"Key '{key}' not found in cache file: {file_path}")
        except Exception as e:
            print(f"Failed to load cached {label or key} (reason: {e}). Recomputing...")
    return None
        

def save2disk(save_path, prefix, **data):
    """
    Generate a save path and save data as a compressed npz file.

    Args:
        save_path (str or Path): Directory to save the file.
        prefix (str): Prefix for the file name (e.g., 'dfc', 'mc').
        **data: Data to save (e.g., dfc_stream=dfc_stream).

    Returns:
        Path or None: The path where data was saved, or None if save_path not given.
    """
    print('here')
    if save_path:
        # file_path = save_path / f"{prefix}_window_size={window_size}_lag={lag}_animals={n_animals}_regions={nodes}.npz"
        print(f"Saving {prefix} stream to: {save_path}")
        np.savez_compressed(save_path, **data, allow_pickle=True)
        return save_path
    return None

#%%
# Check if the prefix files exist and their sizes: using the check_and_rerun_missing_files function and get_missing_files function
def get_missing_files(paths, prefix, time_window_range, lag, n_animals, roi, size_threshold=1_000_000):
    """
    Check if the prefix files exist for all specified window sizes after computing.
    If a file is empty/corrupt, it will be added to the *missing_files* list.
    Args:
        paths (dict): Dictionary containing paths for saving files.
        prefix (str): Prefix for the file names. Now implemented as 'dfc' and 'mc'.
        time_window_range (list): List of time window sizes to check.
        lag (int): Lag parameter used in the computation.
        n_animals (int): Number of animals in the dataset.
        roi (list): List of regions of interest.
        size_threshold (int): Minimum file size threshold to consider a file valid.
    Returns:
        missing_files (list): List of time window sizes for which files are missing or invalid.     
    """
    missing_files = []
    for ws in time_window_range:
        # 1. Check the existence of the file for each window size
        file_path = make_file_path(paths, prefix, ws, lag, n_animals, roi)
        if not file_path.exists():
            missing_files.append(ws)
        # 2. Check if the file is empty or corrupt (less than 1 MB)
        else:
            if file_path.stat().st_size < size_threshold:  # This will raise an error if the file is not valid
                # Remove the file if it's empty or corrupt
                print(f"File {file_path} exists but is empty or corrupt. Removing it.")
                file_path.unlink(missing_ok=True)
                missing_files.append(ws)
    return missing_files

def check_and_rerun_missing_files(paths, prefix, time_window_range, lag, n_animals, roi):
    """
    Check for missing prefix files and compute them if necessary.
    Args:
        paths (dict): Dictionary containing paths for different data types.
        prefix (str): Prefix of the files to check. 'dfc' for DFC stream files, 'mc' for meta-connectivity files.
        time_window_range (np.ndarray): Array of time window sizes to check.
        lag (int): Lag parameter for DFC computation.
        n_animals (int): Number of animals in the dataset.
        roi (str): Region of interest for DFC computation.
    Returns:
        missing_files (list): List of time window sizes for which files are missing or invalid.
    """
    missing_files = get_missing_files(paths, prefix, time_window_range, lag, n_animals, roi)
    if not missing_files:
        print(f"All {prefix} files already exist.")
    else:
        print(f"Missing {prefix} files for window sizes:", missing_files)
        Parallel(n_jobs=min(PROCESSORS, len(missing_files)))(
            delayed(compute4window_new)(ws, prefix) for ws in missing_files
        )
    return missing_files


#%%

#%%
def filename_sort_mat(folder_path):
    """Read and sort MATLAB file names in a given folder path."""
    files_name      = np.sort(os.listdir(folder_path))
    return files_name


def extract_hash_numbers(filenames, prefix='lot3_'):
    """Extract hash numbers from filenames based on a given prefix."""
    hash_numbers    = [int(name.split(prefix)[-1][:4]) for name in filenames if prefix in name]
    return hash_numbers

def load_matdata(folder_data, specific_folder, files_name):
    ts_list = []
    hash_dir        = Path(folder_data) / specific_folder

    for idx,file_name in enumerate(files_name):
        file_path       = hash_dir / file_name
        
        try:
            data = loadmat(file_path)['tc']
            ts_list.append(data)
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
    
    
    # Check if the first dimension is consistent
    first_dim_size = ts_list[0].shape[0]
    if all(data.shape[0] == first_dim_size for data in ts_list):
        # Convert the list to a NumPy array
        ts_array = np.array(ts_list)
        return ts_array
    else:
        print("Error: Inconsistent shapes along the first dimension.")
