#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:26:30 2024

@author: samy
"""
#%%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# import sys
# sys.path.append('../shared_code')

from shared_code.fun_loaddata import extract_hash_numbers
from shared_code.fun_utils import filename_sort_mat, load_matdata, classify_phenotypes, make_combination_masks, make_masks
from shared_code.fun_utils import get_paths
import time
from scipy.io import loadmat
import re
#%%

# =============================================================================
# This code loads the time series data and cognitive data from files
# =============================================================================

# ------------------------Configuration------------------------
# Set the path to the root directory of your dataset
# USE_EXTERNAL_DISK = True
# ROOT = Path('/media/samy/Elements1/Proyectos/LauraHarsan/dataset/julien/') if USE_EXTERNAL_DISK \
#         else Path('/home/samy/Bureau/Proyect/LauraHarsan/dataset/julien/')

paths = get_paths(dataset_name='julien_caillette', 
                  timecourse_folder='time_courses',
                  cognitive_data_file='mice_groups_comp_index.xlsx')

paths['roi'] = paths['timeseries'] / 'all_ROI_coimagine.txt'

# TS_FOLDER = paths['timeseries']
# COG_XLSX = paths['cognitive_data']
# PREPROCESS_DATA = ROOT / 'preprocess_data'
# PREPROCESS_DATA.mkdir(parents=True, exist_ok=True)

#%%



#%%
# -----------------------------------------------------------------------------
def load_mat_timeseries(folder: Path) -> tuple:
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
            print(f"Loaded {fname}: shape {data.shape}")
        except Exception as e:
            print(f"Error loading {fname}: {e}")
    return ts_list, shapes, names

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

def main():
    # Load all time series data
    ts_list, ts_shapes, loaded_files = load_mat_timeseries(paths['timeseries'])
    # Check if all loaded files have the same shape
    if len(set(ts_shapes)) > 1:
        print("Warning: Not all loaded files have the same shape.")
    # Extract mouse IDs from filenames
    ts_ids = extract_mouse_ids(loaded_files)

    # =============================================================================
    # Load cognitive data from .xlsx document
    # =============================================================================
    #Load cognitive data
    cog_data     = pd.read_excel(paths['cog_data'], sheet_name='mice_groups_comp_index')
    cog_data['mouse'] = cog_data['mouse'].astype(str) #Ensure mouse IDs are strings

    region_labels        = np.loadtxt(paths['roi'], dtype=str).tolist()
    region_labels_clean = [label.replace("Both_", "") for label in region_labels]

    matched_ids = [mid for mid in ts_ids if mid in cog_data['mouse'].values]
    # Filter cognitive data to include only matched mouse IDs and sorted by mouse IDs
    cog_data_filtered = cog_data.set_index('mouse').loc[matched_ids].reset_index()

    #List of time series that match the mouse IDs in the cognitive data, preserving the order
    ts_filtered = [ts for ts, id_ in zip(ts_list, ts_ids) if id_ in matched_ids]
    n_animals = len(ts_filtered)
    total_tp, regions = ts_filtered[0].shape

    if len(ts_filtered) != len(cog_data_filtered):
        raise ValueError("Mismatch in time series and cognitive data entries.")

    # Extract group features
    split_grp = cog_data_filtered["grp"].str.split("_", expand=True)
    cog_data_filtered["genotype"] = split_grp[0]
    cog_data_filtered["treatment"] = split_grp[1]
    cog_data_filtered = pd.concat([cog_data_filtered,
                                    pd.get_dummies(split_grp[0]),
                                    pd.get_dummies(split_grp[1])], axis=1)

    print(f"Matched {len(matched_ids)} mice")
    print(f"Region labels loaded: {len(region_labels_clean)}")
    print(f"Filtered cognitive data shape: {cog_data_filtered.shape}")

    #add the number of time points of each animal
    cog_data_filtered['n_timepoints'] = [ts.shape[0] for ts in ts_filtered]

    # Save processed data
    if all(ts.shape == ts_filtered[0].shape for ts in ts_filtered):
        ts_array = np.stack(ts_filtered)
        np.savez(paths['sorted'] / "ts_filtered.npz", 
                 ts=ts_array,
                 n_animals=n_animals,
                 total_tp=total_tp,
                 regions=regions,
                 anat_labels = region_labels_clean)
    #              ts=ts,  
    #      n_animals=n_animals, 
    # total_tp=total_tp, 
    # regions=regions, 
    # is_2month_old=is_2month_old,
    # anat_labels=anat_labels,
        print(f"Saved: ts_filtered.npz with shape {ts_array.shape}")
    else:
        np.savez(paths['sorted'] / "ts_filtered_unstacked.npz", 
                                ts=np.array(ts_filtered, dtype=object),
                                n_animals=n_animals,
                                total_tp=total_tp,
                                regions=regions,
                                anat_labels = region_labels_clean)
        print("Saved: ts_filtered_unstacked.npz")

    cog_data_filtered.to_csv(paths['sorted'] / "cog_data_filtered.csv", index=False)
    print("Saved: cog_data_filtered.csv")
    return ts_filtered

if __name__ == "__main__":
    ts_filtered = main()

# %%
