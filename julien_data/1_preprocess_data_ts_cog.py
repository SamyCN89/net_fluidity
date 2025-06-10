#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:26:30 2024

@author: samy
"""
#%%
from pathlib import Path
import numpy as np
import time
import pandas as pd
import pickle
from scipy.io import loadmat
from collections import defaultdict

from shared_code.fun_loaddata import extract_hash_numbers, load_mat_timeseries, extract_mouse_ids
from shared_code.fun_utils import filename_sort_mat, load_matdata, classify_phenotypes, make_combination_masks, make_masks
from shared_code.fun_paths import get_paths
import matplotlib.pyplot as plt
#%%

# =============================================================================
# This code loads the time series data and cognitive data from files
# =============================================================================

# ------------------------Configuration------------------------

paths = get_paths(dataset_name='julien_caillette', 
                  timecourse_folder='time_courses',
                  cognitive_data_file='mice_groups_comp_index.xlsx',
                  anat_labels_file='all_ROI_coimagine.txt',)

# %%
# -----------------------------------------------------------------------------

def main(filter_mode="exclude_shortest"):
    """    Main function to load and preprocess time series and cognitive data.
    Args:
        filter_mode (str): Mode to filter time series data. Options are:
            - "exclude_shortest": Exclude time series shorter than the shortest one.
            - "truncate": Truncate all time series to the length of the shortest one.
            - "none": Load all time series without filtering.
    Returns:
        ts_filtered (list): List of filtered time series data.
        cog_data_filtered (DataFrame): Filtered cognitive data.
    """
    # Load all time series data
    ts_list, ts_shapes, loaded_files = load_mat_timeseries(paths['timeseries'])
    # Extract mouse IDs from filenames
    ts_ids = extract_mouse_ids(loaded_files)

    # Check if all loaded files have the same shape
    if len(set(ts_shapes)) > 1:
        print("Warning: Not all loaded files have the same shape. \n"
              "They are:")
        #Print only the filename and shape of the smallest time series
        min_shape = min(ts_shapes)
        for file, shape in zip(loaded_files, ts_shapes):
            if shape == min_shape:
                print(f"{file}: {shape}")

    # # Group all time series by shape
    min_timepoints = min(ts.shape[0] for ts in ts_list)

    if filter_mode == "exclude_shortest":
        filtered = [(ts, id_) for ts, id_ in zip(ts_list, ts_ids) if ts.shape[0] > min_timepoints]
        ts_list = [ts for ts, id_ in filtered]
        ts_ids  = [id_ for ts, id_ in filtered]
        print(f"Loaded {len(ts_list)} time series with more than {min_timepoints} time points each.")
    elif filter_mode == "truncate":
        ts_list = [ts[:min_timepoints, :] for ts in ts_list]
        print(f"Truncated all time series to {min_timepoints} time points each.")
    else:
        print(f"Loaded {len(ts_list)} time series with varying time points.")
            
    # =========================================
    # Load cognitive data from .xlsx document
    # =============================================================================
    #Load cognitive data
    cog_data     = pd.read_excel(paths['cog_data'], sheet_name='mice_groups_comp_index')
    cog_data['mouse'] = cog_data['mouse'].astype(str) #Ensure mouse IDs are strings

    #Region labels are loaded from a text file
    region_labels        = np.loadtxt(paths['labels'], dtype=str).tolist()
    region_labels_clean = [label.replace("Both_", "") for label in region_labels]

    # =============================================================================
    # Filter time series data to include only those with mouse IDs present in cognitive data   
    # =============================================================================
    matched_ids = [mid for mid in ts_ids 
                   if mid in cog_data['mouse'].values]

    # Filter cognitive data to include only matched mouse IDs and sorted by mouse IDs
    cog_data_filtered = cog_data.set_index('mouse').loc[matched_ids].reset_index()

    #List of time series that match the mouse IDs in the cognitive data, preserving the order
    ts_filtered = [ts for ts, id_ in zip(ts_list, ts_ids) if id_ in matched_ids]

    # Print excluded mice
    excluded_ts_ids = [mid for mid in ts_ids if mid not in cog_data['mouse'].values]
    excluded_cog_ids = [mid for mid in cog_data['mouse'].astype(str).values if mid not in ts_ids]

    print("Mice with time series but NO cognitive data:", excluded_ts_ids, (set(ts_ids) - set(cog_data['mouse'])))
    print("Mice with cognitive data but NO time series:", excluded_cog_ids, (set(cog_data['mouse']) - set(ts_ids)))

    # Check if the number of time series matches the number of cognitive data entries
    if len(ts_filtered) != len(cog_data_filtered):
        raise ValueError("Mismatch in time series and cognitive data entries.")

    # Animals and cognitive data paramaters    
    n_animals = len(ts_filtered)
    total_tr, regions = ts_filtered[0].shape

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

    # Metadata preparation
    metadata = cog_data_filtered.copy()
    metadata['ts_file'] = [f"ts_{id_}.npy" for id_ in matched_ids]  # For example
    total_tr = np.unique([ts.shape[0] for ts in ts_filtered])[-1]

    # Optionally, save region labels and other info
    metadata_dict = {
        'mouse_metadata': metadata,
        'region_labels': region_labels_clean,
        'n_animals': n_animals,
        'regions': regions,
        'total_tr': int(total_tr),
        'anat_labels': region_labels_clean,
        'filter_mode': filter_mode,
    }

    # Save metadata to a pickle file
    with open(paths['preprocessed'] / f"metadata_animals_{n_animals}_regions_{regions}_tr_{total_tr}.pkl", "wb") as f:   
        pickle.dump(metadata_dict, f)

    # Save processed data
    if all(ts.shape == ts_filtered[0].shape for ts in ts_filtered):
        ts_array = np.stack(ts_filtered)
        np.savez(paths['preprocessed'] / f"ts_filtered_animals_{n_animals}_regions_{regions}_tr_{total_tr}.npz", 
                 ts=ts_array,
                 n_animals=n_animals,
                 total_tr=total_tr,
                 regions=regions,
                 anat_labels = region_labels_clean)
        print(f"Saved: ts_filtered_animals_{n_animals}_regions_{regions}_tr_{total_tr}.npz with shape {ts_array.shape}")
    else:
        np.savez(paths['preprocessed'] / f"ts_filtered_animals_{n_animals}_regions_{regions}_tr_{total_tr}.npz", 
                                ts=np.array(ts_filtered, dtype=object),
                                n_animals=n_animals,
                                total_tr=total_tr,
                                regions=regions,
                                anat_labels = region_labels_clean)
        print(f"Saved: ts_filtered_animals_{n_animals}_regions_{regions}_tr_{total_tr}.npz")

    cog_data_filtered.to_csv(paths['preprocessed'] / f"cog_data_filtered_animals_{n_animals}_regions_{regions}_tr_{total_tr}.csv", index=False)
    print(f"Saved: cog_data_filtered_animals_{n_animals}_regions_{regions}_tr_{total_tr}.csv")
    return ts_filtered, cog_data_filtered, metadata_dict

if __name__ == "__main__":
    # ts_filtered, cog_data_filtered, metadata_dict = main(filter_mode=None)
    ts_filtered, cog_data_filtered, metadata_dict = main(filter_mode='exclude_shortest')
