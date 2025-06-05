#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 14:42:38 2023

@author: samy
"""

#%%
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
data = DFCAnalysis()

#Preprocessed data
data.get_metadata()
data.get_ts_preprocessed()
data.get_cogdata_preprocessed()
data.get_temporal_parameters()

import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
import gc

def merge_speed_fc2_files(data, save_path: Optional[Path] = None, 
                         window_sizes: Optional[List[int]] = None,
                         tau_range: Optional[List[int]] = None,
                         delete_individual_files: bool = False) -> Dict:
    """
    Merge all speed and FC2 data from individual window/tau files into two consolidated files.
    
    Parameters:
    -----------
    data : DFCAnalysis object
        Data analysis object with paths and metadata
    save_path : Path, optional
        Directory to save merged files. If None, uses data.paths['speed']
    window_sizes : List[int], optional
        List of window sizes to merge. If None, uses data.time_window_range
    tau_range : List[int], optional
        List of tau values to merge. If None, uses range(0, data.tau + 1)
    delete_individual_files : bool
        Whether to delete individual files after merging (default: False)
    
    Returns:
    --------
    Dict : Information about merged files
    """
    logger = logging.getLogger(__name__)
    
    # Set defaults
    if save_path is None:
        save_path = Path(data.paths['speed'])
    if window_sizes is None:
        window_sizes = data.time_window_range
    if tau_range is None:
        tau_range = list(range(0, data.tau + 1))
    
    # Get metadata
    n_animals = data.n_animals
    regions = data.regions
    
    # Initialize storage for all data
    all_speed_data = {}  # Will store as {(window, tau): {'median_speeds': ..., 'speeds': ...}}
    all_fc2_data = {}    # Will store as {(window, tau): fc2_array}
    metadata_collection = {}
    
    # Track what we find
    found_files = []
    missing_files = []
    has_fc2 = False
    
    logger.info(f"Starting merge process for {len(window_sizes)} windows and {len(tau_range)} tau values")
    
    # Collect all data
    for window_size in tqdm(window_sizes, desc="Collecting data from windows"):
        for tau_val in tau_range:
            # Construct filename
            filename = f"speed_win{window_size}_tau{tau_val}_all_animals_n{n_animals}_regions{regions}.npz"
            file_path = save_path / f"speed_win{window_size}" / filename
            
            if not file_path.exists():
                missing_files.append((window_size, tau_val))
                logger.warning(f"Missing file: {file_path}")
                continue
            
            try:
                # Load data
                loaded_data = np.load(file_path, allow_pickle=True)
                
                # Store speed data
                all_speed_data[(window_size, tau_val)] = {
                    'median_speeds': loaded_data['median_speeds'],
                    'speeds': loaded_data['speeds']
                }
                
                # Store FC2 data if available
                if 'fc2' in loaded_data:
                    all_fc2_data[(window_size, tau_val)] = loaded_data['fc2']
                    has_fc2 = True
                
                # Store metadata
                metadata_collection[(window_size, tau_val)] = loaded_data['metadata'].item()
                
                found_files.append((window_size, tau_val))
                loaded_data.close()
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
    
    logger.info(f"Found {len(found_files)} files, missing {len(missing_files)} files")
    
    if not found_files:
        raise ValueError("No valid files found to merge!")
    
    # Create merged speed file
    logger.info("Creating merged speed file...")
    speed_merged_file = save_path / f"merged_speed_all_windows_n{n_animals}_regions{regions}.npz"
    
    # Prepare speed data for saving
    speed_save_data = {
        'window_sizes': np.array(window_sizes),
        'tau_range': np.array(tau_range),
        'n_animals': n_animals,
        'regions': regions,
        'found_combinations': found_files,
        'missing_combinations': missing_files,
        'metadata_collection': metadata_collection
    }
    
    # Add speed data with structured keys
    for (window, tau), speed_data in all_speed_data.items():
        speed_save_data[f'median_speeds_win{window}_tau{tau}'] = speed_data['median_speeds']
        speed_save_data[f'speeds_win{window}_tau{tau}'] = speed_data['speeds']
    
    # Save merged speed file
    np.savez_compressed(speed_merged_file, **speed_save_data)
    logger.info(f"✓ Saved merged speed data to: {speed_merged_file}")
    
    # Create merged FC2 file if FC2 data exists
    fc2_merged_file = None
    if has_fc2 and all_fc2_data:
        logger.info("Creating merged FC2 file...")
        fc2_merged_file = save_path / f"merged_fc2_all_windows_n{n_animals}_regions{regions}.npz"
        
        # Prepare FC2 data for saving
        fc2_save_data = {
            'window_sizes': np.array(window_sizes),
            'tau_range': np.array(tau_range),
            'n_animals': n_animals,
            'regions': regions,
            'found_combinations': found_files,
            'has_fc2': [(w, t) for (w, t) in found_files if (w, t) in all_fc2_data]
        }
        
        # Add FC2 data with structured keys
        for (window, tau), fc2_data in all_fc2_data.items():
            fc2_save_data[f'fc2_win{window}_tau{tau}'] = fc2_data
        
        # Save merged FC2 file
        np.savez_compressed(fc2_merged_file, **fc2_save_data)
        logger.info(f"✓ Saved merged FC2 data to: {fc2_merged_file}")
    else:
        logger.info("No FC2 data found to merge")
    
    # Optional: Delete individual files
    if delete_individual_files:
        logger.info("Deleting individual files...")
        deleted_count = 0
        for window_size, tau_val in found_files:
            filename = f"speed_win{window_size}_tau{tau_val}_all_animals_n{n_animals}_regions{regions}.npz"
            file_path = save_path / f"speed_win{window_size}" / filename
            try:
                file_path.unlink()
                deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to delete {file_path}: {e}")
        logger.info(f"Deleted {deleted_count} individual files")
    
    # Clean up memory
    del all_speed_data, all_fc2_data
    gc.collect()
    
    return {
        'speed_file': speed_merged_file,
        'fc2_file': fc2_merged_file,
        'found_files': len(found_files),
        'missing_files': len(missing_files),
        'has_fc2': has_fc2
    }


def load_merged_data(data, data_type: str = 'speed') -> Dict:
    """
    Load merged speed or FC2 data.
    
    Parameters:
    -----------
    data : DFCAnalysis object
        Data analysis object
    data_type : str
        'speed' or 'fc2' to specify which merged file to load
    
    Returns:
    --------
    Dict : Loaded data with utility functions
    """
    save_path = Path(data.paths['speed'])
    n_animals = data.n_animals
    regions = data.regions
    
    if data_type == 'speed':
        file_path = save_path / f"merged_speed_all_windows_n{n_animals}_regions{regions}.npz"
    elif data_type == 'fc2':
        file_path = save_path / f"merged_fc2_all_windows_n{n_animals}_regions{regions}.npz"
    else:
        raise ValueError(f"Unknown data_type: {data_type}. Use 'speed' or 'fc2'")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Merged {data_type} file not found: {file_path}")
    
    # Load data
    loaded_data = np.load(file_path, allow_pickle=True)
    
    # Create result dictionary with metadata
    result = {
        'window_sizes': loaded_data['window_sizes'],
        'tau_range': loaded_data['tau_range'],
        'n_animals': loaded_data['n_animals'].item(),
        'regions': loaded_data['regions'].item(),
        'found_combinations': loaded_data['found_combinations'],
        '_raw_data': dict(loaded_data)  # Keep raw data for direct access
    }
    
    # Add convenience method to extract specific window/tau data
    def get_data(window_size: int, tau_val: int, data_field: str = None):
        """Get data for specific window and tau combination."""
        if data_type == 'speed':
            if data_field == 'median_speeds' or data_field is None:
                key = f'median_speeds_win{window_size}_tau{tau_val}'
                if key in loaded_data:
                    return loaded_data[key]
            if data_field == 'speeds':
                key = f'speeds_win{window_size}_tau{tau_val}'
                if key in loaded_data:
                    return loaded_data[key]
            if data_field is None:
                # Return both if no specific field requested
                return {
                    'median_speeds': loaded_data.get(f'median_speeds_win{window_size}_tau{tau_val}'),
                    'speeds': loaded_data.get(f'speeds_win{window_size}_tau{tau_val}')
                }
        elif data_type == 'fc2':
            key = f'fc2_win{window_size}_tau{tau_val}'
            if key in loaded_data:
                return loaded_data[key]
        
        return None
    
    # Add method to result
    result['get_data'] = get_data
    
    # Add method to list available combinations
    def list_available():
        """List all available (window, tau) combinations."""
        available = []
        for key in loaded_data.keys():
            if data_type == 'speed' and key.startswith('median_speeds_win'):
                parts = key.split('_')
                window = int(parts[1][3:])  # Remove 'win' prefix
                tau = int(parts[2][3:])      # Remove 'tau' prefix
                available.append((window, tau))
            elif data_type == 'fc2' and key.startswith('fc2_win'):
                parts = key.split('_')
                window = int(parts[1][3:])
                tau = int(parts[2][3:])
                available.append((window, tau))
        return sorted(list(set(available)))
    
    result['list_available'] = list_available
    
    loaded_data.close()
    return result


# Example usage functions
def quick_merge(data, delete_old=False):
    """Quick merge function for interactive use."""
    return merge_speed_fc2_files(data, delete_individual_files=delete_old)


def quick_load_merged(data, data_type='speed'):
    """Quick load function for merged data."""
    return load_merged_data(data, data_type)


# Usage examples:
if __name__ == "__main__":
    # Assuming 'data' is your DFCAnalysis object
    
    # 1. Merge all files
    merge_result = quick_merge(data, delete_old=False)
    print(f"Merged {merge_result['found_files']} files")
    print(f"Speed file: {merge_result['speed_file']}")
    print(f"FC2 file: {merge_result['fc2_file']}")
    
    # 2. Load merged speed data
    speed_data = quick_load_merged(data, 'speed')
    print(f"Available combinations: {len(speed_data.list_available())}")
    
    # Get specific data
    median_speeds_50_3 = speed_data.get_data(window_size=50, tau_val=3, data_field='median_speeds')
    print(f"Median speeds for window=50, tau=3: shape={median_speeds_50_3.shape}")
    
    # 3. Load merged FC2 data (if available)
    try:
        fc2_data = quick_load_merged(data, 'fc2')
        fc2_50_3 = fc2_data.get_data(window_size=50, tau_val=3)
        print(f"FC2 for window=50, tau=3: shape={fc2_50_3.shape}")
    except FileNotFoundError:
        print("No FC2 data file found")