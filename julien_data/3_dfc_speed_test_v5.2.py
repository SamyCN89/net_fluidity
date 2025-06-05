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
from typing import Dict, List, Optional, Tuple
import gc
import tempfile
import shutil

def merge_speed_fc2_files_memory_efficient(data, save_path: Optional[Path] = None, 
                                         window_sizes: Optional[List[int]] = None,
                                         tau_range: Optional[List[int]] = None,
                                         delete_individual_files: bool = False,
                                         chunk_size: int = 10) -> Dict:
    """
    Memory-efficient merge of all speed and FC2 data files.
    Processes files in chunks to avoid loading everything into memory at once.
    
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
    chunk_size : int
        Number of files to process at once (default: 10)
    
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
    
    # First pass: scan for available files
    logger.info("Scanning for available files...")
    available_files = []
    missing_files = []
    has_fc2 = False
    
    for window_size in window_sizes:
        for tau_val in tau_range:
            filename = f"speed_win{window_size}_tau{tau_val}_all_animals_n{n_animals}_regions{regions}.npz"
            file_path = save_path / f"speed_win{window_size}" / filename
            
            if file_path.exists():
                # Quick check for FC2 without loading full file
                try:
                    with np.load(file_path, mmap_mode='r') as f:
                        if 'fc2' in f:
                            has_fc2 = True
                    available_files.append((window_size, tau_val, file_path))
                except:
                    logger.warning(f"Could not check file: {file_path}")
            else:
                missing_files.append((window_size, tau_val))
    
    logger.info(f"Found {len(available_files)} files, missing {len(missing_files)} files")
    
    if not available_files:
        raise ValueError("No valid files found to merge!")
    
    # Prepare output files
    speed_merged_file = save_path / f"merged_speed_all_windows_n{n_animals}_regions{regions}.npz"
    fc2_merged_file = save_path / f"merged_fc2_all_windows_n{n_animals}_regions{regions}.npz" if has_fc2 else None
    
    # Use temporary files to avoid corruption if process fails
    with tempfile.NamedTemporaryFile(delete=False) as speed_temp:
        speed_temp_path = Path(speed_temp.name)
    
    fc2_temp_path = None
    if has_fc2:
        with tempfile.NamedTemporaryFile(delete=False) as fc2_temp:
            fc2_temp_path = Path(fc2_temp.name)
    
    try:
        # Initialize data dictionaries that will be saved
        speed_data_to_save = {
            'window_sizes': np.array(window_sizes),
            'tau_range': np.array(tau_range),
            'n_animals': n_animals,
            'regions': regions,
            'found_combinations': [(w, t) for w, t, _ in available_files],
            'missing_combinations': missing_files,
            'metadata_collection': {}
        }
        
        fc2_data_to_save = {
            'window_sizes': np.array(window_sizes),
            'tau_range': np.array(tau_range),
            'n_animals': n_animals,
            'regions': regions,
            'found_combinations': [(w, t) for w, t, _ in available_files],
            'has_fc2': []
        } if has_fc2 else None
        
        # Process files in chunks
        logger.info(f"Processing files in chunks of {chunk_size}...")
        
        for i in tqdm(range(0, len(available_files), chunk_size), desc="Processing chunks"):
            chunk = available_files[i:i + chunk_size]
            
            # Process current chunk
            for window_size, tau_val, file_path in chunk:
                try:
                    # Load single file
                    with np.load(file_path, allow_pickle=True) as loaded_data:
                        # Add speed data
                        speed_data_to_save[f'median_speeds_win{window_size}_tau{tau_val}'] = loaded_data['median_speeds'].copy()
                        speed_data_to_save[f'speeds_win{window_size}_tau{tau_val}'] = loaded_data['speeds'].copy()
                        
                        # Add metadata
                        speed_data_to_save['metadata_collection'][(window_size, tau_val)] = loaded_data['metadata'].item()
                        
                        # Add FC2 data if available
                        if has_fc2 and fc2_data_to_save is not None and 'fc2' in loaded_data:
                            fc2_data_to_save[f'fc2_win{window_size}_tau{tau_val}'] = loaded_data['fc2'].copy()
                            fc2_data_to_save['has_fc2'].append((window_size, tau_val))
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    continue
            
            # Force garbage collection after each chunk
            gc.collect()
            
            # Save intermediate results every 50 files to free memory
            if i > 0 and i % (chunk_size * 5) == 0:
                logger.info(f"Saving intermediate results at file {i}...")
                
                # Save speed data
                np.savez_compressed(speed_temp_path, **speed_data_to_save)
                
                # Reload to free memory (keeps only metadata)
                with np.load(speed_temp_path, allow_pickle=True) as temp_data:
                    metadata_keys = ['window_sizes', 'tau_range', 'n_animals', 'regions', 
                                   'found_combinations', 'missing_combinations', 'metadata_collection']
                    speed_data_to_save = {k: temp_data[k].copy() if k in temp_data else speed_data_to_save[k] 
                                        for k in metadata_keys}
                
                # Similar for FC2
                if has_fc2 and fc2_data_to_save is not None:
                    np.savez_compressed(fc2_temp_path, **fc2_data_to_save)
                    with np.load(fc2_temp_path, allow_pickle=True) as temp_data:
                        fc2_metadata_keys = ['window_sizes', 'tau_range', 'n_animals', 'regions', 
                                           'found_combinations', 'has_fc2']
                        fc2_data_to_save = {k: temp_data[k].copy() if k in temp_data else fc2_data_to_save[k] 
                                          for k in fc2_metadata_keys}
                
                gc.collect()
        
        # Final save
        logger.info("Saving final merged files...")
        
        # Save speed file
        np.savez_compressed(speed_temp_path, **speed_data_to_save)
        shutil.move(str(speed_temp_path), str(speed_merged_file))
        logger.info(f"✓ Saved merged speed data to: {speed_merged_file}")
        
        # Save FC2 file if needed
        if has_fc2 and fc2_data_to_save is not None and fc2_temp_path is not None:
            np.savez_compressed(fc2_temp_path, **fc2_data_to_save)
            shutil.move(str(fc2_temp_path), str(fc2_merged_file))
            logger.info(f"✓ Saved merged FC2 data to: {fc2_merged_file}")
        
        # Optional: Delete individual files
        if delete_individual_files:
            logger.info("Deleting individual files...")
            deleted_count = 0
            for _, _, file_path in available_files:
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}: {e}")
            logger.info(f"Deleted {deleted_count} individual files")
        
        return {
            'speed_file': speed_merged_file,
            'fc2_file': fc2_merged_file,
            'found_files': len(available_files),
            'missing_files': len(missing_files),
            'has_fc2': has_fc2
        }
        
    finally:
        # Clean up temporary files if they still exist
        for temp_path in [speed_temp_path, fc2_temp_path]:
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass


def load_merged_data_lazy(data, data_type: str = 'speed', window_size: Optional[int] = None, 
                         tau_val: Optional[int] = None) -> Dict:
    """
    Memory-efficient loading of merged data with lazy loading.
    
    Parameters:
    -----------
    data : DFCAnalysis object
        Data analysis object
    data_type : str
        'speed' or 'fc2' to specify which merged file to load
    window_size : int, optional
        If provided, only load data for this window size
    tau_val : int, optional
        If provided, only load data for this tau value
    
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
    
    # Use memory mapping for efficient access
    loaded_data = np.load(file_path, mmap_mode='r', allow_pickle=True)
    
    # Create result dictionary with metadata
    result = {
        'window_sizes': np.array(loaded_data['window_sizes']),
        'tau_range': np.array(loaded_data['tau_range']),
        'n_animals': int(loaded_data['n_animals']),
        'regions': int(loaded_data['regions']),
        'found_combinations': loaded_data['found_combinations'],
        '_file_handle': loaded_data  # Keep file handle open
    }
    
    # If specific window/tau requested, load only that data
    if window_size is not None and tau_val is not None:
        if data_type == 'speed':
            result['median_speeds'] = np.array(loaded_data.get(f'median_speeds_win{window_size}_tau{tau_val}'))
            result['speeds'] = np.array(loaded_data.get(f'speeds_win{window_size}_tau{tau_val}'))
        elif data_type == 'fc2':
            result['fc2'] = np.array(loaded_data.get(f'fc2_win{window_size}_tau{tau_val}'))
    
    # Add lazy loading method
    def get_data(window_size: int, tau_val: int, data_field: str = None):
        """Get data for specific window and tau combination (lazy loading)."""
        if data_type == 'speed':
            if data_field == 'median_speeds' or data_field is None:
                key = f'median_speeds_win{window_size}_tau{tau_val}'
                if key in loaded_data:
                    return np.array(loaded_data[key])
            if data_field == 'speeds':
                key = f'speeds_win{window_size}_tau{tau_val}'
                if key in loaded_data:
                    return np.array(loaded_data[key])
            if data_field is None:
                return {
                    'median_speeds': np.array(loaded_data.get(f'median_speeds_win{window_size}_tau{tau_val}')),
                    'speeds': np.array(loaded_data.get(f'speeds_win{window_size}_tau{tau_val}'))
                }
        elif data_type == 'fc2':
            key = f'fc2_win{window_size}_tau{tau_val}'
            if key in loaded_data:
                return np.array(loaded_data[key])
        return None
    
    result['get_data'] = get_data
    
    # Add method to close file handle
    def close():
        """Close the file handle to free resources."""
        if '_file_handle' in result and hasattr(result['_file_handle'], 'close'):
            result['_file_handle'].close()
    
    result['close'] = close
    
    return result


def estimate_merged_file_size(data, save_path: Optional[Path] = None) -> Dict:
    """
    Estimate the size of merged files before creating them.
    
    Parameters:
    -----------
    data : DFCAnalysis object
        Data analysis object
    save_path : Path, optional
        Directory containing individual files
    
    Returns:
    --------
    Dict : Estimated sizes and memory requirements
    """
    if save_path is None:
        save_path = Path(data.paths['speed'])
    
    total_size_speed = 0
    total_size_fc2 = 0
    file_count = 0
    has_fc2 = False
    
    # Scan all possible files
    for window_size in data.time_window_range:
        for tau_val in range(0, data.tau + 1):
            filename = f"speed_win{window_size}_tau{tau_val}_all_animals_n{data.n_animals}_regions{data.regions}.npz"
            file_path = save_path / f"speed_win{window_size}" / filename
            
            if file_path.exists():
                file_size = file_path.stat().st_size
                total_size_speed += file_size
                
                # Check if FC2 exists
                try:
                    with np.load(file_path, mmap_mode='r') as f:
                        if 'fc2' in f:
                            has_fc2 = True
                            # Estimate FC2 portion (usually similar to speed data size)
                            total_size_fc2 += file_size * 0.8  # Rough estimate
                except:
                    pass
                
                file_count += 1
    
    # Add overhead for metadata and compression
    overhead_factor = 1.2
    
    return {
        'estimated_speed_file_size_gb': (total_size_speed * overhead_factor) / (1024**3),
        'estimated_fc2_file_size_gb': (total_size_fc2 * overhead_factor) / (1024**3) if has_fc2 else 0,
        'total_individual_files': file_count,
        'has_fc2_data': has_fc2,
        'peak_memory_required_gb': (total_size_speed + total_size_fc2) * 2 / (1024**3),  # Conservative estimate
        'recommendation': 'Use chunk_size=5 for large datasets' if total_size_speed > 10 * (1024**3) else 'Default settings should work fine'
    }


# Convenience functions
def quick_merge_efficient(data, chunk_size=10, delete_old=False):
    """Memory-efficient merge function."""
    return merge_speed_fc2_files_memory_efficient(data, chunk_size=chunk_size, delete_individual_files=delete_old)


def quick_load_lazy(data, data_type='speed', window_size=None, tau_val=None):
    """Lazy loading function for merged data."""
    return load_merged_data_lazy(data, data_type, window_size, tau_val)


def check_merge_feasibility(data):
    """Check if merging is feasible given current system resources."""
    import psutil
    
    estimates = estimate_merged_file_size(data)
    available_ram = psutil.virtual_memory().available / (1024**3)
    
    print(f"Estimated merged file sizes:")
    print(f"  Speed file: {estimates['estimated_speed_file_size_gb']:.2f} GB")
    print(f"  FC2 file: {estimates['estimated_fc2_file_size_gb']:.2f} GB")
    print(f"  Peak RAM needed: {estimates['peak_memory_required_gb']:.2f} GB")
    print(f"  Available RAM: {available_ram:.2f} GB")
    print(f"  Recommendation: {estimates['recommendation']}")
    
    if estimates['peak_memory_required_gb'] > available_ram * 0.8:
        print("\n⚠️  WARNING: Merge may require more RAM than available!")
        print("  Consider using a smaller chunk_size or closing other applications.")
    
    return estimates


# Example usage:
if __name__ == "__main__":
    # Check feasibility first
    estimates = check_merge_feasibility(data)
    
    # # Merge with appropriate chunk size
    # if estimates['peak_memory_required_gb'] > 20:
    #     merge_result = quick_merge_efficient(data, chunk_size=5)
    # else:
    #     merge_result = quick_merge_efficient(data, chunk_size=10)
    
    # # Load specific data lazily
    # speed_data = quick_load_lazy(data, 'speed', window_size=50, tau_val=3)
    # median_speeds = speed_data.get_data(50, 3, 'median_speeds')
    # speed_data.close()  # Don't forget to close when done!