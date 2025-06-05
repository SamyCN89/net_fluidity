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

# #Run the code with the refactored function

# processors = -1
# save_path = data.paths['speed']  # Directory to save results
# prefix= 'speed'  # Prefix for the DFC speed results


# min_tau_zero =True
# tau_range = np.arange(0, data.tau + 1) if min_tau_zero else np.arange(-data.tau, data.tau + 1)


import numpy as np
import logging
import gc
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import Dict, List, Tuple, Optional, Union
import psutil
import warnings
#%%
class DFCSpeedAnalyzer:
    """
    Optimized DFC Speed Analysis with proper memory management and data handling.
    """
    
    def __init__(self, save_path: Path, max_memory_gb: float = None):
        """
        Initialize the DFC Speed Analyzer.
        
        Parameters:
        -----------
        save_path : Path
            Directory to save results
        max_memory_gb : float, optional
            Maximum memory usage in GB. If None, uses 80% of available RAM.
        """
        self.save_path = Path(save_path)
        self.save_path.mkdir(exist_ok=True, parents=True)
        
        # Memory management
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        self.max_memory_gb = max_memory_gb or (available_memory * 0.8)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def estimate_memory_usage(self, dfc_shape: Tuple, n_animals: int, 
                            return_fc2: bool = False) -> float:
        """
        Estimate memory usage in GB for processing.
        
        Parameters:
        -----------
        dfc_shape : tuple
            Shape of DFC data - can be (time_points, n_regions, n_regions) or (time_points, n_connections)
        n_animals : int
            Number of animals
        return_fc2 : bool
            Whether FC2 will be computed and stored
        
        Returns:
        --------
        float : Estimated memory usage in GB
        """
        if len(dfc_shape) == 3:
            # Full connectivity matrix format: (time_points, n_regions, n_regions)
            time_points, n_regions, _ = dfc_shape
            n_connections = n_regions * n_regions
        elif len(dfc_shape) == 2:
            # Vectorized format: (time_points, n_connections)
            time_points, n_connections = dfc_shape
            n_regions = int(np.sqrt(n_connections))  # Approximate
        else:
            raise ValueError(f"Unexpected DFC shape: {dfc_shape}")
        
        # Base DFC data (float64 = 8 bytes)
        dfc_memory = n_animals * time_points * n_connections * 8
        
        # Speed results (typically much smaller - one value per time point pair)
        speed_memory = n_animals * max(1, time_points - 1) * 8  # rough estimate
        
        # FC2 if requested (similar to DFC)
        fc2_memory = dfc_memory if return_fc2 else 0
        
        # Add 50% buffer for intermediate calculations
        total_bytes = (dfc_memory + speed_memory + fc2_memory) * 1.5
        
        return total_bytes / (1024**3)  # Convert to GB

    def calculate_optimal_batch_size(self, dfc_shape: Tuple, n_animals: int, 
                                   return_fc2: bool = False) -> int:
        """
        Calculate optimal batch size based on memory constraints.
        """
        try:
            single_animal_memory = self.estimate_memory_usage(
                dfc_shape, 1, return_fc2
            )
            
            if single_animal_memory > self.max_memory_gb:
                warnings.warn(
                    f"Single animal requires {single_animal_memory:.2f}GB, "
                    f"but limit is {self.max_memory_gb:.2f}GB"
                )
                return 1
            
            max_batch_size = int(self.max_memory_gb / single_animal_memory)
            optimal_batch_size = min(max_batch_size, n_animals, 32)  # Cap at 32 for practical reasons
            
            return max(1, optimal_batch_size)  # Ensure at least 1
            
        except Exception as e:
            self.logger.warning(f"Memory estimation failed: {e}. Using conservative batch size.")
            # Use conservative approach: start with small batch and let system handle it
            return min(4, n_animals)

    def _process_all_animals_combined(self, dfc_array: np.ndarray, window_size: int, 
                                    tau_val: int, effective_vstep: int, method: str, 
                                    return_fc2: bool, out_dir: Path, nodes: int, 
                                    n_animals: int, load_cache: bool = True) -> Dict:
        """
        Process ALL animals and save results in ONE combined file per tau/window.
        
        Parameters:
        -----------
        dfc_array : np.ndarray
            DFC data for all animals [n_animals, time_points, n_regions, n_regions]
        window_size : int
            Window size for this analysis
        tau_val : int
            Tau value for this analysis
        effective_vstep : int
            Effective step size for DFC speed calculation
        method : str
            Method for DFC speed calculation
        return_fc2 : bool
            Whether to compute and return FC2
        out_dir : Path
            Output directory
        nodes : int
            Number of brain regions
        n_animals : int
            Number of animals
        load_cache : bool
            Whether to check for existing files
        
        Returns:
        --------
        Dict : Processing results and statistics
        """
        from shared_code.fun_dfcspeed import dfc_speed
        
        # COMBINED FILE: All animals in one file per tau/window
        combined_file = out_dir / f"speed_win{window_size}_tau{tau_val}_all_animals_n{n_animals}_regions{nodes}.npz"
        
        # Check cache - if file exists and load_cache is True, skip processing
        if load_cache and combined_file.exists():
            self.logger.info(f"Found cached results for tau={tau_val} at {combined_file}")
            return {
                'processed': n_animals,
                'errors': 0,
                'cached': n_animals,
                'total': n_animals,
                'status': 'loaded_from_cache'
            }
        
        # Initialize lists to store results for ALL animals
        all_median_speeds = []
        all_speeds = []
        all_fc2 = [] if return_fc2 else None
        processing_errors = {}
        
        self.logger.info(f"Processing {n_animals} animals for tau={tau_val}")
        
        # Process each animal
        for i in tqdm(range(n_animals), desc=f"Animals (tau={tau_val})"):
            try:
                # Get animal data
                animal_dfc = dfc_array[i]
                
                # Validate data
                if animal_dfc is None or animal_dfc.size == 0:
                    self.logger.warning(f"Empty DFC data for animal {i}")
                    # Add NaN placeholders
                    all_median_speeds.append(np.nan)
                    all_speeds.append(np.array([np.nan]))
                    if return_fc2:
                        all_fc2.append(np.array([np.nan]))
                    processing_errors[i] = "Empty DFC data"
                    continue
                
                # Compute DFC speed
                if return_fc2:
                    result = dfc_speed(
                        animal_dfc, 
                        vstep=effective_vstep, 
                        method=method, 
                        return_fc2=True
                    )
                    if len(result) == 3:
                        median_speed, speeds, fc2 = result
                        all_median_speeds.append(median_speed)
                        all_speeds.append(speeds)
                        all_fc2.append(fc2)
                    else:
                        raise ValueError(f"Unexpected return format from dfc_speed")
                else:
                    result = dfc_speed(
                        animal_dfc, 
                        vstep=effective_vstep, 
                        method=method, 
                        return_fc2=False
                    )
                    if len(result) == 2:
                        median_speed, speeds = result
                        all_median_speeds.append(median_speed)
                        all_speeds.append(speeds)
                    else:
                        raise ValueError(f"Unexpected return format from dfc_speed")
                
            except Exception as e:
                self.logger.error(f"Error processing animal {i} (tau={tau_val}): {e}")
                # Add NaN placeholders for failed animals
                all_median_speeds.append(np.nan)
                all_speeds.append(np.array([np.nan]))
                if return_fc2:
                    all_fc2.append(np.array([np.nan]))
                processing_errors[i] = str(e)
            
            finally:
                # Memory cleanup for individual animal processing
                if 'median_speed' in locals():
                    del median_speed
                if 'speeds' in locals():
                    del speeds
                if 'fc2' in locals():
                    del fc2
                if 'result' in locals():
                    del result
        
        # Convert lists to arrays and save everything in ONE file
        try:
            # Convert to numpy arrays
            all_median_speeds = np.array(all_median_speeds)  # Shape: (n_animals,)
            
            # For speeds, handle potentially different lengths
            speeds_array = np.array(all_speeds, dtype=object)
            
            # Check if all speed arrays have the same length
            if len(all_speeds) > 0:
                speed_lengths = [len(s) if hasattr(s, '__len__') else 1 for s in all_speeds]
                if len(set(speed_lengths)) == 1:  # All same length
                    speeds_array = np.array(all_speeds)  # Convert to regular 2D array
                    self.logger.info(f"Speeds converted to regular array: {speeds_array.shape}")
                else:
                    self.logger.info(f"Speeds have different lengths, keeping as object array")
            
            # Prepare data to save
            save_data = {
                'median_speeds': all_median_speeds,
                'speeds': speeds_array,
                'metadata': {
                    'window_size': window_size,
                    'tau': tau_val,
                    'effective_vstep': effective_vstep,
                    'method': method,
                    'n_animals': n_animals,
                    'regions': nodes,
                    'processing_errors': processing_errors,
                    'n_successful': np.sum(~np.isnan(all_median_speeds)),
                    'n_errors': len(processing_errors)
                }
            }
            
            # Add FC2 data if computed
            if return_fc2 and all_fc2:
                fc2_array = np.array(all_fc2, dtype=object)
                if len(all_fc2) > 0:
                    try:
                        fc2_array = np.array(all_fc2)  # Try regular array
                        self.logger.info(f"FC2 converted to regular array: {fc2_array.shape}")
                    except ValueError:
                        self.logger.info(f"FC2 arrays have different shapes, keeping as object array")
                save_data['fc2'] = fc2_array
            
            # Save combined results
            np.savez_compressed(combined_file, **save_data)
            
            n_successful = save_data['metadata']['n_successful']
            n_errors = save_data['metadata']['n_errors']
            
            self.logger.info(f"✓ Saved combined results for tau={tau_val} to {combined_file}")
            self.logger.info(f"  Successful: {n_successful}/{n_animals}, Errors: {n_errors}")
            
            return {
                'processed': n_successful,
                'errors': n_errors,
                'cached': 0,
                'total': n_animals,
                'status': 'completed'
            }
            
        except Exception as save_error:
            self.logger.error(f"Failed to save combined results: {save_error}")
            raise
        
        finally:
            # Final memory cleanup
            del all_median_speeds, all_speeds
            if all_fc2:
                del all_fc2
            gc.collect()


def process_dfc_speed_analysis_optimized(data_loader, window_sizes: List[int], 
                                       tau_range: List[int], save_path: Path,
                                       n_jobs: int = -1, load_cache: bool = True,
                                       **analysis_kwargs) -> Dict:
    """
    Optimized main function for DFC speed analysis with proper memory management.
    
    Parameters:
    -----------
    data_loader : object
        Data loader object with methods to load DFC data
    window_sizes : List[int]
        List of window sizes to analyze
    tau_range : List[int]
        List of tau values to analyze
    save_path : Path
        Directory to save results
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    load_cache : bool
        Whether to load cached results
    **analysis_kwargs : dict
        Additional analysis parameters (method, overlap_mode, etc.)
    
    Returns:
    --------
    Dict : Summary of all processing results
    """
    
    analyzer = DFCSpeedAnalyzer(save_path)
    logger = logging.getLogger(__name__)
    
    # Extract analysis parameters
    method = analysis_kwargs.get('method', 'pearson')
    overlap_mode = analysis_kwargs.get('overlap_mode', 'points')
    overlap_value = analysis_kwargs.get('overlap_value', 1)
    return_fc2 = analysis_kwargs.get('return_fc2', False)
    min_tau_zero = analysis_kwargs.get('min_tau_zero', True)
    
    def calculate_vstep(window_size: int, overlap_mode: str, overlap_value: float) -> int:
        """Calculate step size for sliding window."""
        if overlap_mode == 'points':
            return max(1, int(overlap_value))
        elif overlap_mode == 'percentage':
            if not 0 <= overlap_value <= 1:
                raise ValueError(f"Overlap percentage must be [0,1], got {overlap_value}")
            return max(1, int(window_size * (1 - overlap_value)))
        else:
            raise ValueError(f"Unknown overlap_mode: {overlap_mode}")
    
    # Get data metadata once
    data_loader.get_metadata()
    n_animals = data_loader.n_animals
    nodes = data_loader.regions
    lag = data_loader.lag
    
    all_results = {}
    
    # Process each window size
    for window_size in tqdm(window_sizes, desc="Window sizes"):
        logger.info(f"Processing window size: {window_size}")
        
        # Load DFC data once per window size
        try:
            data_loader.load_dfc_1_window(lag=lag, window=window_size)
            dfc_data = data_loader.dfc_stream
            
            if dfc_data is None or len(dfc_data) == 0:
                logger.error(f"No DFC data loaded for window {window_size}")
                continue
                
            # Convert to numpy array for easier handling
            dfc_array = np.array(dfc_data)
            logger.info(f"DFC data shape: {dfc_array.shape}")
            
        except Exception as e:
            logger.error(f"Failed to load DFC data for window {window_size}: {e}")
            continue
                
        # Create output directory
        out_dir = save_path / f"speed_win{window_size}"
        out_dir.mkdir(exist_ok=True, parents=True)
        
        # Process each tau value
        for tau_val in tqdm(tau_range, desc=f"Tau values (win={window_size})"):
            
            # Calculate effective vstep
            base_vstep = calculate_vstep(window_size, overlap_mode, overlap_value)
            effective_vstep = int(base_vstep + tau_val)
            
            if effective_vstep <= 0:
                logger.warning(f"Skipping tau={tau_val}: effective_vstep={effective_vstep} <= 0")
                continue
            
            logger.info(f"Processing tau={tau_val}, effective_vstep={effective_vstep}")
            
            # Process ALL animals at once (no batching needed for combined files)
            tau_result = analyzer._process_all_animals_combined(
                dfc_array=dfc_array,
                window_size=window_size,
                tau_val=tau_val,
                effective_vstep=effective_vstep,
                method=method,
                return_fc2=return_fc2,
                out_dir=out_dir,
                nodes=nodes,
                n_animals=n_animals,
                load_cache=load_cache
            )
            
            # Store results
            tau_summary = {
                'window_size': window_size,
                'tau': tau_val,
                'effective_vstep': effective_vstep,
                'processed': tau_result['processed'],
                'errors': tau_result['errors'],
                'cached': tau_result['cached'],
                'total': tau_result['total'],
                'success_rate': tau_result['processed'] / n_animals if n_animals > 0 else 0,
                'status': tau_result['status']
            }
            
            all_results[f"win{window_size}_tau{tau_val}"] = tau_summary
            
            logger.info(
                f"✓ Window {window_size}, Tau {tau_val}: "
                f"{tau_summary['processed']}/{tau_summary['total']} processed, "
                f"{tau_summary['cached']} cached, {tau_summary['errors']} errors"
            )        
        # Clean up DFC data for this window
        del dfc_array, dfc_data
        data_loader.dfc_stream = None  # Clear from data loader
        gc.collect()
    
    # Save overall summary
    summary_file = save_path / "analysis_summary.npz"
    np.savez_compressed(summary_file, 
                       results=all_results,
                       parameters={
                           'window_sizes': window_sizes,
                           'tau_range': tau_range,
                           'method': method,
                           'overlap_mode': overlap_mode,
                           'overlap_value': overlap_value,
                           'return_fc2': return_fc2,
                           'n_animals': n_animals,
                           'nodes': nodes
                       })
    
    logger.info(f"✓ Analysis complete. Summary saved to {summary_file}")
    return all_results


# Usage example with your original code structure:
def run_optimized_analysis(data, processors=-1):
    """
    Drop-in replacement for your original analysis loop.
    """
    save_path = data.paths['speed']
    min_tau_zero = True
    tau_range = np.arange(0, data.tau + 1) if min_tau_zero else np.arange(-data.tau, data.tau + 1)
    
    analysis_kwargs = {
        'method': 'pearson',
        'overlap_mode': 'points',
        'overlap_value': 1,
        'return_fc2': True,
        'min_tau_zero': min_tau_zero,
    }
    
    results = process_dfc_speed_analysis_optimized(
        data_loader=data,
        window_sizes=data.time_window_range,
        tau_range=tau_range,
        save_path=save_path,
        n_jobs=processors,
        load_cache=True,
        **analysis_kwargs
    )
    
    return results

run_optimized_analysis(data, processors=-1)

#%%
def load_speed_data(data, window_size: int, tau_val: int) -> Dict:
    """
    Load speed and FC2 data for a specific window size and tau value.
    """
    speed_path = data.paths['speed']
    
    # Construct filename
    filename = f"speed_win{window_size}_tau{tau_val}_all_animals_n{data.n_animals}_regions{data.regions}.npz"
    file_path = speed_path / f"speed_win{window_size}" / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"Speed data file not found: {file_path}")
    
    print(f"Loading: {file_path}")
    
    # Load the data
    loaded_data = np.load(file_path, allow_pickle=True)
    
    result = {
        'median_speeds': loaded_data['median_speeds'],
        'speeds': loaded_data['speeds'],
        'metadata': loaded_data['metadata'].item(),
        'file_path': file_path
    }
    
    # Add FC2 if available
    if 'fc2' in loaded_data:
        result['fc2'] = loaded_data['fc2']
        print(f"✓ Loaded FC2 data with shape: {loaded_data['fc2'].shape}")
    else:
        print("! No FC2 data found in file")
    
    
    loaded_data.close()  # Close the file
    return result

# Quick load function for immediate use:
def quick_load(data, window_size, tau_val):
    """Quick load function for interactive use"""
    return load_speed_data(data, window_size, tau_val)

data_50_3 = quick_load(data, window_size=50, tau_val=3)
#%%
# plot histograms of speeds and median speeds, and a flattened version of the speeds
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Plotting function
def plot_speed_histograms(data):
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle(f"Speed Distributions (Window Size: {data['metadata']['window_size']}, Tau: {data['metadata']['tau']})")

    # Histogram of median speeds
    axs[0].hist(data['median_speeds'], bins=30, color='blue', alpha=0.7)
    axs[0].set_title("Median Speeds")
    axs[0].set_xlabel("Speed")
    axs[0].set_ylabel("Frequency")

    # MMatrix of distribution of speeds, y axis= animal, x axis= speed, colors= frequency
    speed_matrix = np.array(data['speeds'])
    # Get the distribution of speeds and bin edges
    speed_bins = np.linspace(0, 1.2, 100)  # Assuming speeds are in the range [0, 1.2]
    speed_matrix = np.array([np.histogram(s, bins=speed_bins)[0] for s in speed_matrix])
    # Normalize the speed matrix
    speed_matrix = speed_matrix / speed_matrix.sum(axis=1, keepdims=True)
    # Ensure the matrix is filled with zeros where no data exists
    speed_matrix = np.nan_to_num(speed_matrix, nan=0.0)
    # Set the x-ticks to be the speed values
    axs[1].set_xticks(np.arange(len(speed_bins)))
    axs[1].set_xticklabels(np.round(speed_bins, 2), rotation=45)
    # Plot the speed matrix
    axs[1].imshow(speed_matrix, aspect='auto', cmap='viridis', origin='lower')
    axs[1].set_title("Speeds")
    axs[1].set_xlabel("Speed")
    axs[1].set_ylabel("Frequency")
    axs[1].set_xticks(np.arange(0, 30, 5))

    # Flattened histogram of speeds
    flattened_speeds = np.concatenate(data['speeds'])
    axs[2].hist(flattened_speeds, bins=30, color='green', alpha=0.7, histtype='step')
    axs[2].set_title("Flattened Speeds")
    axs[2].set_xlabel("Speed")
    axs[2].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

# Quick load and plot
data_10_3 = quick_load(data, window_size=10, tau_val=3)
plot_speed_histograms(data_10_3)

# %%
import numpy as np
from pathlib import Path
import os

def check_float32_conversion_feasibility(data):
    """
    Check feasibility of converting all speed/FC2 data from float64 to float32.
    
    Parameters:
    -----------
    data : DFCAnalysis object
        Data analysis object with paths
    
    Returns:
    --------
    Dict with analysis results
    """
    speed_path = Path(data.paths['speed'])
    total_size_64 = 0
    total_size_32 = 0
    file_count = 0
    
    print("Scanning files...")
    
    # Scan all files
    for window_size in data.time_window_range:
        for tau_val in range(0, data.tau + 1):
            filename = f"speed_win{window_size}_tau{tau_val}_all_animals_n{data.n_animals}_regions{data.regions}.npz"
            file_path = speed_path / f"speed_win{window_size}" / filename
            
            if file_path.exists():
                file_count += 1
                current_size = file_path.stat().st_size
                total_size_64 += current_size
                # Float32 uses half the space for numeric data (rough estimate: 50% reduction)
                total_size_32 += current_size * 0.5
    
    # Calculate savings
    space_saved_gb = (total_size_64 - total_size_32) / (1024**3)
    reduction_percent = ((total_size_64 - total_size_32) / total_size_64 * 100) if total_size_64 > 0 else 0
    
    print(f"\n📊 Float32 Conversion Analysis:")
    print(f"Files found: {file_count}")
    print(f"Current size (float64): {total_size_64/(1024**3):.2f} GB")
    print(f"Estimated size (float32): {total_size_32/(1024**3):.2f} GB")
    print(f"Space saved: {space_saved_gb:.2f} GB ({reduction_percent:.1f}%)")
    print(f"\n✅ Recommendation: {'Worth converting!' if space_saved_gb > 1 else 'Minor savings'}")
    
    return {
        'current_size_gb': total_size_64/(1024**3),
        'estimated_size_gb': total_size_32/(1024**3),
        'space_saved_gb': space_saved_gb,
        'reduction_percent': reduction_percent,
        'file_count': file_count
    }
    
check_float32_conversion_feasibility(data)

# %%
import numpy as np

def check_float32_error(data, window_size=50, tau_val=3):
    """
    Quick check of precision loss when converting to float32.
    
    Parameters:
    -----------
    data : DFCAnalysis object
    window_size : int
        Sample window to test
    tau_val : int
        Sample tau to test
    """
    # Load sample data
    sample = load_speed_data(data, window_size, tau_val)
    
    print(f"Testing on window={window_size}, tau={tau_val}")
    
    # Check median_speeds
    original = sample['median_speeds']
    converted = original.astype(np.float32).astype(np.float64)
    
    abs_error = np.abs(original - converted)
    rel_error = abs_error / (np.abs(original) + 1e-10)
    
    print(f"\n📊 Median Speeds:")
    print(f"Max absolute error: {np.max(abs_error):.2e}")
    print(f"Max relative error: {np.max(rel_error):.2%}")
    print(f"Mean relative error: {np.mean(rel_error):.2%}")
    
    # Check speeds array
    if isinstance(sample['speeds'], np.ndarray) and sample['speeds'].dtype != object:
        speeds_orig = sample['speeds']
        speeds_conv = speeds_orig.astype(np.float32).astype(np.float64)
        speeds_rel_error = np.abs(speeds_orig - speeds_conv) / (np.abs(speeds_orig) + 1e-10)
        print(f"\n📊 Speeds Array:")
        print(f"Max relative error: {np.max(speeds_rel_error):.6%}")
    
    # Check FC2 if available
    if 'fc2' in sample and isinstance(sample['fc2'], np.ndarray):
        fc2_orig = sample['fc2']
        fc2_conv = fc2_orig.astype(np.float32).astype(np.float64)
        fc2_rel_error = np.abs(fc2_orig - fc2_conv) / (np.abs(fc2_orig) + 1e-10)
        print(f"\n📊 FC2:")
        print(f"Max relative error: {np.max(fc2_rel_error):.6%}")
    
    print(f"\n✅ Verdict: {'Safe to convert' if np.max(rel_error) < 0.001 else 'Check if error is acceptable'}")
check_float32_error(data, window_size=50, tau_val=3)
# %%
import numpy as np

def analyze_float16_feasibility(data, window_size=50, tau_val=3):
    """
    Check if float16 conversion is safe for your data.
    
    ⚠️ Float16 limitations:
    - Range: ±65,504 (values outside become inf)
    - Precision: ~3-4 decimal places
    - Smallest positive: 6.10×10⁻⁵ (smaller becomes 0)
    """
    # Load sample data
    sample = load_speed_data(data, window_size, tau_val)
    
    print("🔍 Float16 Feasibility Analysis")
    print("=" * 50)
    
    results = {}
    
    # Check each array type
    for name, array in [('median_speeds', sample['median_speeds']), 
                       ('speeds', sample['speeds'][0] if sample['speeds'].ndim > 1 else sample['speeds'])]:
        
        if not isinstance(array, np.ndarray) or array.dtype == object:
            continue
            
        print(f"\n📊 Analyzing {name}:")
        
        # Check range
        min_val, max_val = np.nanmin(array), np.nanmax(array)
        print(f"  Range: [{min_val:.6f}, {max_val:.6f}]")
        
        # Check if within float16 range
        if abs(min_val) > 65504 or abs(max_val) > 65504:
            print(f"  ❌ OUT OF RANGE for float16!")
            results[name] = 'unsafe'
            continue
        
        # Check precision loss
        original = array.flatten()[:1000]  # Sample first 1000 values
        f16_converted = original.astype(np.float16).astype(np.float64)
        f32_converted = original.astype(np.float32).astype(np.float64)
        
        # Calculate errors
        f16_rel_error = np.abs(original - f16_converted) / (np.abs(original) + 1e-10)
        f32_rel_error = np.abs(original - f32_converted) / (np.abs(original) + 1e-10)
        
        print(f"  Float32 max error: {np.max(f32_rel_error):.2e} ({np.max(f32_rel_error)*100:.4f}%)")
        print(f"  Float16 max error: {np.max(f16_rel_error):.2e} ({np.max(f16_rel_error)*100:.2f}%)")
        
        # Check for values that become zero
        tiny_values = np.sum(np.abs(original) < 6.1e-5)
        if tiny_values > 0:
            print(f"  ⚠️  {tiny_values} values would become 0 in float16")
        
        # Verdict
        if np.max(f16_rel_error) < 0.01:  # Less than 1% error
            print(f"  ✅ Safe for float16")
            results[name] = 'safe'
        else:
            print(f"  ⚠️  High precision loss ({np.max(f16_rel_error)*100:.1f}%)")
            results[name] = 'risky'
    
    # Overall recommendation
    print("\n" + "="*50)
    print("💡 RECOMMENDATION:")
    
    if all(v == 'safe' for v in results.values()):
        print("✅ Float16 appears safe for your data")
        print("   Space savings: ~75% vs float64, ~50% vs float32")
    else:
        print("⚠️  Float16 NOT recommended - significant precision loss")
        print("   Stick with float32 for safety")
    
    return results


def compare_all_formats(data, window_size=50, tau_val=3):
    """
    Compare storage sizes for different formats.
    """
    sample = load_speed_data(data, window_size, tau_val)
    
    print("💾 Storage Comparison (per array):")
    print("=" * 50)
    
    for name, array in [('median_speeds', sample['median_speeds'])]:
        if not isinstance(array, np.ndarray):
            continue
            
        size_64 = array.nbytes / 1024**2  # MB
        size_32 = (array.nbytes / 2) / 1024**2
        size_16 = (array.nbytes / 4) / 1024**2
        
        print(f"\n{name} (shape: {array.shape}):")
        print(f"  float64: {size_64:.2f} MB")
        print(f"  float32: {size_32:.2f} MB (-{(1-size_32/size_64)*100:.0f}%)")
        print(f"  float16: {size_16:.2f} MB (-{(1-size_16/size_64)*100:.0f}%)")


def test_float16_conversion(data, window_size=50, tau_val=3):
    """
    Test actual float16 conversion on sample data.
    """
    sample = load_speed_data(data, window_size, tau_val)
    
    print("🧪 Testing Float16 Conversion:")
    print("=" * 50)
    
    # Test on median speeds
    original = sample['median_speeds']
    
    # Try conversion
    try:
        f16_array = original.astype(np.float16)
        
        # Check for infinities
        n_inf = np.sum(np.isinf(f16_array))
        n_zero = np.sum((original != 0) & (f16_array == 0))
        
        print(f"✅ Conversion successful")
        print(f"   Values turned to inf: {n_inf}")
        print(f"   Non-zero values turned to 0: {n_zero}")
        
        if n_inf > 0 or n_zero > 0:
            print("   ⚠️  DATA LOSS DETECTED!")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")


# Quick check function
def should_use_float16(data):
    """Quick recommendation for float16."""
    results = analyze_float16_feasibility(data)
    return all(v == 'safe' for v in results.values())


# 1. Run feasibility analysis
analyze_float16_feasibility(data)

# 2. Compare storage sizes
compare_all_formats(data)

# 3. Test actual conversion
test_float16_conversion(data)
# %%
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import gc

def convert_all_to_float16(data, backup=True, test_first=True):
    """
    Convert all speed/FC2 files to float16 for maximum space savings.
    
    Parameters:
    -----------
    data : DFCAnalysis object
    backup : bool
        If True, creates backup of original files
    test_first : bool
        If True, tests conversion on first file before proceeding
    """
    speed_path = Path(data.paths['speed'])
    converted_count = 0
    total_saved_gb = 0
    errors = []
    
    print("🚀 Starting float16 conversion...")
    
    # Get all files to process
    files_to_convert = []
    for window_size in data.time_window_range:
        for tau_val in range(0, data.tau + 1):
            filename = f"speed_win{window_size}_tau{tau_val}_all_animals_n{data.n_animals}_regions{data.regions}.npz"
            file_path = speed_path / f"speed_win{window_size}" / filename
            if file_path.exists():
                files_to_convert.append((window_size, tau_val, file_path))
    
    print(f"Found {len(files_to_convert)} files to convert")
    
    # Test first file if requested
    if test_first and files_to_convert:
        print("\n🧪 Testing first file...")
        _, _, test_file = files_to_convert[0]
        
        with np.load(test_file, allow_pickle=True) as loaded_data:
            for key in ['median_speeds', 'speeds', 'fc2']:
                if key in loaded_data:
                    array = loaded_data[key]
                    if isinstance(array, np.ndarray) and array.dtype in [np.float64, np.float32]:
                        test_f16 = array.astype(np.float16)
                        if np.any(np.isinf(test_f16)):
                            raise ValueError(f"Float16 conversion creates infinities in {key}!")
                        
                        # Check max error
                        rel_error = np.abs(array - test_f16.astype(array.dtype)) / (np.abs(array) + 1e-10)
                        max_error = np.max(rel_error)
                        if max_error > 0.01:  # 1% threshold
                            raise ValueError(f"Float16 conversion error too high in {key}: {max_error:.2%}")
        
        print("✅ Test passed! Proceeding with conversion...")
    
    # Process each file
    for window_size, tau_val, file_path in tqdm(files_to_convert, desc="Converting to float16"):
        try:
            original_size = file_path.stat().st_size / (1024**3)  # GB
            
            # Create backup if requested
            if backup:
                backup_path = file_path.with_suffix('.npz.f16bak')
                if not backup_path.exists():
                    shutil.copy2(file_path, backup_path)
            
            # Load data
            with np.load(file_path, allow_pickle=True) as loaded_data:
                save_data = {}
                
                # Convert arrays to float16
                for key in loaded_data.files:
                    data_array = loaded_data[key]
                    
                    if key in ['median_speeds', 'speeds', 'fc2']:
                        if isinstance(data_array, np.ndarray) and data_array.dtype in [np.float64, np.float32]:
                            # Convert to float16
                            f16_array = data_array.astype(np.float16)
                            
                            # Safety check
                            if np.any(np.isinf(f16_array)):
                                raise ValueError(f"Infinity detected in {key}")
                            
                            save_data[key] = f16_array
                        else:
                            save_data[key] = data_array
                    else:
                        # Keep metadata as is
                        save_data[key] = data_array
            
            # Save converted file
            np.savez_compressed(file_path, **save_data)
            
            # Check new size
            new_size = file_path.stat().st_size / (1024**3)  # GB
            saved = original_size - new_size
            total_saved_gb += saved
            converted_count += 1
            
            # Clean up memory
            del save_data
            gc.collect()
            
        except Exception as e:
            errors.append((file_path, str(e)))
            print(f"\n❌ Error converting {file_path.name}: {e}")
    
    # Summary
    print(f"\n✅ Float16 Conversion Complete!")
    print(f"Files converted: {converted_count}/{len(files_to_convert)}")
    print(f"Space saved: {total_saved_gb:.2f} GB")
    print(f"Current precision: ~3-4 decimal places (sufficient for your data)")
    
    if errors:
        print(f"\n⚠️ Errors encountered: {len(errors)}")
        for path, error in errors[:5]:
            print(f"  - {path.name}: {error}")
    
    return {
        'converted': converted_count,
        'total_files': len(files_to_convert),
        'space_saved_gb': total_saved_gb,
        'errors': errors
    }


def verify_float16_files(data, sample_size=5):
    """
    Verify float16 conversion worked correctly by checking random files.
    """
    speed_path = Path(data.paths['speed'])
    
    print("🔍 Verifying float16 conversion...")
    
    # Find some converted files
    checked = 0
    issues = 0
    
    for window_size in data.time_window_range[:sample_size]:
        for tau_val in range(min(3, data.tau + 1)):
            filename = f"speed_win{window_size}_tau{tau_val}_all_animals_n{data.n_animals}_regions{data.regions}.npz"
            file_path = speed_path / f"speed_win{window_size}" / filename
            
            if file_path.exists():
                with np.load(file_path, allow_pickle=True) as loaded_data:
                    for key in ['median_speeds', 'speeds', 'fc2']:
                        if key in loaded_data:
                            array = loaded_data[key]
                            if isinstance(array, np.ndarray) and array.dtype == np.float16:
                                print(f"✅ {filename} - {key}: float16")
                                checked += 1
                            elif isinstance(array, np.ndarray):
                                print(f"⚠️  {filename} - {key}: still {array.dtype}")
                                issues += 1
    
    print(f"\nChecked {checked} arrays, found {issues} issues")
    return checked, issues


def restore_from_float16_backup(data):
    """
    Restore files from .f16bak backups if needed.
    """
    speed_path = Path(data.paths['speed'])
    restored = 0
    
    for backup_file in speed_path.rglob("*.npz.f16bak"):
        original_file = backup_file.with_suffix('')  # Remove .f16bak
        shutil.copy2(backup_file, original_file)
        restored += 1
    
    print(f"✅ Restored {restored} files from float16 backups")
    return restored


def delete_float16_backups(data):
    """
    Delete all .f16bak files after confirming conversion worked.
    """
    speed_path = Path(data.paths['speed'])
    deleted = 0
    
    for backup_file in speed_path.rglob("*.npz.f16bak"):
        backup_file.unlink()
        deleted += 1
    
    print(f"✅ Deleted {deleted} float16 backup files")
    return deleted


# Quick usage function
def quick_convert_float16(data):
    """Quick convert all files to float16 with safety checks."""
    return convert_all_to_float16(data, backup=True, test_first=True)


# Usage
if __name__ == "__main__":
    # Convert to float16
    result = quick_convert_float16(data)
    
    # Verify conversion
    verify_float16_files(data)
    
    # If problems, restore:
    # restore_from_float16_backup(data)
    
    # If all good, delete backups:
    # delete_float16_backups(data)
# %%
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import gc

def check_fc2_range(data, sample_windows=3):
    """
    Quick check of FC2 value ranges to understand why float16 fails.
    """
    print("🔍 Checking FC2 value ranges...")
    
    for window_size in data.time_window_range[:sample_windows]:
        for tau_val in range(min(3, data.tau + 1)):
            try:
                sample = load_speed_data(data, window_size, tau_val)
                if 'fc2' in sample:
                    fc2 = sample['fc2']
                    print(f"\nWindow {window_size}, Tau {tau_val}:")
                    print(f"  FC2 range: [{np.min(fc2):.6f}, {np.max(fc2):.6f}]")
                    print(f"  Contains values < 6e-5: {np.sum(np.abs(fc2) < 6e-5)} values")
                    
                    # Test float16 conversion
                    f16_test = fc2.astype(np.float16)
                    zeros_created = np.sum((fc2 != 0) & (f16_test == 0))
                    print(f"  Float16 would create {zeros_created} false zeros")
            except:
                pass


def convert_all_hybrid(data, backup=True):
    """
    Hybrid conversion: float16 for speed data, float32 for FC2.
    This gives optimal space savings while preserving FC2 precision.
    
    Parameters:
    -----------
    data : DFCAnalysis object
    backup : bool
        If True, creates backup of original files
    """
    speed_path = Path(data.paths['speed'])
    converted_count = 0
    total_saved_gb = 0
    errors = []
    
    print("🚀 Starting hybrid conversion (float16 for speed, float32 for FC2)...")
    
    # Get all files to process
    files_to_convert = []
    for window_size in data.time_window_range:
        for tau_val in range(0, data.tau + 1):
            filename = f"speed_win{window_size}_tau{tau_val}_all_animals_n{data.n_animals}_regions{data.regions}.npz"
            file_path = speed_path / f"speed_win{window_size}" / filename
            if file_path.exists():
                files_to_convert.append((window_size, tau_val, file_path))
    
    print(f"Found {len(files_to_convert)} files to convert")
    
    # Process each file
    for window_size, tau_val, file_path in tqdm(files_to_convert, desc="Hybrid conversion"):
        try:
            original_size = file_path.stat().st_size / (1024**3)  # GB
            
            # Create backup if requested
            if backup:
                backup_path = file_path.with_suffix('.npz.hybridbak')
                if not backup_path.exists():
                    shutil.copy2(file_path, backup_path)
            
            # Load data
            with np.load(file_path, allow_pickle=True) as loaded_data:
                save_data = {}
                
                # Convert arrays with appropriate precision
                for key in loaded_data.files:
                    data_array = loaded_data[key]
                    
                    if key in ['median_speeds', 'speeds'] and isinstance(data_array, np.ndarray):
                        # Convert speed data to float16 (safe)
                        if data_array.dtype in [np.float64, np.float32]:
                            save_data[key] = data_array.astype(np.float16)
                        else:
                            save_data[key] = data_array
                            
                    elif key == 'fc2' and isinstance(data_array, np.ndarray):
                        # Keep FC2 as float32 (for precision)
                        if data_array.dtype == np.float64:
                            save_data[key] = data_array.astype(np.float32)
                        else:
                            save_data[key] = data_array
                            
                    else:
                        # Keep metadata as is
                        save_data[key] = data_array
            
            # Save converted file
            np.savez_compressed(file_path, **save_data)
            
            # Check new size
            new_size = file_path.stat().st_size / (1024**3)  # GB
            saved = original_size - new_size
            total_saved_gb += saved
            converted_count += 1
            
            # Clean up memory
            del save_data
            gc.collect()
            
        except Exception as e:
            errors.append((file_path, str(e)))
            print(f"\n❌ Error converting {file_path.name}: {e}")
    
    # Summary
    print(f"\n✅ Hybrid Conversion Complete!")
    print(f"Files converted: {converted_count}/{len(files_to_convert)}")
    print(f"Space saved: {total_saved_gb:.2f} GB")
    print(f"\nPrecision summary:")
    print(f"  • Speed data: float16 (3-4 decimal places)")
    print(f"  • FC2 data: float32 (7 decimal places)")
    
    if errors:
        print(f"\n⚠️ Errors: {len(errors)}")
    
    return {
        'converted': converted_count,
        'total_files': len(files_to_convert),
        'space_saved_gb': total_saved_gb,
        'errors': errors
    }


def verify_hybrid_conversion(data, sample_size=50):
    """
    Verify hybrid conversion worked correctly.
    """
    speed_path = Path(data.paths['speed'])
    
    print("🔍 Verifying hybrid conversion...")
    print("Expected: speed data as float16, FC2 as float32")
    print("-" * 50)
    
    checked_files = 0
    correct_format = 0
    
    for window_size in data.time_window_range[:sample_size]:
        for tau_val in range(min(2, data.tau + 1)):
            filename = f"speed_win{window_size}_tau{tau_val}_all_animals_n{data.n_animals}_regions{data.regions}.npz"
            file_path = speed_path / f"speed_win{window_size}" / filename
            
            if file_path.exists():
                checked_files += 1
                all_correct = True
                
                with np.load(file_path, allow_pickle=True) as loaded_data:
                    print(f"\n📁 {filename}:")
                    
                    # Check median_speeds (should be float16)
                    if 'median_speeds' in loaded_data:
                        dtype = loaded_data['median_speeds'].dtype
                        is_correct = dtype == np.float16
                        print(f"  median_speeds: {dtype} {'✅' if is_correct else '❌'}")
                        all_correct &= is_correct
                    
                    # Check speeds (should be float16)
                    if 'speeds' in loaded_data:
                        array = loaded_data['speeds']
                        if isinstance(array, np.ndarray) and array.dtype != object:
                            dtype = array.dtype
                            is_correct = dtype == np.float16
                            print(f"  speeds: {dtype} {'✅' if is_correct else '❌'}")
                            all_correct &= is_correct
                    
                    # Check FC2 (should be float32)
                    if 'fc2' in loaded_data:
                        dtype = loaded_data['fc2'].dtype
                        is_correct = dtype == np.float32
                        print(f"  fc2: {dtype} {'✅' if is_correct else '❌'}")
                        all_correct &= is_correct
                
                if all_correct:
                    correct_format += 1
    
    print(f"\n Summary: {correct_format}/{checked_files} files correctly formatted")
    return correct_format == checked_files


def estimate_hybrid_savings(data):
    """
    Estimate space savings with hybrid approach.
    """
    # Rough estimates based on typical data
    n_files = len(data.time_window_range) * (data.tau + 1)
    
    # Assume average file has both speed and FC2 data
    # Speed data: small (48 animals × few values)
    # FC2 data: large (48 × 666 × 111 as in your example)
    
    speed_data_gb = n_files * 0.001  # Very small
    fc2_data_gb = n_files * 0.05     # Larger
    
    # Current (float64)
    current_total = (speed_data_gb + fc2_data_gb) * 1
    
    # Hybrid (speed->float16, fc2->float32)
    hybrid_total = (speed_data_gb * 0.25) + (fc2_data_gb * 0.5)
    
    savings = current_total - hybrid_total
    percent = (savings / current_total) * 100
    
    print(f"💰 Estimated Hybrid Savings:")
    print(f"  Current size: ~{current_total:.1f} GB")
    print(f"  After hybrid: ~{hybrid_total:.1f} GB")
    print(f"  Space saved: ~{savings:.1f} GB ({percent:.0f}%)")


# Convenience functions
def quick_hybrid_convert(data):
    """Quick hybrid conversion with backup."""
    return convert_all_hybrid(data, backup=True)


def restore_hybrid_backups(data):
    """Restore from .hybridbak files."""
    speed_path = Path(data.paths['speed'])
    restored = 0
    
    for backup_file in speed_path.rglob("*.npz.hybridbak"):
        original_file = backup_file.with_suffix('')
        shutil.copy2(backup_file, original_file)
        restored += 1
    
    print(f"✅ Restored {restored} files from hybrid backups")
    return restored


def delete_hybrid_backups(data):
    """Delete .hybridbak files."""
    speed_path = Path(data.paths['speed'])
    deleted = 0
    
    for backup_file in speed_path.rglob("*.npz.hybridbak"):
        backup_file.unlink()
        deleted += 1
    
    print(f"✅ Deleted {deleted} hybrid backup files")
    return deleted
# %%
# 1. First, check why FC2 fails with float16
check_fc2_range(data)

# 2. Estimate savings with hybrid approach
estimate_hybrid_savings(data)

# # 3. Run the hybrid conversion
result = quick_hybrid_convert(data)

# # 4. Verify it worked correctly
verify_hybrid_conversion(data)

# # 5. If all good, delete backups
# delete_hybrid_backups(data)
# %%

import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import gc

def check_fc2_range(data, sample_windows=3):
    """
    Quick check of FC2 value ranges to understand why float16 fails.
    """
    print("🔍 Checking FC2 value ranges...")
    
    for window_size in data.time_window_range[:sample_windows]:
        for tau_val in range(min(3, data.tau + 1)):
            try:
                sample = load_speed_data(data, window_size, tau_val)
                if 'fc2' in sample:
                    fc2 = sample['fc2']
                    print(f"\nWindow {window_size}, Tau {tau_val}:")
                    print(f"  FC2 range: [{np.min(fc2):.6f}, {np.max(fc2):.6f}]")
                    print(f"  Contains values < 6e-5: {np.sum(np.abs(fc2) < 6e-5)} values")
                    
                    # Test float16 conversion
                    f16_test = fc2.astype(np.float16)
                    zeros_created = np.sum((fc2 != 0) & (f16_test == 0))
                    print(f"  Float16 would create {zeros_created} false zeros")
            except:
                pass


def convert_all_hybrid(data, backup=True):
    """
    Hybrid conversion: float16 for speed data, float32 for FC2.
    This gives optimal space savings while preserving FC2 precision.
    
    Parameters:
    -----------
    data : DFCAnalysis object
    backup : bool
        If True, creates backup of original files
    """
    speed_path = Path(data.paths['speed'])
    converted_count = 0
    total_saved_gb = 0
    errors = []
    
    print("🚀 Starting hybrid conversion (float16 for speed, float32 for FC2)...")
    
    # Get all files to process
    files_to_convert = []
    for window_size in data.time_window_range:
        for tau_val in range(0, data.tau + 1):
            filename = f"speed_win{window_size}_tau{tau_val}_all_animals_n{data.n_animals}_regions{data.regions}.npz"
            file_path = speed_path / f"speed_win{window_size}" / filename
            if file_path.exists():
                files_to_convert.append((window_size, tau_val, file_path))
    
    print(f"Found {len(files_to_convert)} files to convert")
    
    # Process each file
    for window_size, tau_val, file_path in tqdm(files_to_convert, desc="Hybrid conversion"):
        try:
            original_size = file_path.stat().st_size / (1024**3)  # GB
            
            # Create backup if requested
            if backup:
                backup_path = file_path.with_suffix('.npz.hybridbak')
                if not backup_path.exists():
                    shutil.copy2(file_path, backup_path)
            
            # Load data
            with np.load(file_path, allow_pickle=True) as loaded_data:
                save_data = {}
                
                # Convert arrays with appropriate precision
                for key in loaded_data.files:
                    data_array = loaded_data[key]
                    
                    if key in ['median_speeds', 'speeds'] and isinstance(data_array, np.ndarray):
                        # Convert speed data to float16 (safe)
                        if data_array.dtype in [np.float64, np.float32]:
                            save_data[key] = data_array.astype(np.float16)
                        else:
                            save_data[key] = data_array
                            
                    elif key == 'fc2' and isinstance(data_array, np.ndarray):
                        # Keep FC2 as float32 (for precision)
                        if data_array.dtype == np.float64:
                            save_data[key] = data_array.astype(np.float32)
                        else:
                            save_data[key] = data_array
                            
                    else:
                        # Keep metadata as is
                        save_data[key] = data_array
            
            # Save converted file
            np.savez_compressed(file_path, **save_data)
            
            # Check new size
            new_size = file_path.stat().st_size / (1024**3)  # GB
            saved = original_size - new_size
            total_saved_gb += saved
            converted_count += 1
            
            # Clean up memory
            del save_data
            gc.collect()
            
        except Exception as e:
            errors.append((file_path, str(e)))
            print(f"\n❌ Error converting {file_path.name}: {e}")
    
    # Summary
    print(f"\n✅ Hybrid Conversion Complete!")
    print(f"Files converted: {converted_count}/{len(files_to_convert)}")
    print(f"Space saved: {total_saved_gb:.2f} GB")
    print(f"\nPrecision summary:")
    print(f"  • Speed data: float16 (3-4 decimal places)")
    print(f"  • FC2 data: float32 (7 decimal places)")
    
    if errors:
        print(f"\n⚠️ Errors: {len(errors)}")
    
    return {
        'converted': converted_count,
        'total_files': len(files_to_convert),
        'space_saved_gb': total_saved_gb,
        'errors': errors
    }


def verify_hybrid_conversion(data, sample_size=5):
    """
    Verify hybrid conversion worked correctly.
    """
    speed_path = Path(data.paths['speed'])
    
    print("🔍 Verifying hybrid conversion...")
    print("Expected: speed data as float16, FC2 as float32")
    print("-" * 50)
    
    checked_files = 0
    correct_format = 0
    
    # Sample evenly across the range
    window_samples = np.linspace(0, len(data.time_window_range)-1, min(sample_size, len(data.time_window_range)), dtype=int)
    tau_samples = np.linspace(0, data.tau, min(sample_size, data.tau+1), dtype=int)
    
    for window_idx in window_samples:
        window_size = data.time_window_range[window_idx]
        for tau_val in tau_samples:
            filename = f"speed_win{window_size}_tau{tau_val}_all_animals_n{data.n_animals}_regions{data.regions}.npz"
            file_path = speed_path / f"speed_win{window_size}" / filename
            
            if file_path.exists():
                checked_files += 1
                all_correct = True
                
                with np.load(file_path, allow_pickle=True) as loaded_data:
                    print(f"\n📁 {filename}:")
                    
                    # Check median_speeds (should be float16)
                    if 'median_speeds' in loaded_data:
                        dtype = loaded_data['median_speeds'].dtype
                        is_correct = dtype == np.float16
                        print(f"  median_speeds: {dtype} {'✅' if is_correct else '❌'}")
                        all_correct &= is_correct
                    
                    # Check speeds (should be float16)
                    if 'speeds' in loaded_data:
                        array = loaded_data['speeds']
                        if isinstance(array, np.ndarray) and array.dtype != object:
                            dtype = array.dtype
                            is_correct = dtype == np.float16
                            print(f"  speeds: {dtype} {'✅' if is_correct else '❌'}")
                            all_correct &= is_correct
                    
                    # Check FC2 (should be float32)
                    if 'fc2' in loaded_data:
                        dtype = loaded_data['fc2'].dtype
                        is_correct = dtype == np.float32
                        print(f"  fc2: {dtype} {'✅' if is_correct else '❌'}")
                        all_correct &= is_correct
                
                if all_correct:
                    correct_format += 1
    
    print(f"\n Summary: {correct_format}/{checked_files} files correctly formatted")
    return correct_format == checked_files


def check_conversion_scope(data):
    """
    Show the full scope of files that will be converted.
    """
    print("📊 Conversion Scope Analysis:")
    print("="*50)
    print(f"Window sizes: {data.time_window_range[0]} to {data.time_window_range[-1]} (n={len(data.time_window_range)})")
    print(f"Tau values: 0 to {data.tau} (n={data.tau + 1})")
    print(f"Total files: {len(data.time_window_range) * (data.tau + 1)}")
    print(f"Animals per file: {data.n_animals}")
    print(f"Regions: {data.regions}")
    
    # Check actual files
    speed_path = Path(data.paths['speed'])
    actual_files = 0
    
    for window_size in data.time_window_range:
        for tau_val in range(0, data.tau + 1):
            filename = f"speed_win{window_size}_tau{tau_val}_all_animals_n{data.n_animals}_regions{data.regions}.npz"
            file_path = speed_path / f"speed_win{window_size}" / filename
            if file_path.exists():
                actual_files += 1
    
    print(f"\nActual files found: {actual_files}")
    print(f"Missing files: {len(data.time_window_range) * (data.tau + 1) - actual_files}")
    
    return actual_files
    """
    Estimate space savings with hybrid approach.
    """
    # Rough estimates based on typical data
    n_files = len(data.time_window_range) * (data.tau + 1)
    
    # Assume average file has both speed and FC2 data
    # Speed data: small (48 animals × few values)
    # FC2 data: large (48 × 666 × 111 as in your example)
    
    speed_data_gb = n_files * 0.001  # Very small
    fc2_data_gb = n_files * 0.05     # Larger
    
    # Current (float64)
    current_total = (speed_data_gb + fc2_data_gb) * 1
    
    # Hybrid (speed->float16, fc2->float32)
    hybrid_total = (speed_data_gb * 0.25) + (fc2_data_gb * 0.5)
    
    savings = current_total - hybrid_total
    percent = (savings / current_total) * 100
    
    print(f"💰 Estimated Hybrid Savings:")
    print(f"  Current size: ~{current_total:.1f} GB")
    print(f"  After hybrid: ~{hybrid_total:.1f} GB")
    print(f"  Space saved: ~{savings:.1f} GB ({percent:.0f}%)")


# Convenience functions
def quick_hybrid_convert(data):
    """Quick hybrid conversion with backup."""
    return convert_all_hybrid(data, backup=True)


def restore_hybrid_backups(data):
    """Restore from .hybridbak files."""
    speed_path = Path(data.paths['speed'])
    restored = 0
    
    for backup_file in speed_path.rglob("*.npz.hybridbak"):
        original_file = backup_file.with_suffix('')
        shutil.copy2(backup_file, original_file)
        restored += 1
    
    print(f"✅ Restored {restored} files from hybrid backups")
    return restored


def delete_hybrid_backups(data):
    """Delete .hybridbak files."""
    speed_path = Path(data.paths['speed'])
    deleted = 0
    
    for backup_file in speed_path.rglob("*.npz.hybridbak"):
        backup_file.unlink()
        deleted += 1
    
    print(f"✅ Deleted {deleted} hybrid backup files")
    return deleted

# %%
check_conversion_scope(data)

# %%
# Run the full conversion (this processes ALL files)
result = quick_hybrid_convert(data)

# Check the result
print(f"\nConverted {result['converted']} files")
print(f"Total space saved: {result['space_saved_gb']:.2f} GB")
# %%
verify_hybrid_conversion(data)
# %%
