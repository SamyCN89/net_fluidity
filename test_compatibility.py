#!/usr/bin/env python3
"""
Comparison test between the unified dfc_speed function and existing implementations.
"""

import numpy as np
import sys
import time
from pathlib import Path

# Add paths for different implementations
sys.path.insert(0, str(Path(__file__).parent / "shared_code"))
sys.path.insert(0, str(Path(__file__).parent / "metaconnectivity"))
sys.path.insert(0, str(Path(__file__).parent / "julien_data"))

def test_implementation_compatibility():
    """Compare results between different dfc_speed implementations."""
    
    print("COMPATIBILITY TESTING: Unified vs Original Implementations")
    print("="*70)
    
    # Create test data
    np.random.seed(42)
    n_rois = 10
    n_frames = 50
    
    # Create 3D test data
    dfc_3d = np.random.randn(n_rois, n_rois, n_frames) * 0.1
    for t in range(n_frames):
        dfc_3d[:, :, t] = (dfc_3d[:, :, t] + dfc_3d[:, :, t].T) / 2
        np.fill_diagonal(dfc_3d[:, :, t], 0.0)
    
    # Convert to 2D for testing
    tril_idx = np.tril_indices(n_rois, k=-1)
    dfc_2d = dfc_3d[tril_idx[0], tril_idx[1], :]
    
    vstep = 2
    
    print(f"Test data: {n_rois}×{n_rois}×{n_frames} -> {dfc_2d.shape[0]} pairs × {n_frames} frames")
    print(f"Using vstep = {vstep}")
    print()
    
    # Test unified implementation
    try:
        from shared_code.fun_dfcspeed import dfc_speed as dfc_speed_unified
        
        start_time = time.time()
        median_unified, speeds_unified = dfc_speed_unified(dfc_2d, vstep=vstep, method='pearson')
        time_unified = time.time() - start_time
        
        print(f"✓ UNIFIED IMPLEMENTATION:")
        print(f"  Median speed: {median_unified:.6f}")
        print(f"  Speed array length: {len(speeds_unified)}")
        print(f"  Speed range: [{speeds_unified.min():.6f}, {speeds_unified.max():.6f}]")
        print(f"  Computation time: {time_unified:.6f} seconds")
        print()
        
        unified_available = True
        
    except Exception as e:
        print(f"✗ UNIFIED IMPLEMENTATION: Failed - {e}")
        unified_available = False
        median_unified = None
        speeds_unified = None
    
    # Test metaconnectivity implementation
    try:
        from fun_dfcspeed import dfc_speed as dfc_speed_meta
        
        start_time = time.time()
        median_meta, speeds_meta = dfc_speed_meta(dfc_2d, vstep=vstep)
        time_meta = time.time() - start_time
        
        print(f"✓ METACONNECTIVITY IMPLEMENTATION:")
        print(f"  Median speed: {median_meta:.6f}")
        print(f"  Speed array length: {len(speeds_meta)}")
        print(f"  Speed range: [{speeds_meta.min():.6f}, {speeds_meta.max():.6f}]")
        print(f"  Computation time: {time_meta:.6f} seconds")
        print()
        
        meta_available = True
        
    except Exception as e:
        print(f"✗ METACONNECTIVITY IMPLEMENTATION: Failed - {e}")
        meta_available = False
        median_meta = None
        speeds_meta = None
    
    # Test julien implementation
    try:
        sys.path.append(str(Path(__file__).parent / "julien_data"))
        # Read the file to get the function
        julien_file = Path(__file__).parent / "julien_data" / "3_dfc_speed_analysis.py"
        if julien_file.exists():
            # Import the dfc_speed and dfc_speed_new functions
            exec(open(julien_file).read(), globals())
            
            # Test original dfc_speed from julien
            if 'dfc_speed' in globals():
                start_time = time.time()
                median_julien, speeds_julien = dfc_speed(dfc_2d, vstep=vstep, method='pearson')
                time_julien = time.time() - start_time
                
                print(f"✓ JULIEN IMPLEMENTATION (original):")
                print(f"  Median speed: {median_julien:.6f}")
                print(f"  Speed array length: {len(speeds_julien)}")
                print(f"  Speed range: [{speeds_julien.min():.6f}, {speeds_julien.max():.6f}]")
                print(f"  Computation time: {time_julien:.6f} seconds")
                print()
                
                julien_available = True
            else:
                print("✗ JULIEN IMPLEMENTATION: dfc_speed function not found")
                julien_available = False
                median_julien = None
                speeds_julien = None
            
            # Test optimized dfc_speed_new from julien
            if 'dfc_speed_new' in globals():
                start_time = time.time()
                median_julien_new, speeds_julien_new = dfc_speed_new(dfc_2d, vstep=vstep)
                time_julien_new = time.time() - start_time
                
                print(f"✓ JULIEN IMPLEMENTATION (optimized):")
                print(f"  Median speed: {median_julien_new:.6f}")
                print(f"  Speed array length: {len(speeds_julien_new)}")
                print(f"  Speed range: [{speeds_julien_new.min():.6f}, {speeds_julien_new.max():.6f}]")
                print(f"  Computation time: {time_julien_new:.6f} seconds")
                print()
                
                julien_new_available = True
            else:
                print("✗ JULIEN IMPLEMENTATION (optimized): dfc_speed_new function not found")
                julien_new_available = False
                median_julien_new = None
                speeds_julien_new = None
                
        else:
            print("✗ JULIEN IMPLEMENTATION: File not found")
            julien_available = False
            julien_new_available = False
            
    except Exception as e:
        print(f"✗ JULIEN IMPLEMENTATION: Failed - {e}")
        julien_available = False
        julien_new_available = False
    
    # Compare results
    print("\nRESULT COMPARISON:")
    print("-" * 50)
    
    if unified_available and meta_available:
        diff_median = abs(median_unified - median_meta)
        print(f"Unified vs Metaconnectivity median difference: {diff_median:.8f}")
        
        if len(speeds_unified) == len(speeds_meta):
            speed_diff = np.mean(np.abs(speeds_unified - speeds_meta))
            max_speed_diff = np.max(np.abs(speeds_unified - speeds_meta))
            print(f"Speed arrays mean difference: {speed_diff:.8f}")
            print(f"Speed arrays max difference: {max_speed_diff:.8f}")
        else:
            print("Speed arrays have different lengths")
    
    if unified_available and 'julien_available' in locals() and julien_available:
        diff_median = abs(median_unified - median_julien)
        print(f"Unified vs Julien median difference: {diff_median:.8f}")
    
    if unified_available and 'julien_new_available' in locals() and julien_new_available:
        diff_median = abs(median_unified - median_julien_new)
        print(f"Unified vs Julien-new median difference: {diff_median:.8f}")
    
    print("\nPERFORMANCE SUMMARY:")
    print("-" * 50)
    
    if unified_available:
        print(f"Unified implementation: {time_unified:.6f} seconds")
    if meta_available:
        print(f"Metaconnectivity implementation: {time_meta:.6f} seconds")
    if 'julien_available' in locals() and julien_available:
        print(f"Julien implementation: {time_julien:.6f} seconds")
    if 'julien_new_available' in locals() and julien_new_available:
        print(f"Julien optimized implementation: {time_julien_new:.6f} seconds")
    
    return True

if __name__ == "__main__":
    test_implementation_compatibility()
    print("\n" + "="*70)
    print("COMPATIBILITY TESTING COMPLETE")
    print("="*70)
