#!/usr/bin/env python3
"""
Test script for the unified dfc_speed function.
Tests different input formats, methods, and parameter combinations.
"""

import numpy as np
import sys
from pathlib import Path

# Add the shared_code module to the path
sys.path.insert(0, str(Path(__file__).parent / "shared_code"))

from shared_code.fun_dfcspeed import dfc_speed

def test_unified_dfc_speed():
    """Test the unified dfc_speed function with various inputs and parameters."""
    
    print("Testing unified dfc_speed function...")
    
    # Create synthetic test data
    np.random.seed(42)
    
    # Test 1: 2D input (n_pairs, frames)
    print("\n1. Testing 2D input (n_pairs, frames)...")
    n_pairs = 45  # 10 choose 2 = 45 pairs
    n_frames = 100
    dfc_2d = np.random.randn(n_pairs, n_frames) * 0.1 + np.sin(np.linspace(0, 4*np.pi, n_frames))
    
    # Basic test
    median_speed, speeds = dfc_speed(dfc_2d, vstep=1, method='pearson')
    print(f"  Median speed: {median_speed:.4f}")
    print(f"  Speed array shape: {speeds.shape}")
    print(f"  Speed range: [{speeds.min():.4f}, {speeds.max():.4f}]")
    
    # Test 2: 3D input (roi, roi, frames)
    print("\n2. Testing 3D input (roi, roi, frames)...")
    n_rois = 10
    dfc_3d = np.random.randn(n_rois, n_rois, n_frames) * 0.1
    # Make it symmetric
    for t in range(n_frames):
        dfc_3d[:, :, t] = (dfc_3d[:, :, t] + dfc_3d[:, :, t].T) / 2
        np.fill_diagonal(dfc_3d[:, :, t], 1.0)  # Set diagonal to 1
    
    median_speed, speeds = dfc_speed(dfc_3d, vstep=1, method='pearson')
    print(f"  Median speed: {median_speed:.4f}")
    print(f"  Speed array shape: {speeds.shape}")
    
    # Test 3: Different methods
    print("\n3. Testing different correlation methods...")
    methods = ['pearson', 'spearman', 'cosine']
    
    for method in methods:
        try:
            median_speed, speeds = dfc_speed(dfc_2d, vstep=1, method=method)
            print(f"  {method.capitalize()}: median_speed = {median_speed:.4f}")
        except Exception as e:
            print(f"  {method.capitalize()}: ERROR - {e}")
    
    # Test 4: Different vstep values
    print("\n4. Testing different vstep values...")
    vsteps = [1, 2, 5, 10]
    
    for vstep in vsteps:
        median_speed, speeds = dfc_speed(dfc_2d, vstep=vstep, method='pearson')
        print(f"  vstep={vstep}: median_speed = {median_speed:.4f}, n_speeds = {len(speeds)}")
    
    # Test 5: Return FC2 option
    print("\n5. Testing return_fc2 option...")
    median_speed, speeds, fc2 = dfc_speed(dfc_2d, vstep=1, return_fc2=True, method='pearson')
    print(f"  Median speed: {median_speed:.4f}")
    print(f"  FC2 shape: {fc2.shape}")
    
    # Test 6: Error handling
    print("\n6. Testing error handling...")
    
    # Invalid dimensions
    try:
        dfc_speed(np.random.randn(10), vstep=1)
        print("  ERROR: Should have raised ValueError for 1D input")
    except ValueError as e:
        print(f"  ✓ Correctly caught 1D input error: {e}")
    
    # Invalid method
    try:
        dfc_speed(dfc_2d, vstep=1, method='invalid')
        print("  ERROR: Should have raised ValueError for invalid method")
    except ValueError as e:
        print(f"  ✓ Correctly caught invalid method error: {e}")
    
    # Invalid vstep
    try:
        dfc_speed(dfc_2d, vstep=0)
        print("  ERROR: Should have raised TypeError for invalid vstep")
    except TypeError as e:
        print(f"  ✓ Correctly caught invalid vstep error: {e}")
    
    # vstep too large
    try:
        dfc_speed(dfc_2d, vstep=n_frames+1)
        print("  ERROR: Should have raised ValueError for vstep > n_frames")
    except ValueError as e:
        print(f"  ✓ Correctly caught vstep too large error: {e}")
    
    print("\n✓ All tests completed successfully!")
    
    return True

def performance_comparison():
    """Compare performance of different implementations if available."""
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    # Create larger test data
    np.random.seed(42)
    n_pairs = 1000
    n_frames = 500
    dfc_large = np.random.randn(n_pairs, n_frames) * 0.1
    
    import time
    
    # Test unified function
    print(f"\nTesting unified dfc_speed with {n_pairs} pairs × {n_frames} frames...")
    
    start_time = time.time()
    median_speed, speeds = dfc_speed(dfc_large, vstep=1, method='pearson')
    unified_time = time.time() - start_time
    
    print(f"Unified function:")
    print(f"  Time: {unified_time:.4f} seconds")
    print(f"  Median speed: {median_speed:.4f}")
    print(f"  Speed range: [{speeds.min():.4f}, {speeds.max():.4f}]")
    
    return unified_time

if __name__ == "__main__":
    # Run tests
    test_unified_dfc_speed()
    
    # Performance comparison
    try:
        performance_comparison()
    except Exception as e:
        print(f"Performance comparison failed: {e}")
    
    print("\n" + "="*60)
    print("UNIFIED DFC_SPEED FUNCTION TESTING COMPLETE")
    print("="*60)
