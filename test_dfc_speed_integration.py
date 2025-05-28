#!/usr/bin/env python3
"""
Test script for DFC speed integration with get_tenet4window_range function.
Tests the complete 'dfc_speed' prefix workflow.
"""

import numpy as np
import sys
import tempfile
from pathlib import Path

# Add the shared_code module to the path
sys.path.insert(0, str(Path(__file__).parent / "shared_code"))

from shared_code.fun_dfcspeed import get_tenet4window_range

def test_dfc_speed_integration():
    """Test the DFC speed integration with get_tenet4window_range."""
    
    print("Testing DFC speed integration...")
    
    # Create synthetic test data
    np.random.seed(42)
    n_animals = 3
    n_regions = 5  
    n_timepoints = 200
    
    # Generate synthetic time series data
    ts_data = np.random.randn(n_animals, n_regions, n_timepoints) * 0.1
    
    # Add some temporal structure
    for i in range(n_animals):
        for j in range(n_regions):
            ts_data[i, j, :] += np.sin(np.linspace(0, 4*np.pi, n_timepoints) + j * 0.5) * 0.2
    
    print(f"Test data shape: {ts_data.shape}")
    
    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Set up paths dictionary
        paths = {
            'dfc_speed': temp_path / 'dfc_speed'
        }
        
        # Test parameters
        time_window_range = [10, 20, 5]  # Small range for quick testing
        prefix = 'dfc_speed'
        lag = 1
        regions = [f'region_{i}' for i in range(n_regions)]
        processors = 1  # Use single processor for testing
        
        # DFC speed specific parameters
        dfc_speed_kwargs = {
            'tau': 3,
            'min_tau_zero': False,
            'method': 'pearson'
        }
        
        print(f"Window range: {time_window_range}")
        print(f"DFC speed parameters: {dfc_speed_kwargs}")
        
        try:
            # Test the complete DFC speed workflow
            print("\nRunning get_tenet4window_range with dfc_speed prefix...")
            
            get_tenet4window_range(
                ts=ts_data,
                time_window_range=time_window_range,
                prefix=prefix,
                paths=paths,
                lag=lag,
                n_animals=n_animals,
                regions=regions,
                processors=processors,
                **dfc_speed_kwargs
            )
            
            print("✓ DFC speed computation completed successfully!")
            
            # Check if output files were created
            output_dir = paths['dfc_speed']
            if output_dir.exists():
                output_files = list(output_dir.glob('*.npz'))
                print(f"✓ Created {len(output_files)} output files:")
                for file in output_files:
                    print(f"  - {file.name}")
                    
                    # Try to load and inspect one file
                    if len(output_files) > 0:
                        data = np.load(output_files[0], allow_pickle=True)
                        print(f"  File keys: {list(data.keys())}")
                        if 'speed_medians' in data:
                            speed_medians = data['speed_medians']
                            print(f"  Speed medians shape: {speed_medians.shape}")
                            print(f"  Speed medians sample: {speed_medians[:3] if len(speed_medians) > 3 else speed_medians}")
            else:
                print("⚠ No output directory created")
            
            return True
            
        except Exception as e:
            print(f"✗ Error during DFC speed integration test: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_dfc_speed_kwargs_validation():
    """Test kwargs parameter validation for DFC speed."""
    
    print("\n" + "="*60)
    print("Testing DFC speed kwargs validation...")
    
    # Create minimal test data
    np.random.seed(42)
    ts_data = np.random.randn(1, 3, 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        paths = {'dfc_speed': temp_path / 'dfc_speed'}
        
        # Test different parameter combinations
        test_cases = [
            {'tau': 2, 'min_tau_zero': True, 'method': 'pearson'},
            {'tau': 5, 'min_tau_zero': False, 'method': 'spearman'},
            {'tau': 1, 'min_tau_zero': True, 'method': 'cosine'},
        ]
        
        for i, kwargs in enumerate(test_cases):
            try:
                print(f"\nTest case {i+1}: {kwargs}")
                
                get_tenet4window_range(
                    ts=ts_data,
                    time_window_range=[10, 10, 1],  # Single window for speed
                    prefix='dfc_speed',
                    paths=paths,
                    lag=1,
                    n_animals=1,
                    regions=['r1', 'r2', 'r3'],
                    processors=1,
                    **kwargs
                )
                
                print(f"  ✓ Parameters accepted: {kwargs}")
                
            except Exception as e:
                print(f"  ✗ Error with parameters {kwargs}: {e}")
                return False
    
    print("✓ All kwargs validation tests passed!")
    return True

if __name__ == "__main__":
    print("="*60)
    print("DFC SPEED INTEGRATION TESTING")
    print("="*60)
    
    # Run integration test
    success1 = test_dfc_speed_integration()
    
    # Run kwargs validation test
    success2 = test_dfc_speed_kwargs_validation()
    
    if success1 and success2:
        print("\n" + "="*60)
        print("✓ ALL DFC SPEED INTEGRATION TESTS PASSED!")
        print("="*60)
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("✗ SOME TESTS FAILED")
        print("="*60)
        sys.exit(1)
