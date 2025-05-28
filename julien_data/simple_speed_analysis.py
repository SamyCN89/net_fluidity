#!/usr/bin/env python3
"""
Simple DFC Speed Results Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # Load the data
    results_file = Path("/media/samy/Elements1/Proyectos/LauraHarsan/results/julien_caillette/speed/speed_dfc_lag=1_tau=5_wmax=100_wmin=5.npz")
    
    print("Loading DFC speed results...")
    data = np.load(results_file, allow_pickle=True)
    
    vel_data = data['vel']
    speed_medians = data['speed_median']
    
    print(f"Number of animals: {len(vel_data)}")
    print(f"Speed medians shape: {speed_medians.shape}")
    
    # Simple statistics on speed medians
    print(f"\n=== SPEED MEDIAN ANALYSIS ===")
    print(f"Overall median speed range: [{np.nanmin(speed_medians):.6f}, {np.nanmax(speed_medians):.6f}]")
    print(f"Overall median speed mean: {np.nanmean(speed_medians):.6f}")
    
    # Try to analyze individual animal speeds
    print(f"\n=== INDIVIDUAL ANIMAL ANALYSIS ===")
    valid_animals = 0
    total_measurements = 0
    
    for i in range(len(vel_data)):
        try:
            vel = vel_data[i]
            if vel is not None:
                # Convert to array if needed
                if not isinstance(vel, np.ndarray):
                    vel = np.array(vel)
                
                n_measurements = len(vel.flatten())
                total_measurements += n_measurements
                valid_animals += 1
                
                if i < 5:  # Show first 5 animals
                    vel_flat = vel.flatten()
                    print(f"Animal {i}: {n_measurements} measurements, range [{vel_flat.min():.6f}, {vel_flat.max():.6f}]")
            else:
                print(f"Animal {i}: No data")
                
        except Exception as e:
            print(f"Animal {i}: Error - {e}")
    
    print(f"\nSummary: {valid_animals}/{len(vel_data)} animals with valid data")
    print(f"Total speed measurements: {total_measurements:,}")
    
    # Create simple visualization
    print(f"\n=== CREATING VISUALIZATION ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Speed medians across animals (averaged across windows and tau)
    animal_avg_speeds = np.nanmean(speed_medians, axis=(1, 2))
    axes[0, 0].bar(range(len(animal_avg_speeds)), animal_avg_speeds)
    axes[0, 0].set_title('Average Speed per Animal')
    axes[0, 0].set_xlabel('Animal ID')
    axes[0, 0].set_ylabel('Average Speed')
    
    # 2. Speed medians across window sizes (averaged across animals and tau)
    window_avg_speeds = np.nanmean(speed_medians, axis=(0, 2))
    axes[0, 1].plot(window_avg_speeds)
    axes[0, 1].set_title('Speed vs Window Size')
    axes[0, 1].set_xlabel('Window Index')
    axes[0, 1].set_ylabel('Average Speed')
    
    # 3. Speed medians across tau values (averaged across animals and windows)
    tau_avg_speeds = np.nanmean(speed_medians, axis=(0, 1))
    axes[1, 0].bar(range(len(tau_avg_speeds)), tau_avg_speeds)
    axes[1, 0].set_title('Speed vs Tau Value')
    axes[1, 0].set_xlabel('Tau Index')
    axes[1, 0].set_ylabel('Average Speed')
    
    # 4. Distribution of all median speeds
    all_medians = speed_medians.flatten()
    all_medians = all_medians[~np.isnan(all_medians)]
    axes[1, 1].hist(all_medians, bins=50, alpha=0.7, histtype='step')
    
    axes[1, 1].set_title('Distribution of Speed Medians')
    axes[1, 1].set_xlabel('Speed')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = "dfc_speed_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as: {output_file}")
    
    # Don't show the plot in headless environment
    # plt.show()

if __name__ == "__main__":
    main()
