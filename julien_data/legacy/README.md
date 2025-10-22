# Legacy Scripts

This directory contains legacy, deprecated, and experimental scripts that are no longer part of the main analysis pipeline but are retained for reference and historical purposes.

## Contents

### Older Versions
- **`3_dfc_local_speed_v1.py`**: Earlier version of DFC speed computation (superseded by v6)
  - Kept for reference and comparison
  - May use different algorithms or parameters

### Test and Development Scripts
- **`test_func_speed.py`**: Test script for speed computation functions
  - Used during development for validation
  - Contains various test cases and benchmarks

### Loading and Processing Scripts
- **`laod_las_speed.py`**: Old loading script with typo in filename
  - Historical script for loading speed results
  - Functionality integrated into current pipeline

### Plotting and Visualization Scripts
- **`local_speed_plot.py`**: First version of speed plotting utilities
- **`local_speed_plot_v2.py`**: Updated version with enhanced features
  - Large comprehensive plotting script
  - Contains various visualization approaches
  
- **`plot_cog_data.py`**: Cognitive data plotting utilities
  - Statistical visualizations
  - Group comparisons

- **`plots.py`**: Generic plotting functions
- **`plts_speed.py`**: Speed-specific plotting utilities
- **`Plot_speed_figures.py`**: Script for generating publication figures

### Experimental Analysis Scripts
- **`dfc_windows_pooling.py`**: Experimental windowing and pooling approaches
  - Alternative methods for DFC analysis
  - Not part of current pipeline

- **`modularity.py`**: Network modularity analysis
  - Experimental community detection
  - Alternative analysis approach

## Why These Scripts Are Here

These scripts have been moved to legacy for various reasons:
1. **Superseded by newer versions**: Better implementations exist in the main `src/` directory
2. **Experimental code**: Not validated for production use
3. **Development artifacts**: Test scripts and prototypes
4. **Redundant functionality**: Features now integrated into core scripts
5. **Historical reference**: Useful for understanding the evolution of the analysis

## Usage Warning

⚠️ **These scripts may not work with the current data structure or dependencies**
- They are retained for reference only
- Use scripts in `src/` for current analysis
- If you need functionality from these scripts, consider porting relevant code to the main pipeline

## Migration Notes

If you need to use any of these scripts:
1. Review the code carefully for outdated dependencies
2. Check if equivalent functionality exists in `src/` scripts
3. Update import paths if needed
4. Test thoroughly before using results

For the current, maintained analysis pipeline, always use scripts from the `../src/` directory.
