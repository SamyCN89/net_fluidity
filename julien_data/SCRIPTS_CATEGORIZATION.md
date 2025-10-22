# Scripts Categorization Summary

This document provides a comprehensive overview of all scripts in the julien_data directory after the migration.

## Core Production Scripts (in `src/`)

These are the **maintained, standalone scripts** that form the main analysis pipeline:

### ✅ Active Scripts

1. **`1_preprocess_data_ts_cog.py`** (8.4 KB)
   - Purpose: Load and preprocess raw time series and cognitive data
   - Status: Core pipeline script
   - Dependencies: shared_code
   - Keep: Yes

2. **`2_compute_dfc_stream.py`** (4.5 KB)
   - Purpose: Compute DFC streams from preprocessed data
   - Status: Core pipeline script
   - Dependencies: shared_code, class_dataanalysis_julien
   - Keep: Yes

3. **`3_dfc_speed_test_v6.py`** (16 KB)
   - Purpose: Latest DFC speed computation
   - Status: Current version, actively maintained
   - Dependencies: shared_code, class_dataanalysis_julien
   - Keep: Yes

4. **`class_dataanalysis_julien.py`** (37 KB)
   - Purpose: Core analysis class with data loading utilities
   - Status: Required by pipeline scripts
   - Dependencies: shared_code
   - Keep: Yes

5. **`simple_speed_analysis.py`** (3.7 KB)
   - Purpose: Standalone analysis and visualization of speed results
   - Status: Standalone tool
   - Dependencies: numpy, matplotlib
   - Keep: Yes

## Legacy/Deprecated Scripts (in `legacy/`)

These scripts are **retained for reference** but not part of active development:

### 📦 Historical Versions

6. **`3_dfc_local_speed_v1.py`** (24 KB)
   - Purpose: Earlier version of DFC speed computation
   - Status: Superseded by v6
   - Reason: Old implementation
   - Can Delete: After confirming no unique functionality

### 🧪 Test/Development Scripts

7. **`test_func_speed.py`** (15 KB)
   - Purpose: Test script for speed functions
   - Status: Development artifact
   - Reason: Testing only
   - Can Delete: If tests are no longer needed

### 📊 Plotting Scripts

8. **`Plot_speed_figures.py`** (6.7 KB)
   - Purpose: Generate publication figures
   - Status: Plotting utility
   - Reason: Visualization only, not core analysis
   - Can Delete: If figures are already generated

9. **`plot_cog_data.py`** (33 KB)
   - Purpose: Cognitive data visualization
   - Status: Large plotting script
   - Reason: Visualization only
   - Can Delete: If no longer needed for analysis

10. **`plots.py`** (4.8 KB)
    - Purpose: Generic plotting functions
    - Status: Utility script
    - Reason: Generic utilities
    - Can Delete: If functionality moved to shared_code

11. **`plts_speed.py`** (4.5 KB)
    - Purpose: Speed-specific plotting
    - Status: Utility script
    - Reason: Plotting utilities
    - Can Delete: If no longer used

12. **`local_speed_plot.py`** (57 KB)
    - Purpose: Speed plotting v1
    - Status: Superseded by v2
    - Reason: Old version
    - Can Delete: Yes, v2 exists

13. **`local_speed_plot_v2.py`** (93 KB)
    - Purpose: Enhanced speed plotting
    - Status: Latest version but large
    - Reason: Not core pipeline
    - Can Delete: If plots are already generated

### 🔬 Experimental Scripts

14. **`dfc_windows_pooling.py`** (22 KB)
    - Purpose: Experimental windowing approaches
    - Status: Experimental
    - Reason: Alternative method not adopted
    - Can Delete: If experiment concluded

15. **`modularity.py`** (7.4 KB)
    - Purpose: Network modularity analysis
    - Status: Experimental
    - Reason: Alternative analysis approach
    - Can Delete: If not part of current research

### 🗃️ Old Loading Scripts

16. **`laod_las_speed.py`** (50 KB)
    - Purpose: Load speed results (note typo in filename)
    - Status: Old version
    - Reason: Typo in name, old implementation
    - Can Delete: Yes, functionality integrated elsewhere

## Already Deleted (Empty Files)

These files were **removed** as they contained no code:

- ❌ `3_dfc_speed_test.py` (0 bytes) - DELETED
- ❌ `demo_before_after.py` (0 bytes) - DELETED
- ❌ `demo_improved_system.py` (0 bytes) - DELETED
- ❌ `demo_practical_usage.py` (0 bytes) - DELETED
- ❌ `test_improved_functions.py` (0 bytes) - DELETED

## Other Files (Not Scripts)

- `PROJECT_SUMMARY.md` - Project documentation (Keep)
- `plots_speed.ipynb` - Jupyter notebook (Keep for reference)
- `figure_dfc_cog_composite*.png/svg` - Generated figures (Keep)
- `fig/` directory - Figure output directory (Keep)
- `results/` directory - Analysis results directory (Keep)

## Recommendations for Deletion

### Safe to Delete (High Confidence)

These can be deleted after verifying they're no longer needed:

1. **`legacy/local_speed_plot.py`** - Superseded by v2
2. **`legacy/laod_las_speed.py`** - Old version with typo
3. **`legacy/3_dfc_local_speed_v1.py`** - Old version, superseded by v6

### Consider Deleting (Medium Confidence)

These are likely no longer needed but review first:

4. **`legacy/test_func_speed.py`** - Development script
5. **`legacy/dfc_windows_pooling.py`** - Experimental code
6. **`legacy/modularity.py`** - Experimental analysis

### Keep for Reference (Low Confidence for Deletion)

These might still be useful for plotting/visualization:

7. **`legacy/Plot_speed_figures.py`** - May be needed for figures
8. **`legacy/plot_cog_data.py`** - May be needed for analysis
9. **`legacy/plots.py`** - Generic utilities
10. **`legacy/plts_speed.py`** - Plotting utilities
11. **`legacy/local_speed_plot_v2.py`** - Latest plotting version

## Migration Impact

### Before Migration
- 21 Python scripts in julien_data root
- 5 empty files
- No clear organization

### After Migration
- **5 core scripts** in `src/` (actively maintained)
- **11 legacy scripts** in `legacy/` (reference only)
- **5 empty files** deleted
- Clear documentation in each directory

### Result
- ✅ Clear separation of active vs deprecated code
- ✅ Standalone scripts easily identifiable
- ✅ Legacy code preserved but separated
- ✅ Empty files removed
- ✅ Documentation added for maintainability

## Next Steps

1. **Review Legacy Scripts**: Determine which legacy scripts can be permanently deleted
2. **Archive vs Delete**: Consider creating a git tag before deletion for historical reference
3. **Test Core Scripts**: Verify the 5 core scripts work with current data
4. **Update Dependencies**: Ensure shared_code package is properly installed
5. **Documentation**: Keep README files updated as scripts are added/removed
