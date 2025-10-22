# Migration Guide: julien_data Code Reorganization

**Date**: October 2025  
**Purpose**: Document the migration of scripts from `julien_data/` to organized `src/` and `legacy/` directories

## Overview

The julien_data directory has been reorganized to improve code maintainability and clarity. This guide documents what was moved, why, and how to adapt your workflows.

## Changes Summary

### New Directory Structure

```
julien_data/
├── src/                      # Core standalone scripts (NEW)
│   ├── 1_preprocess_data_ts_cog.py
│   ├── 2_compute_dfc_stream.py
│   ├── 3_dfc_speed_test_v6.py
│   ├── class_dataanalysis_julien.py
│   ├── simple_speed_analysis.py
│   └── README.md
├── legacy/                   # Deprecated/experimental code (NEW)
│   ├── 3_dfc_local_speed_v1.py
│   ├── Plot_speed_figures.py
│   ├── dfc_windows_pooling.py
│   ├── laod_las_speed.py
│   ├── local_speed_plot.py
│   ├── local_speed_plot_v2.py
│   ├── modularity.py
│   ├── plot_cog_data.py
│   ├── plots.py
│   ├── plts_speed.py
│   ├── test_func_speed.py
│   └── README.md
├── fig/                      # Figures (unchanged)
├── results/                  # Results (unchanged)
├── PROJECT_SUMMARY.md        # Project documentation (unchanged)
└── plots_speed.ipynb         # Jupyter notebook (unchanged)
```

## Migration Details

### 1. Scripts Moved to `src/` (Core Pipeline)

These are the **production-ready, standalone scripts** that form the main analysis pipeline:

| Script | Purpose | Why in src/ |
|--------|---------|-------------|
| `1_preprocess_data_ts_cog.py` | Data preprocessing | Core pipeline step 1 |
| `2_compute_dfc_stream.py` | DFC stream computation | Core pipeline step 2 |
| `3_dfc_speed_test_v6.py` | DFC speed computation | Latest version, core analysis |
| `class_dataanalysis_julien.py` | Analysis utilities class | Required by pipeline scripts |
| `simple_speed_analysis.py` | Results analysis | Standalone analysis tool |

**Action Required**: If you were running these scripts, update your paths:
```bash
# OLD
python 1_preprocess_data_ts_cog.py

# NEW
python src/1_preprocess_data_ts_cog.py
# OR
cd src && python 1_preprocess_data_ts_cog.py
```

### 2. Scripts Moved to `legacy/` (Deprecated/Experimental)

These scripts are **no longer part of the main pipeline** but retained for reference:

| Script | Reason for Legacy Status |
|--------|-------------------------|
| `3_dfc_local_speed_v1.py` | Superseded by v6 |
| `test_func_speed.py` | Development/test script |
| `laod_las_speed.py` | Old version with filename typo |
| `local_speed_plot.py` | Superseded by v2 |
| `local_speed_plot_v2.py` | Large plotting script, not core pipeline |
| `plot_cog_data.py` | Visualization only, not core analysis |
| `plots.py` | Generic utilities, not standalone |
| `plts_speed.py` | Plotting utilities |
| `Plot_speed_figures.py` | Figure generation script |
| `dfc_windows_pooling.py` | Experimental approach |
| `modularity.py` | Experimental analysis |

**Action Required**: If you need these scripts, update paths:
```bash
# OLD
python local_speed_plot_v2.py

# NEW
python legacy/local_speed_plot_v2.py
```

⚠️ **Warning**: Legacy scripts may not work with current data structures or dependencies.

### 3. Files Deleted (Empty/Unused)

The following **empty files** were removed:

- `3_dfc_speed_test.py` (0 bytes)
- `demo_before_after.py` (0 bytes)
- `demo_improved_system.py` (0 bytes)
- `demo_practical_usage.py` (0 bytes)
- `test_improved_functions.py` (0 bytes)

**Action Required**: None. These files contained no code.

## Import Updates

### For Scripts in `src/`

Scripts in `src/` can import from each other using relative imports:

```python
# No changes needed - these already work
from class_dataanalysis_julien import DFCAnalysis
```

### For External Scripts

If you have scripts outside julien_data that import from it:

```python
# OLD
from class_dataanalysis_julien import DFCAnalysis

# NEW - if running from repository root
from julien_data.src.class_dataanalysis_julien import DFCAnalysis

# OR - add src to Python path
import sys
sys.path.insert(0, 'julien_data/src')
from class_dataanalysis_julien import DFCAnalysis
```

## Best Practices Going Forward

### For New Scripts

1. **Core analysis scripts** → Add to `src/`
   - Standalone scripts
   - Part of main pipeline
   - Well-tested and documented

2. **Experimental/test scripts** → Add to `legacy/` or keep in root temporarily
   - Prototype code
   - Alternative approaches
   - Development/debugging scripts

3. **Visualization scripts** → Consider if they belong in `src/` or `legacy/`
   - If critical for results → `src/`
   - If exploratory/optional → `legacy/`

### Naming Conventions

Following the pattern from `allegiance/src/`:
- Use descriptive names: `compute_dfc_stream.py` ✓ not `dfc.py` ✗
- Number sequential pipeline steps: `1_`, `2_`, `3_`
- Version important alternatives: `_v1`, `_v2`, etc.

## Verification

To verify your setup after migration:

```bash
# Check src scripts are present
ls julien_data/src/*.py

# Check legacy scripts are present  
ls julien_data/legacy/*.py

# Test importing the analysis class
python -c "import sys; sys.path.insert(0, 'julien_data/src'); from class_dataanalysis_julien import DFCAnalysis; print('Import successful')"
```

## Rollback (if needed)

If you need to temporarily revert:

```bash
# This would undo the migration (NOT RECOMMENDED)
git revert <commit-hash>
```

However, we recommend adapting to the new structure as it improves organization and maintainability.

## Questions?

If you encounter issues with the migration:
1. Check if the script is in `src/` or `legacy/`
2. Update your import paths
3. Review this guide's Import Updates section
4. Check the README in each directory for specific details

## Related Documentation

- `src/README.md` - Core scripts documentation
- `legacy/README.md` - Legacy scripts documentation
- `PROJECT_SUMMARY.md` - Original project summary
- Repository root `README.md` - Overall repository structure
