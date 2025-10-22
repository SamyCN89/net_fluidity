# Julien Data Analysis - Core Scripts

This directory contains the core standalone scripts for analyzing the Julien Caillette dataset. These scripts form the main analysis pipeline for computing dynamic functional connectivity (DFC) and related metrics.

## Core Scripts

### 1. Data Preprocessing
- **`1_preprocess_data_ts_cog.py`**: Loads and preprocesses raw time series and cognitive data
  - Filters time series based on length
  - Matches time series with cognitive data
  - Saves preprocessed data for downstream analysis

### 2. DFC Stream Computation
- **`2_compute_dfc_stream.py`**: Computes DFC streams from preprocessed time series
  - Uses sliding window correlation
  - Supports various window sizes and lag parameters
  - Outputs DFC streams for speed computation

### 3. DFC Speed Analysis
- **`3_dfc_speed_test_v6.py`**: Latest version of DFC speed computation
  - Computes speed of variation in dynamic functional connectivity
  - Supports multiple correlation methods (Pearson, Spearman, cosine)
  - Includes parallelization for efficient processing

### 4. Analysis Utilities
- **`class_dataanalysis_julien.py`**: Core data analysis class
  - Centralizes data loading and preprocessing
  - Manages paths and metadata
  - Provides utilities for DFC and speed analysis

- **`simple_speed_analysis.py`**: Standalone script for analyzing DFC speed results
  - Loads precomputed speed results
  - Generates summary statistics
  - Creates visualizations

## Usage

### Basic Workflow

1. **Preprocess data**:
   ```bash
   cd /home/runner/work/net_fluidity/net_fluidity/julien_data/src
   python 1_preprocess_data_ts_cog.py
   ```

2. **Compute DFC streams**:
   ```bash
   python 2_compute_dfc_stream.py
   ```

3. **Compute DFC speed**:
   ```bash
   python 3_dfc_speed_test_v6.py
   ```

4. **Analyze results**:
   ```bash
   python simple_speed_analysis.py
   ```

## Dependencies

All scripts depend on the `shared_code` package which should be installed in editable mode:
```bash
pip install -e ../../shared_code
```

## Data Requirements

Scripts expect data to be organized according to paths defined in `shared_code.fun_paths`:
- Time series: `.mat` files containing neural time series
- Cognitive data: Excel file with mouse metadata
- Region labels: Text file with anatomical labels

## Output

Results are saved to directories managed by the path configuration:
- Preprocessed data: `preprocessed/`
- DFC streams: `dfc/`
- Speed results: `speed/`
- Figures: `fig/`

## Notes

- These scripts are designed to be run sequentially as part of the analysis pipeline
- Each script is standalone and can be executed independently if prerequisites are met
- For legacy/deprecated scripts, see the `../legacy/` directory
