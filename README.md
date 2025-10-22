# net_fluidity

Tools for dynamic functional connectivity (DFC), meta‑connectivity, and related analyses. This repo consolidates utilities to compute sliding‑window FC streams, dFC speed, pooling/oversampling summaries, and helper functions for data handling and plotting.

## Overview

- Focus: quantify fast changes in FC over time (dFC speed), build dFC streams, and support meta‑connectivity workflows.
- Core implementation lives under `shared_code/` with a unified, optimized `dfc_speed` and helpers. The `metaconnectivity/` folder contains research scripts and earlier implementations.

## Repository Structure

- `shared_code/`: installable Python package of reusable functions
  - `shared_code/fun_dfcspeed.py`: unified dFC speed, dFC stream, and handler utilities
  - `shared_code/fun_optimization.py`: fast correlation and vectorized speed kernels
  - `shared_code/fun_utils.py`: I/O, grouping, matrices, plotting helpers
  - `shared_code/fun_paths.py`: dataset/result/figures path helpers (env‑driven)
- `metaconnectivity/`: analysis scripts and legacy/experimental functions
- `julien_data/`: Julien Caillette dataset analysis
  - `julien_data/src/`: core standalone scripts for DFC analysis pipeline
  - `julien_data/legacy/`: deprecated and experimental code (reference only)
  - `julien_data/MIGRATION_GUIDE.md`: documentation of recent reorganization
- `allegiance/src/`: allegiance matrix computation scripts
- `test_*.py`: quick tests and comparisons between implementations

## Installation

Python 3.11 is recommended. Optionally create a virtual environment, then install the shared package in editable mode:

```bash
pip install -e shared_code
``

Optional: environment variables for data paths used by `fun_paths` can be placed in a local `.env` file or your shell. For example:

```bash
# used by shared_code/shared_code/fun_paths.py
PROJECT_ROOT_LOCAL=/absolute/path/to/project/root
DATASET_NAME=ines_abdullah
```

If you use `fun_paths.get_paths`, it will read `PROJECT_ROOT_<ENV>` (default `LOCAL`) and build a standard folder layout under `dataset/`, `results/`, and `fig/` within that root.

## Quick Start

Compute a dFC stream from time series and its dFC speed using the optimized unified function:

```python
import numpy as np
from shared_code.fun_dfcspeed import ts2dfc_stream, dfc_speed

# Synthetic time series: T timepoints × N regions
T, N = 300, 10
ts = np.random.randn(T, N)

# Build dFC stream (vectorized lower triangle over time)
dfc_2d = ts2dfc_stream(ts, window_size=30, lag=5, format_data='2D')

# dFC speed between FC_t and FC_{t+vstep}
median_speed, speeds = dfc_speed(dfc_2d, vstep=1, method='pearson')
print(median_speed, speeds.shape)
```

3D dFC input is also supported — pass `roi × roi × frames` arrays directly to `dfc_speed`.

## Running Tests

Simple functional checks and comparisons:

```bash
python test_unified_dfc_speed.py
python test_compatibility.py
```

These scripts generate synthetic data and compare implementations. They do not require external data.

## Notes on Implementations

- Prefer the unified `shared_code.fun_dfcspeed.dfc_speed` which supports Pearson/Spearman/cosine and returns optional FC2 streams. Earlier variants exist under `metaconnectivity/` and `julien_data/` for reference and benchmarking.

## Contributing

- Follow `shared_code/SCIENTIFIC_CODING_GUIDELINES.md` for style and scientific computing practices.
- Add or update docstrings for public functions; include parameter and return types where possible.
- When adding features, include a small usage snippet or expand the relevant README section.

## Citation

Dynamic Functional Connectivity as a complex random walk: Definitions and the dFCwalk toolbox. Lucas Arbabyazd, Diego Lombardo, Olivier Blin, Mira Didic, Demian Battaglia, Viktor Jirsa. MethodsX (2020) doi: 10.1016/j.mex.2020.101168.

