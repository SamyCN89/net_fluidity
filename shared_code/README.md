# shared_code

Reusable Python utilities for dynamic functional connectivity (dFC) and meta‑connectivity analysis, including optimized kernels to compute dFC streams and dFC speed, data I/O helpers, and plotting utilities.

## Installation

Install in editable mode for development:

```bash
pip install -e .
```

## Usage

Core APIs are exposed from `shared_code` after installation. Typical workflow:

1) Build a dFC stream from time series using a sliding window, then 2) compute dFC speed between successive windows.

```python
import numpy as np
from shared_code.fun_dfcspeed import ts2dfc_stream, dfc_speed

T, N = 300, 10
ts = np.random.randn(T, N)  # (timepoints, regions)

# 1) Sliding‑window dFC stream
dfc_2d = ts2dfc_stream(ts, window_size=30, lag=5, format_data='2D')

# 2) dFC speed (supports 'pearson' | 'spearman' | 'cosine')
median_speed, speeds = dfc_speed(dfc_2d, vstep=1, method='pearson')
print(median_speed, speeds.shape)
```

3D dFC input (`roi × roi × frames`) is also supported directly by `dfc_speed`:

```python
dfc_3d = np.random.randn(N, N, 100)
median_speed, speeds = dfc_speed(dfc_3d, vstep=2, method='cosine')
```

## API Highlights

- `fun_dfcspeed.ts2dfc_stream(ts, window_size, lag=None, format_data='2D', method='pearson')`:
  - Builds a dFC stream as lower‑triangular vectors (`'2D'`) or full matrices (`'3D'`).
- `fun_dfcspeed.dfc_speed(dfc_stream, vstep=1, method='pearson', return_fc2=False)`:
  - Vectorized and efficient speed computation; returns median and time series of speeds.
- `fun_optimization.*`:
  - Optimized correlation and speed kernels (`pearson_speed_vectorized`, `cosine_speed_vectorized`, `spearman_speed`).
- `fun_utils.*`:
  - Data I/O helpers (npz/csv), matrix reshape utilities, grouping and plotting helpers.
- `fun_paths.get_paths(...)`:
  - Builds canonical dataset/results/figures paths based on environment variables (see repo `README.md`).

## Testing

Run repository‑level tests that validate this package’s core functions:

```bash
python ../test_unified_dfc_speed.py
python ../test_compatibility.py
```

These scripts generate synthetic data and do not require external datasets.

