# metaconnectivity

Analysis scripts and functions for meta‑connectivity and dFC speed used during research and prototyping. Many newer, unified utilities live under `shared_code/`; this folder keeps earlier implementations and analysis pipelines for reproducibility and reference.

## Contents

- `fun_dfcspeed.py`: legacy dFC stream and dFC speed routines used in early experiments
- `compute_*` scripts: meta‑connectivity and modularity analyses
- `deprecated_fun.py`: older variants retained for reference
- `master_mc.py`: pipeline scaffold for meta‑connectivity workflows

## Recommended Usage

- Prefer the optimized, unified functions from `shared_code`, e.g. `shared_code.fun_dfcspeed.dfc_speed` and `ts2dfc_stream`.
- Use the modules here to reproduce prior results or for exploratory analyses. When migrating code, consider swapping to the `shared_code` APIs for speed and feature parity.

## Minimal Example (legacy variant)

```python
import numpy as np
from metaconnectivity.fun_dfcspeed import ts2dfc_stream, dfc_speed

T, N = 300, 10
ts = np.random.randn(T, N)

dfc_2d = ts2dfc_stream(ts, window_size=30, lag=5, format_data='2D')
median_speed, speeds = dfc_speed(dfc_2d, vstep=1)
```

For a richer, method‑selectable implementation (Pearson/Spearman/Cosine) and optional FC2 returns, use the `shared_code` version.
