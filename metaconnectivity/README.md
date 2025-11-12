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

## Loading Preprocessed Cognitive Data

Preprocessed cognitive metadata lives under the dataset-specific `preprocessed` directory returned by `shared_code.fun_paths.get_paths`. Use the shared loaders to keep path handling consistent:

```python
from shared_code.fun_loaddata import load_timeseries_bundle
from shared_code.fun_utils import load_cognitive_data
from shared_code.fun_paths import get_paths

paths = get_paths(
    dataset_name="ines_abdullah",
    timecourse_folder="Timecourses_updated_03052024",
    cognitive_data_file="ROIs.xlsx",
    anat_labels_file="41_Allen.txt",
)

# Bundled time series + masks + metadata
bundle = load_timeseries_bundle(
    paths["preprocessed"] / "ts_and_meta_2m4m.npz",
    paths["preprocessed"] / "grouping_data_oip.pkl",
)

# Preprocessed cognitive scores
cog_data = load_cognitive_data(
    paths["preprocessed"] / "cog_data_sorted_2m4m.csv"
)
```

`bundle.metadata` exposes derived values such as `total_tr`, `anat_labels`, and `is_2month_old` that older scripts used to fetch from the NPZ manually.

## Legacy Cleanup Plan

These scripts remain for historical reference but are slated for removal once all workflows are migrated to `shared_code` and the modern `metaconnectivity/` entry points. Remove them after confirming no active analyses depend on them:

- `metaconnectivity/old_useful/compute_metaconnectivity.py` — duplicate of the modern modularity pipeline; still hard-codes host paths and parallelises logic already covered (with tests) in `shared_code`.
- `metaconnectivity/old_useful/compute_metaconnectivity_allegiance.py` — legacy variant of the allegiance workflow that writes to external disks; superseded by `metaconnectivity/compute_metaconnectivity_modularity.py`.
- `metaconnectivity/old_useful/compute_genuine_trimers.py` — prototype for trimer analysis now handled by `metaconnectivity/compute_trimers.py` plus `shared_code.fun_metaconnectivity.compute_trimers_genuine`.
- `metaconnectivity/old_useful/master_mc.py` and `Consolidate_data.py` — orchestration scaffolds that rebuild the entire pipeline with hard-coded directories; prefer scripted entry points or Make targets.
- dFC prototypes (`dfc_streams*.py`, `dfc_metaconnectivity.py`, `dfc_windows_pooling.py`, `1.0dfc_speed_data.py`, `fc_individuals.py`) — rely on handwritten loaders and duplicated kernels; modern equivalents exist in `shared_code.fun_dfcspeed`.
- Legacy module copies (`old_useful/fun_dfcspeed.py`, `fun_metaconnectivity.py`, `fun_optimization.py`) — logic already lives in `shared_code`; keeping clones risks divergence.
- Exploratory allegiance scripts (`metaconnectivity_allegiance_matrix_test3.py`, `test4.py`) — consume the same caches without validation; archival only.
- Data snapshots under `metaconnectivity/old_useful/` (`Behaviour_exclusions_ROIs_female.xlsx`, `data.h5`, `data.json`) — relocate outside the repo or document provenance under `reports/` to comply with governance.

Track progress in `metaconnectivity_todo.md` and update this list as soon as each dependency is retired.
