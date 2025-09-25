# Julien Data — Quick Guide

Small, task‑oriented notes for running the Julien Caillette dFC analyses that live in this folder. The core, optimized implementations are in `shared_code/`; these scripts orchestrate preprocessing, dFC stream building, dFC speed computation, and plotting.

## Overview
- Preprocess: align raw time series, cognitive data, and region labels.
- dFC streams: build sliding‑window FC streams for a window range.
- dFC speed: compute speed per animal, per window, per tau (and optional per‑region/communities).
- Plots: pool distributions, group comparisons, correlations with cognition.

## Prerequisites
- Python 3.11 recommended.
- Install shared package: `pip install -e shared_code`
- Paths are resolved via `shared_code.fun_paths.get_paths`. Set env vars if needed, e.g.:
  - `PROJECT_ROOT_LOCAL=/absolute/path/to/project/root`
  - Optional dataset switch via `dataset_name` in scripts (defaults to `julien_caillette`).

## Minimal Workflow
1) Preprocess input data
- Script: `julien_data/1_preprocess_data_ts_cog.py`
- Output (under `paths['preprocessed']`):
  - `metadata_animals_{N}_regions_{R}_tr_{T}.pkl`
  - `ts_filtered_animals_{N}_regions_{R}_tr_{T}.npz`
  - `cog_data_filtered_animals_{N}_regions_{R}_tr_{T}.csv`

2) Build dFC streams across windows
- Script: `julien_data/2_compute_dfc_stream.py`
- Uses `shared_code.fun_dfcspeed.get_tenet4window_range(...)`
- Output (under `paths['dfc']`): one NPZ per window, key `dfc` with shape `(n_animals, n_pairs, n_frames)`

3) Compute dFC speed
- Script (global): `julien_data/3_dfc_speed_test_v6.py`
- Script (per‑region): `julien_data/3_dfc_local_speed_v1.py`
- Output (under `paths['speed']`):
  - Per‑window NPZ with `speeds` (object arrays per animal × tau)
  - Consolidated PKL across windows: `speed_windows{W}_tau{T}_animals_{N}.pkl`

4) Plot and analyze
- Group distributions, medians vs window size: `julien_data/plots_speed.py`
- Cognition (NOR) plots and stats: `julien_data/plot_cog_data.py`
- Community‑wise figures: `julien_data/Plot_speed_figures.py` (requires `allegiance/communities_wt_veh.pkl`)
- Window pooling and QQ tools: `julien_data/dfc_windows_pooling.py`, `julien_data/local_speed_plot*.py`

## Common Parameters
- `lag`, `tau`, and `window_parameter=(wmin, wmax, step)` are loaded from metadata or set in scripts.
- Groups are automatically available as `data.groups` from cognitive data `(genotype, treatment) -> indices`.

## Tips
- Heavy steps (streams/speed) support parallelism (`processors=-1`).
- Some arrays are saved with `dtype=object` due to variable lengths per window/tau; downstream scripts handle this by flattening and removing NaNs.
- Prefer the unified functions in `shared_code.fun_dfcspeed` when extending analyses.

For deeper details, see `README.md` at repo root and `julien_data/PROJECT_SUMMARY.md`.

## Tutorial
See `julien_data/USAGE_TUTORIAL.md` for a step‑by‑step guide (preprocess → dFC streams → dFC speed) and CLI examples for region selection and per‑region runs.
