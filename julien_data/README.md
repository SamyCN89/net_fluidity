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
- Group distributions, medians vs window size: `julien_data/speed_plots.py` (wrapper: `julien_data/src/speed_plots_cli.py`)
- Extras in `speed_plots.py`:
  - Equal-animal weighting (KDE averaging or subsample)
  - Cognition correlation: `--cog-scatter` and `--corr-vs-window` (with `--cog-var`, `--reducer`, `--weighting`, `--equalize-length`)
- Cognition (NOR) plots and stats: `julien_data/plot_cog_data.py`
- Community‑wise figures: `julien_data/community_speed_figures.py` (requires `allegiance/communities_wt_veh.pkl`)
- Window pooling and QQ tools: `julien_data/dfc_windows_pooling.py`, `julien_data/local_speed_plot_v2.py`

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

## Legacy Scripts (archived)

The following exploratory or superseded scripts have been moved to `julien_data/legacy/` to reduce clutter. Use the new CLIs where applicable.

- `plts_speed.py`: early cache inspection of DFC/speed outputs (use `speed_plots.py` / `src/speed_plots_cli.py`).
- `plots_speed.py`: jupytext plotting notebook (covered by `speed_plots.py`).
- `3_dfc_local_speed_v1.py`: older speed compute (superseded by `3_dfc_speed_test_v6.py`).
- `local_speed_plot.py`: early per‑region plotting prototype (use `local_speed_plot_v2.py` or `plot_merged_speed.py`).
- `local_speed_plot_v3.py`: duplicate variant of v2 (keep `local_speed_plot_v2.py`).

Kept for now (not fully replicated by CLIs):
- `local_speed_plot_v2.py`: richer plots with equal‑animal weighting.
- `laod_las_speed.py`: community‑specific speed analysis (reads `communities_wt_veh.pkl`).
- `community_speed_figures.py`: community‑wise distributions and stats.
- `dfc_windows_pooling.py`: window pooling utilities.
- `modularity.py`: community stability and allegiance EDA.

## TODOs (next improvements)

- [done] Equal‑animal weighting in speed plots.
- [done] Community‑speed plotting CLI from merged PKL + communities.
- Migrate remaining v2-only plot variants (per-animal detailed summaries) as needed, then archive v2.

## Quick Commands (400‑TR example)

Shortcuts live in `julien_data/Makefile` — run from repo root. Override variables inline.

Default TR is 500. Use `TR=400` to target the 400‑TR subset as shown below.

- Preprocess only 400‑TR:
  - `make -C julien_data preprocess TR=400`

- dFC streams for 400 (choose one):
  - Padded single pass: `make -C julien_data dfc-all TR=400`
  - Per-length split: `make -C julien_data dfc-split TR=400`

- dFC speed (original vs wrapper) and compare:
  - Original: `make -C julien_data speed-orig TR=400 SELECTED="1,4,9" SUBSET_A=orig`
  - Wrapper:  `make -C julien_data speed-wrap TR=400 SELECTED="1,4,9" SUBSET_B=wrap`
  - Compare:  `make -C julien_data compare SUBSET_A=orig SUBSET_B=wrap WIN=9`

- Plot merged outputs (save figures next to merged PKL):
  - `make -C julien_data plot TR=400 PLOT_SUBSET=wrap TAU=0`

- Community plots (per-community distributions):
  - `make -C julien_data community-plot TR=400 PLOT_SUBSET=wrap POOL=all`
  - Use `POOL=short` or `POOL=long` to focus on window pools.

Notes
- All commands rely on `shared_code.fun_paths.get_paths` to locate data/output roots.
- For a different region selection, set `SELECTED="i,j,k"` or omit it to plot all edges.
- You can also call the underlying CLIs directly as shown in `USAGE_TUTORIAL.md`.
