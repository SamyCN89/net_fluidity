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
- `julien_data/`: demos, plots, and figures used during development
- `test_*.py`: quick tests and comparisons between implementations

## Installation

Python 3.11 is recommended. Optionally create a virtual environment, then install the shared package in editable mode:

```bash
pip install -e shared_code
```

Optional: environment variables for data paths used by `fun_paths` can be placed in a local `.env` file or your shell. For example:

```bash
# used by shared_code/shared_code/fun_paths.py
PROJECT_ROOT_LOCAL=/absolute/path/to/project/root
DATASET_NAME=ines_abdullah
```

If you use `fun_paths.get_paths`, it will read `PROJECT_ROOT_<ENV>` (default `LOCAL`) and build a standard folder layout under `dataset/`, `results/`, and `fig/` within that root.

## Quick Start

### Preprocess datasets

Run the central CLI (shorthand dataset names supported):

```bash
# Julien dataset (500-TR animals only)
python scripts/preprocessing/preprocess.py --dataset-name julien --only-tr 500

# Ines dataset with custom folder mapping
python scripts/preprocessing/preprocess.py \
  --dataset-name ines \
  --ines-folder 2mois=Lot3_2mois \
  --ines-folder 4mois=Lot3_4mois
```

This writes canonical bundles (`ts_and_meta_<dataset>.npz`) under `results/<dataset>/preprocessed/`, plus legacy-compatible artefacts. See `docs/preprocessing.md` for full usage.

### Compute dynamic FC streams

Use the consolidated CLI (now with flexible parallel backends). See `docs/dfc_compute.md` for full parameter reference and troubleshooting tips.

```bash
python scripts/dfc/dfc_compute.py \
  --dataset-name julien \
  --wmin 5 --wmax 20 --wstep 5 \
  --lag 1 --tau 5 --format 3D \
  --jobs 32 --parallel-backend thread --batch-per-animal
```

Parallel tuning cheatsheet:
- `--jobs N` controls worker count (set `N>1` to enable concurrency).
- `--parallel-backend thread|process` chooses thread vs process executors (use `process` if the GIL becomes a bottleneck; default is `thread`).
- `--chunksize K` hints how many tasks each process grabs per batch (useful only with the process backend; try 2–4 on large nodes).
- `--batch-per-animal` gives each worker one animal at a time and computes all requested windows before returning—best when sweeping many window sizes or running on >32 cores.

### Compute dFC speed

The new shared CLI reads the dFC bundles above and writes per-window speed arrays for both datasets:

```bash
python scripts/speed/dfc_speed_compute.py \
  --dataset-name ines \
  --subset-name all \
  --window-min 5 --window-max 15 --window-step 5 \
  --lag 1 --tau-max 3 \
  --jobs 4
```

Specify `--region-labels "ACC,PL"` or `--region-indices 0,4` to focus on subsets, and use `--dry-run` to inspect planned outputs. Artefacts appear under `results/<dataset>/speed/<subset>/`.

Full pipeline guidance (dFC → speed → bootstrap/plots) lives in `docs/speed_pipeline.md` and `docs/Compute_and_Plot_Tutorial.md`.

Bootstrap compute/plot (recommended flow)

```bash
# 1) Configure paths (env vars or .env) and install shared package
pip install -e shared_code
export PATHS_ROOT=/abs/path/to/project/root
export DATASET_NAME=julien_caillette

# 2) Inspect/prepare paths and context
make speed-doctor

# 3) Compute CSVs (quantiles, diffs, optional correlations)
make speed-compute TR=500 SUBSET=regions500 TAU_INDEX=0 N_BOOT=2000 JOBS=8 PAIR_SCOPE=windows

# 4) Plot from CSVs
make speed-plot TR=500 SUBSET=regions500

# Inspect pooled speed distributions (splits windows by median into short/long)
python scripts/bootstrap/plot_speed_distributions.py \
  --dataset-name ines \
  --subset all \
  --group-cols Genotype,Sexe \
  --tau-index 0 \
  --pool-threshold median \
  --plot-format png \
  --include-all-pool \
  --bins 40

# Optional: pool-tests or correlations (if computed)
make speed-pooltest TR=500 SUBSET=regions500
make speed-cor TR=500 SUBSET=regions500
```

Core API example (compute dFC stream and speed directly):

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

## Makefile Targets

- `make help-speed`: lists dFC speed bootstrap targets and variables.
- `make speed-doctor`: shows and prepares paths; runs context checks.
- `make speed-compute`: runs `scripts/compute_speed_bootstrap.py` with common defaults (override variables like `TR`, `SUBSET`, `N_BOOT`).
- `make speed-plot`: plots from CSVs via `scripts/plot_speed_bootstrap.py`.
- `make speed-pooltest`: renders pool-test figures if `speed_bootstrap_pooltest*.csv` exists.
- `make speed-cor`: renders correlation figures if `speed_nor_correlations.csv` exists.

## Notes on Implementations

- Prefer the unified `shared_code.fun_dfcspeed.dfc_speed` which supports Pearson/Spearman/cosine and returns optional FC2 streams. Earlier variants exist under `metaconnectivity/` and `julien_data/` for reference and benchmarking.
- Dynamic FC streams for both datasets should now be generated via `scripts/dfc/dfc_compute.py` (aliased by `allegiance/src/dfc_compute.py`). Legacy scripts in `julien_data/` and `metaconnectivity/old_useful/` are being retired in favour of this shared implementation.

## Phase 3 Workflow

The Julien dataset flow now has a stabilized Phase 3 path with a thin CLI wrapper, a dataset context, notebook helpers, and a bootstrap CLI.

- Tutorial: see `julien_data/USAGE_TUTORIAL.md` (env, preprocess → streams → speed, notebooks, and bootstrap examples).
- Notebook Tutorial (Bootstrap CLI): `docs/Bootstrap_Speed_CLI_Tutorial.md` (end‑to‑end usage, parameters, outputs, troubleshooting).
- Compute Speed (wrapper): `python julien_data/src/speed_compute.py --tr 500 --processors -1`
  - Per‑region outputs: add `--per-region`.
  - Region selection by labels: `--selected-region-labels "ACC,THAL"` and `--region-mode within|touching`.
  - Shared engine (parity checks): `--engine shared`.
- Notebook helpers: `scripts/speed_bootstrap_nb.py`
  - Load all regions/windows: `load_all_speeds_by_region_nb(...)`
  - Pool windows: `pool_short_long_nb(...)`
  - Bootstrap quantiles, diffs, and plotting utilities.
- Bootstrap CLI (CSV + figures):
  ```bash
  python scripts/bootstrap_speed_groups_cli.py \
    --tr 500 --subset shared --tau-index 0 \
    --n-boot 2000 --seed 0 --ci 95 \
    --pool-threshold median --pool-all \
    --plot --grid --grid-cols 2 --progress \
    --jobs 4 --parallel-scope windows \
    --outdir j500_t0 --append-subset-to-outdir --load-cache
  ```
  - Outputs: CSVs under `paths['speed']/<outdir>/*.csv` and figures under `paths['f_speed']/<outdir>/*.<fmt>`.
  - If `--outdir` is omitted, it defaults to `--subset` (or `bootstrap` if no subset).

### Centralized Bootstrap Kernels (Phase 3)

- All bootstrap kernels are centralized in `shared_code/shared_code/fun_bootstrap.py`.
  - Use these from scripts and notebooks; avoid local duplicates.
  - Supports adaptive early stop, vectorized chunking, and memory controls.

- Compute CSVs with `scripts/compute_speed_bootstrap.py` (compute‑only):
  - Important flags:
    - `--reuse-group-boots`: compute per‑group bootstrap replicates once; derive all pairs from them (major speedup for many pairs).
    - `--chunk INT`: batch size for vectorized resampling (throughput vs memory).
    - `--boots-float32`: store bootstrap arrays in float32.
    - `--values-float32`: cast pooled values to float32 before resampling (reduces batch memory).
    - `--index-int32`: use int32 index arrays for resampling (reduces index memory).
    - `--blas-threads INT`: limit BLAS threads per worker (defaults to 1 when `--jobs > 1`).
  - Example (fast + moderate memory):
    ```bash
    python scripts/compute_speed_bootstrap.py \
      --tr 400 --subset regions400 --tau-index 0 \
      --n-boot 2000 --reuse-group-boots \
      --chunk 256 --boots-float32 \
      --jobs 8 --parallel-scope windows --progress
    ```
  - Example (lower memory):
    ```bash
    python scripts/compute_speed_bootstrap.py \
      --tr 400 --subset regions400 --tau-index 0 \
      --n-boot 2000 --reuse-group-boots \
      --chunk 128 --boots-float32 --values-float32 --index-int32 \
      --jobs 4 --parallel-scope windows --progress
    ```

## Docs

- Usage Tutorial: `julien_data/USAGE_TUTORIAL.md`
- Bootstrap Speed CLI Tutorial: `docs/Bootstrap_Speed_CLI_Tutorial.md`
- Compute/Plot Split Tutorial (compute CSVs, then plot): `docs/Compute_and_Plot_Tutorial.md`
- Batch runner: `scripts/run_bootstrap_batches.sh` (compute/plot/list/dry‑run over many subsets)

## Contributing

- Follow `shared_code/SCIENTIFIC_CODING_GUIDELINES.md` for style and scientific computing practices.
- Add or update docstrings for public functions; include parameter and return types where possible.
- When adding features, include a small usage snippet or expand the relevant README section.

## Citation

Dynamic Functional Connectivity as a complex random walk: Definitions and the dFCwalk toolbox. Lucas Arbabyazd, Diego Lombardo, Olivier Blin, Mira Didic, Demian Battaglia, Viktor Jirsa. MethodsX (2020) doi: 10.1016/j.mex.2020.101168.
