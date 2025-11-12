# Dataset Context — DFCAnalysis

This guide documents the repository’s single source of truth (SoT) for dataset paths and cognitive metadata used by the speed bootstrap and plotting scripts: `src/net_fluidity_julien/context.py` with the `DFCAnalysis` class.

## Purpose

- Centralize dataset configuration (roots, dataset name) and derive canonical paths for inputs, results, and figures.
- Provide accessors for preprocessed time series/metadata and cognitive data used for grouping and correlations (e.g., NOR).
- Serve as the SoT for compute/plot scripts, avoiding ad‑hoc path assembly scattered across the codebase.

## Lifecycle

1) Construct and configure
- Import and instantiate inside scripts or notebooks:
```python
from net_fluidity_julien.context import DFCAnalysis
ctx = DFCAnalysis()
```
- The class reads environment variables to resolve roots:
  - `PATHS_ROOT` (hard override), or `PATHS_ENV` + `PROJECT_ROOT_<ENV>`
  - `DATASET_NAME` (dataset subfolder)

2) Resolve paths and metadata
- Call `get_metadata()` once to populate metadata and internal paths:
```python
meta = ctx.get_metadata()  # returns a dict/DataFrame bundle as implemented
paths = ctx.paths          # dict-like with keys used across scripts
```
- Expected keys typically include: `preprocessed`, `speed`, `dfc`, `f_speed`, and related dataset folders.

3) Access cognitive data
- Cognitive CSV (grouping + behavioral scores) is loaded via a dedicated accessor, e.g. `get_cognitive_df()`:
```python
cog = ctx.get_cognitive_df()  # pandas DataFrame
```
- Grouping columns (defaults in scripts: `genotype,treatment`) and behavioral columns (e.g., `index_NOR`) live here. Scripts validate presence.

4) Consume in compute/plot scripts
- Scripts import `DFCAnalysis` as SoT and avoid any legacy fallbacks. Examples:
  - `scripts/compute_speed_bootstrap.py` uses `ctx.paths['speed']` for outputs, reads grouping columns from `cog`, and (optionally) a NOR column via `--nor-col`.
  - `scripts/plot_speed_bootstrap.py` and correlation/pool‑test plotters use the same `paths` to find CSVs and write figures under `paths['f_speed']`.

## Schema (expected fields)

- `paths: Dict[str, str|Path]`
  - `preprocessed`: folder with preprocessed NPZ (time series + meta)
  - `speed`: root for speed-related CSV outputs (per subset/outdir)
  - `dfc`: root for DFC streams (if used by upstream steps)
  - `f_speed`: figures root for speed plots
  - Optional dataset-specific keys are allowed; scripts only rely on the above.

- `metadata`
  - A minimal structure exposing TRs, tau counts, window sizes, and ROI labels as available. Scripts primarily need TR and tau count to interpret `--tr` and `--tau-index`.

- `cognitive_df: pandas.DataFrame`
  - Index/ID column matching animals in NPZs
  - Grouping columns (e.g., `genotype`, `treatment`)
  - Behavioral columns (e.g., `index_NOR`) for correlations

## Examples

Programmatic usage
```python
from net_fluidity_julien.context import DFCAnalysis
ctx = DFCAnalysis()
_ = ctx.get_metadata()        # populate paths/metadata
cog = ctx.get_cognitive_df()  # DataFrame

# Grouping columns discovery (fallback to defaults if not passed by CLI)
group_cols = [c for c in ('genotype', 'treatment') if c in cog.columns]
if not group_cols:
    raise RuntimeError('No default grouping columns found in cognitive CSV')

# NOR column selection
nor_col = 'index_NOR' if 'index_NOR' in cog.columns else None

print('paths.speed:', ctx.paths['speed'])
print('cog columns:', cog.columns.tolist()[:8])
```

CLI integration pattern (compute)
```python
# inside scripts/compute_speed_bootstrap.py
from net_fluidity_julien.context import DFCAnalysis
ctx = DFCAnalysis()
ctx.get_metadata()
cog_df = ctx.get_cognitive_df()

# Resolve outputs and validate columns
out_root = Path(ctx.paths['speed']) / args.outdir
out_root.mkdir(parents=True, exist_ok=True)

for col in args.group_cols:
    if col not in cog_df.columns:
        raise SystemExit(f"Missing group column: {col}")

if args.correlate_nor:
    nor_col = args.nor_col or 'index_NOR'
    if nor_col not in cog_df.columns:
        raise SystemExit(f"Missing NOR column in cognitive CSV: {nor_col}")
```

## Validation

Use `paths_doctor.py` to check context configuration and cognitive CSV readiness:
```bash
python scripts/paths_doctor.py --show --check-write --create
python scripts/paths_doctor.py --check-context  # validates cognitive CSV, group cols, and NOR column
```

## Data Standardization

This repo operates cleanly once “preprocessed” artifacts exist. For workflows that start from raw data, standardize inputs so `DFCAnalysis` and `shared_code.fun_paths` can discover them consistently:

- Raw cognitive data (optional for bootstrap): historically `dataset/<DATASET_NAME>/cog_data/ROIs.xlsx` in `fun_paths` defaults; `DFCAnalysis` uses `mice_groups_comp_index_2.xlsx`. If you rely on raw readers, ensure the expected file is present or adjust filenames in your local context.
- Preprocessed cognitive CSV (required for bootstrap): `results/<DATASET_NAME>/preprocessed_data/cog_data_filtered_animals_<N>_regions_<R>_tr_<TR>.csv` with, at minimum:
  - Grouping columns: `genotype`, `treatment` (or override via `--group-cols`).
  - Behavioral metric for correlations: `index_NOR` (or pass `--nor-col`).
- Metadata pickle: `metadata_animals_<N>_regions_<R>_tr_<TR>.pkl` with keys: `mouse_metadata`, `region_labels`, `n_animals`, `regions`, `total_tr`, `lag`, `tau`, `window_range`.
- Time series NPZ: `ts_filtered_animals_<N>_regions_<R>_tr_<TR>.npz` containing `ts`.

Tip: use `paths_doctor.py` to check directory creation/writability and cognitive CSV readiness. It will warn about missing raw files (e.g., `ROIs.xlsx`) even if your current workflow only needs preprocessed CSV — this is informational and can be ignored for bootstrap.

## Notes

- Keep `DFCAnalysis` backwards-compatible for path keys consumed by scripts; evolve with additive changes when possible.
- Prefer reading environment settings once in `DFCAnalysis` rather than in each script to avoid drift.
- When adding new outputs, extend `paths` and document in this guide and the relevant tutorials.
