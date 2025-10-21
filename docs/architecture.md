# Architecture — Allegiance/Cohesion Pipeline

This document summarizes the analysis flow implemented under `allegiance/src/`, the shared utilities it depends on, and the main artifacts produced. It focuses on making the pipeline reproducible and easy to navigate.

---

## Overview

- Goal: quantify “cohesion” (allegiance) — how often ROI pairs belong to the same community over time — and analyze effects across age/sex/genotype.
- Inputs: dataset-specific preprocessing CLIs in `scripts/preprocessing/` build canonical bundles (`ts_and_meta_<dataset>.npz`) plus grouping metadata. These feed `allegiance/src/dfc_compute.py` to build dynamic FC streams, followed by allegiance/cohesion stages.
- Outputs: compact NPZ summaries for downstream stats, Parquet/CSV events, and figures under `fig/<dataset>/cohesion/`.
- Determinism: preprocessing, dFC, and cohesion scripts expose CLIs; parameters are recorded in file names and manifest JSON.

---

## Key Modules (allegiance/src)

- `dfc_compute.py`
  - Computes dynamic FC streams per animal over a range of window sizes.
  - Uses `shared_code.fun_dfcspeed.ts2dfc_stream` and paths from `shared_code.fun_paths.get_paths`.
  - Emits `dfc_*.npz` in `paths["dfc"]` with shapes `(animals, regions, regions, frames)` or `(animals, pairs, frames)` depending on `--format`.
  - Refer to `docs/dfc_compute.md` for full CLI usage and caching semantics.

- `allegiance_merge.py`
  - Merges per-window allegiance outputs into a single `merged_allegiance_*.npz` file.
  - Loads DFC to infer number of windows; aggregates community labels (`dfc_communities`), sorting indices (`sort_allegiances`), and contingency matrices per animal/window.
  - Downstream code uses `shared_code.fun_metaconnectivity.load_merged_allegiance` to read this artifact.

- `cohesion_compute.py` (primary producer)
  - Loads merged allegiance and reorders communities per window using `sort_allegiances` to ensure label consistency.
  - Computes per-link “time_ratio” (fraction of windows two ROIs share a module) and a binary ATL cube `(animals × time × links)` for event extraction.
  - Extracts activation events (onset/offset/duration per link), then summarizes mean duration, std duration, and a burstiness proxy per animal/link.
  - Saves: `cohesion_data_w{ws}_lag{lag}_tau{tau}_{scope}[_{tag}].npz` with arrays: `time_ratio`, `mean_duration`, `std_duration`, `burstiness`, `pair_labels`, `anat_labels_sorted`, plus counts; and `events_*.parquet` (+ CSV fallback) and a `manifest_*.json`.
  - ROI scoping: supports `--roi`/`--roi-scope` for all/DMN/memory/custom via indices, label substrings, or file.

- `cohesion_stats_plot.py`
  - Consumes cohesion NPZ to compute statistical summaries and generate multi-panel figures (grouped by Sex/Genotype, with age pairing when requested).
  - Supports within-base paired tests (Wilcoxon) and group-based unpaired tests (Mann–Whitney U), with optional multiple-testing corrections.
  - Emits CSV tables of p-values and effect sizes under `fig/<dataset>/cohesion/stats/` and saves figures.

- `plot_cohesion_curves.py`
  - Convenience plotting for a few selected links (matched by ROI name substrings).
  - Draws per-link 2m vs 4m curves with optional grouping by Sex/Genotype (and Sex×Genotype) and adds significance annotations from in-script tests or CSVs.
  - Saves to `fig/<dataset>/cohesion/link_curves/`.

- Other files
  - `cohesion_report.py`: end-to-end entry that loads merged allegiance, reorders communities, computes simple module-count summaries, and can dispatch cohesion and stats steps. Useful for exploration; the production path is via `cohesion_compute.py` + `cohesion_stats_plot.py`.
  - `burst_detection_PBM.py`, `cohesion_playground.py`, files under `legacy/`: exploratory/legacy analyses; not required for the main pipeline.

---

## Data Flow

0) Raw inputs → canonical bundles
   - `scripts/preprocessing/preprocess.py` dispatches to `scripts/preprocessing/julien.py` or `scripts/preprocessing/ines.py` (based on `--dataset-name`) to align time-series and cognitive metadata, write `ts_and_meta_<dataset>.npz`, and emit grouping artefacts. Shims under `julien_data/src/preprocess.py` and `metaconnectivity/cognitive_data_ts_sorted.py` now import these central modules.
   - See [`docs/preprocessing_checks.md`](preprocessing_checks.md) for quick snippets to inspect the generated outputs.

1) Canonical bundles → DFC streams
   - `scripts/dfc/dfc_compute.py` (also re-exported by `allegiance/src/dfc_compute.py`) loads `ts` from the canonical bundle (default `ts_and_meta_<dataset>.npz`) and computes DFC per animal/window size. Use `--jobs` to parallelise per-animal computation when resources allow. See `docs/dfc_compute.md` for command examples.
   - `scripts/speed/dfc_speed_compute.py` consumes those dFC bundles and generates per-window speed artefacts under `results/<dataset>/speed/`, preserving the legacy subset naming (`all`, `shared`, region folders). Supports ROI filtering via `--region-labels` / `--region-indices`.

2) Per-window allegiance → merged allegiance
   - External step (not shown here) produces per-window allegiance artifacts per animal/window.
   - `allegiance_merge.py` aggregates these into `merged_allegiance_*.npz` with fields:
     - `dfc_communities`: `(animals, windows, regions)` integer labels
     - `sort_allegiances`: `(animals, windows, regions)` indices to reorder labels consistently
     - `contingency_matrices`: `(animals, windows, regions, regions)`

3) Merged allegiance → cohesion summaries
   - `cohesion_compute.py` applies `sort_allegiances` to reorder communities, then for a selected ROI scope:
     - Computes `time_ratio` over upper-triangle links.
     - Builds a binary ATL `(animals, time, links)`, extracts events, and computes duration statistics.
     - Writes NPZ, events Parquet/CSV, counts CSV, and a manifest JSON capturing parameters and shapes.

4) Cohesion summaries → stats and figures
   - `cohesion_stats_plot.py` loads the NPZ and grouping masks to compute stats tables and plots.
  - `plot_cohesion_curves.py` renders focused link-level curves with optional significance bars.

---

## Shared Utilities and Configuration

- `shared_code.fun_paths.get_paths` centralizes filesystem layout and resolves:
  - `preprocessed/` for `ts_and_meta_<dataset>.npz` and `grouping_data_*.pkl`
  - `dfc/` for dFC outputs; `allegiance/` for merged data; `f_cohesion/` for figures
  - Paths are controlled via environment variables per governance (`.env`). Set one of:
    - `PATHS_ROOT` (hard override) **or**
    - `PROJECT_ROOT_<ENV>` (e.g. `PROJECT_ROOT_LOCAL`, `PROJECT_ROOT_CLUSTER`) with optional `PATHS_ENV` selector.
  - `DATASET_NAME` chooses the dataset subfolder (default: `ines_abdullah`); override `timecourse_folder`, `cognitive_data_file`, `anat_labels_file` per dataset when needed.
- Raw-input contract for metaconnectivity:
  - Time courses: `dataset/<DATASET_NAME>/<timecourse_folder>/*.mat` (produced upstream).
  - Cognitive table: `dataset/<DATASET_NAME>/cog_data/<cognitive_data_file>`.
  - Anatomical labels: `dataset/<DATASET_NAME>/cog_data/<anat_labels_file>`.
  - Preprocessing emits `results/<DATASET_NAME>/preprocessed/ts_and_meta_<dataset>.npz` plus grouping pickles.
- `shared_code.fun_loaddata.TimeSeriesBundle` wraps the preprocessed NPZ (time series + metadata) and optional grouping masks, providing `n_animals`, `n_regions`, and label accessors for downstream pipelines.

- `shared_code.fun_metaconnectivity.load_merged_allegiance` reads the merged allegiance bundle and returns `(dfc_communities, sort_allegiances, contingency_matrices)`.

- Tests: smoke tests should cover import/CLI execution for `allegiance/src` entry points and small-shape computations; keep runs deterministic (fixed seeds, cached inputs, headless plots).

---

## Typical Usage

- Preprocess canonical bundles:
  - `python scripts/preprocessing/preprocess.py --dataset-name julien --only-tr 500`
  - `python scripts/preprocessing/preprocess.py --dataset-name ines --folder 2mois=Lot3_2mois --folder 4mois=Lot3_4mois`

- Compute cohesion summaries (DMN example):
  - `python allegiance/src/cohesion_compute.py --window-size 9 --lag 1 --tau 3 --roi dmn --emit all --save-plots --no-show`

- Plot stats/figures from summaries:
  - `python allegiance/src/cohesion_stats_plot.py --window-size 9 --lag 1 --tau 3 --roi-scope dmn --save-plots --no-show`

- Plot selected link curves with annotations:
  - `python allegiance/src/plot_cohesion_curves.py --window-size 9 --lag 1 --tau 3 --roi-scope dmn --roi-substrings "d HIP,v HIP,RSP" --save-plots --no-show`

- Visualise pooled dFC speed distributions (short/long split):
  - `python scripts/bootstrap/plot_speed_distributions.py --dataset-name ines --subset all --group-cols Genotype,Sexe --pool-threshold median --tau-index 0`

All scripts honor the figures root from `get_paths()` and support headless execution.

---

## Notes & Conventions

- Determinism and testability: avoid top-level code with side effects; prefer `main()` + `argparse` + logging.
- Reordering communities per window using `sort_allegiances` is essential before pairwise cohesion calculations.
- Upper-triangle link ordering is consistent across modules via the same index convention.
- Keep new environment variables documented here and in `shared_code/README.md` when APIs/paths evolve.

Docstrings & typing
- Public functions and CLIs expose complete type hints for arguments and return values.
- Use short, focused docstrings: 1–2 lines describing purpose; include parameter meaning when non-obvious and specify array shapes when applicable.
- For statistical helpers, state test and tails; for plotting helpers, state coordinate systems (data vs axes) and units.

Link curves — color-by modes
- `--color-by age`: one trace (2m vs 4m); no live p-values; overall bars omitted unless provided by CSV.
- `--color-by sex`: two traces (Female, Male); per-sex 2m vs 4m bars; within-age pairwise comparisons (Female vs Male) at 2m and 4m.
- `--color-by genotype` or `both`: two traces (wt, dKI); per-genotype 2m vs 4m bars; within-age pairwise comparisons (wt vs dKI).
- `--color-by sex_genotype`: four traces (Female wt, Female dKI, Male wt, Male dKI); per-trace 2m vs 4m bars; within-age pairwise comparisons across all cohorts (e.g., Female wt 2m vs Female dKI 2m, Male wt 2m, Male dKI 2m; likewise at 4m).

Scientific note — event extraction:
- Cohesion “events” are contiguous runs where a link is active (same-module=1). We now extract events using a vectorized diff method (pad→diff→pair onsets/offsets) rather than a Python scan loop. This preserves scientific semantics (onset inclusive, offset exclusive, duration=offset−onset) and was verified to produce identical results across randomized tests. The change improves performance without altering outputs.
