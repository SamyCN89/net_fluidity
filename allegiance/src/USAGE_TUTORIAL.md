# Allegiance Scripts — Usage Guide

## Overview
- Scripts under `allegiance/src/` implement a pipeline to compute dynamic FC (DFC), run per-window community detection (“allegiance”), merge results, and perform group-level analyses and plots.
- Core utils (DFC stream, paths, loaders) live in `shared_code/` and are imported by these scripts.

## Setup
- Python 3.11. Install local package: `pip install -e shared_code`.
- Optional: set `PROJECT_DATA_ROOT` to point at your data root. Scripts default to `timecourse_folder="Timecourses_updated_03052024"`.
- Artifacts are created under folders returned by `shared_code.fun_paths.get_paths()` (e.g., `preprocessed/`, `dfc/`, `allegiance/`).

## Pipeline
1) Preprocess and group metadata
- Run: `python allegiance/src/prep_cog_groups.py`
- Outputs (`preprocessed/`):
  - `ts_and_meta_2m4m.npz` (time series, labels),
  - `grouping_data_oip.pkl`, `grouping_data_per_sex(gen_phen).pkl`, `grouping_data_new.pkl`.

2) Compute DFC streams (sliding-window FC)
- Run: `python allegiance/src/dfc_compute.py --format 3D --wmin 5 --wmax 30 --wstep 5 --lag 1 --tau 3`
- Output: `dfc/dfc_window_size={WS}_lag={LAG}_tau={TAU}_animals={A}_regions={N}.npz` (key: `dfc`).

3) Allegiance per animal × window (parallel)
- Run: `python allegiance/src/allegiance_jobs.py --n_jobs 8 --window_size 9 --lag 1`
- Output: `allegiance/temp/dfc_window_size=..._animal_XX_window_YYYY.npz` (communities, sort indices, contingency per slice).

4) Merge allegiance results
- Run: `python allegiance/src/allegiance_merge.py`
- Output: `allegiance/merged_allegiance_window_size=..._lag=..._animals=..._regions=....npz`
- Load via: `from shared_code.fun_metaconnectivity import load_merged_allegiance`.

5) Analysis and reports
- Clean entry point: `python allegiance/src/cohesion_report.py --window-size 9 --lag 1 --tau 3 --save-plots --no-show`
  - Quick plots: community matrix and module-count time series.
  - Cohesion map: add `--compute-cohesion` (same-module=1 over time per link).
  - Events/burstiness: add `--compute-events` (extract on/off durations and plot burstiness).
  - Stats: add `--with-stats` plus controls:
    - `--stats-mode {age,group,all}`: age-paired (2m vs 4m within base), group-based (independent), or both.
    - `--group-compare {sex,genotype,both}`: choose dimensions for group-based.
    - `--cross-age`: include cross-age comparisons (e.g., Female-2m vs Male-4m).
    - `--pool-ages`: add pooled comparisons over ages (e.g., Female vs Male ignoring age).
    - `--include-phenotype {none,oip,nor,both}`: optionally add OiP/NOR to age-paired stats (default none).
    - Outputs: CSVs and heatmaps saved under `allegiance/out/`.
  - Scope: restrict to DMN with `--dmn-index 0,23,13,22,2,28,34,37,39,8,35`, or all regions with `--dmn-index ""`.
- Legacy exploratory scripts: `allegiance_per_animal_v2.py`, `plot_modules_stability.py`, `coherence_analysis.py`.

## CLI Reference (clean analysis)
- Common flags: `--save-plots`, `--no-show`, `--animal <idx>`, `--alpha 0.05`.
- Outputs:
  - Figures: `fig/<dataset>/cohesion/per_animal/*.png` (per-animal) and `fig/<dataset>/cohesion/stats/*.png` (stats heatmaps).
  - Tables (when `--with-stats`): CSVs under `allegiance/out/`:
    - `pvals_wilcoxon_*.csv`, `pvals_ttest_*.csv`, `effects_cohesiondiff_*.csv`.
    - Heatmaps saved alongside with matching tags.

## Common Comparison Commands
- Sex within-age (Female-2m vs Male-2m; Female-4m vs Male-4m):
  - `python allegiance/src/cohesion_report.py --with-stats --stats-mode group --group-compare sex --save-plots --no-show`
- Genotype within-age (dKI-2m vs wt-2m; dKI-4m vs wt-4m):
  - `python allegiance/src/cohesion_report.py --with-stats --stats-mode group --group-compare genotype --save-plots --no-show`
- Both Sex and Genotype within-age:
  - `python allegiance/src/cohesion_report.py --with-stats --stats-mode group --group-compare both --save-plots --no-show`
- Pooled over ages (Female vs Male, dKI vs wt ignoring age):
  - `python allegiance/src/cohesion_report.py --with-stats --stats-mode group --group-compare both --pool-ages --save-plots --no-show`
- Cross-age pairs (e.g., Female-2m vs Male-4m):
  - `python allegiance/src/cohesion_report.py --with-stats --stats-mode group --group-compare sex --cross-age --save-plots --no-show`
- Age-paired within base (2m vs 4m for Sex/Genotype):
  - `python allegiance/src/cohesion_report.py --with-stats --stats-mode age --include-phenotype none --save-plots --no-show`

## Notes & Compatibility
- Logging: set `NET_FLUIDITY_LOGGING=config/logging.yaml` for structured logs.
- Environments: some scripts accept `--env`/`--data_root` (see `run_all_allegiance_local.py`).
- DFC file formats:
  - New (2_compute_dfc_local): `dfc/… .npz` with key `dfc` and `..._tau=...` in filename.
  - Older (expected by run_all_allegiance_local): `mc/… .npz` with key `dfc_stream`.
  - Keep flows consistent, or regenerate in the matching format to avoid mismatches.
  - Weighted heatmaps use a fixed color scale of `[-0.1, 0.1]` for (1−p)×effect.

## Why Mann–Whitney for Male vs Female?
- Male vs Female (and wt vs dKI) compare independent animal groups, not repeated measures, so paired Wilcoxon is invalid.
- Mann–Whitney U tests differences in central tendency without assuming normality; it tolerates unequal group sizes.
- Cautions: unbalanced groups, differing variance/shape, and within-subject dependence across ages. Mitigations: run within-age comparisons and clearly label cross-age results when `--cross-age` is enabled.

## Effect Sizes
- Mean difference (mdiff): μY − μX per link, where μ is the group/condition mean of the cohesion time ratio (fraction of windows two regions share a module). Units: fraction of time (−1…+1).
- Cohesion-diff ratio (cdratio): (μY − μX) / (μY + μX); normalizes by overall magnitude to compare links with different baselines. Symmetric and bounded near [−1, 1]; guarded with a small ε when denominator is near zero.

## Troubleshooting
- “Merged allegiance not found”: ensure step 3 completed and then re-run step 4.
- Shape mismatches in stats: verify group files exist in `preprocessed/` from step 1.
- Headless servers: add `--no-show` to analysis commands and rely on saved figures.

## Cohesion Compute & Stats (separate scripts)
- Compute (summaries + events, Parquet):
  - `python allegiance/src/cohesion_compute.py --window-size 9 --lag 1 --tau 3 --dmn-index "" --min-duration 2`
  - Saves NPZ + Parquet under `results/<dataset>/allegiance/cohesion_data/`.
  - Parquet requires `pyarrow` (`pip install pyarrow`); falls back to CSV if missing.
- Stats + plots:
  - `python allegiance/src/cohesion_stats_plot.py --with-stats --stats-mode all --group-compare both --dmn-index "" --save-plots --no-show`
  - Tables: `results/<dataset>/allegiance/out/`
  - Figures: `fig/<dataset>/cohesion/stats/`

## Active Scripts and Legacy Cleanup
- Active (keep):
  - `allegiance/src/prep_cog_groups.py` — preprocessing and grouping
  - `allegiance/src/dfc_compute.py` — DFC stream computation
  - `allegiance/src/allegiance_jobs.py` — per-window allegiance jobs (parallel)
  - `allegiance/src/allegiance_merge.py` — merge allegiance outputs
  - `allegiance/src/cohesion_report.py` — quick per-animal report (communities, modules)
  - `allegiance/src/cohesion_compute.py` — compute summaries + events (NPZ + Parquet)
  - `allegiance/src/cohesion_stats_plot.py` — stats + heatmaps from summaries
  - `allegiance/src/cohesion_playground.py`, `allegiance/src/events_playground.py` — notebook helpers

- Proposed to move to `allegiance/src/legacy/` (superseded/experimental):
  - `allegiance/src/coherence_analysis.py` — superseded by `cohesion_report.py`
  - `allegiance/src/allegiance_per_animal.py`, `allegiance/src/allegiance_per_animal_v2.py`
  - `allegiance/src/plot_modules_stability.py`
  - `allegiance/src/compute_allegiance_local.py`
  - `allegiance/src/burst_detection_PBM.py`
  - `allegiance/src/test_plt.py`
  - `allegiance/src/alignment_temporal_consensus_method/` (folder)

- Optional renames (for clarity; not yet applied):
  - `1_preprocessed_data_ts_cog_groups.py` → `prep_cog_groups.py`
  - `2_compute_dfc_local.py` → `dfc_compute.py`
  - `run_all_allegiance_local.py` → `allegiance_jobs.py`
  - `merge_allegiance_parallel.py` → `allegiance_merge.py`
  - `coherence_analysis_clean.py` → `cohesion_report.py`


## Make Targets
- A Makefile is provided with pipeline shortcuts. See `allegiance/src/make_tutorial.md` for full details.
