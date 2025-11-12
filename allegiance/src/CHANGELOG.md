# Changelog — Allegiance Folder

## 2025-10-01

- Restructured the allegiance pipeline and folder layout.
  - Moved superseded/experimental scripts to `allegiance/src/legacy/` (kept `burst_detection_PBM.py` active).
  - Renamed active entry points for clarity:
    - `1_preprocessed_data_ts_cog_groups.py` → `prep_cog_groups.py`
    - `2_compute_dfc_local.py` → `dfc_compute.py`
    - `run_all_allegiance_local.py` → `allegiance_jobs.py`
    - `merge_allegiance_parallel.py` → `allegiance_merge.py`
    - `coherence_analysis_clean.py` → `cohesion_report.py`
- Added new, focused scripts:
  - `cohesion_compute.py`: computes `time_ratio`, event durations (Parquet), mean/std durations, and burstiness; includes per‑animal event counts and optional binary ATL plots.
  - `cohesion_stats_plot.py`: runs age‑paired (Wilcoxon/t‑test) and group‑based (Mann–Whitney U) stats; exports CSVs and heatmaps with fixed (1−p)×effect color scale `[-0.1, 0.1]`; supports pooled and cross‑age comparisons.
  - `cohesion_playground.py` and `events_playground.py`: notebook helpers with toy data and exploratory plots.
- Event detection switched to a scan‑based implementation for correctness and parity with the toy reference.
- Figures relocated under `paths["f_cohesion"]/{per_animal,stats}`; CSV tables stay under `results`.
- Added `allegiance/src/Makefile` with a default `pipeline` target and HPC‑friendly `RUN` wrapper.
- Created `allegiance/src/make_tutorial.md` and linked from `USAGE_TUTORIAL.md`.
- Updated `USAGE_TUTORIAL.md` to reflect new names, compute/stats flow, and Make targets.

## 2025-09-30

- Added AGENTS.md contributor guide.
- Introduced a cleaned per‑animal analysis script (previously `coherence_analysis_clean.py`, now `cohesion_report.py`).
- Extended stats: age‑paired (2m vs 4m), group‑based options, phenotype toggle, and fixed `[-0.1, 0.1]` color scale for weighted plots.
- Clarified events (contiguous same‑module runs) and burstiness metric ((std−mean)/(std+mean)).
