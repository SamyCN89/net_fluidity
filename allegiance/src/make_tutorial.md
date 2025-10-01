# Makefile Tutorial — Pipeline Shortcuts

This guide explains the provided Makefile targets to run the DFC → Allegiance → Cohesion pipeline quickly. Override variables inline, e.g., `make dfc WS=7`.

## Variables (override with VAR=value)
- `WS`: window size (default 9)
- `LAG`: lag between windows (default 1)
- `TAU`: tau tag in filenames (default 3)
- `DMN`: region subset in sorted label space; empty string means all regions. Example: `DMN="0,23,13,22,2,28,34,37,39,8,35"`
- `N_JOBS`: parallel jobs for allegiance (default 8)
- `STATS_MODE`: `age|group|all` for stats
- `GROUP`: `sex|genotype|both` for group-based stats
- `ALPHA`: significance threshold (default 0.05)
- `CROSS_AGE`: `1` to include cross-age pairs; `0` otherwise
- `POOL_AGES`: `1` to pool ages (Female vs Male ignoring age); `0` otherwise

## Targets
- `make pipeline`: runs `prep → dfc → allegiance-jobs → allegiance-merge → cohesion-compute → cohesion-stats`
- `make help-pipeline`: brief usage of targets and variables
- `make prep`: preprocessing + grouping (writes `preprocessed/` artifacts)
- `make dfc`: compute DFC streams (3D) for a single `WS`
- `make allegiance-jobs`: run per-window allegiance in parallel (uses `WS` and `LAG`)
- `make allegiance-merge`: merge slice outputs into a single NPZ
- `make cohesion-compute`: compute cohesion summaries (time_ratio, durations, burstiness) and events (Parquet)
- `make cohesion-stats`: run stats (age/group) and save tables/heatmaps
- `make cohesion-report`: quick per-animal plots (communities & module counts)

## Examples
- Compute DFC (WS=9, LAG=1):
  - `make dfc WS=9 LAG=1 TAU=3`
- Run allegiance jobs (8 cores):
  - `make allegiance-jobs WS=9 LAG=1 N_JOBS=8`
- Merge allegiance outputs:
  - `make allegiance-merge`
- Compute cohesion (all regions, keep short events):
  - `make cohesion-compute WS=9 LAG=1 TAU=3 DMN=""`
- Stats: group-based Sex + Genotype within-age, save heatmaps:
  - `make cohesion-stats WS=9 LAG=1 TAU=3 DMN="" STATS_MODE=group GROUP=both`
- Stats: pooled over ages for Sex only:
  - `make cohesion-stats WS=9 LAG=1 TAU=3 DMN="" STATS_MODE=group GROUP=sex POOL_AGES=1`
- Quick report (per-animal):
  - `make cohesion-report WS=9 LAG=1 TAU=3`

## Outputs
- DFC: `results/<dataset>/dfc/dfc_window_size=..._lag=..._tau=..._animals=..._regions=....npz`
- Allegiance temp: `results/<dataset>/allegiance/temp/*.npz` (per animal×window)
- Allegiance merged: `results/<dataset>/allegiance/merged_allegiance_*.npz`
- Cohesion compute: `results/<dataset>/allegiance/cohesion_data/`
  - `cohesion_data_w{WS}_lag{LAG}_tau{TAU}_{scope}.npz` (arrays)
  - `events_w{WS}_lag{LAG}_tau{TAU}_{scope}.parquet` (requires `pyarrow`)
  - `events_count_w{WS}_lag{LAG}_tau{TAU}_{scope}.csv` (per-animal counts)
- Figures: `fig/<dataset>/cohesion/`
  - `per_animal/` (binary maps and report plots)
  - `stats/` (significance and weighted heatmaps)

## Tips

### HPC usage
- Set `RUN` to your launcher to run on a compute node, e.g., `RUN="srun -n 1"` or `RUN="srun --cpu-bind=cores -c 4"`.
- Examples:
  - `make -C allegiance/src pipeline RUN="srun -n 1" WS=9 LAG=1`
  - `make -C allegiance/src cohesion-stats RUN="srun -n 1" STATS_MODE=group GROUP=both`
- For large-scale allegiance across many windows/animals, consider using `allegiance_jobs.py` as a template for job arrays if your scheduler supports them.
- Install `pyarrow` for Parquet: `pip install pyarrow` (or Conda equivalent)
- Use `DMN=""` to include all regions; provide a comma list to restrict
- Keep `WS`/`LAG` consistent across steps; `TAU` is a label in filenames
