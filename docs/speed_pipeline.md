# Speed Pipeline — Dataset-Agnostic CLIs

This note summarises the consolidated dFC speed workflow now shared between the Julien and Ines datasets. All scripts live under `scripts/dfc/` and `scripts/speed/` (computation) and `scripts/bootstrap/` (analysis/plots).

---

## 1. Generate dFC streams

```bash
python scripts/dfc/dfc_compute.py \
  --dataset-name ines \
  --wmin 5 --wmax 100 --wstep 1 \
  --lag 1 --tau 5 \
  --format 2D \
  --jobs 8 \
  --cache overwrite
```

- `--dataset-name`: accepts aliases (`ines`, `julien`, …) and resolves canonical defaults.
- `--format 2D`: stores the dFC stream as `(animals, pairs, frames)`; choose `3D` for full matrices.
- Files land in `results/<dataset>/dfc/` as `dfc_window_size=<w>_lag=<lag>_tau=<tau>_animals=<n>_regions=<r>.npz`.

## 2. Compute speed bundles

```bash
python scripts/speed/dfc_speed_compute.py \
  --dataset-name ines \
  --subset all \
  --window-min 5 --window-max 100 --window-step 1 \
  --lag 1 --tau-max 5 \
  --jobs 8 \
  --cache overwrite
```

- Reuses the dFC NPZs above; emits per-window speed NPZs under `results/<dataset>/speed/<subset>/`.
- `--tau-max` expands `tau_range=[0..tau_max]`; alternatively provide an explicit comma list via `--tau-range`.
- Supports threaded execution (`--jobs`) and region filtering (`--region-labels`, `--region-indices`).

## 3. Bootstrap CSVs

```bash
python scripts/bootstrap/compute_speed_bootstrap.py \
  --dataset-name ines \
  --subset all \
  --group-cols Genotype,Sexe \
  --pairs "(dKI,F)-(WT,F);(dKI,M)-(WT,M)" \
  --tau-index 3 \
  --pool-threshold median --pool-all \
  --n-boot 500 \
  --jobs 6 --parallel-scope windows \
  --progress
```

- Generates `speed_bootstrap_quantiles.csv`, `speed_bootstrap_diffs.csv`, optional correlations and pool-tests in `results/<dataset>/speed/<subset>/`.
- Group names are derived from the cognitive CSV (case-insensitive match).

## 4. Plotting options

| Script | Purpose | Typical command |
| --- | --- | --- |
| `scripts/bootstrap/plot_speed_bootstrap.py` | Per-window / pooled percentile differences & quantiles | `python scripts/bootstrap/plot_speed_bootstrap.py --dataset-name ines --subset all --plot-diffs-by-win --plot-pooled-diffs` |
| `scripts/bootstrap/plot_speed_distributions.py` | Histograms of pooled speed values split by window threshold | `python scripts/bootstrap/plot_speed_distributions.py --dataset-name ines --subset all --group-cols Genotype,Sexe --pool-threshold median --include-all-pool --bins 40` |
| `scripts/bootstrap/plot_speed_correlations.py` | NOR correlation vs window/pools (requires `--correlate-nor` during compute) | `python scripts/bootstrap/plot_speed_correlations.py --dataset-name julien --subset regions500 --metric spearman --plot-by-win --plot-pooled` |
| `scripts/bootstrap/plot_speed_pooltest.py` | Visualise target vs pooled control percentiles | `python scripts/bootstrap/plot_speed_pooltest.py --dataset-name julien --subset regions500 --bywin --pooled --progress` |

Each plot CLI is read-only: it never recomputes speeds or bootstraps, only consumes existing CSV/NPZ artefacts.

---

## Requirements & Tips

- Install `numba` once (`pip install numba`) to satisfy the shared optimisation kernels.
- Ensure environment variables (`PATHS_ROOT` or `PATHS_ENV` + `PROJECT_ROOT_<ENV>`) and `DATASET_NAME` are configured.
- Use `python scripts/paths_doctor.py --show --check-write --create` to verify path setup.
- The new CLIs handle legacy dFC filenames (`dfc_window_size=5_lag=1_animals=…`) and the tau-tagged variants produced by the modern pipeline.

