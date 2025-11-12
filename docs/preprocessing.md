#!/usr/bin/env markdown
# Preprocessing Usage Guide

Centralised CLIs under `scripts/preprocessing/` build canonical bundles for both supported datasets:

- **Julien (`julien_caillette`)** — single-age recordings with optional TR filtering.
- **Ines (`ines_abdullah`)** — paired 2 m / 4 m recordings with phenotype classification and grouping masks.

All entry points ultimately write to `results/<dataset>/preprocessed/` as resolved by `shared_code.fun_paths.get_paths`.

---

## Quick Start

```bash
# Julien dataset (keep only 500‑TR animals; dry run to inspect)
python scripts/preprocessing/preprocess.py --dataset-name julien --only-tr 500 --dry-run

# Ines dataset (override folder mapping; run full write)
python scripts/preprocessing/preprocess.py \
  --dataset-name ines \
  --ines-folder 2mois=Lot3_2mois \
  --ines-folder 4mois=Lot3_4mois
```

The CLI accepts shorthand names: any value starting with `julien` targets `julien_caillette`; `ines*` maps to `ines_abdullah`.

---

## CLI Options

Common flags:

- `--dataset-name {julien|ines}` — select dataset (case-insensitive prefix).
- `--dry-run` — assemble artefacts but skip writing to disk.
- `--log-level` — control verbosity (`DEBUG`..`CRITICAL`).

Julien-specific flags (passed through to `scripts/preprocessing/julien.py`):

- `--filter-mode {exclude_shortest|truncate|none}` — harmonise TS lengths when `--only-tr` is unset.
- `--only-tr N` — keep only animals with exactly `N` time points.
- `--julien-timecourse-folder`, `--julien-cognitive-data-file`, `--julien-cognitive-sheet`, `--julien-anat-labels-file` — override defaults recorded in `shared_code.fun_paths`.

Ines-specific flags (delegated to `scripts/preprocessing/ines.py`):

- `--ines-folder PERIOD=FOLDER` (repeatable) — override default folder mapping (`2mois`, `4mois`).
- `--ines-transient N` — discard `N` initial time points when loading MAT files.
- `--ines-threshold FLOAT` — phenotype threshold passed to `shared_code.fun_utils.classify_phenotypes`.
- `--ines-no-extra-groups` — skip writing exploratory `grouping_data_new.pkl`.
- `--ines-timecourse-folder`, `--ines-cognitive-data-file`, `--ines-anat-labels-file` — override dataset defaults.

---

## Outputs

Successful runs (without `--dry-run`) write under `results/<dataset>/preprocessed/`:

- `ts_and_meta_<dataset>.npz` — canonical bundle consumed by dFC/DFC-speed pipelines. Contains `ts`, `n_animals`, `regions`, `total_tr`, anatomical labels, and dataset metadata. For uniform time lengths, it also includes `mouse_ids`.
- Legacy-compatible NPZs (e.g., `ts_filtered_animals_*`, `ts_and_meta_julien.npz`, `ts_and_meta_2m4m.npz`) when shapes permit; preserved for downstream scripts expecting historical filenames.
- Pickled metadata (`metadata_animals_*.pkl`) and grouping artefacts (`grouping_data_*.pkl` for Ines).
- Filtered cognitive CSV (`cog_data_filtered_*.csv` or `cog_data_sorted_2m4m.csv`).

---

## Integration Notes

- Import helpers directly when scripting:

  ```python
  from scripts.preprocessing.julien import prepare_dataset, write_outputs
  result = prepare_dataset(filter_mode="exclude_shortest", only_tr=500)
  write_outputs(result, dry_run=False)
  ```

- Existing shims (`julien_data/src/preprocess.py`, `metaconnectivity/cognitive_data_ts_sorted.py`) remain for compatibility; they simply call the central modules.
- Downstream consumers (`allegiance/src/dfc_compute.py`, future speed scripts) should load the canonical bundle `ts_and_meta_<dataset>.npz` via `shared_code.fun_utils.load_timeseries_data`. Refer to `docs/dfc_compute.md` for the consolidated DFC CLI.

---

## Known Requirements

- Excel ingestion requires `openpyxl`. Install via `pip install openpyxl`.
- Ines preprocessing emits grouping pickles; ensure downstream code either loads them or documents why they are not needed.
- All scripts rely on correct environment variables (`PATHS_ROOT` or `PROJECT_ROOT_*`) so that `shared_code.fun_paths.get_paths` can resolve datasets and output directories.
