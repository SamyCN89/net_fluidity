# scripts/dfc/dfc_compute.py — Usage Guide

This document describes how to run the consolidated Dynamic Functional Connectivity (DFC) CLI located at `scripts/dfc/dfc_compute.py`. The CLI supersedes previous dataset-specific scripts (`allegiance/src/dfc_compute.py`, `julien_data/2_compute_dfc_stream.py`, `metaconnectivity/old_useful/dfc_streams_wof_clean.py`) and provides a threaded implementation that works for both supported datasets.

---

## Prerequisites
- Run the preprocessing pipeline first so the canonical bundle `ts_and_meta_<dataset>.npz` exists in `results/<dataset>/preprocessed/`. See `docs/preprocessing.md` for details.
- Configure paths via the shared environment variables (`PATHS_ROOT`, `PROJECT_ROOT_LOCAL`, `DATASET_NAME`, etc.). The CLI resolves directories through `shared_code.fun_paths.get_paths`.
- Install the shared package in editable mode (`pip install -e shared_code`) and ensure dependencies listed in `pyproject.toml` are available (`numpy`, `numba`, `tqdm`, etc.).

---

## Core Concepts
- **Datasets**: Pass `--dataset-name julien` or `--dataset-name ines`. Aliases such as `--dataset-name julien_caillette` and `--dataset-name ines_abdullah` are also accepted.
- **Bundles**: By default the CLI looks for `ts_and_meta_<dataset>.npz`. Use `--bundle-name` to override the filename if you generated a custom bundle.
- **Window sweep**: The triple `(wmin, wmax, wstep)` defines the set of sliding windows to compute. Each window size produces a compressed NPZ file named `dfc_window_size=<w>_lag=<lag>_tau=<tau>_animals=<n>_regions=<r>.npz`.
- **Formats**: Use `--format 3D` (default) for `(animals, regions, regions, frames)` arrays or `--format 2D` for the vectorised upper triangle `(animals, pairs, frames)`.
- **Caching**: Existing files are skipped by default. Adjust `--cache` to `load`, `verify`, or `overwrite` depending on whether you want to re-use or regenerate artefacts.
- **Parallelism**: `--jobs` allows per-animal parallelism via a thread pool. Pick a value that matches available CPU cores and memory bandwidth.

---

## Typical Commands
Preprocess first (example for Julien dataset):

```bash
python scripts/preprocessing/preprocess.py --dataset-name julien --only-tr 500
```

Compute DFC for the Julien dataset with a modest window sweep:

```bash
python scripts/dfc/dfc_compute.py \
  --dataset-name julien \
  --wmin 5 --wmax 20 --wstep 5 \
  --lag 1 --tau 5 --format 3D \
  --jobs 4
```

Compute DFC for the Ines dataset while re-using cached files if shapes match:

```bash
python scripts/dfc/dfc_compute.py \
  --dataset-name ines \
  --wmin 9 --wmax 21 --wstep 6 \
  --lag 1 --tau 3 \
  --cache verify \
  --jobs 2
```

Target a custom bundle and write 2D (flattened) DFC streams:

```bash
python scripts/dfc/dfc_compute.py \
  --dataset-name julien \
  --bundle-name ts_and_meta_julien_custom.npz \
  --format 2D \
  --wmin 7 --wmax 7 --lag 1 --tau 4
```

---

## Output Layout
- Files are written under `paths["dfc"]`, typically `results/<dataset>/dfc/`.
- Each NPZ contains a single array stored under the `dfc` key.
- Metadata such as the number of animals/regions/lags is encoded in the filename. Downstream scripts parse these names, so prefer the defaults unless you have a reason to customise them.

---

## Integration Tips
- The CLI is also re-exported by `allegiance/src/dfc_compute.py`, so any legacy workflows continue to function. Prefer calling the consolidated version directly in new automation.
- `shared_code.fun_utils.load_timeseries_data` enforces `allow_pickle=False`; if you load NPZ outputs manually, follow the same policy for safety.
- When running on a scheduler or cluster, set `--jobs 1` per worker and parallelise across window ranges using job arrays to avoid oversubscribing threads.
- Add synthetic smoke tests around your preferred parameter choices to catch regressions, especially when changing default window ranges or format selection.

---

## Troubleshooting
- **“Expected bundle not found”**: Re-run preprocessing or pass `--bundle-name` with the correct filename.
- **“Window X exceeds timeseries length”**: Your window is longer than the available TRs; lower `--wmin/--wmax` or ensure preprocessing outputs matched TR counts.
- **Slow performance**: Increase `--jobs`, but monitor memory usage. For the vectorised format, consider switching to `--format 2D` if downstream consumers only need flattened data.
- **Shape mismatch in verify mode**: Delete or rename the stale file so it can be recomputed, or rerun with `--cache overwrite`.

