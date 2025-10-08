# Julien dFC Speed — Phase 3 Usage

This guide reflects the Phase 3 workflow: stable paths via `shared_code.fun_paths`, unified context (`net_fluidity_julien.context.DFCAnalysis`), per‑region compute, and notebook‑friendly loaders and bootstrapping.

## 0) Environment

- Python 3.11
- Install shared package: `pip install -e shared_code`
- Configure paths via env (profile‑driven):
  - Simple hard override: `PATHS_ROOT=/abs/path/to/repo-root`
  - Or profile label: `PATHS_ENV=CLUSTER_FS` and `PROJECT_ROOT_CLUSTER_FS=/abs/path`
  - Optional: `DATASET_NAME=julien_caillette`
  - Validate: `python scripts/paths_doctor.py --show --check-write --create`
- Optional logging: copy `config/logging.example.yaml` → `config/logging.yaml` and set `NET_FLUIDITY_LOGGING=config/logging.yaml`.
 - Notebooks: ensure package imports by adding the repo `src` to `sys.path` (so `net_fluidity_julien` is importable):
   - In a notebook cell: `import sys; sys.path.insert(0, 'src')`

## 1) Preprocess Data

Align time series, cognitive data, and labels:

```bash
python julien_data/1_preprocess_data_ts_cog.py
```

Outputs (under `paths['preprocessed']`):
- `metadata_animals_{N}_regions_{R}_tr_{T}.pkl`
- `ts_filtered_animals_{N}_regions_{R}_tr_{T}.npz`
- `cog_data_filtered_animals_{N}_regions_{R}_tr_{T}.csv`

## 2) Compute dFC Streams

Build per‑window dFC streams (script selects the TR from metadata):

```bash
python julien_data/2_compute_dfc_stream.py
```

Outputs (under `paths['dfc']`): one NPZ per window (key `dfc`).

## 3) Compute dFC Speed

Use the Phase 3 wrapper (delegates to the legacy engine, keeps outputs stable):

```bash
python julien_data/src/speed_compute.py --tr 500 --processors -1
```

Per‑region outputs (one subfolder per region):

```bash
python julien_data/src/speed_compute.py --tr 500 --per-region --processors -1
```

Selections (optional): `--selected-region-labels "ACC,THAL"`, `--region-mode within|touching`, `--subset-name mySubset`.

Outputs (under `paths['speed']`):
- `all/` if no selection; `regions-<label>/` for per‑region; or your `--subset-name`.
- Per‑window NPZ: `speed_win{W}_..._tau{T}_animals_{N}_regions_{R}.npz` with key `speeds`.

List available region labels to help with selections:

```bash
python julien_data/3_dfc_speed_test_v6.py --tr 500 --list-regions
```

Try the shared engine (keeps naming; used for parity checks during Phase 3):

```bash
python julien_data/src/speed_compute.py --tr 500 --engine shared
```

## 4) Notebook — Load All Regions + Windows (TR=500)

All loaders and plotting helpers are notebook‑friendly (no argparse needed).

```python
import sys
sys.path.insert(0, 'src')  # ensure net_fluidity_julien is importable in notebooks
from scripts.speed_bootstrap_nb import (
    load_all_speeds_by_region_nb, pool_short_long_nb,
    bootstrap_quantiles_by_group, bootstrap_quantile_diffs_by_keys,
    plot_group_quantiles, plot_quantile_diffs, compute_pairs_diffs_nb, plot_pairs_grid_nb,
)

payload = load_all_speeds_by_region_nb(tr=500, subset_name='shared', tau_index=0)
regions = payload['regions']; groups = payload['groups']
r = regions[0]
windows = payload['by_region'][r]['windows']
per_by_win = payload['by_region'][r]['per_animal_by_window']

# Short/long pools by median
pools = pool_short_long_nb(per_by_win, windows, threshold='median')
qa_short = bootstrap_quantiles_by_group(pools['short'], groups, q=[1,5,50,95,99])
plot_group_quantiles(qa_short, title=f'{r} | pool=short (cut={pools["cut"]})')

pairs = [
  (('WT','VEH'), ('WT','LCTB92')),
  (('Dp1Yey','VEH'), ('Dp1Yey','LCTB92')),
  (('WT','VEH'), ('Dp1Yey','VEH')),
  (('WT','LCTB92'), ('Dp1Yey','LCTB92')),
]
qd_map = compute_pairs_diffs_nb(pools['short'], groups, pairs)
plot_pairs_grid_nb(qd_map, pairs, title=f'{r} | pool=short', cols=2)
```

Global (non per‑region) speeds:

```python
from scripts.speed_bootstrap_nb import load_all_speeds_nb
payload_global = load_all_speeds_nb(tr=500, subset_name='shared', region_label=None, tau_index=0)
```

## 5) CLI — Bootstrap Tables + Figures (optional)

Write tidy CSVs and export figures for all regions/windows (and short/long pools):

```bash
python scripts/bootstrap_speed_groups_cli.py \
  --tr 500 --subset shared --tau-index 0 \
  --pool-threshold median --pool-all \
  --plot --grid --grid-cols 2
```

Outputs
- `reports/speed_bootstrap_quantiles.csv`
- `reports/speed_bootstrap_diffs.csv`
- figures under `fig/<dataset>/speed/<subset>/...`

## 6) Baseline & Checks

- Install/lint: `pip install -e shared_code`; `make check`
- Smoke tests: `pytest -q`
- Data presence (Python): list candidates under `paths['dfc']` and `paths['speed']` using the context.

## Notes

- Prefer `net_fluidity_julien.context.DFCAnalysis` and helpers in `scripts/speed_bootstrap_nb.py`.
- Legacy modules in `julien_data/` remain available for back‑compat; Phase 3 prioritizes package imports.
