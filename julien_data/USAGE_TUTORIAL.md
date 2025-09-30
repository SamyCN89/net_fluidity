# Julien dFC Speed — Usage Tutorial

This tutorial shows how to run the Julien dFC speed analysis with region subsetting and per‑region loops using `3_dfc_speed_test_v6.py`. It assumes you’ve prepared data and dFC streams.

## 1) Prepare Data

Run preprocessing to align time series, cognitive data, and labels:

```bash
python julien_data/1_preprocess_data_ts_cog.py
```

Outputs (under `paths['preprocessed']`):
- `metadata_animals_{N}_regions_{R}_tr_{T}.pkl`
- `ts_filtered_animals_{N}_regions_{R}_tr_{T}.npz`
- `cog_data_filtered_animals_{N}_regions_{R}_tr_{T}.csv`

## 2) Compute dFC Streams

Build per‑window dFC streams:

```bash
python julien_data/2_compute_dfc_stream.py
```

Outputs (under `paths['dfc']`): one NPZ per window, key `dfc` with shape `(n_animals, n_pairs, n_frames)`.

## 3) Compute dFC Speed

The script `3_dfc_speed_test_v6.py` computes speed per animal, per window, and per tau (multi‑tau legacy engine) with optional region selection.

Common flags:
- `--method`: `pearson` | `spearman` | `cosine` (default: `pearson`)
- `--processors`: number of parallel jobs (e.g., `-1` for all cores)
- `--load-cache`: reuse existing per‑window NPZs if present
- `--list-regions`: print all region labels with indices and exit
- `--selected-regions`: comma list of region indices (`0,3,7`)
- `--selected-region-labels`: comma list of labels (mapped to indices)
- `--region-mode`: `touching` (edges incident to any selected region) or `within`
- `--return-fc2`: also save second‑frame indices (debugging)
- `--per-region`: loop across all regions, one output set per region
- `--subset-name`: custom subfolder name under `speed/` for outputs (overrides auto-naming)

Examples:

- Whole‑brain (all edges), default params:
```bash
python julien_data/3_dfc_speed_test_v6.py
```

- Selected regions by labels; only edges within the set:
```bash
python julien_data/3_dfc_speed_test_v6.py \
  --selected-region-labels "ACC,THAL" \
  --region-mode within
```

- Selected regions by indices; edges touching those regions:
```bash
python julien_data/3_dfc_speed_test_v6.py \
  --selected-regions "1,4,9" \
  --region-mode touching
```

- Per‑region run (one output per region):
```bash
python julien_data/3_dfc_speed_test_v6.py --per-region
```

- Custom subfolder name (e.g., `myFavoriteSet`):
```bash
python julien_data/3_dfc_speed_test_v6.py \
  --selected-region-labels "ACC,THAL" \
  --region-mode touching \
  --subset-name myFavoriteSet
```

- Outputs are saved under subfolders of `paths['speed']`:
- `all/` if no region selection
- `regions-ACC-THAL/` when selecting by labels (<= 5 labels shown)
- `indices-1_4_9/` when selecting by indices (<= 5 shown)
- `nregs-12/` when many regions are selected
- or a custom folder you provide via `--subset-name`

- Inside each subfolder, per‑window NPZ files named:
  - `speed_win{W}[_subset_mode-<mode>_<desc>]_tau{T}_animals_{N}_regions_{R}.npz` with key `speeds`.
  - Examples of `<desc>`: `region-16-ACC`, `idx-1_4_9`, `lab-ACC-THAL`, or `nregs-12` (when many).

## 4) Quick Load and Plot

Overall pooled (auto‑detect last window across subfolders):

```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from julien_data.class_dataanalysis_julien import DFCAnalysis

data = DFCAnalysis(); data.load_preprocessed_data(); data.get_temporal_parameters()
save_root = data.paths['speed']
W = int(data.time_window_range[-1]); tau_count = int(data.tau + 1)

# Optionally restrict to a subfolder, e.g.:
# subdir = save_root / 'all'
# candidates = sorted(subdir.glob(f"speed_win{W}_*tau{tau_count}_animals_{data.n_animals}_regions_{data.regions}.npz"))

# Auto‑detect across all subfolders
candidates = sorted(save_root.rglob(f"speed_win{W}_*tau{tau_count}_animals_{data.n_animals}_regions_{data.regions}.npz"))
if not candidates:
    raise FileNotFoundError('No speed file found')
npz_path = candidates[-1]
print('Using file:', npz_path)

npz = np.load(npz_path, allow_pickle=True)
win_speeds = npz['speeds']  # object array: len=n_animals; each entry (n_taus, T_w)

pooled = []
for a in range(len(win_speeds)):
    arr = win_speeds[a]
    if arr is None: continue
    arr = np.asarray(arr, dtype=float)
    pooled.append(arr[~np.isnan(arr)])
all_vals = np.concatenate(pooled) if pooled else np.array([])
print('Pooled size:', all_vals.size, 'median:', np.nanmedian(all_vals) if all_vals.size else np.nan)

plt.hist(all_vals, bins=120, density=True, histtype='step', alpha=0.85)
plt.title(f'dFC Speed (W={W}, all animals, all taus)')
plt.xlabel('Speed'); plt.ylabel('Density'); plt.tight_layout(); plt.show()
```

Per‑group (like `local_speed_plot_v2.py`):

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from julien_data.class_dataanalysis_julien import DFCAnalysis

data = DFCAnalysis(); data.load_preprocessed_data(); data.get_temporal_parameters()
save_root = data.paths['speed']
W = int(data.time_window_range[-1]); tau_count = int(data.tau + 1)
npz_path = sorted(save_root.rglob(f"speed_win{W}_*tau{tau_count}_animals_{data.n_animals}_regions_{data.regions}.npz"))[-1]
npz = np.load(npz_path, allow_pickle=True)
win_speeds = npz['speeds']

group_speeds = {}
for group, idxs in data.groups.items():
    pooled = []
    for a in idxs:
        if a >= len(win_speeds): continue
        arr = win_speeds[a]
        if arr is None: continue
        arr = np.asarray(arr, dtype=float)
        pooled.append(arr[~np.isnan(arr)])
    group_speeds[group] = np.concatenate(pooled) if pooled else np.array([])

sns.set_theme(style='white', context='talk'); plt.figure(figsize=(9,6))
palette = sns.color_palette('tab10', n_colors=len(group_speeds))
for (grp, vals), color in zip(group_speeds.items(), palette, strict=False):
    if vals.size == 0: continue
    label = f"{grp[0]}-{grp[1]}"
    plt.hist(vals, bins=120, density=True, histtype='step', lw=1.7, alpha=0.85, label=label, color=color)
    sns.kdeplot(vals, bw_adjust=0.7, color=color, lw=2)
plt.title(f'dFC Speed per group (W={W}, all taus)'); plt.xlabel('Speed'); plt.ylabel('Density')
plt.legend(title='Group'); plt.tight_layout(); sns.despine(trim=True); plt.show()
```

Merged outputs (created automatically at the end of a run):

```python
import pickle
from pathlib import Path
from julien_data.class_dataanalysis_julien import DFCAnalysis

data = DFCAnalysis(); data.load_preprocessed_data(); data.get_temporal_parameters()
save_root = data.paths['speed']
subdir = save_root / 'all'  # or 'regions-ACC-THAL', your --subset-name, etc.

merged_pkl = sorted(subdir.glob(f"speed_windows*_tau{data.tau+1}_animals_{data.n_animals}_regions_{data.regions}.pkl"))[-1]
with open(merged_pkl, 'rb') as fh:
    payload = pickle.load(fh)
all_speed = payload['speeds']  # list length = n_windows; each element is object array per window
meta = payload['meta']         # dict: method, region_mode, selected_regions/labels, etc.
print('Merged windows:', len(all_speed), 'example window shape:', all_speed[-1].shape)
print('Meta:', meta)

# If you saved FC2 with --return-fc2, metadata is stored inside the NPZ
import numpy as np
fc2_npz = sorted(subdir.glob(f"speed_fc_windows*_tau{data.tau+1}_animals_{data.n_animals}_regions_{data.regions}.npz"))
if fc2_npz:
    fc2 = np.load(fc2_npz[-1], allow_pickle=True)
    fc2_meta = fc2['meta'].item()  # retrieve dict stored as object
    print('FC2 meta:', fc2_meta)
```

For richer examples (stats and figures), see `plots_speed.py`, `plot_cog_data.py`, and `local_speed_plot_v2.py`.

## 5) Plot From Merged Outputs (Script)

Use the helper script to load the merged PKL and create publication‑style plots:

```bash
# Overall + per‑group + medians (auto‑detect merged file under speed/)
python julien_data/plot_merged_speed.py --subset-name all

# Focus a specific subset folder; only tau=0; save figures
python julien_data/plot_merged_speed.py \
  --subset-name regions-ACC-THAL \
  --tau 0 \
  --savefig

# Custom subset (if you used --subset-name during compute), no group/medians
python julien_data/plot_merged_speed.py \
  --subset-name myFavoriteSet \
  --no-group --no-medians
```

Flags:
- `--subset-name`: subfolder in `speed/` (e.g., `all`, `regions-ACC-THAL`, or your custom)
- `--tau`: integer tau index to plot; omit to pool all taus
- `--no-group`: skip per‑group distributions
- `--no-medians`: skip median vs window plots
- `--savefig`: save figures next to the merged PKL
- `--groups`: filter to specific groups, comma‑separated `GENOTYPE-TREATMENT` (e.g., `WT-VEH,Dp1Yey-LCTB92`)
- `--split-pools`: also plot per‑group distributions for two window pools (first half vs second half of window sizes)
- `--split-at`: specify the window size threshold (Pool A <= threshold, Pool B > threshold). If omitted, the script splits at mid index; for odd counts it drops the middle to enforce equal group sizes and prints which W was dropped.
 - `--qq`: produce QQ plots between selected groups for a chosen pool and tau (used with `--groups` and `--split-at`)
 - `--qq-pool`: which pool for QQ (`A` or `B`)

Use from Jupyter:

```python
from julien_data.plot_merged_speed import run_plot

# All defaults, auto‑detect 'all' or pass your subfolder
run_plot(subset_name='all')

# Focus specific groups and tau
run_plot(subset_name='regions-ACC-THAL', tau=0, groups=['WT-VEH','Dp1Yey-LCTB92'])

# Only overall, no groups/medians, and save figures; also split pools when groups shown
run_plot(subset_name='myFavoriteSet', no_group=True, no_medians=True, savefig=True)
run_plot(subset_name='myFavoriteSet', split_pools=True)

# Split at a specific window size (consistent for all groups)
run_plot(subset_name='all', split_pools=True, split_at=40)

# QQ plots between groups for Pool A, tau=0
# CLI:  python julien_data/plot_merged_speed.py --subset-name all --groups WT-VEH,Dp1Yey-LCTB92 --split-pools --split-at 40 --tau 0 --qq --qq-pool A
# NB: From Jupyter you can call the QQ helper (see script) or extend run_plot to enable it.

## 6) Quick Reference

- Preprocess data:
  - `python julien_data/1_preprocess_data_ts_cog.py`
- Build dFC streams:
  - `python julien_data/2_compute_dfc_stream.py`
- Compute dFC speed (global):
  - `python julien_data/3_dfc_speed_test_v6.py`
- Compute dFC speed (labels, within-mode, custom folder):
  - `python julien_data/3_dfc_speed_test_v6.py --selected-region-labels "ACC,THAL" --region-mode within --subset-name myFavoriteSet`
- List region labels with indices:
  - `python julien_data/3_dfc_speed_test_v6.py --list-regions`
- Plot merged outputs (script):
  - `python julien_data/plot_merged_speed.py --subset-name myFavoriteSet --split-pools --split-at 40`
- Plot merged outputs (Jupyter):
  - `from julien_data.plot_merged_speed import run_plot; run_plot(subset_name='myFavoriteSet', split_pools=True, split_at=40)`

## 7) Resuming This Work — Suggested Prompt

Copy‑paste this prompt to resume the session with full context:

```
Context: We refactored dFC speed computation to use dfc_speed_split with region subsetting, added per-region loops, explicit subset tags, and per-selection subfolders (including --subset-name). We also implemented merged outputs (PKL with meta; NPZ for FC2) and added a plotting script (plot_merged_speed.py) that supports per‑group plots, quantile bands, and two window pools (equal halves or --split-at threshold). Tutorial is in julien_data/USAGE_TUTORIAL.md.

Please do the following next:
1) Add a dry‑run flag to 3_dfc_speed_test_v6.py that prints planned filenames and subfolder, then exits.
2) In plot_merged_speed.py, add an option to plot QQ plots between selected groups for a given pool and tau.
3) Extend meta saved in merged outputs to include a timestamp and git commit (if available), and surface it in the plotting script summary.
4) Update USAGE_TUTORIAL.md with the new flags and examples.
```
```

## Notes
- The script logs label→index mapping and warns about unknown labels when using `--selected-region-labels`.
- Variable‑length arrays are saved as object dtype; downstream code flattens and drops NaNs before pooling.
