# Bootstrap Speed CLI — Notebook‑Style Tutorial

This tutorial shows how to use `scripts/bootstrap_speed_groups_cli.py` to bootstrap dFC speed distributions per group, per region and window, and optionally pool across windows. You can paste the code blocks into a Jupyter notebook (Markdown + code cells).

## 0) Setup

```bash
pip install -e shared_code
# Optional dev tools (linters/tests/plotting)
pip install -r requirements-dev.txt
```

Paths are profile‑driven. Either set a hard root or select a profile:

```
# Hard override (simple)
export PATHS_ROOT=/abs/path/to/project/root
export DATASET_NAME=julien_caillette

# Or profile based
export PATHS_ENV=CLUSTER_FS
export PROJECT_ROOT_CLUSTER_FS=/scratch/$USER/laura_harsan
export DATASET_NAME=julien_caillette

# Validate
python scripts/paths_doctor.py --show --check-write --create
```

Verify outputs exist:

```python
from shared_code.shared_code.fun_paths import get_paths
paths = get_paths(dataset_name='julien_caillette', create=False, check_write=False)
paths['speed'], paths['dfc']
```

## 1) Quick Tau Check

```bash
python scripts/bootstrap_speed_groups_cli.py --tr 500 --subset regions500 --show-tau
# Example output:
# tau (from metadata) = 3
# tau_count (valid indices) = 4 -> indices 0..3
# Use --tau-index -1 to pool all taus.
```

## 2) Basic Run (Single Tau)

```bash
python scripts/bootstrap_speed_groups_cli.py \
  --tr 500 --subset regions500 --tau-index 0 \
  --n-boot 2000 --seed 0 --ci 95 \
  --pool-threshold median
```

Parameters used:
- taus: only one (tau index 0). Use `--tau-index -1` to pool all available taus.
- bootstraps: `--n-boot 2000` (default).
- groups: `--group-cols genotype,treatment` (default).
- percentiles: `--q 1,5,50,95,99` (default).
- pool threshold: `median` splits windows into short (<= median) and long (> median); per‑window rows are still produced, pools add extra rows.

Outputs are written under dataset paths:
- CSVs: `paths['speed']/<outdir>/speed_bootstrap_quantiles.csv` and `..._diffs.csv`

## 3) Plots and Grids (what you get and how to tune them)

Command example (outdir defaults to subset if not provided):

```bash
python scripts/bootstrap_speed_groups_cli.py \
  --tr 500 --subset regions500 --tau-index 0 \
  --pool-threshold median --pool-all \
  --plot --grid --grid-cols 2 --plot-format png \
  --progress --jobs 4 --parallel-scope windows \
  --seed 123
```

Where the figures go
- Saved under `paths['f_speed']/<outdir>/`.
- File naming and types:
  - Per‑window group quantiles: `quantiles_<region>_win<W>.<fmt>`
  - Per‑window pairwise diffs: `diffs_<region>_win<W>_<A>_vs_<B>.<fmt>`
  - Per‑window grid (if `--grid`): `grid_<region>_win<W>.<fmt>`
  - Pools (“short”, “long”, and optionally “all”):
    - Group quantiles: `quantiles_<region>_pool-<name>.<fmt>`
    - Pairwise diffs: `diffs_<region>_pool-<name>_<A>_vs_<B>.<fmt>`

What each plot shows
- Group quantiles (quantiles_*):
  - X‑axis: group levels (e.g., WT‑VEH, WT‑LCTB92, ...).
  - Whiskers: bootstrap percentile spreads — inner by default is 5–95%; thin caps indicate outer 1–99% when available.
  - Point: the median (q=50) per group; an errorbar overlays the median’s bootstrap CI.
  - You control which percentiles are computed via `--q` (default `1,5,50,95,99`).

- Pairwise diffs (diffs_*):
  - X‑axis: percentiles requested via `--q` (e.g., 1, 5, 50, 95, 99).
  - Y‑axis: percentile(A) − percentile(B).
  - Zero line is drawn; errorbars show the bootstrap CI for each percentile.
  - Markers: filled = significant (CI excludes 0); open = not significant.
  - Pairs are provided via `--pairs` (see defaults below).

- Grids (grid_*):
  - A compact arrangement of all pairwise diffs panels for a given window/pool.
  - Columns controlled by `--grid-cols` (default 2).

Tuning plotting behavior
- `--plot-format`: `png` (default), `pdf`, `svg`.
- `--grid-cols`: layout columns for grid plots (default 2).
- `--progress`: shows tqdm progress bars.
- `--load-cache`: skips saving a figure if the file already exists.
- `--dry-run`: prints what will be processed (regions, windows, pools, output paths) and exits without computing.
- `--list-inputs`: with `--dry-run`, prints the exact NPZ files that would be read.
- `--q`: changes the percentiles shown in both quantiles and diffs plots.
- `--seed`, `--n-boot`, `--ci`: control bootstrap variability and CI width.

Performance tips
- Use `--jobs N` and `--parallel-scope windows` to parallelize per‑window work within each region.
- Combine with `--progress` to keep an eye on progress.

Note on output folder names
- If you omit `--outdir`, it defaults to the value of `--subset` (or to `bootstrap` if no subset). This keeps outputs tidy without extra flags.
- If you provide a custom `--outdir` for multiple subsets, you can optionally add `--append-subset-to-outdir` to avoid collisions (writes to `<outdir>__subset-<subset>`).

## 4) Custom Groups and Pairs

```bash
python scripts/bootstrap_speed_groups_cli.py \
  --tr 500 --subset regions500 --tau-index 0 \
  --group-cols genotype,treatment \
  --pairs "(WT,VEH)-(WT,LCTB92);(Dp1Yey,VEH)-(Dp1Yey,LCTB92);(WT,VEH)-(Dp1Yey,VEH);(WT,LCTB92)-(Dp1Yey,LCTB92)" \
  --pool-threshold median --plot
```

## 5) Read CSV Outputs

Two CSVs are written to the output directory (default `reports/`):

- `speed_bootstrap_quantiles.csv`: per‑group percentile estimates and CIs.
  - Columns:
    - `region`: region label (or `all` if not per‑region).
    - `window`: window size (int), or pool label (`short`, `long`, `all`).
    - `group`: group key (scalar or tuple printed as `(geno, treat)` depending on `--group-cols`).
    - `q`: percentile requested (e.g., 1, 5, 50, 95, 99).
    - `point`: percentile estimate.
    - `lo`, `hi`: bootstrap confidence interval bounds (CI set by `--ci`).
    - `n`: total pooled sample count (after filtering NaNs), across all animals in the group.

- `speed_bootstrap_diffs.csv`: per‑pair percentile differences with CIs and significance.
  - Columns:
    - `region`, `window`: as above.
    - `A`, `B`: group keys compared (A − B).
    - `q`: percentile compared.
    - `diff`: difference of percentiles, i.e., `percentile(A) − percentile(B)`.
    - `lo`, `hi`: bootstrap CI for the difference.
    - `significant`: boolean; true when the CI does not cross zero.
    - `n_a`, `n_b`: pooled sample sizes used for A and B.

Reading and basic inspection:

```python
import pandas as pd
from net_fluidity_julien.context import DFCAnalysis
ctx = DFCAnalysis(); ctx.get_metadata();
from pathlib import Path
out_csv_root = Path(ctx.paths['speed']) / 'j500_t0'
qdf = pd.read_csv(out_csv_root / 'speed_bootstrap_quantiles.csv')
ddf = pd.read_csv(out_csv_root / 'speed_bootstrap_diffs.csv')

# Unique regions and windows/pools present
print(sorted(qdf['region'].unique())[:5])
print(sorted(qdf['window'].unique(), key=str)[:8])

# Focus on median (q=50) for a specific region and pool
sub_q = qdf[(qdf['region'] == 'ACC') & (qdf['q'] == 50) & (qdf['window'].isin(['short', 'long']))]
print(sub_q.head())

# Pivot to compare median by group across windows (numerical windows only)
sub_win = qdf[(qdf['region'] == 'ACC') & (qdf['q'] == 50) & (qdf['window'].apply(lambda x: str(x).isdigit()))]
pivot = sub_win.pivot_table(index='window', columns='group', values='point')
print(pivot.head())

# Significant differences at a given window for a given pair and percentile
pair = (("WT","VEH"), ("WT","LCTB92"))
mask = (
    (ddf['region'] == 'ACC') &
    (ddf['window'] == 9) &
    (ddf['A'] == str(pair[0])) &
    (ddf['B'] == str(pair[1])) &
    (ddf['q'] == 50)
)
print(ddf[mask][['diff','lo','hi','significant','n_a','n_b']])
```

Interpretation tips:
- If `n` (or `n_a`/`n_b`) is 0, that group/pair had no valid values (e.g., empty tau or filters).
- A positive `diff` means group A has a higher percentile than group B.
- `significant=True` indicates the CI excludes zero; always check magnitude and direction.

## 5.1) Pool‑Test (target vs pooled supergroup)

If you want to test a group against a pooled control built from other groups:

- Column‑based pooling (automatic):
  - Use the compute‑only CLI and add `--bootstrap-pool-cols` (subset of `--group-cols`), plus `--pool-exclude-self` if needed:
    ```bash
    python scripts/compute_speed_bootstrap.py \
      --tr 500 --subset regions500 --tau-index 0 \
      --bootstrap-pool-cols genotype --pool-exclude-self \
      --pool-threshold median --pool-all --n-boot 2000 --jobs 8
    ```
  - Outputs `speed_bootstrap_pooltest(_nboot-<N>).csv` with metadata columns identifying the pool.

- Explicit pooling (custom unions):
  - Define the exact pool and targets:
    ```bash
    python scripts/compute_speed_pooltest_explicit.py \
      --tr 500 --subset dmn_within --tau-index 0 \
      --group-cols genotype,treatment \
      --targets "(Dp1Yey,LCTB92);(WT,VEH)" \
      --pool "(WT,VEH);(Dp1Yey,LCTB92)" \
      --n-boot 2000 --jobs 8 --progress \
      --pool-threshold median --pool-all
    ```

Plot pool‑tests directly from CSVs:
```bash
python scripts/plot_speed_pooltest.py --tr 500 --subset dmn_within --bywin --pooled --progress
```

## 6) Troubleshooting

- Tau index: run `--show-tau` first or use `--tau-index -1` to pool all taus.
- Empty groups/windows: rows with `n=0` and NaN CIs; plots skip gracefully.
- Subset path: ensure it points to per‑window NPZs in `regions-<label>/` or `all/`.

## 6.1) What bootstrap is used?

Non‑parametric bootstrapping with replacement over pooled samples:

```python
def bootstrap_ci_1d(x, n_boot=2000, stat='median', ci=95.0, random_state=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if stat == 'median':
        stat_fn = lambda a: float(np.median(a))
    elif stat == 'mean':
        stat_fn = lambda a: float(np.mean(a))
    elif stat.startswith('q'):
        q = float(stat[1:]) / 100.0
        stat_fn = lambda a: float(np.quantile(a, q))
    est = stat_fn(x)
    rng = np.random.default_rng(random_state)
    boots = [stat_fn(x[rng.choice(x.size, x.size, replace=True)]) for _ in range(n_boot)]
    alpha = (100.0 - ci) / 2.0
    lo = np.percentile(boots, alpha)
    hi = np.percentile(boots, 100.0 - alpha)
    return est, lo, hi
```

- Group percentiles `bootstrap_quantiles_by_group` bootstrap pooled per‑group samples to estimate requested percentiles and CIs.
- Pairwise differences `bootstrap_quantile_diffs` resample A and B independently, compute percentile(A) − percentile(B) per resample, and form CIs.

Key parameters:
- `--n-boot`: number of bootstrap resamples (default 2000).
- `--seed`: base random seed (deterministic runs).
- `--ci`: confidence level in percent (default 95).
- `--q`: which percentiles to compute (default 1,5,50,95,99).

## 7) End‑to‑End Example

```bash
python scripts/bootstrap_speed_groups_cli.py --tr 500 --subset regions500 --show-tau

python scripts/bootstrap_speed_groups_cli.py \
  --tr 500 --subset regions500 --tau-index 0 \
  --pool-threshold median --pool-all \
  --plot --grid --grid-cols 2 --progress \
  --jobs 4 --parallel-scope windows \
  --outdir j500_t0 --append-subset-to-outdir --seed 123

## 8) How does "-all" pooling work?

When you pass `--pool-all`, the CLI concatenates raw speed samples across all windows for each animal (per region) and then runs the bootstrap on the pooled samples. It does NOT pool bootstrap summaries; it pools the underlying samples first, which preserves sample‑level variability across windows.
```

## 9) Possible Additions

- Filtering: `--include-regions`, `--window-min/--window-max`.
- Statistics: `--stat` selector, interaction effects export, FDR marking.
- Metadata: write a JSON sidecar describing run parameters and discovered windows.
- UX: `--dry-run` to list work units; richer progress/reporting.
- Visualization: theme/size options; default SVG export.
