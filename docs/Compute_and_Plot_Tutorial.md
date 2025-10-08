# dFC Speed Bootstrap — Compute and Plot Tutorial

This tutorial splits the workflow into two clean steps: compute CSVs only, then plot from CSVs. It uses thin scripts that wrap the central bootstrap kernels.

## 0) Setup

```bash
pip install -e shared_code
# Optional: plotting/dev tools
pip install -r requirements-dev.txt
```

Paths are profile‑driven and cluster‑friendly. You can:
- Set `PATHS_ROOT` (hard override), or
- Set `PATHS_ENV` and define `PROJECT_ROOT_<ENV>`.
Also set `DATASET_NAME` to select the dataset subfolder.

Examples
```
# Simple hard override (recommended on cluster filesystems)
export PATHS_ROOT=/abs/path/to/project/root
export DATASET_NAME=julien_caillette

# Or pick a named profile
export PATHS_ENV=CLUSTER_FS
export PROJECT_ROOT_CLUSTER_FS=/scratch/$USER/laura_harsan
export DATASET_NAME=julien_caillette

# Validate resolved roots/paths and write access
python scripts/paths_doctor.py --show --check-write --create
```

## 1) Compute CSVs (no plotting)

Script: `scripts/compute_speed_bootstrap.py`

Purpose: compute per-ROI, per-window (and pooled short/long/all) bootstrap tables as CSVs. No figures are generated; you can plot later.

Outputs (under dataset paths, with n‑boot suffixed copies):
- `paths['speed']/<outdir>/speed_bootstrap_quantiles.csv` and `_nboot-<N>.csv`
- `paths['speed']/<outdir>/speed_bootstrap_diffs.csv` and `_nboot-<N>.csv`
- `paths['speed']/<outdir>/speed_nor_correlations.csv` and `_nboot-<N>.csv` (when `--correlate-nor`)
- Pool‑test (see below): `speed_bootstrap_pooltest.csv` and `_nboot-<N>.csv`

Fast example (TR=500, subset `regions500`, parallel windows):

```bash
python scripts/compute_speed_bootstrap.py \
  --tr 500 --subset regions500 --tau-index 0 \
  --pool-threshold median --pool-all \
  --n-boot 2000 --jobs 8 --parallel-scope windows \
  --progress
```

Notes:
- `--outdir` defaults to `--subset` (or `bootstrap` if no subset).
- Use `--append-subset-to-outdir` to suffix an explicit outdir as `<outdir>__subset-<subset>`.
- `--progress` shows tqdm bars in the underlying compute.

Options overview (compute)
- `--tr INT`: select TR by metadata; e.g., 500 or 400.
- `--subset NAME`: choose the output subset folder under `paths['speed']`; if omitted, defaults to `bootstrap`.
- `--outdir NAME`: override output folder name (defaults to `--subset` if omitted).
- `--append-subset-to-outdir`: suffix `<outdir>__subset-<subset>` when combining multiple subsets.
- `--tau-index INT`: choose a tau slice; `-1` pools all taus together.
- `--q LIST`: percentiles to compute (comma‑sep), default `1,5,50,95,99`.
- `--pairs STRING`: pairs string `(G1,T1)-(G2,T2);...` to compare for diffs.
- `--n-boot INT`: bootstrap resamples (e.g., 500 for fast scans, 2000 for final).
- `--seed INT`: RNG seed for reproducibility.
- `--ci FLOAT`: CI percent for bootstrap (default 95). 
- `--pool-threshold median|INT`: pool windows into `short` (≤ threshold) and `long` (> threshold).
- `--pool-all`: also add an `all` pool across all windows.
- `--jobs INT`: parallelize across windows per ROI (e.g., `8`).
- `--parallel-scope windows`: current scope; window‑level parallelism.
- `--progress`: show progress bars.
- `--load-cache`: reuse existing CSVs if present (skips recompute).
- `--reuse-group-boots`: reuse per‑group bootstrap replicates across pairs for faster diffs.
- `--boots-float32` / `--values-float32` / `--index-int32`: memory/perf tuning for large jobs.
  
Pool‑test controls
- `--bootstrap-pool-cols COLS`: build pooled supergroups by matching a subset of `--group-cols` (e.g., `genotype`).
- `--pool-exclude-self`: when testing a group, drop its own animals from the pooled control.
- Writes `speed_bootstrap_pooltest(_nboot-<N>).csv` with metadata: `group_label, group_cols, pool_by, pool_match, pool_exclude_self`.

Behavior and outputs
- ROI enforcement: window pools (short/long/all) are computed per ROI (no cross‑ROI mixing).
- CSVs written to `paths['speed']/<outdir>/`:
  - `speed_bootstrap_quantiles.csv`: per‑group percentiles + CIs; columns:
    - `region, roi, window, group, q, point, lo, hi, n`.
  - `speed_bootstrap_diffs.csv`: per‑pair percentile diffs + CIs and p‑values; columns:
    - `region, roi, window, A, B, q, diff, lo, hi, p, p_method, significant, n_a, n_b`.
  - `speed_nor_correlations.csv` (if `--correlate-nor`): pooled‑window correlations of per‑animal percentiles with a NOR score; columns:
    - `region, roi, window, q, pearson_r, pearson_p, spearman_rho, spearman_p, n, nor_col`.

### 1.1) Pool‑Test (target vs pooled supergroup)

Two ways to generate pool‑tests during compute:
- Column‑based pooling (most common):
  - Add `--bootstrap-pool-cols genotype` (or any subset of your `--group-cols`) to construct a pooled control per group by matching only those columns.
  - Optionally add `--pool-exclude-self` to drop the target group’s own animals from the pool.
  - Pool‑tests are computed per window and for pooled windows (`short|long|all` when enabled), and saved in `speed_bootstrap_pooltest(_nboot-<N>).csv`.

- Explicit pooling (custom unions):
  - Use `scripts/compute_speed_pooltest_explicit.py` to define the exact control pool.
  - Example: test `(Dp1Yey,LCTB92)` and `(WT,VEH)` vs a pool `(WT,VEH)+(Dp1Yey,LCTB92)`:
    ```bash
    python scripts/compute_speed_pooltest_explicit.py \
      --tr 500 --subset dmn_within --tau-index 0 \
      --group-cols genotype,treatment \
      --targets "(Dp1Yey,LCTB92);(WT,VEH)" \
      --pool "(WT,VEH);(Dp1Yey,LCTB92)" \
      --n-boot 2000 --jobs 8 --progress \
      --pool-threshold median --pool-all
    ```
  - Writes `speed_bootstrap_pooltest_explicit(_nboot-<N>).csv` with the same schema and `pool_by=explicit`.

### 1.1) Detailed analysis usage (what the compute step does)

- Input discovery:
  - Looks under `paths['speed']/<subset>` for per‑ROI window files named like `speed_win<W>_*.npz`.
  - Infers the ROI label from the filename tag (e.g., `_subset_mode-..._region-3-ACC_...` or `_lab-ACC_...`); falls back to folder name.
  - Region folders can be `regions-<label>` or any folder containing `speed_win*_*.npz`.

- Per‑window analysis (per ROI):
  - Loads the NPZ (key `speeds`), which contains an object array of length `n_animals`, each entry a 2D array `[n_tau, T_window]`.
  - Tau handling:
    - `--tau-index k (k ≥ 0)`: use the k‑th tau row per animal.
    - `--tau-index -1`: pool across all tau rows first, then bootstrap (increases N; mixes tau distributions).
  - Grouping: builds `groups_map` from cognitive data columns (default `genotype,treatment`).
  - Quantiles: for each group, pools all per‑animal values (NaNs dropped) and bootstraps requested percentiles (`--q`) with `--n-boot` resamples; writes rows to quantiles CSV.
- Diffs: for each `(A,B)` pair in `--pairs`, independently bootstraps pooled A and B and computes percentile(A) − percentile(B) with a CI and an empirical two‑sided p‑value; writes rows to diffs CSV.
  - Missing pairs: if a pair key is not present in `groups_map` for this dataset, it is skipped (no error).

- Pooled windows (short/long/all):
  - Window pools are per‑ROI, based on `--pool-threshold`:
    - `median`: `short` = windows ≤ median; `long` = windows > median.
    - `INT`: explicit threshold split.
    - `--pool-all`: adds an `all` pool across all ROI windows.
  - For each pool, concatenates per‑animal arrays across windows, then applies the same quantile/diffs bootstrap as per‑window; writes rows to CSVs with `window=short|long|all`.

- Bootstrap details:
  - Non‑parametric bootstrapping with replacement over pooled samples; vectorized in chunks for speed.
  - `--n-boot` controls the number of resamples; `--ci` sets the CI level.
  - Diffs resample A and B independently; `significant` marks CI excluding 0.
  - p‑values in diffs: empirical two‑sided bootstrap on percentile differences with +1/(B+1) smoothing.
  - p‑values in correlations (when `--correlate-nor`): use SciPy (`pearsonr`, `spearmanr`) when available; otherwise, coefficients are reported and p‑values are `NaN`.

- Edge cases and robustness:
  - Very small datasets (few animals) or very small `--n-boot` yield wider CIs or NaNs; rows are still written.
  - NaNs are dropped before resampling. If a group has no samples, `n=0` and percentiles/diffs are NaN.
  - Pair keys not present in the dataset are skipped silently.

- Performance and reproducibility:
  - Use `--jobs N` with `--parallel-scope windows` to parallelize across windows per ROI.
  - Reduce `--n-boot` or `--q` for exploratory runs; increase for final results.
  - `--seed` ensures reproducible bootstrap; `--load-cache` reuses CSVs.

## 2) Plot from CSVs (plot-only)

Script: `scripts/plot_speed_bootstrap.py`

Purpose: draw figures directly from existing CSVs (no compute). The plot script is self‑contained and does not call other CLIs; ideal for fast iteration and clean separation of concerns.

Inputs expected:
- `paths['speed']/<outdir>/speed_bootstrap_quantiles.csv`
- `paths['speed']/<outdir>/speed_bootstrap_diffs.csv`

Figures (under `paths['f_speed']/<outdir>/`):
- By-window summaries per ROI/pair: `bywin_<roi>_<A>_vs_<B>.<fmt>`
- Grids per ROI aggregating all pairs: `bywin_grid_<roi>.<fmt>`
- Pooled diffs (short/long/all), per ROI and pair: `pooled_diffs_<roi>_pool-<pool>_<A>_vs_<B>.<fmt>`
- Pooled quantiles (short/long/all), per ROI: `pooled_quantiles_<roi>_pool-<pool>.<fmt>`

Example:

```bash
python scripts/plot_speed_bootstrap.py \
  --tr 500 --subset regions500 \
  --plot-diffs-by-win --plot-diffs-bywin-grid --bywin-grid-cols 2
```

Options:
- `--plot-format png|pdf|svg` (default `png`).
- `--outdir` optional; defaults to `--subset` if omitted.
- Use `--append-subset-to-outdir` to suffix when reusing an outdir across subsets.
- `--plot-diffs-by-win`: per ROI and pair, plot diff(A−B) vs window.
- `--plot-diffs-bywin-grid`: per ROI, grid aggregating all pairs (by-window).
- `--plot-pooled-diffs`: per ROI and pair, plot pooled (short/long/all) diffs with CIs.
- `--plot-pooled-quantiles`: per ROI, plot pooled (short/long/all) per‑group quantiles with CIs.
  - Legends on pooled diffs include per‑quantile p‑values (from the diffs CSV). Filled markers indicate CI excludes 0.
- `--progress`: show progress bars (requires tqdm); falls back to plain iteration if not installed.

Correlation plots (from `speed_nor_correlations.csv`)

Script: `scripts/plot_speed_correlations.py`

- What: correlation (Spearman or Pearson) vs window, and pooled (`short|long|all`).
- Inputs: `paths['speed']/<outdir>/speed_nor_correlations.csv` (enable `--correlate-nor` in compute).
- Figures:
  - `cor_bywin_<roi>_<metric>.<fmt>`
  - `cor_pooled_<roi>_<metric>.<fmt>`
- Options:
  - `--metric spearman|pearson` (default `spearman`)
  - `--alpha 0.05`: significance threshold; filled markers indicate `p <= alpha`
  - `--plot-by-win`, `--plot-pooled`, `--progress`, `--plot-format`, `--outdir`, `--append-subset-to-outdir`

Pool‑test plots (from `speed_bootstrap_pooltest*.csv`)

Script: `scripts/plot_speed_pooltest.py`

- Inputs (searched in order): `speed_bootstrap_pooltest_explicit_nboot-*.csv`, `speed_bootstrap_pooltest_nboot-*.csv`, then non‑suffixed.
- What: target percentile vs CI of pooled control, per window and pooled (`short|long|all`).
- Titles automatically annotate the pooled control (e.g., `vs pool by genotype=WT; exclude self`).
- Examples:
  ```bash
  python scripts/plot_speed_pooltest.py --tr 500 --subset dmn_within --bywin --pooled --progress
  python scripts/plot_speed_pooltest.py --tr 500 --subset dmn_within --bywin --q 50
  ```

## 3) Common Flags and Tips

- ROI handling: scripts infer `roi` from NPZ filenames; pooling is enforced per ROI.
- CSV schema:
  - Quantiles: `region, roi, window, group, q, point, lo, hi, n`
  - Diffs: `region, roi, window, A, B, q, diff, lo, hi, p, p_method, significant, n_a, n_b`
  - NOR correlations: `region, roi, window, q, pearson_r, pearson_p, spearman_rho, spearman_p, n, nor_col`
- Plot cosmetics:
  - Per-percentile color is stable and consistent with legends.
  - `bywin_*` plots show filled circles for significant points and open circles for ns.
- Performance on shared nodes: limit BLAS threads to avoid oversubscription.
  - Compute script auto‑limits threads when `--jobs > 1`; you can also export `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.

## 4) Typical Workflow

1) Compute once (parallel), write CSVs:
```bash
python scripts/compute_speed_bootstrap.py \
  --tr 500 --subset regions500 --tau-index 0 \
  --pool-threshold median --pool-all \
  --n-boot 2000 --jobs 8 --parallel-scope windows --progress
```
2) Plot many times (fast), tweak styles/pairs without recompute:
```bash
python scripts/plot_speed_bootstrap.py \
  --tr 500 --subset regions500 \
  --plot-diffs-by-win --plot-diffs-bywin-grid --bywin-grid-cols 2
```

## 5) Troubleshooting

- If `plot_speed_bootstrap.py` complains about missing CSVs, run the compute step first (or confirm `--subset`/`--outdir`).
- If legends/markers look off, re-run plotting with a simpler `--q` (e.g., `--q 5,50,95`) to declutter.
- Performance: reduce `--n-boot` for exploratory runs; use `--jobs` for window‑level parallelism.
- Empty groups/pairs: if a group or pair has no samples (e.g., a subset excludes them), the script writes NaNs rather than failing; diffs CIs/p-values still reflect bootstrap where possible.

## 6) Batch Runs

Script: `scripts/run_bootstrap_batches.sh`

- Actions: `both` (default), `compute`, `plot`, `list` (print subsets), `dry-run` (print commands only).
- Subsets processed by default:
  - `regions500`
  - `<name>_<flag>` for `name in {dmn,memory,sal,lat,1st,2nd,3rd,4rd}` and `flag in {within,touching}`.

Examples
- Run compute+plot for all subsets:
  - `bash scripts/run_bootstrap_batches.sh both`
- Preview planned subsets and commands:
  - `bash scripts/run_bootstrap_batches.sh list`
  - `bash scripts/run_bootstrap_batches.sh dry-run`

Environment overrides (prefix variables before the command)
- Syntax: `VAR=value VAR2=value bash scripts/run_bootstrap_batches.sh ACTION`
- Variables:
  - Compute: `TR`, `TAU_INDEX`, `N_BOOT`, `JOBS`, `CHUNK`, `GROUP_COLS`
  - Pools: `POOL_THRESH` (`median` or int), `POOL_ALL` (1/0)
  - Pool‑test: `POOL_COLS` (subset of group cols, e.g., `genotype`), `POOL_EXCLUDE_SELF` (1/0), `N_ANIMALS`
  - Caching/Perf: `REUSE_GROUP_BOOTS` (1/0), `BOOTS_FLOAT32` (1/0), `LOAD_CACHE` (1/0)
  - Correlations: `CORRELATE_NOR` (1/0), `CORRELATE_NOR_BY_GROUPS` (1/0)
  - Plotting: `BYWIN_GRID_COLS`, `PLOT_FORMAT`, `PLOT_PROGRESS` (1/0), `PLOT_POOLTEST` (1/0)
  - Output routing: `OUTDIR` (empty → use subset)

Examples (overrides)
- Quick smoke: `N_BOOT=500 JOBS=4 CHUNK=128 LOAD_CACHE=0 bash scripts/run_bootstrap_batches.sh both`
- Column‑based pool‑test: `POOL_COLS=genotype POOL_EXCLUDE_SELF=1 bash scripts/run_bootstrap_batches.sh compute`
- Plot only (incl. pool‑tests): `PLOT_POOLTEST=1 bash scripts/run_bootstrap_batches.sh plot`

## 7) Paths and Cluster Notes

- Configure roots with `PATHS_ROOT` or `PATHS_ENV` + `PROJECT_ROOT_<ENV>`.
- Validate with `scripts/paths_doctor.py --show --check-write --create`.
- If raw data are read‑only, you can symlink or mirror them into your root at `dataset/<DATASET_NAME>/...` to keep the layout consistent.
- Long runs without Slurm:
  - `tmux new -s boots && bash scripts/run_bootstrap_batches.sh both |& tee boots.log`
  - or `nohup bash scripts/run_bootstrap_batches.sh both > boots.out 2>&1 & disown`

## 8) Bootstrap Kernels and Centralization Phases

- Kernels live in `shared_code.fun_bootstrap` (preferred import path in scripts). Scripts fall back gracefully when central kernels aren’t available.

- Phases to centralize and simplify:
  - Phase 0 (current): scripts prefer central kernels; local fallbacks exist for compatibility.
  - Phase 1: standardize all imports in scripts/docs to `shared_code.fun_bootstrap`.
  - Phase 2: remove duplicated local bootstrap implementations; rely solely on central kernels with tests.
  - Phase 3: unify performance knobs (dtype/index/chunk), default `--reuse-group-boots`, document thread limits.
  - Phase 4: stabilize API, doc examples for programmatic use, ensure CI coverage for percentile/diff/p‑value semantics.
