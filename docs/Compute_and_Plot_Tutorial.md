# dFC Speed Bootstrap — Compute and Plot Tutorial

This tutorial splits the workflow into two clean steps: compute CSVs only, then plot from CSVs. It uses two thin scripts that wrap the main CLI.

## 0) Setup

```bash
pip install -e shared_code
# Optional: plotting/dev tools
pip install -r requirements-dev.txt
```

Ensure your `.env` has project root:

```
PROJECT_ROOT_LOCAL=/absolute/path/to/project/root
DATASET_NAME=julien_caillette  # optional
```

## 1) Compute CSVs (no plotting)

Script: `scripts/compute_speed_bootstrap.py`

Purpose: compute per-ROI, per-window (and pooled short/long/all) bootstrap tables as CSVs. No figures are generated; you can plot later.

Outputs (under dataset paths):
- `paths['speed']/<outdir>/speed_bootstrap_quantiles.csv`
- `paths['speed']/<outdir>/speed_bootstrap_diffs.csv`

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

Behavior and outputs
- ROI enforcement: window pools (short/long/all) are computed per ROI (no cross‑ROI mixing).
- CSVs written to `paths['speed']/<outdir>/`:
  - `speed_bootstrap_quantiles.csv`: per‑group percentiles + CIs; columns `region, roi, window, group, q, point, lo, hi, n`.
  - `speed_bootstrap_diffs.csv`: per‑pair percentile diffs + CIs; columns `region, roi, window, A, B, q, diff, lo, hi, significant, n_a, n_b`.

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
  - Diffs: for each `(A,B)` pair in `--pairs`, independently bootstraps pooled A and B and computes percentile(A) − percentile(B) with a CI; writes rows to diffs CSV.
  - Missing pairs: if a pair key is not present in `groups_map` for this dataset, it is skipped (no error).

- Pooled windows (short/long/all):
  - Window pools are per‑ROI, based on `--pool-threshold`:
    - `median`: `short` = windows ≤ median; `long` = windows > median.
    - `INT`: explicit threshold split.
    - `--pool-all`: adds an `all` pool across all ROI windows.
  - For each pool, concatenates per‑animal arrays across windows, then applies the same quantile/diffs bootstrap as per‑window; writes rows to CSVs with `window=short|long|all`.

- Bootstrap details:
  - Non‑parametric resampling with replacement over the pooled (per‑group) samples; vectorized in chunks for speed.
  - `--n-boot` controls the number of resamples; `--ci` sets the CI level.
  - Diffs resample A and B independently; significance in CSV means the CI excludes 0.

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

Purpose: draw figures directly from existing CSVs (no compute). Ideal for fast iteration and clean separation of concerns.

Inputs expected:
- `paths['speed']/<outdir>/speed_bootstrap_quantiles.csv`
- `paths['speed']/<outdir>/speed_bootstrap_diffs.csv`

Figures (under `paths['f_speed']/<outdir>/`):
- By-window summaries per ROI/pair: `bywin_<roi>_<A>_vs_<B>.<fmt>`
- Grids per ROI aggregating all pairs: `bywin_grid_<roi>.<fmt>`

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

## 3) Common Flags and Tips

- ROI handling: scripts infer `roi` from NPZ filenames; pooling is enforced per ROI.
- CSV schema:
  - Quantiles: `region, roi, window, group, q, point, lo, hi, n`
  - Diffs: `region, roi, window, A, B, q, diff, lo, hi, significant, n_a, n_b`
- Plot cosmetics:
  - Per-percentile color is stable and consistent with legends.
  - `bywin_*` plots show filled circles for significant points and open circles for ns.

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
