# Bootstrap Speed CLI — Notebook‑Style Tutorial

This tutorial shows how to use `scripts/bootstrap_speed_groups_cli.py` to bootstrap dFC speed distributions per group, per region and window, and optionally pool across windows. You can paste the code blocks into a Jupyter notebook (Markdown + code cells).

## 0) Setup

```bash
pip install -e shared_code
# Optional dev tools (linters/tests/plotting)
pip install -r requirements-dev.txt
```

Ensure `.env` has a valid project root:

```
PROJECT_ROOT_LOCAL=/absolute/path/to/project/root
DATASET_NAME=julien_caillette  # optional; default in code
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
  --pool-threshold median \
  --outdir reports
```

Parameters used:
- taus: only one (tau index 0). Use `--tau-index -1` to pool all available taus.
- bootstraps: `--n-boot 2000` (default).
- groups: `--group-cols genotype,treatment` (default).
- percentiles: `--q 1,5,50,95,99` (default).
- pool threshold: `median` splits windows into short (<= median) and long (> median); per‑window rows are still produced, pools add extra rows.

Outputs:
- `reports/speed_bootstrap_quantiles.csv`
- `reports/speed_bootstrap_diffs.csv`

## 3) With Plots and Grids

```bash
python scripts/bootstrap_speed_groups_cli.py \
  --tr 500 --subset regions500 --tau-index 0 \
  --pool-threshold median --pool-all \
  --plot --grid --grid-cols 2 \
  --outdir reports
```

Figures go to `fig/<dataset>/speed/<subset>/...` (or `reports/figs` fallback).

Progress bars (optional): add `--progress` (requires `tqdm`).

## 4) Custom Groups and Pairs

```bash
python scripts/bootstrap_speed_groups_cli.py \
  --tr 500 --subset regions500 --tau-index 0 \
  --group-cols genotype,treatment \
  --pairs "(WT,VEH)-(WT,LCTB92);(Dp1Yey,VEH)-(Dp1Yey,LCTB92);(WT,VEH)-(Dp1Yey,VEH);(WT,LCTB92)-(Dp1Yey,LCTB92)" \
  --pool-threshold median --plot
```

## 5) Read CSV Outputs

```python
import pandas as pd
qdf = pd.read_csv('reports/speed_bootstrap_quantiles.csv')
ddf = pd.read_csv('reports/speed_bootstrap_diffs.csv')
sorted(qdf['region'].unique())[:5], sorted(qdf['window'].unique(), key=str)[:5]
```

## 6) Troubleshooting

- Tau index: run `--show-tau` first or use `--tau-index -1` to pool all taus.
- Empty groups/windows: rows with `n=0` and NaN CIs; plots skip gracefully.
- Subset path: ensure it points to per‑window NPZs in `regions-<label>/` or `all/`.

## 7) End‑to‑End Example

```bash
python scripts/bootstrap_speed_groups_cli.py --tr 500 --subset regions500 --show-tau

python scripts/bootstrap_speed_groups_cli.py \
  --tr 500 --subset regions500 --tau-index 0 \
  --pool-threshold median --pool-all \
  --plot --grid --grid-cols 2 \
  --progress \
  --outdir reports
```

## 8) Possible Additions

- Filtering: `--include-regions`, `--window-min/--window-max`.
- Statistics: `--stat` selector, interaction effects export, FDR marking.
- Metadata: write a JSON sidecar describing run parameters and discovered windows.
- UX: `--dry-run` to list work units; richer progress/reporting.
- Visualization: theme/size options; default SVG export.

