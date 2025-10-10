# TODO — Next Session

- Cohesion pipeline polish (new CLI + stats)
  - Makefile: expose unified ROI flags (ROI, ROI_INDICES, ROI_LABELS, ROI_FILE), EMIT, TAG, OVERWRITE, PLOT, ANIMAL, SHOW; wire into cohesion-compute target.
  - Deprecations: print clear warnings for legacy compute flags (`--roi-scope/--dmn-index/--memory-index`, `--plot-animal/--save-all-binary/--no-show`) and update help strings.
  - Add `--dfc-preset {default,fast,thorough}` to cohesion_compute (maps ws/lag/tau) with override precedence.
  - Predefined ROI sets: support `--roi memory-narrow|memory-broad` assembled from label patterns; document included labels.
  - Group stats: add optional Welch t-test for independent samples; export `pvals_group_ttest_*.csv` + heatmaps.
  - P-adjust: add `--p-adjust fdr-bh` (Benjamini–Hochberg) alongside existing Bonferroni and Bonferroni-by-age.
  - Matrix figures: add legend snippet and optional auto vmin/vmax per comparison; cache ROI mapping for speed.
  - Validation: check ROI indices in unified/custom modes against sorted labels; print helpful diffs on mismatch.
  - Manifest: include selected ROI labels list; include CLI args and git commit hash if available.
  - Tests: unit tests for `_bonferroni_by_age_in_columns`, `_infer_roi_order_from_pairs`, `compute_time_ratio_and_binary` shapes.

- Paths & Environment
  - Verify PATHS_ROOT or PATHS_ENV + PROJECT_ROOT_<ENV> configuration; run `python scripts/paths_doctor.py --show --check-write --create`.
  - Optionally call `paths_doctor.py` at the start of `scripts/run_bootstrap_batches.sh` behind a `DOCTOR=1` env gate.

- Smoke E2E Validation (small n_boot)
  - Compute: `python scripts/compute_speed_bootstrap.py --tr 500 --subset regions500 --tau-index 0 --pool-threshold median --pool-all --n-boot 200 --reuse-group-boots --chunk 128 --bootstrap-pool-cols genotype --pool-exclude-self --jobs 4 --progress`.
  - Plot bootstrap: `python scripts/plot_speed_bootstrap.py --tr 500 --subset regions500 --plot-diffs-by-win --plot-diffs-bywin-grid --bywin-grid-cols 2`.
  - Plot pool-test: `python scripts/plot_speed_pooltest.py --tr 500 --subset regions500 --bywin --pooled`.
  - Plot correlations (if computed): `python scripts/plot_speed_correlations.py --tr 500 --subset regions500 --metric both --plot-by-win --plot-pooled`.
  - Check outputs: n-boot suffixed CSVs exist; pool-test CSV includes metadata (`group_label, group_cols, pool_by, pool_match, pool_exclude_self`); plotting picks explicit/suffixed files first.

- Explicit Pool-Test Cross-Check
  - Run: `python scripts/compute_speed_pooltest_explicit.py --tr 500 --subset dmn_within --tau-index 0 --group-cols genotype,treatment --targets "(Dp1Yey,LCTB92);(WT,VEH)" --pool "(WT,VEH);(Dp1Yey,LCTB92)" --n-boot 200 --pool-threshold median --pool-all`.
  - Compare rows vs column-based pool-test when equivalent; verify `pool_by=explicit` and readable `pool_match`.

- Batch Runner
  - Dry-run: `bash scripts/run_bootstrap_batches.sh dry-run` (ensure pool-test/plot/correlation commands appear as expected).
  - Partial compute: `N_BOOT=1000 PLOT_POOLTEST=1 bash scripts/run_bootstrap_batches.sh both |& tee boots.log`.
  - Confirm BLAS thread capping on multi-job runs (oversubscription avoidance).

- Performance & Robustness
  - Measure memory/throughput across `--chunk {128,256}` and `--jobs {4,8}`; record recommended defaults.
  - Verify `--n-animals` semantics (0 or negative → use all) and align docs accordingly.
  - Ensure plotting handles missing pools/windows gracefully with warnings (already added; verify messages).

- Enhancements
  - CSVs: propagate `group_label` into quantiles/diffs CSVs for readability.
  - Paths: support separate `RESULTS_ROOT` / `FIGS_ROOT` overrides (avoid symlinks) in `fun_paths`.
  - Compute resume: detect completed regions/windows and skip; allow `--resume`/`--skip-existing`.
  - Metadata sidecar: write a JSON with run parameters and discovered windows (for reproducibility).
  - Unit tests: add tests for `bootstrap_group_from_pool` (inside/CI/p-values), and `bootstrap_diff_percentiles` parity.
  - Batch pre-check: integrate `paths_doctor.py` under `DOCTOR=1` in `run_bootstrap_batches.sh`.

- Docs & Examples
  - Add quick smoke test block (n_boot=200) at the top of tutorials.
  - Add a concise README snippet for pool-tests (column-based vs explicit) with examples.
  - Expand troubleshooting: what to do if pool-test CSVs are missing (compute flags to enable).
  - Ensure terminology consistency (`bywin` vs `by-window`) across docs.
  - Add `.env.sample` with PATHS_ROOT / PATHS_ENV / PROJECT_ROOT_<ENV> / DATASET_NAME.

- Data Tuning for Preprocessing
  - Standardize raw cognitive file naming: provide `dataset/<DATASET_NAME>/cog_data/ROIs.xlsx` (or update `fun_paths`/`DFCAnalysis` to a single filename) to avoid doctor warnings.
  - Document minimum schemas:
    - Preprocessed cognitive CSV: `genotype`, `treatment`, and `index_NOR` (or chosen `--nor-col`).
    - Metadata pickle keys (`mouse_metadata`, `region_labels`, `n_animals`, `regions`, `total_tr`, `lag`, `tau`, `window_range`).
  - Add a small conversion script/notebook to build the preprocessed cognitive CSV from raw Excel (column selection, renaming, filtering), and write alongside metadata.
  - Extend `paths_doctor.py` to recognize alternative raw filenames and report which one DFCAnalysis expects; downgrade warnings when preprocessed CSV is present.
  - Provide a sample dataset scaffold generator for tests/demos (folders + tiny CSV/NPZ) and document how to plug a custom dataset via `PATHS_ROOT`.

- Stretch Goals
  - Pooling rule variants: support include/exclude lists in addition to column-based matching.
  - Plot controls: expose alpha for pool-test “inside CI” markers and per-quantile styling.
  - CSV formats: option to emit wide pivoted tables alongside long format.
  - HPC: optional Slurm submission helpers (separate doc), keeping current nohup/tmux path for non-Slurm clusters.
