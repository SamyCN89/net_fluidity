# Changelog

All notable changes to this repository are documented in this file.

The format follows Conventional Commits and semantic versioning where practical.

## [Unreleased]

- docs: add top-level CHANGELOG.md documenting upcoming cleanup work.
- docs(julien_data): update USAGE_TUTORIAL with baseline verification steps and smoke commands.
- meta: no behavior or API changes; computation code unchanged.
- feat(julien_data): add thin CLI wrapper `julien_data/src/speed_compute.py` delegating to `3_dfc_speed_test_v6.py` (runs original as subprocess to keep joblib behavior).
- chore(scripts): add `scripts/compare_speed_outputs.py` to validate parity between subfolder outputs (shapes, NaN masks, values with tolerance).
- feat(julien_data): add `src/preprocess.py` wrapper CLI (imports and calls original with `--filter-mode`).
- feat(julien_data): add `src/dfc_stream_compute.py` wrapper CLI (subprocess delegate to original stream script).
- feat(julien_data): add `src/dfc_stream_cli.py` single-pass/split CLI for predictable DFC stream runs (no community loops).
- feat(julien_data): add `--tr` to `dfc_stream_cli` to select metadata by total_tr.
- feat(julien_data): add `--only-tr` to preprocess wrapper to generate filtered preprocessed data for a specific timepoint length.
- feat(julien_data): add `--tr` to `3_dfc_speed_test_v6.py` to select metadata by total_tr; no algorithm changes.
- feat(julien_data): add `--tr` to `speed_plots.py` (renamed from plot_merged_speed.py) to plot using a specific metadata set; add wrapper `src/speed_plots_cli.py`.
- feat(julien_data): implement equal‑animal weighting options in `speed_plots.py` (KDE averaging and subsample mode).
- feat(julien_data): add community plotting CLI `src/community_speed_plot.py` (reads merged PKL + communities and generates per‑community plots).
- chore(julien_data): add `community-plot` target to Makefile; update README and tutorial with usage.
- refactor(julien_data): start Phase 2 — add `src/plots_utils.py` and switch `speed_plots.py` to use shared helpers (pooling, equal-animal weighting, window split).
- feat(julien_data): add cognition correlation plots to `speed_plots.py` (scatter and rho vs window), with flags for reducer, weighting, and equalization.
- docs(julien_data): README documents new plotting flags and usage.
- feat(pkg): begin Phase 3 — add `src/net_fluidity_julien` package with `context.py` (DFCAnalysis) and `__init__`; update scripts to prefer package imports with fallbacks.
- docs(julien_data): add Makefile with handy targets (preprocess/dfc/speed/compare/plot) and update README with quick commands.
- feat(shared_code): add `dfc_speed_multi_tau` to `shared_code.shared_code.fun_dfcspeed` matching legacy multi‑tau semantics.
- test: add smoke parity test `tests_smoke/test_multi_tau_parity.py` comparing shared vs legacy on synthetic data (2D and 3D).
- feat(julien_data): add `--engine {legacy,shared}` to `3_dfc_speed_test_v6.py`; default remains legacy. Shared uses `dfc_speed_multi_tau`.
- chore(julien_data): extend Makefile with `speed-shared` and `ab-speed` (orig+shared+compare) targets.
- docs(julien_data): document engine selection in USAGE_TUTORIAL.md.
- refactor(julien_data): move legacy scripts to `julien_data/legacy/` (plts_speed.py, plots_speed.py, 3_dfc_local_speed_v1.py, local_speed_plot.py, local_speed_plot_v3.py) and update README/Tutorial; rename files for clarity (community plots and speed plots).

## [0.1.0] - 2025-10-01

- Initial snapshot for this collaborative session (reference point only).

---

Guidelines

- Use Conventional Commits in entries, e.g. `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`.
- Summaries should be concise; link PRs/issues where relevant.
- For algorithmic changes, add a note on expected effect and attach evidence in PR description (plots/timings).
