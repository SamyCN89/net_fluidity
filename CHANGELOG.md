# Changelog

All notable changes to this repository are documented in this file.

The format follows Conventional Commits and semantic versioning where practical.

## [Unreleased]

### Added
- docs: Context guide `docs/Context_Guide.md` documenting `DFCAnalysis` (SoT), lifecycle, schema, examples, and validation commands.
- build: Makefile targets for dFC speed bootstrap — `speed-doctor`, `speed-compute`, `speed-plot`, `speed-pooltest`, `speed-cor`, and `help-speed`.
- test: Synthetic end-to-end smoke tests — `tests_smoke/test_end_to_end_compute.py` (compute CSVs) and `tests_smoke/test_plot_from_csv.py` (plot from CSVs).

### Changed
- docs: `docs/Compute_and_Plot_Tutorial.md` clarifies SoT usage, `--parallel-scope`/`--region-jobs`, and NOR column source; adds `paths_doctor.py --check-context`.
- docs: `docs/Bootstrap_Speed_CLI_Tutorial.md` marked as legacy with pointers to the split compute/plot flow; added SoT and context notes.
- docs: `docs/ENVIRONMENTS.md` includes a context validation snippet.
- docs/README: Added Makefile quick-start for compute/plot and target descriptions.

### Fixed
- docs: Removed outdated flags/artifacts; aligned CLI help and examples with current scripts.

## [0.1.0] - 2025-10-01

- Initial snapshot for this collaborative session (reference point only).

---

Guidelines

- Use Conventional Commits in entries, e.g. `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`.
- Summaries should be concise; link PRs/issues where relevant.
- For algorithmic changes, add a note on expected effect and attach evidence in PR description (plots/timings).
