# Metaconnectivity Cleanup & Migration Tracker

## Workflow Landmarks
- **Preprocessing bundle** → `metaconnectivity/cognitive_data_ts_sorted.py` (modern version rebuilt from legacy + `allegiance/src/prep_cog_groups.py`). Produces `ts_and_meta_2m4m.npz`, `cog_data_sorted_2m4m.csv`, and mask pickles via `shared_code.fun_paths`.
- **Meta-connectivity compute** → `metaconnectivity/compute_metaconnectivity_allegiance.py` (canonical legacy reference). Reads the preprocessing bundle, runs MC/allegiance with shared helpers, and caches NPZ files under `reports/metaconnectivity/<dataset>/`.
- **Plot/report** → `metaconnectivity/old_useful/plot_metaconnectivity_modularity.py` (to adapt next). Consumes the cached NPZ bundle and writes figures to `reports/metaconnectivity/<dataset>/figures/`.

## Observations (Resolved or In-Flight)
- Path handling now centralised on `shared_code.fun_paths.get_paths`; no more `/media/…` toggles or shadow `get_paths` implementations.
- Preprocessing script used to miss `import pickle` and rewrote `total_tp`; modern version fixes imports, persists `total_tr`, and exposes reusable functions/CLI.
- Duplicate `fun_*` modules remain in `metaconnectivity/old_useful/`; they stay only as thin shims until all callers migrate to `shared_code`.
- Legacy orchestration (`master_mc.py`) provided no unique behaviour; removal eliminates redundant path bootstrap logic.
- Prototype `1.0dfc_speed_data.py` added no functionality beyond existing shared kernels; safe to remove after verifying docs point to `shared_code.fun_dfcspeed`.

## Consolidated Action Items
1. **Preprocessing bundle**
   - [x] Merge legacy and allegiance preprocessing logic into `metaconnectivity/cognitive_data_ts_sorted.py`.
   - [ ] Add CLI tests or smoke coverage that exercises `prepare_cognitive_dataset(..., dry_run=True)` with synthetic data.
   - [ ] Document the bundle API in `docs/metaconnectivity-old-useful-workflow.md`.
2. **Meta-connectivity compute**
   - [ ] Refactor `compute_metaconnectivity_allegiance.py` into callable functions + Typer/argparse CLI.
   - [ ] Replace remaining imports of `metaconnectivity.fun_*` in active pipelines with `shared_code.fun_*`.
3. **Plot/report modernisation**
   - [ ] Convert `plot_metaconnectivity_modularity.py` into a parameterised module that reads from the new cache layout and validates prerequisites.
   - [ ] Move durable plotting utilities into `shared_code.fun_utils` (or new `shared_code.fun_plot`) with tests.
4. **Legacy retirement**
   - [x] Remove `metaconnectivity/master_mc.py` and `metaconnectivity/old_useful/master_mc.py` (no unique logic).
   - [x] Delete `metaconnectivity/old_useful/1.0dfc_speed_data.py`; any useful notes now live in documentation.
   - [ ] Audit remaining `old_useful/*.py` files; either archive truly historical artefacts or port them via the Adopt/Adapt decision tree.
5. **Reporting & governance**
   - [ ] Update `docs/architecture.md` to reference the new preprocessing entry point.
   - [ ] Track progress against `Phase 2 — Consolidate Meta` in `reports/refactor_issues.md`.

## Residual Risks
- Some exploratory scripts still assume cached NPZ/PKL names; a repo-wide search for `grouping_data_oip.pkl` is needed after renaming or schema changes.
- Numerical parity between the modern preprocessing bundle and historical outputs has not been revalidated post-refactor; schedule a small regression check.

## Next Verification Steps
- Run `python metaconnectivity/cognitive_data_ts_sorted.py --help` once the CLI is added.
- Execute `python metaconnectivity/compute_metaconnectivity_allegiance.py --dry-run` on synthetic data to ensure the cached layout stays consistent.
- Confirm the plotting script can render a sample figure using only artefacts produced through the modernised pipeline.
