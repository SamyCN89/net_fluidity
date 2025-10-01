# TODOs — Net Fluidity (Julien)

This is a checkpoint of what’s done and what’s next so we can resume quickly.

## Status (quick)
- Phase 1: CLIs + parity + docs — done
- Phase 2: Plot utils + equal-animal + community plots — done
- Phase 3: Package start (context + plots utils; adapters) — done (initial)

## Next (Phase 3 — finish migration)
- Package moves:
  - Add `net_fluidity_julien/plots.py` (adapter) — done
  - Add `net_fluidity_julien/community.py` (adapter) — done
  - Optionally move plotting logic fully into package modules and keep julien_data/ wrappers as thin delegators
- Packaging (optional but recommended):
  - Add minimal `pyproject.toml` to support `pip install -e .` for `net_fluidity_julien`
  - Switch imports to package form by default (no PYTHONPATH needed)
- Validation:
  - Re-run TR=400/500 flows (dfc → speed → plots) with package imports
  - A/B shared vs legacy engine across multiple windows; keep comparator logs

## Phase 4 — Flip defaults + deprecate
- Change default `--engine` to `shared` in `3_dfc_speed_test_v6.py` after A/B parity on TR=400/500
- Deprecate legacy plotting scripts; mark legacy wrappers with warnings
- Update tutorial/README to emphasize package CLIs and new defaults

## Phase 5 — Polish + consistency
- CLI consistency and help polish across preprocess, dfc, speed, plots
- Community plots: integrate per-animal summaries and significance markers
- Standardize manifests and quick reports (counts/shapes/timings) for compute runs

## Tests & CI
- Expand smoke: plots rendering (headless), manifest presence, shape checks
- (Optional) add CI with pytest smoke and `make report`

## Nice-to-haves
- Entry points to run `python -m net_fluidity_julien.plots ...` and `...community`
- Example notebooks pointing to CLI outputs

---

Short commands reminder:
- Preprocess: `make -C julien_data preprocess TR=400`
- Streams: `make -C julien_data dfc-all TR=400`
- Speed: `make -C julien_data speed-shared TR=400 SELECTED="1,4,9" SUBSET_B=shared`
- Compare: `make -C julien_data compare SUBSET_A=legacy SUBSET_B=shared WIN=9`
- Plots: `python julien_data/src/speed_plots_cli.py --tr 400 --subset-name shared --tau 0 --savefig`
- Community: `make -C julien_data community-plot TR=400 PLOT_SUBSET=shared POOL=all`
