# Allegiance DFC Workflow

This note explains how the legacy `allegiance/src/dfc_compute.py` entrypoint behaves
now that the dFC computation logic lives in `scripts/dfc/dfc_compute.py`. Use it as a
quick reference for running or debugging the allegiance pipeline while the shim
remains in place.

---

## Overview
- `allegiance/src/dfc_compute.py` simply re-exports `main()` and helper functions from
  `scripts/dfc/dfc_compute.py`.
- Invoking `python allegiance/src/dfc_compute.py ...` is functionally identical to
  calling the centralized CLI under `scripts/dfc/`.
- All argument parsing, dataset defaults, bundle loading, caching, and logging are
  provided by the shared implementation. See `docs/dfc_compute.md` for full CLI
  semantics.

---

## Execution Flow
1. **CLI invocation** — Users can continue to call the legacy path:
   ```bash
   python allegiance/src/dfc_compute.py --dataset-name julien --wmin 5 --wmax 15 --tau 0,5 --lag 1
   ```
   The shim forwards the arguments to `scripts.dfc.dfc_compute.main`.

2. **Argument parsing** — `build_parser()` (imported from the shared module) defines all
   options, including multi-τ support (`--tau 0,5,10`), a single lag (`--lag 1`), cache
   policies, and `--dry-run`.

3. **Dataset resolution** — `DATASET_DEFAULTS` maps supported aliases (e.g. `julien`,
   `ines`) to the canonical bundle names and folders. Paths are constructed via
   `shared_code.fun_paths.get_paths`.

4. **Bundle loading** — `load_timeseries()` reads the preprocessed NPZ bundle,
   validates shape, and logs the source path. The allegiance CLI inherits this behaviour.

5. **Computation loop** — For each requested window size and τ value:
   - Skip or reuse existing outputs according to `--cache`.
   - Dispatch `shared_code.fun_dfcspeed.ts2dfc_stream` per animal (threaded if
     `--jobs > 1`).
   - Write `dfc_window_size=<W>_lag=<L>_tau=<T>_animals=<A>_regions=<R>.npz` under
     the dataset’s `paths["dfc"]` directory.

6. **Dry runs** — With `--dry-run`, the CLI only reports which files would be touched,
   which is helpful when validating cache behaviour in legacy allegiance workflows.

---

## Typical Usage
```bash
# Generate dFC outputs for Julien dataset, windows 5-15, τ ∈ {0,5}, skip existing files.
python allegiance/src/dfc_compute.py \
  --dataset-name julien \
  --wmin 5 --wmax 15 --wstep 5 \
  --lag 1 \
  --tau 0,5 \
  --cache skip \
  --jobs 4
```

```bash
# Inspect planned outputs without writing anything.
python allegiance/src/dfc_compute.py \
  --dataset-name ines \
  --wmin 9 --wmax 21 --wstep 6 \
  --lag 1 \
  --tau 3 \
  --dry-run
```

---

## Related Documentation
- `docs/dfc_compute.md` — canonical CLI reference shared by all entrypoints.
- `docs/architecture.md` — high-level overview of shared pipelines.
- `docs/preprocessing.md` — prerequisites for generating the NPZ bundles consumed here.

Keep this shim around until all allegiance automation calls `scripts/dfc/dfc_compute.py`
directly. Once migration is complete, the file can be deleted and callers should switch
to the centralized path.
