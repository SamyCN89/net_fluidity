# Metaconnectivity Legacy Review (old_useful)

## Workflow Snapshot
- `metaconnectivity/old_useful/cognitive_data_ts_sorted.py:33` builds the dataset by intersecting raw time courses and cognitive scores, then persists `cog_data_sorted_2m4m.csv`, `ts_and_meta_2m4m.npz`, and grouping pickles under `root/results/sorted_data`.
- `metaconnectivity/old_useful/compute_metaconnectivity.py:51` and `compute_metaconnectivity_allegiance.py:46` load those artifacts, run the legacy MC + allegiance pipeline, and emit compressed NPZ bundles into `root/results/mc*` plus associated figures.
- `metaconnectivity/old_useful/plot_metaconnectivity_modularity.py:82` expects the NPZ bundles and grouping masks to exist, then generates figure exports into `paths['allegiance']`.
- Legacy orchestration scripts such as `metaconnectivity/old_useful/master_mc.py:80` duplicate the same path bootstrap logic and stitch the previous steps together.

## I/O and Path Blockers
- **Hard-coded roots:** Multiple modules gate execution behind `external_disk` flags but still hard-code user-specific directories (e.g., `/media/samy/…`, `/home/samy/…`). See `cognitive_data_ts_sorted.py:33-48`, `compute_metaconnectivity.py:51-66`, and `master_mc.py:80-106`. These paths bypass the `shared_code.fun_paths` contract and fail for anyone else.
- **Divergent `get_paths` helpers:** `fun_utils.get_paths` in `old_useful` reimplements the path map with a different schema and default roots (`fun_utils.py:36-67`). Other scripts import both the legacy and shared variants (`plot_metaconnectivity_modularity.py:71-97`, `master_mc.py:22-106`), leading to inconsistent keys and directories that may not exist.
- **Writes outside the repository:** Preprocessing saves directly into the external root (`cognitive_data_ts_sorted.py:196-207`), metaconnectivity saves to `root/results/mc*` (`compute_metaconnectivity.py:129-156`), and plotting writes figures into `paths['allegiance']` without guaranteeing the folders exist (`plot_metaconnectivity_modularity.py:192-193`). None of these respect the repo’s `reports/` policy and they silently create folders on the host machine.
- **Missing dependency loading:** `plot_metaconnectivity_modularity.py` now references `mask_groups` and `label_variables` but the loading block was commented out (`plot_metaconnectivity_modularity.py:109-111`), so the script raises a `NameError` before any plotting occurs.
- **Legacy data loaders:** Several scripts still call `load_matdata` and other helpers via `from fun_loaddata import *`, but `fun_loaddata.py` is no longer present in `old_useful`. The imports resolve only if the working directory coincidentally exposes the modern module, making these scripts fragile.

## Refactor Strategy
1. **Codify the dataset contract:** Decide which raw inputs (time courses, cognitive tables, labels) are still required and document them alongside the shared `get_paths` environment variables. Add a thin dataclass (e.g., `TimeSeriesBundle`) that exposes `ts`, `metadata`, and `group_masks`.
2. **Centralize loaders:** Move the reusable pieces of `cognitive_data_ts_sorted.py` into `shared_code.fun_loaddata` or a new `metaconnectivity/io.py`. Use `shared_code.fun_paths.get_paths` exclusively, call `create_directories`, and route new artifacts into `reports/metaconnectivity/<dataset>` rather than external disks.
3. **Modularize pipelines:** Refactor `compute_metaconnectivity*.py` so the computational core is a function (`run_metaconnectivity(bundle, params)`) that returns structured outputs. Expose CLI wrappers or scripts that parse parameters, call the core function, and write results through a single serialization helper.
4. **Modernize plotting:** Update plotting scripts to accept explicit inputs (paths to NPZ, grouping masks) and to write figures under `reports/` or `paths["figures"]`. Remove in-script path guessing and rely on the centralized loader to supply `mask_groups` et al.
5. **Validation:** Add smoke tests that execute the new loader + pipeline stack on tiny synthetic data to ensure the refactored workflow remains deterministic (`tests_smoke/`).

## Immediate Fixes
- Restore the grouping-data load in `plot_metaconnectivity_modularity.py` and align its path handling with `shared_code.fun_paths` so plotting works again.
- Replace legacy `get_paths` definitions with imports from `shared_code.fun_paths` everywhere, and delete the duplicates in `fun_utils.py` once call sites migrate.
- Audit every `np.savez`/`pickle.dump` call to ensure outputs stay inside the repository (ideally under `reports/metaconnectivity/`) and that the destination directories are created via the centralized helper before writing.
