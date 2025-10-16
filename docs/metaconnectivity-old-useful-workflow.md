# metaconnectivity-old_useful Workflow

This guide helps triage code parked in `metaconnectivity/old_useful/` after promoting shared helpers. Follow it to decide when to reuse, adapt, or retire legacy functions while honoring repository governance.

---

## Decision Workflow
- **Clarify target workflow**: Identify the analysis or figure you need to reproduce and note the modern entrypoints under `scripts/` or `allegiance/src/`.
- **Check shared helpers**: Search `shared_code/shared_code/*.py` (use `rg`). If a helper already exists, adopt it rather than forking.
- **Assess gaps**: When shared helpers lack a required feature, decide whether to extend them (preferred) or temporarily wrap the legacy helper.
- **Classify the asset**:
  - `Adopt`: Identical logic already ported—switch imports and delete duplicates.
  - `Adapt`: Needs API/path cleanup before moving into `shared_code`.
  - `Retire`: Superior modern equivalent exists; keep only a pointer.
  - `Archive`: Data snapshots or exploratory scripts; leave untouched.
- **Plan the migration**: Pick a migration scheme (see below) and update docs if APIs move.
- **Validate**: Run `make check && pytest -q tests_smoke` (install `numba` first if missing).
- **Record outcome**: Update this inventory or issue checklist; add retired helpers to the No-Migrate table with their modern replacements.

---

## Migration Schemes

**Scheme 1 – Incremental Refactor**
1. Redirect live scripts in `metaconnectivity/` to import from `shared_code`.
2. Add thin compatibility wrappers where the legacy signature differs.
3. Once smoke tests pass, delete the duplicate helper in `old_useful`.
4. Extend shared helpers only when wrappers become unmanageable.

_When to pick_: Minimal disruption, easy regression tracking, fits ongoing analyses.

**Scheme 2 – Function-by-Function Evaluation (deep clean)**
1. Exercise each legacy helper in isolation and compare outputs with `shared_code`.
2. Equivalent → replace all call sites and remove the legacy function.
3. Non-equivalent → list under “No-Migrate” with the recommended modern substitute.
4. Extend `shared_code` only after documenting behaviour deltas and adding tests.

_When to pick_: Surfaces subtle drift; yields the cleanest repo but takes longer.

---

## Module Inventory

### Core Algorithms

| Legacy module | Key functions | Status | Action | Modern reference |
| --- | --- | --- | --- | --- |
| `fun_dfcspeed.py` | `ts2fc`, `ts2dfc_stream`, `dfc_speed`, `matrix2vec`, `window_pooling_speed`, `sort_modularity` | Adopt except `sort_modularity` | Use `shared_code.fun_dfcspeed.*`; drop `ts2dfc_stream_old`. Keep `window_pooling_speed` only as a wrapper that calls `shared_code.fun_dfcspeed.pool_vel_windows`. Replace modularity sorting with `shared_code.fun_metaconnectivity.allegiance_matrix_analysis`. | `shared_code.fun_dfcspeed`, `shared_code.fun_metaconnectivity` |
| `fun_metaconnectivity.py` | `compute_metaconnectivity`, `allegiance_matrix_analysis`, `fun_allegiance_communities`, `build_trimer_mask`, `trimers_by_apex` | Adopt | Prefer richer vectorised versions in `shared_code.fun_metaconnectivity`. Only `variables_selector` and `fun_mc_viscocity` remain legacy; evaluate need before porting. | `shared_code.fun_metaconnectivity` |
| `fun_optimization.py` | `fast_corrcoef*` family | Retire | `shared_code.fun_optimization` expands the API (`pearson_speed_vectorized`, `cosine_speed_vectorized`, `spearman_speed`). Keep legacy NumPy/Numba variants only if benchmarking proves necessary. | `shared_code.fun_optimization` |
| `fun_utils.py` | Loaders, phenotype helpers, plotting utilities, path helpers | Adapt | Merge only functionality missing from `shared_code.fun_utils` or `shared_code.fun_paths`. Avoid duplicating `dfc_stream2fcd`/`matrix2vec` (already in shared code). | `shared_code.fun_utils`, `shared_code.fun_paths` |
| `functions_analysis.py` | PLI analyses, FCD calculator | Adapt | No modern analogue yet. Port if still required, replacing `brainconn` and other MATLAB-era deps. | Add to `shared_code` once cleaned |

### Pipeline Scripts

| Script | Description | Status | Action | Modern replacement |
| --- | --- | --- | --- | --- |
| `compute_metaconnectivity.py`, `compute_metaconnectivity_allegiance.py` | Legacy MC pipelines | Retire | Use `metaconnectivity/compute_metaconnectivity_modularity.py` hooked into shared helpers. Preserve CLI notes only. | `metaconnectivity/compute_metaconnectivity_modularity.py` |
| `compute_genuine_trimers.py`, `metaconnectivity_allegiance_matrix_test*.py` | Prototype allegiance/trimers workflows | Retire | Modern flow: `metaconnectivity/compute_trimers.py` plus `shared_code.fun_metaconnectivity.compute_trimers_genuine`. | `metaconnectivity/compute_trimers.py` |
| `dfc_streams*.py`, `dfc_metaconnectivity.py` | Early dFC pipelines | Retire | Replace with shared helpers invoked from `allegiance/src` or `scripts/metaconnectivity`. | `shared_code.fun_dfcspeed` |
| `master_mc.py`, `Consolidate_data.py` | Legacy orchestration | Archive | Document historical context; avoid running. Prefer Make/CLI pipelines. | — |
| `plot_*` scripts | Visualization helpers | Adapt | Move reusable pieces into `shared_code.fun_plot` or notebooks once confirmed still needed. | `shared_code.fun_plot` (extend) |

### Data & Artifacts

| Item | Type | Action |
| --- | --- | --- |
| `.xlsx`, `.h5`, `.json`, `.mat` snapshots | Data | Keep ignored or move under `reports/`; never import directly. |
| `cognitive_data_ts_sorted.py`, `fc_individuals.py` | Data munging | Prefer `shared_code.fun_loaddata` or notebooks; reimplement reproducibly if still used. |

---

## No-Migrate List (Use Modern Equivalent)

| Legacy symbol | Rationale | Replacement |
| --- | --- | --- |
| `fun_dfcspeed.ts2dfc_stream_old` | Duplicate, slower sliding window implementation | `shared_code.fun_dfcspeed.ts2dfc_stream` |
| `dfc_streams.dfc_speed_series` | Superseded by oversampled/vectorised paths | `shared_code.fun_dfcspeed.dfc_speed_oversampled_series` |
| `fun_dfcspeed.sort_modularity` | Implicit MATLAB-style community ordering, differs from modern behaviour | `shared_code.fun_metaconnectivity.allegiance_matrix_analysis` |
| `dfc_windows_pooling.window_pooling_speed` | Duplicate pooling logic | `shared_code.fun_dfcspeed.pool_vel_windows` |
| `fun_utils.get_paths`, `fun_utils.get_root_path`, `fun_utils.extract_hash_numbers` | Hard-coded filesystem assumptions | `shared_code.fun_paths` utilities |

---

## Technical Notes

- **`shared_code.fun_dfcspeed.dfc_speed`**: Accepts 2D/3D dFC streams, slices frames via `vstep`, and computes speeds with `pearson`, `spearman`, or `cosine` kernels. Use `return_fc2=True` only when legacy tooling expects trailing FC indices.
- **`pearson_speed_vectorized` vs. `fast_corrcoef`**:  
  - `fast_corrcoef`: builds FC matrices from raw time series (window → FC).  
  - `pearson_speed_vectorized`: compares already vectorised FC frames to yield per-frame speeds.  
  They address different steps; keep both and ensure pipelines call the correct one.
- **Numba dependency**: Speed paths require `numba`. Install it before running checks: `python -m pip install numba`.
- **Pooling helpers**: Prefer `shared_code.fun_dfcspeed.pool_vel_windows`. Add missing behaviours there instead of reviving legacy pooling code.

---

## Testing Guidance

- Extend `tests_smoke/` with regression coverage for:
  - `shared_code.fun_dfcspeed.dfc_speed` (median and distribution checks against fixtures).
  - `shared_code.fun_metaconnectivity.compute_trimers_genuine` (compare with archived outputs stored under `reports/`).
  - Any adapted helpers brought over from `functions_analysis.py` or plotting modules.
- Use deterministic seeds (`np.random.default_rng(42)`) when fabricating time series for tests.
- For CLI scripts, create smoke wrappers that execute on tiny synthetic datasets (≤5 ROI, ≤50 frames) to guard imports.

---

## Open Follow-Ups

1. Confirm whether `variables_selector` or `fun_mc_viscocity` still serve active analyses; port with tests if yes.
2. Decide which plotting utilities should become documented examples or notebooks.
3. Track progress toward removing `metaconnectivity/old_useful` once all call sites migrate to shared helpers (maintain an issue checklist referencing this workflow).

