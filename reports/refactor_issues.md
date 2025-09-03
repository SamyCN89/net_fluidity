# Net Fluidity — Refactor Phases: Issue Bodies

Copy/paste each section as a GitHub issue. Each includes summary, scope, tasks, acceptance criteria, and risks.

---

## Phase 0 — Foundations (settings, exports, hygiene)

- Title: Phase 0 — Foundations (settings, explicit exports, import hygiene)
- Summary: Centralize configuration; make `shared_code` exports explicit; enforce import hygiene via pre-commit/CI.
- Scope:
  - In: settings module; explicit `__all__` in `shared_code`; pre-commit checks; `.env.example`; docs.
  - Out: Functional refactors of compute modules.
- Tasks:
  - [ ] Add `net_fluidity/settings.py` (env parsing, defaults, validation; supports `PROJECT_ROOT_<ENV>`, `DATASET_NAME`).
  - [ ] Replace star re-exports in `shared_code/shared_code/__init__.py` with explicit symbols and `__all__`.
  - [ ] Wire `.tools/check_forbidden_imports.py` into pre-commit/CI; add Ruff rule to forbid star imports.
  - [ ] Add `.env.example` and docs for required env vars.
- Acceptance Criteria:
  - [ ] Importing from `shared_code` uses explicit symbols; no star re-exports remain.
  - [ ] Settings loads from env, validates, and surfaces clear errors.
  - [ ] Pre-commit/CI fails on forbidden imports/star imports; passes on main.
  - [ ] README/docs list env vars and examples.
- Risks/Dependencies: Low risk; enables later phases.
- Owner/Estimate/Labels: [TBD], 2–3 days; labels: `arch`, `infra`, `phase-0`.

---

## Phase 1 — De‑Monolith Core (split fun_dfcspeed)

- Title: Phase 1 — De‑Monolith Core (split `fun_dfcspeed`)
- Summary: Separate FC/DFC stream/speed concerns to improve readability, testability, and reuse.
- Scope:
  - In: Create `fc.py`, `stream.py`, `speed.py`; add types/docstrings; deprecation shims in `fun_dfcspeed`.
  - Out: Algorithmic changes to math.
- Tasks:
  - [ ] Create `shared_code/shared_code/fc.py` (FC/PLV computation).
  - [ ] Create `shared_code/shared_code/stream.py` (sliding windows, `ts2dfc_stream`).
  - [ ] Create `shared_code/shared_code/speed.py` (pearson/spearman/cosine speed functions).
  - [ ] Keep `fun_dfcspeed.py` as thin wrapper with deprecation warnings and pass-through exports.
  - [ ] Update imports in allegiance scripts and primary callers to use new modules.
  - [ ] Add type hints, docstrings, and targeted unit tests (golden outputs).
- Acceptance Criteria:
  - [ ] New modules return bitwise-equal or within tolerance vs prior implementation.
  - [ ] `fun_dfcspeed` warns but remains backward-compatible.
  - [ ] LOC of `fun_dfcspeed.py` reduced by >60%.
  - [ ] Tests pass in CI.
- Risks/Dependencies: Medium (import churn); depends on Phase 0.
- Owner/Estimate/Labels: [TBD], 3–5 days; labels: `arch`, `core`, `phase-1`.

---

## Phase 2 — Consolidate Meta (merge legacy → shared_code)

- Title: Phase 2 — Consolidate Meta (merge legacy into `shared_code.fun_metaconnectivity`)
- Summary: Fold needed functionality from `metaconnectivity/*` into `shared_code`, deprecate legacy modules.
- Scope:
  - In: Migrate stable functions; add tests; deprecation path; update orchestrators.
  - Out: New features in legacy scripts.
- Tasks:
  - [ ] Inventory legacy `metaconnectivity` functions and map to `shared_code` equivalents.
  - [ ] Port missing, stable functions; add unit tests and `tools/compare_shared_vs_meta.py` parity checks.
  - [ ] Add deprecation warnings and README notes in legacy modules.
  - [ ] Update imports in `allegiance/src/*` and `julien_data/*` to `shared_code`.
- Acceptance Criteria:
  - [ ] No code imports stable funcs from `metaconnectivity/*`.
  - [ ] Parity checks pass within tolerances.
  - [ ] Legacy modules emit deprecation on import.
- Risks/Dependencies: Medium; numerical parity and path differences.
- Owner/Estimate/Labels: [TBD], 3–4 days; labels: `arch`, `cleanup`, `phase-2`.

---

## Phase 3 — Orchestration CLI (dfc/mc/allegiance/merge)

- Title: Phase 3 — Orchestration CLI (unified pipelines)
- Summary: Provide a CLI to run core pipelines with consistent args, logging, and outputs.
- Scope:
  - In: `net_fluidity/cli.py` (Typer/Click) with `dfc`, `mc`, `allegiance`, `merge` subcommands; logging; config.
  - Out: SLURM submission (Phase 4).
- Tasks:
  - [ ] Scaffold `net_fluidity` package with `cli.py` and `__main__.py` entrypoint.
  - [ ] Implement subcommands mapping to shared_code functions.
  - [ ] Standardize args: `--env`, `--data-root`, `--dataset`, `--window-size`, `--lag`, `--n-jobs`, `--out`.
  - [ ] Centralize logging setup; ensure non-interactive plotting backend.
  - [ ] Add CLI docs and examples.
- Acceptance Criteria:
  - [ ] `python -m net_fluidity --help` shows expected commands.
  - [ ] Smoke runs write outputs under `fun_paths` locations.
  - [ ] Logs route to file and stderr with consistent format.
- Risks/Dependencies: Low/Medium; depends on Phase 0–1.
- Owner/Estimate/Labels: [TBD], 2–3 days; labels: `cli`, `pipelines`, `phase-3`.

---

## Phase 4 — SLURM/HPC Integration

- Title: Phase 4 — SLURM/HPC integration (sbatch/arrays)
- Summary: Generate sbatch scripts/array jobs with resource presets; align joblib threads.
- Scope:
  - In: `net_fluidity/slurm.py` (or `slurm/submit.py`); examples and docs.
  - Out: Cluster-specific modules.
- Tasks:
  - [ ] Render sbatch scripts for CLI subcommands (single/array jobs).
  - [ ] Presets map to `--time`, `--mem`, `--cpus-per-task` and env capping (`OMP_NUM_THREADS`, etc.).
  - [ ] Provide examples and README section with submission recipes.
- Acceptance Criteria:
  - [ ] Command renders ready-to-run sbatch files.
  - [ ] Scripts align `cpus-per-task` with joblib `n_jobs`.
  - [ ] Docs include array submission examples.
- Risks/Dependencies: Medium; cluster variability.
- Owner/Estimate/Labels: [TBD], 2–3 days; labels: `hpc`, `slurm`, `phase-4`.

---

## Phase 5 — Caching & Versioning

- Title: Phase 5 — Caching & Versioning for outputs
- Summary: Version `.npz` schema; add manifest with input/content hashes to prevent stale cache use.
- Scope:
  - In: Helpers in `shared_code.fun_loaddata` for `version`, `manifest.json`, `hash_inputs`.
  - Out: Changing file formats beyond adding metadata.
- Tasks:
  - [ ] Add version tag and provenance fields to writers (`save2disk`).
  - [ ] Compute content hash from parameters + shapes + seeds; write manifest.
  - [ ] Readers validate version/hash; invalidate cache on mismatch with clear logs.
  - [ ] Document cache policy and semantics.
- Acceptance Criteria:
  - [ ] All new outputs carry version/provenance.
  - [ ] Stale caches trigger recompute with clear messaging.
  - [ ] Tests cover manifest write/read and invalidation paths.
- Risks/Dependencies: Medium; avoid breaking existing caches silently.
- Owner/Estimate/Labels: [TBD], 3 days; labels: `infra`, `caching`, `phase-5`.

---

## Phase 6 — Tests & Benchmarks

- Title: Phase 6 — Tests & Benchmarks (core numerics)
- Summary: Add unit tests for numeric correctness and baseline performance checks.
- Scope:
  - In: `tests_core/` for fc/stream/speed/metaconnectivity; CLI smoke; optional pytest-benchmark.
  - Out: GPU/cluster-scale perf tests.
- Tasks:
  - [ ] Golden tests: FC/DFC/speed vs NumPy/Scipy on random and edge-case inputs.
  - [ ] Property tests (symmetry, unit diagonal, invariants).
  - [ ] CLI smoke tests using small synthetic data; headless plotting check.
  - [ ] Perf baselines for numba-enabled vs disabled paths (if available).
- Acceptance Criteria:
  - [ ] ≥90% coverage on split modules; core invariants tested.
  - [ ] CI passes reliably under CPU-only.
  - [ ] Benchmarks captured (non-gating) for key functions.
- Risks/Dependencies: Medium; numba compilation variance.
- Owner/Estimate/Labels: [TBD], 3–4 days; labels: `tests`, `benchmarks`, `phase-6`.

---

## Phase 7 — EDA Extraction (julien_data)

- Title: Phase 7 — EDA Extraction from `julien_data`
- Summary: Extract reusable logic from large EDA scripts into importable modules; move remaining flows to notebooks.
- Scope:
  - In: `shared_code.analysis` module; convert top N plots; deprecate duplicates (v1/v2).
  - Out: Full rewrite of all EDA.
- Tasks:
  - [ ] Identify top 3 reusable functions per large script; move and add light tests.
  - [ ] Create notebooks calling library functions; retire redundant scripts.
  - [ ] Update docs linking to notebooks and new APIs.
- Acceptance Criteria:
  - [ ] Reduced LOC in `julien_data` by ≥50% across targeted files.
  - [ ] No import-time side effects; plots reproducible via notebooks.
  - [ ] Key plots validated via CI smoke (Agg backend).
- Risks/Dependencies: Medium/High; user workflows may rely on scripts.
- Owner/Estimate/Labels: [TBD], 4–6 days; labels: `eda`, `cleanup`, `phase-7`.

---

## Phase 8 — Logging & Plotting Hygiene

- Title: Phase 8 — Logging & Plotting Hygiene
- Summary: Remove import-time side effects; add plotting and logging context management.
- Scope:
  - In: Plot style context manager; centralized logger config; no `rcParams` changes on import.
  - Out: Major aesthetic redesigns.
- Tasks:
  - [ ] Add `with plot_style(...):` context helper; replace global `rcParams.update` calls.
  - [ ] Ensure libraries do not call `matplotlib.use()`; tests set Agg backend.
  - [ ] Centralize `logging.config` loading; remove ad-hoc setups across modules.
- Acceptance Criteria:
  - [ ] Imports produce no logs or plotting side effects.
  - [ ] Smoke tests verify non-interactive backend usage.
  - [ ] Docs updated with plotting guidelines and examples.
- Risks/Dependencies: Low/Medium; touches common utilities.
- Owner/Estimate/Labels: [TBD], 2–3 days; labels: `logging`, `plots`, `phase-8`.

---

## Phase 9 — Docs & Usage

- Title: Phase 9 — Docs & Usage (architecture, CLI, SLURM)
- Summary: Publish concise architecture/usage guide and SLURM recipes.
- Scope:
  - In: `docs/ARCHITECTURE.md`, README updates, environment setup, CLI usage, SLURM guide.
  - Out: Full Sphinx site (optional later).
- Tasks:
  - [ ] Add repo map, dependency graphs, and module responsibilities.
  - [ ] CLI quickstart and config/env examples.
  - [ ] SLURM submission recipes and presets explanation.
- Acceptance Criteria:
  - [ ] Docs merged; team can run pipelines locally and on HPC from docs alone.
  - [ ] Links to `reports/architecture_refactor_plan.md` and notebooks.
- Risks/Dependencies: Low; consolidates prior phases.
- Owner/Estimate/Labels: [TBD], 2 days; labels: `docs`, `phase-9`.

---

## Optional — GitHub Issue Template

You can add a repo-wide issue template to standardize future issues:

```md
# Summary

# Scope
- In:
- Out:

# Tasks
- [ ] ...

# Acceptance Criteria
- [ ] ...

# Risks/Dependencies

# Owner/Estimate/Labels
```

