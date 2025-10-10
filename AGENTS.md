# Repository Governance & Agent Protocol

This document defines how contributors — human or automated — work safely and consistently inside the repository.
It replaces informal guidelines with explicit rules for coding, testing, reviewing, and documentation.

---

## 1. Intent & Principles

- **Clarity first:** every function and figure must be reproducible from data to output.
- **Determinism over speed:** prefer predictable pipelines; profile before optimizing.
- **Minimalism:** simplest implementation that meets the need (SOLID, KISS, YAGNI, DRY).
<>- **Testability:** all logic in `shared_code/` must have smoke or unit tests.
- **Security:** read/write only within project root; never embed credentials or data.

---

## 2. Project Structure & Code Boundaries

| Path | Purpose | Quality Expectation |
|------|----------|--------------------|
| `shared_code/` | Core package, reusable API | 🧪 Tests + docs required |
| `metaconnectivity/` | Legacy/experimental work | ⚠️ Optional tests |
| `allegiance/src/` | Pipelines & analysis scripts | ✅ Must run smoke tests |
| `julien_data/` | Demos/figures only | 🚫 No imports from here |
| `tools/`, `scripts/` | CI, lint, automation | 💡 No domain logic |
| `tests_smoke/` | Default pytest suite | 🧩 Fast deterministic tests |

## 3. Agent Protocol

Applies to both human developers and automated agents (e.g., Codex, Copilot Workspace).

1. **Read context** → skim `docs/architecture.md`, linked design notes, and the related GitHub issue.
2. **Plan** → post a short “Session Start” comment describing intended changes.
3. **Develop** → keep commits ≤ 400 LOC, run `make check && pytest -q`.
4. **Report** → at completion, post “What landed + Next steps” to the issue.
5. **Sync Docs** → if code layout or APIs change, update `shared_code/README.md` and `docs/architecture.md` in the same PR.
6. **Review** → ensure all changes are reviewed by a human before merging to `main`.

Scientific invariants over implementation
- When optimizing numerics, preserve the scientific definition and verify equivalence on representative data. For example, `allegiance/src/cohesion_compute.py` now uses a vectorized diff-based event extraction (pairing rising/falling edges) instead of a Python scan loop; this was validated to produce identical onsets/offsets/durations across randomized tests and multiple `min_duration` thresholds. Prefer such transparent optimizations that do not alter results.

---

## 4. Workflow & Quality Gates

- No commit of generated data, caches, or logs.
- Write clear, descriptive commit messages; prefer Conventional Commits style.
- Tie issues to roadmap goals in `docs/ROADMAP.md`.
- All changes to `shared_code/` must include or update tests in `tests_smoke/`.
- All public APIs must have docstrings and type hints.
- No unused imports or variables.
---

## 5. Testing & Reproducibility

- Use **pytest**; test files `test_*.py`.
- Fix random seeds (`np.random.seed` or `rng=...`).
- Smoke tests reproduce key analysis paths end-to-end.
- Integration tests mirror published figures or metrics.
- Use environment variables (`DATASET_NAME`, `PROJECT_ROOT_LOCAL`, `PROJECT_ROOT_CLUSTER`) for input paths; never hardcode.
<>- Store test results or metadata in `reports/` (ignored by git).
- Require report output.

---

## 6. Coding Style & Naming

- Formatter: **Black 88 chars**, target `py311`.
- Linter: **Ruff** rules E/F/W/I/UP/B; isort via Ruff.
- Type hints mandatory for public APIs.
- Naming → `snake_case` (fns), `PascalCase` (classes), `UPPER_SNAKE` (constants).
- No shadow imports from `metaconnectivity`; prefer `shared_code.*`.
  - Good: `from shared_code.fun_dfcspeed import ts2dfc_stream`
  - Avoid: `from metaconnectivity.fun_dfcspeed import ts2dfc_stream`
  - Exception: stable, non-domain utilities (e.g., `from metaconnectivity.fun_utils import save_npz`).


---

## 7. Security & Configuration

- Keep data paths external → `shared_code/fun_paths.py`.
- Configure `PROJECT_ROOT_LOCAL` (+ optional `DATASET_NAME`) in `.env`.
- Never commit data or credentials.
- Copy `config/logging.example.yaml` to customize logging locally.
- Use `.gitignore` to exclude local configs, data, logs.
- Avoid large files; use Git LFS if necessary.
- Prefer relative imports within `shared_code/`.
- Document new environment variables in `docs/architecture.md`.

---

## 8. Ambiguity Resolution

- Choose the smallest sufficient solution.
- Follow existing naming and folder patterns.
- When uncertain, document rationale in PR description.
- Human instructions in issue comments override defaults.
- For complex changes, discuss in issue comments before implementation.
- Prefer explicit over implicit; avoid “magic” behavior.
- Update docs immediately.
- Default to simplicity.
- Follow existing patterns, unless there’s a strong reason to change.

---

## 9. References

- Architecture: `docs/architecture.md`
- Roadmap: `docs/ROADMAP.md`
- Tests: `pytest.ini`, `tests_smoke/`
- Lint config: `pyproject.toml`


