# Repository Guidelines

## Project Structure & Module Organization
- Code: `shared_code/shared_code/` (primary package, stable APIs). Install with `pip install -e shared_code`.
- Legacy/experiments: `metaconnectivity/`; pipeline scripts: `allegiance/src/`.
- Demos and figures: `julien_data/`.
- Tests: `tests_smoke/` (default pytest path), plus ad‑hoc scripts `test_*.py` in repo root.
- Utilities: `tools/` (linters, import guards), `scripts/` (checks, smoke runner), `docs/`, `reports/`, `config/`.

## Build, Test, and Development Commands
- Create env (Python 3.11) and install: `pip install -e shared_code`.
- Format/lint: `make check` (Black+Ruff), `make format` (apply fixes), or `bash scripts/run_checks.sh check`.
- Tests (smoke by default): `pytest -q` or `pytest -q tests_smoke/`.
- Smoke without pytest: `python scripts/run_smoke.py`.
- Reports: `make report` or `bash scripts/run_checks.sh report` (writes to `reports/`).

## Coding Style & Naming Conventions
- Formatter: Black, 88 chars; Python target `py311`.
- Lint: Ruff with rules E/F/W/I/UP/B; isort via Ruff; see `pyproject.toml`.
- Indentation: 4 spaces; prefer type hints and docstrings for public APIs.
- Naming: `snake_case` for modules/functions, `PascalCase` for classes, `UPPER_SNAKE` for constants.
- Imports: prefer stable APIs in `shared_code.*`. Pre‑commit forbids some `metaconnectivity` imports. Example:
  - Good: `from shared_code.fun_dfcspeed import ts2dfc_stream`
  - Avoid: `from metaconnectivity.fun_dfcspeed import ts2dfc_stream`

## Testing Guidelines
- Framework: `pytest` (see `pytest.ini` → `tests_smoke/`).
- Naming: files `test_*.py`, functions `test_*`.
- Quick run: `pytest -q tests_smoke/test_dfc_speed_smoke.py`.
- Integration/examples: `python test_unified_dfc_speed.py`, `python test_dfc_speed_integration.py`.

## Commit & Pull Request Guidelines
- Commits: imperative, present tense; prefer Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`). Example: `feat(shared_code): add cosine option to dfc_speed`.
- PRs: clear description, linked issues, before/after plots or timings when algorithms change, note new env vars. Include test evidence and ensure `make check` passes.
- Install hooks: `pre-commit install`; run locally with `pre-commit run -a`.

## Security & Configuration Tips
- Do not commit data; use paths from `shared_code/shared_code/fun_paths.py`. Set `PROJECT_ROOT_LOCAL` (and optional `DATASET_NAME`) in a `.env` or shell.
- Optional logging config: copy `config/logging.example.yaml` and adjust locally.

