.PHONY: check format lint report

CHECK_DIRS = shared_code/shared_code metaconnectivity allegiance/src julien_data

check:
	black --check $(CHECK_DIRS)
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check $(CHECK_DIRS); \
	else \
		echo "[skip] Ruff not found. Install via conda/pip."; \
	fi

format:
	black $(CHECK_DIRS)
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check --fix $(CHECK_DIRS) || true; \
	else \
		echo "[skip] Ruff not found. Install via conda/pip."; \
	fi

lint:
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check $(CHECK_DIRS); \
	else \
		echo "[skip] Ruff not found. Install via conda/pip."; \
	fi

report:
	mkdir -p reports
	black --check $(CHECK_DIRS) > reports/style_check.md 2>&1 || true
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check $(CHECK_DIRS) > reports/lint_ruff.md 2>&1 || true; \
	else \
		echo "Ruff not found. Install via conda/pip and re-run." > reports/lint_ruff.md; \
	fi

