.PHONY: check format lint report help-pipeline prep dfc allegiance-jobs allegiance-merge cohesion-compute cohesion-stats cohesion-report

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

# ---------------- Pipeline helpers (DFX/Allegiance/Cohesion) ----------------

# Defaults (override via `make VAR=value`)
PY ?= python
WS ?= 9
LAG ?= 1
TAU ?= 3
DMN ?=
N_JOBS ?= 8
STATS_MODE ?= all           # age|group|all
GROUP ?= both               # sex|genotype|both
ALPHA ?= 0.05
CROSS_AGE ?= 0              # 1 to enable
POOL_AGES ?= 0              # 1 to enable

help-pipeline:
	@echo "make prep                # preprocess + grouping"
	@echo "make dfc WS=$(WS) LAG=$(LAG) TAU=$(TAU)"
	@echo "make allegiance-jobs WS=$(WS) LAG=$(LAG) N_JOBS=$(N_JOBS)"
	@echo "make allegiance-merge     # merge allegiance outputs"
	@echo "make cohesion-compute WS=$(WS) LAG=$(LAG) TAU=$(TAU) DMN=\"$(DMN)\""
	@echo "make cohesion-stats WS=$(WS) LAG=$(LAG) TAU=$(TAU) STATS_MODE=$(STATS_MODE) GROUP=$(GROUP)"
	@echo "make cohesion-report WS=$(WS) LAG=$(LAG) TAU=$(TAU)"

prep:
	$(PY) allegiance/src/prep_cog_groups.py

dfc:
	$(PY) allegiance/src/dfc_compute.py --format 3D --wmin $(WS) --wmax $(WS) --wstep 1 --lag $(LAG) --tau $(TAU)

allegiance-jobs:
	$(PY) allegiance/src/allegiance_jobs.py --n_jobs $(N_JOBS) --window_size $(WS) --lag $(LAG)

allegiance-merge:
	$(PY) allegiance/src/allegiance_merge.py

cohesion-compute:
	$(PY) allegiance/src/cohesion_compute.py --window-size $(WS) --lag $(LAG) --tau $(TAU) --dmn-index "$(DMN)" --min-duration 1

cohesion-stats:
	$(PY) allegiance/src/cohesion_stats_plot.py \
	  --window-size $(WS) --lag $(LAG) --tau $(TAU) --dmn-index "$(DMN)" \
	  --with-stats --stats-mode $(STATS_MODE) --group-compare $(GROUP) \
	  $(if $(filter 1,$(CROSS_AGE)),--cross-age,) \
	  $(if $(filter 1,$(POOL_AGES)),--pool-ages,) \
	  --alpha $(ALPHA) --save-plots --no-show

cohesion-report:
	$(PY) allegiance/src/cohesion_report.py --window-size $(WS) --lag $(LAG) --tau $(TAU) --save-plots --no-show
