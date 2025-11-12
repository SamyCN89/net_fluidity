.PHONY: check format lint report help-pipeline prep dfc allegiance-jobs allegiance-merge cohesion-compute cohesion-stats cohesion-report \
        help-speed speed-compute speed-plot speed-pooltest speed-cor speed-doctor test-smoke

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

# ---------------- dFC Speed Bootstrap (compute/plot) ----------------

# Defaults (override on the command line)
TR ?= 500
SUBSET ?= regions500
TAU_INDEX ?= 0
N_BOOT ?= 2000
JOBS ?= 8
REGION_JOBS ?= 1
OUTDIR ?=
PAIR_SCOPE ?= windows
Q ?= 1,5,50,95,99

help-speed:
	@echo "make speed-compute TR=$(TR) SUBSET=$(SUBSET) TAU_INDEX=$(TAU_INDEX) N_BOOT=$(N_BOOT) JOBS=$(JOBS) REGION_JOBS=$(REGION_JOBS)"
	@echo "make speed-plot TR=$(TR) SUBSET=$(SUBSET)"
	@echo "make speed-pooltest TR=$(TR) SUBSET=$(SUBSET)  # plot pool-tests if present"
	@echo "make speed-cor TR=$(TR) SUBSET=$(SUBSET)       # plot correlations if present"
	@echo "make speed-doctor                             # show/check/create paths; context checks"
	@echo "make test-smoke                                # run smoke tests if available"

speed-compute:
	$(PY) scripts/compute_speed_bootstrap.py \
	  --tr $(TR) --subset $(SUBSET) --tau-index $(TAU_INDEX) \
	  --n-boot $(N_BOOT) --q $(Q) \
	  --pool-threshold median --pool-all \
	  --jobs $(JOBS) --region-jobs $(REGION_JOBS) --parallel-scope $(PAIR_SCOPE) \
	  $(if $(OUTDIR),--outdir $(OUTDIR),) \
	  --progress

speed-plot:
	$(PY) scripts/plot_speed_bootstrap.py \
	  --tr $(TR) --subset $(SUBSET) \
	  --plot-diffs-by-win --plot-diffs-bywin-grid --bywin-grid-cols 2

speed-pooltest:
	$(PY) scripts/plot_speed_pooltest.py \
	  --tr $(TR) --subset $(SUBSET) --bywin --pooled --progress

speed-cor:
	$(PY) scripts/plot_speed_correlations.py \
	  --tr $(TR) --subset $(SUBSET) --plot-by-win --plot-pooled --progress

speed-doctor:
	$(PY) scripts/paths_doctor.py --show --check-write --create --check-context

test-smoke:
	pytest -q || true
