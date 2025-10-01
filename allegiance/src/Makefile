.DEFAULT_GOAL := help

.PHONY: help pipeline prep dfc allegiance-jobs allegiance-merge cohesion-compute cohesion-stats cohesion-report

# Configurable variables (override: make target VAR=value)
PY ?= python
RUN ?=
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

help:
	@echo "Targets:"
	@echo "  make pipeline                 # prep -> dfc -> jobs -> merge -> cohesion-compute -> cohesion-stats"
	@echo "  make prep                     # preprocessing + grouping"
	@echo "  make dfc WS=$(WS) LAG=$(LAG) TAU=$(TAU)"
	@echo "  make allegiance-jobs WS=$(WS) LAG=$(LAG) N_JOBS=$(N_JOBS)"
	@echo "  make allegiance-merge         # merge allegiance outputs"
	@echo "  make cohesion-compute WS=$(WS) LAG=$(LAG) TAU=$(TAU) DMN=\"$(DMN)\""
	@echo "  make cohesion-stats WS=$(WS) LAG=$(LAG) TAU=$(TAU) STATS_MODE=$(STATS_MODE) GROUP=$(GROUP) [CROSS_AGE=1] [POOL_AGES=1]"
	@echo "  make cohesion-report WS=$(WS) LAG=$(LAG) TAU=$(TAU)"
	@echo "Variables: PY=$(PY) RUN=$(RUN) WS=$(WS) LAG=$(LAG) TAU=$(TAU) DMN=\"$(DMN)\" N_JOBS=$(N_JOBS)"
	@echo "Note: Set RUN to 'srun -n 1' or similar on HPC if desired."

# Full pipeline
pipeline: prep dfc allegiance-jobs allegiance-merge cohesion-compute cohesion-stats

prep:
	$(RUN) $(PY) prep_cog_groups.py

dfc:
	$(RUN) $(PY) dfc_compute.py --format 3D --wmin $(WS) --wmax $(WS) --wstep 1 --lag $(LAG) --tau $(TAU)

allegiance-jobs:
	$(RUN) $(PY) allegiance_jobs.py --n_jobs $(N_JOBS) --window_size $(WS) --lag $(LAG)

allegiance-merge:
	$(RUN) $(PY) allegiance_merge.py

cohesion-compute:
	$(RUN) $(PY) cohesion_compute.py --window-size $(WS) --lag $(LAG) --tau $(TAU) --dmn-index "$(DMN)" --min-duration 1

cohesion-stats:
	$(RUN) $(PY) cohesion_stats_plot.py \
	  --window-size $(WS) --lag $(LAG) --tau $(TAU) --dmn-index "$(DMN)" \
	  --with-stats --stats-mode $(STATS_MODE) --group-compare $(GROUP) \
	  $(if $(filter 1,$(CROSS_AGE)),--cross-age,) \
	  $(if $(filter 1,$(POOL_AGES)),--pool-ages,) \
	  --alpha $(ALPHA) --save-plots --no-show

cohesion-report:
	$(RUN) $(PY) cohesion_report.py --window-size $(WS) --lag $(LAG) --tau $(TAU) --save-plots --no-show

