.PHONY: help preprocess dfc-all dfc-split speed-orig speed-wrap speed-shared ab-speed compare plot community-plot

# Defaults: TR=500 (example override: make TR=400 SELECTED="1,4,9" SUBSET_A=orig SUBSET_B=wrap)
PY ?= python
TR ?= 500
SELECTED ?=
SUBSET_A ?= orig
SUBSET_B ?= wrap
WIN ?=
PLOT_SUBSET ?= $(SUBSET_B)
TAU ?=
POOL ?= all

help:
	@echo "make preprocess       # preprocess only TR=$(TR)"
	@echo "make dfc-all          # DFC streams (pad all) for TR=$(TR)"
	@echo "make dfc-split        # DFC streams per-length split for TR=$(TR)"
	@echo "make speed-orig       # speed with original script (subset=$(SUBSET_A))"
	@echo "make speed-wrap       # speed with wrapper (subset=$(SUBSET_B))"
	@echo "make speed-shared     # speed with shared engine (subset=$(SUBSET_B))"
	@echo "make ab-speed         # run orig + shared and compare (WIN=$(WIN) optional)"
	@echo "make compare          # compare speed outputs (WIN=$(WIN) optional)"
	@echo "make plot             # plot merged outputs (subset=$(PLOT_SUBSET), TR=$(TR), TAU=$(TAU) optional)"
	@echo "make community-plot   # plot community-wise speeds (subset=$(PLOT_SUBSET), TR=$(TR), POOL=$(POOL))"

preprocess:
	$(PY) src/preprocess.py --only-tr $(TR)

dfc-all:
	$(PY) src/dfc_stream_cli.py --mode all --processors -1 --load-cache --tr $(TR)

dfc-split:
	$(PY) src/dfc_stream_cli.py --mode split --processors -1 --load-cache --tr $(TR)

speed-orig:
	$(PY) 3_dfc_speed_test_v6.py --tr $(TR) $(if $(SELECTED),--selected-regions "$(SELECTED)",) --region-mode touching --subset-name $(SUBSET_A)

speed-wrap:
	$(PY) src/speed_compute.py --tr $(TR) $(if $(SELECTED),--selected-regions "$(SELECTED)",) --region-mode touching --subset-name $(SUBSET_B)

speed-shared:
	$(PY) src/speed_compute.py --tr $(TR) $(if $(SELECTED),--selected-regions "$(SELECTED)",) --engine shared --region-mode touching --subset-name $(SUBSET_B)

compare:
	$(PY) ../scripts/compare_speed_outputs.py --subset-a $(SUBSET_A) --subset-b $(SUBSET_B) $(if $(WIN),--window-size $(WIN),)

ab-speed:
	$(MAKE) speed-orig TR=$(TR) SELECTED="$(SELECTED)" SUBSET_A=$(SUBSET_A)
	$(MAKE) speed-shared TR=$(TR) SELECTED="$(SELECTED)" SUBSET_B=$(SUBSET_B)
	$(MAKE) compare SUBSET_A=$(SUBSET_A) SUBSET_B=$(SUBSET_B) WIN=$(WIN)

plot:
	$(PY) src/speed_plots_cli.py --tr $(TR) --subset-name $(PLOT_SUBSET) $(if $(TAU),--tau $(TAU),) --savefig

community-plot:
	$(PY) src/community_speed_plot.py --tr $(TR) --subset-name $(PLOT_SUBSET) --pool $(POOL) --savefig
