Session Summary — 2025-09-30

- Added AGENTS.md contributor guide tailored to this repo.
- Created allegiance/src/USAGE_TUTORIAL.md with pipeline, commands, and notes.
- Implemented allegiance/src/coherence_analysis_clean.py (clean analysis entry point):
  - Wrapped in CLI with logging; headless plotting support.
  - Reordered communities, module-count summaries, and core plots.
  - Cohesion metrics, event extraction, and burstiness heatmap.
  - Age-paired stats (Wilcoxon, t-test) with mean-diff and cohesion-diff ratio effects.
  - Group-based stats via Mann–Whitney U (Sex/Genotype), within-age, cross-age (optional).
  - Pooled-age comparisons (Female vs Male, dKI vs wt ignoring age).
  - Phenotype toggle: `--include-phenotype {none,oip,nor,both}` for age-paired stats.
  - Fixed weighted heatmap color scale to [-0.1, 0.1].
- Removed OiP from default factor spec; controlled via CLI include-phenotype.
- Updated tutorial with comparison recipes, new flags, and rationale for Mann–Whitney.

Session Summary — 2025-10-01

- Allegiance folder restructuring and new pipeline components:
  - Added cohesion_compute.py (summaries + events Parquet) with:
    - time_ratio, mean/std duration, burstiness NPZ; Parquet/CSV events
    - events-per-animal preview + counts CSV; binary ATL plotting (--plot-animal/--save-all-binary)
    - scan-based event detection (validated via toy reference)
  - Added cohesion_stats_plot.py (stats + heatmaps) with:
    - age-paired (Wilcoxon/t-test) and group-based (Mann–Whitney) modes
    - effects: mean difference and cohesion-diff ratio; fixed (1−p)×effect color scale [-0.1, 0.1]
    - pooled-ages and cross-age comparisons; phenotype toggle for age mode
  - Added notebook helpers: cohesion_playground.py (new plots incl. burstiness vs std/mean) and events_playground.py (toy dataset + quick_demo).
  - Moved legacy scripts to allegiance/src/legacy/ (kept burst_detection_PBM.py active).
  - Renamed scripts:
    - 1_preprocessed_data_ts_cog_groups.py → prep_cog_groups.py
    - 2_compute_dfc_local.py → dfc_compute.py
    - run_all_allegiance_local.py → allegiance_jobs.py
    - merge_allegiance_parallel.py → allegiance_merge.py
    - coherence_analysis_clean.py → cohesion_report.py
  - Figures now saved under paths["f_cohesion"]/per_animal and paths["f_cohesion"]/stats; CSVs remain under results.
- Makefile additions:
  - Root Makefile: pipeline helpers; dedicated allegiance/src/Makefile with default pipeline target.
  - HPC-friendly RUN wrapper (e.g., RUN="srun -n 1"); added make_tutorial.md and linked from USAGE_TUTORIAL.md.
- Updated USAGE_TUTORIAL.md with new script names, sections for compute/stats, notebook playground, and Makefile link.
