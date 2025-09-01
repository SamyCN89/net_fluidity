# Duplicate-Module Census (shared_code vs metaconnectivity)

Scope: `shared_code/shared_code/fun_*.py` vs `metaconnectivity/fun_*.py`.
Recommendation: use `shared_code` as the single source of truth and deprecate duplicates in `metaconnectivity`.

## Overlap Map
- fun_bootstrap.py: Only in shared_code
- fun_dfcspeed.py: Differs
  - Shared functions (6): compute_plv_matrix_vectorized, dfc_speed, dfc_speed_oversampled_series, parallel_dfc_speed_oversampled_series, ts2dfc_stream, ts2fc
  - Only in shared_code (7): _handle_dfc_speed_analysis, check_and_rerun_missing_files, compute4window, get_population_wpooling, get_tenet4window_range, handler_get_tenet, pool_vel_windows
  - Only in metaconnectivity (5): dfc_stream2fcd, matrix2vec, sort_modularity, ts2dfc_stream_old, window_pooling_speed
- fun_loaddata.py: Differs
  - Only in shared_code (13): extract_hash_numbers, extract_mouse_ids, filename_sort_mat, get_missing_files, load_fc2_npz, load_from_cache, load_mat_timeseries, load_matdata, load_npz_dict, load_pickle, make_file_path, save2disk, save_pickle
- fun_metaconnectivity.py: Differs
  - Shared functions (18): _build_agreement_matrix, _run_louvain, allegiance_matrix_analysis, allegiance_wrapper_, build_agreement_matrix_vectorized, build_trimer_mask, compute_mc_nplets_mask_and_index, compute_metaconnectivity, compute_trimers_identity, contingency_matrix_fun, fun_allegiance_communities, fun_allegiance_communities2, get_fc_mc_indices, get_mc_region_identities, intramodule_indices_mask, trimers_by_apex, trimers_leaves_fc, trimers_root_fc
  - Only in shared_code (3): animal_mc, compute_metaconnectivity_old, load_merged_allegiance
- fun_network.py: Only in shared_code
- fun_optimization.py: Differs
  - Shared functions (3): fast_corrcoef, fast_corrcoef_numba, fast_corrcoef_numba_parallel
  - Only in shared_code (5): cosine_speed, cosine_speed_vectorized, pearson_speed, pearson_speed_vectorized, spearman_speed
- fun_paths.py: Only in shared_code
- fun_utils.py: Differs
  - Shared functions (10): classify_phenotypes, filename_sort_mat, load_cognitive_data, load_grouping_data, load_matdata, load_timeseries_data, make_combination_masks, make_masks, set_figure_params, split_groups_by_age
  - Only in shared_code (5): check_symmetric, dfc_stream2fcd, load_timeseries, matrix2vec, validate_alignment
  - Only in metaconnectivity (2): get_paths, get_root_path

## Single Source-of-Truth Plan
- Keep `shared_code` modules canonical and import from them in analyses.
- For any divergent research scripts in `metaconnectivity`, move to a clearly named `experimental/` or `deprecated/` area or refactor to call `shared_code`.
- Add thin wrappers or deprecation warnings where needed, then remove duplicates after migration.