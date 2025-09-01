# Duplicate-Module Census

This report compares fun_*.py modules between shared_code and metaconnectivity.

Recommendation: keep shared_code as canonical; refactor or deprecate duplicates in metaconnectivity.


## Overlap Map

- Bold entries: identical content (SHA-1 match)
- Differs: same basename, different contents
- Orphan(A/B): exists only in one side

- fun_bootstrap.py: Orphan(A) in shared_code
- fun_dfcspeed.py: Differs
  - Shared functions: compute_plv_matrix_vectorized, dfc_speed, dfc_speed_oversampled_series, parallel_dfc_speed_oversampled_series, ts2dfc_stream, ts2fc
  - Only in shared_code: _handle_dfc_speed_analysis, check_and_rerun_missing_files, compute4window, get_population_wpooling, get_tenet4window_range, handler_get_tenet, pool_vel_windows
  - Only in metaconnectivity: dfc_stream2fcd, matrix2vec, sort_modularity, ts2dfc_stream_old, window_pooling_speed
- fun_loaddata.py: Differs
  - Only in shared_code: extract_hash_numbers, extract_mouse_ids, filename_sort_mat, get_missing_files, load_fc2_npz, load_from_cache, load_mat_timeseries, load_matdata, load_npz_dict, load_pickle, make_file_path, save2disk, save_pickle
- fun_metaconnectivity.py: Differs
  - Shared functions: _build_agreement_matrix, _run_louvain, allegiance_matrix_analysis, allegiance_wrapper_, build_agreement_matrix_vectorized, build_trimer_mask, compute_mc_nplets_mask_and_index, compute_metaconnectivity, compute_trimers_identity, contingency_matrix_fun, fun_allegiance_communities, fun_allegiance_communities2, get_fc_mc_indices, get_mc_region_identities, intramodule_indices_mask, trimers_by_apex, trimers_leaves_fc, trimers_root_fc
  - Only in shared_code: animal_mc, compute_metaconnectivity_old, load_merged_allegiance
- fun_network.py: Orphan(A) in shared_code
- fun_optimization.py: Differs
  - Shared functions: fast_corrcoef, fast_corrcoef_numba, fast_corrcoef_numba_parallel
  - Only in shared_code: cosine_speed, cosine_speed_vectorized, pearson_speed, pearson_speed_vectorized, spearman_speed
- fun_paths.py: Orphan(A) in shared_code
- fun_utils.py: Differs
  - Shared functions: classify_phenotypes, filename_sort_mat, load_cognitive_data, load_grouping_data, load_matdata, load_timeseries_data, make_combination_masks, make_masks, set_figure_params, split_groups_by_age
  - Only in shared_code: check_symmetric, dfc_stream2fcd, load_timeseries, matrix2vec, validate_alignment
  - Only in metaconnectivity: get_paths, get_root_path

## Recommendation
- Keep `shared_code` versions authoritative.
- If metaconnectivity copies diverge, either import from `shared_code` or move specialized research-only variants into a `deprecated/` or `experimental/` submodule with clear docstrings.
- Add unit tests against `shared_code` APIs and deprecate duplicate entry points.
