import numpy as np

from shared_code.shared_code.fun_metaconnectivity import (
    compute_metaconnectivity,
    fun_allegiance_communities,
)


def test_meta_connectivity_and_allegiance_tiny():
    # Tiny dataset: 1 animal, few regions, short time series
    A, N, T = 1, 6, 50
    rng = np.random.default_rng(42)
    # Expected ts layout for ts2dfc_stream is (timepoints, regions);
    # compute_metaconnectivity expects ts_data as (animals, timepoints, regions)
    ts = rng.standard_normal((A, T, N)).astype(np.float32)

    # Compute meta-connectivity quickly (small window, lag)
    mc = compute_metaconnectivity(ts, window_size=10, lag=5, save_path=None, n_jobs=1)
    assert mc.shape == (A, N * (N - 1) // 2, N * (N - 1) // 2)

    # Use a single matrix for allegiance analysis; keep settings very small
    communities, sort_idx, contingency = fun_allegiance_communities(
        mc_data=mc[0], n_runs=2, gamma_pt=2, ref_name="smoke", save_path=None, n_jobs=1
    )

    assert communities.size == (N * (N - 1)) // 2
    assert sort_idx.size == (N * (N - 1)) // 2
    assert contingency.shape == (communities.size, communities.size)
