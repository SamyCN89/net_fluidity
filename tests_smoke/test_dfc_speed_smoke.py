import numpy as np

from shared_code.shared_code.fun_dfcspeed import ts2dfc_stream, dfc_speed


def test_dfc_speed_synthetic_small():
    # Small synthetic time series: T x N
    T, N = 60, 6
    rng = np.random.default_rng(0)
    ts = rng.standard_normal((T, N))

    # Build a short dFC stream and compute speed
    dfc2d = ts2dfc_stream(ts, window_size=20, lag=5, format_data="2D")
    med, speeds = dfc_speed(dfc2d, vstep=1, method="pearson")

    assert np.isfinite(med)
    assert speeds.ndim == 1 and speeds.size > 0

