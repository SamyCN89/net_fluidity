import importlib.util
from pathlib import Path

import numpy as np

try:
    from shared_code.fun_dfcspeed import dfc_speed_multi_tau
except ModuleNotFoundError:  # fallback to local package path
    import sys as _sys
    _root = Path(__file__).resolve().parents[1]
    _pkg_root = _root / "shared_code"
    if str(_pkg_root) not in _sys.path:
        _sys.path.insert(0, str(_pkg_root))
    from shared_code.fun_dfcspeed import dfc_speed_multi_tau  # type: ignore


def _load_legacy_v6():
    here = Path(__file__).resolve().parents[1]  # repo root
    v6 = here / "julien_data" / "3_dfc_speed_test_v6.py"
    # Ensure julien_data dir is importable for fallback imports inside the script
    import sys as _sys
    if str(v6.parent) not in _sys.path:
        _sys.path.insert(0, str(v6.parent))
    spec = importlib.util.spec_from_file_location("julien_speed_v6", v6)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to import legacy v6 speed script")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rand_dfc_stream(seed: int, T: int, N: int):
    rng = np.random.default_rng(seed)
    # Build a random time series and form a simplistic dFC stream for test purposes
    # Here we directly create a pair-stream (2D) with shape (P, T)
    P = N * (N - 1) // 2
    return rng.standard_normal((P, T)).astype(np.float32)


def test_multi_tau_parity_2d_small():
    mod = _load_legacy_v6()
    dfc2d = _rand_dfc_stream(seed=0, T=60, N=6)
    vstep = 3
    tau_range = np.array([0, 1, 2], dtype=int)

    ref = mod.dfc_speed_split(dfc2d, vstep=vstep, tau_range=tau_range, method="pearson")
    new = dfc_speed_multi_tau(dfc2d, vstep=vstep, tau_range=tau_range, method="pearson")

    assert ref.shape == new.shape
    assert np.allclose(ref, new, rtol=1e-6, atol=1e-8)


def test_multi_tau_parity_3d_small():
    mod = _load_legacy_v6()
    rng = np.random.default_rng(1)
    N, T = 6, 50
    dfc3d = rng.standard_normal((N, N, T)).astype(np.float32)
    # make symmetric with zeros on diag, as typical FC frames
    for t in range(T):
        A = dfc3d[:, :, t]
        A = 0.5 * (A + A.T)
        np.fill_diagonal(A, 0.0)
        dfc3d[:, :, t] = A

    vstep = 2
    tau_range = np.array([0, 2], dtype=int)

    ref = mod.dfc_speed_split(dfc3d, vstep=vstep, tau_range=tau_range, method="pearson")
    new = dfc_speed_multi_tau(dfc3d, vstep=vstep, tau_range=tau_range, method="pearson")

    assert ref.shape == new.shape
    assert np.allclose(ref, new, rtol=1e-6, atol=1e-8)
