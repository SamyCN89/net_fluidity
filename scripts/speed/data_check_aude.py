#!/usr/bin/env python3

# %%
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# from shared_code.fun_dfcspeed import dfc_speed_split, #dfc_stream2fcd, ts2dfc_stream


# %%
def fast_corrcoef(ts):
    """
    Numba-accelerated Pearson correlation matrix using z-score and dot product.
    ts: np.ndarray (timepoints, features)
    """
    n_samples, n_features = ts.shape
    mean = np.mean(ts, axis=0)
    std = np.std(ts, axis=0, ddof=1)
    # Avoid division by zero for constant columns
    std[std == 0] = 1.0
    z = (ts - mean) / std
    return np.dot(z.T, z) / (n_samples - 1)


def ts2dfc_stream(ts, window_size, lag=None, format_data="2D", method="pearson"):
    """
    Compute dynamic functional connectivity (DFC) stream using a sliding window approach.

    Parameters:
        ts (np.ndarray): Time series data (timepoints x regions).
        window_size (int): Size of the sliding window.
        lag (int): Step size between windows (default = window_size).
        format_data (str): '2D' for vectorized FC, '3D' for FC matrices.
        method (str): Correlation method (currently only 'pearson').

    Returns:
        np.ndarray: DFC stream, either in 2D (n_pairs x frames) or 3D (n_regions x n_regions x frames).
    """
    t_total, n = ts.shape
    lag = lag or window_size
    frames = (t_total - window_size) // lag + 1
    n_pairs = n * (n - 1) // 2

    # Preallocate DFC stream
    dfc_stream = None
    tril_idx = None

    if format_data == "2D":
        dfc_stream = np.empty((n_pairs, frames))
        tril_idx = np.tril_indices(n, k=-1)  # Precompute once
    elif format_data == "3D":
        dfc_stream = np.empty((n, n, frames))
    else:
        raise ValueError(f"Unsupported format_data '{format_data}'. Use '2D' or '3D'")

    for k in range(frames):
        wstart = k * lag
        wstop = wstart + window_size
        window = ts[wstart:wstop, :]
        fc = fast_corrcoef(window)

        if format_data == "2D":
            dfc_stream[:, k] = fc[tril_idx]
        else:
            dfc_stream[:, :, k] = fc

    return dfc_stream


def dfc_stream2fcd(dfc_stream: np.ndarray) -> np.ndarray:
    """Convert a dFC stream into a functional connectivity dynamics (FCD) matrix.

    Parameters
    ----------
    dfc_stream : np.ndarray
        2D (n_pairs, n_frames) or 3D (n_roi, n_roi, n_frames) dFC stream.

    Returns
    -------
    np.ndarray
        Correlation matrix between dFC frames.
    """
    if dfc_stream.ndim not in (2, 3):
        raise ValueError("dfc_stream must be 2D or 3D")

    if dfc_stream.ndim == 3:
        dfc_vec = matrix2vec(dfc_stream)
    else:
        dfc_vec = dfc_stream

    return np.corrcoef(dfc_vec.T)


def matrix2vec(matrix3d: np.ndarray) -> np.ndarray:
    """Vectorize each frame of a 3D matrix into columns.

    Parameters
    ----------
    matrix3d : np.ndarray
        Array shaped (n_roi, n_roi, n_frames).

    Returns
    -------
    np.ndarray
        Array shaped (n_roi * n_roi, n_frames) with column-wise vectorised frames.
    """
    if matrix3d.ndim != 3:
        raise ValueError("matrix3d must be 3D")
    n_roi = matrix3d.shape[0]
    return matrix3d.reshape(n_roi * n_roi, matrix3d.shape[2], order="C")


def dfc_speed_split(
    dfc_stream: np.ndarray,
    *,
    vstep: int = 1,
    tau_range: Sequence[int] | np.ndarray = (0,),
    method: str = "pearson",
    return_fc2: bool = False,
    triu_indices: tuple[np.ndarray, np.ndarray] | None = None,
    time_offset: int = 0,
) -> np.ndarray:
    """Compute dFC speed using window-step offsets and optional tau shifts.

    This variant mirrors the computation used in the exploratory ``data_check`` script:
    it advances comparisons by the sliding window step (`vstep`) and applies additional
    offsets derived from ``time_offset`` and each value in ``tau_range``.

    Parameters
    ----------
    dfc_stream : np.ndarray
        2D (n_pairs, n_frames) vectorized dFC stream or 3D (n_rois, n_rois, n_frames) matrices.
    vstep : int
        Sliding step (in frames/TRs) between consecutive windows. Must match the lag used to
        generate the dFC stream.
    tau_range : sequence of int
        Additional non-negative offsets applied on top of the base window step.
    method : {"pearson","spearman","cosine"}
        Similarity metric for speed computation.
    return_fc2 : bool
        If True, return the FC2 indices used for comparisons instead of speed values.
    triu_indices : tuple of np.ndarray, optional
        Precomputed upper-triangle indices when working with 3D streams.
    time_offset : int
        Additional offset (in frames/TRs) applied before converting to window units.
        Positive values are rounded up to the nearest multiple of ``vstep``.

    Returns
    -------
    np.ndarray
        Array of shape (len(tau_range), T_eff) with speeds per tau. If ``return_fc2`` is True,
        the function returns a 1D array of FC2 indices instead.
    """
    if not isinstance(dfc_stream, np.ndarray):
        raise TypeError("dfc_stream must be a numpy array")
    if dfc_stream.ndim not in (2, 3):
        raise ValueError("dfc_stream must be 2D or 3D")

    vstep_int = int(vstep)
    if vstep_int <= 0:
        raise ValueError("vstep must be a positive integer")

    if method not in ("pearson", "spearman", "cosine"):
        raise ValueError("Unsupported method")

    tau_arr = np.atleast_1d(np.asarray(tau_range, dtype=int))
    if tau_arr.size == 0:
        tau_arr = np.array([0], dtype=int)
    if (tau_arr < 0).any():
        raise ValueError("tau_range values must be non-negative")

    time_offset_int = int(time_offset)
    if time_offset_int < 0:
        raise ValueError("time_offset must be non-negative")

    # Convert optional time offset (frames/TRs) to multiples of vstep
    time_window = (
        int(np.ceil(time_offset_int / vstep_int)) if time_offset_int > 0 else 0
    )

    if dfc_stream.ndim == 3:
        n_rois = dfc_stream.shape[0]
        if triu_indices is None:
            triu_indices = np.triu_indices(n_rois, k=1)
        fc_stream = dfc_stream[triu_indices[0], triu_indices[1], :]
    else:
        fc_stream = dfc_stream

    n_frames = fc_stream.shape[1]
    tau_max = int(tau_arr.max())
    indices_max = n_frames - (vstep_int + tau_max + time_offset_int)
    if indices_max <= 1:
        raise ValueError(
            "Not enough frames for given parameters "
            f"(n_frames={n_frames}, vstep={vstep_int}, tau_max={tau_max}, time_offset={time_offset_int})"
        )

    indices = np.arange(indices_max, dtype=int)
    if indices.size < 2:
        raise ValueError(
            "Not enough indices to compute speed (need at least two frames)."
        )

    head = indices[:-1]
    tail = indices[1:]
    fc1_idx = np.repeat(head[np.newaxis, :], tau_arr.size, axis=0)
    fc2_idx = np.vstack([tail + int(tau) + time_window for tau in tau_arr])

    fc1_matrices = fc_stream[:, fc1_idx.flatten()]
    fc2_matrices = fc_stream[:, fc2_idx.flatten()]

    if return_fc2:
        return fc2_idx.flatten().astype(int)

    if method == "pearson":
        speeds = pearson_speed_vectorized(fc1_matrices, fc2_matrices)
    elif method == "spearman":
        speeds = spearman_speed(fc1_matrices, fc2_matrices)
    else:  # cosine
        speeds = cosine_speed_vectorized(fc1_matrices, fc2_matrices)

    speeds = np.clip(speeds, 0.0, 2.0)
    return speeds.reshape(tau_arr.size, -1)


def pearson_speed_vectorized(fc1, fc2):
    """
    Fully vectorized Pearson correlation speed computation using einsum.
    This is the most optimized version for large datasets.

    Parameters:
        fc1, fc2: 2D arrays (n_pairs, n_frames)

    Returns:
        speeds: 1D array (n_frames,) of speed values
    """
    # Center the data
    fc1_centered = fc1 - np.mean(fc1, axis=0, keepdims=True)
    fc2_centered = fc2 - np.mean(fc2, axis=0, keepdims=True)

    # Compute correlations using einsum (fully vectorized)
    numerator = np.einsum("ij,ij->j", fc1_centered, fc2_centered)
    fc1_ss = np.einsum("ij,ij->j", fc1_centered, fc1_centered)
    fc2_ss = np.einsum("ij,ij->j", fc2_centered, fc2_centered)

    denominator = np.sqrt(fc1_ss * fc2_ss)

    # Handle numerical edge cases
    valid_mask = denominator > np.finfo(float).eps
    correlations = np.zeros(fc1.shape[1])
    correlations[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

    return 1.0 - correlations


# %%
# Load data HERE CHANGE THE PATH TO YOUR DATA
data = loadmat("/home/samy/Bureau/vscode/net_fluidity/scripts/dfc/test.mat")["TS"]

# sample frequencies in Hz
fr = 1 / 0.72  # HERE CHANGE THE SAMPLING FREQUENCY
sf = (fr * np.arange(55, 65)).astype(int)

# %% Compute dFC
speed_pool = []
for s in sf:
    # lag in number of frames
    lag = max(1, int(round(s * 0.005)))
    dfc_stream = ts2dfc_stream(data, s, lag, format_data="2D")
    fcd = dfc_stream2fcd(dfc_stream)
    speed = dfc_speed_split(
        dfc_stream,
        vstep=int(lag),
        tau_range=np.arange(2),
        method="pearson",
        return_fc2=False,
        time_offset=s,
    )

    print(
        s,
        lag,
        dfc_stream.min(),
        dfc_stream.max(),
        speed.min(),
        speed.max(),
        fcd.min(),
        fcd.max(),
    )
    speed_pool.append(speed)

# %%
tril_idx = np.tril_indices(data.shape[1], k=-1)

# from shared_code.fun_optimization import fast_corrcoef

for i, k in enumerate(range(200)):
    wstart = k * lag
    wstop = wstart + s
    window = data[wstart:wstop, :]
    fc = fast_corrcoef(window)
    print(i, wstart, wstop, np.shape(window), np.shape(fc))

    dfc_stream[:, k] = fc[tril_idx]

# %%
# %%
# imshow of results
plt.imshow(dfc_stream, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")

# %%
speed_flat = np.array([x.flatten() for x in speed_pool], dtype=object)
speed_flat = np.concatenate(speed_flat)


# plot hist of speed
plt.hist(speed_flat.ravel(), bins=100, histtype="step", density=True)
# %%
plt.imshow(fcd, aspect="auto", cmap="jet", vmin=0, vmax=1)
plt.colorbar()

# %%
