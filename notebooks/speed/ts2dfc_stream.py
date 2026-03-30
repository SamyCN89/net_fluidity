
#%%
from __future__ import annotations

import importlib
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path so both packages are importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path("__file__").resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

print("REPO_ROOT:", REPO_ROOT)
print("Python:", sys.version)
#%%
# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
T          = 300    # timepoints
N          = 10     # regions
WINDOW     = 30     # sliding window size (frames)
LAG        = 1      # lag between windows (frames)
VSTEP      = 1      # speed step
SEED       = 42
THRESHOLD  = 1e-6   # discrepancy flag threshold

rng = np.random.default_rng(SEED)

n_pairs  = N * (N - 1) // 2
n_frames = (T - WINDOW) // LAG + 1
# A / C produce n_frames - VSTEP - 1 speeds; B produces n_frames - VSTEP
n_speeds_AC = n_frames - VSTEP - 1
n_speeds_B  = n_frames - VSTEP

print(f"T={T}, N={N}, window={WINDOW}, lag={LAG}, vstep={VSTEP}")
print(f"n_frames = {n_frames}, n_pairs = {n_pairs}")
print(f"Expected speeds: A/C = {n_speeds_AC}, B = {n_speeds_B}")

# %%
# Synthetic AR(1) time series — mild autocorrelation makes speed non-trivial
phi = 0.7
ts  = np.zeros((T, N))
ts[0] = rng.standard_normal(N)
for t in range(1, T):
    ts[t] = phi * ts[t - 1] + np.sqrt(1 - phi**2) * rng.standard_normal(N)

print("ts shape:", ts.shape, "  mean:", ts.mean().round(4), "  std:", ts.std().round(4))
#%%
# Build ONE canonical 2D dfc_stream (lower triangle, shape n_pairs × n_frames)
# All three implementations will receive this identical array as input.
# from shared_code.fun_dfcspeed import ts2dfc_stream

###############################################################################################################
# Metaconnectivity folder's original implementations (for comparison)
###############################################################################################################
from numba import njit, prange
import numpy as np

# @njit(fastmath=True)
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



def ts2fc(timeseries, format_data = '2D', method='pearson'):
    """
    Calculate functional connectivity from time series data.

    Parameters:
    timeseries (array): Time series data of shape (timepoints, nodes).
    format_data (str): Output format, '2D' for full matrix or '1D' for lower-triangular vector.

    Returns:
    fc (array): Functional connectivity matrix ('2D') or vector ('1D').

    Adapted from Lucas Arbabyazd et al 2020. Methods X, doi: 10.1016/j.neuroimage.2020.117156
    """
    # Calculate correlation coefficient matrix
    if method=='pearson':
        fc = fast_corrcoef(timeseries)
        # fc = fast_corrcoef2(timeseries)
        # fc = fast_corrcoef_numba(timeseries)

        # fc = np.corrcoef(timeseries.T)
    elif method=='plv':
        fc = compute_plv_matrix_vectorized(timeseries.T)

    # Optionally zero out the diagonal for '2D' format
    if format_data=='2D':
        np.fill_diagonal(fc,0)#fill the diagonal with 0
        return fc
    elif format_data=='1D':
        # Return the lower-triangular part excluding the diagonal
        return fc[np.tril_indices_from(fc, k=-1)]
#%%
def ts2dfc_stream_metaconnectivityfolder(ts, window_size, lag=None, format_data='2D', method='pearson'):
    """
    Calculate dynamic functional connectivity stream (dfc_stream) from time series.

    Parameters:
    - ts: np.ndarray (timepoints x regions)
    - window_size: int
    - lag: int (defaults to window_size)
    - format_data: '2D' for vectorized, '3D' for matrices

    Returns:
    - dfc_stream: np.ndarray
    """
    t_total, n = ts.shape
    lag = lag or window_size
    frames = (t_total - window_size) // lag + 1
    n_pairs = n * (n - 1) // 2

    if format_data == '2D':
        dfc_stream = np.empty((n_pairs, frames))
        tril_idx = np.tril_indices(n, k=-1)  # Precompute once
    elif format_data == '3D':
        dfc_stream = np.empty((n, n, frames))

    for k in range(frames):
        wstart = k * lag
        wstop = wstart + window_size
        window = ts[wstart:wstop, :]
        fc = fast_corrcoef(window)

        if format_data == '2D':
            dfc_stream[:, k] = fc[tril_idx]
        else:
            dfc_stream[:, :, k] = fc

    return dfc_stream

def ts2dfc_stream_old_metaconnectivityfolder(ts, windows_size, lag=None, format_data='2D', method='pearson'):
    """
    Calculate dynamic functional connectivity stream (dfc_stream) from time series data.

    Parameters:
    ts (array): Time series data of shape (t, n), where t is timepoints, n is regions.
    windows_size (int): Window size to slide over the ts.
    lag (int): Shift value for the window. Defaults to W if not specified.
    format (str): Output format. '2D' for a (l, F) shape, '3D' for a (n, n, F) shape.

    Returns:
    dFCstream (array): Dynamic functional connectivity stream.
    """

    t_total, n = np.shape(ts)
    #Not overlap
    lag = lag or windows_size
    # if lag is None:
    #     lag = windows_size
    # Calculate the number of frames/windows
    frames = (t_total - windows_size)//lag + 1
    n_pairs               = n * (n-1)//2 #number of pairwise correlations

    if format_data=='2D':
        dfc_stream = np.empty((n_pairs, frames))
    elif format_data=='3D':
        dfc_stream = np.empty((n, n, frames))


    for k in range(frames):
        wstart = k * lag
        wstop = wstart + windows_size
        if format_data =='2D':
            dfc_stream[:, k]    = ts2fc(ts[wstart:wstop, :], '1D', method=method)  # Assuming TS2FC returns a vector
        elif format_data == '3D':
            dfc_stream[:, :, k] = ts2fc(ts[wstart:wstop, :], '2D',method=method)  # Assuming TS2FC returns a matrix
    #         dfc_stream[:, :, k] = fc
    return dfc_stream

#%%
###################################################################################################################
# shared code implementations (for comparison)
###################################################################################################################

def ts2dfc_stream_shared(ts, window_size, lag=None, format_data="2D", method="pearson"):
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




#%%


dfc_stream_2d_metaconnectivity = ts2dfc_stream_metaconnectivityfolder(ts, window_size=WINDOW, lag=LAG, format_data="2D")
print("dfc_stream_2d_metaconnectivity shape:", dfc_stream_2d_metaconnectivity.shape,
      "  min:", dfc_stream_2d_metaconnectivity.min().round(4),
      "  max:", dfc_stream_2d_metaconnectivity.max().round(4))


dfc_stream_2d_metaconnectivity_old = ts2dfc_stream_old_metaconnectivityfolder(ts, windows_size=WINDOW, lag=LAG, format_data="2D")
print("dfc_stream_2d_metaconnectivity_old shape:", dfc_stream_2d_metaconnectivity_old.shape,
      "  min:", dfc_stream_2d_metaconnectivity_old.min().round(4),
      "  max:", dfc_stream_2d_metaconnectivity_old.max().round(4))

dfc_stream_2d_shared = ts2dfc_stream_shared(ts, window_size=WINDOW, lag=LAG, format_data="2D")
print("dfc_stream_2d_shared shape:", dfc_stream_2d_shared.shape,
      "  min:", dfc_stream_2d_shared.min().round(4),
      "  max:", dfc_stream_2d_shared.max().round(4))
#%%
np.allclose(dfc_stream_2d_metaconnectivity, dfc_stream_2d_metaconnectivity_old, atol=THRESHOLD)
np.allclose(dfc_stream_2d_metaconnectivity, dfc_stream_2d_shared, atol=THRESHOLD)
# np.testing.assert_allclose(dfc_stream_2d_metaconnectivity, dfc_stream_2d_metaconnectivity_old, atol=THRESHOLD)
# print("✅ Metaconnectivity folder's new and old implementations produce the same results within the threshold")

#%%
#plot the dfc_stream_2d to visualize the data
plt.figure(figsize=(10, 6))
plt.imshow(dfc_stream_2d_metaconnectivity, aspect='auto', interpolation='none', cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(label=r'CC(BOLD$_{i, i\neq j}$, BOLD$_{j, j\neq i}$)')
plt.title('dFC Stream ')
plt.xlabel('Time Windows')
plt.ylabel('Region Pairs')
plt.show()
# %%

