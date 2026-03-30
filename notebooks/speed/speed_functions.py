# %%
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sys
import time
from tkinter import W
from tracemalloc import start

import matplotlib.pyplot as plt
import numpy as np
import math

from scripts.mc.FP7c_fdr_by_contrast import ALPHA

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path so both packages are importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path("__file__").resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

print("REPO_ROOT:", REPO_ROOT)
print("Python:", sys.version)
# %%
# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
T = 500  # timepoints
N = 10  # regions
WINDOW = 25  # sliding window size (frames)
# LAG        = WINDOW      # lag between windows (frames)
LAG = 1  # lag between windows (frames)
VSTEP = 1  # speed step
SEED = 42
THRESHOLD = 1e-6  # discrepancy flag threshold

TAU_RANGE = (-2,-1,0,1,2)

rng = np.random.default_rng(SEED)

n_pairs = N * (N - 1) // 2
n_frames = (T - WINDOW) // LAG + 1
# A / C produce n_frames - VSTEP - 1 speeds; B produces n_frames - VSTEP
n_speeds_AC = n_frames - VSTEP - 1
n_speeds_B = n_frames - VSTEP

print(f"T={T}, N={N}, window={WINDOW}, lag={LAG}, vstep={VSTEP}")
print(f"n_frames = {n_frames}, n_pairs = {n_pairs}")
print(f"Expected speeds: A/C = {n_speeds_AC}, B = {n_speeds_B}")

# %%
# Synthetic AR(1) time series — mild autocorrelation makes speed non-trivial
phi = 0.9  #  strong memory, slow drift, speed is low and smooth
ts = np.zeros((T, N))
ts[0] = rng.standard_normal(N)
for t in range(1, T):
    ts[t] = phi * ts[t - 1] + np.sqrt(1 - phi**2) * rng.standard_normal(N)

print("ts shape:", ts.shape, "  mean:", ts.mean().round(4), "  std:", ts.std().round(4))
# %%
# Build ONE canonical 2D dfc_stream (lower triangle, shape n_pairs × n_frames)
# All three implementations will receive this identical array as input.
# from shared_code.fun_dfcspeed import ts2dfc_stream

# %%

# ===========================================================================
# Functions to compute dFC stream
# ===========================================================================


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


# %%
# ===========================================================================
# Compute dFC stream using the ts2dfc_stream function
# ===========================================================================

# Run the ts2dfc_stream function and time it

start = time.time()
dfc_stream_2d = ts2dfc_stream(ts, window_size=WINDOW, lag=LAG, format_data="2D")
end = time.time()
print(
    "dfc_stream_2d shape:",
    dfc_stream_2d.shape,
    "  min:",
    dfc_stream_2d.min().round(4),
    "  max:",
    dfc_stream_2d.max().round(4),
)
print("Time elapsed:", end - start, "seconds")

# %%
# ===========================================================================
# plot the dfc_stream_2d to visualize the data
# ===========================================================================

plt.figure(figsize=(10, 6))
plt.imshow(
    dfc_stream_2d, aspect="auto", interpolation="none", cmap="RdBu_r", vmin=-1, vmax=1
)
plt.colorbar(label=r"CC(BOLD$_{i, i\neq j}$, BOLD$_{j, j\neq i}$)")
plt.title("dFC Stream ")
plt.xlabel("Time Windows")
plt.ylabel("Region Pairs")
plt.show()
# %%
# ===========================================================================
# Functions to compute speed dFC
# ===========================================================================


def pearson_speed_vectorized(fc1: np.ndarray, fc2: np.ndarray) -> np.ndarray:
    """Vectorized Pearson-based speed via einsum.

    Parameters
    ----------
    fc1, fc2 : np.ndarray, shape (n_pairs, n_frames)

    Returns
    -------
    speeds : np.ndarray, shape (n_frames,)
        1 - corr(fc1[:, t], fc2[:, t]) for each t.
    """
    fc1c  = fc1 - fc1.mean(axis=0, keepdims=True)
    fc2c  = fc2 - fc2.mean(axis=0, keepdims=True)
    num   = np.einsum("ij,ij->j", fc1c, fc2c)
    ss1   = np.einsum("ij,ij->j", fc1c, fc1c)
    ss2   = np.einsum("ij,ij->j", fc2c, fc2c)
    denom = np.sqrt(ss1 * ss2)
    corr  = np.where(denom > np.finfo(float).eps, num / denom, 0.0)
    return 1.0 - corr


def dfc_speed_split(
    dfc_stream: np.ndarray,
    *,
    window_size: int,
    lag: int = 1,
    vstep: int = 1,
    tau_range: Sequence[int] | np.ndarray = (0,),
    method: str = "pearson",
    return_fc2: bool = False,
) -> np.ndarray:
    """Compute dFC speed across multiple tau offsets (window oversampling).

    For each tau in tau_range and each FC1 frame t, speed is:

        speed[t, tau] = 1 - corr(FC[t], FC[t + time_window + tau])

    where:
        time_window = ceil(window_size / lag)   — minimum frame separation
                                                  guaranteeing no overlap between
                                                  the windows of FC1 and FC2.

    vstep controls how FC1 frames are sampled:
        vstep=1            → dense: every frame (FC1s overlap each other)
        vstep=time_window  → sparse: FC1s are fully non-overlapping

    In both cases the FC1→FC2 separation is always exactly (time_window + tau)
    frames — vstep does NOT affect that gap.

    tau=0   → FC1 and FC2 are adjacent (minimum non-overlap).
    tau>0   → FC2 is further in time (longer-scale oversampling).
    tau<0   → FC2 partially overlaps FC1 (controlled overlap regime).
              tau must satisfy tau > -time_window to avoid full collapse (FC1==FC2).

    Parameters
    ----------
    dfc_stream : np.ndarray
        2D (n_pairs, n_frames) vectorized dFC stream, lower triangle, as produced
        by ts2dfc_stream(..., format_data='2D').
        3D (n_rois, n_rois, n_frames) is also accepted; the lower triangle is
        extracted automatically, consistent with ts2dfc_stream.
    window_size : int
        Size of the sliding window used to build the dFC stream (in TRs).
    lag : int
        Lag between consecutive windows used to build the dFC stream (in TRs).
        Default is 1.
    vstep : int
        Step between consecutive FC1 frames (in stream frames).
        Recommended values:
            1            → dense speed series (default)
            time_window  → fully non-overlapping FC1s
        Default is 1.
    tau_range : sequence of int
        Frame offsets applied on top of time_window. May include negative values
        for partial overlap. Must satisfy:
            tau > -ceil(window_size / lag)   for all tau in tau_range.
    method : {"pearson"}
        Similarity metric. Currently only "pearson" is supported.
    return_fc2 : bool
        If True, return the FC2 index array instead of speed values (useful
        for debugging index construction).

    Returns
    -------
    np.ndarray
        Shape (len(tau_range), T_eff) — one row of speed values per tau.
        T_eff = len(np.arange(0, indices_max, vstep)) - 1
        If return_fc2 is True, returns a 1D int array of FC2 indices instead.

    Raises
    ------
    ValueError
        If any tau causes full collapse (FC1 == FC2), or if there are not enough
        frames for the requested parameters.
    TypeError
        If dfc_stream is not a numpy array.

    Examples
    --------
     # Dense speed series, minimum non-overlap
     speeds = dfc_speed_split(stream, window_size=15, lag=1, vstep=1, tau_range=(0,))
     speeds.shape  # (1, T_eff)

     # Multi-tau oversampling with partial overlap
     speeds = dfc_speed_split(stream, window_size=15, lag=1, tau_range=(-2, -1, 0, 1, 2))
     speeds.shape  # (5, T_eff)

     # Fully non-overlapping FC1s
     tw = math.ceil(15 / 1)  # = 15
     speeds = dfc_speed_split(stream, window_size=15, lag=1, vstep=tw, tau_range=(0,))
    """
    # ── Input validation ──────────────────────────────────────────────────────
    if not isinstance(dfc_stream, np.ndarray):
        raise TypeError("dfc_stream must be a numpy array")
    if dfc_stream.ndim not in (2, 3):
        raise ValueError(
            "dfc_stream must be 2D (n_pairs, n_frames) or 3D (n_rois, n_rois, n_frames)"
        )
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if lag < 1:
        raise ValueError("lag must be >= 1")
    if vstep < 1:
        raise ValueError("vstep must be >= 1")
    if method != "pearson":
        raise ValueError(
            f"Unsupported method '{method}'. Currently only 'pearson' is supported."
        )

    # ── Flatten 3D → 2D (lower triangle, consistent with ts2dfc_stream) ──────
    if dfc_stream.ndim == 3:
        n_rois = dfc_stream.shape[0]
        tril_idx = np.tril_indices(n_rois, k=-1)
        fc_stream = dfc_stream[tril_idx[0], tril_idx[1], :]
    else:
        fc_stream = dfc_stream

    n_frames = fc_stream.shape[1]

    # ── time_window: minimum frame separation for non-overlap ─────────────────
    # lag=1 → time_window = window_size (1 frame = 1 TR)
    # lag>1 → fewer frames needed (1 frame = lag TRs)
    time_window = math.ceil(window_size / lag)

    # ── tau_range validation ──────────────────────────────────────────────────
    tau_arr = np.atleast_1d(np.asarray(tau_range, dtype=int))
    if tau_arr.size == 0:
        tau_arr = np.array([0], dtype=int)

    invalid = tau_arr[tau_arr <= -time_window]
    if invalid.size > 0:
        raise ValueError(
            f"tau values {invalid.tolist()} would cause FC1 == FC2 (full collapse). "
            f"All tau must be > {-time_window} (i.e., >= {-time_window + 1})."
        )

    # ── Index construction ────────────────────────────────────────────────────
    # largest FC1 index t such that FC2 index (t + time_window + tau_max) < n_frames
    tau_max     = int(tau_arr.max())
    indices_max = n_frames - (time_window + tau_max)

    if indices_max <= vstep:
        raise ValueError(
            f"Not enough frames for the requested parameters "
            f"(n_frames={n_frames}, window_size={window_size}, lag={lag}, "
            f"vstep={vstep}, tau_max={tau_max}). "
            f"Need at least {time_window + tau_max + vstep + 1} frames."
        )

    # FC1 frames: sub-sampled with step vstep
    # vstep=1          → [0, 1, 2, ..., indices_max-2]
    # vstep=time_window → [0, tw, 2*tw, ...]  fully non-overlapping FC1s
    head = np.arange(0, indices_max - vstep, vstep, dtype=int)

    if head.size < 1:
        raise ValueError("Not enough FC1 frames after sub-sampling. Reduce vstep.")

    # FC2 frames: always exactly (time_window + tau) ahead of the corresponding FC1
    # vstep does NOT appear here — the FC1→FC2 gap is solely (time_window + tau)
    fc1_idx = np.tile(head, (tau_arr.size, 1))                          # (n_tau, T_eff)
    fc2_idx = np.vstack([head + time_window + int(tau) for tau in tau_arr])  # (n_tau, T_eff)

    if return_fc2:
        return fc2_idx.flatten().astype(int)

    # ── Speed computation ─────────────────────────────────────────────────────
    # Flatten across tau for a single batched einsum call
    fc1_mat = fc_stream[:, fc1_idx.flatten()]        # (n_pairs, T_eff * n_tau)
    fc2_mat = fc_stream[:, fc2_idx.flatten()]

    speeds_flat = pearson_speed_vectorized(fc1_mat, fc2_mat)
    speeds      = np.clip(speeds_flat, 0.0, 2.0).reshape(tau_arr.size, -1)

    return speeds   # shape: (len(tau_range), T_eff)

# %%
speed_split = dfc_speed_split(
    dfc_stream_2d,
    window_size=WINDOW,
    lag=LAG,
    vstep=1,  # fully non-overlapping FC1s
    # vstep=WINDOW,  # fully non-overlapping FC1s
    tau_range=TAU_RANGE,
    method="pearson",
    return_fc2=False,
)
print("speed_split shape:", speed_split.shape)
#%%#plot the speed_split to visualize the data
plt.figure(figsize=(10, 6))
for i in range(speed_split.shape[0]):
    plt.plot(speed_split[i], marker="o", label=f"tau={i}")
plt.title("Speeds from dfc_speed_split")
plt.xlabel("Time Steps")
plt.ylabel("Speed")
plt.grid()
plt.legend()
plt.show()

# plot the histogram of speed_split to visualize the distribution
plt.figure(figsize=(10, 6))
for i in range(speed_split.shape[0]):
    plt.hist(
        speed_split[i],
        bins=20,
        alpha=0.5,
        label=f"tau={i}",
        histtype="step",
        linewidth=3,
        density=True,
    )
plt.title("Histogram of Speeds from dfc_speed_split")
plt.xlabel("Speed")
plt.ylabel("Frequency")
plt.grid()
plt.legend()
plt.show()

# plot kde of speed_split to visualize the distribution
plt.figure(figsize=(10, 6))
for i in range(speed_split.shape[0]):
    sns.kdeplot(
        speed_split[i],
        bw_adjust=0.3,
        common_norm=True,
        # cumulative=True,
        label=f"tau={i}",
        fill=False,
        linewidth=3,
    )
plt.title("KDE of Speeds from dfc_speed_split")
plt.xlabel("Speed")
plt.ylabel("Density")
plt.yscale("log")
plt.grid()
plt.legend()
plt.show()

# %%

fc_speed_stream_idx = dfc_speed_split(
    dfc_stream_2d,
    window_size=WINDOW,
    lag=LAG,
    vstep=1,
    tau_range=(0,),
    method="pearson",
    return_fc2=True,
)

fc_speed_stream = dfc_stream_2d[:, fc_speed_stream_idx]

#%%
#TSNE visualization of the fc_speed_stream
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=SEED)
fc_speed_stream_2d = tsne.fit_transform(fc_speed_stream.T)

#%%
plt.figure(figsize=(10, 6))
plt.plot(fc_speed_stream_2d[:, 0], fc_speed_stream_2d[:, 1], "o-", alpha=0.7)

plt.title("t-SNE of FC Speed Stream")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.grid()
plt.show()

# %%


# %%
