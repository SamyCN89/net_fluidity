#!/usr/bin/env python3

# %%
from scipy.io import loadmat
from shared_code.fun_dfcspeed import ts2dfc_stream, dfc_stream2fcd
import matplotlib.pyplot as plt
import numpy as np



#%%
# Compute the speed of dFC
def dfc_speed_split(
    dfc_stream,
    vstep=1,
    tau_range=0,
    method="pearson",
    return_fc2=False,
    triu_indices=None,
    time_offset=0,
):
    """
    Unified function to calculate the speed of variation in dynamic functional connectivity (dFC).

    ----------
    dfc_stream : numpy.ndarray
        Dynamic functional connectivity stream. Can be either: 2D array (n_pairs, n_frames) 3D array (n_rois, n_rois, n_frames): Full FC matrices over time
    vstep : int, optional
        Time step for computing FC speed (default=1). Must be positive and < n_frames.
    method : str, optional
        Correlation method to use for speed computation (default='pearson').
        Supported methods:
        - 'pearson': Pearson correlation coefficient
        - 'spearman': Spearman rank correlation
        - 'cosine': Cosine similarity
    tril_indices : tuple, optional
        Pre-computed triangular indices for 3D input (default=None).
        If None, will be computed automatically for 3D input.
    return_fc2 : bool, optional
        If True, also return the second FC matrix for each time step (default=False).

    Returns
    -------
    speed_median : float
        Median of the computed speed distribution.
    speeds : numpy.ndarray
        Time series of computed speeds with shape (n_frames - vstep,).
    fc2_stream : numpy.ndarray, optional
        Second FC matrix for each time step. Only returned if return_fc2=True.
        Shape: (n_pairs, n_frames - vstep) for vectorized output.

    References
    ----------
    Dynamic Functional Connectivity as a complex random walk: Definitions and the dFCwalk toolbox
    Lucas Arbabyazd, Diego Lombardo, Olivier Blin, Mira Didic, Demian Battaglia, Viktor Jirsa
    MethodsX 2020, doi: 10.1016/j.mex.2020.101168
    """
    from shared_code.fun_optimization import (
        cosine_speed_vectorized,
        pearson_speed_vectorized,
        spearman_speed,
    )

    # Input validation
    if not isinstance(dfc_stream, np.ndarray):
        raise TypeError("dfc_stream must be a numpy array")

    if dfc_stream.ndim not in [2, 3]:
        raise ValueError(
            "dfc_stream must be 2D (n_pairs, frames) or 3D (roi, roi, frames)"
        )

    if not isinstance(vstep, int) or vstep <= 0:
        raise TypeError("vstep must be a positive integer")

    if method not in ["pearson", "spearman", "cosine"]:
        raise ValueError(
            f"Unsupported method '{method}'. Use 'pearson', 'spearman', or 'cosine'"
        )

    # Handle input format conversion
    # 3D input: (n_rois, n_rois, n_frames)
    if dfc_stream.ndim == 3:
        n_rois = dfc_stream.shape[0]
        n_frames = dfc_stream.shape[2]

        # Generate triangular indices if not provided
        if triu_indices is None:
            triu_indices = np.triu_indices(n_rois, k=1)

        # Extract upper triangular values efficiently
        fc_stream = dfc_stream[triu_indices[0], triu_indices[1], :]
    else:
        # 2D input: (n_pairs, n_frames)
        fc_stream = dfc_stream
        n_frames = fc_stream.shape[1]

    # Validate frame count vs vstep
    if vstep >= n_frames:
        raise ValueError(
            f"vstep ({vstep}) must be less than number of frames ({n_frames})"
        )

    fc1_indices = []
    fc2_indices = []

    # Determine maximum tau shift from provided tau_range
    tau_max = int(np.max(tau_range)) if np.size(tau_range) > 0 else 0
    indices_max = n_frames - (vstep + tau_max + time_offset)
    indices = np.arange(0, indices_max, 1)

    time_window = int(np.ceil(time_offset / vstep)) if time_offset > 0 else 0

    print(f"indices_max: {indices_max}, n_frames: {n_frames}, vstep: {vstep}, tau_max: {tau_max}, time_offset: {time_offset}")
    # Generate index pairs for FC matrices based on tau_range
    if np.size(tau_range) > 1:
        for tau_aux in tau_range:
            fc1_indices.append(indices[:-1])  # Indices for the first FC matrix
            fc2_indices.append(
                indices[1:] + tau_aux + time_window
            )  # Indices for the second FC matrix
            # print(indices[:-1], indices[1:]+tau_aux+time_offset+vstep-1)
    else:
        tau_aux = tau_range
        fc1_indices.append(indices[:-1])
        fc2_indices.append(
            # (indices[1:] // vstep) + tau_aux + time_offset + vstep - 1
            (indices[1:] + tau_aux + time_window )
        )  # Indices for the second FC matrix

    print(f"fc1_indices: {fc1_indices}, fc2_indices: {fc2_indices}")
    n_speeds = (len(indices) - 1) * np.size(tau_range)
    n_pairs = fc_stream.shape[0]

    # Pre-allocate output arrays for efficiency
    speeds = np.empty((n_speeds, np.size(tau_range)), dtype=np.float32)
    fc2_stream = None

    # Extract FC matrices for vectorized computation
    fc1_matrices = fc_stream[
        :, np.array(fc1_indices).flatten()
    ]  # Shape: (n_pairs, n_speeds)
    fc2_matrices = fc_stream[
        :, np.array(fc2_indices).flatten()
    ]  # Shape: (n_pairs, n_speeds)
    if return_fc2:
        fc2_stream_indices = np.empty(
            n_speeds, dtype=int
        )  # Pre-allocate for second FC matrix indices
        # fc2_stream[:, :] = fc2_matrices
        fc2_stream_indices[:] = (np.array(fc2_indices).flatten()).astype(int)
        return fc2_stream_indices

    # Use optimized speed computation functions for maximum performance
    if method == "pearson":
        speeds = pearson_speed_vectorized(fc1_matrices, fc2_matrices)
    elif method == "spearman":
        speeds = spearman_speed(fc1_matrices, fc2_matrices)
    elif method == "cosine":
        speeds = cosine_speed_vectorized(fc1_matrices, fc2_matrices)
    else:
        raise ValueError(
            f"Unsupported method '{method}'. Use 'pearson', 'spearman', or 'cosine'"
        )

    # Ensure speeds are within valid range [0, 2] for numerical stability
    speeds = np.clip(speeds, 0, 2.0)
    speeds_mat = speeds.reshape(len(tau_range), -1)  # Reshape to (n_pairs, n_speeds)

    return speeds_mat
speed = dfc_speed_split(dfc_stream, vstep=int(s), tau_range=np.arange(1), method='pearson', return_fc2=False, time_offset=lag)

# %%
# Load data
data = loadmat('/home/samy/Bureau/vscode/net_fluidity/scripts/speed/test.mat')['TS']

#sample frequencies in Hz
sf=((1/.72)*np.arange(55,65)).astype(int)

# %% Compute dFC
speed_pool = []
for s in sf:

    #lag in number of frames
    lag = int((s*.72) *.05)

    dfc_stream = ts2dfc_stream(data, s, lag, format_data='2D')
    fcd = dfc_stream2fcd(dfc_stream)
    speed = dfc_speed_split(dfc_stream, vstep=int(lag), tau_range=np.arange(2), method='pearson', return_fc2=False, time_offset=s)

    print(s, lag, dfc_stream.min(), dfc_stream.max(), speed.min(), speed.max(), fcd.min(), fcd.max())
    speed_pool.append(speed)

# %%
from shared_code.fun_optimization import fast_corrcoef
for i,k in enumerate(range(200)):
    wstart = k * lag
    wstop = wstart + s
    window = data[wstart:wstop, :]
    fc = fast_corrcoef(window)
    print(i,wstart, wstop, np.shape(window), np.shape(fc))

    dfc_stream[:, k] = fc[tril_idx]

#%%
#%%
# imshow of results
plt.imshow(dfc_stream,
           aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(label='Correlation')

#%%
speed_flat = np.array([x.flatten() for x in speed_pool], dtype=object)
speed_flat = np.concatenate(speed_flat)


#plot hist of speed
plt.hist(
    speed_flat.ravel(), bins=50, histtype='step', density=True)
#%%
plt.imshow(fcd,
           aspect='auto', cmap='jet',
           vmin=0, vmax=1)
plt.colorbar()

# %%
