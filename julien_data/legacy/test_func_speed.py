#%%
# import pickle
from pathlib import Path
from networkx import density
import numpy as np
from class_dataanalysis_julien import DFCAnalysis
import pickle

data = DFCAnalysis()
#%%
# Load raw data
# data.load_raw_timeseries()
# data.load_raw_cognitive_data()
# data.load_raw_region_labels()

# Load preprocessed data
data.load_preprocessed_data()


data.get_temporal_parameters()
#%%
# Match these variables to your last run:
prefix = "speed"
save_path = data.paths['speed']  # <-- update this!
time_window_range = data.time_window_range           # <-- list of window sizes, same as in your analysis
tau_range = np.arange(0,data.tau+ 1)                   # <-- as above
n_animals = data.n_animals                # <-- as above
data.load_preprocessed_data()

window_file_total = save_path / f"{prefix}_windows{len(time_window_range)}_tau{len(tau_range)}_animals_{n_animals}.pkl"
#%%
with open(window_file_total, 'rb') as f:
    all_speed = pickle.load(f)
# %%
all_speed[0]


#%%
def dfc_speed(dfc_stream, 
            vstep=1, 
            method='pearson', 
            return_fc2=False,
            tril_indices=None
            ):
    """
    Unified function to calculate the speed of variation in dynamic functional connectivity (dFC).
    
    This function computes the speed of dFC variation as 1 - correlation between FC states 
    separated by a specified time step. It combines optimized vectorized computation using 
    einsum with support for multiple correlation methods and efficient handling of both 
    2D and 3D input formats.
    
    Parameters
    ----------
    dfc_stream : numpy.ndarray
        Dynamic functional connectivity stream. Can be either:
        - 2D array (n_pairs, n_frames): Lower triangular FC values over time
        - 3D array (n_rois, n_rois, n_frames): Full FC matrices over time
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
        
    Raises
    ------
    ValueError
        If dfc_stream dimensions are invalid, vstep is out of bounds, or method is unsupported.
    TypeError
        If vstep is not a positive integer.
        
    Notes
    -----
    This unified implementation combines the best features from multiple existing dfc_speed 
    implementations across the project:

    1. **Vectorized Computation**: Uses optimized einsum operations for correlation computation
    2. **Multiple Methods**: Supports Pearson, Spearman, and cosine correlation methods
    3. **Flexible Input**: Handles both 2D vectorized and 3D matrix formats efficiently
    4. **Memory Efficient**: Pre-allocates arrays and uses numerical stability improvements
    5. **Robust Validation**: Comprehensive input validation and error handling
    
    The speed is computed as: speed = 1 - correlation(FC_t, FC_{t+vstep})
    where correlation method is specified by the 'method' parameter.
    
    For 3D input, the function efficiently extracts lower triangular values using 
    pre-computed or automatically generated indices, then processes as 2D data.
    
    Examples
    --------
    >>> # 2D input (vectorized FC)
    >>> dfc_2d = np.random.randn(45, 100)  # 45 pairs, 100 time frames
    >>> median_speed, speeds = dfc_speed(dfc_2d, vstep=1, method='pearson')
    >>> print(f"Median speed: {median_speed:.3f}")
    
    >>> # 3D input (FC matrices)  
    >>> dfc_3d = np.random.randn(10, 10, 100)  # 10x10 FC matrices, 100 frames
    >>> median_speed, speeds = dfc_speed(dfc_3d, vstep=2, method='spearman')
    
    >>> # With second FC matrix return
    >>> median_speed, speeds, fc2 = dfc_speed(dfc_2d, vstep=1, return_fc2=True)
    
    References
    ----------
    Dynamic Functional Connectivity as a complex random walk: Definitions and the dFCwalk toolbox
    Lucas Arbabyazd, Diego Lombardo, Olivier Blin, Mira Didic, Demian Battaglia, Viktor Jirsa
    MethodsX 2020, doi: 10.1016/j.mex.2020.101168
    """
    from shared_code.fun_dfcspeed import (
        pearson_speed_vectorized,
        spearman_speed,
        cosine_speed_vectorized
    )
    # Input validation
    if not isinstance(dfc_stream, np.ndarray):
        raise TypeError("dfc_stream must be a numpy array")
    
    if dfc_stream.ndim not in [2, 3]:
        raise ValueError("dfc_stream must be 2D (n_pairs, frames) or 3D (roi, roi, frames)")
    
    if not isinstance(vstep, int) or vstep <= 0:
        raise TypeError("vstep must be a positive integer")
        
    if method not in ['pearson', 'spearman', 'cosine']:
        raise ValueError(f"Unsupported method '{method}'. Use 'pearson', 'spearman', or 'cosine'")
    
    # Handle input format conversion
    if dfc_stream.ndim == 3:
        n_rois = dfc_stream.shape[0]
        n_frames = dfc_stream.shape[2]
        
        # Generate triangular indices if not provided
        if tril_indices is None:
            tril_indices = np.tril_indices(n_rois, k=-1)
        
        # Extract lower triangular values efficiently
        fc_stream = dfc_stream[tril_indices[0], tril_indices[1], :]
    else:
        # 2D input: (n_pairs, n_frames)
        fc_stream = dfc_stream
        n_frames = fc_stream.shape[1]
    
    # Validate frame count vs vstep
    if vstep >= n_frames:
        raise ValueError(f"vstep ({vstep}) must be less than number of frames ({n_frames})")
    
    indices = np.arange(0, n_frames - vstep, 1)
    n_speeds = len(indices)-1
    n_pairs = fc_stream.shape[0]
    
    # Pre-allocate output arrays for efficiency
    speeds = np.empty(n_speeds)
    fc2_stream = None
    
    # Extract FC matrices for vectorized computation
    fc1_matrices = fc_stream[:, indices[:-1]]  # Shape: (n_pairs, n_speeds)
    fc2_matrices = fc_stream[:, indices[1:]+vstep-1]   # Shape: (n_pairs, n_speeds)

    print(np.shape(indices[:-1]), np.shape(indices[1:]+vstep-1))
    print(indices[:-1], indices[1:]+vstep-1)

    if return_fc2:
        fc2_stream = np.empty((n_pairs, n_speeds))
        fc2_stream[:, :] = fc2_matrices
    
    # Use optimized speed computation functions for maximum performance
    if method == 'pearson':
        speeds = pearson_speed_vectorized(fc1_matrices, fc2_matrices)
    elif method == 'spearman':
        speeds = spearman_speed(fc1_matrices, fc2_matrices)
    elif method == 'cosine':
        speeds = cosine_speed_vectorized(fc1_matrices, fc2_matrices)
    else:
        raise ValueError(f"Unsupported method '{method}'. Use 'pearson', 'spearman', or 'cosine'")

    # Ensure speeds are within valid range [-1, 2] for numerical stability
    speeds = np.clip(speeds, -1.0, 2.0)
    
    # Compute median speed
    speed_median = np.median(speeds)
    
    # Return results based on options
    if return_fc2:
        return speed_median, speeds, fc2_stream
    else:
        return speeds
    
#%%

ws_5 = data.load_dfc_1_window(lag=1,window=50)
speed = dfc_speed(ws_5[0], 
            vstep=50, 
            method='pearson', 
            return_fc2=False,
            tril_indices=None
            )
# %%

all_speed_5_0 = all_speed[-1][0][0]

# hist of all_speed_5_0 abd speed
import matplotlib.pyplot as plt
plt.hist(all_speed_5_0, bins=10, alpha=0.5, label='all_speed_5_0')
plt.hist(speed, bins=10, alpha=0.5, label='speed')
plt.xlabel('Speed')
plt.ylabel('Frequency')
plt.title('Histogram of Speed Values')
plt.legend()
# %%

#%%
import logging
from joblib import Parallel, delayed
from tqdm import tqdm
logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

def dfc_speed_split(dfc_stream, 
            vstep=1, 
            tau=0,
            tau_range=0,
            method='pearson', 
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
        pearson_speed_vectorized,
        spearman_speed,
        cosine_speed_vectorized
    )
    
    # Input validation
    if not isinstance(dfc_stream, np.ndarray):
        raise TypeError("dfc_stream must be a numpy array")
    
    if dfc_stream.ndim not in [2, 3]:
        raise ValueError("dfc_stream must be 2D (n_pairs, frames) or 3D (roi, roi, frames)")
    
    if not isinstance(vstep, int) or vstep <= 0:
        raise TypeError("vstep must be a positive integer")
        
    if method not in ['pearson', 'spearman', 'cosine']:
        raise ValueError(f"Unsupported method '{method}'. Use 'pearson', 'spearman', or 'cosine'")
    
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
        raise ValueError(f"vstep ({vstep}) must be less than number of frames ({n_frames})")

    fc1_indices = []
    fc2_indices = []    

    indices_max = n_frames - (vstep + np.max(tau) + time_offset) 
    indices = np.arange(0, indices_max, 1)
    if np.size(tau_range) > 1:
        for tau_aux in tau_range:
            fc1_indices.append(indices[:-1])  # Indices for the first FC matrix
            fc2_indices.append(indices[1:]+tau_aux+time_offset+vstep-1)   # Indices for the second FC matrix
            print(indices[:-1], indices[1:]+tau_aux+time_offset+vstep-1)
    else:
        tau_aux = tau_range
        fc1_indices.append(indices[:-1])
        fc2_indices.append(indices[1:]+tau_aux+time_offset+vstep-1)   # Indices for the second FC matrix

    n_speeds = (len(indices)-1) * np.size(tau_range)
    n_pairs = fc_stream.shape[0]
    
    # Pre-allocate output arrays for efficiency
    speeds = np.empty((n_speeds, np.size(tau_range)), dtype=np.float32)
    fc2_stream = None
    
    # Extract FC matrices for vectorized computation
    fc1_matrices = fc_stream[:, np.array(fc1_indices).flatten()]  # Shape: (n_pairs, n_speeds)
    fc2_matrices = fc_stream[:, np.array(fc2_indices).flatten()]  # Shape: (n_pairs, n_speeds)
    if return_fc2:
        fc2_stream_indices = np.empty(n_speeds, dtype=int)  # Pre-allocate for second FC matrix indices
        # fc2_stream[:, :] = fc2_matrices
        fc2_stream_indices[:] = (np.array(fc2_indices).T.flatten()).astype(int)
        return fc2_stream_indices

    # Use optimized speed computation functions for maximum performance
    if method == 'pearson':
        speeds = pearson_speed_vectorized(fc1_matrices, fc2_matrices)
    elif method == 'spearman':
        speeds = spearman_speed(fc1_matrices, fc2_matrices)
    elif method == 'cosine':
        speeds = cosine_speed_vectorized(fc1_matrices, fc2_matrices)
    else:
        raise ValueError(f"Unsupported method '{method}'. Use 'pearson', 'spearman', or 'cosine'")

    # Ensure speeds are within valid range [-1, 2] for numerical stability
    speeds = np.clip(speeds, -1.0, 2.0)
    speeds_mat = speeds.reshape(len(tau_range), -1)  # Reshape to (n_pairs, n_speeds)

    return speeds_mat


speed2 = dfc_speed_split(ws_5[0], 
            vstep=50, 
            tau_range=np.arange(0, data.tau + 1),  # <-- tau range from your analysis
            tau=data.tau,  # <-- tau from your analysis
            method='pearson', 
            return_fc2=False,
            triu_indices=None,
            time_offset=0,
            )
# %%

# hist of speed2 and speed
plt.hist((speed, speed2[3]), bins=50, alpha=0.5, histtype='step', label=('speed', 'speed2'))
# plt.hist(speed, bins=50, alpha=0.5, label='speed')
# plt.hist(speed2[0], bins=50, alpha=0.5, label='speed2')
plt.xlabel('Speed')
plt.ylabel('Frequency')
plt.title('Histogram of Speed Values')
plt.legend()
# %%
