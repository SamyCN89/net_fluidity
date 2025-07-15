
#%%
from pathlib import Path
import numpy as np
import pandas as pd
import time

from shared_code.fun_paths import get_paths
from shared_code.fun_loaddata import (load_mat_timeseries, extract_mouse_ids, load_npz_dict, make_file_path)
#%%
# ------------------------ Path's Configuration ------------------------
paths = get_paths(dataset_name='julien_caillette', 
                  timecourse_folder='time_courses',
                  cognitive_data_file='mice_groups_comp_index.xlsx',
                  anat_labels_file='all_ROI_coimagine.txt')

# ------------------------ 1.1 Load raw time series data ----------------------
# 1.1 Load raw time series data from .mat files
# (n_animals x n_timepoints x n_regions) -> 23 and 49 400 timepoints (Dp1Yey-VEH and Dp1Yey-LCTB92)
ts_list, ts_shapes, loaded_files = load_mat_timeseries(paths['timeseries'])
ts_ids = extract_mouse_ids(loaded_files)

# ---------------------- 1.2 Load Raw cognitive data ------------------------
# 1.2 Load cognitive data from .xlsx file
cog_data     = pd.read_excel(paths['cog_data'], sheet_name='mice_groups_comp_index')

# ---------------------- 1.3 Load Region labels ------------------------
# 1.3 Clean region labels
region_labels = np.loadtxt(paths['labels'], dtype=str).tolist()
# %%
# ---------------------- 2 Load Preprocessed data ------------------------
# 2.1 Load preprocessed time series data
data_ts_pre = load_npz_dict(paths['preprocessed'] / Path('ts_filtered_unstacked.npz'))
ts = data_ts_pre['ts']
n_animals = data_ts_pre['n_animals']
total_tr = data_ts_pre['total_tr']
regions = data_ts_pre['regions']
anat_labels = data_ts_pre['anat_labels']

# 2.2 Load preprocessed cognitive data
cog_data_filtered = pd.read_csv(paths['preprocessed'] / Path("cog_data_filtered.csv"))

print(f"Loaded {len(ts)} time series")
print(f"Loaded cognitive data for {len(cog_data_filtered)} animals")

# 2.3 Load groups
groups = cog_data_filtered.groupby(['genotype', 'treatment']).groups

# %%
# ---------------------- 3 DFC stream anaylisys ------------------------
# Load 1 time window size (TW_i) dfc_stream data from <path> 
# Shape (n_animals x n_pairs x n_windows)

lag = 1
tau = 3
window_parameter = (5, 50, 1)
time_window_range = np.arange(window_parameter[0], window_parameter[1] + 1, window_parameter[2])

print(f"\ndFC parameters: lag={lag}, tau={tau}, window_range={time_window_range[0]}-{time_window_range[-1]}")

# Load the computed results
for window_size in time_window_range:

    prefix = 'dfc'
    file_path = make_file_path(paths['dfc'] , prefix, window_size, lag, n_animals, regions)

    print(np.load(file_path, allow_pickle=True).files)
    # Check available keys
    results = load_npz_dict(file_path)
    dfc = results[prefix]
#%%
# ---------------------- 4 DFC speed analysis ------------------------
# Load Speed (S) dfc data from <path> 
# Shape (n_windows x (n_animals x n_tau) x n_pairs)

speed=[]
fc_speed = []
for window_size in time_window_range:

    prefix = 'speed'
    file_path = make_file_path(paths['speed'] , prefix, window_size, tau, n_animals, regions)
    # file_path = make_file_path(paths['results'] / 'speed', prefix, window_size, lag, n_animals, regions)

    print(np.load(file_path, allow_pickle=True).files)
    # Check available keys
    results = load_npz_dict(file_path)
    # speed = results[prefix]


    speed.append(results['speed'])   # Convert to numpy array if needed
    fc_speed.append(results['fc2'])   # Convert to numpy array if needed
# speed = np.array(speed, dtype=object) # Convert to numpy array if needed
# %%

speeds_all = [speed[ws] for ws in np.arange(len(time_window_range))]
# %%
n_windows = len(speeds_all)
n_animals = speeds_all[0].shape[0]

# Transpose so speeds_all_T[animal][window] = speeds array
speeds_all_T = [
    [speeds_all[win][animal] for win in range(n_windows)]
    for animal in range(n_animals)
]

n_tau = 5
n_animals = int(speeds_all[0].shape[0] // n_tau)

# For each window, split by animal and tau
# speeds_all[win][animal * n_tau + tau_idx] is the row for animal "animal", tau "tau_idx"
speeds_by_animal_tau = [
    [
        [
            speeds_all[win][animal * n_tau + tau_idx]
            for tau_idx in range(n_tau)
        ]
        for animal in range(n_animals)
    ]
    for win in range(n_windows)
]
# speeds_by_animal_tau[win][animal][tau] gives you a 1D array of speeds

# If you want per-animal, all windows, all tau, flatten window dimension:
speeds_all_T = [
    [speeds_by_animal_tau[win][animal][tau]
     for win in range(n_windows) for tau in range(n_tau)]
    for animal in range(n_animals)
]










