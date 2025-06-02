
#%%
from pathlib import Path
import numpy as np
import pandas as pd
import time

from shared_code.fun_utils import get_paths
from shared_code.fun_loaddata import (load_mat_timeseries, extract_mouse_ids, load_npz_dict)
import time
from scipy.io import loadmat
import re
#%%

paths = get_paths(dataset_name='julien_caillette', 
                  timecourse_folder='time_courses',
                  cognitive_data_file='mice_groups_comp_index.xlsx')

paths['roi'] = paths['timeseries'] / 'all_ROI_coimagine.txt'


# 1 Load raw time series data from .mat files
ts_list, ts_shapes, loaded_files = load_mat_timeseries(paths['timeseries'])
ts_ids = extract_mouse_ids(loaded_files)
# %%
# 2 Load cognitive data from .xlsx file
cog_data     = pd.read_excel(paths['cog_data'], sheet_name='mice_groups_comp_index')

# %%
# 3 Clean region labels
region_labels = np.loadtxt(paths['roi'], dtype=str).tolist()
# %%
# ------------------------ Load Preprocessed Data ------------------------

data_ts_pre = load_npz_dict(paths['preprocessed'] / Path('ts_filtered_unstacked.npz'))
ts = data_ts_pre['ts']
n_animals = data_ts_pre['n_animals']
total_tp = data_ts_pre['total_tp']
regions = data_ts_pre['regions']
anat_labels = data_ts_pre['anat_labels']
# Load cognitive data
cog_data_filtered = pd.read_csv(paths['preprocessed'] / Path("cog_data_filtered.csv"))


print(f"Loaded {len(ts)} time series")
print(f"Loaded cognitive data for {len(cog_data_filtered)} animals")

# %%
