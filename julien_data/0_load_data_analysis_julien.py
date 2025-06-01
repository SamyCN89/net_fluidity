
#%%
from pathlib import Path
import numpy as np
import pandas as pd
import time

from shared_code.fun_utils import get_paths 
from shared_code.fun_loaddata import load_mat_timeseries, extact_mouse_ids

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
