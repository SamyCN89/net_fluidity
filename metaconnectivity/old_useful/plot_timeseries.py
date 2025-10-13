#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 02:59:41 2025

@author: samy
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fun_utils import get_paths, set_figure_params

# ========================== Figure parameters ================================
# Set figure parameters globally
save_fig = set_figure_params(True)

# =================== Paths and folders =======================================

paths = get_paths()
data_ts = np.load(paths['results'] /  'ts_and_meta_2m4m.npz')

# ========================== Load data =========================

#Parameters and indices of variables
ts          = data_ts['ts']
n_animals   = int(data_ts['n_animals'])
total_tp    = data_ts['total_tp']
regions     = data_ts['regions']
is_2month_old = data_ts['is_2month_old']
anat_labels = data_ts['anat_labels']

#%%

plt.figure(figsize=(12, 8))
offset = 0.07  # vertical offset between time series
for i, ts1 in enumerate(ts[0].T):
    plt.plot(ts1 + i * offset, label=f"TS {i+1}")
plt.ylim(-0.1,0.75)
plt.title("Stacked Time Series")
plt.xlabel("Time Points")
plt.ylabel("Signal + Offset")
plt.tight_layout()
plt.show()
plt.savefig(paths['figures'] / 'ts/ts_extract.png')