#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:26:30 2024

@author: samy
"""

#%%
from click import group
import numpy as np
import time
# from functions_analysis import *
from pathlib import Path

import scipy

from fun_loaddata import *
from fun_dfcspeed import *

from fun_metaconnectivity import (compute_metaconnectivity, 
                                  intramodule_indices_mask, 
                                  get_fc_mc_indices, 
                                  get_mc_region_identities, 
                                  fun_allegiance_communities,
                                  compute_trimers_identity,
                                    build_trimer_mask,
                                  )

from fun_utils import (set_figure_params, 
                       get_paths, 
                       load_cognitive_data,
                       load_timeseries_data,
                       load_grouping_data,
                       )
# =============================================================================
# This code compute 
# Load the data
# Intersect the 2 and 4 months to have data that have the two datapoints
# ========================== Figure parameters ================================
save_fig = set_figure_params(False)

# =================== Paths and folders =======================================
timeseries_folder = 'Timecourses_updated_03052024'
external_disk = True
if external_disk==True:
    root = Path('/media/samy/Elements1/Proyectos/LauraHarsan/script_mc/')
else:    
    root = Path('/home/samy/Bureau/Proyect/LauraHarsan/Ines/')

paths = get_paths(external_disk=True,
                  external_path=root,
                  timecourse_folder=timeseries_folder)

# ========================== Load data =========================
cog_data_filtered = load_cognitive_data(paths['sorted'] / 'cog_data_sorted_2m4m.csv')
data_ts = load_timeseries_data(paths['sorted'] / 'ts_and_meta_2m4m.npz')
mask_groups, label_variables = load_grouping_data(paths['results'] / "grouping_data_oip.pkl")


# ========================== Indices ==========================================
ts=data_ts['ts']
n_animals = data_ts['n_animals']
regions = data_ts['regions']
anat_labels = data_ts['anat_labels']


#%%
# ======================== Fluidity analysis ==========================================
#Parameters speed


def extremal_Sueveges(Y, p):
    u = np.quantile(Y, p)              # Same: p-th quantile threshold
    q = 1 - p
    Li = np.where(Y > u)[0]            # Same: get indices where Y > threshold
    Ti = np.diff(Li)                   # Time between exceedances
    Si = Ti - 1                        # Inter-event times
    Nc = np.sum(Si > 0)                # Count of clusters (Si > 0)
    N = len(Ti)                        # Number of inter-exceedance intervals
    sum_qSi = np.sum(q * Si)           # Sum for formula

    if sum_qSi == 0:                   # Prevent division by 0
        return np.nan

    numerator = sum_qSi + N + Nc - np.sqrt((sum_qSi + N + Nc)**2 - 8 * Nc * sum_qSi)
    theta = numerator / (2 * sum_qSi)  # Suveges' closed-form formula
    return theta


from scipy.spatial.distance import cdist
from scipy.stats import genpareto

def MA_EEG_Man_Dim_Flui(TS, quanti=0.98, step=1):
    L = TS.shape[0]
    Dimension = np.zeros(L)
    Fluidity = np.zeros(L)

    for j in range(0, L, step):
        idx_others = np.setdiff1d(np.arange(0, L, step), [j])
        if len(idx_others) == 0:
            continue

        distance = cdist(TS[j:j+1], TS[idx_others])[0]
        logdista = -np.log(distance + np.finfo(float).eps)  # Avoid log(0)

        Fluidity[j] = extremal_Sueveges(logdista, quanti)
        thresh = np.quantile(logdista, quanti)

        sorted_logdista = np.sort(logdista)
        findidx = np.argmax(sorted_logdista > thresh)
        logextr = sorted_logdista[findidx:-1] if (findidx is not None and findidx < len(sorted_logdista)-1) else np.array([])


        if logextr.size > 0:
            try:
                c, loc, scale = genpareto.fit(logextr - thresh, floc=0)
                Dimension[j] = 1. / (scale + np.finfo(float).eps)
            except Exception:
                Dimension[j] = np.nan
        else:
            Dimension[j] = np.nan

    return Fluidity, Dimension

#%%

fluidity = np.zeros((n_animals, len(ts[0])))
dimension = np.zeros((n_animals, len(ts[0])))

results = Parallel(n_jobs=-1)(
    delayed(MA_EEG_Man_Dim_Flui)(ts[xx]) for xx in tqdm(range(n_animals))
)   
for xx, (f, d) in enumerate(results):
    fluidity[xx] = f
    dimension[xx] = d

#%%
import matplotlib.pyplot as plt


# Plotting the results
plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.plot(fluidity, label='Fluidity')
plt.title('Fluidity')
plt.xlabel('Time')
plt.ylabel('Fluidity')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(dimension, label='Dimension')
plt.title('Dimension')
plt.xlabel('Time')
plt.ylabel('Dimension')
plt.legend()
#%%
# ======================== Statistics ==========================================

#comapration of q5 and q50 percentiles in dimension and fluidity 
q_range=(5,50,95)

qq_dimension = np.nanmean(np.nanpercentile(dimension, q_range, axis=1),axis=1)
print(f'dimension Q{q_range[0]} = {qq_dimension[0]}')
print(f'dimension Q{q_range[1]} = {qq_dimension[1]}')
print(f'dimension Q{q_range[2]} = {qq_dimension[2]}')

qq_fluidity = np.nanmean(np.nanpercentile(fluidity, q_range, axis=1),axis=1)
print(f'fluidity Q{q_range[0]} = {qq_fluidity[0]}')
print(f'fluidity Q{q_range[1]} = {qq_fluidity[1]}')
print(f'fluidity Q{q_range[2]} = {qq_fluidity[2]}')

# %%
#Save ts[1] in .mat format
# from scipy.io import savemat
# data_share ={'ts':ts[1]}
# savemat('ts1.mat', data_share)

# dimension_group = np.zeros(np.array(mask_groups))

dimension_group = {}
fluidity_group = {}
qq_dimension_group = {}
qq_fluidity_group = {}

for label, mask in zip(label_variables[0], mask_groups[0]):
    dimension_group[label] = dimension[mask]
    fluidity_group[label] = fluidity[mask]
    qq_dimension_group[label] = np.nanpercentile(dimension_group[label], q_range, axis=1)
    qq_fluidity_group[label] = np.nanpercentile(fluidity_group[label], q_range, axis=1)

# # group the dimension and fluidity by genotype and then compute the percentile
# for xx in range(len(label_variables[1])):
#     dimension_group[label_variables[1][xx]] = dimension[mask_groups[1][xx]]
#     fluidity_group[label_variables[1][xx]] = fluidity[mask_groups[1][xx]]
#     qq_dimension_group[label_variables[1][xx]] = np.nanpercentile(dimension_group[label_variables[1][xx]], q_range, axis=1)
#     qq_fluidity_group[label_variables[1][xx]] = np.nanpercentile(fluidity_group[label_variables[1][xx]], q_range, axis=1)

#%%

# groups_aux = [ 'wt 2m', 'dKI 2m']
# groups_aux = ['Good 2m', 'Impaired 2m', 'Bad 2m']
groups_aux = ['Good 4m', 'Impaired 4m', 'Bad 4m']
# groups_aux = ['Good 2m', 'Good 4m', 'Impaired 2m', 'Impaired 4m', 'Learners 2m', 'Learners 4m', 'Bad 2m', 'Bad 4m'])


#Plot histograms of qq_dimension and qq_fluidity of the mask_groups
plt.figure(figsize=(10,8))
for xx in range(len(q_range)):
    plt.subplot(3,1,1+xx)   
    plt.hist([qq_dimension_group[label][xx] for label in groups_aux], bins=10,
             histtype='step',
             label=groups_aux,
             density=True)
    plt.title(f'QQ Dimension - {q_range[xx]}th Percentile')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
plt.legend()
plt.show()
#%%
plt.figure(figsize=(10,8))
for xx in range(len(q_range)):
    plt.subplot(3,1,1+xx)   
    plt.hist([qq_fluidity_group[label][xx] for label in groups_aux  ], bins=10,
             histtype='step',
             label=groups_aux,
             density=True)
    plt.title(f'QQ Fluidity - {q_range[xx]}th Percentile')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
plt.legend()
plt.show()
