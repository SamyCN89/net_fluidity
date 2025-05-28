#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:26:30 2024

@author: samy
"""
#%%
import numpy as np
import os
import pandas as pd
import pickle
from pathlib import Path

from shared_code.fun_loaddata import extract_hash_numbers
from shared_code.fun_utils import filename_sort_mat, load_matdata, classify_phenotypes, make_combination_masks, make_masks, get_paths
import matplotlib.pyplot as plt
import time

# =============================================================================
# This code computes 
# Load the data
# Intersect the 2 and 4 months to have data that have the two datapoints

# =============================================================================
TRANSIENT = 50 #Transient to remove
THRESHOLD = 0.2 #Threshold for phenotype classification

#%% Define paths, folders and hash
# =============================================================================
# #Define paths, folders and hash
# =============================================================================


# Define the timeseries directory
timeseries_folder = 'Timecourses_updated_03052024'
# Will prioritize PROJECT_DATA_ROOT if set
paths = get_paths(timecourse_folder=timeseries_folder)
folders = {'2mois': 'TC_2months', '4mois': 'TC_4months'}

#%% Load cog data and intersect if there is in 2M and 4M
# =============================================================================
# Load cognitive data from .xlsx document
# =============================================================================
#Load cognitive data
cog_data_df     = pd.read_excel(paths['cog_data'], sheet_name='Exclusions')
data_roi        = pd.read_excel(paths['cog_data'], sheet_name='41_Allen').to_numpy()

# =============================================================================
# Intersect the functional filenames hash for 2 and 4 months
# =============================================================================

# Retrieve filenames and hash numbers
filenames       = {period: filename_sort_mat(os.path.join(paths['timeseries'], folder)) for period, folder in folders.items()}
hash_numbers    = {period: extract_hash_numbers(filenames[period]) for period in filenames}
common_ids, intind_2m, intind_4m = np.intersect1d(hash_numbers['2mois'], hash_numbers['4mois'], return_indices=True)
print('Number of intersected elements in 2m and 4m :' , len(common_ids))

cog_data_df['oip_4m-2m']  = cog_data_df.loc[:,'OiP_4M'] - cog_data_df.loc[:,'OiP_2M']
cog_data_df['oip_4m+2m']  = cog_data_df.loc[:,'OiP_4M'] + cog_data_df.loc[:,'OiP_2M']

cog_data_df['ro24h_4m-2m']  = cog_data_df.loc[:,'RO24h_4M']-cog_data_df.loc[:,'RO24h_2M']
cog_data_df['ro24h_4m+2m']  = cog_data_df.loc[:,'RO24h_4M']+cog_data_df.loc[:,'RO24h_2M']

# Filtering based on sex (Male/Female), genotype (wt/dKI) and TC (ok/Excluded)
cog_data_df['TC_2M']    = cog_data_df['TC_2M'].map({'ok': 0, 'Excluded': 1})
cog_data_df['TC_4M']    = cog_data_df['TC_4M'].map({'ok': 0, 'Excluded': 1})

# ===================== cog_data_filtered =====================================
# # Create a new dataframe filtering cognitive data for animals with the intersected functional data
#Intersection cognitive and time series hash
cog_data_filtered       = cog_data_df[cog_data_df['Name'].isin(common_ids)].sort_values(by='Name').copy()

#Remove the TC 'excluded' from the data
cog_data_filtered       = cog_data_filtered[(cog_data_filtered['TC_2M'] == 0) & (cog_data_filtered['TC_4M'] == 0)].copy()
#%% The pre-procesecisng for time series sorted
# =============================================================================
# The pre-procesecisng for time series sorted
# =============================================================================
# Generating boolean indices for various filters
mouse_hash_cog      = cog_data_filtered['Name'].to_numpy()

#Intersection of functional and cognitive data
inter_cogfun    = np.intersect1d(common_ids, mouse_hash_cog, return_indices=True)
print('Number of intersected cognitive and functional elements :' , len(inter_cogfun[0]))

#Generating sorted index of functional data (2m and 4m) 
# index_tsintcog  = np.array(int_2m4m)[1:,inter_cogfun[1]] #intersection of 2m,4m and coginfo
index_tsintcog  = np.array((intind_2m, intind_4m))[:,inter_cogfun[1]] #intersection of 2m,4m and coginfo

#sorted time sries file name of intersected funcog
common_files_2m = np.array(filenames['2mois'])[index_tsintcog[0]]
common_files_4m = np.array(filenames['4mois'])[index_tsintcog[1]]

#Loading the time series of the intersected data
ts2m = load_matdata(paths['timeseries'], folders['2mois'], common_files_2m	)
ts4m = load_matdata(paths['timeseries'], folders['4mois'], common_files_4m)

#Remove the first transient of data
ts2m = ts2m[:,TRANSIENT:]
ts4m = ts4m[:,TRANSIENT:]

# Preallocate and stack
n2, n4 = ts2m.shape[0], ts4m.shape[0]
total_tp, regions = ts2m.shape[1:]
n_animals = n2 + n4
ts = np.empty((n_animals, total_tp, regions), dtype=ts2m.dtype)
ts[:n2] = ts2m
ts[n2:] = ts4m

#Some important variables
anat_labels     = [xx[1].replace('.', ' ') for xx in data_roi] # Anatomical labels
is_2month_old = np.arange(n_animals) < n2

#%% Genrate list of groups
# ========================== Genrate list of groups ===========================

# =============================================================================
# #Setting of the threshold for phenotype classification
# =============================================================================
#Clasiffy phenotypes using threshold (default = 0.2)
cog_data_filtered = classify_phenotypes(cog_data_filtered, metric_prefix='OiP', threshold=THRESHOLD)
cog_data_filtered = classify_phenotypes(cog_data_filtered, metric_prefix='RO24h', threshold=THRESHOLD)

genotype = cog_data_filtered['Genotype']
sexe = cog_data_filtered['Sexe']
phenotype_oip = cog_data_filtered['Phenotype_OiP']
phenotype_nor = cog_data_filtered['Phenotype_RO24h']

group_phenotype_oip = (phenotype_oip=='good', phenotype_oip=='impaired', phenotype_oip=='learners', phenotype_oip=='bad')
prelab_phenotype_oip = ('Good', 'Impaired', 'Learners', 'Bad')

group_phenotype_nor = (phenotype_nor=='good', phenotype_nor=='impaired', phenotype_nor=='learners', phenotype_nor=='bad')
prelab_phenotype_nor = ('Good', 'Impaired', 'Learners', 'Bad')

group_genotype = (genotype=='wt', genotype=='dKI')
prelab_genotype = ('wt', 'dKI')

group_sex = (sexe=='F', sexe=='M')
prelab_sex = ('Female', 'Male')

group_info = [
    (group_phenotype_oip, prelab_phenotype_oip),
    (group_phenotype_nor, prelab_phenotype_nor),
    (group_genotype, prelab_genotype),
    (group_sex, prelab_sex)
]

mask_groups, label_variables = make_masks(group_info, is_2month_old)
#%% Genrate list of groups per sex

phenotypes_test = ['good', 'impaired', 'learners', 'bad']
sexes_test = ['F', 'M']
mask_combo_oip, label_combo_oip = make_combination_masks(
    cog_data_filtered,
    primary_col='Phenotype_OiP',
    by_col='Sexe',
    primary_levels=phenotypes_test,
    by_levels=sexes_test,
    is_2month_old=is_2month_old
)

phenotypes_test = ['good', 'impaired', 'learners', 'bad']
sexes_test = ['F', 'M']
mask_combo_nor, label_combo_nor = make_combination_masks(
    cog_data_filtered,
    primary_col='Phenotype_RO24h',
    by_col='Sexe',
    primary_levels=phenotypes_test,
    by_levels=sexes_test,
    is_2month_old=is_2month_old
)

genotypes_test = ['wt', 'dKI']
mask_combo_gen, label_combo_gen = make_combination_masks(
    cog_data_filtered,
    primary_col='Genotype',
    by_col='Sexe',
    primary_levels=genotypes_test,
    by_levels=sexes_test,
    is_2month_old=is_2month_old
)

mask_groups_per_sex = (mask_combo_oip, mask_combo_nor, mask_combo_gen)
label_variables_per_sex = (label_combo_oip, label_combo_nor, label_combo_gen)
#%% Save results
# =============================================================================
# Save results
# =============================================================================
#Cognitive data
cog_data_filtered.to_csv(paths['sorted'] / 'cog_data_sorted_2m4m.csv', index=False)

#time series plus metadata
np.savez(paths['sorted'] / 'ts_and_meta_2m4m.npz',
         ts=ts,  
         n_animals=n_animals, 
    total_tp=total_tp, 
    regions=regions, 
    is_2month_old=is_2month_old,
    anat_labels=anat_labels,
    )

#mask and labels for groups
with open(paths['sorted'] / "grouping_data_oip.pkl", "wb") as f:
    pickle.dump((mask_groups, label_variables), f)

with open(paths['sorted'] / "grouping_data_per_sex(gen_phen).pkl", "wb") as f:
    pickle.dump((mask_groups_per_sex, label_variables_per_sex), f)

#%% Load pre-process data
# =============================================================================
# # Load result
# =============================================================================

cog_data_filtered = pd.read_csv(paths['sorted'] / 'cog_data_sorted_2m4m.csv')
data_ts = np.load(paths['sorted'] / 'ts_and_meta_2m4m.npz')

ts=data_ts['ts']
n_animals = data_ts['n_animals']
total_tp = data_ts['total_tp']
regions = data_ts['regions']
is_2month_old = data_ts['is_2month_old']
anat_labels= data_ts['anat_labels']

# %%
