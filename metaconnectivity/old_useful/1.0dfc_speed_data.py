#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 14:42:38 2023

@author: samy
"""

import numpy as np
import matplotlib.pyplot as plt
import brainconn as bct
import os
import time
import pandas as pd
# from functions_analysis import *
from scipy.io import loadmat, savemat
from scipy.special import erfc
from scipy.stats import pearsonr, spearmanr

from fun_loaddata import *
from fun_dfcspeed import *

#%%Figure's parameters
# =============================================================================
# Figure's parameters
# =============================================================================

# Set figure parameters globally
plt.rcParams.update({'axes.labelsize': 15, 
                     'axes.titlesize': 13,
                     #'xtick.labelsize' : 13,
                     # 'ytick.labelsize' : 13,
                     # 'axes.spines.left': False, 
                     #'axes.spines.bottom': False,
                     'axes.spines.right': False, 
                     'axes.spines.top': False})

save_fig    = False
save_data   = False
# plt.style.use('seaborn-white')

#%% Define paths, folders and hash
root = '/home/samy/Bureau/Proyect/LauraHarsan/Ines/Timecourses_updated_03052024/'
# folders = {'2mois': 'Lot3_2mois', '4mois': 'Lot3_4mois'}
folders = {'2mois': 'TC_2months', '4mois': 'TC_4months'}
folder_results = '/home/samy/Bureau/Proyect/LauraHarsan/Ines/results/'

#%% Parameters speed
window_parameter = (5,100,1)
lag=1
tau=3
tau_array       = np.append(np.arange(0,tau), tau ) 
lentau          = len(tau_array)
#%% Load data - Intersect the functional data for 2 and 4 months
# =============================================================================
# Load fMRI data - Intersect the functional data for 2 and 4 months
# =============================================================================
#hash data
hash_parameters = ('lag=%s_tau=%s_wmax=%s_wmin=%s'%(lag,tau,window_parameter[1],window_parameter[0]))

# Load filenames and hash numbers
filenames       = {period: filename_sort_mat(os.path.join(root, folder)) for period, folder in folders.items()}
hash_numbers    = {period: extract_hash_numbers(filenames[period]) for period in filenames}

int_2m4m = np.intersect1d(hash_numbers['2mois'], hash_numbers['4mois'], return_indices=True)
print('Number of intersected elements in 2m and 4m :' , len(int_2m4m[0]))
#%% Load cognitive data 
# =============================================================================
# Load cognitive data from .xlsx document
# =============================================================================
#Load cognitive data
#Old data
# cog_data_path   = os.path.join(root, 'Behaviour_exclusions_ROIs_female.xlsx')
# cog_data_path   = os.path.join(root, 'Behaviour_exclusions_ROIs.xlsx')
# cog_data_df     = pd.read_excel(cog_data_path, sheet_name='Feuil1')
# data_roi        = pd.read_excel(cog_data_path, sheet_name='40_Allen_ROIs_list').to_numpy()
#Update 03/05/2024
cog_data_path   = os.path.join(root, 'ROIs.xlsx')
cog_data_df     = pd.read_excel(cog_data_path, sheet_name='Exclusions')
data_roi        = pd.read_excel(cog_data_path, sheet_name='41_Allen').to_numpy()

# Anatomical labels
anat_labels     = np.array([xx[1] for xx in data_roi])

cog_data_df['sexe_label']  = cog_data_df.loc[:,'Sexe']
cog_data_df['gen_label']  = cog_data_df.loc[:,'Genotype']

cog_data_df['oip_4m-2m']  = cog_data_df.loc[:,'OiP_4M']-cog_data_df.loc[:,'OiP_2M']
cog_data_df['oip_4m+2m']  = cog_data_df.loc[:,'OiP_4M']+cog_data_df.loc[:,'OiP_2M']

cog_data_df['ro24h_4m-2m']  = cog_data_df.loc[:,'RO24h_4M']-cog_data_df.loc[:,'RO24h_2M']
cog_data_df['ro24h_4m+2m']  = cog_data_df.loc[:,'RO24h_4M']+cog_data_df.loc[:,'RO24h_2M']


# Filtering based on sex (Male/Female), genotype (wt/dKI) and TC (ok/Excluded)
# Convert sex and genotype to numerical values for easier processing
cog_data_df['Sexe']     = cog_data_df['Sexe'].map({'M': 0, 'F': 1})
cog_data_df['Genotype'] = cog_data_df['Genotype'].map({'wt': 0, 'dKI': 1})
cog_data_df['TC_2M']    = cog_data_df['TC_2M'].map({'ok': 0, 'Excluded': 1})
cog_data_df['TC_4M']    = cog_data_df['TC_4M'].map({'ok': 0, 'Excluded': 1})

# Filter cognitive data for animals with the intersected functional data
cog_data_filtered       = cog_data_df[cog_data_df['Name'].isin(int_2m4m[0])].sort_values(by='Name')

#Remove the TC 'excluded' from the data
cog_data_filtered       = cog_data_filtered[(cog_data_filtered['TC_2M'] == 0) & (cog_data_filtered['TC_4M'] == 0)]

# Generating boolean indices for various filters
mouse_hash_cog      = cog_data_filtered['Name'].to_numpy()
male_index          = cog_data_filtered['Sexe'] == 0
female_index        = cog_data_filtered['Sexe'] == 1
wt_index          = cog_data_filtered['Genotype'] == 0
dki_index           = cog_data_filtered['Genotype'] == 1

# Further filtering based on specific criteria, e.g., males with wt genotype
male_wt_data    = cog_data_filtered[(cog_data_filtered['Sexe'] == 0) & (cog_data_filtered['Genotype'] == 0)]
male_dki_data   = cog_data_filtered[(cog_data_filtered['Sexe'] == 0) & (cog_data_filtered['Genotype'] == 1)]
female_wt_data    = cog_data_filtered[(cog_data_filtered['Sexe'] == 1) & (cog_data_filtered['Genotype'] == 0)]
female_dki_data   = cog_data_filtered[(cog_data_filtered['Sexe'] == 1) & (cog_data_filtered['Genotype'] == 1)]

wt_data = cog_data_filtered[(cog_data_filtered['Genotype'] == 0)]
dki_data = cog_data_filtered[(cog_data_filtered['Genotype'] == 1)]

# Generate a label list
sex_label       = cog_data_filtered['sexe_label'].to_numpy()
gen_label       = cog_data_filtered['gen_label'].to_numpy()
#%% Extracting the intersection of functional and cognitive data

#Extracting the intersection of functional and cognitive data
inter_cogfun    = np.intersect1d(int_2m4m[0], mouse_hash_cog, return_indices=True)
print('Number of intersected cognitive and functional elements :' , len(inter_cogfun[0]))

#Generating sorted index of functional data (2m and 4m) 
index_tsintcog  = np.array(int_2m4m)[1:,inter_cogfun[1]] #intersection of 2m,4m and coginfo

#Extracting the file name of functional time series that are intersected
filename_int2m = filenames['2mois'][index_tsintcog[0]]
filename_int4m = filenames['4mois'][index_tsintcog[1]]
        
#Loading the time series of the intersected data
ts2m = load_matdata(root, folders['2mois'], filename_int2m)
ts4m = load_matdata(root, folders['4mois'], filename_int4m)

#Remove the first transient of data
transient=50
ts2m = ts2m[:,transient:]
ts4m = ts4m[:,transient:]

# 

# plt.figure(911)
# plt.clf()
# for i in range(63):
#     plt.subplot(7,9,i+1)
#     plt.imshow(dfc_stream_2m[i], interpolation='none',aspect='auto',cmap='coolwarm')
#     plt.clim(-1,1)

#Some important variables
n_animals, total_tp, regions = ts2m.shape
#%%
# =============================================================================
# Plot cognitive data scores
# =============================================================================
aux_data_type = 2

cog_wt_aux = (male_wt_data, female_wt_data, wt_data)[aux_data_type]
cog_dki_aux =(male_dki_data, female_dki_data, dki_data)[aux_data_type]

aux_l1 = ('Male','Female','All')
cog_aux_labels = ['%s WT 2M'%(aux_l1[aux_data_type]), '%s WT 4M'%(aux_l1[aux_data_type]), '%s dKI 2M'%(aux_l1[aux_data_type]), '%s dKI 4M'%(aux_l1[aux_data_type])]

#plot the aux_data_type
plt.figure(1,figsize=(10, 6.5))
plt.clf()

plt.subplot(211)
plt.title('Distribution of OiP scores for %s'%aux_l1[aux_data_type])
plt.violinplot((cog_wt_aux['OiP_2M'], cog_wt_aux['OiP_4M'], cog_dki_aux['OiP_2M'], cog_dki_aux['OiP_4M']))#,cmap='C1')
plt.xticks([1, 2, 3, 4], cog_aux_labels, fontsize=12)
plt.axhline(0,c='k')
plt.axhline(0.2,c='k',ls='--')
plt.axhline(-0.2,c='k',ls='--')
plt.ylabel('OiP score')

plt.subplot(212)
plt.title('Distribution of RO24h for %s'%aux_l1[aux_data_type])
plt.violinplot((cog_wt_aux['RO24h_2M'], cog_wt_aux['RO24h_4M'], cog_dki_aux['RO24h_2M'], cog_dki_aux['RO24h_4M']))
plt.xticks([1, 2, 3, 4], cog_aux_labels, fontsize=12)
plt.axhline(0,c='k')
plt.axhline(0.2,c='k',ls='--')
plt.axhline(-0.2,c='k',ls='--')
plt.ylabel('RO24h score')
plt.tight_layout()
if save_fig==True:
    plt.savefig('fig/cog_data/oip_ro24h_%s_wt_dki.png'%aux_l1[aux_data_type])
    plt.savefig('fig/cog_data/oip_ro24h_%s_wt_dki.pdf'%aux_l1[aux_data_type])
#%%
# =============================================================================
# Save cognitive data
# =============================================================================
#data saved
if save_data==True:
    data_save = {}
    data_save['cog_data'] = cog_data_filtered
    
    data_save['ts2m'] = ts2m
    data_save['ts4m'] = ts4m
    
    data_save['male_index'] = male_index
    data_save['female_index'] = female_index
    data_save['wt_index'] = wt_index
    data_save['dki_index'] = dki_index
    
    data_save['mouse_hash'] = mouse_hash_cog
    data_save['gen_label'] = gen_label
    data_save['sex_label'] = sex_label
    data_save['anat_labels'] = anat_labels
    
    # savemat(folder_results+'speed_lag=%s_tau=%s_wmax=%s_wmin=%s.mat'%(lag,tau,window_parameter[0],window_parameter[1]), data_save)
    savemat(folder_results+'order_data/list_'+ hash_parameters, data_save)

#%% # Compute speed dFC
# =============================================================================
# Speed analysis
# Compute the dfc speed distribution using wondow oversampling method for each animal. Also retrieve median speed for each tau, in multiple W, for each animal
# =============================================================================

time_windows_min, time_windows_max, time_window_step = window_parameter
time_windows_range = np.arange(time_windows_min,time_windows_max+1,time_window_step)

vel_list2m = []
vel_list4m = []
speed_median_2m = []
speed_median_4m = []

start = time.time()
for xx in ts2m:
# for xx in ts2m[:3]:
    aux_speed_median, aux_speed_dist  = parallel_dfc_speed_oversampled_series(xx, window_parameter, lag,tau=tau, min_tau_zero=True, get_speed_dist=True)
    vel_list2m.append(aux_speed_dist)  
    speed_median_2m.append(aux_speed_median)

for xx in ts4m:
# for xx in ts4m[:3]:
    aux_speed_median, aux_speed_dist  = parallel_dfc_speed_oversampled_series(xx, window_parameter, lag,tau=tau, min_tau_zero=True, get_speed_dist=True)
    vel_list4m.append(aux_speed_dist)  
    speed_median_4m.append(aux_speed_median)

speed_median_2m = np.array(speed_median_2m)
speed_median_4m = np.array(speed_median_4m)

stop= time.time()
print('speed dist windows oversampling analysis time', stop-start,'s')

#%% Save speed data
# =============================================================================
# Save speed analysis
# =============================================================================

if save_data==True:
    vel_list2m_asar = np.asarray(vel_list2m,dtype=object)
    vel_list4m_asar = np.asarray(vel_list4m,dtype=object)
    # np.savez(folder_results + 'speed/speed2m4m_dist' + hash_parameters, vel_2m = vel_list2m_asar, vel_4m=vel_list4m_asar)
    np.savez(folder_results + 'speed/speed2m4m_dist_03052024' + hash_parameters, 
             vel_2m     = vel_list2m_asar, 
             vel_4m     = vel_list4m_asar)
    
    #Check the load dataset
    load_vel = np.load(folder_results + 'speed/speed2m4m_dist' + hash_parameters+'.npz', allow_pickle=True)
    
    vel_2m = load_vel['vel_2m']
    vel_4m = load_vel['vel_4m']

