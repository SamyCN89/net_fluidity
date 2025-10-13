#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:26:30 2024

@author: samy
"""
from pathlib import Path
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

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

from itertools import combinations_with_replacement

import copy
import pickle

from fun_loaddata import *
from fun_dfcspeed import *
from fun_metaconnectivity import (compute_metaconnectivity, 
                                  allegiance_matrix_analysis, 
                                  intramodule_indices_mask, 
                                  get_fc_mc_indices, 
                                  get_mc_region_identities, 
                                  compute_trimers_identity, 
                                  build_trimer_mask,
                                  )
from fun_utils import (split_groups_by_age, 
                       get_paths,
                       set_figure_params,
                       )


# =============================================================================
# This code uses
#   cog_data_filtered, data_ts, grouping_data, and data_mc_mod

# Plot
    # PLOT MC using allegiance reference
    # Plot MC for each individual
    # Plot MC intra/inter modular per group
    # Plot MC intra/inter for each individual
    # Plot MC modules for each individual
    # plot MC modules per group
# =============================================================================

#%%Load and plot parameters

# ========================== Figure parameters ================================
#Figure parameters
save_fig =  set_figure_params(True)
bins_parameter=200

# =================== Paths and folders =======================================
external_disk = True
paths = get_paths(external_disk)

#%% Load sorted data
# ========================== Load data =========================
cog_data_filtered = pd.read_csv(paths['sorted'] / 'cog_data_sorted_2m4m.csv')

# ts=data_ts['ts']
data_ts = np.load(paths['sorted'] / 'ts_and_meta_2m4m.npz')
n_animals   = int(data_ts['n_animals'])
total_tp = data_ts['total_tp']
regions = data_ts['regions']
is_2month_old = data_ts['is_2month_old']
anat_labels= data_ts['anat_labels']

results_path = paths['results'] / "grouping_data_oip.pkl"
with results_path.open("rb") as f:
    mask_groups, label_variables = pickle.load(f)
#%% MEtaconnectivity computing
# =============================================================================
# Metaconnectivity
# =============================================================================
#Parameters speed

PROCESSORS =-1

lag=1
tau=5
window_size = 7
window_parameter = (5,100,1)

#Parameters allegiance analysis
n_runs_allegiance = 1000
gamma_pt_allegiance = 100
ind_ref = mask_groups[0][0] # the mask of the reference matrix
# label_ref = label_variables[0][0] #The label of the reference matrix
label_ref = 'Good2m' #The label of the reference matrix
# label_ref = 'wt2M_recurrecy' #The label of the reference matrix

tau_array       = np.append(np.arange(0,tau), tau ) 
lentau          = len(tau_array)

time_window_min, time_window_max, time_window_step = window_parameter
time_window_range = np.arange(time_window_min,time_window_max+1,time_window_step)

mc_data_filename = f"mc_allegiance_ref(runs={label_ref}_gammaval={n_runs_allegiance})={gamma_pt_allegiance}_lag={lag}_windowsize={window_size}_animals={n_animals}_regions={regions}.npz".replace(' ','')
#%% Load data metaconnectivity, and modularity
# =============================================================================
# Load data
# =============================================================================
data_mc_mod_filename = paths['mc_mod'] / mc_data_filename 
data_mc_mod = np.load(data_mc_mod_filename, allow_pickle=True)

mc_allegiance = data_mc_mod['mc']
mc_ref_allegiance_communities           = data_mc_mod['mc_ref_allegiance_communities']
# mc_ref_allegiance_sort   = data_mc_mod['mc_ref_allegiance_sort']

# mc_modules_mask                 = data_mc_mod['mc_modules_mask']
# mc_idx = data_mc_mod['mc_idx_tril']

mc_val = data_mc_mod['mc_val_tril']
# mc_reg_idx             = data_mc_mod['mc_reg_idx']
mc_mod_idx             = data_mc_mod['mc_mod_idx'],
mc_mod_idx = np.squeeze(mc_mod_idx)

#%%Figures
# =========================== Labels figures ==================================
label_mclinks = r'Inter-regional links'
label_mc_formula = r'MC$_{[ij, kl]} = CC[FC_{ij}(t), FC_{kl}(t)]$'
label_yhist = 'Probability density'
#%%Metaconnectivity
# =============================================================================
# MC allegiance template (using now template as good ones)

# =============================================================================
for idx, setb in enumerate(mask_groups):
# idx=0
# setb = mask_groups[0]
    aux_label= label_variables[idx]
    num_group = int(len(aux_label)/2)

    plt.figure(1+idx, figsize=(13,10))
    plt.clf()
    for xx in range(num_group*2):
        plt.subplot(num_group,2,1+xx)
        plt.title('MC ref allegiance sorted %s '%aux_label[xx])
        plt.imshow(np.mean(mc_allegiance[setb[idx]],axis=0), interpolation='none', aspect='auto', cmap = 'coolwarm')
        plt.xticks((25,700), labels=['1 ...', r'... $N^2-N$'], fontsize=12)
        plt.yticks((25, 150, 580, 779), labels=['1',' .\n.\n.', ' .\n.\n.', r'$N^2-N$'], fontsize=12)
        plt.xlabel(label_mclinks, labelpad=-3, fontsize=11)
        plt.ylabel(label_mclinks, labelpad=-27, fontsize=11)
    
        cbar = plt.colorbar()
        cbar.set_label(label_mc_formula, rotation=270, labelpad=25, fontsize=11)  # <- your colorbar label
        plt.clim((-0.1, 0.1))
        cbar.set_ticks([-0.1,0,0.1])
        cbar.ax.tick_params(labelsize=15)
    plt.tight_layout()
    if save_fig==True:
        plt.savefig(paths['allegiance'] / f'Allegiance_consensus_{aux_label}.png')
#%%

# =============================================================================
# MC for each individual
# =============================================================================

for idx, setb in enumerate(mask_groups):
    aux_label= label_variables[idx]
    num_group = int(len(aux_label)/2)
    for ii, ind in enumerate(setb):
        # ind= mask_groups_good_impaired[0]
        plt.figure(4+ii+(4*idx), figsize=(13,10))
        plt.clf()
    
        aux_numplot_ = mc_allegiance[ind].shape[0]
        if np.sqrt(aux_numplot_)/np.round(np.sqrt(aux_numplot_))==1:
            aux2=int(np.sqrt(aux_numplot_))
        else:
            aux2 = int(np.ceil(np.sqrt(aux_numplot_)))
        # print(aux_numplot_, aux2)
        plt.title('Consensus module MC %s '%aux_label[ii])
        for xx in range(aux_numplot_):
            plt.subplot(aux2,aux2,1+xx)
            plt.imshow(mc_allegiance[xx], interpolation='none', aspect='auto', cmap = 'coolwarm')
            plt.xticks([])
            plt.yticks([])
            plt.clim((-0.25, 0.25))
        plt.tight_layout()
        if save_fig==True:
            plt.savefig(paths['allegiance'] / f'Allegiance_consensus_{aux_label[ii]}.png')

#%%Modularity
# =============================================================================
# MC intra/inter modular per group
# =============================================================================
plt.figure(8, figsize=(13,9.5))
plt.clf()
plt.subplot(3,3,1)
plt.hist(( (mc_val[:,mc_mod_idx>0]).ravel(), (mc_val[:,mc_mod_idx==0]).ravel()), 
         bins=bins_parameter, 
         density=True,
         histtype='step',
         label=('Intra-module', 'Inter-module')
         )
plt.yscale('log')
plt.xlabel(label_mc_formula)
plt.ylabel(label_yhist, fontsize=10)
plt.xticks([-0.7,0,0.7], fontsize=13)
plt.ylim(10e-6,10e0)
plt.legend()

for idx, setb in enumerate(mask_groups):
    plt.subplot(3,3,2+(idx*3))
    plt.title('Intra-module')
    plt.hist([mc_val[xx][: ,mc_mod_idx>0].ravel() for xx in setb], 
              bins=bins_parameter, 
             density=True,
             histtype='step',
             label=label_variables[idx])
    plt.xticks([-0.7,0,0.7], fontsize=13)
    plt.ylim(10e-6,10e0)
    plt.yscale('log')
    # plt.legend()

    plt.subplot(3,3,3+(idx*3))
    plt.title('Inter-module')
    plt.hist([mc_val[xx][: ,mc_mod_idx==0].ravel() for xx in setb], 
              bins=bins_parameter, 
             density=True,
              histtype='step',
              label=label_variables[idx])
    plt.xticks([-0.7,0,0.7], fontsize=13)
    plt.ylim(10e-6,10e0)
    plt.yscale('log')
    plt.legend()
plt.tight_layout()
if save_fig==True:
    plt.savefig(paths['fmodularity'] / f'Intra_intermodularity_mc_values.png')
#%%
# =============================================================================
# Plot MC intra/inter values of each individual
# =============================================================================


plt.figure(9)
plt.clf()

plt.subplot(211)
plt.title('Individual Intra-module values')
plt.hist([mc_val[ind,mc_mod_idx>0].ravel() for ind in range(n_animals)],
         bins=bins_parameter, 
         density=True,
         histtype='step',
         alpha=0.2,
         color=np.full(n_animals, 'Gray')
        )
plt.hist((mc_val[:,mc_mod_idx>0].ravel()), 
         bins=bins_parameter, 
         density=True,
         histtype='step',
         linewidth=1.5,
         color='k'
        )
plt.yscale('log')
plt.ylabel(label_yhist)
plt.xticks([-0.7,0,0.7], fontsize=13)

plt.subplot(212)
plt.title('Individual Inter-module values')
plt.hist([mc_val[ind,mc_mod_idx==0].ravel() for ind in range(n_animals)], 
         bins=bins_parameter, 
         density=True,
         histtype='step',
         alpha=0.2,
         color=np.full(n_animals, 'Gray')
        )

plt.hist((mc_val[:,mc_mod_idx==0].ravel()), 
         bins=bins_parameter, 
         density=True,
         histtype='step',
         linewidth=1.5,
         color='k'
        )
plt.xticks([-0.7,0,0.7], fontsize=13)
plt.yscale('log')
plt.ylabel(label_mc_formula)

# plt.legend()
plt.xlabel(label_mc_formula)
plt.ylabel(label_yhist)
plt.tight_layout()
if save_fig==True:
    # plt.savefig(paths['figures'] + 'modularity/Allmice_Intra_intermodularity_mc_values.png')
    plt.savefig(paths['fmodularity'] / f'Allmice_Intra_intermodularity_mc_values.png')

#%%
# =============================================================================
# Plot MC modules values of each individual
# =============================================================================



plt.figure(10)
plt.clf()

aux_numplot_ = len(np.unique(mc_ref_allegiance_communities))
if np.sqrt(aux_numplot_)/np.round(np.sqrt(aux_numplot_))==1:
    aux2=int(np.sqrt(aux_numplot_))
else:
    aux2 = int(np.ceil(np.sqrt(aux_numplot_)))
# print(aux_numplot_, aux2)

for xx in range(len(np.unique(mc_ref_allegiance_communities))):
    # plt.subplot(3,3, 1+xx)
    plt.subplot(aux2,2,1+xx)
    plt.title('Module %s'%(xx+1))
    plt.hist(mc_val[:, mc_mod_idx==xx+1].T, 
             bins=bins_parameter, 
             density=True,
             histtype='step',
             alpha=0.2,
              # color=np.full(n_animals, 'Gray')
            )
    plt.yscale('log')
    
    plt.hist((mc_val[:, mc_mod_idx==xx+1]).ravel(), 
             bins=bins_parameter, 
             density=True,
             histtype='step',
             linewidth=1.1,
             color='k'
            )
    plt.yscale('log')
    plt.xticks([-0.7,0,0.7], fontsize=13)
    plt.ylabel(label_yhist)
    plt.xlabel(label_mc_formula)
plt.tight_layout()
if save_fig==True:
    plt.savefig(paths['fmodularity'] / f'Allmice_Intramodularity_mc_values.png')
    # plt.savefig(paths['figures'] + 'modularity/Allmice_Intramodularity_alone_mc_values.png')
    # plt.savefig(paths['figures'] + 'modularity/Allmice_Intramodularity_alone_mc_values.png')

#%%
# =============================================================================
# plot MC modules values for each group
# =============================================================================

for idx, setb in enumerate(mask_groups):
    aux_label= label_variables[idx]

    plt.figure(11+idx, figsize=(13,8))
    plt.clf()
    for xx in np.unique(mc_ref_allegiance_communities):
        plt.subplot(aux2,2,xx)
        # plt.subplot(2,2, xx)
        plt.title('Module %s'%(xx))
        plt.hist(([mc_val[ind][:, mc_mod_idx==xx].ravel() for ind in setb]),
             bins=70, 
             density=True,
             histtype='step',
             label=aux_label
             
             # color=np.full(124, 'Gray')
            )
        plt.yscale('log')
        plt.ylabel(label_yhist, fontsize=12)
        plt.xlabel(label_mc_formula, fontsize=12)
        plt.xticks([-0.7,0,0.7], fontsize=13)
    plt.legend()
    plt.tight_layout()
    if save_fig==True:
        plt.savefig(paths['fmodularity'] / f'Intramodularity_mc_values_{aux_label[0]}_{aux_label[2]}.png')
        # plt.savefig(paths['figures'] + 'modularity/Intramodularity_mc_values_%s_%s.png'%(aux_label[0], aux_label[2]))

#%%

