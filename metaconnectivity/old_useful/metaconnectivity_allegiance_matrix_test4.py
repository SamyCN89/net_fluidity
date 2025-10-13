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
from fun_metaconnectivity import compute_metaconnectivity, allegiance_matrix_analysis, intramodule_indices_mask, get_fc_mc_indices, get_mc_region_identities, compute_trimers_identity, build_trimer_mask
from fun_utils import split_groups_by_age


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
# Set figure parameters globally
plt.rcParams.update({'axes.labelsize': 15, 
                     'axes.titlesize': 13,
                     # 'axes.spines.left': False, 'axes.spines.bottom': False,
                     'axes.spines.right': False, 
                     'axes.spines.top': False,
                     })
#Save options
save_fig =True
# save_data = False
bins_parameter=200

# =================== Paths and folders =======================================
external_disk = True
if external_disk==True:
    root = Path('/media/samy/Elements1/Proyectos/LauraHarsan/script_mc/')
else:    
    root = Path('/home/samy/Bureau/Proyect/LauraHarsan/Ines/')

folders = {'2mois': 'TC_2months', '4mois': 'TC_4months'}

path_results = root / 'results'
path_figures = root / 'fig'

path_sorted = path_results / 'sorted_data/'
path_mc_mod = path_results / 'mc_mod/'
path_timeseries = path_results / 'Timecourses_updated_03052024'
path_cog_data   = path_timeseries / 'ROIs.xlsx'

path_allegiance = path_figures / 'allegiance'
path_modularity = path_figures / 'modularity'
path_motif = path_figures / 'motif'
# save_filename = path_mc_mod / 'mc_allegiance_ref(runs=%s_gammaval=%s)=%s_lag=%s_windowsize=%s_animals=%s_regions=%s.npz'%(label_ref, n_runs_allegiance, gamma_pt_allegiance, lag, window_size, n_animals, regions)


#%%
# ========================== Load data =========================
#cog data
cog_data_filtered = pd.read_csv(path_sorted / 'cog_data_sorted_2m4m.csv')

#ts data and metadata
data_ts = np.load(path_sorted / 'ts_and_meta_2m4m.npz')
# ts=data_ts['ts']
n_animals = data_ts['n_animals']
total_tp = data_ts['total_tp']
regions = data_ts['regions']
is_2month_old = data_ts['is_2month_old']
anat_labels= data_ts['anat_labels']

#Grouping data
with open(path_results / "grouping_data_oip.pkl", "rb") as f:
    mask_groups, label_variables = pickle.load(f)

#Parameters and indices of variables
n_animals   = int(data_ts['n_animals'])
total_tp    = data_ts['total_tp']
regions     = data_ts['regions']
anat_labels = data_ts['anat_labels']
is_2month_old = data_ts['is_2month_old'] #time groups

#%%
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
#%% Load data metaconnectivity, modularity and trimers
# =============================================================================
# Load data
# =============================================================================
data_mc_mod_filename = path_mc_mod /mc_data_filename 
# data_mc_mod = np.load(os.path.join(path_results, 'mc/mc_allegiance_ref=%s_lag=%s_windowsize=%s_.npz'%(lag, window_size)), allow_pickle=True)
data_mc_mod = np.load(data_mc_mod_filename, allow_pickle=True)
# data_mc_mod = np.load(os.path.join(path_results, 'mc/mc_analysis_data_lag=%s_windowsize=%s_.npz'%(lag, window_size)), allow_pickle=True)

mc_allegiance = data_mc_mod['mc']
mc_ref_allegiance_communities   = data_mc_mod['mc_ref_allegiance_communities']
mc_ref_allegiance_sort          = data_mc_mod['mc_ref_allegiance_sort']

mc_modules_mask                 = data_mc_mod['mc_modules_mask']
mc_nplets_mask                  = data_mc_mod['mc_nplets_mask']
mc_idx                          = data_mc_mod['mc_idx_tril']

mc_val                          = data_mc_mod['mc_val_tril']
fc_reg_idx                      = data_mc_mod['fc_reg_idx']
mc_reg_idx                      = data_mc_mod['mc_reg_idx']
mc_mod_idx                      = data_mc_mod['mc_mod_idx'],
mc_nplets_index                 = data_mc_mod['mc_nplets_idx']

mc_mod_idx                      = np.squeeze(mc_mod_idx)


#%%Figures


# =========================== Labels figures ==================================
label_mclinks = r'Inter-regional links'
label_mc_formula = r'MC$_{[ij, kl]} = CC[FC_{ij}(t), FC_{kl}(t)]$'
label_yhist = 'Probability density'
label_mc_trimers = r'Trimer Meta-strengths $MC(i) = \sum_{jk} MC_{[ij, il]}$'

label_trimer = r'Trimer = MC$_{[ir, jr]}$'


#%%
# =============================================================================
# Plot MC trimer intra/inter per group
# =============================================================================
plt.figure(1, figsize=(15,13))

plt.clf()
# for xx in range(len(np.unique(mc_ref_allegiance_communities))):
for idx, setb in enumerate(mask_groups):
    aux_label= label_variables[idx]
    plt.subplot(3,3, 1+idx)

    plt.title('Trimer Motif')
    
    mask_intramod_i, mask_intramod_j = np.argwhere(mc_nplets_mask>0).T
    plt.hist(([mc_val[mod][:, mc_nplets_index>0].ravel() for mod in setb]), 
             histtype='step', 
             bins=bins_parameter, 
             density=True,
             label=aux_label)
    
    plt.yscale('log')
    plt.ylabel(label_yhist, fontsize=12)
    plt.xlabel(label_trimer, fontsize=12)
    plt.xticks([-0.8,0,0.8], fontsize=13)
    plt.ylim(10e-5,10e0)
    # plt.legend()
    plt.subplot(3,3, 4+idx)

    plt.title('Trimers Intra-module')
    
    mask_intermod_i, mask_intermod_j = np.argwhere(mc_nplets_mask>0).T
    # plt.hist(([(mc_allegiance[ind][:, mask_intermod_i, mask_intermod_j]).ravel() for ind in setb]), 
    # plt.hist(([trimer_mc_values[mod][:,trimer_intramod_id].ravel() for mod in setb]), 
    plt.hist(([mc_val[mod][:, (mc_nplets_index>0) * (mc_mod_idx>0)].ravel() for mod in setb]), 
    # plt.hist(([(masked_values(mc_allegiance[ind], aux_mask)).ravel() for ind in setb]), 
             histtype='step', 
             bins=bins_parameter, 
             density=True,
             label=aux_label)
    # plt.legend()
    plt.yscale('log')
    plt.ylabel(label_yhist, fontsize=12)
    plt.xlabel(label_trimer, fontsize=12)
    plt.xticks([-0.8,0,0.8], fontsize=13)
    plt.ylim(10e-5,10e0)

    plt.subplot(3,3, 7+idx)

    plt.title('Trimers inter-module')
    
    # plt.hist(([(masked_values(mc_allegiance[ind], aux_mask)).ravel() for ind in setb]), 
    plt.hist(([mc_val[mod][:, (mc_nplets_index>0) * (mc_mod_idx==0)].ravel() for mod in setb]), 
    # plt.hist(([trimer_mc_values[ind][:,trimer_intermod_id].ravel() for ind in setb]),
             histtype='step', 
             bins=bins_parameter, 
             density=True,
             label=aux_label)
    plt.legend()
    plt.yscale('log')
    plt.ylabel(label_yhist, fontsize=12)
    plt.xlabel(label_trimer, fontsize=12)
    plt.xticks([-0.8,0,0.8], fontsize=13)
    plt.ylim(10e-5,10e0)
    
plt.tight_layout()
if save_fig==True:
    plt.savefig(path_motif / f'hist_trimer_mcvalues_mod(intra_inter).png')

#%%
# =============================================================================
# Plot MC trimer intra/inter per group and per module
# =============================================================================
plt.figure(2, figsize=(15,12))
plt.clf()
subplotindex = (0,5, 10)

aux_numplot_ = len(np.unique(mc_ref_allegiance_communities))
if np.sqrt(aux_numplot_)/np.round(np.sqrt(aux_numplot_))==1:
    aux2=int(np.sqrt(aux_numplot_))
else:
    aux2 = int(np.ceil(np.sqrt(aux_numplot_)))
    
for idx, setb in enumerate(mask_groups):
    for xx in range(len(np.unique(mc_ref_allegiance_communities))):
        aux_label= label_variables[idx]
        
        # plt.subplot(5, aux2, 1+idx+subplotindex[xx])
        print(aux2, 5, 1+xx+subplotindex[idx])
        plt.subplot(aux2, 5, 1+xx + subplotindex[idx])
    
        plt.title('Trimer and Module(%s) %s'%(np.sum(mc_ref_allegiance_communities==xx+1),xx+1))

        # plt.hist(([trimer_mc_values[mod][:,tr].ravel() for mod in setb]), 
        plt.hist(([mc_val[mod][:,mc_mod_idx==xx+1].ravel() for mod in setb]), 
                  histtype='step', 
                  bins = 50, 
                  density=True,
                  label=aux_label)
        
        plt.yscale('log')
        if xx==0:
            plt.ylabel(label_yhist, fontsize=12)
            plt.xlabel(label_trimer, fontsize=12)
            plt.xticks([-0.7,0,0.8], fontsize=13)
            plt.legend()
        else:
            plt.xticks([])
            
        plt.xlim(-.9,.9)
    
plt.tight_layout()
if save_fig==True:
    plt.savefig(path_motif / f'hist_trimer_mcvalues_mod(intra_num).png')
    # plt.savefig(path_figures + 'trimers/hist_trimer_mcvalues_mod(intra_num).png')
    # plt.savefig(path_figures + 'trimers/hist_trimer_mcvalues_mod(intra_num).pdf')

#%%Triemers per module per region
# =============================================================================
# Plot MC trimer intra/inter per group and per module
# Trimers per intramodule per region
# =============================================================================
# subplotindex = (0,3,6,9)
# for xx in range(len(np.unique(mc_ref_allegiance_communities))):
for idx, setb in enumerate(mask_groups):
# setb= mask_groups[0]
# idx=0
    plt.figure(18+idx, figsize=(15,12))
    plt.clf()
    aux_label= label_variables[idx]
    for tr in range(regions):
            plt.subplot(6, 7, 1+tr)
            plt.title(anat_labels[tr],fontsize=10)
            # plt.hist(([trimer_per_region[tr,mod].ravel() for mod in setb]), 
            # plt.hist(([mc_val[mod][:, (mc_nplets_index>0) * (mc_mod_idx==0)].ravel()     for mod in setb]), 
            plt.hist(([mc_val[mod][:, (mc_nplets_index>0) * (mc_mod_idx>0) * ((mc_nplets_index-1)==tr)].ravel() for mod in setb]), 
                       histtype='step', 
                       bins=50, 
                       density=True,
                       label=aux_label
                      )
            plt.yscale('log')
            plt.ylim(10e-4,10e0)
            plt.yticks([])
            if tr+1> (5*7):
                plt.xticks([-0.8,0,0.8], fontsize=13)
            else:
                plt.xticks([])
    plt.legend()
    plt.subplot(6,7,1+tr+1)
    plt.title('trimer per region')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if save_fig==True:
        # plt.savefig(path_figures + 'trimers/hist_trimer_intramodule_per_region_%s_%s.png'%(aux_label[0],aux_label[2]))
        plt.savefig(path_motif / f'hist_trimer_intramodule_per_region_{aux_label[0]}_{aux_label[2]}.png')
#%%
# =============================================================================
# Trimers per intramodule per region
# =============================================================================
# subplotindex = (0,3,6,9)
# for xx in range(len(np.unique(mc_ref_allegiance_communities))):
for idx, setb in enumerate(mask_groups):
# setb= mask_groups[0]
# idx=0
    plt.figure(18+idx, figsize=(15,12))
    plt.clf()
    aux_label= label_variables[idx]
    for tr in range(regions):
            plt.subplot(6, 7, 1+tr)
            plt.title(anat_labels[tr],fontsize=10)
            # plt.hist(([trimer_per_region[tr,mod].ravel() for mod in setb]), 
            # plt.hist(([mc_val[mod][:, (mc_nplets_index>0) * (mc_mod_idx==0)].ravel()     for mod in setb]), 
            plt.hist(([mc_val[mod][:, (mc_nplets_index>0) * (mc_mod_idx>0) * (mc_nplets_index==tr+1)].ravel() for mod in setb]), 
                       histtype='step', 
                       bins=50, 
                       density=True,
                       label=aux_label
                      )
            plt.yscale('log')
            plt.ylim(10e-4,10e0)
            plt.yticks([])
            if tr+1> (5*7):
                plt.xticks([-0.8,0,0.8], fontsize=13)
            else:
                plt.xticks([])
    plt.legend()
    plt.subplot(6,7,1+tr+1)
    plt.title('trimer per region')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    # if save_fig==True:
    #     plt.savefig(path_figures + 'trimers/hist_trimer_intramodule_per_region_%s_%s.png'%(aux_label[0],aux_label[2]))
#%%
# =============================================================================
# Trimers per inter-module per region
# =============================================================================
# subplotindex = (0,3,6,9)
# for xx in range(len(np.unique(mc_ref_allegiance_communities))):
for idx, setb in enumerate(mask_groups):
# setb= mask_groups[0]
# idx=0
    plt.figure(21+idx, figsize=(15,12))
    plt.clf()
    aux_label= label_variables[idx]
    for tr in range(regions):
            print(tr)
            plt.subplot(6, 7, 1+tr)
            plt.title(anat_labels[tr],fontsize=10)
            # plt.hist(([trimer_per_region[tr,mod].ravel() for mod in setb]), 
            plt.hist(([mc_val[mod][:, (mc_nplets_index>0) * (mc_mod_idx==0)* (mc_nplets_index==tr+1)].ravel()     for mod in setb]), 
            # plt.hist(([mc_val[mod][:, (mc_nplets_index>0) * (mc_mod_idx>0) * (mc_nplets_index==tr+1)].ravel()     for mod in setb]), 
                       histtype='step', 
                       bins=50, 
                       density=True,
                       label=aux_label
                      )
            plt.yscale('log')
            plt.ylim(10e-4,10e0)
            plt.yticks([])
            if tr+1> (5*7):
                plt.xticks([-0.8,0,0.8], fontsize=13)
            else:
                plt.xticks([])
    plt.legend()
    # plt.subplot(6,7,1+tr+1)
    plt.tight_layout()
    # if save_fig==True:
    #     plt.savefig(path_figures + 'trimers/hist_trimer_intermodule_per_region_%s_%s.png'%(aux_label[0],aux_label[2]))
#%%Aca voy - Hay que terminar de eliminar mc_mod and trimer_per_region
# =============================================================================
# Trimers per inter-module per region
# =============================================================================
for idx, setb in enumerate(mask_groups):
# idx=0
# setb=mask_groups[1]
    plt.figure(24+idx, figsize=(5,5))
    plt.clf()
    aux_label= label_variables[idx]
    #     aux_label= label_variables[idx]
    for tr in range(regions):
            plt.subplot(6, 7, 1+tr)
            plt.title(anat_labels[tr],fontsize=10)
            # plt.hist(([trimer_2m4m[tr][:,mod[:62]].ravel() for mod in setb]), 
            # plt.hist((trimer_2m4m[tr].ravel()), 
            
            aux_values2m = np.array(
                        [mc_val[mod * is_2month_old][:, 
                                 (mc_nplets_index==tr+1) * 
                                 (mc_nplets_index>0) * 
                                 (mc_mod_idx>0)
                         ].ravel() 
                         for mod in setb
                         ], object).ravel()
            aux_values4m = np.array(
                        [mc_val[mod * ~is_2month_old][:, 
                                 (mc_nplets_index==tr+1) * 
                                 (mc_nplets_index>0) * 
                                 (mc_mod_idx>0)
                         ].ravel() 
                         for mod in setb
                         ], object).ravel()
                
            # plt.hist((np.array(
            #             [mc_val[mod * is_2month_old][:, 
            #                      (mc_nplets_index==tr+1) * 
            #                      (mc_nplets_index>0) * 
            #                      (mc_mod_idx>0)
            #              ].ravel() 
            #              for mod in setb
            #              ], object).ravel()), 
            #           # [mc_val[mod * ~is_2month_old][:, (mc_nplets_index==tr+1) * (mc_nplets_index>0)* (mc_mod_idx>0)].ravel() for mod in setb]), 
            plt.hist((np.concatenate([a for a in aux_values2m if a.size > 0]), np.concatenate([a for a in aux_values4m if a.size > 0])), 
                      # [mc_val[mod * ~is_2month_old][:, (mc_nplets_index==tr+1) * (mc_nplets_index>0)* (mc_mod_idx>0)].ravel() for mod in setb]), 
                      # mc_val[62:, (mc_nplets_index==tr+1)* (mc_nplets_index>0)* (mc_mod_idx>0)].ravel()), 
                      histtype='step', 
                      bins=60, 
                      density=True,
                       label=(('%s/%s 2M'%(aux_label[0], aux_label[2]), '%s/%s 4M'%(aux_label[1], aux_label[3])))
                      )
            plt.yscale('log')
            plt.ylim(10e-4,10e0)
            plt.yticks([])
            if tr+1> (5*7):
                plt.xticks([-0.8,0,0.8], fontsize=13)
            else:
                plt.xticks([])
            # plt.subplot(6,7,1+tr+1)
    plt.legend()
    # plt.title('trimer-intra per region')
    plt.tight_layout()
#%%
#%% LAtest
# =============================================================================
# #%% Latest - Search for the top values
# =============================================================================


# mc_allegiance_10p = np.nanpercentile(mc_allegiance,q=90,axis=(1,2))
# mc_top10_mask = mc_allegiance>mc_allegiance_10p[:,None,None]
mc_allegiance_thr_10pos = np.nanpercentile(mc_allegiance,q=90,axis=(1,2))
mc_top10_mask = mc_allegiance>mc_allegiance_thr_10pos[:,None,None]

mc_allegiance_thr_10neg = np.nanpercentile(mc_allegiance,q=10,axis=(1,2))
mc_bottom10_mask = mc_allegiance<mc_allegiance_thr_10neg[:,None,None]
#%%
mc_top10_idx = mc_top10_mask[:,mc_idx[:,0],mc_idx[:,1]]
mc_bottom10_idx = mc_bottom10_mask[:,mc_idx[:,0],mc_idx[:,1]]


#%%

# to_plot = (mc_val[mc_top10_idx], 
#            mc_val[mc_top10_idx * (mc_mod_idx>0)],
#            mc_val[mc_top10_idx * (mc_mod_idx==0)])

plt.figure(60)
plt.clf()
# plt.hist(mc_val[mc_top10_idx].ravel(), 

plt.subplot(311)

plt.hist((mc_val[mc_top10_idx].ravel(), 
          mc_val[mc_top10_idx *(mc_mod_idx==0)[None,:] ],
          mc_val[mc_top10_idx *(mc_mod_idx>0)[None,:] ],
         
            ), 
         histtype='step', 
          bins=70, 
         density=True,
         label=('top 10', 't10 intermod', 't10 intramod'),
         )
         # histtype='step',bins=50, density=True)
# plt.boxplot(mc_allegiance_10p)
plt.yscale('log')
plt.legend()
plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)
plt.ylabel('Probability density', fontsize=10)

plt.subplot(312)
plt.hist((mc_val[mc_top10_idx].ravel(), 
          mc_val[mc_top10_idx *(mc_nplets_index==0)[None,:] ],
          mc_val[mc_top10_idx *(mc_nplets_index>0)[None,:] ],
         
            ), 
         histtype='step', 
          bins=70, 
         density=True,
         label=('top 10', 't10 4-plets', 't10 3-plets')#, 't10 4-plets intramod', 't10 3-plets intramod'),
         )
# plt.boxplot(mc_allegiance_10p)
plt.yscale('log')
plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)
plt.ylabel('Probability density', fontsize=10)

plt.legend()
plt.subplot(313)
plt.hist((mc_val[mc_top10_idx].ravel(), 
          mc_val[mc_top10_idx *(mc_nplets_index==0)[None,:] *(mc_mod_idx==0)[None,:] ],
          mc_val[mc_top10_idx *(mc_nplets_index>0)[None,:] * (mc_mod_idx==0)[None,:] ],

          mc_val[mc_top10_idx *(mc_nplets_index==0)[None,:] *(mc_mod_idx>0)[None,:] ],
          mc_val[mc_top10_idx *(mc_nplets_index>0)[None,:] * (mc_mod_idx>0)[None,:] ]),

         histtype='step', 
          bins=70, 
         density=True,
         label=('top 10', 't10 4-plets intermod', 't10 3-plets intermod', 't10 4-plets intramod', 't10 3-plets intramod'),
         )
# plt.boxplot(mc_allegiance_10p)
plt.yscale('log')
plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)
plt.ylabel('Probability density', fontsize=10)
plt.ylim(10e-5,10e1)
plt.legend()
# plt.subplot(6,7,1+tr+1)
plt.tight_layout()
if save_fig==True:
    plt.savefig(path_figures + 'trimers/top10_trimer_intermodule_%s_%s.png'%(aux_label[0],aux_label[2]))
#%%
plt.figure(61)
plt.clf()
# plt.hist(mc_val[mc_top10_idx].ravel(), 

plt.subplot(311)

plt.hist((mc_val[mc_bottom10_idx].ravel(), 
          mc_val[mc_bottom10_idx *(mc_mod_idx==0)[None,:] ],
          mc_val[mc_bottom10_idx *(mc_mod_idx>0)[None,:] ],
         
            ), 
         histtype='step', 
          bins=70, 
         density=True,
         label=('top 10', 't10 intermod', 't10 intramod'),
         )
         # histtype='step',bins=50, density=True)
# plt.boxplot(mc_allegiance_10p)
plt.yscale('log')
plt.legend()
plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)
plt.ylabel('Probability density', fontsize=10)

plt.subplot(312)
plt.hist((mc_val[mc_bottom10_idx].ravel(), 
          mc_val[mc_bottom10_idx *(mc_nplets_index==0)[None,:] ],
          mc_val[mc_bottom10_idx *(mc_nplets_index>0)[None,:] ],
         
            ), 
         histtype='step', 
          bins=70, 
         density=True,
         label=('top 10', 't10 4-plets', 't10 3-plets')#, 't10 4-plets intramod', 't10 3-plets intramod'),
         )
# plt.boxplot(mc_allegiance_10p)
plt.yscale('log')
plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)
plt.ylabel('Probability density', fontsize=10)

plt.legend()
plt.subplot(313)
plt.hist((mc_val[mc_bottom10_idx].ravel(), 
          mc_val[mc_bottom10_idx *(mc_nplets_index==0)[None,:] *(mc_mod_idx==0)[None,:] ],
          mc_val[mc_bottom10_idx *(mc_nplets_index>0)[None,:] * (mc_mod_idx==0)[None,:] ],

          mc_val[mc_bottom10_idx *(mc_nplets_index==0)[None,:] *(mc_mod_idx>0)[None,:] ],
          mc_val[mc_bottom10_idx *(mc_nplets_index>0)[None,:] * (mc_mod_idx>0)[None,:] ]),

         histtype='step', 
          bins=70, 
         density=True,
         label=('bottom 10', 'b10 4-plets intermod', 'b10 3-plets intermod', 'b10 4-plets intramod', 'b10 3-plets intramod'),
         )
# plt.boxplot(mc_allegiance_10p)
plt.yscale('log')
plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)
plt.ylabel('Probability density', fontsize=10)
# plt.ylim(10e-4,10e0)
plt.legend()
# plt.subplot(6,7,1+tr+1)
plt.tight_layout()
if save_fig==True:
    plt.savefig(path_figures + 'trimers/bottom10_trimer_intermodule_%s_%s.png'%(aux_label[0],aux_label[2]))
#%%

for idx, setb in enumerate(mask_groups):
# idx=0
# setb = mask_groups[0]
    aux_label= label_variables[idx]
    num_group = int(len(aux_label)/2)

    plt.figure(62+idx, figsize=(13,10))
    plt.clf()

    # plt.hist(mc_val[mc_top10_idx].ravel(), 
    
    plt.subplot(321)
    plt.title('top 10')
    plt.hist((
            [mc_val[mc_top10_idx * mod[:,None]].ravel() for mod in setb]
              # mc_val[mc_top10_idx *(mc_mod_idx==0)[None,:] ],
              # mc_val[mc_top10_idx *(mc_mod_idx>0)[None,:] ])
             # for mod in setb]
                ), 
             histtype='step', 
              bins=70, 
             density=True,
             # label=('top 10', 't10 intermod', 't10 intramod'),
             label=aux_label,
             )
             # histtype='step',bins=50, density=True)
    # plt.boxplot(mc_allegiance_10p)
    plt.yscale('log')
    plt.legend()
    plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)
    plt.ylabel('Probability density', fontsize=10)
    
    plt.subplot(322)
    plt.title('Inter-modularity')
    plt.hist(([mc_val[mc_top10_idx * mod[:,None] * (mc_mod_idx==0)[None,:] ].ravel() for mod in setb]
             
                ), 
             histtype='step', 
              bins=70, 
             density=True,
             # label=('top 10', 't10 4-plets', 't10 3-plets')#, 't10 4-plets intramod', 't10 3-plets intramod'),
             label=aux_label,
             )
    # plt.boxplot(mc_allegiance_10p)
    plt.yscale('log')
    plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)
    plt.ylabel('Probability density', fontsize=10)
    
    plt.legend()

    plt.subplot(323)
    plt.title('Intra-modularity')
    plt.hist(([mc_val[mc_top10_idx * mod[:,None] * (mc_mod_idx>0)[None,:] ].ravel() for mod in setb]
             
                ), 
             histtype='step', 
              bins=70, 
             density=True,
             # label=('top 10', 't10 4-plets', 't10 3-plets')#, 't10 4-plets intramod', 't10 3-plets intramod'),
             label=aux_label,
             )
    # plt.boxplot(mc_allegiance_10p)
    plt.yscale('log')
    plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)
    plt.ylabel('Probability density', fontsize=10)
    
    plt.legend()
    
    plt.subplot(324)
    plt.title('Intra-modularity 4plets')
    plt.hist(([mc_val[mc_top10_idx * mod[:,None] * (mc_mod_idx>0)[None,:] *(mc_nplets_index==0)[None,:] ].ravel() for mod in setb]
             
                ), 
             histtype='step', 
              bins=70, 
             density=True,
             # label=('top 10', 't10 4-plets', 't10 3-plets')#, 't10 4-plets intramod', 't10 3-plets intramod'),
             label=aux_label,
             )
    # plt.boxplot(mc_allegiance_10p)
    plt.yscale('log')
    plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)
    plt.ylabel('Probability density', fontsize=10)
    
    plt.legend()
    plt.subplot(325)
    plt.title('Intra-modularity 3plets')
    plt.hist(([mc_val[mc_top10_idx * mod[:,None] * (mc_mod_idx>0)[None,:] *(mc_nplets_index>0)[None,:] ].ravel() for mod in setb]
             
                ), 
             histtype='step', 
              bins=70, 
             density=True,
             # label=('top 10', 't10 4-plets', 't10 3-plets')#, 't10 4-plets intramod', 't10 3-plets intramod'),
             label=aux_label,
             )
    # plt.boxplot(mc_allegiance_10p)
    plt.yscale('log')
    plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)
    plt.ylabel('Probability density', fontsize=10)

    plt.subplot(326)
    plt.title('3plets')
    plt.hist(([mc_val[mc_top10_idx * mod[:,None] * (mc_nplets_index>0)[None,:] ].ravel() for mod in setb]
             
                ), 
             histtype='step', 
              bins=70, 
             density=True,
             # label=('top 10', 't10 4-plets', 't10 3-plets')#, 't10 4-plets intramod', 't10 3-plets intramod'),
             label=aux_label,
             )
    # plt.boxplot(mc_allegiance_10p)
    plt.yscale('log')
    plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)
    plt.ylabel('Probability density', fontsize=10)
    
    plt.legend()
    
    # plt.subplot(313)
    # plt.hist((mc_val[mc_top10_idx].ravel(), 
    #           mc_val[mc_top10_idx *(mc_nplets_index==0)[None,:] *(mc_mod_idx==0)[None,:] ],
    #           mc_val[mc_top10_idx *(mc_nplets_index>0)[None,:] * (mc_mod_idx==0)[None,:] ],
    
    #           mc_val[mc_top10_idx *(mc_nplets_index==0)[None,:] *(mc_mod_idx>0)[None,:] ],
    #           mc_val[mc_top10_idx *(mc_nplets_index>0)[None,:] * (mc_mod_idx>0)[None,:] ]),
    
    #          histtype='step', 
    #           bins=70, 
    #          density=True,
    #          label=aux_label,
    #          # label=('top 10', 't10 4-plets intermod', 't10 3-plets intermod', 't10 4-plets intramod', 't10 3-plets intramod'),
    #          )
    # # plt.boxplot(mc_allegiance_10p)
    # plt.yscale('log')
    # plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)
    # plt.ylabel('Probability density', fontsize=10)
    # plt.ylim(10e-4,10e0)
    # plt.legend()
    # plt.subplot(6,7,1+tr+1)
    plt.tight_layout()
    if save_fig==True:
        plt.savefig(path_figures + 'trimers/top10_trimer_intermodule_%s_%s.png'%(aux_label[0],aux_label[2]))
#%%
for idx, setb in enumerate(mask_groups):
# idx=0
# setb = mask_groups[0]
    aux_label= label_variables[idx]
    num_group = int(len(aux_label)/2)

    plt.figure(66+idx, figsize=(13,10))
    plt.clf()

    # plt.hist(mc_val[mc_top10_idx].ravel(), 
    
    plt.subplot(321)
    plt.title('bottom 10')
    plt.hist((
            [mc_val[mc_bottom10_idx* mod[:,None]].ravel() for mod in setb]
              # mc_val[mc_bottom10_idx*(mc_mod_idx==0)[None,:] ],
              # mc_val[mc_bottom10_idx*(mc_mod_idx>0)[None,:] ])
             # for mod in setb]
                ), 
             histtype='step', 
              bins=70, 
             density=True,
             # label=('top 10', 't10 intermod', 't10 intramod'),
             label=aux_label,
             )
             # histtype='step',bins=50, density=True)
    # plt.boxplot(mc_allegiance_10p)
    plt.yscale('log')
    plt.legend()
    plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)
    plt.ylabel('Probability density', fontsize=10)
    
    plt.subplot(322)
    plt.title('Inter-modularity')
    plt.hist(([mc_val[mc_bottom10_idx* mod[:,None] * (mc_mod_idx==0)[None,:] ].ravel() for mod in setb]
             
                ), 
             histtype='step', 
              bins=70, 
             density=True,
             # label=('top 10', 't10 4-plets', 't10 3-plets')#, 't10 4-plets intramod', 't10 3-plets intramod'),
             label=aux_label,
             )
    # plt.boxplot(mc_allegiance_10p)
    plt.yscale('log')
    plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)
    plt.ylabel('Probability density', fontsize=10)
    
    plt.legend()

    plt.subplot(323)
    plt.title('Intra-modularity')
    plt.hist(([mc_val[mc_bottom10_idx* mod[:,None] * (mc_mod_idx>0)[None,:] ].ravel() for mod in setb]
             
                ), 
             histtype='step', 
              bins=70, 
             density=True,
             # label=('top 10', 't10 4-plets', 't10 3-plets')#, 't10 4-plets intramod', 't10 3-plets intramod'),
             label=aux_label,
             )
    # plt.boxplot(mc_allegiance_10p)
    plt.yscale('log')
    plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)
    plt.ylabel('Probability density', fontsize=10)
    
    plt.legend()
    
    plt.subplot(324)
    plt.title('Intra-modularity 4plets')
    plt.hist(([mc_val[mc_bottom10_idx* mod[:,None] * (mc_mod_idx>0)[None,:] *(mc_nplets_index==0)[None,:] ].ravel() for mod in setb]
             
                ), 
             histtype='step', 
              bins=70, 
             density=True,
             # label=('top 10', 't10 4-plets', 't10 3-plets')#, 't10 4-plets intramod', 't10 3-plets intramod'),
             label=aux_label,
             )
    # plt.boxplot(mc_allegiance_10p)
    plt.yscale('log')
    plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)
    plt.ylabel('Probability density', fontsize=10)
    
    plt.legend()
    plt.subplot(325)
    plt.title('Intra-modularity 3plets')
    plt.hist(([mc_val[mc_bottom10_idx* mod[:,None] * (mc_mod_idx>0)[None,:] *(mc_nplets_index>0)[None,:] ].ravel() for mod in setb]
             
                ), 
             histtype='step', 
              bins=70, 
             density=True,
             # label=('top 10', 't10 4-plets', 't10 3-plets')#, 't10 4-plets intramod', 't10 3-plets intramod'),
             label=aux_label,
             )
    # plt.boxplot(mc_allegiance_10p)
    plt.yscale('log')
    plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)
    plt.ylabel('Probability density', fontsize=10)

    plt.subplot(326)
    plt.title('3plets')
    plt.hist(([mc_val[mc_bottom10_idx* mod[:,None] * (mc_nplets_index>0)[None,:] ].ravel() for mod in setb]
             
                ), 
             histtype='step', 
              bins=70, 
             density=True,
             # label=('top 10', 't10 4-plets', 't10 3-plets')#, 't10 4-plets intramod', 't10 3-plets intramod'),
             label=aux_label,
             )
    # plt.boxplot(mc_allegiance_10p)
    plt.yscale('log')
    plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)
    plt.ylabel('Probability density', fontsize=10)
    
    plt.legend()
    
    # plt.subplot(313)
    # plt.hist((mc_val[mc_top10_idx].ravel(), 
    #           mc_val[mc_bottom10_idx*(mc_nplets_index==0)[None,:] *(mc_mod_idx==0)[None,:] ],
    #           mc_val[mc_bottom10_idx*(mc_nplets_index>0)[None,:] * (mc_mod_idx==0)[None,:] ],
    
    #           mc_val[mc_bottom10_idx*(mc_nplets_index==0)[None,:] *(mc_mod_idx>0)[None,:] ],
    #           mc_val[mc_bottom10_idx*(mc_nplets_index>0)[None,:] * (mc_mod_idx>0)[None,:] ]),
    
    #          histtype='step', 
    #           bins=70, 
    #          density=True,
    #          label=aux_label,
    #          # label=('top 10', 't10 4-plets intermod', 't10 3-plets intermod', 't10 4-plets intramod', 't10 3-plets intramod'),
    #          )
    # # plt.boxplot(mc_allegiance_10p)
    # plt.yscale('log')
    # plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)
    # plt.ylabel('Probability density', fontsize=10)
    # plt.ylim(10e-4,10e0)
    # plt.legend()
    # plt.subplot(6,7,1+tr+1)
    plt.tight_layout()
    if save_fig==True:
        plt.savefig(path_figures + 'trimers/top10_trimer_intermodule_%s_%s.png'%(aux_label[0],aux_label[2]))
#%%

from statsmodels.distributions.empirical_distribution import ECDF
# --- Define ECDF Function ---
def fun_ecdf(data,side='right'):
    ecdf = ECDF(data)
    x = np.sort(data)
    if side == 'right':
        y = 1- ecdf(x)
    elif side =='left':
        y = ecdf(x)
    return x, y


#%% ecdf top

for idx, setb in enumerate(mask_groups):
# idx=0
# setb = mask_groups[0]
    aux_label= label_variables[idx]
    num_group = int(len(aux_label)/2)

    plt.figure(76+idx, figsize=(13,10))
    plt.clf()

    # --- Plot ---

    plt.subplot(331)
    plt.title('Top 10')
    aux_x = [fun_ecdf(mc_val[mc_top10_idx * mod[:, None]].ravel()) for mod in setb]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label='%s'%aux_label[i])
    plt.ylabel(' ECDF', fontsize=10)
    plt.yscale('log')
    plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)
    aux_x=None
    # plt.legend()

    plt.subplot(332)
    plt.title('Inter-modularity')
    aux_x = [fun_ecdf(mc_val[mc_top10_idx * mod[:, None] * (mc_mod_idx==0)[None,:] ].ravel()) for mod in setb]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label='%s'%aux_label[i])
    plt.ylabel(' ECDF', fontsize=10)
    plt.yscale('log')
    plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)
    

    plt.subplot(333)
    plt.title('Intra-modularity')
    aux_x = [fun_ecdf(mc_val[mc_top10_idx * mod[:, None] * (mc_mod_idx>0)[None,:] ].ravel()) for mod in setb]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label='%s'%aux_label[i])
    plt.ylabel(' ECDF', fontsize=10)
    plt.yscale('log')
    plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)
    
    plt.subplot(334)
    plt.title('Tetramers')
    aux_x = [fun_ecdf(mc_val[mc_top10_idx * mod[:, None] * (mc_nplets_index==0)[None,:]].ravel()) for mod in setb]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label='%s'%aux_label[i])
    plt.ylabel(' ECDF', fontsize=10)
    plt.yscale('log')
    plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)

    plt.subplot(335)
    plt.title('Inter-modularity tetramers')
    aux_x = [fun_ecdf(mc_val[mc_top10_idx * mod[:, None] * (mc_mod_idx>0)[None,:] * (mc_nplets_index==0)[None,:]].ravel()) for mod in setb]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label='%s'%aux_label[i])
    plt.ylabel(' ECDF', fontsize=10)
    plt.yscale('log')
    plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)

    plt.subplot(336)
    plt.title('Intra-modularity tetramers')
    aux_x = [fun_ecdf(mc_val[mc_top10_idx * mod[:, None] * (mc_mod_idx==0)[None,:] * (mc_nplets_index==0)[None,:]].ravel()) for mod in setb]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label='%s'%aux_label[i])
    plt.ylabel(' ECDF', fontsize=10)
    plt.yscale('log')
    plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)

    plt.subplot(337)
    plt.title('Trimers')
    aux_x = [fun_ecdf(mc_val[mc_top10_idx * mod[:, None] * (mc_nplets_index>0)[None,:]].ravel()) for mod in setb]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label='%s'%aux_label[i])
    plt.ylabel(' ECDF', fontsize=10)
    plt.yscale('log')
    plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)

    plt.subplot(338)
    plt.title('Inter-modularity Trimers')
    aux_x = [fun_ecdf(mc_val[mc_top10_idx * mod[:, None] * (mc_mod_idx==0)[None,:] * (mc_nplets_index>0)[None,:]].ravel()) for mod in setb]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label='%s'%aux_label[i])
    plt.ylabel(' ECDF', fontsize=10)
    plt.yscale('log')
    plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)

    plt.subplot(339)
    plt.title('Intra-modularity Trimers')
    aux_x = [fun_ecdf(mc_val[mc_top10_idx * mod[:, None] * (mc_mod_idx>0)[None,:] * (mc_nplets_index>0)[None,:]].ravel()) for mod in setb]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label='%s'%aux_label[i])
    plt.ylabel(' ECDF', fontsize=10)
    plt.yscale('log')
    plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)

    
    plt.legend()
    plt.tight_layout()
    # if save_fig==True:
    #     plt.savefig(path_figures + 'trimers/ecdf_top10_trimer_intermodule_%s_%s.png'%(aux_label[0],aux_label[2]))


#%% ecdf bottom


for idx, setb in enumerate(mask_groups):
# idx=0
# setb = mask_groups[0]
    aux_label= label_variables[idx]
    num_group = int(len(aux_label)/2)

    plt.figure(86+idx, figsize=(13,10))
    plt.clf()

    # --- Plot ---

    plt.subplot(331)
    plt.title('bottom 10')
    aux_x = [fun_ecdf(mc_val[mc_bottom10_idx * mod[:, None]].ravel(),side='left') for mod in setb]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label='%s'%aux_label[i])
    plt.ylabel(' ECDF', fontsize=10)
    plt.yscale('log')
    plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)

    plt.subplot(332)
    plt.title('Inter-modularity')
    aux_x = [fun_ecdf(mc_val[mc_bottom10_idx * mod[:, None] * (mc_mod_idx==0)[None,:] ].ravel(),side='left') for mod in setb]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label='%s'%aux_label[i])
    plt.ylabel(' ECDF', fontsize=10)
    plt.yscale('log')
    plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)
    

    plt.subplot(333)
    plt.title('Intra-modularity')
    aux_x = [fun_ecdf(mc_val[mc_bottom10_idx * mod[:, None] * (mc_mod_idx>0)[None,:] ].ravel(),side='left') for mod in setb]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label='%s'%aux_label[i])
    plt.ylabel(' ECDF', fontsize=10)
    plt.yscale('log')
    plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)
    
    plt.subplot(334)
    plt.title('Tetramers')
    aux_x = [fun_ecdf(mc_val[mc_bottom10_idx * mod[:, None] * (mc_nplets_index==0)[None,:]].ravel(),side='left') for mod in setb]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label='%s'%aux_label[i])
    plt.ylabel(' ECDF', fontsize=10)
    plt.yscale('log')
    plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)

    plt.subplot(335)
    plt.title('Inter-modularity tetramers')
    aux_x = [fun_ecdf(mc_val[mc_bottom10_idx * mod[:, None] * (mc_mod_idx>0)[None,:] * (mc_nplets_index==0)[None,:]].ravel(),side='left') for mod in setb]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label='%s'%aux_label[i])
    plt.ylabel(' ECDF', fontsize=10)
    plt.yscale('log')
    plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)

    plt.subplot(336)
    plt.title('Intra-modularity tetramers')
    aux_x = [fun_ecdf(mc_val[mc_bottom10_idx * mod[:, None] * (mc_mod_idx==0)[None,:] * (mc_nplets_index==0)[None,:]].ravel(),side='left') for mod in setb]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label='%s'%aux_label[i])
    plt.ylabel(' ECDF', fontsize=10)
    plt.yscale('log')
    plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)

    plt.subplot(337)
    plt.title('Trimers')
    aux_x = [fun_ecdf(mc_val[mc_bottom10_idx * mod[:, None] * (mc_nplets_index>0)[None,:]].ravel(),side='left') for mod in setb]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label='%s'%aux_label[i])
    plt.ylabel(' ECDF', fontsize=10)
    plt.yscale('log')
    plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)

    plt.subplot(338)
    plt.title('Inter-modularity Trimers')
    aux_x = [fun_ecdf(mc_val[mc_bottom10_idx * mod[:, None] * (mc_mod_idx==0)[None,:] * (mc_nplets_index>0)[None,:]].ravel(),side='left') for mod in setb]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label='%s'%aux_label[i])
    plt.ylabel(' ECDF', fontsize=10)
    plt.yscale('log')
    plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)

    plt.subplot(339)
    plt.title('Intra-modularity Trimers')
    aux_x = [fun_ecdf(mc_val[mc_bottom10_idx * mod[:, None] * (mc_mod_idx>0)[None,:] * (mc_nplets_index>0)[None,:]].ravel(),side='left') for mod in setb]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label='%s'%aux_label[i])
    plt.ylabel(' ECDF', fontsize=10)
    plt.yscale('log')
    plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)

    
    plt.legend()
    plt.tight_layout()
    # if save_fig==True:
    #     plt.savefig(path_figures + 'trimers/ecdf_bottom10_trimer_intermodule_%s_%s.png'%(aux_label[0],aux_label[2]))

#%%
import seaborn as sns


# original, subset = mc_val[mc_top10_idx].ravel(), mc_val[mc_top10_idx*(mc_mod_idx>0)[None,:]].ravel()
original, subset = mc_val[mc_top10_idx].ravel(), mc_val[mc_top10_idx*(mc_nplets_index>0)[None,:]].ravel()

plt.figure(91)
plt.clf()
sns.boxplot(data=[original, subset])
plt.xticks([0, 1], ['Original', 'Subset'])
plt.title('Boxplot')

plt.tight_layout()
plt.show()

from scipy.stats import ks_2samp, ttest_ind

ks_stat, ks_p = ks_2samp(original, subset)
print(f"KS test: statistic={ks_stat:.3f}, p-value={ks_p:.3f}")

# # Cohen's d
# cohen_d = (original.mean() - subset.mean()) / ((original.std()**2 + subset.std()**2)/2)**0.5
# print(f"Cohen's d: {cohen_d:.3f}")
# #%%

# import scipy.stats as stats

# # Make sure both datasets are sorted and of the same length
# # If not, you can interpolate to match sizes
# min_len = min(len(original), len(subset))
# orig_sorted = np.sort(original)
# sub_sorted = np.sort(subset)

# # Interpolate if needed
# if len(original) > len(subset):
#     orig_interp = np.percentile(orig_sorted, np.linspace(0, 100, min_len))
#     sub_interp = sub_sorted
# elif len(subset) > len(original):
#     orig_interp = orig_sorted
#     sub_interp = np.percentile(sub_sorted, np.linspace(0, 100, min_len))
# else:
#     orig_interp = orig_sorted
#     sub_interp = sub_sorted

# # Q-Q Plot
# plt.figure(8890,figsize=(6,6))
# # plt.clf()
# plt.plot(orig_interp, sub_interp, 'o')
# min_val = min(orig_interp.min(), sub_interp.min())
# max_val = max(orig_interp.max(), sub_interp.max())
# plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
# plt.xlabel('Original Quantiles')
# plt.ylabel('Subset Quantiles')
# plt.title('Q-Q Plot: Original vs Subset (Raw Values)')
# plt.legend()
# plt.grid(True)
# plt.show()

#%%

_mask = mask_groups[0]
tr=1
# test = np.array([mc_val[mod][:, (mc_nplets_index>0) * (mc_mod_idx>0) * (mc_nplets_index==tr+1)].ravel() for mod in setb] )
# test = mc_val[_mask[0]][:, (mc_nplets_index>0) * (mc_mod_idx>0) * (mc_nplets_index==tr+1)]
test = mc_val[_mask[0]][:, (mc_nplets_index>0) * (mc_mod_idx>0) * (mc_reg_idx==tr+1)]
print(np.shape(test))
#%%
                 # for idx, setb in enumerate(mask_groups):
setb= mask_groups[1]
idx=1
plt.figure(180+idx, figsize=(15,12))
plt.clf()
aux_label= label_variables[idx]
for tr in range(regions):
        plt.subplot(6, 7, 1+tr)
        plt.title(anat_labels[tr],fontsize=10)
        # plt.hist(([trimer_per_region[tr,mod].ravel() for mod in setb]), 
        # plt.hist(([mc_val[mod][:, (mc_nplets_index>0) * (mc_mod_idx==0)].ravel()     for mod in setb]), 
        plt.bar(aux_label,
                np.array([np.nanpercentile(mc_val[mod][:, (mc_nplets_index>0) * (mc_mod_idx>0) * (mc_nplets_index==tr+1)].ravel(),90) for mod in setb]).ravel(), 
                # np.array([np.nanpercentile(mc_val[mod][:, (mc_mod_idx>0) * (mc_nplets_index==tr+1)].ravel(),90) for mod in setb]).ravel()).flatten(), 
                  )
        # plt.yscale('log')
        # plt.ylim(10e-4,10e0)
        plt.yticks([])
        if tr+1> (5*7):
            plt.xticks([-0.8,0,0.8], fontsize=13)
        else:
            plt.xticks([])
# plt.legend()
plt.subplot(6,7,1+tr+1)
plt.title('trimer per region')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
    
    
#%%
    # aux_label= label_variables[idx]
    # num_group = int(len(aux_label)/2)

    # plt.figure(76+idx, figsize=(13,10))
    # plt.clf()

    # # --- Plot ---

    # plt.subplot(331)
    # plt.title('Top 10')
    # aux_x = [fun_ecdf(mc_val[mc_top10_idx * mod[:, None]].ravel()) for mod in setb]
    # for i, (x, y) in enumerate(aux_x):
    #     plt.plot(x, y, label='%s'%aux_label[i])
    # plt.ylabel(' ECDF', fontsize=10)
    # plt.yscale('log')
    # plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)
    # aux_x=None
    # # plt.legend()

    # plt.subplot(332)
    # plt.title('Inter-modularity')
    # aux_x = [fun_ecdf(mc_val[mc_top10_idx * mod[:, None] * (mc_mod_idx==0)[None,:] ].ravel()) for mod in setb]
    # for i, (x, y) in enumerate(aux_x):
    #     plt.plot(x, y, label='%s'%aux_label[i])
    # plt.ylabel(' ECDF', fontsize=10)
    # plt.yscale('log')
    # plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)
    

    # plt.subplot(333)
    # plt.title('Intra-modularity')
    # aux_x = [fun_ecdf(mc_val[mc_top10_idx * mod[:, None] * (mc_mod_idx>0)[None,:] ].ravel()) for mod in setb]
    # for i, (x, y) in enumerate(aux_x):
    #     plt.plot(x, y, label='%s'%aux_label[i])
    # plt.ylabel(' ECDF', fontsize=10)
    # plt.yscale('log')
    # plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)
    
    # plt.subplot(334)
    # plt.title('Tetramers')
    # aux_x = [fun_ecdf(mc_val[mc_top10_idx * mod[:, None] * (mc_nplets_index==0)[None,:]].ravel()) for mod in setb]
    # for i, (x, y) in enumerate(aux_x):
    #     plt.plot(x, y, label='%s'%aux_label[i])
    # plt.ylabel(' ECDF', fontsize=10)
    # plt.yscale('log')
    # plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)

    # plt.subplot(335)
    # plt.title('Inter-modularity tetramers')
    # aux_x = [fun_ecdf(mc_val[mc_top10_idx * mod[:, None] * (mc_mod_idx>0)[None,:] * (mc_nplets_index==0)[None,:]].ravel()) for mod in setb]
    # for i, (x, y) in enumerate(aux_x):
    #     plt.plot(x, y, label='%s'%aux_label[i])
    # plt.ylabel(' ECDF', fontsize=10)
    # plt.yscale('log')
    # plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)

    # plt.subplot(336)
    # plt.title('Intra-modularity tetramers')
    # aux_x = [fun_ecdf(mc_val[mc_top10_idx * mod[:, None] * (mc_mod_idx==0)[None,:] * (mc_nplets_index==0)[None,:]].ravel()) for mod in setb]
    # for i, (x, y) in enumerate(aux_x):
    #     plt.plot(x, y, label='%s'%aux_label[i])
    # plt.ylabel(' ECDF', fontsize=10)
    # plt.yscale('log')
    # plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)

    # plt.subplot(337)
    # plt.title('Trimers')
    # aux_x = [fun_ecdf(mc_val[mc_top10_idx * mod[:, None] * (mc_nplets_index>0)[None,:]].ravel()) for mod in setb]
    # for i, (x, y) in enumerate(aux_x):
    #     plt.plot(x, y, label='%s'%aux_label[i])
    # plt.ylabel(' ECDF', fontsize=10)
    # plt.yscale('log')
    # plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)

    # plt.subplot(338)
    # plt.title('Inter-modularity Trimers')
    # aux_x = [fun_ecdf(mc_val[mc_top10_idx * mod[:, None] * (mc_mod_idx==0)[None,:] * (mc_nplets_index>0)[None,:]].ravel()) for mod in setb]
    # for i, (x, y) in enumerate(aux_x):
    #     plt.plot(x, y, label='%s'%aux_label[i])
    # plt.ylabel(' ECDF', fontsize=10)
    # plt.yscale('log')
    # plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)

    # plt.subplot(339)
    # plt.title('Intra-modularity Trimers')
    # aux_x = [fun_ecdf(mc_val[mc_top10_idx * mod[:, None] * (mc_mod_idx>0)[None,:] * (mc_nplets_index>0)[None,:]].ravel()) for mod in setb]
    # for i, (x, y) in enumerate(aux_x):
    #     plt.plot(x, y, label='%s'%aux_label[i])
    # plt.ylabel(' ECDF', fontsize=10)
    # plt.yscale('log')
    # plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)

    
    # plt.legend()
    # plt.tight_layout()
# =============================================================================
# Index of regions, modularity and nplets
# =============================================================================
#mc indices tril
# fc_indx = np.array(np.tril_indices(regions,k=-1)).T
# mc_idx = np.array(np.tril_indices(fc_indx.shape[0], k=-1)).T
# aux_indentity_region_mc =fc_indx[mc_idx]
# mc_reg_idx = np.transpose(np.reshape(aux_indentity_region_mc, 
#                           (aux_indentity_region_mc.shape[0], 4))
#                           ,(1,0)) #identity of the regios in the MC_i

# #mc values tril
# mc_idx = np.transpose(mc_idx,(1,0))
# mc_val  = mc_allegiance[:,mc_idx[0],mc_idx[1]]

# #mc modules indices and mask
# mc_mod_idx = (mc_modules_mask[mc_idx[0], mc_idx[1]])
# mc_mod_idx = mc_mod_idx.astype(int)

## %% Trimers

# =============================================================================intramodules_idx
# trimer - values and identity
# trimer_index, trimer_reg, trimer_reg_apex, trimer_values
# =============================================================================