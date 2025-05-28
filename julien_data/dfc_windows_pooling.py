#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 14:42:38 2023

@author: samy
"""
#%% Import libraries
from cProfile import label
from pathlib import Path
from tkinter import W
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import brainconn as bct
import os
import time
import pandas as pd
# import sys
# sys.path.append('../shared_code')
# from functions_analysis import *
from scipy.io import loadmat, savemat
from scipy.special import erfc
from scipy.stats import pearsonr, spearmanr

from shared_code.fun_loaddata import *
from shared_code.fun_dfcspeed import pool_vel_windows, get_population_wpooling

# from fun_utils import set_figure_params
from shared_code.fun_bootstrap import handler_bootstrap_permutation
from shared_code.fun_utils import get_paths, set_figure_params

from joblib import Parallel, delayed


#%% Figure parameters 
# =============================================================================
# Figure's parameters
# =============================================================================


save_fig = set_figure_params(savefig=True)
PROCESSORS = -1 # Use all available processors
#%%
# ------------------------Configuration------------------------
# Set the path to the root directory of your dataset
USE_EXTERNAL_DISK = True
ROOT = Path('/media/samy/Elements1/Proyectos/LauraHarsan/dataset/julien/') if USE_EXTERNAL_DISK \
        else Path('/home/samy/Bureau/Proyect/LauraHarsan/dataset/julien/')

PREPROCESS_DATA = ROOT / 'preprocess_data'
RESULTS_DIR = ROOT / Path('results')
SPEED_DIR = RESULTS_DIR / 'speed'

FIG_DIR = ROOT / 'fig'
SPEED_FIG_DIR = FIG_DIR / 'speed'
SPEED_FIG_DIR.mkdir(parents=True, exist_ok=True)   

# Parameters speed
WINDOW_PARAM = (5,100,1)
LAG=1
TAU=5

HASH_TAG = f"lag={LAG}_tau={TAU}_wmax={WINDOW_PARAM[1]}_wmin={WINDOW_PARAM[0]}"

#%%
paths = get_paths(dataset_name='julien_caillette', 
                  timecourse_folder='time_courses',
                  cognitive_data_file='mice_groups_comp_index.xlsx')

#%%
# USE_EXTERNAL_DISK = True
# ROOT = Path('/media/samy/Elements1/Proyectos/LauraHarsan/dataset/julien_caillette/') if USE_EXTERNAL_DISK \
#         else Path('/home/samy/Bureau/Proyect/LauraHarsan/dataset/julien_caillette/')
# RESULTS_DIR = ROOT / Path('results')
paths['speed'] = paths['results'] / 'speed'
# paths['speed'].mkdir(parents=True, exist_ok=True)

TS_FILE = paths['sorted'] / Path("ts_filtered_unstacked.npz")
COG_FILE = paths['sorted'] / Path("cog_data_filtered.csv")

SAVE_DATA = True
#OLD REMOVE

#%%
# ------------------------ Load Data ------------------------

#Here we load the preprocessed cognitive and time- series data

cog_data = pd.read_csv(paths['sorted'] / "cog_data_filtered.csv")
data = np.load(paths['sorted'] / "ts_filtered_unstacked.npz", allow_pickle=True)
ts_filtered = data["ts"]

vel_data = np.load(paths['speed'] / f'speed_dfc_{HASH_TAG}.npz', allow_pickle=True)
vel = vel_data['vel']
speed_median = vel_data['speed_median']

paths = get_paths(dataset_name='julien_caillette', 
                  timecourse_folder='time_courses',
                  cognitive_data_file='mice_groups_comp_index.xlsx')

#%%
# USE_EXTERNAL_DISK = True
# ROOT = Path('/media/samy/Elements1/Proyectos/LauraHarsan/dataset/julien_caillette/') if USE_EXTERNAL_DISK \
#         else Path('/home/samy/Bureau/Proyect/LauraHarsan/dataset/julien_caillette/')
# RESULTS_DIR = ROOT / Path('results')
paths['speed'] = paths['results'] / 'speed'
# paths['speed'].mkdir(parents=True, exist_ok=True)

TS_FILE = paths['sorted'] / Path("ts_filtered_unstacked.npz")
COG_FILE = paths['sorted'] / Path("cog_data_filtered.csv")

SAVE_DATA = True


# ------------------------ Preprocessing ------------------------

wt_index          = cog_data['WT']
mut_index          = cog_data['Dp1Yey']

veh_index        = cog_data['VEH']
treat_index        = cog_data['LCTB92']

tau_array           = np.append(np.arange(0,TAU), TAU) 
lentau              = len(tau_array)

short_mid = 13
mid_long = 35

# short_mid = 13
# mid_long = 40
limits = (short_mid, mid_long)  # match your original: short/mid/long split

# #Some important variables
n_animals = len(ts_filtered)
regions = ts_filtered[0].shape[1]
#%%

time_windows_range = np.arange(WINDOW_PARAM[0],WINDOW_PARAM[1]+1,WINDOW_PARAM[2])
aux_timewr = time_windows_range*2

vel_label = ('%s-%ss (short)'%(aux_timewr[0], aux_timewr[limits[0]]),
             '%s-%ss (mid)'%(aux_timewr[limits[0]], aux_timewr[limits[1]]),
             '%s-%ss (long)'%(aux_timewr[limits[1]], aux_timewr[-1]))
# n_animals = int(np.sum(male_index))
#%%
pooled_vel = pool_vel_windows(vel, lentau, limits, strategy="drop")
# pooled_vel = pool_vel_windows(vel, lentau, limits, strategy="pad")

# Remove NaNs inside each pooled array — Case 1 (per-animal, per-window cleanup)
aux_short2m_list = [arr[~np.isnan(arr)] for arr in pooled_vel["short"]]
aux_mid2m_list   = [arr[~np.isnan(arr)] for arr in pooled_vel["mid"]]
aux_long2m_list  = [arr[~np.isnan(arr)] for arr in pooled_vel["long"]]

# Replace wp_list with cleaned lists
wp_list = (aux_short2m_list, aux_mid2m_list, aux_long2m_list)

#%%
#For the dfc speed distribution window oversampling, get a windows pooling


#Genotyped based
wp_wt = get_population_wpooling(wp_list, wt_index)
wp_wt_veh = get_population_wpooling(wp_list, wt_index & veh_index)
wp_wt_treat = get_population_wpooling(wp_list, wt_index & treat_index)

wp_mut = get_population_wpooling(wp_list, mut_index)
wp_mut_veh = get_population_wpooling(wp_list, mut_index & veh_index)
wp_mut_treat = get_population_wpooling(wp_list, mut_index & treat_index) 

# =============================================================================

#%%
# Compute the normalization histogram of the speed for each animal

def compute_normalized_histograms(*all_speeds, bins=250, range_=(0, 1.2)):
    """
    Compute normalized histograms for multiple speed data arrays, normalized to the global max.

    Args:
        *all_speeds: Variable number of 1D arrays (e.g., WT, Mut, etc.)
        bins: Number of histogram bins.
        range_: Tuple specifying the histogram range.

    Returns:
        bin_centers, list_of_normalized_counts
    """
    # Compute raw histograms for each group
    counts_list = []
    bin_edges = None
    for speeds in all_speeds:
        counts, bin_edges = np.histogram(speeds, bins=bins, range=range_, density=False)
        counts_list.append(counts)
    # Normalize by global max
    global_max = max([counts.max() for counts in counts_list])
    norm_counts_list = [counts / global_max for counts in counts_list]
    # norm_counts_list = [counts / counts.max() for counts in counts_list]
    # Compute bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, norm_counts_list


# =============================================================================
# Plot windows pool
# =============================================================================
# def plot_wpool(wp_var1, wp_var2, name_data = 'all'):
#     plt.figure(1, figsize=(12,10))
#     plt.clf()
#     # vel_label = ('10-30s (short)','30-72s (mid)','72-160s (long)')
    
#     # wp_var1 = wp_wt
#     # wp_var2 = wp_mut
#     for i in range(3):
#         #Normalize the histograms
#         bin_centers, norm_counts_list = compute_normalized_histograms(
#                 np.array(wp_var1[i]), 
#                 np.array(wp_var2[i])
#             )
#         # Plot the Normalized histogram
#         plt.subplot(3,2,2*i+1)
#         if i==0:
#             plt.title('%s %s'%(vel_label[i],name_data))
#             plt.ylabel('Rel Freq (to global max)')
#         else:
#             plt.title('%s'%vel_label[i])
#         plt.plot(bin_centers, norm_counts_list[0], label="WT", linestyle='-', alpha=0.9)
#         plt.plot(bin_centers, norm_counts_list[1], label="Mut", linestyle='-', alpha=0.9)
#         # plt.hist((wp_var1[i], wp_var2[i]),label=('wt', 'mut'), histtype='step',bins=150, density=True)
        
#         plt.xlim(0,1.2)

#         plt.subplot(3,2,2*i+2)
#         if i==0:
#             plt.title('%s %s'%(vel_label[i],name_data))
#         else:
#             plt.title('%s'%vel_label[i])
#         # plt.hist((wp_var1[i], wp_var2[i]),label=('wt', 'mut'), histtype='step',bins=150, density=True)
#         plt.plot(bin_centers, norm_counts_list[0], label="WT", linestyle='-', alpha=0.9)
#         plt.plot(bin_centers, norm_counts_list[1], label="Mut", linestyle='-', alpha=0.9)
#         # plt.hist((wp_var1[i], wp_var2[i]),label=('wt', 'mut'), histtype='step',bins=150, density=True, log=True)
#         plt.xlim(0,1.2)
#         plt.ylim(10e-5, 10e-1)
#         plt.yscale('log')
#     # plt.xlabel('Freq[v]')
#     plt.xlabel("dFC Speed")
#     plt.legend()
#     plt.tight_layout()
    
#     if save_fig ==True:
#         plt.savefig(SPEED_FIG_DIR / f'hist_speed_{HASH_TAG}_{name_data}.png')
#         plt.savefig(SPEED_FIG_DIR / f'hist_speed_{HASH_TAG}_{name_data}.pdf')
#     plt.show()
def plot_wpool(*wp_vars, labels=None, name_data='all'):
    """
    Plot normalized histograms of pooled dFC speed windows for multiple groups.

    Args:
        *wp_vars: Variable number of tuples/lists with three pooled arrays (short, mid, long) per group.
        labels: List of labels for each group (e.g., ['WT', 'Mut', 'Treat']).
        name_data: String used for figure title and saving.
    """
    if labels is None:
        labels = [f'Group {i+1}' for i in range(len(wp_vars))]

    assert len(wp_vars) == len(labels), "Each data group must have a corresponding label"

    plt.figure(figsize=(14, 10))
    plt.clf()

    for i in range(3):  # short, mid, long
        # Extract and convert each group's pooled data for this window
        window_data = [np.array(wp[i]) for wp in wp_vars]

        # Compute normalized histograms
        bin_centers, norm_counts_list = compute_normalized_histograms(*window_data)

        # Linear scale plot
        plt.subplot(3, 2, 2*i + 1)
        for counts, label in zip(norm_counts_list, labels):
            plt.plot(bin_centers, counts, label=label, alpha=0.9)
        plt.title(f'{vel_label[i]} {name_data}' if i == 0 else vel_label[i])
        if i == 0:
            plt.ylabel('Rel Freq (to global max)')
        plt.xlim(0, 1.2)

        # Log scale plot
        plt.subplot(3, 2, 2*i + 2)
        for counts, label in zip(norm_counts_list, labels):
            plt.plot(bin_centers, counts, label=label, alpha=0.9)
        plt.title(f'{vel_label[i]} {name_data}' if i == 0 else vel_label[i])
        plt.xlim(0, 1.2)
        # plt.ylim(1e-4, 1e-1)
        plt.yscale('log')

    plt.xlabel("dFC Speed")
    plt.legend()
    plt.tight_layout()

    if save_fig:
        plt.savefig(SPEED_FIG_DIR / f'hist_speed_{HASH_TAG}_{name_data}.png')
        plt.savefig(SPEED_FIG_DIR / f'hist_speed_{HASH_TAG}_{name_data}.pdf')

    plt.show()

plot_wpool(wp_wt, wp_mut, labels=['WT', 'Dp1Yey'], name_data='wt_LCTB92')
plot_wpool(wp_wt_veh, wp_mut_veh, labels=['WT VEH', 'Dp1Yey VEH'], name_data='wt_LCTB92_veh')
plot_wpool(wp_wt_treat, wp_mut_treat, labels=['WT Treat', 'Dp1Yey Treat'], name_data='wt_LCTB92_treat')
plot_wpool(wp_wt_treat, wp_wt_veh, wp_mut_treat, wp_mut_veh, labels=['WT Treat', 'WT VEH', 'Dp1Yey Treat', 'Dp1Yey VEH'], name_data='multi_group')


# plot_wpool(wp_wt, wp_mut, name_data = 'wt_mut')
# plot_wpool(wp_wt_veh, wp_mut_veh, name_data = 'wt_mut_veh')
# plot_wpool(wp_wt_treat, wp_mut_treat, name_data = 'wt_mut_treat')
# plot_wpool(wp_wt_female, wp_mut_female, name_data = 'female')
# plot_wpool(wp_wt_male, wp_mut_male, name_data = 'male')
#%%
# =============================================================================
# Cumsum
# =============================================================================

def compute_cumulative_distribution(wp_list):
    """
    Compute cumulative distribution function (CDF) for each pooled speed array.

    Args:
        wp_list: List of pooled speed arrays for different groups.

    Returns:
        cdf_list: List of tuples (sorted_speeds, cdf_values) for each group.
    """
    cdf_list = []
    for speeds in wp_list:
        sorted_speeds = np.sort(speeds)
        cdf_values = np.arange(1, len(sorted_speeds) + 1) / len(sorted_speeds)
        cdf_list.append((sorted_speeds, cdf_values))
    return cdf_list

cdf_wt = compute_cumulative_distribution(wp_wt)
cdf_wt_treat = compute_cumulative_distribution(wp_wt_treat)
cdf_wt_veh = compute_cumulative_distribution(wp_wt_veh)

cdf_mut = compute_cumulative_distribution(wp_mut)
cdf_mut_treat = compute_cumulative_distribution(wp_mut_treat)
cdf_mut_veh = compute_cumulative_distribution(wp_mut_veh)


#%%

def cdf_plot(*wp_vars, labels=None):
    
    if labels is None:
        labels = [f'Group {i+1}' for i in range(len(wp_vars))]

    assert len(wp_vars) == len(labels), "Each data group must have a corresponding label"
    
    
    plt.figure(11, figsize=(8, 8))
    plt.clf()

    for i in range(3):  # short, mid, long
        plt.subplot(3, 1, i + 1)
        plt.title(f'Cumulative Distribution Function ({vel_label[i]})')
        for cdf, label in zip(wp_vars, labels):
            plt.plot(cdf[i][0], cdf[i][1], label=label)
        # plt.yscale('log')
    plt.ylabel('Cumulative probability')
    plt.xlabel('dFC Speed')
    plt.legend()
    plt.tight_layout()
    plt.show()

cdf_plot(cdf_wt_treat, cdf_wt_veh, cdf_mut_treat, cdf_mut_veh, labels=['WT Treat', 'WT VEH', 'Dp1Yey Treat', 'Dp1Yey VEH'])
# Plot the cumulative distribution function (CDF) for both groups
# plt.plot(cdf_wt[0], 1 - cdf_wt[1], label='wt')
# plt.plot(cdf_mut[0], 1 - cdf_mut[1], label='mut')
# plt.xlabel('dFC Speed')
# plt.ylabel('1 - Cumulative probability')
# plt.legend()
# # plt.xscale('log')
# plt.yscale('log')
#%%

#%%# =============================================================================
# Save windows pooling data
# =============================================================================

# load_wpool = np.load(folder_results + 'speed/windowspooling_' + hash_parameters+'.npz', allow_pickle=True)

#%%
# =============================================================================
# Compute velocity statistics
# =============================================================================


bin_centers, norm_counts_list = compute_normalized_histograms(
    np.concatenate(wp_wt), 
    np.concatenate(wp_mut)
)
#%%
# Plot
plt.figure(figsize=(10, 6))
plt.plot(bin_centers, norm_counts_list[0], label="WT", linestyle='-', alpha=0.9)
plt.plot(bin_centers, norm_counts_list[1], label="Mut", linestyle='-', alpha=0.9)
plt.xlabel("dFC Speed")
plt.ylabel("Relative Frequency (normalized to global max)")
plt.title("Histogram of dFC Speeds (WT vs Mut, shared normalization)")
plt.legend()
plt.yscale('log')
# plt.grid(True)
plt.tight_layout()
plt.show()


#%%
# Save the data in optimal format .npz 

np.savez(SPEED_DIR / f'windows_pooling_{HASH_TAG}.npz',
         wpool_wt       = wp_wt, 
         wpool_mut      = wp_mut, 
         )


#%%



# =============================================================================
# Compute and plot quantile range
# =============================================================================

def quantiles_across_groups(group_list, q_range):
    """
    Compute quantile ranges across multiple groups and window pools.

    Parameters:
    - group_list: array-like of shape (n_groups, n_wpools), each entry is a 1D array of speed values
    - q_range: array-like of quantiles to compute (e.g., np.linspace(0.01, 0.99, 99))

    Returns:
    - np.ndarray of shape (n_groups, len(q_range), n_wpools)
    """
    group_list = np.asarray(group_list, dtype=object)
    n_groups, n_wpools = group_list.shape
    n_quantiles = len(q_range)

    # Allocate output: shape (n_groups, n_wpools, n_quantiles)
    output = np.empty((n_groups, n_wpools, n_quantiles))

    for g in range(n_groups):
        for w in range(n_wpools):
            output[g, w, :] = np.quantile(group_list[g, w], q_range)

    return output

 #%%
 #distribution difference
def qq_diff(qq_data):
    qq_data = np.transpose(qq_data, (1, 0, 2))  # Shape: (n_groups, n_wpools, n_quantiles) 
    num_group = qq_data.shape[0] # Number of groups
    qq_diff =[] # Initialize list to store slopes
    for qq_aux in qq_data:
        diff_qq = np.squeeze(np.diff(qq_aux, axis=0))
        # print(diff_qq.shape)
        qq_diff.append(diff_qq)
    return qq_diff

# Slope difference
def qq_slope(qq_data):
    qq_data = np.transpose(qq_data, (1, 0, 2))  # Shape: (n_groups, n_wpools, n_quantiles) 
    num_group = qq_data.shape[0] # Number of groups
    qq_slope =[] # Initialize list to store slopes
    for qq_aux in qq_data:
        
        slope = np.diff(qq_aux, axis=1)
        diff_slope = np.squeeze(np.diff(slope, axis=0))
        # print(np.shape(diff_slope))

        qq_slope.append(diff_slope)
    return qq_slope

#%%
# ============================ Quantile per phenotype ==========================

#Parameters
q_range = np.linspace(0.01, 0.99,99)
# label_gen = ('WT','Dp1Yey')
label_gen = ('WT', 'Dp1Yey Treat')
label_gen_veh_treat = ('WT Treat', 'WT VEH', 'Dp1Yey Treat', 'Dp1Yey VEH')
label_vel = ('short', 'mid', 'long')
#%%
#quantile per genotype
# wp_genotype = (wp_wt, wp_mut)  #window pooling for genotype
wp_genotype = (wp_wt_treat, wp_mut_treat)  #window pooling for genotype
qq_genotype = quantiles_across_groups(wp_genotype, q_range)
wp_genotype_veh_treat = (wp_wt_veh, wp_wt_treat, wp_mut_veh, wp_mut_treat)  #window pooling for genotype
qq_genotype_veh_treat = quantiles_across_groups(wp_genotype_veh_treat, q_range)


# quantile per group
# qq_genotype = quantile_per_group(wp_genotype, q_range)
#%%
# Slope per group
slope_qq_gen = qq_slope(qq_genotype)
slope_qq_gen_veh_treat = qq_slope(qq_genotype_veh_treat)

# Diff per group
diff_qq_gen = qq_diff(qq_genotype)
diff_qq_gen_veh_treat = qq_diff(qq_genotype_veh_treat)

#%%
# Plots 
# Plot slopes
plt.figure(2, figsize=(12,5))
for vv in range(np.shape(slope_qq_gen)[0]):
    plt.plot(np.arange(len(np.diff(q_range)))/len(np.diff(q_range)), slope_qq_gen[vv],'.-',label=label_vel[vv])
plt.title(r"$\Delta$ Slope of Percentiles (Genotype)")
plt.xlabel("Quantiles")
plt.ylabel(r"$\Delta$ Slope (Dp1Yey treat - WT treat)")
plt.axhline(0, color='k', linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()
#%%
plt.figure(21, )
for vv in range(np.shape(slope_qq_gen_veh_treat)[1]):
    plt.plot(np.arange(len(np.diff(q_range)))/len(np.diff(q_range)), slope_qq_gen_veh_treat[vv],'.-',label=label_vel[vv])
plt.title(r"$\Delta$ Slope of Percentiles (Genotype)")
plt.xlabel("Quantiles")
plt.ylabel(r"$\Delta$ Slope (Mut-WT)")
plt.axhline(0, color='k', linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()
#%%
# Plot differences
plt.figure(3, figsize=(12,5))
for vv in range(np.shape(diff_qq_gen)[0]):
    plt.plot(q_range, diff_qq_gen[vv],'.-',label=label_vel[vv])
plt.title(r"$\Delta$ Difference of Percentiles (Genotype)")
plt.xlabel("Quantiles")
plt.ylabel(r"$\Delta$ Difference (Mut-WT)")
plt.axhline(0, color='k', linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()

#QQ plot
plt.figure(4, figsize=(12,5))
plt.clf()
for vv in range(np.shape(qq_genotype)[1]):
    print(vv)
    plt.subplot(1,3,vv+1)
    plt.title('Q-Q plot %s'%label_vel[vv])
    plt.scatter(qq_genotype[0, vv], qq_genotype[1, vv], label='%s'%label_vel[vv], facecolors='none', edgecolors='C%s'%vv, s=40)
    plt.xlabel("WT Quantiles")
    plt.ylabel("Mut Quantiles")
    # plot a diagonal line
    plt.plot([0.1, 1], [0.1, 1], color='k', linestyle='--', alpha=0.5)
    plt.legend()
    
#%% Compute Bootstrap permutation

#%%

# # ================== Bootstrap permutation ==========================
# # Parameters
n_replicas = 50
qq_gen_bootstrap = np.array(handler_bootstrap_permutation(wp_genotype,q_range, replicas=n_replicas))

#%%

def plot_bootstrap_ci_comparison(q_range, qq_data, qq_bootstrap, group_labels=('WT', 'Mut'), vel_labels=('short', 'mid', 'long'), title='Bootstrap CI per Window'):
    """
    Plot quantile curves and their confidence intervals for two groups.

    Parameters:
    - q_range: array of quantile values
    - qq_data: shape (2, len(q_range), 3), median quantiles for each group
    - qq_bootstrap: shape (2, 3, 2, len(q_range)), low/high CIs: [group, window, low/high, quantiles]
    """
    n_windows = len(vel_labels)
    
    plt.figure(figsize=(5 * n_windows, 5))
    for i in range(n_windows):
        plt.subplot(1, n_windows, i + 1)
        for g, label in enumerate(group_labels):
            # Plot median quantile line
            plt.plot(q_range, qq_data[g, :, i], label=f"{label} median", color=f"C{g}")
            
            # Plot CI band
            low = qq_bootstrap[g, i, 0]
            high = qq_bootstrap[g, i, 1]
            plt.fill_between(q_range, low, high, color=f"C{g}", alpha=0.2, label=f"{label} CI")

        plt.title(f"{vel_labels[i].capitalize()} Window")
        plt.xlabel("Quantile")
        plt.ylabel("dFC Speed")
        plt.grid(True)
        plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
qq_data = np.transpose(qq_genotype,(0,2,1)) # Shape: (n_groups, n_wpools, n_quantiles)
plot_bootstrap_ci_comparison(q_range, qq_data, qq_gen_bootstrap, group_labels=('WT', 'Mut'))

#%%
# =============================================================================
# Save data quantiles and bootstrap
# =============================================================================
data_quantile = {}
data_quantile['q_range'] = q_range
#quartile
data_quantile['qq_gen_all'] = qq_genotype
#quartile slope
data_quantile['qq_slope_gen_all'] = slope_qq_gen
#quartile bootstrap
data_quantile['qq_boot_gen'] = qq_gen_bootstrap
#quartile bootstrap slope
data_quantile['qq_slope_boot_gen'] = qq_gen_bootstrap
#quartile difference
data_quantile['qq_diff_gen'] = diff_qq_gen
#quartile difference slope
data_quantile['qq_slope_diff_gen'] = slope_qq_gen
#Save the data in optimal format .npz
np.savez(SPEED_DIR / f'qq_genotype_boot={n_replicas}.npz', **data_quantile)
# savemat('results/statistics/qq_animals_gen_phe_boot=%s.mat'%n_replicas, data_quantile)

#%%
