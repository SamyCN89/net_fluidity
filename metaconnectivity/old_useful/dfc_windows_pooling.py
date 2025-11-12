#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 14:42:38 2023

@author: samy
"""
from tqdm import tqdm
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

#%% Figure parameters 
# =============================================================================
# Figure's parameters
# =============================================================================

# Set figure parameters globally
plt.rcParams.update({'axes.labelsize': 15, 'axes.titlesize': 13,
                     # 'axes.spines.left': False, 'axes.spines.bottom': False,
                     'axes.spines.right': False, 'axes.spines.top': False})
# plt.style.use('seaborn-white')
save_fig =True

#%% Define paths, folders and hash
# root = '/home/samy/Bureau/Proyect/LauraHarsan/Ines/Timecourses_updated/'
# folders = {'2mois': 'Lot3_2mois', '4mois': 'Lot3_4mois'}
root = '/home/samy/Bureau/Proyect/LauraHarsan/Ines/Timecourses_updated_03052024/'
folders = {'2mois': 'TC_2months', '4mois': 'TC_4months'}
folder_results = '/home/samy/Bureau/Proyect/LauraHarsan/Ines/results/'

#%% Parameters speed
window_parameter    = (5,100,1)
time_windows_min, time_windows_max, time_window_step = window_parameter
time_windows_range = np.arange(time_windows_min,time_windows_max+1,time_window_step)
lag                 = 1
tau                 = 3
tau_array           = np.append(np.arange(0,tau), tau) 
lentau              = len(tau_array)
#%% Load data - Intersect the functional data for 2 and 4 months
# =============================================================================
# Load data - Intersect the functional data for 2 and 4 months
# =============================================================================

#hash data
hash_parameters = ('lag=%s_tau=%s_wmax=%s_wmin=%s'%(lag,tau,window_parameter[1],window_parameter[0]))
# Load filenames and hash numbers
filenames       = {period: filename_sort_mat(os.path.join(root, folder)) for period, folder in folders.items()}
hash_numbers    = {period: extract_hash_numbers(filenames[period]) for period in filenames}

int_2m4m = np.intersect1d(hash_numbers['2mois'], hash_numbers['4mois'], return_indices=True)
print('Number of intersected elements in 2m and 4m :' , len(int_2m4m[0]))

# =============================================================================
# Load cognitive data from .xlsx document
# =============================================================================
#Load cognitive data
#Old data
# cog_data_path   = os.path.join(root, 'Behaviour_exclusions_ROIs_female.xlsx')
# # cog_data_path   = os.path.join(root, 'Behaviour_exclusions_ROIs.xlsx')
# cog_data_df     = pd.read_excel(cog_data_path, sheet_name='Feuil1')
# data_roi        = pd.read_excel(cog_data_path, sheet_name='40_Allen_ROIs_list').to_numpy()
#Update 03/05/2024
cog_data_path   = os.path.join(root, 'ROIs.xlsx')
cog_data_df     = pd.read_excel(cog_data_path, sheet_name='Exclusions')
data_roi        = pd.read_excel(cog_data_path, sheet_name='41_Allen').to_numpy()

# Anatomical labels
anat_labels     = np.array([xx[0] for xx in data_roi])

cog_data_df['sexe_label']  = cog_data_df.loc[:,'Sexe']
cog_data_df['gen_label']  = cog_data_df.loc[:,'Genotype']

# cog_data_df['oip_4m-2m']  = ((cog_data_df.loc[:,'OiP_4M']+1)/2)-((cog_data_df.loc[:,'OiP_2M']+1)/2)
cog_data_df['oip_4m-2m']  = cog_data_df.loc[:,'OiP_4M']-cog_data_df.loc[:,'OiP_2M']
cog_data_df['oip_4m+2m']  = cog_data_df.loc[:,'OiP_4M']+cog_data_df.loc[:,'OiP_2M']

cog_data_df['ro24h_4m-2m']  = cog_data_df.loc[:,'RO24h_4M']-cog_data_df.loc[:,'RO24h_2M']
# cog_data_df['ro24h_4m-2m']  = ((cog_data_df.loc[:,'RO24h_4M']+1)/2)-((cog_data_df.loc[:,'RO24h_2M']+1)/2)
cog_data_df['ro24h_4m+2m']  = cog_data_df.loc[:,'RO24h_4M']+cog_data_df.loc[:,'RO24h_2M']


# Filtering based on sex (Male/Female), genotype (wt/dKI) and TC (ok/Excluded)
# Convert sex and genotype to numerical values for easier processing
cog_data_df['Sexe']     = cog_data_df['Sexe'].map({'M': 0, 'F': 1})
cog_data_df['Genotype'] = cog_data_df['Genotype'].map({'wt': 0, 'dKI': 1})
cog_data_df['TC_2M']    = cog_data_df['TC_2M'].map({'ok': 0, 'Excluded': 1})
cog_data_df['TC_4M']    = cog_data_df['TC_4M'].map({'ok': 0, 'Excluded': 1})

#%%

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
#%%

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

#Some important variables
n_animals, total_tp, regions = ts2m.shape
#%%
# =============================================================================
# #Load functional dataset
# =============================================================================
#Old
# load_vel = np.load(folder_results + 'speed/speed2m4m_dist' + hash_parameters+'.npz', allow_pickle=True)
#Update 03/05/2024
load_vel = np.load(folder_results + 'speed/speed2m4m_dist_03052024' + hash_parameters +'.npz', allow_pickle=True)

vel_2m = load_vel['vel_2m']
vel_4m = load_vel['vel_4m']

#%%
# =============================================================================
# Windows pooling of window oversampled speeds 
# =============================================================================

vel_list2m=vel_2m
vel_list4m=vel_4m

# limit_short_mid = 10
# limit_mid_long = 31

limit_short_mid = 13
limit_mid_long = 53

aux_timewr = time_windows_range*2


vel_label = ('%s-%ss (short)'%(aux_timewr[0], aux_timewr[limit_short_mid]),
             '%s-%ss (mid)'%(aux_timewr[limit_short_mid], aux_timewr[limit_mid_long]),
             '%s-%ss (long)'%(aux_timewr[limit_mid_long], aux_timewr[-1]))
# n_animals = int(np.sum(male_index))

#For the dfc speed distribution window oversampling, get a windows pooling
aux_short2m = np.array([np.hstack(vel_list2m[xx][0*lentau : limit_short_mid*lentau]) for xx in range(n_animals)])
aux_mid2m = np.array([np.hstack(vel_list2m[xx][limit_short_mid*lentau:limit_mid_long*lentau]) for xx in range(n_animals)])
aux_long2m = np.array([np.hstack(vel_list2m[xx][limit_mid_long*lentau:]) for xx in range(n_animals)])

aux_short4m = np.array([np.hstack(vel_list4m[xx][0*lentau:limit_short_mid*lentau]) for xx in range(n_animals)])
aux_mid4m = np.array([np.hstack(vel_list4m[xx][limit_short_mid*lentau:limit_mid_long*lentau]) for xx in range(n_animals)])
aux_long4m = np.array([np.hstack(vel_list4m[xx][limit_mid_long*lentau:]) for xx in range(n_animals)])

wp_list = (aux_short2m, aux_mid2m, aux_long2m, aux_short4m, aux_mid4m, aux_long4m)

def wpool_impaired(wp_list, index_group):
    print('Group:', np.sum(index_group))
    wp_wt = np.asarray([np.hstack(wp_list[xx][index_group]) for xx in range(6)], dtype=object)
    return wp_wt

#%%
# =============================================================================
# #Index
# =============================================================================
#female
fem_wt_index = wt_index&female_index
fem_dki_index = dki_index&female_index

#male
male_wt_index = wt_index&male_index
male_dki_index = dki_index&male_index

#Oip index
thr_cog=0.2
good_oip = np.logical_and((cog_data_filtered['OiP_2M'] > thr_cog) , (cog_data_filtered['OiP_4M'] >thr_cog)) 
learners_oip = np.logical_and((cog_data_filtered['OiP_2M'] < thr_cog) , (cog_data_filtered['OiP_4M'] >thr_cog)) 
impaired_oip = np.logical_and((cog_data_filtered['OiP_2M'] > thr_cog) , (cog_data_filtered['OiP_4M'] <thr_cog)) 
bad_oip  = np.logical_and((cog_data_filtered['OiP_2M'] < thr_cog) , (cog_data_filtered['OiP_4M'] <thr_cog)) 

#or index
good_ro24 = np.logical_and((cog_data_filtered['RO24h_2M'] > thr_cog) , (cog_data_filtered['RO24h_4M'] >thr_cog)) 
learners_ro24 = np.logical_and((cog_data_filtered['RO24h_2M'] < thr_cog) , (cog_data_filtered['RO24h_4M'] >thr_cog)) 
impaired_ro24 = np.logical_and((cog_data_filtered['RO24h_2M'] > thr_cog) , (cog_data_filtered['RO24h_4M'] <thr_cog)) 
bad_ro24  = np.logical_and((cog_data_filtered['RO24h_2M'] < thr_cog) , (cog_data_filtered['RO24h_4M'] <thr_cog)) 


#%%
# =============================================================================
# #Window pooled datasets
# =============================================================================
#Genotyped based
wp_wt = wpool_impaired(wp_list, wt_index)
wp_dki = wpool_impaired(wp_list, dki_index)

#Gender/Genotyped based
wp_wt_female = wpool_impaired(wp_list, fem_wt_index)
wp_dki_female = wpool_impaired(wp_list, fem_dki_index)
wp_wt_male = wpool_impaired(wp_list, male_wt_index)
wp_dki_male = wpool_impaired(wp_list, male_dki_index)

#Phenotype based on oip
wp_good_oip = wpool_impaired(wp_list, good_oip)
wp_bad_oip = wpool_impaired(wp_list, bad_oip)
wp_learners_oip = wpool_impaired(wp_list, learners_oip)
wp_impaired_oip = wpool_impaired(wp_list, impaired_oip)

#Phenotype based on ro24h
wp_good_ro24 = wpool_impaired(wp_list, good_ro24)
wp_bad_ro24 = wpool_impaired(wp_list, bad_ro24)
wp_learners_ro24 = wpool_impaired(wp_list, learners_ro24)
wp_impaired_ro24 = wpool_impaired(wp_list, impaired_ro24)

#%%
# =============================================================================
# Plot windows pool
# =============================================================================
def plot_wpool(wp_var1, wp_var2, name_data = 'all'):
    plt.figure(1, figsize=(12,10))
    plt.clf()
    # vel_label = ('10-30s (short)','30-72s (mid)','72-160s (long)')
    
    # wp_var1 = wp_wt
    # wp_var2 = wp_dki
    for i in range(3):
        plt.subplot(3,2,2*i+1)
        if i==0:
            plt.title('%s %s'%(vel_label[i],name_data))
        else:
            plt.title('%s'%vel_label[i])
        plt.hist((wp_var1[i], wp_var2[i], wp_var1[i+3], wp_var2[i+3]),label=('2m wt', '2m dki', '4m wt', '4m dki'), histtype='step',bins=200, density=True)
        plt.ylabel('Counts')
    
        plt.subplot(3,2,2*i+2)
        plt.hist((wp_var1[i], wp_var2[i], wp_var1[i+3], wp_var2[i+3]),label=('2m wt', '2m dki', '4m wt', '4m dki'), histtype='step',bins=500, density=True, log=True)
        plt.xlim(0.2,1.2)
    plt.xlabel('Freq[v]')
    plt.legend()
    plt.tight_layout()
    
    if save_fig ==True:
        plt.savefig('fig/speed/speed_window_pooling_and_oversampling_%s_dki_vs_wt_lag=%s_tau=%s.png'%(name_data,lag,tau))
        plt.savefig('fig/speed/speed_window_pooling_and_oversampling_%s_dki_vs_wt_lag=%s_tau=%s.pdf'%(name_data,lag,tau))

plot_wpool(wp_wt, wp_dki, name_data = 'all')
plot_wpool(wp_wt_female, wp_dki_female, name_data = 'female')
plot_wpool(wp_wt_male, wp_dki_male, name_data = 'male')
#%%
# =============================================================================
# Cumsum
# =============================================================================

a = np.sort(wp_wt_male[0])
b = np.sort(wp_wt_male[3])

c = np.sort(wp_dki_male[0])
d = np.sort(wp_dki_male[3])
p = 1 * np.arange(len(a))/(len(a)-1)
q = 1 * np.arange(len(c))/(len(c)-1)

plt.figure(11)
plt.clf()
plt.plot(a,p, linewidth=3)
plt.plot(b,p)
plt.plot(c,q)
plt.plot(d,q)
plt.legend(('2m wt', '2m dki', '4m wt', '4m dki'))
# plt.xscale('log')
# plt.yscale('log')
#%%

label_wp = ('good', 'learners', 'bad', 'impaired')

label_cog = 'oip'

plt.figure(2, figsize=(12,10))
plt.clf()

wp_var1 = wp_good_oip
wp_var2 = wp_learners_oip
wp_var3 = wp_bad_oip
wp_var4 = wp_impaired_oip

for i in range(3):
    plt.subplot(3,2,2*i+1)
    if i==0:
        plt.title('%s %s'%(vel_label[i],'OiP 2m'))
    else:
        plt.title('%s'%vel_label[i])
    plt.title('OiP 2m')
    plt.hist((wp_var1[i], wp_var2[i], wp_var3[i], wp_var4[i]), histtype='step',bins=150, density=True)
    plt.ylabel('Counts')
    plt.xlim(0.2,1.2)

    plt.subplot(3,2,2*i+2)
    if i==0:
        plt.title('%s %s'%(vel_label[i],'OiP 4m'))
    else:
        plt.title('%s'%vel_label[i])
    plt.hist((wp_var1[i+3], wp_var2[i+3], wp_var3[i+3], wp_var4[i+3]),label=label_wp, histtype='step',bins=150, density=True)
    plt.xlim(0.2,1.2)
plt.xlabel('Freq[v]')
plt.legend()
plt.tight_layout()

if save_fig ==True:
    plt.savefig('fig/speed/%s_speed_window_pool_and_oversampling_%s.png'%(label_cog, hash_parameters))
    plt.savefig('fig/speed/%s_speed_window_pool_and_oversampling_%s.pdf'%(label_cog, hash_parameters))
    # plt.savefig('fig/speed/oip_speed_window_pool_and_oversampling_%s_dki_vs_wt_lag=%s_tau=%s.png'%(name_data,lag,tau))
#%%
label_cog='ro24h'
plt.figure(3, figsize=(12,10))
plt.clf()

wp_var1 = wp_good_ro24
wp_var2 = wp_learners_ro24
wp_var3 = wp_impaired_ro24
wp_var4 = wp_bad_ro24

label_wp = ('good', 'learners', 'bad', 'impaired')

for i in range(3):
    plt.subplot(3,2,2*i+1)
    if i==0:
        plt.title('%s %s'%(vel_label[i],'RO24h 2m'))
    else:
        plt.title('%s'%vel_label[i])
    plt.hist((wp_var1[i], wp_var2[i], wp_var3[i], wp_var4[i]), histtype='step',bins=200, density=True)
    plt.ylabel('Counts')
    plt.xlim(0.2,1.2)

    plt.subplot(3,2,2*i+2)
    if i==0:
        plt.title('%s %s'%(vel_label[i],'RO24h 4m'))
    else:
        plt.title('%s'%vel_label[i])
    plt.hist((wp_var1[i+3], wp_var2[i+3], wp_var3[i+3], wp_var4[i+3]),label=label_wp, histtype='step',bins=200, density=True)
    plt.xlim(0.2,1.2)
plt.xlabel('Freq[v]')
plt.legend()
plt.tight_layout()
if save_fig ==True:
    plt.savefig('fig/speed/%s_speed_window_pool_and_oversampling_%s.png'%(label_cog, hash_parameters))
    plt.savefig('fig/speed/%s_speed_window_pool_and_oversampling_%s.pdf'%(label_cog, hash_parameters))


#%%# =============================================================================
# Save windows pooling data
# =============================================================================
np.savez(folder_results + 'speed/windowspooling_' + hash_parameters,
         wpool_wt       = wp_wt, 
         wpool_dki      = wp_dki, 
         wpool_wt_fem   = wp_wt_female,
         wpool_dki_fem  = wp_dki_female, 
         wpool_wt_male  = wp_wt_male, 
         wpool_dki_male = wp_dki_male)
# load_wpool = np.load(folder_results + 'speed/windowspooling_' + hash_parameters+'.npz', allow_pickle=True)

#%%
# =============================================================================
# Compute velocity statistics
# =============================================================================


#%%
# =============================================================================
# Compute and plot quantile range
# =============================================================================
def chauvenet(array):
    mean = array.mean()           # Mean of incoming array
    stdv = array.std()            # Standard deviation
    N = len(array)                # Lenght of incoming array

    criterion = 1.0/(2*N)         # Chauvenet's criterion
    d = abs(array-mean)/stdv      # Distance of a value to mean in stdv's

    prob = erfc(d)                # Area normal dist.    
    # Calculate probability of each data point
    # prob = 2 * (1 - norm.cdf(d))
    
    return prob < criterion       # Use boolean array outside this function

def compute_quantile_range(wp_aux, q_range):

    qq_wp = np.array([np.quantile(wp_aux[xx], q_range) for xx in range(6)])
    qq_wp_reshaped = np.reshape(qq_wp,(2,3,99)).transpose(2,0,1)
    return qq_wp_reshaped

# def compute_quantile_range(wp_aux, q_range):

#     qq_wp= []
#     for qq in range(len(q_range)):
#         q_aux = [np.quantile(wp_aux[xx], q_range[qq]) for xx in range(6)]
#         q_aux = np.reshape(q_aux, ((2,3)))
#         qq_wp.append(q_aux)
#     return np.array(qq_wp)

def quantile_per_group(group_list, q_range):
    len_group = np.array(group_list).shape[0]
    # wp_phenotype_oip = (wp_good_oip, wp_learners_oip,  wp_bad_oip, wp_impaired_oip)
    qq_group = np.array([compute_quantile_range(group_list[xx], q_range) for xx in range(len_group)])
    return qq_group

def qq_slope(qq_data):
    nv = qq_data.shape[0]
    oip_qq_slope =[]
    for tt in range(nv):
        qq_aux = qq_data[tt]
        # good_oip_qq_slope = np.array([(np.diff(qq_aux[qq], axis=0)/np.sum(qq_aux[qq], axis=0))[0] for qq in range(len(q_range))])
        good_oip_qq_slope = np.array([(np.diff(qq_aux[qq], axis=0))[0] for qq in range(len(q_range))])
        oip_qq_slope.append(good_oip_qq_slope)
    return oip_qq_slope

#%%
def bootstrap_permutation(data_true, q_range, replicas=1000):
    start = time.time()
    
    n_len = data_true.shape[0]
    q_values = np.array(q_range)
    
    # Preallocate arrays for quantiles to minimize memory usage
    quantiles = np.zeros((len(q_values), replicas))
    
    for i in tqdm(range(replicas)):
        # Sample indices without creating a large temporary array of sampled data
        indices = np.random.randint(0, n_len, n_len)
        # print(indices.shape)
        
        # Calculate quantiles directly from selected indices to avoid duplicating data
        sampled_data = np.take(data_true, indices, axis=0)
        quantiles[:, i] = np.quantile(sampled_data, q_values, axis=0)
    
    # Calculate confidence intervals without creating a large bootstrap sample array
    low_q, high_q = np.quantile(quantiles, [0.025, 0.975], axis=1)

    stop = time.time()
    print('time for bootstrap: ', stop - start)

    return low_q, high_q

def bootstrap_permutation_old(data_true, q_range, replicas=3):
    start = time.time()
    
    n_len= data_true.shape[0]
    
    # Using np.random.choice for more direct sampling
    indices = np.random.randint(0, n_len, (replicas, n_len))
    data_test = data_true[indices]
    
    bootstrap = np.quantile(data_test, q_range, axis=1)
    
    #Confidance interval
    low_q, high_q = np.quantile(bootstrap, [0.025, 0.975], axis=1)

    stop = time.time()
    print('time for bootstrap: ', stop-start)

    return low_q,high_q
# a = bootstrap_permutation(wp_wt[0], q_range, replicas=1000)
# b = bootstrap_permutation_old(wp_wt[0], q_range, replicas=1000)

#%%
def handler_bootstrap_permutation(wp_type, q_range, replicas=10):
    n_type = np.array(wp_type).shape[0]
    aux_qq_data = []
    for wp_ in tqdm(wp_type):
        n = wp_.shape[0]
        wp_boot = np.array([bootstrap_permutation(wp_[xx], q_range, replicas) for xx in range(n)])
        # qq_confint = np.array(wp_boot[3:]-wp_boot[:3])
        aux_qq_data.append(wp_boot)
    return aux_qq_data

q_range = np.linspace(0.01, 0.99,99)
label_gen = ('wt','dki')
label_vel = ('short', 'mid', 'long')

#quantile per phenotype
wp_phenotype_oip = (wp_good_oip, wp_learners_oip,  wp_bad_oip, wp_impaired_oip)
wp_phenotype_ro24 = (wp_good_ro24, wp_learners_ro24,wp_bad_ro24,  wp_impaired_ro24)

qq_phenotype_oip = quantile_per_group(wp_phenotype_oip, q_range)
qq_phenotype_ro24 = quantile_per_group(wp_phenotype_ro24, q_range)

#quantile per genotype and male/female
wp_genotype = (wp_wt, wp_dki)  
wp_genotype_male = (wp_wt_male, wp_dki_male)  
wp_genotype_female = (wp_wt_female, wp_dki_female)  

qq_genotype = quantile_per_group(wp_genotype, q_range)
qq_genotype_male = quantile_per_group(wp_genotype_male, q_range)
qq_genotype_female = quantile_per_group(wp_genotype_female, q_range)
#%%
#Slope between 4m and 2m
oip_qq_slope = qq_slope(qq_phenotype_oip)
ro24_qq_slope = qq_slope(qq_phenotype_ro24)

gen_qq_slope = qq_slope(qq_genotype)
gen_male_qq_slope = qq_slope(qq_genotype_male)
gen_female_qq_slope = qq_slope(qq_genotype_female)

#%% Compute Bootstrap permutation
n_replicas = 500
qq_phe_oip_bootstrap = np.array(handler_bootstrap_permutation(wp_phenotype_oip,q_range, replicas=n_replicas))
qq_phe_ro24_bootstrap = np.array(handler_bootstrap_permutation(wp_phenotype_ro24,q_range, replicas=n_replicas))

qq_gen_bootstrap = np.array(handler_bootstrap_permutation(wp_genotype,q_range, replicas=n_replicas))
qq_gen_male_bootstrap = np.array(handler_bootstrap_permutation(wp_genotype_male,q_range, replicas=n_replicas))
qq_gen_female_bootstrap = np.array(handler_bootstrap_permutation(wp_genotype_female,q_range, replicas=n_replicas))

#%%
# =============================================================================
# Save data
# =============================================================================
data_quantile = {}
data_quantile['q_range'] = q_range
#quartile
data_quantile['qq_oip'] = qq_phenotype_oip
data_quantile['qq_ro24'] = qq_phenotype_ro24
data_quantile['qq_gen_all'] = qq_genotype
data_quantile['qq_gen_male'] = qq_genotype_male
data_quantile['qq_gen_female'] = qq_genotype_female
#quartile slope
data_quantile['qq_slope_oip'] = oip_qq_slope
data_quantile['qq_slope_ro24h'] = ro24_qq_slope
data_quantile['qq_slope_gen_all'] = gen_qq_slope
data_quantile['qq_slope_gen_male'] = gen_male_qq_slope
data_quantile['qq_slope_gen_female'] = gen_female_qq_slope
#quartile bootstrap
data_quantile['qq_boot_oip'] = qq_phe_oip_bootstrap
data_quantile['qq_boot_ro24h'] = qq_phe_ro24_bootstrap
data_quantile['qq_boot_gen'] = qq_gen_bootstrap
data_quantile['qq_boot_gen_male'] = qq_gen_male_bootstrap
data_quantile['qq_boot_gen_female'] = qq_gen_female_bootstrap

savemat('results/statistics/qq_animals_gen_phe_boot=%s.mat'%n_replicas, data_quantile)

#%%

#quantile per phenotype at 2 and 4 months
for qq in range(len(q_range)):
    plt.figure(41, figsize=(10,5))
    plt.clf()
    for vv in range(3):
        plt.subplot(2,3,1+vv)
        plt.title('oip %s qtl wp %s'%(np.round(q_range[qq],3), label_vel[vv]))
        for phe in range(4):
            plt.plot(qq_phenotype_oip[phe, qq, :, vv], '.-', label=label_wp[phe])

            plt.plot(0, qq_phe_oip_bootstrap[phe, vv,1,qq], color='C%s'%phe, marker='^',alpha=0.3)
            plt.plot(0, qq_phe_oip_bootstrap[phe, vv,0,qq], color='C%s'%phe, marker='v',alpha=0.3)
            plt.plot(1, qq_phe_oip_bootstrap[phe, vv+3,1,qq], color='C%s'%phe, marker='^',alpha=0.3)
            plt.plot(1, qq_phe_oip_bootstrap[phe, vv+3,0,qq], color='C%s'%phe, marker='v',alpha=0.3)
        plt.xticks((0,1), ('2m', '4m'),fontsize=12)

        plt.subplot(2,3,4+vv)
        plt.title('ro24 %s qtl wp %s'%(np.round(q_range[qq],3), label_vel[vv]))
        for phe in range(4):
            plt.plot(qq_phenotype_ro24[phe, qq, :, vv], '.-', label=label_wp[phe])
            plt.plot(0, qq_phe_ro24_bootstrap[phe, vv,1,qq], color='C%s'%phe, marker='^',alpha=0.3)
            plt.plot(0, qq_phe_ro24_bootstrap[phe, vv,0,qq], color='C%s'%phe, marker='v',alpha=0.3)
            plt.plot(1, qq_phe_ro24_bootstrap[phe, vv+3,1,qq], color='C%s'%phe, marker='^',alpha=0.3)
            plt.plot(1, qq_phe_ro24_bootstrap[phe, vv+3,0,qq], color='C%s'%phe, marker='v',alpha=0.3)
        plt.xticks((0,1), ('2m', '4m'),fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig/statistics/qq/phenotype/wp_phenotype_statst_quantile_%s_%s.png'%(qq, hash_parameters))

#%%
#quantile per genotype 2 and 4 months
def quantile_plot_dki_wt(qq_genotype, qq_gen_bootstrap, gen_type='all'):
    for qq in tqdm(range(len(q_range))):
        plt.figure(42, figsize=(10,5))
        plt.clf()
        for vv in range(3):
            plt.subplot(1,3,1+vv)
            plt.title('oip %s qtl wp %s'%(np.round(q_range[qq],3), label_vel[vv]))
            for phe in range(2):
                plt.plot(qq_genotype[phe, qq, :, vv], 'o-', label=label_gen[phe])
                plt.plot(0,qq_gen_bootstrap[phe, vv,1,qq], color='C%s'%phe, marker='^',alpha=0.3)
                plt.plot(0,qq_gen_bootstrap[phe, vv,0,qq], color='C%s'%phe, marker='v',alpha=0.3)
                plt.plot(1,qq_gen_bootstrap[phe, vv+3,1,qq], color='C%s'%phe, marker='^',alpha=0.3)
                plt.plot(1,qq_gen_bootstrap[phe, vv+3,0,qq], color='C%s'%phe, marker='^',alpha=0.3)
            plt.xticks((0,1), ('2m', '4m'),fontsize=12)
            # plt.ylim(np.min(qq_phenotype_oip[:, qq, :]), np.max(qq_phenotype_oip[:, qq, :]))
        
        plt.legend()
        plt.tight_layout()
        plt.savefig('fig/statistics/qq/genotype/%s/wp_statst_quantile_%s_%s.png'%(gen_type,qq,hash_parameters))


quantile_plot_dki_wt(qq_genotype, qq_gen_bootstrap, gen_type='all')
quantile_plot_dki_wt(qq_genotype_male, qq_gen_male_bootstrap, gen_type='male')
quantile_plot_dki_wt(qq_genotype_female, qq_gen_female_bootstrap, gen_type='female')
#%%,

plt.figure(44, figsize=(12,7))
plt.clf()
for vv in range(3):
    plt.subplot(2,3,1+vv)
    plt.title('oip wp %s'%(vel_label[vv]))
    for phe in range(4):
        plt.plot(q_range, oip_qq_slope[phe][:,vv],'.--', label=label_wp[phe])
        plt.plot(q_range, qq_phe_oip_bootstrap[phe, vv+3,0]-qq_phe_oip_bootstrap[phe, vv,0],'C%s'%(phe))
        plt.plot(q_range, qq_phe_oip_bootstrap[phe, vv+3,1]-qq_phe_oip_bootstrap[phe, vv,1],'C%s'%(phe))
        # plt.plot(q_range, qq_phe_oip_bootstrap[phe, vv,1],'C%s'%(phe))
    plt.ylim(-0.08,0.05)
    # plt.ylim(-0.2,0.1)
    plt.axhline(0,c='k')
    
    plt.ylabel(r'q$_{slope(4m,2m)}$')
for vv in range(3):
    plt.subplot(2,3,4+vv)
    plt.title('ro24 wp %s'%(vel_label[vv]))
    for phe in range(4):
        plt.plot(q_range, ro24_qq_slope[phe][:,vv],'.--', label=label_wp[phe])
        plt.plot(q_range, qq_phe_ro24_bootstrap[phe, vv+3,0]-qq_phe_ro24_bootstrap[phe, vv,0],'C%s'%(phe))
        plt.plot(q_range, qq_phe_ro24_bootstrap[phe, vv+3,1]-qq_phe_ro24_bootstrap[phe, vv,1],'C%s'%(phe))
        # plt.plot(q_range, qq_phe_ro24_bootstrap[phe, vv,0],'C%s'%(phe))
        # plt.plot(q_range, qq_phe_ro24_bootstrap[phe, vv,1],'C%s'%(phe))
    plt.ylim(-0.08,0.15)
    # plt.ylim(-0.2,0.1)
    plt.axhline(0,c='k')
    plt.ylabel(r'q$_{slope(4m,2m)}$')
    plt.xlabel(r'quantile')
plt.legend()
plt.tight_layout()
plt.savefig('fig/statistics/special/wp_slope_phenotype_quantile_%s.png'%(hash_parameters))
#%%

# q_range = np.log(q_range)
plt.figure(45, figsize=(12,9))
plt.clf()
for vv in range(3):
    plt.subplot(3,3,1+vv)
    plt.title('all wp %s'%(vel_label[vv]))
    for phe in range(2):
        print('C%s'%(phe))
        plt.plot(q_range, gen_qq_slope[phe][:,vv],'.--',label=label_gen[phe])
        plt.plot(q_range, qq_gen_bootstrap[phe, vv+3,0] - qq_gen_bootstrap[phe, vv,0],'C%s'%(phe))
        plt.plot(q_range, qq_gen_bootstrap[phe, vv+3,1] - qq_gen_bootstrap[phe, vv,1],'C%s'%(phe))
        # plt.plot(q_range, qq_gen_bootstrap[phe, vv,0],'C%s'%(phe))
        # plt.plot(q_range, qq_gen_bootstrap[phe, vv,1],'C%s'%(phe))
    plt.ylim(-0.08,0.05)
    plt.axhline(0,c='k')
    plt.ylabel(r'q$_{slope(4m,2m)}$')

    plt.subplot(3,3,4+vv)
    plt.title('female wp %s'%(vel_label[vv]))
    for phe in range(2):
        plt.plot(q_range, gen_female_qq_slope[phe][:,vv],'.--', label=label_gen[phe])
        plt.plot(q_range, qq_gen_female_bootstrap[phe, vv+3,0] - qq_gen_female_bootstrap[phe, vv,0],'C%s'%(phe))
        plt.plot(q_range, qq_gen_female_bootstrap[phe, vv+3,1] - qq_gen_female_bootstrap[phe, vv,1],'C%s'%(phe))
        # plt.plot(q_range, qq_gen_female_bootstrap[phe, vv,0],'C%s'%(phe))
        # plt.plot(q_range, qq_gen_female_bootstrap[phe, vv,1],'C%s'%(phe))
    plt.ylim(-0.08,0.05)
    # plt.ylim(-0.2,0.1)
    plt.axhline(0,c='k')
    plt.ylabel(r'q$_{slope(4m,2m)}$')

    plt.subplot(3,3,7+vv)
    plt.title('male wp %s'%(vel_label[vv]))
    for phe in range(2):
        plt.plot(q_range, gen_male_qq_slope[phe][:,vv],'.--', label=label_gen[phe])
        plt.plot(q_range, qq_gen_male_bootstrap[phe, vv+3,0]-qq_gen_male_bootstrap[phe, vv,0],'C%s'%(phe))
        plt.plot(q_range, qq_gen_male_bootstrap[phe, vv+3,1]-qq_gen_male_bootstrap[phe, vv,1],'C%s'%(phe))
        # plt.plot(q_range, qq_gen_male_bootstrap[phe, vv,0],'C%s'%(phe))
        # plt.plot(q_range, qq_gen_male_bootstrap[phe, vv,1],'C%s'%(phe))
    plt.ylim(-0.08,0.05)
    # plt.ylim(-0.2,0.1)
    plt.axhline(0,c='k')
    plt.ylabel(r'q$_{slope(4m,2m)}$')
    plt.xlabel(r'quantile')
    # plt.plot(q_range, wt_qq_slope[:,vv],'.-', label=label_gen[0])
    # plt.plot(q_range, dki_qq_slope[:,vv],'.-', label=label_gen[1])
plt.legend()
plt.tight_layout()
plt.savefig('fig/statistics/special/wp_slope_genotype_quantile_%s.png'%(hash_parameters))
#%%
#%%
qq_phenotype_oip2 = qq_phenotype_oip.transpose(0,3,2,1) #Condition, quantiles, months, window_pool(slow,middel,fast)
qq_phenotype_ro242 = qq_phenotype_ro24.transpose(0,3,2,1) #Condition, window_pool(slow,middel,fast),months, ,quantiles

x=np.linspace(0.2, 1.06)

plt.figure(5)
plt.clf()
for yy in range(4):
    plt.subplot(2,2,yy+1)
    plt.title('Q-Q plot %s'%label_wp[yy])
    for xx in range(3):
        plt.scatter(qq_phenotype_ro242[yy,xx,0], qq_phenotype_ro242[yy,xx,1], label='%s'%vel_label[xx],facecolors='none', 
                    edgecolors='C%s'%xx,
                    s=15)
    plt.plot(x,x)
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('4 months')
    plt.xlabel('2 months')
plt.legend()
plt.xlim(0.2,1.06)
plt.ylim(0.2,1.06)

#%%
plt.figure(51)
plt.clf()
for xx in range(3):
    plt.subplot(1,3,xx+1)
    plt.title('Q-Q plot %s'%label_vel[xx])
    for yy in range(4):
        plt.scatter(qq_phenotype_oip2[yy,xx,0], qq_phenotype_oip2[yy,xx,1], 
                    label='%s'%label_wp[yy],
                    facecolors='none', 
                    edgecolors='C%s'%yy,
                    s=15,
                    alpha=0.5)
    plt.plot(x,x,alpha=0.3,color='k')
    plt.xlim(0.2,1.06)
    plt.ylim(0.2,1.06)
    plt.xticks((0.2,0.6,1))
    plt.yticks((0.2,0.6,1))
    # plt.yscale('log')
    # plt.xscale('log')
    plt.ylabel('4 months')
    plt.xlabel('2 months')
plt.legend()
#%%
qq_genotype2 = qq_genotype.transpose(0,3,2,1)
qq_genotype_female2 = qq_genotype_female.transpose(0,3,2,1)
qq_genotype_male2 = qq_genotype_male.transpose(0,3,2,1)

plt.figure(52,figsize=(12,5))
plt.clf()
for xx in range(3):
    plt.subplot(1,3,xx+1)
    plt.title('Q-Q plot %s'%label_vel[xx])
    for yy in range(2):
        # plt.scatter(qq_genotype_male2[yy,xx,0], qq_genotype_male2[yy,xx,1], 
        # plt.scatter(qq_genotype_female2[yy,xx,0], qq_genotype_female2[yy,xx,1], 
        plt.scatter(qq_genotype2[yy,xx,0], qq_genotype2[yy,xx,1], 
                    label='%s'%label_gen[yy],
                    facecolors='none', 
                    edgecolors='C%s'%yy,
                    s=15,
                    alpha=0.5)
    plt.plot(x,x,alpha=0.3,color='k')
    # plt.yscale('log')
    # plt.xscale('log')
    plt.ylabel('4 months')
    plt.xlabel('2 months')
    plt.xlim(0.2,1.06)
    plt.ylim(0.2,1.06)
    plt.xticks((0.2,0.6,1))
    plt.yticks((0.2,0.6,1))
plt.legend()
plt.tight_layout()