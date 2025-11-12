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
# =============================================================================
# Figure's parameters
# =============================================================================

# Set figure parameters globally
plt.rcParams.update({'axes.labelsize': 15, 'axes.titlesize': 13,
                     # 'axes.spines.left': False, 'axes.spines.bottom': False,
                     'axes.spines.right': False, 'axes.spines.top': False})
# plt.style.use('seaborn-white')
save_fig =False
save_data = False
#Define paths, folders and hash
root = '/home/samy/Bureau/Proyect/LauraHarsan/Ines/Timecourses_updated/'
folders = {'2mois': 'Lot3_2mois', '4mois': 'Lot3_4mois'}
folder_results = '/home/samy/Bureau/Proyect/LauraHarsan/Ines/results/'

#Parameters speed
window_parameter = (5,100,1)
lag=1
tau=5
tau_array       = np.append(np.arange(0,tau), tau ) 
lentau          = len(tau_array)
#hash data
hash_parameters = ('lag=%s_tau=%s_wmax=%s_wmin=%s'%(lag,tau,window_parameter[1],window_parameter[0]))
#%%
# =============================================================================
# Load data - Intersect the functional data for 2 and 4 months
# =============================================================================

# Load filenames and hash numbers
filenames       = {period: filename_sort_mat(os.path.join(root, folder)) for period, folder in folders.items()}
hash_numbers    = {period: extract_hash_numbers(filenames[period]) for period in filenames}

int_2m4m = np.intersect1d(hash_numbers['2mois'], hash_numbers['4mois'], return_indices=True)
print('Number of intersected elements in 2m and 4m :' , len(int_2m4m[0]))

# =============================================================================
# Load cognitive data from .xlsx document
# =============================================================================
#Load cognitive data
cog_data_path   = os.path.join(root, 'Behaviour_exclusions_ROIs_female.xlsx')
# cog_data_path   = os.path.join(root, 'Behaviour_exclusions_ROIs.xlsx')
cog_data_df     = pd.read_excel(cog_data_path, sheet_name='Feuil1')
data_roi        = pd.read_excel(cog_data_path, sheet_name='40_Allen_ROIs_list').to_numpy()

# Anatomical labels
anat_labels     = np.array([xx[0] for xx in data_roi])

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
#Load dataset
load_vel = np.load(folder_results + 'speed/speed2m4m_dist' + hash_parameters+'.npz', allow_pickle=True)

vel_2m = load_vel['vel_2m']
vel_4m = load_vel['vel_4m']
# np.savez(folder_results + hash_parameters+'.npy',np.asanyarray(vel_list2m))

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

#For the dfc speed distribution window oversampling, get a windows pooling
aux_short2m = np.array([np.hstack(vel_list2m[xx][0*lentau : limit_short_mid*lentau]) for xx in range(n_animals)])
aux_mid2m = np.array([np.hstack(vel_list2m[xx][limit_short_mid*lentau:limit_mid_long*lentau]) for xx in range(n_animals)])
aux_long2m = np.array([np.hstack(vel_list2m[xx][limit_mid_long*lentau:]) for xx in range(n_animals)])

aux_short4m = np.array([np.hstack(vel_list4m[xx][0*lentau:limit_short_mid*lentau]) for xx in range(n_animals)])
aux_mid4m = np.array([np.hstack(vel_list4m[xx][limit_short_mid*lentau:limit_mid_long*lentau]) for xx in range(n_animals)])
aux_long4m = np.array([np.hstack(vel_list4m[xx][limit_mid_long*lentau:]) for xx in range(n_animals)])

#Index
fem_wt_index = wt_index&female_index
fem_dki_index = dki_index&female_index

male_wt_index = wt_index&male_index
male_dki_index = dki_index&male_index

wp_list = (aux_short2m, aux_mid2m, aux_long2m, aux_short4m, aux_mid4m, aux_long4m)

wp_wt = np.asarray([np.hstack(wp_list[xx][wt_index]) for xx in range(6)], dtype=object)
wp_dki = np.asarray([np.hstack(wp_list[xx][dki_index]) for xx in range(6)], dtype=object)

wp_wt_female = np.asarray([np.hstack(wp_list[xx][fem_wt_index]) for xx in range(6)], dtype=object)
wp_dki_female = np.asarray([np.hstack(wp_list[xx][fem_dki_index]) for xx in range(6)], dtype=object)

wp_wt_male = np.asarray([np.hstack(wp_list[xx][male_wt_index]) for xx in range(6)], dtype=object)
wp_dki_male = np.asarray([np.hstack(wp_list[xx][male_dki_index]) for xx in range(6)], dtype=object)


def plot_wpool(wp_var1, wp_var2, name_data = 'all'):
    plt.figure(1, figsize=(12,10))
    plt.clf()
    vel_label = ('10-30s (short)','30-72s (mid)','72-160s (long)')
    
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
        # plt.savefig('fig/speed/speed_window_pooling_and_oversampling_male_dki_vs_wt_lag=%s_tau=%s.pdf'%(lag,tau))

plot_wpool(wp_wt, wp_dki, name_data = 'all')
plot_wpool(wp_wt_female, wp_dki_female, name_data = 'female')
plot_wpool(wp_wt_male, wp_dki_male, name_data = 'male')

#%%# =============================================================================
# Save windows pooling data
# =============================================================================
if save_data ==True:
    np.savez(folder_results + 'speed/windowspooling_' + hash_parameters,
             wpool_wt = wp_wt, 
             wpool_wt_fem = wp_wt_female,
             wpool_wt_male=wp_wt_male, 
             wpool_dki=wp_dki, 
             wpool_dki_fem=wp_dki_female, 
             wpool_dki_male=wp_dki_male)
    
    load_wpool = np.load(folder_results + 'speed/windowspooling_' + hash_parameters+'.npz', allow_pickle=True)

#%%
# =============================================================================
# Compute velocity statistics
# =============================================================================

def vel_statistics(vel_array, bins_number=150):
    hist = np.array([np.histogram(vel_array[xx], density=True, bins=150)[0] for xx in range(n_animals)])
    aux_vel = np.array([np.histogram(vel_array[xx], density=True, bins=150)[1] for xx in range(n_animals)])

    v_median = np.array([np.median(vel_array[xx]) for xx in range(n_animals)])#np.median(hist,axis=1)
    v_typ= np.array([aux_vel[xx,np.argmax(hist[xx])] for xx in range(n_animals)])
    vel_q5 = np.quantile(vel_array, 0.05, axis=1)
    vel_q95 = np.quantile(vel_array, 0.95, axis=1)
    
    return v_median, v_typ,vel_q5, vel_q95

short2m_statistics = vel_statistics(aux_short2m,bins_number=150)
short4m_statistics = vel_statistics(aux_short4m,bins_number=150)

mid2m_statistics = vel_statistics(aux_mid2m,bins_number=100)
mid4m_statistics = vel_statistics(aux_mid4m,bins_number=100)

long2m_statistics = vel_statistics(aux_long2m,bins_number=100)
long4m_statistics = vel_statistics(aux_long4m,bins_number=100)

#%%
# =============================================================================
# Compute velocity statistics for difference between 2 and 4 monthsdata
# =============================================================================
vel_list_all = ((short2m_statistics,short4m_statistics), (mid2m_statistics, mid4m_statistics), (long2m_statistics,long4m_statistics))
label_statistic = ('v_median', 'v_typ','vel_q5', 'vel_q95')
label_vel = ('short', 'mid', 'long')
kk=[2,4]
# label_vel = 'short'
for vv in range(3):
    for xx in range(2):
        for pp in range(len(label_statistic)):
            cog_data_filtered['%s %s %sm'%(label_statistic[pp],label_vel[vv],kk[xx])] = vel_list_all[vv][xx][pp]

for vv in range(3):
    for pp in range(len(label_statistic)):
        cog_data_filtered['delta4m2m %s %s '%(label_statistic[pp],label_vel[vv])] = vel_list_all[vv][1][pp]-vel_list_all[vv][0][pp]/(vel_list_all[vv][0][pp]+vel_list_all[vv][1][pp])


#%%
# =============================================================================
# Plot individual velocity performance for each statistic
# =============================================================================

# All veocities
for idx_vel, vel_sts in enumerate(vel_list_all):
    print(idx_vel,np.shape(vel_sts[0]))
    #all the statistics
    for xx in range(len(label_statistic)):
    
        plt.figure(2,figsize=(8,6))
        plt.clf()
        
        statistic = xx
        var2m = vel_sts[0][statistic]#long2m_statistics[0]
        var4m = vel_sts[1][statistic]#long4m_statistics[0]
        velmedian2m4m = np.array((var2m, var4m))
        
        plt.subplot(221)
        plt.title('WT %s %s'%(label_statistic[statistic], label_vel[idx_vel]))
        plt.plot(velmedian2m4m[:,wt_index],'o--')
        plt.ylabel('<v>')
        plt.xlim(-0.2,1.2)
        plt.xticks((0,1), ('2m', '4m'))
        plt.ylim(np.min(velmedian2m4m)-0.1, np.max(velmedian2m4m)+0.1)
        
        plt.subplot(222)
        plt.title('dKI %s % s'%(label_statistic[statistic], label_vel[idx_vel]))
        plt.plot(velmedian2m4m[:,dki_index],'o--')
        plt.xlim(-0.2,1.2)
        plt.xticks((0,1), ('2m', '4m'))
        plt.ylim(np.min(velmedian2m4m)-0.1, np.max(velmedian2m4m)+0.1)
        
        delta_vel2m4m = (np.array(var4m) - np.array(var2m)) / (np.array(var4m) + np.array(var2m))
        delta_vel2m4m = np.array((np.zeros(len(delta_vel2m4m)), delta_vel2m4m))
        
        plt.subplot(223)
        plt.plot(delta_vel2m4m[:,wt_index],'o--', c='C0')
        plt.ylabel(r'$\Delta$ v$_{4m-2m}$')
        plt.axhline(0,color='k',ls='--')
        plt.ylim(np.min(delta_vel2m4m[1])-0.01,np.max(delta_vel2m4m[1])+0.01)
        plt.xticks([])
        
        plt.subplot(224)
        plt.plot(delta_vel2m4m[:,dki_index],'o--', c='C1')
        plt.ylim(np.min(delta_vel2m4m[1])-0.01,np.max(delta_vel2m4m[1])+0.01)
        plt.axhline(0,color='k',ls='--')
        plt.tight_layout()
        plt.xticks([])
        
        if save_fig ==True:
            plt.savefig('fig/statistics/vel_statistics_%s_%s.png'%(label_statistic[statistic], label_vel[idx_vel]))
            plt.savefig('fig/statistics/vel_statistics_%s_%s.pdf'%(label_statistic[statistic], label_vel[idx_vel]))
        # plt.savefig('fig/statistics/fc_all_dki_vs_wt.pdf')
#%%

# =============================================================================
# Plot all individual cognitive data
# =============================================================================

plt.figure(3,figsize=(8,5))
plt.clf()

plt.subplot(221)
plt.title('All OiP')
plt.scatter(cog_data_filtered['OiP_2M'][dki_index], cog_data_filtered['OiP_4M'][dki_index], label='dki')
plt.scatter(cog_data_filtered['OiP_2M'][wt_index], cog_data_filtered['OiP_4M'][wt_index], label='wt')
plt.axhline(0.2,color='k')
plt.axvline(0.2,color='k')
plt.ylabel('4M')
plt.xlabel('2M')
plt.legend()

plt.subplot(222)
plt.title('All RO24h')
plt.scatter(cog_data_filtered['RO24h_2M'][dki_index], cog_data_filtered['RO24h_4M'][dki_index], label='dki')
plt.scatter(cog_data_filtered['RO24h_2M'][wt_index], cog_data_filtered['RO24h_4M'][wt_index], label='wt')
plt.axhline(0.2,color='k')
plt.axvline(0.2,color='k')
plt.ylabel('4M')
plt.xlabel('2M')
plt.legend()

plt.subplot(223)
plt.title('OiP')
plt.scatter(male_dki_data['OiP_2M'], male_dki_data['OiP_4M'], label='dki male')
plt.scatter(male_wt_data['OiP_2M'], male_wt_data['OiP_4M'], label='wt male')
plt.scatter(female_dki_data['OiP_2M'], female_dki_data['OiP_4M'], label='dki female')
plt.scatter(female_wt_data['OiP_2M'], female_wt_data['OiP_4M'], label='wt female')
plt.axhline(0.2,color='k')
plt.axvline(0.2,color='k')
plt.ylabel('4M')
plt.xlabel('2M')
plt.legend()

plt.subplot(224)
plt.title('RO24h')
plt.scatter(male_dki_data['RO24h_2M'], male_dki_data['RO24h_4M'], label='dki male')
plt.scatter(male_wt_data['RO24h_2M'], male_wt_data['RO24h_4M'], label='wt male')
plt.scatter(female_dki_data['RO24h_2M'], female_dki_data['RO24h_4M'], label='dki female')
plt.scatter(female_wt_data['RO24h_2M'], female_wt_data['RO24h_4M'], label='wt female')
plt.axhline(0.2,color='k')
plt.axvline(0.2,color='k')
plt.ylabel('4M')
plt.xlabel('2M')
plt.legend()

if save_fig ==True:
    plt.savefig('fig/cog_data/scatter_plot_wt_ski_male_female.png')
    plt.savefig('fig/cog_data/scatter_plot_wt_ski_male_female.pdf')

#%%
# =============================================================================
# Excluding in impaired, normal and dumb groups 
#Group into the dKI mice that lost is spatial memory at 4m
# =============================================================================
wp_list = (aux_short2m, aux_mid2m, aux_long2m, aux_short4m, aux_mid4m, aux_long4m)

def wpool_impaired(wp_list, index_group):

    print('Group:', np.sum(index_group))
    wp_wt = np.asarray([np.hstack(wp_list[xx][index_group]) for xx in range(6)], dtype=object)
    return wp_wt

good = np.logical_and((cog_data_filtered['OiP_2M'] > 0.2) , (cog_data_filtered['OiP_4M'] >0.2))
learners = np.logical_and((cog_data_filtered['OiP_2M'] < 0.2) , (cog_data_filtered['OiP_4M'] >0.2))
impaired = np.logical_and((cog_data_filtered['OiP_2M'] > 0.2) , (cog_data_filtered['OiP_4M'] <0.2))
bad  = np.logical_and((cog_data_filtered['OiP_2M'] < 0.2) , (cog_data_filtered['OiP_4M'] <0.2))

wp_good = wpool_impaired(wp_list, good)
wp_bad = wpool_impaired(wp_list, bad)
wp_learners = wpool_impaired(wp_list, learners)
wp_impaired = wpool_impaired(wp_list, impaired)


# oip_impaired = (cog_data_filtered['OiP_2M']>0.2)&(cog_data_filtered['OiP_4M']<0.2)
# ro24h_impaired = (cog_data_filtered['RO24h_2M']>0.2)&(cog_data_filtered['RO24h_4M']<0.2)

# # oip_normal = (cog_data_filtered['OiP_4M']>0.2 & cog_data_filtered['OiP_2M']>0.2)
# ro24h_normal = (cog_data_filtered['RO24h_4M']>0.2) & (cog_data_filtered['RO24h_2M']>0.2)

# oip_dumb = (cog_data_filtered['OiP_2M']<0.2)&(cog_data_filtered['OiP_4M']<0.2)
# ro24h_dumb = (cog_data_filtered['RO24h_2M']<0.2)&(cog_data_filtered['RO24h_4M']<0.2)

# impaired = ro24h_impaired
# normal = ro24h_normal
# dumb = ro24h_dumb

# normal = oip_normal
# impaired = oip_impaired
# dumb = oip_dumb


# print('Excluded:', np.sum(impaired))

# wp_list = (aux_short2m, aux_mid2m, aux_long2m, aux_short4m, aux_mid4m, aux_long4m)

# def wpool_impaired(wp_list, index_group, index_test):

#     print('Group:', np.sum(index_test), np.sum(index_group), np.sum(index_test))
#     wp_wt = np.asarray([np.hstack(wp_list[xx][index_test]) for xx in range(6)], dtype=object)
#     # wp_wt = np.asarray([np.hstack(wp_list[xx][index_group & index_test]) for xx in range(6)], dtype=object)
    
#     return wp_wt

# wp_wt = wpool_impaired(wp_list, wt_index, normal)

# wp_dki_normal = wpool_impaired(wp_list, dki_index, normal)
# wp_dki_imp = wpool_impaired(wp_list, dki_index, impaired)
# wp_dki_dumb = wpool_impaired(wp_list, dki_index, dumb)


plt.figure(4, figsize=(12,10))
plt.clf()
vel_label = ('10-30s (short)','30-72s (mid)','72-160s (long)')

wp_var1 = wp_good
wp_var2 = wp_bad
wp_var3 = wp_impaired
wp_var4 = wp_learners


label_wp = ('good', 'bad', 'impaired', 'learners')

for i in range(3):
    plt.subplot(3,2,2*i+1)
    plt.title('OiP 2m')
    plt.hist((wp_var1[i], wp_var2[i], wp_var3[i], wp_var4[i]), histtype='step',bins=200, density=True)
    plt.ylabel('Counts')
    plt.xlim(0.2,1.2)

    plt.subplot(3,2,2*i+2)
    plt.title('OiP 4m')
    plt.hist((wp_var1[i+3], wp_var2[i+3], wp_var3[i+3], wp_var4[i+3]),label=label_wp, histtype='step',bins=200, density=True)
    plt.xlim(0.2,1.2)
plt.xlabel('Freq[v]')
plt.legend()
plt.tight_layout()

#%%
good = np.logical_and((cog_data_filtered['RO24h_2M'] > 0.2) , (cog_data_filtered['RO24h_4M'] >0.2))
learners = np.logical_and((cog_data_filtered['RO24h_2M'] < 0.2) , (cog_data_filtered['RO24h_4M'] >0.2))
impaired = np.logical_and((cog_data_filtered['RO24h_2M'] > 0.2) , (cog_data_filtered['RO24h_4M'] <0.2))
bad  = np.logical_and((cog_data_filtered['RO24h_2M'] < 0.2) , (cog_data_filtered['RO24h_4M'] <0.2))

wp_good_ro = wpool_impaired(wp_list, good)
wp_bad_ro = wpool_impaired(wp_list, bad)
wp_learners_ro = wpool_impaired(wp_list, learners)
wp_impaired_ro = wpool_impaired(wp_list, impaired)

plt.figure(5, figsize=(12,10))
plt.clf()
vel_label = ('10-30s (short)','30-72s (mid)','72-160s (long)')

wp_var1 = wp_good_ro
wp_var2 = wp_bad_ro
wp_var3 = wp_impaired_ro
wp_var4 = wp_learners_ro

label_wp = ('good', 'bad', 'impaired', 'learners')

for i in range(3):
    plt.subplot(3,2,2*i+1)
    plt.title('RO24h 2m')
    plt.hist((wp_var1[i], wp_var2[i], wp_var3[i], wp_var4[i]), histtype='step',bins=200, density=True)
    plt.ylabel('Counts')
    plt.xlim(0.2,1.2)

    plt.subplot(3,2,2*i+2)
    plt.title('RO24h 4m')
    plt.hist((wp_var1[i+3], wp_var2[i+3], wp_var3[i+3], wp_var4[i+3]),label=label_wp, histtype='step',bins=200, density=True)
    plt.xlim(0.2,1.2)
plt.xlabel('Freq[v]')
plt.legend()
plt.tight_layout()


#%%
# =============================================================================
# Metaconnectivity
# =============================================================================
lag=1
window_parameter = (5,80,1)
time_windows_min, time_windows_max, time_window_step = window_parameter
time_windows_range = np.arange(time_windows_min,time_windows_max+1,time_window_step)

# windows_size = time_windows_range[limit_mid_long//2]
windows_size = 10
dfc_stream_2m   = np.array([ts2dfc_stream(ts2m[xx], windows_size, lag, format_data='2D') for xx in range(n_animals)])
dfc_stream_4m   = np.array([ts2dfc_stream(ts4m[xx], windows_size, lag, format_data='2D') for xx in range(n_animals)])
#%%
#metaconnectivity
# index1 = good
# index2 = impaired
index1 = wt_index
index2 = dki_index

#2m
mc_2m = np.array([np.corrcoef(dfc_stream_2m[xx]) for xx in range(n_animals)])
mc_2m_wt =mc_2m[index1]
mc_2m_wt_vis =mc_2m_wt[mc_2m_wt<0]

mc_2m_dki =mc_2m[index2]
mc_2m_dki_vis =mc_2m_dki[mc_2m_dki<0]

#4m
mc_4m       = np.array([np.corrcoef(dfc_stream_4m[xx]) for xx in range(n_animals)])
mc_4m_wt    = mc_4m[index1]
mc_4m_wt_vis = mc_4m_wt[mc_4m_wt<0]

mc_4m_dki =mc_4m[index2]
mc_4m_dki_vis =mc_4m_dki[mc_4m_dki<0]

mc = np.corrcoef(dfc_stream_2m[0])
# community_structure,q_statistic = bct.modularity.community_louvain(np.abs(mc),gamma=0.9)
community_structure,q_statistic = bct.modularity.modularity_louvain_und_sign(mc,gamma=0.9)
print(np.unique(community_structure),q_statistic)

sorted_community_structure = np.argsort(community_structure)
mc_mod = mc[:,sorted_community_structure][sorted_community_structure,:]


b =[np.mean(abs(mc_2m_dki[xx][mc_2m_dki[xx]<0])) for xx in range(np.sum(index1))]
a = [np.mean(abs(mc_4m_wt[xx][mc_4m_dki[xx]<0])) for xx in range(np.sum(index1))]




#%%
def metaconnectivity(fct):
    """
    Calculate the metaconnectivity from a functional connectivity in time.
    Data strucuture TxNxN (time windows, regions, regions)

    Parameters
    ----------
    fct : TxNxN np.array
        The arrray with T time windows, and N by N nodes.

    Returns
    -------
    metaconn : MxM np.array
        The metaconnectivity of each pair of fct in time W. The dimension is M=N*(N-1)/2
    m_index : np.array
        DESCRIPTION.
    p_index : TYPE
        DESCRIPTION.

    """
    #parameters
    # N = data.shape[1]
    T,N,_=fct.shape #time windows in data
    M = int((N*(N-1))/2)
    
    #index structure
    m_index = np.array(np.triu_indices(N,1)).T
    p_index = np.array(np.triu_indices(M,1)).T
    
    #Metaconnectivity
    pcorr = np.zeros((T,M))
    for t,fc in enumerate(fct):
        pcorr[t] = fc[np.triu_indices_from(fc,1)]
    pcorr=pcorr.T
    
    #metaconnectivity
    metaconn = np.corrcoef(pcorr)
    return metaconn, m_index, p_index

def metaconnectivity_structure(fct):
    #metaconnectivity
    metaconn, m_index, p_index= metaconnectivity(fct)
    #Metaconnectivity data structure
    P,_=  np.shape(p_index)
    _,N,_=fct.shape #time windows in data
    # P,_=  np.shape(m_index)
    
    data_structure = np.zeros((P,9)) #i,j,k,l, apex(1/0-true/false), ap1, ap2, ap3
    data_structure[:,5:] = -1 #i,j,k,l, apex(1/0-true/false), ap1, ap2, ap3
    
    for i in range(P):
        #the data list
        meta_indx = p_index[i]
        ind_mc = np.ndarray.flatten(m_index[meta_indx])
        data_structure[i,:4] = ind_mc 
        print(meta_indx,i)
    
        #Is trimer, or quatrimer? 1 or 0
        data_structure[i,4] = ((len(np.unique(data_structure[i,:4]))<4)*1)
        data_structure[i,8] = np.abs(metaconn[meta_indx[0],meta_indx[1]])
        aux_i = 0
        for xx in ind_mc:
            
            #trimers
            if data_structure[i,4]==1:
                #Apex identity 
                aux_apex = np.sum(np.isin(ind_mc, xx))>1
                if aux_apex==True:
                    data_structure[i,5] = xx
                else:
                    data_structure[i,6+aux_i] = xx
                    aux_i=+1
    return data_structure, metaconn


# b=metaconnectivity_structu1re(ts2dfc_stream(ts2m[0], windows_size, lag, format_data='3D').T)
# c =b[1][np.triu_indices(780,1)]

a=ts2fc(ts2dfc_stream(ts2m[0], windows_size, lag, format_data='2D').T)
a_der = ts2dfc_stream(ts2m[0], windows_size, lag, format_data='2D')

N=ts2m[0].shape[1]
# P = a.shape[0]
P,_ = a_der.shape

m_index = np.array(np.triu_indices(N,1)).T
p_index = np.array(np.triu_indices(P,1)).T

data_structure = np.zeros((len(p_index),9)) #i,j,k,l, apex(1/0-true/false), ap1, ap2, ap3
data_structure[:,5:] = -1 #i,j,k,l, apex(1/0-true/false), ap1, ap2, ap3

for i in range(len(p_index)):
    #the data list
    meta_indx = p_index[i]
    ind_mc = np.ndarray.flatten(m_index[meta_indx])
    # print(meta_indx,i)
    data_structure[i,:4] = ind_mc



#%%
plt.figure(9)
plt.clf()
plt.subplot(311)
# plt.imshow(mc, aspect='auto', interpolation='none',cmap='RdBu_r')
plt.imshow(mc_mod, aspect='auto', interpolation='none',cmap='viridis')
plt.colorbar()

plt.subplot(312)
# plt.hist((mc_2m_wt_vis, mc_2m_dki_vis), histtype='step',bins=10000)
# plt.hist((mc_2m_wt_vis, mc_2m_dki_vis, mc_4m_wt_vis, mc_4m_dki_vis), histtype='step',bins=150)#, log=True)
# plt.hist((mc_2m_wt_vis, mc_2m_dki_vis, mc_4m_wt_vis, mc_4m_dki_vis), density=True, histtype='step',bins=300, log=True, label=('2m wt', '2m dki', '4m wt', '4m dki'))
plt.hist((mc_2m_wt_vis, mc_2m_dki_vis, mc_4m_wt_vis, mc_4m_dki_vis),  histtype='step',bins=300, label=('2m wt', '2m dki', '4m wt', '4m dki'))
# plt.hist((mc_2m_wt_vis, mc_2m_dki_vis), density=True, histtype='step',bins=150)#, log=True)
# plt.clim(-0.5,0.5)
plt.legend(loc=2)





#%%
plt.figure(10)
plt.clf()
plt.subplot(321)
# plt.plot(timeseries)
# plt.imshow(data, aspect='auto')

plt.subplot(322)
plt.title('dFC stream')
plt.imshow(dfc_stream_2m[0], aspect='auto', interpolation='none',cmap='RdBu_r')
plt.colorbar()

plt.subplot(323)
plt.title('fc sorted mod')
plt.imshow(fc_2m_mod[0], aspect='auto', interpolation='none',cmap='RdBu_r')
plt.colorbar()
# plt.clim(-0.5,0.5)
plt.clim(0,0.5)

plt.subplot(324)
plt.title('fcd')
plt.imshow(fcd_2m[0], aspect='auto', interpolation='none',cmap='RdBu_r')
plt.clim(0,0.5)
plt.colorbar()
plt.subplot(325)
plt.title('speed(1-corr)')
plt.hist(speed_oversampl.flatten() ,histtype='step')#, aspect='auto', interpolation='none',cmap='RdBu_r')

plt.subplot(326)
plt.title('Metaconnectivity')
plt.imshow(mc_mod, aspect='auto', interpolation='none',cmap='RdBu_r')
plt.colorbar()
#%%







