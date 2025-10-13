#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 14:42:38 2023

@author: samy
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import brainconn as bct
import os
import numpy.linalg as LA
import scipy.stats 
from scipy.special import kl_div
import time
import pandas as pd
# from functions_analysis import *
from fun_loaddata import *
from fun_dfcspeed import *

from scipy.special import erfc
from scipy.stats import pearsonr, spearmanr

# =============================================================================
# Figure's parameters
# =============================================================================

# Set figure parameters globally
plt.rcParams.update({'axes.labelsize': 15, 'axes.titlesize': 13,
                     # 'axes.spines.left': False, 'axes.spines.bottom': False,
                     'axes.spines.right': False, 'axes.spines.top': False})
# plt.style.use('seaborn-white')
save_fig =False

#%%
# =============================================================================
# Load data - Intersect the functional data for 2 and 4 months
# =============================================================================
# Define paths and folders
root = '/home/samy/Bureau/Proyect/LauraHarsan/Ines/Timecourses_updated/'
folders = {'2mois': 'Lot3_2mois', '4mois': 'Lot3_4mois'}

# Load filenames and hash numbers
filenames       = {period: filename_sort_mat(os.path.join(root, folder)) for period, folder in folders.items()}
hash_numbers    = {period: extract_hash_numbers(filenames[period]) for period in filenames}

int_2m4m = np.intersect1d(hash_numbers['2mois'], hash_numbers['4mois'], return_indices=True)
print('Number of intersected elements in 2m and 4m :' , len(int_2m4m[0]))

# =============================================================================
# Load cognitive data from .xlsx document
# =============================================================================
#Load cognitive data
cog_data_path   = os.path.join(root, 'Behaviour_exclusions_ROIs.xlsx')
cog_data_df     = pd.read_excel(cog_data_path, sheet_name='Feuil1')
data_roi        = pd.read_excel(cog_data_path, sheet_name='40_Allen_ROIs_list').to_numpy()

# Anatomical labels
anat_labels     = np.array([xx[0] for xx in data_roi])

cog_data_df['sexe_label']  = cog_data_df.loc[:,'Sexe']
cog_data_df['gen_label']  = cog_data_df.loc[:,'Genotype']

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
# female_wt_data    = cog_data_filtered[(cog_data_filtered['Sexe'] == 1) & (cog_data_filtered['Genotype'] == 0)]
# female_dki_data   = cog_data_filtered[(cog_data_filtered['Sexe'] == 1) & (cog_data_filtered['Genotype'] == 1)]

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
# =============================================================================
# Optional
# =============================================================================
#Sexe
ts2m_male, ts2m_female  = ts2m[male_index], ts2m[female_index]
#Genotype
ts2m_wt, ts2m_dki     = ts2m[wt_index], ts2m[dki_index]

# =============================================================================
#ERASE THIS!!!!!!!
# Could create justr for this time ts female
# =============================================================================
filename_int2mf = filenames['2mois'][~index_tsintcog[0]]
filename_int4mf = filenames['4mois'][~index_tsintcog[1]]

ts2mf = load_matdata(root, folders['2mois'], filename_int2mf)
ts4mf = load_matdata(root, folders['4mois'], filename_int4mf)

ts2m_female = ts2mf[:,transient:]
ts4m_female = ts4mf[:,transient:]

#%%
# Visualization: Plotting data based on the filters applied
plt.figure(1,figsize=(10, 6))
plt.clf()

plt.subplot(211)
plt.violinplot((male_wt_data['OiP_2M'], male_wt_data['OiP_4M'], male_dki_data['OiP_2M'], male_dki_data['OiP_4M']))
plt.xticks([1, 2, 3, 4], ['Male WT 2M', 'Male WT 4M', 'Male dKI 2M', 'Male dKI 4M'])
plt.axhline(0,c='k')
plt.ylabel('OiP score')
plt.title('Distribution of OiP scores for Male')

plt.subplot(212)
plt.violinplot((male_wt_data['RO24h_2M'], male_wt_data['RO24h_4M'], male_dki_data['RO24h_2M'], male_dki_data['RO24h_4M']))
plt.xticks([1, 2, 3, 4], ['Male WT 2M', 'Male WT 4M', 'Male dKI 2M', 'Male dKI 4M'])
plt.axhline(0,c='k')
plt.ylabel('RO24h score')
plt.title('Distribution of RO24h for Male')
plt.tight_layout()
if save_fig==True:
    plt.savefig('fig/cog_data/oip_ro24h_male_wt_dki.png')
    plt.savefig('fig/cog_data/oip_ro24h_male_wt_dki.pdf')

#%%    
# =============================================================================
# Save data
# =============================================================================

#data saved
data_save = {}
data_save['ts2m'] = ts2m
data_save['ts4m'] = ts4m

data_save['male_index'] = male_index
data_save['female_index'] = female_index
data_save['wt_index'] = wt_index
data_save['dki_index'] = dki_index

data_save['mouse_hash'] = mouse_hash_cog
data_save['gen_label'] = gen_label
data_save['sex_label'] = sex_label
data_save['cog_data'] = cog_data_filtered
data_save['anat_labels'] = anat_labels
#%%






# =============================================================================
# FC and modularity
# -There is another modularity algorithm which claims to be better than Louvain. It's called leiden algorithm
#https://www.nature.com/articles/s41598-019-41695-z
# =============================================================================
















#%%
# =============================================================================
# static FC analysis
# =============================================================================
#Functional connectivity
fc_2m = np.array([ts2fc(ts2m[xx]) for xx in range(n_animals)])
fc_4m = np.array([ts2fc(ts4m[xx]) for xx in range(n_animals)])

#Modularity
fc_2m_mod = np.array([sort_modularity(fc_2m[xx]) for xx in range(n_animals)])
fc_4m_mod = np.array([sort_modularity(fc_4m[xx]) for xx in range(n_animals)])

#superior triangular (maybe fucntion)
ind_fctri_2m = np.triu_indices(fc_2m.shape[2],1)
ind_fctri_4m = np.triu_indices(fc_4m.shape[2],1)

#Vector of correlations
tri_2m = np.array([fc_2m[tt, ind_fctri_2m[0], ind_fctri_2m[1]] for tt in range(n_animals)])
tri_4m = np.array([fc_4m[tt, ind_fctri_4m[0], ind_fctri_4m[1]] for tt in range(n_animals)])

#%%
# # Plot individual analysis of static fc
for idx_mice in range(n_animals):
# for idx_mice in range(2):
    
    aux_ts2m = ts2m[idx_mice]
    aux_ts4m = ts4m[idx_mice]
    
    plt.figure(2, figsize=(9,10))
    plt.clf()
    plt.subplot(321)
    plt.title('2m mouse #%s %s %s'%(mouse_hash_cog[idx_mice], gen_label[idx_mice], sex_label[idx_mice]))
    plt.plot(ts2m[idx_mice])
    plt.ylabel('Bold')
    plt.xlabel('time')
    
    plt.subplot(322)
    plt.title('4m mouse #%s %s %s'%(mouse_hash_cog[idx_mice], gen_label[idx_mice], sex_label[idx_mice]))
    plt.plot(ts4m[idx_mice])
    plt.ylabel('Bold')
    plt.xlabel('time')
    
    plt.subplot(323)
    plt.title('FC')
    plt.imshow(fc_2m_mod[idx_mice], aspect='auto', interpolation='none',cmap='RdBu_r')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("regions")
    plt.ylabel("regions")
    
    plt.subplot(324)
    plt.title('FC')
    plt.imshow(fc_4m_mod[idx_mice], aspect='auto', interpolation='none',cmap='RdBu_r')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("regions")
    plt.ylabel("regions")
    
    plt.subplot(325)
    # Fit linear regression via least squares with numpy.polyfit
    # It returns an slope (b) and intercept (a)
    # deg=1 means linear fit (i.e. polynomial of degree 1)
    # Create sequence of 100 numbers from 0 to 100 
    b,a=np.polyfit(tri_2m[idx_mice], tri_4m[idx_mice],deg=1)
    xseq = np.linspace(-1, 1, num=100)
    
    plt.title('slope:%s'%np.round(b,3))
    plt.scatter(tri_2m[idx_mice], tri_4m[idx_mice])
    
    # Plot regression line
    plt.plot(xseq, a + b * xseq, color="k", lw=2.5);
    
    plt.xlabel('2m')
    plt.ylabel('4m')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    
    
    plt.subplot(326)
    
    plt.hist((tri_2m[idx_mice], tri_4m[idx_mice]),histtype='step',bins=50)
    plt.legend(('2m','4m'))
    plt.xlabel('CC')
    plt.ylabel('Counts #')
    
    plt.tight_layout()
    if save_fig ==True:
        plt.savefig('fig/fc/mouse_#%s_%s_%s.png'%(mouse_hash_cog[idx_mice], gen_label[idx_mice], sex_label[idx_mice]))
        plt.savefig('fig/fc/mouse_#%s_%s_%s.pdf'%(mouse_hash_cog[idx_mice], gen_label[idx_mice], sex_label[idx_mice]))
#%%
# =============================================================================
# Speed analysis

# Compute the dfc speed distribution using wondow oversampling method for each animal. Also retrieve median speed for each tau, in multiple W, for each animal
# =============================================================================
window_parameter = (5,80,1)
lag=1
tau=4
tau_array       = np.append(np.arange(-tau,tau), tau ) 

vel_list2m = []
vel_list4m = []
speed_median_2m = []
speed_median_4m = []

start = time.time()
for xx in ts2m:
    aux_speed_median, aux_speed_dist  = parallel_dfc_speed_oversampled_series(xx, window_parameter, lag,tau=tau, get_speed_dist=True)
    vel_list2m.append(aux_speed_dist)  
    speed_median_2m.append(aux_speed_median)

for xx in ts4m:
    aux_speed_median, aux_speed_dist  = parallel_dfc_speed_oversampled_series(xx, window_parameter, lag,tau=tau, get_speed_dist=True)
    vel_list4m.append(aux_speed_dist)  
    speed_median_4m.append(aux_speed_median)

speed_median_2m = np.array(speed_median_2m)
speed_median_4m = np.array(speed_median_4m)

stop= time.time()
print('speed dist windows oversampling analysis time', stop-start,'s')

#%%
# =============================================================================
# Windows pooling of window oversampled speeds 
# =============================================================================

lentau = len(tau_array)

#For the dfc speed distribution window oversampling, get a windows pooling
aux_short2m = np.array([np.hstack(vel_list2m[xx][0*lentau:10*lentau]) for xx in range(n_animals)])
aux_mid2m = np.array([np.hstack(vel_list2m[xx][10*lentau:31*lentau]) for xx in range(n_animals)])
# aux_long2m = np.array([np.hstack(vel_list2m[xx][31*lentau:61*lentau]) for xx in range(n_animals)])
aux_long2m = np.array([np.hstack(vel_list2m[xx][31*lentau:]) for xx in range(n_animals)])

aux_short4m = np.array([np.hstack(vel_list4m[xx][0*lentau:10*lentau]) for xx in range(n_animals)])
aux_mid4m = np.array([np.hstack(vel_list4m[xx][10*lentau:31*lentau]) for xx in range(n_animals)])
# aux_long4m = np.array([np.hstack(vel_list4m[xx][31*lentau:61*lentau]) for xx in range(n_animals)])
aux_long4m = np.array([np.hstack(vel_list4m[xx][31*lentau:]) for xx in range(n_animals)])

window_pooling_speed_wt2m = (np.hstack(aux_short2m[wt_index]), np.hstack(aux_mid2m[wt_index]), np.hstack(aux_long2m[wt_index]))
window_pooling_speed_dki2m = (np.hstack(aux_short2m[dki_index]), np.hstack(aux_mid2m[dki_index]), np.hstack(aux_long2m[dki_index]))

window_pooling_speed_wt4m = (np.hstack(aux_short4m[wt_index]), np.hstack(aux_mid4m[wt_index]), np.hstack(aux_long4m[wt_index]))
window_pooling_speed_dki4m = (np.hstack(aux_short4m[dki_index]), np.hstack(aux_mid4m[dki_index]), np.hstack(aux_long4m[dki_index]))

plt.figure(8, figsize=(12,10))
plt.clf()
vel_label = ('10-30s (short)','30-72s (mid)','72-160s (long)')

for i in range(3):
    plt.subplot(3,2,2*i+1)
    plt.title('%s'%vel_label[i])
    plt.hist((window_pooling_speed_wt2m[i], window_pooling_speed_dki2m[i],window_pooling_speed_wt4m[i], window_pooling_speed_dki4m[i]),label=('2m wt', '2m dki', '4m wt', '4m dki'), histtype='step',bins=200, density=True)
    # plt.hist((window_pooling_speed_wt2m[i], window_pooling_speed_dki2m[i],window_pooling_speed_wt4m[i], window_pooling_speed_dki4m[i]),label=('2m wt', '2m dki', '4m wt', '4m dki'), histtype='step',bins=500, density=True, log=True)
    plt.ylabel('Counts')

    plt.subplot(3,2,2*i+2)
    plt.hist((window_pooling_speed_wt2m[i], window_pooling_speed_dki2m[i],window_pooling_speed_wt4m[i], window_pooling_speed_dki4m[i]),label=('2m wt', '2m dki', '4m wt', '4m dki'), histtype='step',bins=500, density=True, log=True)
    plt.xlim(0.2,1.2)
plt.xlabel('Freq[v]')
plt.legend()
plt.tight_layout()

if save_fig ==True:
    plt.savefig('fig/speed/speed_window_pooling_and_oversampling_male_dki_vs_wt_lag=%s_tau=%s.png'%(lag,tau))
    plt.savefig('fig/speed/speed_window_pooling_and_oversampling_male_dki_vs_wt_lag=%s_tau=%s.pdf'%(lag,tau))
#%%
# =============================================================================
# Calculate the FCD and dFC_stream for a given W
# =============================================================================
#Windows FCD
windows_size = 30
lag = 1

dfc_stream_2m   = np.array([ts2dfc_stream(ts2m[xx], windows_size, lag, format_data='2D') for xx in range(n_animals)])
fcd_2m          = np.array([dfc_stream2fcd(dfc_stream_2m[xx]) for xx in range(n_animals)])

dfc_stream_4m   = np.array([ts2dfc_stream(ts4m[xx], windows_size, lag, format_data='2D') for xx in range(n_animals)])
fcd_4m          = np.array([dfc_stream2fcd(dfc_stream_4m[xx]) for xx in range(n_animals)])

#%%
# =============================================================================
# Velocity statistics
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
vel_list_all = ((short2m_statistics,short4m_statistics), (mid2m_statistics, mid4m_statistics), (long2m_statistics,long4m_statistics))

label_statistic = ('v_median', 'v_typical', 'q5', 'q95')
label_vel = ('short', 'mid', 'long')

for idx_vel, vel_sts in enumerate(vel_list_all):
    print(idx_vel,np.shape(vel_sts[0]))

    for xx in range(len(label_statistic)):
    
        plt.figure(11,figsize=(8,6))
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
# Speead average and cognitive metrics
#-chauvenet criterion
#-person
# =============================================================================
# statistics_df = pd.DataFrame()

oip_dki,ro24_dki = np.array(abs_cog_data(male_dki_data))
oip_wt, ro24_wt = np.array(abs_cog_data(male_wt_data))

oip_saux_dki = pd.Series(oip_dki,index=mouse_hash_cog[dki_index],name='delta_oip')
oip_saux_wt = pd.Series(oip_wt,index=mouse_hash_cog[wt_index],name='delta_oip')

ro24_saux_wt = pd.Series(ro24_wt,index=mouse_hash_cog[wt_index],name='delta_ro24')
ro24_saux_dki = pd.Series(ro24_dki,index=mouse_hash_cog[dki_index],name='delta_ro24')

statistic_df = pd.DataFrame(pd.concat( [oip_saux_wt,oip_saux_dki],axis=0))
statistic_df['delta_ro24'] = pd.concat( [ro24_saux_wt,ro24_saux_dki],axis=0)

statistic_df['genotype'] = pd.concat([pd.Series(np.ones(len(mouse_hash_cog[dki_index])), index=mouse_hash_cog[dki_index]),
                                     pd.Series(np.zeros(len(mouse_hash_cog[wt_index])), index=mouse_hash_cog[wt_index])]
                                     , axis=0)
#%%

aux_short2m4m = (np.array(short4m_statistics)-np.array(short2m_statistics)) / (np.array(short4m_statistics)+np.array(short2m_statistics))
aux_mid2m4m = (np.array(mid4m_statistics)-np.array(mid2m_statistics)) / (np.array(mid4m_statistics)+np.array(mid2m_statistics))
aux_long2m4m = (np.array(long4m_statistics)-np.array(long2m_statistics)) / (np.array(long4m_statistics)+np.array(long2m_statistics))


short_df = pd.DataFrame(aux_short2m4m.T, 
                    columns=['v_median_short2m4m','v_typ_short2m4m','v_q5_short2m4m','v_q95_short2m4m'], 
                    index=mouse_hash_cog
                    )
mid_df = pd.DataFrame(aux_mid2m4m.T, 
                    columns=['v_median_mid4m','v_typ_mid4m','v_q5_mid4m','v_q95_mid4m'], 
                    index=mouse_hash_cog
                   )
long_df = pd.DataFrame(aux_long2m4m.T, 
                    columns=['v_median_long4m','v_typ_long4m','v_q5_long4m','v_q95_long4m'], 
                    index=mouse_hash_cog
                   )

c = pd.merge(short_df, mid_df, left_index=True,right_index=True)
d = pd.merge(long_df,c, left_index=True,right_index=True)
statistics = pd.merge(statistic_df,d, left_index=True,right_index=True)
# statistic_df['v_median_short2m']

wt_sts = statistics[statistics.genotype ==0]
dki_sts = statistics[statistics.genotype ==1]

#%%
cog =male_wt_data#male_dki_data
cog =male_dki_data
# cog_data_abs2m = (cog['OiP_2M'])
# cog_data_abs4m = (cog['OiP_4M'])
cog_data_abs2m = (cog['RO24h_2M'])
cog_data_abs4m = (cog['RO24h_4M'])

plt.figure(12,figsize=(8,5))
plt.clf()
plt.subplot(121)
plt.title('OiP')
# plt.scatter(np.log2(cog_data_abs2m), np.log2(cog_data_abs4m))
plt.scatter(male_dki_data['OiP_2M'], male_dki_data['OiP_4M'], label='dki')
plt.scatter(male_wt_data['OiP_2M'], male_wt_data['OiP_4M'], label='wt')
plt.axhline(0.2,color='k')
plt.axvline(0.2,color='k')

plt.ylabel('4M')
plt.xlabel('2M')
plt.legend()

plt.subplot(122)
plt.title('RO24h')
# plt.scatter(np.log2(cog_data_abs2m), np.log2(cog_data_abs4m))
plt.scatter(male_dki_data['RO24h_2M'], male_dki_data['RO24h_4M'], label='dki')
plt.scatter(male_wt_data['RO24h_2M'], male_wt_data['RO24h_4M'], label='wt')
plt.axhline(0.2,color='k')
plt.axvline(0.2,color='k')
plt.ylabel('4M')
plt.xlabel('2M')
plt.legend()


oip_dki_impaired = (male_dki_data['OiP_2M']>0.1)&(male_dki_data['OiP_4M']<0.1)
ro_dki_impaired = (male_dki_data['RO24h_2M']>0.2)&(male_dki_data['RO24h_4M']<0.2)
#%%

for idx_vel, vel_sts in enumerate(vel_list_all):

    for xx in range(len(label_statistic)):
        statistic = xx
        var2m = vel_sts[0][statistic]#long2m_statistics[0]
        var4m = vel_sts[1][statistic]#long4m_statistics[0]
        velmedian2m4m = np.array((var2m, var4m))

        delta_vel2m4m = (np.array(var4m) - np.array(var2m)) / (np.array(var4m) + np.array(var2m))

        oip_dki,ro24_dki = abs_cog_data(male_dki_data)
        oip_wt, ro24_wt = abs_cog_data(male_wt_data)
        
        
        oip_dki = oip_dki[oip_dki_impaired]
        ro24_dki = ro24_dki[ro_dki_impaired]

        b_dki,a_dki = np.polyfit(delta_vel2m4m[dki_index][oip_dki_impaired], oip_dki,deg=1)
        b_dki_ro,a_dki_ro = np.polyfit(delta_vel2m4m[dki_index][ro_dki_impaired], ro24_dki,deg=1)
        
        xseq = np.linspace(np.min(delta_vel2m4m), np.max(delta_vel2m4m), num=100)
        
        plt.figure(12,figsize=(8,5))
        plt.clf()
        
        plt.subplot(121)
        plt.title('%s %s'%(label_statistic[statistic], label_vel[idx_vel]))
        plt.scatter(delta_vel2m4m[dki_index][oip_dki_impaired], oip_dki, label='dki slope = %s'%np.round(b_dki,3))
        plt.axhline(0,color='k')
        plt.axvline(0,color='k')
        
        plt.plot(xseq, a_dki + b_dki * xseq, color="C1")
        
        plt.ylabel(r'$\Delta <OiP_{4m-2m}>$')
        plt.xlabel(r'$\Delta <v_{4m-2m}>$')
        plt.legend()

        plt.subplot(122)
        plt.title('%s %s'%(label_statistic[statistic], label_vel[idx_vel]))
        plt.scatter(delta_vel2m4m[dki_index][ro_dki_impaired], ro24_dki, label='dki slope = %s'%np.round(b_dki,3))
        plt.axhline(0,color='k')
        plt.axvline(0,color='k')
        
        plt.plot(xseq, a_dki + b_dki * xseq, color="C1")
        
        plt.ylabel(r'$\Delta <OiP_{4m-2m}>$')
        plt.xlabel(r'$\Delta <v_{4m-2m}>$')
        plt.legend()
        
        plt.tight_layout()
        
        if save_fig ==True:
            plt.savefig('fig/statistics/velcog_dki_impaired_statistics2_%s_%s.png'%(label_statistic[statistic], label_vel[idx_vel]))
            # plt.savefig('fig/statistics/velcog_statistics2_%s_%s.pdf'%(label_statistic[statistic], label_vel[idx_vel]))


#%%


def abs_cog_data(cog_data):
    # cog_data_abs2m = (cog_data['OiP_2M'] + 1)/2
    # cog_data_abs4m = (cog_data['OiP_4M'] + 1)/2

    # RO24_abs2m = (cog_data['RO24h_2M'] + 1)/2
    # RO24h_abs4m = (cog_data['RO24h_4M'] + 1)/2

    # oip = (cog_data_abs4m-cog_data_abs2m)/(cog_data_abs4m+cog_data_abs2m)
    # ro24 = (RO24h_abs4m-RO24_abs2m)/ (RO24h_abs4m+RO24_abs2m)
    cog_data_abs2m = (cog_data['OiP_2M'])
    cog_data_abs4m = (cog_data['OiP_4M'])

    RO24_abs2m = (cog_data['RO24h_2M'])
    RO24h_abs4m = (cog_data['RO24h_4M'])
    
    # oip = (cog_data_abs4m+cog_data_abs2m)/2#(cog_data_abs4m+cog_data_abs2m)
    # ro24 = (RO24h_abs4m+RO24_abs2m)/ 2#(RO24h_abs4m+RO24_abs2m)
    oip = (cog_data_abs4m-cog_data_abs2m)#/(cog_data_abs4m+cog_data_abs2m)
    ro24 = (RO24h_abs4m-RO24_abs2m)#/ (RO24h_abs4m+RO24_abs2m)
    return oip,ro24

for idx_vel, vel_sts in enumerate(vel_list_all):
    print(idx_vel,np.shape(vel_sts))

    for xx in range(len(label_statistic)):
        statistic = xx
        var2m = vel_sts[0][statistic]#long2m_statistics[0]
        var4m = vel_sts[1][statistic]#long4m_statistics[0]
        velmedian2m4m = np.array((var2m, var4m))

        delta_vel2m4m = (np.array(var4m) - np.array(var2m)) / (np.array(var4m) + np.array(var2m))

        oip_dki,ro24_dki = abs_cog_data(male_dki_data)
        oip_wt, ro24_wt = abs_cog_data(male_wt_data)
        
        
        b_wt,a_wt = np.polyfit(delta_vel2m4m[wt_index], oip_wt,deg=1)
        b_dki,a_dki = np.polyfit(delta_vel2m4m[dki_index], oip_dki,deg=1)
        
        b_wt_ro,a_wt_ro = np.polyfit(delta_vel2m4m[wt_index], ro24_wt,deg=1)
        b_dki_ro,a_dki_ro = np.polyfit(delta_vel2m4m[dki_index], ro24_dki,deg=1)
        # xseq = np.linspace(-0.3, 0.2, num=100)
        xseq = np.linspace(np.min(delta_vel2m4m), np.max(delta_vel2m4m), num=100)
        
        plt.figure(12,figsize=(8,5))
        plt.clf()
        
        plt.subplot(121)
        plt.title('%s %s'%(label_statistic[statistic], label_vel[idx_vel]))
        plt.scatter(delta_vel2m4m[wt_index], oip_wt, label='wt slope = %s'%np.round(b_wt,3))
        plt.scatter(delta_vel2m4m[dki_index], oip_dki, label='dki slope = %s'%np.round(b_dki,3))
        plt.axhline(0,color='k')
        plt.axvline(0,color='k')
        
        plt.plot(xseq, a_wt + b_wt * xseq, color="C0")
        plt.plot(xseq, a_dki + b_dki * xseq, color="C1")
        
        plt.ylabel(r'$\Delta <OiP_{4m-2m}>$')
        plt.xlabel(r'$\Delta <v_{4m-2m}>$')
        plt.legend()
        
        plt.subplot(122)
        plt.scatter(delta_vel2m4m[wt_index],ro24_wt, label='wt slope = %s'%np.round(b_wt_ro,3))
        plt.scatter(delta_vel2m4m[dki_index],ro24_dki, label='dki slope = %s'%np.round(b_dki_ro,3))
        plt.axhline(0,color='k')
        plt.axvline(0,color='k')
        
        plt.plot(xseq, a_wt_ro + b_wt_ro * xseq, color="C0")
        plt.plot(xseq, a_dki_ro + b_dki_ro * xseq, color="C1")
        plt.ylabel(r'$\Delta <RO24_{4m-2m}>$')
        plt.xlabel(r'$\Delta <v_{4m-2m}>$')
        plt.legend()
        
        plt.tight_layout()
        
        if save_fig ==True:
            plt.savefig('fig/statistics/velcog_statistics2_%s_%s.png'%(label_statistic[statistic], label_vel[idx_vel]))
            plt.savefig('fig/statistics/velcog_statistics2_%s_%s.pdf'%(label_statistic[statistic], label_vel[idx_vel]))


#%%
# =============================================================================
# Chauvenet criterion of exclusion
# =============================================================================
oip_dki,ro24_dki = abs_cog_data(male_dki_data)
oip_wt, ro24_wt = abs_cog_data(male_wt_data)


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


mask_chauvenet_oip_wt = np.array(chauvenet(oip_wt)) | np.array(chauvenet(delta_vel2m4m[wt_index]))
mask_chauvenet_oip_dki = np.array(chauvenet(oip_dki)) | np.array(chauvenet(delta_vel2m4m[dki_index]))

mask_chauvenet_ro_wt = np.array(chauvenet(ro24_wt)) | np.array(chauvenet(delta_vel2m4m[wt_index]))
mask_chauvenet_ro_dki = np.array(chauvenet(ro24_dki)) | np.array(chauvenet(delta_vel2m4m[dki_index]))


#%%

idx_sts=0

idx_v=0
for idx_sts in range(len(label_statistic)):
    for idx_v in range(len(label_vel)):
        statistic = idx_sts
        vel_sts = vel_list_all[idx_v]
        
        var2m = vel_sts[0][statistic]#long2m_statistics[0]
        var4m = vel_sts[1][statistic]#long4m_statistics[0]

        delta_vel2m4m = (np.array(var4m) - np.array(var2m)) / (np.array(var4m) + np.array(var2m))

        mask_chauvenet_oip_wt = np.array(chauvenet(oip_wt)) | np.array(chauvenet(delta_vel2m4m[wt_index]))
        mask_chauvenet_oip_dki = np.array(chauvenet(oip_dki)) | np.array(chauvenet(delta_vel2m4m[dki_index]))
        
        mask_chauvenet_ro_wt = np.array(chauvenet(ro24_wt)) | np.array(chauvenet(delta_vel2m4m[wt_index]))
        mask_chauvenet_ro_dki = np.array(chauvenet(ro24_dki)) | np.array(chauvenet(delta_vel2m4m[dki_index]))
        
        
        # cc_array = np.zeros((2,2))
        cc_array = np.zeros((2,4,2))
        cc_array_wo_outl = np.zeros((2,4,2))
        
        cc_used = (pearsonr,spearmanr)
        
        for idx_cc, ccr in enumerate(cc_used):
            print(idx_cc)#ccr(delta_vel2m4m_wt[~mask_chauvenet_oip_wt], oip_wt[~mask_chauvenet_oip_wt]))
            cc_array[idx_cc,0] = ccr(delta_vel2m4m_wt, oip_wt)
            cc_array[idx_cc,1] = ccr(delta_vel2m4m[dki_index], oip_dki)
            cc_array[idx_cc,2] = ccr(delta_vel2m4m_wt, ro24_wt)
            cc_array[idx_cc,3] = ccr(delta_vel2m4m[dki_index], ro24_dki)
        
            cc_array_wo_outl[idx_cc,0] = ccr(delta_vel2m4m_wt[~mask_chauvenet_oip_wt], oip_wt[~mask_chauvenet_oip_wt])
            cc_array_wo_outl[idx_cc,1] = ccr(delta_vel2m4m[dki_index][~mask_chauvenet_oip_dki], oip_dki[~mask_chauvenet_oip_dki])
            cc_array_wo_outl[idx_cc,2] = ccr(delta_vel2m4m_wt[~mask_chauvenet_ro_wt], ro24_wt[~mask_chauvenet_ro_wt])
            cc_array_wo_outl[idx_cc,3] = ccr(delta_vel2m4m[dki_index][~mask_chauvenet_ro_dki], ro24_dki[~mask_chauvenet_ro_dki])
            # cc_array[idx_cc] = aux_cc_array
            # cc_array[idx,0,1] = (ccr(delta_vel2m4m_wt[~mask_chauvenet_oip_wt], oip_wt[~mask_chauvenet_oip_wt]))[1]
        
        label_statistic = ('v_median', 'v_typical', 'q5', 'q95')
        label_vel = ('short', 'mid', 'long')
        
        plt.figure(999)
        plt.clf()
        plt.subplot(221)
        plt.title('Pearson %s %s'%(label_statistic[idx_sts], label_vel[idx_v]))
        plt.imshow(cc_array[0],aspect='auto', cmap='tab10')
        plt.xticks([0,1], ('cc', 'p_v'))
        plt.yticks([0,1,2,3], ('2m4m_wt_oip', '2m4m_dki_oip', '2m4m_wt_ro24', '2m4m_dki_ro24'),rotation=20)
        plt.colorbar()
        plt.clim(0,0.5)
        
        plt.subplot(222)
        plt.title('Pearson w/out')
        plt.imshow(cc_array_wo_outl[0],aspect='auto', cmap='tab10')
        plt.xticks([0,1], ('cc', 'p_v'))
        plt.yticks([])
        plt.colorbar()
        
        plt.subplot(223)
        plt.title('Spearman')
        plt.imshow(cc_array[1],aspect='auto', cmap='tab10')
        plt.xticks([0,1], ('cc', 'p_v'))
        plt.yticks([0,1,2,3], ('2m4m_wt_oip', '2m4m_dki_oip', '2m4m_wt_ro24', '2m4m_dki_ro24'),rotation=20)
        plt.colorbar()
        
        
        plt.subplot(224)
        plt.title('Spearman w/out')
        plt.imshow(cc_array_wo_outl[0],aspect='auto', cmap='tab10')
        plt.xticks([0,1], ('cc', 'p_v'))
        plt.yticks([])
        # plt.yticks([0,1,2,3], ('2m4m_wt_oip', '2m4m_dki_oip', '2m4m_wt_ro24', '2m4m_dki_ro24'))
        plt.colorbar()
        
        plt.savefig('fig/statistics/spearman_pearson_funcog_%s_%s.png'%(label_statistic[idx_sts], label_vel[idx_v]))
#%%
#define array of data values
from scipy.stats import bootstrap
idx=1

#convert array to sequence
plt.figure(98765)
plt.clf()
label_cogfun= ('oip_wt','oip_dki','ro24_wt','ro24_dki', 'delta2m4m_wt','delta2m4m_dki')
data_cogfun = ((oip_wt,), (oip_dki,), (ro24_wt,), (ro24_dki,), (delta_vel2m4m[wt_index],), (delta_vel2m4m[dki_index],))
for idx, data in enumerate(data_cogfun):

    mat_idx=2
    
    mat_label= ('std','mean','median')
    aux=(np.std, np.mean, np.median)
    for idx_mat, mat_obj in enumerate(aux):
    
        #calculate 95% bootstrapped confidence interval for median
        # bootstrap_ci = bootstrap(data[idx], mat_obj[mat_idx], confidence_level=0.975, method='bca')
        bootstrap_ci = bootstrap(data, mat_obj, confidence_level=0.975, method='bca')
        
        #view 95% boostrapped confidence interval
        print(np.shape(bootstrap_ci.bootstrap_distribution))
        print(mat_obj(data))
        
        # ConfidenceInterval(low=10.0, high=20.0)
        plt.subplot(3,6, (idx_mat*6) +1+ (idx))
        plt.title('%s %s'%(mat_label[idx_mat], label_cogfun[idx]))
        plt.hist(bootstrap_ci.bootstrap_distribution, bins=50)
        # plt.axhline()
        plt.axvline(bootstrap_ci.confidence_interval[0])
        plt.axvline(bootstrap_ci.confidence_interval[1])
        plt.axvline(mat_obj(data),color='k')
        # print(np.median(data))

#%%


#%%
# =============================================================================
# Metaconnectivity
# =============================================================================


#metaconnectivity
mc = np.corrcoef(dfc_stream_2m[1])
# mc = np.corrcoef(Pcorr)

# community_structure,q_statistic = bct.modularity.community_louvain(np.abs(mc),gamma=0.9)
community_structure,q_statistic = bct.modularity.modularity_louvain_und_sign(mc,gamma=1.25)
print(np.unique(community_structure),q_statistic)

sorted_community_structure = np.argsort(community_structure)

mc_mod = mc[:,sorted_community_structure][sorted_community_structure,:]


plt.figure(14)
plt.clf()
# plt.imshow(mc, aspect='auto', interpolation='none',cmap='RdBu_r')
plt.imshow(mc_mod, aspect='auto', interpolation='none',cmap='viridis')
plt.colorbar()
# plt.clim(-0.5,0.5)





#%%
plt.figure(15)
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




