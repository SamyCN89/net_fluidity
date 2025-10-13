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
save_fig =True

# Define paths, folders and hash
root = '/home/samy/Bureau/Proyect/LauraHarsan/Ines/Timecourses_updated/'
folders = {'2mois': 'Lot3_2mois', '4mois': 'Lot3_4mois'}
folder_results = '/home/samy/Bureau/Proyect/LauraHarsan/Ines/results/'

#Parameters speed
window_parameter = (5,100,1)
lag=1
tau=3
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
    
    plt.figure(1, figsize=(9,10))
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
# # Population plot of static fc
plt.figure(2,figsize=(8,8))
plt.clf()

plt.subplot(211)
b1,a1=np.polyfit((tri_2m[dki_index]).flatten(), (tri_4m[dki_index].flatten()),deg=1)
b2,a2=np.polyfit((tri_2m[wt_index]).flatten(), (tri_4m[wt_index].flatten()),deg=1)
xseq = np.linspace(-1, 1, num=100)

plt.title('dKi vs wt ')
plt.scatter(tri_2m[dki_index], tri_4m[dki_index],marker='.',label='dki slope=%s'%np.round(b1,3))
plt.scatter(tri_2m[wt_index], tri_4m[wt_index],marker='.', alpha=0.5,label='wt slope=%s'%np.round(b2,3))

plt.plot(xseq, a1 + b1 * xseq, color="C0", lw=2.5)
plt.plot(xseq, a2 + b2 * xseq, color="C1", lw=2.5)
plt.xlabel('2m')
plt.ylabel('4m')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.legend()

plt.subplot(212)
plt.hist((tri_2m[dki_index].flatten(), tri_2m[wt_index].flatten(), tri_4m[dki_index].flatten(), tri_4m[wt_index].flatten()),histtype='step',bins=100)
plt.legend(('2m_dki', '2m_wt', '4m_dki','4m_wt'))
plt.xlabel('Pairwise correlation')
plt.ylabel('Counts #')

plt.tight_layout()
if save_fig ==True:
    plt.savefig('fig/fc/fc_all_dki_vs_wt.png')
    plt.savefig('fig/fc/fc_all_dki_vs_wt.pdf')

#%%

# =============================================================================
# Calculate the FCD and dFC_stream for a given W
# =============================================================================
#Windows FCD
windows_size = 30
lag = 1
hash_parameters = ('lag=%s_wlength=%s'%(lag, windows_size))

start = time.time()
dfc_stream_2m   = np.array([ts2dfc_stream(ts2m[xx], windows_size, lag, format_data='2D') for xx in range(n_animals)])
fcd_2m          = np.array([dfc_stream2fcd(dfc_stream_2m[xx]) for xx in range(n_animals)])

dfc_stream_4m   = np.array([ts2dfc_stream(ts4m[xx], windows_size, lag, format_data='2D') for xx in range(n_animals)])
fcd_4m          = np.array([dfc_stream2fcd(dfc_stream_4m[xx]) for xx in range(n_animals)])
stop = time.time()
print(stop-start)

mc_4m = np.array([ts2fc(dfc_stream_4m[xx].T) for xx in range(n_animals)])
mc_4m_mod = np.array([sort_modularity(mc_4m[xx]) for xx in range(n_animals)])

mc_2m = np.array([ts2fc(dfc_stream_2m[xx].T) for xx in range(n_animals)])
mc_2m_mod = np.array([sort_modularity(mc_2m[xx]) for xx in range(n_animals)])
#%%
for idx_mice in range(n_animals):
# for idx_mice in range(2):
    
    plt.figure(3, figsize=(9,10))

    plt.clf()
    plt.subplot(321)
    # plt.title('4m mouse #%s %s %s'%(mouse_hash_cog[idx_mice], gen_label[idx_mice], sex_label[idx_mice])
    plt.title('dFC stream 2m mouse #%s %s %s'%(mouse_hash_cog[idx_mice], gen_label[idx_mice], sex_label[idx_mice]))
    plt.imshow(dfc_stream_2m[idx_mice], aspect='auto', interpolation='none',cmap='RdBu_r')
    plt.ylabel(r'(region$_{i}$,region$_{j}$)')
    plt.xlabel(r't$_{w}$')
    plt.colorbar()
    
    plt.subplot(322)
    plt.title('dFC stream 4m mouse #%s %s %s'%(mouse_hash_cog[idx_mice], gen_label[idx_mice], sex_label[idx_mice]))
    plt.imshow(dfc_stream_4m[idx_mice], aspect='auto', interpolation='none',cmap='RdBu_r')
    plt.ylabel(r'(region$_{i}$,region$_{j}$)')
    plt.xlabel(r't$_{w}$')
    plt.colorbar()

    plt.subplot(323)
    plt.title('FCD (W=%s, lag=%s)'%(windows_size,lag))
    plt.imshow(fcd_2m[idx_mice], aspect='auto', interpolation='none',cmap='jet')
    plt.xlabel(r't$_{b}$')
    plt.ylabel(r't$_{a}$')
    plt.clim(0,0.5)
    plt.colorbar()
    
    plt.subplot(324)
    plt.title('FCD (W=%s, lag=%s)'%(windows_size,lag))
    plt.imshow(fcd_4m[idx_mice], aspect='auto', interpolation='none',cmap='jet')
    plt.xlabel(r't$_{b}$')
    plt.ylabel(r't$_{a}$')
    plt.clim(0,0.5)
    plt.colorbar()
    
    plt.subplot(325)
    plt.title('Metaconnectivity')
    plt.imshow(mc_2m_mod[idx_mice], aspect='auto', interpolation='none',cmap='RdBu')
    plt.ylabel(r'CC$_{ij}$,CC$_{kl}$')
    plt.xlabel(r'$W$')
    plt.clim(-0.75,0.75)
    plt.colorbar()

    plt.subplot(326)
    plt.title('Metaconnectivity')
    plt.imshow(mc_4m_mod[idx_mice], aspect='auto', interpolation='none',cmap='RdBu')
    plt.ylabel(r'CC$_{ij}$,CC$_{kl}$')
    plt.xlabel(r'$W$')
    plt.clim(-0.75,0.75)
    plt.colorbar()
    
    plt.tight_layout()
    if save_fig ==True:
        plt.savefig('fig/fcd/fcd_mouse_#%s_%s.png'%(mouse_hash_cog[idx_mice], hash_parameters))
        plt.savefig('fig/fcd/fcd_mouse_#%s_%s.pdf'%(mouse_hash_cog[idx_mice], hash_parameters))
