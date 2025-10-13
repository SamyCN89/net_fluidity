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

# =============================================================================
# Figure's parameters
# =============================================================================

# Set figure parameters globally
plt.rcParams.update({'axes.labelsize': 15, 'axes.titlesize': 13,
                     # 'axes.spines.left': False, 'axes.spines.bottom': False,
                     'axes.spines.right': False, 'axes.spines.top': False})

# plt.style.use('seaborn-white')
save_fig =True

# =============================================================================
# Funcitons - to move
# =============================================================================

# def load_matdata(folder_data, specific_folder, files_name):
#     ts_list = []
#     hash_dir        = os.path.join(folder_data, specific_folder)

#     for idx,file_name in enumerate(files_name):
#         file_path       = os.path.join(hash_dir, file_name)
        
#         try:
#             data = loadmat(file_path)['tc']
#             ts_list.append(data)
#         except Exception as e:
#             print(f"Error loading data from {file_path}: {e}")
    
    
#     # Check if the first dimension is consistent
#     first_dim_size = ts_list[0].shape[0]
#     if all(data.shape[0] == first_dim_size for data in ts_list):
#         # Convert the list to a NumPy array
#         ts_array = np.array(ts_list)
#         return ts_array
#     else:
#         print("Error: Inconsistent shapes along the first dimension.")

# =============================================================================
# Load data - Intersect the data for 2 and 4 months
# =============================================================================
root = '/home/samy/Bureau/Proyect/LauraHarsan/Ines/Timecourses_updated/'

# Define paths and folders
folders = {'2mois': 'Lot3_2mois', '4mois': 'Lot3_4mois'}

# Load filenames and hash numbers
filenames       = {period: filename_sort_mat(os.path.join(root, folder)) for period, folder in folders.items()}
hash_numbers    = {period: extract_hash_numbers(filenames[period]) for period in filenames}

# common_hashes, ind_2m, ind_4m = np.intersect1d(hash_numbers['2mois'], hash_numbers['4mois'], return_indices=True)
int_2m4m = np.intersect1d(hash_numbers['2mois'], hash_numbers['4mois'], return_indices=True)

print('Number of intersected elements in 2m and 4m :' , len(int_2m4m[0]))

#%%
# =============================================================================
# Load cognitive data from .xlsx document
# =============================================================================
#Load cognitive data
cog_data_path   = os.path.join(root, 'Behaviour_exclusions_ROIs.xlsx')
cog_data_df     = pd.read_excel(cog_data_path, sheet_name='Feuil1')
data_roi        = pd.read_excel(cog_data_path, sheet_name='40_Allen_ROIs_list').to_numpy()

# Anatomical labels
anat_labels     = np.array([xx[0] for xx in data_roi])

# sex_label       = np.array(pd.DataFrame(cog_data_df, columns=['Sexe']))[:,0].T[inter_cogfun[2]]
cog_data_df['sexe_label']  = cog_data_df.loc[:,'Sexe']
cog_data_df['gen_label']  = cog_data_df.loc[:,'Genotype']

# Filtering based on sex (Male/Female), genotype (wt/dKI) and TC (ok/Excluded)
# Convert sex and genotype to numerical values for easier processing
cog_data_df['Sexe']     = cog_data_df['Sexe'].map({'M': 0, 'F': 1})
cog_data_df['Genotype'] = cog_data_df['Genotype'].map({'wt': 0, 'dKI': 1})
cog_data_df['TC_2M']    = cog_data_df['TC_2M'].map({'ok': 0, 'Excluded': 1})
cog_data_df['TC_4M']    = cog_data_df['TC_4M'].map({'ok': 0, 'Excluded': 1})

# Filter cognitive data for animals present in the intersected functional data
cog_data_filtered       = cog_data_df[cog_data_df['Name'].isin(int_2m4m[0])].sort_values(by='Name')

#Remove the TC 'excluded' from the data
cog_data_filtered       = cog_data_filtered[(cog_data_filtered['TC_2M'] == 0) & (cog_data_filtered['TC_4M'] == 0)]

# Generating boolean indices for various filters
mouse_hash_cog      = cog_data_filtered['Name'].to_numpy()
male_index          = cog_data_filtered['Sexe'] == 0
female_index        = cog_data_filtered['Sexe'] == 1
ctrl_index          = cog_data_filtered['Genotype'] == 0
dki_index           = cog_data_filtered['Genotype'] == 1

# Further filtering based on specific criteria, e.g., males with wt genotype
male_wt_data    = cog_data_filtered[(cog_data_filtered['Sexe'] == 0) & (cog_data_filtered['Genotype'] == 0)]
male_dki_data   = cog_data_filtered[(cog_data_filtered['Sexe'] == 0) & (cog_data_filtered['Genotype'] == 1)]

#Extracting the intersection of functional and cognitive data
inter_cogfun    = np.intersect1d(int_2m4m[0], mouse_hash_cog, return_indices=True)
print('Number of intersected cognitive and functional elements :' , len(inter_cogfun[0]))

#Generating sorted index of functional data (2m and 4m) 
index_tsintcog  = np.array(int_2m4m)[1:,inter_cogfun[1]] #intersection of 2m,4m and coginfo

#%%
# Create the arrays

#Extracting the file name of functional time series that are intersected
filename_int2m = filenames['2mois'][index_tsintcog[0]]
filename_int4m = filenames['4mois'][index_tsintcog[1]]
        
#Loading the time series of the intersected data
ts2m = load_matdata(root, folders['2mois'], filename_int2m)
ts4m = load_matdata(root, folders['4mois'], filename_int4m)

#Remove the first transient of data
ts2m = ts2m[:,50:]
ts4m = ts4m[:,50:]

#Sexe
ts2m_male, ts2m_female  = ts2m[male_index], ts2m[female_index]
#Genotype
ts2m_ctrl, ts2m_dki     = ts2m[ctrl_index], ts2m[dki_index]

#Some variables
n_animals, total_tp, regions = ts2m.shape

# Generate a label list
sex_label       = cog_data_filtered['sexe_label'].to_numpy()
gen_label       = cog_data_filtered['gen_label'].to_numpy()

# =============================================================================
#ERASE THIS!!!!!!!
# Could create justr for this time ts female
# =============================================================================
filename_int2mf = filenames['2mois'][~index_tsintcog[0]]
filename_int4mf = filenames['4mois'][~index_tsintcog[1]]

ts2mf = load_matdata(root, folders['2mois'], filename_int2mf)
ts4mf = load_matdata(root, folders['4mois'], filename_int4mf)

ts2m_female = ts2mf[:,50:]
ts4m_female = ts4mf[:,50:]

#%%
# Visualization: Plotting data based on the filters applied
plt.figure(1,figsize=(10, 6))
plt.clf()
plt.subplot(211)
# plt.hist((male_wt_data['OiP_2M'], male_wt_data['OiP_4M'], male_dki_data['OiP_2M'], male_dki_data['OiP_4M']), bins=4, 
#          alpha=0.7, 
#           histtype='step', 
#          label=('Male WT 2M', 'Male WT 4M', 'Male dKI 2M', 'Male dKI 4M'))
plt.violinplot((male_wt_data['OiP_2M'], male_wt_data['OiP_4M'], male_dki_data['OiP_2M'], male_dki_data['OiP_4M']))
            # labels=('Male WT 2M', 'Male WT 4M', 'Male dKI 2M', 'Male dKI 4M'))
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
    plt.savefig('fig/cog_data/oip_ro24h_male_ctrl_dki.png')
    plt.savefig('fig/cog_data/oip_ro24h_male_ctrl_dki.pdf')
#%%
# =============================================================================
# FC and modularity
# -There is another modularity algorithm which claims to be better than Louvain. It's called leiden algorithm
#https://www.nature.com/articles/s41598-019-41695-z
# =============================================================================

#static functions
def ts2fc(timeseries, format_data = '2D'):
    """
    Calculate functional connectivity from time series data.
    
    Parameters:
    timeseries (array): Time series data of shape (timepoints, nodes).
    format_data (str): Output format, '2D' for full matrix or '1D' for lower-triangular vector.
    
    Returns:
    fc (array): Functional connectivity matrix ('2D') or vector ('1D').
    
    Adapted from Lucas Arbabyazd et al 2020. Methods X, doi: 10.1016/j.neuroimage.2020.117156
    """
    # Calculate correlation coefficient matrix
    fc = np.corrcoef(timeseries.T)

    # Optionally zero out the diagonal for '2D' format
    if format_data=='2D':
        np.fill_diagonal(fc,0)#fill the diagonal with 0
        return fc
    elif format_data=='1D':
        # Return the lower-triangular part excluding the diagonal
        return fc[np.tril_indices_from(fc, k=-1)]
    
def sort_modularity(fc):
    #Modularity of Louvain
    # modules, louvain = bct.modularity.modularity_louvain_dir(fc)
    # modules, louvain = bct.modularity.modularity_louvain_dir(fc)
    modules, louvain = bct.modularity.modularity_louvain_und_sign(fc, gamma=1.1)
    # print(np.unique(modules),louvain)
    
    #sort accord the modularity
    sort_modules = np.argsort(modules)
    # print(sort_modules)
    fc_mod = fc[:,sort_modules][sort_modules,:] #fc sorted by modularity
    
    return fc_mod

#dynamics functions
def ts2dfc_stream(ts, windows_size, lag=None, format_data='2D'):
    """
    Calculate dynamic functional connectivity stream (dfc_stream) from time series data.

    Parameters:
    ts (array): Time series data of shape (t, n), where t is timepoints, n is regions.
    windows_size (int): Window size to slide over the ts.
    lag (int): Shift value for the window. Defaults to W if not specified.
    format (str): Output format. '2D' for a (l, F) shape, '3D' for a (n, n, F) shape.

    Returns:
    dFCstream (array): Dynamic functional connectivity stream.
    """
    
    if lag is None:
        lag = windows_size
    
    
    ts_total, regions = np.shape(ts)
    
    cc2               = regions * (regions-1)//2 #number of pairwise correlations
    # Calculate the number of frames/windows
    frames = (ts_total-windows_size)//lag + 1
    
    if format_data=='3D':
        dfc_stream = np.zeros((regions,regions, frames))
    elif format_data=='2D':
        dfc_stream = np.zeros((cc2, frames))

    for k in range(frames):
        wstart = k * lag
        wstop = wstart + windows_size
        if format_data == '3D':
            dfc_stream[:, :, k] = ts2fc(ts[wstart:wstop, :], '2D')  # Assuming TS2FC returns a matrix
        elif format_data =='2D':
            dfc_stream[:, k]    = ts2fc(ts[wstart:wstop, :], '1D')  # Assuming TS2FC returns a vector

    return dfc_stream

def matrix2vec(matrix3d):
    """
    Convert a 3D matrix into a 2D matrix by vectorizing each 2D matrix along the third dimension.
    
    Parameters:
    matrix3d (numpy.ndarray): 3D numpy array.
    
    Returns:
    numpy.ndarray: 2D numpy array where each column is the vectorized form of the 2D matrices from the 3D input.
    """
    F, n, _ = matrix3d.T.shape  # Assuming matrix3d shape is [F, n, n]
    return matrix3d.reshape((n*n,F))

def dfc_stream2fcd(dfc_stream):
    """
    Calculate the dynamic functional connectivity (dFC) matrix from a dfc_stream.
    
    Parameters:
    dfc_stream (numpy.ndarray): Input dynamic functional connectivity stream, can be 2D or 3D.
    
    Returns:
    numpy.ndarray: The dFC matrix computed as the correlation of the dfc_stream.
    """
    if dfc_stream.ndim < 2 or dfc_stream.ndim > 3:
        raise ValueError("Provide a valid size dfc_stream (2D or 3D)!")
    # Convert 3D dfc_stream to 2D if necessary
  
    if dfc_stream.ndim == 3:
        dfc_stream_2D = matrix2vec(dfc_stream)
    else:
        dfc_stream_2D = dfc_stream

    # Compute dFC
    dfc_stream_2D = dfc_stream_2D.T
    dfc = np.corrcoef(dfc_stream_2D)
    
    return dfc

def dfc_speed(dfc_stream, vstep=1):
    """
    Calculate speeds of variation in dynamic functional connectivity over a specified step size.
    
    Parameters:
    dfc_stream (numpy.ndarray): Input dynamic functional connectivity stream (2D or 3D).
    vstep (int): Step size for computing speed of variation (default=1).
    
    Returns:
    speed_median (float): Median of computed distribution of speeds.
    Speeds (numpy.ndarray): Time series of computed speeds.
    """
    # Check the dimensionality of dfc_stream and process accordingly
    if dfc_stream.ndim == 3:
        # Assuming a reshapedfc_stream function exists to convert 3D dfc_stream to 2D
        FCstr = dfc_stream.reshape(dfc_stream.shape[0]*dfc_stream.shape[1], dfc_stream.shape[2])
    elif dfc_stream.ndim == 2:
        FCstr = dfc_stream
    else:
        raise ValueError("Provide a valid size dFCstream (2D or 3D)!")
    
    nslices = FCstr.shape[1]
    speeds = []

    # Compute speeds using correlation distance
    # for sp in range(nslices - vstep):
    for sp in range(nslices - vstep):
        if (sp + vstep)>0:
        # if (sp + vstep)>=0 or (nslices - vstep)>sp:
            fc1 = FCstr[:, sp]
            fc2 = FCstr[:, sp + vstep]
            speed = 1 - np.corrcoef(fc1, fc2)[0, 1]
            speeds.append(speed)

    # Calculate median speed
    speed_median    = np.median(speeds)
    speed_all       = np.array(speeds)

    return speed_median, speed_all

def dfc_speed_series(ts, window_parameter, lag=1, tau=3, get_speed_dist=False):
    
    time_windows_min, time_windows_max, time_window_step = window_parameter
    time_windows_range = np.arange(time_windows_min,time_windows_max+1,time_window_step)
    tau_array       = np.append(np.arange(-tau,tau), tau ) 
    
    speed_windows = np.zeros((len(time_windows_range), len(tau_array)))
    speed_dist    = []
    
    for idx_tt, tt in enumerate(time_windows_range):
    
        windows_size    = tt
    
        dfc_streamaux   = ts2dfc_stream(ts, windows_size, lag, format_data='2D')
        width_stripe      = dfc_streamaux.shape[1]-windows_size-tau
    
        speed_array    = np.array([dfc_speed(dfc_streamaux, vstep=windows_size + sp)[1][:width_stripe] for sp in tau_array])
        speed_windows[idx_tt] = np.mean(speed_array,axis=1)

        if get_speed_dist==True:        # speed_dist = np.mean(speed_array,axis=1)
            speed_dist.append(speed_array.flatten())
        
    if get_speed_dist==True:        # speed_dist = np.mean(speed_array,axis=1)
        return speed_windows, speed_dist
    else:
        return speed_windows

def window_pooling_speed(filter_listed, vel_list):
    short_vel_list = []
    mid_vel_list = []
    long_vel_list = []
    
    filter_list = np.where(filter_listed==True)[0]
    
    # for tt in range(29): 
    for tt in filter_list: 
        for yy in range(10):
            short_vel_list.append(vel_list[tt][yy])  
        for yy in range(10,31):
            mid_vel_list.append(vel_list[tt][yy])  
        for yy in range(31,61):
            long_vel_list.append(vel_list[tt][yy])  
    
    short_vel = np.concatenate(short_vel_list) if short_vel_list else np.array([])
    mid_vel = np.concatenate(mid_vel_list) if mid_vel_list else np.array([])
    long_vel = np.concatenate(long_vel_list) if long_vel_list else np.array([])
    
    return short_vel , mid_vel, long_vel
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
#Plot individual static fc
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
# Population plot

plt.figure(3,figsize=(8,8))
plt.clf()

plt.subplot(211)
b1,a1=np.polyfit((tri_2m[dki_index]).flatten(), (tri_4m[dki_index].flatten()),deg=1)
b2,a2=np.polyfit((tri_2m[ctrl_index]).flatten(), (tri_4m[ctrl_index].flatten()),deg=1)
xseq = np.linspace(-1, 1, num=100)

plt.title('dKi vs ctrl ')
plt.scatter(tri_2m[dki_index], tri_4m[dki_index],marker='.',label='dki slope=%s'%np.round(b1,3))
plt.scatter(tri_2m[ctrl_index], tri_4m[ctrl_index],marker='.', alpha=0.5,label='ctrl slope=%s'%np.round(b2,3))

plt.plot(xseq, a1 + b1 * xseq, color="C0", lw=2.5)
plt.plot(xseq, a2 + b2 * xseq, color="C1", lw=2.5)
plt.xlabel('2m')
plt.ylabel('4m')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.legend()

plt.subplot(212)
plt.hist((tri_2m[dki_index].flatten(), tri_2m[ctrl_index].flatten(), tri_4m[dki_index].flatten(), tri_4m[ctrl_index].flatten()),histtype='step',bins=100)
plt.legend(('2m_dki', '2m_ctrl', '4m_dki','4m_ctrl'))
plt.xlabel('Pairwise correlation')
plt.ylabel('Counts #')

plt.tight_layout()
if save_fig ==True:
    plt.savefig('fig/fc/fc_all_dki_vs_ctrl.png')
    plt.savefig('fig/fc/fc_all_dki_vs_ctrl.pdf')

#%%
#%%
# =============================================================================
# speed by window oversampling and fcd for one conditions and animal
# =============================================================================
idx_mice = 1
ts_aux = ts2m[idx_mice]

windows_size = 55
lag = 1
tau = 3
tau_array       = np.append(np.arange(-tau,tau), tau ) 

dfc_streamaux   = ts2dfc_stream(ts_aux, windows_size, lag, format_data='2D')
fcd             = dfc_stream2fcd(dfc_streamaux)

width_stripe    = dfc_streamaux.shape[1]-windows_size-tau

start = time.time()
#Window oversampling
#get the stripe of speeds around W
speed_array    = np.array([dfc_speed(dfc_streamaux, vstep=windows_size + sp)[1][:width_stripe] for sp in tau_array])
stop= time.time()
print('sim time', stop-start)

# fcd_2m             = [dfc_stream2fcd(ts2dfc_stream(xx, windows_size, lag, format_data='2D')) for xx in ts2m]
#%%Fcd and stream plot
plt.figure(4)
plt.clf()
plt.subplot(211)
# plt.title('2m mouse #%s %s %s'%(mouse_hash_cog[idx_mice], gen_label[idx_mice], sex_label[idx_mice]))
plt.title('dFC stream mouse #%s %s %s'%(mouse_hash_cog[idx_mice], gen_label[idx_mice], sex_label[idx_mice]))
plt.imshow(dfc_streamaux, aspect='auto', interpolation='none', cmap='jet')
plt.colorbar()

plt.subplot(212)
plt.title('FCD')
plt.imshow(fcd, aspect='auto', interpolation='none', cmap='jet')
plt.colorbar()
plt.clim(0,0.5)
plt.tight_layout()

#Speed plot
plt.figure(5)
plt.clf()
plt.subplot(211)
plt.title('dFC speed mouse #%s %s %s'%(mouse_hash_cog[idx_mice], gen_label[idx_mice], sex_label[idx_mice]))
plt.imshow(speed_array, aspect='auto', interpolation='none', cmap='jet')
# plt.colorbar()
plt.clim(0,1)
plt.subplot(212)
plt.plot(np.mean(speed_array,axis=0), label='mean')
plt.plot(np.mean(speed_array,axis=0)-np.std(speed_array,axis=0), 'C1', label='mean-std')
plt.plot(np.mean(speed_array,axis=0)+np.std(speed_array,axis=0), 'C1', label='mean+std')
plt.axhline(np.mean(speed_array))
# plt.xlim(0, 178)
plt.legend()
plt.tight_layout()
#%%
# =============================================================================
# dFC speed average via window oversampling for one animal, with multiple windows length and tau
# - Resume this part to dfc_speed_series
# =============================================================================
idx_mice = 1
ts_aux = ts2m[idx_mice]

lag             = 1
tau             = 3
tau_array       = np.append(np.arange(-tau,tau), tau ) 

#windows parameter range
time_windows_min, time_windows_max, time_window_step = 5, 80,1
time_windows_range = np.arange(time_windows_min,time_windows_max+1,time_window_step)
speed_windows = np.zeros((len(time_windows_range), len(tau_array)))

start = time.time()
for idx_tt, tt in enumerate(time_windows_range):

    windows_size        = tt

    dfc_streamaux       = ts2dfc_stream(ts_aux, windows_size, lag, format_data='2D')
    # dfc_streamaux   = ts2dfc_stream(ts_aux, windows_size, format_data='2D')
    width_stripe        = dfc_streamaux.shape[1]-windows_size-tau

    speed_array         = np.array([dfc_speed(dfc_streamaux, vstep=windows_size + sp)[1][:width_stripe] for sp in tau_array])
    speed_windows[idx_tt] = np.mean(speed_array,axis=1)
    # print(np.shape(speed_array),np.shape(speed_windows))

stop= time.time()
print('sim time', stop-start)

plt.figure(6)
plt.clf()
plt.title('dFC speed average mouse #%s %s %s'%(mouse_hash_cog[idx_mice], gen_label[idx_mice], sex_label[idx_mice]))
plt.plot(time_windows_range,speed_windows,'.-')#, label='%s'%tau_array)
plt.xlabel('W')
plt.ylabel('<v>')
plt.legend(tau_array,title=r'$\Delta\tau$', title_fontsize=20, fontsize=15)
plt.ylim((0,1))

plt.tight_layout()

# =============================================================================
# #Check for one window length and multiple tau
# =============================================================================
windows_size_test    = 5

dfc_streamaux   = ts2dfc_stream(ts_aux, windows_size_test, lag, format_data='2D')
width_stripe      = dfc_streamaux.shape[1]-windows_size-tau
speed_array2    = np.array([dfc_speed(dfc_streamaux, vstep=windows_size     + sp)[1][:width_stripe] for sp in tau_array])

plt.figure(7)
plt.clf()
# plt.violinplot(speed_array2.T, showmeans=True)
plt.boxplot(speed_array2.T, showmeans=True)
#%%
# =============================================================================
# For all the animal and conditions, the speed median for different tau
# =============================================================================

window_parameter = (5,80,1)
lag=1

start = time.time()
speed_median_2m = np.array([dfc_speed_series(tsx, window_parameter,lag) for tsx in ts2m])
speed_median_4m = np.array([dfc_speed_series(tsx, window_parameter,lag) for tsx in ts4m])
stop= time.time()
print('sim time', stop-start)

#%%
# =============================================================================
# Take distributions for each W for one animal, mixing taus
# =============================================================================

vel_list2m = []
vel_list4m = []

start = time.time()
for xx in ts2m:
    aux_speed,aux_speed_dist  = dfc_speed_series(xx, window_parameter, lag, get_speed_dist=True)
    vel_list2m.append(aux_speed_dist)  
for xx in ts4m:
    aux_speed,aux_speed_dist2 = dfc_speed_series(xx, window_parameter, lag, get_speed_dist=True)
    vel_list4m.append(aux_speed_dist2)  
stop= time.time()
print('speed dist analysis time', stop-start)

# vel_median_list2m = []
# for xx in ts2m[:2]:
#     aux_speed,aux_speed_dist  = dfc_speed_series(xx, window_parameter, lag, get_speed_dist=True)
    # vel_list2m.append(aux_speed_dist)  
    # vel_median_list2m.extend()
#%%
# =============================================================================
# Windows pooling of window oversampling data
# =============================================================================
#For the dfc speed distribution window oversampling, get a windows pooling
window_pooling_speed_ctrl2m     = window_pooling_speed(ctrl_index,  vel_list2m)
window_pooling_speed_dki2m      = window_pooling_speed(dki_index,   vel_list2m)
window_pooling_speed_ctrl4m     = window_pooling_speed(ctrl_index,  vel_list4m)
window_pooling_speed_dki4m      = window_pooling_speed(dki_index,   vel_list4m)

plt.figure(8, figsize=(6,10))
plt.clf()
vel_label = ('10-30s (short)','30-72s (mid)','72-132s (long)')

for i in range(3):
    plt.subplot(3,1,i+1)
    plt.title('%s'%vel_label[i])
    plt.hist((window_pooling_speed_ctrl2m[i], window_pooling_speed_dki2m[i],window_pooling_speed_ctrl4m[i], window_pooling_speed_dki4m[i]),label=('2m ctrl', '2m dki', '4m ctrl', '4m dki'), histtype='step',bins=500, density=True)
    # plt.hist((window_pooling_speed_ctrl2m[i], window_pooling_speed_dki2m[i],window_pooling_speed_ctrl4m[i], window_pooling_speed_dki4m[i]),label=('2m ctrl', '2m dki', '4m ctrl', '4m dki'), histtype='step',bins=500, density=True, log=True)
    plt.ylabel('Counts')
    plt.xlim(0.2,1.2)
plt.xlabel('Freq[v]')
plt.legend()
plt.tight_layout()

if save_fig ==True:
    plt.savefig('fig/speed/speed_window_pooling_and_oversampling_male_dki_vs_ctrl_lag=%s_tau=%s.png'%(lag,tau))
    plt.savefig('fig/speed/speed_window_pooling_and_oversampling_male_dki_vs_ctrl_lag=%s_tau=%s.pdf'%(lag,tau))
#%%
# =============================================================================
# Windows pooling of window not-oversampled data
#-Falta
# =============================================================================


#%%
# =============================================================================
# Calculate the FCD and dFC_stream for a given W
# =============================================================================
#Windows FCD
windows_size = 30
lag = 1

start = time.time()
dfc_stream_2m   = np.array([ts2dfc_stream(ts2m[xx], windows_size, lag, format_data='2D') for xx in range(n_animals)])
fcd_2m          = np.array([dfc_stream2fcd(dfc_stream_2m[xx]) for xx in range(n_animals)])

dfc_stream_4m   = np.array([ts2dfc_stream(ts4m[xx], windows_size, lag, format_data='2D') for xx in range(n_animals)])
fcd_4m          = np.array([dfc_stream2fcd(dfc_stream_4m[xx]) for xx in range(n_animals)])
stop = time.time()
print(stop-start)

#%%
for idx_mice in range(n_animals):
# for idx_mice in range(2):
    
    plt.figure(9)

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
    plt.title('FCD (W=%s)'%windows_size)
    plt.imshow(fcd_2m[idx_mice], aspect='auto', interpolation='none',cmap='jet')
    plt.xlabel(r't$_{b}$')
    plt.ylabel(r't$_{a}$')
    plt.clim(0,0.5)
    plt.colorbar()
    
    plt.subplot(324)
    plt.title('FCD (W=%s)'%windows_size)
    plt.imshow(fcd_4m[idx_mice], aspect='auto', interpolation='none',cmap='jet')
    plt.xlabel(r't$_{b}$')
    plt.ylabel(r't$_{a}$')
    plt.clim(0,0.5)
    plt.colorbar()
    
    plt.subplot(313)
    plt.title('FCD velocity, overlap=%s, k+-=%s'%(lag, tau))
    plt.plot(time_windows_range, np.mean(speed_median_2m[idx_mice],axis=1),'.-',label='2m')
    plt.plot(time_windows_range, np.mean(speed_median_4m[idx_mice],axis=1),'.-',label='4m')
    plt.ylabel(r'$< v >$')
    plt.xlabel(r'$W$')
    plt.ylim(0,1)
    plt.legend()
    if save_fig ==True:
        plt.savefig('fig/dKI/fcd_mouse_#%s.png'%mouse_hash_cog[idx_mice])
        plt.savefig('fig/dKI/fcd_mouse_#%s.pdf'%mouse_hash_cog[idx_mice])
#%%
idx_mice=8

# idx_mice = 0
plt.figure(10)
plt.clf()


plt.subplot(231)
plt.title('dFC velocity 2m, overlap=%s, k+-=%s'%(lag,tau))
plt.plot(time_windows_range, speed_median_2m[idx_mice],'.--',label='2m')
plt.ylabel(r'$< v >$')
plt.xlabel(r'$W$')
plt.ylim(0,1)

plt.subplot(232)
plt.title('dFC velocity 4m, overlap=%s, k+-=%s'%(lag,tau))
plt.plot(time_windows_range, speed_median_4m[idx_mice],'.--',label='2m')
plt.xlabel(r'$W$')
plt.ylim(0,1)

plt.subplot(233)
plt.plot(np.linspace(0.2,1),np.linspace(0.2,1,),'--')
plt.scatter(speed_median_2m[idx_mice], speed_median_4m[idx_mice])# c=np.arange(497), marker='.')
plt.xlabel('2m')
plt.ylabel('4m')

# plt.subplot(234)
# plt.hist((short_vel, short_vel2),histtype='step')
# plt.subplot(235)
# plt.hist((mid_vel, mid_vel2),histtype='step')
# plt.subplot(236)
# plt.hist((long_vel, long_vel2),histtype='step')
# plt.plot(time_windows_range, np.mean(speed_median_2m[idx_mice],axis=1),'.--',label='2m')
# plt.plot(time_windows_range, np.mean(speed_median_4m[idx_mice],axis=1),'.--',label='2m')
# plt.plot(time_windows_range_4m, mean_velocity_4m,'.--',label='4m')
#%%
# =============================================================================
# The typical velocity for each mouse
# =============================================================================



#%%
# =============================================================================
# Velocity statistics
# =============================================================================
# idx_mice = 2

shortvel_ind2m = np.array([np.hstack(vel_list2m[xx][0:10]) for xx in range(n_animals)])
midvel_ind2m   = np.array([np.hstack(vel_list2m[xx][10:31]) for xx in range(n_animals)])
longvel_ind2m   = np.array([np.hstack(vel_list2m[xx][31:61]) for xx in range(n_animals)])

shortvel_ind4m = np.array([np.hstack(vel_list4m[xx][0:10]) for xx in range(n_animals)])
midvel_ind4m   = np.array([np.hstack(vel_list4m[xx][10:31]) for xx in range(n_animals)])
longvel_ind4m   = np.array([np.hstack(vel_list4m[xx][31:61]) for xx in range(n_animals)])

def vel_statistics(vel_array, bins_number=150):
    hist = np.array([np.histogram(vel_array[xx], density=True, bins=150)[0] for xx in range(n_animals)])
    aux_vel = np.array([np.histogram(vel_array[xx], density=True, bins=150)[1] for xx in range(n_animals)])

    v_median = np.array([np.median(vel_array[xx]) for xx in range(n_animals)])#np.median(hist,axis=1)
    v_typ= np.array([aux_vel[xx,np.argmax(hist[xx])] for xx in range(n_animals)])
    # v_typ = np.max(vel_array,axis=1)
    vel_q5 = np.quantile(vel_array, 0.05, axis=1)
    vel_q95 = np.quantile(vel_array, 0.95, axis=1)
    # vel_q95 = [np.quantile(vel_array[xx], 0.95,axis=1) for xx in range(n_animals)]#np.median(hist,axis=1)
    
    return v_median, v_typ,vel_q5, vel_q95

short2m_statistics = vel_statistics(shortvel_ind2m,bins_number=150)
short4m_statistics = vel_statistics(shortvel_ind4m,bins_number=150)

mid2m_statistics = vel_statistics(midvel_ind2m,bins_number=100)
mid4m_statistics = vel_statistics(midvel_ind4m,bins_number=100)

long2m_statistics = vel_statistics(longvel_ind2m,bins_number=100)
long4m_statistics = vel_statistics(longvel_ind4m,bins_number=100)
#%%
hist = np.array([np.histogram(shortvel_ind2m[xx], density=True, bins=150)[0] for xx in range(n_animals)])
aux_vel = np.array([np.histogram(shortvel_ind2m[xx], density=True, bins=150)[1] for xx in range(n_animals)])
[aux_vel[xx,np.argmax(hist[xx])] for xx in range(n_animals)]
# np.argmax(hist,axis=1)
#%%
statistic = 3
# var4m = short4m_statistics[statistic]#long4m_statistics[0]
# var2m = short2m_statistics[statistic]#long2m_statistics[0]

# var4m = mid4m_statistics[statistic]#long4m_statistics[0]
# var2m = mid2m_statistics[statistic]#long2m_statistics[0]

var4m = long4m_statistics[statistic]#long4m_statistics[0]
var2m = long2m_statistics[statistic]#long2m_statistics[0]
velmedian2m4m = np.array((var2m, var4m))

print(np.mean(velmedian2m4m[:,ctrl_index]))
print(np.mean(velmedian2m4m[:,dki_index]))

plt.figure(234)
plt.clf()

plt.subplot(221)
plt.title('WT')

plt.plot(velmedian2m4m[:,ctrl_index],'o--')
plt.ylabel('<v>')
plt.xlim(-0.2,1.2)
plt.xticks((0,1), ('2m', '4m'))

plt.subplot(222)
plt.title('dKI')
plt.plot(velmedian2m4m[:,dki_index],'o--')
# plt.ylim(0.3,1)
plt.xlim(-0.2,1.2)
plt.xticks((0,1), ('2m', '4m'))

delta_vel2m4m = (np.array(var4m) - np.array(var2m)) / (np.array(var4m) + np.array(var2m))
delta_vel2m4m = np.array((np.zeros(len(delta_vel2m4m)), delta_vel2m4m))

plt.subplot(223)
# plt.plot(delta_vel2m4m[:,ctrl_index],'o--', c='C0')
plt.plot(delta_vel2m4m[:,ctrl_index],'o--', c='C0')
plt.axhline(0)
# plt.ylim(-0.4,0.4)
plt.ylim(np.min(delta_vel2m4m[1]),np.max(delta_vel2m4m[1]))

plt.subplot(224)
plt.plot(delta_vel2m4m[:,dki_index],'o--', c='C1')
# plt.plot(np.zeros(len(delta_vel2m4m[dki_index]))+1,delta_vel2m4m[dki_index],'o--')
plt.ylim(np.min(delta_vel2m4m[1]),np.max(delta_vel2m4m[1]))
# plt.ylim(-0.4,0.4)
# plt.ylim(-0.15,0.15)
plt.axhline(0)
# plt.ylabel(r'$\Delta$ v')
# plt.xlim(np.min(delta_vel2m4m[1]),np.max(delta_vel2m4m[1]))
# plt.xticks((0,1), ('2m', '4m'))

#%%
# =============================================================================
# Speead average and cognitive metrics
# =============================================================================

# plt.violinplot((male_wt_data['OiP_2M'], male_wt_data['OiP_4M'], male_dki_data['OiP_2M'], male_dki_data['OiP_4M']))
speedmean_tau_2m = np.median(speed_median_2m[ctrl_index],axis=2)
speedmean_tau_4m = np.median(speed_median_4m[ctrl_index],axis=2)
speedmean_tau_2m_dki = np.median(speed_median_2m[dki_index],axis=2)
speedmean_tau_4m_dki = np.median(speed_median_4m[dki_index],axis=2)

plt.figure(10)
plt.clf()
w_aux = 25
# np.mean(speed_median_2m[idx_mice],axis=1)
plt.scatter(speedmean_tau_2m[:, w_aux], male_wt_data['OiP_2M'], label='2m wt')
plt.scatter(speedmean_tau_4m[:, w_aux], male_wt_data['OiP_4M'], label='4m wt')
plt.scatter(speedmean_tau_2m_dki[:, w_aux], male_dki_data['OiP_2M'], label='2m dki')
plt.scatter(speedmean_tau_4m_dki[:, w_aux], male_dki_data['OiP_4M'], label='4m dki')
plt.legend()
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


plt.figure(11)
plt.clf()
# plt.imshow(mc, aspect='auto', interpolation='none',cmap='RdBu_r')
plt.imshow(mc_mod, aspect='auto', interpolation='none',cmap='viridis')
plt.colorbar()
# plt.clim(-0.5,0.5)





#%%
plt.figure(12)
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
plt.hist(speed_array.flatten() ,histtype='step')#, aspect='auto', interpolation='none',cmap='RdBu_r')

plt.subplot(326)
plt.title('Metaconnectivity')
plt.imshow(mc_mod, aspect='auto', interpolation='none',cmap='RdBu_r')
plt.colorbar()
#%%




plt.figure(13)
plt.clf()
# plt.subplot
# (321)
plt.boxplot(np.median(speed_windows_2m[ctrl_index],axis=0).T)#,'.-',label='2m ctrl')
plt.boxplot(np.median(speed_windows_4m[ctrl_index],axis=0).T)#,'.-',label='2m ctrl')
plt.boxplot(np.median(speed_windows_2m[dki_index],axis=0).T)#,'.-',label='2m ctrl')
# plt.plot(time_windows_range,np.mean(np.median(speed_windows_2m[ctrl_index],axis=0),axis=1),'.-',label='2m ctrl')
# plt.plot(time_windows_range,np.mean(np.median(speed_windows_4m[ctrl_index],axis=0),axis=1),'.-',label='4m ctrl')
# plt.plot(time_windows_range,np.mean(np.median(speed_windows_2m[dki_index],axis=0),axis=1),'.-',label='2m dki')
# plt.plot(time_windows_range,np.mean(np.median(speed_windows_4m[dki_index],axis=0),axis=1),'.-',label='4m dki')
plt.ylim(0.4,1)
plt.legend()

#%%
# =============================================================================
# Save data
# =============================================================================
timeseries=ts_aux

#data saved
data_save = {}
data_save['ts2m'] = ts2m
data_save['names_2m'] = filename_int2m
data_save['fc_2m'] = fc_2m

data_save['ts4m'] = ts4m
data_save['names_4m'] = filename_int4m
data_save['fc_4m'] = fc_4m
