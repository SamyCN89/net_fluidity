#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 14:42:38 2023

@author: samy
"""
#%%
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
# from .functions_analysis import *

from shared_code.fun_paths import get_paths
from shared_code.fun_utils import set_figure_params
from shared_code.fun_utils import filename_sort_mat
# =============================================================================
# Figure's parameters
# =============================================================================

# Set figure parameters globally
plt.rcParams.update({'axes.labelsize': 17, 'axes.titlesize': 13,
                     # 'axes.spines.left': False, 'axes.spines.bottom': False,
                     'axes.spines.right': False, 'axes.spines.top': False})

# plt.style.use('seaborn-white')
save_fig =True

# =============================================================================
# Funcitons - to move
# =============================================================================

def load_matdata(folder_data, specific_folder, files_name):
    ts_list = []
    hash_dir        = os.path.join(folder_data, specific_folder)

    for idx,file_name in enumerate(files_name):
        file_path       = os.path.join(hash_dir, file_name)

        try:
            data = loadmat(file_path)['tc']
            ts_list.append(data)
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")


    # Check if the first dimension is consistent
    first_dim_size = ts_list[0].shape[0]
    if all(data.shape[0] == first_dim_size for data in ts_list):
        # Convert the list to a NumPy array
        ts_array = np.array(ts_list)
        return ts_array
    else:
        print("Error: Inconsistent shapes along the first dimension.")

# =============================================================================
# Load data - Intersect the data for 2 and 4 months
# =============================================================================

paths = get_paths(
    dataset_name="ines_abdullah",
    timecourse_folder=timecourse_folder,
    cognitive_data_file="ROIs.xlsx",
    anat_labels_file="41_Allen.txt",
)

root = '/home/samy/Bureau/Proyect/LauraHarsan/Ines/Timecourses_updated_03052024/'

# Define paths and folders
folders = {'2mois': 'Lot3_2mois', '4mois': 'Lot3_4mois'}

# Load filenames and hash numbers
filenames       = {period: filename_sort_mat(os.path.join(root, folder)) for period, folder in folders.items()}
hash_numbers    = {period: extract_hash_numbers(filenames[period]) for period in filenames}

# common_hashes, ind_2m, ind_4m = np.intersect1d(hash_numbers['2mois'], hash_numbers['4mois'], return_indices=True)
int_2m4m = np.intersect1d(hash_numbers['2mois'], hash_numbers['4mois'], return_indices=True)

print('Number of intersected elements in 2m and 4m :' , len(int_2m4m[0]))

#%%
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
#%%
#Generate a cog_mapinfo and cog_data array
# cog_data        = pd.DataFrame(cog_data_df, columns=["Name","OiP_2M","OiP_4M","RO24h_2M", "RO24h_4M"]).to_numpy().T
# cog_data_df        = pd.DataFrame(cog_data_df, columns=["Name","Sexe","Genotype","TC_2M","TC_4M"])

#Redifining the variables, from string to numbers
mapping         = {'M':0, 'F':1, 'wt':0, 'dKI':1, 'ok':0,'Excluded':1} #Name, sexe,genotype, tc_2m,tc_4m
cog_mapinfo     =  cog_data_df.replace({'Sexe':mapping, 'Genotype':mapping, 'TC_2M':mapping, 'TC_4M':mapping}).to_numpy().T


#Extracting the intersection of functional and cognitive data
inter_cogfun    = np.intersect1d(int_2m4m[0], mouse_hash_cog, return_indices=True)
print('Number of intersected cognitive and functional elements :' , len(inter_cogfun[0]))

#Generating sorted index of functional data (2m and 4m) and the related cognitive info (sex, genotype,tc_2m and tc_4m)
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

# Generate a label list
#Generating exclusion booleans for :
sex_label       = cog_data_filtered['sexe_label'].to_numpy()
gen_label       = cog_data_filtered['gen_label'].to_numpy()
# sex_label       = np.array(pd.DataFrame(cog_data_df, columns=['Sexe']))[:,0].T[inter_cogfun[2]]
# gen_label       = np.array(pd.DataFrame(cog_data_df, columns=['Genotype']))[:,0].T[inter_cogfun[2]]
# tc2m_label      = np.array(pd.DataFrame(cog_data_df, columns=['TC_2M']))[:,0].T[inter_cogfun[2]]
# tc4m_label      = np.array(pd.DataFrame(cog_data_df, columns=['TC_4M']))[:,0].T[inter_cogfun[2]]
#%%
# Visualization: Plotting data based on the filters applied
# Example: Plotting a histogram of a cognitive score for male_wt_data
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
plt.legend()

plt.subplot(212)
plt.violinplot((male_wt_data['RO24h_2M'], male_wt_data['RO24h_4M'], male_dki_data['RO24h_2M'], male_dki_data['RO24h_4M']))
plt.xticks([1, 2, 3, 4], ['Male WT 2M', 'Male WT 4M', 'Male dKI 2M', 'Male dKI 4M'])
plt.axhline(0,c='k')
plt.ylabel('RO24h score')
plt.title('Distribution of RO24h for Male')
plt.legend()
#%%
# =============================================================================
# FC and modularity
# -There is another modularity algorithm which claims to be better than Louvain. It's called leiden algorithm
#https://www.nature.com/articles/s41598-019-41695-z
# =============================================================================

#FC
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

#%%
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

#Functional connectivity
fc_2m = np.array([ts2fc(ts2m[xx]) for xx in range(n_animals)])
fc_4m = np.array([ts2fc(ts4m[xx]) for xx in range(n_animals)])

#Modularity
fc_2m_mod = np.array([sort_modularity(fc_2m[xx]) for xx in range(n_animals)])
fc_4m_mod = np.array([sort_modularity(fc_4m[xx]) for xx in range(n_animals)])

#superior triangular (maybe fucntion)
ind_fctri_2m = np.triu_indices(fc_2m.shape[2],1)
ind_fctri_4m = np.triu_indices(fc_4m.shape[2],1)

tri_2m = np.array([fc_2m[tt, ind_fctri_2m[0], ind_fctri_2m[1]] for tt in range(n_animals)])
tri_4m = np.array([fc_4m[tt, ind_fctri_4m[0], ind_fctri_4m[1]] for tt in range(n_animals)])

#%%
for idx_mice in range(n_animals):
# for idx_mice in range(2):

    aux_ts2m = ts2m[idx_mice]
    aux_ts4m = ts4m[idx_mice]

    plt.figure(2)
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

# for idx_mice in range(n_animals):
    if save_fig ==True:
        # plt.title('2m mouse #%s %s %s'%(mouse_hash_cog[idx_mice], gen_label[idx_mice], sex_label[idx_mice]))
        plt.savefig('fig/fc/mouse_#%s_%s_%s.png'%(mouse_hash_cog[idx_mice], gen_label[idx_mice], sex_label[idx_mice]))
        plt.savefig('fig/fc/mouse_#%s_%s_%s.pdf'%(mouse_hash_cog[idx_mice], gen_label[idx_mice], sex_label[idx_mice]))
        # plt.savefig('fig/fc/mouse_#%s.pdf'%mouse_hash_cog[idx_mice])
#%%

plt.figure(3)
plt.clf()


plt.subplot(211)
b1,a1=np.polyfit((tri_2m[dki_index]).flatten(), (tri_4m[dki_index].flatten()),deg=1)
b2,a2=np.polyfit((tri_2m[ctrl_index]).flatten(), (tri_4m[ctrl_index].flatten()),deg=1)
xseq = np.linspace(-1, 1, num=100)


# plt.title('dKi vs ctrl slope:%s'%np.round(b,3))
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
plt.hist((tri_2m[dki_index].flatten(), tri_2m[ctrl_index].flatten(), tri_4m[dki_index].flatten(), tri_4m[ctrl_index].flatten()),histtype='step',bins=50)
# plt.hist((tri_2m[dki_index], tri_2m[ctrl_index], tri_4m[dki_index], tri_4m[ctrl_index]),histtype='step',bins=50)
plt.legend(('2m_dki', '2m_ctrl', '4m_dki','4m_ctrl'))
plt.xlabel('CC')
plt.ylabel('Counts #')

if save_fig ==True:
    plt.savefig('fig/fc/fc_all_dki_vs_ctrl.png')
    plt.savefig('fig/fc/fc_all_dki_vs_ctrl.pdf')

#%%
# =============================================================================
# FCD stream
# =============================================================================
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

lag=1
windows_size=25
start = time.time()
dfc_streamaux = ts2dfc_stream(ts2m[0], windows_size, lag, format_data='2D')
_,dfc_streamaux2 = fcd_analysis(ts2m[0], windows_size, lag)
# [ts2dfc_stream(xx, windows_size, lag, format_data='2D') for xx in ts2m]
# [fcd_analysis(xx, windows_size, lag) for xx in ts2m]
stop= time.time()
print('sim time', stop-start)
#%%
# =============================================================================
# FCD
# =============================================================================
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

def dFCstream2dFC(dFCstream):
    """
    Calculate the dynamic functional connectivity (dFC) matrix from a dFCstream.

    Parameters:
    dFCstream (numpy.ndarray): Input dynamic functional connectivity stream, can be 2D or 3D.

    Returns:
    numpy.ndarray: The dFC matrix computed as the correlation of the dFCstream.
    """
    if dFCstream.ndim < 2 or dFCstream.ndim > 3:
        raise ValueError("Provide a valid size dFCstream (2D or 3D)!")
    # Convert 3D dFCstream to 2D if necessary

    if dFCstream.ndim == 3:
        dFCstream_2D = matrix2vec(dFCstream)
    else:
        dFCstream_2D = dFCstream

    # Compute dFC
    dFCstream_2D = dFCstream_2D.T
    dFC = np.corrcoef(dFCstream_2D)

    return dFC

lag=1
windows_size=30
start = time.time()
dfc_streamaux = ts2dfc_stream(ts2m[0], windows_size, lag, format_data='2D')
fcd = dFCstream2dFC(dfc_streamaux)

dfc_streamaux2 = ts2dfc_stream(ts2m[0], windows_size, lag, format_data='3D')
fcd2 = dFCstream2dFC(dfc_streamaux2)

fcd3,dfc_streamaux3 = fcd_analysis(ts2m[0], windows_size, lag)
stop= time.time()
print('sim time', stop-start)


plt.figure(5646)
plt.clf()
plt.subplot(311)
plt.imshow(fcd, aspect='auto', interpolation='none', cmap='jet')
plt.subplot(312)
plt.imshow(fcd2, aspect='auto', interpolation='none', cmap='jet')
plt.subplot(313)
plt.imshow(fcd3, aspect='auto', interpolation=None, cmap='jet')
#%%
def dFC_Speeds(dFCstream, vstep=1):
    """
    Calculate speeds of variation in dynamic functional connectivity over a specified step size.

    Parameters:
    dFCstream (numpy.ndarray): Input dynamic functional connectivity stream (2D or 3D).
    vstep (int): Step size for computing speed of variation (default=1).

    Returns:
    typSpeed (float): Median of computed distribution of speeds.
    Speeds (numpy.ndarray): Time series of computed speeds.
    """
    # Check the dimensionality of dFCstream and process accordingly
    if dFCstream.ndim == 3:
        # Assuming a reshapedFCstream function exists to convert 3D dFCstream to 2D
        FCstr = dFCstream.reshape(dFCstream.shape[0]*dFCstream.shape[1], dFCstream.shape[2])
    elif dFCstream.ndim == 2:
        FCstr = dFCstream
    else:
        raise ValueError("Provide a valid size dFCstream (2D or 3D)!")

    nslices = FCstr.shape[1]
    speeds = []

    # Compute speeds using correlation distance
    for s in range(nslices - vstep):
        FC1 = FCstr[:, s]
        FC2 = FCstr[:, s + vstep]
        speed = 1 - np.corrcoef(FC1, FC2)[0, 1]
        speeds.append(speed)

    # Calculate median speed
    typSpeed = np.median(speeds)
    Speeds = np.array(speeds)

    return typSpeed, Speeds

tau = 3
tau_array = np.arange(-tau,tau)
tau_array =  np.append(tau_array, tau)

len_ribbon = dfc_streamaux.shape[1]-windows_size-tau

speed_array = np.zeros((len(tau_array), len_ribbon))

for idx, sp in enumerate(tau_array):
    dfc_streamaux = ts2dfc_stream(ts2m[0], windows_size, lag, format_data='2D')
    mean_speed, speeds = dFC_Speeds(dfc_streamaux,vstep=windows_size)
    speed_array[idx] = dFC_Speeds(dfc_streamaux,vstep=windows_size+sp)[1][:len_ribbon]
    # print(dFC_Speeds(dfc_streamaux,vstep=windows_size)[1][:len_ribbon].shape)
    # print(speeds.shape, np.arange(-tau,tau))
#%%
plt.figure(5646)
plt.clf()
plt.subplot(311)
plt.imshow(speed_array, aspect='auto', interpolation='none', cmap='jet')
plt.colorbar()
plt.clim(0,1)
plt.subplot(312)
plt.plot(np.mean(speed_array,axis=0), label='mean')
plt.plot(np.mean(speed_array,axis=0)-np.std(speed_array,axis=0), 'C1', label='mean-std')
plt.plot(np.mean(speed_array,axis=0)+np.std(speed_array,axis=0), 'C1', label='mean+std')

plt.axhline(np.mean(speed_array))

plt.legend()
#%%
# start = time.time()
# stop= time.time()
# print('sim time', stop-start)
def fcd_analysis(ts_aux, windows_size, overlap):
    ts_total, regions = np.shape(ts_aux)


    wtotal = int(round( (ts_total / windows_size) ,0))
    l_points = np.arange(0, ts_total-windows_size, overlap)

    aux_dfc = []

    for w0 in l_points:
        w1 = w0 + windows_size
        # wtotal = int(round(np.shape(ts_aux)[1] / windows_size,0))
        # print(w0,w1)
        aux_pcorr = np.corrcoef(ts_aux[w0:w1].T)
        np.fill_diagonal(aux_pcorr, 0)
        aux_dfc.append(aux_pcorr)

    dfc_stream  = np.array([allPm[np.tril_indices(regions, k = -1)] for allPm in aux_dfc])

    #fcd
    fcd         = np.corrcoef(dfc_stream)
    return fcd, dfc_stream
#%%

# =============================================================================
# Velocity FCD
# =============================================================================

#Windows FCD
def fcd_velocity_old(aux_ts,time_windows_min=5,time_windows_max=50,time_window_step=1, overlap = 1, k_range=3):

    time_windows_range = np.arange(time_windows_min,time_windows_max+1,time_window_step)
    mean_velocity = np.zeros(np.shape(time_windows_range))

    for num_ind, aux_tw in enumerate(time_windows_range):


        fcd, dfc_stream = fcd_analysis(aux_ts, aux_tw, overlap)

        window_not_overlap = round(aux_tw/overlap)

        size = fcd.shape[0]
        ind_ribbon_max = np.tril_indices(size,k=window_not_overlap+k_range)
        ind_ribbon_min = np.tril_indices(size,k=window_not_overlap-k_range)


        mask_ribbon = np.zeros((size, size))
        mask_ribbon[ind_ribbon_max] =1
        mask_ribbon[ind_ribbon_min] =0

        ind_ribbon = mask_ribbon>0
        fcd_velocity = 1-fcd[ind_ribbon]
        mean_velocity[num_ind] = np.nanmean(fcd_velocity)

    return time_windows_range,mean_velocity

#Windows FCD
def fcd_velocity(aux_ts,time_windows_min=5,time_windows_max=50,time_window_step=1, overlap = 1, k_range=3):

    time_windows_range = np.arange(time_windows_min,time_windows_max+1,time_window_step)
    mean_velocity = np.zeros(np.shape(time_windows_range))

    fcd_velocity  =[]
    dfc_walk = []
    for num_ind, aux_tw in enumerate(time_windows_range):


        fcd, dfc_stream = fcd_analysis(aux_ts, aux_tw, overlap)

        window_not_overlap = round(aux_tw/overlap)

        size = fcd.shape[0]
        ind_ribbon_max = np.tril_indices(size,k=window_not_overlap+k_range)
        ind_ribbon_min = np.tril_indices(size,k=window_not_overlap-k_range)

        mask_ribbon = np.zeros((size, size))
        mask_ribbon[ind_ribbon_max] =1
        mask_ribbon[ind_ribbon_min] =0

        ind_ribbon = mask_ribbon>0
        aux_fcd_velocity = fcd[ind_ribbon]
        mean_velocity[num_ind] = np.nanmean(aux_fcd_velocity)

        #walk
        fcd_velocity.append((aux_fcd_velocity))

        mask_ribbon[mask_ribbon==0] = np.nan
        dfc_walk_mean = 1-np.nanmean(fcd*mask_ribbon, axis=1)
        dfc_walk_std = 1-np.nanstd(fcd*mask_ribbon, axis=1)
        dfc_walk.append((dfc_walk_mean,dfc_walk_std))


    return time_windows_range, mean_velocity, fcd_velocity, dfc_walk

# b = fcd_velocity(ts2m[0],time_windows_min=5,time_windows_max=50,time_window_step=1,overlap=1,k_range=3)
time_windows_range, mean_velocity, fcd_velocity, dfc_walk =  fcd_velocity(ts2m[0],time_windows_min=5,time_windows_max=50,time_window_step=1,overlap=1,k_range=3)
#%%
# time_windows_range, mean_velocity, fcd_velocity, dfc_walk = fcd_velocity(ts2m[0],time_windows_min=5,time_windows_max=50,time_window_step=1,overlap=1,k_range=3)

#short 5-14,mid 15-35,long 36-50
aux_vel = fcd_velocity[5:14]

short_vel = np.array([])
mid_vel = np.array([])
long_vel = np.array([])

for yy in fcd_velocity[:9]:
    short_vel = np.concatenate((short_vel, yy))

for yy in fcd_velocity[10:30]:
    mid_vel = np.concatenate((mid_vel, yy))

for yy in fcd_velocity[30:]:
    long_vel = np.concatenate((long_vel, yy))
    # aux1.append(yy)

# mean_fcd_velocity_2m = np.array([fcd_velocity(ts2m[idx_mice],time_windows_min=5,time_windows_max=50,time_window_step=1,overlap=overlap,k_range=k_range)
                                           # for idx_mice in range(n_animals)])
#%%
plt.figure(9841352)
plt.clf()
# for xx in range(10):
plt.hist((1-short_vel,1-mid_vel,1-long_vel), histtype='step')
plt.legend(('short','mid','long'))
#%%
#Windows FCD
windows_size=30
overlap = 1
k_range=3


# ts_total, regions = np.shape(ts_aux)


wtotal = int(round( (total_tp / windows_size) ,0))
l_points = np.arange(0, total_tp - windows_size, overlap)
# start = time.time()
# fcd2m = [fcd_analysis(ts2m[xx], windows_size, overlap) for xx in range(n_animals)]
# stop = time.time()
# print(stop-start)
# fcd2m, dfc2m_stream =np.array([fcd_analysis(ts2m[xx], windows_size, overlap) for xx in range(n_animals)])

fcd = np.zeros((n_animals,2,len(l_points), len(l_points)))
dfc_stream = np.zeros((n_animals,2,len(l_points), int(regions*(regions-1)/2)))

start = time.time()
for idx, xx in enumerate(range(n_animals)):
    fcd[idx, 0], dfc_stream[idx, 0] = fcd_analysis(ts2m[idx], windows_size, overlap)
    fcd[idx, 1], dfc_stream[idx, 1] = fcd_analysis(ts4m[idx], windows_size, overlap)
    # print(np.shape(dfc2m_stream), np.shape(dfc4m_stream))
stop = time.time()
print(stop-start)

# fc_2m_mod = np.array([sort_modularity(fc_2m[xx]) for xx in range(n_animals)])
#%%
mean_fcd_velocity_2m = np.array([fcd_velocity_old(ts2m[idx_mice],time_windows_min=5,time_windows_max=50,time_window_step=1,overlap=overlap,k_range=k_range)
                                           for idx_mice in range(n_animals)])
mean_fcd_velocity_2m = np.array([fcd_velocity(ts2m[idx_mice],time_windows_min=5,time_windows_max=50,time_window_step=1,overlap=overlap,k_range=k_range)
                                           for idx_mice in range(n_animals)])
mean_fcd_velocity_4m = np.array([fcd_velocity(ts4m[idx_mice],time_windows_min=5,time_windows_max=50,time_window_step=1,overlap=overlap,k_range=k_range)
                                           for idx_mice in range(n_animals)])

#%%
# idx_mice =20

# for idx_mice in range(n_animals):
for idx_mice in range(2):

    plt.figure(25)
    plt.clf()
    plt.subplot(321)
    # plt.title('4m mouse #%s %s %s'%(mouse_hash_cog[idx_mice], gen_label[idx_mice], sex_label[idx_mice])
    plt.title('dFC stream 2m mouse #%s %s %s'%(mouse_hash_cog[idx_mice], gen_label[idx_mice], sex_label[idx_mice]))
    # plt.title('dFC stream')
    plt.imshow(dfc_stream[idx_mice,0].T, aspect='auto', interpolation='none',cmap='RdBu_r')
    plt.ylabel(r'(region$_{i}$,region$_{j}$)')
    plt.xlabel(r't$_{w}$')
    plt.colorbar()

    plt.subplot(322)
    plt.title('dFC stream 4m mouse #%s %s %s'%(mouse_hash_cog[idx_mice], gen_label[idx_mice], sex_label[idx_mice]))
    # plt.title('dFC stream 4m mouse #%s %s'%(mouse_hash_cog[idx_mice], gen_label))
    # plt.imshow(corr_vectors.T, aspect='auto', interpolation='none',cmap='RdBu_r')
    plt.imshow(dfc_stream[idx_mice,1].T, aspect='auto', interpolation='none',cmap='RdBu_r')
    plt.ylabel(r'(region$_{i}$,region$_{j}$)')
    plt.xlabel(r't$_{w}$')
    plt.colorbar()

    plt.subplot(323)
    plt.title('FCD (W=%s)'%windows_size)
    plt.imshow(fcd[idx_mice,0], aspect='auto', interpolation='none',cmap='jet')
    plt.xlabel(r't$_{b}$')
    plt.ylabel(r't$_{a}$')
    plt.clim(0,0.5)
    plt.colorbar()

    plt.subplot(324)
    plt.title('FCD (W=%s)'%windows_size)
    plt.imshow(fcd[idx_mice,1], aspect='auto', interpolation='none',cmap='jet')
    plt.xlabel(r't$_{b}$')
    plt.ylabel(r't$_{a}$')
    plt.clim(0,0.5)
    plt.colorbar()
    # dfc_stream =

    plt.subplot(325)
    plt.title('FCD velocity, overlap=%s, k+-=%s'%(overlap,k_range))
    plt.plot(mean_fcd_velocity_2m[idx_mice,0], mean_fcd_velocity_2m[idx_mice,1],'.--',label='2m')
    plt.plot(mean_fcd_velocity_4m[idx_mice,0], mean_fcd_velocity_4m[idx_mice,1],'.--',label='4m')
    # plt.plot(time_windows_range_2m, mean_velocity_2m,'.--',label='2m')
    # plt.plot(time_windows_range_4m, mean_velocity_4m,'.--',label='4m')
    plt.ylabel(r'$< v >$')
    plt.xlabel(r'$W$')
    plt.ylim(0,1)
    plt.legend()
    # print(np.shape(dfc_stream), wtotal)
    if save_fig ==True:
        plt.savefig('fig/dKI/fcd_mouse_#%s.png'%mouse_hash_cog[idx_mice])
        plt.savefig('fig/dKI/fcd_mouse_#%s.pdf'%mouse_hash_cog[idx_mice])
#%%
# =============================================================================
# Save data
# =============================================================================
timeseries=ts2m[idx_mice]

#data saved
data_save = {}
data_save['ts2m'] = ts2m
data_save['names_2m'] = filename_int2m
data_save['fc_2m'] = fc_2m

data_save['ts4m'] = ts4m
data_save['names_4m'] = filename_int4m
data_save['fc_4m'] = fc_4m
names = ['corr', 'clarksondist']
#%%
def extract_FCD(data, L = 50, mode = 1, dt = 1, steps = 1):
    """
    Computes the FCD matrix. First, calculates all the Functional Connectivity
    (FC) matrices over time. Next, Uses the FCs matrices to build the FCD matrix.

    Parameters
    ----------
    data : txN numpy array.
           time series. Rows represent the time, and columns the nodes.
    L : integer.
        Time windows' length in time units (e.g., seconds).
    mode : integer.
           1: Pearson Correlation based distance (values between 0 and 1).
           2: Clarkson distance (values between 0 and 1).
    dt : float.
         inverse of the sampling rate in time units (e.g., seconds).
    steps : integer > 0.
            number of points to advance for calculating the next FC.
    Returns
    -------
    FCD : LxL numpy array.
          Functional Connectivity Dynamics matrix with L total time windows.
    L_points : integer.
               Total number of time windows.
    steps : integer > 0.
            The selected step.
    """

    nnodes = data.shape[1] #Number of nodes

    L_points = int(np.round(L / dt, 0)) #Time windows' length in points

    N_windows = (data.shape[0] - L_points) // steps + 1 #Total number of time windows

    all_corr_matrix = [] #vector to append the FCs matrices

    #FCs built using the Pearson Correlation
    #and neglecting negative values.
    for i in range(0,N_windows):
        idx1, idx2 = 0 + i * steps, L_points + i *steps
        corr_matrix = np.corrcoef(data[idx1:idx2,:].T)
        np.fill_diagonal(corr_matrix, 0)
        all_corr_matrix.append(corr_matrix)

    #Vectorized versions of FCs matrices.
    corr_vectors = np.array([allPm[np.tril_indices(nnodes, k = -1)] for allPm in all_corr_matrix])

    X = np.shape(corr_vectors)[0]
    FCD = np.zeros((X,X))

    if mode in [1,2]:
        modeFCD = names[mode-1]
    else:
        raise ValueError('Select a valid mode for the FCD')

    #Computing the FCD
    if modeFCD == 'corr': #Correlation-based distance
        CV_centered=corr_vectors - np.mean(corr_vectors,-1)[:,None]
        # FCD = 1 - np.abs(np.corrcoef(CV_centered))
        FCD = (np.corrcoef(CV_centered))
    elif modeFCD == 'clarksondist': #Clarkson distance
        for ii in range(X):
            for jj in range(ii):
                FCD[ii,jj]= LA.norm(corr_vectors[ii,:]/LA.norm(corr_vectors[ii,:]) - corr_vectors[jj,:]/LA.norm(corr_vectors[jj,:]))
                FCD[jj,ii]=FCD[ii,jj]
        FCD /= np.sqrt(2)

    return(FCD, L_points, steps,corr_vectors)

fcd,l_points, steps,corr_vectors = extract_FCD(timeseries,L=30,mode=1,steps=1)


plt.figure(4)
plt.clf()
# plt.imshow(corr_vectors.T, aspect='auto', interpolation='none',cmap='RdBu_r')
plt.imshow(fcd, aspect='auto', interpolation='none',cmap='RdBu_r')
plt.colorbar()
# plt.clim(-0.5,0.5)


def FCD_vars(FCD, L_points, steps, bins = 20, vmin = 0, vmax = 1):
    """
    Calculates the FCD's speed (dtyp) and FCD's variance (varFCD). The typical
    FCD speed corresponds to the median of the histogram of the FCD values,
    through the diagonal of the FCD with a L_points/steps offset. The varFCD
    is computed as the variance of FCD values of the upper triangle of the FCD
    matrix (using the same offset).

    Parameters
    ----------
    FCD : LxL numpy array.
          Functional Connectivity Dynamics matrix with L total time windows.
    L_points: integer.
              Total number of time windows.
    steps : integer > 0.
            number of time points used to advance between consecutives FCs.
    bins : integer > 0.
           number of bins (intervals) for building the histogram of FCD values.
    vmin, vmax : float, vmax > vmin.
                 Limits of the histogram.
    Returns
    -------
    dtyp : float.
           Typical FCD speed.
    varFCD : float.
             variance of the FCD.
    """


    offset = int(L_points / steps) #FCD values away from the diagonal

    distance = [FCD[XY,XY + offset] for XY in range(len(FCD) - offset)]
    histogram = np.histogram(distance, bins, range = (vmin,vmax))
    dtyp = histogram[1][np.argmax(histogram[0])] #FCD speed

    varFCD = np.var(FCD[np.triu_indices(FCD.shape[0], k = offset)]) #FCD variance

    return([dtyp,varFCD,distance])

dtyp,varFCD,distance=FCD_vars(fcd, l_points, 30, bins = 3, vmin = 0, vmax = 1)


#%%


def tapis_FC(FCtime):
    """
    Generate one triangle part of the FC in time windows

    Parameters
    ----------
    FCtime : numpy array
        W x M x M. The FC matrices, W is the total number of FCs

    Raises
    ------
    ValueError
        If the array is not 3-dimension, or if the FCs are not square.

    Returns
    -------
    Pcorr : numpy array
        Unraveled FC matrices, with dimension W x (Mx(M-1)/2). Each row is a couple
        of channels.
    Code Samy Castro and Patricio Orio 2023
    """
    if len(FCtime.shape) != 3:
        raise ValueError("Array must be 3-dimension")
    T,M,N = FCtime.shape # time window, electrode PLI1, electrode PLI2
    if M!=N:
        raise ValueError("Last 2 dimensions must match (expecting an array of square matrices)")

    Pcorr = np.zeros((T,int(M*(M-1)/2)))
    for t,fc in enumerate(FCtime):
        Pcorr[t] = np.abs(fc[np.triu_indices_from(fc,1)])
    return Pcorr


def calc_FCD(FCtime):
    """
    Calculate the FCD matrix from an array of FCs

    Parameters
    ----------
    FCtime : numpy array
        W x M x M. The FC matrices, W is the total number of FCs

    Raises
    ------
    ValueError
        If the array is not 3-dimension, or if the FCs are not square.

    Returns
    -------
    Pcorr : numpy array
        Unraveled FC matrices, with dimension W x (Mx(M-1)/2). Each row is a couple
        of channels.
    FCDmat : numpy array
        W x W. FCD matrix calculated as Pearson correlation between abs(FCs).

    """
    if len(FCtime.shape) != 3:
        raise ValueError("Array must be 3-dimension")
    T,M,N = FCtime.shape
    if M!=N:
        raise ValueError("Last 2 dimensions must match (expecting an array of square matrices)")

    Pcorr = np.zeros((T,int(M*(M-1)/2)))
    for t,fc in enumerate(FCtime):
        Pcorr[t] = np.abs(fc[np.triu_indices_from(fc,1)])
    FCDmat = np.corrcoef(Pcorr)
    Pcorr = Pcorr.T
    # Pcorr = Pcorr

    return Pcorr, FCDmat



#%%
def conn_dyn(data, sf=100, nper = 50, overlap=0.9):
    """
    Connectivity dynamics. Calculate a time-resolved Phase Lag Index functional connectivity.
    Needs to be provided in a frequency band.

    Parameters
    ----------
    data : numpy array
        T x M array. T is time points and M are channels
    rh : list
        List containing rhythm [min freq, max freq, period in s] of the band to analyze.
    sf : float, optional
        Sampling frequency (samples /s). The default is 1000.
    nper : float, optional
        Number of periods to be contained in the windows. The default is 10.
    overlap : float, optional
        Fraction of overlap between successive windows. 0 <= overlap < 1. The default is 0.5.

    Returns
    -------
    time_points : numpy array
        Length W. Array containing the time at the mid-point of each window.
    PLI_mat : numpy array
        Time-resolved FC. The value is signed PLI, so for most applications the abs
        must be taken. W x M x M. W is the number of windows, M the number of channels

    """
    # minF, maxF, perL    = rh
    windowlength        = int(nper) #1cycleXseg, number of cycles in windows, sampling frequency
    P,N                 = data.shape #time, electrodes

    points_to_ext       = int(np.maximum(P*0.05, windowlength*2)) #Number of points in the windows
    signal_ext          = np.concatenate((data[points_to_ext:0:-1], data,
                                 data[-2:-points_to_ext:-1]),axis=0) #reverse point_to_ext, data,

    # b,a = signal.bessel(2, [minF * 2/sf, maxF * 2/sf], btype='bandpass')
    # b,a                 = signal.bessel(2, [minF /(sf/2), maxF /( sf/2)], btype='bandpass') #The dessign of the filter
    # Vfilt               = signal.filtfilt(b,a,signal_ext,axis=0)

    # analytic            = signal.hilbert(Vfilt,axis=0)
    # analytic            = analytic[points_to_ext:-points_to_ext+2]

    windowstart         = np.arange(0,len(data)-windowlength,windowlength*(1-overlap), dtype=int)
    windowend           = windowstart + windowlength
    time_points         = ((windowend+windowstart)/2)/sf

    PLI_mat             = np.zeros((len(windowstart),N,N))

    #PLI Calculation
    for i,(i_init,i_end) in enumerate(zip(windowstart,windowend)):
        # cross_signal    = analytic[i_init:i_end,:,None] * np.conjugate(analytic[i_init:i_end,None,:])
        # pearson    = data[i_init:i_end,:,None] * np.conjugate(analytic[i_init:i_end,None,:])
        # PLI_mat[i]      = np.mean(np.sign(cross_signal.imag),0)
        pearson    = np.corrcoef(data[i_init:i_end].T)# * np.conjugate(analytic[i_init:i_end,None,:])

        PLI_mat[i]      = np.abs(pearson)

    return time_points, PLI_mat
#%%
# =============================================================================
# Metaconnectivity
# =============================================================================

def calc_FCD(FCtime):
    """
    Calculate the FCD matrix from an array of FCs

    Parameters
    ----------
    FCtime : numpy array
        W x M x M. The FC matrices, W is the total number of FCs

    Raises
    ------
    ValueError
        If the array is not 3-dimension, or if the FCs are not square.

    Returns
    -------
    Pcorr : numpy array
        Unraveled FC matrices, with dimension W x (Mx(M-1)/2). Each row is a couple
        of channels.
    FCDmat : numpy array
        W x W. FCD matrix calculated as Pearson correlation between abs(FCs).

    """
    if len(FCtime.shape) != 3:
        raise ValueError("Array must be 3-dimension")
    T,M,N = FCtime.shape
    if M!=N:
        raise ValueError("Last 2 dimensions must match (expecting an array of square matrices)")

    Pcorr = np.zeros((T,int(M*(M-1)/2)))
    for t,fc in enumerate(FCtime):
        Pcorr[t] = np.abs(fc[np.triu_indices_from(fc,1)])
    FCDmat = np.corrcoef(Pcorr)
    Pcorr = Pcorr.T
    # Pcorr = Pcorr

    return Pcorr, FCDmat

time_points, PLI_mat = conn_dyn(timeseries,1)
Pcorr, FCDmat =calc_FCD(PLI_mat)

# corr_vectors


plt.figure(6)
plt.clf()
# plt.imshow(corr_vectors.T, aspect='auto', interpolation='none',cmap='RdBu_r')
plt.imshow(FCDmat, aspect='auto', interpolation='none',cmap='RdBu_r')
plt.colorbar()
# plt.clim(-0.5,0.5)


#%%
#metaconnectivity
# pcorr = tapis_FC(PLI_mat)

# mc = np.abs(np.corrcoef(Pcorr))
mc = np.corrcoef(Pcorr)

# community_structure,q_statistic = bct.modularity.community_louvain(np.abs(mc),gamma=0.9)
community_structure,q_statistic = bct.modularity.modularity_louvain_und_sign(mc,gamma=1.25)
print(np.unique(community_structure),q_statistic)

sorted_community_structure = np.argsort(community_structure)

mc_mod = mc[:,sorted_community_structure][sorted_community_structure,:]


plt.figure(7)
plt.clf()
plt.imshow(mc, aspect='auto', interpolation='none',cmap='RdBu_r')
# plt.imshow(mc_mod, aspect='auto', interpolation='none',cmap='RdBu_r')
plt.colorbar()
plt.clim(-1,1)





#%%}










plt.figure(1)
plt.clf()
plt.subplot(321)
plt.plot(timeseries)
# plt.imshow(data, aspect='auto')

plt.subplot(322)
plt.title('dFC stream')
plt.imshow(corr_vectors.T, aspect='auto', interpolation='none',cmap='RdBu_r')
plt.colorbar()

plt.subplot(323)
plt.title('fc sorted mod')
# plt.imshow(fc_mod, aspect='auto', interpolation='none',cmap='RdBu_r')
plt.colorbar()
# plt.clim(-0.5,0.5)
plt.clim(0,0.5)

plt.subplot(324)
plt.title('fcd')
plt.imshow(fcd, aspect='auto', interpolation='none',cmap='RdBu_r')
plt.clim(0,0.5)
plt.colorbar()

plt.subplot(325)
# plt.title('speed(1-corr)')
plt.hist(distance,histtype='step')#, aspect='auto', interpolation='none',cmap='RdBu_r')

plt.subplot(326)
plt.title('Metaconnectivity')
plt.imshow(mc_mod, aspect='auto', interpolation='none',cmap='viridis')
plt.colorbar()



