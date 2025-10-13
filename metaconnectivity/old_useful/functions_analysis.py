#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 17:51:30 2023

@author: samy
"""

import numpy as np
from scipy.io import savemat
from scipy import signal
import copy 
from h5py import File
import os


# =============================================================================
# Shufled functions
# =============================================================================

def randomShift(data,N=50,cols=True):
    """
    Generate a N number of surrogate time-series, by random shifting of the time series

    Parameters
    ----------
    data : NxM array 
        the datapoints os large N in the different time-series M.
    N : int, optional
        The number of surrogates to generate. The default is 50.
    cols : Bool, optional
        Swipes the firs and last axes of the data. The default is True.

    Returns
    -------
    outSeries : TYPE
        DESCRIPTION.
        
    Codigo Patricio Orio Abril 2023 
    """
    if cols:
        data2   = data.T
    else:
        data2   = np.copy(data)
    
    D,L         = data2.shape
    outSeries   = []
    
    for i in range(N):
        shifts  = np.random.randint(1,L,D)
        serie   = [np.r_[d[s:],d[:s]] for d,s in zip(data2,shifts)]
        print(serie)
        outSeries.append(serie)
    
    outSeries=np.array(outSeries)
    
    if cols:
        outSeries=np.swapaxes(outSeries,1,2)
    return outSeries 


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

def conn_dyn(data, rh, sf=1000, nper = 10, overlap=0.5):
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
    minF, maxF, perL    = rh
    windowlength        = int(perL*nper*sf) #1cycleXseg, number of cycles in windows, sampling frequency
    P,N                 = data.shape #time, electrodes
    
    points_to_ext       = int(np.maximum(P*0.05, windowlength*2)) #Number of points in the windows
    signal_ext          = np.concatenate((data[points_to_ext:0:-1], data, 
                                 data[-2:-points_to_ext:-1]),axis=0) #reverse point_to_ext, data,    
    
    # b,a = signal.bessel(2, [minF * 2/sf, maxF * 2/sf], btype='bandpass')
    b,a                 = signal.bessel(2, [minF /(sf/2), maxF /( sf/2)], btype='bandpass') #The dessign of the filter
    Vfilt               = signal.filtfilt(b,a,signal_ext,axis=0)
    
    analytic            = signal.hilbert(Vfilt,axis=0)
    analytic            = analytic[points_to_ext:-points_to_ext+2]
    
    windowstart         = np.arange(0,len(data)-windowlength,windowlength*(1-overlap), dtype=int)
    windowend           = windowstart + windowlength
    time_points         = ((windowend+windowstart)/2)/sf

    PLI_mat             = np.zeros((len(windowstart),N,N))
    
    #PLI Calculation
    for i,(i_init,i_end) in enumerate(zip(windowstart,windowend)):
        cross_signal    = analytic[i_init:i_end,:,None] * np.conjugate(analytic[i_init:i_end,None,:])
        
        PLI_mat[i]      = np.mean(np.sign(cross_signal.imag),0)
        
    return time_points, PLI_mat

def thr_std(surr, FC_t, std_times=1):
    """
    Generate the threshold by the surrogate data, by N times the standard deviation, i.e. std_times

    Parameters
    ----------
    surr : TYPE
        DESCRIPTION.
    std_times : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    FC_t_surr : TYPE
        DESCRIPTION.

    """
    surr_mean = np.mean(surr,axis=0)
    surr_std = np.std(surr,axis=0)
    surr_mean_t = tapis_FC(surr_mean)
    surr_std_t = tapis_FC(surr_std)
    
    thr_high = (surr_mean_t+(surr_std_t*std_times))
    thr_low = (surr_mean_t-(surr_std_t*std_times))

    mask = np.logical_or((FC_t>thr_high), (FC_t<thr_low))    
    FC_t_surr = copy.deepcopy(FC_t)
    
    FC_t_surr[mask==0] = 0
    
    return FC_t_surr


def PLI(data, file_name, sampling_frequency = 1000, nperc=2, overlap = 0.5, NumSurr=50, folder_to_save='wPLI'):
    """
    Calculate a time-resolved functional connectivity of PLI in different frequency bands and then save it in the folder_to_save

    Parameters
    ----------
    data : numpy array
        T x M array. T is time points and M are channels
    file name : string
        The name of the mice.
    sampling_frequency : float, optional
        Sampling frequency (samples /s). The default is 1000.
    nperc : float, optional
        Number of periods to be contained in the windows. The default is 10.
    overlap : float, optional
        Fraction of overlap between successive windows. 0 <= overlap < 1. The default is 0.5.
    folder_to_save : string
        Name of the folder to save. Returns error if not exist the folder    

    Returns
    -------
    In the folder_to_save returs the .mat with the analyzed data

    """
    #Data info
    n_signals = int(len(data[0,:]))
    
    time_max = len(data[:,0])/sampling_frequency
    time_extent = (0, time_max)
    
    n_time_samples = int((time_extent[1] - time_extent[0]) * sampling_frequency) 
    time = np.linspace(time_extent[0], time_extent[1], num=int(n_time_samples), endpoint=True)
    
    #WT signals
    signal = np.zeros((n_time_samples, n_signals))
    signal[:,:] = data[:,:]
    
    #For rhythms: min,max and time_window
    delta = [0.5,4, 0.5*nperc]
    theta = [4,12, 0.125*nperc]
    beta = [15,30, 0.05*nperc]
    gamma = [30,100, 0.016666*nperc]
    #List of rhythms to analyze
    rhythm = (delta, theta, beta, gamma) 
    label_rhythm = ('delta', 'theta', 'beta', 'gamma')
    
    for ind, rh in enumerate(rhythm):	
        print(ind)
        
        time_points, PLI_mat = conn_dyn(signal, rh=rh, sf=sampling_frequency, nper=nperc, overlap=overlap)
        PLI_abs = abs(PLI_mat)

        #FCt analysis
        FCt = tapis_FC(PLI_abs)

        #Surrogate analysis
        surrX = randomShift(data, N=NumSurr)
        _,surrA = zip(*[conn_dyn(sX,rh=rh) for sX in surrX])
        surrA2 = abs(np.array(surrA))
        
        #Threshold of FCt by std 
        FC_t_surr1 = thr_std(surrA2, FCt, std_times=1)
        FC_t_surr2 = thr_std(surrA2, FCt, std_times=2)
        FC_t_surr3 = thr_std(surrA2, FCt, std_times=3)        
        
        #Save data
        data_PLI = {}
        data_PLI['wPLI'] = PLI_abs
        data_PLI['time_points'] = time_points
        data_PLI['FCt'] = FCt
        data_PLI['FCt_surr1'] = FC_t_surr1
        data_PLI['FCt_surr2'] = FC_t_surr2
        data_PLI['FCt_surr3'] = FC_t_surr3
        
        savemat('../results/%s/%s_PLI_rhythm_%s_nperc=%s_overlapx%s.mat'%(folder_to_save,file_name, label_rhythm[ind], nperc, overlap), data_PLI)

def PLI_cluster(file_name, sampling_frequency = 1000, nperc=16, overlap = 0.1, NumSurr=20, folder_to_save='wPLI'):
    """
    Calculate a time-resolved functional connectivity of PLI adn save it in the folder_to_save

    Parameters
    ----------
    data : numpy array
        T x M array. T is time points and M are channels
    file name : string
        The name of the mice.
    sampling_frequency : float, optional
        Sampling frequency (samples /s). The default is 1000.
    nperc : float, optional
        Number of periods to be contained in the windows. The default is 10.
    overlap : float, optional
        Fraction of overlap between successive windows. 0 <= overlap < 1. The default is 0.5.
    folder_to_save : string
        Name of the folder to save. Returns error if not exist the folder    

    Returns
    -------
    In the folder_to_save returs the .mat with the analyzed data

    """
    hash_dir = '/home/samy/Bureau/Proyect/Matthieu/data/example_info/'
    dict_load = File(hash_dir+file_name,'r')
    dict_key = list(dict_load.keys())[0]
    data = np.array(dict_load[dict_key])

    PLI(data, file_name, sampling_frequency = 1000, nperc=16, overlap = 0.1, NumSurr=20, folder_to_save='wPLI')
        
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

def PLI_rate(PLI, pairwise_num, thr_num = 100, thr_max =0.3):

    #Pcorr
    PLI = abs(PLI)
    Pcorr, _ = calc_FCD(PLI)
    
    thr_range = np.linspace(0.001, thr_max,thr_num)
    
    rate_pli = np.zeros((thr_num, pairwise_num))
    
    for thr in range(thr_num):
        aux_true_pcorr = (Pcorr >thr_range[thr])*1# (PLI_theta_nshape>0.4)*1
        rate_pli[thr] = np.sum(aux_true_pcorr,axis=1)/Pcorr.shape[1]
        
    return rate_pli, thr_range, Pcorr

def metaconn_pli(pli, thr_pli=0.2):
    mask_pcorr_thr = ((pli > thr_pli)*1)
    mc_pli = np.corrcoef(pli * mask_pcorr_thr, pli * mask_pcorr_thr)
    return mc_pli
