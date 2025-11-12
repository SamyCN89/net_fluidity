#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 00:16:53 2025

@author: samy
"""

import numpy as np
import matplotlib.pyplot as plt
import brainconn as bct
import os
import pandas as pd
from pathlib import Path
import copy
import pickle
from tqdm import tqdm

from itertools import combinations_with_replacement
from joblib import Parallel, delayed, parallel_backend

from fun_dfcspeed import ts2dfc_stream
from fun_loaddata import *
from fun_optimization import fast_corrcoef#, fast_corrcoef_numba, fast_corrcoef_numba_parallel
# import time
# from functions_analysis import *
# from scipy.io import loadmat, savemat
# from scipy.special import erfc
# from scipy.stats import pearsonr, spearmanr

# from scipy.spatial.distance import squareform
# from scipy.cluster.hierarchy import linkage, fcluster




def variables_selector(bool_index1, bool_index2, index1_label = 'Good', index2_label = 'Impaired'):
    
    index1 = np.tile(bool_index1,2)
    index2 = np.tile(bool_index2,2)
    
    #animals specific index
    index1_cond_age1 = np.logical_and(index1, is_2month_old)
    index1_cond_age2 = np.logical_and(index1, ~is_2month_old)
    index2_cond_age1 = np.logical_and(index2, is_2month_old)
    index2_cond_age2 = np.logical_and(index2, ~is_2month_old)
    
    variables_setbool = (index1_cond_age1, index1_cond_age2, 
                         index2_cond_age1, index2_cond_age2)
    
    label_variables = (index1_label + ' 2m', index1_label + ' 4m', index2_label + ' 2m', index2_label + ' 4m')

    return index1, index2, index1_label, index2_label, variables_setbool, label_variables

#%%Metaconnectivity
def compute_metaconnectivity(ts_data, window_size=7, lag=1, return_dfc=False, save_path=None, n_jobs=-1):
    """
    Compute or load cached meta-connectivity (MC) from time-series data in parallel.

    Parameters:
    - ts_data: np.ndarray (n_animals, n_regions, n_timepoints)
    - window_size: int
    - lag: int
    - return_dfc: bool
    - save_path: str or None
    - n_jobs: int, number of parallel jobs (-1 = all cores)

    Returns:
    - mc: np.ndarray
    - dfc_stream: np.ndarray (optional)
    """

    n_animals, tr_points, nodes  = ts_data.shape
    dfc_stream  = None
    mc          = None

    # File path setup
    save_path = Path(save_path) if save_path else None
    full_save_path = (
        save_path / f"mc_window_size={window_size}_lag={lag}_animals={n_animals}_regions={nodes}.npz"
        if save_path else None
    )
    if full_save_path:
        full_save_path.parent.mkdir(parents=True, exist_ok=True)
        # full_save_path = os.path.join(save_path, f'mc_window_size={window_size}_lag={lag}_animals={n_animals}_regions={nodes}.npz')
        # os.makedirs(os.path.dirname(full_save_path), exist_ok=True)

    # Load from cache
    if full_save_path and full_save_path.exists():
        print(f"Loading meta-connectivity from: {full_save_path}")
        data = np.load(full_save_path, allow_pickle=True)
        mc = data['mc']
        dfc_stream = data['dfc_stream'] if return_dfc and 'dfc_stream' in data else None

    else:
        print(f"Computing meta-connectivity in parallel (window_size={window_size}, lag={lag})...")

        # Parallel DFC stream computation per animal
        with parallel_backend("loky", n_jobs=n_jobs):
            dfc_stream_list = Parallel()(
                delayed(ts2dfc_stream)(ts_data[i], window_size, lag, format_data='2D')
                # for i in tqdm(range(n_animals), desc="DFC Streams")
                for i in range(n_animals)
            )
        dfc_stream = np.stack(dfc_stream_list)

        # Parallel MC matrices per animal
        with parallel_backend("loky", n_jobs=n_jobs):
            mc_list = Parallel()(
                delayed(fast_corrcoef)(dfc.T)
                # for dfc in tqdm(dfc_stream, desc="Meta-connectivity")
                for dfc in dfc_stream
                )
        mc = np.stack(mc_list)

        # Save results if path is provided
        if full_save_path:
            print(f"Saving meta-connectivity to: {full_save_path}")
            if return_dfc:
                np.savez_compressed(full_save_path, mc=mc, dfc_stream=dfc_stream if return_dfc else None)
            else:
                np.savez_compressed(full_save_path, mc=mc)
    # print(f"Max RAM usage during run: {max(all_mem_use):.2f} MB")
    return (mc, dfc_stream) if return_dfc else mc


#%%
# =============================================================================
# Analysis on Metaconnectivity 
# =============================================================================
def fun_mc_viscocity(data):
    """
    Compute viscocity from array of trials and their MC, the first dimension must be the trials

    Parameters
    ----------
    data : N,M,M np.array 
        the trials MC.

    Returns
    -------
    mc_viscocity_val : N, MC neg values
        The first dimesion is trials, the next is variable and are list of the negative values.
    mc_viscocity_mask : M,M bool
        The mask is a boolean of the data that is true for negative value.
    """
    n_trials = data.shape[0]
    
    data = copy.deepcopy(data)
    mc_viscocity_mask = (data<0)
    mc_viscocity_val = np.array([data[i,mc_viscocity_mask[i]] for i in range(n_trials)] ,dtype='object')
    return mc_viscocity_val, mc_viscocity_mask

#%%
# =============================================================================
# Module allegiance Using Louvain method 
#Maybe add Leiden
# =============================================================================


def _run_louvain(mc_data, gamma):
    Ci, Q = bct.modularity.modularity_louvain_und_sign(mc_data, gamma=gamma)
    return np.asanyarray(Ci, dtype=np.int32), Q

def _build_agreement_matrix(communities):
    """
    Vectorized computation of agreement matrix from community labels.
    """
    n_nodes = communities[0].shape[0]
    agreement = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    for Ci in communities:
        # agreement += (Ci[:, None] == Ci[None, :])
        agreement += (Ci[:, None] == Ci)

    return agreement

def contingency_matrix_fun(n_runs, mc_data, gamma_range=10, gmin= 0.8, gmax=1.3, cache_path=None, ref_name='', n_jobs=-1):
    """
    Compute or load a contingency matrix from community detection runs using joblib and vectorized agreement matrix.
    """

    n_nodes = mc_data.shape[0]
    gamma_mod = np.linspace(gmin, gmax, gamma_range)
    
    if cache_path:
        cache_dir = Path(cache_path)
        cache_dir.mkdir(parents=True, exist_ok = True)
        full_cache_path = cache_dir / f'contingency_matrix_ref={ref_name}_regions={n_nodes}_nruns={n_runs}_gamma_repetitions={gamma_range}'
        if full_cache_path.exists():
            with full_cache_path.open('rb') as f:
                print(f"[cache] Loading contingency matrix from {full_cache_path}")
                return pickle.load(f)
    else:
        full_cache_path = None

    contingency_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    gamma_qmod_val = np.zeros((gamma_range, n_runs), dtype=np.float64)
    gamma_agreement_mat = np.zeros((gamma_range, n_nodes, n_nodes), dtype=np.float64)

    for idx, gamma in enumerate(tqdm(gamma_mod, desc="Gamma values")):
        # Louvain with per-run progress bar
        results = list(tqdm(
            Parallel(n_jobs=n_jobs)(
                delayed(_run_louvain)(mc_data, gamma) for _ in range(n_runs)
            ),
            total=n_runs,
            desc=f"Gamma {gamma:.2f}"
        ))

        communities, modularities = zip(*results)
        communities = np.array([np.array(c) for c in communities])
        gamma_qmod_val[idx] = modularities

        # Efficient agreement accumulation
        agreement =_build_agreement_matrix(communities)
        gamma_agreement_mat[idx] = agreement

        contingency_matrix += agreement
        

    contingency_matrix /= (n_runs * gamma_range)

    # Save to cache
    if full_cache_path is not None:
        with full_cache_path.open('wb') as f:
            pickle.dump((contingency_matrix, gamma_qmod_val, gamma_agreement_mat), f)
            print(f"[cache] Saved to {full_cache_path}")

    return contingency_matrix, gamma_qmod_val, gamma_agreement_mat

#%%
def allegiance_matrix_analysis(mc_mean_template, n_runs=100, gamma_pt=10, cache_path=None, ref_name='', n_jobs=-1):
    """
    Wrapper to compute allegiance communities and sorting indices.

    Parameters
    ----------
    mc_mean_template : ndarray
        Mean meta-connectivity matrix.
    n_runs : int
        Number of repetitions per gamma.
    gamma_pt : int
        Number of gamma values.
    cache_path : str or None
        Path to cache contingency matrix result.

    Returns
    -------
    allegancy_communities : ndarray
        Final Louvain community labels.
    argsort_allegancy_communities : ndarray
        Sorting indices by community.
    """
    print('here',cache_path)
    # contingency_matrix, gamma_mean, gamma_std = contingency_matrix_fun_old(
    contingency_matrix, _, _ = contingency_matrix_fun(
        n_runs=n_runs, 
        mc_data=mc_mean_template, 
        gamma_range=gamma_pt, 
        cache_path=cache_path, 
        ref_name=ref_name,
        n_jobs=n_jobs
    )

    

    allegancy_communities, allegancy_modularity_q = bct.modularity.modularity_louvain_und(contingency_matrix, gamma=1.2)
    argsort_allegancy_communities = np.argsort(allegancy_communities)

    return allegancy_communities, argsort_allegancy_communities, allegancy_modularity_q, contingency_matrix

#%%
def fun_allegiance_communities(mc_data, n_runs=1000, gamma_pt=100, ref_name=None, save_path=None, n_jobs=-1):
    """
    Compute allegiance communities from a single or multiple mc matrices.

    Parameters:
        mc_data: 2D or 3D ndarray
        n_runs: int
        gamma_pt: float
        ref_name: str
        save_path: Path
        n_jobs: int
    Returns:
        communities, sort_idx, contingency_matrix
    """

    def process_single(mc_matrix):#, n_runs = 10, gamma_pt = 10, ref_name='', save_path=None, n_jobs=-1): # gamma number of points in the defined range
        #allegiance index, argsort, Q value
        communities, sort_idx, _, contingency = allegiance_matrix_analysis(mc_matrix, 
                                                                           n_runs=n_runs, 
                                                                           gamma_pt=gamma_pt, 
                                                                           cache_path=save_path, 
                                                                           ref_name=ref_name, 
                                                                           n_jobs=n_jobs,
                                                                           )
        communities = communities[sort_idx]
        return communities, sort_idx, contingency
        

    if mc_data.ndim == 3:
        allegiances = []
        for i in range(mc_data.shape[0]):
            _, allegiance, _ = process_single(mc_data[i])
            allegiances.append(allegiance)
        mean_allegiance = np.mean(allegiances, axis=0)
        communities, sort_idx, contingency = process_single(mean_allegiance)
    elif mc_data.ndim == 2:
        communities, sort_idx, contingency = process_single(mc_data)
    else:
        raise ValueError("Input mc_data must be 2D or 3D.")

    if save_path and ref_name:
        np.savez_compressed(
            Path(save_path) / f"allegiance_{ref_name}.npz",
            communities=communities,
            sort_idx=sort_idx,
            contingency=contingency
        )

    return communities, sort_idx, contingency

#%%
# =============================================================================
# Modularity
# =============================================================================
def intramodule_indices_mask(allegancy_communities):
    
    n_2 = len(allegancy_communities)

    # Dictionary mapping module → list of node indices in that module
    intramodules_idx = {
        mod: np.where(mod == allegancy_communities)[0] 
        for mod in np.unique(allegancy_communities)}
    
    # Build an array of (mod, i, j) for every intra-module pair (i, j)
    # Uses combinations_with_replacement to include self-connections (i == j)
    # pairs = [(mod, list(combinations_with_replacement(intramodules_idx[1], 2))) for mod in np.unique(allegancy_communities)]
    intramodule_indices = np.array([
        (mod, i, j)
        for mod in np.unique(allegancy_communities)
        for i, j in combinations_with_replacement(intramodules_idx[mod], 2)
        ]).T
    
    # intramodule_indices[:,intramodule_indices[0]==1]
    
    mc_modules_mask = np.zeros((n_2, n_2))
    for ind, mod in enumerate(range(1, np.max(np.unique(allegancy_communities))+1)):
        idx = np.abs(intramodules_idx[mod])
        mc_modules_mask[np.ix_(idx, idx)] = ind + 1
    
    return intramodules_idx, intramodule_indices, mc_modules_mask

# =============================================================================
# Trimers
# =============================================================================
def get_fc_mc_indices(regions):
    fc_idx = np.array(np.tril_indices(regions, k=-1)).T
    mc_idx = np.array(np.tril_indices(fc_idx.shape[0], k=-1)).T
    return fc_idx, mc_idx

def get_mc_region_identities(fc_idx, mc_idx, sort_ref):
    aux_fc = fc_idx[sort_ref]
    fc_reg_idx = aux_fc[mc_idx]  # shape: (n_mc, 2, 2)
    mc_reg_idx = fc_reg_idx.reshape(-1, 4).T  # shape: (4, n_mc)
    return mc_reg_idx, fc_reg_idx

def compute_trimers_identity(regions):
    fc_idx, mc_idx = get_fc_mc_indices(regions)
    aux_identity = fc_idx[mc_idx]
    mc_reg_idx = aux_identity.reshape(-1, 4)  # shape: (n_mc, 4)

    # Find trimers: exactly 3 unique nodes among the 4 defining a meta-connection
    unique_counts = np.array([len(np.unique(row)) for row in mc_reg_idx])
    trimer_mask = unique_counts == 3
    trimer_idx = mc_idx[trimer_mask].T
    trimer_reg_id = mc_reg_idx[trimer_mask]

    # Find apex node (node that appears twice)
    trimer_apex = np.array([
        np.unique(row, return_counts=True)[0][
            np.unique(row, return_counts=True)[1] > 1
        ][0] if len(np.unique(row, return_counts=True)[0][
            np.unique(row, return_counts=True)[1] > 1]) > 0 else np.nan
        for row in trimer_reg_id
    ])

    return trimer_idx, trimer_reg_id, trimer_apex

def build_trimer_mask(trimer_idx, trimer_apex, n_fc_edges):
    mask = np.zeros((n_fc_edges, n_fc_edges))
    np.fill_diagonal(mask, np.nan)
    for i in range(trimer_idx.shape[1]):
        a, b = trimer_idx[0, i], trimer_idx[1, i]
        apex_val = trimer_apex[i] + 1  # optional +1 offset
        mask[a, b] = mask[b, a] = apex_val
    return mask

#%%
def trimers_by_apex(trimer_values, trimer_reg_apex):
    """
    Splits trimer MC values by apex region and group.
    
    Parameters
    ----------
    mc_values : ndarray, shape (n_animals, n_trimers)
        Trimer values per subject.
    trimer_reg_apex : ndarray, shape (n_trimers,)
        Apex region for each trimer.
    index1, index2 : boolean arrays
        Group masks (e.g., Good vs Impaired)
    
    Returns
    -------
    regval_index1 : list of arrays
        Each entry: trimer values for group 1 animals, per apex.
    regval_index2 : list of arrays
        Each entry: trimer values for group 2 animals, per apex.
    apex_ids : ndarray
        Unique apex region IDs, in order.
    """
    unique_apexes = np.unique(trimer_reg_apex)
    
    regval = [
        trimer_values[:, trimer_reg_apex == apex]
        for apex in unique_apexes
        ]  # List of shape (n_animals, n_trimers_per_apex)

    return regval

# trimers_per_region = np.array(trimers_by_apex(trimers_mc_values, trimer_reg_apex))