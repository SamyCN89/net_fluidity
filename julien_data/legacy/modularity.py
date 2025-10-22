#%%
# 
from pathlib import Path
import numpy as np
import pickle
import logging
from tqdm import tqdm
import gc

from joblib import Parallel, delayed
from typing import Dict, List, Tuple, Optional, Union

from webcolors import name_to_rgb_percent

from class_dataanalysis_julien import DFCAnalysis
from shared_code.fun_loaddata import save_pickle
from shared_code.fun_utils import set_figure_params
from shared_code.fun_dfcspeed import ts2fc, ts2dfc_stream
from shared_code.fun_metaconnectivity import contingency_matrix_fun
processors = -1  # Use all available processors

data = DFCAnalysis()

#Preprocessed data
data.get_metadata()
data.get_ts_preprocessed()
data.get_cogdata_preprocessed()
data.get_temporal_parameters()




# %%
ts = data.ts
fc = np.array([ts2fc(ts[xx]) for xx in tqdm(range(len(ts)))])

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.imshow(fc[0], cmap='viridis', vmin=0, vmax=1)

wt_veh = data.groups[('WT', 'VEH')]


fc_wt_veh = fc[wt_veh]
gamma_n = 20
n_runs=500
n_regions = data.regions
allegiance_mat = np.zeros((len(fc_wt_veh), n_regions, n_regions))
gamma_qmod_val = np.zeros((len(fc_wt_veh), gamma_n, n_runs))
gamma_agreement_mat = np.zeros((len(fc_wt_veh), gamma_n, n_regions, n_regions))
communities_mat = np.zeros((len(fc_wt_veh), gamma_n, n_runs,n_regions))

#%%

# Consensus matrix
gmin= 0.5
gmax=2
gamma = np.linspace(gmin, gmax, gamma_n)   
for i, fc_mat in enumerate(tqdm(fc_wt_veh)):
    allegiance_mat[i], gamma_qmod_val[i], gamma_agreement_mat[i], communities_mat[i] = contingency_matrix_fun(n_runs, fc_mat, gamma_range=gamma_n, gmin=gmin, gmax=gmax, cache_path=None, ref_name='', n_jobs=-1, return_='community')

gamma= np.linspace(gmin, gmax, gamma_n)
#%%
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from itertools import combinations
from joblib import Parallel, delayed
from tqdm import tqdm

def compute_mean_nmi_parallel(communities_mat, n_jobs=-1):
    """
    communities_mat: shape (n_animals, n_gamma, n_runs, n_regions)
    n_jobs: int, number of CPU cores (default: all)
    Returns:
        mean_nmi: shape (n_gamma,) - average NMI across all animals and runs per gamma
    """
    n_animals, n_gamma, n_runs, n_regions = communities_mat.shape
    mean_nmi = np.zeros(n_gamma)
    for g in tqdm(range(n_gamma)):
        partitions = communities_mat[:, g, :, :].reshape(-1, n_regions)
        n_partitions = partitions.shape[0]
        pairs = list(combinations(range(n_partitions), 2))
        nmi_scores = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(normalized_mutual_info_score)(partitions[i], partitions[j]) 
            for i, j in pairs
        )
        mean_nmi[g] = np.mean(nmi_scores)
    return mean_nmi

# Usage example:
# mean_nmi = compute_mean_nmi_parallel(communities_mat, n_jobs=8)

mean_nmi = compute_mean_nmi_parallel(communities_mat)

#%%
def plot_stability(gammas, mean_nmi):
    plt.figure(figsize=(6,4))
    plt.plot(gammas, mean_nmi, marker='o')
    plt.xlabel('Gamma')
    plt.ylabel('Mean NMI (Stability)')
    plt.title('Partition Stability Across Gamma')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
gamma= np.linspace(gmin, gmax, 500)
plot_stability(gamma, mean_nmi)
# ====== Example Usage ======

# Suppose you have:
# communities_mat.shape = (n_subjects, n_runs, n_gamma, n_nodes)
gammas = np.linspace(0.5, 2.0, gamma_n)

best_gamma_idx, consensus_partition, allegiance = consensus_pipeline(communities_mat, gammas)

# consensus_partition is your robust, group-level community assignment
# allegiance is your co-assignment matrix for visualization

#%%


#Plot gamma_qmod_val
plt.figure(figsize=(10, 8))
plt.plot(gamma,np.mean(gamma_qmod_val, axis=1))#, label='gamma_qmod_val')
plt.fill_between(gamma,
                  np.mean(gamma_qmod_val, axis=1)-np.std(gamma_qmod_val, axis=1),
                  np.mean(gamma_qmod_val, axis=1)+np.std(gamma_qmod_val, axis=1), alpha=0.2)

plt.plot(gamma, np.mean(gamma_qmod_val2, axis=1), label='gamma_qmod_val2')
plt.fill_between(gamma,
                  np.mean(gamma_qmod_val2, axis=1)-np.std(gamma_qmod_val2, axis=1),
                  np.mean(gamma_qmod_val2, axis=1)+np.std(gamma_qmod_val2, axis=1), alpha=0.2)
plt.plot(gamma, np.mean(gamma_qmod_val3, axis=1), label='gamma_qmod_val3')
plt.fill_between(gamma,
                  np.mean(gamma_qmod_val3, axis=1)-np.std(gamma_qmod_val3, axis=1),
                  np.mean(gamma_qmod_val3, axis=1)+np.std(gamma_qmod_val3, axis=1), alpha=0.2)  
plt.xlabel('Gamma')
plt.ylabel('Modularity')
plt.title('Modularity vs Gamma')
plt.legend()











# %%
# Spearman Correlation between allegiance matrices
from scipy.stats import pearsonr, spearmanr
def compute_correlation(allegiance_matrices: np.ndarray) -> np.ndarray:
    n_matrices = allegiance_matrices.shape[0]
    correlation_matrix = np.zeros((n_matrices, n_matrices))
    
    for i in range(n_matrices):
        for j in range(i, n_matrices):
            corr, _ = spearmanr(allegiance_matrices[i].flatten(), allegiance_matrices[j].flatten())
            # corr, _ = pearsonr(allegiance_matrices[i].flatten(), allegiance_matrices[j].flatten())
            correlation_matrix[i, j] = corr
            correlation_matrix[j, i] = corr  # Symmetric matrix

    return correlation_matrix

# Compute the correlation matrix
correlation_matrix = compute_correlation(allegiance_mat)

#Plot the correlation matrix
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='viridis', vmin=0, vmax=1)
plt.colorbar()

agreement_matrices = np.sum(allegiance_mat, axis=0) / len(fc_wt_veh)

data.region_labels_preprocessed
#plot the agreement matrix
plt.figure(figsize=(10, 8))
plt.imshow(agreement_matrices, cmap='viridis', vmin=0, vmax=1)
plt.xticks(ticks=np.arange(len(data.region_labels_preprocessed)), labels=data.region_labels_preprocessed, rotation=90)
plt.yticks(ticks=np.arange(len(data.region_labels_preprocessed)), labels=data.region_labels_preprocessed)
plt.colorbar()


import brainconn as bct


communities, _ = bct.modularity.modularity_louvain_und_sign(agreement_matrices, gamma=1., seed=42)  

#save the communities
save_pickle(communities, Path(data.paths['allegiance']) / 'communities_wt_veh.pkl')

#LOAD communities
with open(Path(data.paths['allegiance']) / 'communities_wt_veh.pkl', 'rb') as f:
    communities = pickle.load(f)
# %%
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

# %%
#%%

fcd = np.array([dfc_stream2fcd(ts2dfc_stream(ts1, window_size=20, lag=1)) for ts1 in tqdm(data.ts)])
# %%
#plot fcd
plt.figure(figsize=(10, 8))
plt.imshow(fcd[33], cmap='viridis', vmin=0, vmax=1)
# plt.colorbar()
plt.title('FCD for first animal')
plt.ylabel(r'$FC(t_i)$', fontsize=14)
plt.xlabel(r'$FC(t_j)$', fontsize=14)
plt.xticks([])
plt.yticks([])    
plt.colorbar(label=r'CC($FC(t_i)$, $FC(t_j)$)')
data.cog_data_filtered



# %%
