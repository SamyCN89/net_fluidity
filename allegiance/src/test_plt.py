#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pickle

from shared_code.fun_paths import get_paths
from shared_code.fun_metaconnectivity import load_merged_allegiance, contingency_matrix_fun, 

# --- CONFIG ---
window_size = 9
lag = 1
timecourse_folder = 'Timecourses_updated_03052024'

# --- DATA TS LOAD ---
paths = get_paths(timecourse_folder=timecourse_folder)
data_ts = np.load(paths['sorted'] / 'ts_and_meta_2m4m.npz', allow_pickle=True)
ts = data_ts['ts']
n_animals = len(ts)
n_regions = ts[0].shape[1]
anat_labels = data_ts['anat_labels']

filename_dfc = f'window_size={window_size}_lag={lag}_animals={n_animals}_regions={n_regions}'
dfc_data = np.load(paths['dfc'] / f'dfc_{filename_dfc}.npz')
n_windows = np.transpose(dfc_data['dfc_stream'], (0, 3, 2, 1)).shape[-1]

# --- GROUPING DATA ---
with open(paths['sorted'] / "grouping_data_oip.pkl", "rb") as f:
    mask_groups, label_variables = pickle.load(f)
with open(paths['sorted'] / "grouping_data_per_sex(gen_phen).pkl", "rb") as f:
    mask_groups_per_sex, label_variables_per_sex = pickle.load(f)
#%%
#%%
#
# # ------Trimers -------
# mc_nplets_mask, mc_nplets_index = compute_mc_nplets_mask_and_index(
#     n_regions, allegiance_sort=sort_allegiances
# )

# # --- Modularituy and Allegiance Data ---
# save_filename = paths["mc_mod"] / 'mc_allegiance_ref(runs=wt2m_gammaval=1000)=9_lag=1_windowsize=100_animals=126_regions=41.npz.'
# data_analysis = np.load(save_filename, allow_pickle=True)
# sort_allegiance = data_analysis["sort_allegiance"]
# mc_modules_mask = data_analysis["mc_modules_mask"]
# mc_idx = data_analysis["mc_idx_tril"]
# fc_idx = data_analysis["fc_idx_tril"]
# mc_val = data_analysis["mc_val_tril"]
# mc_mod_idx = data_analysis["mc_mod_idx"]
# mc_reg_idx = data_analysis["mc_reg_idx"]
# fc_reg_idx = data_analysis["fc_reg_idx"]

#%%
# --- ALLEGIANCE DATA ---
dfc_communities, sort_allegiances, contingency_matrices = load_merged_allegiance(
    paths, window_size=window_size, lag=lag
)
dfc_communities_sorted = np.take_along_axis(dfc_communities, sort_allegiances.astype(int), axis=-1)
# dfc_communities_sorted = dfc_communities[sort_allegiances[:,:,0].astype(int), sort_allegiances[1].astype(int), sort_allegiances[2].astype(int)]
# dfc_communities_sorted = dfc_communities_sorted[:, :, sort_allegiances[0, 0].astype(int)]
#%%
plt.figure(figsize=(10, 8))
plt.title("Sorted DFC Communities")
# plt.imshow(dfc_communities_sorted[0].T, aspect='auto', interpolation='none', cmap='viridis')
plt.imshow(contingency_matrices[0, 0, sort_allegiances[0, 0].astype(int)][:, sort_allegiances[0, 0].astype(int)], aspect='auto', interpolation='none', cmap='viridis')  
plt.colorbar()
# print(np.unique(dfc_communities_sorted[0].T))
#%%
# --- UPPER TRIANGLE STACK ---
triu = np.triu_indices(n_regions, k=1)
cont_mat_n_pairs = contingency_matrices[:, :, triu[0], triu[1]]
cont_mat_n_pairs = np.concatenate(cont_mat_n_pairs, axis=0)
cont_mat_n_pairs_animal = np.repeat(np.arange(n_windows), n_animals)

#%%
#Aggregation matrix
cont_mat_n_pairs_agg = np.stack([(np.sum(contingency_matrices[ind], axis=0)/n_windows) for ind in range(n_animals)])

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

def matrix2vec(matrix3d):
    """
    Convert a 3D matrix into a 2D matrix by vectorizing each 2D matrix along the third dimension.
    
    Parameters:
    matrix3d (numpy.ndarray): 3D numpy array.
    
    Returns:
    numpy.ndarray: 2D numpy array where each column is the vectorized form of the 2D matrices from the 3D input.
    """
    #F: Frame, n: node
    F, n, _ = matrix3d.T.shape  # Assuming matrix3d shape is [F, n, n]
    return matrix3d.reshape((n*n,F))

# sort by mask_group[2] the cont_mat_n_pairs_agg_fcd
cont_mat_n_pairs_agg_fcd = dfc_stream2fcd(cont_mat_n_pairs_agg.T)

ind_grp_sort = np.concatenate([np.squeeze(np.argwhere(grp==True)) for grp in mask_groups[2]])
# sort the cont_mat_n_pairs_agg_fcd by ind_grp_sort
cont_mat_n_pairs_agg_fcd_sortgrp = cont_mat_n_pairs_agg_fcd[ind_grp_sort, :][:, ind_grp_sort]

#plot matrix cont_mat_n_pairs_agg_fcd_sortgrp, 
#put in the axes marks of the mask_groups    

plt.figure(figsize=(10, 8))
plt.title("Contingency Matrix Aggregated")
plt.imshow(cont_mat_n_pairs_agg_fcd_sortgrp, aspect='auto', interpolation='none', cmap='viridis')
plt.xlabel("Animal")
plt.ylabel("Animal")
plt.colorbar()

#%%
# --- t-SNE (dimensionality reduction) ---
scaler = StandardScaler()

dfc_communities_sorted_scaled = scaler.fit_transform(
    dfc_communities_sorted.reshape(-1, dfc_communities_sorted.shape[-1])
)
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_jobs=-1, verbose=1, metric="euclidean")
cont_mat_n_pairs_tsne = tsne.fit_transform(cont_mat_n_pairs)
cont_mat_n_pairs_tsne2 = tsne.fit_transform(cont_mat_n_pairs.T)
#%%
plt.figure(figsize=(10, 8))
plt.scatter(
    cont_mat_n_pairs_tsne[:, 0], cont_mat_n_pairs_tsne[:, 1],
    c=cont_mat_n_pairs_animal, s=15, marker='.', cmap='tab20', alpha=0.5
)
plt.title("t-SNE of Contingency Matrix Pairs")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()
#%%
plt.figure(figsize=(10, 8))
plt.scatter(
    cont_mat_n_pairs_tsne2[:, 0], cont_mat_n_pairs_tsne2[:, 1],
     s=50, marker='.', cmap='tab20', alpha=0.5
)
plt.title("t-SNE of Contingency Matrix Pairs (Transposed)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()
#%%
# --- COMMUNITY LABEL MEDIAN ---
# dfc_communities_sorted_median = np.median(dfc_communities_sorted.T, axis=2)
dfc_communities_sorted_median = dfc_communities_sorted[91].T
plt.figure(figsize=(10, 8))
plt.title("Community label - Median")
plt.imshow(dfc_communities_sorted_median, aspect='auto', interpolation='none', cmap='viridis')
plt.colorbar()
plt.yticks(np.arange(n_regions), labels=anat_labels[sort_allegiances[0, 0].astype(int)])
plt.ylabel("Regions")
plt.xlabel("Time Windows")
plt.show()
#%%
# --- AGREEMENT MATRIX (vectorized) ---
# def build_agreement_matrix_vectorized(communities):
#     n_runs, n_nodes = communities.shape
#     eq = (communities[:, :, None] == communities[:, None, :])
#     return np.sum(eq, axis=0).astype(np.float32)

def build_agreement_matrix_vectorized(communities):
    """
    Compute the agreement matrix for a 2D numpy array of community labels using vectorization.
    communities: array of shape (n_runs, n_nodes)
    Returns:
        agreement: 2D array (n_nodes, n_nodes)
    """
    # communities shape: (n_runs, n_nodes)
    # compare all node pairs for each run, shape becomes (n_runs, n_nodes, n_nodes)
    equal_matrix = (communities[:, :, None] == communities[:, None, :])
    # Sum over runs
    agreement = np.sum(equal_matrix, axis=0)
    return agreement.astype(np.float64)
#%%
agreement = np.stack([build_agreement_matrix_vectorized(dfc_communities_sorted[indv]) for indv in tqdm(range(n_animals))])
#%%
plt.figure(figsize=(10, 8))
plt.title("Agreement Matrix")
plt.clf()
# for ii, aux_group in enumerate(mask_groups[2]):
    # plt.subplot(2, 2, 1+ii)
np.shape(agreement[aux_group])
aux_agreement = agreement[91]
plt.imshow(aux_agreement, aspect='auto', interpolation='none', cmap='viridis')
plt.colorbar()
    # plt.show()
# aux_group = mask_groups[0][0]
# aux_agreement = np.median(agreement[aux_group], axis=0)
# plt.imshow(aux_agreement, aspect='auto', interpolation='none', cmap='viridis')
# plt.colorbar()
# plt.show()
#%%
# --- CONSENSUS COMMUNITY STRUCTURE ---
contingency_matrix, gamma_qmod_val, gamma_agreement_mat = contingency_matrix_fun(
    1000, mc_data=dfc_communities_sorted.T, gamma_range=10, gmin=0.5, gmax=1, cache_path=None, ref_name='', n_jobs=-1
)
plt.figure(figsize=(10, 8))
plt.title("Consensus Community Structure")
plt.imshow(contingency_matrix, aspect='auto', interpolation='none', cmap='Greys')
plt.colorbar()
plt.show()

# --- CRITICAL ERROR CHECKS ---
assert ts.shape[0] == n_animals, "n_animals mismatch"
assert dfc_communities_sorted.shape[1] == n_windows, "Window mismatch"
assert contingency_matrices.shape[0] == n_animals, "Contingency: n_animals mismatch"

# %%
