#%%
from math import e
from re import M
from matplotlib import cm, markers
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from shared_code.fun_paths import get_paths
from shared_code.fun_metaconnectivity import load_merged_allegiance#%%
from math import e
from re import M
from matplotlib import cm, markers
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from shared_code.fun_paths import get_paths
from shared_code.fun_metaconnectivity import load_merged_allegiance

# Set consistent config to match previous run
window_size = 9
lag = 1
timecourse_folder = 'Timecourses_updated_03052024'

# Load meta info to determine shape
paths = get_paths(timecourse_folder=timecourse_folder)
#%%
data_ts = np.load(paths['sorted'] / 'ts_and_meta_2m4m.npz', allow_pickle=True)
ts = data_ts['ts']
n_animals = len(ts)
n_regions = ts[0].shape[1]
anat_labels = data_ts['anat_labels']

filename_dfc = f'window_size={window_size}_lag={lag}_animals={n_animals}_regions={n_regions}'
dfc_data = np.load(paths['dfc'] / f'dfc_{filename_dfc}.npz')
n_windows = np.transpose(dfc_data['dfc_stream'], (0, 3, 2, 1)).shape[-1]

#%%
import pickle
with open(paths['sorted'] / "grouping_data_oip.pkl", "rb") as f:
    mask_groups, label_variables = pickle.load(f)
with open(paths['sorted'] / "grouping_data_per_sex(gen_phen).pkl", "rb") as f:
    mask_groups_per_sex, label_variables_per_sex = pickle.load(f)#%%

# Load the merged allegiance data of all animals
dfc_communities, sort_allegiances, contingency_matrices = load_merged_allegiance(paths, window_size=9, lag=1)
dfc_communities_sorted = dfc_communities[:, :, sort_allegiances[0, 0].astype(int)] # REaorder the labelling of the communities (deprecated soon)

#%%
triu = np.triu_indices(n_regions, k=1)  # Get upper triangle indices for n_regions x n_regions matrix


mask_groups_2 = mask_groups[2]  # Use the first group for demonstration
label_variables_2 = label_variables[2]  # Use the first label set for demonstration

#Concatenate or stack cont_mat_n_pairs in the first two dimensions
# cont_mat_n_pairs = contingency_matrices[mask_groups_2[0]][:,:, triu[0], triu[1]]  # Extract the upper triangle of the first contingency matrix
# cont_mat_n_pairs = np.concatenate(cont_mat_n_pairs, axis=0)  # Shape: (n_animals * n_windows, n_pairs)
# cont_mat_n_pairs_animal = np.repeat(np.arange(n_windows), np.sum(mask_groups_2[0]))  # Shape: (n_animals * n_windows,)

cont_mat_n_pairs = contingency_matrices[:,:, triu[0], triu[1]]  # Extract the upper triangle of the first contingency matrix
cont_mat_n_pairs = np.concatenate(cont_mat_n_pairs, axis=0)  # Shape: (n_animals * n_windows, n_pairs)
cont_mat_n_pairs_animal = np.repeat(np.arange(n_windows), n_animals)  # Shape: (n_animals * n_windows,)

# List of contingency matrices for each animal in the contatenated array to color the t-SNE plot
# cont_mat_n_pairs_animal = np.repeat(np.arange(np.sum(mask_groups_2[0])), np.sum(mask_groups_2[0]))  # Shape: (n_animals * n_windows,)
#%%
#TSNE on one animal of contingency matrix (contingency_matrices[0])
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
# Standardize the data
scaler = StandardScaler()
dfc_communities_sorted_scaled = scaler.fit_transform(dfc_communities_sorted.reshape(-1, dfc_communities_sorted.shape[-1]))

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
cont_mat_n_pairs_tsne = tsne.fit_transform(cont_mat_n_pairs)
#%%
# Plot the t-SNE results
plt.figure(figsize=(10, 8))
plt.scatter(cont_mat_n_pairs_tsne[:, 0], cont_mat_n_pairs_tsne[:, 1], 
            c=cont_mat_n_pairs_animal,
            s=15,
            marker='.',
            cmap='tab20',  # Use a colormap to differentiate animals
            alpha=0.5)#, cmap='viridis', alpha=0.5)
# plt.plot(cont_mat_n_pairs_tsne[:, 0], cont_mat_n_pairs_tsne[:, 1], '.', alpha=0.5)#, cmap='viridis', alpha=0.5)
plt.title("t-SNE of Contingency Matrix Pairs")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
# plt.grid(True)
plt.show()
#%%
# Plot the mean matrix
dfc_communities_sorted_median = np.median(dfc_communities_sorted.T, axis=2) # Take the median across the time windows

#%%
plt.figure(figsize=(10, 8))
plt.subplot(1, 1, 1)
plt.clf()
plt.title("Community label - Animal 0, Window 0")
# plt.imshow(cm_0_mean.T , aspect='auto', interpolation='none', cmap='Greys')
# aux_argsort = np.argsort(dfc_communities_sorted)

plt.imshow(dfc_communities_sorted_median, aspect='auto', interpolation='none', cmap='viridis')
# plt.clim(0, 1)
plt.colorbar()
plt.yticks(np.arange(n_regions), labels=anat_labels[sort_allegiances[0, 0].astype(int)])
plt.ylabel("Regions")
plt.xlabel("Time Windows")
plt.show()

#%%
#Plot median community labels for mask_groups
plt.figure(figsize=(10, 8))
plt.clf()
# plt.imshow(cm_0_mean.T , aspect='auto', interpolation='none', cmap='Greys')
aux_argsort = np.argsort(dfc_communities_sorted)    
dfc_groups = np.array([np.median(dfc_communities_sorted[mask_groups[2][xx]], axis=0) for xx in range(len(mask_groups[2]))])

# plt.subplots(2, 2, ii + 1)
for ii, mat in enumerate(dfc_groups):
    aux_labels = label_variables[2]
    plt.subplot(2, 2, ii + 1)
    plt.title(f"Community label - Group {aux_labels[ii]}, Window {ii+1}")
    plt.imshow(mat.T, aspect='auto', interpolation='none', cmap='viridis')
    # plt.clim(0, 1)
    # plt.colorbar()
    # plt.yticks(np.arange(len(mask_groups[0])), labels=label_variables[0][mask_groups[0]])
    # plt.ylabel("Regions")

#%%%# Check the shape of the loaded data

cm_0 = np.array(contingency_matrices[0])

triu_indices = np.array(np.triu_indices(n_regions, k=1))


cm_0_triu = cm_0[:, triu_indices[0], triu_indices[1]]


cm_0_mean_triu = np.mean(cm_0_triu, axis=0) 

#Reshape the mean matrix to the original shape
cm_0_mean = np.zeros((n_regions, n_regions))
cm_0_mean[triu_indices[0], triu_indices[1]] = cm_0_mean_triu
cm_0_mean = cm_0_mean + cm_0_mean.T
cm_0_mean[np.diag_indices_from(cm_0_mean)] = 1

# Plot the mean matrix
plt.figure(figsize=(10, 8))
plt.subplot(1, 1, 1)
plt.clf()
plt.title("Allegiance Matrix - Animal 0, Window 0")
plt.imshow(cm_0_mean.T , aspect='auto', interpolation='none', cmap='Greys')
# plt.imshow(dfc_communities_sorted , aspect='auto', interpolation='none', cmap='Greys')
# plt.clim(0, 1)
plt.colorbar()
plt.xlabel("DFT Frequency")
plt.ylabel("DFT Frequency")
plt.show()


#%%
# Plot matrices 9 matrices in a grid
plt.figure(figsize=(10, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.clf()
    plt.title(f"Contingency Matrix - Animal 0, Window {i}")
    plt.imshow(cm_0[i].T , aspect='auto', interpolation='none', cmap='Greys')
    plt.clim(0, 1)
    plt.colorbar()
    plt.xlabel("DFT Frequency")
    plt.ylabel("DFT Frequency")
#%%
# plot imshow cm_0_triu
plt.figure(figsize=(10, 8))
plt.subplot(1, 1, 1)
plt.clf()
plt.title("Contingency Matrix - Animal 0, Window 0")
plt.imshow(cm_0_triu.T , aspect='auto', interpolation='none', cmap='Greys')
plt.clim(0, 1)
plt.colorbar()
plt.xlabel("DFT Frequency")
plt.ylabel("DFT Frequency")
plt.show()

#%%
# Plot the cumsum of cm_0_triu contingency matrix for all the windows
plt.figure(figsize=(10, 8))
plt.subplot(1, 1, 1)
plt.clf()
plt.title("Contingency Matrix - Animal 0, Window 0")
plt.plot(np.sort(cm_0_triu.ravel()))# aspect='auto', interpolation='none', cmap='Greys')
#%%
# Plot the histogram of the contingency matrix for all windows
plt.figure(figsize=(12, 12))
plt.title("Contingency Matrix - Animal 0, Window 0")

# One histogram per row (i.e., each region pair)
plt.hist(cm_0_triu[cm_0_triu > 0.1], bins=100, density=True, histtype='step')
plt.xlabel("Contingency Matrix Value")
plt.ylabel("Frequency") 
plt.ylim(0, 2)
plt.tight_layout()
plt.show()
# %%
# Plot imshow of the contingency matrix for  9 windows in one animal
plt.figure(figsize=(12, 12))
plt.clf()
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.title(f"Contingency Matrix - Animal 0, Window {i}")
    plt.imshow(cm_0[i].T , aspect='auto', interpolation='none', cmap='viridis')
    plt.clim(0, 1)
    plt.colorbar()
    plt.xlabel("DFT Frequency")
    plt.ylabel("DFT Frequency")
# %%
def _build_agreement_matrix(communities):
    """
    Compute the agreement matrix for a list of community labels.
    Parameters
    ----------
    communities : list of 1D arrays
        List of community labels for each run. Each array should have the same length.
    Returns
    -------
    agreement : 2D array
        The agreement matrix, where entry (i, j) represents the number of communities
        that nodes i and j belong to.
    """
    n_runs, n_nodes = communities.shape
    agreement = np.zeros((n_nodes, n_nodes), dtype=np.uint16)

    for Ci in communities:
        # agreement += (Ci[:, None] == Ci[None, :])
        agreement += (Ci[:, None] == Ci)

    return agreement.astype(np.float32)
#%%
# Global Allegiance Matrix

# Average modular structure over all windows and animals
allegiance_matrices = cm_0
allegiance_avg = allegiance_matrices.mean(axis=0)

# "consensus" community structure over the whole period with Louvain method
from shared_code.fun_metaconnectivity import contingency_matrix_fun
# contingency_matrix, gamma_qmod_val, gamma_agreement_mat =contingency_matrix_fun(1000, mc_data=allegiance_avg, gamma_range=10, gmin=0.1, gmax=1, cache_path=None, ref_name='', n_jobs=-1)
contingency_matrix, gamma_qmod_val, gamma_agreement_mat =contingency_matrix_fun(1000, mc_data=dfc_communities_sorted.T, gamma_range=10, gmin=0.5, gmax=1, cache_path=None, ref_name='', n_jobs=-1)
#%%
consensus_community =contingency_matrix

#Plot consensus community
plt.figure(figsize=(10, 8))
plt.subplot(1, 1, 1)
plt.clf()
plt.imshow(consensus_community , aspect='auto', interpolation='none', cmap='Greys')
plt.colorbar()
plt.xlabel("DFT Frequency")
plt.ylabel("DFT Frequency")
plt.show()
# Plot the mean matrix
# %%

agreement = _build_agreement_matrix(dfc_communities_sorted.T)

plt.figure(figsize=(10, 8))
plt.subplot(1, 1, 1)
plt.clf()
plt.imshow(agreement , aspect='auto', interpolation='none', cmap='viridis')
#%%

























# %%


# Set consistent config to match previous run
window_size = 9
lag = 1
timecourse_folder = 'Timecourses_updated_03052024'

# Load meta info to determine shape
paths = get_paths(timecourse_folder=timecourse_folder)
#%%
data_ts = np.load(paths['sorted'] / 'ts_and_meta_2m4m.npz', allow_pickle=True)
ts = data_ts['ts']
n_animals = len(ts)
n_regions = ts[0].shape[1]
anat_labels = data_ts['anat_labels']

filename_dfc = f'window_size={window_size}_lag={lag}_animals={n_animals}_regions={n_regions}'
dfc_data = np.load(paths['dfc'] / f'dfc_{filename_dfc}.npz')
n_windows = np.transpose(dfc_data['dfc_stream'], (0, 3, 2, 1)).shape[-1]

#%%
import pickle
with open(paths['sorted'] / "grouping_data_oip.pkl", "rb") as f:
    mask_groups, label_variables = pickle.load(f)
with open(paths['sorted'] / "grouping_data_per_sex(gen_phen).pkl", "rb") as f:
    mask_groups_per_sex, label_variables_per_sex = pickle.load(f)#%%

# Load the merged allegiance data of all animals
dfc_communities, sort_allegiances, contingency_matrices = load_merged_allegiance(paths, window_size=9, lag=1)
dfc_communities_sorted = dfc_communities[:, :, sort_allegiances[0, 0].astype(int)] # REaorder the labelling of the communities (deprecated soon)

#%%
triu = np.triu_indices(n_regions, k=1)  # Get upper triangle indices for n_regions x n_regions matrix


mask_groups_2 = mask_groups[2]  # Use the first group for demonstration
label_variables_2 = label_variables[2]  # Use the first label set for demonstration

#Concatenate or stack cont_mat_n_pairs in the first two dimensions
# cont_mat_n_pairs = contingency_matrices[mask_groups_2[0]][:,:, triu[0], triu[1]]  # Extract the upper triangle of the first contingency matrix
# cont_mat_n_pairs = np.concatenate(cont_mat_n_pairs, axis=0)  # Shape: (n_animals * n_windows, n_pairs)
# cont_mat_n_pairs_animal = np.repeat(np.arange(n_windows), np.sum(mask_groups_2[0]))  # Shape: (n_animals * n_windows,)

cont_mat_n_pairs = contingency_matrices[:,:, triu[0], triu[1]]  # Extract the upper triangle of the first contingency matrix
cont_mat_n_pairs = np.concatenate(cont_mat_n_pairs, axis=0)  # Shape: (n_animals * n_windows, n_pairs)
cont_mat_n_pairs_animal = np.repeat(np.arange(n_windows), n_animals)  # Shape: (n_animals * n_windows,)

# List of contingency matrices for each animal in the contatenated array to color the t-SNE plot
# cont_mat_n_pairs_animal = np.repeat(np.arange(np.sum(mask_groups_2[0])), np.sum(mask_groups_2[0]))  # Shape: (n_animals * n_windows,)
#%%
#TSNE on one animal of contingency matrix (contingency_matrices[0])
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
# Standardize the data
scaler = StandardScaler()
dfc_communities_sorted_scaled = scaler.fit_transform(dfc_communities_sorted.reshape(-1, dfc_communities_sorted.shape[-1]))

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
cont_mat_n_pairs_tsne = tsne.fit_transform(cont_mat_n_pairs)
#%%
# Plot the t-SNE results
plt.figure(figsize=(10, 8))
plt.scatter(cont_mat_n_pairs_tsne[:, 0], cont_mat_n_pairs_tsne[:, 1], 
            c=cont_mat_n_pairs_animal,
            s=15,
            marker='.',
            cmap='tab20',  # Use a colormap to differentiate animals
            alpha=0.5)#, cmap='viridis', alpha=0.5)
# plt.plot(cont_mat_n_pairs_tsne[:, 0], cont_mat_n_pairs_tsne[:, 1], '.', alpha=0.5)#, cmap='viridis', alpha=0.5)
plt.title("t-SNE of Contingency Matrix Pairs")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
# plt.grid(True)
plt.show()
#%%
# Plot the mean matrix
dfc_communities_sorted_median = np.median(dfc_communities_sorted.T, axis=2) # Take the median across the time windows

#%%
plt.figure(figsize=(10, 8))
plt.subplot(1, 1, 1)
plt.clf()
plt.title("Community label - Animal 0, Window 0")
# plt.imshow(cm_0_mean.T , aspect='auto', interpolation='none', cmap='Greys')
# aux_argsort = np.argsort(dfc_communities_sorted)

plt.imshow(dfc_communities_sorted_median, aspect='auto', interpolation='none', cmap='viridis')
# plt.clim(0, 1)
plt.colorbar()
plt.yticks(np.arange(n_regions), labels=anat_labels[sort_allegiances[0, 0].astype(int)])
plt.ylabel("Regions")
plt.xlabel("Time Windows")
plt.show()

#%%
#Plot median community labels for mask_groups
plt.figure(figsize=(10, 8))
plt.clf()
# plt.imshow(cm_0_mean.T , aspect='auto', interpolation='none', cmap='Greys')
aux_argsort = np.argsort(dfc_communities_sorted)    
dfc_groups = np.array([np.median(dfc_communities_sorted[mask_groups[2][xx]], axis=0) for xx in range(len(mask_groups[2]))])

# plt.subplots(2, 2, ii + 1)
for ii, mat in enumerate(dfc_groups):
    aux_labels = label_variables[2]
    plt.subplot(2, 2, ii + 1)
    plt.title(f"Community label - Group {aux_labels[ii]}, Window {ii+1}")
    plt.imshow(mat.T, aspect='auto', interpolation='none', cmap='viridis')
    # plt.clim(0, 1)
    # plt.colorbar()
    # plt.yticks(np.arange(len(mask_groups[0])), labels=label_variables[0][mask_groups[0]])
    # plt.ylabel("Regions")

#%%%# Check the shape of the loaded data

cm_0 = np.array(contingency_matrices[0])

triu_indices = np.array(np.triu_indices(n_regions, k=1))


cm_0_triu = cm_0[:, triu_indices[0], triu_indices[1]]


cm_0_mean_triu = np.mean(cm_0_triu, axis=0) 

#Reshape the mean matrix to the original shape
cm_0_mean = np.zeros((n_regions, n_regions))
cm_0_mean[triu_indices[0], triu_indices[1]] = cm_0_mean_triu
cm_0_mean = cm_0_mean + cm_0_mean.T
cm_0_mean[np.diag_indices_from(cm_0_mean)] = 1

# Plot the mean matrix
plt.figure(figsize=(10, 8))
plt.subplot(1, 1, 1)
plt.clf()
plt.title("Allegiance Matrix - Animal 0, Window 0")
plt.imshow(cm_0_mean.T , aspect='auto', interpolation='none', cmap='Greys')
# plt.imshow(dfc_communities_sorted , aspect='auto', interpolation='none', cmap='Greys')
# plt.clim(0, 1)
plt.colorbar()
plt.xlabel("DFT Frequency")
plt.ylabel("DFT Frequency")
plt.show()


#%%
# Plot matrices 9 matrices in a grid
plt.figure(figsize=(10, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.clf()
    plt.title(f"Contingency Matrix - Animal 0, Window {i}")
    plt.imshow(cm_0[i].T , aspect='auto', interpolation='none', cmap='Greys')
    plt.clim(0, 1)
    plt.colorbar()
    plt.xlabel("DFT Frequency")
    plt.ylabel("DFT Frequency")
#%%
# plot imshow cm_0_triu
plt.figure(figsize=(10, 8))
plt.subplot(1, 1, 1)
plt.clf()
plt.title("Contingency Matrix - Animal 0, Window 0")
plt.imshow(cm_0_triu.T , aspect='auto', interpolation='none', cmap='Greys')
plt.clim(0, 1)
plt.colorbar()
plt.xlabel("DFT Frequency")
plt.ylabel("DFT Frequency")
plt.show()

#%%
# Plot the cumsum of cm_0_triu contingency matrix for all the windows
plt.figure(figsize=(10, 8))
plt.subplot(1, 1, 1)
plt.clf()
plt.title("Contingency Matrix - Animal 0, Window 0")
plt.plot(np.sort(cm_0_triu.ravel()))# aspect='auto', interpolation='none', cmap='Greys')
#%%
# Plot the histogram of the contingency matrix for all windows
plt.figure(figsize=(12, 12))
plt.title("Contingency Matrix - Animal 0, Window 0")

# One histogram per row (i.e., each region pair)
plt.hist(cm_0_triu[cm_0_triu > 0.1], bins=100, density=True, histtype='step')
plt.xlabel("Contingency Matrix Value")
plt.ylabel("Frequency") 
plt.ylim(0, 2)
plt.tight_layout()
plt.show()
# %%
# Plot imshow of the contingency matrix for  9 windows in one animal
plt.figure(figsize=(12, 12))
plt.clf()
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.title(f"Contingency Matrix - Animal 0, Window {i}")
    plt.imshow(cm_0[i].T , aspect='auto', interpolation='none', cmap='viridis')
    plt.clim(0, 1)
    plt.colorbar()
    plt.xlabel("DFT Frequency")
    plt.ylabel("DFT Frequency")
# %%
def _build_agreement_matrix(communities):
    """
    Compute the agreement matrix for a list of community labels.
    Parameters
    ----------
    communities : list of 1D arrays
        List of community labels for each run. Each array should have the same length.
    Returns
    -------
    agreement : 2D array
        The agreement matrix, where entry (i, j) represents the number of communities
        that nodes i and j belong to.
    """
    n_runs, n_nodes = communities.shape
    agreement = np.zeros((n_nodes, n_nodes), dtype=np.uint16)

    for Ci in communities:
        # agreement += (Ci[:, None] == Ci[None, :])
        agreement += (Ci[:, None] == Ci)

    return agreement.astype(np.float32)
#%%
# Global Allegiance Matrix

# Average modular structure over all windows and animals
allegiance_matrices = cm_0
allegiance_avg = allegiance_matrices.mean(axis=0)

# "consensus" community structure over the whole period with Louvain method
from shared_code.fun_metaconnectivity import contingency_matrix_fun
# contingency_matrix, gamma_qmod_val, gamma_agreement_mat =contingency_matrix_fun(1000, mc_data=allegiance_avg, gamma_range=10, gmin=0.1, gmax=1, cache_path=None, ref_name='', n_jobs=-1)
contingency_matrix, gamma_qmod_val, gamma_agreement_mat =contingency_matrix_fun(1000, mc_data=dfc_communities_sorted.T, gamma_range=10, gmin=0.5, gmax=1, cache_path=None, ref_name='', n_jobs=-1)
#%%
consensus_community =contingency_matrix

#Plot consensus community
plt.figure(figsize=(10, 8))
plt.subplot(1, 1, 1)
plt.clf()
plt.imshow(consensus_community , aspect='auto', interpolation='none', cmap='Greys')
plt.colorbar()
plt.xlabel("DFT Frequency")
plt.ylabel("DFT Frequency")
plt.show()
# Plot the mean matrix
# %%

agreement = _build_agreement_matrix(dfc_communities_sorted.T)

plt.figure(figsize=(10, 8))
plt.subplot(1, 1, 1)
plt.clf()
plt.imshow(agreement , aspect='auto', interpolation='none', cmap='viridis')
#%%

























# %%
