#%%
from calendar import c
from math import e
from re import M
from turtle import st
from unittest import result
from joblib import Parallel
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
paths = get_paths(timecourse_folder=timecourse_folder)

#%%
# Load meta info to determine shape
data_ts = np.load(paths['preprocessed'] / 'ts_and_meta_2m4m.npz', allow_pickle=True)
ts = data_ts['ts']
n_animals = len(ts)
n_regions = ts[0].shape[1]
anat_labels = data_ts['anat_labels']

filename_dfc = f'window_size={window_size}_lag={lag}_animals={n_animals}_regions={n_regions}'
dfc_data = np.load(paths['dfc'] / f'dfc_{filename_dfc}.npz')
n_windows = np.transpose(dfc_data['dfc_stream'], (0, 3, 2, 1)).shape[-1]


#%%


#%%
import pickle
with open(paths['preprocessed'] / "grouping_data_oip.pkl", "rb") as f:
    mask_groups, label_variables = pickle.load(f)
with open(paths['preprocessed'] / "grouping_data_per_sex(gen_phen).pkl", "rb") as f:
    mask_groups_per_sex, label_variables_per_sex = pickle.load(f)
with open(paths['preprocessed'] / "grouping_data_new.pkl", "rb") as f:
    groups_sex_geno, groups_sex_pheno_oip, groups_sex_pheno_nor = pickle.load(f)

# ----- Load the mc data -----
from shared_code.fun_metaconnectivity import compute_mc_nplets_mask_and_index
# Load the regions and allegiance data
# Check if the regions and allegiance data are loaded correctly

label_ref = label_variables[2][0]  # Use the first label set for demonstration
n_runs_allegiance = 1000
gamma_pt_allegiance = 100
mc_mod_dataset = paths['mc_mod'] / f"mc_allegiance_ref(runs={label_ref}_gammaval={n_runs_allegiance})={gamma_pt_allegiance}_lag={lag}_windowsize={window_size}_animals={n_animals}_regions={n_regions}.npz".replace(' ','')
#%%
# Load the merged allegiance data of all animals
dfc_communities, sort_allegiances, contingency_matrices = load_merged_allegiance(paths, window_size=9, lag=1)

#%%

#%%

dfc_communities_sorted = np.zeros_like(dfc_communities)

for ani in tqdm(range(n_animals), desc="Animals"):
    for ws in range(n_windows):
        dfc_communities_sorted[ani, ws] = dfc_communities[ani, ws, sort_allegiances[ani, ws].astype(int)]
#%%
#plot the dfc_communities_sorted matrix of 1st animal
plt.figure(figsize=(10, 8))
plt.imshow(dfc_communities_sorted[0].T, aspect='auto', interpolation='none', cmap='viridis')
plt.colorbar()
plt.title("DFC Communities - Animal 0")
plt.ylabel("Regions")
plt.xlabel("Time Windows")
plt.show()

module_num = np.zeros(n_windows)
for i in range(n_windows):
    module_num[i] = len(np.unique(dfc_communities_sorted[1,i]))  # Check the unique values in the sorted communities for the first animal

#plot the number of modules per time window
plt.figure(figsize=(20, 6))
plt.plot(module_num, marker='o')
plt.title("Number of Modules per Time Window - Animal 0")
plt.xlabel("Time Windows")
plt.ylabel("Number of Modules")
plt.grid()
plt.show()

#%%
# Plot the triu contingency matrices for the first animal
n_regions = dfc_communities_sorted.shape[2]
# Extract the upper triangle indices for n_regions x n_regions matrix
triu_indices = np.triu_indices(n_regions, k=1)  # Get upper triangle indices for n_regions x n_regions matrix
# Extract the upper triangle of the contingency matrices for the first animal
contingency_matrices_0 = contingency_matrices[0]  # Get the contingency matrices for the first animal
contingency_matrices_0_triu = contingency_matrices_0[:, triu_indices[0], triu_indices[1]]

#plot the hist of contingency_matrices_0_triu 
plt.figure(figsize=(12, 12))
plt.clf()
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.title(f"Contingency Matrix - Animal 0, Window {i}")
    # Plot the histogram of the upper triangle of the contingency matrix for the first animal
    plt.hist(contingency_matrices_0_triu[i], bins=50, color='blue', alpha=0.7)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid()
# plt.hist(contingency_matrices_0_triu[0], bins=50, color='blue', alpha=0.7)
# # plt.hist2d(contingency_matrices_0_triu, bins=50, color='blue', alpha=0.7)
# plt.title("Histogram of Contingency Matrices - Animal 0")
# plt.xlabel("Value")
# plt.ylabel("Frequency")
# plt.grid()
# plt.show()
#%%
#plot sorted_cmat 1
plt.figure(figsize=(10, 8))
plt.imshow(contingency_matrices_0_triu>0, aspect='auto', interpolation='none', cmap='viridis')
plt.colorbar()
plt.title("Sorted DFC Communities - Animal 0")
plt.ylabel("Regions")
plt.xlabel("Time Windows")
plt.show()



#%%
cmat = contingency_matrices_0
sort_idx = sort_allegiances[0].astype(int)  # Get the sorting indices for the first animal

# Create an empty array for the sorted matrices
sorted_cmat = np.empty_like(cmat)

for i in range(cmat.shape[0]):
    idx = sort_idx[0]
    # Reorder both rows and columns using idx
    sorted_cmat[i] = cmat[i][idx, :][:, idx]
#%%
#plot sorted_cmat 1
plt.figure(figsize=(10, 8))
plt.imshow(sorted_cmat[0], aspect='auto', interpolation='none', cmap='viridis')
plt.colorbar()
plt.title("Sorted DFC Communities - Animal 0")
plt.ylabel("Regions")
plt.xlabel("Time Windows")
plt.show()
#plot sorted_cmat 1
plt.figure(figsize=(10, 8))
plt.imshow(contingency_matrices_0[0], aspect='auto', interpolation='none', cmap='viridis')
plt.colorbar()
plt.title("Sorted DFC Communities - Animal 0")
plt.ylabel("Regions")
plt.xlabel("Time Windows")
plt.show()


#%%
# contingency_matrices_0_triu = contingency_matrices_0[:, triu_indices[0], triu_indices[1]]
contingency_matrices_0_triu = sorted_cmat[:, triu_indices[0], triu_indices[1]]
# Compute the mean of the upper triangle across all windows for the first animal
# Plot the mean contingency matrix for the first animal
plt.figure(figsize=(10, 8))
plt.imshow(contingency_matrices_0_triu.T, aspect='auto', interpolation='none', cmap='viridis')
plt.colorbar()
plt.title("Mean Contingency Matrix - Animal 0")
plt.ylabel("Regions")
plt.xlabel("Regions")
plt.show()
#%%
# the spearman correlation between time points in contingency_matrices_0_triu (time_points x n_pairs)
from scipy.stats import spearmanr
time_corr_agreement = np.zeros(n_windows-1)  # Initialize the correlation matrix
for i in range(n_windows-1):
    time_corr_agreement[i] = spearmanr(contingency_matrices_0_triu[i], contingency_matrices_0_triu[i+1])[0]  # Compute the correlation matrix of the upper triangle of the contingency matrices for the first animal






















































#%%
# plot the time correlation of the agreement matrices
plt.figure(figsize=(25, 7))
plt.plot(time_corr_agreement, 'o-', markersize=5, alpha=0.7)
#%%
from scipy.stats import spearmanr, pearsonr
#adjusted rand index
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
# Suppose agreement_matrices: shape (n_windows, n_regions, n_regions)
similarities = np.zeros((n_windows, n_windows))
# similarities = np.zeros((n_regions, n_regions))

contingency_matrices_0 = contingency_matrices[0]  # Get the contingency matrices for the first animal

for t1 in tqdm(range(n_windows), desc="Windows"):
    for t2 in range(n_windows):
        
        # Example: Pearson correlation of upper triangles
        # Extract the contingency matrices for the two time windows
        #upper triangle indices
        triu_indices = np.triu_indices(n_regions, k=1)  # Get upper triangle indices for n_regions x n_regions matrix
        mat1 = contingency_matrices_0[t1][triu_indices[0], triu_indices[1]]
        mat2 = contingency_matrices_0[t2][triu_indices[0], triu_indices[1]]

        # mat1 = contingency_matrices_0[t1]
        # mat2 = contingency_matrices_0[t2]
        #Pearson correlation
        # similarities[t1, t2] = np.corrcoef(mat1, mat2)[0, 1]
        # spearman correlation
        # similarities[t1, t2] = spearmanr(mat1, mat2)[0]
        # # Frobenius norm
        similarities[t1, t2] = spearmanr(mat1, mat2)[0]
        # # Normalized Mutual Information
        # similarities[t1, t2] = normalized_mutual_info_score(mat1, mat2)
        # similarities[t1, t2] = np.linalg.norm(mat1 - mat2)
        
# similarities[t1, t2]: similarity between agreement matrices at window t1 and t2
#%%
# plot the similarities matrix
plt.figure(figsize=(10, 8))
plt.imshow(similarities, aspect='auto', interpolation='none', cmap='viridis')
plt.colorbar()
plt.title("Similarities Matrix - Animal 0")
plt.ylabel("Regions")
plt.xlabel("Regions")
plt.clim(0, .1)
plt.show()
#%%























































































#%%
# ------------------ communities allegiances through time windows ------------------


contingency_matrices_0 = contingency_matrices[0]  # Get the contingency matrices for the first animal

upper_tri_indices = np.triu_indices(n_regions, k=1)  # Get upper triangle indices for n_regions x n_regions matrix
# Extract the upper triangle of the contingency matrix for the first animal
contingency_matrices_0_triu = contingency_matrices_0[:, upper_tri_indices[0], upper_tri_indices[1]]
# Compute the mean of the upper triangle across all windows for the first animal
#plot the contingency matrix for the first animal
plt.figure(figsize=(10, 8))

plt.imshow(contingency_matrices_0_triu.T, aspect='auto', interpolation='none', cmap='viridis')
plt.colorbar()
plt.title("Contingency Matrix - Animal 0")
plt.ylabel("n pairs")
plt.xlabel("TWs")
plt.show()

#%%

dfc_communities_sorted_0 = dfc_communities_sorted[0]  # Get the sorted communities for the first animal

# Compute the mean of the sorted communities across all windows for the first animal
corr_mat = np.zeros((n_windows, n_windows))
for ii in tqdm(range(n_windows), desc="Computing correlations"):
    for jj in range(n_windows):
       corr, _ = pearsonr(dfc_communities_sorted_0[ii], dfc_communities_sorted_0[jj])
       corr_mat[ii, jj] = corr

# plot the correlation matrix
plt.figure(figsize=(10, 8))
plt.imshow(corr_mat, aspect='auto', interpolation='none', cmap='viridis')
plt.colorbar()
plt.title("Correlation Matrix - Animal 0")
plt.ylabel("Windows")
plt.xlabel("Windows")
plt.show()

#%%
# Mutual Information between columns of the dfc_communities_sorted matrix
from sklearn.metrics import mutual_info_score
mi_mat = np.zeros((n_windows, n_windows))
for ii in tqdm(range(n_windows), desc="Computing mutual information"):
    for jj in range(n_windows):
        mi = mutual_info_score(dfc_communities_sorted_0[ii], dfc_communities_sorted_0[jj])
        mi_mat[ii, jj] = mi

# plot the mutual information matrix
plt.figure(figsize=(10, 8))
plt.imshow(mi_mat, aspect='auto', interpolation='none', cmap='viridis')
plt.colorbar()
plt.title("Mutual Information Matrix - Animal 0")
plt.ylabel("Windows")
plt.xlabel("Windows")
plt.clim(0, 0.3)
plt.show()
#%%

#%%

from matplotlib.colors import ListedColormap
from mizani.palettes import brewer_pal
# Choose a categorical palette: 'Set1', 'Set2', 'Pastel1', etc.
palette_func = brewer_pal(type='qual', palette='Set1')
n_categories = int(dfc_communities_sorted.max() + 1)
colors = palette_func(n_categories)


# Paul Tol's bright palette (7 colors)
tol_bright = ["#BBBBBB",
    "#4477AA", "#EE6677", "#228833", "#CCBB44",
    "#66CCEE", "#AA3377", 
]


n_categories = int(dfc_communities_sorted.max() + 1)
cmap = ListedColormap(tol_bright[:n_categories])
#%%
# for animal in range(n_animals):
for animal in range(2):
    #plot one dfc_communities_sorted matrix
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 1, 1)
    plt.clf()
    # plt.title("Community label - Animal 0, Window 0")
    # plt.imshow(cm_0_mean.T , aspect='auto', interpolation='none', cmap='Greys')
    # aux_argsort = np.argsort(dfc_communities_sorted)
    plt.imshow(dfc_communities_sorted[animal].T, aspect='auto', interpolation='none', cmap=cmap)
    # plt.imshow(contingency_0[sorting_0][:, sorting_0], aspect='auto', interpolation='none', cmap='viridis')
    # plt.clim(0, 1)
    plt.colorbar()
    plt.yticks(np.arange(n_regions), labels=anat_labels[sort_allegiances[0, 0].astype(int)])
    plt.ylabel("Regions")
    plt.xlabel(r"Time Windows (TW$_{1}$, TW$_{2}$, ..., TW$_{n}$)")
    plt.title(f"Community labels - Animal {animal}")
    plt.savefig(paths['f_mod'] / f"dfc_communities_per_animal_{animal}.png", dpi=300, bbox_inches='tight')
    plt.show()
# ]

#%%

# ----------------- Consensus Clustering -----------------
#Compute the consensus clustering from the temporal aggregation of the contingency matrices
from shared_code.fun_metaconnectivity import build_agreement_matrix_vectorized

temporal_aggregation_mat = np.sum(contingency_matrices, axis=1) /n_windows   # Average across animals and windows

#Plot the allegiance matrix
plt.figure(figsize=(10, 8))
plt.imshow(temporal_aggregation_mat[0], aspect='auto', interpolation='none', cmap='Set1')
plt.colorbar()
plt.title("Temporal Aggregation Matrix")
plt.ylabel("Regions")
plt.xlabel("Regions")
# plt.clim(0,0.5)
plt.show()

#%%
import brainconn as bct  # or bctpy equivalent
from joblib import Parallel, delayed
import time
from scipy.stats import pearsonr

_runs=100
temporal_agreement_matrix = np.zeros((n_animals, n_regions, n_regions))  # Initialize agreement matrix
start_time = time.time()
for animal in tqdm(range(n_animals), desc="Animals"):
    partitions = []
    q_values = []
    # agreement_matrix: n_nodes x n_nodes, values in [0,1]
    results = Parallel(n_jobs=6)(  
        delayed(bct.modularity.modularity_louvain_und_sign)(temporal_aggregation_mat[animal], gamma=1)
        for _ in range(_runs)
        )
    
    for partition, q in results:
        partitions.append(partition)
        q_values.append(q)
    # print(f"Average modularity (Q): {np.mean(q_values)}")
    # ...and then cluster *that* matrix to get final consensus.

    # Build consensus agreement matrix from these partitions...
    temporal_agreement_matrix[animal] = build_agreement_matrix_vectorized(np.array(partitions))/ _runs
# temporal_agreement_matrix = temporal_agreement_matrix / _runs  # Normalize the agreement matrix by the number of runs
stop_time = time.time()
print(f"Time taken for consensus clustering: {stop_time - start_time} seconds /n {n_animals} animals")
#%%
# for animal in range(n_animals):
for animal in range(2):
    #plot one dfc_communities_sorted matrix
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 1, 1)
    plt.clf()
    plt.title(f"Temporal agreement matrix - Animal {animal}")
    # plt.imshow(cm_0_mean.T , aspect='auto', interpolation='none', cmap='Greys')
    # aux_argsort = np.argsort(dfc_communities_sorted)
    plt.imshow(temporal_agreement_matrix[animal].T, aspect='auto', interpolation='none', cmap=cmap)
    # plt.imshow(contingency_0[sorting_0][:, sorting_0], aspect='auto', interpolation='none', cmap='viridis')
    # plt.clim(0, 1)
    plt.colorbar()
    plt.yticks(np.arange(n_regions), labels=anat_labels[sort_allegiances[0, 0].astype(int)])
    plt.ylabel("Regions")
    plt.xlabel(r"Time Windows (TW$_{1}$, TW$_{2}$, ..., TW$_{n}$)")
    # plt.title(f"Consensus Clustering - Animal {animal}")
    plt.savefig(paths['f_mod'] / f"dfc_temp_agreement_per_animal_{animal}.png", dpi=300, bbox_inches='tight')
    plt.show()
#%%
# # Compute Pearson correlation between the agreement matrix and the temporal aggregation matrix
# pearson_val =pearsonr(temporal_agreement_matrix.flatten(), temporal_aggregation_mat[animal].flatten())
#   # Compute Pearson correlation between the agreement matrix and the temporal aggregation matrix
# pearson_val = pearsonr(temporal_agreement_matrix.flatten(), temporal_aggregation_mat[animal].flatten())
#   # Compute Pearson correlation between the agreement matrix and the temporal aggregation matrix
# pearson_val = pearsonr(temporal_agreement_matrix.flatten(), temporal_aggregation_mat[animal].flatten())
# temporal_agreement_matrix: n_nodes x n_nodes, values in [0,1]
results = Parallel(n_jobs=6)(  
    delayed(bct.modularity.modularity_louvain_und_sign)(temporal_agreement_matrix[animal], gamma=1)
    for animal in range(n_animals)
    )

community_agreement_labels, q_values = zip(*results)
# for partition, q in results:
#     partitions.append(partition)
#     q_values.append(q)


#%%
#community label alignment 

# For each window, label “1” should refer to (as much as possible) the same set of 
# regions across all windows. This process is called community label alignment 
# (or “label matching” or “tracking community identities”).

# Steps for community label alignment:
# 1. Identify the regions that correspond to label “1” in each window.
# 2. Create a mapping of these regions across all windows.
# 3. Apply this mapping to ensure consistent labeling.


from scipy.optimize import linear_sum_assignment
def align_community_labels(communities):
    """
    Align community labels across multiple windows.
    
    Parameters
    ----------
    communities : 2D array
        Array of shape (n_windows, n_regions) where each row represents the community labels for a window.
        
    Returns
    -------
    aligned_communities : 2D array
        Aligned community labels.
    """
    n_windows, n_regions = communities.shape
    aligned_communities = np.zeros_like(communities)

    for i in range(n_windows):
        # Create a cost matrix for the current window against all others
        cost_matrix = np.zeros((n_regions, n_regions))
        for j in range(n_windows):
            if i != j:
                for k in range(n_regions):
                    cost_matrix[k, :] += (communities[i] == k) * (communities[j] != k)
        
        # Solve the assignment problem to minimize the cost
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        aligned_communities[i] = col_ind[communities[i]]
    
    return aligned_communities

# Align community labels across all windows
aligned_communities_temporal = align_community_labels(dfc_communities_sorted[0].astype(int))
#%%
# Plot the aligned communities for one animal
plt.figure(figsize=(10, 8))
plt.imshow(aligned_communities_temporal.T, aspect='auto', interpolation='none', cmap=cmap)
plt.colorbar()
plt.yticks(np.arange(n_regions), labels=anat_labels[sort_allegiances[0, 0].astype(int)])
plt.ylabel("Regions")
plt.xlabel(r"Time Windows (TW$_{1}$, TW$_{2}$, ..., TW$_{n}$)")
plt.clim(0, np.max(dfc_communities_sorted[0]))
plt.title(f"Aligned Communities - Animal {animal}")
plt.savefig(paths['f_mod'] / f"dfc_aligned_communities_per_animal_{animal}.png", dpi=300, bbox_inches='tight')
plt.show()

#%%
#%# Alognment of temporal partitions using a 
# Consensus clustering with temporal aggregation matrix as reference

from scipy.optimize import linear_sum_assignment


reference = community_agreement_labels[0]  # Use the first window as reference

def align_partition_to_reference(partition, reference):
    
    n_comm = max(partition.max(), reference.max()) + 1
    
    # Build cost matrix: how many nodes overlap between community i (in partition) and community j (in reference)
    cost_matrix = np.zeros((n_comm, n_comm))
    for i in range(n_comm):
        for j in range(n_comm):
            cost_matrix[i, j] = -np.sum((partition == i) & (reference == j))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # Create new partition with remapped labels
    aligned = np.zeros_like(partition)
    for i, j in zip(row_ind, col_ind):
        aligned[partition == i] = j
    return aligned

# Align all windows to the first window
aligned_partitions = np.zeros_like(dfc_communities_sorted)
for animal in tqdm(range(n_animals)):
    partitions = dfc_communities_sorted[animal].astype(int)  # Use the current animal's communities as partitions
    reference = community_agreement_labels[animal]  # Use the current animal's first window as reference

    aligned_partitions[animal] = align_partition_to_reference(partitions, reference)


#%%
# Plot the aligned partitions for one animal
plt.figure(figsize=(10, 8))
plt.imshow(aligned_partitions[7].T, aspect='auto', interpolation='none', cmap=cmap)
plt.colorbar()
plt.yticks(np.arange(n_regions), labels=anat_labels[sort_allegiances[0, 0].astype(int)])
plt.ylabel("Regions")
plt.xlabel(r"Time Windows (TW$_{1}$, TW$_{2}$, ..., TW$_{n}$)")
plt.title(f"Aligned Partitions - Animal {animal}")
plt.savefig(paths['f_mod'] / f"dfc_aligned_partitions_per_animal_{animal}.png", dpi=300, bbox_inches='tight')
plt.show()


for animal in range(n_animals):
    print(np.unique(community_agreement_labels[animal]), np.unique(dfc_communities_sorted[animal]))
#%%
































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

agreement = build_agreement_matrix_vectorized(dfc_communities_sorted.T)

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
    return agreement.astype(np.float32)

agreement = build_agreement_matrix_vectorized(dfc_communities_sorted[0])

community_agreement = [build_agreement_matrix_vectorized(dfc_communities[indv]) for indv in range(n_animals)]
#%%

#plot the agreement matrix for different groups of animals
for i in range(9):
    plt.figure(figsize=(10, 8))
    plt.subplot(3, 3, i + 1)
    # plt.clf()
    plt.title(f"Agreement Matrix - Animal {i}")
    plt.imshow(community_agreement[i] / np.max(community_agreement[i]) , aspect='auto', interpolation='none', cmap='viridis')
    plt.colorbar()
    plt.tight_layout()
    # plt.xlabel("DFT Frequency")
    # plt.ylabel("DFT Frequency")
    plt.show()
# plt.figure(figsize=(10, 8))
# # plt.subplot(1, 1, 1)
# plt.clf()
# plt.title("Agreement Matrix - Animal 0")
# # plt.imshow(agreement/np.max(agreement) , aspect='auto', interpolation='none', cmap='viridis')
# plt.imshow(community_agreement/np.max(community_agreement) , aspect='auto', interpolation='none', cmap='viridis')
# plt.colorbar()
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

agreement = build_agreement_matrix_vectorized(dfc_communities_sorted.T)

plt.figure(figsize=(10, 8))
plt.subplot(1, 1, 1)
plt.clf()
plt.imshow(agreement , aspect='auto', interpolation='none', cmap='viridis')
#%%

























# %%
