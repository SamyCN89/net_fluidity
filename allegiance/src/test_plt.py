#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pickle

from shared_code.fun_paths import get_paths
from shared_code.fun_metaconnectivity import load_merged_allegiance, contingency_matrix_fun

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
# --- ALLEGIANCE DATA ---
dfc_communities, sort_allegiances, contingency_matrices = load_merged_allegiance(
    paths, window_size=window_size, lag=lag
)
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
# --- t-SNE (dimensionality reduction) ---
scaler = StandardScaler()
dfc_communities_sorted_scaled = scaler.fit_transform(
    dfc_communities_sorted.reshape(-1, dfc_communities_sorted.shape[-1])
)
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_jobs=-1, verbose=1, metric="euclidean")
cont_mat_n_pairs_tsne = tsne.fit_transform(cont_mat_n_pairs)

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
# --- COMMUNITY LABEL MEDIAN ---
dfc_communities_sorted_median = np.median(dfc_communities_sorted.T, axis=2)
plt.figure(figsize=(10, 8))
plt.title("Community label - Median")
plt.imshow(dfc_communities_sorted_median, aspect='auto', interpolation='none', cmap='viridis')
plt.colorbar()
plt.yticks(np.arange(n_regions), labels=anat_labels[sort_allegiances[0, 0].astype(int)])
plt.ylabel("Regions")
plt.xlabel("Time Windows")
plt.show()

# --- AGREEMENT MATRIX (vectorized) ---
def build_agreement_matrix_vectorized(communities):
    n_runs, n_nodes = communities.shape
    eq = (communities[:, :, None] == communities[:, None, :])
    return np.sum(eq, axis=0).astype(np.float32)

agreement = build_agreement_matrix_vectorized(dfc_communities_sorted.T)
plt.figure(figsize=(10, 8))
plt.title("Agreement Matrix")
plt.imshow(agreement, aspect='auto', interpolation='none', cmap='viridis')
plt.colorbar()
plt.show()

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
