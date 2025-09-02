# %%
from pathlib import Path as _Path
import pickle
import time

from joblib import Parallel, delayed
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
import logging
import argparse
import sys

# Optional/extra dependencies
try:
    from mizani.palettes import brewer_pal
except Exception:  # pragma: no cover
    brewer_pal = None  # type: ignore

try:
    import brainconn as bct
except Exception:  # pragma: no cover
    bct = None  # type: ignore

try:
    from sklearn.manifold import TSNE
    from sklearn.metrics import mutual_info_score
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    TSNE = None  # type: ignore
    StandardScaler = None  # type: ignore
    mutual_info_score = None  # type: ignore
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr

from shared_code.fun_metaconnectivity import (
    build_agreement_matrix_vectorized,
    load_merged_allegiance,  # %%
    contingency_matrix_fun,
)
from shared_code.fun_paths import get_paths

#%%
# -----------------------------------------------------------------------------
# Step 1: Centralized configuration
# -----------------------------------------------------------------------------
CONFIG = {
    "window_size": 9,
    "lag": 1,
    "tau": 3,
    "tsne": {"n_components": 2, "perplexity": 30, "random_state": 42},
    "consensus": {"n_runs": 1000, "gamma_pt": 10, "gmin": 0.5, "gmax": 1.0},
}

# Step 2: Lightweight logging setup
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

# Step 3: Small helpers to reduce duplication
def upper_triangle(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[-1]
    tri = np.triu_indices(n, k=1)
    return mat[..., tri[0], tri[1]]


def build_agreement_matrix(communities_2d: np.ndarray) -> np.ndarray:
    """Wrapper over vectorized agreement; communities_2d shape: (runs, nodes)."""
    return build_agreement_matrix_vectorized(communities_2d)


def validate_shapes(ts: np.ndarray,
                    dfc_communities: np.ndarray,
                    contingency_matrices: np.ndarray,
                    n_animals: int,
                    n_windows: int) -> None:
    """Centralized shape checks; raises AssertionError with clear messages."""
    assert ts.shape[0] == n_animals, "ts: n_animals mismatch"
    assert dfc_communities.shape[:2] == (n_animals, n_windows), "dfc_communities: shape mismatch"
    assert contingency_matrices.shape[0] == n_animals, "contingency_matrices: n_animals mismatch"




def plot_matrix(mat: np.ndarray, title: str, cmap: str = "viridis",
                save: bool = False, out_path: _Path | None = None, figsize=(10, 8)) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect="auto", interpolation="none", cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    if save and out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_tsne(embedding: np.ndarray, colors: np.ndarray | None,
              title: str, save: bool = False, out_path: _Path | None = None,
              figsize=(10, 8), s: int = 15) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=s, marker=".", cmap="tab20", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    if save and out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


parser = argparse.ArgumentParser(description="Allegiance analysis (v2)")
parser.add_argument("--window-size", type=int, default=CONFIG["window_size"], dest="window_size")
parser.add_argument("--lag", type=int, default=CONFIG["lag"], dest="lag")
parser.add_argument("--tau", type=int, default=CONFIG["tau"], dest="tau")
parser.add_argument("--overwrite-cache", action="store_true", dest="overwrite_cache",
                    help="Recompute cached intermediates (agreement, TSNE, etc.)")
parser.add_argument("--tsne-perp", type=float, default=CONFIG["tsne"]["perplexity"], dest="tsne_perp",
                    help="t-SNE perplexity")
parser.add_argument("--tsne-seed", type=int, default=CONFIG["tsne"]["random_state"], dest="tsne_seed",
                    help="t-SNE random seed")
parser.add_argument("--tsne-dim", type=int, default=CONFIG["tsne"]["n_components"], dest="tsne_dim",
                    help="t-SNE output dimensions")
parser.add_argument("--consensus-runs", type=int, default=CONFIG["consensus"]["n_runs"], dest="cons_n_runs",
                    help="Number of Louvain runs for consensus")
parser.add_argument("--consensus-gamma-pt", type=int, default=CONFIG["consensus"]["gamma_pt"], dest="cons_gamma_pt",
                    help="Number of gamma points")
parser.add_argument("--consensus-gmin", type=float, default=CONFIG["consensus"]["gmin"], dest="cons_gmin",
                    help="Minimum gamma value")
parser.add_argument("--consensus-gmax", type=float, default=CONFIG["consensus"]["gmax"], dest="cons_gmax",
                    help="Maximum gamma value")
parser.add_argument("--save-plots", action="store_true", dest="save_plots",
                    help="Save figures to disk under allegiance/fig")
# ARGS = parser.parse_args([]) if globals().get("__name__") != "__main__" else parser.parse_args()
ARGS, _ = parser.parse_known_args()

def compute_agreement_cached(dfc_sorted, cache_dir, window_size, lag, tau, overwrite=False):
    cache_dir.mkdir(parents=True, exist_ok=True)
    agree_cache = cache_dir / f"agreement_w{window_size}_lag{lag}_tau{tau}.npz"

    # agree_cache = agree_cache
    if agree_cache.exists() and not overwrite:
        try:
            arr = np.load(agree_cache)["agreement"]
            logger.info("[cache] Loaded agreement: %s", agree_cache)
            return arr
        except Exception:
            logger.warning("[cache] Failed to load %s; recomputing", agree_cache)
    arr = build_agreement_matrix_vectorized(dfc_sorted.T)
    np.savez_compressed(agree_cache, agreement=arr)
    return arr


def compute_tsne_cached(X, cache_dir, prefix, tsne_cfg, window_size, lag, tau, overwrite=False):
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = (cache_dir / (f"{prefix}_window={window_size}_lag={lag}_tau={tau}"
                          f"_perp={tsne_cfg['perplexity']}_seed={tsne_cfg['random_state']}.npz"))
    if fname.exists() and not overwrite:
        try:
            emb = np.load(fname)["embedding"]
            logger.info("[cache] Loaded %s", fname)
            return emb
        except Exception:
            logger.warning("[cache] Failed to load %s; recomputing", fname)
    tsne = TSNE(n_components=tsne_cfg["n_components"], perplexity=tsne_cfg["perplexity"],
                random_state=tsne_cfg["random_state"])
    emb = tsne.fit_transform(X)
    np.savez_compressed(fname, embedding=emb)
    return emb


# Set consistent config to match previous run / CLI
window_size = ARGS.window_size
lag = ARGS.lag
tau = ARGS.tau
timecourse_folder = "Timecourses_updated_03052024"
paths = get_paths(timecourse_folder=timecourse_folder)
plot_dir = (paths["allegiance"] / "fig").expanduser()
plot_dir.mkdir(parents=True, exist_ok=True)

# %% Load meta info to determine shape
data_ts = np.load(paths["preprocessed"] / "ts_and_meta_2m4m.npz", allow_pickle=True)
ts = data_ts["ts"]
n_animals = len(ts)
n_regions = ts[0].shape[1]
anat_labels = data_ts["anat_labels"]
logger.info("Loaded time series: n_animals=%d, n_regions=%d", n_animals, n_regions)

filename_dfc = (
    # f"window_size={window_size}_lag={lag}_animals={n_animals}_regions={n_regions}"
    f"window_size={window_size}_lag={lag}_tau={tau}_animals={n_animals}_regions={n_regions}"
)
dfc_data = np.load(paths["dfc"] / f"dfc_{filename_dfc}.npz")
n_windows = dfc_data["dfc"].shape[-1]
logger.info("Detected %d time windows from DFC cache", n_windows)
# %%
with open(paths["preprocessed"] / "grouping_data_oip.pkl", "rb") as f:
    mask_groups, label_variables = pickle.load(f)
with open(paths["preprocessed"] / "grouping_data_per_sex(gen_phen).pkl", "rb") as f:
    mask_groups_per_sex, label_variables_per_sex = pickle.load(f)
with open(paths["preprocessed"] / "grouping_data_new.pkl", "rb") as f:
    groups_sex_geno, groups_sex_pheno_oip, groups_sex_pheno_nor = pickle.load(f)

# ----- Load the mc data -----
# Load the regions and allegiance data
# Check if the regions and allegiance data are loaded correctly

label_ref = label_variables[2][0]  # Use the first label set for demonstration
n_runs_allegiance = 1000
gamma_pt_allegiance = 100
mc_mod_dataset = paths[
    "mc_mod"
] / f"mc_allegiance_ref(runs={label_ref}_gammaval={n_runs_allegiance})={gamma_pt_allegiance}_lag={lag}_windowsize={window_size}_animals={n_animals}_regions={n_regions}.npz".replace(
    " ", ""
)
# %%
# Load the merged allegiance data of all animals
dfc_communities, sort_allegiances, contingency_matrices = load_merged_allegiance(
    paths, window_size=9, lag=1
)
validate_shapes(ts, dfc_communities, contingency_matrices, n_animals, n_windows)
# %%
dfc_communities_sorted = np.zeros_like(dfc_communities)

for ani in tqdm(range(n_animals), desc="Animals"):
    for ws in range(n_windows):
        dfc_communities_sorted[ani, ws] = dfc_communities[
            ani, ws, sort_allegiances[ani, ws].astype(int)
        ]
# %%
# plot the dfc_communities_sorted matrix of 1st animal
plot_matrix(dfc_communities_sorted[0].T, "DFC Communities - Animal 0", cmap="viridis", save=ARGS.save_plots, out_path=plot_dir / f"dfc_communities_animal0_w{window_size}_l{lag}_t{tau}.png")

module_num = np.zeros(n_windows)
for i in range(n_windows):
    module_num[i] = len(
        np.unique(dfc_communities_sorted[1, i])
    )  # Check the unique values in the sorted communities for the first animal

# plot the number of modules per time window
plt.figure(figsize=(20, 6))
plt.plot(module_num, marker="o")
plt.title("Number of Modules per Time Window - Animal 0")
plt.xlabel("Time Windows")
plt.ylabel("Number of Modules")
plt.grid()
plt.show()

# %% Plot the triu contingency matrices for the first animal
n_regions = dfc_communities_sorted.shape[2]
# Extract the upper triangle indices for n_regions x n_regions matrix
triu_indices = np.triu_indices(
    n_regions, k=1
)  # Get upper triangle indices for n_regions x n_regions matrix
# Extract the upper triangle of the contingency matrices for the first animal
contingency_matrices_0 = contingency_matrices[
    0
]  # Get the contingency matrices for the first animal
contingency_matrices_0_triu = contingency_matrices_0[
    :, triu_indices[0], triu_indices[1]
]

# plot the hist of contingency_matrices_0_triu
plt.figure(figsize=(12, 12))
plt.clf()
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.title(f"Contingency Matrix - Animal 0, Window {i}")
    # Plot the histogram of the upper triangle of the contingency matrix for the first animal
    plt.hist(contingency_matrices_0_triu[i], bins=50, color="blue", alpha=0.7)
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
# %%
# plot sorted_cmat 1
plt.figure(figsize=(10, 8))
plt.imshow(
    contingency_matrices_0_triu > 0, aspect="auto", interpolation="none", cmap="viridis"
)
plt.colorbar()
plt.title("Sorted DFC Communities - Animal 0")
plt.ylabel("Regions")
plt.xlabel("Time Windows")
plt.show()


# %%
cmat = contingency_matrices_0
sort_idx = sort_allegiances[0].astype(
    int
)  # Get the sorting indices for the first animal

# Create an empty array for the sorted matrices
sorted_cmat = np.empty_like(cmat)

for i in range(cmat.shape[0]):
    idx = sort_idx[0]
    # Reorder both rows and columns using idx
    sorted_cmat[i] = cmat[i][idx, :][:, idx]
# %%
# plot sorted_cmat 1
plt.figure(figsize=(10, 8))
plt.imshow(sorted_cmat[0], aspect="auto", interpolation="none", cmap="viridis")
plt.colorbar()
plt.title("Sorted DFC Communities - Animal 0")
plt.ylabel("Regions")
plt.xlabel("Time Windows")
plt.show()
# plot sorted_cmat 1
plt.figure(figsize=(10, 8))
plt.imshow(
    contingency_matrices_0[0], aspect="auto", interpolation="none", cmap="viridis"
)
plt.colorbar()
plt.title("Sorted DFC Communities - Animal 0")
plt.ylabel("Regions")
plt.xlabel("Time Windows")
plt.show()


# %%
# contingency_matrices_0_triu = contingency_matrices_0[:, triu_indices[0], triu_indices[1]]
contingency_matrices_0_triu = sorted_cmat[:, triu_indices[0], triu_indices[1]]
# Compute the mean of the upper triangle across all windows for the first animal
# Plot the mean contingency matrix for the first animal
plt.figure(figsize=(10, 8))
plt.imshow(
    contingency_matrices_0_triu.T, aspect="auto", interpolation="none", cmap="viridis"
)
plt.colorbar()
plt.title("Mean Contingency Matrix - Animal 0")
plt.ylabel("Regions")
plt.xlabel("Regions")
plt.show()
# %%
# the spearman correlation between time points in contingency_matrices_0_triu (time_points x n_pairs)

time_corr_agreement = np.zeros(n_windows - 1)  # Initialize the correlation matrix
for i in range(n_windows - 1):
    time_corr_agreement[i] = spearmanr(
        contingency_matrices_0_triu[i], contingency_matrices_0_triu[i + 1]
    )[
        0
    ]  # Compute the correlation matrix of the upper triangle of the contingency matrices for the first animal


# %%
# plot the time correlation of the agreement matrices
plt.figure(figsize=(25, 7))
plt.plot(time_corr_agreement, "o-", markersize=5, alpha=0.7)
# %%
# already imported at top

# adjusted rand index
# Suppose agreement_matrices: shape (n_windows, n_regions, n_regions)
similarities = np.zeros((n_windows, n_windows))
# similarities = np.zeros((n_regions, n_regions))

contingency_matrices_0 = contingency_matrices[
    0
]  # Get the contingency matrices for the first animal

for t1 in tqdm(range(n_windows), desc="Windows"):
    for t2 in range(n_windows):

        # Example: Pearson correlation of upper triangles
        # Extract the contingency matrices for the two time windows
        # upper triangle indices
        triu_indices = np.triu_indices(
            n_regions, k=1
        )  # Get upper triangle indices for n_regions x n_regions matrix
        mat1 = contingency_matrices_0[t1][triu_indices[0], triu_indices[1]]
        mat2 = contingency_matrices_0[t2][triu_indices[0], triu_indices[1]]

        # mat1 = contingency_matrices_0[t1]
        # mat2 = contingency_matrices_0[t2]
        # Pearson correlation
        # similarities[t1, t2] = np.corrcoef(mat1, mat2)[0, 1]
        # spearman correlation
        # similarities[t1, t2] = spearmanr(mat1, mat2)[0]
        # # Frobenius norm
        similarities[t1, t2] = spearmanr(mat1, mat2)[0]
        # # Normalized Mutual Information
        # similarities[t1, t2] = normalized_mutual_info_score(mat1, mat2)
        # similarities[t1, t2] = np.linalg.norm(mat1 - mat2)

# similarities[t1, t2]: similarity between agreement matrices at window t1 and t2
# %%
# plot the similarities matrix
plt.figure(figsize=(10, 8))
plt.imshow(similarities, aspect="auto", interpolation="none", cmap="viridis")
plt.colorbar()
plt.title("Similarities Matrix - Animal 0")
plt.ylabel("Regions")
plt.xlabel("Regions")
plt.clim(0, 0.1)
plt.show()
# %%


# %%
# ------------------ communities allegiances through time windows ------------------


contingency_matrices_0 = contingency_matrices[
    0
]  # Get the contingency matrices for the first animal

upper_tri_indices = np.triu_indices(
    n_regions, k=1
)  # Get upper triangle indices for n_regions x n_regions matrix
# Extract the upper triangle of the contingency matrix for the first animal
contingency_matrices_0_triu = contingency_matrices_0[
    :, upper_tri_indices[0], upper_tri_indices[1]
]
# Compute the mean of the upper triangle across all windows for the first animal
# plot the contingency matrix for the first animal
plt.figure(figsize=(10, 8))

plt.imshow(
    contingency_matrices_0_triu.T, aspect="auto", interpolation="none", cmap="viridis"
)
plt.colorbar()
plt.title("Contingency Matrix - Animal 0")
plt.ylabel("n pairs")
plt.xlabel("TWs")
plt.show()

# %%

dfc_communities_sorted_0 = dfc_communities_sorted[
    0
]  # Get the sorted communities for the first animal

# Compute the mean of the sorted communities across all windows for the first animal
corr_mat = np.zeros((n_windows, n_windows))
for ii in tqdm(range(n_windows), desc="Computing correlations"):
    for jj in range(n_windows):
        corr, _ = pearsonr(dfc_communities_sorted_0[ii], dfc_communities_sorted_0[jj])
        corr_mat[ii, jj] = corr

# plot the correlation matrix
plt.figure(figsize=(10, 8))
plt.imshow(corr_mat, aspect="auto", interpolation="none", cmap="viridis")
plt.colorbar()
plt.title("Correlation Matrix - Animal 0")
plt.ylabel("Windows")
plt.xlabel("Windows")
plt.show()

# %%
# Mutual Information between columns of the dfc_communities_sorted matrix
if mutual_info_score is None:
    raise RuntimeError(
        "scikit-learn is required for mutual_info_score; install scikit-learn to run this section."
    )

mi_mat = np.zeros((n_windows, n_windows))
for ii in tqdm(range(n_windows), desc="Computing mutual information"):
    for jj in range(n_windows):
        mi = mutual_info_score(
            dfc_communities_sorted_0[ii], dfc_communities_sorted_0[jj]
        )
        mi_mat[ii, jj] = mi

# plot the mutual information matrix
plt.figure(figsize=(10, 8))
plt.imshow(mi_mat, aspect="auto", interpolation="none", cmap="viridis")
plt.colorbar()
plt.title("Mutual Information Matrix - Animal 0")
plt.ylabel("Windows")
plt.xlabel("Windows")
plt.clim(0, 0.3)
plt.show()
# %%

# %%

# Paul Tol's bright palette (7 colors)
tol_bright = [
    "#BBBBBB",
    "#4477AA",
    "#EE6677",
    "#228833",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
]
# Choose a categorical palette: 'Set1', 'Set2', 'Pastel1', etc. (fallback if mizani missing)
palette_func = (
    brewer_pal(type="qual", palette="Set1") if brewer_pal else (lambda n: tol_bright[:n])
)
n_categories = int(dfc_communities_sorted.max() + 1)
colors = palette_func(n_categories)




n_categories = int(dfc_communities_sorted.max() + 1)
cmap = ListedColormap(tol_bright[:n_categories])
# %%
# for animal in range(n_animals):
for animal in range(2):
    # plot one dfc_communities_sorted matrix
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 1, 1)
    plt.clf()
    # plt.title("Community label - Animal 0, Window 0")
    # plt.imshow(cm_0_mean.T , aspect='auto', interpolation='none', cmap='Greys')
    # aux_argsort = np.argsort(dfc_communities_sorted)
    plt.imshow(
        dfc_communities_sorted[animal].T, aspect="auto", interpolation="none", cmap=cmap
    )
    # plt.imshow(contingency_0[sorting_0][:, sorting_0], aspect='auto', interpolation='none', cmap='viridis')
    # plt.clim(0, 1)
    plt.colorbar()
    plt.yticks(
        np.arange(n_regions), labels=anat_labels[sort_allegiances[0, 0].astype(int)]
    )
    plt.ylabel("Regions")
    plt.xlabel(r"Time Windows (TW$_{1}$, TW$_{2}$, ..., TW$_{n}$)")
    plt.title(f"Community labels - Animal {animal}")
    plt.savefig(
        paths["f_mod"] / f"dfc_communities_per_animal_{animal}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
# ]

# %%

# ----------------- Consensus Clustering -----------------
# Compute the consensus clustering from the temporal aggregation of the contingency matrices

temporal_aggregation_mat = (
    np.sum(contingency_matrices, axis=1) / n_windows
)  # Average across animals and windows

# Plot the allegiance matrix
plot_matrix(temporal_aggregation_mat[0], "Temporal Aggregation Matrix", cmap="viridis", save=ARGS.save_plots, out_path=plot_dir / f"temporal_agg_w{window_size}_l{lag}_t{tau}.png")

# %%
_runs = 100
temporal_agreement_matrix = np.zeros(
    (n_animals, n_regions, n_regions)
)  # Initialize agreement matrix
start_time = time.time()
for animal in tqdm(range(n_animals), desc="Animals"):
    partitions = []
    q_values = []
    # agreement_matrix: n_nodes x n_nodes, values in [0,1]
    results = Parallel(n_jobs=6)(
        delayed(bct.modularity.modularity_louvain_und_sign)(
            temporal_aggregation_mat[animal], gamma=1
        )
        for _ in range(_runs)
    )

    for partition, q in results:
        partitions.append(partition)
        q_values.append(q)
    # print(f"Average modularity (Q): {np.mean(q_values)}")
    # ...and then cluster *that* matrix to get final consensus.

    # Build consensus agreement matrix from these partitions...
    temporal_agreement_matrix[animal] = (
        build_agreement_matrix_vectorized(np.array(partitions)) / _runs
    )
# temporal_agreement_matrix = temporal_agreement_matrix / _runs  # Normalize the agreement matrix by the number of runs
stop_time = time.time()
print(
    f"Time taken for consensus clustering: {stop_time - start_time} seconds /n {n_animals} animals"
)
# %%
# for animal in range(n_animals):
for animal in range(2):
    # plot one dfc_communities_sorted matrix
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 1, 1)
    plt.clf()
    plt.title(f"Temporal agreement matrix - Animal {animal}")
    # plt.imshow(cm_0_mean.T , aspect='auto', interpolation='none', cmap='Greys')
    # aux_argsort = np.argsort(dfc_communities_sorted)
    plt.imshow(
        temporal_agreement_matrix[animal].T,
        aspect="auto",
        interpolation="none",
        cmap=cmap,
    )
    # plt.imshow(contingency_0[sorting_0][:, sorting_0], aspect='auto', interpolation='none', cmap='viridis')
    # plt.clim(0, 1)
    plt.colorbar()
    plt.yticks(
        np.arange(n_regions), labels=anat_labels[sort_allegiances[0, 0].astype(int)]
    )
    plt.ylabel("Regions")
    plt.xlabel(r"Time Windows (TW$_{1}$, TW$_{2}$, ..., TW$_{n}$)")
    # plt.title(f"Consensus Clustering - Animal {animal}")
    plt.savefig(
        paths["f_mod"] / f"dfc_temp_agreement_per_animal_{animal}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
# %%
# # Compute Pearson correlation between the agreement matrix and the temporal aggregation matrix
# pearson_val =pearsonr(temporal_agreement_matrix.flatten(), temporal_aggregation_mat[animal].flatten())
#   # Compute Pearson correlation between the agreement matrix and the temporal aggregation matrix
# pearson_val = pearsonr(temporal_agreement_matrix.flatten(), temporal_aggregation_mat[animal].flatten())
#   # Compute Pearson correlation between the agreement matrix and the temporal aggregation matrix
# pearson_val = pearsonr(temporal_agreement_matrix.flatten(), temporal_aggregation_mat[animal].flatten())
# temporal_agreement_matrix: n_nodes x n_nodes, values in [0,1]
results = Parallel(n_jobs=6)(
    delayed(bct.modularity.modularity_louvain_und_sign)(
        temporal_agreement_matrix[animal], gamma=1
    )
    for animal in range(n_animals)
)

community_agreement_labels, q_values = zip(*results, strict=False)
# for partition, q in results:
#     partitions.append(partition)
#     q_values.append(q)


# %%
# community label alignment

# For each window, label “1” should refer to (as much as possible) the same set of
# regions across all windows. This process is called community label alignment
# (or “label matching” or “tracking community identities”).

# Steps for community label alignment:
# 1. Identify the regions that correspond to label “1” in each window.
# 2. Create a mapping of these regions across all windows.
# 3. Apply this mapping to ensure consistent labeling.

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
aligned_communities_temporal = align_community_labels(
    dfc_communities_sorted[0].astype(int)
)
# %%
# Plot the aligned communities for one animal
plt.figure(figsize=(10, 8))
plt.imshow(
    aligned_communities_temporal.T, aspect="auto", interpolation="none", cmap=cmap
)
plt.colorbar()
plt.yticks(np.arange(n_regions), labels=anat_labels[sort_allegiances[0, 0].astype(int)])
plt.ylabel("Regions")
plt.xlabel(r"Time Windows (TW$_{1}$, TW$_{2}$, ..., TW$_{n}$)")
plt.clim(0, np.max(dfc_communities_sorted[0]))
plt.title(f"Aligned Communities - Animal {animal}")
plt.savefig(
    paths["f_mod"] / f"dfc_aligned_communities_per_animal_{animal}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# %%
# %# Alognment of temporal partitions using a
# Consensus clustering with temporal aggregation matrix as reference



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
    for i, j in zip(row_ind, col_ind, strict=False):
        aligned[partition == i] = j
    return aligned


# Align all windows to the first window
aligned_partitions = np.zeros_like(dfc_communities_sorted)
for animal in tqdm(range(n_animals)):
    partitions = dfc_communities_sorted[animal].astype(
        int
    )  # Use the current animal's communities as partitions
    reference = community_agreement_labels[
        animal
    ]  # Use the current animal's first window as reference

    aligned_partitions[animal] = align_partition_to_reference(partitions, reference)


# %%
# Plot the aligned partitions for one animal
plt.figure(figsize=(10, 8))
plt.imshow(aligned_partitions[7].T, aspect="auto", interpolation="none", cmap=cmap)
plt.colorbar()
plt.yticks(np.arange(n_regions), labels=anat_labels[sort_allegiances[0, 0].astype(int)])
plt.ylabel("Regions")
plt.xlabel(r"Time Windows (TW$_{1}$, TW$_{2}$, ..., TW$_{n}$)")
plt.title(f"Aligned Partitions - Animal {animal}")
plt.savefig(
    paths["f_mod"] / f"dfc_aligned_partitions_per_animal_{animal}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()


for animal in range(n_animals):
    print(
        np.unique(community_agreement_labels[animal]),
        np.unique(dfc_communities_sorted[animal]),
    )
# %%


# %%
triu = np.triu_indices(
    n_regions, k=1
)  # Get upper triangle indices for n_regions x n_regions matrix


mask_groups_2 = mask_groups[2]  # Use the first group for demonstration
label_variables_2 = label_variables[2]  # Use the first label set for demonstration

# Concatenate or stack cont_mat_n_pairs in the first two dimensions
# cont_mat_n_pairs = contingency_matrices[mask_groups_2[0]][:,:, triu[0], triu[1]]  # Extract the upper triangle of the first contingency matrix
# cont_mat_n_pairs = np.concatenate(cont_mat_n_pairs, axis=0)  # Shape: (n_animals * n_windows, n_pairs)
# cont_mat_n_pairs_animal = np.repeat(np.arange(n_windows), np.sum(mask_groups_2[0]))  # Shape: (n_animals * n_windows,)

cont_mat_n_pairs = contingency_matrices[
    :, :, triu[0], triu[1]
]  # Extract the upper triangle of the first contingency matrix
cont_mat_n_pairs = np.concatenate(
    cont_mat_n_pairs, axis=0
)  # Shape: (n_animals * n_windows, n_pairs)
cont_mat_n_pairs_animal = np.repeat(
    np.arange(n_windows), n_animals
)  # Shape: (n_animals * n_windows,)

# List of contingency matrices for each animal in the contatenated array to color the t-SNE plot
# cont_mat_n_pairs_animal = np.repeat(np.arange(np.sum(mask_groups_2[0])), np.sum(mask_groups_2[0]))  # Shape: (n_animals * n_windows,)
# %%
# TSNE on one animal of contingency matrix (contingency_matrices[0])
if TSNE is None or StandardScaler is None:
    raise RuntimeError(
        "scikit-learn is required for TSNE/StandardScaler; install scikit-learn to run this section."
    )

tsne_perp = CONFIG["tsne"]["perplexity"]
tsne_seed = CONFIG["tsne"]["random_state"]

# Standardize the data (kept for potential later usage)
scaler = StandardScaler()
dfc_communities_sorted_scaled = scaler.fit_transform(
    dfc_communities_sorted.reshape(-1, dfc_communities_sorted.shape[-1])
)

# Perform t-SNE with caching
tsne_cache_pairs = (
    (paths["allegiance"] / "cache").expanduser()
    / f"tsne_pairs_window={window_size}_lag={lag}_tau={tau}_perp={tsne_perp}_seed={tsne_seed}.npz"
)
tsne_cache_pairs.parent.mkdir(parents=True, exist_ok=True)
if tsne_cache_pairs.exists() and not ARGS.overwrite_cache:
    try:
        cont_mat_n_pairs_tsne = np.load(tsne_cache_pairs)["embedding"]
        logger.info("[cache] Loaded TSNE pairs embedding: %s", tsne_cache_pairs)
    except Exception:
        logger.warning("[cache] Failed to load %s; recomputing", tsne_cache_pairs)
        tsne = TSNE(
            n_components=CONFIG["tsne"]["n_components"],
            random_state=tsne_seed,
            perplexity=tsne_perp,
        )
        cont_mat_n_pairs_tsne = compute_tsne_cached(cont_mat_n_pairs, tsne_cache_pairs.parent, "tsne_pairs", {"n_components": CONFIG["tsne"]["n_components"], "perplexity": tsne_perp, "random_state": tsne_seed}, window_size, lag, tau, overwrite=ARGS.overwrite_cache)
        np.savez_compressed(tsne_cache_pairs, embedding=cont_mat_n_pairs_tsne)
else:
    tsne = TSNE(
        n_components=CONFIG["tsne"]["n_components"],
        random_state=tsne_seed,
        perplexity=tsne_perp,
    )
    cont_mat_n_pairs_tsne = compute_tsne_cached(cont_mat_n_pairs, tsne_cache_pairs.parent, "tsne_pairs", {"n_components": CONFIG["tsne"]["n_components"], "perplexity": tsne_perp, "random_state": tsne_seed}, window_size, lag, tau, overwrite=ARGS.overwrite_cache)
    np.savez_compressed(tsne_cache_pairs, embedding=cont_mat_n_pairs_tsne)
# %%
# Plot the t-SNE results
plt.figure(figsize=(10, 8))
plt.scatter(
    cont_mat_n_pairs_tsne[:, 0],
    cont_mat_n_pairs_tsne[:, 1],
    c=cont_mat_n_pairs_animal,
    s=15,
    marker=".",
    cmap="tab20",  # Use a colormap to differentiate animals
    alpha=0.5,
)  # , cmap='viridis', alpha=0.5)
# plt.plot(cont_mat_n_pairs_tsne[:, 0], cont_mat_n_pairs_tsne[:, 1], '.', alpha=0.5)#, cmap='viridis', alpha=0.5)
plt.title("t-SNE of Contingency Matrix Pairs")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
# plt.grid(True)
plt.show()
# %%
# Compute and plot TSNE on transposed pairs (cached)
tsne_cache_pairs_T = (
    (paths["allegiance"] / "cache").expanduser()
    / f"tsne_pairsT_window={window_size}_lag={lag}_tau={tau}_perp={tsne_perp}_seed={tsne_seed}.npz"
)
if tsne_cache_pairs_T.exists() and not ARGS.overwrite_cache:
    try:
        cont_mat_n_pairs_tsne2 = np.load(tsne_cache_pairs_T)["embedding"]
        logger.info("[cache] Loaded TSNE pairs^T embedding: %s", tsne_cache_pairs_T)
    except Exception:
        logger.warning("[cache] Failed to load %s; recomputing", tsne_cache_pairs_T)
        tsne2 = TSNE(
            n_components=CONFIG["tsne"]["n_components"],
            random_state=tsne_seed,
            perplexity=tsne_perp,
        )
        cont_mat_n_pairs_tsne2 = compute_tsne_cached(cont_mat_n_pairs.T, tsne_cache_pairs_T.parent, "tsne_pairsT", {"n_components": CONFIG["tsne"]["n_components"], "perplexity": tsne_perp, "random_state": tsne_seed}, window_size, lag, tau, overwrite=ARGS.overwrite_cache)
        np.savez_compressed(tsne_cache_pairs_T, embedding=cont_mat_n_pairs_tsne2)
else:
    tsne2 = TSNE(
        n_components=CONFIG["tsne"]["n_components"],
        random_state=tsne_seed,
        perplexity=tsne_perp,
    )
    cont_mat_n_pairs_tsne2 = compute_tsne_cached(cont_mat_n_pairs.T, tsne_cache_pairs_T.parent, "tsne_pairsT", {"n_components": CONFIG["tsne"]["n_components"], "perplexity": tsne_perp, "random_state": tsne_seed}, window_size, lag, tau, overwrite=ARGS.overwrite_cache)
    np.savez_compressed(tsne_cache_pairs_T, embedding=cont_mat_n_pairs_tsne2)

plt.figure(figsize=(10, 8))
plt.scatter(
    cont_mat_n_pairs_tsne2[:, 0],
    cont_mat_n_pairs_tsne2[:, 1],
    s=50,
    marker=".",
    cmap="tab20",
    alpha=0.5,
)
plt.title("t-SNE of Contingency Matrix Pairs (Transposed)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()
# %%
# Plot the mean matrix

dfc_communities_sorted_median = np.median(
    dfc_communities_sorted.T, axis=2
)  # Take the median across the time windows

# %%
plot_matrix(dfc_communities_sorted_median, "Community label - Animal 0, Window 0", cmap="viridis", save=ARGS.save_plots, out_path=plot_dir / f"community_median_w{window_size}_l{lag}_t{tau}.png")

# %%
# Plot median community labels for mask_groups
plt.figure(figsize=(10, 8))
plt.clf()
# plt.imshow(cm_0_mean.T , aspect='auto', interpolation='none', cmap='Greys')
aux_argsort = np.argsort(dfc_communities_sorted)
dfc_groups = np.array(
    [
        np.median(dfc_communities_sorted[mask_groups[2][xx]], axis=0)
        for xx in range(len(mask_groups[2]))
    ]
)

# plt.subplots(2, 2, ii + 1)
for ii, mat in enumerate(dfc_groups):
    aux_labels = label_variables[2]
    plt.subplot(2, 2, ii + 1)
    plt.title(f"Community label - Group {aux_labels[ii]}, Window {ii+1}")
    plt.imshow(mat.T, aspect="auto", interpolation="none", cmap="viridis")
    # plt.clim(0, 1)
    # plt.colorbar()
    # plt.yticks(np.arange(len(mask_groups[0])), labels=label_variables[0][mask_groups[0]])
    # plt.ylabel("Regions")

# %%%# Check the shape of the loaded data

cm_0 = np.array(contingency_matrices[0])

triu_indices = np.array(np.triu_indices(n_regions, k=1))


cm_0_triu = cm_0[:, triu_indices[0], triu_indices[1]]


cm_0_mean_triu = np.mean(cm_0_triu, axis=0)

# Reshape the mean matrix to the original shape
cm_0_mean = np.zeros((n_regions, n_regions))
cm_0_mean[triu_indices[0], triu_indices[1]] = cm_0_mean_triu
cm_0_mean = cm_0_mean + cm_0_mean.T
cm_0_mean[np.diag_indices_from(cm_0_mean)] = 1

# Plot the mean matrix
plt.figure(figsize=(10, 8))
plt.subplot(1, 1, 1)
plt.clf()
plt.title("Allegiance Matrix - Animal 0, Window 0")
plt.imshow(cm_0_mean.T, aspect="auto", interpolation="none", cmap="Greys")
# plt.imshow(dfc_communities_sorted , aspect='auto', interpolation='none', cmap='Greys')
# plt.clim(0, 1)
plt.colorbar()
plt.xlabel("DFT Frequency")
plt.ylabel("DFT Frequency")
plt.show()


# %%
# Plot matrices 9 matrices in a grid
plt.figure(figsize=(10, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.clf()
    plt.title(f"Contingency Matrix - Animal 0, Window {i}")
    plt.imshow(cm_0[i].T, aspect="auto", interpolation="none", cmap="Greys")
    plt.clim(0, 1)
    plt.colorbar()
    plt.xlabel("DFT Frequency")
    plt.ylabel("DFT Frequency")
# %%
# plot imshow cm_0_triu
plt.figure(figsize=(10, 8))
plt.subplot(1, 1, 1)
plt.clf()
plt.title("Contingency Matrix - Animal 0, Window 0")
plt.imshow(cm_0_triu.T, aspect="auto", interpolation="none", cmap="Greys")
plt.clim(0, 1)
plt.colorbar()
plt.xlabel("DFT Frequency")
plt.ylabel("DFT Frequency")
plt.show()

# %%
# Plot the cumsum of cm_0_triu contingency matrix for all the windows
plt.figure(figsize=(10, 8))
plt.subplot(1, 1, 1)
plt.clf()
plt.title("Contingency Matrix - Animal 0, Window 0")
plt.plot(
    np.sort(cm_0_triu.ravel())
)  # aspect='auto', interpolation='none', cmap='Greys')
# %%
# Plot the histogram of the contingency matrix for all windows
plt.figure(figsize=(12, 12))
plt.title("Contingency Matrix - Animal 0, Window 0")

# One histogram per row (i.e., each region pair)
plt.hist(cm_0_triu[cm_0_triu > 0.1], bins=100, density=True, histtype="step")
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
    plt.imshow(cm_0[i].T, aspect="auto", interpolation="none", cmap="viridis")
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
        agreement += Ci[:, None] == Ci

    return agreement.astype(np.float32)


# %%
# Global Allegiance Matrix
# Average modular structure over all windows and animals
def _assert_square_symmetric(A, name="W"):
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"{name} must be square 2D: got shape {A.shape}")
    # not strictly necessary, but nice to enforce
    if not np.allclose(A, A.T, atol=1e-8, equal_nan=True):
        logging.warning("%s is not perfectly symmetric; symmetrizing.", name)
        return (A + A.T) / 2
    return A

# before calling Louvain
W = _assert_square_symmetric(W, "mc_data")

# "consensus" community structure over the whole period with Louvain method
contingency_matrix, gamma_qmod_val, gamma_agreement_mat = contingency_matrix_fun(
    1000,
    mc_data=W,
    gamma_range=10,
    gmin=0.5,
    gmax=1.0,
    cache_path=None,
    ref_name="",
    n_jobs=-1,
)
# %%
consensus_community = contingency_matrix

# Plot consensus community
plot_matrix(consensus_community, "Consensus Community (Contingency)", cmap="viridis", save=ARGS.save_plots, out_path=plot_dir / f"consensus_w{window_size}_l{lag}_t{tau}.png")
# Plot the mean matrix
# %%

cache_dir = paths["allegiance"] / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)
agree_cache = cache_dir / f"agreement_window={window_size}_lag={lag}_tau={tau}_animals={n_animals}_regions={n_regions}.npz"
if False:
    try:
        agreement = np.load(agree_cache)["agreement"]
        logger.info("[cache] Loaded agreement matrix: %s", agree_cache)
    except Exception:
        logger.warning("[cache] Failed to load %s; recomputing", agree_cache)
        agreement = compute_agreement_cached(dfc_communities_sorted, cache_dir, window_size, lag, tau, overwrite=ARGS.overwrite_cache)
        np.savez_compressed(agree_cache, agreement=agreement)
else:
    agreement = compute_agreement_cached(dfc_communities_sorted, cache_dir, window_size, lag, tau, overwrite=ARGS.overwrite_cache)
    np.savez_compressed(agree_cache, agreement=agreement)

plt.figure(figsize=(10, 8))
plt.subplot(1, 1, 1)
plt.clf()
plt.imshow(agreement[:,:,0], aspect="auto", interpolation="none", cmap="viridis")
plt.colorbar()
plt.xlabel("Regions")
plt.ylabel("Regions")
plt.title(f"Agreement Matrix - window={window_size}, lag={lag}, tau={tau}")
plt.show()

# %%


# # Set consistent config to match previous run
# window_size = 9
# lag = 1
# timecourse_folder = "Timecourses_updated_03052024"

# # Load meta info to determine shape
# paths = get_paths(timecourse_folder=timecourse_folder)
plot_dir = (paths["allegiance"] / "fig").expanduser()
plot_dir.mkdir(parents=True, exist_ok=True)
# %%
data_ts = np.load(paths["sorted"] / "ts_and_meta_2m4m.npz", allow_pickle=True)
ts = data_ts["ts"]
n_animals = len(ts)
n_regions = ts[0].shape[1]
anat_labels = data_ts["anat_labels"]

filename_dfc = (
    f"window_size={window_size}_lag={lag}_animals={n_animals}_regions={n_regions}"
)
dfc_data = np.load(paths["dfc"] / f"dfc_{filename_dfc}.npz")
n_windows = np.transpose(dfc_data["dfc_stream"], (0, 3, 2, 1)).shape[-1]

# %%
# already imported at top

with open(paths["sorted"] / "grouping_data_oip.pkl", "rb") as f:
    mask_groups, label_variables = pickle.load(f)
with open(paths["sorted"] / "grouping_data_per_sex(gen_phen).pkl", "rb") as f:
    mask_groups_per_sex, label_variables_per_sex = pickle.load(f)  # %%

# Load the merged allegiance data of all animals
dfc_communities, sort_allegiances, contingency_matrices = load_merged_allegiance(
    paths, window_size=9, lag=1
)
dfc_communities_sorted = dfc_communities[
    :, :, sort_allegiances[0, 0].astype(int)
]  # REaorder the labelling of the communities (deprecated soon)

# %%
triu = np.triu_indices(
    n_regions, k=1
)  # Get upper triangle indices for n_regions x n_regions matrix


mask_groups_2 = mask_groups[2]  # Use the first group for demonstration
label_variables_2 = label_variables[2]  # Use the first label set for demonstration

# Concatenate or stack cont_mat_n_pairs in the first two dimensions
# cont_mat_n_pairs = contingency_matrices[mask_groups_2[0]][:,:, triu[0], triu[1]]  # Extract the upper triangle of the first contingency matrix
# cont_mat_n_pairs = np.concatenate(cont_mat_n_pairs, axis=0)  # Shape: (n_animals * n_windows, n_pairs)
# cont_mat_n_pairs_animal = np.repeat(np.arange(n_windows), np.sum(mask_groups_2[0]))  # Shape: (n_animals * n_windows,)

cont_mat_n_pairs = contingency_matrices[
    :, :, triu[0], triu[1]
]  # Extract the upper triangle of the first contingency matrix
cont_mat_n_pairs = np.concatenate(
    cont_mat_n_pairs, axis=0
)  # Shape: (n_animals * n_windows, n_pairs)
cont_mat_n_pairs_animal = np.repeat(
    np.arange(n_windows), n_animals
)  # Shape: (n_animals * n_windows,)

# List of contingency matrices for each animal in the contatenated array to color the t-SNE plot
# cont_mat_n_pairs_animal = np.repeat(np.arange(np.sum(mask_groups_2[0])), np.sum(mask_groups_2[0]))  # Shape: (n_animals * n_windows,)
# %%
# TSNE on one animal of contingency matrix (contingency_matrices[0])
# already imported at top

# Standardize the data
scaler = StandardScaler()
dfc_communities_sorted_scaled = scaler.fit_transform(
    dfc_communities_sorted.reshape(-1, dfc_communities_sorted.shape[-1])
)

tsne_perp = CONFIG["tsne"]["perplexity"]
tsne_seed = CONFIG["tsne"]["random_state"]

# Perform t-SNE with caching
tsne_cache_pairs = (
    (paths["allegiance"] / "cache").expanduser()
    / f"tsne_pairs_window={window_size}_lag={lag}_tau={tau}_perp={tsne_perp}_seed={tsne_seed}.npz"
)
tsne_cache_pairs.parent.mkdir(parents=True, exist_ok=True)
if tsne_cache_pairs.exists() and not ARGS.overwrite_cache:
    try:
        cont_mat_n_pairs_tsne = np.load(tsne_cache_pairs)["embedding"]
        logger.info("[cache] Loaded TSNE pairs embedding: %s", tsne_cache_pairs)
    except Exception:
        logger.warning("[cache] Failed to load %s; recomputing", tsne_cache_pairs)
        tsne = TSNE(
            n_components=CONFIG["tsne"]["n_components"],
            random_state=tsne_seed,
            perplexity=tsne_perp,
        )
        cont_mat_n_pairs_tsne = compute_tsne_cached(cont_mat_n_pairs, tsne_cache_pairs.parent, "tsne_pairs", {"n_components": CONFIG["tsne"]["n_components"], "perplexity": tsne_perp, "random_state": tsne_seed}, window_size, lag, tau, overwrite=ARGS.overwrite_cache)
        np.savez_compressed(tsne_cache_pairs, embedding=cont_mat_n_pairs_tsne)
else:
    tsne = TSNE(
        n_components=CONFIG["tsne"]["n_components"],
        random_state=tsne_seed,
        perplexity=tsne_perp,
    )
    cont_mat_n_pairs_tsne = compute_tsne_cached(cont_mat_n_pairs, tsne_cache_pairs.parent, "tsne_pairs", {"n_components": CONFIG["tsne"]["n_components"], "perplexity": tsne_perp, "random_state": tsne_seed}, window_size, lag, tau, overwrite=ARGS.overwrite_cache)
    np.savez_compressed(tsne_cache_pairs, embedding=cont_mat_n_pairs_tsne)
# %%
# Plot the t-SNE results
plt.figure(figsize=(10, 8))
plt.scatter(
    cont_mat_n_pairs_tsne[:, 0],
    cont_mat_n_pairs_tsne[:, 1],
    c=cont_mat_n_pairs_animal,
    s=15,
    marker=".",
    cmap="tab20",  # Use a colormap to differentiate animals
    alpha=0.5,
)  # , cmap='viridis', alpha=0.5)
# plt.plot(cont_mat_n_pairs_tsne[:, 0], cont_mat_n_pairs_tsne[:, 1], '.', alpha=0.5)#, cmap='viridis', alpha=0.5)
plt.title("t-SNE of Contingency Matrix Pairs")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
# plt.grid(True)
plt.show()
# %%
# Compute and plot TSNE on transposed pairs (cached)
tsne_cache_pairs_T = (
    (paths["allegiance"] / "cache").expanduser()
    / f"tsne_pairsT_window={window_size}_lag={lag}_tau={tau}_perp={tsne_perp}_seed={tsne_seed}.npz"
)
if tsne_cache_pairs_T.exists() and not ARGS.overwrite_cache:
    try:
        cont_mat_n_pairs_tsne2 = np.load(tsne_cache_pairs_T)["embedding"]
        logger.info("[cache] Loaded TSNE pairs^T embedding: %s", tsne_cache_pairs_T)
    except Exception:
        logger.warning("[cache] Failed to load %s; recomputing", tsne_cache_pairs_T)
        tsne2 = TSNE(
            n_components=CONFIG["tsne"]["n_components"],
            random_state=tsne_seed,
            perplexity=tsne_perp,
        )
        cont_mat_n_pairs_tsne2 = compute_tsne_cached(cont_mat_n_pairs.T, tsne_cache_pairs_T.parent, "tsne_pairsT", {"n_components": CONFIG["tsne"]["n_components"], "perplexity": tsne_perp, "random_state": tsne_seed}, window_size, lag, tau, overwrite=ARGS.overwrite_cache)
        np.savez_compressed(tsne_cache_pairs_T, embedding=cont_mat_n_pairs_tsne2)
else:
    tsne2 = TSNE(
        n_components=CONFIG["tsne"]["n_components"],
        random_state=tsne_seed,
        perplexity=tsne_perp,
    )
    cont_mat_n_pairs_tsne2 = compute_tsne_cached(cont_mat_n_pairs.T, tsne_cache_pairs_T.parent, "tsne_pairsT", {"n_components": CONFIG["tsne"]["n_components"], "perplexity": tsne_perp, "random_state": tsne_seed}, window_size, lag, tau, overwrite=ARGS.overwrite_cache)
    np.savez_compressed(tsne_cache_pairs_T, embedding=cont_mat_n_pairs_tsne2)

plt.figure(figsize=(10, 8))
plt.scatter(
    cont_mat_n_pairs_tsne2[:, 0],
    cont_mat_n_pairs_tsne2[:, 1],
    s=50,
    marker=".",
    cmap="tab20",
    alpha=0.5,
)
plt.title("t-SNE of Contingency Matrix Pairs (Transposed)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.show()
# %%
# Plot the mean matrix

dfc_communities_sorted_median = np.median(
    dfc_communities_sorted.T, axis=2
)  # Take the median across the time windows

# %%
plot_matrix(dfc_communities_sorted_median, "Community label - Animal 0, Window 0", cmap="viridis", save=ARGS.save_plots, out_path=plot_dir / f"community_median_w{window_size}_l{lag}_t{tau}.png")

# %%
# Plot median community labels for mask_groups
plt.figure(figsize=(10, 8))
plt.clf()
# plt.imshow(cm_0_mean.T , aspect='auto', interpolation='none', cmap='Greys')
aux_argsort = np.argsort(dfc_communities_sorted)
dfc_groups = np.array(
    [
        np.median(dfc_communities_sorted[mask_groups[2][xx]], axis=0)
        for xx in range(len(mask_groups[2]))
    ]
)

# plt.subplots(2, 2, ii + 1)
for ii, mat in enumerate(dfc_groups):
    aux_labels = label_variables[2]
    plt.subplot(2, 2, ii + 1)
    plt.title(f"Community label - Group {aux_labels[ii]}, Window {ii+1}")
    plt.imshow(mat.T, aspect="auto", interpolation="none", cmap="viridis")
    # plt.clim(0, 1)
    # plt.colorbar()
    # plt.yticks(np.arange(len(mask_groups[0])), labels=label_variables[0][mask_groups[0]])
    # plt.ylabel("Regions")

# %%%# Check the shape of the loaded data

cm_0 = np.array(contingency_matrices[0])

triu_indices = np.array(np.triu_indices(n_regions, k=1))


cm_0_triu = cm_0[:, triu_indices[0], triu_indices[1]]


cm_0_mean_triu = np.mean(cm_0_triu, axis=0)

# Reshape the mean matrix to the original shape
cm_0_mean = np.zeros((n_regions, n_regions))
cm_0_mean[triu_indices[0], triu_indices[1]] = cm_0_mean_triu
cm_0_mean = cm_0_mean + cm_0_mean.T
cm_0_mean[np.diag_indices_from(cm_0_mean)] = 1

# Plot the mean matrix
plt.figure(figsize=(10, 8))
plt.subplot(1, 1, 1)
plt.clf()
plt.title("Allegiance Matrix - Animal 0, Window 0")
plt.imshow(cm_0_mean.T, aspect="auto", interpolation="none", cmap="Greys")
# plt.imshow(dfc_communities_sorted , aspect='auto', interpolation='none', cmap='Greys')
# plt.clim(0, 1)
plt.colorbar()
plt.xlabel("DFT Frequency")
plt.ylabel("DFT Frequency")
plt.show()


# %%
# Plot matrices 9 matrices in a grid
plt.figure(figsize=(10, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.clf()
    plt.title(f"Contingency Matrix - Animal 0, Window {i}")
    plt.imshow(cm_0[i].T, aspect="auto", interpolation="none", cmap="Greys")
    plt.clim(0, 1)
    plt.colorbar()
    plt.xlabel("DFT Frequency")
    plt.ylabel("DFT Frequency")
# %%
# plot imshow cm_0_triu
plt.figure(figsize=(10, 8))
plt.subplot(1, 1, 1)
plt.clf()
plt.title("Contingency Matrix - Animal 0, Window 0")
plt.imshow(cm_0_triu.T, aspect="auto", interpolation="none", cmap="Greys")
plt.clim(0, 1)
plt.colorbar()
plt.xlabel("DFT Frequency")
plt.ylabel("DFT Frequency")
plt.show()

# %%
# Plot the cumsum of cm_0_triu contingency matrix for all the windows
plt.figure(figsize=(10, 8))
plt.subplot(1, 1, 1)
plt.clf()
plt.title("Contingency Matrix - Animal 0, Window 0")
plt.plot(
    np.sort(cm_0_triu.ravel())
)  # aspect='auto', interpolation='none', cmap='Greys')
# %%
# Plot the histogram of the contingency matrix for all windows
plt.figure(figsize=(12, 12))
plt.title("Contingency Matrix - Animal 0, Window 0")

# One histogram per row (i.e., each region pair)
plt.hist(cm_0_triu[cm_0_triu > 0.1], bins=100, density=True, histtype="step")
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
    plt.imshow(cm_0[i].T, aspect="auto", interpolation="none", cmap="viridis")
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
        agreement += Ci[:, None] == Ci

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
    equal_matrix = communities[:, :, None] == communities[:, None, :]
    # Sum over runs
    agreement = np.sum(equal_matrix, axis=0)
    return agreement.astype(np.float32)


agreement = build_agreement_matrix_vectorized(dfc_communities_sorted[0])

community_agreement = [
    build_agreement_matrix_vectorized(dfc_communities[indv])
    for indv in range(n_animals)
]
# %%

# plot the agreement matrix for different groups of animals
for i in range(9):
    plt.figure(figsize=(10, 8))
    plt.subplot(3, 3, i + 1)
    # plt.clf()
    plt.title(f"Agreement Matrix - Animal {i}")
    plt.imshow(
        community_agreement[i] / np.max(community_agreement[i]),
        aspect="auto",
        interpolation="none",
        cmap="viridis",
    )
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
# %%
# Global Allegiance Matrix

# Average modular structure over all windows and animals
allegiance_matrices = cm_0
allegiance_avg = allegiance_matrices.mean(axis=0)

# "consensus" community structure over the whole period with Louvain method
# already imported at top

# contingency_matrix, gamma_qmod_val, gamma_agreement_mat =contingency_matrix_fun(1000, mc_data=allegiance_avg, gamma_range=10, gmin=0.1, gmax=1, cache_path=None, ref_name='', n_jobs=-1)
contingency_matrix, gamma_qmod_val, gamma_agreement_mat = contingency_matrix_fun(
    1000,
    mc_data=dfc_communities_sorted.T,
    gamma_range=10,
    gmin=0.5,
    gmax=1,
    cache_path=None,
    ref_name="",
    n_jobs=-1,
)
# %%
consensus_community = contingency_matrix

# Plot consensus community
plot_matrix(consensus_community, "Consensus Community (Contingency)", cmap="viridis", save=ARGS.save_plots, out_path=plot_dir / f"consensus_w{window_size}_l{lag}_t{tau}.png")
# Plot the mean matrix
# %%

agreement = compute_agreement_cached(dfc_communities_sorted, cache_dir, window_size, lag, tau, overwrite=ARGS.overwrite_cache)

plot_matrix(agreement, f"Agreement Matrix - window={window_size}, lag={lag}, tau={tau}", cmap="viridis", save=ARGS.save_plots, out_path=plot_dir / f"agreement_w{window_size}_l{lag}_t{tau}.png")
# Placeholder to keep subsequent code aligned
# %%


# %%
