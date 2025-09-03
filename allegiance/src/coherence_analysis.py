# %%
import argparse
import logging
from operator import index
from pathlib import Path as _Path
import pickle

import matplotlib.pyplot as plt
import numpy as np

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
    TSNE = None  # type: ignoredmn_labels_index = (2,13,22,23,28,34,39,37)
    StandardScaler = None  # type: ignore
    mutual_info_score = None  # type: ignore

from shared_code.fun_metaconnectivity import (
    build_agreement_matrix_vectorized,
    load_merged_allegiance,  # %%
)
from shared_code.fun_paths import get_paths

# %%
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


def validate_shapes(
    ts: np.ndarray,
    dfc_communities: np.ndarray,
    contingency_matrices: np.ndarray,
    n_animals: int,
    n_windows: int,
) -> None:
    """Centralized shape checks; raises AssertionError with clear messages."""
    assert ts.shape[0] == n_animals, "ts: n_animals mismatch"
    assert dfc_communities.shape[:2] == (
        n_animals,
        n_windows,
    ), "dfc_communities: shape mismatch"
    assert (
        contingency_matrices.shape[0] == n_animals
    ), "contingency_matrices: n_animals mismatch"


def plot_matrix(
    mat: np.ndarray,
    title: str,
    cmap: str = "viridis",
    save: bool = False,
    out_path: _Path | None = None,
    figsize=(10, 8),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect="auto", interpolation="none", cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    if save and out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_tsne(
    embedding: np.ndarray,
    colors: np.ndarray | None,
    title: str,
    save: bool = False,
    out_path: _Path | None = None,
    figsize=(10, 8),
    s: int = 15,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors,
        s=s,
        marker=".",
        cmap="tab20",
        alpha=0.5,
    )
    ax.set_title(title)
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    if save and out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


parser = argparse.ArgumentParser(description="Allegiance analysis (v2)")
parser.add_argument(
    "--window-size", type=int, default=CONFIG["window_size"], dest="window_size"
)
parser.add_argument("--lag", type=int, default=CONFIG["lag"], dest="lag")
parser.add_argument("--tau", type=int, default=CONFIG["tau"], dest="tau")
parser.add_argument(
    "--overwrite-cache",
    action="store_true",
    dest="overwrite_cache",
    help="Recompute cached intermediates (agreement, TSNE, etc.)",
)
parser.add_argument(
    "--tsne-perp",
    type=float,
    default=CONFIG["tsne"]["perplexity"],
    dest="tsne_perp",
    help="t-SNE perplexity",
)
parser.add_argument(
    "--tsne-seed",
    type=int,
    default=CONFIG["tsne"]["random_state"],
    dest="tsne_seed",
    help="t-SNE random seed",
)
parser.add_argument(
    "--tsne-dim",
    type=int,
    default=CONFIG["tsne"]["n_components"],
    dest="tsne_dim",
    help="t-SNE output dimensions",
)
parser.add_argument(
    "--consensus-runs",
    type=int,
    default=CONFIG["consensus"]["n_runs"],
    dest="cons_n_runs",
    help="Number of Louvain runs for consensus",
)
parser.add_argument(
    "--consensus-gamma-pt",
    type=int,
    default=CONFIG["consensus"]["gamma_pt"],
    dest="cons_gamma_pt",
    help="Number of gamma points",
)
parser.add_argument(
    "--consensus-gmin",
    type=float,
    default=CONFIG["consensus"]["gmin"],
    dest="cons_gmin",
    help="Minimum gamma value",
)
parser.add_argument(
    "--consensus-gmax",
    type=float,
    default=CONFIG["consensus"]["gmax"],
    dest="cons_gmax",
    help="Maximum gamma value",
)
parser.add_argument(
    "--save-plots",
    action="store_true",
    dest="save_plots",
    help="Save figures to disk under allegiance/fig",
)
# ARGS = parser.parse_args([]) if globals().get("__name__") != "__main__" else parser.parse_args()
ARGS, _ = parser.parse_known_args()


def compute_agreement_cached(
    dfc_sorted, cache_dir, window_size, lag, tau, overwrite=False
):
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


def compute_tsne_cached(
    X, cache_dir, prefix, tsne_cfg, window_size, lag, tau, overwrite=False
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = cache_dir / (
        f"{prefix}_window={window_size}_lag={lag}_tau={tau}"
        f"_perp={tsne_cfg['perplexity']}_seed={tsne_cfg['random_state']}.npz"
    )
    if fname.exists() and not overwrite:
        try:
            emb = np.load(fname)["embedding"]
            logger.info("[cache] Loaded %s", fname)
            return emb
        except Exception:
            logger.warning("[cache] Failed to load %s; recomputing", fname)
    tsne = TSNE(
        n_components=tsne_cfg["n_components"],
        perplexity=tsne_cfg["perplexity"],
        random_state=tsne_cfg["random_state"],
    )
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
# Reorder community labels using vectorized indexing for efficiency
dfc_communities_sorted = np.take_along_axis(
    dfc_communities, sort_allegiances.astype(int), axis=2
)


# %%
def plot_matrix(
    mat: np.ndarray,
    title: str,
    cmap: str = "viridis",
    save: bool = False,
    out_path: _Path | None = None,
    figsize=(10, 8),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect="auto", interpolation="none", cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    plt.yticks(
        np.arange(n_regions), labels=anat_labels[sort_allegiances[0, 0].astype(int)]
    )
    plt.xlabel(r"Time Windows (TW$_{1}$, TW$_{2}$, ..., TW$_{n}$)")
    if save and out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


# plot the dfc_communities_sorted matrix of 1st animal
plot_matrix(
    dfc_communities_sorted[0].T,
    "dFC Communities - Animal 0",
    cmap="viridis",
    save=ARGS.save_plots,
    out_path=plot_dir / f"dfc_communities_animal0_w{window_size}_l{lag}_t{tau}.png",
)

# the module_num for all the animals
module_num = np.zeros((n_animals, n_windows))
for animal in range(n_animals):
    for i in range(n_windows):
        module_num[animal, i] = len(
            np.unique(dfc_communities_sorted[animal, i])
        )  # Check the unique values in the sorted communities for each animal

# plot the number of modules per time window
plt.figure(figsize=(20, 6))
plt.plot(module_num[0], marker="o")
plt.title("Number of Modules per Time Window - Animal 0")
plt.xlabel(r"Time Windows (TW$_{1}$, TW$_{2}$, ..., TW$_{n}$)")
plt.ylabel("Number of Modules")
plt.show()

# %%
# Compute for each animal the min, max and mean number of modules across time windows
min_modules = np.min(module_num, axis=1)
max_modules = np.max(module_num, axis=1)
mean_modules = np.mean(module_num, axis=1)
for animal in range(n_animals):
    print(
        f"Animal {animal} - Min: {min_modules[animal]}, Max: {max_modules[animal]}, Mean: {mean_modules[animal]}"
    )

# %%
# plot the mean_modules per animal, sorted by group
plt.figure(figsize=(20, 6))
plt.title("Mean Number of Modules per Animal")
for i, label in enumerate(label_variables):
    plt.subplot(len(label_variables), 1, i + 1)
    for j, lbl in enumerate(label):
        plt.plot(mean_modules[mask_groups[0][j]], marker="o", label=f"{lbl}")
plt.xlabel("Animals")
plt.ylabel("Mean Number of Modules")
# plt.xticks(ticks=np.arange(n_animals), labels=[f"Animal {i}" for i in range(n_animals)], rotation=90)
plt.legend()
plt.show()

# %%
# Assumptions:
# - mean_modules: shape (n_animals,)
# - mask_groups has the same structure as label_variables:
#   mask_groups[factor_idx][level_idx] is a boolean mask (n_animals,) for that level
# - Each factor has labels like "... 2m" and "... 4m" that pair naturally


def sem(x):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    return np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def build_xy_for_factor(labels_i, masks_i, mean_modules):
    """
    Pair levels that end with '2m' and '4m' into base groups.
    Returns lists: base_names, x_means (2m), y_means (4m), x_sem, y_sem.
    """
    # Parse base name + age
    parsed = []
    for lbl, m in zip(labels_i, masks_i, strict=False):
        lbl = str(lbl)
        if lbl.endswith("2m"):
            base = lbl[:-2].strip()
            age = "2m"
        elif lbl.endswith("4m"):
            base = lbl[:-2].strip()
            age = "4m"
        else:
            # ignore unrecognized ages
            continue
        parsed.append((base, age, m))

    # Group by base and collect 2m/4m masks
    by_base = {}
    for base, age, m in parsed:
        by_base.setdefault(base, {})[age] = m

    base_names, x_means, y_means, x_err, y_err = [], [], [], [], []
    for base, ages in by_base.items():
        m2 = ages.get("2m", None)
        m4 = ages.get("4m", None)
        if m2 is None or m4 is None:
            # skip bases that don't have both ages
            continue

        vals2 = mean_modules[m2]
        vals4 = mean_modules[m4]

        # compute group means and SEMs
        x_mu, y_mu = float(np.nanmean(vals2)), float(np.nanmean(vals4))
        x_se, y_se = sem(vals2), sem(vals4)

        base_names.append(base)
        x_means.append(x_mu)
        y_means.append(y_mu)
        x_err.append(x_se)
        y_err.append(y_se)

    return (
        base_names,
        np.array(x_means),
        np.array(y_means),
        np.array(x_err),
        np.array(y_err),
    )


# ---- Plot: one panel per factor ----
n_factors = len(label_variables)
fig, axes = plt.subplots(
    1, n_factors, figsize=(6 * n_factors, 6), sharex=True, sharey=True
)
if n_factors == 1:
    axes = [axes]

factors_labels = ("oip", "nor", "genotype", "sex")

for ax, (labels_i, masks_i), fi in zip(
    axes,
    zip(label_variables, mask_groups, strict=False),
    range(n_factors),
    strict=False,
):
    base_names, x_mu, y_mu, x_se, y_se = build_xy_for_factor(
        labels_i, masks_i, mean_modules
    )

    # Scatter with error bars
    ax.errorbar(x_mu, y_mu, xerr=x_se, yerr=y_se, fmt="o", capsize=3, lw=1, alpha=0.9)

    # Annotate each point
    for bn, x, y in zip(base_names, x_mu, y_mu, strict=False):
        ax.annotate(bn, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=10)

    # Reference line y=x
    lim_lo = min(np.min(x_mu - x_se), np.min(y_mu - y_se)) if len(x_mu) else 0
    lim_hi = max(np.max(x_mu + x_se), np.max(y_mu + y_se)) if len(x_mu) else 1
    pad = 0.05 * (lim_hi - lim_lo if lim_hi > lim_lo else 1.0)
    lo, hi = lim_lo - pad, lim_hi + pad
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)

    ax.set_title(f"{factors_labels[fi]}")
    ax.set_xlabel("Mean modules @ 2m")
    ax.set_ylabel("Mean modules @ 4m")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

plt.suptitle("2m vs 4m mean modules (group-level)", y=1.02)
plt.tight_layout()
plt.show()

# %%


def plot_matrix(
    mat: np.ndarray,
    title: str,
    cmap: str = "viridis",
    save: bool = False,
    out_path: _Path | None = None,
    figsize=(10, 8),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect="auto", interpolation="none", cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    plt.yticks(
        np.arange(n_regions), labels=anat_labels[sort_allegiances[0, 0].astype(int)]
    )
    plt.xlabel(r"Time Windows (TW$_{1}$, TW$_{2}$, ..., TW$_{n}$)")
    if save and out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


# plot the dfc_communities_sorted matrix of 1st animal
plot_matrix(
    dfc_communities_sorted[0].T,
    "dFC Communities - Animal 0",
    cmap="viridis",
    save=False,
)
# %%

############################################################################################
#- Cohesion analysis
#############################################################################################
# for the first animal, compute the cohesion timeseries between region 1 and region 2


#Default Mode Network regions
dmn_labels_index = [0, 23, 13, 22, 2, 28, 34, 37, 39, 8, 35]


def cohesion_probability(communities):
    cohesion_probability = np.zeros((len(dmn_labels_index), len(dmn_labels_index)))
    # cohesion_probability = np.zeros((n_regions, n_regions))
    for reg1 in range(len(dmn_labels_index)):
        for reg2 in range(reg1 + 1, len(dmn_labels_index)):
            aux1 = communities[:, dmn_labels_index[reg1]]
            aux2 = communities[:, dmn_labels_index[reg2]]
            # aux1 = communities[dmn_labels_index[reg1]]
            # aux2 = communities[dmn_labels_index[reg2]]
            # print(f'size aux1: {aux1.shape}, size aux2: {aux2.shape}')
            cohesion_timeseries = aux1 - aux2
            # count the number of time windows where the two regions are in the same module
            cohesion_probability[reg1, reg2] = (cohesion_timeseries == 0).sum() / len(
                cohesion_timeseries
            )
            # print(cohesion_probability[reg1, reg2])
    # print(aux1)
    return cohesion_probability


cohesion_probability_all = np.zeros(
    (n_animals, len(dmn_labels_index), len(dmn_labels_index))
)

for animal in range(n_animals):
    cohesion_probability_all[animal] = cohesion_probability(
        dfc_communities_sorted[animal]
    )

# %%
# plot imshow of cohesion_probability
plt.figure(figsize=(10, 8))
plt.imshow(
    cohesion_probability_all[1],
    aspect="auto",
    interpolation="none",
    cmap="viridis",
    vmin=0,
    vmax=1,
)
plt.colorbar()
plt.title("Cohesion Probability")
plt.xlabel("Region 2")
plt.ylabel("Region 1")
plt.xticks(
    np.arange(len(dmn_labels_index)),
    labels=anat_labels[sort_allegiances[0, 0].astype(int)][dmn_labels_index],
    rotation=90,
)
plt.yticks(
    np.arange(len(dmn_labels_index)),
    labels=anat_labels[sort_allegiances[0, 0].astype(int)][dmn_labels_index],
)
plt.show()

# %%

plt.figure(figsize=(10, 8))
plt.clf()
for i in range(n_animals):
    plt.subplot(12, 11, i + 1)
    plt.imshow(
        cohesion_probability_all[i],
        aspect="auto",
        interpolation="none",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    plt.xticks([])

    plt.yticks([])
    # plt.colorbar()
    # plt.title(f"Cohesion Probability - Animal {i}")
# plt.xlabel("Region 2")
# plt.ylabel("Region 1")
# plt.yticks(np.arange(n_regions), labels=anat_labels[sort_allegiances[0, 0].astype(int)])
plt.show()

# %%


#plot cohesion_probability_mean_group

for xx, label in zip(mask_groups, label_variables):
    print(label)
    for i, group in enumerate(xx):
        plt.figure(figsize=(10, 8))
        plt.imshow(
            np.mean(cohesion_probability_all[group], axis=0),
            aspect="auto",
            interpolation="none",
            cmap="viridis",
            vmin=0.1,
            vmax=0.7,
        )
        plt.colorbar()
        plt.title(f"Mean Cohesion {label[i]} ")
        plt.yticks(
            np.arange(len(dmn_labels_index)),
            labels=anat_labels[sort_allegiances[0, 0].astype(int)][dmn_labels_index],
        )
        plt.xticks(
            np.arange(len(dmn_labels_index)),
            labels=anat_labels[sort_allegiances[0, 0].astype(int)][dmn_labels_index],
            rotation=90,
        )
        plt.show()

plt.figure(figsize=(10, 8))

for i, group in enumerate(mask_groups[0]):
    plt.subplot(4, 2, 1+i)
    cohesion_probability_mean_group = np.mean(cohesion_probability_all[group], axis=0)

    plt.imshow(
        cohesion_probability_mean_group,
        aspect="auto",
        interpolation="none",
        cmap="viridis",
        vmin=0.1,
        vmax=0.7,
    )
    plt.colorbar()
    plt.title(f"Mean Cohesion {label_variables[0][i]} ")
    plt.yticks(
        np.arange(len(dmn_labels_index)),
        labels=anat_labels[sort_allegiances[0, 0].astype(int)][dmn_labels_index],
    )
plt.xticks(
    np.arange(len(dmn_labels_index)),
    labels=anat_labels[sort_allegiances[0, 0].astype(int)][dmn_labels_index],
    rotation=90,
)

plt.tight_layout()
plt.show()

# %%
#------------Cohesion time series-------------------------------



def cohesion_timeseries(communities, region_index=None):

    if region_index==None:
        region_index = np.arange(communities.shape[1])
    cohesion_timeseries = np.zeros((len(region_index), len(region_index), communities.shape[0]))
    # cohesion_probability = np.zeros((n_regions, n_regions))
    for reg1 in range(len(region_index)):
        for reg2 in range(reg1 + 1, len(region_index)):
            aux1 = communities[:,region_index[reg1]]
            aux2 = communities[:,region_index[reg2]]
            cohesion_timeseries[reg1, reg2, :] = aux1 - aux2
            # count the number of time windows where the two regions are in the same module
    return cohesion_timeseries

cohesion_timeseries_all = np.zeros(
    (n_animals, n_regions, n_regions, n_windows)
)

cohesion_timeseries_dmn = np.zeros(
    (n_animals, len(dmn_labels_index), len(dmn_labels_index), n_windows)
)

for animal in range(n_animals):
    cohesion_timeseries_dmn[animal] = cohesion_timeseries(
        dfc_communities_sorted[animal], region_index=dmn_labels_index
    )
    cohesion_timeseries_all[animal] = cohesion_timeseries(
        dfc_communities_sorted[animal]    )


plt.figure(figsize=(17, 8))
index_timeseries = np.triu_indices(len(dmn_labels_index), k=1)
plt.plot(cohesion_timeseries_all[0,0,2,:])
plt.xlim(-1,50)
plt.figure(figsize=(10, 8))
index_timeseries = np.triu_indices(len(dmn_labels_index), k=1)
for idx, (ii, jj) in enumerate(zip(index_timeseries[0], index_timeseries[1], strict=False)):
    plt.subplot(11, 10, idx + 1)
    plt.plot(cohesion_timeseries_all[0,ii,jj,:])
#%%


#------------------------Burst of cohesion--------------------------

def extract_link_activations(binary_fc_data):
    """
    Extract onset, offset, and duration of active (1-valued) FC links over time.

    Parameters:
    -----------
    binary_fc_data : np.ndarray
        Shape (n_animals, time_points, n_links), binary values (0 or 1).

    Returns:
    --------
    all_events : list of lists of dicts
        all_events[animal][link] is a list of event dicts with keys: onset, offset, duration
    """
    n_animals, n_timepoints, n_links = binary_fc_data.shape
    all_events = []

    for animal in range(n_animals):
        animal_events = []
        for link in range(n_links):
            signal = binary_fc_data[animal, :, link]
            diff = np.diff(np.concatenate([[0], signal, [0]]))
            onsets = np.where(diff == 1)[0]
            offsets = np.where(diff == -1)[0]
            durations = offsets - onsets

            events = [
                {"onset": int(o), "offset": int(f), "duration": int(d)}
                for o, f, d in zip(onsets, offsets, durations, strict=False)
            ]
            animal_events.append(events)
        all_events.append(animal_events)

    return all_events
#%%
index_timeseries = np.triu_indices(n_regions, k=1)
index_timeseries_dmn = np.triu_indices(len(dmn_labels_index), k=1)
cohesion_timeseries_all_triu = cohesion_timeseries_all[:, index_timeseries[0], index_timeseries[1], :]
cohesion_timeseries_dmn_triu = cohesion_timeseries_dmn[:, index_timeseries_dmn[0], index_timeseries_dmn[1], :]
cohesion_timeseries_all_binary = (cohesion_timeseries_all_triu == 0).astype(int)
cohesion_timeseries_dmn_binary = (cohesion_timeseries_dmn_triu == 0).astype(int)

cohesion_timeseries_all_binary = np.transpose(cohesion_timeseries_all_binary, (0, 2, 1))
cohesion_timeseries_dmn_binary = np.transpose(cohesion_timeseries_dmn_binary, (0, 2, 1))
# burst_cohesion = extract_link_activations(cohesion_timeseries_all_binary)

# duration = burst_cohesion[0][0][0]["duration"]

# %%

import numpy as np
import pandas as pd

def extract_link_activations_df(binary_fc_data: np.ndarray, min_duration: int = 1) -> pd.DataFrame:

# def extract_link_activations_df(binary_fc_data: np.ndarray) -> pd.DataFrame:
    """
    Vectorized extraction of onsets, offsets, and durations of active (1-valued) FC links.

    Parameters
    ----------
    binary_fc_data : np.ndarray
        Shape (n_animals, n_timepoints, n_links), dtype {0,1}

    Returns
    -------
    events_df : pd.DataFrame
        Columns: ['animal', 'link', 'onset', 'offset', 'duration']
        One row per activation burst.
    """
    # Expect shape (A, T, L)
    A, T, L = binary_fc_data.shape

    # Pad a zero at both ends along time, then diff along time
    z = np.zeros((A, 1, L), dtype=binary_fc_data.dtype)
    xpad = np.concatenate((z, binary_fc_data, z), axis=1)
    d = np.diff(xpad, axis=1)  # shape (A, T+1, L)

    # Find all onsets/offsets across the whole array
    # Each row in *_idx is [animal, time_index, link]
    on_idx  = np.argwhere(d == 1)
    off_idx = np.argwhere(d == -1)

    # Build DataFrames; use a group key to pair onsets/offsets per (animal, link)
    on = pd.DataFrame(on_idx, columns=["animal", "time", "link"])
    off = pd.DataFrame(off_idx, columns=["animal", "time", "link"])

    on["gid"]  = on["animal"] * L + on["link"]
    off["gid"] = off["animal"] * L + off["link"]

    # Order by group then time; assign a within-group index (0,1,2,...) to pair events
    on = on.sort_values(["gid", "time"]).reset_index(drop=True)
    off = off.sort_values(["gid", "time"]).reset_index(drop=True)

    on["idx"]  = on.groupby("gid").cumcount()
    off["idx"] = off.groupby("gid").cumcount()

    # Merge on (gid, idx) to align each onset with its corresponding offset
    events = on.merge(off, on=["gid", "idx"], suffixes=("_on", "_off"))

    # Sanity: animals/links must match after merge
    # (They do, but keep columns from the onset side)
    events = events.rename(columns={
        "animal_on": "animal",
        "link_on": "link",
        "time_on": "onset",
        "time_off": "offset",
    })[["animal", "link", "onset", "offset"]]

    # Duration in original (unpadded) time indexing
    events["duration"] = events["offset"] - events["onset"]

    # Keep only long enough events
    events = events[events["duration"] >= min_duration]
    return events

def events_df_to_nested(events: pd.DataFrame, n_animals: int, n_links: int):
    """
    Optional: convert the tidy DataFrame back to the original nested list structure.
    Returns: list[n_animals][n_links] -> list of dicts {onset, offset, duration}
    """
    nested = [[[] for _ in range(n_links)] for _ in range(n_animals)]
    for row in events.itertuples(index=False):
        nested[row.animal][row.link].append(
            {"onset": int(row.onset), "offset": int(row.offset), "duration": int(row.duration)}
        )
    return nested

# Your binary array: (n_animals, time_points, n_links)
# cohesion_timeseries_all_binary already has that shape if you computed the upper triangle per time.
# events_df = extract_link_activations_df(cohesion_timeseries_all_binary, min_duration=2)
events_df = extract_link_activations_df(cohesion_timeseries_dmn_binary, min_duration=2)

# If you still want the old structure:
all_events = events_df_to_nested(events_df,
                                 n_animals=cohesion_timeseries_dmn_binary.shape[0],
                                 n_links=cohesion_timeseries_dmn_binary.shape[2])

# Equivalent to your old example of grabbing the first duration:
duration = all_events[0][0][0]["duration"]

# Or, directly from the DataFrame (e.g., first event of animal 0, link 0):
duration_df = (events_df.query("animal == 0 and link == 1")
                        .sort_values("onset")
                        )

# %%
def mean_duration_matrix(events_df: pd.DataFrame, n_animals: int, n_links: int, fill=0.0):
    """
    Returns a (n_animals, n_links) array with the mean duration of bursts.
    Links with no bursts get `fill` (default 0.0).
    """
    m = (events_df
         .groupby(["animal", "link"])["duration"]
         .mean()
         .unstack("link"))
    m = m.reindex(index=range(n_animals), columns=range(n_links))  # ensure full grid
    return m.fillna(fill).to_numpy()

def std_duration_matrix(events_df: pd.DataFrame, n_animals: int, n_links: int, fill=0.0):
    """
    Returns a (n_animals, n_links) array with the standard deviation of burst durations.
    Links with no bursts get `fill` (default 0.0).
    """
    m = (events_df
         .groupby(["animal", "link"])["duration"]
         .std()
         .unstack("link"))
    m = m.reindex(index=range(n_animals), columns=range(n_links))  # ensure full grid
    return m.fillna(fill).to_numpy()


mean_dur = mean_duration_matrix(events_df, n_animals=cohesion_timeseries_dmn_binary.shape[0], n_links=cohesion_timeseries_dmn_binary.shape[2])
std_dur = std_duration_matrix(events_df, n_animals=cohesion_timeseries_dmn_binary.shape[0], n_links=cohesion_timeseries_dmn_binary.shape[2])


# Burstiness coefficient
burstiness = (std_dur - mean_dur) / (std_dur + mean_dur)
burstiness[mean_dur == 0] = 0  # avoid division by zero

#%%
import cmocean as cm
plt.figure(figsize=(17, 8))
plt.imshow(burstiness, interpolation='none', aspect='auto', cmap=cm.cm.balance,
           vmin=-1, vmax=1)
plt.colorbar()


#%%
anat_labels_sorted  =anat_labels[sort_allegiances[0, 0].astype(int)]

links_label =()
for xx in index_timeseries_dmn[0]:
    for yy in index_timeseries_dmn[1]:
        links_label.append(f'{anat_labels_sorted[dmn_labels_index[xx]]}-{anat_labels_sorted[dmn_labels_index[yy]]}' )

mask=mask_groups[2][2]  # 4m XY
label = label_variables[2][2]
plt.figure(figsize=(17, 8))
plt.imshow(burstiness[mask], interpolation='none', aspect='auto', cmap=cm.cm.balance,
           vmin=-1, vmax=1)
plt.ylabel('animals')
# plt.xticks(np.arange(burstiness.shape[1]), labels=links_label, rotation=90)
plt.colorbar()
# %%
per_animal_mean = np.nanmean(np.where(burstiness==0, np.nan, burstiness), axis=1)


for i in range(4):
    for label, group in zip(label_variables[i], mask_groups[i]):
        print(f'{label} burst mean duration {np.mean(per_animal_mean[group])}')

# %%
