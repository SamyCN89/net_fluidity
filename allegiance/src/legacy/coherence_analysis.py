# %%
import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import logging
from pathlib import Path as _Path
import pickle
import re

import brainconn as bct
import cmocean as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from shared_code.fun_metaconnectivity import load_merged_allegiance
from shared_code.fun_paths import get_paths


# %%
# -----------------------------------------------------------------------------
# Config & logging
# -----------------------------------------------------------------------------
# Step 1: Centralized configuration
#
@dataclass(frozen=True)
class Config:
    window_size: int = 9
    lag: int = 1
    tau: int = 3
    timecourse_folder = "Timecourses_updated_03052024"
    dmn_labels_index = [0, 23, 13, 22, 2, 28, 34, 37, 39, 8, 35]


CONFIG = Config()
timecourse_folder = CONFIG.timecourse_folder
dmn_labels_index = CONFIG.dmn_labels_index

# --- Block spec: single source of truth (order matters) ---
# ('block title', 'single', factor_idx)  OR  ('block title', 'pair', factorA_idx, factorB_idx)
BLOCK_SPEC = [
    ("Sex", "single", 3),
    ("Genotype", "single", 2),
    ("OiP", "single", 0),
    ("Sex×Genotype", "pair", 3, 2),
    ("Sex×OiP", "pair", 3, 0),
    ("Genotype×OiP", "pair", 2, 0),
]

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
# Step 3: Small helpers to reduce duplication
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


def sem(x):
    """Standard Error of the Mean, ignoring NaNs."""
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    return np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def split_base_age(label: str) -> tuple[str, str | None]:
    """'Female 2m' -> ('Female','2m'); 'Good 4m' -> ('Good','4m')"""
    parts = str(label).split()
    if len(parts) >= 2 and re.fullmatch(r"\d+m", parts[-1]):
        return " ".join(parts[:-1]), parts[-1]
    return label, None


def build_paths(timecourse_folder: str):
    paths = get_paths(timecourse_folder=timecourse_folder)
    plot_dir = (paths["allegiance"] / "fig").expanduser()
    plot_dir.mkdir(parents=True, exist_ok=True)
    return paths, plot_dir


def load_base_data(paths, window_size: int, lag: int, tau: int):
    data_ts = np.load(paths["preprocessed"] / "ts_and_meta_2m4m.npz", allow_pickle=True)
    ts = data_ts["ts"]
    anat_labels = data_ts["anat_labels"]

    dfc_file = f"dfc_window_size={window_size}_lag={lag}_tau={tau}_animals={len(ts)}_regions={ts[0].shape[1]}.npz"
    dfc_data = np.load(paths["dfc"] / dfc_file)

    n_animals = len(ts)
    n_regions = ts[0].shape[1]
    n_windows = dfc_data["dfc"].shape[-1]
    return ts, anat_labels, n_animals, n_regions, n_windows


def load_sorted_communities(paths, window_size: int, lag: int):
    dfc_communities, sort_allegiances, contingency_matrices = load_merged_allegiance(
        paths, window_size=window_size, lag=lag
    )
    dfc_communities_sorted = np.take_along_axis(
        dfc_communities, sort_allegiances.astype(int), axis=2
    )
    return dfc_communities_sorted, sort_allegiances, contingency_matrices


# -----------------------------------------------------------------------------
# ---- Plot helpers functions ----
# -----------------------------------------------------------------------------
def plot_matrix(
    mat: np.ndarray,
    title: str,
    *,
    ytick_labels: list[str] | None = None,
    cmap: str = "viridis",
    save: bool = False,
    out_path: _Path | None = None,
    figsize=(10, 8),
) -> None:
    """Plot a matrix with optional y-tick labels."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect="auto", interpolation="none", cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    if ytick_labels is not None:
        ax.set_yticks(np.arange(len(ytick_labels)))
        ax.set_yticklabels(ytick_labels)
    ax.set_xlabel(r"Time Windows (TW$_{1}$, TW$_{2}$, ..., TW$_{n}$)")
    if save and out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


def separators_from_multiindex(mi: pd.MultiIndex):
    """From a MultiIndex, return list of (end_idx, level_value) for each level change."""
    if mi.nlevels == 0 or mi.size == 0:
        return []
    blocks = mi.get_level_values(0)
    seps, cur, count = [], blocks[0], 0
    for b in blocks:
        if b != cur:
            seps.append((count, cur))
            cur = b
        count += 1
    seps.append((count, cur))
    return seps


def plot_sig_pvals_multi(
    pvals_df: pd.DataFrame,
    alpha: float = 0.05,
    title: str = "Paired Wilcoxon — significant only",
    show_grid: bool = True,
):
    data = pvals_df.values
    mask = np.where(data <= alpha, data, np.nan)
    n_rows, n_cols = mask.shape
    fig, ax = plt.subplots(figsize=(max(15, 0.22 * n_cols), max(0.01, 0.16 * n_rows)))
    im = ax.imshow(
        mask, aspect="auto", interpolation="none", cmap="viridis_r", vmin=0, vmax=alpha
    )
    fig.colorbar(im, ax=ax).set_label("p-value")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(pvals_df.index, fontsize=6)
    col_labels = [f"{b} | {c}" for b, c in pvals_df.columns.to_list()]
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=8)
    if show_grid:
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which="minor", color="k", linestyle="-", linewidth=0.3, alpha=0.25)
        ax.grid(which="major", visible=False)
    for x_end, _ in separators_from_multiindex(pvals_df.columns)[:-1]:
        ax.axvline(x_end - 0.5, color="k", lw=1.0, alpha=0.6)
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def plot_weighted_multi(
    pvals_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    alpha: float = 0.05,
    title: str = "(1 - p) × mean cohesion diff",
    vmin=-0.1,
    vmax=0.1,
):
    assert tuple(pvals_df.columns) == tuple(weights_df.columns)
    assert list(pvals_df.index) == list(weights_df.index)
    p, w = pvals_df.values, weights_df.values
    Z = np.where(p <= alpha, 1 - p, np.nan) * w
    n_rows, n_cols = Z.shape
    fig, ax = plt.subplots(figsize=(max(15, 0.22 * n_cols), max(0.01, 0.16 * n_rows)))
    im = ax.imshow(
        Z, aspect="auto", interpolation="none", cmap="RdBu", vmin=vmin, vmax=vmax
    )
    fig.colorbar(im, ax=ax).set_label("(1 - p) × mean cohesion diff")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(pvals_df.index, fontsize=6)
    col_labels = [f"{b} | {c}" for b, c in pvals_df.columns.to_list()]
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=8)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="k", linestyle="-", linewidth=0.3, alpha=0.25)
    ax.grid(which="major", visible=False)
    for x_end, _ in separators_from_multiindex(pvals_df.columns)[:-1]:
        ax.axvline(x_end - 0.5, color="k", lw=1.0, alpha=0.6)
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# -------------------- Grouping helpers (cached) --------------------
# -----------------------------------------------------------------------------

# %%
# Global memo for factor_base_indices
_FBI_CACHE: dict[tuple, dict[str, dict[str, np.ndarray | None]]] = {}


def factor_base_indices(
    factor_idx: int,
    label_variables: Sequence[Sequence[str]],
    mask_groups: Sequence[Sequence[np.ndarray]],
) -> dict[str, dict[str, np.ndarray | None]]:
    """
    Build base-> {'2m': idx, '4m': idx} once per (factor, labels, masks) signature.
    Caches by a hashable signature (labels as tuples of str, masks as bytes).
    """
    # Build a hashable signature
    labels_sig = tuple(tuple(map(str, lv)) for lv in label_variables)
    masks_sig = tuple(
        tuple(np.asarray(m, dtype=bool).tobytes() for m in mg) for mg in mask_groups
    )
    key = (factor_idx, labels_sig, masks_sig)
    if key in _FBI_CACHE:
        return _FBI_CACHE[key]

    # Compute once if not cached
    bases: dict[str, dict[str, np.ndarray | None]] = {}
    labels = label_variables[factor_idx]
    masks = mask_groups[factor_idx]
    for lbl, m in zip(labels, masks, strict=False):
        base, age = split_base_age(lbl)
        if age in {"2m", "4m"}:
            idx = np.flatnonzero(m)
            ent = bases.setdefault(base, {"2m": None, "4m": None})
            ent[age] = idx

    _FBI_CACHE[key] = bases
    return bases


# %%


# -----------------------------------------------------------------------------
# ---- Command line arguments ----
# -----------------------------------------------------------------------------
#   Parse command line arguments
def build_arg_parser(cfg: Config) -> argparse.ArgumentParser:
    """Builds the argument parser for command line execution."""
    parser = argparse.ArgumentParser(description="Allegiance analysis (v3)")
    parser.add_argument(
        "--window-size", type=int, default=cfg.window_size, dest="window_size"
    )
    parser.add_argument("--lag", type=int, default=cfg.lag, dest="lag")
    parser.add_argument("--tau", type=int, default=cfg.tau, dest="tau")
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save figures to disk under allegiance/fig",
    )
    return parser


# ARGS = parser.parse_args([]) if globals().get("__name__") != "__main__" else parser.parse_args()
ARGS, _ = build_arg_parser(CONFIG).parse_known_args()

# %%

# ---- Main analysis ----
window_size = ARGS.window_size
lag = ARGS.lag
tau = ARGS.tau
paths, plot_dir = build_paths(timecourse_folder=timecourse_folder)

# %% Load meta info to determine shape

ts, anat_labels, n_animals, n_regions, n_windows = load_base_data(
    paths, window_size, lag, tau
)

logger.info("Loaded time series: n_animals=%d, n_regions=%d", n_animals, n_regions)

filename_dfc = (
    # f"window_size={window_size}_lag={lag}_animals={n_animals}_regions={n_regions}"
    f"window_size={window_size}_lag={lag}_tau={tau}_animals={n_animals}_regions={n_regions}"
)
dfc_data = np.load(paths["dfc"] / f"dfc_{filename_dfc}.npz")
logger.info("Detected %d time windows from DFC cache", n_windows)


# ---- Load grouping data ----
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
dfc_sorted, sort_allegiances, cont = load_sorted_communities(paths, window_size, lag)

anat_labels = anat_labels[sort_allegiances[0, 0].astype(int)]
anat_labels_sorted = anat_labels
# %%

plt.figure(figsize=(17, 8))
plt.imshow(cont[0, 0], aspect="auto", interpolation="none", cmap="viridis")
plt.colorbar()
plt.title("dFC Communities - Animal 0")
plt.xlabel("Time Windows")
plt.ylabel("Regions")
plt.show()

# %%
# --------------- Compute the number of modules in each time window for all the animals
module_num = np.zeros((n_animals, n_windows))
for animal in range(n_animals):
    for i in range(n_windows):
        module_num[animal, i] = len(
            np.unique(dfc_sorted[animal, i])
        )  # Check the unique values in the sorted communities for each animal


# plot the dfc_sorted matrix of 1st animal
plot_matrix(
    dfc_sorted[2].T,
    "dFC Communities - Animal 0",
    cmap="viridis",
    ytick_labels=list(anat_labels_sorted),
    save=ARGS.save_plots,
    out_path=plot_dir / f"dfc_communities_animal0_w{window_size}_l{lag}_t{tau}.png",
)

# plot the number of modules per time window
plt.figure(figsize=(20, 6))
plt.plot(module_num[0], marker="o")
plt.title("Number of Modules per Time Window - Animal 0")
plt.xlabel(r"Time Windows (TW$_{1}$, TW$_{2}$, ..., TW$_{n}$)")
plt.ylabel("Number of Modules")
plt.show()

# scatter plot of 2m vs 4m of number of modules per time window for groups

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
        plt.plot(mean_modules[mask_groups[0][j]], label=f"{lbl}", alpha=0.5, marker=".")
    plt.legend(fontsize=4)
plt.xlabel("Animals")
plt.ylabel("Mean Number of Modules")
# plt.xticks(ticks=np.arange(n_animals), labels=[f"Animal {i}" for i in range(n_animals)], rotation=90)
plt.show()


# %%
# Difference of mean modules between 2m and 4m for each animal in each group

# %%

# Assuming that the first half of the animals are 2m and the second half are 4m
half_aux = int(len(mean_modules) / 2)

diff_mean_modules = mean_modules[:half_aux] - mean_modules[half_aux:]
normed = (diff_mean_modules - diff_mean_modules.min()) / (
    diff_mean_modules.max() - diff_mean_modules.min()
)


plt.figure(figsize=(10, 6))

for factor_idx in range(len(label_variables)):
    # factor_idx = 0  # choose which grouping
    plt.subplot(4, 1, 1 + factor_idx)

    labels = label_variables[factor_idx]
    masks = mask_groups[factor_idx]
    for lbl, mask in zip(labels, masks, strict=False):
        idx = np.where(mask)[0]
        print(lbl, idx)
        # if 2m in lbl:
        if "2m" in lbl:
            # plt.plot(normed[idx], label=lbl, marker='.', alpha=0.3)
            plt.plot(diff_mean_modules[idx], label=lbl, marker=".", alpha=0.3)
            print(f"Found 2m in {lbl}: {idx}")
        elif "4m" in lbl:
            print(f"Found 4m in {lbl}: {idx}")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.legend()


# %%
# diff_mean_modules = mean_modules[mask_groups[0][1]] - mean_modules[mask_groups[0][0]]
print(f"Difference of mean modules between 2m and 4m: {diff_mean_modules}")
# plot the difference of mean modules between 2m and 4m for each animal in each group
plt.figure(figsize=(10, 6))
plt.bar(np.arange(len(diff_mean_modules)), diff_mean_modules, alpha=0.7)
plt.xticks(
    np.arange(len(diff_mean_modules)),
    [f"Animal {i}" for i in range(len(diff_mean_modules))],
    rotation=45,
)
plt.xlabel("Animals")
plt.ylabel("Difference in Mean Modules (4m - 2m)")
plt.title("Difference of Mean Modules between 2m and 4m for Each Animal")
plt.tight_layout()
plt.show()
# %%
# Assumptions:
# - mean_modules: shape (n_animals,)
# - mask_groups has the same structure as label_variables:
#   mask_groups[factor_idx][level_idx] is a boolean mask (n_animals,) for that level
# - Each factor has labels like "... 2m" and "... 4m" that pair naturally


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


# %%

# ---- Plot: one panel per factor ----
n_factors = len(label_variables)
fig, axes = plt.subplots(
    1, n_factors, figsize=(6 * n_factors, 8), sharex=True, sharey=True
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
        ax.annotate(bn, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=12)

    # Reference line y=x
    lim_lo = min(np.min(x_mu - x_se), np.min(y_mu - y_se)) if len(x_mu) else 0
    lim_hi = max(np.max(x_mu + x_se), np.max(y_mu + y_se)) if len(x_mu) else 1
    pad = 0.05 * (lim_hi - lim_lo if lim_hi > lim_lo else 1.0)
    lo, hi = lim_lo - pad, lim_hi + pad
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)

    ax.set_title(f"{factors_labels[fi]}", fontsize=18)
    ax.set_xlabel("Mean modules @ 2m", fontsize=14)
    ax.set_ylabel("Mean modules @ 4m", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

plt.suptitle("2m vs 4m mean modules (group-level)", y=1.02)
plt.tight_layout()
plt.show()
# %%

factors_labels = ("oip", "nor", "genotype", "sex")


colors = plt.cm.tab10.colors  # categorical palette

for f_idx, (factor_name, labels, masks) in enumerate(
    zip(factors_labels, label_variables, mask_groups, strict=False)
):
    groups = {}
    for lbl, m in zip(labels, masks, strict=False):
        base, age = split_base_age(lbl)
        if age is None:
            continue
        groups.setdefault(base, {})[age] = m

    plt.figure(figsize=(7, 6))

    for k, (base, ages) in enumerate(groups.items()):
        m2 = ages.get("2m")
        m4 = ages.get("4m")
        if m2 is None or m4 is None:
            continue

        x = mean_modules[m2].astype(float)
        y = mean_modules[m4].astype(float)

        plt.scatter(x, y, label=base, color=colors[k % len(colors)], alpha=0.7, s=60)
        plt.axhline(np.mean(y), c=colors[k % len(colors)])
        plt.axvline(np.mean(x), c=colors[k % len(colors)])

    all_vals = mean_modules
    lo = float(np.nanmin(all_vals)) - 0.25
    hi = float(np.nanmax(all_vals)) + 0.25
    plt.plot([lo, hi], [lo, hi], "--", color="gray", alpha=0.6)

    plt.xlim(2.9, 3.4)
    plt.ylim(2.9, 3.4)

    plt.gca().set_aspect("equal", adjustable="box")

    plt.xlabel("Mean modules @ 2m")
    plt.ylabel("Mean modules @ 4m")
    plt.title(f"Per-animal averages (2m vs 4m) — {factor_name}")
    plt.legend(
        title=factor_name, frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left"
    )
    plt.tight_layout()
    plt.show()


# %%

############################################################################################
# - Cohesion analysis
#############################################################################################
# for the first animal, compute the cohesion timeseries between region 1 and region 2


# Default Mode Network regions


def cohesion_timeseries(communities, region_index=None):

    if region_index == None:
        region_index = np.arange(communities.shape[1])
    cohesion_timeseries = np.zeros(
        (len(region_index), len(region_index), communities.shape[0])
    )
    # cohesion_probability = np.zeros((n_regions, n_regions))
    for reg1 in range(len(region_index)):
        for reg2 in range(reg1 + 1, len(region_index)):
            aux1 = communities[:, region_index[reg1]]
            aux2 = communities[:, region_index[reg2]]
            cohesion_timeseries[reg1, reg2, :] = aux1 - aux2
            # count the number of time windows where the two regions are in the same module
    return cohesion_timeseries


def cohesion_probability(communities):
    # communities: (n_windows, n_regions) of community labels
    d = communities[:, dmn_labels_index]  # (T, D)
    same = (d[:, :, None] == d[:, None, :]).sum(axis=0)  # (D, D)
    tri = np.triu_indices(len(dmn_labels_index), k=1)
    prob = np.zeros((len(dmn_labels_index), len(dmn_labels_index)))
    prob[tri] = same[tri] / d.shape[0]
    return prob


cohesion_probability_all = np.zeros(
    (n_animals, len(dmn_labels_index), len(dmn_labels_index))
)

for animal in range(n_animals):
    cohesion_probability_all[animal] = cohesion_probability(dfc_sorted[animal])


cohesion_timeseries_all = np.zeros((n_animals, n_regions, n_regions, n_windows))
cohesion_timeseries_dmn = np.zeros(
    (n_animals, len(dmn_labels_index), len(dmn_labels_index), n_windows)
)

for animal in range(n_animals):
    cohesion_timeseries_dmn[animal] = cohesion_timeseries(
        dfc_sorted[animal], region_index=dmn_labels_index
    )
    cohesion_timeseries_all[animal] = cohesion_timeseries(dfc_sorted[animal])

# %%
# Suppose we already have:
# - n_regions
# - anat_labels (array of region names, length = n_regions)
# - dmn_labels_index (list of indices for DMN regions)

# All upper-triangle pairs in global space
index_timeseries = np.triu_indices(n_regions, k=1)

# Restrict to DMN pairs
dmn_set = set(dmn_labels_index)
pairs = list(zip(index_timeseries[0], index_timeseries[1], strict=False))
mask = [(i in dmn_set) and (j in dmn_set) for i, j in pairs]
dmn_pairs = (index_timeseries[0][mask], index_timeseries[1][mask])

# Map to labels in dmn
dmn_pairs_labels = [
    (anat_labels_sorted[i], anat_labels_sorted[j])
    for i, j in zip(dmn_pairs[0], dmn_pairs[1], strict=False)
]

# map to labels in anat_labels
all_pairs_labels = [
    (anat_labels_sorted[i], anat_labels_sorted[j])
    for i, j in zip(index_timeseries[0], index_timeseries[1], strict=False)
]

# Example output
# print("Numeric DMN pairs:", list(zip(dmn_pairs[0], dmn_pairs[1]))[:5])
# print("Label DMN pairs:", dmn_pairs_labels[:5])

# for i, val in enumerate(aux_dmn_x):
cohesion_timeseries_all_triu = cohesion_timeseries_all[
    :, index_timeseries[0], index_timeseries[1], :
]
cohesion_timeseries_all_binary = (cohesion_timeseries_all_triu == 0).astype(int)


cohesion_timeseries_dmn_triu = cohesion_timeseries_all[:, dmn_pairs[0], dmn_pairs[1], :]
cohesion_timeseries_dmn_binary = (cohesion_timeseries_dmn_triu == 0).astype(int)

# cohesion_timeseries_all[ind_x, ind_y, :]
# %%
cohesion_timeseries_aux = cohesion_timeseries_dmn_triu
cohesion_timeseries_aux = cohesion_timeseries_all_triu
# plot imshow of cohesion_timeseries_all_triu for one animal
plt.figure(figsize=(10, 8))
plt.imshow(
    cohesion_timeseries_aux[6],
    aspect="auto",
    interpolation="none",
    cmap="Greys",
    vmin=0,
    vmax=1,
)
plt.colorbar()
plt.title(f"Cohesion Timeseries - Animal {0}")
plt.xlabel(r"Time Windows (TW$_{1}$, TW$_{2}$, ..., TW$_{n}$)")
plt.ylabel(r"Links Pairs ($N (N-1)/2$)")
# plt.yticks([])
# plt.yticks(np.arange(len(dmn_pairs_labels)), labels=[f"{i[0]}-{i[1]}" for i in dmn_pairs_labels], fontsize=6)
plt.yticks(
    np.arange(len(all_pairs_labels)),
    labels=[f"{i[0]}-{i[1]}" for i in all_pairs_labels],
    fontsize=2,
)
# plt.xticks(np.arange(n_windows), labels=[f"TW{i+1}" for i in range(n_windows)], rotation=90)
plt.show()
# %%


def symmetrize_mean_cohesion(coh_all: np.ndarray) -> np.ndarray:
    mean_mat = np.mean(coh_all, axis=3)
    return 0.5 * (mean_mat + np.swapaxes(mean_mat, -1, -2))


cohesion_timeseries_all_sym = symmetrize_mean_cohesion(cohesion_timeseries_all)

# %%

q = np.zeros(n_animals)

for i in range(n_animals):
    comm, q[i] = bct.modularity.modularity_louvain_und_sign(
        cohesion_timeseries_all_sym[i]
    )


plt.figure(figsize=(10, 8))
plt.imshow(
    cohesion_timeseries_all_sym[1],
    aspect="auto",
    interpolation="none",
    cmap="Greys",
    vmin=0,
    vmax=1,
)
plt.colorbar()
plt.xticks(
    np.arange(n_regions),
    labels=anat_labels[sort_allegiances[0, 0].astype(int)],
    rotation=90,
)
plt.yticks(np.arange(n_regions), labels=anat_labels[sort_allegiances[0, 0].astype(int)])
plt.title("Cohesion Timeseries Sum - Animal 1")

# %%

# aux_mask = mask_groups[0]
aux_mask = mask_groups[0]
for xx in range(4):
    for i, aux_mask in enumerate(mask_groups[xx]):

        aux_label_variables = label_variables[xx][i]

        plt.figure(figsize=(10, 8))
        plt.clf()
        plt.imshow(
            np.mean(cohesion_timeseries_all_sym[aux_mask], axis=0),
            aspect="auto",
            interpolation="none",
            cmap="Greys",
            vmin=0,
            vmax=1,
        )
        plt.colorbar()
        plt.xticks(
            np.arange(n_regions),
            labels=anat_labels[sort_allegiances[0, 0].astype(int)],
            rotation=90,
        )
        plt.yticks(
            np.arange(n_regions), labels=anat_labels[sort_allegiances[0, 0].astype(int)]
        )
        # plt.colorbar()
        plt.title(
            f"Mean Cohesion Probability - {aux_label_variables} q={np.mean(q[aux_mask]):.3f}"
        )


# %%

for group in range(len(mask_groups)):
    # plt.xlabel("Region 2")
    # plt.ylabel("Region 1")
    # plt.yticks(np.arange(n_regions), labels=anat_labels[sort_allegiances[0, 0].astype(int)])
    plt.show()

cohesion_timeseries_all_sym_wth = cohesion_timeseries_all_sym.copy()

# %%

# plot imshow of cohesion_timeseries_all_binary for one animal
plt.figure(figsize=(10, 8))
plt.title(f"Cohesion Probability - Animal {0}")
plt.imshow(
    cohesion_timeseries_all_binary[0],
    aspect="auto",
    interpolation="none",
    cmap="gray_r",
    vmin=0,
    vmax=1,
)
plt.colorbar()

# %%
# plot imshow of cohesion_timeseries_all_binary for all animals
plt.figure(figsize=(10, 8))
for i in range(n_animals):
    plt.subplot(12, 11, i + 1)
    plt.imshow(
        cohesion_timeseries_all_binary[i],
        aspect="auto",
        interpolation="none",
        cmap="gray_r",
        vmin=0,
        vmax=1,
    )
    plt.xticks([])

    plt.yticks([])
    # plt.colorbar()
    # plt.title(f"Cohesion Probability - Animal {i}")
# plt.subplot(11, 5, 1)
# plt.imshow(cohesion_timeseries_dmn_binary[5], aspect="auto", interpolation="none", cmap='gray_r', vmin=0, vmax=1)
# plt.colorbar()
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
plt.title("Cohesion Probability DMN")
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
aux_time_ratio = (
    np.sum(cohesion_timeseries_dmn_binary, axis=2)
    / cohesion_timeseries_dmn_binary.shape[2]
)  # (n_animals, n_links) or (A, L)

# --- rows (links) ---
link_labels = [f"{a}–{b}" for (a, b) in dmn_pairs_labels]  # length L

# --- data_T: (L, A) ---
data_T = aux_time_ratio.T  # (n_links, n_animals)
# %%
# plot imshow of aux_time_ratio
plt.figure(figsize=(10, 8))
plt.clf()
# for i in range(n_animals):
plt.imshow(
    data_T,
    aspect="auto",
    interpolation="none",
    cmap="viridis",
    vmin=0,
    vmax=1,
)
plt.ylabel("Links Pairs")
plt.yticks(
    np.arange(len(dmn_pairs_labels)),
    labels=[f"{i[0]}-{i[1]}" for i in dmn_pairs_labels],
    fontsize=6,
)
plt.xlabel("Animals")
# plt.xticks([])

# plt.yticks([])

# plt.colorbar()
# plt.title(f"Cohesion Probability - Animal {i}")
# plt.xlabel("Region 2")
# plt.ylabel("Region 1")
# plt.yticks(np.arange(n_regions), labels=anat_labels[sort_allegiances[0, 0].astype(int)])
plt.show()

# %%
n_links_aux = aux_time_ratio.shape[1]
# plot violin plot of aux_time_ratio for each group in mask_groups[3]
for i in range(n_links_aux):
    plt.figure(figsize=(10, 8))
    plt.violinplot(
        (
            aux_time_ratio.T[i, mask_groups[3][0]],
            aux_time_ratio.T[i, mask_groups[3][1]],
            aux_time_ratio.T[i, mask_groups[3][2]],
            aux_time_ratio.T[i, mask_groups[3][3]],
        )
    )

# %%


# ============================================================================
# -------------------------------Wilcoxon code---------------------------------
# ============================================================================
# -------------------- helpers --------------------


def _cols_single_factor_keys_and_data(
    block_title: str,
    factor_idx: int,
    data_T: np.ndarray,
    link_labels: list[str],
    label_variables,
    mask_groups,
    value_fn,
) -> tuple[list[tuple[str, str]], list[np.ndarray]]:
    """
    Build columns for a single factor. Returns:
      keys: list of (Block, ColumnLabel)
      cols: list of 1D arrays (n_links,)
    value_fn(X, Y) is called with shape (n_links, n_pairs) arrays and must return (n_links,)
    """
    F = factor_base_indices(factor_idx, label_variables, mask_groups)
    keys, cols = [], []
    for base, ages in F.items():
        idx2 = ages.get("2m")
        idx4 = ages.get("4m")
        if (
            idx2 is None
            or idx4 is None
            or len(idx2) == 0
            or len(idx4) == 0
            or len(idx2) != len(idx4)
        ):
            continue
        X = data_T[:, idx2]
        Y = data_T[:, idx4]
        v = value_fn(X, Y)
        keys.append((block_title, base))
        cols.append(v)
    return keys, cols


def _cols_two_factors_keys_and_data(
    block_title: str,
    factorA_idx: int,
    factorB_idx: int,
    data_T: np.ndarray,
    link_labels: list[str],
    label_variables,
    mask_groups,
    value_fn,
) -> tuple[list[tuple[str, str]], list[np.ndarray]]:
    """
    Build columns for a pair of factors. ColumnLabel is like "Female·wt".
    """
    A = factor_base_indices(factorA_idx, label_variables, mask_groups)
    B = factor_base_indices(factorB_idx, label_variables, mask_groups)
    keys, cols = [], []
    for a, agesA in A.items():
        idx2A, idx4A = agesA.get("2m"), agesA.get("4m")
        if idx2A is None or idx4A is None:
            continue
        for b, agesB in B.items():
            idx2B, idx4B = agesB.get("2m"), agesB.get("4m")
            if idx2B is None or idx4B is None:
                continue
            keep = np.isin(idx2A, idx2B) & np.isin(idx4A, idx4B)
            idx2, idx4 = idx2A[keep], idx4A[keep]
            if len(idx2) == 0 or len(idx4) == 0 or len(idx2) != len(idx4):
                continue
            X = data_T[:, idx2]
            Y = data_T[:, idx4]
            v = value_fn(X, Y)
            keys.append((block_title, f"{a}·{b}"))
            cols.append(v)
    return keys, cols


def _factor_base_masks(factor_idx, label_variables, mask_groups):
    """
    Returns: dict base -> {'2m': mask, '4m': mask}
    Keeps only bases that have explicit 2m/4m masks.
    """
    bases = {}
    for lbl, m in zip(
        label_variables[factor_idx], mask_groups[factor_idx], strict=False
    ):
        base, age = split_base_age(lbl)
        if age in {"2m", "4m"}:
            bases.setdefault(base, {})[age] = m
    return bases


def wilcoxon_pvals_single_factor(
    data_T: np.ndarray,  # (n_links, n_animals)
    link_labels: list[str],  # len = n_links
    label_variables,
    mask_groups,
    factor_idx: int,  # 0=oip, 1=nor, 2=genotype, 3=sex
    include_empty: bool = False,  # include NaN columns for bases w/o valid pairs
):
    """
    Paired Wilcoxon 2m vs 4m within each base of a single factor.
    Returns DataFrame of raw p-values: columns are bases (e.g., Female, Male).
    """
    F = _factor_base_masks(factor_idx, label_variables, mask_groups)
    col_names, cols = [], []
    for base, ages in F.items():
        m2, m4 = ages.get("2m"), ages.get("4m")
        if (
            (m2 is None)
            or (m4 is None)
            or (m2.sum() == 0)
            or (m4.sum() == 0)
            or (m2.sum() != m4.sum())
        ):
            if include_empty:
                col_names.append(base)
                cols.append([np.nan] * data_T.shape[0])
            continue

        pvals = []
        for i in range(data_T.shape[0]):
            x, y = data_T[i, m2], data_T[i, m4]
            try:
                _, p = wilcoxon(x, y, zero_method="zsplit", alternative="two-sided")
            except ValueError:
                p = 1.0
            pvals.append(p)
        col_names.append(base)
        cols.append(pvals)

    if not cols:
        return pd.DataFrame(np.nan, index=link_labels, columns=[])
    mat = np.column_stack(cols)
    return pd.DataFrame(mat, index=link_labels, columns=col_names)


def wilcoxon_pvals_two_factors(
    data_T: np.ndarray,
    link_labels: list[str],
    label_variables,
    mask_groups,
    factorA_idx: int,
    factorB_idx: int,
    include_empty: bool = False,
):
    """
    Paired Wilcoxon 2m vs 4m within each (FactorA × FactorB) stratum.
    Returns DataFrame of raw p-values: columns are 'A·B' (e.g., Female·wt).
    """
    A = _factor_base_masks(factorA_idx, label_variables, mask_groups)
    B = _factor_base_masks(factorB_idx, label_variables, mask_groups)

    col_names, cols = [], []
    for a in A.keys():
        for b in B.keys():
            m2 = (
                A[a].get("2m") & B[b].get("2m")
                if ("2m" in A[a] and "2m" in B[b])
                else None
            )
            m4 = (
                A[a].get("4m") & B[b].get("4m")
                if ("4m" in A[a] and "4m" in B[b])
                else None
            )
            if (
                (m2 is None)
                or (m4 is None)
                or (m2.sum() == 0)
                or (m4.sum() == 0)
                or (m2.sum() != m4.sum())
            ):
                if include_empty:
                    col_names.append(f"{a}·{b}")
                    cols.append([np.nan] * data_T.shape[0])
                continue

            pvals = []
            for i in range(data_T.shape[0]):
                x, y = data_T[i, m2], data_T[i, m4]
                try:
                    _, p = wilcoxon(x, y, zero_method="zsplit", alternative="two-sided")
                except ValueError:
                    p = 1.0
                pvals.append(p)
            col_names.append(f"{a}·{b}")
            cols.append(pvals)

    if not cols:
        return pd.DataFrame(np.nan, index=link_labels, columns=[])
    mat = np.column_stack(cols)
    return pd.DataFrame(mat, index=link_labels, columns=col_names)


def combine_blocks(blocks: list[tuple[str, pd.DataFrame]]):
    """
    Concatenate multiple p-value DataFrames horizontally.
    Returns combined df and a list of separator positions for plotting.
    """
    dfs, seps, total = [], [], 0
    for title, df in blocks:
        dfs.append(df)
        total += df.shape[1]
        seps.append((total, title))
    combined = pd.concat(dfs, axis=1)

    # prefix column names with block titles
    prefixed = []
    for title, df in blocks:
        prefixed.extend([f"{title}·{col}" for col in df.columns])
    combined.columns = prefixed
    print(combined.columns)
    return combined, seps


# combine_blocks([])


# %%
# -------------------- vectorized Wilcoxon --------------------
def _wilcoxon_rows(X, Y, zero_method="wilcox"):
    """
    Vectorized Wilcoxon along rows if SciPy supports axis; otherwise fallback loop.
    X, Y: (n_links, n_pairs)
    Returns p-values, shape (n_links,)
    """
    try:
        # With SciPy >= 1.9, wilcoxon supports axis argument
        res = wilcoxon(
            X,
            Y,
            zero_method=zero_method,
            alternative="two-sided",
            axis=1,
            method="asymptotic",
        )
        return np.asarray(res.pvalue)

    except TypeError:
        # Fallback: minimal Python loop, still faster because we sliced once
        pvals = np.empty(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            try:
                _, p = wilcoxon(
                    X[i],
                    Y[i],
                    zero_method=zero_method,
                    alternative="two-sided",
                    method="asymptotic",
                )
            except ValueError:
                p = 1.0
            pvals[i] = p
        return pvals


def _cohesion_diff_rows(X, Y):
    """Return mean over pairs of (Y-X)/(Y+X) per link (n_links,)."""
    eps = 1e-9
    return np.mean((Y - X) / np.maximum(Y + X, eps), axis=1)


def _ttest_rows(X, Y):
    """
    Vectorized paired t-test over rows (links).
    X, Y: (n_links, n_pairs)
    Returns p-values, shape (n_links,)
    """
    # scipy.stats.ttest_rel supports axis broadcasting
    _, p = ttest_rel(X, Y, axis=1, nan_policy="propagate", alternative="two-sided")
    p = np.asarray(p, dtype=float)
    # map NaNs (degenerate rows) to 1.0 so they won't appear as significant
    return np.where(np.isnan(p), 1.0, p)


def build_table_from_spec_pvals(
    data_T: np.ndarray,
    link_labels: list[str],
    label_variables,
    mask_groups,
    block_spec=BLOCK_SPEC,
) -> pd.DataFrame:
    """
    Build a p-value table with MultiIndex columns (Block, Column).
    Uses the single source of truth BLOCK_SPEC.
    """
    all_keys, all_cols = [], []
    for item in block_spec:
        if item[1] == "single":
            block, _, fidx = item
            keys, cols = _cols_single_factor_keys_and_data(
                block,
                fidx,
                data_T,
                link_labels,
                label_variables,
                mask_groups,
                value_fn=lambda X, Y: _wilcoxon_rows(X, Y),
            )
        else:
            block, _, fA, fB = item
            keys, cols = _cols_two_factors_keys_and_data(
                block,
                fA,
                fB,
                data_T,
                link_labels,
                label_variables,
                mask_groups,
                value_fn=lambda X, Y: _wilcoxon_rows(X, Y),
            )
        all_keys += keys
        all_cols += cols

    if not all_cols:
        return pd.DataFrame(index=link_labels, columns=pd.MultiIndex.from_tuples([]))

    M = np.column_stack(all_cols)
    columns = pd.MultiIndex.from_tuples(all_keys, names=["Block", "Column"])
    return pd.DataFrame(M, index=link_labels, columns=columns)


def build_table_from_spec_cohesiondiff(
    data_T: np.ndarray,
    link_labels: list[str],
    label_variables,
    mask_groups,
    block_spec=BLOCK_SPEC,
) -> pd.DataFrame:
    """
    Build a cohesion-difference table with the SAME MultiIndex columns (Block, Column).
    """
    all_keys, all_cols = [], []
    for item in block_spec:
        if item[1] == "single":
            block, _, fidx = item
            keys, cols = _cols_single_factor_keys_and_data(
                block,
                fidx,
                data_T,
                link_labels,
                label_variables,
                mask_groups,
                value_fn=lambda X, Y: _cohesion_diff_rows(X, Y),
            )
        else:
            block, _, fA, fB = item
            keys, cols = _cols_two_factors_keys_and_data(
                block,
                fA,
                fB,
                data_T,
                link_labels,
                label_variables,
                mask_groups,
                value_fn=lambda X, Y: _cohesion_diff_rows(X, Y),
            )
        all_keys += keys
        all_cols += cols

    if not all_cols:
        return pd.DataFrame(index=link_labels, columns=pd.MultiIndex.from_tuples([]))

    M = np.column_stack(all_cols)
    columns = pd.MultiIndex.from_tuples(all_keys, names=["Block", "Column"])
    return pd.DataFrame(M, index=link_labels, columns=columns)


def build_table_from_spec_ttest_pvals(
    data_T: np.ndarray,
    link_labels: list[str],
    label_variables,
    mask_groups,
    block_spec=BLOCK_SPEC,
) -> pd.DataFrame:
    """
    Build a paired t-test p-value table with MultiIndex columns (Block, Column).
    Uses the same BLOCK_SPEC so columns match cohesion diff.
    """
    all_keys, all_cols = [], []
    for item in block_spec:
        if item[1] == "single":
            block, _, fidx = item
            keys, cols = _cols_single_factor_keys_and_data(
                block,
                fidx,
                data_T,
                link_labels,
                label_variables,
                mask_groups,
                value_fn=lambda X, Y: _ttest_rows(X, Y),
            )
        else:
            block, _, fA, fB = item
            keys, cols = _cols_two_factors_keys_and_data(
                block,
                fA,
                fB,
                data_T,
                link_labels,
                label_variables,
                mask_groups,
                value_fn=lambda X, Y: _ttest_rows(X, Y),
            )
        all_keys += keys
        all_cols += cols

    if not all_cols:
        return pd.DataFrame(index=link_labels, columns=pd.MultiIndex.from_tuples([]))

    M = np.column_stack(all_cols)
    columns = pd.MultiIndex.from_tuples(all_keys, names=["Block", "Column"])
    return pd.DataFrame(M, index=link_labels, columns=columns)


# -------------------- factor indexing --------------------
def wilcoxon_pvals_single_factor_fast(
    data_T: np.ndarray,  # (n_links, n_animals)
    link_labels: list[str],
    factor_idx: int,  # 0=oip, 1=nor, 2=genotype, 3=sex
    include_empty: bool = False,
    zero_method: str = "wilcox",
) -> pd.DataFrame:
    """
    Paired Wilcoxon 2m vs 4m within each base of a single factor.
    Returns DataFrame of raw p-values: columns are bases (e.g. "oip·2m" vs "oip·4m").
    """
    F = factor_base_indices(
        factor_idx,
        label_variables,
        mask_groups,
    )
    cols = []
    names = []
    n_links = data_T.shape[0]

    for base, ages in F.items():
        idx2, idx4 = ages.get("2m"), ages.get("4m")
        if (
            idx2 is None
            or idx4 is None
            or len(idx2) == 0
            or len(idx4) == 0
            or len(idx2) != len(idx4)
        ):
            # Incompatible age groups
            if include_empty:  # keep column of NaNs
                cols.append(np.full(n_links, np.nan))
                names.append(base)
            continue

        # Slice once: (n_links, n_pairs)
        X = data_T[:, idx2]
        Y = data_T[:, idx4]
        p = _wilcoxon_rows(X, Y, zero_method=zero_method)
        cols.append(p)
        names.append(base)  # e.g., "oip·2m" vs "oip·4m"

    # No valid columns found (e.g., bad factor_idx)
    if not cols:
        return pd.DataFrame(np.nan, index=link_labels, columns=[])

    # Stack columns into a matrix
    mat = np.column_stack(cols)  # (n_links, n_cols)
    return pd.DataFrame(mat, index=link_labels, columns=names)


def wilcoxon_pvals_two_factors_fast(
    data_T: np.ndarray,
    link_labels: list[str],
    factorA_idx: int,
    factorB_idx: int,
    include_empty: bool = False,
    zero_method: str = "wilcox",
) -> pd.DataFrame:
    """Paired Wilcoxon 2m vs 4m within each base of two factors.
    Returns DataFrame of raw p-values: columns are bases (e.g. "oip·2m" vs "oip·4m").
    """

    A = factor_base_indices(factorA_idx, label_variables, mask_groups)
    B = factor_base_indices(factorB_idx, label_variables, mask_groups)

    cols = []
    names = []
    n_links = data_T.shape[0]

    for a, agesA in A.items():
        idx2A, idx4A = agesA.get("2m"), agesA.get("4m")
        if idx2A is None or idx4A is None:
            continue
        for b, agesB in B.items():
            idx2B, idx4B = agesB.get("2m"), agesB.get("4m")
            if idx2B is None or idx4B is None:
                continue

            # intersect paired indices (assumes same ordering across masks)
            # idx2 = np.intersect1d(idx2A, idx2B, assume_unique=False)
            # idx4 = np.intersect1d(idx4A, idx4B, assume_unique=False)

            keep = np.isin(idx2A, idx2B) & np.isin(idx4A, idx4B)
            idx2 = idx2A[keep]
            idx4 = idx4A[keep]

            if len(idx2) == 0 or len(idx4) == 0 or len(idx2) != len(idx4):
                if include_empty:
                    cols.append(np.full(n_links, np.nan))
                    names.append(f"{a}·{b}")
                continue

            X = data_T[:, idx2]
            Y = data_T[:, idx4]
            p = _wilcoxon_rows(X, Y, zero_method=zero_method)
            cols.append(p)
            names.append(f"{a}·{b}")

    if not cols:
        return pd.DataFrame(np.nan, index=link_labels, columns=[])

    mat = np.column_stack(cols)
    return pd.DataFrame(mat, index=link_labels, columns=names)


# %%
# -------------------- build everything --------------------


# %%
def single_factor_fast(
    data_T: np.ndarray,  # (n_links, n_animals)
    link_labels: list[str],
    factor_idx: int,  # 0=oip, 1=nor, 2=genotype, 3=sex
    include_empty: bool = False,
    zero_method: str = "wilcox",
) -> pd.DataFrame:
    """Cohesion difference (mean((Y-X)/(Y+X))) within each base of a single factor."""
    F = factor_base_indices(
        factor_idx,
        label_variables,
        mask_groups,
    )
    cols = []
    names = []
    n_links = data_T.shape[0]
    print(F)

    for base, ages in F.items():
        idx2 = ages.get("2m")
        idx4 = ages.get("4m")
        if (
            idx2 is None
            or idx4 is None
            or len(idx2) == 0
            or len(idx4) == 0
            or len(idx2) != len(idx4)
        ):
            if include_empty:
                cols.append(np.full(n_links, np.nan))
                names.append(base)
            continue

        # Slice once: (n_links, n_pairs)
        X = data_T[:, idx2]
        Y = data_T[:, idx4]
        # return Y-X
        eps = 1e-9
        cols.append(np.mean((Y - X) / np.maximum(Y + X, eps), axis=1))
        names.append(base)

    if not cols:
        return pd.DataFrame(np.nan, index=link_labels, columns=[])

    mat = np.column_stack(cols)  # (n_links, n_cols)
    return pd.DataFrame(mat, index=link_labels, columns=names)


def two_factors_fast(
    data_T: np.ndarray,
    link_labels: list[str],
    factorA_idx: int,
    factorB_idx: int,
    include_empty: bool = False,
    zero_method: str = "wilcox",
) -> pd.DataFrame:
    """Cohesion difference (mean((Y-X)/(Y+X))) within each (FactorA × FactorB) stratum."""

    A = factor_base_indices(
        factorA_idx,
        label_variables,
        mask_groups,
    )
    B = factor_base_indices(
        factorB_idx,
        label_variables,
        mask_groups,
    )

    cols = []
    names = []
    n_links = data_T.shape[0]

    for a, agesA in A.items():
        idx2A, idx4A = agesA.get("2m"), agesA.get("4m")
        if idx2A is None or idx4A is None:
            continue
        for b, agesB in B.items():
            idx2B, idx4B = agesB.get("2m"), agesB.get("4m")
            if idx2B is None or idx4B is None:
                continue

            # intersect paired indices (assumes same ordering across masks)
            # idx2 = np.intersect1d(idx2A, idx2B, assume_unique=False)
            # idx4 = np.intersect1d(idx4A, idx4B, assume_unique=False)

            keep = np.isin(idx2A, idx2B) & np.isin(idx4A, idx4B)
            idx2 = idx2A[keep]
            idx4 = idx4A[keep]

            if len(idx2) == 0 or len(idx4) == 0 or len(idx2) != len(idx4):
                if include_empty:
                    cols.append(np.full(n_links, np.nan))
                    names.append(f"{a}·{b}")
                continue

            X = data_T[:, idx2]
            Y = data_T[:, idx4]
            # return Y-X
            eps = 1e-9
            cols.append(np.mean((Y - X) / np.maximum(Y + X, eps), axis=1))
            names.append(f"{a}·{b}")

    if not cols:
        return pd.DataFrame(np.nan, index=link_labels, columns=[])

    mat = np.column_stack(cols)
    return pd.DataFrame(mat, index=link_labels, columns=names)


link_labels = [f"{a}–{b}" for (a, b) in dmn_pairs_labels]
# If you want *all* pairs instead:
# link_labels = [f"{a}–{b}" for (a, b) in all_pairs_labels]


# # Combine all blocks horizontally: singles + combos
# combined_df, separators = combine_blocks([
#     block_sex, block_geno, block_oip,   # single factors
#     block_sex_geno, block_sex_oip,  # combinations
#     block_geno_oip
# ])


cohesion_diff_oip = single_factor_fast(aux_time_ratio.T, link_labels, factor_idx=0)
# cohesion_diff_nor  = single_factor_fast(aux_time_ratio.T, link_labels, factor_idx=1)
cohesion_diff_geno = single_factor_fast(aux_time_ratio.T, link_labels, factor_idx=2)
cohesion_diff_sex = single_factor_fast(aux_time_ratio.T, link_labels, factor_idx=3)

cohesion_diff_sex_geno = two_factors_fast(
    aux_time_ratio.T, link_labels, factorA_idx=3, factorB_idx=2
)
cohesion_diff_sex_oip = two_factors_fast(
    aux_time_ratio.T, link_labels, factorA_idx=3, factorB_idx=0
)
# cohesion_diff_sex_nor  = two_factors_fast(aux_time_ratio.T, link_labels, factorA_idx=3, factorB_idx=1)
cohesion_diff_geno_oip = two_factors_fast(
    aux_time_ratio.T, link_labels, factorA_idx=2, factorB_idx=0
)

# %%
# ============================================================================
# --- Build Wilcoxon p-values & cohesion-diff tables from the same spec ---
pvals_df = build_table_from_spec_pvals(
    data_T, link_labels, label_variables, mask_groups, block_spec=BLOCK_SPEC
)
cohdiff_df = build_table_from_spec_cohesiondiff(
    data_T, link_labels, label_variables, mask_groups, block_spec=BLOCK_SPEC
)

# --- Plots ---
plot_sig_pvals_multi(pvals_df, alpha=0.05, title="Wilcoxon 2m vs 4m — significant only")
plot_weighted_multi(
    pvals_df,
    cohdiff_df,
    alpha=0.05,
    title="Wilcoxon — (1 - p) × mean cohesion difference",
    vmin=-0.1,
    vmax=0.1,
)
# %%
# --- Build t-test p-values from the same spec (columns align with cohdiff_df) ---
ttest_pvals_df = build_table_from_spec_ttest_pvals(
    data_T, link_labels, label_variables, mask_groups, block_spec=BLOCK_SPEC
)

# Optional quick checks
print("Wilcoxon & diff aligned:", tuple(pvals_df.columns) == tuple(cohdiff_df.columns))
print(
    "t-test & diff aligned:", tuple(ttest_pvals_df.columns) == tuple(cohdiff_df.columns)
)

# Separators from MultiIndex (same for all since same spec)
seps = separators_from_multiindex(ttest_pvals_df.columns)

# --- Plot t-test significance only ---
plot_sig_pvals_multi(
    ttest_pvals_df, alpha=0.05, title="Paired t-test 2m vs 4m — significant only"
)

# --- Plot t-test: significant cells colored by effect (cohesion diff) ---
plot_weighted_multi(
    ttest_pvals_df,
    cohdiff_df,
    alpha=0.05,
    title="t-test — (1 - p) × mean cohesion difference",
    vmin=-0.1,
    vmax=0.1,
)


# %%
group = mask_groups[0][0]
cohesion_probability_all[group]
# %%
# plot cohesion_probability_mean_group

for xx, label in zip(mask_groups, label_variables, strict=False):
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
    plt.subplot(4, 2, 1 + i)
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
# ------------Cohesion time series-------------------------------


plt.figure(figsize=(17, 8))
index_timeseries = np.triu_indices(len(dmn_labels_index), k=1)
plt.plot(cohesion_timeseries_all[0, 0, 2, :])
plt.xlim(-1, 50)
plt.figure(figsize=(10, 8))
index_timeseries = np.triu_indices(len(dmn_labels_index), k=1)
for idx, (ii, jj) in enumerate(
    zip(index_timeseries[0], index_timeseries[1], strict=False)
):
    plt.subplot(11, 10, idx + 1)
    plt.plot(cohesion_timeseries_all[0, ii, jj, :])
# %%


# ============================================================================
# ------------------------Burst of cohesion--------------------------
# ============================================================================


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


# %%
index_timeseries = np.triu_indices(n_regions, k=1)
index_timeseries_dmn = np.triu_indices(len(dmn_labels_index), k=1)
cohesion_timeseries_all_triu = cohesion_timeseries_all[
    :, index_timeseries[0], index_timeseries[1], :
]
cohesion_timeseries_dmn_triu = cohesion_timeseries_dmn[
    :, index_timeseries_dmn[0], index_timeseries_dmn[1], :
]
cohesion_timeseries_all_binary = (cohesion_timeseries_all_triu == 0).astype(int)
cohesion_timeseries_dmn_binary = (cohesion_timeseries_dmn_triu == 0).astype(int)

cohesion_timeseries_all_binary = np.transpose(cohesion_timeseries_all_binary, (0, 2, 1))
cohesion_timeseries_dmn_binary = np.transpose(cohesion_timeseries_dmn_binary, (0, 2, 1))
# burst_cohesion = extract_link_activations(cohesion_timeseries_all_binary)

# duration = burst_cohesion[0][0][0]["duration"]

# %%


def extract_link_activations_df(
    binary_fc_data: np.ndarray, min_duration: int = 1
) -> pd.DataFrame:

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
    on_idx = np.argwhere(d == 1)
    off_idx = np.argwhere(d == -1)

    # Build DataFrames; use a group key to pair onsets/offsets per (animal, link)
    on = pd.DataFrame(on_idx, columns=["animal", "time", "link"])
    off = pd.DataFrame(off_idx, columns=["animal", "time", "link"])

    on["gid"] = on["animal"] * L + on["link"]
    off["gid"] = off["animal"] * L + off["link"]

    # Order by group then time; assign a within-group index (0,1,2,...) to pair events
    on = on.sort_values(["gid", "time"]).reset_index(drop=True)
    off = off.sort_values(["gid", "time"]).reset_index(drop=True)

    on["idx"] = on.groupby("gid").cumcount()
    off["idx"] = off.groupby("gid").cumcount()

    # Merge on (gid, idx) to align each onset with its corresponding offset
    events = on.merge(off, on=["gid", "idx"], suffixes=("_on", "_off"))

    # Sanity: animals/links must match after merge
    # (They do, but keep columns from the onset side)
    events = events.rename(
        columns={
            "animal_on": "animal",
            "link_on": "link",
            "time_on": "onset",
            "time_off": "offset",
        }
    )[["animal", "link", "onset", "offset"]]

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
            {
                "onset": int(row.onset),
                "offset": int(row.offset),
                "duration": int(row.duration),
            }
        )
    return nested


# Your binary array: (n_animals, time_points, n_links)
# cohesion_timeseries_all_binary already has that shape if you computed the upper triangle per time.
# events_df = extract_link_activations_df(cohesion_timeseries_all_binary, min_duration=2)
events_df = extract_link_activations_df(cohesion_timeseries_dmn_binary, min_duration=2)

# If you still want the old structure:
all_events = events_df_to_nested(
    events_df,
    n_animals=cohesion_timeseries_dmn_binary.shape[0],
    n_links=cohesion_timeseries_dmn_binary.shape[2],
)

# Equivalent to your old example of grabbing the first duration:
duration = all_events[0][0][0]["duration"]

# Or, directly from the DataFrame (e.g., first event of animal 0, link 0):
duration_df = events_df.query("animal == 0 and link == 1").sort_values("onset")


# %%
def mean_duration_matrix(
    events_df: pd.DataFrame, n_animals: int, n_links: int, fill=0.0
):
    """
    Returns a (n_animals, n_links) array with the mean duration of bursts.
    Links with no bursts get `fill` (default 0.0).
    """
    m = events_df.groupby(["animal", "link"])["duration"].mean().unstack("link")
    m = m.reindex(index=range(n_animals), columns=range(n_links))  # ensure full grid
    return m.fillna(fill).to_numpy()


def std_duration_matrix(
    events_df: pd.DataFrame, n_animals: int, n_links: int, fill=0.0
):
    """
    Returns a (n_animals, n_links) array with the standard deviation of burst durations.
    Links with no bursts get `fill` (default 0.0).
    """
    m = events_df.groupby(["animal", "link"])["duration"].std().unstack("link")
    m = m.reindex(index=range(n_animals), columns=range(n_links))  # ensure full grid
    return m.fillna(fill).to_numpy()


mean_dur = mean_duration_matrix(
    events_df,
    n_animals=cohesion_timeseries_dmn_binary.shape[0],
    n_links=cohesion_timeseries_dmn_binary.shape[2],
)
std_dur = std_duration_matrix(
    events_df,
    n_animals=cohesion_timeseries_dmn_binary.shape[0],
    n_links=cohesion_timeseries_dmn_binary.shape[2],
)


# Burstiness coefficient
burstiness = (std_dur - mean_dur) / (std_dur + mean_dur)
burstiness[mean_dur == 0] = 0  # avoid division by zero

# %%


plt.figure(figsize=(17, 8))
plt.imshow(
    burstiness, interpolation="none", aspect="auto", cmap=cm.cm.balance, vmin=-1, vmax=1
)
plt.colorbar()


# %%

links_label = []
for xx in index_timeseries_dmn[0]:
    for yy in index_timeseries_dmn[1]:
        links_label.append(
            f"{anat_labels_sorted[dmn_labels_index[xx]]}-{anat_labels_sorted[dmn_labels_index[yy]]}"
        )

mask = mask_groups[2][2]  # 4m XY
label = label_variables[2][2]
plt.figure(figsize=(17, 8))
plt.imshow(
    burstiness[mask],
    interpolation="none",
    aspect="auto",
    cmap=cm.cm.balance,
    vmin=-1,
    vmax=1,
)
plt.ylabel("animals")
# plt.xticks(np.arange(burstiness.shape[1]), labels=links_label, rotation=90)
plt.colorbar()
# %%
per_animal_mean = np.nanmean(np.where(burstiness == 0, np.nan, burstiness), axis=1)


for i in range(4):
    for label, group in zip(label_variables[i], mask_groups[i], strict=False):
        print(f"{label} burst mean duration {np.mean(per_animal_mean[group])}")

# %%
