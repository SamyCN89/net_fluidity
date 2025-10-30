# %%

from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple, List

from scripts.dfc.dfc_compute import DATASET_DEFAULTS, _canonical_dataset
from shared_code.fun_loaddata import load_timeseries_bundle
from shared_code.fun_paths import get_paths

# from scripts import speed
from shared_code.fun_utils import load_cognitive_data, set_figure_params

# from src.plots_utils import per_animal_summary

# %%
# timecourse_folder = "Timecourses_updated_03052024"

# %%

def combo_color(genotype: str, treatment: str) -> str:
    """Consistent color mapping for genotype+treatment."""
    key = (genotype, treatment)
    table = {
        ("WT", "VEH"): "C0",
        ("WT", "LCTB92"): "C1",
        ("Dp1Yey", "VEH"): "C2",
        ("Dp1Yey", "LCTB92"): "C3",
    }
    return table.get(key, "gray")


def combo_label(genotype: str, treatment: str) -> str:
    return f"{genotype}_{treatment}"



# %%
# -------- speed loading --------
# Speed loading: speed for each time window size

# -------------------------------
# ====== LOAD LOGIC =====
# -------------------------------

def load_speed_stack(
    paths_speed_root: Path, time_windows_range: Sequence[int], template: str
) -> list[np.ndarray]:
    """
    Load speed arrays for each window size.
    Returns a list S where S[j] is an array of shape (n_animals,) of object arrays,
    each entry S[j][i] is 1D array of samples for animal i at window j.
    """
    speeds: list[np.ndarray] = []
    for w in time_windows_range:
        filepath = paths_speed_root / template.format(w=w)
        a = np.load(filepath, allow_pickle=True)
        s = a["speeds"]  # iterable of arrays per animal
        # s_flat = np.array([x.ravel() for x in s], dtype=object)
        s_flat = [np.asarray(x, dtype=float).ravel() for x in s]
        speeds.append(s_flat)
    return speeds


def robust_percentiles(x: np.ndarray, qs=(1, 5, 95, 99)) -> dict[int, float]:
    if x.size == 0 or not np.isfinite(x).any():
        return {q: np.nan for q in qs}
    x = x[np.isfinite(x)]
    ps = np.percentile(x, qs)
    return {int(q): float(p) for q, p in zip(qs, ps, strict=False)}


# -------------------------------
# ====== WINDOW SPLIT LOGIC =====
# -------------------------------


def count_samples_per_window(speeds: list[np.ndarray]) -> np.ndarray:
    """
    For each window j, return the total number of samples across animals.
    """
    return np.array([sum(len(x) for x in speed) for speed in speeds], dtype=int)


def cdf_split_indices(speeds: list[np.ndarray]) -> tuple[int, int, int]:
    """
    Compute indices for 1/3, 1/2, 2/3 thresholds of cumulative samples.
    """
    counts = count_samples_per_window(speeds)
    cdf = (
        np.cumsum(counts) / counts.sum()
        if counts.sum() > 0
        else np.zeros_like(counts, dtype=float)
    )
    # robust indices
    i_third = int(np.searchsorted(cdf, 1.0 / 3.0))
    i_half = int(np.searchsorted(cdf, 0.5))
    i_two_third = int(np.searchsorted(cdf, 2.0 / 3.0))
    # avoid empty slices
    i_third = max(1, i_third)
    i_half = max(1, i_half)
    i_two_third = max(i_third + 1, i_two_third)
    return i_third, i_half, i_two_third


def select_windows(
    pool_split: str, n_windows: int, i_third: int, i_half: int, i_two_third: int
) -> dict[str, range]:
    """
    Returns dict of ranges for 'short' | 'mid' | 'long' depending on split.
    """
    if pool_split == "all":
        return {"all": range(0, n_windows)}
    if pool_split == "half":
        return {
            "short": range(0, i_half),
            "long": range(i_half, n_windows),
        }
    # 'third'
    return {
        "short": range(0, i_third),
        "mid": range(i_third, i_two_third),
        "long": range(i_two_third, n_windows),
    }


def flatten_windows(speeds, start, end):
    """Flatten all animals' samples across windows [start:end)."""
    # speeds[start:end] is a list of length (end-start), each element = list over animals
    arrays = [
        np.asarray(s, dtype=float).ravel() for speed in speeds[start:end] for s in speed
    ]
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=float)


def global_min_max(arrs: Iterable[np.ndarray]) -> tuple[float, float]:
    """Compute global min and max across all arrays."""
    # Compute global min and max of the arrays
    vals_min = [np.nanmin(a) for a in arrs if a.size]
    vals_max = [np.nanmax(a) for a in arrs if a.size]
    vmin = min(vals_min) if vals_min else 0.0
    vmax = max(vals_max) if vals_max else 1.0
    if vmin == vmax:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)

# -------------------------------
# ====== HIST LOGIC =====
# -------------------------------

def safe_hist(
    x: np.ndarray, bins: int, rng: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Count histogram (density=False). Normalize later as desired."""
    if x.size == 0:
        edges = np.linspace(rng[0], rng[1], bins + 1)
        return np.zeros(bins), edges
    h, edges = np.histogram(x, bins=bins, range=rng, density=False)
    return h, edges

def hist_prob(x, bins, rng):
    h, e = np.histogram(x, bins=bins, range=rng, density=False)
    s = h.sum()
    return (h / s if s > 0 else np.zeros_like(h)), e


def normalize_counts_to_prob(h: np.ndarray) -> np.ndarray:
    """Make the sum of bin heights = 1 (probabilities per bin)."""
    s = h.sum()
    return (h / s) if s > 0 else np.zeros_like(h, dtype=float)


def build_per_animal_normalized_hists(
    speeds: list[np.ndarray],
    selected_windows: range,
    bins: int,
    hist_range: tuple[float, float],
) -> np.ndarray:
    """
    Returns array of shape (n_animals, bins) where each row is an animal's
    histogram normalized to sum=1 over the selected windows.
    """
    n_animals = len(speeds[0])
    H = np.zeros((n_animals, bins), dtype=float)
    for i in range(n_animals):
        flat_i = (
            np.concatenate([speeds[j][i].ravel() for j in selected_windows])
            if selected_windows
            else np.array([], dtype=float)
        )
        h_i, _ = safe_hist(flat_i, bins, hist_range)
        H[i] = normalize_counts_to_prob(h_i)
    return H


def flatten_group_animals_over_windows(
    animal_speeds: list[list[np.ndarray]], indices: Sequence[int], w_range: range
) -> np.ndarray:
    """Flatten selected animals over a window range into one 1D array."""
    arrays = []
    for i in indices:
        arrays.extend([animal_speeds[i][j] for j in w_range])
    flat = (
        np.concatenate([a.ravel() for a in arrays])
        if arrays
        else np.array([], dtype=float)
    )
    return flat

#%%
# -------------------------------
# ====== PLOT LOGIC =====
# -------------------------------

def plot_group_median_vs_window(
    ax: plt.Axes,
    time_windows_range: Sequence[int],
    group_genotype_treatment: dict[tuple[str, str], Sequence[int]],
    speeds: list[np.ndarray],
) -> None:
    """Group mean of per-animal means vs window."""
    for (genotype, treatment), indices in group_genotype_treatment.items():
        y = []
        for j in range(len(time_windows_range)):
            per_animal_medians = [float(np.median(speeds[j][i])) for i in indices]
            y.append(
                float(np.mean(per_animal_medians)) if per_animal_medians else np.nan
            )
        ax.plot(time_windows_range, y, ".-", label=combo_label(genotype, treatment))
    ax.set_xlabel("Time Window Size")
    ax.set_ylabel("Mean dFC Speed")
    ax.set_title("dFC Speed vs Window per Genotype–Treatment")
    ax.legend()


def plot_group_histograms(
    ax: plt.Axes,
    centers: np.ndarray,
    group_hists: dict[tuple[str, str], np.ndarray],
    title: str,
    ylog: bool = False,
) -> None:
    for (genotype, treatment), hist in group_hists.items():
        ax.plot(centers, hist, lw=1, alpha=0.7, label=combo_label(genotype, treatment))
    ax.set_title(title)
    ax.set_xlabel("Speed")
    ax.set_ylabel("Probability per bin")
    if ylog:
        ax.set_yscale("log")
    ax.legend()


def pooled_group_histogram(
    animal_speeds: list[list[np.ndarray]],
    group_indices: Sequence[int],
    w_range: range,
    bins: int,
    hist_range: tuple[float, float],
) -> np.ndarray:
    """
    Pool all samples from animals in the group over selected windows,
    then normalize to probability per bin (sum=1).
    """
    flat = flatten_group_animals_over_windows(animal_speeds, group_indices, w_range)
    h, _ = safe_hist(flat, bins, hist_range)
    return normalize_counts_to_prob(h)


# %%
# --------- Load data ---------
# dataset2 = _canonical_dataset("ines")          # honours aliases
# cfg2 = DATASET_DEFAULTS[dataset2]
# Get paths for data loading ines_abdullah dataset
save_fig = set_figure_params(False)
dataset = _canonical_dataset("julien")  # honours aliases
cfg = DATASET_DEFAULTS[dataset]
paths = get_paths(
    dataset_name=dataset,
    timecourse_folder=cfg["timecourse_folder"],
    cognitive_data_file=cfg["cognitive_data_file"],
    anat_labels_file=cfg["anat_labels_file"],
)
speed_root = Path(paths["speed"])

# Load timeseries bundle and grouping data
bundle = load_timeseries_bundle(
    paths["preprocessed"] / "ts_and_meta_2m4m.npz",
    # paths["preprocessed"] / "grouping_data_oip.pkl",
)

# Extract relevant data from bundle
n_animals = bundle.n_animals
total_tr = bundle.total_tr
anat_labels = bundle.anat_labels
regions = bundle.n_regions
# Create masks for each region group
mask_groups = bundle.mask_groups
label_variables = bundle.label_variables
# %%
# # Load cognitive data
cog_data = load_cognitive_data(
    paths["preprocessed"]
    / f"cog_data_filtered_animals_{n_animals}_regions_{regions}_tr_{total_tr}.csv"
)

# Define grouping based on cognitive data
group_treatment = cog_data.groupby("treatment").groups
group_genotype = cog_data.groupby("genotype").groups
group_genotype_treatment = cog_data.groupby(["genotype", "treatment"]).groups

# Load speed data
time_windows_range = np.arange(5, 100, 1)
speeds = load_speed_stack(
    paths["speed"],
    time_windows_range,
    # "dmn_within/nregs-6/speed_win{w}_lag1_tau4_animals_48_regions_37.npz",
    "all/all/speed_win{w}_lag1_tau4_animals_48_regions_37.npz",
)

# Reshape speeds for easier access

# animal_speeds = []
# for i in range(n_animals):
#     animal_s = [speeds[j][i] for j in range(len(time_windows_range))]
#     animal_speeds.append(animal_s)

n_windows = len(speeds)
n_animals = len(speeds[0])  # derive from data, not metadata
animal_speeds = [[speeds[j][i] for j in range(n_windows)] for i in range(n_animals)]
# Precompute mean speed for each animal at each window
per_window_animal_means = [
    np.array([float(np.mean(speeds[j][i])) for i in range(n_animals)], dtype=float)
    for j in range(n_windows)
]

# %%
# -------- Pooled speed distribution per group_genotype_treatment and window range --------
pool_split = "third"  # or 'half'
# # -------- Pooled speed distribution per equal windows range--------
counts = count_samples_per_window(speeds)
pooled_speeds_cdf = np.cumsum(counts) / np.sum(counts)
indice_third, indice_half, indice_two_third = cdf_split_indices(speeds)
ranges = select_windows(
    pool_split, len(time_windows_range), indice_third, indice_half, indice_two_third
)
all_speeds_flat = flatten_windows(speeds, 0, len(speeds))
if pool_split == "half":
    short_speeds_flat = flatten_windows(speeds, 0, indice_half)
    long_speeds_flat = flatten_windows(speeds, indice_half, len(speeds))
elif pool_split == "third":
    short_speeds_flat = flatten_windows(speeds, 0, indice_third)
    mid_speeds_flat = flatten_windows(speeds, indice_third, indice_two_third)
    long_speeds_flat = flatten_windows(speeds, indice_two_third, len(speeds))

# %%
# --------- Histograms ---------
bins_hist = 200
all_speeds_min, all_speeds_max = global_min_max([all_speeds_flat])
edges = np.linspace(all_speeds_min, all_speeds_max, bins_hist + 1)
centers = 0.5 * (edges[:-1] + edges[1:])


# %%
for seg_name, w_range in ranges.items():
    print(
        f"Building per-animal histograms for segment '{seg_name}' with windows {list(w_range)}"
    )
    H_per_animal = build_per_animal_normalized_hists(
        speeds, w_range, bins_hist, (all_speeds_min, all_speeds_max)
    )
    group_means: dict[tuple[str, str], np.ndarray] = {}
    for gt, idxs in group_genotype_treatment.items():
        print(f"Processing group: {gt} with indices: {idxs}")
        mean_hist = (
            np.mean(H_per_animal[idxs], axis=0) if len(idxs) else np.zeros(bins_hist)
        )
        group_means[gt] = mean_hist

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_group_histograms(
        ax[0], centers, group_means, f"Per-animal mean hists ({seg_name})", ylog=False
    )
    plot_group_histograms(
        ax[1],
        centers,
        group_means,
        f"Per-animal mean hists (log, {seg_name})",
        ylog=True,
    )
    for a in ax:
        a.set_ylim(bottom=max(1e-5, a.get_ylim()[0]))


for seg_name, w_range in ranges.items():
    pooled_group: dict[tuple[str, str], np.ndarray] = {}
    for gt, idxs in group_genotype_treatment.items():
        pooled_group[gt] = pooled_group_histogram(
            animal_speeds, idxs, w_range, bins_hist, (all_speeds_min, all_speeds_max)
        )

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_group_histograms(
        ax[0], centers, pooled_group, f"Pooled hist ({seg_name})", ylog=False
    )
    plot_group_histograms(
        ax[1], centers, pooled_group, f"Pooled hist (log, {seg_name})", ylog=True
    )
    for a in ax:
        a.set_ylim(bottom=max(1e-5, a.get_ylim()[0]))

plt.show()

# %%

all_speeds_hist, bin_edge = hist_prob(
    all_speeds_flat, bins_hist, (all_speeds_min, all_speeds_max)
)

if pool_split == "half":
    short_speeds_hist, _ = hist_prob(
        short_speeds_flat, bins_hist, (all_speeds_min, all_speeds_max)
    )
    long_speeds_hist, _ = hist_prob(
        long_speeds_flat, bins_hist, (all_speeds_min, all_speeds_max)
    )
elif pool_split == "third":
    short_speeds_hist, _ = hist_prob(
        short_speeds_flat, bins_hist, (all_speeds_min, all_speeds_max)
    )
    mid_speeds_hist, _ = hist_prob(
        mid_speeds_flat, bins_hist, (all_speeds_min, all_speeds_max)
    )
    long_speeds_hist, _ = hist_prob(
        long_speeds_flat, bins_hist, (all_speeds_min, all_speeds_max)
    )


# %%
# Group speeds by label
all_speeds_grp_hist = {}
# for label, indices in group_genotype_treatment.items():
for label, indices in group_genotype_treatment.items():
    print(f"Processing group: {label} with indices: {indices}")
    group_speeds = []
    aux_animal_group = [animal_speeds[idx] for idx in indices]
    for animal_s in aux_animal_group:
        # print(len(animal_s[0]))
        # print(f"  Animal speeds shape: {[animal_s[j].shape for j in range(len(time_windows_range))]}")
        group_vals = np.concatenate(
            [animal_s[j] for j in range(len(time_windows_range))]
        )
        group_speeds.append(group_vals)
    group_speeds_flat = np.concatenate(group_speeds)
    group_hist, _ = np.histogram(
        group_speeds_flat,
        bins=bins_hist,
        range=(all_speeds_min, all_speeds_max),
        density=False,
    )
    all_speeds_grp_hist[label] = group_hist  # / np.sum(group_hist)
    # all_speeds_grp_hist[label] = group_hist


# %%
# Plotting mean speeds vs window size for each animal
plt.figure(figsize=(8, 6))
for i in range(n_animals):
    animal_speed = [speeds[j][i] for j in range(len(time_windows_range))]
    # mean_speeds = [np.mean(animal_speed[j]) for j in range(len(time_windows_range))]
    mean_speeds = [per_window_animal_means[j][i] for j in range(len(time_windows_range))]

    # color based on genotype and treatment of the animal
    # group_genotype = cog_data.groupby('genotype').groups
    # group_treatment = cog_data.groupby('treatment').groups

    # find genotype label by checking which list i belongs to
    genotype = next(
        (g for g, idx_list in group_genotype.items() if i in idx_list), "Unknown"
    )

    # optionally, do the same for treatment if you have group_treatment defined
    treatment = next(
        (t for t, idx_list in group_treatment.items() if i in idx_list), "Unknown"
    )

    # assign colors based on genotype
    color = (
        "C0"
        if genotype == "WT" and treatment == "VEH"
        else (
            "C1"
            if genotype == "WT" and treatment == "LCTB92"
            else (
                "C2"
                if genotype == "Dp1Yey" and treatment == "VEH"
                else "C3" if genotype == "Dp1Yey" and treatment == "LCTB92" else "gray"
            )
        )
    )

    plt.plot(time_windows_range, mean_speeds, color=color, alpha=0.3)
    # print(i, np.shape(mean_speeds))
plt.xlabel("Time Window Size")
plt.ylabel("Mean dFC Speed")
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color="C0", lw=2, label="WT_VEH"),
    Line2D([0], [0], color="C1", lw=2, label="WT_LCTB92"),
    Line2D([0], [0], color="C2", lw=2, label="Dp1Yey_VEH"),
    Line2D([0], [0], color="C3", lw=2, label="Dp1Yey_LCTB92"),
]
plt.legend(handles=legend_elements, loc="upper right")

plt.title("dFC Speed vs Time Window Size for Each Animal")

# %%
# --------  Plotting mean speeds vs window size per group_genotype_treatment  --------
# B) group mean vs window
fig2, ax2 = plt.subplots(figsize=(8, 6))
plot_group_median_vs_window(ax2, time_windows_range, group_genotype_treatment, speeds)

# %%
# Plotting in a subplot (mean speeds, percentile 1, 5, 95 and 99 of speed) vs window size for each genotype and treatment group
# Create a new figure for the subplots
alpha_aux = 0.5
# plt.figure(figsize=(10, 8))

# Plot percentiles vs window size (1,5,median,95,99) per group across 5 subplots
fig, axs = plt.subplots(2, 3, figsize=(11, 8), sharex=True)
axes = axs.ravel()
titles = ["1st pct", "5th pct", "Median", "95th pct", "99th pct"]
stat_keys = [1, 5, "median", 95, 99]

# Initialize storage
series_per_stat: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
for (genotype, treatment), indices in group_genotype_treatment.items():
    color = combo_color(genotype, treatment)
    s1, s5, sm, s95, s99 = [], [], [], [], []
    for j in range(len(time_windows_range)):
        group_speeds_flat = np.concatenate([speeds[j][i].ravel() for i in indices]) if len(indices) else np.array([], dtype=float)
        p = robust_percentiles(group_speeds_flat, qs=(1, 5, 95, 99))
        s1.append(p[1]); s5.append(p[5]); s95.append(p[95]); s99.append(p[99])
        sm.append(float(np.nanmedian(group_speeds_flat)) if group_speeds_flat.size else np.nan)

    # plot each stat in its own axis
    axes[0].plot(time_windows_range, s1, ".-", alpha=0.6, color=color, label=combo_label(genotype, treatment))
    axes[1].plot(time_windows_range, s5, ".-", alpha=0.6, color=color)
    axes[2].plot(time_windows_range, sm, ".-", alpha=0.6, color=color)
    axes[3].plot(time_windows_range, s95, ".-", alpha=0.6, color=color)
    axes[4].plot(time_windows_range, s99, ".-", alpha=0.6, color=color)

# Titles + cosmetics
for ax, t in zip(axes[:5], titles):
    ax.set_title(f"dFC speed {t}")
for ax in axes:
    ax.grid(alpha=0.2)
axes[3].set_xlabel("Time Window Size")
axes[4].set_xlabel("Time Window Size")
axes[2].set_ylabel("dFC Speed")
axes[0].legend(ncol=1, fontsize=10)
fig.delaxes(axes[5])  # remove the empty 6th cell
fig.tight_layout()
#%%
# for (genotype, treatment), indices in group_genotype_treatment.items():

#     mean_speeds_group = []
#     p1_speeds_group  = []
#     p5_speeds_group  = []
#     p95_speeds_group = []
#     p99_speeds_group = []
#     for j in range(len(time_windows_range)):
#         group_speeds_flat = (
#             np.concatenate([speeds[j][i].ravel() for i in indices])
#             if len(indices)
#             else np.array([], dtype=float)
#         )
#         p = robust_percentiles(group_speeds_flat, qs=(1, 5, 95, 99))
#         p1_speeds_group.append(p[1])
#         p5_speeds_group.append(p[5])
#         p95_speeds_group.append(p[95])
#         p99_speeds_group.append(p[99])

#         # print(np.allclose( np.percentile(group_speeds_flat, 1), p[1]))
#         mean_speeds_group.append(
#             float(np.median(group_speeds_flat)) if group_speeds_flat.size else np.nan
#         )
#     # (
#     #     mean_speeds_group,
#     #     p1_speeds_group,
#     #     p5_speeds_group,
#     #     p95_speeds_group,
#     #     p99_speeds_group,
#     # ) = ([], [], [], [], [])

#     #     p1_speeds_group.append(
#     #         float(np.percentile(group_speeds_flat, 1))
#     #         if group_speeds_flat.size
#     #         else np.nan
#     #     )
#     #     p5_speeds_group.append(
#     #         float(np.percentile(group_speeds_flat, 5))
#     #         if group_speeds_flat.size
#     #         else np.nan
#     #     )
#     #     p95_speeds_group.append(
#     #         float(np.percentile(group_speeds_flat, 95))
#     #         if group_speeds_flat.size
#     #         else np.nan
#     #     )
#     #     p99_speeds_group.append(
#     #         float(np.percentile(group_speeds_flat, 99))
#     #         if group_speeds_flat.size
#     #         else np.nan
#     #     )

#     # Plotting mean speeds and percentiles
#     plt.subplot(2, 3, 1)
#     plt.plot(time_windows_range, p1_speeds_group, ".-", alpha=alpha_aux)

#     plt.title("dFC Speed 1st Percentile")
#     plt.subplot(2, 3, 2)
#     plt.plot(time_windows_range, p5_speeds_group, ".-", alpha=alpha_aux)
#     plt.title("dFC Speed 5th Percentile")
#     plt.subplot(2, 3, 3)
#     plt.plot(time_windows_range, mean_speeds_group, ".-", alpha=alpha_aux)
#     plt.title("dFC Speed mean ")
#     plt.subplot(2, 3, 4)
#     plt.plot(time_windows_range, p95_speeds_group, ".-", alpha=alpha_aux)
#     plt.title("dFC Speed 95th Percentile")
#     plt.subplot(2, 3, 5)
#     plt.plot(
#         time_windows_range,
#         p99_speeds_group,
#         ".-",
#         label=f"{genotype} {treatment} ",
#         alpha=alpha_aux,
#     )
#     plt.title("dFC Speed 99th Percentile")
# plt.xlabel("Time Window Size")
# plt.ylabel("dFC Speed")
# plt.legend()
# plt.tight_layout()
# %%
# cumulative distribution function (CDF)
plt.figure(figsize=(7, 5))
plt.title("Cumulative Distribution of dFC Speeds across Time Windows")
plt.axvline(
    x=time_windows_range[indice_half],
    color="red",
    linestyle="--",
    label="Median Window Size",
    alpha=0.5,
)
plt.axvline(
    x=time_windows_range[indice_third],
    color="green",
    linestyle="--",
    label="1/3 Window Size",
    alpha=0.5,
)
plt.axvline(
    x=time_windows_range[indice_two_third],
    color="blue",
    linestyle="--",
    label="2/3 Window Size",
    alpha=0.5,
)

plt.axhline(y=0.5, color="red", linestyle="--", alpha=0.5)
plt.axhline(y=1 / 3, color="green", linestyle="--", alpha=0.5)
plt.axhline(y=2 / 3, color="blue", linestyle="--", alpha=0.5)
plt.plot(time_windows_range, pooled_speeds_cdf, color="orange", lw=2, alpha=0.8)

plt.xlabel("Time Window Size")
plt.ylabel("Cumulative Frequency")
# plt.xticks(time_windows_range[::5])
step = max(1, len(time_windows_range)//12)
plt.xticks(time_windows_range[::step])

plt.legend()
plt.tight_layout()

# Plot overall distribution of speeds

plt.figure(figsize=(7, 5))
# plt.subplot(3, 1, 1)
plt.title("Pooled Speed (all windows pooled)")
plt.plot(
    bin_edge[:-1],
    all_speeds_hist,
    color="dodgerblue",
    lw=2,
    alpha=0.8,
    label="all animals",
)
if pool_split == "half":
    plt.plot(
        bin_edge[:-1],
        short_speeds_hist,
        color="orange",
        lw=2,
        alpha=0.8,
        label="short windows",
    )
    plt.plot(
        bin_edge[:-1],
        long_speeds_hist,
        color="green",
        lw=2,
        alpha=0.8,
        label="long windows",
    )
elif pool_split == "third":
    plt.plot(
        bin_edge[:-1],
        short_speeds_hist,
        color="orange",
        lw=2,
        alpha=0.8,
        label="short windows",
    )
    plt.plot(
        bin_edge[:-1],
        mid_speeds_hist,
        color="purple",
        lw=2,
        alpha=0.8,
        label="mid windows",
    )
    plt.plot(
        bin_edge[:-1],
        long_speeds_hist,
        color="green",
        lw=2,
        alpha=0.8,
        label="long windows",
    )
# # Alternative plotting method
# plt.plot(all_speeds_hist[1][:-1], all_speeds_hist[0], color='dodgerblue', lw=2, alpha=0.8, label='all animals')
# plt.plot(short_speeds_hist[1][:-1], short_speeds_hist[0], color='orange', lw=2, alpha=0.8, label='short windows')
# plt.plot(mid_speeds_hist[1][:-1], mid_speeds_hist[0], color='purple', lw=2, alpha=0.8, label='mid windows')
# plt.plot(long_speeds_hist[1][:-1], long_speeds_hist[0], color='green', lw=2, alpha=0.8, label='long windows')
plt.legend()
plt.xlabel("Speed")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
# %%
# Plot distribution by region groups
# for label, hist in all_speeds_grp_hist.items():
# for label_big in label_sets:
plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
for label in group_genotype_treatment.keys():
    hist = all_speeds_grp_hist[label]
    plt.plot(bin_edge[:-1], hist, lw=1, alpha=0.4, label=label)
plt.title("Distribution of dFC Speed")
plt.xlabel("Speed")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()

plt.subplot(1, 2, 2)
for label in group_genotype_treatment.keys():
    hist = all_speeds_grp_hist[label]
    plt.plot(
        bin_edge[:-1],
        hist,
        # Alternative plotting method
        # plt.plot(hist[1][:-1], hist[0],
        lw=1,
        alpha=0.4,
        label=label,
    )
plt.title("Distribution of dFC Speed (Log Scale)")
plt.xlabel("Speed")
plt.ylabel("Frequency")
plt.yscale("log")
plt.legend()
plt.tight_layout()


# plt.savefig(paths['f_speed'] / f'speed_distribution_by_region_groups_{label_big}_windows_ines.png', dpi=300)
# %%

# %%
