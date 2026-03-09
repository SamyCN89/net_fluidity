# %% ========================== IMPORTS & CONFIG ==========================
# Generate group-wise Spearman correlation plots with bootstrapped CIs & significance shading


from collections.abc import Iterable, Sequence
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np

# Optional for Parquet saving (Option B)
try:
    import pandas as pd
except Exception:
    pd = None

from scripts.dfc.dfc_compute import DATASET_DEFAULTS, _canonical_dataset
from shared_code.fun_loaddata import load_timeseries_bundle
from shared_code.fun_paths import get_paths
from shared_code.fun_utils import load_cognitive_data, set_figure_params

# %%
# ----------------- User toggles -----------------
# SAVE_MODE = {
#     "npz_pack": True,  # Option A
#     "parquet": False,
# }  # Option B (requires pandas)
# RNG_SEED = 123  # for future bootstrap reproducibility
# -----------------------------------------------

GROUP_RECIPES = {
    # single factors
    "sex": ["Sexe"],
    "age": ["Age"],
    "genotype": ["Genotype"],
    "phenotype_oip": ["Phenotype_OiP"],
    "phenotype_nor": ["Phenotype_RO24h"],
    # 2-way
    "age_sex": ["Age", "Sexe"],
    "age_genotype": ["Age", "Genotype"],
    "age_phenotype_oip": ["Age", "Phenotype_OiP"],
    "age_phenotype_nor": ["Age", "Phenotype_RO24h"],
    "sex_genotype": ["Sexe", "Genotype"],
    "sex_phenotype_oip": ["Sexe", "Phenotype_OiP"],
    "sex_phenotype_nor": ["Sexe", "Phenotype_RO24h"],
    # 3-way
    "age_sex_genotype": ["Sexe", "Age", "Genotype"],
    "age_sex_phenotype_oip": ["Sexe", "Age", "Phenotype_OiP"],
    "age_sex_phenotype_nor": ["Sexe", "Age", "Phenotype_RO24h"],
    # default used in your “julien” branch originally
    "genotype_treatment": ["genotype", "treatment"],  # only if those cols exist
}


# fixed palette, independent of rcParams
AGE_CONTRAST_PALETTE = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]


def make_age_contrast_color_map(group_keys, groups_selected: str) -> dict[str, str]:
    """
    Map each non-age label → a fixed color from AGE_CONTRAST_PALETTE.
    """
    # pure age: only one contrast
    example_key = next(iter(group_keys))
    if groups_selected == "age" and isinstance(example_key, str):
        return {"4M-2M": AGE_CONTRAST_PALETTE[0]}

    labels = []
    for k in group_keys:
        if isinstance(k, str):
            continue
        if "4M" not in k:
            continue
        lbl = age_contrast_label(k)
        if lbl not in labels:
            labels.append(lbl)

    return {
        lbl: AGE_CONTRAST_PALETTE[i % len(AGE_CONTRAST_PALETTE)]
        for i, lbl in enumerate(labels)
    }


# %%
# # %% ========================== SMALL HELPERS ==========================


def load_speed_stack(
    paths_speed_root: Path, time_windows_range: Sequence[int]
) -> list[np.ndarray]:
    """Return list S where S[j][i] is 1D np.array of samples for animal i at window j."""
    speeds: list[np.ndarray] = []
    for w in time_windows_range:
        a = np.load(
            paths_speed_root.format(w=w, n_animals=n_animals, regions=regions),
            allow_pickle=True,
        )
        s = a["speeds"]
        s_flat = [np.asarray(x, dtype=float).ravel() for x in s]
        speeds.append(s_flat)
    return speeds


# # --- pooling helpers ---


def count_samples_per_window(speeds: list[np.ndarray]) -> np.ndarray:
    """Count total samples per time window across all animals.
    speeds:
        list S where S[j][i] is 1D np.array of samples for animal i at window j.
    Returns:
        counts: 1D np.array where counts[j] = total samples at window j across animals.
    """
    return np.array([sum(len(x) for x in speed) for speed in speeds], dtype=int)


def cdf_split_indices(speeds: list[np.ndarray]) -> tuple[int, int, int]:
    counts = count_samples_per_window(speeds)
    cdf = (
        np.cumsum(counts) / counts.sum()
        if counts.sum() > 0
        else np.zeros_like(counts, dtype=float)
    )
    i_third = int(np.searchsorted(cdf, 1.0 / 3.0))
    i_half = int(np.searchsorted(cdf, 0.5))
    i_two_third = int(np.searchsorted(cdf, 2.0 / 3.0))
    i_third = max(1, i_third)
    i_half = max(1, i_half)
    i_two_third = max(i_third + 1, i_two_third)
    return i_third, i_half, i_two_third


def select_windows(
    pool_split: str, n_windows: int, i_third: int, i_half: int, i_two_third: int
) -> dict[str, range]:
    """Return dict of pool name → range of window indices."""
    if pool_split == "all":
        return {"all": range(0, n_windows)}
    if pool_split == "half":
        return {"short": range(0, i_half), "long": range(i_half, n_windows)}
    return {
        "short": range(0, i_third),
        "mid": range(i_third, i_two_third),
        "long": range(i_two_third, n_windows),
    }


def flatten_windows(speeds: list[np.ndarray], start: int, end: int) -> np.ndarray:
    arrays = [
        np.asarray(s, dtype=float).ravel() for speed in speeds[start:end] for s in speed
    ]
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=float)


def global_min_max(arrs: Iterable[np.ndarray]) -> tuple[float, float]:
    vals_min = [np.nanmin(a) for a in arrs if a.size]
    vals_max = [np.nanmax(a) for a in arrs if a.size]
    vmin = min(vals_min) if vals_min else 0.0
    vmax = max(vals_max) if vals_max else 1.0
    if vmin == vmax:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


# %%
def plot_group_median_vs_window(
    ax: plt.Axes,
    time_windows_range: Sequence[int],
    group_data: dict[tuple[str, str], Sequence[int]],
    speeds: list[np.ndarray],
) -> None:
    """Mean of per-animal medians vs window (group curve)."""
    for (genotype, treatment, x), indices in group_data.items():
        print(f"[INFO] Processing group: {genotype}, {treatment}")
        y = []
        for j in range(len(time_windows_range)):
            per_animal_medians = [float(np.median(speeds[j][i])) for i in indices]
            y.append(
                float(np.mean(per_animal_medians)) if per_animal_medians else np.nan
            )
        ax.plot(time_windows_range, y, ".-", label=combo_label(genotype, treatment))
    ax.set_xlabel("Time Window Size")
    ax.set_ylabel("Mean of per-animal medians (dFC speed)")
    ax.set_title("dFC Speed vs Window per Genotype–Treatment")
    ax.legend()


# %% ================ GROUPING HELPERS ==========================
# Group indices from cognitive data


def make_long_cog(cog_data: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Return a long-form dataframe with columns:
    Name, Sexe, Genotype, Age, oip, ro24h, tc, Phenotype_OiP, Phenotype_RO24h
    """
    if dataset_name == "julien":
        # already a single age or different schema? adapt here if needed
        df = cog_data.copy()
        # ensure consistent columns exist; fill missing if required
        for c in ["oip", "ro24h", "tc", "Phenotype_OiP", "Phenotype_RO24h"]:
            if c not in df.columns:
                df[c] = np.nan
        if "Age" not in df.columns:
            df["Age"] = "NA"
        df = df[
            [
                "Name",
                "Sexe",
                "Genotype",
                "Age",
                "oip",
                "ro24h",
                "tc",
                "Phenotype_OiP",
                "Phenotype_RO24h",
            ]
        ]

    elif dataset_name == "ines":
        cols_common = ["Name", "Sexe", "Genotype", "Phenotype_OiP", "Phenotype_RO24h"]
        df2 = cog_data[cols_common + ["OiP_2M", "RO24h_2M", "TC_2M"]].copy()
        df4 = cog_data[cols_common + ["OiP_4M", "RO24h_4M", "TC_4M"]].copy()

        df2["Age"] = "2M"
        df4["Age"] = "4M"
        df2 = df2.rename(columns={"OiP_2M": "oip", "RO24h_2M": "ro24h", "TC_2M": "tc"})
        df4 = df4.rename(columns={"OiP_4M": "oip", "RO24h_4M": "ro24h", "TC_4M": "tc"})
        df = pd.concat([df2, df4], ignore_index=True)

    else:
        raise ValueError(f"Unknown dataset_name={dataset_name}")

    # Normalize values
    df["Sexe"] = df["Sexe"].map({"F": "female", "M": "male"}).fillna(df["Sexe"])
    # Optional: categorical types (faster grouping, stable order)
    for col in ["Sexe", "Age", "Genotype", "Phenotype_OiP", "Phenotype_RO24h"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def group_indices(df: pd.DataFrame, by: Sequence[str], index_col: str = "Name"):
    """
    Return a dict { group_key_tuple_or_scalar : np.ndarray[int] } of row indices.
    """
    if not by:
        # single group: "all"
        return {"all": np.arange(len(df), dtype=int)}
    gb = df.groupby(list(by), sort=False)
    return {k: v.values for k, v in gb.groups.items()}


def get_group_data(cog_data: pd.DataFrame, dataset_name: str, groups_selected: str):
    """Return group indices dict based on selected grouping recipe."""
    df_long = make_long_cog(cog_data, dataset_name)

    # If a recipe references missing columns, raise a helpful error.
    cols = GROUP_RECIPES.get(groups_selected)
    if cols is None:
        raise ValueError(
            f"Unknown groups_selected='{groups_selected}'. "
            f"Choose from: {sorted(GROUP_RECIPES.keys())}"
        )

    missing = [c for c in cols if c not in df_long.columns]
    if missing:
        raise ValueError(
            f"Grouping '{groups_selected}' needs columns {missing} "
            f"missing in df_long.columns={list(df_long.columns)}"
        )

    return group_indices(df_long, cols)


# %%

# # =============================================================================
# # -----------------------------------------------------------------------------
# # ============================= Main Code ==============================
# # -----------------------------------------------------------------------------
# # =============================================================================


# # %% ========================== LOAD DATA ==========================
dataset_name = "ines"
# dataset_name = "julien"
save_fig = set_figure_params(True)
dataset = _canonical_dataset(dataset_name)
cfg = DATASET_DEFAULTS[dataset]

time_windows_range = np.arange(5, 100, 1)

POOL_SPLIT = "third"  # 'half' | 'third' | 'all'
BINS_HIST = 200

SAVE_GROUP_HISTS = True
# %% ============= Get paths & output locations ====================
paths = get_paths(
    dataset_name=dataset,
    timecourse_folder=cfg["timecourse_folder"],
    cognitive_data_file=cfg["cognitive_data_file"],
    anat_labels_file=cfg["anat_labels_file"],
)


# # Root locations
speed_root = Path(paths["speed"])
preprocessed_root = Path(paths["preprocessed"])

# # Load location
loaddir_ts_meta = preprocessed_root / "ts_and_meta_2m4m.npz"
loaddir_cog_data = str(
    preprocessed_root
    / "cog_data_filtered_animals_{n_animals}_regions_{regions}_tr_{total_tr}.csv"
)
# loaddir_speed = str(speed_root / "all/all/speed_win{w}_lag1_tau4_animals_48_regions_37.npz")
# loaddir_speed = str(speed_root / "dmn_within/nregs-6/speed_win{w}_lag1_tau4_animals_{n_animals}_regions_{regions}.npz")
loaddir_speed = str(
    speed_root / "all/speed_win{w}_lag1_tau2_animals_{n_animals}_regions_{regions}.npz"
)

# # Output location for group histograms
# # outdir_save_group_hists = speed_root / f"{dataset}_pool_{POOL_SPLIT}_bins{BINS_HIST}" / f"pooled_group_hists__{seg_name}.npz"
# # bootstrap CI folder
bootstrap_folder = paths["speed"] / "bootstrap"
bootstrap_folder.mkdir(parents=True, exist_ok=True)


outdir_bootstrap_repeat = str(
    bootstrap_folder
    / "bootstrap_downsample_repeat_group_{groups_selected}_nresamples_{n_resamples}_downsample_factor_{downsample_factor}_seed_{seed}.pkl"
)

# outdir_speed_bootstrap_cis = str(
#     bootstrap_folder / f"bootstrap_cis_{POOL_SPLIT}_bins{BINS_HIST}.npz"
# )

# # Savedir location
# # acceleration plots
distribution_folder = paths["f_speed"] / "distribution"
distribution_folder.mkdir(parents=True, exist_ok=True)

# # acceleration plots
acceleration_folder = paths["f_speed"] / "acceleration"
acceleration_folder.mkdir(parents=True, exist_ok=True)

# savedir_dfc_speed_per_animal = str(
#     time_window_folder
#     / "dFC_speed_per_animal_{n_animals}_regions_{regions}_tr_{total_tr}.png"
# )
# savedir_dfc_speed_group_median_vs_window = str(
#     time_window_folder
#     / "dFC_speed_group_median_vs_window_{n_animals}_regions_{regions}_tr_{total_tr}.png"
# )
# savedir_dfc_speed_percentiles_vs_window = str(
#     time_window_folder
#     / "dFC_speed_percentiles_vs_window_{n_animals}_regions_{regions}_tr_{total_tr}.png"
# )

# # pooling plots
# pooling_folder = paths["f_speed"] / "pooling"
# pooling_folder.mkdir(parents=True, exist_ok=True)

# savedir_dfc_speed_cdf_windows = str(
#     pooling_folder
#     / "dFC_speed_cdf_windows_animals_{n_animals}_regions_{regions}_tr_{total_tr}.png"
# )
# savedir_pooled_speed_hist_bins = str(
#     pooling_folder
#     / "pooled_speed_hist_bins{BINS_HIST}_animals_{n_animals}_regions{regions}_tr{total_tr}.png"
# )
# savedir_pooled_group_hists = str(
#     speed_root
#     / "pooled_group_hists_{POOL_SPLIT}_bins{BINS_HIST}_animals_{n_animals}_regions{regions}_tr{total_tr}.png"
# )


# # %% ========================== LOAD DATA ==========================
# # Load timeseries bundle to get n_animals, n_regions, total_tr
bundle = load_timeseries_bundle(loaddir_ts_meta)
n_animals = bundle.n_animals
total_tr = bundle.total_tr
regions = bundle.n_regions

# Load cognitive data for grouping
cog_data = load_cognitive_data(
    loaddir_cog_data.format(n_animals=n_animals, regions=regions, total_tr=total_tr)
)
# %%
# Load speed data
speeds = load_speed_stack(
    loaddir_speed,
    time_windows_range,
)
# Basic dimensions
n_windows = len(speeds)
n_animals = len(speeds[0])

# speeds[j][i] = 1D np.array of all speed samples for animal i, at window j (time_windows_range)

#AMO MUCHO A MARINE <3


# %%
# ========================== PRECOMPUTE DIMENSIONS ==========================


# # Precompute mean speed for each animal at each window (handy for mean-based bootstrap later)
per_window_animal_means = [
    np.array([float(np.mean(speeds[j][i])) for i in range(n_animals)], dtype=float)
    for j in range(n_windows)
]

# %% ========================== SPLITS IN POOLINGS & HIST SETUP ==========================

# Get the split indices and ranges
counts = count_samples_per_window(speeds)
pooled_speeds_cdf = (
    np.cumsum(counts) / np.sum(counts) if counts.sum() else np.zeros_like(counts)
)

# Get split indices
i_third, i_half, i_two_third = cdf_split_indices(speeds)
ranges = select_windows(
    POOL_SPLIT, len(time_windows_range), i_third, i_half, i_two_third
)

# Get global min/max for histogram binning
all_speeds_flat = flatten_windows(speeds, 0, len(speeds))
all_speeds_min, all_speeds_max = global_min_max([all_speeds_flat])
edges = np.linspace(all_speeds_min, all_speeds_max, BINS_HIST + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
# %%


# # ================================================================================
# # --------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------
# # ================================================================================
# # ----------------------------------- group_data start ---------------------------
# # ================================================================================
# # --------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------
# # ================================================================================

# percentiles_ = np.linspace(0, 100, 100)
n_resamples = 10_000
downsample_factor = 10
seed = 42

# # %% ========================== GROUP INDICES ==========================
# # build once
df_long = make_long_cog(cog_data, dataset_name)

# # pick a grouping
# # groups_selected = "age_sex_genotype"  # or "sex", "age", "phenotype_oip", ...

groups_list = [
    "sex",
    "age",
    "genotype",
    "phenotype_oip",
    "phenotype_nor",
    "age_sex",
    "age_genotype",
    "age_phenotype_nor",
    "age_phenotype_oip",
    "age_sex_genotype",
    "age_sex_phenotype_oip",
    "age_sex_phenotype_nor",
]

# %%
# ============================================================
#   PLOTTING: group mean histograms (linear + log panels)
# ============================================================


def pretty_group_label(gt):
    if isinstance(gt, tuple):
        return " | ".join(str(x) for x in gt)
    return str(gt)


def age_contrast_label(gt_4m):
    """
    Build the label used for 4M–2M contrasts.
    For tuple keys: drop the age ('2M'/'4M') and join other factors.
    For pure age: just return '4M-2M'.
    """
    # pure age grouping will handle label separately
    if isinstance(gt_4m, str):
        return "4M-2M"

    # tuple like ('female', '4M', 'wt', ...)
    parts = [str(v) for v in gt_4m if v not in ("2M", "4M")]
    return " | ".join(parts) if parts else "all"


# %%
# ========================== PLOTTING GROUP MEAN HISTOGRAMS ==========================

for groups_selected in groups_list:
    print(f"\nProcessing grouping: {groups_selected}")

    # --- Load bootstrap pack ---
    outdir_bootstrap_repeat_aux = Path(
        outdir_bootstrap_repeat.format(
            groups_selected=groups_selected,
            n_resamples=n_resamples,
            downsample_factor=downsample_factor,
            seed=seed,
        )
    )

    if not outdir_bootstrap_repeat_aux.exists():
        print("  -> bootstrap missing")
        continue

    with open(outdir_bootstrap_repeat_aux, "rb") as f:
        data_loaded = pickle.load(f)

    ranges = data_loaded["ranges"]
    group_means_by_segment = data_loaded["group_means_by_segment"]
    pooled_group_hists_by_segment = data_loaded["pooled_group_hists_by_segment"]
    pooled_group_speed_by_segment = data_loaded["pooled_group_speed_by_segment"]
    group_speed_by_segment = data_loaded["group_speed_by_segment"]

    seg_names = list(ranges.keys())
    n_seg = len(seg_names)

    # -----------------------------
    # Two-row figure:
    #   Top: linear scale
    #   Bottom: log scale
    # -----------------------------
    fig, axes = plt.subplots(
        2,
        n_seg,
        figsize=(6 * n_seg, 8),
        sharex=True,
    )

    # If n_seg = 1 → normalize axes to 2 lists
    if n_seg == 1:
        axes = np.array([axes]).reshape(2, 1)

    # -----------------------------------------
    # Prepare clean, readable color cycle
    # -----------------------------------------
    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        color=plt.cm.tab20(np.linspace(0, 1, 20))
    )

    for col, seg_name in enumerate(seg_names):
        # group_means = group_means_by_segment[seg_name]
        group_means = pooled_group_hists_by_segment[seg_name]

        # ---------- TOP ROW (linear) ----------
        ax_lin = axes[0, col]
        ax_lin.set_title(f"{seg_name} (linear scale)", fontsize=14)

        for gt, mean_hist in group_means.items():
            ax_lin.plot(
                centers, mean_hist, lw=1.2, alpha=0.8, label=pretty_group_label(gt)
            )

        ax_lin.set_xlabel("Speed")
        ax_lin.set_ylabel("Density")
        ax_lin.grid(True, which="both", ls="--", lw=0.4)

        # ---------- BOTTOM ROW (log scale) ----------
        ax_log = axes[1, col]
        ax_log.set_title(f"{seg_name} (log scale)", fontsize=14)

        for gt, mean_hist in group_means.items():
            ax_log.plot(centers, mean_hist, lw=1.2, alpha=0.8)

        ax_log.set_xlabel("Speed")
        ax_log.set_ylabel("Density (log)")
        ax_log.set_yscale("log")
        ax_log.grid(True, which="both", ls="--", lw=0.4)

    # -----------------------------
    # Single consolidated legend
    # -----------------------------
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))

    fig.legend(
        unique.values(),
        unique.keys(),
        title="Groups",
        loc="center left",
        bbox_to_anchor=(0.92, 0.5),
        fontsize=11,
        frameon=False,
        handlelength=1.0,
        handleheight=0.8,
        borderpad=0.4,
        labelspacing=0.3,
        handletextpad=0.4,
    )

    # only ~10–12% of width reserved instead of 20%
    plt.tight_layout(rect=[0.02, 0.02, 0.88, 0.95])

    plt.savefig(
        distribution_folder
        / f"group_means_dist_{groups_selected}_{POOL_SPLIT}_bins{BINS_HIST}.png",
        bbox_inches="tight",
    )
    plt.show()

# %%


# ========================== PLOTTING CIs ==========================
for groups_selected in groups_list:
    print(f"\nProcessing grouping (CIs): {groups_selected}")

    # --- Load bootstrap pack ---
    outdir_bootstrap_repeat_aux = Path(
        outdir_bootstrap_repeat.format(
            groups_selected=groups_selected,
            n_resamples=n_resamples,
            downsample_factor=downsample_factor,
            seed=seed,
        )
    )

    if not outdir_bootstrap_repeat_aux.exists():
        print("  -> bootstrap file missing, skipping")
        continue

    with open(outdir_bootstrap_repeat_aux, "rb") as f:
        data_loaded = pickle.load(f)
    print(f"Loading bootstrap: {outdir_bootstrap_repeat_aux}")

    # metadata / results
    ranges = data_loaded["ranges"]
    percentiles_ = data_loaded["percentiles_"]
    group_data = data_loaded["group_data"]
    ci_low_repeat = data_loaded["ci_low_repeat"]
    ci_high_repeat = data_loaded["ci_high_repeat"]

    seg_names = list(ranges.keys())
    n_seg = len(seg_names)

    # -----------------------------
    # Two-row figure:
    #   Top: linear y-scale
    #   Bottom: log y-scale
    # -----------------------------
    fig, axes = plt.subplots(
        2,
        n_seg,
        figsize=(6 * n_seg, 8),
        # sharex=True,
    )
    if n_seg == 1:
        axes = np.array([axes]).reshape(2, 1)

    # consistent color cycle
    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        color=plt.cm.tab20(np.linspace(0, 1, 20))
    )

    # for legend consolidation
    legend_handles = []
    legend_labels = []

    for col, seg_name in enumerate(seg_names):

        # ---------- TOP ROW (linear y) ----------
        ax_lin = axes[0, col]
        ax_lin.set_title(f"{seg_name} (linear scale)", fontsize=14)

        for gt in group_data.keys():
            lo = ci_low_repeat[seg_name][gt]
            hi = ci_high_repeat[seg_name][gt]
            label = pretty_group_label(gt)

            band = ax_lin.fill_between(
                percentiles_,
                lo,
                hi,
                alpha=0.6,
                label=label,
            )

            legend_handles.append(band)
            legend_labels.append(label)

        ax_lin.set_xlabel("Percentiles")
        ax_lin.set_ylabel("Speed")
        ax_lin.set_ylim(0.2, 1.5)
        ax_lin.set_xlim(0, 100)
        ax_lin.set_xscale("linear")
        ax_lin.set_yscale("linear")
        ax_lin.grid(True, which="both", ls="--", lw=0.4)

        # ---------- BOTTOM ROW (log y) ----------
        ax_log = axes[1, col]
        ax_log.set_title(f"{seg_name} (log scale)", fontsize=14)

        for gt in group_data.keys():
            lo = ci_low_repeat[seg_name][gt]
            hi = ci_high_repeat[seg_name][gt]

            ax_log.fill_between(
                percentiles_,
                lo,
                hi,
                alpha=0.6,
            )

        ax_log.set_xlabel("Percentiles")
        ax_log.set_ylabel("Speed (log)")
        ax_log.set_yscale("log")
        ax_log.set_xscale("log")
        ax_log.grid(True, which="both", ls="--", lw=0.4)

    # -----------------------------
    # Single consolidated legend
    # -----------------------------
    uniq = {}
    for h, l in zip(legend_handles, legend_labels, strict=False):
        if l not in uniq:
            uniq[l] = h

    fig.legend(
        uniq.values(),
        uniq.keys(),
        title="Groups",
        loc="center left",
        bbox_to_anchor=(0.92, 0.5),
        fontsize=11,
        frameon=False,
        handlelength=1.0,
        handleheight=0.8,
        borderpad=0.4,
        labelspacing=0.3,
        handletextpad=0.4,
    )

    plt.tight_layout(rect=[0.02, 0.02, 0.88, 0.95])

    plt.savefig(
        distribution_folder
        / f"ci_comparison_{groups_selected}_{POOL_SPLIT}_bins{BINS_HIST}.png",
        bbox_inches="tight",
    )
    plt.show()


# %%
# % ========================== PLOTTING ==========================
for groups_selected in groups_list:
    print(f"Processing grouping: {groups_selected}")

    # folder load bootstrap
    outdir_bootstrap_repeat_aux = Path(
        outdir_bootstrap_repeat.format(
            groups_selected=groups_selected,
            n_resamples=n_resamples,
            downsample_factor=downsample_factor,
            seed=seed,
        )
    )
    # check exists
    if not outdir_bootstrap_repeat_aux.exists():
        print("  -> bootstrap file missing, skipping")
        continue
    # load bootstrap data
    with open(outdir_bootstrap_repeat_aux, "rb") as f:
        data_loaded = pickle.load(f)
    print(f"Loading bootstrap: {outdir_bootstrap_repeat_aux}")

    # metadata
    groups_selected = data_loaded["groups_selected"]
    group_data = data_loaded["group_data"]
    ranges = data_loaded["ranges"]
    percentiles_ = data_loaded["percentiles_"]

    # bootstrap results
    ci_low_repeat = data_loaded["ci_low_repeat"]
    ci_high_repeat = data_loaded["ci_high_repeat"]
    vals_btr_downsample_repeat = data_loaded["ci_btr_downsample_repeat"]

    # speed data
    group_means_by_segment = data_loaded["group_means_by_segment"]
    pooled_group_hists_by_segment = data_loaded["pooled_group_hists_by_segment"]
    pooled_group_speed_by_segment = data_loaded["pooled_group_speed_by_segment"]
    group_speed_by_segment = data_loaded["group_speed_by_segment"]

    print(f"Group data keys: {group_data.keys()}")

    # plot confidence intervals comparison
    plt.figure(figsize=(16, 8))
    for seg_name, w_range in ranges.items():
        plt.subplot(1, len(ranges), list(ranges.keys()).index(seg_name) + 1)
        plt.title(f"Confidence Intervals Comparison - {seg_name}")
        for gt in group_data.keys():
            # print("Plotting GT:", gt)
            plt.fill_between(
                percentiles_,
                ci_low_repeat[seg_name][gt],
                ci_high_repeat[seg_name][gt],
                alpha=0.5,
                label=gt,
            )

        # plt.plot(percentiles_, group_flat_speed_perc, label='Pooled Group Histogram')
        plt.xlabel("Percentiles")
        plt.ylabel("Speed")
        plt.yscale("log")
        plt.xscale("log")
        plt.ylim(2e-1, 1.5e0)
        # plt.xlim(0,3)

        # plt.xscale('log')
        plt.legend()
    plt.tight_layout()
    plt.savefig(
        distribution_folder
        / f"ci_comparison_downsampled_classic_bootstrap_{groups_selected}_{POOL_SPLIT}_bins{BINS_HIST}.png"
    )
    plt.savefig(
        distribution_folder
        / f"ci_comparison_downsampled_classic_bootstrap__loglog_{groups_selected}_{POOL_SPLIT}_bins{BINS_HIST}.png"
    )
    plt.show()

    for seg_name, w_range in ranges.items():
        # Group loops
        ci_low_i = {}
        ci_high_i = {}
        for gt, idxs in group_data.items():
            print(pooled_group_speed_by_segment[seg_name][gt].shape, seg_name, gt)
            group_flat_speed = pooled_group_speed_by_segment[seg_name][gt]

            # for classic resampling of distribution
            group_flat_speed_perc = np.percentile(group_flat_speed, percentiles_)

            # for each animal histogram
            animal_flat_speed_perc = np.empty(
                (len(group_speed_by_segment[seg_name][gt][0]), len(percentiles_)),
                dtype=float,
            )

            group_speed_by_segment_i = group_speed_by_segment[seg_name][gt][0]

            plt.figure(figsize=(12, 10))
            for i in range(len(group_speed_by_segment_i)):
                # print(f"Animal {i} speed shape: {np.shape(group_speed_by_segment_i[i])}")
                aux_animal_s = group_speed_by_segment_i[i]
                aux_animal_i_s_flat = [
                    aux_animal_s[j].tolist() for j in range(len(aux_animal_s))
                ]
                # flatten aux_animal_i_s_flat
                aux_animal_i_s_flat = np.array(
                    [item for sublist in aux_animal_i_s_flat for item in sublist]
                )
                # print(f"Animal {i} flat speed shape: {np.shape(aux_animal_i_s_flat)}")

                hist_aux = np.histogram(
                    np.ravel(aux_animal_i_s_flat),
                    bins=100,
                    range=(group_flat_speed.min(), group_flat_speed.max()),
                )

                plt.plot(
                    hist_aux[1][:-1], hist_aux[0], ".-", label=f"Animal {i}", alpha=0.5
                )  # nor {nor_index[idxs[i]]}')
                # plt.plot()

                plt.xlabel("Speed")
                plt.ylabel("Count")
                plt.title(f"Animal Speed Histogram - {seg_name} - {gt}")
                plt.xlim(0.1, 1.2)

                aux_animal_i_perc = np.percentile(aux_animal_i_s_flat, percentiles_)
                # print(f"Animal {i} percentiles: {aux_animal_i_perc}")
                animal_flat_speed_perc[i] = aux_animal_i_perc
            # plt.legend()
            plt.tight_layout()
            gt_format = str(
                str(gt)
                .replace("(", "")
                .replace(")", "")
                .replace(",", "_")
                .replace("'", "")
                .replace(" ", "")
            )
            plt.savefig(
                distribution_folder
                / f"animal_speed_histograms_{seg_name}_gt_{gt_format}.png"
            )
            plt.show()
# %%

# %%
# %%
groups_list = [
    # "sex",
    "age",
    # "genotype",
    # "phenotype_oip",
    # "phenotype_nor",
    "age_sex",
    "age_genotype",
    "age_phenotype_nor",
    "age_phenotype_oip",
    "age_sex_genotype",
    "age_sex_phenotype_oip",
    "age_sex_phenotype_nor",
]

# ============================================================
#   CI DIFF BANDS: 4M - 2M AS SUBPLOTS PER SEGMENT
# ============================================================

for groups_selected in groups_list:
    print(f"Processing grouping: {groups_selected}")

    # folder load bootstrap
    outdir_bootstrap_repeat_aux = Path(
        outdir_bootstrap_repeat.format(
            groups_selected=groups_selected,
            n_resamples=n_resamples,
            downsample_factor=downsample_factor,
            seed=seed,
        )
    )

    if not outdir_bootstrap_repeat_aux.exists():
        print("  -> bootstrap file missing, skipping")
        continue

    with open(outdir_bootstrap_repeat_aux, "rb") as f:
        data_loaded = pickle.load(f)
    print(f"Loading bootstrap: {outdir_bootstrap_repeat_aux}")

    # metadata
    groups_selected = data_loaded["groups_selected"]
    group_data = data_loaded["group_data"]
    ranges = data_loaded["ranges"]
    percentiles_ = data_loaded["percentiles_"]

    # bootstrap results
    ci_low_repeat = data_loaded["ci_low_repeat"]
    ci_high_repeat = data_loaded["ci_high_repeat"]

    print(f"Group data keys: {group_data.keys()}")

    # ---------- one figure per groups_selected, subplots per segment ----------
    seg_names = list(ranges.keys())
    n_seg = len(seg_names)

    fig, axes = plt.subplots(
        1,
        n_seg,
        figsize=(6 * n_seg, 5),
        sharex=True,
        sharey=True,
    )
    if n_seg == 1:
        axes = [axes]

    # fixed color map: label (female, male, …) -> color from AGE_CONTRAST_PALETTE
    color_map = make_age_contrast_color_map(group_data.keys(), groups_selected)

    for ax, seg_name in zip(axes, seg_names, strict=False):
        plotted_any = False

        for gt in ci_low_repeat[seg_name].keys():
            # ----- CASE A: pure age grouping: keys are '2M', '4M' -----
            if groups_selected == "age":
                if gt != "4M":
                    continue  # only contrast 4M vs 2M

                gt_4m = "4M"
                gt_2m = "2M"

                if gt_2m not in ci_low_repeat[seg_name]:
                    print(
                        f"Skipping pair: {gt_4m} vs {gt_2m} (missing CI for seg={seg_name})"
                    )
                    continue

                label = "4M-2M"  # must match key in color_map
            # ----- CASE B: age + other factors: keys are tuples -----
            else:
                # e.g. gt = ('female', '2M', 'wt') or ('female', '4M', 'wt')
                if not isinstance(gt, tuple) or "4M" not in gt:
                    continue

                age_idx = gt.index("4M")
                gt_4m = gt

                gt2_list = list(gt)
                gt2_list[age_idx] = "2M"
                gt_2m = tuple(gt2_list)

                if gt_2m not in ci_low_repeat[seg_name]:
                    print(
                        f"Skipping pair: {gt_4m} vs {gt_2m} (missing CI for seg={seg_name})"
                    )
                    continue

                # label: all factors except age, consistent with color_map
                label = age_contrast_label(gt_4m)  # e.g. "female | wt"

            # ---- color for this label ----
            color = color_map[label]

            # ----- compute diff CI: always 4M – 2M from here -----
            ci_low_4m = ci_low_repeat[seg_name][gt_4m]
            ci_low_2m = ci_low_repeat[seg_name][gt_2m]
            ci_high_4m = ci_high_repeat[seg_name][gt_4m]
            ci_high_2m = ci_high_repeat[seg_name][gt_2m]

            ci_low_diff = ci_low_4m - ci_low_2m
            ci_high_diff = ci_high_4m - ci_high_2m

            # low diff + high diff + band, all with the same explicit color
            ax.plot(
                percentiles_,
                ci_low_diff,
                label=label,
                color=color,
                alpha=0.7,
            )
            ax.plot(
                percentiles_,
                ci_high_diff,
                color=color,
                alpha=0.7,
            )
            ax.fill_between(
                percentiles_,
                ci_low_diff,
                ci_high_diff,
                color=color,
                alpha=0.3,
            )

            plotted_any = True

        # ----- common formatting -----
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(seg_name)
        ax.set_xlabel("Percentiles")
        ax.set_ylim(-0.2, 0.2)
        ax.set_xlim(0, 100)
        ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
        ax.tick_params(axis="both", labelsize=15)

        if ax is axes[0]:
            ax.set_ylabel("Speed Difference (4M - 2M)")

    # -------- single legend for whole figure --------
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles, strict=False))

    fig.legend(
        uniq.values(),
        uniq.keys(),
        title="Age contrast" if groups_selected == "age" else "Group (non-age factors)",
        loc="center right",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10,
        frameon=False,
    )

    fig.suptitle(f"dFC acceleration low/high diff 4M-2M   groups={groups_selected}")
    plt.tight_layout(rect=[0.02, 0.02, 0.85, 0.95])

    plt.savefig(
        acceleration_folder
        / f"ci_diff_band_4M_minus_2M_{seg_name}_{groups_selected}.png"
    )
    plt.show()


# %%
def sex_contrast_label(gt_male):
    """
    Build the label used for male–female contrasts.
    For tuple keys: drop the sex ('male'/'female') and join other factors.
    For pure sex: just return 'male-female'.
    """
    if isinstance(gt_male, str):
        return "male-female"

    parts = [str(v) for v in gt_male if v not in ("male", "female")]
    return " | ".join(parts) if parts else "all"


def make_sex_contrast_color_map(group_keys, groups_selected: str) -> dict[str, str]:
    """
    Map each non-sex label → a fixed color from AGE_CONTRAST_PALETTE.
    Mirrors make_age_contrast_color_map, but for sex contrasts.
    """
    example_key = next(iter(group_keys))

    # pure sex: only one contrast
    if groups_selected == "sex" and isinstance(example_key, str):
        return {"male-female": AGE_CONTRAST_PALETTE[0]}

    labels: list[str] = []
    for k in group_keys:
        if isinstance(k, str):
            # keys 'male'/'female' by themselves are handled in the loop
            continue
        if "male" not in k and "female" not in k:
            continue
        if isinstance(k, tuple) and "male" in k:
            lbl = sex_contrast_label(k)
            if lbl not in labels:
                labels.append(lbl)

    return {
        lbl: AGE_CONTRAST_PALETTE[i % len(AGE_CONTRAST_PALETTE)]
        for i, lbl in enumerate(labels)
    }


# ============================================================
#   CI DIFF BANDS: MALE - FEMALE AS SUBPLOTS PER SEGMENT
# ============================================================

# Age = '2M' | '4M'
# Sex = 'male' | 'female'

groups_list = [
    "sex",
    "age_sex",
    "age_sex_genotype",
    "age_sex_phenotype_oip",
    "age_sex_phenotype_nor",
    "sex_genotype",
    "sex_phenotype_oip",
    "sex_phenotype_nor",
]

for groups_selected in groups_list:
    print(f"\nProcessing grouping (sex diff): {groups_selected}")

    # folder load bootstrap (same pattern as 4M-2M block)
    outdir_bootstrap_repeat_aux = Path(
        outdir_bootstrap_repeat.format(
            groups_selected=groups_selected,
            n_resamples=n_resamples,
            downsample_factor=downsample_factor,
            seed=seed,
        )
    )
    if not outdir_bootstrap_repeat_aux.exists():
        print("  -> bootstrap missing")
        continue

    with open(outdir_bootstrap_repeat_aux, "rb") as f:
        data_loaded = pickle.load(f)
    print(f"Loading bootstrap: {outdir_bootstrap_repeat_aux}")

    # metadata
    groups_selected = data_loaded["groups_selected"]
    group_data = data_loaded["group_data"]
    ranges = data_loaded["ranges"]
    percentiles_ = data_loaded["percentiles_"]

    # bootstrap results
    ci_low_repeat = data_loaded["ci_low_repeat"]
    ci_high_repeat = data_loaded["ci_high_repeat"]

    print(f"Group data keys: {group_data.keys()}")

    seg_names = list(ranges.keys())
    n_seg = len(seg_names)

    fig, axes = plt.subplots(
        1,
        n_seg,
        figsize=(6 * n_seg, 5),
        sharex=True,
        sharey=True,
    )
    if n_seg == 1:
        axes = [axes]

    # fixed color map: label (non-sex factors) -> color
    color_map = make_sex_contrast_color_map(group_data.keys(), groups_selected)

    for ax, seg_name in zip(axes, seg_names, strict=False):
        plotted_any = False

        for gt in ci_low_repeat[seg_name].keys():
            # ----- CASE A: pure sex grouping: keys are 'male', 'female' -----
            if groups_selected == "sex":
                if gt != "male":
                    # only contrast male vs female
                    continue

                gt_m = "male"
                gt_f = "female"

                if gt_f not in ci_low_repeat[seg_name]:
                    print(
                        f"Skipping pair: {gt_m} vs {gt_f} "
                        f"(missing CI for seg={seg_name})"
                    )
                    continue

                label = "male-female"  # must match key in color_map

            # ----- CASE B: sex + other factors: keys are tuples -----
            else:
                # e.g. ('female', '2M', 'wt') or ('male', '2M', 'wt')
                if not isinstance(gt, tuple) or "male" not in gt:
                    # we only start the pair from the 'male' version
                    continue

                sex_idx = gt.index("male")
                gt_m = gt

                gt_f_list = list(gt)
                gt_f_list[sex_idx] = "female"
                gt_f = tuple(gt_f_list)

                if gt_f not in ci_low_repeat[seg_name]:
                    print(
                        f"Skipping pair: {gt_m} vs {gt_f} "
                        f"(missing CI for seg={seg_name})"
                    )
                    continue

                # label: all factors except sex, consistent with color_map
                label = sex_contrast_label(gt_m)  # e.g. "2M | wt"

            # ---- color for this label ----
            color = color_map[label]

            # ----- compute diff CI: always male – female from here -----
            ci_low_m = ci_low_repeat[seg_name][gt_m]
            ci_low_f = ci_low_repeat[seg_name][gt_f]
            ci_high_m = ci_high_repeat[seg_name][gt_m]
            ci_high_f = ci_high_repeat[seg_name][gt_f]

            ci_low_diff = ci_low_m - ci_low_f
            ci_high_diff = ci_high_m - ci_high_f

            # low diff + high diff + band, all with the same explicit color
            ax.plot(
                percentiles_,
                ci_low_diff,
                label=label,
                color=color,
                alpha=0.7,
            )
            ax.plot(
                percentiles_,
                ci_high_diff,
                color=color,
                alpha=0.7,
            )
            ax.fill_between(
                percentiles_,
                ci_low_diff,
                ci_high_diff,
                color=color,
                alpha=0.3,
            )

            plotted_any = True

        # ----- common formatting (match 4M–2M block) -----
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(seg_name)
        ax.set_xlabel("Percentiles")
        ax.set_ylim(-0.2, 0.2)
        ax.set_xlim(0, 100)
        ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
        ax.tick_params(axis="both", labelsize=15)

        if ax is axes[0]:
            ax.set_ylabel("Speed Difference (male - female)")

    # -------- single legend for whole figure --------
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles, strict=False))

    fig.legend(
        uniq.values(),
        uniq.keys(),
        title="Sex contrast" if groups_selected == "sex" else "Group (non-sex factors)",
        loc="center right",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10,
        frameon=False,
    )

    fig.suptitle(
        f"dFC acceleration low/high diff male-female   groups={groups_selected}"
    )
    plt.tight_layout(rect=[0.02, 0.02, 0.85, 0.95])

    plt.savefig(
        acceleration_folder
        / f"ci_diff_band_male_minus_female_{seg_name}_{groups_selected}.png"
    )
    plt.show()


# %%
# ============================================================
#   CI DIFF BANDS: WT - dKI AS SUBPLOTS PER SEGMENT
# ============================================================
def genotype_contrast_label(gt_ref, mut="dKI", ref="wt"):
    """
    Label for genotype contrast *wt - dKI*.

    gt_ref is the key for the reference genotype (wt, or tuple containing 'wt').
    For pure genotype: returns 'wt-dKI'.
    For tuples: drops genotype and joins the remaining factors with ' | '.
    """
    if isinstance(gt_ref, str):
        # pure genotype: just 'wt-dKI'
        return f"{ref}-{mut}"

    parts = [str(v) for v in gt_ref if v not in (ref, mut)]
    return " | ".join(parts) if parts else "all"


def make_genotype_contrast_color_map(
    group_keys,
    groups_selected: str,
    mut: str = "dKI",
    ref: str = "wt",
) -> dict[str, str]:
    """
    Map each non-genotype label → a fixed color from AGE_CONTRAST_PALETTE,
    for the contrast *wt - dKI*.

    IMPORTANT: we now enumerate labels from the *reference* (wt) keys,
    so the map contains exactly the same labels used in the plotting code.
    """
    example_key = next(iter(group_keys))

    # pure genotype: only one contrast 'wt-dKI'
    if groups_selected == "genotype" and isinstance(example_key, str):
        return {f"{ref}-{mut}": AGE_CONTRAST_PALETTE[0]}

    labels: list[str] = []
    for k in group_keys:
        if isinstance(k, str):
            # 'wt', 'dKI' handled in the plotting loop
            continue
        if ref not in k and mut not in k:
            continue
        # we take labels from the reference-genotype version (contains 'wt')
        if isinstance(k, tuple) and ref in k:
            lbl = genotype_contrast_label(k, mut=mut, ref=ref)
            if lbl not in labels:
                labels.append(lbl)

    return {
        lbl: AGE_CONTRAST_PALETTE[i % len(AGE_CONTRAST_PALETTE)]
        for i, lbl in enumerate(labels)
    }


# ============================================================
#   CI DIFF BANDS: wt - dKI AS SUBPLOTS PER SEGMENT
# ============================================================

groups_list = [
    "genotype",
    "age_genotype",
    "sex_genotype",
    "age_sex_genotype",
]

for groups_selected in groups_list:
    print(f"\nProcessing grouping (genotype diff wt-dKI): {groups_selected}")

    # Load bootstrap
    outdir_bootstrap_repeat_aux = Path(
        outdir_bootstrap_repeat.format(
            groups_selected=groups_selected,
            n_resamples=n_resamples,
            downsample_factor=downsample_factor,
            seed=seed,
        )
    )
    if not outdir_bootstrap_repeat_aux.exists():
        print("  -> bootstrap missing")
        continue

    with open(outdir_bootstrap_repeat_aux, "rb") as f:
        data_loaded = pickle.load(f)
    print(f"Loading bootstrap: {outdir_bootstrap_repeat_aux}")

    # metadata
    groups_selected = data_loaded["groups_selected"]
    group_data = data_loaded["group_data"]
    ranges = data_loaded["ranges"]
    percentiles_ = data_loaded["percentiles_"]

    # bootstrap results
    ci_low_repeat = data_loaded["ci_low_repeat"]
    ci_high_repeat = data_loaded["ci_high_repeat"]

    seg_names = list(ranges.keys())
    n_seg = len(seg_names)

    fig, axes = plt.subplots(
        1,
        n_seg,
        figsize=(6 * n_seg, 5),
        sharex=True,
        sharey=True,
    )
    if n_seg == 1:
        axes = [axes]

    # Color map for *wt-dKI* (same grammar as sex/age contrasts)
    color_map = make_genotype_contrast_color_map(
        group_data.keys(),
        groups_selected,
        mut="dKI",  # mutant
        ref="wt",  # reference
    )

    for ax, seg_name in zip(axes, seg_names, strict=False):
        plotted_any = False

        for gt in ci_low_repeat[seg_name].keys():
            # ---- CASE A: pure genotype (strings 'wt', 'dKI') ----
            if groups_selected == "genotype":
                if gt != "wt":
                    continue  # we start from reference = wt

                gt_ref = "wt"
                gt_mut = "dKI"

                if gt_mut not in ci_low_repeat[seg_name]:
                    print(f"Skipping: missing dKI for seg {seg_name}")
                    continue

                label = "wt-dKI"

            # ---- CASE B: tuple with multiple factors ----
            else:
                # only start from wt
                if not isinstance(gt, tuple) or "wt" not in gt:
                    continue

                geno_idx = gt.index("wt")
                gt_ref = gt

                # build dKI tuple
                gt_mut_list = list(gt)
                gt_mut_list[geno_idx] = "dKI"
                gt_mut = tuple(gt_mut_list)

                if gt_mut not in ci_low_repeat[seg_name]:
                    print(f"Skipping pair {gt_ref} vs {gt_mut}: missing")
                    continue

                label = genotype_contrast_label(gt_ref, mut="dKI", ref="wt")

            # ---- Color ----
            color = color_map[label]

            # ---- Compute diff: wt – dKI ----
            ci_low_ref = ci_low_repeat[seg_name][gt_ref]
            ci_low_mut = ci_low_repeat[seg_name][gt_mut]
            ci_high_ref = ci_high_repeat[seg_name][gt_ref]
            ci_high_mut = ci_high_repeat[seg_name][gt_mut]

            ci_low_diff = ci_low_ref - ci_low_mut
            ci_high_diff = ci_high_ref - ci_high_mut

            # ---- Plot band ----
            ax.plot(
                percentiles_,
                ci_low_diff,
                label=label,
                color=color,
                alpha=0.7,
            )
            ax.plot(
                percentiles_,
                ci_high_diff,
                color=color,
                alpha=0.7,
            )
            ax.fill_between(
                percentiles_,
                ci_low_diff,
                ci_high_diff,
                color=color,
                alpha=0.3,
            )

            plotted_any = True

        # Formatting
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(seg_name)
        ax.set_xlabel("Percentiles")
        ax.set_ylim(-0.2, 0.2)
        ax.set_xlim(0, 100)
        ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
        ax.tick_params(axis="both", labelsize=15)

        if ax is axes[0]:
            ax.set_ylabel("Speed Difference (wt - dKI)")

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles, strict=False))

    fig.legend(
        uniq.values(),
        uniq.keys(),
        title="Genotype contrast",
        loc="center right",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10,
        frameon=False,
    )

    fig.suptitle(f"dFC acceleration low/high diff wt - dKI   groups={groups_selected}")
    plt.tight_layout(rect=[0.02, 0.02, 0.85, 0.95])

    plt.savefig(
        acceleration_folder
        / f"ci_diff_band_wt_minus_dKI_{seg_name}_{groups_selected}.png"
    )
    plt.show()


# %%
# ========================== PERCENTILE TRACKS ==========================


# Generate speed percentiles per segment
def build_per_animal_flat_speed(
    speeds: list[np.ndarray],
    selected_windows: range,
    bins: int,
    hist_range: tuple[float, float],
) -> np.ndarray:
    """Build per-animal normalized histograms over selected windows.

    speeds:
        list S where S[j][i] is 1D np.array of samples for animal i at window j.

    selected_windows: range of window indices to include.
    Returns H where H[i] is normalized histogram for animal i over selected windows.
    """
    n_animals = len(speeds[0])
    flat_speeds_per_segment = np.zeros((n_animals,), dtype=object)
    for i in range(n_animals):
        # Pool samples for animal i over selected windows
        flat_i = (
            np.concatenate([speeds[j][i].ravel() for j in selected_windows])
            if selected_windows
            else np.array([], dtype=float)
        )
        flat_speeds_per_segment[i] = flat_i
    flat_speeds_per_segment = np.vstack(flat_speeds_per_segment)
    return flat_speeds_per_segment


if dataset_name == "julien":
    nor_index = cog_data["index_NOR"].values
elif dataset_name == "ines":
    nor_index = df_long["ro24h"].values

percentiles_ = np.linspace(0, 100, 100)
n_windows = len(time_windows_range)
flat_speeds_per_segment_i = build_per_animal_flat_speed(
    speeds,
    range(0, n_windows - 1),
    BINS_HIST,
    hist_range=(all_speeds_min, all_speeds_max),
)
# speeds_percentile_per_segment = np.percentile(
speeds_percentile_per_segment = np.percentile(
    flat_speeds_per_segment_i, q=percentiles_, axis=1
)


speeds_percentile_per_segment = {}
for seg_name, w_range in ranges.items():
    flat_speeds_per_segment_i = build_per_animal_flat_speed(
        speeds, w_range, BINS_HIST, hist_range=(all_speeds_min, all_speeds_max)
    )
    print(
        f"[INFO] flat_speeds_per_segment_i shape for segment {seg_name}:",
        flat_speeds_per_segment_i.shape,
    )
    speeds_percentile_per_segment[seg_name] = np.percentile(
        flat_speeds_per_segment_i, q=percentiles_, axis=1
    )

    # %%


# %% ========================== PLOTS (unchanged style) ==========================
# 1) Per-animal curves
plt.figure(figsize=(8, 6))
for i in range(n_animals):
    mean_speeds = [
        per_window_animal_means[j][i] for j in range(len(time_windows_range))
    ]
    genotype = next(
        (g for g, idx_list in group_genotype.items() if i in idx_list), "Unknown"
    )
    treatment = next(
        (t for t, idx_list in group_treatment.items() if i in idx_list), "Unknown"
    )
    color = combo_color(genotype, treatment)
    plt.plot(time_windows_range, mean_speeds, color=color, alpha=0.3)
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
if save_fig:
    outpath_fig_aux = savedir_dfc_speed_per_animal.format(
        n_animals=n_animals, regions=regions, total_tr=total_tr
    )
    plt.savefig(outpath_fig_aux, dpi=300)
    print(f"[INFO] Figure saved to: {outpath_fig_aux}")
# %%
# 2) Group curve (mean of per-animal medians)
fig2, ax2 = plt.subplots(figsize=(8, 6))
plot_group_median_vs_window(ax2, time_windows_range, group_data, speeds)
if save_fig:
    outpath_fig2 = savedir_dfc_speed_group_median_vs_window.format(
        n_animals=n_animals, regions=regions, total_tr=total_tr
    )
    fig2.savefig(outpath_fig2, dpi=300)
    print(f"[INFO] Figure saved to: {outpath_fig2}")

# %%
# 3) Percentile tracks per group (1,5,median,95,99) on 5 subplots
fig, axs = plt.subplots(2, 3, figsize=(11, 8), sharex=True)
axes = axs.ravel()
titles = ["1st pct", "5th pct", "Median", "95th pct", "99th pct"]
for (genotype, treatment), indices in group_data.items():
    print(f"[INFO] Processing group: {genotype}, {treatment}")
    color = combo_color(genotype, treatment)
    s1, s5, sm, s95, s99 = [], [], [], [], []
    for j in range(len(time_windows_range)):
        gflat = (
            np.concatenate([speeds[j][i].ravel() for i in indices])
            if len(indices)
            else np.array([], dtype=float)
        )
        p = robust_percentiles(gflat, qs=(1, 5, 95, 99))
        s1.append(p[1])
        s5.append(p[5])
        s95.append(p[95])
        s99.append(p[99])
        sm.append(float(np.nanmedian(gflat)) if gflat.size else np.nan)
    axes[0].plot(
        time_windows_range,
        s1,
        ".-",
        alpha=0.6,
        color=color,
        label=combo_label(genotype, treatment),
    )
    axes[1].plot(time_windows_range, s5, ".-", alpha=0.6, color=color)
    axes[2].plot(time_windows_range, sm, ".-", alpha=0.6, color=color)
    axes[3].plot(time_windows_range, s95, ".-", alpha=0.6, color=color)
    axes[4].plot(time_windows_range, s99, ".-", alpha=0.6, color=color)
for ax, t in zip(axes[:5], titles, strict=False):
    ax.set_title(f"dFC speed {t}")
for ax in axes:
    ax.grid(alpha=0.2)
axes[3].set_xlabel("Time Window Size")
axes[4].set_xlabel("Time Window Size")
axes[2].set_ylabel("dFC Speed")
axes[0].legend(ncol=1, fontsize=10)
fig.delaxes(axes[5])
fig.tight_layout()
if save_fig:
    outpath_fig3 = savedir_dfc_speed_percentiles_vs_window.format(
        n_animals=n_animals, regions=regions, total_tr=total_tr
    )
    fig.savefig(outpath_fig3, dpi=300)
    print(f"[INFO] Figure saved to: {outpath_fig3}")

# %%
# 4) CDF across windows
plt.figure(figsize=(7, 5))
plt.title("Cumulative Distribution of dFC Speeds across Time Windows")
plt.plot(time_windows_range, pooled_speeds_cdf, color="orange", lw=2, alpha=0.8)
plt.axvline(
    x=time_windows_range[i_half],
    color="red",
    linestyle="--",
    label="Median Window Size",
    alpha=0.5,
)
plt.axvline(
    x=time_windows_range[i_third],
    color="green",
    linestyle="--",
    label="1/3 Window Size",
    alpha=0.5,
)
plt.axvline(
    x=time_windows_range[i_two_third],
    color="blue",
    linestyle="--",
    label="2/3 Window Size",
    alpha=0.5,
)
plt.axhline(y=0.5, color="red", linestyle="--", alpha=0.5)
plt.axhline(y=1 / 3, color="green", linestyle="--", alpha=0.5)
plt.axhline(y=2 / 3, color="blue", linestyle="--", alpha=0.5)
plt.xlabel("Time Window Size")
plt.ylabel("Cumulative Frequency")
step = max(1, len(time_windows_range) // 12)
plt.xticks(time_windows_range[::step])
plt.legend()
plt.tight_layout()

if save_fig:
    outpath_fig4 = savedir_dfc_speed_cdf_windows.format(
        n_animals=n_animals, regions=regions, total_tr=total_tr
    )
    plt.savefig(outpath_fig4, dpi=300)
    print(f"[INFO] Figure saved to: {outpath_fig4}")
# %%
# 5) Example: pooled histograms (all/short/mid/long)
all_speeds_hist, bin_edge = hist_prob(
    all_speeds_flat, BINS_HIST, (all_speeds_min, all_speeds_max)
)
if POOL_SPLIT == "half":
    short_speeds_flat = flatten_windows(speeds, 0, i_half)
    long_speeds_flat = flatten_windows(speeds, i_half, len(speeds))
elif POOL_SPLIT == "third":
    short_speeds_flat = flatten_windows(speeds, 0, i_third)
    mid_speeds_flat = flatten_windows(speeds, i_third, i_two_third)
    long_speeds_flat = flatten_windows(speeds, i_two_third, len(speeds))

plt.figure(figsize=(7, 5))
plt.title("Pooled Speed (all windows pooled)")
plt.plot(
    bin_edge[:-1],
    all_speeds_hist,
    color="dodgerblue",
    lw=2,
    alpha=0.8,
    label="all animals",
)
if POOL_SPLIT == "half":
    plt.plot(
        bin_edge[:-1],
        hist_prob(short_speeds_flat, BINS_HIST, (all_speeds_min, all_speeds_max))[0],
        color="orange",
        lw=2,
        alpha=0.8,
        label="short windows",
    )
    plt.plot(
        bin_edge[:-1],
        hist_prob(long_speeds_flat, BINS_HIST, (all_speeds_min, all_speeds_max))[0],
        color="green",
        lw=2,
        alpha=0.8,
        label="long windows",
    )
elif POOL_SPLIT == "third":
    plt.plot(
        bin_edge[:-1],
        hist_prob(short_speeds_flat, BINS_HIST, (all_speeds_min, all_speeds_max))[0],
        color="orange",
        lw=2,
        alpha=0.8,
        label="short windows",
    )
    plt.plot(
        bin_edge[:-1],
        hist_prob(mid_speeds_flat, BINS_HIST, (all_speeds_min, all_speeds_max))[0],
        color="purple",
        lw=2,
        alpha=0.8,
        label="mid windows",
    )
    plt.plot(
        bin_edge[:-1],
        hist_prob(long_speeds_flat, BINS_HIST, (all_speeds_min, all_speeds_max))[0],
        color="green",
        lw=2,
        alpha=0.8,
        label="long windows",
    )
plt.legend()
plt.xlabel("Speed")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

if save_fig:
    outpath_fig5 = savedir_pooled_speed_hist_bins.format(
        BINS_HIST=BINS_HIST, n_animals=n_animals, regions=regions, total_tr=total_tr
    )
    plt.savefig(outpath_fig5, dpi=300)
    print(f"[INFO] Figure saved to: {outpath_fig5}")
# %%
# 6) (Optional) Group histograms per segment (per-animal mean & pooled) — no extra saving
PLOT_GROUP_HISTS = True
if PLOT_GROUP_HISTS:
    for seg_name, seg_range in ranges.items():
        aux_range = (
            f"{time_windows_range[seg_range[0]]}-{time_windows_range[seg_range[-1]]} tr"
        )
        print(f"[INFO] Plotting group histograms for segment: {seg_name} {aux_range}")
        # pooled hist by group
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
        plot_group_histograms(
            ax[0],
            centers,
            pooled_group_hists_by_segment[seg_name],
            f"Pooled {seg_name}: {aux_range}",
            ylog=False,
        )
        plot_group_histograms(
            ax[1], centers, pooled_group_hists_by_segment[seg_name], "", ylog=True
        )
        ymin, ymax = ax[1].get_ylim()
        ax[1].set_ylim(bottom=max(1e-5, ymin), top=ymax)
        fig.tight_layout()
        plt.legend()

    if save_fig:
        outpath_fig6 = savedir_pooled_group_hists.format(
            POOL_SPLIT=POOL_SPLIT,
            BINS_HIST=BINS_HIST,
            n_animals=n_animals,
            regions=regions,
            total_tr=total_tr,
        )
        fig.savefig(outpath_fig6, dpi=300)
        print(f"[INFO] Figure saved to: {outpath_fig6}")

plt.show()
# %%
# ---- optional save of data (one NPZ per segment + a JSON meta)

if SAVE_GROUP_HISTS:
    np.savez_compressed(
        outdir_save_group_hists,
        centers=centers,
        **{f"{g}__{t}": v for (g, t), v in group_means.items()},
    )


# %% ========================== SAVE OPTIONS ==========================
def make_meta(
    dataset: str,
    pool_split: str,
    time_windows_range: Sequence[int],
    ranges: dict[str, range],
    groups: dict[tuple[str, str], Sequence[int]],
    bins_hist: int,
    seed: int | None = None,
) -> dict:
    return {
        "dataset": dataset,
        "pool_split": pool_split,
        "time_windows_range": list(map(int, time_windows_range)),
        "ranges": {k: [int(x) for x in v] for k, v in ranges.items()},
        "groups": {f"{k[0]}__{k[1]}": [int(i) for i in v] for k, v in groups.items()},
        "bins_hist": int(bins_hist),
        "rng_seed": int(seed) if seed is not None else None,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }


# ---------- Option A: NPZ Pack ----------
def save_npz_pack(
    base_dir: Path,
    dataset: str,
    meta: dict,
    per_window_animal_means: list[np.ndarray],
    H_per_segment: dict[str, np.ndarray],
    edges: np.ndarray,
    pooled_group_hists_by_segment: dict[str, dict[tuple[str, str], np.ndarray]],
):
    out_dir = (
        base_dir
        / "bootstrap_packs"
        / f"{dataset}_{meta['pool_split']}_bins{meta['bins_hist']}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # per-window per-animal means
    np.savez_compressed(
        out_dir / "per_window_animal_means.npz",
        **{
            f"w{j}": per_window_animal_means[j]
            for j in range(len(per_window_animal_means))
        },
    )
    # histograms per segment
    for seg, H in H_per_segment.items():
        np.savez_compressed(out_dir / f"per_animal_hists__{seg}.npz", H=H, edges=edges)
    # pooled group hists per segment
    for seg, d in pooled_group_hists_by_segment.items():
        np.savez_compressed(
            out_dir / f"pooled_group_hists__{seg}.npz",
            **{f"{g[0]}__{g[1]}": v for g, v in d.items()},
        )


# ---------- Option B: Parquet Long-Form ----------
def save_parquet_longform(
    base_dir: Path,
    dataset: str,
    meta: dict,
    centers: np.ndarray,
    H_per_segment: dict[str, np.ndarray],
    group_means_by_segment: dict[str, dict[tuple[str, str], np.ndarray]],
):
    if pd is None:
        print("[WARN] pandas not available; skipping parquet save.")
        return
    out_dir = (
        base_dir
        / "bootstrap_packs"
        / f"{dataset}_{meta['pool_split']}_bins{meta['bins_hist']}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Per-animal histograms (long form): one row per (segment, animal, bin)
    rows = []
    for seg, H in H_per_segment.items():
        n_animals, n_bins = H.shape
        for i_an in range(n_animals):
            rows.extend(
                [
                    {
                        "segment": seg,
                        "animal": int(i_an),
                        "bin": int(k),
                        "center": float(centers[k]),
                        "p": float(H[i_an, k]),
                    }
                    for k in range(n_bins)
                ]
            )
    df_anim = pd.DataFrame(rows)
    df_anim.to_parquet(out_dir / "per_animal_hists.parquet", index=False)

    # Group-average histograms (long form): one row per (segment, group, bin)
    rows = []
    for seg, d in group_means_by_segment.items():
        for (g, t), hist in d.items():
            for k, p in enumerate(hist):
                rows.append(
                    {
                        "segment": seg,
                        "group": f"{g}__{t}",
                        "bin": int(k),
                        "center": float(centers[k]),
                        "p": float(p),
                    }
                )
    df_group = pd.DataFrame(rows)
    df_group.to_parquet(out_dir / "group_mean_hists.parquet", index=False)


# ---------- Execute selected save modes ----------
meta = make_meta(
    dataset,
    POOL_SPLIT,
    time_windows_range,
    ranges,
    group_data,
    BINS_HIST,
    RNG_SEED,
)
base_dir = Path(paths["speed"])

if SAVE_MODE.get("npz_pack", False):
    save_npz_pack(
        base_dir,
        dataset,
        meta,
        per_window_animal_means,
        H_per_segment,
        edges,
        pooled_group_hists_by_segment,
    )

if SAVE_MODE.get("parquet", False):
    save_parquet_longform(
        base_dir, dataset, meta, centers, H_per_segment, group_means_by_segment
    )

print("[INFO] Save completed. Modes:", SAVE_MODE)


# # --- Drop-in replacement for the two plotting loops (with optional saving) ---
# out_dir = Path(paths["speed"]) / "bootstrap_packs" / f"{dataset}_{POOL_SPLIT}_bins{BINS_HIST}"
# out_dir.mkdir(parents=True, exist_ok=True)
# SAVE_GROUP_HISTS = True  # toggle saving of group histograms per segment

# # 1) Per-animal MEAN histograms by group, per segment
# for seg_name, w_range in ranges.items():
#     print(f"Building per-animal histograms for segment '{seg_name}' with windows {list(w_range)}")
#     H_per_animal = build_per_animal_normalized_hists(
#         speeds, w_range, BINS_HIST, (all_speeds_min, all_speeds_max)
#     )

#     # average over animals per group -> a histogram curve per group
#     group_means: dict[tuple[str, str], np.ndarray] = {}
#     for gt, idxs in group_data.items():
#         print(f"Processing group: {gt} with indices: {idxs}")
#         group_means[gt] = np.mean(H_per_animal[idxs], axis=0) if len(idxs) else np.zeros(BINS_HIST)

#     # ---- plot
#     fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
#     plot_group_histograms(ax[0], centers, group_means, f"Per-animal mean hists ({seg_name})", ylog=False)
#     plot_group_histograms(ax[1], centers, group_means, f"Per-animal mean hists (log, {seg_name})", ylog=True)
#     # keep log strictly positive
#     ymin, ymax = ax[1].get_ylim()
#     ax[1].set_ylim(bottom=max(1e-5, ymin), top=ymax)
#     fig.tight_layout()

#     # ---- optional save of data (one NPZ per segment + a JSON meta)
#     if SAVE_GROUP_HISTS:
#         np.savez_compressed(
#             out_dir / f"group_mean_hists__{seg_name}.npz",
#             centers=centers,
#             **{f"{g}__{t}": v for (g, t), v in group_means.items()},
#         )

# # 2) POOLED histograms by group, per segment (flatten all values over animals & windows)
# for seg_name, w_range in ranges.items():
#     pooled_group: dict[tuple[str, str], np.ndarray] = {}
#     for gt, idxs in group_data.items():
#         pooled_group[gt] = pooled_group_histogram(
#             animal_speeds, idxs, w_range, BINS_HIST, (all_speeds_min, all_speeds_max)
#         )

#     # ---- plot
#     fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
#     plot_group_histograms(ax[0], centers, pooled_group, f"Pooled hist ({seg_name})", ylog=False)
#     plot_group_histograms(ax[1], centers, pooled_group, f"Pooled hist (log, {seg_name})", ylog=True)
#     ymin, ymax = ax[1].get_ylim()
#     ax[1].set_ylim(bottom=max(1e-5, ymin), top=ymax)
#     fig.tight_layout()

#     # ---- optional save of data
#     if SAVE_GROUP_HISTS:
#         np.savez_compressed(
#             out_dir / f"pooled_group_hists__{seg_name}.npz",
#             centers=centers,
#             **{f"{g}__{t}": v for (g, t), v in pooled_group.items()},
#         )

# plt.show()

# %%

# %%

import os

from scipy.stats import spearmanr


# --------------------------
# Helper: bootstrap Spearman correlation
# --------------------------
def bootstrap_spearman(x, y, n_resamples=1000, random_state=0):
    """Return mean Spearman rho and 95% CI via bootstrapping."""
    rng = np.random.default_rng(random_state)
    n = len(x)
    r_boot = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        r_boot[i], _ = spearmanr(x[idx], y[idx])
    return np.mean(r_boot), np.percentile(r_boot, [2.5, 97.5])


# --------------------------
# Helper: bootstrap difference between two correlations
# --------------------------
def bootstrap_diff(x1, y1, x2, y2, n_resamples=1000, random_state=0):
    rng = np.random.default_rng(random_state)
    n1, n2 = len(x1), len(x2)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx1 = rng.integers(0, n1, n1)
        idx2 = rng.integers(0, n2, n2)
        r1, _ = spearmanr(x1[idx1], y1[idx1])
        r2, _ = spearmanr(x2[idx2], y2[idx2])
        diffs[i] = r1 - r2
    return np.mean(diffs), np.percentile(diffs, [2.5, 97.5])


# %%
# Generate Δρ plots with bootstrapped CIs for all group pairs & segments
# --------------------------
# Main: Iterate over all group pairs
# --------------------------
savedir_corr_plots = paths["f_speed"] / "correlation_plots"
savedir_corr_plots.mkdir(parents=True, exist_ok=True)
results_dir = str(savedir_corr_plots)
os.makedirs(results_dir, exist_ok=True)

group_keys = list(group_data.keys())
pairs = list(combinations(group_keys, 2))

# Bootstrap Δρ plots
for speed_seg_name, speeds_ppsegment in speeds_percentile_per_segment.items():
    print(
        f"[INFO] Processing segment: {speed_seg_name} with shape {speeds_ppsegment.shape}"
    )

    for group_a, group_b in pairs:
        print(f"\n=== Comparing {group_a} vs {group_b} in pool {speed_seg_name} ===")

        idx_a = group_data[group_a]
        idx_b = group_data[group_b]

        # Skip if any group has too few animals
        if len(idx_a) < 3 or len(idx_b) < 3:
            print(f"⚠️ Skipping {group_a} vs {group_b} (too few animals)")
            continue

        diff_means, diff_ci_low, diff_ci_high = [], [], []

        for i in range(speeds_ppsegment.shape[0]):
            print("Processing percentile index:", i)
            y1 = speeds_ppsegment[i, idx_a]
            y2 = speeds_ppsegment[i, idx_b]
            x1 = nor_index[idx_a]
            x2 = nor_index[idx_b]
            mean_diff, (ci_low, ci_high) = bootstrap_diff(
                x1, y1, x2, y2, n_resamples=1000
            )
            diff_means.append(mean_diff)
            diff_ci_low.append(ci_low)
            diff_ci_high.append(ci_high)

        diff_means, diff_ci_low, diff_ci_high = map(
            np.array, (diff_means, diff_ci_low, diff_ci_high)
        )

        # Plot Δρ curve
        plt.figure(figsize=(10, 5))
        plt.plot(
            percentiles_,
            diff_means,
            lw=2,
            color="purple",
            label=f"{group_a} − {group_b}",
        )
        plt.fill_between(
            percentiles_, diff_ci_low, diff_ci_high, color="purple", alpha=0.3
        )
        plt.axhline(0, color="black", lw=1)

        # Highlight significant regions (CI excludes 0)
        plt.fill_between(
            percentiles_,
            diff_ci_low,
            diff_ci_high,
            where=(diff_ci_low > 0) | (diff_ci_high < 0),
            color="purple",
            alpha=0.2,
            label="Significant Δρ (95% CI excludes 0)",
        )

        plt.xlabel("Percentiles")
        plt.ylabel("Δ Spearman ρ (Group A − Group B)")
        plt.title(
            f"Difference in Spearman Correlation between {group_a} and {group_b}\nwith Bootstrapped 95% CI"
        )
        plt.legend()
        plt.tight_layout()

        # Save each figure automatically
        fname = f"delta_rho_{group_a[0]}_{group_a[1]}__vs__{group_b[0]}_{group_b[1]}_{speed_seg_name}.png".replace(
            "'", ""
        )
        plt.savefig(os.path.join(results_dir, fname), dpi=300)
        plt.close()

        print(f"✅ Saved plot → {fname}")


# %%
# Save summary CSV of Δρ results for all group pairs & segments


# Ensure results folder exists
resultdir_speed_delta_rho = paths["speed"] / "delta_rho_results"
resultdir_speed_delta_rho.mkdir(parents=True, exist_ok=True)

summary_data = []
for speed_seg_name, speeds_ppsegment in speeds_percentile_per_segment.items():
    print(
        f"[INFO] Processing segment: {speed_seg_name} with shape {speeds_ppsegment.shape}"
    )

    for group_a, group_b in combinations(group_data.keys(), 2):
        fname_base = f"{group_a[0]}_{group_a[1]}__vs__{group_b[0]}_{group_b[1]}_{speed_seg_name}".replace(
            "'", ""
        )
        filepath_png = os.path.join(
            resultdir_speed_delta_rho, f"delta_rho_{fname_base}.png"
        )
        filepath_npz = os.path.join(
            resultdir_speed_delta_rho, f"delta_rho_{fname_base}.npz"
        )

        # Skip if one group too small or not computed
        idx_a = group_data[group_a]
        idx_b = group_data[group_b]
        if len(idx_a) < 3 or len(idx_b) < 3:
            continue

        print(f"Processing Δρ for {group_a} vs {group_b}...")

        diff_means, diff_ci_low, diff_ci_high = [], [], []

        for i in range(speeds_ppsegment.shape[0]):
            y1 = speeds_ppsegment[i, idx_a]
            y2 = speeds_ppsegment[i, idx_b]
            x1 = nor_index[idx_a]
            x2 = nor_index[idx_b]
            mean_diff, (ci_low, ci_high) = bootstrap_diff(
                x1, y1, x2, y2, n_resamples=1000
            )
            diff_means.append(mean_diff)
            diff_ci_low.append(ci_low)
            diff_ci_high.append(ci_high)

        diff_means, diff_ci_low, diff_ci_high = map(
            np.array, (diff_means, diff_ci_low, diff_ci_high)
        )

        # Compute significance mask (CI excludes 0)
        sig_mask = (diff_ci_low > 0) | (diff_ci_high < 0)

        # Save to NPZ (for future replotting or analysis)
        np.savez(
            filepath_npz,
            percentiles=percentiles_,
            delta_rho_mean=diff_means,
            delta_rho_ci_low=diff_ci_low,
            delta_rho_ci_high=diff_ci_high,
            significant_mask=sig_mask,
            groupA=group_a,
            groupB=group_b,
        )

        # Summaries for CSV
        mean_delta = np.nanmean(diff_means)
        ci_global = (np.nanmin(diff_ci_low), np.nanmax(diff_ci_high))

        # Identify contiguous significant percentile ranges
        sig_ranges = []
        in_block = False
        start = None
        for i, val in enumerate(sig_mask):
            if val and not in_block:
                start = percentiles_[i]
                in_block = True
            elif not val and in_block:
                end = percentiles_[i - 1]
                sig_ranges.append(f"{start:.1f}-{end:.1f}")
                in_block = False
        if in_block:
            sig_ranges.append(f"{start:.1f}-{percentiles_[-1]:.1f}")
        sig_range_str = ", ".join(sig_ranges) if sig_ranges else "None"

        summary_data.append(
            {
                "Group A": f"{group_a[0]} {group_a[1]}",
                "Group B": f"{group_b[0]} {group_b[1]}",
                "Mean Δρ": f"{mean_delta:.3f}",
                "95% CI (min,max)": f"[{ci_global[0]:.3f}, {ci_global[1]:.3f}]",
                "Significant percentile ranges": sig_range_str,
                "NPZ file": os.path.basename(filepath_npz),
            }
        )

    # Build summary table and export to CSV
    summary_df = pd.DataFrame(summary_data)
    csv_path = os.path.join(results_dir, "delta_rho_summary.csv")
    summary_df.to_csv(csv_path, index=False)

    print(f"\n✅ Summary CSV saved to: {csv_path}")
    display(summary_df)

# %%


# --------------------------
# Helper: bootstrap Spearman correlation
# --------------------------
def bootstrap_spearman(x, y, n_resamples=1000, random_state=0):
    """Return mean Spearman rho and 95% CI via bootstrapping."""
    rng = np.random.default_rng(random_state)
    n = len(x)
    r_boot = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        r_boot[i], _ = spearmanr(x[idx], y[idx])
    return np.mean(r_boot), np.percentile(r_boot, [2.5, 97.5])


# --------------------------
# Group-wise Spearman correlations
# --------------------------
for speed_seg_name, speeds_ppsegment in speeds_percentile_per_segment.items():
    print(
        f"[INFO] Processing segment: {speed_seg_name} with shape {speeds_ppsegment.shape}"
    )

    group_results = {}

    for (
        gt,
        idxs,
    ) in group_data.items():  # e.g. {"WT": [0,1,...], "Mut": [24,...]}
        r_means, ci_lows, ci_highs, p_vals = [], [], [], []

        for i in range(speeds_ppsegment.shape[0]):
            y = speeds_ppsegment[i, idxs]
            x = nor_index[idxs]
            if np.std(y) == 0:
                r_means.append(np.nan)
                ci_lows.append(np.nan)
                ci_highs.append(np.nan)
                p_vals.append(np.nan)
                continue

            r, p = spearmanr(x, y)
            r_mean, (r_low, r_high) = bootstrap_spearman(x, y, n_resamples=1000)
            r_means.append(r_mean)
            ci_lows.append(r_low)
            ci_highs.append(r_high)
            p_vals.append(p)

        group_results[gt] = {
            "r_means": np.array(r_means),
            "ci_lows": np.array(ci_lows),
            "ci_highs": np.array(ci_highs),
            "p_vals": np.array(p_vals),
        }

    # --------------------------
    # Plot group-wise results with significance shading
    # --------------------------
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10.colors  # distinct colors for groups

    for idx, (gt, res) in enumerate(group_results.items()):
        color = colors[idx % len(colors)]
        plt.plot(
            percentiles_, res["r_means"], color=color, lw=2, label=f"{gt} (ρ mean)"
        )
        plt.fill_between(
            percentiles_, res["ci_lows"], res["ci_highs"], color=color, alpha=0.25
        )

        # Shade non-significant regions (p > 0.05)
        plt.fill_between(
            percentiles_,
            res["ci_lows"],
            res["ci_highs"],
            where=(res["p_vals"] > 0.05),
            color=color,
            alpha=0.1,
            label=f"{gt} p > 0.05",
        )

    plt.axhline(0, color="black", lw=1)
    plt.xlabel("Percentiles")
    plt.ylabel("Spearman Correlation (ρ)")
    plt.title(
        "Spearman Correlation - NOR vs Speed \nwith Bootstrapped 95% CI and p>0.05 Shading"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()
    savedir_spearman = savedir_corr_plots / f"spearman_correlation_{speed_seg_name}.png"
    plt.savefig(savedir_spearman, dpi=300)

# %%
