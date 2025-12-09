#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group-wise dFC speed distributions & bootstrap CIs
Works with both:
  - dataset_name = "ines"
  - dataset_name = "julien"
"""
#%%
from collections.abc import Iterable, Sequence
import dis
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np

# Optional for Parquet saving (not used here but kept for compatibility)
try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from scripts.dfc.dfc_compute import DATASET_DEFAULTS, _canonical_dataset
from shared_code.fun_loaddata import load_timeseries_bundle
from shared_code.fun_paths import get_paths
from shared_code.fun_utils import load_cognitive_data, set_figure_params


# must match the compute script
BASE_SPEED_SUBSETS = [
    "all",
    # "regions500",
    # "per_region",
    "dmn_touching",
    "1st_touching",
    "2nd_touching",
    "3rd_touching",
    "4th_touching",
    "lat_touching",
    "mem_touching",
    "sal_touching",
    "dmn_within",
    "1st_within",
    "2nd_within",
    "3rd_within",
    "4th_within",
    "lat_within",
    "mem_within",
    "sal_within",
]




import re

def discover_per_region_subset_labels(bootstrap_folder: Path) -> list[str]:
    """
    Look in the bootstrap dir for files of the form:

      bootstrap_downsample_repeat_subset_per_region_region-XXX_group_...

    and return the list of subset labels:
      ['per_region_region-AI', 'per_region_region-PL', ...]
    """
    labels = set()
    pattern = "bootstrap_downsample_repeat_subset_per_region_region-*_group_*.pkl"

    for p in bootstrap_folder.glob(pattern):
        m = re.search(r"subset_(per_region_region-[^_]+)_group_", p.name)
        if m:
            labels.add(m.group(1))

    return sorted(labels)


#%%


# =============================================================================
# ------------------------- DATASET SELECTION ---------------------------------
# =============================================================================

# dataset_name = "ines"
dataset_name = "julien"

dataset = _canonical_dataset(dataset_name)
cfg = DATASET_DEFAULTS[dataset]

# genotype reference / mutant labels per dataset (for contrasts)
if dataset_name == "ines":
    GENO_REF = "wt"
    GENO_MUT = "dKI"
elif dataset_name == "julien":
    GENO_REF = "WT"
    GENO_MUT = "Dp1Yey"
else:
    raise ValueError(f"Unknown dataset_name={dataset_name}")

# =============================================================================
# ------------------------- GROUP RECIPES -------------------------------------
# =============================================================================

# Recipes are in terms of *df_long* columns returned by make_long_cog

GROUP_RECIPES_INES = {
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
}

GROUP_RECIPES_JULIEN = {
    "genotype": ["genotype"],
    "treatment": ["treatment"],
    "genotype_treatment": ["genotype", "treatment"],
    # extend later if you want grp/index_NOR etc.
}


def get_group_recipes_for_dataset(ds_name: str) -> dict[str, list[str]]:
    if ds_name == "ines":
        return GROUP_RECIPES_INES
    if ds_name == "julien":
        return GROUP_RECIPES_JULIEN
    raise ValueError(f"Unknown dataset_name={ds_name}")


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


# =============================================================================
# ------------------------- BASIC LOAD HELPERS --------------------------------
# =============================================================================
def load_speed_stack(
    paths_speed_root: Path, time_windows_range: Sequence[int], n_animals: int, regions: int
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


def count_samples_per_window(speeds: list[np.ndarray]) -> np.ndarray:
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


def pretty_group_label(gt):
    if isinstance(gt, tuple):
        return " | ".join(str(x) for x in gt)
    return str(gt)


# =============================================================================
# ------------------------- COGNITIVE DATA HELPERS ----------------------------
# =============================================================================
def make_long_cog(cog_data: pd.DataFrame, ds_name: str) -> pd.DataFrame:
    """
    Standardize cognitive / grouping data.

    For 'julien':
      - one row per animal
      - columns: name, genotype, treatment, grp, index_NOR (if present)

    For 'ines':
      - long 2M/4M format with sex/age/phenotypes.
    """
    if ds_name == "julien":
        df = cog_data.copy()

        if "mouse" in df.columns:
            df = df.rename(columns={"mouse": "name"})

        cols_keep = [
            c
            for c in ["name", "genotype", "treatment", "grp", "index_NOR"]
            if c in df.columns
        ]
        df = df[cols_keep]

        for col in ["genotype", "treatment", "grp"]:
            if col in df.columns:
                df[col] = df[col].astype("category")

        return df

    elif ds_name == "ines":
        cols_common = ["Name", "Sexe", "Genotype", "Phenotype_OiP", "Phenotype_RO24h"]
        df2 = cog_data[cols_common + ["OiP_2M", "RO24h_2M", "TC_2M"]].copy()
        df4 = cog_data[cols_common + ["OiP_4M", "RO24h_4M", "TC_4M"]].copy()

        df2["Age"] = "2M"
        df4["Age"] = "4M"
        df2 = df2.rename(columns={"OiP_2M": "oip", "RO24h_2M": "ro24h", "TC_2M": "tc"})
        df4 = df4.rename(columns={"OiP_4M": "oip", "RO24h_4M": "ro24h", "TC_4M": "tc"})
        df = pd.concat([df2, df4], ignore_index=True)

        df["Sexe"] = df["Sexe"].map({"F": "female", "M": "male"}).fillna(df["Sexe"])

        for col in ["Sexe", "Age", "Genotype", "Phenotype_OiP", "Phenotype_RO24h"]:
            if col in df.columns:
                df[col] = df[col].astype("category")

        return df

    else:
        raise ValueError(f"Unknown dataset_name={ds_name}")


def group_indices(df: pd.DataFrame, by: Sequence[str]) -> dict:
    """
    Return a dict { group_key_tuple_or_scalar : np.ndarray[int] } of row indices.
    """
    if not by:
        return {"all": np.arange(len(df), dtype=int)}
    gb = df.groupby(list(by), sort=False)
    return {k: v.values for k, v in gb.groups.items()}


def get_group_data(cog_data: pd.DataFrame, ds_name: str, groups_selected: str):
    df_long = make_long_cog(cog_data, ds_name)
    recipes = get_group_recipes_for_dataset(ds_name)

    cols = recipes.get(groups_selected)
    if cols is None:
        raise ValueError(
            f"Unknown groups_selected='{groups_selected}' for dataset '{ds_name}'. "
            f"Choose from: {sorted(recipes.keys())}"
        )

    missing = [c for c in cols if c not in df_long.columns]
    if missing:
        raise ValueError(
            f"Grouping '{groups_selected}' needs columns {missing} "
            f"missing in df_long.columns={list(df_long.columns)}"
        )

    return group_indices(df_long, cols)


# =============================================================================
# ------------------------- CONTRAST LABEL HELPERS ----------------------------
# =============================================================================
def age_contrast_label(gt_4m):
    """Label for 4M–2M contrasts (INES only)."""
    if isinstance(gt_4m, str):
        return "4M-2M"
    parts = [str(v) for v in gt_4m if v not in ("2M", "4M")]
    return " | ".join(parts) if parts else "all"


def make_age_contrast_color_map(group_keys, groups_selected: str) -> dict[str, str]:
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


def sex_contrast_label(gt_male):
    """Label for male–female contrasts (INES only)."""
    if isinstance(gt_male, str):
        return "male-female"
    parts = [str(v) for v in gt_male if v not in ("male", "female")]
    return " | ".join(parts) if parts else "all"


def make_sex_contrast_color_map(group_keys, groups_selected: str) -> dict[str, str]:
    example_key = next(iter(group_keys))
    if groups_selected == "sex" and isinstance(example_key, str):
        return {"male-female": AGE_CONTRAST_PALETTE[0]}

    labels: list[str] = []
    for k in group_keys:
        if isinstance(k, str):
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


def genotype_contrast_label(gt_ref, mut: str, ref: str):
    """
    Label for genotype contrast *ref - mut*.

    For pure genotype: 'ref-mut'.
    For tuples: drop genotype and join remaining factors.
    """
    if isinstance(gt_ref, str):
        return f"{ref}-{mut}"
    parts = [str(v) for v in gt_ref if v not in (ref, mut)]
    return " | ".join(parts) if parts else "all"


def make_genotype_contrast_color_map(
    group_keys,
    groups_selected: str,
    mut: str,
    ref: str,
) -> dict[str, str]:
    example_key = next(iter(group_keys))
    if groups_selected == "genotype" and isinstance(example_key, str):
        return {f"{ref}-{mut}": AGE_CONTRAST_PALETTE[0]}

    labels: list[str] = []
    for k in group_keys:
        if isinstance(k, str):
            continue
        if ref not in k and mut not in k:
            continue
        if isinstance(k, tuple) and ref in k:
            lbl = genotype_contrast_label(k, mut=mut, ref=ref)
            if lbl not in labels:
                labels.append(lbl)

    return {
        lbl: AGE_CONTRAST_PALETTE[i % len(AGE_CONTRAST_PALETTE)]
        for i, lbl in enumerate(labels)
    }


# =============================================================================
# ------------------------- PATHS & DATA LOADING ------------------------------
# =============================================================================

save_fig = set_figure_params(True)

time_windows_range = np.arange(5, 100, 1)
POOL_SPLIT = "third"  # 'half' | 'third' | 'all'
BINS_HIST = 200
n_resamples = 10_000
downsample_factor = 10
seed = 42

paths = get_paths(
    dataset_name=dataset,
    timecourse_folder=cfg["timecourse_folder"],
    cognitive_data_file=cfg["cognitive_data_file"],
    anat_labels_file=cfg["anat_labels_file"],
)







speed_root = Path(paths["speed"])
preprocessed_root = Path(paths["preprocessed"])

loaddir_ts_meta = preprocessed_root / "ts_and_meta_2m4m.npz"
loaddir_cog_data = str(
    preprocessed_root
    / "cog_data_filtered_animals_{n_animals}_regions_{regions}_tr_{total_tr}.csv"
)
loaddir_speed = str(
    speed_root / "all/speed_win{w}_lag1_tau2_animals_{n_animals}_regions_{regions}.npz"
)

bootstrap_folder = paths["speed"] / "bootstrap"
bootstrap_folder.mkdir(parents=True, exist_ok=True)

# same pattern as in the compute script
outdir_bootstrap_repeat = str(
    paths["speed"]
    / "bootstrap"
    / "bootstrap_downsample_repeat_subset_{subset}_group_{groups_selected}"
      "_nresamples_{n_resamples}_downsample_factor_{downsample_factor}_seed_{seed}.pkl"
)


distribution_folder = paths["f_speed"] / "distribution"
distribution_folder.mkdir(parents=True, exist_ok=True)

acceleration_folder = paths["f_speed"] / "acceleration"
acceleration_folder.mkdir(parents=True, exist_ok=True)

# --- Load timeseries/meta ---
bundle = load_timeseries_bundle(loaddir_ts_meta)
n_animals = bundle.n_animals
total_tr = bundle.total_tr
regions = bundle.n_regions

# --- Load cognitive data ---
cog_data = load_cognitive_data(
    loaddir_cog_data.format(n_animals=n_animals, regions=regions, total_tr=total_tr)
)

# --- Load speed data ---
speeds = load_speed_stack(
    loaddir_speed,
    time_windows_range,
    n_animals=n_animals,
    regions=regions,
)

n_windows = len(speeds)

ALL_PER_REGION_SUBSETS = discover_per_region_subset_labels(bootstrap_folder)
print("[INFO] per_region subsets found:", ALL_PER_REGION_SUBSETS)


SPEED_SUBSETS = BASE_SPEED_SUBSETS + ALL_PER_REGION_SUBSETS

#%%
# splits & histogram grid
counts = count_samples_per_window(speeds)
pooled_speeds_cdf = (
    np.cumsum(counts) / np.sum(counts) if counts.sum() else np.zeros_like(counts)
)
i_third, i_half, i_two_third = cdf_split_indices(speeds)
ranges = select_windows(
    POOL_SPLIT, len(time_windows_range), i_third, i_half, i_two_third
)
all_speeds_flat = flatten_windows(speeds, 0, len(speeds))
all_speeds_min, all_speeds_max = global_min_max([all_speeds_flat])
edges = np.linspace(all_speeds_min, all_speeds_max, BINS_HIST + 1)
centers = 0.5 * (edges[:-1] + edges[1:])


# =============================================================================
# ------------------------- GROUP LISTS PER DATASET ---------------------------
# =============================================================================

recipes = get_group_recipes_for_dataset(dataset_name)

if dataset_name == "ines":
    groups_dist = [
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
    groups_age_diff = [
        "age",
        "age_sex",
        "age_genotype",
        "age_phenotype_nor",
        "age_phenotype_oip",
        "age_sex_genotype",
        "age_sex_phenotype_oip",
        "age_sex_phenotype_nor",
    ]
    groups_sex_diff = [
        "sex",
        "age_sex",
        "age_sex_genotype",
        "age_sex_phenotype_oip",
        "age_sex_phenotype_nor",
        "sex_genotype",
        "sex_phenotype_oip",
        "sex_phenotype_nor",
    ]
    groups_genotype_diff = [
        "genotype",
        "age_genotype",
        "sex_genotype",
        "age_sex_genotype",
    ]
elif dataset_name == "julien":
    groups_dist = [
        # "genotype",
        # "treatment",
        "genotype_treatment",
    ]
    groups_age_diff = []   # not defined for julien
    groups_sex_diff = []   # not defined for julien
    groups_genotype_diff = [
        # "genotype",
        "genotype_treatment",
    ]
else:
    raise ValueError(f"Unknown dataset_name={dataset_name}")

groups_ci = groups_dist  # same set for CI envelopes
#%%

# =============================================================================
# --------------------- PLOT GROUP MEAN HISTOGRAMS ----------------------------
# =============================================================================
for subset in SPEED_SUBSETS:
    print(f"\n=== [MEAN HIST] Subset: {subset} ===")

    # where to save plots for this subset
    distribution_folder = paths["f_speed"] / subset / "distribution"
    dist_folder = paths["f_speed"] / "dist"
    dist_folder.mkdir(parents=True, exist_ok=True)
    distribution_folder.mkdir(parents=True, exist_ok=True)

    for groups_selected in groups_dist:
        print(f"  [MEAN HIST] Processing grouping: {groups_selected}")

        outdir_bootstrap_repeat_aux = Path(
            outdir_bootstrap_repeat.format(
                subset=subset,
                groups_selected=groups_selected,
                n_resamples=n_resamples,
                downsample_factor=downsample_factor,
                seed=seed,
            )
        )

        if not outdir_bootstrap_repeat_aux.exists():
            print(f"    -> bootstrap missing for subset={subset}, group={groups_selected}")
            continue

        with open(outdir_bootstrap_repeat_aux, "rb") as f:
            data_loaded = pickle.load(f)

        ranges = data_loaded["ranges"]
        pooled_group_hists_by_segment = data_loaded["pooled_group_hists_by_segment"]
        centers = data_loaded["centers"]   # use from file to be safe

        seg_names = list(ranges.keys())
        n_seg = len(seg_names)

        fig, axes = plt.subplots(
            2,
            n_seg,
            figsize=(6 * n_seg, 8),
            sharex=True,
        )
        if n_seg == 1:
            axes = np.array([axes]).reshape(2, 1)

        plt.rcParams["axes.prop_cycle"] = plt.cycler(
            color=plt.cm.tab20(np.linspace(0, 1, 20))
        )

        for col, seg_name in enumerate(seg_names):
            group_means = pooled_group_hists_by_segment[seg_name]

            # linear y
            ax_lin = axes[0, col]
            ax_lin.set_title(f"{seg_name} (linear)", fontsize=14)
            for gt, mean_hist in group_means.items():
                ax_lin.plot(
                    centers, mean_hist, lw=1.2, alpha=0.8, label=pretty_group_label(gt)
                )
            ax_lin.set_xlabel("Speed")
            ax_lin.set_ylabel("Density")
            ax_lin.grid(True, which="both", ls="--", lw=0.4)

            # log y
            ax_log = axes[1, col]
            ax_log.set_title(f"{seg_name} (log)", fontsize=14)
            for gt, mean_hist in group_means.items():
                ax_log.plot(centers, mean_hist, lw=1.2, alpha=0.8)
            ax_log.set_xlabel("Speed")
            ax_log.set_ylabel("Density (log)")
            ax_log.set_yscale("log")
            ax_log.grid(True, which="both", ls="--", lw=0.4)

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

        plt.tight_layout(rect=[0.02, 0.02, 0.88, 0.95])

        plt.savefig(
            dist_folder
            / f"group_means_dist_subset_{subset}_group_{groups_selected}_{POOL_SPLIT}_bins{BINS_HIST}.png",
            bbox_inches="tight",
            dpi=150,
        )
        plt.show()
#%%
# =============================================================================

# for groups_selected in groups_dist:
#     print(f"\n[MEAN HIST] Processing grouping: {groups_selected}")

#     outdir_bootstrap_repeat_aux = Path(
#         outdir_bootstrap_repeat.format(
#             groups_selected=groups_selected,
#             n_resamples=n_resamples,
#             downsample_factor=downsample_factor,
#             seed=seed,
#         )
#     )

#     if not outdir_bootstrap_repeat_aux.exists():
#         print("  -> bootstrap missing")
#         continue

#     with open(outdir_bootstrap_repeat_aux, "rb") as f:
#         data_loaded = pickle.load(f)

#     ranges = data_loaded["ranges"]
#     pooled_group_hists_by_segment = data_loaded["pooled_group_hists_by_segment"]

#     seg_names = list(ranges.keys())
#     n_seg = len(seg_names)

#     fig, axes = plt.subplots(
#         2,
#         n_seg,
#         figsize=(6 * n_seg, 8),
#         sharex=True,
#     )

#     if n_seg == 1:
#         axes = np.array([axes]).reshape(2, 1)

#     plt.rcParams["axes.prop_cycle"] = plt.cycler(
#         color=plt.cm.tab20(np.linspace(0, 1, 20))
#     )

#     for col, seg_name in enumerate(seg_names):
#         group_means = pooled_group_hists_by_segment[seg_name]

#         # linear y
#         ax_lin = axes[0, col]
#         ax_lin.set_title(f"{seg_name} (linear)", fontsize=14)
#         for gt, mean_hist in group_means.items():
#             ax_lin.plot(
#                 centers, mean_hist, lw=1.2, alpha=0.8, label=pretty_group_label(gt)
#             )
#         ax_lin.set_xlabel("Speed")
#         ax_lin.set_ylabel("Density")
#         ax_lin.grid(True, which="both", ls="--", lw=0.4)

#         # log y
#         ax_log = axes[1, col]
#         ax_log.set_title(f"{seg_name} (log)", fontsize=14)
#         for gt, mean_hist in group_means.items():
#             ax_log.plot(centers, mean_hist, lw=1.2, alpha=0.8)
#         ax_log.set_xlabel("Speed")
#         ax_log.set_ylabel("Density (log)")
#         ax_log.set_yscale("log")
#         ax_log.grid(True, which="both", ls="--", lw=0.4)

#     handles, labels = axes[0, 0].get_legend_handles_labels()
#     unique = dict(zip(labels, handles, strict=False))

#     fig.legend(
#         unique.values(),
#         unique.keys(),
#         title="Groups",
#         loc="center left",
#         bbox_to_anchor=(0.92, 0.5),
#         fontsize=11,
#         frameon=False,
#         handlelength=1.0,
#         handleheight=0.8,
#         borderpad=0.4,
#         labelspacing=0.3,
#         handletextpad=0.4,
#     )

#     plt.tight_layout(rect=[0.02, 0.02, 0.88, 0.95])

#     plt.savefig(
#         distribution_folder
#         / f"group_means_dist_{groups_selected}_{POOL_SPLIT}_bins{BINS_HIST}.png",
#         bbox_inches="tight",
#     )
#     plt.show()


# =============================================================================
# --------------------- PLOT CI ENVELOPES (ABS VALUES) ------------------------
# =============================================================================

# for groups_selected in groups_ci:
#     print(f"\n[CI ENVELOPES] Processing grouping: {groups_selected}")

#     outdir_bootstrap_repeat_aux = Path(
#         outdir_bootstrap_repeat.format(
#             groups_selected=groups_selected,
#             n_resamples=n_resamples,
#             downsample_factor=downsample_factor,
#             seed=seed,
#         )
#     )

#     if not outdir_bootstrap_repeat_aux.exists():
#         print("  -> bootstrap file missing, skipping")
#         continue

#     with open(outdir_bootstrap_repeat_aux, "rb") as f:
#         data_loaded = pickle.load(f)
#     print(f"  Loading bootstrap: {outdir_bootstrap_repeat_aux}")

#     ranges = data_loaded["ranges"]
#     percentiles_ = data_loaded["percentiles_"]
#     group_data = data_loaded["group_data"]
#     ci_low_repeat = data_loaded["ci_low_repeat"]
#     ci_high_repeat = data_loaded["ci_high_repeat"]

#     seg_names = list(ranges.keys())
#     n_seg = len(seg_names)

#     fig, axes = plt.subplots(
#         2,
#         n_seg,
#         figsize=(6 * n_seg, 8),
#     )
#     if n_seg == 1:
#         axes = np.array([axes]).reshape(2, 1)

#     plt.rcParams["axes.prop_cycle"] = plt.cycler(
#         color=plt.cm.tab20(np.linspace(0, 1, 20))
#     )

#     legend_handles = []
#     legend_labels = []

#     for col, seg_name in enumerate(seg_names):
#         ax_lin = axes[0, col]
#         ax_lin.set_title(f"{seg_name} (linear)", fontsize=14)

#         for gt in group_data.keys():
#             lo = ci_low_repeat[seg_name][gt]
#             hi = ci_high_repeat[seg_name][gt]
#             label = pretty_group_label(gt)

#             band = ax_lin.fill_between(
#                 percentiles_,
#                 lo,
#                 hi,
#                 alpha=0.6,
#                 label=label,
#             )
#             legend_handles.append(band)
#             legend_labels.append(label)

#         ax_lin.set_xlabel("Percentiles")
#         ax_lin.set_ylabel("Speed")
#         ax_lin.set_ylim(0.2, 1.5)
#         ax_lin.set_xlim(0, 100)
#         ax_lin.grid(True, which="both", ls="--", lw=0.4)

#         ax_log = axes[1, col]
#         ax_log.set_title(f"{seg_name} (log)", fontsize=14)
#         for gt in group_data.keys():
#             lo = ci_low_repeat[seg_name][gt]
#             hi = ci_high_repeat[seg_name][gt]
#             ax_log.fill_between(
#                 percentiles_,
#                 lo,
#                 hi,
#                 alpha=0.6,
#             )
#         ax_log.set_xlabel("Percentiles")
#         ax_log.set_ylabel("Speed (log)")
#         ax_log.set_yscale("log")
#         ax_log.set_xscale("log")
#         ax_log.grid(True, which="both", ls="--", lw=0.4)

#     uniq = {}
#     for h, l in zip(legend_handles, legend_labels, strict=False):
#         if l not in uniq:
#             uniq[l] = h

#     fig.legend(
#         uniq.values(),
#         uniq.keys(),
#         title="Groups",
#         loc="center left",
#         bbox_to_anchor=(0.92, 0.5),
#         fontsize=11,
#         frameon=False,
#         handlelength=1.0,
#         handleheight=0.8,
#         borderpad=0.4,
#         labelspacing=0.3,
#         handletextpad=0.4,
#     )

#     plt.tight_layout(rect=[0.02, 0.02, 0.88, 0.95])

#     plt.savefig(
#         distribution_folder
#         / f"ci_comparison_{groups_selected}_{POOL_SPLIT}_bins{BINS_HIST}.png",
#         bbox_inches="tight",
#     )
#     plt.show()
#%%
for subset in SPEED_SUBSETS:
    print(f"\n=== [CI ENVELOPES] Subset: {subset} ===")

    dist_folder = paths["f_speed"] / "dist"
    dist_folder.mkdir(parents=True, exist_ok=True)
    print(f"  Distribution folder: {dist_folder}")

    for groups_selected in groups_ci:
        print(f"  [CI ENVELOPES] Processing grouping: {groups_selected}")

        outdir_bootstrap_repeat_aux = Path(
            outdir_bootstrap_repeat.format(
                subset=subset,
                groups_selected=groups_selected,
                n_resamples=n_resamples,
                downsample_factor=downsample_factor,
                seed=seed,
            )
        )
        if not outdir_bootstrap_repeat_aux.exists():
            print("    -> bootstrap file missing, skipping")
            continue

        with open(outdir_bootstrap_repeat_aux, "rb") as f:
            data_loaded = pickle.load(f)
        print(f"    Loading bootstrap: {outdir_bootstrap_repeat_aux}")

        ranges = data_loaded["ranges"]
        percentiles_ = data_loaded["percentiles_"]
        group_data = data_loaded["group_data"]
        ci_low_repeat = data_loaded["ci_low_repeat"]
        ci_high_repeat = data_loaded["ci_high_repeat"]

        seg_names = list(ranges.keys())
        n_seg = len(seg_names)

        fig, axes = plt.subplots(
            2,
            n_seg,
            figsize=(6 * n_seg, 8),
        )
        if n_seg == 1:
            axes = np.array([axes]).reshape(2, 1)

        plt.rcParams["axes.prop_cycle"] = plt.cycler(
            color=plt.cm.tab20(np.linspace(0, 1, 20))
        )

        legend_handles = []
        legend_labels = []

        for col, seg_name in enumerate(seg_names):
            ax_lin = axes[0, col]
            ax_lin.set_title(f"{seg_name} (linear)", fontsize=14)

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
            # ax_lin.set_ylim(0.2, 1.5)
            ax_lin.set_xlim(0, 100)
            ax_lin.grid(True, which="both", ls="--", lw=0.4)

            ax_log = axes[1, col]
            ax_log.set_title(f"{seg_name} (log)", fontsize=14)
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
            dist_folder
            / f"ci_comparison_subset_{subset}_group_{groups_selected}_{POOL_SPLIT}_bins{BINS_HIST}.png",
            bbox_inches="tight",
            dpi=150,
        )
        plt.show()

#%%
# =============================================================================
# ------------------ CI DIFF BANDS: AGE CONTRAST (INES ONLY) ------------------
# =============================================================================

if dataset_name == "ines":
    for groups_selected in groups_age_diff:
        print(f"\n[AGE DIFF] Processing grouping: {groups_selected}")

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
        print(f"  Loading bootstrap: {outdir_bootstrap_repeat_aux}")

        groups_selected = data_loaded["groups_selected"]
        group_data = data_loaded["group_data"]
        ranges = data_loaded["ranges"]
        percentiles_ = data_loaded["percentiles_"]
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

        color_map = make_age_contrast_color_map(group_data.keys(), groups_selected)

        for ax, seg_name in zip(axes, seg_names, strict=False):
            for gt in ci_low_repeat[seg_name].keys():
                if groups_selected == "age":
                    if gt != "4M":
                        continue
                    gt_4m = "4M"
                    gt_2m = "2M"
                    if gt_2m not in ci_low_repeat[seg_name]:
                        continue
                    label = "4M-2M"
                else:
                    if not isinstance(gt, tuple) or "4M" not in gt:
                        continue
                    age_idx = gt.index("4M")
                    gt_4m = gt
                    gt2_list = list(gt)
                    gt2_list[age_idx] = "2M"
                    gt_2m = tuple(gt2_list)
                    if gt_2m not in ci_low_repeat[seg_name]:
                        continue
                    label = age_contrast_label(gt_4m)

                color = color_map[label]

                ci_low_4m = ci_low_repeat[seg_name][gt_4m]
                ci_low_2m = ci_low_repeat[seg_name][gt_2m]
                ci_high_4m = ci_high_repeat[seg_name][gt_4m]
                ci_high_2m = ci_high_repeat[seg_name][gt_2m]

                ci_low_diff = ci_low_4m - ci_low_2m
                ci_high_diff = ci_high_4m - ci_high_2m

                ax.plot(percentiles_, ci_low_diff, label=label, color=color, alpha=0.7)
                ax.plot(percentiles_, ci_high_diff, color=color, alpha=0.7)
                ax.fill_between(
                    percentiles_, ci_low_diff, ci_high_diff, color=color, alpha=0.3
                )

            ax.axhline(0, color="black", linestyle="--", linewidth=1)
            ax.set_title(seg_name)
            ax.set_xlabel("Percentiles")
            ax.set_ylim(-0.2, 0.2)
            ax.set_xlim(0, 100)
            ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
            ax.tick_params(axis="both", labelsize=15)
            if ax is axes[0]:
                ax.set_ylabel("Speed Difference (4M - 2M)")

        handles, labels = axes[0].get_legend_handles_labels()
        uniq = dict(zip(labels, handles, strict=False))

        fig.legend(
            uniq.values(),
            uniq.keys(),
            title="Age contrast"
            if groups_selected == "age"
            else "Group (non-age factors)",
            loc="center right",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=10,
            frameon=False,
        )

        fig.suptitle(f"dFC acceleration diff 4M-2M   groups={groups_selected}")
        plt.tight_layout(rect=[0.02, 0.02, 0.85, 0.95])

        plt.savefig(
            acceleration_folder
            / f"ci_diff_band_4M_minus_2M_{seg_name}_{groups_selected}.png"
        )
        plt.show()


# =============================================================================
# ------------------ CI DIFF BANDS: SEX CONTRAST (INES ONLY) ------------------
# =============================================================================

if dataset_name == "ines":
    groups_list_sex = groups_sex_diff

    for groups_selected in groups_list_sex:
        print(f"\n[SEX DIFF] Processing grouping: {groups_selected}")

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
        print(f"  Loading bootstrap: {outdir_bootstrap_repeat_aux}")

        groups_selected = data_loaded["groups_selected"]
        group_data = data_loaded["group_data"]
        ranges = data_loaded["ranges"]
        percentiles_ = data_loaded["percentiles_"]
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

        color_map = make_sex_contrast_color_map(group_data.keys(), groups_selected)

        for ax, seg_name in zip(axes, seg_names, strict=False):
            for gt in ci_low_repeat[seg_name].keys():
                if groups_selected == "sex":
                    if gt != "male":
                        continue
                    gt_m = "male"
                    gt_f = "female"
                    if gt_f not in ci_low_repeat[seg_name]:
                        continue
                    label = "male-female"
                else:
                    if not isinstance(gt, tuple) or "male" not in gt:
                        continue
                    sex_idx = gt.index("male")
                    gt_m = gt
                    gt_f_list = list(gt)
                    gt_f_list[sex_idx] = "female"
                    gt_f = tuple(gt_f_list)
                    if gt_f not in ci_low_repeat[seg_name]:
                        continue
                    label = sex_contrast_label(gt_m)

                color = color_map[label]

                ci_low_m = ci_low_repeat[seg_name][gt_m]
                ci_low_f = ci_low_repeat[seg_name][gt_f]
                ci_high_m = ci_high_repeat[seg_name][gt_m]
                ci_high_f = ci_high_repeat[seg_name][gt_f]

                ci_low_diff = ci_low_m - ci_low_f
                ci_high_diff = ci_high_m - ci_high_f

                ax.plot(percentiles_, ci_low_diff, label=label, color=color, alpha=0.7)
                ax.plot(percentiles_, ci_high_diff, color=color, alpha=0.7)
                ax.fill_between(
                    percentiles_, ci_low_diff, ci_high_diff, color=color, alpha=0.3
                )

            ax.axhline(0, color="black", linestyle="--", linewidth=1)
            ax.set_title(seg_name)
            ax.set_xlabel("Percentiles")
            ax.set_ylim(-0.2, 0.2)
            ax.set_xlim(0, 100)
            ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
            ax.tick_params(axis="both", labelsize=15)
            if ax is axes[0]:
                ax.set_ylabel("Speed Difference (male - female)")

        handles, labels = axes[0].get_legend_handles_labels()
        uniq = dict(zip(labels, handles, strict=False))

        fig.legend(
            uniq.values(),
            uniq.keys(),
            title="Sex contrast"
            if groups_selected == "sex"
            else "Group (non-sex factors)",
            loc="center right",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=10,
            frameon=False,
        )

        fig.suptitle(
            f"dFC acceleration diff male-female   groups={groups_selected}"
        )
        plt.tight_layout(rect=[0.02, 0.02, 0.85, 0.95])

        plt.savefig(
            acceleration_folder
            / f"ci_diff_band_male_minus_female_{seg_name}_{groups_selected}.png"
        )
        plt.show()

#%%
# =============================================================================
# ---------------- CI DIFF BANDS: GENOTYPE CONTRAST (BOTH DATASETS) -----------
# =============================================================================

for subset in SPEED_SUBSETS:
    print(f"\n=== [GENOTYPE DIFF {GENO_REF}-{GENO_MUT}] Subset: {subset} ===")

    for groups_selected in groups_genotype_diff:
        print(f"  Processing grouping: {groups_selected}")

        outdir_bootstrap_repeat_aux = Path(
            outdir_bootstrap_repeat.format(
                subset=subset,
                groups_selected=groups_selected,
                n_resamples=n_resamples,
                downsample_factor=downsample_factor,
                seed=seed,
            )
        )

        if not outdir_bootstrap_repeat_aux.exists():
            print(f"    -> bootstrap missing for subset={subset}, group={groups_selected}")
            continue

        with open(outdir_bootstrap_repeat_aux, "rb") as f:
            data_loaded = pickle.load(f)
        print(f"    Loading bootstrap: {outdir_bootstrap_repeat_aux}")

        groups_selected_loaded = data_loaded["groups_selected"]
        group_data = data_loaded["group_data"]
        ranges = data_loaded["ranges"]
        percentiles_ = data_loaded["percentiles_"]
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

        color_map = make_genotype_contrast_color_map(
            group_data.keys(),
            groups_selected_loaded,
            mut=GENO_MUT,
            ref=GENO_REF,
        )

        for ax, seg_name in zip(axes, seg_names, strict=False):
            for gt in ci_low_repeat[seg_name].keys():
                if groups_selected_loaded == "genotype":
                    if gt != GENO_REF:
                        continue
                    gt_ref = GENO_REF
                    gt_mut = GENO_MUT
                    if gt_mut not in ci_low_repeat[seg_name]:
                        continue
                    label = f"{GENO_REF}-{GENO_MUT}"
                else:
                    if not isinstance(gt, tuple) or GENO_REF not in gt:
                        continue
                    geno_idx = gt.index(GENO_REF)
                    gt_ref = gt
                    gt_mut_list = list(gt)
                    gt_mut_list[geno_idx] = GENO_MUT
                    gt_mut = tuple(gt_mut_list)
                    if gt_mut not in ci_low_repeat[seg_name]:
                        continue
                    label = genotype_contrast_label(gt_ref, mut=GENO_MUT, ref=GENO_REF)

                color = color_map[label]

                ci_low_ref = ci_low_repeat[seg_name][gt_ref]
                ci_low_mut = ci_low_repeat[seg_name][gt_mut]
                ci_high_ref = ci_high_repeat[seg_name][gt_ref]
                ci_high_mut = ci_high_repeat[seg_name][gt_mut]

                ci_low_diff = ci_low_ref - ci_low_mut
                ci_high_diff = ci_high_ref - ci_high_mut

                ax.plot(percentiles_, ci_low_diff, label=label, color=color, alpha=0.7)
                ax.plot(percentiles_, ci_high_diff, color=color, alpha=0.7)
                ax.fill_between(
                    percentiles_, ci_low_diff, ci_high_diff, color=color, alpha=0.3
                )

            ax.axhline(0, color="black", linestyle="--", linewidth=1)
            ax.set_title(seg_name)
            ax.set_xlabel("Percentiles")
            ax.set_ylim(-0.4, 0.4)
            ax.set_xlim(0, 100)
            ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
            ax.tick_params(axis="both", labelsize=15)
            if ax is axes[0]:
                ax.set_ylabel(f"Speed Difference ({GENO_REF} - {GENO_MUT})")

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

        fig.suptitle(
            f"dFC acceleration diff {GENO_REF} - {GENO_MUT}   "
            f"groups={groups_selected_loaded}   subset={subset}"
        )
        plt.tight_layout(rect=[0.02, 0.02, 0.85, 0.95])

        plt.savefig(
            acceleration_folder
            / f"ci_diff_band_{GENO_REF}_minus_{GENO_MUT}_subset_{subset}_{groups_selected_loaded}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()
#%%

# =============================================================================
# ---- CI DIFF BANDS: (WT-VEH) minus other genotype_treatment groups (JULIEN) --
# =============================================================================

if dataset_name == "julien":
    ref_group = (GENO_REF, "VEH")  # ('WT', 'VEH')
    other_groups = [
        (GENO_REF, "LCTB92"),
        (GENO_MUT, "VEH"),
        (GENO_MUT, "LCTB92"),
    ]
    diff_labels = {
        (GENO_REF, "LCTB92"): f"{GENO_REF}-VEH - {GENO_REF}-LCTB92",
        (GENO_MUT, "VEH"):    f"{GENO_REF}-VEH - {GENO_MUT}-VEH",
        (GENO_MUT, "LCTB92"): f"{GENO_REF}-VEH - {GENO_MUT}-LCTB92",
    }

    for subset in SPEED_SUBSETS:
        print(
            f"\n[GENOTYPE_TREATMENT DIFF vs {GENO_REF}-VEH] "
            f"subset={subset}"
        )

        # we force groups_selected="genotype_treatment" here
        groups_selected = "genotype_treatment"

        outdir_bootstrap_repeat_aux = Path(
            outdir_bootstrap_repeat.format(
                subset=subset,
                groups_selected=groups_selected,
                n_resamples=n_resamples,
                downsample_factor=downsample_factor,
                seed=seed,
            )
        )

        if not outdir_bootstrap_repeat_aux.exists():
            print(
                f"  -> bootstrap missing for subset={subset}, "
                f"group={groups_selected}"
            )
            continue

        with open(outdir_bootstrap_repeat_aux, "rb") as f:
            data_loaded = pickle.load(f)
        print(f"  Loading bootstrap: {outdir_bootstrap_repeat_aux}")

        group_data = data_loaded["group_data"]
        ranges = data_loaded["ranges"]
        percentiles_ = data_loaded["percentiles_"]
        ci_low_repeat = data_loaded["ci_low_repeat"]
        ci_high_repeat = data_loaded["ci_high_repeat"]

        # segments: short / mid / long
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

        # sanity check: reference group must exist in all segments
        missing_ref = [
            seg for seg in seg_names
            if ref_group not in ci_low_repeat[seg]
        ]
        if missing_ref:
            print(
                f"  ⚠️ reference group {ref_group} missing in segments: "
                f"{missing_ref} – skipping subset={subset}"
            )
            plt.close(fig)
            continue

        # color cycle for the 3 contrasts
        colors = plt.cm.tab10(np.linspace(0, 1, 3))

        legend_handles = []
        legend_labels = []

        for ax, seg_name in zip(axes, seg_names, strict=False):
            lo_ref = ci_low_repeat[seg_name][ref_group]
            hi_ref = ci_high_repeat[seg_name][ref_group]

            for idx, grp in enumerate(other_groups):
                if grp not in ci_low_repeat[seg_name]:
                    print(f"    -> group {grp} missing in segment {seg_name}, skipping")
                    continue

                lo_g = ci_low_repeat[seg_name][grp]
                hi_g = ci_high_repeat[seg_name][grp]

                lo_diff = lo_ref - lo_g
                hi_diff = hi_ref - hi_g

                label = diff_labels[grp]
                color = colors[idx]

                line_lo = ax.plot(
                    percentiles_,
                    lo_diff,
                    color=color,
                    alpha=0.9,
                    lw=1.8,
                )[0]
                ax.plot(
                    percentiles_,
                    hi_diff,
                    color=color,
                    alpha=0.9,
                    lw=1.8,
                )
                ax.fill_between(
                    percentiles_,
                    lo_diff,
                    hi_diff,
                    color=color,
                    alpha=0.25,
                )

                # only collect legend handle once per label
                if label not in legend_labels:
                    legend_labels.append(label)
                    legend_handles.append(line_lo)

            ax.axhline(0, color="black", linestyle="--", linewidth=1)
            ax.set_title(seg_name, fontsize=14)
            ax.set_xlabel("Percentiles")
            ax.set_xlim(0, 100)
            ax.set_ylim(-0.4, 0.4)  # adjust if needed
            ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
            ax.tick_params(axis="both", labelsize=12)
            if ax is axes[0]:
                ax.set_ylabel(f"Speed Difference ({GENO_REF}-VEH - other)", fontsize=12)

        fig.legend(
            legend_handles,
            legend_labels,
            title="Genotype × Treatment contrasts\n(ref = WT-VEH)",
            loc="center right",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=11,
            frameon=False,
        )

        fig.suptitle(
            f"dFC acceleration diff vs {GENO_REF}-VEH   "
            f"groups=genotype_treatment   subset={subset}",
            fontsize=14,
        )
        plt.tight_layout(rect=[0.02, 0.02, 0.85, 0.95])

        plt.savefig(
            acceleration_folder
            / f"ci_diff_band_{GENO_REF}_VEH_vs_others_subset_{subset}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()

#%%




















#%%






















# %%
# ========================== PERCENTILE TRACKS ==========================


# ============================================================
#   SPEED PERCENTILES PER SEGMENT (for NOR correlations)
# ============================================================

def build_per_animal_flat_speed(
    speeds: list[np.ndarray],
    selected_windows: range,
) -> np.ndarray:
    """
    For a given window segment, pool all dFC speed samples per animal.

    speeds[j][i] = 1D array of samples for animal i at window j.
    selected_windows: range of window indices belonging to a segment
                      (e.g. short / mid / long).

    Returns
    -------
    flat_speeds : np.ndarray, shape (n_animals, n_samples_in_segment)
        Row i = all speed samples of animal i across the selected windows.
    """
    n_animals = len(speeds[0])

    flat_speeds = []
    for i in range(n_animals):
        if selected_windows:
            flat_i = np.concatenate(
                [speeds[j][i].ravel() for j in selected_windows]
            )
        else:
            flat_i = np.array([], dtype=float)
        flat_speeds.append(flat_i)

    # assumes same number of samples per animal in the segment
    return np.vstack(flat_speeds)


# --- behavioural index (one value per animal) ---
if dataset_name == "julien":
    nor_index = cog_data["index_NOR"].to_numpy()
elif dataset_name == "ines":
    # make_long_cog is already defined above
    df_long = make_long_cog(cog_data, dataset_name)
    nor_index = df_long["ro24h"].to_numpy()
else:
    raise ValueError(f"Unknown dataset_name={dataset_name!r} for NOR index.")


# --- percentiles grid & per-segment percentiles ---
percentiles_ = np.linspace(0, 100, 100)  # 0–100th percentile

speeds_percentile_per_segment: dict[str, np.ndarray] = {}

for seg_name, w_range in ranges.items():
    flat_speeds = build_per_animal_flat_speed(speeds, w_range)
    print(
        f"[INFO] flat_speeds shape for segment {seg_name}: {flat_speeds.shape}"
    )

    # shape: (len(percentiles_), n_animals)
    speeds_percentile_per_segment[seg_name] = np.percentile(
        flat_speeds,
        q=percentiles_,
        axis=1,  # percentile across samples (columns)
    )



# %% ========================== PLOTS (unchanged style) ==========================

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

# pooling plots
pooling_folder = paths["f_speed"] / "pooling"
pooling_folder.mkdir(parents=True, exist_ok=True)

savedir_dfc_speed_cdf_windows = str(
    pooling_folder
    / "dFC_speed_cdf_windows_animals_{n_animals}_regions_{regions}_tr_{total_tr}.png"
)

if save_fig:
    outpath_fig4 = savedir_dfc_speed_cdf_windows.format(
        n_animals=n_animals, regions=regions, total_tr=total_tr
    )
    plt.savefig(outpath_fig4, dpi=300)
    print(f"[INFO] Figure saved to: {outpath_fig4}")



# # %%

# import os

# from scipy.stats import spearmanr


# # --------------------------
# # Helper: bootstrap Spearman correlation
# # --------------------------
# def bootstrap_spearman(x, y, n_resamples=1000, random_state=0):
#     """Return mean Spearman rho and 95% CI via bootstrapping."""
#     rng = np.random.default_rng(random_state)
#     n = len(x)
#     r_boot = np.empty(n_resamples)
#     for i in range(n_resamples):
#         idx = rng.integers(0, n, n)
#         r_boot[i], _ = spearmanr(x[idx], y[idx])
#     return np.mean(r_boot), np.percentile(r_boot, [2.5, 97.5])


# # --------------------------
# # Helper: bootstrap difference between two correlations
# # --------------------------
# def bootstrap_diff(x1, y1, x2, y2, n_resamples=1000, random_state=0):
#     rng = np.random.default_rng(random_state)
#     n1, n2 = len(x1), len(x2)
#     diffs = np.empty(n_resamples)
#     for i in range(n_resamples):
#         idx1 = rng.integers(0, n1, n1)
#         idx2 = rng.integers(0, n2, n2)
#         r1, _ = spearmanr(x1[idx1], y1[idx1])
#         r2, _ = spearmanr(x2[idx2], y2[idx2])
#         diffs[i] = r1 - r2
#     return np.mean(diffs), np.percentile(diffs, [2.5, 97.5])


# # %%
# # Generate Δρ plots with bootstrapped CIs for all group pairs & segments
# # --------------------------
# # Main: Iterate over all group pairs
# # --------------------------
# savedir_corr_plots = paths["f_speed"] / "correlation_plots"
# savedir_corr_plots.mkdir(parents=True, exist_ok=True)
# results_dir = str(savedir_corr_plots)
# os.makedirs(results_dir, exist_ok=True)

# group_keys = list(group_data.keys())
# pairs = list(combinations(group_keys, 2))

# # Bootstrap Δρ plots
# for speed_seg_name, speeds_ppsegment in speeds_percentile_per_segment.items():
#     print(
#         f"[INFO] Processing segment: {speed_seg_name} with shape {speeds_ppsegment.shape}"
#     )

#     for group_a, group_b in pairs:
#         print(f"\n=== Comparing {group_a} vs {group_b} in pool {speed_seg_name} ===")

#         idx_a = group_data[group_a]
#         idx_b = group_data[group_b]

#         # Skip if any group has too few animals
#         if len(idx_a) < 3 or len(idx_b) < 3:
#             print(f"⚠️ Skipping {group_a} vs {group_b} (too few animals)")
#             continue

#         diff_means, diff_ci_low, diff_ci_high = [], [], []

#         for i in range(speeds_ppsegment.shape[0]):
#             print("Processing percentile index:", i)
#             y1 = speeds_ppsegment[i, idx_a]
#             y2 = speeds_ppsegment[i, idx_b]
#             x1 = nor_index[idx_a]
#             x2 = nor_index[idx_b]
#             mean_diff, (ci_low, ci_high) = bootstrap_diff(
#                 x1, y1, x2, y2, n_resamples=1000
#             )
#             diff_means.append(mean_diff)
#             diff_ci_low.append(ci_low)
#             diff_ci_high.append(ci_high)

#         diff_means, diff_ci_low, diff_ci_high = map(
#             np.array, (diff_means, diff_ci_low, diff_ci_high)
#         )

#         # Plot Δρ curve
#         plt.figure(figsize=(10, 5))
#         plt.plot(
#             percentiles_,
#             diff_means,
#             lw=2,
#             color="purple",
#             label=f"{group_a} − {group_b}",
#         )
#         plt.fill_between(
#             percentiles_, diff_ci_low, diff_ci_high, color="purple", alpha=0.3
#         )
#         plt.axhline(0, color="black", lw=1)

#         # Highlight significant regions (CI excludes 0)
#         plt.fill_between(
#             percentiles_,
#             diff_ci_low,
#             diff_ci_high,
#             where=(diff_ci_low > 0) | (diff_ci_high < 0),
#             color="purple",
#             alpha=0.2,
#             label="Significant Δρ (95% CI excludes 0)",
#         )

#         plt.xlabel("Percentiles")
#         plt.ylabel("Δ Spearman ρ (Group A − Group B)")
#         plt.title(
#             f"Difference in Spearman Correlation between {group_a} and {group_b}\nwith Bootstrapped 95% CI"
#         )
#         plt.legend()
#         plt.tight_layout()

#         # Save each figure automatically
#         fname = f"delta_rho_{group_a[0]}_{group_a[1]}__vs__{group_b[0]}_{group_b[1]}_{speed_seg_name}.png".replace(
#             "'", ""
#         )
#         plt.savefig(os.path.join(results_dir, fname), dpi=300)
#         plt.close()

#         print(f"✅ Saved plot → {fname}")


# # %%
# # Save summary CSV of Δρ results for all group pairs & segments


# # Ensure results folder exists
# resultdir_speed_delta_rho = paths["speed"] / "delta_rho_results"
# resultdir_speed_delta_rho.mkdir(parents=True, exist_ok=True)

# summary_data = []
# for speed_seg_name, speeds_ppsegment in speeds_percentile_per_segment.items():
#     print(
#         f"[INFO] Processing segment: {speed_seg_name} with shape {speeds_ppsegment.shape}"
#     )

#     for group_a, group_b in combinations(group_data.keys(), 2):
#         fname_base = f"{group_a[0]}_{group_a[1]}__vs__{group_b[0]}_{group_b[1]}_{speed_seg_name}".replace(
#             "'", ""
#         )
#         filepath_png = os.path.join(
#             resultdir_speed_delta_rho, f"delta_rho_{fname_base}.png"
#         )
#         filepath_npz = os.path.join(
#             resultdir_speed_delta_rho, f"delta_rho_{fname_base}.npz"
#         )

#         # Skip if one group too small or not computed
#         idx_a = group_data[group_a]
#         idx_b = group_data[group_b]
#         if len(idx_a) < 3 or len(idx_b) < 3:
#             continue

#         print(f"Processing Δρ for {group_a} vs {group_b}...")

#         diff_means, diff_ci_low, diff_ci_high = [], [], []

#         for i in range(speeds_ppsegment.shape[0]):
#             y1 = speeds_ppsegment[i, idx_a]
#             y2 = speeds_ppsegment[i, idx_b]
#             x1 = nor_index[idx_a]
#             x2 = nor_index[idx_b]
#             mean_diff, (ci_low, ci_high) = bootstrap_diff(
#                 x1, y1, x2, y2, n_resamples=1000
#             )
#             diff_means.append(mean_diff)
#             diff_ci_low.append(ci_low)
#             diff_ci_high.append(ci_high)

#         diff_means, diff_ci_low, diff_ci_high = map(
#             np.array, (diff_means, diff_ci_low, diff_ci_high)
#         )

#         # Compute significance mask (CI excludes 0)
#         sig_mask = (diff_ci_low > 0) | (diff_ci_high < 0)

#         # Save to NPZ (for future replotting or analysis)
#         np.savez(
#             filepath_npz,
#             percentiles=percentiles_,
#             delta_rho_mean=diff_means,
#             delta_rho_ci_low=diff_ci_low,
#             delta_rho_ci_high=diff_ci_high,
#             significant_mask=sig_mask,
#             groupA=group_a,
#             groupB=group_b,
#         )

#         # Summaries for CSV
#         mean_delta = np.nanmean(diff_means)
#         ci_global = (np.nanmin(diff_ci_low), np.nanmax(diff_ci_high))

#         # Identify contiguous significant percentile ranges
#         sig_ranges = []
#         in_block = False
#         start = None
#         for i, val in enumerate(sig_mask):
#             if val and not in_block:
#                 start = percentiles_[i]
#                 in_block = True
#             elif not val and in_block:
#                 end = percentiles_[i - 1]
#                 sig_ranges.append(f"{start:.1f}-{end:.1f}")
#                 in_block = False
#         if in_block:
#             sig_ranges.append(f"{start:.1f}-{percentiles_[-1]:.1f}")
#         sig_range_str = ", ".join(sig_ranges) if sig_ranges else "None"

#         summary_data.append(
#             {
#                 "Group A": f"{group_a[0]} {group_a[1]}",
#                 "Group B": f"{group_b[0]} {group_b[1]}",
#                 "Mean Δρ": f"{mean_delta:.3f}",
#                 "95% CI (min,max)": f"[{ci_global[0]:.3f}, {ci_global[1]:.3f}]",
#                 "Significant percentile ranges": sig_range_str,
#                 "NPZ file": os.path.basename(filepath_npz),
#             }
#         )

#     # Build summary table and export to CSV
#     summary_df = pd.DataFrame(summary_data)
#     csv_path = os.path.join(results_dir, "delta_rho_summary.csv")
#     summary_df.to_csv(csv_path, index=False)

#     print(f"\n✅ Summary CSV saved to: {csv_path}")
#     display(summary_df)

# # %%


# # --------------------------
# # Helper: bootstrap Spearman correlation
# # --------------------------
# def bootstrap_spearman(x, y, n_resamples=1000, random_state=0):
#     """Return mean Spearman rho and 95% CI via bootstrapping."""
#     rng = np.random.default_rng(random_state)
#     n = len(x)
#     r_boot = np.empty(n_resamples)
#     for i in range(n_resamples):
#         idx = rng.integers(0, n, n)
#         r_boot[i], _ = spearmanr(x[idx], y[idx])
#     return np.mean(r_boot), np.percentile(r_boot, [2.5, 97.5])


# # --------------------------
# # Group-wise Spearman correlations
# # --------------------------
# for speed_seg_name, speeds_ppsegment in speeds_percentile_per_segment.items():
#     print(
#         f"[INFO] Processing segment: {speed_seg_name} with shape {speeds_ppsegment.shape}"
#     )

#     group_results = {}

#     for (
#         gt,
#         idxs,
#     ) in group_data.items():  # e.g. {"WT": [0,1,...], "Mut": [24,...]}
#         r_means, ci_lows, ci_highs, p_vals = [], [], [], []

#         for i in range(speeds_ppsegment.shape[0]):
#             y = speeds_ppsegment[i, idxs]
#             x = nor_index[idxs]
#             if np.std(y) == 0:
#                 r_means.append(np.nan)
#                 ci_lows.append(np.nan)
#                 ci_highs.append(np.nan)
#                 p_vals.append(np.nan)
#                 continue

#             r, p = spearmanr(x, y)
#             r_mean, (r_low, r_high) = bootstrap_spearman(x, y, n_resamples=1000)
#             r_means.append(r_mean)
#             ci_lows.append(r_low)
#             ci_highs.append(r_high)
#             p_vals.append(p)

#         group_results[gt] = {
#             "r_means": np.array(r_means),
#             "ci_lows": np.array(ci_lows),
#             "ci_highs": np.array(ci_highs),
#             "p_vals": np.array(p_vals),
#         }

#     # --------------------------
#     # Plot group-wise results with significance shading
#     # --------------------------
#     plt.figure(figsize=(10, 6))
#     colors = plt.cm.tab10.colors  # distinct colors for groups

#     for idx, (gt, res) in enumerate(group_results.items()):
#         color = colors[idx % len(colors)]
#         plt.plot(
#             percentiles_, res["r_means"], color=color, lw=2, label=f"{gt} (ρ mean)"
#         )
#         plt.fill_between(
#             percentiles_, res["ci_lows"], res["ci_highs"], color=color, alpha=0.25
#         )

#         # Shade non-significant regions (p > 0.05)
#         plt.fill_between(
#             percentiles_,
#             res["ci_lows"],
#             res["ci_highs"],
#             where=(res["p_vals"] > 0.05),
#             color=color,
#             alpha=0.1,
#             label=f"{gt} p > 0.05",
#         )

#     plt.axhline(0, color="black", lw=1)
#     plt.xlabel("Percentiles")
#     plt.ylabel("Spearman Correlation (ρ)")
#     plt.title(
#         "Spearman Correlation - NOR vs Speed \nwith Bootstrapped 95% CI and p>0.05 Shading"
#     )
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
#     savedir_spearman = savedir_corr_plots / f"spearman_correlation_{speed_seg_name}.png"
#     plt.savefig(savedir_spearman, dpi=300)

# # %%

# %%
