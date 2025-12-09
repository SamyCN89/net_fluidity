#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QC: dFC speed vs window size, for all speed subsets (julien dataset)

For each subset in SPEED_SUBSETS this script produces a 3-panel figure:
  1) Per-animal mean speed vs window
  2) Pooled median dFC speed vs window (per group)
  3) Spread (99th - 1st percentile) of the pooled distribution vs window (per group)
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from glob import glob

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from scripts.dfc.dfc_compute import DATASET_DEFAULTS, _canonical_dataset
from shared_code.fun_loaddata import load_timeseries_bundle
from shared_code.fun_paths import get_paths
from shared_code.fun_utils import load_cognitive_data, set_figure_params


# ------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------

# Only implemented / tested for julien
DATASET_NAME = "julien"

# must match the compute script
SPEED_SUBSETS = [
    "all",
    # "regions500",
    "per_region",
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

time_windows_range = np.arange(5, 100, 1)  # must match the speed compute script

dataset = _canonical_dataset(DATASET_NAME)
cfg = DATASET_DEFAULTS[dataset]

# set fonts / style & get save flag
save_fig = set_figure_params(True)


# ------------------------------------------------------------------------
# BASIC HELPERS
# ------------------------------------------------------------------------

def load_speed_stack(
    paths_speed_root: str,
    time_windows_range: Sequence[int],
    n_animals: int,
    regions: int,
) -> list[np.ndarray]:
    """
    Return list S where S[j][i] is 1D np.array of samples for animal i at window j.
    `paths_speed_root` is a format string with {w}, {n_animals}, {regions}.
    """
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

def discover_per_region_descriptors(
    subset_dir: Path,
    w0: int,
    n_animals: int,
    regions: int,
    lag: int = 1,
    tau_count: int = 2,
) -> list[str]:
    """
    Inspect one window under per_region/ and list all region descriptors.

    Returns
    -------
    list[str]  e.g. ['region-AI', 'region-PL', ...]
    """
    pattern = (
        subset_dir
        / f"speed_win{w0}_lag{lag}_tau{tau_count}_animals_{n_animals}_regions_{regions}_region-*.npz"
    )
    files = sorted(glob(str(pattern)))
    if not files:
        raise FileNotFoundError(
            f"No per-region speed files found for window={w0} with pattern {pattern}"
        )

    descriptors: list[str] = []
    for fpath in files:
        name = Path(fpath).name
        suffix = name.split(f"regions_{regions}_", 1)[1]  # region-XXX.npz
        descriptor = suffix[:-4]  # strip '.npz'
        descriptors.append(descriptor)

    return sorted(set(descriptors))


def load_speed_stack_single_region(
    subset_dir: Path,
    time_windows_range: Sequence[int],
    n_animals: int,
    regions: int,
    region_desc: str,
    lag: int = 1,
    tau_count: int = 2,
) -> list[list[np.ndarray]]:
    """
    Load speeds for ONE region (e.g. 'region-AI') across all windows.

    Returns
    -------
    speeds_per_window : list[list[np.ndarray]]
        speeds[w][i], with w = window index, i = animal index.
    """
    speeds_per_window: list[list[np.ndarray]] = []

    for w in time_windows_range:
        fpath = (
            subset_dir
            / f"speed_win{w}_lag{lag}_tau{tau_count}_animals_{n_animals}_regions_{regions}_{region_desc}.npz"
        )
        if not fpath.exists():
            raise FileNotFoundError(f"Missing per-region file: {fpath}")

        with np.load(fpath, allow_pickle=True) as z:
            if "speeds" not in z.files:
                raise KeyError(f"{fpath} missing 'speeds' array")
            s = z["speeds"]  # object array, len n_animals

        if len(s) != n_animals:
            raise ValueError(f"{fpath}: expected {n_animals} animals, got {len(s)}")

        window_speeds: list[np.ndarray] = []
        for i in range(n_animals):
            arr = np.asarray(s[i], dtype=float)
            window_speeds.append(arr.ravel())
        speeds_per_window.append(window_speeds)

    return speeds_per_window


def robust_percentiles(x: np.ndarray, qs=(1, 50, 99)) -> dict[int, float]:
    """Percentiles that survive empty / non-finite inputs."""
    if x.size == 0 or not np.isfinite(x).any():
        return {int(q): np.nan for q in qs}
    x = x[np.isfinite(x)]
    ps = np.percentile(x, qs)
    return {int(q): float(p) for q, p in zip(qs, ps, strict=False)}


def make_long_cog_julien(cog_data: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise julien cognitive data to have one row per animal with:
      name, genotype, treatment
    """
    df = cog_data.copy()
    if "mouse" in df.columns:
        df = df.rename(columns={"mouse": "name"})
    cols_keep = [c for c in ["name", "genotype", "treatment"] if c in df.columns]
    df = df[cols_keep]
    for col in ["genotype", "treatment"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def group_indices(df: pd.DataFrame, by: Sequence[str]) -> dict:
    """Return {group_key_tuple : np.ndarray[int]} of row indices."""
    gb = df.groupby(list(by), sort=False)
    return {k: v.values for k, v in gb.groups.items()}


def get_group_data_genotype_treatment(cog_data: pd.DataFrame) -> dict:
    df_long = make_long_cog_julien(cog_data)
    return group_indices(df_long, ["genotype", "treatment"])


# ------------------------------------------------------------------------
# PATHS & SHARED DATA
# ------------------------------------------------------------------------

paths = get_paths(
    dataset_name=dataset,
    timecourse_folder=cfg["timecourse_folder"],
    cognitive_data_file=cfg["cognitive_data_file"],
    anat_labels_file=cfg["anat_labels_file"],
)

speed_root = Path(paths["speed"])
preprocessed_root = Path(paths["preprocessed"])

# meta / timeseries bundle (to get n_animals, regions, total_tr)
loaddir_ts_meta = preprocessed_root / "ts_and_meta_2m4m.npz"
bundle = load_timeseries_bundle(loaddir_ts_meta)
n_animals = bundle.n_animals
regions = bundle.n_regions
total_tr = bundle.total_tr

# cognitive data
loaddir_cog_data = preprocessed_root / (
    f"cog_data_filtered_animals_{n_animals}_regions_{regions}_tr_{total_tr}.csv"
)
cog_data = load_cognitive_data(str(loaddir_cog_data))

# genotype × treatment groups (same for all subsets)
group_data = get_group_data_genotype_treatment(cog_data)
group_keys = list(group_data.keys())
print("[INFO] genotype_treatment groups:", group_keys)

# output folder
qc_folder = Path(paths["f_speed"]) / "qc_speed_vs_window"
qc_folder.mkdir(parents=True, exist_ok=True)

# template for NPZ paths; only {subset} is formatted here,
# {w}, {n_animals}, {regions} stay for load_speed_stack
subset_speed_template = str(
    speed_root
    / "{subset}/speed_win{{w}}_lag1_tau2_animals_{{n_animals}}_regions_{{regions}}.npz"
)


def run_qc_for_subset_label(
    subset_label: str,
    speeds: list[list[np.ndarray]],
    time_windows_range: np.ndarray,
    group_data: dict,
    group_keys: list,
    qc_folder: Path,
    save_fig: bool,
):
    n_windows = len(speeds)
    if n_windows == 0:
        print(f"  -> skipping subset '{subset_label}' (no windows)")
        return

    # per-window mean speed per animal: list[window] -> (n_animals,)
    n_animals = len(speeds[0])
    per_window_animal_means: list[np.ndarray] = []
    for j in range(n_windows):
        means_j = np.array(
            [float(np.mean(speeds[j][i])) for i in range(n_animals)], dtype=float
        )
        per_window_animal_means.append(means_j)

    # global flatten for percentiles
    all_flat = np.concatenate(
        [s.ravel() for win in speeds for s in win if np.asarray(s).size]
    )
    all_flat = all_flat[np.isfinite(all_flat)]
    if all_flat.size == 0:
        print(f"  -> subset '{subset_label}': all speeds empty / non-finite, skipping")
        return

    # colour mapping per group (same for all subsets)
    cmap = plt.cm.get_cmap("tab10", len(group_keys))
    group_colors = {gt: cmap(i) for i, gt in enumerate(group_keys)}

    # pooled percentile tracks per group
    qs = (1, 50, 99)
    percentile_tracks: dict = {}

    for gt, idxs in group_data.items():
        qdict = {q: [] for q in qs}

        for j in range(n_windows):
            if len(idxs) == 0:
                for q in qs:
                    qdict[q].append(np.nan)
                continue

            gflat = np.concatenate([speeds[j][i].ravel() for i in idxs])
            p = robust_percentiles(gflat, qs=qs)
            for q in qs:
                qdict[q].append(p[q])

        percentile_tracks[gt] = {
            q: np.array(vals, dtype=float) for q, vals in qdict.items()
        }

    # 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    axA, axB, axC = axes

    # Panel A: per-animal curves
    for gt, idxs in group_data.items():
        col = group_colors[gt]
        for i in idxs:
            mean_speeds_i = [per_window_animal_means[j][i] for j in range(n_windows)]
            axA.plot(
                time_windows_range,
                mean_speeds_i,
                color=col,
                alpha=0.25,
                lw=1.0,
            )

    axA.set_title("Per-animal mean dFC speed vs window")
    axA.set_xlabel("Time-window size (TR)")
    axA.set_ylabel("Mean dFC speed")
    axA.grid(alpha=0.3)

    # Panel B: pooled median vs window
    for gt, qdict in percentile_tracks.items():
        col = group_colors[gt]
        median_track = qdict[50]
        label = f"{gt[0]} | {gt[1]}"
        axB.plot(
            time_windows_range,
            median_track,
            marker="o",
            ms=2,
            lw=1.5,
            color=col,
            label=label,
        )

    axB.set_title("Pooled median dFC speed vs window")
    axB.set_xlabel("Time-window size (TR)")
    axB.set_ylabel("Median dFC speed")
    axB.grid(alpha=0.3)

    # Panel C: spread (99th - 1st) vs window
    for gt, qdict in percentile_tracks.items():
        col = group_colors[gt]
        width_track = qdict[99] - qdict[1]
        label = f"{gt[0]} | {gt[1]}"
        axC.plot(
            time_windows_range,
            width_track,
            marker="o",
            ms=2,
            lw=1.5,
            color=col,
            label=label,
        )

    axC.set_title("Spread of distribution (99th - 1st pct) vs window")
    axC.set_xlabel("Time-window size (TR)")
    axC.set_ylabel("Width of distribution")
    axC.grid(alpha=0.3)

    axC.legend(
        title="genotype | treatment",
        loc="upper right",
        frameon=False,
        fontsize=9,
    )

    fig.suptitle(
        f"QC: dFC speed vs window size ({DATASET_NAME}, subset='{subset_label}')",
        y=1.02,
        fontsize=14,
    )
    plt.tight_layout()

    if save_fig:
        outpath_fig = qc_folder / f"qc_speed_vs_window_subset_{subset_label}.png"
        fig.savefig(outpath_fig, dpi=200, bbox_inches="tight")
        print(f"  -> saved {outpath_fig}")

    plt.close(fig)


# ------------------------------------------------------------------------
# MAIN LOOP OVER SUBSETS
# ------------------------------------------------------------------------
for subset in SPEED_SUBSETS:
    print(f"\n[QC 3-panel] subset = {subset}")

    # ----------------------------------------------------
    # SPECIAL CASE: per_region → loop over all ROIs
    # ----------------------------------------------------
    if subset == "per_region":
        subset_dir = speed_root / "per_region"
        try:
            region_descs = discover_per_region_descriptors(
                subset_dir=subset_dir,
                w0=int(time_windows_range[0]),
                n_animals=n_animals,
                regions=regions,
                lag=1,
                tau_count=2,
            )
        except FileNotFoundError as e:
            print(f"  -> per_region: no per-region files found: {e}")
            continue

        print(f"  -> per_region: found {len(region_descs)} regions")

        for region_desc in region_descs:
            subset_label = f"per_region_{region_desc}"
            print(f"    [QC per_region] region = {region_desc}")

            try:
                speeds = load_speed_stack_single_region(
                    subset_dir=subset_dir,
                    time_windows_range=time_windows_range,
                    n_animals=n_animals,
                    regions=regions,
                    region_desc=region_desc,
                    lag=1,
                    tau_count=2,
                )
            except FileNotFoundError as e:
                print(f"      -> skipping region {region_desc} (missing files): {e}")
                continue

            run_qc_for_subset_label(
                subset_label=subset_label,
                speeds=speeds,
                time_windows_range=time_windows_range,
                group_data=group_data,
                group_keys=group_keys,
                qc_folder=qc_folder,
                save_fig=save_fig,
            )

        continue  # don't fall through to the standard logic

    # ----------------------------------------------------
    # STANDARD CASE: global subset (all, regions500, within, touching)
    # ----------------------------------------------------
    paths_speed_root = subset_speed_template.format(subset=subset)

    try:
        speeds = load_speed_stack(
            paths_speed_root,
            time_windows_range,
            n_animals=n_animals,
            regions=regions,
        )
    except FileNotFoundError as e:
        print(f"  -> skipping subset '{subset}' (missing files): {e}")
        continue

    run_qc_for_subset_label(
        subset_label=subset,
        speeds=speeds,
        time_windows_range=time_windows_range,
        group_data=group_data,
        group_keys=group_keys,
        qc_folder=qc_folder,
        save_fig=save_fig,
    )

print("\n[INFO] QC script finished.")
