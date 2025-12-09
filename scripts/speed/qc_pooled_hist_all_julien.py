#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick QC for pooled dFC speed histograms (Block 5 reimplementation) for ALL subsets.

- Loops over SPEED_SUBSETS (all, regions500, dmn_touching, etc.)
- For each subset:
    * loads the speed stack
    * pools speeds across windows
    * optionally splits windows into short/mid/long (or short/long)
    * plots pooled histograms

Run from repo root:

    python scripts/speed/qc_pooled_hist_block5_all_subsets.py

Adjust DATASET_NAME / POOL_SPLIT / BINS_HIST below if needed.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.dfc.dfc_compute import DATASET_DEFAULTS, _canonical_dataset
from shared_code.fun_loaddata import load_timeseries_bundle
from shared_code.fun_paths import get_paths


# ---------------------------------------------------------
# Subsets – must match the compute script
# ---------------------------------------------------------
SPEED_SUBSETS = [
    "all",
    "regions500",
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


# ---------------------------------------------------------
# Helpers copied from the big script
# ---------------------------------------------------------
def load_speed_stack(
    template: str,
    time_windows_range: Sequence[int],
    n_animals: int,
    regions: int,
) -> list[np.ndarray]:
    """
    Return list S where S[j][i] is 1D np.array of samples for animal i at window j.

    template: string with placeholders {subset}, {w}, {n_animals}, {regions}
              here we pass a template already filled with subset.
    """
    speeds: list[np.ndarray] = []
    for w in time_windows_range:
        fname = template.format(w=w, n_animals=n_animals, regions=regions)
        a = np.load(fname, allow_pickle=True)
        s = a["speeds"]
        # ensure 1D float arrays
        s_flat = [np.asarray(x, dtype=float).ravel() for x in s]
        speeds.append(s_flat)
    return speeds


def count_samples_per_window(speeds: list[np.ndarray]) -> np.ndarray:
    return np.array([sum(len(x) for x in speed) for speed in speeds], dtype=int)


def cdf_split_indices(speeds: list[np.ndarray]) -> tuple[int, int, int]:
    """
    Given speeds[j][i] arrays, compute the window indices corresponding to
    1/3, 1/2, and 2/3 of the cumulative sample count.
    """
    counts = count_samples_per_window(speeds)
    if counts.sum() > 0:
        cdf = np.cumsum(counts) / counts.sum()
    else:
        cdf = np.zeros_like(counts, dtype=float)

    i_third = int(np.searchsorted(cdf, 1.0 / 3.0))
    i_half = int(np.searchsorted(cdf, 0.5))
    i_two_third = int(np.searchsorted(cdf, 2.0 / 3.0))

    # minimal sanity
    i_third = max(1, i_third)
    i_half = max(1, i_half)
    i_two_third = max(i_third + 1, i_two_third)
    return i_third, i_half, i_two_third


def flatten_windows(
    speeds: list[np.ndarray],
    start: int,
    end: int,
) -> np.ndarray:
    """
    Flatten all animal samples between window indices [start, end)
    into a single 1D array.
    """
    arrays = [
        np.asarray(s, dtype=float).ravel()
        for speed in speeds[start:end]
        for s in speed
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


def hist_prob(x: np.ndarray, bins: int, rng: tuple[float, float]):
    """
    Histogram -> probability per bin (not density), plus bin edges.
    """
    h, e = np.histogram(x, bins=bins, range=rng, density=False)
    s = h.sum()
    p = h / s if s > 0 else np.zeros_like(h)
    return p, e


# ---------------------------------------------------------
# Block-5 QC: pooled histograms for one subset
# ---------------------------------------------------------
def plot_pooled_speed_histograms(
    speeds: list[np.ndarray],
    pool_split: str,
    bins_hist: int,
    hist_range: tuple[float, float] | None = None,
    title_suffix: str = "",
    save_path: Path | None = None,
):
    """
    Reimplementation of your Block 5 for a single subset:

    - Compute pooled histogram over *all* windows
    - Optionally also "short/mid/long" or "short/long" depending on pool_split
    - Plot everything on one figure
    """
    n_windows = len(speeds)

    # 1) global min/max for histogram range
    if hist_range is None:
        all_flat = flatten_windows(speeds, 0, n_windows)
        vmin, vmax = global_min_max([all_flat])
    else:
        vmin, vmax = hist_range

    # 2) compute window split indices
    i_third, i_half, i_two_third = cdf_split_indices(speeds)

    # 3) pooled arrays depending on split
    all_speeds_flat = flatten_windows(speeds, 0, n_windows)
    all_speeds_hist, bin_edge = hist_prob(
        all_speeds_flat, bins_hist, (vmin, vmax)
    )

    if pool_split == "half":
        short_speeds_flat = flatten_windows(speeds, 0, i_half)
        long_speeds_flat = flatten_windows(speeds, i_half, n_windows)
    elif pool_split == "third":
        short_speeds_flat = flatten_windows(speeds, 0, i_third)
        mid_speeds_flat = flatten_windows(speeds, i_third, i_two_third)
        long_speeds_flat = flatten_windows(speeds, i_two_third, n_windows)
    else:
        short_speeds_flat = mid_speeds_flat = long_speeds_flat = None

    # 4) plot
    plt.figure(figsize=(7, 5))
    plt.title(f"Pooled dFC speed {title_suffix}".strip())

    # all animals, all windows
    plt.plot(
        bin_edge[:-1],
        all_speeds_hist,
        color="dodgerblue",
        lw=2,
        alpha=0.8,
        label="all windows",
    )

    if pool_split == "half":
        plt.plot(
            bin_edge[:-1],
            hist_prob(short_speeds_flat, bins_hist, (vmin, vmax))[0],
            color="orange",
            lw=2,
            alpha=0.8,
            label="short windows",
        )
        plt.plot(
            bin_edge[:-1],
            hist_prob(long_speeds_flat, bins_hist, (vmin, vmax))[0],
            color="green",
            lw=2,
            alpha=0.8,
            label="long windows",
        )
    elif pool_split == "third":
        plt.plot(
            bin_edge[:-1],
            hist_prob(short_speeds_flat, bins_hist, (vmin, vmax))[0],
            color="orange",
            lw=2,
            alpha=0.8,
            label="short windows",
        )
        plt.plot(
            bin_edge[:-1],
            hist_prob(mid_speeds_flat, bins_hist, (vmin, vmax))[0],
            color="purple",
            lw=2,
            alpha=0.8,
            label="mid windows",
        )
        plt.plot(
            bin_edge[:-1],
            hist_prob(long_speeds_flat, bins_hist, (vmin, vmax))[0],
            color="green",
            lw=2,
            alpha=0.8,
            label="long windows",
        )

    plt.legend()
    plt.xlabel("dFC speed")
    plt.ylabel("Probability per bin")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] QC pooled hist figure saved to: {save_path}")

    plt.show()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    # === USER SETTINGS HERE ===
    DATASET_NAME = "julien"   # "julien" or "ines"
    POOL_SPLIT = "third"      # "half", "third" or "all"
    BINS_HIST = 200
    # ==========================

    dataset = _canonical_dataset(DATASET_NAME)
    cfg = DATASET_DEFAULTS[dataset]

    paths = get_paths(
        dataset_name=dataset,
        timecourse_folder=cfg["timecourse_folder"],
        cognitive_data_file=cfg["cognitive_data_file"],
        anat_labels_file=cfg["anat_labels_file"],
    )

    speed_root = Path(paths["speed"])
    preprocessed_root = Path(paths["preprocessed"])

    # load meta to know n_animals / regions
    loaddir_ts_meta = preprocessed_root / "ts_and_meta_2m4m.npz"
    bundle = load_timeseries_bundle(loaddir_ts_meta)
    n_animals = bundle.n_animals
    regions = bundle.n_regions

    # same window range as the big script
    time_windows_range = np.arange(5, 100, 1)

    qc_base_dir = Path(paths["f_speed"]) / "qc_pooled_hist"

    for subset in SPEED_SUBSETS:
        print(f"\n[QC pooled hist] subset = {subset}")

        # template with this subset
        loaddir_speed_template = str(
            speed_root
            / subset
            / "speed_win{w}_lag1_tau2_animals_{n_animals}_regions_{regions}.npz"
        )

        # try to load; skip if missing
        try:
            speeds = load_speed_stack(
                loaddir_speed_template,
                time_windows_range,
                n_animals=n_animals,
                regions=regions,
            )
        except FileNotFoundError:
            print(f"  -> speed files not found for subset '{subset}', skipping.")
            continue
        except Exception as e:
            print(f"  -> error loading subset '{subset}': {e}")
            continue

        qc_outdir = qc_base_dir / subset
        qc_fig_path = qc_outdir / f"pooled_hist_{DATASET_NAME}_{subset}_{POOL_SPLIT}_bins{BINS_HIST}.png"

        plot_pooled_speed_histograms(
            speeds=speeds,
            pool_split=POOL_SPLIT,
            bins_hist=BINS_HIST,
            hist_range=None,
            title_suffix=f"({DATASET_NAME}, subset={subset}, split={POOL_SPLIT})",
            save_path=qc_fig_path,
        )
