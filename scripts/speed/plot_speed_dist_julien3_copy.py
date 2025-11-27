#!/usr/bin/env python3
"""
Lightweight version of the legacy speed distribution plotter.

This script keeps only the pieces needed to:
  - load speed NPZ files for a window range,
  - derive simple pooled segments (all/half/third),
  - plot pooled histograms and basic percentile tracks per group.

Heavy bootstrap utilities and unused helpers from the original script were removed
to keep the entrypoint runnable and easier to maintain.
"""
#%%
from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None

from scripts.dfc.dfc_compute import DATASET_DEFAULTS, _canonical_dataset
from shared_code.fun_loaddata import load_timeseries_bundle
from shared_code.fun_paths import get_paths
from shared_code.fun_utils import load_cognitive_data, set_figure_params


# ----------------- Small helpers kept from the original script -----------------
def combo_color(genotype: str, treatment: str) -> str:
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


def load_speed_stack(
    template: Path, time_windows_range: Sequence[int], n_animals: int, n_regions: int
) -> list[np.ndarray]:
    """Return list S where S[j][i] is 1D array of samples for animal i at window j."""
    speeds: list[np.ndarray] = []
    for w in time_windows_range:
        path = Path(str(template).format(w=w, n_animals=n_animals, regions=n_regions))
        with np.load(path, allow_pickle=True) as a:
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


def hist_prob(x: np.ndarray, bins: int, rng: tuple[float, float]):
    h, e = np.histogram(x, bins=bins, range=rng, density=False)
    s = h.sum()
    return (h / s if s > 0 else np.zeros_like(h)), e


def robust_percentiles(x: np.ndarray, qs=(1, 5, 95, 99)) -> dict[int, float]:
    if x.size == 0 or not np.isfinite(x).any():
        return {int(q): np.nan for q in qs}
    x = x[np.isfinite(x)]
    ps = np.percentile(x, qs)
    return {int(q): float(p) for q, p in zip(qs, ps, strict=False)}


# --------------------------- Grouping utilities ---------------------------
def make_long_cog(cog_data: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Return a long-form cognitive dataframe with Age and phenotypes normalized."""
    if dataset_name == "julien":
        df = cog_data.copy()
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

    df["Sexe"] = df["Sexe"].map({"F": "female", "M": "male"}).fillna(df["Sexe"])
    for col in ["Sexe", "Age", "Genotype", "Phenotype_OiP", "Phenotype_RO24h"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def group_indices(df: pd.DataFrame, by: Sequence[str]):
    """Return a dict {group_key: indices}."""
    if not by:
        return {"all": np.arange(len(df), dtype=int)}
    gb = df.groupby(list(by), sort=False)
    return {k: v.values for k, v in gb.groups.items()}


GROUP_RECIPES = {
    "sex": ["Sexe"],
    "age": ["Age"],
    "genotype": ["Genotype"],
    "phenotype_oip": ["Phenotype_OiP"],
    "phenotype_nor": ["Phenotype_RO24h"],
    "age_sex": ["Age", "Sexe"],
    "age_genotype": ["Age", "Genotype"],
    "age_phenotype_oip": ["Age", "Phenotype_OiP"],
    "age_phenotype_nor": ["Age", "Phenotype_RO24h"],
    "sex_genotype": ["Sexe", "Genotype"],
    "age_sex_genotype": ["Sexe", "Age", "Genotype"],
    "age_sex_phenotype_oip": ["Sexe", "Age", "Phenotype_OiP"],
    "age_sex_phenotype_nor": ["Sexe", "Age", "Phenotype_RO24h"],
    "genotype_treatment": ["genotype", "treatment"],  # keep legacy hook
}


def get_group_data(cog_data: pd.DataFrame, dataset_name: str, groups_selected: str):
    df_long = make_long_cog(cog_data, dataset_name)
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


# ----------------------------- Plotting helpers -----------------------------
def plot_percentile_tracks(
    ax,
    time_windows_range: Sequence[int],
    group_data: dict,
    speeds: list[np.ndarray],
    percentiles=(1, 5, 50, 95, 99),
):
    for (genotype, treatment), indices in group_data.items():
        color = combo_color(genotype, treatment)
        tracks = {p: [] for p in percentiles}
        for j in range(len(time_windows_range)):
            pooled = (
                np.concatenate([speeds[j][i].ravel() for i in indices])
                if len(indices)
                else np.array([], dtype=float)
            )
            stats = robust_percentiles(pooled, qs=percentiles)
            for p in percentiles:
                tracks[p].append(stats[p])
        ax.plot(
            time_windows_range,
            tracks[50],
            ".-",
            alpha=0.7,
            color=color,
            label=combo_label(genotype, treatment),
        )
        ax.fill_between(
            time_windows_range,
            tracks[5],
            tracks[95],
            color=color,
            alpha=0.15,
        )
    ax.set_xlabel("Time window")
    ax.set_ylabel("dFC speed")
    ax.set_title("Median ± (5th,95th) percentiles per group")
    ax.grid(alpha=0.2)
    ax.legend()


def plot_pooled_hist(
    ax,
    segments: dict[str, range],
    speeds: list[np.ndarray],
    edges: np.ndarray,
    pool_split: str,
):
    centers = 0.5 * (edges[:-1] + edges[1:])
    for name, seg in segments.items():
        pooled = flatten_windows(speeds, seg.start, seg.stop)
        hist, _ = hist_prob(pooled, len(edges) - 1, (edges[0], edges[-1]))
        ax.plot(centers, hist, label=name, alpha=0.8)
    ax.set_xlabel("Speed")
    ax.set_ylabel("Probability per bin")
    ax.set_title(f"Pooled speed histograms ({pool_split})")
    ax.legend()
    ax.grid(alpha=0.2)


# ----------------------------- CLI and main -----------------------------
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-name", default="ines", help="Dataset alias (ines|julien)")
    ap.add_argument(
        "--window-min", type=int, default=5, help="Minimum window size (inclusive)"
    )
    ap.add_argument(
        "--window-max", type=int, default=99, help="Maximum window size (inclusive)"
    )
    ap.add_argument(
        "--window-step", type=int, default=1, help="Window step for sweep"
    )
    ap.add_argument(
        "--speed-template",
        default="all/speed_win{w}_lag1_tau2_animals_{n_animals}_regions_{regions}.npz",
        help="Path under paths['speed'] with placeholders {w},{n_animals},{regions}",
    )
    ap.add_argument(
        "--pool-split",
        choices=["all", "half", "third"],
        default="third",
        help="How to pool windows for pooled histograms",
    )
    ap.add_argument(
        "--bins-hist", type=int, default=200, help="Histogram bin count for pooled plots"
    )
    ap.add_argument(
        "--grouping",
        default="age_sex",
        help=f"Grouping recipe ({', '.join(sorted(GROUP_RECIPES.keys()))})",
    )
    ap.add_argument(
        "--no-show", action="store_true", help="Skip plt.show() (useful for batch runs)"
    )
    return ap.parse_args()


def main(argv=None) -> int:
    args = parse_args(argv)
    dataset = _canonical_dataset(args.dataset_name)
    cfg = DATASET_DEFAULTS[dataset]

    # Figure setup (uses global matplotlib RC adjustments)
    save_fig = set_figure_params(True)
    _ = save_fig  # kept for parity; not saving to disk here

    time_windows_range = np.arange(args.window_min, args.window_max + 1, args.window_step)

    paths = get_paths(
        dataset_name=dataset,
        timecourse_folder=cfg["timecourse_folder"],
        cognitive_data_file=cfg["cognitive_data_file"],
        anat_labels_file=cfg["anat_labels_file"],
    )
    speed_root = Path(paths["speed"])
    preprocessed_root = Path(paths["preprocessed"])

    bundle = load_timeseries_bundle(preprocessed_root / "ts_and_meta_2m4m.npz")
    n_animals = bundle.n_animals
    n_regions = bundle.n_regions
    total_tr = bundle.total_tr

    cog_data = load_cognitive_data(
        preprocessed_root
        / f"cog_data_filtered_animals_{n_animals}_regions_{n_regions}_tr_{total_tr}.csv"
    )

    speed_template = speed_root / args.speed_template
    speeds = load_speed_stack(speed_template, time_windows_range, n_animals, n_regions)
    n_windows = len(speeds)

    counts = count_samples_per_window(speeds)
    pooled_speeds_cdf = (
        np.cumsum(counts) / np.sum(counts) if counts.sum() else np.zeros_like(counts)
    )
    i_third, i_half, i_two_third = cdf_split_indices(speeds)
    ranges = select_windows(args.pool_split, n_windows, i_third, i_half, i_two_third)

    all_speeds_flat = flatten_windows(speeds, 0, len(speeds))
    all_speeds_min, all_speeds_max = global_min_max([all_speeds_flat])
    edges = np.linspace(all_speeds_min, all_speeds_max, args.bins_hist + 1)

    group_data = get_group_data(cog_data, dataset, args.grouping)

    # ---- Plots ----
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    plot_percentile_tracks(ax1, time_windows_range, group_data, speeds)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    plot_pooled_hist(ax2, ranges, speeds, edges, args.pool_split)

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.plot(time_windows_range, pooled_speeds_cdf, color="orange", lw=2, alpha=0.8)
    ax3.set_title("Cumulative samples across windows")
    ax3.set_xlabel("Time window")
    ax3.set_ylabel("Cumulative fraction")
    step = max(1, len(time_windows_range) // 12)
    ax3.set_xticks(time_windows_range[::step])
    ax3.grid(alpha=0.2)

    if not args.no_show:
        plt.show()

    # Console summary
    print(
        f"[summary] dataset={dataset}, windows={len(time_windows_range)}, "
        f"animals={n_animals}, regions={n_regions}, grouping={args.grouping}"
    )
    print(
        "[summary] pooled segments:",
        {k: (time_windows_range[v.start], time_windows_range[v.stop - 1]) for k, v in ranges.items()},
    )
    print(
        "[summary] group sizes:",
        {str(k): len(v) for k, v in group_data.items()},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

# %%
