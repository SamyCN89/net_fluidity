#!/usr/bin/env python3
"""
Plot merged dFC speed outputs (speeds across windows) similar to local_speed_plot_v2.py.

Reads the merged PKL produced by 3_dfc_speed_test_v6.py and generates:
- Overall pooled distribution (all animals, all taus pooled for the last window)
- Per-group distributions (genotype × treatment)
- Median speed vs window size per group (optionally for a specific tau)

Usage examples:

  python julien_data/speed_plots.py --subset-name all
  python julien_data/speed_plots.py --subset-name regions-ACC-THAL --tau 0

If --subset-name is omitted, the script will auto-detect a merged PKL under paths['speed'].
"""
# %%
import argparse
from collections.abc import Iterable, Sequence
from itertools import combinations
import logging
from pathlib import Path
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from scripts.bootstrap.compute_speed_bootstrap import (
    BootstrapConfig,
    _concat_per_animal,
    _find_region_folders,
    _list_window_files,
    _pool_windows_indices,
    _resolve_group_columns,
    build_groups_from_columns,
    load_dataset_context,
    load_per_animal_from_npz,
)
from scripts.dfc.dfc_compute import _canonical_dataset
from shared_code.fun_bootstrap import pool_per_animal
from shared_code.fun_loaddata import load_timeseries_bundle

LOGGER = logging.getLogger(__name__)


# Robust import of DFCAnalysis (prefer src package during migration)
try:
    from net_fluidity_julien.context import DFCAnalysis
except ModuleNotFoundError:
    try:
        from julien_data.class_dataanalysis_julien import DFCAnalysis
    except ModuleNotFoundError:
        from class_dataanalysis_julien import DFCAnalysis

# Shared plotting utilities (import from src/, with fallback if not installed as a package)
try:
    from net_fluidity_julien.plots_utils import (
        pool_speeds_per_animal as _pu_per_animal,
        pool_window_speeds as _pu_pool_window_speeds,
        split_window_indices as _pu_split,
        subsample_equal_length as _pu_subsample,
    )
except ModuleNotFoundError:
    try:
        from julien_data.src.plots_utils import (
            pool_speeds_per_animal as _pu_per_animal,
            pool_window_speeds as _pu_pool_window_speeds,
            split_window_indices as _pu_split,
            subsample_equal_length as _pu_subsample,
        )
    except ModuleNotFoundError:
        try:
            from src.plots_utils import (
                pool_speeds_per_animal as _pu_per_animal,
                pool_window_speeds as _pu_pool_window_speeds,
                split_window_indices as _pu_split,
                subsample_equal_length as _pu_subsample,
            )
        except ModuleNotFoundError:
            # Last resort: define no-op fallbacks
            def _pu_pool_window_speeds(win_array, tau=None):
                return np.array([], float)

            def _pu_per_animal(win_array, idxs, tau=None):
                return [np.array([], float) for _ in idxs]

            def _pu_subsample(
                per_animal, n_per_animal=None, replace=False, random_state=0
            ):
                return np.array([], float)

            def _pu_split(window_sizes, split_at=None):
                return (
                    list(range(len(window_sizes) // 2)),
                    list(range(len(window_sizes) // 2, len(window_sizes))),
                    "fallback",
                )


# %%
def find_merged_file(
    save_root: Path,
    n_animals: int,
    regions: int,
    tau_count: int,
    subset_name: str | None,
):
    """Find the merged speeds PKL in a specific subfolder (subset_name) or auto-detect under save_root."""
    if subset_name:
        subdir = save_root / subset_name
        if not subdir.exists():
            raise FileNotFoundError(f"Subset folder not found: {subdir}")
        cands = sorted(
            subdir.glob(
                f"speed_windows*_tau{tau_count}_animals_{n_animals}_regions_{regions}.pkl"
            )
        )
    else:
        cands = sorted(
            save_root.rglob(
                f"speed_windows*_tau{tau_count}_animals_{n_animals}_regions_{regions}.pkl"
            )
        )
    if not cands:
        where = (save_root / subset_name) if subset_name else save_root
        raise FileNotFoundError(f"No merged speeds PKL found under {where}")
    return cands[-1]


def pool_window_speeds(win_array, tau: int | None = None):
    return _pu_pool_window_speeds(win_array, tau=tau)


def plot_overall_distribution(last_window_vals, window_size: int, ax=None):
    ax = ax or plt.gca()
    ax.hist(last_window_vals, bins=120, density=True, histtype="step", alpha=0.9)
    ax.set_title(f"dFC Speed (W={window_size}, all animals)")
    ax.set_xlabel("Speed")
    ax.set_ylabel("Density")
    return ax


# %%
def _pool_speeds_per_animal(win_array, idxs, tau: int | None = None):
    return _pu_per_animal(win_array, idxs, tau=tau)


def _subsample_equal_length(
    per_animal,
    n_per_animal: int | None = None,
    replace: bool = False,
    random_state: int | None = 0,
):
    return _pu_subsample(
        per_animal,
        n_per_animal=n_per_animal,
        replace=replace,
        random_state=random_state,
    )


def plot_group_distributions(
    win_array,
    groups: dict,
    window_size: int,
    tau: int | None = None,
    equal_animal_weight: bool = False,
    equal_method: str = "kde",
    n_per_animal: int | None = None,
    replace: bool = False,
    normalize_density: bool = True,
    random_state: int | None = 0,
):
    import seaborn as sns  # optional; only needed for KDE

    try:
        import statsmodels.api as sm
    except Exception:
        sm = None

    sns.set_theme(style="white", context="talk")
    plt.figure(figsize=(9, 6))
    palette = sns.color_palette("tab10", n_colors=len(groups))
    for (grp, idxs), color in zip(groups.items(), palette, strict=False):
        label = f"{grp[0]}-{grp[1]}"
        if equal_animal_weight:
            per_animal = _pool_speeds_per_animal(win_array, idxs, tau=tau)
            if equal_method == "kde" and sm is not None:
                xs = np.linspace(0, 2, 500)
                curves = []
                for arr in per_animal:
                    if arr.size < 5:
                        continue
                    kde = sm.nonparametric.KDEUnivariate(arr)
                    kde.fit()
                    y = np.interp(xs, kde.support, kde.density, left=0, right=0)
                    if not normalize_density:
                        y *= arr.size
                    curves.append(y)
                if curves:
                    stacked = np.vstack(curves)
                    mean_y = (
                        np.nanmean(stacked, axis=0)
                        if normalize_density
                        else np.nansum(stacked, axis=0) / stacked.shape[0]
                    )
                    plt.plot(xs, mean_y, lw=2.5, label=label, color=color)
            elif equal_method == "subsample":
                pooled = _subsample_equal_length(
                    per_animal,
                    n_per_animal=n_per_animal,
                    replace=replace,
                    random_state=random_state,
                )
                if pooled.size:
                    sns.kdeplot(
                        pooled,
                        bw_adjust=0.5,
                        label=label,
                        color=color,
                        linewidth=2.5,
                        clip=(0, 2),
                        common_norm=normalize_density,
                    )
        else:
            pooled = []
            for a in idxs:
                if a >= len(win_array):
                    continue
                arr = win_array[a]
                if arr is None:
                    continue
                arr = np.asarray(arr, dtype=float)
                if arr.ndim == 2:
                    if tau is None:
                        pooled.append(arr[~np.isnan(arr)])
                    else:
                        if 0 <= tau < arr.shape[0]:
                            pooled.append(arr[tau][~np.isnan(arr[tau])])
            vals = np.concatenate(pooled) if pooled else np.array([])
            if vals.size == 0:
                continue
            sns.kdeplot(
                vals,
                bw_adjust=0.5,
                label=label,
                color=color,
                linewidth=2.5,
                clip=(0, 2),
                common_norm=normalize_density,
            )

    plt.title(
        f"dFC Speed per group (W={window_size}{', tau='+str(tau) if tau is not None else ', all taus'})"
    )
    plt.xlabel("Speed")
    plt.ylabel("Density")
    plt.legend(title="Group")
    plt.tight_layout()
    sns.despine(trim=True)


def plot_median_vs_window(
    all_speed, groups: dict, window_sizes: list[int], tau: int | None = None
):
    """Plot median with 25–75% quantile bands per group across window sizes."""
    plt.figure(figsize=(10, 6))
    for grp_idx, (grp, idxs) in enumerate(groups.items()):
        medians = []
        q25 = []
        q75 = []
        for w_idx, win_array in enumerate(all_speed):
            pooled = []
            for a in idxs:
                if a >= len(win_array):
                    continue
                arr = np.asarray(win_array[a], dtype=float)
                if arr.ndim == 2:
                    if tau is None:
                        pooled.append(arr[~np.isnan(arr)])
                    else:
                        if 0 <= tau < arr.shape[0]:
                            pooled.append(arr[tau][~np.isnan(arr[tau])])
            vals = np.concatenate(pooled) if pooled else np.array([])
            if vals.size:
                medians.append(np.nanmedian(vals))
                q25.append(np.nanpercentile(vals, 25))
                q75.append(np.nanpercentile(vals, 75))
            else:
                medians.append(np.nan)
                q25.append(np.nan)
                q75.append(np.nan)
        label = f"{grp[0]}-{grp[1]}"
        plt.plot(window_sizes, medians, marker=".", label=label)
        # Quantile band
        plt.fill_between(window_sizes, q25, q75, alpha=0.15)
    plt.title(
        f"Median dFC Speed vs Window Size{' (all taus)' if tau is None else f' (tau={tau})'}"
    )
    plt.xlabel("Window size")
    plt.ylabel("Median speed")
    plt.legend(title="Group")
    plt.tight_layout()


def split_window_indices(window_sizes: list[int], split_at: int | None = None):
    """Split windows into two pools.

    - If split_at is provided, Pool A = sizes <= split_at, Pool B = sizes > split_at.
    - Else, split into two equal-count halves by index. If odd count, drop the middle index
      to enforce equal sizes and report which window was dropped.

    Returns (first_idx, second_idx, info_str)
    """
    sizes = list(map(int, window_sizes))
    sizes_sorted = sizes  # assumed sorted
    n = len(sizes_sorted)
    if split_at is not None:
        first = [i for i, w in enumerate(sizes_sorted) if w <= split_at]
        second = [i for i, w in enumerate(sizes_sorted) if w > split_at]
        info = f"split_at={split_at} (A: <= {split_at}, B: > {split_at})"
        return first, second, info

    # Equal-count split by index
    if n % 2 == 0:
        mid = n // 2
        first = list(range(0, mid))
        second = list(range(mid, n))
        info = f"equal-count split between W={sizes_sorted[mid-1]} and W={sizes_sorted[mid]}"
        return first, second, info
    else:
        mid = n // 2
        # Drop the middle index to enforce equal count
        dropped = sizes_sorted[mid]
        first = list(range(0, mid))
        second = list(range(mid + 1, n))
        info = (
            f"equal-count split by index; dropped middle W={dropped}; "
            f"A up to W={sizes_sorted[mid-1]}, B from W={sizes_sorted[mid+1]}"
        )
        return first, second, info


def plot_group_distributions_two_pools(
    all_speed,
    groups: dict,
    window_sizes: list[int],
    tau: int | None = None,
    split_at: int | None = None,
):
    """Plot per-group distributions for two window pools (equal count halves)."""
    import seaborn as sns

    first_idx, second_idx, info = _pu_split(window_sizes, split_at=split_at)
    print("Window split:", info)

    def pool_vals(win_idx_list, idxs):
        pooled = []
        for w_idx in win_idx_list:
            win_array = all_speed[w_idx]
            for a in idxs:
                if a >= len(win_array):
                    continue
                arr = np.asarray(win_array[a], dtype=float)
                if arr.ndim == 2:
                    if tau is None:
                        pooled.append(arr[~np.isnan(arr)])
                    else:
                        if 0 <= tau < arr.shape[0]:
                            pooled.append(arr[tau][~np.isnan(arr[tau])])
        return np.concatenate(pooled) if pooled else np.array([])

    # First pool figure
    sns.set_theme(style="white", context="talk")
    plt.figure(figsize=(9, 6))
    palette = sns.color_palette("tab10", n_colors=len(groups))
    for (grp, idxs), color in zip(groups.items(), palette, strict=False):
        vals = pool_vals(first_idx, idxs)
        if vals.size == 0:
            continue
        label = f"{grp[0]}-{grp[1]}"
        plt.hist(
            vals,
            bins=120,
            density=True,
            histtype="step",
            lw=1.7,
            alpha=0.85,
            label=label,
            color=color,
        )
        sns.kdeplot(vals, bw_adjust=0.7, color=color, lw=2)
    plt.title("dFC Speed per group — Pool A")
    plt.xlabel("Speed")
    plt.ylabel("Density")
    plt.legend(title="Group")
    plt.tight_layout()
    sns.despine(trim=True)

    # Second pool figure
    plt.figure(figsize=(9, 6))
    palette = sns.color_palette("tab10", n_colors=len(groups))
    for (grp, idxs), color in zip(groups.items(), palette, strict=False):
        vals = pool_vals(second_idx, idxs)
        if vals.size == 0:
            continue
        label = f"{grp[0]}-{grp[1]}"
        plt.hist(
            vals,
            bins=120,
            density=True,
            histtype="step",
            lw=1.7,
            alpha=0.85,
            label=label,
            color=color,
        )
        sns.kdeplot(vals, bw_adjust=0.7, color=color, lw=2)
    plt.title("dFC Speed per group — Pool B")
    plt.xlabel("Speed")
    plt.ylabel("Density")
    plt.legend(title="Group")
    plt.tight_layout()
    sns.despine(trim=True)


# %%


def plot_qq_between_groups(
    all_speed,
    groups: dict,
    window_sizes: list[int],
    tau: int | None = None,
    pool: str = "A",
    split_at: int | None = None,
):
    """QQ plots between selected groups for a given pool (A or B) and tau."""
    first_idx, second_idx, info = _pu_split(window_sizes, split_at=split_at)
    idx_list = first_idx if str(pool).upper() == "A" else second_idx
    if not idx_list:
        print("No windows in selected pool.")
        return

    def pool_vals(win_idx_list, idxs):
        pooled = []
        for w_idx in win_idx_list:
            win_array = all_speed[w_idx]
            for a in idxs:
                if a >= len(win_array):
                    continue
                arr = np.asarray(win_array[a], dtype=float)
                if arr.ndim == 2:
                    if tau is None:
                        pooled.append(arr[~np.isnan(arr)])
                    else:
                        if 0 <= tau < arr.shape[0]:
                            pooled.append(arr[tau][~np.isnan(arr[tau])])
        return np.concatenate(pooled) if pooled else np.array([])

    # Build group arrays
    group_vals = []
    group_names = []
    for grp, idxs in groups.items():
        vals = pool_vals(idx_list, idxs)
        if vals.size:
            group_vals.append(vals)
            group_names.append(f"{grp[0]}-{grp[1]}")
    if len(group_vals) < 2:
        print("Need at least two non-empty groups for QQ plots.")
        return

    # Plot grid of QQ for all pairs
    n_pairs = len(group_vals) * (len(group_vals) - 1) // 2
    ncols = 3
    nrows = int(np.ceil(n_pairs / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 5.2 * nrows), squeeze=False
    )
    pairs = list(combinations(range(len(group_vals)), 2))
    q = np.linspace(0, 1, 400)
    for ax, (i, j) in zip(axes.flat, pairs, strict=False):
        x = np.quantile(group_vals[i], q)
        y = np.quantile(group_vals[j], q)
        ax.plot(x, y, color="k", lw=2)
        m = min(x.min(), y.min())
        M = max(x.max(), y.max())
        ax.plot([m, M], [m, M], "k--", lw=1)
        ax.set_title(f"QQ: {group_names[j]} vs {group_names[i]}")
        ax.set_xlabel(group_names[i])
        ax.set_ylabel(group_names[j])
        ax.grid(True, ls=":", alpha=0.3)
    # Hide unused axes
    for ax in axes.flat[len(pairs) :]:
        ax.axis("off")
    fig.suptitle(f"QQ plots (Pool {pool}, tau={'all' if tau is None else tau})", y=0.98)
    plt.tight_layout()


def run_plot(
    subset_name: str | None = None,
    tau: int | None = None,
    no_group: bool = False,
    no_medians: bool = False,
    savefig: bool = False,
    groups: list[str] | None = None,
    split_pools: bool = False,
    split_at: int | None = None,
    tr: int | None = None,
    equal_animal_weight: bool = False,
    equal_method: str = "kde",
    n_per_animal: int | None = None,
    replace: bool = False,
    normalize_density: bool = True,
    dataset_name: str = "julien_caillette",
    pooled_only: bool = False,
):
    """Run plotting end-to-end with parameters (usable from Jupyter)."""
    original_dataset = dataset_name
    try:
        dataset_name = _canonical_dataset(dataset_name)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported dataset '{original_dataset}'. Expected names like 'julien' or 'ines'."
        ) from exc

    use_legacy_flow = dataset_name == "julien_caillette"
    use_pooled_only = pooled_only or not use_legacy_flow
    if use_pooled_only and not pooled_only and not use_legacy_flow:
        LOGGER.info(
            "Dataset %s does not ship legacy merged PKLs; switching to pooled-only workflow.",
            dataset_name,
        )

    if use_pooled_only:
        tau_idx = tau if tau is not None else -1
        pool_subset = subset_name or "all"
        LOGGER.info(
            "Running pooled-only speed plots for dataset=%s subset=%s tau_index=%s",
            dataset_name,
            pool_subset,
            tau_idx,
        )
        pooled_results = plot_dataset_all_windows(
            dataset_name=dataset_name,
            subset=pool_subset,
            tau_index=tau_idx,
            include_all_pool=True,
            include_groups=groups,
            pool_threshold="median",
            plot_all_animals=True,
            tr=tr,
        )
        figures_found = False
        save_dir: Path | None = None
        ctx = None
        if savefig:
            try:
                ctx = load_dataset_context(dataset_name, tr_hint=tr)
            except FileNotFoundError as err:
                LOGGER.error("%s", err)
                raise
            subset_dir = Path(ctx.paths["speed"])
            if pool_subset:
                subset_dir = subset_dir / pool_subset
            subset_dir.mkdir(parents=True, exist_ok=True)
            save_dir = subset_dir / "pooled_plots"
            save_dir.mkdir(parents=True, exist_ok=True)

        tau_label = "all" if tau_idx < 0 else str(tau_idx)
        for pool_name, details in pooled_results.items():
            if not isinstance(details, dict):
                continue
            fig_list = details.get("figures", [])
            if not fig_list:
                continue
            figures_found = True
            if save_dir is not None:
                for idx, fig in enumerate(fig_list):
                    filename = f"{dataset_name}_{pool_subset}_tau{tau_label}_{pool_name}_{idx:02d}.png"
                    fig.savefig(save_dir / filename, dpi=200)
        if figures_found:
            plt.show()
        else:
            LOGGER.warning(
                "No figures produced for pooled-only plotting (dataset=%s subset=%s tau=%s)",
                dataset_name,
                pool_subset,
                tau_label,
            )
        result: dict[str, object] = {"pooled_results": pooled_results}
        if save_dir is not None:
            result["save_dir"] = save_dir
        if ctx is not None:
            result["context"] = ctx
        return result

    # Load dataset context
    data = DFCAnalysis(dataset_name=dataset_name)
    if tr is None:
        data.get_metadata()
    else:
        from pathlib import Path as _Path

        preproc = _Path(data.paths["preprocessed"])  # type: ignore[index]
        cands = sorted(preproc.glob(f"metadata_animals_*_tr_{int(tr)}.pkl"))
        if not cands:
            raise FileNotFoundError(f"No metadata file for tr={tr} under {preproc}")
        data.get_metadata(meta_filename=cands[0].name)
    data.get_ts_preprocessed()
    data.get_cogdata_preprocessed()
    data.get_temporal_parameters()

    save_root = Path(data.paths["speed"])  # contains subfolders
    tau_count = int(data.tau + 1)

    try:
        merged_path = find_merged_file(
            save_root, data.n_animals, data.regions, tau_count, subset_name
        )
    except FileNotFoundError as err:
        LOGGER.error("%s", err)
        LOGGER.error(
            "Legacy merged PKL not found. Re-run with --pooled-only to use bootstrap NPZ pools instead."
        )
        raise
    print("Merged speeds PKL:", merged_path)
    with open(merged_path, "rb") as fh:
        payload = pickle.load(fh)
    all_speed = payload["speeds"]  # list per window
    meta = payload.get("meta", {})
    print("Meta:", meta)

    # Get window sizes
    window_sizes = meta.get("window_sizes")
    if window_sizes is None:
        window_sizes = list(map(int, data.time_window_range))
    else:
        window_sizes = list(map(int, window_sizes))

    # Optionally filter groups
    plot_groups = data.groups
    if groups:
        wanted = {g.strip() for g in groups}
        # Map 'GENOTYPE-TREATMENT' strings to keys
        filtered = {}
        for k, idxs in data.groups.items():
            key_str = f"{k[0]}-{k[1]}"
            if key_str in wanted:
                filtered[k] = idxs
        if not filtered:
            print("Warning: no matching groups from:", groups)
        else:
            plot_groups = filtered

    # Overall distribution on last window
    last_window = all_speed[-1]
    vals = pool_window_speeds(last_window, tau=tau)
    print(
        "Last window pooled:",
        vals.size,
        "median:",
        np.nanmedian(vals) if vals.size else np.nan,
    )

    # Overall distribution
    plt.figure(figsize=(7, 5))
    plot_overall_distribution(vals, window_sizes[-1])
    if savefig:
        out = merged_path.with_suffix("")
        plt.savefig(out.as_posix() + f"_overall_W{window_sizes[-1]}.png", dpi=200)

    # Per-group distributions
    if not no_group:
        plot_group_distributions(
            last_window,
            plot_groups,
            window_sizes[-1],
            tau=tau,
            equal_animal_weight=equal_animal_weight,
            equal_method=equal_method,
            n_per_animal=n_per_animal,
            replace=replace,
            normalize_density=normalize_density,
        )
        if savefig:
            out = merged_path.with_suffix("")
            plt.savefig(out.as_posix() + f"_groups_W{window_sizes[-1]}.png", dpi=200)

    # Median vs window size per group with quantile bands
    if not no_medians:
        plot_median_vs_window(all_speed, plot_groups, window_sizes, tau=tau)
        if savefig:
            out = merged_path.with_suffix("")
            plt.savefig(
                out.as_posix() + f"_medians_tau{'all' if tau is None else tau}.png",
                dpi=200,
            )
    plt.show()

    # Two pools (equal halves) per-group distributions
    if split_pools and not no_group:
        plot_group_distributions_two_pools(
            all_speed, plot_groups, window_sizes, tau=tau, split_at=split_at
        )
        if savefig:
            out = merged_path.with_suffix("")
            plt.savefig(out.as_posix() + "_poolsA_groups.png", dpi=200)
            plt.savefig(out.as_posix() + "_poolsB_groups.png", dpi=200)
        plt.show()

    return {
        "all_speed": all_speed,
        "window_sizes": window_sizes,
        "groups": plot_groups,
        "cog_df": data.cog_data_filtered,
        "merged_path": merged_path,
    }


def plot_speed_distribution_ines(
    subset="all",
    window=None,
    tau_index=3,
    group_cols=("Genotype", "Sexe"),
    bins=120,
    figsize=(8, 5),
    save_path=None,
):
    """
    Plot the pooled dFC-speed histogram for the ines dataset.

    subset      -> folder under paths['speed'] (e.g. "all", "regions-ACC-THAL")
    window      -> window size; if None we take the largest available
    tau_index   -> row index inside each tau stack (-1 means pool all taus)
    group_cols  -> grouping columns from the cognitive CSV
    bins        -> histogram bins
    save_path   -> optional Path/str to save the figure
    """
    cfg = BootstrapConfig(dataset_name="ines", subset=subset, tau_index=tau_index)
    ctx = load_dataset_context(cfg.dataset_name, tr_hint=cfg.tr)

    region_dirs = _find_region_folders(Path(ctx.paths["speed"]) / (subset or ""))
    if not region_dirs:
        raise FileNotFoundError("No speed folders found for the given subset")

    # Pick the first region folder (usually “regions-…”) and list the windows
    windows = _list_window_files(region_dirs[0])
    if not windows:
        raise FileNotFoundError("No NPZ files in region folder")

    if window is None:
        window, npz_path = max(windows, key=lambda pair: pair[0])
    else:
        try:
            window, npz_path = next(pair for pair in windows if pair[0] == int(window))
        except StopIteration:
            raise ValueError(
                f"Window {window} not found; available: {[w for w,_ in windows]}"
            )

    per_animal = load_per_animal_from_npz(npz_path, tau_index=tau_index)
    pooled = pool_per_animal(per_animal, range(len(per_animal)))
    if pooled.size == 0:
        raise RuntimeError("Selected window/tau has no samples")

    groups = build_groups_from_columns(ctx.cog_df, list(group_cols))

    fig, ax = plt.subplots(figsize=figsize)
    # ax.hist(pooled, bins=bins, alpha=0.4, density=True, label="All animals", histtype='step')
    for grp, idxs in groups.items():
        vals = pool_per_animal(per_animal, idxs)
        if vals.size:
            ax.hist(
                vals,
                bins=bins,
                alpha=0.5,
                density=True,
                label=str(grp),
                histtype="step",
                lw=1.7,
            )

    ax.set(
        title=f"ines dFC speed distribution (W={window}, tau={tau_index})",
        xlabel="Speed",
        ylabel="Density",
    )
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig, ax


# %%


def _normalize_group_label(label: object) -> str:
    """Normalize group identifiers to a canonical string like 'WT-VEH'."""
    if isinstance(label, tuple):
        parts = [str(part).strip() for part in label if str(part).strip()]
    else:
        text = str(label).strip()
        if "-" in text:
            parts = [chunk.strip() for chunk in text.split("-")]
        else:
            parts = [text] if text else []
    return "-".join(parts)


def plot_dataset_all_windows(
    dataset_name: str,
    *,
    subset: str = "all",
    tau_index: int = 3,
    group_cols: tuple[str, ...] | None = None,
    group_builder=None,
    include_groups: Sequence[str] | None = None,
    pool_threshold: str | int | None = "median",
    include_all_pool: bool = True,
    figsize: tuple[float, float] = (9, 6),
    bins: int = 150,
    density: bool = True,
    alpha: float = 0.35,
    plot_all_animals: bool = False,
    tr: int | None = None,
) -> dict[str, dict]:
    """Plot pooled dFC-speed distributions for every window in the dataset.

    Optionally pass `tr` to resolve cognition metadata when multiple TR variants exist.
    `include_groups` accepts a sequence of group labels such as "WT-VEH" to filter the plotted cohorts.
    """
    dataset_key = _canonical_dataset(dataset_name)
    dataset_name = dataset_key
    cfg_kwargs = dict(dataset_name=dataset_key, subset=subset, tau_index=tau_index)
    if tr is not None:
        cfg_kwargs["tr"] = int(tr)
    cfg = BootstrapConfig(**cfg_kwargs)
    ctx = load_dataset_context(cfg.dataset_name, tr_hint=cfg.tr)

    speed_root = Path(ctx.paths["speed"]) / (subset or "")
    region_dirs = _find_region_folders(speed_root)
    if not region_dirs:
        raise FileNotFoundError(f"No region folders in {speed_root}")

    win_files = _list_window_files(region_dirs[0])
    if not win_files:
        raise FileNotFoundError(f"No speed NPZ files found in {region_dirs[0]}")
    win_lookup = {int(w): path for w, path in win_files}

    pools = (
        _pool_windows_indices([w for w, _ in win_files], pool_threshold)
        if pool_threshold
        else {}
    )
    if include_all_pool:
        pools.setdefault("all", [w for w, _ in win_files])
    if not pools:
        pools = {"all": [w for w, _ in win_files]}

    resolved_cols = list(group_cols) if group_cols is not None else []
    if group_builder is None:
        groups, group_sets = _default_group_builder(ctx, tuple(resolved_cols) or None)
    else:
        groups, group_sets = group_builder(ctx, tuple(resolved_cols) or None)

    if not groups:
        raise ValueError("Could not construct any groups for plotting.")

    if include_groups:
        normalized_lookup = {_normalize_group_label(key): key for key in groups}
        matched_keys = []
        missing: list[str] = []
        for requested in include_groups:
            normalized_req = _normalize_group_label(requested)
            if not normalized_req:
                continue
            match = normalized_lookup.get(normalized_req)
            if match is None:
                missing.append(requested)
                continue
            if match not in matched_keys:
                matched_keys.append(match)
        if matched_keys:
            groups = {key: groups[key] for key in matched_keys}
            filtered_sets = []
            for labels in group_sets:
                subset_labels = [label for label in labels if label in groups]
                if subset_labels:
                    filtered_sets.append(subset_labels)
            if not filtered_sets:
                filtered_sets = [matched_keys]
            group_sets = filtered_sets
            if missing:
                LOGGER.debug(
                    "Ignoring unmatched group filters for dataset=%s subset=%s: %s",
                    dataset_name,
                    subset,
                    ", ".join(sorted(_normalize_group_label(name) for name in missing)),
                )
        else:
            LOGGER.warning(
                "No pooled groups matched requested filters %s (dataset=%s subset=%s); plotting all groups.",
                include_groups,
                dataset_name,
                subset,
            )

    def pool_for(pool_windows: list[int]) -> list[np.ndarray]:
        per_animals = []
        for w in pool_windows:
            npz_path = win_lookup.get(int(w))
            if npz_path is None:
                LOGGER.warning("Window %s not found; skipping.", w)
                continue
            series = load_per_animal_from_npz(
                npz_path,
                tau_index=tau_index if tau_index >= 0 else None,
                n_animals=cfg.n_animals if cfg.n_animals > 0 else None,
            )
            if any(getattr(arr, "size", 0) > 0 for arr in series):
                per_animals.append(series)
        if not per_animals:
            return []
        return _concat_per_animal(per_animals)

    results: dict[str, dict] = {}
    LOGGER.info(
        "Plotting pooled distributions for dataset=%s subset=%s (tau=%s) across %d pools",
        dataset_key,
        subset,
        tau_index,
        len(pools),
    )
    for pool_name, window_list in pools.items():
        per_animal = pool_for(window_list)
        if not per_animal:
            LOGGER.warning(
                "Skipping pool '%s' (dataset=%s, subset=%s): no samples after pooling %s windows.",
                pool_name,
                dataset_name,
                subset,
                len(window_list),
            )
            continue

        figures = []
        for label_group in group_sets:
            labels = (
                [label_group] if isinstance(label_group, str) else list(label_group)
            )
            fig, ax = plt.subplots(figsize=figsize)
            if plot_all_animals:
                all_vals = pool_per_animal(per_animal, range(len(per_animal)))
                if all_vals.size:
                    ax.hist(
                        all_vals,
                        bins=bins,
                        density=density,
                        alpha=0.25,
                        color="grey",
                        label="All animals",
                        histtype="step",
                        lw=2,
                    )
            for label in labels:
                idxs = groups.get(label)
                if idxs is None:
                    LOGGER.warning("Group '%s' not present; skipping.", label)
                    continue
                vals = pool_per_animal(per_animal, idxs)
                if vals.size == 0:
                    LOGGER.debug("Group '%s' is empty in pool '%s'.", label, pool_name)
                    continue
                ax.hist(
                    vals,
                    bins=bins,
                    density=density,
                    alpha=alpha,
                    label=str(label),
                    histtype="step",
                    lw=2,
                )
            ax.set(
                title=f"{dataset_name} dFC speed – pool '{pool_name}' (tau={tau_index})",
                xlabel="Speed",
                ylabel="Density" if density else "Count",
            )
            ax.legend()
            fig.tight_layout()
            figures.append(fig)
        results[pool_name] = {"windows": sorted(window_list), "figures": figures}
    return results


def plot_ines_all_windows(
    group_cols=("Genotype", "Sexe"),
    subset="all",
    tau_index=3,
    pool_threshold="median",
    include_all_pool=True,
    figsize=(9, 6),
    bins=150,
    density=True,
    plot_all_animals=False,
):
    """Convenience wrapper that applies the pooled-window plot to the Ines dataset."""
    return plot_dataset_all_windows(
        dataset_name="ines",
        subset=subset,
        tau_index=tau_index,
        group_cols=group_cols,
        group_builder=_build_groups_ines,
        pool_threshold=pool_threshold,
        include_all_pool=include_all_pool,
        figsize=figsize,
        bins=bins,
        density=density,
        plot_all_animals=plot_all_animals,
    )


def _default_group_builder(ctx, requested_cols: tuple[str, ...] | None):
    dataset_key = getattr(ctx, "dataset_key", "")
    if requested_cols:
        columns = list(requested_cols)
        resolved = _resolve_group_columns(ctx.cog_df, columns)
    else:
        candidate_sets: list[list[str]] = []

        def _add_candidates(names: Iterable[str] | None):
            if not names:
                return
            cleaned = [str(n).strip() for n in names if str(n).strip()]
            if cleaned and cleaned not in candidate_sets:
                candidate_sets.append(cleaned)

        dataset_defaults = {
            "julien_caillette": ("Genotype", "Treatment"),
            "ines_abdallah": ("Genotype", "Sexe"),
        }
        _add_candidates(dataset_defaults.get(dataset_key))

        cfg = BootstrapConfig(dataset_name=dataset_key or "ines")
        _add_candidates(str(cfg.group_cols).split(","))

        heuristic_pool = [
            "Genotype",
            "Sexe",
            "Treatment",
            "Phenotype",
            "Phenotype_OiP",
            "Phenotype_RO24h",
        ]
        _add_candidates(heuristic_pool)

        categorical_cols = [
            str(col)
            for col in ctx.cog_df.columns
            if getattr(ctx.cog_df[col], "dtype", None) is not None
            and (
                str(ctx.cog_df[col].dtype).startswith(("object", "category"))
                or ctx.cog_df[col].dtype == "O"
            )
        ]
        _add_candidates(categorical_cols[:3])
        _add_candidates([str(col) for col in ctx.cog_df.columns[:3]])

        resolved = None
        last_error: Exception | None = None
        for cols in candidate_sets:
            try:
                resolved = _resolve_group_columns(ctx.cog_df, cols)
                break
            except (KeyError, ValueError) as err:
                last_error = err
        if resolved is None:
            if last_error:
                raise last_error
            raise ValueError(
                "Unable to determine default grouping columns for pooled plots."
            )

    groups = build_groups_from_columns(ctx.cog_df, resolved)
    return groups, [list(groups.keys())]


def _build_groups_ines(ctx, _=None):
    """Construct Ines-specific grouping masks using the preprocessed bundle."""
    pre_dir = Path(ctx.paths["preprocessed"])
    candidates = [
        pre_dir / "ts_and_meta_ines_abdallah.npz",
        pre_dir / "ts_and_meta_2m4m.npz",
    ]
    bundle_path = next((p for p in candidates if p.exists()), None)
    if bundle_path is None:
        matches = sorted(pre_dir.glob("ts_and_meta*.npz"))
        bundle_path = matches[-1] if matches else None
    if bundle_path is None:
        raise FileNotFoundError(
            f"Could not locate a ts_and_meta bundle under {pre_dir}"
        )

    grouping_path = next(
        (p for p in sorted(pre_dir.glob("grouping_data*.pkl")) if p.exists()), None
    )
    bundle = load_timeseries_bundle(bundle_path, grouping_path)
    mask_groups = bundle.mask_groups or ()
    label_variables = bundle.label_variables or ()
    if not mask_groups or not label_variables:
        LOGGER.warning(
            "Ines bundle %s lacks grouping masks; falling back to default genotype/treatment groups.",
            bundle_path,
        )
        return _default_group_builder(ctx, ("Genotype", "Sexe"))

    group_dict: dict[str, list[int]] = {}
    label_sets: list[list[str]] = []
    total_sets = len(label_variables)
    for i, labels in enumerate(label_variables):
        suffix = ""
        if total_sets == 2:
            suffix = " OiP" if i == 0 else " NOR"
        elif total_sets > 2:
            suffix = f" #{i+1}"
        label_group: list[str] = []
        for lbl, mask in zip(labels, mask_groups[i], strict=False):
            mask = np.asarray(mask, dtype=bool)
            indices = np.flatnonzero(mask)
            if indices.size == 0:
                continue
            name = f"{lbl}{suffix}"
            group_dict[name] = indices.tolist()
            label_group.append(name)
        if label_group:
            label_sets.append(label_group)
    if not label_sets:
        label_sets = [list(group_dict.keys())]
    return group_dict, label_sets


# %%


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Plot merged dFC speed outputs",
        allow_abbrev=False,
    )
    ap.add_argument(
        "--subset-name",
        type=str,
        default=None,
        help="Subfolder in speed/ (e.g., 'all', 'regions-ACC-THAL', or custom via --subset-name during compute)",
    )
    ap.add_argument(
        "--tau",
        type=int,
        default=None,
        help="Tau index to plot (default: pool all taus)",
    )
    ap.add_argument(
        "--no-group",
        action="store_true",
        help="Skip per-group plots; only show overall",
    )
    ap.add_argument(
        "--no-medians", action="store_true", help="Skip median vs window plot"
    )
    ap.add_argument(
        "--savefig", action="store_true", help="Save figures next to merged file"
    )
    ap.add_argument(
        "--split-pools",
        action="store_true",
        help="Also plot per-group distributions for two window pools (equal halves or at a specified split)",
    )
    ap.add_argument(
        "--split-at",
        type=int,
        default=None,
        help="Window size threshold to split pools: Pool A <= split_at, Pool B > split_at",
    )
    ap.add_argument(
        "--groups",
        type=str,
        default=None,
        help="Comma-separated group names 'GENOTYPE-TREATMENT' to include (e.g., 'WT-VEH,Dp1Yey-LCTB92')",
    )
    ap.add_argument(
        "--tr",
        type=int,
        default=None,
        help="Select metadata by total_tr for plotting context (e.g., 400 or 500)",
    )
    ap.add_argument(
        "--qq",
        action="store_true",
        help="Produce QQ plots between selected groups for the chosen pool and tau",
    )
    ap.add_argument(
        "--qq-pool",
        type=str,
        default="A",
        choices=["A", "B"],
        help="Pool to use for QQ plots (A or B)",
    )
    # Cognition correlations
    ap.add_argument(
        "--cog-scatter",
        action="store_true",
        help="Scatter of per-animal dFC speed summary vs cognition",
    )
    ap.add_argument(
        "--cog-var",
        type=str,
        default="index_NOR",
        help="Cognitive variable (column in cog df)",
    )
    ap.add_argument(
        "--weighting",
        type=str,
        default="animal",
        choices=["animal", "sample"],
        help="Per-animal summary weighting",
    )
    ap.add_argument(
        "--equalize-length",
        action="store_true",
        help="Equalize sample count per animal before summary",
    )
    ap.add_argument(
        "--reducer", type=str, default="median", help="Reducer: median|mean|qXX"
    )
    ap.add_argument(
        "--corr-vs-window",
        action="store_true",
        help="Compute and plot Spearman correlation vs window size",
    )
    ap.add_argument(
        "--equal-animal-weight",
        action="store_true",
        help="Equal animal weighting in group distributions (average per-animal KDE or subsample)",
    )
    ap.add_argument(
        "--equal-method",
        type=str,
        default="kde",
        choices=["kde", "subsample"],
        help="Equal-animal method",
    )
    ap.add_argument(
        "--n-per-animal",
        type=int,
        default=None,
        help="Equal length per animal when using subsample method",
    )
    ap.add_argument(
        "--replace",
        action="store_true",
        help="Allow replacement during subsample method",
    )
    ap.add_argument(
        "--normalize-density",
        action="store_true",
        help="Normalize KDEs to density; unset to sum per-animal curves",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default="julien_caillette",
        help="Dataset key (e.g., 'julien_caillette' or 'ines_abdallah')",
    )
    ap.add_argument(
        "--pooled-only",
        action="store_true",
        help="Skip legacy merged-PKL workflow and plot pooled speeds from bootstrap NPZ files only",
    )
    if argv is None:
        args, _ = ap.parse_known_args(sys.argv[1:])  # ← key fix
    else:
        args = ap.parse_args(argv)
    groups_list = [s.strip() for s in args.groups.split(",")] if args.groups else None

    try:
        dataset_key = _canonical_dataset(args.dataset)
    except ValueError as exc:
        print(exc)
        return 2

    # Run main plots
    ctx = run_plot(
        subset_name=args.subset_name,
        tau=args.tau,
        no_group=args.no_group,
        no_medians=args.no_medians,
        savefig=args.savefig,
        groups=groups_list,
        split_pools=args.split_pools,
        split_at=args.split_at,
        tr=args.tr,
        equal_animal_weight=args.equal_animal_weight,
        equal_method=args.equal_method,
        n_per_animal=args.n_per_animal,
        replace=args.replace,
        normalize_density=args.normalize_density,
        dataset_name=dataset_key,
        pooled_only=args.pooled_only,
    )

    pooled_ctx = "all_speed" not in ctx

    # Optional: Cognition scatter
    if args.cog_scatter:
        if pooled_ctx:
            print(
                "Cognition scatter is unavailable in pooled-only mode; rerun without --pooled-only."
            )
            return 0
        all_speed = ctx["all_speed"]
        window_sizes = ctx["window_sizes"]
        groups = ctx["groups"]
        cog_df = ctx["cog_df"]
        # per-animal summary over selected taus and all windows
        try:
            from julien_data.src.plots_utils import (
                per_animal_summary as _pu_per_animal_summary,
            )
        except ModuleNotFoundError:
            from src.plots_utils import (
                per_animal_summary as _pu_per_animal_summary,  # type: ignore
            )
        x = _pu_per_animal_summary(
            all_speed,
            reducer=args.reducer,
            windows=None,
            taus=None if args.tau is None else [args.tau],
            weighting=args.weighting,
            equalize_length=args.equalize_length,
        )
        y = cog_df[args.cog_var].values
        plt.figure(figsize=(7, 5))
        plt.scatter(x, y, c="k", alpha=0.85)
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() >= 3:
            rho, p = spearmanr(x[mask], y[mask])
            plt.title(
                f"dFC speed ({args.reducer}) vs {args.cog_var} — rho={rho:.2f}, p={p:.2g}"
            )
        else:
            plt.title(f"dFC speed ({args.reducer}) vs {args.cog_var}")
        plt.xlabel(f"{args.reducer} dFC speed per animal")
        plt.ylabel(args.cog_var)
        if args.savefig:
            out = ctx["merged_path"].with_suffix("")
            plt.savefig(out.as_posix() + f"_cog_scatter_{args.cog_var}.png", dpi=200)
        else:
            plt.show()

    # Optional: Correlation vs window
    if args.corr_vs_window:
        if pooled_ctx:
            print(
                "Correlation vs window is unavailable in pooled-only mode; rerun without --pooled-only."
            )
            return 0
        all_speed = ctx["all_speed"]
        window_sizes = ctx["window_sizes"]
        groups = ctx["groups"]
        cog_df = ctx["cog_df"]
        try:
            from julien_data.src.plots_utils import (
                per_animal_summary as _pu_per_animal_summary,
            )
        except ModuleNotFoundError:
            from src.plots_utils import (
                per_animal_summary as _pu_per_animal_summary,  # type: ignore
            )
        rows = []
        for w_idx, wsize in enumerate(window_sizes):
            x = _pu_per_animal_summary(
                all_speed,
                reducer=args.reducer,
                windows=[w_idx],
                taus=None if args.tau is None else [args.tau],
                weighting=args.weighting,
                equalize_length=args.equalize_length,
            )
            y = cog_df[args.cog_var].values
            for grp, idxs in groups.items():
                idx = np.array(idxs)
                mask = ~np.isnan(x[idx]) & ~np.isnan(y[idx])
                n = int(mask.sum())
                rho, p = (
                    spearmanr(x[idx][mask], y[idx][mask])
                    if n >= 3
                    else (np.nan, np.nan)
                )
                rows.append(
                    {
                        "window_idx": w_idx,
                        "window_size": wsize,
                        "group": "-".join(grp),
                        "rho": rho,
                        "p": p,
                        "n": n,
                        "weighting": args.weighting,
                        "equalize_length": args.equalize_length,
                        "reducer": args.reducer,
                        "cog_var": args.cog_var,
                    }
                )
        import pandas as pd

        df = pd.DataFrame(rows)
        # Plot
        import seaborn as sns

        plt.figure(figsize=(12, 6))
        order = ["-".join(g) for g in groups.keys()]
        palette = sns.color_palette("tab10", n_colors=len(order))
        for color, lab in zip(palette, order, strict=False):
            sub = df[df["group"] == lab].sort_values("window_size")
            if sub.empty:
                continue
            plt.plot(
                sub["window_size"], sub["rho"], "-o", color=color, label=lab, zorder=2
            )
            sig = sub[(sub["p"] < 0.05) & sub["rho"].notna()]
            if not sig.empty:
                plt.scatter(
                    sig["window_size"],
                    sig["rho"],
                    marker="*",
                    s=120,
                    color=color,
                    edgecolor="k",
                    linewidth=0.6,
                    zorder=4,
                )
        plt.axhline(0, color="grey", linestyle="--", linewidth=1, zorder=1)
        plt.xlabel("Window Size")
        plt.ylabel(f"Spearman ρ (dFC speed, {args.cog_var})")
        plt.ylim(-1, 1)
        plt.legend(title="Group")
        plt.tight_layout()
        if args.savefig:
            out = ctx["merged_path"].with_suffix("")
            plt.savefig(out.as_posix() + f"_corr_vs_window_{args.cog_var}.png", dpi=200)

    # Optional QQ plots
    if args.qq:
        if pooled_ctx:
            print(
                "QQ plots require the legacy merged-PKL workflow; rerun without --pooled-only."
            )
            return 0
        # Reuse same context
        data = DFCAnalysis(dataset_name=dataset_key)
        if args.tr is None:
            data.get_metadata()
        else:
            preproc = Path(data.paths["preprocessed"])  # type: ignore[index]
            cands = sorted(preproc.glob(f"metadata_animals_*_tr_{int(args.tr)}.pkl"))
            if not cands:
                raise FileNotFoundError(
                    f"No metadata file for tr={args.tr} under {preproc}"
                )
            data.get_metadata(meta_filename=cands[0].name)
        data.get_ts_preprocessed()
        data.get_cogdata_preprocessed()
        data.get_temporal_parameters()
        save_root = Path(data.paths["speed"])
        tau_count = int(data.tau + 1)
        merged_path = find_merged_file(
            save_root, data.n_animals, data.regions, tau_count, args.subset_name
        )
        with open(merged_path, "rb") as fh:
            payload = pickle.load(fh)
        all_speed = payload["speeds"]
        meta = payload.get("meta", {})
        window_sizes = meta.get("window_sizes") or list(
            map(int, data.time_window_range)
        )
        # Build groups (filtered if provided)
        plot_groups = data.groups
        if groups_list:
            wanted = {g.strip() for g in groups_list}
            filtered = {}
            for k, idxs in data.groups.items():
                key_str = f"{k[0]}-{k[1]}"
                if key_str in wanted:
                    filtered[k] = idxs
            if filtered:
                plot_groups = filtered
        plot_qq_between_groups(
            all_speed,
            plot_groups,
            list(map(int, window_sizes)),
            tau=args.tau,
            pool=args.qq_pool,
            split_at=args.split_at,
        )


if __name__ == "__main__":
    main()


# %%
