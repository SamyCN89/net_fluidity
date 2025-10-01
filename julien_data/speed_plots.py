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
#%%
import argparse
from pathlib import Path
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import spearmanr

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
        pool_window_speeds as _pu_pool_window_speeds,
        pool_speeds_per_animal as _pu_per_animal,
        subsample_equal_length as _pu_subsample,
        split_window_indices as _pu_split,
    )
except ModuleNotFoundError:
    try:
        from julien_data.src.plots_utils import (
            pool_window_speeds as _pu_pool_window_speeds,
            pool_speeds_per_animal as _pu_per_animal,
            subsample_equal_length as _pu_subsample,
            split_window_indices as _pu_split,
        )
    except ModuleNotFoundError:
        try:
            from src.plots_utils import (
                pool_window_speeds as _pu_pool_window_speeds,
                pool_speeds_per_animal as _pu_per_animal,
                subsample_equal_length as _pu_subsample,
                split_window_indices as _pu_split,
            )
        except ModuleNotFoundError:
            # Last resort: define no-op fallbacks
            def _pu_pool_window_speeds(win_array, tau=None):
                return np.array([], float)

            def _pu_per_animal(win_array, idxs, tau=None):
                return [np.array([], float) for _ in idxs]

            def _pu_subsample(per_animal, n_per_animal=None, replace=False, random_state=0):
                return np.array([], float)

            def _pu_split(window_sizes, split_at=None):
                return list(range(len(window_sizes)//2)), list(range(len(window_sizes)//2, len(window_sizes))), "fallback"

#%%
def find_merged_file(save_root: Path, n_animals: int, regions: int, tau_count: int, subset_name: str | None):
    """Find the merged speeds PKL in a specific subfolder (subset_name) or auto-detect under save_root."""
    if subset_name:
        subdir = save_root / subset_name
        if not subdir.exists():
            raise FileNotFoundError(f"Subset folder not found: {subdir}")
        cands = sorted(
            subdir.glob(f"speed_windows*_tau{tau_count}_animals_{n_animals}_regions_{regions}.pkl")
        )
    else:
        cands = sorted(
            save_root.rglob(f"speed_windows*_tau{tau_count}_animals_{n_animals}_regions_{regions}.pkl")
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

#%%
def _pool_speeds_per_animal(win_array, idxs, tau: int | None = None):
    return _pu_per_animal(win_array, idxs, tau=tau)


def _subsample_equal_length(per_animal, n_per_animal: int | None = None, replace: bool = False, random_state: int | None = 0):
    return _pu_subsample(per_animal, n_per_animal=n_per_animal, replace=replace, random_state=random_state)


def plot_group_distributions(win_array, groups: dict, window_size: int, tau: int | None = None,
                             equal_animal_weight: bool = False, equal_method: str = "kde",
                             n_per_animal: int | None = None, replace: bool = False,
                             normalize_density: bool = True, random_state: int | None = 0):
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
                    mean_y = (np.nanmean(stacked, axis=0) if normalize_density else np.nansum(stacked, axis=0) / stacked.shape[0])
                    plt.plot(xs, mean_y, lw=2.5, label=label, color=color)
            elif equal_method == "subsample":
                pooled = _subsample_equal_length(per_animal, n_per_animal=n_per_animal, replace=replace, random_state=random_state)
                if pooled.size:
                    sns.kdeplot(pooled, bw_adjust=0.5, label=label, color=color, linewidth=2.5, clip=(0, 2), common_norm=normalize_density)
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
            sns.kdeplot(vals, bw_adjust=0.5, label=label, color=color, linewidth=2.5, clip=(0, 2), common_norm=normalize_density)

    plt.title(f"dFC Speed per group (W={window_size}{', tau='+str(tau) if tau is not None else ', all taus'})")
    plt.xlabel("Speed"); plt.ylabel("Density"); plt.legend(title="Group")
    plt.tight_layout(); sns.despine(trim=True)


def plot_median_vs_window(all_speed, groups: dict, window_sizes: list[int], tau: int | None = None):
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
    plt.title(f"Median dFC Speed vs Window Size{' (all taus)' if tau is None else f' (tau={tau})'}")
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


def plot_group_distributions_two_pools(all_speed, groups: dict, window_sizes: list[int], tau: int | None = None, split_at: int | None = None):
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
        plt.hist(vals, bins=120, density=True, histtype="step", lw=1.7, alpha=0.85, label=label, color=color)
        sns.kdeplot(vals, bw_adjust=0.7, color=color, lw=2)
    plt.title("dFC Speed per group — Pool A")
    plt.xlabel("Speed"); plt.ylabel("Density"); plt.legend(title="Group"); plt.tight_layout(); sns.despine(trim=True)

    # Second pool figure
    plt.figure(figsize=(9, 6))
    palette = sns.color_palette("tab10", n_colors=len(groups))
    for (grp, idxs), color in zip(groups.items(), palette, strict=False):
        vals = pool_vals(second_idx, idxs)
        if vals.size == 0:
            continue
        label = f"{grp[0]}-{grp[1]}"
        plt.hist(vals, bins=120, density=True, histtype="step", lw=1.7, alpha=0.85, label=label, color=color)
        sns.kdeplot(vals, bw_adjust=0.7, color=color, lw=2)
    plt.title("dFC Speed per group — Pool B")
    plt.xlabel("Speed"); plt.ylabel("Density"); plt.legend(title="Group"); plt.tight_layout(); sns.despine(trim=True)
#%%

def plot_qq_between_groups(all_speed, groups: dict, window_sizes: list[int], tau: int | None = None, pool: str = "A", split_at: int | None = None):
    """QQ plots between selected groups for a given pool (A or B) and tau."""
    import seaborn as sns
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
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 5.2 * nrows), squeeze=False)
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
        ax.set_xlabel(group_names[i]); ax.set_ylabel(group_names[j])
        ax.grid(True, ls=":", alpha=0.3)
    # Hide unused axes
    for ax in axes.flat[len(pairs):]:
        ax.axis('off')
    fig.suptitle(f"QQ plots (Pool {pool}, tau={'all' if tau is None else tau})", y=0.98)
    plt.tight_layout()


def run_plot(subset_name: str | None = None,
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
             normalize_density: bool = True):
    """Run plotting end-to-end with parameters (usable from Jupyter)."""
    # Load dataset context
    data = DFCAnalysis()
    if tr is None:
        data.get_metadata()
    else:
        from pathlib import Path as _Path
        preproc = _Path(data.paths["preprocessed"])  # type: ignore[index]
        cands = sorted(preproc.glob(f"metadata_animals_*_tr_{int(tr)}.pkl"))
        if not cands:
            raise FileNotFoundError(f"No metadata file for tr={tr} under {preproc}")
        data.get_metadata(meta_filename=cands[0].name)
    data.get_ts_preprocessed(); data.get_cogdata_preprocessed(); data.get_temporal_parameters()

    save_root = Path(data.paths["speed"])  # contains subfolders
    tau_count = int(data.tau + 1)

    merged_path = find_merged_file(save_root, data.n_animals, data.regions, tau_count, subset_name)
    print("Merged speeds PKL:", merged_path)
    with open(merged_path, "rb") as fh:
        payload = pickle.load(fh)
    all_speed = payload["speeds"]  # list per window
    meta = payload.get("meta", {})
    print("Meta:", meta)

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
    print("Last window pooled:", vals.size, "median:", np.nanmedian(vals) if vals.size else np.nan)

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
            plt.savefig(out.as_posix() + f"_medians_tau{'all' if tau is None else tau}.png", dpi=200)
    plt.show()

    # Two pools (equal halves) per-group distributions
    if split_pools and not no_group:
        plot_group_distributions_two_pools(all_speed, plot_groups, window_sizes, tau=tau, split_at=split_at)
        if savefig:
            out = merged_path.with_suffix("")
            plt.savefig(out.as_posix() + f"_poolsA_groups.png", dpi=200)
            plt.savefig(out.as_posix() + f"_poolsB_groups.png", dpi=200)
        plt.show()

    return {
        "all_speed": all_speed,
        "window_sizes": window_sizes,
        "groups": plot_groups,
        "cog_df": data.cog_data_filtered,
        "merged_path": merged_path,
    }
#%%

def main(argv=None):
    ap = argparse.ArgumentParser(description="Plot merged dFC speed outputs",
                                 allow_abbrev=False,
                                 )
    ap.add_argument("--subset-name", type=str, default=None, help="Subfolder in speed/ (e.g., 'all', 'regions-ACC-THAL', or custom via --subset-name during compute)")
    ap.add_argument("--tau", type=int, default=None, help="Tau index to plot (default: pool all taus)")
    ap.add_argument("--no-group", action="store_true", help="Skip per-group plots; only show overall")
    ap.add_argument("--no-medians", action="store_true", help="Skip median vs window plot")
    ap.add_argument("--savefig", action="store_true", help="Save figures next to merged file")
    ap.add_argument("--split-pools", action="store_true", help="Also plot per-group distributions for two window pools (equal halves or at a specified split)")
    ap.add_argument("--split-at", type=int, default=None, help="Window size threshold to split pools: Pool A <= split_at, Pool B > split_at")
    ap.add_argument("--groups", type=str, default=None, help="Comma-separated group names 'GENOTYPE-TREATMENT' to include (e.g., 'WT-VEH,Dp1Yey-LCTB92')")
    ap.add_argument("--tr", type=int, default=None, help="Select metadata by total_tr for plotting context (e.g., 400 or 500)")
    ap.add_argument("--qq", action="store_true", help="Produce QQ plots between selected groups for the chosen pool and tau")
    ap.add_argument("--qq-pool", type=str, default="A", choices=["A","B"], help="Pool to use for QQ plots (A or B)")
    # Cognition correlations
    ap.add_argument("--cog-scatter", action="store_true", help="Scatter of per-animal dFC speed summary vs cognition")
    ap.add_argument("--cog-var", type=str, default="index_NOR", help="Cognitive variable (column in cog df)")
    ap.add_argument("--weighting", type=str, default="animal", choices=["animal","sample"], help="Per-animal summary weighting")
    ap.add_argument("--equalize-length", action="store_true", help="Equalize sample count per animal before summary")
    ap.add_argument("--reducer", type=str, default="median", help="Reducer: median|mean|qXX")
    ap.add_argument("--corr-vs-window", action="store_true", help="Compute and plot Spearman correlation vs window size")
    ap.add_argument("--equal-animal-weight", action="store_true", help="Equal animal weighting in group distributions (average per-animal KDE or subsample)")
    ap.add_argument("--equal-method", type=str, default="kde", choices=["kde","subsample"], help="Equal-animal method")
    ap.add_argument("--n-per-animal", type=int, default=None, help="Equal length per animal when using subsample method")
    ap.add_argument("--replace", action="store_true", help="Allow replacement during subsample method")
    ap.add_argument("--normalize-density", action="store_true", help="Normalize KDEs to density; unset to sum per-animal curves")
    if argv is None:
        args, _ = ap.parse_known_args(sys.argv[1:])  # ← key fix
    else:
        args = ap.parse_args(argv)
    groups_list = [s.strip() for s in args.groups.split(",")] if args.groups else None
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
    )

    # Optional: Cognition scatter
    if args.cog_scatter:
        all_speed = ctx["all_speed"]; window_sizes = ctx["window_sizes"]; groups = ctx["groups"]; cog_df = ctx["cog_df"]
        # per-animal summary over selected taus and all windows
        try:
            from julien_data.src.plots_utils import per_animal_summary as _pu_per_animal_summary
        except ModuleNotFoundError:
            from src.plots_utils import per_animal_summary as _pu_per_animal_summary  # type: ignore
        x = _pu_per_animal_summary(
            all_speed,
            reducer=args.reducer,
            windows=None,
            taus=None if args.tau is None else [args.tau],
            weighting=args.weighting,
            equalize_length=args.equalize_length,
        )
        y = cog_df[args.cog_var].values
        plt.figure(figsize=(7,5))
        plt.scatter(x, y, c="k", alpha=0.85)
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() >= 3:
            rho, p = spearmanr(x[mask], y[mask])
            plt.title(f"dFC speed ({args.reducer}) vs {args.cog_var} — rho={rho:.2f}, p={p:.2g}")
        else:
            plt.title(f"dFC speed ({args.reducer}) vs {args.cog_var}")
        plt.xlabel(f"{args.reducer} dFC speed per animal"); plt.ylabel(args.cog_var)
        if args.savefig:
            out = ctx["merged_path"].with_suffix("")
            plt.savefig(out.as_posix() + f"_cog_scatter_{args.cog_var}.png", dpi=200)
        else:
            plt.show()

    # Optional: Correlation vs window
    if args.corr_vs_window:
        all_speed = ctx["all_speed"]; window_sizes = ctx["window_sizes"]; groups = ctx["groups"]; cog_df = ctx["cog_df"]
        try:
            from julien_data.src.plots_utils import per_animal_summary as _pu_per_animal_summary
        except ModuleNotFoundError:
            from src.plots_utils import per_animal_summary as _pu_per_animal_summary  # type: ignore
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
                rho, p = (spearmanr(x[idx][mask], y[idx][mask]) if n >= 3 else (np.nan, np.nan))
                rows.append({
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
                })
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
            plt.plot(sub["window_size"], sub["rho"], "-o", color=color, label=lab, zorder=2)
            sig = sub[(sub["p"] < 0.05) & sub["rho"].notna()]
            if not sig.empty:
                plt.scatter(sig["window_size"], sig["rho"], marker="*", s=120, color=color, edgecolor="k", linewidth=0.6, zorder=4)
        plt.axhline(0, color="grey", linestyle="--", linewidth=1, zorder=1)
        plt.xlabel("Window Size"); plt.ylabel(f"Spearman ρ (dFC speed, {args.cog_var})")
        plt.ylim(-1, 1); plt.legend(title="Group"); plt.tight_layout()
        if args.savefig:
            out = ctx["merged_path"].with_suffix("")
            plt.savefig(out.as_posix() + f"_corr_vs_window_{args.cog_var}.png", dpi=200)

    # Optional QQ plots
    if args.qq:
        # Reuse same context
        data = DFCAnalysis()
        if args.tr is None:
            data.get_metadata()
        else:
            preproc = Path(data.paths["preprocessed"])  # type: ignore[index]
            cands = sorted(preproc.glob(f"metadata_animals_*_tr_{int(args.tr)}.pkl"))
            if not cands:
                raise FileNotFoundError(f"No metadata file for tr={args.tr} under {preproc}")
            data.get_metadata(meta_filename=cands[0].name)
        data.get_ts_preprocessed(); data.get_cogdata_preprocessed(); data.get_temporal_parameters()
        save_root = Path(data.paths["speed"])
        tau_count = int(data.tau + 1)
        merged_path = find_merged_file(save_root, data.n_animals, data.regions, tau_count, args.subset_name)
        with open(merged_path, "rb") as fh:
            payload = pickle.load(fh)
        all_speed = payload["speeds"]
        meta = payload.get("meta", {})
        window_sizes = meta.get("window_sizes") or list(map(int, data.time_window_range))
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
        plot_qq_between_groups(all_speed, plot_groups, list(map(int, window_sizes)), tau=args.tau, pool=args.qq_pool, split_at=args.split_at)


if __name__ == "__main__":
    main()


# %%
