#!/usr/bin/env python3
"""
Plot merged dFC speed outputs (speeds across windows) similar to local_speed_plot_v2.py.

Reads the merged PKL produced by 3_dfc_speed_test_v6.py and generates:
- Overall pooled distribution (all animals, all taus pooled for the last window)
- Per-group distributions (genotype × treatment)
- Median speed vs window size per group (optionally for a specific tau)

Usage examples:

  python julien_data/plot_merged_speed.py --subset-name all
  python julien_data/plot_merged_speed.py --subset-name regions-ACC-THAL --tau 0

If --subset-name is omitted, the script will auto-detect a merged PKL under paths['speed'].
"""
#%%
import argparse
from pathlib import Path
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt

# Robust import of DFCAnalysis
try:
    from julien_data.class_dataanalysis_julien import DFCAnalysis
except ModuleNotFoundError:
    from class_dataanalysis_julien import DFCAnalysis

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
    """Pool speeds for one window across animals, optionally selecting a tau index.
    win_array: object array or ndarray-like, len = n_animals, each entry shaped (n_taus, T_w)
    Returns 1D float array of pooled values with NaNs removed.
    """
    pooled = []
    for a in range(len(win_array)):
        arr = win_array[a]
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 2:
            if tau is None:
                pooled.append(arr[~np.isnan(arr)])
            else:
                if tau < 0 or tau >= arr.shape[0]:
                    continue
                pooled.append(arr[tau][~np.isnan(arr[tau])])
    return np.concatenate(pooled) if pooled else np.array([])


def plot_overall_distribution(last_window_vals, window_size: int, ax=None):
    ax = ax or plt.gca()
    ax.hist(last_window_vals, bins=120, density=True, histtype="step", alpha=0.9)
    ax.set_title(f"dFC Speed (W={window_size}, all animals)")
    ax.set_xlabel("Speed")
    ax.set_ylabel("Density")
    return ax

#%%
def plot_group_distributions(win_array, groups: dict, window_size: int, tau: int | None = None):
    import seaborn as sns  # optional; only needed for KDE

    sns.set_theme(style="white", context="talk")
    plt.figure(figsize=(9, 6))
    palette = sns.color_palette("tab10", n_colors=len(groups))
    for (grp, idxs), color in zip(groups.items(), palette, strict=False):
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
        label = f"{grp[0]}-{grp[1]}"
        plt.hist(vals, bins=120, density=True, histtype="step", lw=1.7, alpha=0.85, label=label, color=color)
        sns.kdeplot(vals, bw_adjust=0.7, color=color, lw=2)
    plt.title(f"dFC Speed per group (W={window_size}{', tau='+str(tau) if tau is not None else ', all taus'})")
    plt.xlabel("Speed"); plt.ylabel("Density"); plt.legend(title="Group")
    plt.tight_layout(); sns.despine(trim=True)


def plot_median_vs_window(all_speed, groups: dict, window_sizes: list[int], tau: int | None = None):
    plt.figure(figsize=(10, 6))
    for grp_idx, (grp, idxs) in enumerate(groups.items()):
        medians = []
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
            medians.append(np.nanmedian(vals) if vals.size else np.nan)
        plt.plot(window_sizes, medians, marker=".", label=f"{grp[0]}-{grp[1]}")
    plt.title(f"Median dFC Speed vs Window Size{' (all taus)' if tau is None else f' (tau={tau})'}")
    plt.xlabel("Window size")
    plt.ylabel("Median speed")
    plt.legend(title="Group")
    plt.tight_layout()
#%%

def run_plot(subset_name: str | None = None, tau: int | None = None, no_group: bool = False, no_medians: bool = False, savefig: bool = False, groups: list[str] | None = None):
    """Run plotting end-to-end with parameters (usable from Jupyter)."""
    # Load dataset context
    data = DFCAnalysis()
    data.get_metadata(); data.get_ts_preprocessed(); data.get_cogdata_preprocessed(); data.get_temporal_parameters()

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
        plot_group_distributions(last_window, plot_groups, window_sizes[-1], tau=tau)
        if savefig:
            out = merged_path.with_suffix("")
            plt.savefig(out.as_posix() + f"_groups_W{window_sizes[-1]}.png", dpi=200)

    # Median vs window size per group
    if not no_medians:
        plot_median_vs_window(all_speed, plot_groups, window_sizes, tau=tau)
        if savefig:
            out = merged_path.with_suffix("")
            plt.savefig(out.as_posix() + f"_medians_tau{'all' if tau is None else tau}.png", dpi=200)
    plt.show()
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
    ap.add_argument("--groups", type=str, default=None, help="Comma-separated group names 'GENOTYPE-TREATMENT' to include (e.g., 'WT-VEH,Dp1Yey-LCTB92')")
    if argv is None:
        args, _ = ap.parse_known_args(sys.argv[1:])  # ← key fix
    else:
        args = ap.parse_args(argv)
    groups_list = [s.strip() for s in args.groups.split(",")] if args.groups else None
    run_plot(
        subset_name=args.subset_name,
        tau=args.tau,
        no_group=args.no_group,
        no_medians=args.no_medians,
        savefig=args.savefig,
        groups=groups_list,
    )


if __name__ == "__main__":
    main()


# %%
