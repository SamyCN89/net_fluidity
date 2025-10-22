
#%%
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union

from scipy.stats import theilslopes, spearmanr, kruskal, mannwhitneyu  
from scipy import stats
import statsmodels.api as sm

# from matplotlib import scale
# from networkx import density
from class_dataanalysis_julien import DFCAnalysis
from itertools import combinations
import string


data = DFCAnalysis()
data.load_preprocessed_data()# %%
data.get_temporal_parameters()

# Context class for data loading and processing
@dataclass
class SpeedContext:
    all_speed: list                # len = n_windows; each is (n_animals, n_taus, T_w)
    window_sizes: np.ndarray       # same as time_window_range
    groups: dict                   # {(genotype,treatment): [animal indices]}
    cog_df: pd.DataFrame           # data.cog_data_filtered (aligned to animals)
    region_label: str              # e.g., "ACC" or whatever label
    n_animals: int                 # convenience
    n_taus: int                    # convenience




#%%


# Create a context instance
def build_speed_context(data, ind_reg, prefix="speed") -> SpeedContext:
    save_path = data.paths['speed']
    time_window_range = data.time_window_range
    tau_range = np.arange(0, data.tau + 1)
    n_animals = data.n_animals

    pkl = save_path / f"{prefix}_region{ind_reg}_windows{len(time_window_range)}_tau{np.size(tau_range)}_animals_{n_animals}.pkl"
    with open(pkl, 'rb') as f:
        all_speed = pickle.load(f)

    # sanity: infer n_taus from first window
    n_taus = all_speed[0].shape[1]

    return SpeedContext(
        all_speed=all_speed,
        window_sizes=np.array(time_window_range),
        groups=data.groups,
        cog_df=data.cog_data_filtered.reset_index(drop=True),
        region_label=data.region_labels_preprocessed[ind_reg],
        n_animals=n_animals,
        n_taus=n_taus
    )
    
# # --- TEMP: test one region end-to-end, no plotting changes yet ---
# test_region = 16  # pick one you like
# ctx = build_speed_context(data, test_region)

# print(f"[Context] Region: {ctx.region_label}")
# print(f"[Context] n_windows={len(ctx.all_speed)}, window_sizes[:5]={ctx.window_sizes[:5]}")
# for i, arr in enumerate(ctx.all_speed[:3]):  # show first 3 windows
#     print(f"  window[{i}] shape = {arr.shape}")  # (n_animals, n_taus, T_w)
# print(f"[Context] groups: { {k: len(v) for k,v in ctx.groups.items()} }")
# print(f"[Context] n_animals={ctx.n_animals}, n_taus={ctx.n_taus}")
#%%
# --- Pooling utilities ---


def pool_speeds(
    all_speed: List[np.ndarray],
    animals: List[int],
    windows: Optional[Union[List[int], np.ndarray]] = None,
    taus: Optional[Union[List[int], np.ndarray]] = None,
    weighting: str = "sample",   # "sample" (current behavior) or "animal"
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Gather speed values for given animals, window indices, and tau indices.
    - weighting="sample": returns a single 1D array concatenating all samples (current behavior).
    - weighting="animal": returns a list of per-animal arrays (for equal-animal-weight uses).
    """
    if windows is None:
        windows = range(len(all_speed))
    if taus is None:
        taus = range(all_speed[0].shape[1])

    per_animal = []
    for a in animals:
        chunks = []
        for w in windows:
            arr3d = all_speed[w]  # (n_animals, n_taus, T_w)
            for t in taus:
                arr = np.asarray(arr3d[a, t, :], float)
                arr = arr[~np.isnan(arr)]
                if arr.size:
                    chunks.append(arr)
        per_animal.append(np.concatenate(chunks) if chunks else np.array([], float))

    if weighting == "sample":
        non_empty = [x for x in per_animal if x.size]
        return np.concatenate(non_empty) if non_empty else np.array([], float)
    elif weighting == "animal":
        return per_animal
    else:
        raise ValueError("weighting must be 'sample' or 'animal'")


def subsample_equal_length(
    per_animal_arrays: list[np.ndarray],
    n_per_animal: int | None = None,
    replace: bool = False,
    random_state: int | None = 0,
) -> np.ndarray:
    """
    Take a list of per-animal arrays and return one pooled array where each animal
    contributes exactly n_per_animal samples (or the minimum available if None).
    """
    rng = np.random.default_rng(random_state)
    non_empty = [arr for arr in per_animal_arrays if arr.size > 0]
    if not non_empty:
        return np.array([], float)

    if n_per_animal is None:
        n_per_animal = min(arr.size for arr in non_empty)

    pooled = []
    for arr in non_empty:
        size = arr.size
        # if array is shorter than requested length and replace=False, fall back to replace=True for that animal
        use_replace = replace or (size < n_per_animal)
        idx = rng.choice(size, size=n_per_animal, replace=use_replace)
        pooled.append(arr[idx])
    return np.concatenate(pooled) if pooled else np.array([], float)


def per_animal_summary(
    all_speed: list[np.ndarray],
    reducer: str = "median",
    windows=None,
    taus=None,
    weighting: str = "sample",   # "sample" or "animal"
    equalize_length: bool = False,
    replace: bool = False,
    random_state: int | None = 0,
) -> np.ndarray:
    """
    Compute a summary per animal for selected windows/taus.

    weighting="sample": pool all samples (default).
    weighting="animal": treat each animal equally.
    equalize_length=True: force each animal to contribute same # samples
    before computing reducer.
    """
    n_animals = all_speed[0].shape[0]
    out = np.full(n_animals, np.nan)

    if windows is None:
        windows = range(len(all_speed))
    if taus is None:
        taus = range(all_speed[0].shape[1])

    rng = np.random.default_rng(random_state)

    # Precompute min_len if needed
    min_len = None
    if equalize_length:
        lengths = []
        for a in range(n_animals):
            pooled_a = pool_speeds(all_speed, [a], windows, taus, weighting=weighting)
            # Flatten list if weighting="animal"
            if isinstance(pooled_a, list):
                pooled_a = np.concatenate(pooled_a) if pooled_a else np.array([])
            lengths.append(len(pooled_a))
        min_len = min(l for l in lengths if l > 0)

    for a in range(n_animals):
        pooled = pool_speeds(all_speed, [a], windows, taus, weighting=weighting)
        if isinstance(pooled, list):
            pooled = np.concatenate(pooled) if pooled else np.array([])

        if pooled.size > 0:
            if equalize_length and min_len is not None and pooled.size >= min_len:
                pooled = rng.choice(pooled, size=min_len, replace=replace)

            if reducer == "median":
                out[a] = np.median(pooled)
            elif reducer == "mean":
                out[a] = np.mean(pooled)
            elif reducer.startswith("q"):  # e.g., "q95"
                q = float(reducer[1:]) / 100.0
                out[a] = np.quantile(pooled, q)
            else:
                raise ValueError("Unknown reducer.")

    return out


#%%
# --- Density plot with equal-weight options ---

def plot_group_distributions(
    ctx: SpeedContext,
    windows=None,
    taus=None,
    equal_animal_weight: bool = True,
    equal_method: str = "kde",        # "kde" or "subsample"
    n_per_animal: int | None = None,  # used in "subsample"
    replace: bool = False,
    random_state: int | None = 0,
    scale: str = "linear",
    normalize_density: bool = True,   # True=area=1, False=absolute counts
    save_fig: bool = False,
):
    sns.set_theme(style='white', palette='deep', context='talk')
    palette = sns.color_palette('tab10', n_colors=len(ctx.groups))

    for (i, (group, animal_idx)) in enumerate(ctx.groups.items()):
        label = f"{group[0]}-{group[1]}".lower()
        color = palette[i]

        if equal_animal_weight:
            per_animal = pool_speeds(ctx.all_speed, animal_idx, windows, taus, weighting="animal")

            if equal_method == "kde":
                xs = np.linspace(0, 2, 500)

                # Auto equalize sample size if not normalizing density
                if not normalize_density:
                    min_len = min(arr.size for arr in per_animal if arr.size > 0)
                    rng = np.random.default_rng(random_state)
                    per_animal = [
                        arr[rng.choice(arr.size, size=min_len, replace=replace)]
                        if arr.size > 0 else arr
                        for arr in per_animal
                    ]

                curves = []
                for arr in per_animal:
                    if arr.size < 5:
                        continue
                    kde = sm.nonparametric.KDEUnivariate(arr)
                    kde.fit()
                    y = np.interp(xs, kde.support, kde.density, left=0, right=0)
                    if not normalize_density:
                        y *= arr.size  # scale by count
                    curves.append(y)

                if curves:
                    stacked = np.vstack(curves)  # (n_animals_kept, len(xs))
                    if normalize_density:
                        mean_y = np.nanmean(stacked, axis=0)
                    else:
                        mean_y = np.nansum(stacked, axis=0) / stacked.shape[0]
                    plt.plot(xs, mean_y, lw=2.5, label=label, color=color)

                    pooled = np.concatenate([a for a in per_animal if a.size]) if per_animal else np.array([])
                    if pooled.size:
                        for v, ls, a in [(np.median(pooled), '-', .9),
                                         (np.quantile(pooled, .05), '--', .6),
                                         (np.quantile(pooled, .95), '--', .6)]:
                            plt.axvline(v, color=color, linestyle=ls, linewidth=1, alpha=a)

            elif equal_method == "subsample":
                pooled = subsample_equal_length(per_animal, n_per_animal=n_per_animal, replace=replace, random_state=random_state)
                if pooled.size:
                    sns.kdeplot(pooled, bw_adjust=.5, label=label, color=color, linewidth=2.5, clip=(0, 2),
                                common_norm=normalize_density)
                    for v, ls, a in [(np.median(pooled), '-', .9),
                                     (np.quantile(pooled, .05), '--', .6),
                                     (np.quantile(pooled, .95), '--', .6)]:
                        plt.axvline(v, color=color, linestyle=ls, linewidth=1, alpha=a)
            else:
                raise ValueError("equal_method must be 'kde' or 'subsample'.")

        else:
            pooled = pool_speeds(ctx.all_speed, animal_idx, windows, taus, weighting="sample")
            if pooled.size:
                sns.kdeplot(pooled, bw_adjust=.5, label=label, color=color, linewidth=2.5, clip=(0, 2),
                            common_norm=normalize_density)
                for v, ls, a in [(np.median(pooled), '-', .9),
                                 (np.quantile(pooled, .05), '--', .6),
                                 (np.quantile(pooled, .95), '--', .6)]:
                    plt.axvline(v, color=color, linestyle=ls, linewidth=1, alpha=a)

    plt.xlabel("dFC Speed")
    plt.ylabel("Density" if normalize_density else "Count")
    plt.yscale(scale)
    plt.title(f"dFC Speeds by Group (region={ctx.region_label})\n"
              f"{'Equal animal weight: ' + equal_method if equal_animal_weight else 'Sample-weighted'}"
              + ("" if normalize_density else " (absolute counts)"))
    plt.legend(title='group', frameon=True)
    plt.tight_layout()
    sns.despine(trim=True)
    if save_fig:
        suffix = f"{scale}_{'equalA-'+equal_method if equal_animal_weight else 'sample'}_{'norm' if normalize_density else 'abs'}"
        out = data.paths['f_speed'] / f'{ctx.region_label}_dFC_speed_dist_{suffix}.png'
        plt.savefig(out, dpi=200)

# --- Correlation plots using the new summary ---

def plot_dfc_speed_vs_cog_scores(
    ctx: SpeedContext,
    cog_var="index_NOR",
    reducer="median",
    windows=None,
    taus=None,
    weighting="sample",
    equalize_length=False,
    save_fig=False,
    random_state=0,
):
    per_animal_vals = per_animal_summary(
        ctx.all_speed, reducer=reducer, windows=windows, taus=taus,
        weighting=weighting, equalize_length=equalize_length, random_state=random_state
    )
    cog_scores = ctx.cog_df[cog_var].values

    plt.figure(figsize=(7,5))
    plt.scatter(per_animal_vals, cog_scores, c='k', alpha=0.85)
    plt.xlabel(f'{reducer} dFC Speed per animal')
    plt.ylabel(cog_var)
    plt.title(f'Region={ctx.region_label} — dFC speed vs {cog_var}\n(weighting={weighting}, equal_len={equalize_length})')

    mask = ~np.isnan(per_animal_vals) & ~np.isnan(cog_scores)
    rho, pval = spearmanr(per_animal_vals[mask], cog_scores[mask])
    plt.text(0.05, 0.95, f"Spearman r={rho:.2f}, p={pval:.3g}",
             transform=plt.gca().transAxes, va='top', ha='left', fontsize=12)

    plt.tight_layout()
    if save_fig:
        out = data.paths['f_speed'] / f'{ctx.region_label}_dfc_vs_{cog_var}_{weighting}_eq{equalize_length}.png'
        plt.savefig(out, dpi=200)

def plot_dfc_speed_vs_cog_scores_per_group(
    ctx: SpeedContext,
    cog_var="index_NOR",
    reducer="median",
    windows=None,
    taus=None,
    weighting="sample",
    equalize_length=False,
    save_fig=False,
    fig_path=None,
    random_state=0
):
    per_animal_vals = per_animal_summary(
        ctx.all_speed, reducer=reducer, windows=windows, taus=taus,
        weighting=weighting, equalize_length=equalize_length, random_state=random_state
    )

    cog_scores = ctx.cog_df[cog_var].values
    group_items = list(ctx.groups.items())
    palette = sns.color_palette("tab10", len(group_items))

    fig, ax = plt.subplots(figsize=(7, 5))

    for (grp_name, animal_indices), color in zip(group_items, palette):
        idx = np.array(animal_indices)
        mask = ~np.isnan(per_animal_vals[idx]) & ~np.isnan(cog_scores[idx])
        if not np.any(mask):
            continue

        x = per_animal_vals[idx][mask]
        y = cog_scores[idx][mask]
        ax.scatter(x, y, label="-".join(grp_name), alpha=0.85, color=color)

        rho, pval = spearmanr(x, y)
        ax.text(0.04, 0.96 - 0.08 * group_items.index((grp_name, animal_indices)),
                f"{'-'.join(grp_name)}: ρ={rho:.2f}, p={pval:.3g}",
                color=color, transform=ax.transAxes)

        slope, intercept, _, _ = stats.theilslopes(y, x, 0.95)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(xx, intercept + slope * xx, color=color, linestyle="--", alpha=0.8)

    ax.set_xlabel(f"{reducer} dFC Speed")
    ax.set_ylabel(cog_var)
    ax.set_title(f"{ctx.region_label} — dFC speed vs {cog_var}\n(weighting={weighting}, equal_len={equalize_length})")
    ax.legend()
    plt.tight_layout()

    if save_fig:
        if fig_path is None:
            fig_path = data.paths['f_speed'] / f"dfc_vs_{cog_var}_{ctx.region_label}_{weighting}_eq{equalize_length}.png"
        plt.savefig(fig_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
        

#%%

def corr_speed_cog_vs_window(
    ctx,
    cog_var="index_NOR",
    reducer="median",
    by_group=True,
    weighting="animal",
    equalize_length=True,
    taus=None,
    random_state=0,
):
    """
    Returns a DataFrame with Spearman rho/p per window size.
    If by_group=True, one row per (window, group). Else, one row per window (ALL).
    """
    rows = []
    win_sizes = ctx.window_sizes

    for w_idx, wsize in enumerate(win_sizes):
        x = per_animal_summary(
            ctx.all_speed,
            reducer=reducer,
            windows=[w_idx],
            taus=taus,
            weighting=weighting,
            equalize_length=equalize_length,
            random_state=random_state,
        )
        y = ctx.cog_df[cog_var].values

        if by_group:
            for grp, idxs in ctx.groups.items():
                idx = np.array(idxs)
                mask = ~np.isnan(x[idx]) & ~np.isnan(y[idx])
                n = int(mask.sum())
                if n >= 3:
                    rho, p = spearmanr(x[idx][mask], y[idx][mask])
                else:
                    rho, p = np.nan, np.nan
                rows.append({
                    "window_idx": w_idx,
                    "window_size": wsize,
                    "group": "-".join(grp),
                    "rho": rho, "p": p, "n": n,
                    "weighting": weighting,
                    "equalize_length": equalize_length,
                    "reducer": reducer,
                    "cog_var": cog_var,
                })
        else:
            mask = ~np.isnan(x) & ~np.isnan(y)
            n = int(mask.sum())
            if n >= 3:
                rho, p = spearmanr(x[mask], y[mask])
            else:
                rho, p = np.nan, np.nan
            rows.append({
                "window_idx": w_idx,
                "window_size": wsize,
                "group": "ALL",
                "rho": rho, "p": p, "n": n,
                "weighting": weighting,
                "equalize_length": equalize_length,
                "reducer": reducer,
                "cog_var": cog_var,
            })

    return pd.DataFrame(rows)


def plot_corr_vs_window(
    df,
    ctx,
    alpha=0.05,
    by_group=True,
    save_fig=False,
    fig_path=None,
):
    """
    Line plot of rho vs window size with stars for p<alpha.
    Expects df from corr_speed_cog_vs_window().
    """
    plt.figure(figsize=(12, 6))
    if by_group:
        # keep group order consistent with ctx.groups
        order = ["-".join(g) for g in ctx.groups.keys()]
        palette = sns.color_palette("tab10", n_colors=len(order))
        for color, lab in zip(palette, order):
            sub = df[df["group"] == lab].sort_values("window_size")
            if sub.empty:
                continue
            plt.plot(sub["window_size"], sub["rho"], "-o", color=color, label=lab, zorder=2)
            sig = sub[(sub["p"] < alpha) & sub["rho"].notna()]
            if not sig.empty:
                plt.scatter(sig["window_size"], sig["rho"], marker="*", s=120,
                            color=color, edgecolor="k", linewidth=0.6, zorder=4)
    else:
        sub = df.sort_values("window_size")
        plt.plot(sub["window_size"], sub["rho"], "-o", color="k", label="ALL", zorder=2)
        sig = sub[(sub["p"] < alpha) & sub["rho"].notna()]
        if not sig.empty:
            plt.scatter(sig["window_size"], sig["rho"], marker="*", s=120,
                        color="k", edgecolor="k", linewidth=0.6, zorder=4)

    plt.axhline(0, color="grey", linestyle="--", linewidth=1, zorder=1)
    plt.xlabel("Window Size")
    cog_var = df["cog_var"].iloc[0]
    plt.ylabel(f"Spearman ρ (dFC speed, {cog_var})")
    plt.ylim(-1, 1)
    plt.xlim(df["window_size"].min() - 1, df["window_size"].max() + 1)

    w = df["weighting"].iloc[0]
    eq = df["equalize_length"].iloc[0]
    reducer = df["reducer"].iloc[0]
    title_base = f"Correlation vs Window Size — {ctx.region_label}\n(weighting={w}, equal_len={eq}, reducer={reducer})"
    plt.title(title_base)
    if by_group:
        plt.legend(title="Group")
    plt.tight_layout()

    if save_fig:
        if fig_path is None:
            fig_path = data.paths["f_speed"] / f"{ctx.region_label}_corr_vs_window_{w}_eq{eq}.png"
        plt.savefig(fig_path, dpi=200)



def plot_group_distributions(
    ctx: SpeedContext,
    windows=None,
    taus=None,
    equal_animal_weight: bool = True,
    equal_method: str = "kde",        # "kde" or "subsample"
    n_per_animal: int | None = None,  # used in "subsample"
    replace: bool = False,
    random_state: int | None = 0,
    scale: str = "linear",
    normalize_density: bool = True,   # True=area=1, False=absolute counts
    save_fig: bool = False,
    ax: plt.Axes | None = None,
    add_title: bool = True,
    add_legend: bool = True,
    tight: bool = True,
):
    sns.set_theme(style='white', palette='deep', context='talk')
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    palette = sns.color_palette('tab10', n_colors=len(ctx.groups))

    for (i, (group, animal_idx)) in enumerate(ctx.groups.items()):
        label = f"{group[0]}-{group[1]}".lower()
        color = palette[i]

        if equal_animal_weight:
            per_animal = pool_speeds(ctx.all_speed, animal_idx, windows, taus, weighting="animal")

            if equal_method == "kde":
                xs = np.linspace(0, 2, 500)

                # Auto equalize sample size if not normalizing density
                if not normalize_density:
                    min_len = min(arr.size for arr in per_animal if arr.size > 0)
                    rng = np.random.default_rng(random_state)
                    per_animal = [
                        arr[rng.choice(arr.size, size=min_len, replace=replace)]
                        if arr.size > 0 else arr
                        for arr in per_animal
                    ]

                curves = []
                for arr in per_animal:
                    if arr.size < 5:
                        continue
                    kde = sm.nonparametric.KDEUnivariate(arr)
                    kde.fit()
                    y = np.interp(xs, kde.support, kde.density, left=0, right=0)
                    if not normalize_density:
                        y *= arr.size  # scale by count
                    curves.append(y)

                if curves:
                    stacked = np.vstack(curves)  # (n_animals_kept, len(xs))
                    if normalize_density:
                        mean_y = np.nanmean(stacked, axis=0)
                    else:
                        mean_y = np.nansum(stacked, axis=0) / stacked.shape[0]
                    ax.plot(xs, mean_y, lw=2.5, label=label, color=color)

                    pooled = np.concatenate([a for a in per_animal if a.size]) if per_animal else np.array([])
                    if pooled.size:
                        for v, ls, a_ in [(np.median(pooled), '-', .9),
                                          (np.quantile(pooled, .05), '--', .6),
                                          (np.quantile(pooled, .95), '--', .6)]:
                            ax.axvline(v, color=color, linestyle=ls, linewidth=1, alpha=a_)

            elif equal_method == "subsample":
                pooled = subsample_equal_length(per_animal, n_per_animal=n_per_animal, replace=replace, random_state=random_state)
                if pooled.size:
                    sns.kdeplot(pooled, bw_adjust=.5, label=label, color=color, linewidth=2.5, clip=(0, 2),
                                common_norm=normalize_density, ax=ax)
                    for v, ls, a_ in [(np.median(pooled), '-', .9),
                                      (np.quantile(pooled, .05), '--', .6),
                                      (np.quantile(pooled, .95), '--', .6)]:
                        ax.axvline(v, color=color, linestyle=ls, linewidth=1, alpha=a_)
            else:
                raise ValueError("equal_method must be 'kde' or 'subsample'.")

        else:
            pooled = pool_speeds(ctx.all_speed, animal_idx, windows, taus, weighting="sample")
            if pooled.size:
                sns.kdeplot(pooled, bw_adjust=.5, label=label, color=color, linewidth=2.5, clip=(0, 2),
                            common_norm=normalize_density, ax=ax)
                for v, ls, a_ in [(np.median(pooled), '-', .9),
                                  (np.quantile(pooled, .05), '--', .6),
                                  (np.quantile(pooled, .95), '--', .6)]:
                    ax.axvline(v, color=color, linestyle=ls, linewidth=1, alpha=a_)

    ax.set_xlabel("dFC Speed")
    ax.set_ylabel("Density" if normalize_density else "Count")
    ax.set_yscale(scale)
    if add_title:
        ax.set_title(f"{'Equal animal weight: ' + equal_method if equal_animal_weight else 'Sample-weighted'}"
                     + ("" if normalize_density else " (absolute counts)"))
    if add_legend:
        ax.legend(title='group', frameon=True)
    sns.despine(ax=ax, trim=True)
    if tight:
        fig.tight_layout()

    if save_fig:
        suffix = f"{scale}_{'equalA-'+equal_method if equal_animal_weight else 'sample'}_{'norm' if normalize_density else 'abs'}"
        out = data.paths['f_speed'] / f'{ctx.region_label}_dFC_speed_dist_{suffix}.png'
        fig.savefig(out, dpi=200)


def get_window_pools(ctx: SpeedContext, mode: str = "half", split_at: int | None = None, quantile: float = 0.5):
    """
    Returns dict with two entries: {'short': idxs, 'long': idxs}
    - mode='half': first half vs second half of ctx.window_sizes
    - mode='threshold': split at absolute window size 'split_at'
    - mode='quantile': split at given quantile of window sizes (e.g., 0.5)
    """
    ws = np.asarray(ctx.window_sizes)
    if mode == "half":
        k = len(ws) // 2
        return {"short": np.arange(0, k), "long": np.arange(k, len(ws))}
    elif mode == "threshold":
        if split_at is None:
            raise ValueError("split_at must be set for mode='threshold'")
        return {"short": np.where(ws < split_at)[0], "long": np.where(ws >= split_at)[0]}
    elif mode == "quantile":
        t = np.quantile(ws, quantile)
        return {"short": np.where(ws <= t)[0], "long": np.where(ws > t)[0]}
    else:
        raise ValueError("mode must be 'half', 'threshold', or 'quantile'")

def plot_short_long_distributions(
    ctx: SpeedContext,
    mode: str = "half",
    split_at: int | None = None,
    quantile: float = 0.5,
    equal_animal_weight: bool = True,
    equal_method: str = "kde",
    n_per_animal: int | None = None,
    replace: bool = False,
    random_state: int | None = 0,
    scale: str = "linear",
    normalize_density: bool = True,
    save_fig: bool = False,
):
    pools = get_window_pools(ctx, mode=mode, split_at=split_at, quantile=quantile)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Short
    plot_group_distributions(
        ctx,
        windows=pools["short"],
        equal_animal_weight=equal_animal_weight,
        equal_method=equal_method,
        n_per_animal=n_per_animal,
        replace=replace,
        random_state=random_state,
        scale=scale,
        normalize_density=normalize_density,
        save_fig=False,
        ax=axes[0],
        add_title=False,
        add_legend=True,
        tight=False,
    )
    axes[0].set_title("Short windows")

    # Long
    plot_group_distributions(
        ctx,
        windows=pools["long"],
        equal_animal_weight=equal_animal_weight,
        equal_method=equal_method,
        n_per_animal=n_per_animal,
        replace=replace,
        random_state=random_state,
        scale=scale,
        normalize_density=normalize_density,
        save_fig=False,
        ax=axes[1],
        add_title=False,
        add_legend=False,  # keep legend only on the left panel
        tight=False,
    )
    axes[1].set_title("Long windows")

    fig.suptitle(f"Distribution of dFC Speeds by Group — {ctx.region_label}\n"
                 f"{'Equal animal weight: ' + equal_method if equal_animal_weight else 'Sample-weighted'}"
                 + ("" if normalize_density else " (absolute counts)"), y=1.02)
    fig.tight_layout()

    if save_fig:
        suffix = f"{scale}_{'equalA-'+equal_method if equal_animal_weight else 'sample'}_{'norm' if normalize_density else 'abs'}"
        out = data.paths['f_speed'] / f"{ctx.region_label}_short_long_{mode}_{suffix}.png"
        fig.savefig(out, dpi=200)


#%%

def build_groups_for_tests(
    ctx: SpeedContext,
    windows=None,
    taus=None,
    approach: str = "per-animal",     # "per-animal" (recommended) or "per-sample"
    weighting: str = "animal",        # only used when approach="per-sample" (for direct pooling)
    equalize_length: bool = True,     # recommended for fairness
    reducer: str = "median",          # only used when approach="per-animal"
    n_per_animal: int | None = None,
    replace: bool = False,
    random_state: int | None = 0,
) -> dict:
    """
    Returns {group_tuple: 1D np.array} ready for nonparametric tests.
    - approach="per-animal": builds an array of per-animal summaries (e.g., medians) per group.
      This treats animals as the independent unit (recommended).
    - approach="per-sample": pools raw samples per group (risk of pseudo-replication).
    """
    if approach == "per-animal":
        vals = per_animal_summary(
            ctx.all_speed, reducer=reducer, windows=windows, taus=taus,
            weighting="sample",              # single-animal pooling; weighting doesn't matter
            equalize_length=equalize_length, # make each animal contribute same #samples
            random_state=random_state
        )
        out = {}
        for g, idxs in ctx.groups.items():
            idx = np.array(idxs)
            grp_vals = vals[idx]
            grp_vals = grp_vals[~np.isnan(grp_vals)]
            out[g] = grp_vals
        return out

    elif approach == "per-sample":
        # Use the same pooling rules you used elsewhere
        return build_group_values(
            ctx, windows=windows, taus=taus,
            weighting=weighting, equalize_length=(equalize_length if weighting=="animal" else False),
            n_per_animal=n_per_animal, replace=replace, random_state=random_state
        )

    else:
        raise ValueError("approach must be 'per-animal' or 'per-sample'")

from scipy import stats

def kruskal_speed_groups(group_arrays: dict) -> dict:
    """
    group_arrays: {group_tuple: np.array}
    Returns a small dict with H, p, Ns, k, df.
    """
    labels = list(group_arrays.keys())
    arrays = [np.asarray(group_arrays[g], float) for g in labels]
    arrays = [a[~np.isnan(a)] for a in arrays]
    nonempty = [(g, a) for g, a in zip(labels, arrays) if a.size > 0]

    if len(nonempty) < 2:
        return {"ok": False, "reason": "Need at least two groups with data."}

    labs, arrs = zip(*nonempty)
    H, p = stats.kruskal(*arrs)
    Ns = {g: int(a.size) for g, a in nonempty}
    return {
        "ok": True,
        "H": float(H),
        "p": float(p),
        "k": len(arrs),
        "df": len(arrs) - 1,
        "Ns": Ns,
        "groups": list(labs),
    }
    
import numpy as np
import pandas as pd
from itertools import combinations

def _rank_biserial_from_u(U, n1, n2, med1, med2):
    # rank-biserial correlation; sign via median difference
    r = 1.0 - 2.0 * (U / (n1 * n2))
    sign = np.sign(med1 - med2)
    return float(sign * abs(r))

def _p_adjust_bonf(pvals):
    p = np.asarray(pvals, float)
    return np.minimum(p * len(p), 1.0)

def _p_adjust_fdr_bh(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    ranked = np.arange(1, len(p) + 1)
    q = p[order] * len(p) / ranked
    q = np.minimum.accumulate(q[::-1])[::-1]  # monotone
    p_adj = np.empty_like(p)
    p_adj[order] = np.minimum(q, 1.0)
    return p_adj

def pairwise_mwu_speed_groups(group_arrays: dict, correction="bonferroni") -> pd.DataFrame:
    rows = []
    keys = list(group_arrays.keys())
    for g1, g2 in combinations(keys, 2):
        a = np.asarray(group_arrays[g1], float)
        b = np.asarray(group_arrays[g2], float)
        a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
        if (a.size < 1) or (b.size < 1):
            continue
        U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rbs = _rank_biserial_from_u(U, len(a), len(b), np.median(a), np.median(b))
        rows.append({
            "group1": "-".join(g1), "group2": "-".join(g2),
            "n1": len(a), "n2": len(b),
            "U": float(U), "p": float(p),
            "rank_biserial": rbs,
            "median1": float(np.median(a)), "median2": float(np.median(b))
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if correction.lower() in ("bonferroni", "bonf"):
        df["p_adj"] = _p_adjust_bonf(df["p"].values)
        df["p_adj_method"] = "bonferroni"
    elif correction.lower() in ("fdr", "bh", "fdr_bh"):
        df["p_adj"] = _p_adjust_fdr_bh(df["p"].values)
        df["p_adj_method"] = "fdr_bh"
    else:
        df["p_adj"] = df["p"].values
        df["p_adj_method"] = "none"

    return df.sort_values("p_adj").reset_index(drop=True)


#%%

        
        
def build_group_values(
    ctx: SpeedContext,
    windows=None,
    taus=None,
    weighting: str = "animal",     # "animal" (fair) or "sample" (old pooled)
    equalize_length: bool = True,  # only relevant for "animal"
    n_per_animal: int | None = None,
    replace: bool = False,
    random_state: int | None = 0,
) -> dict:
    """
    Returns {group_tuple: 1D np.array of pooled values} using the same fairness rules
    you use in other plots.
    """
    out = {}
    for group, animal_idxs in ctx.groups.items():
        if weighting == "animal":
            per_animal = pool_speeds(ctx.all_speed, animal_idxs, windows, taus, weighting="animal")
            per_animal = [a for a in per_animal if a.size > 0]
            if not per_animal:
                out[group] = np.array([])
                continue
            if equalize_length:
                pooled = subsample_equal_length(
                    per_animal, n_per_animal=n_per_animal,
                    replace=replace, random_state=random_state
                )
            else:
                pooled = np.concatenate(per_animal)
            out[group] = pooled
        elif weighting == "sample":
            out[group] = pool_speeds(ctx.all_speed, animal_idxs, windows, taus, weighting="sample")
        else:
            raise ValueError("weighting must be 'animal' or 'sample'")
    return out

from itertools import combinations
import string

def plot_qq_groups(
    ctx: SpeedContext,
    windows=None,
    taus=None,
    weighting: str = "animal",
    equalize_length: bool = True,
    n_per_animal: int | None = None,
    replace: bool = False,
    random_state: int | None = 0,
    n_points: int = 1000,
    n_cols: int = 3,
    save_fig: bool = False,
    fig_path=None,
):
    """
    Pairwise Q–Q plots across groups using the same pooling/equalization rules
    as the rest of the workflow.
    """
    # build pooled arrays per group
    gvals = build_group_values(
        ctx, windows=windows, taus=taus,
        weighting=weighting, equalize_length=equalize_length,
        n_per_animal=n_per_animal, replace=replace, random_state=random_state
    )
    groups_list = list(gvals.keys())
    valid = {g: v for g, v in gvals.items() if v.size > 0}
    if len(valid) < 2:
        print("Not enough groups with data for Q–Q.")
        return

    # global axis limits
    all_vals = np.concatenate(list(valid.values()))
    gmin, gmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))

    # grid layout
    pairs = list(combinations(groups_list, 2))
    n_pairs = len(pairs)
    n_rows = int(np.ceil(n_pairs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.6*n_cols, 5.6*n_rows), squeeze=False)
    q = np.linspace(0, 1, n_points)

    handles = None
    for idx, ((g1, g2), ax) in enumerate(zip(pairs, axes.flat)):
        a1, a2 = gvals[g1], gvals[g2]
        if a1.size == 0 or a2.size == 0:
            ax.axis('off'); continue

        q1 = np.quantile(a1, q)
        q2 = np.quantile(a2, q)

        above = q2 > q1
        below = q2 < q1
        h1 = ax.fill_between(q1, q1, q2, where=above, color='firebrick', alpha=0.35, label='Group2 > Group1')
        h2 = ax.fill_between(q1, q1, q2, where=below, color='dodgerblue', alpha=0.35, label='Group2 < Group1')

        if handles is None:
            handles = [h1, h2]

        ax.plot(q1, q2, color='k', lw=2)                 # Q–Q curve
        ax.plot([gmin, gmax], [gmin, gmax], 'k--', lw=1.2)  # 45° line

        def lab(g): return f"{g[0]}-{g[1]}"
        ax.set_xlabel(f'Quantiles: {lab(g1)}', fontsize=12)
        ax.set_ylabel(f'Quantiles: {lab(g2)}', fontsize=12)
        ax.set_title(f"Q–Q: {lab(g2)} vs {lab(g1)}", fontsize=14)
        ax.set_xlim(gmin, gmax); ax.set_ylim(gmin, gmax)
        ax.grid(True, which='both', linestyle=':', linewidth=0.8, alpha=0.15)
        ax.text(-0.10, 1.05, string.ascii_lowercase[idx], transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='left')

    # hide unused axes
    for ax in axes.flat[n_pairs:]:
        ax.axis('off')

    # legend + titles
    fig.legend(handles, ['Group2 > Group1', 'Group2 < Group1'],
               loc='lower center', bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=12, frameon=True)

    title = (f"Q–Q plots — {ctx.region_label}\n"
             f"weighting={weighting}, equal_len={equalize_length}, "
             f"n_per_animal={'min' if n_per_animal is None else n_per_animal}")
    fig.suptitle(title, fontsize=16, y=1.02)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.12, wspace=0.28, hspace=0.28)

    if save_fig:
        if fig_path is None:
            fig_path = data.paths['f_speed'] / f"{ctx.region_label}_qq_weight-{weighting}_eq{equalize_length}.png"
        fig.savefig(fig_path, dpi=200)
        
#%%

# Build your analysis object and context
data = DFCAnalysis()
data.load_preprocessed_data()
data.get_temporal_parameters()

ctx = build_speed_context(data, ind_reg=16)  # ENT (your example)

# Distributions (equal animal weight; absolute counts)
plot_group_distributions(ctx, equal_animal_weight=True, equal_method="kde", normalize_density=False, save_fig=True)

# Correlation overall (equal-length fairness)
plot_dfc_speed_vs_cog_scores(ctx, weighting="animal", equalize_length=True)

# Correlation per group (equal-length fairness)
plot_dfc_speed_vs_cog_scores_per_group(ctx, weighting="animal", equalize_length=True)
#%%


# compute
df_corr = corr_speed_cog_vs_window(
    ctx,
    cog_var="index_NOR",
    reducer="median",
    by_group=True,                # or False for pooled-all
    weighting="animal",           # fairness by default
    equalize_length=True,         # same #samples per animal
    taus=None                     # or pass a list of tau indices
)

# plot
plot_corr_vs_window(df_corr, ctx, alpha=0.05, by_group=True, save_fig=False)

#%%
# Split by half; plot in absolute counts with equal animal weight (KDE averaging)
plot_short_long_distributions(
    ctx,
    mode="half",
    equal_animal_weight=True,
    equal_method="kde",
    normalize_density=False,  # absolute counts
    scale="linear",
    save_fig=True
)

# Same but normalized densities
plot_short_long_distributions(
    ctx,
    mode="half",
    equal_animal_weight=True,
    equal_method="kde",
    normalize_density=True,
    save_fig=False
)

# Threshold at a specific window length (e.g., <30 vs >=30)
plot_short_long_distributions(
    ctx,
    mode="threshold",
    split_at=30,
    equal_animal_weight=True,
    equal_method="subsample",
    n_per_animal=None,   # min length per animal
    normalize_density=True,
    save_fig=False
)
#%%
# 1) All windows, fair per-animal weighting with equal-length samples
plot_qq_groups(
    ctx,
    windows=None,
    weighting="animal",
    equalize_length=True,
    random_state=0,
    save_fig=False
)

# 2) Only LONG windows (reuse your helper from Step 5b)
pools = get_window_pools(ctx, mode="half")
plot_qq_groups(
    ctx,
    windows=pools["long"],
    weighting="animal",
    equalize_length=True,
    random_state=0,
    save_fig=True
)

#%%
import numpy as np
import pandas as pd
from itertools import combinations

def _rank_biserial_from_u(U, n1, n2, med1, med2):
    # rank-biserial correlation; sign via median difference
    r = 1.0 - 2.0 * (U / (n1 * n2))
    sign = np.sign(med1 - med2)
    return float(sign * abs(r))

def _p_adjust_bonf(pvals):
    p = np.asarray(pvals, float)
    return np.minimum(p * len(p), 1.0)

def _p_adjust_fdr_bh(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    ranked = np.arange(1, len(p) + 1)
    q = p[order] * len(p) / ranked
    q = np.minimum.accumulate(q[::-1])[::-1]  # monotone
    p_adj = np.empty_like(p)
    p_adj[order] = np.minimum(q, 1.0)
    return p_adj

def pairwise_mwu_speed_groups(group_arrays: dict, correction="bonferroni") -> pd.DataFrame:
    rows = []
    keys = list(group_arrays.keys())
    for g1, g2 in combinations(keys, 2):
        a = np.asarray(group_arrays[g1], float)
        b = np.asarray(group_arrays[g2], float)
        a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
        if (a.size < 1) or (b.size < 1):
            continue
        U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rbs = _rank_biserial_from_u(U, len(a), len(b), np.median(a), np.median(b))
        rows.append({
            "group1": "-".join(g1), "group2": "-".join(g2),
            "n1": len(a), "n2": len(b),
            "U": float(U), "p": float(p),
            "rank_biserial": rbs,
            "median1": float(np.median(a)), "median2": float(np.median(b))
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if correction.lower() in ("bonferroni", "bonf"):
        df["p_adj"] = _p_adjust_bonf(df["p"].values)
        df["p_adj_method"] = "bonferroni"
    elif correction.lower() in ("fdr", "bh", "fdr_bh"):
        df["p_adj"] = _p_adjust_fdr_bh(df["p"].values)
        df["p_adj_method"] = "fdr_bh"
    else:
        df["p_adj"] = df["p"].values
        df["p_adj_method"] = "none"

    return df.sort_values("p_adj").reset_index(drop=True)

#%%
# Choose windows: all / short / long
pools = get_window_pools(ctx, mode="half")
win_idxs = pools["long"]  # or "short", or None for all

# Build arrays per group – recommended: per-animal medians
group_arrays = build_groups_for_tests(
    ctx,
    windows=win_idxs,
    approach="per-animal",
    equalize_length=True,   # each animal contributes same # samples to its median
    reducer="median",
    random_state=0
)

# Kruskal–Wallis across groups
kw = kruskal_speed_groups(group_arrays)
print("Kruskal–Wallis:", kw)

# Pairwise MWU with FDR correction
df_mwu = pairwise_mwu_speed_groups(group_arrays, correction="fdr_bh")
print(df_mwu.head())

# (Optional) Save results
# df_mwu.to_csv(data.paths['f_speed'] / f"{ctx.region_label}_pairwise_mwu_peranimal_eq.csv", index=False)


 #%%
# Load raw data
# data.load_raw_timeseries()
# data.load_raw_cognitive_data()
# data.load_raw_region_labels()

# Load preprocessed data
data.load_preprocessed_data()

cog_data_filtered=data.cog_data_filtered
df_cog = cog_data_filtered.copy()
data.get_temporal_parameters()
#%%
# Match these variables to your last run:
prefix = "speed"
save_path = data.paths['speed']  # <-- update this!
time_window_range = data.time_window_range           # <-- list of window sizes, same as in your analysis
tau_range = np.arange(0,data.tau+ 1)                   # <-- as above
n_animals = data.n_animals                # <-- as above
data.load_preprocessed_data()

groups = data.groups  # Dictionary of groups, e.g., {'WT': [0, 1, 2], 'KO': [3, 4]}



#%%

#%%
# Plot a histogram of dFC speed for each group, pooling all windows and taus


# Set publication style (can customize further)
sns.set_theme(style='white', palette='deep', context='talk')

plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 20, 'legend.fontsize': 14})

# Use a color palette with distinct colors for groups
palette = sns.color_palette('tab10', n_colors=len(data.groups))

#%%%
#Borrar
ind_reg = 16
reg=16
# for ind_reg, reg in enumerate(range(data.regions)): 
print(ind_reg,data.region_labels_preprocessed[reg])
# window_file_total = save_path / f"{prefix}_windows{len(time_window_range)}_tau{len(tau_range)}_animals_{n_animals}.pkl"
window_file_total = save_path / f"{prefix}_region{ind_reg}_windows{len(time_window_range)}_tau{np.size(tau_range)}_animals_{n_animals}.pkl"

#All the speed values for all windows and taus
with open(window_file_total, 'rb') as f:
    all_speed = pickle.load(f)

# Now all_speed is a list (or similar) with each entry for one window_size.
# The last one:
last_speed = all_speed[-1]  # This is the speed array for the last window size

# Example: print shape/info
print(f"Loaded speed for window {time_window_range[-1]}: shape = {last_speed.shape}")




#print the shape of each time windows
for i, speed in enumerate(all_speed):
    print(f"Window size {time_window_range[i]}: shape = {speed.shape}")

# Plot a hist distribution that pools (ravel or flatten) all the speed together

# -----------------  Pool all speed values from all windows -----------------
all_speeds_flat = np.concatenate([np.concatenate([s.flatten() for s in speed]) 
                                for speed in all_speed])
([np.shape([s.flatten() for s in speed]) 
for speed in all_speed])


np.shape(all_speeds_flat)

window_sizes = time_window_range  # Your array/list of window sizes
n_windows = len(all_speed)


#%%


#helper to clean arrays

def get_valid_array(arr):
    """Convert to float, remove NaNs, and return as a 1D array."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    return arr

for (group_name, animal_indices), color in zip(data.groups.items(), palette):
    print(f"Processing group {group_name} with n animals {len(animal_indices)}")
    pooled = []
    for win_idx, win_list in enumerate(all_speed):  # Each window size
        print(f"Processing window size {window_sizes[win_idx]} for group {group_name}")
        print(f"Shape of win_list: {win_list.shape}")        
        for animal_idx in animal_indices:
            print(f"Processing animal index {animal_idx} in group {group_name}")
            for tau in range(win_list.shape[1]):
                print(f"Processing tau {tau} for animal {animal_idx} in group {group_name}")
                arr = win_list[animal_idx, tau, :]
                arr = get_valid_array(arr)
                if arr.size > 0:
                    pooled.append(arr)

#%%
def get_pooled_speed_arrays(all_speed, animal_indices, 
                           window_indices=None, tau_indices=None, 
                           verbose=False, window_sizes=None):
    """
    Pool valid speed arrays for given animals, selected windows, and selected taus.
    
    Parameters
    ----------
    all_speed : list of ndarray
        Each entry is (n_animals, n_taus, n_timepoints) for a given window size.
    animal_indices : list of int
        Indices of animals to pool.
    window_indices : list or array-like or None
        Indices of windows to include (default: all).
    tau_indices : list or array-like or None
        Indices of taus to include (default: all).
    verbose : bool
        Print details.
    window_sizes : list or None
        If provided, print human-readable window size.
    
    Returns
    -------
    pooled : list of 1D ndarray
    """
    def get_valid_array(arr):
        """Convert to float, remove NaNs, and return as a 1D array."""
        arr = np.asarray(arr, dtype=float)
        return arr[~np.isnan(arr)]
    
    pooled = []
    # Set defaults
    if window_indices is None:
        window_indices = range(len(all_speed))
    if tau_indices is None:
        # Get taus from first window (assume all the same shape)
        tau_indices = range(all_speed[0].shape[1])
    
    for w_idx in window_indices:
        win_list = all_speed[w_idx]
        if verbose and window_sizes is not None:
            print(f"Processing window size {window_sizes[w_idx]} (index {w_idx})")
            print(f"Shape: {win_list.shape}")
        for animal_idx in animal_indices:
            if verbose:
                print(f"  Animal index {animal_idx}")
            for tau in tau_indices:
                if verbose:
                    print(f"    Tau {tau}")
                arr = get_valid_array(win_list[animal_idx, tau, :])
                if arr.size > 0:
                    pooled.append(arr)
    return pooled
pooled = get_pooled_speed_arrays(all_speed, animal_indices, 
                           window_indices=None, tau_indices=None, 
                           verbose=False, window_sizes=None)
    # if pooled:
    #     group_speeds = np.concatenate(pooled)
    #     if group_speeds.size > 0:
    #         # KDE plot for smooth, publication-ready curves
    #         sns.kdeplot(group_speeds, 
    #                     bw_adjust=.5, 
    #                     label=f"{group_name[0]}-{group_name[1]}".lower(),
    #                     color=color, linewidth=2.5, clip=(0, 2))

            # # Stats lines: not in legend (set label to "_nolegend_")
            # median = np.median(group_speeds)
            # q05 = np.quantile(group_speeds, 0.05)
            # q95 = np.quantile(group_speeds, 0.95)
            # plt.axvline(median, color=color, linestyle='-', linewidth=1, alpha=0.8, label='_nolegend_')
            # plt.axvline(q05, color=color, linestyle='--', linewidth=1, alpha=0.6, label='_nolegend_')
            # plt.axvline(q95, color=color, linestyle='--', linewidth=1, alpha=0.6, label='_nolegend_')
#%%%

def plot_flatten_speed_array(data, scale='linear', save_fig=False):
    """Flatten speed array for a given animal index across all taus."""

    for (group_name, animal_indices), color in zip(data.groups.items(), palette):
        pooled = []
        for win_list in all_speed:  # Each window size
            for animal_idx in animal_indices:
                for tau in range(win_list.shape[1]):
                    arr = win_list[animal_idx, tau, :]
                    arr = np.asarray(arr, dtype=float)
                    arr = arr[~np.isnan(arr)]
                    if arr.size > 0:
                        pooled.append(arr)
        if pooled:
            group_speeds = np.concatenate(pooled)
            if group_speeds.size > 0:
                # KDE plot for smooth, publication-ready curves
                sns.kdeplot(group_speeds, 
                            bw_adjust=.5, 
                            label=f"{group_name[0]}-{group_name[1]}".lower(),
                            color=color, linewidth=2.5, clip=(0, 2))

                # Stats lines: not in legend (set label to "_nolegend_")
                median = np.median(group_speeds)
                q05 = np.quantile(group_speeds, 0.05)
                q95 = np.quantile(group_speeds, 0.95)
                plt.axvline(median, color=color, linestyle='-', linewidth=1, alpha=0.8, label='_nolegend_')
                plt.axvline(q05, color=color, linestyle='--', linewidth=1, alpha=0.6, label='_nolegend_')
                plt.axvline(q95, color=color, linestyle='--', linewidth=1, alpha=0.6, label='_nolegend_')
                            
    plt.xlabel("dFC Speed", labelpad=10)
    plt.ylabel("Density", labelpad=10)
    plt.yscale(scale)  # Log scale for better visibility of tails
    plt.title("Distribution of dFC Speeds by Group\n(All taus, all windows pooled)", pad=15)
    plt.legend(frameon=True, loc='best', title='Group')
    plt.tight_layout()

    # Remove top/right spines for a cleaner look
    sns.despine(trim=True)
    if save_fig:
        # Save the figure to the specified path
        plt.savefig(data.paths['f_speed'] / f'{data.region_labels_preprocessed[ind_reg]}_dFC_speed_distribution_{scale}.png')

def plot_median_speed_vs_window(data, scale='linear', save_fig=False):

    for idx, (group_name, animal_indices) in enumerate(data.groups.items()):
        medians_per_window = []
        q25_per_window = []
        q75_per_window = []
        for win_idx in range(n_windows):
            win_arr = all_speed[win_idx]  # shape: (n_animals, n_taus, n_timepoints)
            # Pool all animals and all taus for this group and window
            speeds_this_window = []
            for animal_idx in animal_indices:
                for tau in range(win_arr.shape[1]):
                    arr = win_arr[animal_idx, tau, :]
                    arr = np.asarray(arr, dtype=float)
                    arr = arr[~np.isnan(arr)]
                    if arr.size > 0:
                        speeds_this_window.append(arr)
            if speeds_this_window:
                flat_speeds = np.concatenate(speeds_this_window)
                median = np.median(flat_speeds)
                q25 = np.quantile(flat_speeds, 0.40)
                q75 = np.quantile(flat_speeds, 0.60)
            else:
                median = np.nan
                q25 = np.nan
                q75 = np.nan
            medians_per_window.append(median)
            q25_per_window.append(q25)
            q75_per_window.append(q75)
        color = palette[idx]
        label = f"{group_name[0]}-{group_name[1]}".lower()
        plt.plot(window_sizes, medians_per_window, marker='.', label=label, color=color, linewidth=2)
        plt.fill_between(window_sizes, q25_per_window, q75_per_window, color=color, alpha=0.1)

    plt.xlabel("Time Window Size")
    plt.ylabel("Median dFC Speed (group, all tau pooled)")
    plt.yscale(scale)  # Log scale for better visibility of tails
    plt.title("Median dFC Speed vs. Window Size by Group\nShading = 25–75% quantile")
    plt.legend(title='group', fontsize=10, ncol=2)
    plt.tight_layout()
    sns.despine(trim=True)
    if save_fig:
        plt.savefig(data.paths['f_speed'] / f'{data.region_labels_preprocessed[ind_reg]}_dFC_speed_vs_window_size_{scale}.png')
#%%

# ------------------------ NOR scores vs dFC speed ------------------------

def plot_dfc_speed_vs_cog_scores(data, save_fig=False):

    # Load cognitive data
    cog_scores = data.cog_data_filtered['index_NOR'].values

    # 1. Compute per-animal dFC speed median
    # Assume all_speed: list of window arrays (n_animals, n_taus, n_timepoints)
    per_animal_speeds = []
    for animal_idx in range(n_animals):
        pooled = []
        for win_arr in all_speed:
            for tau in range(win_arr.shape[1]):
                arr = win_arr[animal_idx, tau, :].astype(float)
                arr = arr[~np.isnan(arr)]
                pooled.append(arr)
        if pooled:
            flat = np.concatenate(pooled)
            per_animal_speeds.append(np.median(flat))
        else:
            per_animal_speeds.append(np.nan)

    per_animal_speeds = np.array(per_animal_speeds)

    # 2. Extract cognitive scores, aligned to animals in the same order!
    # cog_scores = data.cog_data_filtered.loc[:n_animals-1, 'index_NOR'].values  # adjust column as needed

    # 3. Scatter plot
    plt.scatter(per_animal_speeds, cog_scores, c='k', alpha=0.8)
    plt.xlabel('Median dFC Speed per animal')
    plt.ylabel('Cognitive score (e.g., NOR index)')
    plt.title('Relationship between dFC speed and cognitive score')

    # 4. Correlation
    mask = ~np.isnan(per_animal_speeds) & ~np.isnan(cog_scores)
    rho, pval = spearmanr(per_animal_speeds[mask], cog_scores[mask])
    plt.text(0.05, 0.95, f"Spearman r={rho:.2f}, p={pval:.3g}",
            transform=plt.gca().transAxes, va='top', ha='left', fontsize=12)

    plt.tight_layout()
    if save_fig:
        plt.savefig(data.paths['f_speed'] / f'{data.region_labels_preprocessed[ind_reg]}_dFC_speed_vs_cog_scores.png')
        
#%%



def plot_dfc_speed_vs_cog_scores_per_group(data, save_fig=False):
    """
    Plot dFC speed vs cognitive scores, stratified by genotype and treatment.
    
    Parameters:
    - data: Data object containing groups and paths.
    - all_speed: List of speed arrays for each time window.
    - time_window_range: Range of time windows used in the analysis.
    """

    
    # Ensure data is filtered correctly
    cog_data_filtered = data.cog_data_filtered  # Assuming this is a DataFrame with 'genotype', 'treatment', and 'index_NOR'


    # Median dFC speed per animal
    n_animals = all_speed[0].shape[0]
    per_animal_speeds = []
    for animal_idx in range(n_animals):
        pooled = []
        for win_arr in all_speed:
            for tau in range(win_arr.shape[1]):
                arr = win_arr[animal_idx, tau, :].astype(float)
                arr = arr[~np.isnan(arr)]
                pooled.append(arr)
        if pooled:
            flat = np.concatenate(pooled)
            per_animal_speeds.append(np.median(flat))
        else:
            per_animal_speeds.append(np.nan)
    per_animal_speeds = np.array(per_animal_speeds)

    # Cognitive scores and group labels
    cog_df = cog_data_filtered.reset_index(drop=True)
    cog_scores = cog_df['index_NOR'].values
    group_labels = list(zip(cog_df['genotype'], cog_df['treatment']))

    # Assign color/marker per group
    groups = sorted(set(group_labels))
    palette = sns.color_palette('tab10', n_colors=len(groups))
    group2color = {g: palette[i] for i, g in enumerate(groups)}
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '+', 'x']
    group2marker = {g: markers[i % len(markers)] for i, g in enumerate(groups)}

    for i, group in enumerate(groups):
        idxs = [j for j, g in enumerate(group_labels) if g == group]
        speeds = per_animal_speeds[idxs]
        scores = cog_scores[idxs]
        plt.scatter(
            speeds, scores,
            color=group2color[group], marker=group2marker[group],
            label=f"{group[0]}-{group[1]}", s=70, alpha=0.85
        )
        # Only fit if enough data
        mask = ~np.isnan(speeds) & ~np.isnan(scores)
        if np.sum(mask) > 2:
            # Theil-Sen regression (robust to outliers)
            ts_slope, ts_intercept, ts_low, ts_high = theilslopes(scores[mask], speeds[mask])
            xfit = np.linspace(np.nanmin(speeds[mask]), np.nanmax(speeds[mask]), 100)
            yfit = ts_slope * xfit + ts_intercept
            plt.plot(
                xfit, yfit, color=group2color[group],
                linestyle='-', linewidth=2,
                alpha=0.75
            )
            # Spearman correlation
            rho, pval = spearmanr(speeds[mask], scores[mask])
            plt.text(
                0.98, 0.98-i*0.09,
                f"{group[0]}-{group[1]}: ρ={rho:.2f}, p={pval:.2g}",
                color=group2color[group],
                transform=plt.gca().transAxes, fontsize=10, ha='right', va='top'
            )

    plt.xlabel('Median dFC Speed per animal')
    plt.ylabel('Cognitive score (NOR index)')
    plt.title('dFC speed vs. cognitive score, stratified by group\n(Theil-Sen + Spearman)')
    plt.legend(title='Genotype-Treatment', fontsize=10, title_fontsize=12)
    plt.tight_layout()
    if save_fig:
        plt.savefig(data.paths['f_speed'] / f'{data.region_labels_preprocessed[ind_reg]}_dFC_speed_vs_cog_scores_per_group.png')

#%%
# Plot dFC speed vs cognitive scores across different window sizes


def plot_dfc_speed_vs_cog_scores_per_window(data, save_fig=False):
    """
    Plot dFC speed vs cognitive scores across different window sizes.
    """
    window_sizes = time_window_range
    n_windows = len(window_sizes)
    group_dict = data.groups
    cog_df = cog_data_filtered.reset_index(drop=True)

    palette = sns.color_palette('tab10', n_colors=len(group_dict))

    #Plotting correlations between dFC speed and cognitive scores across different window sizes

    for idx, (group, animal_indices) in enumerate(group_dict.items()):
        correlations = []
        pvalues = []
        group_scores = cog_df.loc[animal_indices, 'index_NOR'].values

        for win_idx in range(n_windows):
            win_arr = all_speed[win_idx]
            medians = []
            for animal_idx in animal_indices:
                pooled = []
                for tau in range(win_arr.shape[1]):
                    arr = win_arr[animal_idx, tau, :].astype(float)
                    arr = arr[~np.isnan(arr)]
                    pooled.append(arr)
                if pooled:
                    flat = np.concatenate(pooled)
                    medians.append(np.median(flat))
                else:
                    medians.append(np.nan)
            medians = np.array(medians)
            mask = ~np.isnan(medians) & ~np.isnan(group_scores)
            if np.sum(mask) > 2:
                rho, pval = spearmanr(medians[mask], group_scores[mask])
            else:
                rho, pval = np.nan, np.nan
            correlations.append(rho)
            pvalues.append(pval)
            


        label = f"{group[0]}-{group[1]}".lower()
        plt.plot(window_sizes, correlations, '-o', color=palette[idx], label=label, zorder=2)
        # Overlay significance marker for p < 0.05
        correlations = np.array(correlations)
        pvalues = np.array(pvalues)
        sig_idx = np.where((pvalues < 0.05) & ~np.isnan(pvalues) & ~np.isnan(correlations))[0]
        # Plot filled stars at significant points
        plt.scatter(np.array(window_sizes)[sig_idx], correlations[sig_idx],
                    color=palette[idx], marker='*', s=110, edgecolor='k', linewidth=0.8, zorder=4, label=None)
        


    plt.axhline(0, color='grey', linestyle='--', linewidth=1, zorder=1)
    plt.xlabel('Window Size')
    plt.ylabel("Spearman correlation (dFC speed, cognitive score)")
    plt.title("Correlation vs. Window Size by Group\nStars = p < 0.05")
    plt.ylim(-1, 1)
    plt.xlim(window_sizes[0]-1, window_sizes[-1]+1)
    plt.legend(title='Group')
    plt.tight_layout()
    if save_fig:
        plt.savefig(data.paths['f_speed'] / f'{data.region_labels_preprocessed[ind_reg]}_dFC_speed_vs_cog_scores_per_window.png')
#%%
def plot_dfc_speed_distribution(data, scale='linear', save_fig=False):
    """Plot the distribution of dFC speeds for short and long window pools.
    """
    short_idx = np.arange(len(window_sizes) // 2)
    long_idx = np.arange(len(window_sizes) // 2, len(window_sizes))
    pool_defs = {'Short windows': short_idx, 'Long windows': long_idx}
    palette = sns.color_palette('tab10', n_colors=len(groups))


    for pool_i, (pool_name, idxs) in enumerate(pool_defs.items()):
        plt.subplot(1, 2, pool_i+1)
        for g_idx, (group, animal_idxs) in enumerate(data.groups.items()):


            # Pool all speeds for this group and this pool
            group_speeds = []
            for win_idx in idxs:
                win_arr = all_speed[win_idx]  # (n_animals, n_tau, n_timepoints)
                for animal_idx in animal_idxs:
                    for tau in range(win_arr.shape[1]):
                        arr = win_arr[animal_idx, tau, :].astype(float)
                        arr = arr[~np.isnan(arr)]
                        if arr.size > 0:
                            group_speeds.append(arr)
            group_speeds = np.concatenate(group_speeds) if group_speeds else np.array([])
            # Histogram (step)
            sns.histplot(group_speeds, bins=60, stat='density', element='step', fill=False,
                        color=palette[g_idx], linewidth=1.6, label=f'{group}', alpha=0.6)
            # KDE (over histogram)
            if group_speeds.size > 10:  # Avoid noise for tiny samples
                sns.kdeplot(group_speeds, color=palette[g_idx], lw=2.1, label=None)
            # Median
            plt.axvline(np.median(group_speeds), color=palette[g_idx], linestyle='--', lw=1)
        plt.xlabel("dFC Speed")
        plt.ylabel("Density")
        plt.yscale(scale)
        plt.title(f"{pool_name}")
        # plt.yscale('log')  # Log scale for better visibility
        if pool_i == 0:
            plt.legend(title='Group', fontsize=10)
        else:
            plt.legend().set_visible(False)
        plt.tight_layout()

    plt.suptitle("Distribution of dFC Speeds by Group\nShort vs. Long Window Pools", fontsize=15, y=1.02)
    sns.despine()
    plt.tight_layout()
    if save_fig:
        plt.savefig(data.paths['f_speed'] / f'{data.region_labels_preprocessed[ind_reg]}_dfc_speed_distribution_short_long_pools_{scale}.png')

#%%


# Suppose you have: all_speed, window_sizes, groups from previous code

def plot_qq_plot(data, scale='linear', save_fig=False):
    long_win_indices = np.arange(len(window_sizes)//2, len(window_sizes))
    group_speeds_dict = {}

    for group, animal_idxs in data.groups.items():
        pooled_speeds = []
        for animal_idx in animal_idxs:
            for win_idx in long_win_indices:
                win_arr = all_speed[win_idx]  # shape (n_animals, n_tau, n_timepoints)
                for tau in range(win_arr.shape[1]):
                    arr = win_arr[animal_idx, tau, :].astype(float)
                    arr = arr[~np.isnan(arr)]
                    if arr.size > 0:
                        pooled_speeds.append(arr)
        if pooled_speeds:
            group_speeds_dict[group] = np.concatenate(pooled_speeds)
        else:
            group_speeds_dict[group] = np.array([])


    # Helper: tuple to label
    def group_to_str(group):
        if isinstance(group, tuple):
            return f"{group[0]}-{group[1]}"
        else:
            return str(group)

    groups_list = list(group_speeds_dict.keys())
    n_pairs = len(groups_list) * (len(groups_list) - 1) // 2
    n_cols = 3
    n_rows = int(np.ceil(n_pairs / n_cols))
    n_points = 1000

    # All values for axis limits
    all_vals = np.concatenate([v for v in group_speeds_dict.values() if len(v) > 0])
    global_min = float(np.nanmin(all_vals))
    global_max = float(np.nanmax(all_vals))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.7*n_cols, 5.7*n_rows), squeeze=False)
    legend_handles = []

    for panel_idx, (ax, (g1, g2)) in enumerate(zip(axes.flat, combinations(groups_list, 2))):
        arr1 = group_speeds_dict[g1]
        arr2 = group_speeds_dict[g2]
        if len(arr1) == 0 or len(arr2) == 0:
            ax.axis('off')
            continue
        q = np.linspace(0, 1, n_points)
        quant1 = np.quantile(arr1, q)
        quant2 = np.quantile(arr2, q)
        above = quant2 > quant1
        below = quant2 < quant1

        # Fill areas
        h1 = ax.fill_between(quant1, quant1, quant2, where=above, color='firebrick', alpha=0.40, label='Group2 > Group1')
        h2 = ax.fill_between(quant1, quant1, quant2, where=below, color='dodgerblue', alpha=0.40, label='Group2 < Group1')
        if not legend_handles:
            legend_handles = [h1, h2]
        # Q-Q and diagonal
        ax.plot(quant1, quant2, color='k', lw=2)
        ax.plot([global_min, global_max], [global_min, global_max], 'k--', lw=1.3)

        # Labels and title
        lab1 = group_to_str(g1)
        lab2 = group_to_str(g2)
        ax.set_xlabel(f'Quantiles: {lab1}', fontsize=15, fontweight='bold')
        ax.set_ylabel(f'Quantiles: {lab2}', fontsize=15, fontweight='bold')
        ax.set_title(f"Q-Q: {lab2} vs {lab1}", fontsize=15, fontweight='bold', pad=13)
        # Axis scale
        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)
        ax.tick_params(axis='both', labelsize=15, width=1.2)
        # Panel label (a, b, c, ...)
        ax.text(-0.10, 1.05, string.ascii_lowercase[panel_idx],
                fontsize=19, fontweight='bold', transform=ax.transAxes, va='top', ha='left')
        # Optional faint grid
        ax.grid(True, which='both', linestyle=':', linewidth=0.8, alpha=0.15)

    # Hide unused axes
    for ax in axes.flat[n_pairs:]:
        ax.axis('off')

    # Shared legend below all panels
    fig.legend(legend_handles, ['Group2 > Group1', 'Group2 < Group1'],
            loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=2,
            fontsize=14, frameon=True, borderaxespad=1.0)

    # Interpretation hint below legend
    fig.text(0.5, -0.13,
            "Red fill: Group2 > Group1 (Q–Q curve above diagonal). Blue fill: Group2 < Group1.",
            ha='center', va='center', fontsize=13, color='dimgray')

    # Supertitle above panels
    fig.suptitle('Q–Q Plots (Filled): All Group Pairwise Comparisons',
                fontsize=18, fontweight='semibold', y=1.03)

    plt.subplots_adjust(left=0.09, right=0.96, bottom=0.13, top=0.94, wspace=0.25, hspace=0.28)
    if save_fig==True:
        plt.savefig(data.paths['f_speed'] / f'{data.region_labels_preprocessed[ind_reg]}_dfc_speed_qq_plots_{scale}.png')
# data.region_labels_preprocessed[ind_reg]

#%%
# Build the per-group pooled speed dictionary for the long windows


def plot_qq_plot_long(data, save_fig=False):

    # Split window indices into long windows (second half)

    long_win_indices = np.arange(len(window_sizes)//2, len(window_sizes))

    group_speeds_dict_long = {}
    for group, animal_idxs in data.groups.items():
        pooled_speeds = []
        for animal_idx in animal_idxs:
            for win_idx in long_win_indices:
                win_arr = all_speed[win_idx]
                for tau in range(win_arr.shape[1]):
                    arr = win_arr[animal_idx, tau, :].astype(float)
                    arr = arr[~np.isnan(arr)]
                    if arr.size > 0:
                        pooled_speeds.append(arr)
        if pooled_speeds:
            group_speeds_dict_long[group] = np.concatenate(pooled_speeds)
        else:
            group_speeds_dict_long[group] = np.array([])

    # Now re-use the Q–Q grid code (with supertitle tweak)
    def group_to_str(group):
        if isinstance(group, tuple):
            return f"{group[0]}-{group[1]}"
        else:
            return str(group)

    groups_list = list(group_speeds_dict_long.keys())
    n_pairs = len(groups_list) * (len(groups_list) - 1) // 2
    n_cols = 3
    n_rows = int(np.ceil(n_pairs / n_cols))
    n_points = 1000

    all_vals = np.concatenate([v for v in group_speeds_dict_long.values() if len(v) > 0])
    global_min = float(np.nanmin(all_vals))
    global_max = float(np.nanmax(all_vals))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.7*n_cols, 5.7*n_rows), squeeze=False)
    legend_handles = []

    for panel_idx, (ax, (g1, g2)) in enumerate(zip(axes.flat, combinations(groups_list, 2))):
        arr1 = group_speeds_dict_long[g1]
        arr2 = group_speeds_dict_long[g2]
        if len(arr1) == 0 or len(arr2) == 0:
            ax.axis('off')
            continue
        q = np.linspace(0, 1, n_points)
        quant1 = np.quantile(arr1, q)
        quant2 = np.quantile(arr2, q)
        above = quant2 > quant1
        below = quant2 < quant1

        h1 = ax.fill_between(quant1, quant1, quant2, where=above, color='firebrick', alpha=0.40, label='Group2 > Group1')
        h2 = ax.fill_between(quant1, quant1, quant2, where=below, color='dodgerblue', alpha=0.40, label='Group2 < Group1')
        if not legend_handles:
            legend_handles = [h1, h2]

        ax.plot(quant1, quant2, color='k', lw=2)
        ax.plot([global_min, global_max], [global_min, global_max], 'k--', lw=1.3)

        lab1 = group_to_str(g1)
        lab2 = group_to_str(g2)
        ax.set_xlabel(f'Quantiles: {lab1}', fontsize=15, fontweight='bold')
        ax.set_ylabel(f'Quantiles: {lab2}', fontsize=15, fontweight='bold')
        ax.set_title(f"Quantile–Quantile: {lab2} vs {lab1}", fontsize=15, fontweight='bold', pad=13)
        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)
        ax.tick_params(axis='both', labelsize=15, width=1.2)
        ax.text(-0.10, 1.05, string.ascii_lowercase[panel_idx],
                fontsize=19, fontweight='bold', transform=ax.transAxes, va='top', ha='left')
        ax.grid(True, which='both', linestyle=':', linewidth=0.8, alpha=0.15)

    for ax in axes.flat[n_pairs:]:
        ax.axis('off')

    fig.legend(legend_handles, ['Group2 > Group1', 'Group2 < Group1'],
            loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=2,
            fontsize=14, frameon=True, borderaxespad=1.0)

    fig.text(0.5, -0.13,
            "Red fill: Group2 > Group1 (Q–Q curve above diagonal). Blue fill: Group2 < Group1.",
            ha='center', va='center', fontsize=13, color='dimgray')

    fig.suptitle('Q–Q Plots (Filled): Long Window Pool, All Group Pairwise Comparisons',
                fontsize=18, fontweight='semibold', y=1.03)

    plt.subplots_adjust(left=0.09, right=0.96, bottom=0.13, top=0.94, wspace=0.25, hspace=0.28)
    if save_fig:
        plt.savefig(data.paths['f_speed'] / f'{data.region_labels_preprocessed[ind_reg]}_dfc_speed_qq_plots_long.png')


#%%
ind_reg = 16
reg=16
for ind_reg, reg in enumerate(range(data.regions)): 
    print(ind_reg,data.region_labels_preprocessed[reg])
    # window_file_total = save_path / f"{prefix}_windows{len(time_window_range)}_tau{len(tau_range)}_animals_{n_animals}.pkl"
    window_file_total = save_path / f"{prefix}_region{ind_reg}_windows{len(time_window_range)}_tau{np.size(tau_range)}_animals_{n_animals}.pkl"

    #All the speed values for all windows and taus
    with open(window_file_total, 'rb') as f:
        all_speed = pickle.load(f)

    # Now all_speed is a list (or similar) with each entry for one window_size.
    # The last one:
    last_speed = all_speed[-1]  # This is the speed array for the last window size

    # Example: print shape/info
    print(f"Loaded speed for window {time_window_range[-1]}: shape = {last_speed.shape}")




    #print the shape of each time windows
    for i, speed in enumerate(all_speed):
        print(f"Window size {time_window_range[i]}: shape = {speed.shape}")

    # Plot a hist distribution that pools (ravel or flatten) all the speed together

    # -----------------  Pool all speed values from all windows -----------------
    all_speeds_flat = np.concatenate([np.concatenate([s.flatten() for s in speed]) 
                                    for speed in all_speed])
    ([np.shape([s.flatten() for s in speed]) 
    for speed in all_speed])


    np.shape(all_speeds_flat)

    window_sizes = time_window_range  # Your array/list of window sizes
    n_windows = len(all_speed)

    plt.figure(1, figsize=(10, 6))
    plot_flatten_speed_array(data, scale='linear', save_fig=True)
    plt.figure(2, figsize=(10, 6))
    plot_flatten_speed_array(data, scale='log', save_fig=True)

    # Plot median dFC speed vs. time window size, pooling all taus for each group
    palette = sns.color_palette('tab10', n_colors=len(data.groups))

    plt.figure(3, figsize=(13,6))
    plot_median_speed_vs_window(data, scale='linear', save_fig=True)
    plt.figure(4, figsize=(13,6))
    plot_median_speed_vs_window(data, scale='log', save_fig=True)
    plt.show()

    plt.figure(5, figsize=(9,6))
    plot_dfc_speed_vs_cog_scores(data, save_fig=True)
    plt.show()

    plt.figure(6, figsize=(9, 7))
    plot_dfc_speed_vs_cog_scores_per_group(data, save_fig=True)
    plt.show()


    plt.figure(7,figsize=(13,8))
    plot_dfc_speed_vs_cog_scores_per_window(data, save_fig=True)
    plt.show()

    plt.figure(8, figsize=(12, 6))
    plot_dfc_speed_distribution(data, scale='linear', save_fig=True)
    plt.show()

    plt.figure(9, figsize=(12, 6))
    plot_dfc_speed_distribution(data, scale='log', save_fig=True)
    plt.show()

    plt.figure(10, figsize=(12, 6))
    plot_qq_plot(data, scale='linear', save_fig=True)
    plt.show()

    plt.figure(11, figsize=(12, 6))
    plot_qq_plot_long(data, save_fig=True)   
    plt.show()

    print(f"Plots for region {data.region_labels_preprocessed[ind_reg]} saved successfully.")

# %%

# quantile_levels = np.linspace(0, 1, 20)
# n_windows = len(window_sizes)
# n_q = len(quantile_levels)

# group_names = list(data.groups.keys())[::-1]
# n_groups = len(group_names)
# n_rows = int(np.ceil(np.sqrt(n_groups)))
# n_cols = int(np.ceil(n_groups / n_rows))

# fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 10), sharex=True, sharey=True)
# axes = axes.flatten()

# vmin = np.inf
# vmax = -np.inf
# speed_matrices = []

# for group_name in group_names:
#     animal_indices = data.groups[group_name]
#     speed_matrix = np.full((n_q, n_windows), np.nan)
#     for win_idx in range(n_windows):
#         win_arr = all_speed[win_idx]
#         speeds_this_window = []
#         for animal_idx in animal_indices:
#             for tau in range(win_arr.shape[1]):
#                 arr = win_arr[animal_idx, tau, :].astype(float)


#                 arr = arr[~np.isnan(arr)]
#                 if arr.size > 0:
#                     speeds_this_window.append(arr)
#         if speeds_this_window:
#             flat_speeds = np.concatenate(speeds_this_window)
#             if flat_speeds.size > 0:
#                 speed_matrix[:, win_idx] = [np.quantile(flat_speeds, q) for q in quantile_levels]
#     valid = ~np.isnan(speed_matrix)
#     if np.any(valid):
#         vmin = min(vmin, np.nanmin(speed_matrix))
#         vmax = max(vmax, np.nanmax(speed_matrix))
#     speed_matrices.append(speed_matrix)

# # --- Overlay IQR and median curves ---
# q25_idx = np.argmin(np.abs(quantile_levels - 0.25))
# q50_idx = np.argmin(np.abs(quantile_levels - 0.5))
# q75_idx = np.argmin(np.abs(quantile_levels - 0.75))

# for idx, group_name in enumerate(group_names):
#     ax = axes[idx]
#     mat = speed_matrices[idx]
#     im = ax.imshow(
#         mat,
#         aspect='auto',
#         origin='lower',
#         extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],
#         cmap='magma',
#         vmin=vmin, vmax=vmax
#     )
#     label = f"{group_name[0]}-{group_name[1]}".lower()
#     ax.set_title(label)
#     ax.set_xlabel('Window Size')
#     ax.set_ylabel('Quantile')
#     ax.label_outer()
#     # Overlay median and IQR
#     ax.plot(window_sizes, mat[q50_idx, :], color='w', lw=2.2, label='Median')
#     ax.plot(window_sizes, mat[q25_idx, :], color='w', lw=1.3, ls='--', label='IQR')
#     ax.plot(window_sizes, mat[q75_idx, :], color='w', lw=1.3, ls='--')

# # Place colorbar on the left
# fig.subplots_adjust(left=0.15, right=0.95)
# cbar_ax = fig.add_axes([0.05, 0.25, 0.02, 0.5])
# fig.colorbar(im, cax=cbar_ax, orientation='vertical', label='dFC Speed')

# # Add legend to the first panel (remove duplicate labels)
# handles, labels = axes[0].get_legend_handles_labels()
# axes[0].legend(handles[:2], ['Median', 'IQR'], loc='upper right', frameon=True)

# # plt.tight_layout(rect=[0.15, 0, 1, 1])
# plt.show()

# #%%


# quantile_levels = np.linspace(0, 1, 100)
# n_windows = len(window_sizes)
# n_q = len(quantile_levels)

# group_names = list(data.groups.keys())[::-1]
# n_groups = len(group_names)
# n_rows = int(np.ceil(np.sqrt(n_groups)))
# n_cols = int(np.ceil(n_groups / n_rows))

# fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 10), sharex=True, sharey=True)
# axes = axes.flatten()

# vmin = np.inf
# vmax = -np.inf
# speed_matrices = []

# for group_name in group_names:
#     animal_indices = data.groups[group_name]
#     speed_matrix = np.full((n_q, n_windows), np.nan)
#     for win_idx in range(n_windows):
#         win_arr = all_speed[win_idx]
#         speeds_this_window = []
#         for animal_idx in animal_indices:
#             for tau in range(win_arr.shape[1]):
#                 arr = win_arr[animal_idx, tau, :].astype(float)


#                 arr = arr[~np.isnan(arr)]
#                 if arr.size > 0:
#                     speeds_this_window.append(arr)
#         if speeds_this_window:
#             flat_speeds = np.concatenate(speeds_this_window)
#             if flat_speeds.size > 0:
#                 speed_matrix[:, win_idx] = [np.quantile(flat_speeds, q) for q in quantile_levels]
#     valid = ~np.isnan(speed_matrix)
#     if np.any(valid):
#         vmin = min(vmin, np.nanmin(speed_matrix))
#         vmax = max(vmax, np.nanmax(speed_matrix))
#     speed_matrices.append(speed_matrix)

# # --- Overlay IQR and median curves ---
# q25_idx = np.argmin(np.abs(quantile_levels - 0.25))
# q50_idx = np.argmin(np.abs(quantile_levels - 0.5))
# q75_idx = np.argmin(np.abs(quantile_levels - 0.75))

# for idx, group_name in enumerate(group_names):
#     ax = axes[idx]
#     mat = speed_matrices[idx]
#     im = ax.imshow(
#         mat,
#         aspect='auto',
#         origin='lower',
#         extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],
#         cmap='magma',
#         vmin=vmin, vmax=vmax
#     )
#     label = f"{group_name[0]}-{group_name[1]}".lower()
#     ax.set_title(label)
#     ax.set_xlabel('Window Size')
#     ax.set_ylabel('Quantile')
#     ax.label_outer()
#     # Overlay median and IQR
#     ax.plot(window_sizes, mat[q50_idx, :], color='w', lw=2.2, label='Median')
#     ax.plot(window_sizes, mat[q25_idx, :], color='w', lw=1.3, ls='--', label='IQR')
#     ax.plot(window_sizes, mat[q75_idx, :], color='w', lw=1.3, ls='--')

# # Place colorbar on the left
# fig.subplots_adjust(left=0.15, right=0.95)
# cbar_ax = fig.add_axes([0.05, 0.25, 0.02, 0.5])
# fig.colorbar(im, cax=cbar_ax, orientation='vertical', label='dFC Speed')

# # Add legend to the first panel (remove duplicate labels)
# handles, labels = axes[0].get_legend_handles_labels()
# axes[0].legend(handles[:2], ['Median', 'IQR'], loc='upper right', frameon=True)

# plt.tight_layout(rect=[0.15, 0, 1, 1])
# plt.show()





# # %%


# # Calculate pairwise differences (assumes 4 groups: A, B, C, D)
# diff_AB = speed_matrices[0] - speed_matrices[1]
# diff_AC = speed_matrices[0] - speed_matrices[2]
# diff_AD = speed_matrices[0] - speed_matrices[3]
# diff_BC = speed_matrices[1] - speed_matrices[2]
# diff_BD = speed_matrices[1] - speed_matrices[3]
# diff_CD = speed_matrices[2] - speed_matrices[3]

# diff_vmax = np.nanmax(np.abs([diff_AB, diff_AC, diff_AD, diff_BC, diff_BD, diff_CD]))
# diff_cmap = 'bwr'

# fig = plt.figure(figsize=(20, 18))
# gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1])

# # Shared color limits for all original matrices
# vmin = min(np.nanmin(m) for m in speed_matrices)
# vmax = max(np.nanmax(m) for m in speed_matrices)

# # Row 1: A, B, A-B
# ax1 = fig.add_subplot(gs[0, 0])
# im1 = ax1.imshow(speed_matrices[0], aspect='auto', origin='lower',
#                  extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],
#                  cmap='magma', vmin=vmin, vmax=vmax)
# ax1.set_title(f'{group_names[0][0]}-{group_names[0][1]}'.lower())
# ax1.set_ylabel('Quantile')

# ax2 = fig.add_subplot(gs[0, 1])
# im2 = ax2.imshow(speed_matrices[1], aspect='auto', origin='lower',
#                  extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],
#                  cmap='magma', vmin=vmin, vmax=vmax)
# ax2.set_title(f'{group_names[1][0]}-{group_names[1][1]}'.lower())

# ax3 = fig.add_subplot(gs[0, 2])
# im3 = ax3.imshow(diff_AB, aspect='auto', origin='lower',
#                  extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],
#                  cmap=diff_cmap, vmin=-diff_vmax, vmax=diff_vmax)
# ax3.set_title(f'Diff: {group_names[0][0]}-{group_names[0][1]} - {group_names[1][0]}-{group_names[1][1]}')
# ax3.set_ylabel('Quantile')

# # Row 2: C, D, C-D
# ax4 = fig.add_subplot(gs[1, 0])
# im4 = ax4.imshow(speed_matrices[2], aspect='auto', origin='lower',
#                  extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],
#                  cmap='magma', vmin=vmin, vmax=vmax)
# ax4.set_title(f'{group_names[2][0]}-{group_names[2][1]}'.lower())
# ax4.set_ylabel('Quantile')

# ax5 = fig.add_subplot(gs[1, 1])
# im5 = ax5.imshow(speed_matrices[3], aspect='auto', origin='lower',
#                  extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],
#                  cmap='magma', vmin=vmin, vmax=vmax)
# ax5.set_title(f'{group_names[3][0]}-{group_names[3][1]}'.lower())

# ax6 = fig.add_subplot(gs[1, 2])
# im6 = ax6.imshow(diff_CD, aspect='auto', origin='lower',
#                  extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],
#                  cmap=diff_cmap, vmin=-diff_vmax, vmax=diff_vmax)
# ax6.set_title(f'Diff: {group_names[2][0]}-{group_names[2][1]} - {group_names[3][0]}-{group_names[3][1]}')

# # Row 3: A-C, B-D, A-D
# ax7 = fig.add_subplot(gs[2, 0])
# im7 = ax7.imshow(diff_AC, aspect='auto', origin='lower',
#                  extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],
#                  cmap=diff_cmap, vmin=-diff_vmax, vmax=diff_vmax)
# ax7.set_title(f'Diff: {group_names[0][0]}-{group_names[0][1]} - {group_names[2][0]}-{group_names[2][1]}')
# ax7.set_xlabel('Window Size')
# ax7.set_ylabel('Quantile')

# ax8 = fig.add_subplot(gs[2, 1])
# im8 = ax8.imshow(diff_BD, aspect='auto', origin='lower',
#                  extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],
#                  cmap=diff_cmap, vmin=-diff_vmax, vmax=diff_vmax)
# ax8.set_title(f'Diff: {group_names[1][0]}-{group_names[1][1]} - {group_names[3][0]}-{group_names[3][1]}')
# ax8.set_xlabel('Window Size')

# ax9 = fig.add_subplot(gs[2, 2])
# im9 = ax9.imshow(diff_AD, aspect='auto', origin='lower',
#                  extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],
#                  cmap=diff_cmap, vmin=-diff_vmax, vmax=diff_vmax)
# ax9.set_title(f'Diff: {group_names[0][0]}-{group_names[0][1]} - {group_names[3][0]}-{group_names[3][1]}')
# ax9.set_xlabel('Window Size')

# # Shared colorbars
# fig.subplots_adjust(left=0.07, right=0.91, wspace=0.27, hspace=0.23)
# cbar_ax1 = fig.add_axes([0.93, 0.65, 0.015, 0.27])
# fig.colorbar(im1, cax=cbar_ax1, orientation='vertical', label='dFC Speed')

# cbar_ax2 = fig.add_axes([0.93, 0.12, 0.015, 0.35])
# fig.colorbar(im3, cax=cbar_ax2, orientation='vertical', label='dFC Speed Diff')

# plt.show()



# # %%

# # Assume speed_matrices, group_names, window_sizes, quantile_levels are defined
# N = len(speed_matrices)
# diff_pairs = list(itertools.combinations(range(N), 2))  # All unique pairs
# n_diffs = len(diff_pairs)

# ncols = N  # One column per group
# nrows = 1 + math.ceil(n_diffs / ncols)  # 1 row for originals, rest for differences

# fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), sharex=True, sharey=True)
# axes = axes.flatten()

# # Color scaling for original and difference matrices
# vmin = min(np.nanmin(m) for m in speed_matrices)
# vmax = max(np.nanmax(m) for m in speed_matrices)
# diff_matrices = []
# for i, j in diff_pairs:
#     diff_matrices.append(speed_matrices[i] - speed_matrices[j])
# diff_vmax = np.nanmax(np.abs(diff_matrices))

# # Row 1: original groups
# for idx, mat in enumerate(speed_matrices):
#     ax = axes[idx]
#     im = ax.imshow(mat, aspect='auto', origin='lower',
#                    extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],
#                    cmap='magma', vmin=vmin, vmax=vmax)
#     label = f"{group_names[idx][0]}-{group_names[idx][1]}".lower()
#     ax.set_title(label)
#     ax.set_ylabel('Quantile')
#     ax.set_xlabel('Window Size')
#     ax.label_outer()

# # Next rows: all pairwise differences
# for d_idx, (i, j) in enumerate(diff_pairs):
#     ax_idx = N + d_idx
#     ax = axes[ax_idx]
#     im_diff = ax.imshow(
#         speed_matrices[i] - speed_matrices[j],
#         aspect='auto', origin='lower',
#         extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],
#         cmap='bwr', vmin=-diff_vmax, vmax=diff_vmax
#     )
#     label = f"Diff: {group_names[i][0]}-{group_names[i][1]} - {group_names[j][0]}-{group_names[j][1]}"
#     ax.set_title(label.lower())
#     ax.set_ylabel('Quantile')
#     ax.set_xlabel('Window Size')
#     ax.label_outer()

# # Hide unused axes
# for ax in axes[N + n_diffs:]:
#     ax.axis('off')

# # Colorbars
# fig.subplots_adjust(right=0.92, hspace=0.38, wspace=0.18)
# cbar_ax1 = fig.add_axes([0.93, 0.77, 0.015, 0.17])
# fig.colorbar(im, cax=cbar_ax1, orientation='vertical', label='dFC Speed')
# cbar_ax2 = fig.add_axes([0.93, 0.15, 0.015, 0.57])
# fig.colorbar(im_diff, cax=cbar_ax2, orientation='vertical', label='dFC Speed Diff')

# plt.tight_layout(rect=[0, 0, 0.92, 1])
# plt.show()

#%%


#%%



# %%
# %%


# %%




# #---------------------------- Two timescales --------------------------------


# # Split window indices into two pools (first half, second half)
# n_windows = len(window_sizes)
# first_half_idx = np.arange(n_windows // 2)
# second_half_idx = np.arange(n_windows // 2, n_windows)

# pools = [first_half_idx, second_half_idx]
# pool_labels = ['short', 'long']

# # Prepare
# n_animals = cog_data_filtered.shape[0]
# per_animal_summaries = {label: [] for label in pool_labels}

# for pool_idx, idxs in enumerate(pools):
#     for animal_idx in range(n_animals):
#         pooled_speeds = []
#         for win_idx in idxs:
#             win_arr = all_speed[win_idx]  # shape: (n_animals, n_tau, n_timepoints)
#             for tau in range(win_arr.shape[1]):
#                 arr = win_arr[animal_idx, tau, :].astype(float)
#                 arr = arr[~np.isnan(arr)]
#                 if arr.size > 0:
#                     pooled_speeds.append(arr)
#         if pooled_speeds:
#             all_pooled = np.concatenate(pooled_speeds)
#             per_animal_summaries[pool_labels[pool_idx]].append(np.median(all_pooled))  # Use median, or mean, or quantile
#         else:
#             per_animal_summaries[pool_labels[pool_idx]].append(np.nan)

# # Build a DataFrame for downstream analysis
# df_summary = pd.DataFrame({
#     'index_NOR': cog_data_filtered['index_NOR'].values,
#     'genotype': cog_data_filtered['genotype'].values,
#     'treatment': cog_data_filtered['treatment'].values,
#     'dFC_speed_short': per_animal_summaries['short'],
#     'dFC_speed_long': per_animal_summaries['long']
# })

# %%

# short_idx = np.arange(n_windows // 2)
# long_idx = np.arange(n_windows // 2, n_windows)

# all_speeds_short = []
# all_speeds_long = []

# for idxs, pool in zip([short_idx, long_idx], ['short', 'long']):
#     pool_speeds = []
#     for win_idx in idxs:
#         win_arr = all_speed[win_idx]  # (n_animals, n_tau, n_timepoints)
#         for animal_idx in range(win_arr.shape[0]):
#             for tau in range(win_arr.shape[1]):
#                 arr = win_arr[animal_idx, tau, :].astype(float)
#                 arr = arr[~np.isnan(arr)]
#                 if arr.size > 0:
#                     pool_speeds.append(arr)
#     flat = np.concatenate(pool_speeds) if pool_speeds else np.array([])
#     if pool == 'short':
#         all_speeds_short = flat
#     else:
#         all_speeds_long = flat


# plt.figure(figsize=(8,5))
# sns.histplot(all_speeds_short, bins=75, color='royalblue', label='Short windows',
#              stat='density', element='step', fill=False, linewidth=1.7)
# sns.histplot(all_speeds_long, bins=75, color='firebrick', label='Long windows',
#              stat='density', element='step', fill=False, linewidth=1.7)

# # Optional: add median lines
# plt.axvline(np.median(all_speeds_short), color='royalblue', linestyle='--', lw=1)
# plt.axvline(np.median(all_speeds_long), color='firebrick', linestyle='--', lw=1)

# plt.xlabel("dFC Speed")
# plt.ylabel("Density")
# plt.title("Distribution of dFC Speeds: Short vs. Long Window Pools")
# plt.legend()
# plt.tight_layout()
# plt.show()

# %%
# %%

# Assume: groups = df_summary.groupby(['genotype', 'treatment']).groups
#         window_sizes, all_speed, etc. already defined



# data.region_labels_preprocessed[ind_reg]
# %%



# %%

# %%

# # Assume: groups = df_summary.groupby(['genotype', 'treatment']).groups
# #         window_sizes, all_speed, etc. already defined

# short_idx = np.arange(len(window_sizes) // 2)
# long_idx = np.arange(len(window_sizes) // 2, len(window_sizes))
# pool_defs = {'Short windows': short_idx, 'Long windows': long_idx}
# palette = sns.color_palette('tab10', n_colors=len(groups))

# plt.figure(figsize=(12, 6))

# for pool_i, (pool_name, idxs) in enumerate(pool_defs.items()):
#     plt.subplot(1, 2, pool_i+1)
#     for g_idx, (group, animal_idxs) in enumerate(data.groups.items()):


#         # Pool all speeds for this group and this pool
#         group_speeds = []
#         for win_idx in idxs:
#             win_arr = all_speed[win_idx]  # (n_animals, n_tau, n_timepoints)
#             for animal_idx in animal_idxs:
#                 for tau in range(win_arr.shape[1]):
#                     arr = win_arr[animal_idx, tau, :].astype(float)
#                     arr = arr[~np.isnan(arr)]
#                     if arr.size > 0:
#                         group_speeds.append(arr)
#         group_speeds = np.concatenate(group_speeds) if group_speeds else np.array([])
#         # Histogram (step)
#         sns.histplot(group_speeds, bins=60, stat='density', element='step', fill=False,
#                      color=palette[g_idx], linewidth=1.6, label=f'{group}', alpha=0.6)
#         # KDE (over histogram)
#         if group_speeds.size > 10:  # Avoid noise for tiny samples
#             sns.kdeplot(group_speeds, color=palette[g_idx], lw=2.1, label=None)
#         # Median
#         plt.axvline(np.median(group_speeds), color=palette[g_idx], linestyle='--', lw=1)
#     plt.xlabel("dFC Speed")
#     plt.ylabel("Density")
#     plt.yscale('log')  # Log scale for better visibility
#     plt.title(f"{pool_name}")
#     if pool_i == 0:
#         plt.legend(title='Group', fontsize=10)
#     else:
#         plt.legend().set_visible(False)
#     plt.tight_layout()

# plt.suptitle("Distribution of dFC Speeds by Group\nShort vs. Long Window Pools", fontsize=15, y=1.02)
# plt.tight_layout()
# plt.show()


# %%
#----------- Kruskal-Wallis test for long window speeds -----------
# %%

# Prepare data for test (lists of arrays)
data_for_test = [arr for arr in group_speeds_dict.values()]
stat, pval = kruskal(*data_for_test)

print(f"Kruskal–Wallis H = {stat:.3f}, p = {pval:.3g}")


# %%


# Bonferroni correction for multiple comparisons
n_comps = len(group_speeds_dict) * (len(group_speeds_dict)-1) // 2
for g1, g2 in combinations(group_speeds_dict.keys(), 2):
    u, p = mannwhitneyu(group_speeds_dict[g1], group_speeds_dict[g2], alternative='two-sided')
    print(f"{g1} vs {g2}: U = {u:.2g}, uncorrected p = {p:.4f}, Bonferroni-corrected p = {min(p*n_comps,1):.4f}")

# %%

groups = df_summary.groupby(['genotype', 'treatment'])

results = []

for name, subdf in groups:
    for pool in ['short', 'long']:
        # Prepare
        X = subdf[['dFC_speed_' + pool]].copy()
        X = sm.add_constant(X)
        y = subdf['index_NOR']
        mask = (~X.isnull().any(axis=1)) & (~y.isnull())
        X_clean = X.loc[mask]
        y_clean = y.loc[mask]
        if X_clean.shape[0] > 3:  # Avoid crashing with tiny groups
            model = sm.OLS(y_clean, X_clean).fit()
            coef = model.params['dFC_speed_' + pool]
            pval = model.pvalues['dFC_speed_' + pool]
        else:
            coef = np.nan
            pval = np.nan
        results.append({'group': name, 'window': pool, 'coef': coef, 'pval': pval})

df_group_results = pd.DataFrame(results)

# %%
print(df_group_results)


# %%
