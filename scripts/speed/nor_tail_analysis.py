#!/usr/bin/env python

"""
Tail-focused dFC speed vs NOR analysis for 'julien'.

Requirements:
- Existing net_fluidity layout (DATASET_DEFAULTS, get_paths, etc.)
- Precomputed speed files:
    <speed_root>/<subset>/speed_win{w}_lag1_tau2_animals_{n_animals}_regions_{regions}.npz
- Cognitive CSV with 'index_NOR', 'genotype', 'treatment' aligned with timeseries order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from joblib import Memory

import json
import hashlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.formula.api as smf

from scripts.dfc.dfc_compute import DATASET_DEFAULTS, _canonical_dataset
from shared_code.fun_loaddata import load_timeseries_bundle
from shared_code.fun_paths import get_paths
from shared_code.fun_utils import load_cognitive_data, set_figure_params



# =============================================================================
# CONFIG
# =============================================================================

RNG_SEED = 123

# Choose what you consider "primary" for the paper
PRIMARY_SUBSETS = ["sal_within", "dmn_touching"]  # adapt
PRIMARY_METRICS = [
    "speed_q01",
    "speed_q05",
    "speed_median",
    "speed_q95",
    "speed_q99",
    "speed_width50",
    "speed_width_extreme",
    "speed_asymmetry",
]

# Which speed subsets to analyze
SPEED_SUBSETS = [
    "all",
    # "regions500",
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

memory = Memory(location=None, verbose=0)  # stub
_cached_bootstrap = None

# ---------------------------------------------------------------------
# CACHING HELPERS (Option 1)
# ---------------------------------------------------------------------

def make_cache_key(config: dict) -> str:
    """
    Stable hash from a small JSON-serializable config dict.
    This defines when cached metrics/summary can be safely reused.
    """
    cfg_json = json.dumps(config, sort_keys=True, default=str)
    return hashlib.md5(cfg_json.encode("utf-8")).hexdigest()


# =============================================================================
# PHASE 0 – LOAD DATA HELPERS
# =============================================================================


def load_speed_stack_template(
    template: str,
    time_windows_range: Sequence[int],
    n_animals: int,
    regions: int,
) -> list[list[np.ndarray]]:
    """
    Load speed stacks for one subset.

    Parameters
    ----------
    template : str
        Path template with placeholders {w}, {n_animals}, {regions}.
    time_windows_range : sequence of ints
        Window sizes used in the simulation.
    n_animals : int
        Number of animals.
    regions : int
        Number of regions.

    Returns
    -------
    speeds : list of length n_windows
        speeds[j][i] is a 1D np.array of speed samples for animal i at window j.
    """
    speeds: list[list[np.ndarray]] = []
    for w in time_windows_range:
        fname = template.format(w=w, n_animals=n_animals, regions=regions)
        arr = np.load(fname, allow_pickle=True)
        s = arr["speeds"]  # s[i] is array for animal i at this window
        s_flat = [np.asarray(x, dtype=float).ravel() for x in s]
        if len(s_flat) != n_animals:
            raise ValueError(
                f"Expected {n_animals} animals for w={w}, got {len(s_flat)}"
            )
        speeds.append(s_flat)
    return speeds


def build_julien_group_data(cog_data: pd.DataFrame) -> dict[tuple[str, str], list[int]]:
    """
    Build group_data mapping (genotype, treatment) -> list of animal indices.
    Assumes cog_data rows are aligned with timeseries / speeds ordering.
    """
    required_cols = ["genotype", "treatment"]
    missing = [c for c in required_cols if c not in cog_data.columns]
    if missing:
        raise ValueError(f"Missing columns in cog_data: {missing}")

    group_data: dict[tuple[str, str], list[int]] = {}
    for i, row in cog_data.iterrows():
        gt = str(row["genotype"])
        tx = str(row["treatment"])
        key = (gt, tx)
        group_data.setdefault(key, []).append(i)
    return group_data


def count_samples_per_window(speeds: list[np.ndarray]) -> np.ndarray:
    """
    speeds[j] is list of per-animal arrays for window j.
    Return total number of samples per window.
    """
    return np.array([sum(len(x) for x in speed) for speed in speeds], dtype=int)


def cdf_split_indices(speeds: list[np.ndarray]) -> tuple[int, int, int]:
    """
    Use cumulative sample counts to find cut indices for thirds/half.

    Returns
    -------
    i_third, i_half, i_two_third
    """
    counts = count_samples_per_window(speeds)
    if counts.sum() > 0:
        cdf = np.cumsum(counts) / counts.sum()
    else:
        cdf = np.zeros_like(counts, dtype=float)

    i_third = int(np.searchsorted(cdf, 1.0 / 3.0))
    i_half = int(np.searchsorted(cdf, 0.5))
    i_two_third = int(np.searchsorted(cdf, 2.0 / 3.0))

    # a bit of sanity
    i_third = max(1, i_third)
    i_half = max(1, i_half)
    i_two_third = max(i_third + 1, i_two_third)
    return i_third, i_half, i_two_third


def select_windows(
    pool_split: str,
    n_windows: int,
    i_third: int,
    i_half: int,
    i_two_third: int,
) -> dict[str, range]:
    """
    Map segment name -> range of window indices.
    """
    if pool_split == "all":
        return {"all": range(0, n_windows)}
    if pool_split == "half":
        return {
            "short": range(0, i_half),
            "long": range(i_half, n_windows),
        }
    # default: thirds
    return {
        "short": range(0, i_third),
        "mid": range(i_third, i_two_third),
        "long": range(i_two_third, n_windows),
    }


POOL_SPLIT = "third"  # "all" | "half" | "third"

def compute_subset_metrics_with_segments(
    subset: str,
    speeds: Sequence[Sequence[np.ndarray]],
    nor_index: np.ndarray,
    group_data: dict,
    pool_split: str = POOL_SPLIT,
) -> pd.DataFrame:
    """
    Compute per-animal metrics for a single subset across segments,
    returning a tidy DataFrame with 'subset' labels like 'subset__short'.

    This is the per-subset building block used by the Option 2 cache.
    """
    n_windows = len(speeds)
    if n_windows == 0:
        return pd.DataFrame()

    i_third, i_half, i_two_third = cdf_split_indices(list(speeds))
    ranges = select_windows(pool_split, n_windows, i_third, i_half, i_two_third)

    dfs = []
    for seg_name, w_range in ranges.items():
        subset_label = f"{subset}__{seg_name}"
        df_seg = build_subset_metrics_df(
            subset_name=subset_label,
            speeds=speeds,
            nor_index=nor_index,
            group_data=group_data,
            window_indices=w_range,
        )
        dfs.append(df_seg)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def build_metrics_with_segments(
    speeds_by_subset: dict[str, Sequence[Sequence[np.ndarray]]],
    SPEED_SUBSETS: Sequence[str],
    nor_index: np.ndarray,
    group_data: dict,
    pool_split: str = POOL_SPLIT,
) -> pd.DataFrame:
    """
    Build a tidy DataFrame with per-animal metrics for each subset AND
    each window segment.
    """
    dfs = []
    for subset, speeds in speeds_by_subset.items():
        df_sub = compute_subset_metrics_with_segments(
            subset=subset,
            speeds=speeds,
            nor_index=nor_index,
            group_data=group_data,
            pool_split=pool_split,
        )
        if not df_sub.empty:
            dfs.append(df_sub)

    if not dfs:
        raise ValueError("No metrics built — check speeds_by_subset / file availability.")
    return pd.concat(dfs, ignore_index=True)



# =============================================================================
# PHASE 1 – DATA AGGREGATION: speeds[*][i] → pooled per-animal distributions
# =============================================================================


def pool_speeds_per_animal(
    speeds: Sequence[Sequence[np.ndarray]],
    window_indices: Sequence[int] | None = None,
) -> list[np.ndarray]:
    """
    Pool dFC speeds across windows for each animal.

    speeds[j][i] is a 1D array of samples for animal i at window j.
    """
    if window_indices is None:
        window_indices = range(len(speeds))

    first_win = next(iter(window_indices))
    n_animals = len(speeds[first_win])

    pooled = []
    for i in range(n_animals):
        all_samples = [np.ravel(speeds[j][i]) for j in window_indices]
        pooled.append(np.concatenate(all_samples))
    return pooled


# =============================================================================
# PHASE 2 – METRIC EXTRACTION: MEDIAN, TAIL QUANTILES, WIDTH
# =============================================================================


@dataclass
class SpeedMetrics:
    # basic quantiles
    q01: float
    q05: float
    q50: float
    q90: float
    q95: float
    q99: float

    # derived “tail shape” metrics
    width50: float  # q95 - q05 (old width)
    width_extreme: float  # q99 - q01
    asymmetry: float  # (q95 - q50) - (q50 - q05)


def compute_speed_metrics(samples: np.ndarray) -> SpeedMetrics:
    """Compute central and tail metrics for a 1D array of dFC speeds."""
    q01, q05, q50, q90, q95, q99 = np.percentile(samples, [1, 5, 50, 90, 95, 99])

    width50 = q95 - q05
    width_extreme = q99 - q01
    asymmetry = (q95 - q50) - (q50 - q05)

    return SpeedMetrics(
        q01=q01,
        q05=q05,
        q50=q50,
        q90=q90,
        q95=q95,
        q99=q99,
        width50=width50,
        width_extreme=width_extreme,
        asymmetry=asymmetry,
    )


def infer_animal_group_labels(
    group_data: dict,
    n_animals: int,
) -> tuple[list[str], list[str | None], list[str | None]]:
    """
    From group_data (group_label -> list of animal indices),
    build per-animal group labels and, for ('WT','VEH') tuples,
    genotype / treatment columns.
    """
    group_labels = [None] * n_animals
    genotypes = [None] * n_animals
    treatments = [None] * n_animals

    for gkey, idxs in group_data.items():
        if isinstance(gkey, tuple):
            label = "_".join(map(str, gkey))  # ("WT","VEH") → "WT_VEH"
        else:
            label = str(gkey)

        for i in idxs:
            if group_labels[i] is not None:
                raise ValueError(f"Animal {i} assigned to multiple groups")
            group_labels[i] = label

            if isinstance(gkey, tuple) and len(gkey) == 2:
                genotypes[i] = str(gkey[0])
                treatments[i] = str(gkey[1])

    for i in range(n_animals):
        if group_labels[i] is None:
            group_labels[i] = "UNASSIGNED"

    return group_labels, genotypes, treatments


def build_subset_metrics_df(
    subset_name: str,
    speeds: Sequence[Sequence[np.ndarray]],
    nor_index: np.ndarray,
    group_data: dict,
    window_indices: Sequence[int] | None = None,
) -> pd.DataFrame:
    """
    Build a tidy DataFrame of per-animal speed metrics for one subset.
    """
    pooled = pool_speeds_per_animal(speeds, window_indices)
    n_animals = len(pooled)

    if len(nor_index) != n_animals:
        raise ValueError("nor_index length does not match number of animals")

    group_labels, genotypes, treatments = infer_animal_group_labels(
        group_data, n_animals
    )

    rows = []
    for i in range(n_animals):
        metrics = compute_speed_metrics(pooled[i])
        rows.append(
            {
                "animal_id": i,
                "group": group_labels[i],
                "genotype": genotypes[i],
                "treatment": treatments[i],
                "subset": subset_name,
                "nor": float(nor_index[i]),
                # raw quantiles
                "speed_q01": metrics.q01,
                "speed_q05": metrics.q05,
                "speed_median": metrics.q50,
                "speed_q90": metrics.q90,
                "speed_q95": metrics.q95,
                "speed_q99": metrics.q99,
                # tail-shape metrics
                "speed_width": metrics.width50,
                "speed_width50": metrics.width50,  # q95 - q05 (old width)
                "speed_width_extreme": metrics.width_extreme,  # q99 - q01
                "speed_asymmetry": metrics.asymmetry,
            }
        )
    return pd.DataFrame(rows)


# or whatever subset you want to test


def build_full_metrics_df(
    speeds_by_subset: dict[str, Sequence[Sequence[np.ndarray]]],
    SPEED_SUBSETS: Sequence[str],
    nor_index: np.ndarray,
    group_data: dict,
    window_indices_by_subset: dict[str, Sequence[int]] | None = None,
) -> pd.DataFrame:
    """Loop over all subsets and concatenate the per-animal metrics."""
    dfs = []
    for subset in SPEED_SUBSETS:
        if subset not in speeds_by_subset:
            continue
        speeds = speeds_by_subset[subset]
        win_idx = None
        if window_indices_by_subset is not None:
            win_idx = window_indices_by_subset.get(subset, None)

        df_subset = build_subset_metrics_df(
            subset_name=subset,
            speeds=speeds,
            nor_index=nor_index,
            group_data=group_data,
            window_indices=win_idx,
        )
        dfs.append(df_subset)

    if not dfs:
        raise ValueError("No subsets found in speeds_by_subset")
    return pd.concat(dfs, ignore_index=True)


# =============================================================================
# PHASE 3 – WITHIN-GROUP CORRELATIONS (SPEARMAN + BOOTSTRAP CIs)
# =============================================================================


def _bootstrap_spearman_pure(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int,
    ci_low: float,
    ci_high: float,
    random_state: int | None,
) -> dict:
    rng = np.random.default_rng(random_state)
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    n = len(x)
    rho_obs, _ = spearmanr(x, y)

    boot_rhos = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_rhos[b], _ = spearmanr(x[idx], y[idx])

    lower, upper = np.percentile(boot_rhos, [ci_low, ci_high])
    return {
        "rho": float(rho_obs),
        "rho_boot_mean": float(boot_rhos.mean()),
        "ci_low": float(lower),
        "ci_high": float(upper),
    }


def bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 2000,
    ci: tuple[float, float] = (2.5, 97.5),
    random_state: int | None = None,
) -> dict:
    """
    Thin wrapper using joblib cache if memory.location is set,
    otherwise falls back to pure computation.
    """
    global _cached_bootstrap

    # Decide which function to call
    if memory.location is not None:
        # Initialize the cached function once
        if _cached_bootstrap is None:
            _cached_bootstrap = memory.cache(_bootstrap_spearman_pure)
        func = _cached_bootstrap
    else:
        func = _bootstrap_spearman_pure

    return func(
        np.asarray(x, float),
        np.asarray(y, float),
        int(n_boot),
        float(ci[0]),
        float(ci[1]),
        random_state,
    )

def compute_within_group_correlations(
    df: pd.DataFrame,
    subset: str,
    metrics: Sequence[str] = ("speed_median", "speed_q95"),
    n_boot: int = 2000,
    random_state: int | None = RNG_SEED,
) -> pd.DataFrame:
    """Spearman + bootstrap CI for NOR vs metric within each group."""
    df_sub = df[df["subset"] == subset].copy()

    results = []
    for group, df_g in df_sub.groupby("group"):
        if len(df_g) < 3:
            continue

        y = df_g["nor"].to_numpy()
        for metric in metrics:
            x = df_g[metric].to_numpy()
            stats = bootstrap_spearman(x, y, n_boot=n_boot, random_state=random_state)
            results.append(
                {
                    "subset": subset,
                    "group": group,
                    "metric": metric,
                    "n": len(df_g),
                    **stats,
                }
            )

    return pd.DataFrame(results)


# =============================================================================
# PHASE 4 – BETWEEN-GROUP INTERACTION: NOR ~ metric * group
# =============================================================================


def fit_speed_nor_interaction(
    df: pd.DataFrame,
    subset: str,
    metric: str = "speed_q95",
    ref_group: str = "WT_VEH",
):
    """
    Fit linear model:
        NOR ~ metric * group
    with 'group' as categorical and ref_group as reference.
    """
    df_sub = df[df["subset"] == subset].copy()

    df_sub["group"] = pd.Categorical(
        df_sub["group"],
        categories=sorted(df_sub["group"].unique()),
    )
    if ref_group not in df_sub["group"].cat.categories:
        raise ValueError(f"Reference group {ref_group!r} not in data groups")

    formula = f"nor ~ {metric} * C(group, Treatment(reference='{ref_group}'))"
    model = smf.ols(formula, data=df_sub).fit()

    params = model.params
    cov = model.cov_params()

    groups = list(df_sub["group"].cat.categories)
    slope_rows = []

    for g in groups:
        if g == ref_group:
            param_name = metric
            L = np.zeros(len(params))
            L[params.index.get_loc(param_name)] = 1.0
        else:
            param_main = metric
            param_int = f"{metric}:C(group, Treatment(reference='{ref_group}'))[T.{g}]"

            L = np.zeros(len(params))
            L[params.index.get_loc(param_main)] = 1.0
            if param_int in params.index:
                L[params.index.get_loc(param_int)] = 1.0

        slope = float(L @ params.to_numpy())
        var = float(L @ cov.to_numpy() @ L)
        se = float(np.sqrt(max(var, 0.0)))
        ci_low = slope - 1.96 * se
        ci_high = slope + 1.96 * se

        slope_rows.append(
            {
                "subset": subset,
                "metric": metric,
                "group": g,
                "slope": slope,
                "se": se,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ref_group": ref_group,
            }
        )

    slopes_df = pd.DataFrame(slope_rows)
    return model, slopes_df


# =============================================================================
# PHASE 5 – PLOTTING
# =============================================================================


def plot_nor_vs_metric_by_group(
    df: pd.DataFrame,
    subset: str,
    metric: str = "speed_q95",
    model=None,
    ax: plt.Axes | None = None,
):
    """Scatter NOR vs metric, colored by group, plus regression lines."""
    df_sub = df[df["subset"] == subset].copy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    groups = sorted(df_sub["group"].unique())
    for g in groups:
        dfg = df_sub[df_sub["group"] == g]
        ax.scatter(
            dfg[metric],
            dfg["nor"],
            label=g,
            alpha=0.8,
            edgecolor="none",
        )

    if model is None:
        model, _ = fit_speed_nor_interaction(
            df, subset=subset, metric=metric, ref_group="WT_VEH"
        )

    xs = np.linspace(df_sub[metric].min(), df_sub[metric].max(), 100)
    for g in groups:
        df_pred = pd.DataFrame({metric: xs, "group": g})
        y_pred = model.predict(df_pred)
        ax.plot(xs, y_pred, alpha=0.9)

    ax.set_xlabel(metric)
    ax.set_ylabel("NOR index")
    ax.set_title(f"NOR vs {metric} – subset: {subset}")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


def plot_group_slopes(
    slopes_df: pd.DataFrame,
    subset: str,
    metric: str = "speed_q95",
    ax: plt.Axes | None = None,
):
    """Point + errorbar plot of slopes per group with 95% CI."""
    df_sub = slopes_df[
        (slopes_df["subset"] == subset) & (slopes_df["metric"] == metric)
    ].copy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.figure

    groups = df_sub["group"].tolist()
    x = np.arange(len(groups))
    y = df_sub["slope"].to_numpy()
    yerr = np.vstack(
        [y - df_sub["ci_low"].to_numpy(), df_sub["ci_high"].to_numpy() - y]
    )

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        capsize=3,
        linestyle="none",
    )
    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_ylabel(f"Slope NOR vs {metric}")
    ax.set_title(f"Group slopes – subset: {subset}")
    fig.tight_layout()
    return fig, ax


# =============================================================================
# PHASE 6 – HIGH-LEVEL DRIVER
# =============================================================================


def run_primary_analysis(
    speeds_by_subset: dict[str, Sequence[Sequence[np.ndarray]]],
    SPEED_SUBSETS: Sequence[str],
    nor_index: np.ndarray,
    group_data: dict,
    primary_subsets: Sequence[str] | None = None,
    primary_metrics: Sequence[str] | None = PRIMARY_METRICS,
    pool_split: str = POOL_SPLIT,
):
    """
    Build df_metrics (with segments), then compute:

      - within-group Spearman correlations (with bootstrap CIs)
      - group-wise interaction slopes NOR ~ metric * group

    for *all* subsets and *all* metrics.

    No plotting here; plotting is handled in __main__ after caching.
    """
    # Build metrics with segments (short/mid/long)
    df_metrics = build_metrics_with_segments(
        speeds_by_subset=speeds_by_subset,
        SPEED_SUBSETS=SPEED_SUBSETS,
        nor_index=nor_index,
        group_data=group_data,
        pool_split=pool_split,
    )

    # Ensure subset_base / segment columns exist
    if "subset_base" not in df_metrics.columns or "segment" not in df_metrics.columns:
        base_list = []
        seg_list = []
        for s in df_metrics["subset"]:
            if "__" in s:
                base, seg = s.rsplit("__", 1)
            else:
                base, seg = s, "all"
            base_list.append(base)
            seg_list.append(seg)
        df_metrics["subset_base"] = base_list
        df_metrics["segment"] = seg_list

    # If primary_subsets not given, use ALL subset labels present
    if primary_subsets is None:
        primary_subsets = sorted(df_metrics["subset"].unique())

    # If primary_metrics not given, use all speed_* columns
    if primary_metrics is None:
        primary_metrics = [
            c for c in df_metrics.columns
            if c.startswith("speed_")
        ]

    all_corr: list[pd.DataFrame] = []
    all_slopes: list[pd.DataFrame] = []

    for subset in primary_subsets:
        if subset not in df_metrics["subset"].unique():
            print(f"[WARN] subset {subset} not present in df_metrics; skipping.")
            continue

        df_sub = df_metrics[df_metrics["subset"] == subset]
        groups_here = sorted(df_sub["group"].unique())
        if "WT_VEH" not in groups_here:
            print(
                f"[WARN] ref_group WT_VEH not present in {subset}; "
                f"groups={groups_here}. Skipping interactions for this subset."
            )
            continue

        # 1) within-group correlations for ALL metrics
        corr_df = compute_within_group_correlations(
            df_metrics,
            subset=subset,
            metrics=primary_metrics,
            n_boot=2000,
            random_state=RNG_SEED,
        )
        if not corr_df.empty:
            all_corr.append(corr_df)

        # 2) interaction models for ALL metrics
        for metric in primary_metrics:
            if metric not in df_metrics.columns:
                print(f"[WARN] metric {metric} not in df_metrics; skipping.")
                continue

            print(f"\n=== Interaction: subset={subset}, metric={metric} ===")
            model, slopes_df = fit_speed_nor_interaction(
                df_metrics,
                subset=subset,
                metric=metric,
                ref_group="WT_VEH",
            )
            slopes_df["subset_base"], slopes_df["segment"] = subset.split("__")
            all_slopes.append(slopes_df)

    corr_summary = pd.concat(all_corr, ignore_index=True) if all_corr else None
    slopes_summary = pd.concat(all_slopes, ignore_index=True) if all_slopes else None
    return df_metrics, corr_summary, slopes_summary


def add_subset_segment_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    From subset labels like 'sal_within__short', create:
      - subset_base: 'sal_within'
      - segment: 'short'

    If there is no '__' in the subset, segment='all'.
    Works in-place and also returns the df for convenience.
    """
    base_list = []
    seg_list = []

    for s in df["subset"]:
        if "__" in s:
            base, seg = s.rsplit("__", 1)
        else:
            base, seg = s, "all"
        base_list.append(base)
        seg_list.append(seg)

    df["subset_base"] = base_list
    df["segment"] = seg_list
    return df


def fit_segment_group_interaction(
    df: pd.DataFrame,
    subset_base: str,
    metric: str = "speed_q95",
    ref_group: str = "WT_VEH",
    ref_segment: str = "mid",
):
    """
    Fit a 3-way interaction model for a given base subset:

        NOR ~ metric * group * segment

    with specified reference group and reference segment.

    Returns:
        model  (statsmodels OLS result)
    """
    # Ensure we have the helper columns
    if "subset_base" not in df.columns or "segment" not in df.columns:
        df = add_subset_segment_columns(df)

    df_sub = df[df["subset_base"] == subset_base].copy()
    if df_sub.empty:
        raise ValueError(f"No rows for subset_base={subset_base!r}")

    # Categorical coding with references
    df_sub["group"] = pd.Categorical(
        df_sub["group"],
        categories=sorted(df_sub["group"].unique()),
    )
    df_sub["segment"] = pd.Categorical(
        df_sub["segment"],
        categories=sorted(df_sub["segment"].unique()),
    )

    if ref_group not in df_sub["group"].cat.categories:
        raise ValueError(f"Reference group {ref_group!r} not in data groups")
    if ref_segment not in df_sub["segment"].cat.categories:
        raise ValueError(f"Reference segment {ref_segment!r} not in segments")

    formula = (
        f"nor ~ {metric} * "
        f"C(group, Treatment(reference='{ref_group}')) * "
        f"C(segment, Treatment(reference='{ref_segment}'))"
    )

    model = smf.ols(formula, data=df_sub).fit()
    return model


def leave_one_out_slopes(
    df: pd.DataFrame,
    subset: str,
    metric: str = "speed_q95",
    ref_group: str = "WT_VEH",
):
    df_sub = df[df["subset"] == subset].copy()
    if df_sub.empty:
        raise ValueError(f"No rows for subset={subset!r}")

    animal_ids = sorted(df_sub["animal_id"].unique())
    rows = []

    for aid in animal_ids:
        df_loo = df_sub[df_sub["animal_id"] != aid].copy()
        # use the LOO dataframe here
        model, slopes_df = fit_speed_nor_interaction(
            df_loo,
            subset=subset,
            metric=metric,
            ref_group=ref_group,
        )
        for _, r in slopes_df.iterrows():
            rows.append(
                {
                    "subset": subset,
                    "metric": metric,
                    "group": r["group"],
                    "animal_id": aid,
                    "slope": r["slope"],
                    "se": r["se"],
                    "ci_low": r["ci_low"],
                    "ci_high": r["ci_high"],
                }
            )

    loo_df = pd.DataFrame(rows)
    return loo_df

def plot_multi_segment_scatter_row(
    df: pd.DataFrame,
    subset_base: str,
    metric: str = "speed_q95",
    ref_group: str = "WT_VEH",
    segments: Sequence[str] = ("short", "mid", "long"),
):
    """
    For a given base subset (e.g. 'sal_within'), make a row of panels
    (short, mid, long), each being NOR vs metric, colored by group with
    regression lines from the interaction model.

    Returns (fig, axes).
    """
    fig, axes = plt.subplots(
        1, len(segments), figsize=(5 * len(segments), 4), sharey=True
    )

    if len(segments) == 1:
        axes = [axes]

    for ax, seg in zip(axes, segments, strict=False):
        subset_label = f"{subset_base}__{seg}"
        if subset_label not in df["subset"].unique():
            ax.set_visible(False)
            continue

        # fit model for this subset/segment
        model, _ = fit_speed_nor_interaction(
            df, subset=subset_label, metric=metric, ref_group=ref_group
        )

        plot_nor_vs_metric_by_group(
            df,
            subset=subset_label,
            metric=metric,
            model=model,
            ax=ax,
        )
        ax.set_title(f"{subset_base} – {seg}")

    fig.suptitle(f"NOR vs {metric} by segment – {subset_base}", y=1.02)
    fig.tight_layout()
    return fig, axes


# =============================================================================
# MAIN – REAL JULIEN PIPELINE
# =============================================================================

if __name__ == "__main__":
    dataset_name = "julien"
    dataset = _canonical_dataset(dataset_name)
    cfg = DATASET_DEFAULTS[dataset]
    save_fig = set_figure_params(True)

    # Time windows must match the speed files you already generated
    time_windows_range = np.arange(5, 100, 1)

    # Paths
    paths = get_paths(
        dataset_name=dataset,
        timecourse_folder=cfg["timecourse_folder"],
        cognitive_data_file=cfg["cognitive_data_file"],
        anat_labels_file=cfg["anat_labels_file"],
    )
    speed_root = Path(paths["speed"])
    preprocessed_root = Path(paths["preprocessed"])

    # Load ts bundle to know n_animals, regions, total_tr
    loaddir_ts_meta = preprocessed_root / "ts_and_meta_2m4m.npz"
    bundle = load_timeseries_bundle(loaddir_ts_meta)
    n_animals = bundle.n_animals
    regions = bundle.n_regions
    total_tr = bundle.total_tr

    # Load cognitive data
    loaddir_cog_data = preprocessed_root / (
        "cog_data_filtered_animals_{n_animals}_regions_{regions}_tr_{total_tr}.csv"
    )
    cog_data = load_cognitive_data(
        str(loaddir_cog_data).format(
            n_animals=n_animals, regions=regions, total_tr=total_tr
        )
    )

    # NOR index and group_data
    if "index_NOR" not in cog_data.columns:
        raise ValueError("cog_data is missing 'index_NOR' column")
    nor_index = cog_data["index_NOR"].to_numpy()
    group_data = build_julien_group_data(cog_data)

    # Build speeds_by_subset in the **correct shape**: [window][animal] of 1D arrays
    speeds_by_subset: dict[str, list[list[np.ndarray]]] = {}

    speed_template = str(
        speed_root
        / "{subset}/speed_win{w}_lag1_tau2_animals_{n_animals}_regions_{regions}.npz"
    )

    for subset in SPEED_SUBSETS:
        template = speed_template.format(
            subset=subset, n_animals=n_animals, regions=regions, w="{w}"
        )
        try:
            speeds = load_speed_stack_template(
                template=template,
                time_windows_range=time_windows_range,
                n_animals=n_animals,
                regions=regions,
            )
        except FileNotFoundError as e:
            print(f"[WARN] Skipping subset {subset}: {e}")
            continue
        speeds_by_subset[subset] = speeds
        print(f"[INFO] Loaded speeds for subset '{subset}' with {len(speeds)} windows.")


    all_subset_labels = list(speeds_by_subset.keys())
    print(f"[INFO] Available subsets: {all_subset_labels}")
    # ---------------- Option 1: global cache for df_metrics & summaries ----------------
    cache_root = Path(paths["f_speed"]) / "nor_tail_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    # Option 3: joblib cache directory (keep this on a local/ext4 disk!)
    joblib_cache_root = Path("~/.cache/net_fluidity_nor_bootstrap").expanduser()
    joblib_cache_root.mkdir(parents=True, exist_ok=True)

    memory = Memory(location=joblib_cache_root, verbose=0)


    cache_config = {
        "dataset": dataset_name,
        "pool_split": POOL_SPLIT,
        "time_windows": [int(w) for w in time_windows_range],
        "subsets": sorted(all_subset_labels),
        "primary_metrics": PRIMARY_METRICS,
        "rng_seed": RNG_SEED,
    }

    cache_key = make_cache_key(cache_config)

    cache_df = cache_root / f"df_metrics_{cache_key}.parquet"
    cache_corr = cache_root / f"corr_summary_{cache_key}.parquet"
    cache_slopes = cache_root / f"slopes_summary_{cache_key}.parquet"
    cache_cfg_path = cache_root / f"config_{cache_key}.json"


    # ---------------- Option 0: per-subset metrics cache ----------------
    subset_cache_root = cache_root / "per_subset_metrics"
    subset_cache_root.mkdir(parents=True, exist_ok=True)

    per_subset_dfs = []

    for subset in all_subset_labels:
        subset_speeds = speeds_by_subset[subset]
        subset_cfg = {
            "dataset": dataset_name,
            "subset": subset,
            "pool_split": POOL_SPLIT,
            "time_windows": [int(w) for w in time_windows_range],
            "rng_seed": RNG_SEED,  # if you ever make metrics stochastic
        }
        subset_key = make_cache_key(subset_cfg)
        subset_path = subset_cache_root / f"metrics_{subset_key}.parquet"

        if subset_path.exists():
            print(f"[INFO] Loading cached metrics for subset '{subset}'")
            df_sub = pd.read_parquet(subset_path)
        else:
            print(f"[INFO] Computing metrics for subset '{subset}'")
            df_sub = compute_subset_metrics_with_segments(
                subset=subset,
                speeds=subset_speeds,
                nor_index=nor_index,
                group_data=group_data,
                pool_split=POOL_SPLIT,
            )
            if not df_sub.empty:
                df_sub.to_parquet(subset_path, index=False)

        if not df_sub.empty:
            per_subset_dfs.append(df_sub)

    if not per_subset_dfs:
        raise ValueError("No subset metrics available – check speeds_by_subset")

    df_metrics = pd.concat(per_subset_dfs, ignore_index=True)
    df_metrics = add_subset_segment_columns(df_metrics)
    print(f"[INFO] Built full df_metrics with {len(df_metrics)} rows from per-subset cache.")


    # ---------------- Global cache for summaries only (Option 1) ----------------
    cache_config = {
        "dataset": dataset_name,
        "pool_split": POOL_SPLIT,
        "time_windows": [int(w) for w in time_windows_range],
        "subsets": sorted(all_subset_labels),
        "primary_metrics": PRIMARY_METRICS,
        "rng_seed": RNG_SEED,
    }

    cache_key = make_cache_key(cache_config)
    cache_corr = cache_root / f"corr_summary_{cache_key}.parquet"
    cache_slopes = cache_root / f"slopes_summary_{cache_key}.parquet"
    cache_cfg_path = cache_root / f"config_{cache_key}.json"

    corr_summary = None
    slopes_summary = None

    if cache_corr.exists() and cache_slopes.exists():
        print(f"[INFO] Loading cached summaries from {cache_root}")
        corr_summary = pd.read_parquet(cache_corr)
        slopes_summary = pd.read_parquet(cache_slopes)

        if cache_cfg_path.exists():
            with cache_cfg_path.open("r") as f:
                stored_cfg = json.load(f)
            if stored_cfg != cache_config:
                print("[WARN] Cache config mismatch – forcing recompute of summaries")
                corr_summary = slopes_summary = None
    else:
        print("[INFO] No summary cache found – computing")

    if corr_summary is None or slopes_summary is None:
        # reuse existing df_metrics; run_primary_analysis will rebuild its own df_metrics,
        # but we only care about corr_summary and slopes_summary it returns
        _, corr_summary, slopes_summary = run_primary_analysis(
            speeds_by_subset=speeds_by_subset,
            SPEED_SUBSETS=all_subset_labels,
            nor_index=nor_index,
            group_data=group_data,
            primary_subsets=None,
            primary_metrics=PRIMARY_METRICS,
            pool_split=POOL_SPLIT,
        )

        if corr_summary is not None:
            corr_summary.to_parquet(cache_corr, index=False)
        if slopes_summary is not None:
            slopes_summary.to_parquet(cache_slopes, index=False)

        with cache_cfg_path.open("w") as f:
            json.dump(cache_config, f, indent=2)
    else:
        print("[INFO] Using cached corr_summary / slopes_summary")


    # ---------------- Save plots for ALL subsets × ALL metrics ----------------
    plots_root = cache_root / "plots_all_metrics"
    scatter_dir = plots_root / "scatter"
    slopes_dir = plots_root / "slopes"
    scatter_dir.mkdir(parents=True, exist_ok=True)
    slopes_dir.mkdir(parents=True, exist_ok=True)

    # Loop over each subset label (e.g., 'sal_within__short', 'dmn_touching__mid', ...)
    for subset in sorted(df_metrics["subset"].unique()):
        df_sub = df_metrics[df_metrics["subset"] == subset]
        groups_here = sorted(df_sub["group"].unique())
        if "WT_VEH" not in groups_here:
            print(
                f"[WARN] ref_group WT_VEH not in subset {subset}; "
                f"groups={groups_here}. Skipping plots for this subset."
            )
            continue

        for metric in PRIMARY_METRICS:
            if metric not in df_metrics.columns:
                print(f"[WARN] metric {metric} not in df_metrics; skipping {subset}.")
                continue

            print(f"[PLOT] subset={subset}, metric={metric}")

            # Fit interaction model once (used for both scatter and slopes plots)
            model, slopes_df = fit_speed_nor_interaction(
                df_metrics,
                subset=subset,
                metric=metric,
                ref_group="WT_VEH",
            )

            # 1) Scatter + regression lines
            fig_scatter, _ = plot_nor_vs_metric_by_group(
                df_metrics,
                subset=subset,
                metric=metric,
                model=model,
            )
            scatter_path = scatter_dir / f"scatter_{subset}_{metric}.png"
            fig_scatter.savefig(scatter_path, dpi=300, bbox_inches="tight")
            plt.close(fig_scatter)

            # 2) Group slopes + 95% CI
            fig_slopes, _ = plot_group_slopes(
                slopes_df,
                subset=subset,
                metric=metric,
            )
            slopes_path = slopes_dir / f"slopes_{subset}_{metric}.png"
            fig_slopes.savefig(slopes_path, dpi=300, bbox_inches="tight")
            plt.close(fig_slopes)


    # Example: salience within
    fig_sal, _ = plot_multi_segment_scatter_row(
        df_metrics,
        subset_base="sal_within",
        metric="speed_q95",
        ref_group="WT_VEH",
    )

    # Example: DMN-touching
    fig_dmn, _ = plot_multi_segment_scatter_row(
        df_metrics,
        subset_base="dmn_touching",
        metric="speed_q95",
        ref_group="WT_VEH",
    )

    # Segment × group interaction for salience within
    model_sal_seg = fit_segment_group_interaction(
        df_metrics,
        subset_base="sal_within",
        metric="speed_q95",
        ref_group="WT_VEH",
        ref_segment="mid",
    )

    # Segment × group interaction for DMN-touching
    model_dmn_seg = fit_segment_group_interaction(
        df_metrics,
        subset_base="dmn_touching",
        metric="speed_q95",
        ref_group="WT_VEH",
        ref_segment="mid",
    )

    print(model_sal_seg.summary())
    print(model_dmn_seg.summary())

    loo_sal_mid = leave_one_out_slopes(
        df_metrics,
        subset="sal_within__mid",
        metric="speed_q95",
        ref_group="WT_VEH",
    )

    loo_dmn_long = leave_one_out_slopes(
        df_metrics,
        subset="dmn_touching__long",
        metric="speed_q95",
        ref_group="WT_VEH",
    )

    print("\nLOO slopes – sal_within__mid")
    print(loo_sal_mid.head())

    print("\nLOO slopes – dmn_touching__long")
    print(loo_dmn_long.head())

    print("\n=== METRICS (HEAD) ===")
    print(df_metrics.head())

    print("\n=== WITHIN-GROUP CORRELATIONS ===")
    print(corr_summary)

    print("\n=== GROUP SLOPES ===")
    print(slopes_summary)

    plt.show()
