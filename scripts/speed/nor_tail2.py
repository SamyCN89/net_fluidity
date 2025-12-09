# %%
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

from glob import glob

import json
import hashlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as student_t
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from scripts.dfc.dfc_compute import DATASET_DEFAULTS, _canonical_dataset
from shared_code.fun_loaddata import load_timeseries_bundle
from shared_code.fun_paths import get_paths
from shared_code.fun_utils import load_cognitive_data, set_figure_params


# =============================================================================
# CONFIG
# =============================================================================

RNG_SEED = 123

# Tail / shape metrics we systematically use
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

POOL_SPLIT = "third"  # "all" | "half" | "third"

# global joblib Memory (overwritten in main when cache dir is known)
memory = Memory(location=None, verbose=0)
_cached_bootstrap = None


# ---------------------------------------------------------------------
# CACHING HELPERS
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

# =============================================================================
# PER-REGION SUPPORT
# =============================================================================


def add_per_region_speeds(
    speeds_by_subset: dict[str, list[list[np.ndarray]]],
    speed_root: Path,
    time_windows_range: Sequence[int],
    n_animals: int,
    regions: int,
) -> dict[str, list[list[np.ndarray]]]:
    """
    Discover all per-region descriptors and append them as extra 'subsets'
    in speeds_by_subset, so the rest of the pipeline treats each region
    exactly like any other subset.

    We assume per-region files live under:
        speed_root / "per_region" / <region_desc> /
            speed_win{w}_lag1_tau2_animals_{n_animals}_regions_{regions}.npz

    Adapt the template below to match your actual file naming if needed.
    """
    subset_dir = speed_root / "per_region"
    if not subset_dir.exists():
        print(f"[WARN] per_region directory not found at {subset_dir}; skipping per-region analysis.")
        return speeds_by_subset

    region_descriptors = discover_per_region_descriptors(
        subset_dir=subset_dir,
        w0=int(time_windows_range[0]),
        n_animals=n_animals,
        regions=regions,
        lag=1,        # must match dfc_speed_compute
        tau_count=2,  # TAU_RANGE=0,4 → 2 tau values, adjust if different
    )

    print(f"[per_region] Found {len(region_descriptors)} regions: {region_descriptors}")

    for region_desc in region_descriptors:
        print(f"[per_region] Found {len(region_descriptors)} regions: {region_descriptors}")

        for region_desc in region_descriptors:
            # Files live directly under subset_dir:
            #   speed_win{w}_lag1_tau2_animals_{n_animals}_regions_{regions}_{region_desc}.npz
            template = str(
                subset_dir
                / f"speed_win{{w}}_lag1_tau2_animals_{n_animals}_regions_{regions}_{region_desc}.npz"
            )

            try:
                speeds = load_speed_stack_template(
                    template=template,
                    time_windows_range=time_windows_range,
                    n_animals=n_animals,
                    regions=regions,
                )
            except FileNotFoundError as e:
                print(f"[WARN] Skipping per_region {region_desc}: {e}")
                continue

            # Optional: cleaner subset name without "region-"
            clean_desc = region_desc.replace("region-", "")
            subset_name = f"per_region_{clean_desc}"

            speeds_by_subset[subset_name] = speeds
            print(f"[INFO] Loaded per_region speeds for '{subset_name}' with {len(speeds)} windows.")

    return speeds_by_subset


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
    rho_obs, p_obs = spearmanr(x, y)

    boot_rhos = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_rhos[b], _ = spearmanr(x[idx], y[idx])

    lower, upper = np.percentile(boot_rhos, [ci_low, ci_high])
    return {
        "rho": float(rho_obs),
        "p_value": float(p_obs),
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
            if metric not in df_g.columns:
                continue
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

    Returns
    -------
    model : statsmodels OLS result
    slopes_df : DataFrame with per-group slopes, CIs, p-values.
    """
    df_sub = df[df["subset"] == subset].copy()
    if df_sub.empty:
        raise ValueError(f"No rows for subset={subset!r}")

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
    df_resid = model.df_resid

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

        if se > 0 and np.isfinite(se):
            t_val = slope / se
            p_value = 2 * student_t.sf(abs(t_val), df_resid)
        else:
            p_value = np.nan

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
                "df_resid": df_resid,
                "p_value": p_value,
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
    if df_sub.empty:
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))
        else:
            fig = ax.figure
        ax.set_visible(False)
        return fig, ax

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


def _p_to_star(p: float) -> str:
    """Convert p-value to significance star string."""
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def plot_group_slopes(
    slopes_df: pd.DataFrame,
    subset: str,
    metric: str = "speed_q95",
    ax: plt.Axes | None = None,
):
    """
    Point + errorbar plot of slopes per group with 95% CI + significance stars.
    Stars are based on per-group p-values stored in slopes_df['p_value'].
    """
    df_sub = slopes_df[
        (slopes_df["subset"] == subset) & (slopes_df["metric"] == metric)
    ].copy()

    if df_sub.empty:
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))
        else:
            fig = ax.figure
        ax.set_visible(False)
        return fig, ax

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

    # stars based on p_value (if present)
    if "p_value" in df_sub.columns:
        stars = [_p_to_star(p) for p in df_sub["p_value"].to_numpy()]
    else:
        stars = ["" for _ in groups]

    # vertical offset for star placement
    y_min = np.nanmin(y - yerr[0])
    y_max = np.nanmax(y + yerr[1])
    span = y_max - y_min if np.isfinite(y_max - y_min) and (y_max - y_min) > 0 else 1.0
    offset = 0.05 * span

    for xi, yi, star in zip(x, y, stars):
        if star:
            ax.text(
                xi,
                yi + offset,
                star,
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_ylabel(f"Slope NOR vs {metric}")
    ax.set_title(f"Group slopes – subset: {subset}")
    fig.tight_layout()
    return fig, ax


# =============================================================================
# PHASE 6 – HIGH-LEVEL DRIVER
# =============================================================================


def run_primary_analysis_from_df(
    df_metrics: pd.DataFrame,
    primary_subsets: Sequence[str] | None = None,
    primary_metrics: Sequence[str] = PRIMARY_METRICS,
    save_plots: bool = False,
    fig_root: Path | None = None,
):
    """
    High-level analysis starting from an existing df_metrics that already
    contains:
      - one row per animal × subset__segment
      - columns: 'subset', 'group', 'nor', PRIMARY_METRICS, etc.
      - subset labels like 'sal_within__short'
      - 'subset_base' and 'segment' columns (if not present, they will be added)

    Computes, for all subset__segment:
      - within-group correlations (Spearman + bootstrap CI)
      - group-wise slopes (NOR ~ metric * group, per metric)
      - (optional) scatter + slope plots with significance stars
    """
    # Make sure we have subset_base / segment
    if "subset_base" not in df_metrics.columns or "segment" not in df_metrics.columns:
        df_metrics = add_subset_segment_columns(df_metrics)

    # Default = all subset labels present
    if primary_subsets is None:
        primary_subsets = sorted(df_metrics["subset"].unique())

    all_corr = []
    all_slopes = []

    scatter_dir = None
    slopes_dir = None
    if save_plots and fig_root is not None:
        scatter_dir = fig_root / "scatter"
        slopes_dir = fig_root / "slopes"
        scatter_dir.mkdir(parents=True, exist_ok=True)
        slopes_dir.mkdir(parents=True, exist_ok=True)

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

        # Within-group correlations for ALL metrics requested
        corr_df = compute_within_group_correlations(
            df_metrics,
            subset=subset,
            metrics=primary_metrics,
            n_boot=2000,
            random_state=RNG_SEED,
        )
        if not corr_df.empty:
            all_corr.append(corr_df)

        # Interaction models and slopes for ALL metrics
        for metric in primary_metrics:
            if metric not in df_metrics.columns:
                print(f"[WARN] metric {metric} not in df_metrics; skipping.")
                continue

            print(f"[PLOT] subset={subset}, metric={metric}")
            model, slopes_df = fit_speed_nor_interaction(
                df_metrics,
                subset=subset,
                metric=metric,
                ref_group="WT_VEH",
            )
            # add parsing for later grouping
            slopes_df["subset_base"], slopes_df["segment"] = subset.split("__")
            all_slopes.append(slopes_df)

            if save_plots and scatter_dir is not None and slopes_dir is not None:
                fig1, _ = plot_nor_vs_metric_by_group(
                    df_metrics,
                    subset=subset,
                    metric=metric,
                    model=model,
                )
                fig1_path = scatter_dir / f"scatter_{subset}_{metric}.png"
                fig1.savefig(fig1_path, dpi=300, bbox_inches="tight")
                plt.close(fig1)

                fig2, _ = plot_group_slopes(
                    slopes_df,
                    subset=subset,
                    metric=metric,
                )
                fig2_path = slopes_dir / f"slopes_{subset}_{metric}.png"
                fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
                plt.close(fig2)

    corr_summary = pd.concat(all_corr, ignore_index=True) if all_corr else None
    slopes_summary = pd.concat(all_slopes, ignore_index=True) if all_slopes else None
    return corr_summary, slopes_summary


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

def summarize_segment_group_models(
    df: pd.DataFrame,
    metrics: Sequence[str],
    ref_group: str = "WT_VEH",
    ref_segment: str = "mid",
) -> pd.DataFrame:
    """
    For each subset_base and each metric, fit:

        nor ~ metric * group * segment

    and extract:
      - R², adj R², F, p(F)
      - min p-value for:
          * metric main effect
          * metric × group interactions
          * metric × segment interactions
          * metric × group × segment (3-way)

    Returns a tidy DataFrame (one row per subset_base × metric).
    """
    if "subset_base" not in df.columns or "segment" not in df.columns:
        df = add_subset_segment_columns(df)

    subset_bases = sorted(df["subset_base"].unique())
    rows = []

    for subset_base in subset_bases:
        for metric in metrics:
            if metric not in df.columns:
                print(f"[WARN] Metric {metric} not in df; skipping subset_base={subset_base}")
                continue

            try:
                model = fit_segment_group_interaction(
                    df,
                    subset_base=subset_base,
                    metric=metric,
                    ref_group=ref_group,
                    ref_segment=ref_segment,
                )
            except ValueError as e:
                print(f"[WARN] Could not fit model for subset_base={subset_base}, metric={metric}: {e}")
                continue

            params_index = model.params.index
            pvals = model.pvalues

            def min_p_for(prefix: str, must_contain: str | None = None) -> float:
                vals = []
                for name in params_index:
                    if name.startswith(prefix):
                        if must_contain is None or must_contain in name:
                            vals.append(pvals[name])
                return float(np.min(vals)) if vals else np.nan

            # main effect of metric
            p_metric = float(pvals.get(metric, np.nan))

            # metric × group (any group level)
            p_metric_group = min_p_for(f"{metric}:C(group")

            # metric × segment (short/long vs mid)
            p_metric_segment = min_p_for(f"{metric}:C(segment")

            # metric × group × segment (3-way, any combination)
            p_metric_group_segment = min_p_for(
                f"{metric}:C(group", must_contain="C(segment"
            )

            rows.append(
                {
                    "subset_base": subset_base,
                    "metric": metric,
                    "n": int(model.nobs),
                    "df_model": float(model.df_model),
                    "df_resid": float(model.df_resid),
                    "rsq": float(model.rsquared),
                    "rsq_adj": float(model.rsquared_adj),
                    "fvalue": float(model.fvalue) if model.fvalue is not None else np.nan,
                    "f_pvalue": float(model.f_pvalue) if model.f_pvalue is not None else np.nan,
                    "p_metric": p_metric,
                    "p_metric_group_min": p_metric_group,
                    "p_metric_segment_min": p_metric_segment,
                    "p_metric_group_segment_min": p_metric_group_segment,
                    "ref_group": ref_group,
                    "ref_segment": ref_segment,
                }
            )

    if not rows:
        raise ValueError("No segment×group models could be fitted.")
    return pd.DataFrame(rows)

def leave_one_out_slopes(
    df: pd.DataFrame,
    subset: str,
    metric: str = "speed_q95",
    ref_group: str = "WT_VEH",
) -> pd.DataFrame:
    """
    LOO robustness for slopes in a given subset and metric.

    For each animal, drop it, refit:
        nor ~ metric * group
    and store per-group slopes and CIs.

    Returns a DataFrame with columns:
        subset, metric, group, animal_id, slope, se, ci_low, ci_high
    """
    df_sub = df[df["subset"] == subset].copy()
    if df_sub.empty:
        raise ValueError(f"No rows for subset={subset!r}")

    animal_ids = sorted(df_sub["animal_id"].unique())
    rows = []

    for aid in animal_ids:
        df_loo = df_sub[df_sub["animal_id"] != aid].copy()
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

    return pd.DataFrame(rows)


def leave_one_out_slopes_all(
    df: pd.DataFrame,
    metrics: Sequence[str] = PRIMARY_METRICS,
    ref_group: str = "WT_VEH",
    subsets: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Compute LOO slopes for *all* subset labels and metrics (or a restricted set).

    For each subset in `subsets` (defaults to all df['subset'].unique())
    and each metric in `metrics`, it runs leave_one_out_slopes and
    concatenates everything into one big DataFrame.

    WARNING: This is heavy (overnight job) because it fits one model per
    animal × subset × metric.
    """
    if subsets is None:
        subsets = sorted(df["subset"].unique())

    loo_dfs = []

    for subset in subsets:
        df_sub = df[df["subset"] == subset]
        if df_sub.empty:
            continue

        for metric in metrics:
            if metric not in df.columns:
                print(f"[WARN] metric {metric} not in df; skipping LOO for subset={subset}")
                continue

            print(f"[LOO] subset={subset}, metric={metric}")
            loo_df = leave_one_out_slopes(
                df,
                subset=subset,
                metric=metric,
                ref_group=ref_group,
            )
            loo_df["subset_base"], loo_df["segment"] = subset.split("__")
            loo_dfs.append(loo_df)

    if not loo_dfs:
        raise ValueError("No LOO slopes computed – check subsets/metrics.")
    return pd.concat(loo_dfs, ignore_index=True)



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
# PHASE 7 – META-SUMMARIES: EFFECT RANKING & TOP CANDIDATES
# =============================================================================

def build_effect_summary(
    corr_summary: pd.DataFrame,
    slopes_summary: pd.DataFrame,
    loo_all: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a tidy per-(subset_base, segment, metric, group) summary that combines:

      - Within-group correlations (corr_summary)
      - Group slopes (slopes_summary)
      - LOO robustness (loo_all)

    Returns a DataFrame with columns like:
        subset, subset_base, segment, metric, group,
        corr_rho_boot, corr_q, corr_signif,
        slope, slope_q, slope_signif,
        loo_n, loo_mean_slope, loo_std_slope, loo_same_sign_rate
    """

    if corr_summary is None or slopes_summary is None or loo_all is None:
        raise ValueError("corr_summary, slopes_summary, and loo_all must all be non-None")

    # Make sure subset_base/segment exist everywhere
    for df in (corr_summary, slopes_summary, loo_all):
        if "subset_base" not in df.columns or "segment" not in df.columns:
            add_subset_segment_columns(df)

    # Ensure q_value / signif_fdr_05 exist (if FDR not applied for some reason)
    corr_df = corr_summary.copy()
    if "q_value" not in corr_df.columns:
        corr_df["q_value"] = np.nan
    if "signif_fdr_05" not in corr_df.columns:
        corr_df["signif_fdr_05"] = False

    slopes_df = slopes_summary.copy()
    if "q_value" not in slopes_df.columns:
        slopes_df["q_value"] = np.nan
    if "signif_fdr_05" not in slopes_df.columns:
        slopes_df["signif_fdr_05"] = False

    # Keep only the useful columns
    corr_sel = corr_df[
        [
            "subset",
            "subset_base",
            "segment",
            "group",
            "metric",
            "rho_boot_mean",
            "q_value",
            "signif_fdr_05",
        ]
    ].rename(
        columns={
            "rho_boot_mean": "corr_rho_boot",
            "q_value": "corr_q",
            "signif_fdr_05": "corr_signif",
        }
    )

    slopes_sel = slopes_df[
        [
            "subset",
            "subset_base",
            "segment",
            "group",
            "metric",
            "slope",
            "q_value",
            "signif_fdr_05",
        ]
    ].rename(
        columns={
            "q_value": "slope_q",
            "signif_fdr_05": "slope_signif",
        }
    )

    # Aggregate LOO per subset/metric/group
    loo_df = loo_all.copy()
    if "subset_base" not in loo_df.columns or "segment" not in loo_df.columns:
        add_subset_segment_columns(loo_df)

    # Join full-sample slope to LOO to compute same-sign rate
    slopes_min = slopes_df[
        ["subset", "metric", "group", "slope"]
    ].drop_duplicates()

    loo_with_full = loo_df.merge(
        slopes_min,
        on=["subset", "metric", "group"],
        how="left",
        suffixes=("", "_full"),
    )

    def _safe_sign(x: pd.Series) -> pd.Series:
        # Treat extremely small values as zero to avoid numerical noise flips
        x = x.copy()
        x[np.isclose(x, 0.0)] = 0.0
        return np.sign(x)

    loo_with_full["same_sign"] = (
        _safe_sign(loo_with_full["slope"]) == _safe_sign(loo_with_full["slope_full"])
    )

    loo_agg = (
        loo_with_full.groupby(["subset", "subset_base", "segment", "metric", "group"])
        .agg(
            loo_n=("animal_id", "nunique"),
            loo_mean_slope=("slope", "mean"),
            loo_std_slope=("slope", "std"),
            loo_same_sign_rate=("same_sign", "mean"),
        )
        .reset_index()
    )

    # Merge everything
    eff = slopes_sel.merge(
        corr_sel,
        on=["subset", "subset_base", "segment", "metric", "group"],
        how="left",
    ).merge(
        loo_agg,
        on=["subset", "subset_base", "segment", "metric", "group"],
        how="left",
    )

    # A simple combined effect score (for convenience):
    #  - big |slope| and |rho| help
    #  - require positive LOO same-sign rate (else 0)
    eff["score_raw"] = np.abs(eff["slope"]) * np.abs(eff["corr_rho_boot"])
    eff["score"] = eff["score_raw"] * eff["loo_same_sign_rate"].fillna(0.0)

    return eff


def get_top_effects(
    effect_summary: pd.DataFrame,
    n: int = 5,
    min_loo_same_sign: float = 0.7,
    require_both_sig: bool = True,
) -> pd.DataFrame:
    """
    Select top 'n' (subset_base, segment, metric, group) entries that:

      - optionally have BOTH FDR-significant slope AND corr
      - have LOO same-sign rate >= min_loo_same_sign
      - are ranked by 'score' = |slope| × |rho| × LOO_same_sign_rate

    Returns a DataFrame sorted by descending score.
    """
    df = effect_summary.copy()

    if require_both_sig:
        df = df[df["slope_signif"] & df["corr_signif"]]

    df = df[df["loo_same_sign_rate"].fillna(0.0) >= float(min_loo_same_sign)]

    if df.empty:
        print("[WARN] No effects survived the filters in get_top_effects.")
        return df

    df = df.sort_values("score", ascending=False)
    return df.head(n)

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
        descriptor = suffix[:-4]  # remove ".npz"
        descriptors.append(descriptor)

    return sorted(set(descriptors))

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

    # ---------------- Load speeds for all subsets ----------------
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

    # ---------------- Add per-region as extra subsets ----------------
    speeds_by_subset = add_per_region_speeds(
        speeds_by_subset=speeds_by_subset,
        speed_root=speed_root,
        time_windows_range=time_windows_range,
        n_animals=n_animals,
        regions=regions,
    )

    all_subset_labels = list(speeds_by_subset.keys())
    print(f"[INFO] Available subsets (including per_region): {all_subset_labels}")

    # ---------------- Option 1: global cache root ----------------
    cache_root = Path(paths["f_speed"]) / "nor_tail_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    # joblib cache directory (keep this on a local/ext4 disk!)
    joblib_cache_root = Path("~/.cache/net_fluidity_nor_bootstrap").expanduser()
    joblib_cache_root.mkdir(parents=True, exist_ok=True)

    # override global Memory
    memory = Memory(location=joblib_cache_root, verbose=0)

    cache_config = {
        "dataset": dataset_name,
        "pool_split": POOL_SPLIT,
        "time_windows": [int(w) for w in time_windows_range],
        "subsets": sorted(all_subset_labels),
        "primary_metrics": PRIMARY_METRICS,
        "rng_seed": RNG_SEED,
        "analysis_version": 2,  # <-- bump this whenever you change analysis logic
    }

    cache_key = make_cache_key(cache_config)

    cache_corr = cache_root / f"corr_summary_{cache_key}.parquet"
    cache_slopes = cache_root / f"slopes_summary_{cache_key}.parquet"
    cache_cfg_path = cache_root / f"config_{cache_key}.json"
    loo_cache_path = cache_root / f"loo_slopes_{cache_key}.parquet"
    segment_models_path = cache_root / f"segment_group_models_{cache_key}.parquet"

    # ---------------- Per-subset metrics cache ----------------cache_key
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
            "rng_seed": RNG_SEED,
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

    # ---------------- Global cache for summaries ----------------
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

    # ---------------- LOO robustness for all subset×segment×metric ----------------

    if loo_cache_path.exists():
        print(f"[INFO] Loading cached LOO slopes from {loo_cache_path}")
        loo_all = pd.read_parquet(loo_cache_path)
    else:
        print("[INFO] Computing LOO slopes for all subset×segment×metric (heavy)")
        loo_all = leave_one_out_slopes_all(
            df_metrics,
            metrics=PRIMARY_METRICS,
            ref_group="WT_VEH",
            subsets=None,  # all subset labels
        )
        loo_all.to_parquet(loo_cache_path, index=False)
        print(f"[INFO] Saved LOO slopes to {loo_cache_path}")


    # --------------------------------------------------------------------
    # Segment × group interaction summary FOR ALL subset_bases & metrics
    # --------------------------------------------------------------------

    if segment_models_path.exists():
        print(f"[INFO] Loading cached segment×group models from {segment_models_path}")
        segment_models_df = pd.read_parquet(segment_models_path)
    else:
        print("[INFO] Fitting segment×group×metric interaction models for all subset_bases")
        segment_models_df = summarize_segment_group_models(
            df_metrics,
            metrics=PRIMARY_METRICS,
            ref_group="WT_VEH",
            ref_segment="mid",
        )
        segment_models_df.to_parquet(segment_models_path, index=False)
        print(f"[INFO] Saved segment×group models summary to {segment_models_path}")

    print("\n=== SEGMENT × GROUP MODEL SUMMARY (HEAD) ===")
    print(segment_models_df.head())


    # Directory for per-subset×segment×metric plots
    plots_root = cache_root / "plots_segments_allmetrics"
    plots_root.mkdir(parents=True, exist_ok=True)

    if corr_summary is None or slopes_summary is None:
        # recompute correlations and slopes AND save all plots
        corr_summary, slopes_summary = run_primary_analysis_from_df(
            df_metrics=df_metrics,            # <-- reuse metrics already computed
            primary_subsets=None,            # ALL subset__segment combinations
            primary_metrics=PRIMARY_METRICS,
            save_plots=True,
            fig_root=plots_root,
        )

        # FDR correction for correlations
        if corr_summary is not None and "p_value" in corr_summary.columns:
            mask = corr_summary["p_value"].notna()
            if mask.any():
                _, qvals, _, _ = multipletests(
                    corr_summary.loc[mask, "p_value"], method="fdr_bh"
                )
                corr_summary.loc[mask, "q_value"] = qvals
                corr_summary["signif_fdr_05"] = corr_summary["q_value"] < 0.05

        # FDR correction for slopes
        if slopes_summary is not None and "p_value" in slopes_summary.columns:
            mask = slopes_summary["p_value"].notna()
            if mask.any():
                _, qvals, _, _ = multipletests(
                    slopes_summary.loc[mask, "p_value"], method="fdr_bh"
                )
                slopes_summary.loc[mask, "q_value"] = qvals
                slopes_summary["signif_fdr_05"] = slopes_summary["q_value"] < 0.05

        if corr_summary is not None:
            corr_summary.to_parquet(cache_corr, index=False)
        if slopes_summary is not None:
            slopes_summary.to_parquet(cache_slopes, index=False)

        with cache_cfg_path.open("w") as f:
            json.dump(cache_config, f, indent=2)
    else:
        print("[INFO] Using cached corr_summary / slopes_summary")


    # --------------------------------------------------------------------
    # EFFECT SUMMARY + TOP CANDIDATES
    # --------------------------------------------------------------------
    effect_summary = build_effect_summary(
        corr_summary=corr_summary,
        slopes_summary=slopes_summary,
        loo_all=loo_all,
    )

    print("\n=== EFFECT SUMMARY (HEAD) ===")
    print(effect_summary.head())

    top5 = get_top_effects(
        effect_summary,
        n=5,
        min_loo_same_sign=0.2,
        # require_both_sig=True,
    )

    print("\n=== TOP 5 CANDIDATES (FDR-sig corr & slope, LOO same_sign>=0.2) ===")
    print(top5[
        [
            "subset_base",
            "segment",
            "metric",
            "group",
            "corr_rho_boot",
            "corr_q",
            "slope",
            "slope_q",
            "loo_same_sign_rate",
            "score",
        ]
    ])


    # ---------------- Extra plots: multi-segment scatter rows for speed_q95 ----------------
    fig_dir_rows = cache_root / "plots_segments_speed_q95"
    fig_dir_rows.mkdir(parents=True, exist_ok=True)

    subset_bases = sorted(df_metrics["subset_base"].unique())
    for base in subset_bases:
        fig, _ = plot_multi_segment_scatter_row(
            df_metrics,
            subset_base=base,
            metric="speed_q95",
            ref_group="WT_VEH",
        )
        out_path = fig_dir_rows / f"nor_vs_speed_q95_segments_{base}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Segment × group interaction for salience within and DMN-touching (speed_q95)
    model_sal_seg = fit_segment_group_interaction(
        df_metrics,
        subset_base="sal_within",
        metric="speed_q95",
        ref_group="WT_VEH",
        ref_segment="mid",
    )

    model_dmn_seg = fit_segment_group_interaction(
        df_metrics,
        subset_base="dmn_touching",
        metric="speed_q95",
        ref_group="WT_VEH",
        ref_segment="mid",
    )

    print(model_sal_seg.summary())
    print(model_dmn_seg.summary())

    # LOO robustness for two key cells
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

    print("\n=== WITHIN-GROUP CORRELATIONS (HEAD) ===")
    print(corr_summary.head() if corr_summary is not None else "None")

    print("\n=== GROUP SLOPES (HEAD) ===")
    print(slopes_summary.head() if slopes_summary is not None else "None")

    plt.show()


