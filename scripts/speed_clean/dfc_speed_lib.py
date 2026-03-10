#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dfc_speed_lib.py
================
Core library for dFC speed distribution analysis.

Consolidates helpers from:
  plot_speed_dist_ines.py / plot_speed_dist_julien5.py / plot_speed_dist_acc.py
  nor_tail_analysis.py / nor_tail2.py
  save_session_window_quantiles.py / plot_qc_speed_vs_window_allsubset_julien.py

Public API (all importable):
  -- I/O --
  load_speed_stack(template, windows, n_animals, regions)  →  list[list[ndarray]]
  load_speed_stack_single_region(...)                       →  list[list[ndarray]]
  discover_per_region_descriptors(subset_dir, ...)          →  list[str]

  -- Grouping --
  make_long_cog(cog_data, dataset_name)    →  DataFrame
  group_indices(df, by)                   →  dict[key, ndarray[int]]

  -- Windowing / pooling --
  count_samples_per_window(speeds)              →  ndarray
  cdf_split_indices(speeds)                    →  (i_third, i_half, i_two_third)
  select_windows(pool_split, n_windows, …)     →  dict[str, range]
  flatten_windows(speeds, start, end)          →  ndarray 1D
  global_min_max(arrs)                         →  (float, float)
  pool_speeds_per_animal(speeds, …)            →  list[ndarray]
  build_per_animal_normalized_hists(…)         →  ndarray (n_animals, bins)
  flatten_group_animals_over_windows(…)        →  ndarray 1D
  get_group_animals_over_windows(…)            →  ndarray (n_animals, n_windows) object

  -- Metrics --
  SpeedMetrics (dataclass)
  compute_speed_metrics(samples)           →  SpeedMetrics
  build_subset_metrics_df(...)             →  DataFrame
  compute_subset_metrics_with_segments(…)  →  DataFrame
  build_metrics_with_segments(…)           →  DataFrame

  -- Statistics --
  bootstrap_spearman(x, y, …)             →  dict
  compute_within_group_correlations(…)     →  DataFrame
  fit_speed_nor_interaction(…)             →  (model, slopes_df)
  fit_segment_group_interaction(…)         →  model
  leave_one_out_slopes(…)                 →  DataFrame
  leave_one_out_slopes_all(…)             →  DataFrame
  summarize_segment_group_models(…)        →  DataFrame
  build_effect_summary(…)                 →  DataFrame
  get_top_effects(…)                      →  DataFrame

  -- Downsampling bootstrap (Inès CI bands) --
  bootstrap_downsampling_repeat(data, percentiles, …)  →  (ci_low, ci_high, ci_matrix)
  compute_bootstrap_ci_bands(ranges, pooled_speeds, …) →  (ci_low_dict, ci_high_dict, ci_matrix_dict)

  -- Window × percentile correlation --
  compute_window_nor_correlations(…)  →  DataFrame (roi, window, q, group, spearman_rho, …)
  plot_window_nor_correlations(df_cor, …)  →  saves PNG figures

  -- Quantile tensor --
  compute_quantile_tensor(speeds, q_grid)  →  ndarray (n_sessions, n_windows, n_q)
  save_quantile_npz(outpath, Q, …)

  -- Plotting --
  plot_group_speed_distributions(…)
  plot_ci_bands(…)
  plot_age_contrasts(…)
  plot_per_animal_histograms(…)
  plot_qc_3panel(…)
  plot_nor_vs_metric_by_group(…)
  plot_group_slopes(…)
  plot_multi_segment_scatter_row(…)
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, t as student_t

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    from joblib import Memory
    _joblib_available = True
except ImportError:
    _joblib_available = False

try:
    import statsmodels.formula.api as smf
    from statsmodels.stats.multitest import multipletests
    _statsmodels_available = True
except ImportError:
    _statsmodels_available = False

# Global joblib cache (disabled by default; set via configure_cache())
_memory = None if not _joblib_available else Memory(location=None, verbose=0)
_cached_bootstrap_fn = None

RNG_SEED = 123

# ---------------------------------------------------------------------------
# GROUP RECIPES
# ---------------------------------------------------------------------------

GROUP_RECIPES_INES: dict[str, list[str]] = {
    "sex":                  ["Sexe"],
    "age":                  ["Age"],
    "genotype":             ["Genotype"],
    "phenotype_oip":        ["Phenotype_OiP"],
    "phenotype_nor":        ["Phenotype_RO24h"],
    "age_sex":              ["Age", "Sexe"],
    "age_genotype":         ["Age", "Genotype"],
    "age_phenotype_oip":    ["Age", "Phenotype_OiP"],
    "age_phenotype_nor":    ["Age", "Phenotype_RO24h"],
    "sex_genotype":         ["Sexe", "Genotype"],
    "sex_phenotype_oip":    ["Sexe", "Phenotype_OiP"],
    "sex_phenotype_nor":    ["Sexe", "Phenotype_RO24h"],
    "age_sex_genotype":     ["Sexe", "Age", "Genotype"],
    "age_sex_phenotype_oip":["Sexe", "Age", "Phenotype_OiP"],
    "age_sex_phenotype_nor":["Sexe", "Age", "Phenotype_RO24h"],
}

GROUP_RECIPES_JULIEN: dict[str, list[str]] = {
    "genotype_treatment": ["genotype", "treatment"],
}

GROUP_RECIPES: dict[str, dict[str, list[str]]] = {
    "ines":   GROUP_RECIPES_INES,
    "julien": GROUP_RECIPES_JULIEN,
}

PRIMARY_METRICS: list[str] = [
    "speed_q01",
    "speed_q05",
    "speed_median",
    "speed_q95",
    "speed_q99",
    "speed_width50",
    "speed_width_extreme",
    "speed_asymmetry",
]

# Standard speed subsets
SPEED_SUBSETS: list[str] = [
    "all",
    "dmn_touching", "1st_touching", "2nd_touching", "3rd_touching",
    "4th_touching",  "lat_touching", "mem_touching", "sal_touching",
    "dmn_within",   "1st_within",   "2nd_within",   "3rd_within",
    "4th_within",   "lat_within",   "mem_within",   "sal_within",
]


# =============================================================================
# CACHING HELPERS
# =============================================================================

def configure_cache(cache_dir: str | Path) -> None:
    """Enable joblib caching for bootstrap_spearman to the given directory."""
    global _memory, _cached_bootstrap_fn
    if not _joblib_available:
        raise ImportError("joblib is required for caching")
    _memory = Memory(location=str(cache_dir), verbose=0)
    _cached_bootstrap_fn = None  # will be re-created on next call


def make_cache_key(config: dict) -> str:
    """Stable MD5 hash from a JSON-serializable config dict."""
    return hashlib.md5(
        json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


# =============================================================================
# I/O
# =============================================================================

def load_speed_stack(
    template: str,
    time_windows_range: Sequence[int],
    n_animals: int,
    regions: int,
) -> list[list[np.ndarray]]:
    """
    Load dFC speed stacks for one subset across all windows.

    Parameters
    ----------
    template : str
        Path template with {w}, {n_animals}, {regions} placeholders.
    time_windows_range : sequence of int
        Window sizes present on disk.
    n_animals, regions : int
        Dataset dimensions (used to fill placeholders).

    Returns
    -------
    speeds : list[n_windows] of list[n_animals] of 1D ndarray
        speeds[j][i] = speed samples for animal i at window j.
    """
    speeds: list[list[np.ndarray]] = []
    for w in time_windows_range:
        fp = template.format(w=w, n_animals=n_animals, regions=regions)
        arr = np.load(fp, allow_pickle=True)
        if "speeds" not in arr.files:
            raise KeyError(f"Key 'speeds' missing in {fp}")
        s = arr["speeds"]
        s_flat = [np.asarray(x, dtype=float).ravel() for x in s]
        if len(s_flat) != n_animals:
            raise ValueError(
                f"{fp}: expected {n_animals} animals, got {len(s_flat)}"
            )
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
    Return sorted list of region descriptors (e.g. ['region-AI', 'region-PL'])
    by inspecting existing NPZ files in subset_dir for a single window w0.
    """
    pattern = str(
        subset_dir
        / f"speed_win{w0}_lag{lag}_tau{tau_count}_animals_{n_animals}_regions_{regions}_region-*.npz"
    )
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No per-region files found matching: {pattern}")
    descriptors = []
    for fpath in files:
        suffix = Path(fpath).name.split(f"regions_{regions}_", 1)[1]
        descriptors.append(suffix[:-4])  # strip '.npz'
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
    """Like load_speed_stack but for a single region descriptor under per_region/."""
    speeds: list[list[np.ndarray]] = []
    for w in time_windows_range:
        fpath = (
            subset_dir
            / f"speed_win{w}_lag{lag}_tau{tau_count}_animals_{n_animals}_regions_{regions}_{region_desc}.npz"
        )
        if not fpath.exists():
            raise FileNotFoundError(f"Missing per-region file: {fpath}")
        with np.load(fpath, allow_pickle=True) as z:
            s = z["speeds"]
        if len(s) != n_animals:
            raise ValueError(f"{fpath}: expected {n_animals} animals, got {len(s)}")
        speeds.append([np.asarray(s[i], dtype=float).ravel() for i in range(n_animals)])
    return speeds


def discover_per_region_subset_labels(bootstrap_folder: Path) -> list[str]:
    """
    Scan bootstrap_folder for per-region bootstrap PKLs and return
    the corresponding subset labels (e.g. ['per_region_region-AI', ...]).
    """
    labels: set[str] = set()
    for p in bootstrap_folder.glob("bootstrap_downsample_repeat_subset_per_region_region-*_group_*.pkl"):
        m = re.search(r"subset_(per_region_region-[^_]+)_group_", p.name)
        if m:
            labels.add(m.group(1))
    return sorted(labels)


# =============================================================================
# GROUPING HELPERS
# =============================================================================

def make_long_cog(cog_data: "pd.DataFrame", dataset_name: str) -> "pd.DataFrame":
    """
    Standardise cognitive data to a long-form DataFrame.

    For 'ines': pivots 2M/4M age-stratified columns into rows.
    For 'julien': passes through with light column normalisation.

    Columns returned (dataset-dependent):
      ines  : Name, Sexe, Genotype, Age, oip, ro24h, tc, Phenotype_OiP, Phenotype_RO24h
      julien: name (or mouse), genotype, treatment
    """
    if pd is None:
        raise ImportError("pandas is required")

    df = cog_data.copy()

    if dataset_name == "ines":
        cols_common = ["Name", "Sexe", "Genotype", "Phenotype_OiP", "Phenotype_RO24h"]
        df2 = df[cols_common + ["OiP_2M", "RO24h_2M", "TC_2M"]].copy()
        df4 = df[cols_common + ["OiP_4M", "RO24h_4M", "TC_4M"]].copy()
        df2["Age"] = "2M"
        df4["Age"] = "4M"
        df2 = df2.rename(columns={"OiP_2M": "oip", "RO24h_2M": "ro24h", "TC_2M": "tc"})
        df4 = df4.rename(columns={"OiP_4M": "oip", "RO24h_4M": "ro24h", "TC_4M": "tc"})
        df = pd.concat([df2, df4], ignore_index=True)
        df["Sexe"] = df["Sexe"].map({"F": "female", "M": "male"}).fillna(df["Sexe"])
        for col in ["Sexe", "Age", "Genotype", "Phenotype_OiP", "Phenotype_RO24h"]:
            if col in df.columns:
                df[col] = df[col].astype("category")

    elif dataset_name == "julien":
        if "mouse" in df.columns and "name" not in df.columns:
            df = df.rename(columns={"mouse": "name"})
        cols_keep = [c for c in ["name", "genotype", "treatment"] if c in df.columns]
        df = df[cols_keep].copy()
        for col in ["genotype", "treatment"]:
            if col in df.columns:
                df[col] = df[col].astype("category")

    else:
        raise ValueError(f"Unknown dataset_name={dataset_name!r}")

    return df


def group_indices(
    df: "pd.DataFrame",
    by: Sequence[str],
) -> dict:
    """
    Return {group_key : np.ndarray[int]} mapping group tuples to row indices.
    If `by` is empty, returns {"all": arange(len(df))}.
    """
    if pd is None:
        raise ImportError("pandas is required")
    if not by:
        return {"all": np.arange(len(df), dtype=int)}
    return {k: v.values for k, v in df.groupby(list(by), sort=False).groups.items()}


def get_group_data(
    cog_data: "pd.DataFrame",
    dataset_name: str,
    groups_selected: str,
) -> dict:
    """Build group_data dict using a named recipe from GROUP_RECIPES."""
    recipes = GROUP_RECIPES.get(dataset_name)
    if recipes is None:
        raise ValueError(f"No GROUP_RECIPES for dataset_name={dataset_name!r}")
    cols = recipes.get(groups_selected)
    if cols is None:
        raise ValueError(
            f"Unknown groups_selected={groups_selected!r}. "
            f"Available: {sorted(recipes.keys())}"
        )
    df_long = make_long_cog(cog_data, dataset_name)
    missing = [c for c in cols if c not in df_long.columns]
    if missing:
        raise ValueError(f"Grouping {groups_selected!r} needs missing columns: {missing}")
    return group_indices(df_long, cols)


# =============================================================================
# WINDOWING / POOLING
# =============================================================================

def count_samples_per_window(speeds: list) -> np.ndarray:
    """Total number of speed samples per time window (across all animals)."""
    return np.array([sum(len(x) for x in win) for win in speeds], dtype=int)


def cdf_split_indices(speeds: list) -> tuple[int, int, int]:
    """
    Find window indices at 1/3, 1/2 and 2/3 of the cumulative sample count.
    Used to split windows into short / mid / long segments.
    """
    counts = count_samples_per_window(speeds)
    if counts.sum() > 0:
        cdf = np.cumsum(counts) / counts.sum()
    else:
        cdf = np.zeros_like(counts, dtype=float)

    i_third     = max(1, int(np.searchsorted(cdf, 1 / 3)))
    i_half      = max(1, int(np.searchsorted(cdf, 0.5)))
    i_two_third = max(i_third + 1, int(np.searchsorted(cdf, 2 / 3)))
    return i_third, i_half, i_two_third


def select_windows(
    pool_split: str,
    n_windows: int,
    i_third: int,
    i_half: int,
    i_two_third: int,
) -> dict[str, range]:
    """
    Map segment name → range of window indices.

    pool_split : "all" | "half" | "third"
    """
    if pool_split == "all":
        return {"all": range(n_windows)}
    if pool_split == "half":
        return {"short": range(i_half), "long": range(i_half, n_windows)}
    return {
        "short": range(i_third),
        "mid":   range(i_third, i_two_third),
        "long":  range(i_two_third, n_windows),
    }


def flatten_windows(
    speeds: list[list[np.ndarray]],
    start: int,
    end: int,
) -> np.ndarray:
    """Concatenate all speed samples across windows [start, end) and all animals."""
    arrays = [
        np.asarray(s, dtype=float).ravel()
        for win in speeds[start:end]
        for s in win
    ]
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=float)


def global_min_max(arrs: Iterable[np.ndarray]) -> tuple[float, float]:
    """Finite min and max across a collection of arrays."""
    arrs = list(arrs)
    valid = [a for a in arrs if np.asarray(a).size]
    vmin = float(min(np.nanmin(a) for a in valid)) if valid else 0.0
    vmax = float(max(np.nanmax(a) for a in valid)) if valid else 1.0
    if vmin == vmax:
        vmax = vmin + 1.0
    return vmin, vmax


def pool_speeds_per_animal(
    speeds: list[list[np.ndarray]],
    window_indices: Sequence[int] | None = None,
) -> list[np.ndarray]:
    """
    Pool speed samples across windows for each animal.

    Returns a list of length n_animals, where each element is the
    concatenated 1D array of all samples for that animal over the
    selected windows.
    """
    if window_indices is None:
        window_indices = range(len(speeds))
    n_animals = len(speeds[next(iter(window_indices))])
    return [
        np.concatenate([np.ravel(speeds[j][i]) for j in window_indices])
        for i in range(n_animals)
    ]


# ── Histogram helpers for bootstrap distribution plots ────────────────────────

def build_per_animal_normalized_hists(
    speeds: list[list[np.ndarray]],
    w_range: range,
    bins: int,
    speed_range: tuple[float, float],
) -> np.ndarray:
    """
    Build a (n_animals, bins) array of normalised histograms.

    Each row is the probability density for one animal, pooled across all
    windows in w_range.  Used to compute per-group mean distribution curves.

    Parameters
    ----------
    speeds      : nested list  speeds[window][animal] → 1D float array
    w_range     : range of window indices to pool
    bins        : number of histogram bins
    speed_range : (min, max) shared across all animals

    Returns
    -------
    H : ndarray shape (n_animals, bins)  — each row sums to 1 (or 0 if empty)
    """
    n_animals = len(speeds[0])
    H = np.zeros((n_animals, bins), dtype=float)
    for i in range(n_animals):
        data = np.concatenate([
            np.ravel(speeds[j][i])
            for j in w_range
            if j < len(speeds) and len(speeds[j][i]) > 0
        ]) if any(j < len(speeds) for j in w_range) else np.array([])
        if data.size == 0:
            continue
        h, _ = np.histogram(data, bins=bins, range=speed_range)
        total = h.sum()
        if total > 0:
            H[i] = h / total
    return H


def flatten_group_animals_over_windows(
    animal_speeds: list[list[np.ndarray]],
    animal_indices: list[int],
    w_range: range,
) -> np.ndarray:
    """
    Concatenate all speed samples for a group of animals across a window range.

    Parameters
    ----------
    animal_speeds  : animal_speeds[i][j] → 1D array of speeds for animal i at window j
    animal_indices : list of animal indices belonging to the group
    w_range        : range of window indices to include

    Returns
    -------
    flat : 1D ndarray of all speed values (used as input to bootstrap)
    """
    parts = [
        np.ravel(animal_speeds[i][j])
        for i in animal_indices
        for j in w_range
        if j < len(animal_speeds[i]) and len(animal_speeds[i][j]) > 0
    ]
    return np.concatenate(parts) if parts else np.empty(0, dtype=float)


def get_group_animals_over_windows(
    animal_speeds: list[list[np.ndarray]],
    animal_indices: list[int],
    w_range: range,
) -> np.ndarray:
    """
    Return a (n_group_animals, n_windows) object array of speed arrays.

    Each cell [i, j] holds the 1D speed array for animal i at window j.
    This is stored as ``group_speed_by_segment`` in the bootstrap PKL and
    used by ``plot_per_animal_histograms``.

    Parameters
    ----------
    animal_speeds  : animal_speeds[i][j] → 1D float array
    animal_indices : animal indices for the group
    w_range        : window indices for the segment

    Returns
    -------
    out : ndarray shape (n_animals, n_windows)  dtype=object
    """
    n_a = len(animal_indices)
    n_w = len(w_range)
    out = np.empty((n_a, n_w), dtype=object)
    for row, i in enumerate(animal_indices):
        for col, j in enumerate(w_range):
            out[row, col] = (
                np.ravel(animal_speeds[i][j])
                if j < len(animal_speeds[i])
                else np.empty(0, dtype=float)
            )
    return out


# =============================================================================
# METRICS
# =============================================================================

@dataclass
class SpeedMetrics:
    """Per-animal dFC speed distribution metrics."""
    q01: float
    q05: float
    q50: float
    q90: float
    q95: float
    q99: float
    width50: float         # q95 - q05
    width_extreme: float   # q99 - q01
    asymmetry: float       # (q95 - q50) - (q50 - q05)


def compute_speed_metrics(samples: np.ndarray) -> SpeedMetrics:
    """Compute central and tail metrics for a 1D array of dFC speeds."""
    q01, q05, q50, q90, q95, q99 = np.percentile(
        samples[np.isfinite(samples)], [1, 5, 50, 90, 95, 99]
    )
    return SpeedMetrics(
        q01=q01, q05=q05, q50=q50, q90=q90, q95=q95, q99=q99,
        width50=q95 - q05,
        width_extreme=q99 - q01,
        asymmetry=(q95 - q50) - (q50 - q05),
    )


def _infer_animal_group_labels(
    group_data: dict,
    n_animals: int,
) -> tuple[list[str], list[str | None], list[str | None]]:
    group_labels: list[str | None] = [None] * n_animals
    genotypes:    list[str | None] = [None] * n_animals
    treatments:   list[str | None] = [None] * n_animals

    for gkey, idxs in group_data.items():
        label = "_".join(map(str, gkey)) if isinstance(gkey, tuple) else str(gkey)
        for i in idxs:
            group_labels[i] = label
            if isinstance(gkey, tuple) and len(gkey) == 2:
                genotypes[i]  = str(gkey[0])
                treatments[i] = str(gkey[1])

    for i in range(n_animals):
        if group_labels[i] is None:
            group_labels[i] = "UNASSIGNED"

    return group_labels, genotypes, treatments  # type: ignore[return-value]


def build_subset_metrics_df(
    subset_name: str,
    speeds: list[list[np.ndarray]],
    nor_index: np.ndarray,
    group_data: dict,
    window_indices: Sequence[int] | None = None,
) -> "pd.DataFrame":
    """
    Build a tidy DataFrame of per-animal speed metrics for one subset.

    Parameters
    ----------
    subset_name : str   Label such as 'sal_within__mid'.
    speeds      : list[n_windows][n_animals] of 1D ndarrays.
    nor_index   : 1D ndarray of NOR scores aligned with animals.
    group_data  : {group_key: list[int]} mapping groups to animal indices.
    window_indices : which windows to pool; defaults to all.
    """
    if pd is None:
        raise ImportError("pandas is required")

    pooled = pool_speeds_per_animal(speeds, window_indices)
    n_animals = len(pooled)

    group_labels, genotypes, treatments = _infer_animal_group_labels(group_data, n_animals)

    rows = []
    for i, samples in enumerate(pooled):
        if samples.size == 0:
            continue
        m = compute_speed_metrics(samples)
        rows.append({
            "animal_id":           i,
            "group":               group_labels[i],
            "genotype":            genotypes[i],
            "treatment":           treatments[i],
            "subset":              subset_name,
            "nor":                 float(nor_index[i]),
            "speed_q01":           m.q01,
            "speed_q05":           m.q05,
            "speed_median":        m.q50,
            "speed_q90":           m.q90,
            "speed_q95":           m.q95,
            "speed_q99":           m.q99,
            "speed_width50":       m.width50,
            "speed_width_extreme": m.width_extreme,
            "speed_asymmetry":     m.asymmetry,
        })
    return pd.DataFrame(rows)


def compute_subset_metrics_with_segments(
    subset: str,
    speeds: list[list[np.ndarray]],
    nor_index: np.ndarray,
    group_data: dict,
    pool_split: str = "third",
) -> "pd.DataFrame":
    """
    Compute per-animal metrics for one subset, splitting windows into
    short / mid / long (or all / half) segments.

    Returns a tidy DataFrame with subset labels like 'subset__short'.
    """
    if pd is None:
        raise ImportError("pandas is required")

    n_windows = len(speeds)
    if n_windows == 0:
        return pd.DataFrame()

    i_third, i_half, i_two_third = cdf_split_indices(speeds)
    ranges = select_windows(pool_split, n_windows, i_third, i_half, i_two_third)

    dfs = []
    for seg_name, w_range in ranges.items():
        df_seg = build_subset_metrics_df(
            subset_name=f"{subset}__{seg_name}",
            speeds=speeds,
            nor_index=nor_index,
            group_data=group_data,
            window_indices=w_range,
        )
        dfs.append(df_seg)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def build_metrics_with_segments(
    speeds_by_subset: dict[str, list[list[np.ndarray]]],
    nor_index: np.ndarray,
    group_data: dict,
    pool_split: str = "third",
) -> "pd.DataFrame":
    """Build a combined tidy DataFrame across all subsets and window segments."""
    if pd is None:
        raise ImportError("pandas is required")

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
        raise ValueError("No metrics built — check speeds_by_subset.")
    return pd.concat(dfs, ignore_index=True)


def add_subset_segment_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Derive 'subset_base' and 'segment' columns from 'subset' labels
    like 'sal_within__short'. If no '__', segment='all'.
    Operates in-place and returns df for chaining.
    """
    parts = df["subset"].str.rsplit("__", n=1, expand=True)
    df["subset_base"] = parts.iloc[:, 0]
    df["segment"]     = parts.iloc[:, 1].fillna("all") if parts.shape[1] > 1 else "all"
    return df


# =============================================================================
# STATISTICS
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
    n = len(x)
    rho_obs, p_obs = spearmanr(x, y)
    boot_rhos = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_rhos[b], _ = spearmanr(x[idx], y[idx])
    lo, hi = np.percentile(boot_rhos, [ci_low, ci_high])
    return {
        "rho":           float(rho_obs),
        "p_value":       float(p_obs),
        "rho_boot_mean": float(boot_rhos.mean()),
        "ci_low":        float(lo),
        "ci_high":       float(hi),
    }


def bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 2000,
    ci: tuple[float, float] = (2.5, 97.5),
    random_state: int | None = RNG_SEED,
) -> dict:
    """
    Bootstrapped Spearman ρ with 95% CI.

    Uses joblib cache if configure_cache() has been called.
    Returns dict with keys: rho, p_value, rho_boot_mean, ci_low, ci_high.
    """
    global _cached_bootstrap_fn
    if _memory is not None and _memory.location is not None:
        if _cached_bootstrap_fn is None:
            _cached_bootstrap_fn = _memory.cache(_bootstrap_spearman_pure)
        func = _cached_bootstrap_fn
    else:
        func = _bootstrap_spearman_pure

    return func(
        np.asarray(x, float), np.asarray(y, float),
        int(n_boot), float(ci[0]), float(ci[1]), random_state,
    )


def compute_within_group_correlations(
    df: "pd.DataFrame",
    subset: str,
    metrics: Sequence[str] = ("speed_median", "speed_q95"),
    n_boot: int = 2000,
    random_state: int | None = RNG_SEED,
) -> "pd.DataFrame":
    """
    Spearman ρ + bootstrapped 95% CI for NOR vs each metric, within each group.

    Returns a tidy DataFrame with columns:
        subset, group, metric, n, rho, p_value, rho_boot_mean, ci_low, ci_high
    """
    if pd is None:
        raise ImportError("pandas is required")

    df_sub = df[df["subset"] == subset]
    results = []
    for group, dfg in df_sub.groupby("group"):
        if len(dfg) < 3:
            continue
        y = dfg["nor"].to_numpy()
        for metric in metrics:
            if metric not in dfg.columns:
                continue
            x = dfg[metric].to_numpy()
            stats = bootstrap_spearman(x, y, n_boot=n_boot, random_state=random_state)
            results.append({"subset": subset, "group": group, "metric": metric,
                             "n": len(dfg), **stats})
    return pd.DataFrame(results)


def fit_speed_nor_interaction(
    df: "pd.DataFrame",
    subset: str,
    metric: str = "speed_q95",
    ref_group: str = "WT_VEH",
) -> tuple:
    """
    Fit: NOR ~ metric * group  (OLS, ref_group as reference category).

    Returns
    -------
    model      : statsmodels OLS result
    slopes_df  : DataFrame with per-group slopes, SE, 95% CI, p-value
    """
    if not _statsmodels_available:
        raise ImportError("statsmodels is required for regression models")
    if pd is None:
        raise ImportError("pandas is required")

    df_sub = df[df["subset"] == subset].copy()
    if df_sub.empty:
        raise ValueError(f"No rows for subset={subset!r}")

    df_sub["group"] = pd.Categorical(df_sub["group"],
                                      categories=sorted(df_sub["group"].unique()))
    formula = f"nor ~ {metric} * C(group, Treatment(reference='{ref_group}'))"
    model = smf.ols(formula, data=df_sub).fit()

    params  = model.params
    cov     = model.cov_params()
    df_resid = model.df_resid
    groups  = list(df_sub["group"].cat.categories)

    rows = []
    for g in groups:
        L = np.zeros(len(params))
        L[params.index.get_loc(metric)] = 1.0
        if g != ref_group:
            key = f"{metric}:C(group, Treatment(reference='{ref_group}'))[T.{g}]"
            if key in params.index:
                L[params.index.get_loc(key)] = 1.0

        slope = float(L @ params.to_numpy())
        var   = float(L @ cov.to_numpy() @ L)
        se    = float(np.sqrt(max(var, 0.0)))
        t_val = slope / se if se > 0 else np.nan
        p_val = float(2 * student_t.sf(abs(t_val), df_resid)) if np.isfinite(t_val) else np.nan

        rows.append({
            "subset": subset, "metric": metric, "group": g, "ref_group": ref_group,
            "slope": slope, "se": se,
            "ci_low": slope - 1.96 * se, "ci_high": slope + 1.96 * se,
            "p_value": p_val, "df_resid": df_resid,
        })

    return model, pd.DataFrame(rows)


def fit_segment_group_interaction(
    df: "pd.DataFrame",
    subset_base: str,
    metric: str = "speed_q95",
    ref_group: str = "WT_VEH",
    ref_segment: str = "mid",
):
    """
    Fit: NOR ~ metric * group * segment  (3-way OLS interaction model).

    Returns statsmodels OLS result.
    """
    if not _statsmodels_available:
        raise ImportError("statsmodels is required")
    if pd is None:
        raise ImportError("pandas is required")

    if "subset_base" not in df.columns:
        df = add_subset_segment_columns(df)

    df_sub = df[df["subset_base"] == subset_base].copy()
    if df_sub.empty:
        raise ValueError(f"No rows for subset_base={subset_base!r}")

    df_sub["group"]   = pd.Categorical(df_sub["group"],   categories=sorted(df_sub["group"].unique()))
    df_sub["segment"] = pd.Categorical(df_sub["segment"], categories=sorted(df_sub["segment"].unique()))

    formula = (
        f"nor ~ {metric} * "
        f"C(group, Treatment(reference='{ref_group}')) * "
        f"C(segment, Treatment(reference='{ref_segment}'))"
    )
    return smf.ols(formula, data=df_sub).fit()


def leave_one_out_slopes(
    df: "pd.DataFrame",
    subset: str,
    metric: str = "speed_q95",
    ref_group: str = "WT_VEH",
) -> "pd.DataFrame":
    """
    LOO robustness: drop each animal in turn, refit NOR ~ metric * group,
    return per-group slopes for each leave-one-out iteration.
    """
    if pd is None:
        raise ImportError("pandas is required")

    df_sub = df[df["subset"] == subset].copy()
    if df_sub.empty:
        raise ValueError(f"No rows for subset={subset!r}")

    rows = []
    for aid in sorted(df_sub["animal_id"].unique()):
        _, slopes_df = fit_speed_nor_interaction(
            df_sub[df_sub["animal_id"] != aid],
            subset=subset, metric=metric, ref_group=ref_group,
        )
        for _, r in slopes_df.iterrows():
            rows.append({"subset": subset, "metric": metric,
                         "group": r["group"], "animal_id": aid,
                         "slope": r["slope"], "se": r["se"],
                         "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    return pd.DataFrame(rows)


def leave_one_out_slopes_all(
    df: "pd.DataFrame",
    metrics: Sequence[str] = PRIMARY_METRICS,
    ref_group: str = "WT_VEH",
    subsets: Sequence[str] | None = None,
) -> "pd.DataFrame":
    """Run leave_one_out_slopes for all subset × metric combinations."""
    if pd is None:
        raise ImportError("pandas is required")

    if subsets is None:
        subsets = sorted(df["subset"].unique())

    dfs = []
    for subset in subsets:
        if df[df["subset"] == subset].empty:
            continue
        for metric in metrics:
            if metric not in df.columns:
                continue
            print(f"[LOO] {subset} / {metric}")
            loo = leave_one_out_slopes(df, subset=subset, metric=metric, ref_group=ref_group)
            parts = subset.rsplit("__", 1)
            loo["subset_base"] = parts[0]
            loo["segment"]     = parts[1] if len(parts) > 1 else "all"
            dfs.append(loo)

    if not dfs:
        raise ValueError("No LOO slopes computed.")
    return pd.concat(dfs, ignore_index=True)


def summarize_segment_group_models(
    df: "pd.DataFrame",
    metrics: Sequence[str] = PRIMARY_METRICS,
    ref_group: str = "WT_VEH",
    ref_segment: str = "mid",
) -> "pd.DataFrame":
    """
    For each subset_base × metric, fit the 3-way interaction model and
    extract model fit statistics + minimum p-values for key contrasts.

    Returns one row per (subset_base, metric).
    """
    if pd is None:
        raise ImportError("pandas is required")

    if "subset_base" not in df.columns:
        df = add_subset_segment_columns(df)

    rows = []
    for subset_base in sorted(df["subset_base"].unique()):
        for metric in metrics:
            if metric not in df.columns:
                continue
            try:
                model = fit_segment_group_interaction(
                    df, subset_base=subset_base, metric=metric,
                    ref_group=ref_group, ref_segment=ref_segment,
                )
            except ValueError as e:
                print(f"[WARN] {subset_base} / {metric}: {e}")
                continue

            pvals = model.pvalues

            def _min_p(prefix: str, must_contain: str | None = None) -> float:
                ps = [pvals[n] for n in pvals.index
                      if n.startswith(prefix)
                      and (must_contain is None or must_contain in n)]
                return float(np.min(ps)) if ps else np.nan

            rows.append({
                "subset_base":               subset_base,
                "metric":                    metric,
                "n":                         int(model.nobs),
                "rsq":                       float(model.rsquared),
                "rsq_adj":                   float(model.rsquared_adj),
                "fvalue":                    float(model.fvalue) if model.fvalue is not None else np.nan,
                "f_pvalue":                  float(model.f_pvalue) if model.f_pvalue is not None else np.nan,
                "p_metric":                  float(pvals.get(metric, np.nan)),
                "p_metric_group_min":        _min_p(f"{metric}:C(group"),
                "p_metric_segment_min":      _min_p(f"{metric}:C(segment"),
                "p_metric_group_segment_min":_min_p(f"{metric}:C(group", must_contain="C(segment"),
                "ref_group":                 ref_group,
                "ref_segment":               ref_segment,
            })

    if not rows:
        raise ValueError("No segment×group models could be fitted.")
    return pd.DataFrame(rows)


def build_effect_summary(
    corr_summary: "pd.DataFrame",
    slopes_summary: "pd.DataFrame",
    loo_all: "pd.DataFrame",
) -> "pd.DataFrame":
    """
    Merge within-group correlations, group slopes, and LOO robustness into a
    single ranked summary per (subset_base, segment, metric, group).
    """
    if pd is None:
        raise ImportError("pandas is required")

    for df_part in (corr_summary, slopes_summary, loo_all):
        if "subset_base" not in df_part.columns:
            add_subset_segment_columns(df_part)

    # Ensure q_value / signif columns exist
    for df_part in (corr_summary, slopes_summary):
        df_part.setdefault("q_value",      np.nan)
        df_part.setdefault("signif_fdr_05", False)

    # LOO: aggregate per (subset, metric, group)
    loo_agg = (
        loo_all.groupby(["subset_base", "segment", "metric", "group"])["slope"]
        .agg(
            loo_n="count",
            loo_mean_slope="mean",
            loo_std_slope="std",
        )
        .reset_index()
    )
    loo_agg["loo_same_sign_rate"] = (
        loo_all.groupby(["subset_base", "segment", "metric", "group"])
        .apply(lambda g: (np.sign(g["slope"]) == np.sign(g["slope"].mean())).mean())
        .values
    )

    # Merge
    key_cols = ["subset_base", "segment", "metric", "group"]
    merged = (
        corr_summary[key_cols + ["rho_boot_mean", "q_value", "signif_fdr_05"]]
        .rename(columns={"rho_boot_mean": "corr_rho_boot",
                         "q_value": "corr_q",
                         "signif_fdr_05": "corr_signif"})
        .merge(
            slopes_summary[key_cols + ["slope", "q_value", "signif_fdr_05"]]
            .rename(columns={"q_value": "slope_q", "signif_fdr_05": "slope_signif"}),
            on=key_cols, how="outer",
        )
        .merge(loo_agg, on=key_cols, how="left")
    )

    merged["score"] = (
        merged["corr_signif"].fillna(False).astype(int)
        + merged["slope_signif"].fillna(False).astype(int)
        + merged["loo_same_sign_rate"].fillna(0)
    )
    return merged


def get_top_effects(
    effect_summary: "pd.DataFrame",
    n: int = 10,
    min_loo_same_sign: float = 0.0,
) -> "pd.DataFrame":
    """
    Return the top-n rows from effect_summary ranked by score,
    optionally filtered by minimum LOO same-sign rate.
    """
    df = effect_summary.copy()
    if min_loo_same_sign > 0:
        df = df[df["loo_same_sign_rate"].fillna(0) >= min_loo_same_sign]
    return df.sort_values("score", ascending=False).head(n)


# =============================================================================
# DOWNSAMPLING BOOTSTRAP  (Inès dataset — CI bands over full distribution)
# =============================================================================

def _downsample_once(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw k samples without replacement from 1-D array x."""
    n = x.size
    if k is None or k >= n:
        return x
    # rng.choice is faster when k/n < ~0.2; otherwise a permutation slice is faster
    if k / n < 0.2:
        return x[rng.choice(n, size=k, replace=False)]
    return x[rng.permutation(n)[:k]]


def _one_downsample_repeat(
    data: np.ndarray,
    q: np.ndarray,
    downsample_n: int | None,
    seed_seq: "np.random.SeedSequence",
) -> np.ndarray:
    """One repeat: optional downsample then compute percentiles q."""
    rng = np.random.default_rng(seed_seq)
    x = np.ravel(data)
    if downsample_n is not None and x.size > downsample_n:
        x = _downsample_once(x, downsample_n, rng)
    return np.percentile(x, q)


def bootstrap_downsampling_repeat(
    data: np.ndarray,
    percentiles: np.ndarray,
    repeat: int = 10_000,
    downsample_n: int | None = None,
    seed: int | None = 0,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate the variability of the speed percentile curve by repeated
    downsampling without replacement.

    For each of `repeat` independent draws:
      - subsample `downsample_n` points from the pooled speed distribution
      - compute the full percentile curve over `percentiles`

    The envelope across repeats (0th and 100th percentile) gives the CI band
    shown in the distribution figures.  Using downsampling (rather than
    resampling with replacement) controls for bias introduced by unequal group
    sizes — the effective sample size is equalised across groups via
    `downsample_factor`.

    Parameters
    ----------
    data        : 1-D array of speed samples (all animals × all windows pooled)
    percentiles : grid of percentile values, e.g. np.linspace(0, 100, 100)
    repeat      : number of downsampling repeats (default 10 000)
    downsample_n: target subsample size.  If None, no downsampling is applied.
    seed        : integer seed for reproducibility
    n_jobs      : parallel workers (joblib)

    Returns
    -------
    ci_low   : shape (K,) — 0th-percentile envelope across repeats
    ci_high  : shape (K,) — 100th-percentile envelope across repeats
    ci_matrix: shape (repeat, K) — all repeat curves (stored in PKL for plotting)
    """
    try:
        from joblib import Parallel, delayed as jdelayed
        _joblib = True
    except ImportError:
        _joblib = False

    q = np.asarray(percentiles, dtype=float).ravel()
    if data.size == 0:
        nan = np.full(q.shape, np.nan, float)
        return nan, nan, np.empty((0, q.size), float)

    base_ss = np.random.SeedSequence(None if seed is None else int(seed))
    child_ss = base_ss.spawn(repeat)

    if _joblib and n_jobs != 1:
        rows = Parallel(n_jobs=n_jobs, prefer="processes")(
            jdelayed(_one_downsample_repeat)(data, q, downsample_n, child_ss[i])
            for i in range(repeat)
        )
    else:
        rows = [_one_downsample_repeat(data, q, downsample_n, child_ss[i])
                for i in range(repeat)]

    ci_matrix = np.vstack(rows)                                  # (repeat, K)
    ci_low, ci_high = np.percentile(ci_matrix, [0, 100], axis=0)
    return ci_low, ci_high, ci_matrix


def compute_bootstrap_ci_bands(
    ranges: dict[str, range],
    pooled_group_speed_by_segment: dict,
    group_data: dict,
    percentiles: np.ndarray,
    repeat: int = 10_000,
    downsample_factor: int = 10,
    seed: int = 0,
    n_jobs: int = 1,
    verbose: bool = False,
) -> tuple[dict, dict, dict]:
    """
    Run ``bootstrap_downsampling_repeat`` for every (segment, group) combination.

    For each group the pooled speed distribution (all animals × all windows in
    the segment) is downsampled to ``N // downsample_factor`` per repeat.  This
    equalises effective sample size across groups that differ in animal count.

    Parameters
    ----------
    ranges        : segment → range of window indices (from ``select_windows``)
    pooled_group_speed_by_segment
                  : {seg: {group_key: 1-D ndarray of all speed samples}}
    group_data    : {group_key: array of animal indices} — used for iteration order
    percentiles   : grid passed to ``bootstrap_downsampling_repeat``
    repeat        : bootstrap repeats per (segment, group)
    downsample_factor : divide N by this to get subsample size
    seed          : base random seed
    n_jobs        : joblib workers inside each bootstrap call
    verbose       : print timing per (segment, group)

    Returns
    -------
    ci_low_repeat  : {seg: {group: ndarray shape (K,)}}
    ci_high_repeat : {seg: {group: ndarray shape (K,)}}
    ci_matrix_dict : {seg: {group: ndarray shape (repeat, K)}}
                     stored as ``ci_btr_downsample_repeat`` in the PKL
    """
    import time as _time

    q = np.asarray(percentiles, dtype=float).ravel()

    ci_low_repeat: dict  = {}
    ci_high_repeat: dict = {}
    ci_matrix_dict: dict = {}

    base_ss = np.random.SeedSequence(None if seed is None else int(seed))
    pair_list = [(seg, gt) for seg in ranges for gt in group_data]
    child_ss  = iter(base_ss.spawn(len(pair_list)))

    for seg_name in ranges:
        ci_low_seg:    dict = {}
        ci_high_seg:   dict = {}
        ci_matrix_seg: dict = {}

        for gt in group_data:
            if seg_name not in pooled_group_speed_by_segment or \
               gt not in pooled_group_speed_by_segment[seg_name]:
                raise KeyError(
                    f"Missing pooled speed data for segment='{seg_name}', group='{gt}'"
                )

            data = np.ravel(pooled_group_speed_by_segment[seg_name][gt])
            ds_n = max(1, int(data.size // downsample_factor)) \
                   if (downsample_factor and downsample_factor > 1) else None

            ss   = next(child_ss)
            t0   = _time.time()
            lo, hi, mat = bootstrap_downsampling_repeat(
                data=data,
                percentiles=q,
                repeat=repeat,
                downsample_n=ds_n,
                seed=ss.entropy,
                n_jobs=n_jobs,
            )
            if verbose:
                print(
                    f"  [{seg_name} | {_pretty_label(gt)}] "
                    f"n={data.size:,}  ds_n={ds_n}  "
                    f"repeats={repeat}  {_time.time()-t0:.1f}s"
                )

            ci_low_seg[gt]    = lo
            ci_high_seg[gt]   = hi
            ci_matrix_seg[gt] = mat

        ci_low_repeat[seg_name]  = ci_low_seg
        ci_high_repeat[seg_name] = ci_high_seg
        ci_matrix_dict[seg_name] = ci_matrix_seg

    return ci_low_repeat, ci_high_repeat, ci_matrix_dict


# =============================================================================
# WINDOW × PERCENTILE CORRELATION ANALYSIS
# =============================================================================

def compute_window_nor_correlations(
    speeds_by_subset: dict,
    nor_index: np.ndarray,
    group_data: dict,
    time_windows_range: np.ndarray,
    ranges: dict[str, range],
    q_grid: np.ndarray | None = None,
) -> "pd.DataFrame":
    """
    For every subset × window × percentile × group, compute Spearman and Pearson
    correlation between the per-animal speed percentile value and NOR.

    This produces the data that feeds the 'correlation vs window size' figures —
    a higher-resolution view than the collapsed scalar metrics, showing exactly
    at which window size and which part of the speed distribution the NOR
    relationship is strongest.

    Parameters
    ----------
    speeds_by_subset : dict
        subset → list[list[ndarray]]   (windows × animals × speed samples)
    nor_index : ndarray shape (n_animals,)
        NOR index aligned with animal order.
    group_data : dict
        group_key → array of animal indices.
    time_windows_range : ndarray
        Window sizes (e.g. np.arange(5, 100, 1)).  Must align with speeds axis 0.
    ranges : dict[str, range]
        Segment name → range of window indices, e.g. {"short": range(0,26), ...}
        Used to add pooled (short/mid/long) rows alongside per-window rows.
    q_grid : ndarray, optional
        Percentiles to compute (0–100).  Defaults to [1,5,25,50,75,95,99].

    Returns
    -------
    DataFrame with columns:
        roi, window, q, group,
        spearman_rho, spearman_p,
        pearson_r,    pearson_p
    where ``window`` is either an integer (specific window size) or a segment
    name string ('short', 'mid', 'long') for the pooled summary rows.
    """
    if pd is None:
        raise ImportError("pandas required")

    from scipy.stats import pearsonr

    if q_grid is None:
        q_grid = np.array([1, 5, 25, 50, 75, 95, 99], dtype=float)

    rows = []

    for roi, speeds in speeds_by_subset.items():
        n_windows = len(speeds)
        n_animals = len(speeds[0]) if n_windows > 0 else 0

        # ── Per-window rows ──────────────────────────────────────────────────
        for w_idx, win_size in enumerate(time_windows_range[:n_windows]):
            # shape (n_animals, n_q): percentile values per animal at this window
            q_vals = np.full((n_animals, len(q_grid)), np.nan)
            for i in range(n_animals):
                samples = speeds[w_idx][i]
                if samples.size >= 3:
                    q_vals[i, :] = np.percentile(samples, q_grid)

            for qi, q_val in enumerate(q_grid):
                x = q_vals[:, qi]

                # All-animals row (group = '__ALL__')
                _append_cor_row(rows, roi=roi, window=int(win_size), q=q_val,
                                group="__ALL__", x=x, y=nor_index)

                # Per-group rows
                for grp, idxs in group_data.items():
                    idxs = np.asarray(idxs, dtype=int)
                    mask = idxs[idxs < n_animals]
                    if len(mask) < 3:
                        continue
                    _append_cor_row(rows, roi=roi, window=int(win_size), q=q_val,
                                    group=_pretty_label(grp),
                                    x=x[mask], y=nor_index[mask])

        # ── Pooled (segment) rows ────────────────────────────────────────────
        for seg_name, seg_range in ranges.items():
            # Pool all speed samples across windows in segment, per animal
            q_vals_seg = np.full((n_animals, len(q_grid)), np.nan)
            for i in range(n_animals):
                pooled = np.concatenate([
                    speeds[j][i]
                    for j in seg_range
                    if j < n_windows and speeds[j][i].size > 0
                ]) if any(j < n_windows for j in seg_range) else np.array([])
                if pooled.size >= 3:
                    q_vals_seg[i, :] = np.percentile(pooled, q_grid)

            for qi, q_val in enumerate(q_grid):
                x = q_vals_seg[:, qi]

                _append_cor_row(rows, roi=roi, window=seg_name, q=q_val,
                                group="__ALL__", x=x, y=nor_index)

                for grp, idxs in group_data.items():
                    idxs = np.asarray(idxs, dtype=int)
                    mask = idxs[idxs < n_animals]
                    if len(mask) < 3:
                        continue
                    _append_cor_row(rows, roi=roi, window=seg_name, q=q_val,
                                    group=_pretty_label(grp),
                                    x=x[mask], y=nor_index[mask])

    return pd.DataFrame(rows)


def _append_cor_row(
    rows: list,
    roi: str,
    window,
    q: float,
    group: str,
    x: np.ndarray,
    y: np.ndarray,
) -> None:
    """Compute Spearman + Pearson and append one row dict to rows in-place."""
    from scipy.stats import pearsonr

    valid = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[valid], y[valid]
    n = int(valid.sum())

    if n < 3:
        rows.append({
            "roi": roi, "window": window, "q": q, "group": group,
            "n": n,
            "spearman_rho": np.nan, "spearman_p": np.nan,
            "pearson_r":    np.nan, "pearson_p":  np.nan,
        })
        return

    sp_rho, sp_p = spearmanr(xv, yv)
    pe_r,   pe_p = pearsonr(xv, yv)

    rows.append({
        "roi": roi, "window": window, "q": q, "group": group,
        "n": n,
        "spearman_rho": float(sp_rho), "spearman_p": float(sp_p),
        "pearson_r":    float(pe_r),   "pearson_p":  float(pe_p),
    })


def plot_window_nor_correlations(
    df_cor: "pd.DataFrame",
    metric: str = "spearman",
    alpha: float = 0.05,
    fdr: bool = True,
    grid_cols: int = 3,
    save_dir: Path | None = None,
) -> None:
    """
    Generate three families of figures from the window×percentile correlation
    DataFrame produced by ``compute_window_nor_correlations``:

    1. ``cor_bywin_<roi>_<metric>.png``
       One figure per ROI.  X = window size, Y = correlation coefficient.
       One line per percentile (q).  Filled markers = significant after FDR.

    2. ``cor_pooled_<roi>_<metric>.png``
       One figure per ROI.  X = segment (short/mid/long), Y = correlation.
       One line per percentile.

    3. ``cor_bywin_group_<roi>_<metric>.png``
       Grid figure, one subplot per percentile.  Inside each subplot: one
       line per experimental group vs window size.

    Parameters
    ----------
    df_cor   : output of compute_window_nor_correlations
    metric   : 'spearman' | 'pearson'
    alpha    : significance threshold (after FDR if fdr=True)
    fdr      : apply Benjamini-Hochberg correction per ROI
    grid_cols: columns in the group-grid figure
    save_dir : directory for PNGs (skips saving if None)
    """
    if pd is None:
        raise ImportError("pandas required")

    rho_col = "spearman_rho" if metric == "spearman" else "pearson_r"
    p_col   = "spearman_p"   if metric == "spearman" else "pearson_p"

    def _fdr_bh(pvals: np.ndarray) -> np.ndarray:
        """Benjamini-Hochberg FDR correction."""
        finite = np.isfinite(pvals)
        adj = np.full_like(pvals, np.nan)
        pv = pvals[finite]
        n = pv.size
        if n == 0:
            return adj
        order = np.argsort(pv)
        pv_s = pv[order]
        cmin = 1.0
        adj_s = np.empty(n)
        for i in range(n - 1, -1, -1):
            cmin = min(cmin, pv_s[i] * n / (i + 1))
            adj_s[i] = cmin
        adj_full = np.empty(n)
        adj_full[order] = adj_s
        adj[finite] = adj_full
        return adj

    prop_cycle = plt.rcParams.get("axes.prop_cycle", None)
    BASE_COLORS = (
        prop_cycle.by_key()["color"] if prop_cycle is not None
        else ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
              "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
    )

    def _color_map(keys):
        return {k: BASE_COLORS[i % len(BASE_COLORS)] for i, k in enumerate(sorted(keys))}

    seg_order = ["short", "mid", "long", "all"]

    # Split into by-window rows (window is int) and pooled rows (window is str)
    df_cor = df_cor.copy()
    df_cor["_is_pooled"] = df_cor["window"].apply(lambda w: isinstance(w, str))

    for roi, df_roi in df_cor.groupby("roi"):
        qs = sorted(df_roi["q"].unique())
        color_q     = _color_map(qs)
        groups_all  = sorted(df_roi[df_roi["group"] != "__ALL__"]["group"].unique())
        color_group = _color_map(groups_all) if groups_all else {}

        df_bywin   = df_roi[~df_roi["_is_pooled"]]
        df_pooled  = df_roi[df_roi["_is_pooled"]]

        # ── Figure 1: by-window, all animals ─────────────────────────────────
        df_all = df_bywin[df_bywin["group"] == "__ALL__"]
        if not df_all.empty:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.set_title(f"{roi} | {metric.capitalize()} vs window")

            # Collect all p-values for FDR across this ROI
            all_p   = df_all[p_col].to_numpy()
            all_adj = _fdr_bh(all_p) if fdr else all_p
            df_all  = df_all.copy()
            df_all["_p_adj"] = all_adj

            for q_val in qs:
                sub = df_all[df_all["q"] == q_val].sort_values("window")
                wins  = sub["window"].to_numpy()
                corrs = sub[rho_col].to_numpy()
                padj  = sub["_p_adj"].to_numpy()
                color = color_q[q_val]
                ax.plot(wins, corrs, color=color, label=f"q{int(q_val)}")
                sig = np.isfinite(padj) & (padj <= alpha)
                ax.scatter(wins[sig],  corrs[sig],  color=color, s=18, zorder=3)
                ax.scatter(wins[~sig], corrs[~sig], color=color, s=18,
                           facecolors="none", zorder=3)
            ax.axhline(0, color="k", lw=0.8, alpha=0.4)
            ax.set_xlabel("Window size (TRs)")
            ax.set_ylabel(f"{metric.capitalize()} ρ")
            ax.legend(title="Percentile", loc="best", fontsize=7)
            fig.tight_layout()
            if save_dir:
                fig.savefig(save_dir / f"cor_bywin_{roi}_{metric}.png",
                            dpi=150, bbox_inches="tight")
            plt.close(fig)

        # ── Figure 2: pooled (short/mid/long) ────────────────────────────────
        df_pool_all = df_pooled[df_pooled["group"] == "__ALL__"]
        if not df_pool_all.empty:
            pools_present = [s for s in seg_order if s in df_pool_all["window"].values]
            x_pos = np.arange(len(pools_present))
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.set_title(f"{roi} | {metric.capitalize()} pooled segments")
            for q_val in qs:
                sub_q = df_pool_all[df_pool_all["q"] == q_val]
                corrs = [sub_q[sub_q["window"] == p][rho_col].values[0]
                         if p in sub_q["window"].values else np.nan
                         for p in pools_present]
                padj  = _fdr_bh(sub_q[p_col].to_numpy()) if fdr \
                        else sub_q[p_col].to_numpy()
                sig   = [np.isfinite(v) and v <= alpha for v in padj]
                color = color_q[q_val]
                ax.plot(x_pos, corrs, "o-", color=color, label=f"q{int(q_val)}")
                for xi, yi, s in zip(x_pos, corrs, sig):
                    if not np.isfinite(yi):
                        continue
                    ax.scatter([xi], [yi], color=color, s=30,
                               facecolors=(color if s else "none"), zorder=3)
            ax.axhline(0, color="k", lw=0.8, alpha=0.4)
            ax.set_xticks(x_pos, pools_present)
            ax.set_xlabel("Segment")
            ax.set_ylabel(f"{metric.capitalize()} ρ")
            ax.legend(title="Percentile", loc="best", fontsize=7)
            fig.tight_layout()
            if save_dir:
                fig.savefig(save_dir / f"cor_pooled_{roi}_{metric}.png",
                            dpi=150, bbox_inches="tight")
            plt.close(fig)

        # ── Figure 3: group grid (one subplot per percentile) ─────────────────
        df_grp = df_bywin[df_bywin["group"] != "__ALL__"]
        if not df_grp.empty and groups_all:
            n_q   = len(qs)
            cols  = max(1, grid_cols)
            rows_n = int(np.ceil(n_q / cols))
            fig, axes = plt.subplots(rows_n, cols,
                                     figsize=(5 * cols, 3.8 * rows_n),
                                     squeeze=False)
            for qi_idx, q_val in enumerate(qs):
                r, c = divmod(qi_idx, cols)
                ax = axes[r, c]
                sub_q = df_grp[df_grp["q"] == q_val]
                # FDR across all groups × windows for this quantile
                if fdr:
                    all_p_q  = sub_q[p_col].to_numpy()
                    all_adj_q = _fdr_bh(all_p_q)
                    sub_q = sub_q.copy()
                    sub_q["_p_adj"] = all_adj_q
                else:
                    sub_q = sub_q.copy()
                    sub_q["_p_adj"] = sub_q[p_col]

                for grp in groups_all:
                    sg = sub_q[sub_q["group"] == grp].sort_values("window")
                    if sg.empty:
                        continue
                    wins  = sg["window"].to_numpy()
                    corrs = sg[rho_col].to_numpy()
                    padj  = sg["_p_adj"].to_numpy()
                    color = color_group.get(grp, BASE_COLORS[0])
                    ax.plot(wins, corrs, color=color, label=str(grp))
                    sig = np.isfinite(padj) & (padj <= alpha)
                    ax.scatter(wins[sig],  corrs[sig],  color=color, s=14, zorder=3)
                    ax.scatter(wins[~sig], corrs[~sig], color=color, s=14,
                               facecolors="none", zorder=3)
                ax.axhline(0, color="k", lw=0.8, alpha=0.4)
                ax.set_title(f"q{int(q_val)}", fontsize=9)
                ax.set_xlabel("Window (TRs)", fontsize=8)
                ax.set_ylabel(metric.capitalize(), fontsize=8)

            # Hide unused subplots
            for k in range(n_q, rows_n * cols):
                r, c = divmod(k, cols)
                axes[r, c].axis("off")

            # Shared legend
            handles, labels = axes[0, 0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper right",
                           title="Group", fontsize=7)
            fig.suptitle(f"{roi} | {metric.capitalize()} vs window (by group)")
            fig.tight_layout(rect=[0, 0, 0.97, 0.95])
            if save_dir:
                fig.savefig(save_dir / f"cor_bywin_group_{roi}_{metric}.png",
                            dpi=150, bbox_inches="tight")
            plt.close(fig)


# =============================================================================
# QUANTILE TENSOR
# =============================================================================

def _safe_percentiles(x: np.ndarray, q_grid: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.full(len(q_grid), np.nan, dtype=float)
    return np.percentile(x, q_grid)


def compute_quantile_tensor(
    speeds: list[list[np.ndarray]],
    q_grid: np.ndarray,
) -> np.ndarray:
    """
    Compute Q tensor of shape (n_sessions, n_windows, n_q).

    Q[i, j, k] = speed value at percentile q_grid[k] for session i at window j.
    """
    n_windows  = len(speeds)
    n_sessions = len(speeds[0])
    Q = np.full((n_sessions, n_windows, len(q_grid)), np.nan, dtype=np.float32)
    for j in range(n_windows):
        for i in range(n_sessions):
            Q[i, j, :] = _safe_percentiles(speeds[j][i], q_grid).astype(np.float32)
    return Q


def save_quantile_npz(
    outpath: Path | str,
    Q: np.ndarray,
    q_grid: np.ndarray,
    time_windows_range: np.ndarray,
    session_name: np.ndarray,
    genotype: np.ndarray | None = None,
    treatment: np.ndarray | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save quantile tensor + metadata to a compressed NPZ file."""
    payload: dict[str, Any] = {
        "Q":                  Q.astype(np.float32),
        "q_grid":             q_grid.astype(np.float32),
        "time_windows_range": time_windows_range.astype(np.int32),
        "session_name":       session_name.astype(str),
    }
    if genotype  is not None: payload["genotype"]  = genotype.astype(str)
    if treatment is not None: payload["treatment"] = treatment.astype(str)
    if extra:
        payload.update(extra)
    np.savez_compressed(outpath, **payload)
    print(f"[OK] Saved quantile tensor → {outpath}")


# =============================================================================
# PLOTTING
# =============================================================================

_TAB20 = plt.cm.tab20(np.linspace(0, 1, 20))
_TAB10 = plt.cm.tab10.colors

AGE_CONTRAST_PALETTE = [
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
    "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
]


def _pretty_label(gt) -> str:
    if isinstance(gt, tuple):
        return " | ".join(str(v) for v in gt)
    return str(gt)


def _age_contrast_label(gt_4m) -> str:
    """Label for a 4M group key — drop age tokens and join remaining factors."""
    if isinstance(gt_4m, str):
        return "4M-2M"
    parts = [str(v) for v in gt_4m if v not in ("2M", "4M")]
    return " | ".join(parts) if parts else "all"


def _make_age_contrast_color_map(
    group_keys, groups_selected: str
) -> dict[str, str]:
    """Map each non-age label to a fixed color from AGE_CONTRAST_PALETTE."""
    example = next(iter(group_keys))
    if groups_selected == "age" and isinstance(example, str):
        return {"4M-2M": AGE_CONTRAST_PALETTE[0]}
    labels = []
    for k in group_keys:
        if isinstance(k, str) or "4M" not in k:
            continue
        lbl = _age_contrast_label(k)
        if lbl not in labels:
            labels.append(lbl)
    return {lbl: AGE_CONTRAST_PALETTE[i % len(AGE_CONTRAST_PALETTE)]
            for i, lbl in enumerate(labels)}


def plot_group_speed_distributions(
    group_means: dict,
    centers: np.ndarray,
    seg_names: list[str],
    groups_selected: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Two-row figure (linear + log scale) of group mean speed distributions
    per window segment.

    Parameters
    ----------
    group_means : dict mapping seg_name → {group_key: mean_hist ndarray}
    centers     : bin-center values for the speed axis
    seg_names   : ordered list of segment names to plot as columns
    save_path   : if given, figure is saved here (300 dpi)
    """
    n_seg = len(seg_names)
    fig, axes = plt.subplots(2, n_seg, figsize=(6 * n_seg, 8), sharex=True)
    if n_seg == 1:
        axes = np.array([axes]).reshape(2, 1)

    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=_TAB20)

    for col, seg in enumerate(seg_names):
        means = group_means.get(seg, {})

        for scale_row, (ax, use_log) in enumerate(
            zip(axes[:, col], [False, True])
        ):
            for gt, hist in means.items():
                ax.plot(centers, hist, lw=1.2, alpha=0.8,
                        label=_pretty_label(gt) if not scale_row else None)

            ax.set_xlabel("Speed")
            ax.set_ylabel("Density" + (" (log)" if use_log else ""))
            ax.set_title(f"{seg} ({'log' if use_log else 'linear'} scale)")
            ax.grid(True, which="both", ls="--", lw=0.4)
            if use_log:
                ax.set_yscale("log")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        dict(zip(labels, handles)).values(),
        dict(zip(labels, handles)).keys(),
        title=groups_selected,
        loc="center left", bbox_to_anchor=(0.92, 0.5),
        fontsize=10, frameon=False,
    )
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_ci_bands(
    ci_low: dict,   # seg_name → {group_key: ndarray}
    ci_high: dict,
    percentiles_: np.ndarray,
    seg_names: list[str],
    groups_selected: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Two-row figure (linear + log y-scale) of bootstrapped CI bands
    (percentile-vs-speed) per segment, per group.
    """
    n_seg = len(seg_names)
    fig, axes = plt.subplots(2, n_seg, figsize=(6 * n_seg, 8))
    if n_seg == 1:
        axes = np.array([axes]).reshape(2, 1)

    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=_TAB20)

    handles_all: list = []
    labels_all:  list = []

    for col, seg in enumerate(seg_names):
        lo = ci_low.get(seg, {})
        hi = ci_high.get(seg, {})

        for scale_row, (ax, use_log) in enumerate(zip(axes[:, col], [False, True])):
            ax.set_title(f"{seg} ({'log' if use_log else 'linear'})")
            ax.set_xlabel("Percentiles")
            ax.set_ylabel("Speed" + (" (log)" if use_log else ""))
            ax.grid(True, which="both", ls="--", lw=0.4)
            if use_log:
                ax.set_yscale("log")

            for gt in lo:
                band = ax.fill_between(
                    percentiles_, lo[gt], hi[gt],
                    alpha=0.6, label=_pretty_label(gt),
                )
                if not scale_row:
                    handles_all.append(band)
                    labels_all.append(_pretty_label(gt))

    uniq = dict(zip(labels_all, handles_all))
    fig.legend(
        uniq.values(), uniq.keys(),
        title=groups_selected, loc="center left", bbox_to_anchor=(0.92, 0.5),
        fontsize=10, frameon=False,
    )
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_age_contrasts(
    ci_low: dict,   # seg_name → {group_key: ndarray}
    ci_high: dict,
    percentiles_: np.ndarray,
    seg_names: list[str],
    group_data: dict,
    groups_selected: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    One-row figure showing 4M − 2M bootstrapped CI difference bands per segment.
    Each non-age sub-group gets its own color from AGE_CONTRAST_PALETTE.
    """
    n_seg = len(seg_names)
    fig, axes = plt.subplots(1, n_seg, figsize=(6 * n_seg, 5),
                              sharex=True, sharey=True)
    if n_seg == 1:
        axes = [axes]

    color_map = _make_age_contrast_color_map(group_data.keys(), groups_selected)

    for ax, seg in zip(axes, seg_names):
        lo = ci_low.get(seg, {})
        hi = ci_high.get(seg, {})

        for gt in lo:
            # ---- pure-age grouping ----
            if groups_selected == "age":
                if gt != "4M":
                    continue
                gt_2m = "2M"
                if gt_2m not in lo:
                    continue
                diff_lo = lo["4M"] - hi["2M"]
                diff_hi = hi["4M"] - lo["2M"]
                color = color_map.get("4M-2M", "tab:blue")
                label = "4M−2M"

            # ---- multi-factor groupings ----
            else:
                if not isinstance(gt, tuple) or "4M" not in gt:
                    continue
                gt_2m = tuple("2M" if v == "4M" else v for v in gt)
                if gt_2m not in lo:
                    continue
                diff_lo = lo[gt] - hi[gt_2m]
                diff_hi = hi[gt] - lo[gt_2m]
                label   = _age_contrast_label(gt)
                color   = color_map.get(label, "gray")

            ax.fill_between(percentiles_, diff_lo, diff_hi,
                            alpha=0.45, color=color, label=label)
            ax.axhline(0, color="black", lw=0.8)

        ax.set_title(seg)
        ax.set_xlabel("Percentiles")
        ax.set_ylabel("Δ Speed (4M − 2M)")
        ax.grid(True, which="both", ls="--", lw=0.4)
        ax.legend(frameon=False, fontsize=9)

    fig.suptitle(f"Age contrast (4M − 2M) — {groups_selected}", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def _flatten_animal_entry(entry) -> np.ndarray:
    """
    Flatten one animal's speed entry into a 1D float array.

    The confirmed PKL structure is:
        group_speed_by_segment[seg][gt]
          -> tuple(1,)
            -> ndarray shape (n_animals, n_windows)  dtype=object
              -> cell [i, j]: ndarray shape (n_samples,)  dtype=float64

    So when iterating over animals, each `entry` is a 1D object array
    of length n_windows, where entry[j] is a float64 array of samples.
    This function concatenates all windows into one flat float array.
    """
    if isinstance(entry, np.ndarray):
        if entry.dtype.kind in ("f", "i", "u"):
            # Already a numeric array — just flatten
            return entry.ravel().astype(float)
        # Object array: each element is a numeric array (one per window)
        parts = [np.asarray(entry[j], dtype=float).ravel()
                 for j in range(entry.size)]
        return np.concatenate(parts) if parts else np.empty(0, dtype=float)

    # Plain Python list/tuple of arrays (fallback)
    parts = [np.asarray(item, dtype=float).ravel() for item in entry]
    return np.concatenate(parts) if parts else np.empty(0, dtype=float)


def plot_per_animal_histograms(
    group_speed_by_segment: dict,
    seg_names: list[str],
    group_data: dict,
    save_dir: Path | None = None,
    bins: int = 100,
) -> None:
    """
    For each group × segment, plot overlapping per-animal speed histograms.
    Saves one PNG per (group, segment) into save_dir if given.

    Accepts group_speed_by_segment structures from the bootstrap PKL, which
    may be:
      seg → gt → list[n_animals]  of 1D arrays          (flat)
      seg → gt → (list[n_animals] of lists of arrays,)   (wrapped in tuple)
    """
    for seg in seg_names:
        seg_data = group_speed_by_segment.get(seg, {})
        for gt, speeds_entry in seg_data.items():

            # Structure: tuple(1,) -> ndarray(n_animals, n_windows, dtype=object)
            # Unwrap the outer tuple
            if isinstance(speeds_entry, (tuple, list)) and len(speeds_entry) == 1:
                animal_matrix = speeds_entry[0]  # shape (n_animals, n_windows), dtype=object
            else:
                animal_matrix = speeds_entry

            # animal_matrix[i] is a 1D object array of n_windows speed arrays
            # _flatten_animal_entry concatenates them into one flat float array
            flat_per_animal: list[np.ndarray] = [
                _flatten_animal_entry(animal_matrix[i])
                for i in range(len(animal_matrix))
            ]
            flat_per_animal = [a for a in flat_per_animal if a.size > 0]
            if not flat_per_animal:
                continue

            all_flat = np.concatenate(flat_per_animal)
            all_flat = all_flat[np.isfinite(all_flat)]
            if all_flat.size == 0:
                continue

            sp_min = float(np.nanmin(all_flat))
            sp_max = float(np.nanmax(all_flat))

            fig, ax = plt.subplots(figsize=(10, 5))
            for i, flat in enumerate(flat_per_animal):
                flat = flat[np.isfinite(flat)]
                if flat.size == 0:
                    continue
                counts, edges = np.histogram(flat, bins=bins, range=(sp_min, sp_max))
                ax.plot(edges[:-1], counts, ".-", alpha=0.5, label=f"Animal {i}")

            ax.set_xlabel("Speed")
            ax.set_ylabel("Count")
            ax.set_title(f"Per-animal speed histograms — {seg} — {_pretty_label(gt)}")
            ax.set_xlim(0.1, 1.2)
            plt.tight_layout()

            if save_dir:
                gt_str = str(gt).translate(str.maketrans("", "", "(),'\" "))
                fig.savefig(save_dir / f"animal_histograms_{seg}_{gt_str}.png", dpi=200)
            plt.close(fig)


def plot_qc_3panel(
    speeds: list[list[np.ndarray]],
    time_windows_range: np.ndarray,
    group_data: dict,
    subset_label: str,
    dataset_name: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    3-panel QC figure: per-animal mean speed, pooled median, and distribution
    spread (99th−1st pct) vs window size — one curve per group.
    """
    n_windows = len(speeds)
    n_animals = len(speeds[0])
    group_keys = list(group_data.keys())

    # Per-window per-animal means
    animal_means = [
        np.array([float(np.mean(speeds[j][i])) for i in range(n_animals)])
        for j in range(n_windows)
    ]

    # Percentile tracks per group
    qs = (1, 50, 99)
    pct_tracks: dict = {}
    for gt, idxs in group_data.items():
        qdict = {q: [] for q in qs}
        for j in range(n_windows):
            if len(idxs) == 0:
                for q in qs:
                    qdict[q].append(np.nan)
                continue
            flat = np.concatenate([speeds[j][i].ravel() for i in idxs])
            flat = flat[np.isfinite(flat)]
            ps = np.percentile(flat, qs) if flat.size else [np.nan] * 3
            for q, p in zip(qs, ps):
                qdict[q].append(p)
        pct_tracks[gt] = {q: np.array(v) for q, v in qdict.items()}

    cmap = plt.cm.get_cmap("tab10", len(group_keys))
    colors = {gt: cmap(i) for i, gt in enumerate(group_keys)}

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

    # Panel A: per-animal
    for gt, idxs in group_data.items():
        col = colors[gt]
        for i in idxs:
            axA.plot(time_windows_range,
                     [animal_means[j][i] for j in range(n_windows)],
                     color=col, alpha=0.25, lw=1.0)
    axA.set_title("Per-animal mean speed vs window")
    axA.set_xlabel("Window size (TR)")
    axA.set_ylabel("Mean dFC speed")
    axA.grid(alpha=0.3)

    # Panel B: pooled median
    for gt, qd in pct_tracks.items():
        axB.plot(time_windows_range, qd[50], "o-", ms=2, lw=1.5,
                 color=colors[gt], label=_pretty_label(gt))
    axB.set_title("Pooled median dFC speed vs window")
    axB.set_xlabel("Window size (TR)")
    axB.set_ylabel("Median dFC speed")
    axB.grid(alpha=0.3)

    # Panel C: spread
    for gt, qd in pct_tracks.items():
        axC.plot(time_windows_range, qd[99] - qd[1], "o-", ms=2, lw=1.5,
                 color=colors[gt], label=_pretty_label(gt))
    axC.set_title("Spread (99th−1st pct) vs window")
    axC.set_xlabel("Window size (TR)")
    axC.set_ylabel("Distribution width")
    axC.legend(title="Group", frameon=False, fontsize=9)
    axC.grid(alpha=0.3)

    fig.suptitle(f"QC — {dataset_name} / subset='{subset_label}'", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def _p_to_star(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def plot_nor_vs_metric_by_group(
    df: "pd.DataFrame",
    subset: str,
    metric: str = "speed_q95",
    model=None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Scatter plot of NOR vs dFC speed metric, one series per group,
    with fitted regression lines from the interaction model.
    """
    df_sub = df[df["subset"] == subset].copy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    if df_sub.empty:
        ax.set_visible(False)
        return fig, ax

    groups = sorted(df_sub["group"].unique())
    cmap = plt.cm.get_cmap("tab10", len(groups))
    for i, g in enumerate(groups):
        dfg = df_sub[df_sub["group"] == g]
        ax.scatter(dfg[metric], dfg["nor"],
                   label=g, alpha=0.8, color=cmap(i), edgecolor="none")

    if model is None and _statsmodels_available:
        model, _ = fit_speed_nor_interaction(df, subset=subset,
                                              metric=metric, ref_group="WT_VEH")

    if model is not None:
        xs = np.linspace(df_sub[metric].min(), df_sub[metric].max(), 100)
        for i, g in enumerate(groups):
            y_pred = model.predict(pd.DataFrame({metric: xs, "group": g}))
            ax.plot(xs, y_pred, color=cmap(i), alpha=0.9)

    ax.set_xlabel(metric)
    ax.set_ylabel("NOR index")
    ax.set_title(f"NOR vs {metric}\n{subset}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_group_slopes(
    slopes_df: "pd.DataFrame",
    subset: str,
    metric: str = "speed_q95",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Point + errorbar plot of per-group slopes with 95% CI and significance stars.
    """
    df_sub = slopes_df[
        (slopes_df["subset"] == subset) & (slopes_df["metric"] == metric)
    ]

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.figure

    if df_sub.empty:
        ax.set_visible(False)
        return fig, ax

    groups = df_sub["group"].tolist()
    x   = np.arange(len(groups))
    y   = df_sub["slope"].to_numpy()
    lo  = df_sub["ci_low"].to_numpy()
    hi  = df_sub["ci_high"].to_numpy()

    ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]),
                fmt="o", capsize=3, linestyle="none")
    ax.axhline(0, linestyle="--", lw=1)

    if "p_value" in df_sub.columns:
        span = (max(hi) - min(lo)) or 1.0
        offset = 0.05 * span
        for xi, yi, p in zip(x, y, df_sub["p_value"]):
            s = _p_to_star(p)
            if s:
                ax.text(xi, yi + offset, s, ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_ylabel(f"Slope NOR vs {metric}")
    ax.set_title(f"Group slopes — {subset}")
    fig.tight_layout()
    return fig, ax


def plot_multi_segment_scatter_row(
    df: "pd.DataFrame",
    subset_base: str,
    metric: str = "speed_q95",
    ref_group: str = "WT_VEH",
    segments: Sequence[str] = ("short", "mid", "long"),
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Row of scatter panels (one per segment) for NOR vs metric, with regression
    lines, for a given base subset and metric.
    """
    fig, axes = plt.subplots(1, len(segments),
                              figsize=(5 * len(segments), 4), sharey=True)
    if len(segments) == 1:
        axes = [axes]

    for ax, seg in zip(axes, segments):
        label = f"{subset_base}__{seg}"
        if label not in df["subset"].unique():
            ax.set_visible(False)
            continue
        model = None
        if _statsmodels_available:
            try:
                model, _ = fit_speed_nor_interaction(
                    df, subset=label, metric=metric, ref_group=ref_group
                )
            except Exception:
                pass
        plot_nor_vs_metric_by_group(df, subset=label, metric=metric,
                                    model=model, ax=ax)
        ax.set_title(f"{subset_base} — {seg}")

    fig.suptitle(f"NOR vs {metric} by segment — {subset_base}", y=1.02)
    fig.tight_layout()
    return fig, axes


# =============================================================================
# HIGH-LEVEL ANALYSIS RUNNER
# =============================================================================

def run_primary_analysis_from_df(
    df_metrics: "pd.DataFrame",
    primary_subsets: Sequence[str] | None = None,
    primary_metrics: Sequence[str] = PRIMARY_METRICS,
    ref_group: str = "WT_VEH",
    save_plots: bool = False,
    fig_root: Path | None = None,
) -> tuple["pd.DataFrame | None", "pd.DataFrame | None"]:
    """
    High-level driver starting from a pre-built df_metrics.

    For each subset × metric combination:
      - Computes within-group Spearman correlations (with bootstrap CI)
      - Fits NOR ~ metric * group interaction model and extracts slopes
      - Optionally saves scatter + slope plots

    Applies FDR correction (BH) across all p-values before returning.

    Returns
    -------
    corr_summary   : DataFrame of within-group correlations (or None)
    slopes_summary : DataFrame of group slopes (or None)
    """
    if pd is None:
        raise ImportError("pandas is required")

    if "subset_base" not in df_metrics.columns:
        df_metrics = add_subset_segment_columns(df_metrics)

    subsets = primary_subsets if primary_subsets is not None \
        else sorted(df_metrics["subset"].unique())

    scatter_dir = slopes_dir = None
    if save_plots and fig_root is not None:
        scatter_dir = fig_root / "scatter"
        slopes_dir  = fig_root / "slopes"
        scatter_dir.mkdir(parents=True, exist_ok=True)
        slopes_dir.mkdir(parents=True, exist_ok=True)

    all_corr:   list["pd.DataFrame"] = []
    all_slopes: list["pd.DataFrame"] = []

    for subset in subsets:
        df_sub = df_metrics[df_metrics["subset"] == subset]
        if df_sub.empty:
            print(f"[WARN] subset {subset} not in df_metrics; skipping")
            continue
        if ref_group not in df_sub["group"].unique():
            print(f"[WARN] ref_group {ref_group!r} missing in {subset}; skipping")
            continue

        corr_df = compute_within_group_correlations(
            df_metrics, subset=subset, metrics=primary_metrics,
        )
        if not corr_df.empty:
            all_corr.append(corr_df)

        for metric in primary_metrics:
            if metric not in df_metrics.columns:
                continue
            try:
                model, slopes_df = fit_speed_nor_interaction(
                    df_metrics, subset=subset, metric=metric, ref_group=ref_group,
                )
            except Exception as e:
                print(f"[WARN] {subset}/{metric}: {e}")
                continue

            parts = subset.rsplit("__", 1)
            slopes_df["subset_base"] = parts[0]
            slopes_df["segment"]     = parts[1] if len(parts) > 1 else "all"
            all_slopes.append(slopes_df)

            if save_plots and scatter_dir and slopes_dir:
                fig1, _ = plot_nor_vs_metric_by_group(
                    df_metrics, subset=subset, metric=metric, model=model)
                fig1.savefig(scatter_dir / f"scatter_{subset}_{metric}.png",
                             dpi=300, bbox_inches="tight")
                plt.close(fig1)

                fig2, _ = plot_group_slopes(slopes_df, subset=subset, metric=metric)
                fig2.savefig(slopes_dir / f"slopes_{subset}_{metric}.png",
                             dpi=300, bbox_inches="tight")
                plt.close(fig2)

    corr_summary   = pd.concat(all_corr,   ignore_index=True) if all_corr   else None
    slopes_summary = pd.concat(all_slopes, ignore_index=True) if all_slopes else None

    # FDR correction
    if _statsmodels_available:
        for summary in (corr_summary, slopes_summary):
            if summary is not None and "p_value" in summary.columns:
                mask = summary["p_value"].notna()
                if mask.any():
                    _, qvals, _, _ = multipletests(
                        summary.loc[mask, "p_value"], method="fdr_bh"
                    )
                    summary.loc[mask, "q_value"]      = qvals
                    summary["signif_fdr_05"] = summary.get("q_value", pd.Series(dtype=float)) < 0.05

    return corr_summary, slopes_summary
