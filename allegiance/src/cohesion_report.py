#!/usr/bin/env python3
"""
Cleaned coherence/allegiance analysis entry point (Option A).

Goals:
- Wrap execution in main()+CLI; avoid top-level side effects
- Use logging instead of print; consistent labels
- Toggle plotting via --save-plots and --no-show (headless friendly)

This script reads preprocessed data and merged allegiance results, reorders
communities consistently, computes simple module-count summaries, and
optionally generates a couple of standard plots.
"""
#%%
from __future__ import annotations

import argparse
import pickle
import logging
from pathlib import Path
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu

from shared_code.shared_code.fun_metaconnectivity import load_merged_allegiance
from shared_code.shared_code.fun_paths import get_paths

#%%

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    if logger.handlers:
        return
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean coherence/allegiance analysis")
    p.add_argument("--window-size", type=int, default=9, dest="window_size")
    p.add_argument("--lag", type=int, default=1, dest="lag")
    p.add_argument("--tau", type=int, default=3, dest="tau")
    p.add_argument(
        "--timecourse-folder",
        type=str,
        default="Timecourses_updated_03052024",
        dest="timecourse_folder",
    )
    p.add_argument("--alpha", type=float, default=0.05, help="significance level (unused, reserved)")
    p.add_argument("--save-plots", action="store_true", help="save figures under allegiance/fig")
    p.add_argument("--no-show", action="store_true", help="do not display figures (useful for batch runs)")
    p.add_argument("--animal", type=int, default=0, help="animal index for example plots")
    p.add_argument(
        "--dmn-index",
        type=str,
        default="0,23,13,22,2,28,34,37,39,8,35",
        help="comma-separated indices (in sorted label space) defining the DMN; use empty string to disable",
    )
    p.add_argument("--compute-cohesion", action="store_true", help="compute cohesion time series and probabilities")
    p.add_argument("--compute-events", action="store_true", help="extract link activation events and burstiness")
    p.add_argument("--with-stats", action="store_true", help="compute stats (age-paired and/or group-based) and plots")
    p.add_argument(
        "--stats-mode",
        choices=["age", "group", "all"],
        default="all",
        help="which stats to compute: within-base 2m vs 4m (age), group-based (group), or both",
    )
    p.add_argument(
        "--group-compare",
        choices=["sex", "genotype", "both", "sex_genotype"],
        default="both",
        help="which grouping dimensions to compare for group-based stats; use 'sex_genotype' to build Female/Male × wt/dKI intersections",
    )
    p.add_argument(
        "--cross-age",
        action="store_true",
        help="include cross-age comparisons in group-based stats (e.g., Female-2m vs Male-4m)",
    )
    p.add_argument(
        "--pool-ages",
        action="store_true",
        help="add pooled comparisons over age in group-based stats (e.g., Female vs Male ignoring age)",
    )
    p.add_argument(
        "--include-phenotype",
        choices=["none", "oip", "nor", "both"],
        default="none",
        help="optionally include phenotype (OiP/NOR) in age-paired stats",
    )
    p.add_argument(
        "--p-adjust",
        choices=["none", "bonferroni", "bonferroni-age"],
        default="none",
        help="Multiple-testing correction: 'bonferroni' across links per column; 'bonferroni-age' across columns within same age (2m/4m)",
    )
    return p.parse_args()


def build_paths(timecourse_folder: str) -> Tuple[dict, Path, Path]:
    """Build and create paths for data and plots.

    Returns (paths, per_animal_dir, stats_dir) where per_animal_dir and stats_dir
    are under the figures tree (paths["f_cohesion"]). CSVs remain under results.
    """
    paths = get_paths(timecourse_folder=timecourse_folder)
    per_animal_dir = (paths["f_cohesion"] / "per_animal").expanduser()
    stats_dir = (paths["f_cohesion"] / "stats").expanduser()
    per_animal_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    return paths, per_animal_dir, stats_dir


def load_meta(paths: dict) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(paths["preprocessed"] / "ts_and_meta_2m4m.npz", allow_pickle=True)
    ts = data["ts"]
    anat_labels = np.asarray(data["anat_labels"])
    return ts, anat_labels


def reorder_communities(paths: dict, window_size: int, lag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load merged allegiance and apply sorting indices to communities."""
    dfc_communities, sort_allegiances, contingency_matrices = load_merged_allegiance(
        paths, window_size=window_size, lag=lag
    )
    # Vectorized reorder: align community labels per-window using provided sort indices
    dfc_sorted = np.take_along_axis(dfc_communities, sort_allegiances.astype(int), axis=2)
    return dfc_sorted, sort_allegiances, contingency_matrices


def compute_module_counts(dfc_sorted: np.ndarray) -> np.ndarray:
    """Return number of unique modules per (animal, window)."""
    n_animals, n_windows, _ = dfc_sorted.shape
    out = np.zeros((n_animals, n_windows), dtype=int)
    for a in range(n_animals):
        for w in range(n_windows):
            out[a, w] = int(np.unique(dfc_sorted[a, w]).size)
    return out


def _upper_tri_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    tri = np.triu_indices(n, k=1)
    return tri[0], tri[1]


def _compute_cohesion_artifacts(
    dfc_sorted: np.ndarray,
    *,
    region_index: list[int] | None,
    anat_labels_sorted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    """
    Compute cohesion probability and time series on upper-triangle pairs.
    Returns:
    - cohesion_probability: (A, D, D) if DMN else (A, N, N) but with zeros on diagonal and lower triangle
    - cohesion_ts_triu: (A, L, T) differences of community labels per link (0 means same module)
    - pair_labels: list[(label_i,label_j)] for upper-triangle pairs
    """
    A, T, N = dfc_sorted.shape[0], dfc_sorted.shape[1], dfc_sorted.shape[2]
    if region_index is None:
        idx = list(range(N))
    else:
        idx = region_index

    # Build pair labels in selected index space
    pairs_i, pairs_j = _upper_tri_indices(len(idx))
    pair_labels = [(anat_labels_sorted[idx[i]], anat_labels_sorted[idx[j]]) for i, j in zip(pairs_i, pairs_j)]

    # Cohesion probability per animal
    def _coh_prob_one(comm_2d: np.ndarray) -> np.ndarray:
        d = comm_2d[:, idx]  # (T, D)
        same = (d[:, :, None] == d[:, None, :]).sum(axis=0)  # (D, D)
        return same / float(d.shape[0])

    coh_prob = np.stack([_coh_prob_one(dfc_sorted[a]) for a in range(A)], axis=0)

    # Cohesion time series over upper-triangle pairs
    coh_ts_triu = np.empty((A, pairs_i.size, T), dtype=int)
    for a in range(A):
        comm = dfc_sorted[a][:, idx]  # (T, D)
        # For each pair, store label difference over time
        # Vectorize: gather index arrays
        x = comm[:, pairs_i]
        y = comm[:, pairs_j]
        coh_ts_triu[a] = (x - y).T  # (L, T)

    return coh_prob, coh_ts_triu, pair_labels


def _extract_link_activations_df(binary_ATL: np.ndarray, min_duration: int = 1):
    """ Extract link activation events from binary ATL array.
    Returns a DataFrame with columns: animal, link, onset, offset, duration

    Input: binary_ATL: (A, T, L) array of 0/1 link activation time series
    min_duration: minimum duration (in time points) to keep an event

    Returns: DataFrame with columns: animal, link, onset, offset, duration


    """
    import pandas as pd

    A, T, L = binary_ATL.shape
    z = np.zeros((A, 1, L), dtype=binary_ATL.dtype)
    d = np.diff(np.concatenate((z, binary_ATL, z), axis=1), axis=1)
    on_idx = np.argwhere(d == 1)
    off_idx = np.argwhere(d == -1)
    on = pd.DataFrame(on_idx, columns=["animal", "time", "link"])
    off = pd.DataFrame(off_idx, columns=["animal", "time", "link"])
    on["gid"], off["gid"] = on["animal"] * L + on["link"], off["animal"] * L + off["link"]
    on = on.sort_values(["gid", "time"]).reset_index(drop=True)
    off = off.sort_values(["gid", "time"]).reset_index(drop=True)
    on["idx"], off["idx"] = on.groupby("gid").cumcount(), off.groupby("gid").cumcount()
    events = on.merge(off, on=["gid", "idx"], suffixes=("_on", "_off"))
    events = events.rename(columns={
        "animal_on": "animal",
        "link_on": "link",
        "time_on": "onset",
        "time_off": "offset",
    })[["animal", "link", "onset", "offset"]]
    events["duration"] = events["offset"] - events["onset"]
    return events[events["duration"] >= min_duration]


def _mean_duration_matrix(events_df, n_animals: int, n_links: int, fill: float = 0.0) -> np.ndarray:
    import pandas as pd  # noqa: F401

    m = events_df.groupby(["animal", "link"])['duration'].mean().unstack('link')
    m = m.reindex(index=range(n_animals), columns=range(n_links))
    return m.fillna(fill).to_numpy()


def _std_duration_matrix(events_df, n_animals: int, n_links: int, fill: float = 0.0) -> np.ndarray:
    import pandas as pd  # noqa: F401

    m = events_df.groupby(["animal", "link"])['duration'].std().unstack('link')
    m = m.reindex(index=range(n_animals), columns=range(n_links))
    return m.fillna(fill).to_numpy()


# -------------------- Grouping + stats helpers --------------------

BLOCK_SPEC: list[tuple] = [
    ("Sex", "single", 3),
    ("Genotype", "single", 2),
    ("Sex×Genotype", "pair", 3, 2),
]


def extend_block_spec_with_phenotype(include: str) -> list[tuple]:
    spec = list(BLOCK_SPEC)
    if include in {"oip", "both"}:
        spec.insert(1, ("OiP", "single", 0))
    if include in {"nor", "both"}:
        # place after OiP if included
        idx = 2 if include == "both" else 1
        spec.insert(idx, ("NOR", "single", 1))
    return spec


def _split_base_age(label: str) -> tuple[str, str | None]:
    parts = str(label).split()
    if len(parts) >= 2 and parts[-1] in {"2m", "4m"}:
        return " ".join(parts[:-1]), parts[-1]
    return label, None


def factor_base_indices(
    factor_idx: int,
    label_variables,
    mask_groups,
) -> dict[str, dict[str, np.ndarray | None]]:
    bases: dict[str, dict[str, np.ndarray | None]] = {}
    labels = label_variables[factor_idx]
    masks = mask_groups[factor_idx]
    for lbl, m in zip(labels, masks, strict=False):
        base, age = _split_base_age(lbl)
        if age in {"2m", "4m"}:
            idx = np.flatnonzero(np.asarray(m, dtype=bool))
            ent = bases.setdefault(base, {"2m": None, "4m": None})
            ent[age] = idx
    return bases


def _wilcoxon_rows(X: np.ndarray, Y: np.ndarray, zero_method: str = "wilcox") -> np.ndarray:
    try:
        res = wilcoxon(X, Y, zero_method=zero_method, alternative="two-sided", axis=1, method="asymptotic")
        return np.asarray(res.pvalue)
    except TypeError:
        p = np.empty(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            try:
                _, pv = wilcoxon(X[i], Y[i], zero_method=zero_method, alternative="two-sided", method="asymptotic")
            except ValueError:
                pv = 1.0
            p[i] = pv
        return p


def _ttest_rows(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    _, p = ttest_rel(X, Y, axis=1, nan_policy="propagate", alternative="two-sided")
    p = np.asarray(p, dtype=float)
    return np.where(np.isnan(p), 1.0, p)


def _mwu_rows(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Row-wise Mann–Whitney U two-sided p-values for independent samples.
    X, Y: (n_links, n_samples_x/y) possibly unequal lengths.
    """
    n = X.shape[0]
    out = np.empty(n, dtype=float)
    for i in range(n):
        xi = X[i]
        yi = Y[i]
        try:
            res = mannwhitneyu(xi, yi, alternative="two-sided")
            out[i] = float(res.pvalue)
        except Exception:
            out[i] = 1.0
    return out


def _cohesion_diff_rows(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    eps = 1e-9
    return np.mean((Y - X) / np.maximum(Y + X, eps), axis=1)


def _cols_single_factor_keys_and_data(
    block: str,
    fidx: int,
    data_T: np.ndarray,
    link_labels: list[str],
    label_variables,
    mask_groups,
    value_fn,
):
    F = factor_base_indices(fidx, label_variables, mask_groups)
    keys, cols = [], []
    for base, ages in F.items():
        idx2, idx4 = ages.get("2m"), ages.get("4m")
        if idx2 is None or idx4 is None or len(idx2) == 0 or len(idx4) == 0 or len(idx2) != len(idx4):
            continue
        X = data_T[:, idx2]
        Y = data_T[:, idx4]
        vals = value_fn(X, Y)
        keys.append((block, base))
        cols.append(vals)
    return keys, cols


def _cols_two_factors_keys_and_data(
    block: str,
    fA: int,
    fB: int,
    data_T: np.ndarray,
    link_labels: list[str],
    label_variables,
    mask_groups,
    value_fn,
):
    A = factor_base_indices(fA, label_variables, mask_groups)
    B = factor_base_indices(fB, label_variables, mask_groups)
    keys, cols = [], []
    for a, agesA in A.items():
        idx2A, idx4A = agesA.get("2m"), agesA.get("4m")
        if idx2A is None or idx4A is None:
            continue
        for b, agesB in B.items():
            idx2B, idx4B = agesB.get("2m"), agesB.get("4m")
            if idx2B is None or idx4B is None:
                continue
            # Intersect to keep matched pairs
            keep = np.isin(idx2A, idx2B) & np.isin(idx4A, idx4B)
            idx2 = idx2A[keep]
            idx4 = idx4A[keep]
            if len(idx2) == 0 or len(idx4) == 0 or len(idx2) != len(idx4):
                continue
            X = data_T[:, idx2]
            Y = data_T[:, idx4]
            vals = value_fn(X, Y)
            keys.append((block, f"{a}·{b}"))
            cols.append(vals)
    return keys, cols


def build_table_from_spec(
    data_T: np.ndarray,
    link_labels: list[str],
    label_variables,
    mask_groups,
    block_spec=BLOCK_SPEC,
    value_fn=_wilcoxon_rows,
) -> pd.DataFrame:
    all_keys, all_cols = [], []
    for item in block_spec:
        if item[1] == "single":
            block, _, fidx = item
            keys, cols = _cols_single_factor_keys_and_data(
                block, fidx, data_T, link_labels, label_variables, mask_groups, value_fn
            )
        else:
            block, _, fA, fB = item
            keys, cols = _cols_two_factors_keys_and_data(
                block, fA, fB, data_T, link_labels, label_variables, mask_groups, value_fn
            )
        all_keys += keys
        all_cols += cols
    if not all_cols:
        return pd.DataFrame(index=link_labels, columns=pd.MultiIndex.from_tuples([]))
    M = np.column_stack(all_cols)
    columns = pd.MultiIndex.from_tuples(all_keys, names=["Block", "Column"])
    return pd.DataFrame(M, index=link_labels, columns=columns)


def build_group_comparisons(
    data_T: np.ndarray,
    link_labels: list[str],
    label_variables,
    mask_groups,
    *,
    factors: list[str],  # 'sex', 'genotype', optionally 'sex_genotype'
    cross_age: bool,
    pooled: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build group-based (independent) comparisons using Mann–Whitney U.
    Returns three aligned DataFrames (p-values, mean-diff effect, cohesion-diff ratio effect).
    """
    factor_map = {"sex": ("Sex", 3), "genotype": ("Genotype", 2)}
    cols_p, names = [], []
    cols_mdiff, cols_cdr = [], []

    for key in factors:
        if key not in factor_map:
            continue
        title, fidx = factor_map[key]
        F = factor_base_indices(fidx, label_variables, mask_groups)  # base->{2m,4m}
        bases = list(F.keys())
        ages = ["2m", "4m"]
        # Build group-age entries
        entries = []  # (label, idx)
        for b in bases:
            for age in ages:
                idx = F[b].get(age)
                if idx is None or len(idx) == 0:
                    continue
                entries.append((f"{b}-{age}", idx))

        # Build pooled over age entries
        pooled_entries = []
        if pooled:
            for b in bases:
                idx2 = F[b].get("2m")
                idx4 = F[b].get("4m")
                parts = []
                if idx2 is not None and len(idx2) > 0:
                    parts.append(idx2)
                if idx4 is not None and len(idx4) > 0:
                    parts.append(idx4)
                if parts:
                    pooled_idx = np.unique(np.concatenate(parts))
                    pooled_entries.append((f"{b} (all-ages)", pooled_idx))

        # Decide pairs
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                name_i, idx_i = entries[i]
                name_j, idx_j = entries[j]
                # If not cross-age, require same age suffix
                if not cross_age:
                    ai = name_i.split("-")[-1]
                    aj = name_j.split("-")[-1]
                    if ai != aj:
                        continue
                # Require different base (e.g., Female vs Male; dKI vs wt)
                base_i = name_i.rsplit("-", 1)[0]
                base_j = name_j.rsplit("-", 1)[0]
                if base_i == base_j:
                    continue
                # Skip if any is empty or identical set
                if len(idx_i) == 0 or len(idx_j) == 0:
                    continue

                # Slice and compute stats
                X = data_T[:, idx_i]
                Y = data_T[:, idx_j]
                p = _mwu_rows(X, Y)
                muX = np.mean(X, axis=1)
                muY = np.mean(Y, axis=1)
                mdiff = muY - muX
                cdr = (muY - muX) / np.maximum(muY + muX, 1e-9)

                cols_p.append(p)
                cols_mdiff.append(mdiff)
                cols_cdr.append(cdr)
                names.append((f"{title}", f"{name_i} vs {name_j}"))

        # Pooled comparisons between different bases
        for i in range(len(pooled_entries)):
            for j in range(i + 1, len(pooled_entries)):
                name_i, idx_i = pooled_entries[i]
                name_j, idx_j = pooled_entries[j]
                base_i = name_i.split(" ")[0]
                base_j = name_j.split(" ")[0]
                if base_i == base_j:
                    continue
                if len(idx_i) == 0 or len(idx_j) == 0:
                    continue
                X = data_T[:, idx_i]
                Y = data_T[:, idx_j]
                p = _mwu_rows(X, Y)
                muX = np.mean(X, axis=1)
                muY = np.mean(Y, axis=1)
                mdiff = muY - muX
                cdr = (muY - muX) / np.maximum(muY + muX, 1e-9)
                cols_p.append(p)
                cols_mdiff.append(mdiff)
                cols_cdr.append(cdr)
                names.append((f"{title}", f"{name_i} vs {name_j}"))

    # Optional: combined Sex×Genotype intersections (8 groups: Female/Male × wt/dKI × 2m/4m)
    if "sex_genotype" in factors:
        title = "Sex×Genotype"
        F_sex = factor_base_indices(3, label_variables, mask_groups)
        F_geno = factor_base_indices(2, label_variables, mask_groups)

        # Build entries like "Female dKI-2m" with intersected indices
        entries = []  # (label, idx)
        for sex_base, agesS in F_sex.items():
            for geno_base, agesG in F_geno.items():
                for age in ("2m", "4m"):
                    idxS = agesS.get(age)
                    idxG = agesG.get(age)
                    if idxS is None or idxG is None or len(idxS) == 0 or len(idxG) == 0:
                        continue
                    # Intersection of animals that satisfy both sex and genotype at a given age
                    idx = np.intersect1d(idxS, idxG, assume_unique=False)
                    if idx.size == 0:
                        continue
                    entries.append((f"{sex_base} {geno_base}-{age}", idx))

        # Pooled entries over ages per Sex×Genotype base
        pooled_entries = []
        if pooled:
            for sex_base, agesS in F_sex.items():
                for geno_base, agesG in F_geno.items():
                    parts = []
                    for age in ("2m", "4m"):
                        idxS = agesS.get(age)
                        idxG = agesG.get(age)
                        if idxS is None or idxG is None:
                            continue
                        inter = np.intersect1d(idxS, idxG, assume_unique=False)
                        if inter.size:
                            parts.append(inter)
                    if parts:
                        pooled_idx = np.unique(np.concatenate(parts))
                        pooled_entries.append((f"{sex_base} {geno_base} (all-ages)", pooled_idx))

        # Pairwise comparisons among entries (respect cross_age flag)
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                name_i, idx_i = entries[i]
                name_j, idx_j = entries[j]
                # Within-age only unless cross_age requested
                if not cross_age:
                    ai = name_i.split("-")[-1]
                    aj = name_j.split("-")[-1]
                    if ai != aj:
                        continue
                # Skip comparisons within the exact same base (e.g., Female wt vs Female wt)
                base_i = name_i.rsplit("-", 1)[0]
                base_j = name_j.rsplit("-", 1)[0]
                if base_i == base_j:
                    continue
                if len(idx_i) == 0 or len(idx_j) == 0:
                    continue
                X = data_T[:, idx_i]
                Y = data_T[:, idx_j]
                p = _mwu_rows(X, Y)
                muX = np.mean(X, axis=1)
                muY = np.mean(Y, axis=1)
                mdiff = muY - muX
                cdr = (muY - muX) / np.maximum(muY + muX, 1e-9)
                cols_p.append(p)
                cols_mdiff.append(mdiff)
                cols_cdr.append(cdr)
                names.append((title, f"{name_i} vs {name_j}"))

        # Pooled all-ages pairwise comparisons across different Sex×Genotype bases
        for i in range(len(pooled_entries)):
            for j in range(i + 1, len(pooled_entries)):
                name_i, idx_i = pooled_entries[i]
                name_j, idx_j = pooled_entries[j]
                base_i = name_i.split(" (all-ages)")[0]
                base_j = name_j.split(" (all-ages)")[0]
                if base_i == base_j:
                    continue
                if len(idx_i) == 0 or len(idx_j) == 0:
                    continue
                X = data_T[:, idx_i]
                Y = data_T[:, idx_j]
                p = _mwu_rows(X, Y)
                muX = np.mean(X, axis=1)
                muY = np.mean(Y, axis=1)
                mdiff = muY - muX
                cdr = (muY - muX) / np.maximum(muY + muX, 1e-9)
                cols_p.append(p)
                cols_mdiff.append(mdiff)
                cols_cdr.append(cdr)
                names.append((title, f"{name_i} vs {name_j}"))

    if not cols_p:
        empty = pd.DataFrame(index=link_labels, columns=pd.MultiIndex.from_tuples([]))
        return empty, empty, empty

    P = np.column_stack(cols_p)
    MD = np.column_stack(cols_mdiff)
    CDR = np.column_stack(cols_cdr)
    columns = pd.MultiIndex.from_tuples(names, names=["Block", "Column"])
    return (
        pd.DataFrame(P, index=link_labels, columns=columns),
        pd.DataFrame(MD, index=link_labels, columns=columns),
        pd.DataFrame(CDR, index=link_labels, columns=columns),
    )


def separators_from_multiindex(mi: pd.MultiIndex):
    if mi.nlevels == 0 or mi.size == 0:
        return []
    blocks = mi.get_level_values(0)
    seps, cur, count = [], blocks[0], 0
    for b in blocks:
        if b != cur:
            seps.append((count, cur))
            cur = b
        count += 1
    seps.append((count, cur))
    return seps


def plot_sig_pvals_multi(
    pvals_df: pd.DataFrame,
    alpha: float,
    title: str,
    show_grid: bool = True,
    save: bool = False,
    out_path: Path | None = None,
):
    data = pvals_df.values
    mask = np.where(data <= alpha, data, np.nan)
    n_rows, n_cols = mask.shape if mask.size else (0, 0)
    fig, ax = plt.subplots(figsize=(max(15, 0.22 * n_cols), max(2, 0.16 * max(n_rows, 1))))
    im = ax.imshow(mask, aspect="auto", interpolation="none", cmap="viridis_r", vmin=0, vmax=alpha)
    fig.colorbar(im, ax=ax).set_label("p-value")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(pvals_df.index, fontsize=6)
    col_labels = [f"{b} | {c}" for b, c in pvals_df.columns.to_list()]
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=8)
    if show_grid:
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which="minor", color="k", linestyle="-", linewidth=0.3, alpha=0.25)
        ax.grid(which="major", visible=False)
    for x_end, _ in separators_from_multiindex(pvals_df.columns)[:-1]:
        ax.axvline(x_end - 0.5, color="k", lw=1.0, alpha=0.6)
    ax.set_title(title)
    fig.tight_layout()
    if save and out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig


def plot_weighted_multi(
    pvals_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    alpha: float,
    title: str,
    vmin: float = -0.1,
    vmax: float = 0.1,
    save: bool = False,
    out_path: Path | None = None,
):
    assert tuple(pvals_df.columns) == tuple(weights_df.columns)
    assert list(pvals_df.index) == list(weights_df.index)
    p, w = pvals_df.values, weights_df.values
    Z = np.where(p <= alpha, 1 - p, np.nan) * w
    n_rows, n_cols = Z.shape if Z.size else (0, 0)
    fig, ax = plt.subplots(figsize=(max(15, 0.22 * n_cols), max(2, 0.16 * max(n_rows, 1))))
    im = ax.imshow(Z, aspect="auto", interpolation="none", cmap="RdBu", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax).set_label("(1 - p) × effect")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(pvals_df.index, fontsize=6)
    col_labels = [f"{b} | {c}" for b, c in pvals_df.columns.to_list()]
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=8)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="k", linestyle="-", linewidth=0.3, alpha=0.25)
    ax.grid(which="major", visible=False)
    for x_end, _ in separators_from_multiindex(pvals_df.columns)[:-1]:
        ax.axvline(x_end - 0.5, color="k", lw=1.0, alpha=0.6)
    ax.set_title(title)
    fig.tight_layout()
    if save and out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig


def _bonferroni_by_age_in_columns(pvals_df: pd.DataFrame) -> pd.DataFrame:
    if pvals_df.empty:
        return pvals_df.copy()
    cols = list(pvals_df.columns)
    group_map: dict[str, list[int]] = {"2m": [], "4m": []}
    for j, col in enumerate(cols):
        label = str(col[1])
        try:
            lhs, rhs = label.split(" vs ")
            age_l = lhs.split("-")[-1]
            age_r = rhs.split("-")[-1]
            if age_l in {"2m", "4m"} and age_l == age_r:
                group_map[age_l].append(j)
        except Exception:
            continue
    A = np.asarray(pvals_df.values, dtype=float)
    for age, idxs in group_map.items():
        if not idxs:
            continue
        k = float(len(idxs))
        A[:, idxs] = np.minimum(1.0, A[:, idxs] * k)
    return pd.DataFrame(A, index=pvals_df.index, columns=pvals_df.columns)


def plot_communities_for_animal(
    dfc_sorted: np.ndarray,
    anat_labels_sorted: np.ndarray,
    animal_idx: int,
    *,
    save: bool,
    show: bool,
    out_dir: Path,
    fname: str,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(dfc_sorted[animal_idx].T, aspect="auto", interpolation="none", cmap="viridis")
    ax.set_title(f"dFC Communities — Animal {animal_idx}")
    ax.set_xlabel(r"Time Windows (TW$_{1}$, TW$_{2}$, ..., TW$_{n}$)")
    ax.set_yticks(np.arange(len(anat_labels_sorted)))
    ax.set_yticklabels(anat_labels_sorted)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    if save:
        fig.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
        logger.info("Saved figure: %s", out_dir / fname)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_modules_timeseries(
    module_counts: np.ndarray,
    animal_idx: int,
    *,
    save: bool,
    show: bool,
    out_dir: Path,
    fname: str,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(module_counts[animal_idx], marker="o", lw=1)
    ax.set_title(f"Number of Modules per Time Window — Animal {animal_idx}")
    ax.set_xlabel(r"Time Windows (TW$_{1}$, TW$_{2}$, ..., TW$_{n}$)")
    ax.set_ylabel("#Modules")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save:
        fig.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
        logger.info("Saved figure: %s", out_dir / fname)
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> int:
    setup_logging()
    args = parse_args()

    # Headless plotting if requested
    if args.no_show:
        matplotlib.use("Agg", force=True)

    paths, per_animal_dir, stats_dir = build_paths(args.timecourse_folder)
    ts, anat_labels = load_meta(paths)
    n_animals = len(ts)
    logger.info("Loaded time series: n_animals=%d, n_regions=%d", n_animals, ts[0].shape[1])

    dfc_sorted, sort_idx, contingency = reorder_communities(paths, args.window_size, args.lag)
    n_windows = dfc_sorted.shape[1]
    logger.info("Loaded allegiance: windows=%d, regions=%d", n_windows, dfc_sorted.shape[2])

    # Consistent labels (use first animal/window ordering)
    # anat_labels_sorted = anat_labels[sort_idx[0, 0].astype(int)]
    anat_labels_sorted = anat_labels
    # Simple metric: number of modules per window
    module_counts = compute_module_counts(dfc_sorted)
    logger.info("Module counts: min=%s max=%s mean=%.2f",
                int(module_counts.min()), int(module_counts.max()), float(module_counts.mean()))

    # Plots for a representative animal
    animal = int(args.animal)
    base = f"w{args.window_size}_lag{args.lag}_tau{args.tau}_animal{animal}"
    plot_communities_for_animal(
        dfc_sorted,
        anat_labels_sorted,
        animal,
        save=args.save_plots,
        show=not args.no_show,
        out_dir=per_animal_dir,
        fname=f"dfc_communities_{base}.png",
    )
    plot_modules_timeseries(
        module_counts,
        animal,
        save=args.save_plots,
        show=not args.no_show,
        out_dir=per_animal_dir,
        fname=f"module_counts_{base}.png",
    )

    # Optional: Cohesion and events
    # Parse DMN index (in sorted label space) if provided
    dmn_index: list[int] | None
    if args.dmn_index.strip():
        try:
            dmn_index = [int(x) for x in args.dmn_index.split(",") if str(x).strip() != ""]
        except Exception:
            logger.warning("Invalid --dmn-index; falling back to None (all regions)")
            dmn_index = None
    else:
        dmn_index = None
    scope = "dmn" if dmn_index is not None else "all"

    if args.compute_cohesion or args.compute_events or args.with_stats:
        # Compute cohesion timeseries/probabilities for chosen set (DMN or all regions)
        coh_prob, coh_ts_triu, pair_labels = _compute_cohesion_artifacts(
            dfc_sorted,
            region_index=dmn_index,
            anat_labels_sorted=anat_labels_sorted,
        )
        logger.info("Cohesion: pairs=%d windows=%d", coh_ts_triu.shape[1], coh_ts_triu.shape[2])

        if args.compute_cohesion:
            # Plot binary cohesion (same community) for selected animal
            binary = (coh_ts_triu == 0).astype(int)  # 1 if same module
            fig, ax = plt.subplots(figsize=(12, 6))
            im = ax.imshow(binary[animal].T, aspect="auto", interpolation="none", cmap="gray_r", vmin=0, vmax=1)
            ax.set_title(f"Cohesion (same-module=1) — Animal {animal}")
            ax.set_xlabel(r"Time Windows")
            ax.set_ylabel("Link index (upper-tri)")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            if args.save_plots:
                fig_path = per_animal_dir / f"cohesion_binary_{base}_{scope}.png"
                fig.savefig(fig_path, dpi=300, bbox_inches="tight")
                logger.info("Saved figure: %s", fig_path)
            if not args.no_show:
                plt.show()
            else:
                plt.close(fig)

        if args.compute_events:
            import pandas as pd  # local import to keep base path light

            # Events expect shape (A, T, L)
            binary = (coh_ts_triu == 0).astype(int)
            binary_ATL = np.transpose(binary, (0, 2, 1))
            events = _extract_link_activations_df(binary_ATL, min_duration=2)

            # Aggregate burstiness
            n_animals = binary.shape[0]
            n_links = binary.shape[1]
            mean_dur = _mean_duration_matrix(events, n_animals=n_animals, n_links=n_links)
            std_dur = _std_duration_matrix(events, n_animals=n_animals, n_links=n_links)
            burstiness = (std_dur - mean_dur) / np.maximum(std_dur + mean_dur, 1e-9)
            burstiness[mean_dur == 0] = 0

            fig, ax = plt.subplots(figsize=(14, 6))
            im = ax.imshow(burstiness, interpolation="none", aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_title("Burstiness per animal × link")
            ax.set_xlabel("Link index (upper-tri)")
            ax.set_ylabel("Animal")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            if args.save_plots:
                fig_path = per_animal_dir / f"burstiness_{base}_{scope}.png"
                fig.savefig(fig_path, dpi=300, bbox_inches="tight")
                logger.info("Saved figure: %s", fig_path)
            if not args.no_show:
                plt.show()
            else:
                plt.close(fig)

        if args.with_stats:
            # Inputs for stats tables
            # data_T: (n_links, n_animals)
            binary = (coh_ts_triu == 0).astype(int)
            time_ratio = binary.sum(axis=2) / binary.shape[2]
            data_T = time_ratio.T
            link_labels = [f"{a}–{b}" for (a, b) in pair_labels]

            # Load grouping
            with open(paths["preprocessed"] / "grouping_data_oip.pkl", "rb") as f:
                mask_groups, label_variables = pickle.load(f)

            out_dir = (paths["allegiance"] / "out").expanduser()
            out_dir.mkdir(parents=True, exist_ok=True)

            tag = f"w{args.window_size}_lag{args.lag}_tau{args.tau}_{scope}"

            # AGE-PAIRED (2m vs 4m within base)
            if args.stats_mode in {"age", "all"}:
                spec_age = extend_block_spec_with_phenotype(args.include_phenotype)
                pvals_wil = build_table_from_spec(
                    data_T, link_labels, label_variables, mask_groups, block_spec=spec_age, value_fn=_wilcoxon_rows
                )
                effects_age_ratio = build_table_from_spec(
                    data_T, link_labels, label_variables, mask_groups, block_spec=spec_age, value_fn=_cohesion_diff_rows
                )
                pvals_t = build_table_from_spec(
                    data_T, link_labels, label_variables, mask_groups, block_spec=spec_age, value_fn=_ttest_rows
                )
                # Mean-diff effect for age: compute using means of X/Y internally; reuse ratio as above.
                # For age-paired, we approximate mean-diff via means over animals in each age group.
                effects_age_mdiff = build_table_from_spec(
                    data_T, link_labels, label_variables, mask_groups, block_spec=spec_age,
                    value_fn=lambda X, Y: np.mean(Y, axis=1) - np.mean(X, axis=1)
                )

                # Save raw p-values
                pvals_wil.to_csv(out_dir / f"pvals_age_wilcoxon_{tag}.csv")
                pvals_t.to_csv(out_dir / f"pvals_age_ttest_{tag}.csv")
                effects_age_ratio.to_csv(out_dir / f"effects_age_cdratio_{tag}.csv")
                effects_age_mdiff.to_csv(out_dir / f"effects_age_mdiff_{tag}.csv")

                # Optional Bonferroni across links per column
                title_suffix = ""
                p_wil_plot = pvals_wil
                if args.p_adjust == "bonferroni":
                    p_b = np.minimum(1.0, pvals_wil.values * pvals_wil.shape[0])
                    p_wil_b = pd.DataFrame(p_b, index=pvals_wil.index, columns=pvals_wil.columns)
                    p_wil_b.to_csv(out_dir / f"pvals_age_wilcoxon_bonferroni_{tag}.csv")
                    p_wil_plot = p_wil_b
                    title_suffix = " (Bonferroni)"

                fig1 = plot_sig_pvals_multi(
                    p_wil_plot, alpha=args.alpha, title=f"Age (2m vs 4m) — Wilcoxon significant only{title_suffix}",
                    save=args.save_plots, out_path=stats_dir / f"pvals_age_wilcoxon_sig_{tag}.png"
                )
                if not args.no_show:
                    plt.show()
                else:
                    plt.close(fig1)

                fig2 = plot_weighted_multi(
                    p_wil_plot, effects_age_ratio, alpha=args.alpha,
                    title=f"Age (2m vs 4m) — (1 - p) × cohesion-diff ratio{title_suffix}",
                    vmin=-0.1, vmax=0.1, save=args.save_plots,
                    out_path=stats_dir / f"weighted_age_wilcoxon_cdratio_{tag}.png"
                )
                if not args.no_show:
                    plt.show()
                else:
                    plt.close(fig2)

            # GROUP-BASED (independent samples; Mann–Whitney U)
            if args.stats_mode in {"group", "all"}:
                factors = []
                if args.group_compare in {"sex", "both"}:
                    factors.append("sex")
                if args.group_compare in {"genotype", "both"}:
                    factors.append("genotype")
                if args.group_compare == "sex_genotype":
                    factors.append("sex_genotype")
                pvals_grp, effects_grp_mdiff, effects_grp_cdr = build_group_comparisons(
                    data_T, link_labels, label_variables, mask_groups,
                    factors=factors, cross_age=args.cross_age, pooled=args.pool_ages,
                )
                pvals_grp.to_csv(out_dir / f"pvals_group_mwu_{tag}.csv")
                effects_grp_mdiff.to_csv(out_dir / f"effects_group_mdiff_{tag}.csv")
                effects_grp_cdr.to_csv(out_dir / f"effects_group_cdratio_{tag}.csv")

                title_suffix = ""
                p_grp_plot = pvals_grp
                if args.p_adjust == "bonferroni":
                    p_b = np.minimum(1.0, pvals_grp.values * pvals_grp.shape[0])
                    p_grp_b = pd.DataFrame(p_b, index=pvals_grp.index, columns=pvals_grp.columns)
                    p_grp_b.to_csv(out_dir / f"pvals_group_mwu_bonferroni_{tag}.csv")
                    p_grp_plot = p_grp_b
                    title_suffix = " (Bonferroni)"
                elif args.p_adjust == "bonferroni-age":
                    p_grp_ba = _bonferroni_by_age_in_columns(pvals_grp)
                    p_grp_ba.to_csv(out_dir / f"pvals_group_mwu_bonferroni_age_{tag}.csv")
                    p_grp_plot = p_grp_ba
                    title_suffix = " (Bonferroni by age group)"

                figg1 = plot_sig_pvals_multi(
                    p_grp_plot, alpha=args.alpha, title=f"Group (MWU) — significant only{title_suffix}",
                    save=args.save_plots, out_path=stats_dir / f"pvals_group_mwu_sig_{tag}.png"
                )
                if not args.no_show:
                    plt.show()
                else:
                    plt.close(figg1)

                figg2 = plot_weighted_multi(
                    p_grp_plot, effects_grp_mdiff, alpha=args.alpha,
                    title=f"Group (MWU) — (1 - p) × mean difference{title_suffix}",
                    vmin=-0.1, vmax=0.1, save=args.save_plots,
                    out_path=stats_dir / f"weighted_group_mwu_mdiff_{tag}.png"
                )
                if not args.no_show:
                    plt.show()
                else:
                    plt.close(figg2)

                figg3 = plot_weighted_multi(
                    p_grp_plot, effects_grp_cdr, alpha=args.alpha,
                    title=f"Group (MWU) — (1 - p) × cohesion-diff ratio{title_suffix}",
                    vmin=-0.1, vmax=0.1, save=args.save_plots,
                    out_path=stats_dir / f"weighted_group_mwu_cdratio_{tag}.png"
                )
                if not args.no_show:
                    plt.show()
                else:
                    plt.close(figg3)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
