#!/usr/bin/env python3
"""
Load cohesion summaries (Option 3) and produce statistics + plots.

Inputs: NPZ (time_ratio etc.) from cohesion_compute.py
Outputs:
- CSV tables to results/<dataset>/allegiance/out
- Figures to fig/<dataset>/cohesion/stats
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_rel, wilcoxon

from shared_code.fun_paths import get_paths

logger = logging.getLogger(__name__)

def setup_logging() -> None:
    if logger.handlers:
        return
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cohesion stats and plots")
    p.add_argument("--window-size", type=int, default=9, dest="window_size")
    p.add_argument("--lag", type=int, default=1, dest="lag")
    p.add_argument("--tau", type=int, default=3, dest="tau")
    p.add_argument(
        "--timecourse-folder",
        type=str,
        default="Timecourses_updated_03052024",
        dest="timecourse_folder",
    )
    p.add_argument(
        "--dmn-index",
        type=str,
        default="0,23,13,22,2,28,34,37,39,8,35",
        help="comma-separated indices (sorted label space) for DMN; empty string for all regions",
    )
    p.add_argument(
        "--roi-scope",
        choices=["all", "dmn", "memory", "custom"],
        default="all",
        help=(
            "Scope of ROIs used in the precomputed NPZ: 'all' (default), 'dmn', "
            "'memory', or 'custom' (to load files produced with custom ROI sets)"
        ),
    )
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--save-plots", action="store_true")
    p.add_argument("--no-show", action="store_true")
    p.add_argument(
        "--emit",
        choices=["stats", "plots", "both"],
        default="both",
        help="Select which artifacts to emit: stats CSVs, plots, or both",
    )

    p.add_argument(
        "--with-stats",
        action="store_true",
        help="run stats and export tables + heatmaps",
    )
    p.add_argument("--stats-mode", choices=["age", "group", "all"], default="all")
    p.add_argument(
        "--group-compare",
        choices=["sex", "genotype", "both", "sex_genotype"],
        default="both",
    )
    p.add_argument("--cross-age", action="store_true")
    p.add_argument("--pool-ages", action="store_true")
    p.add_argument(
        "--include-phenotype", choices=["none", "oip", "nor", "both"], default="none"
    )
    p.add_argument(
        "--p-adjust",
        choices=["none", "bonferroni", "bonferroni-age"],
        default="none",
        help="Multiple-testing correction: 'bonferroni' across links per column; 'bonferroni-age' across columns within same age (2m/4m)",
    )
    # Optional: per-comparison D×D matrices
    p.add_argument(
        "--matrix-per-comparison",
        action="store_true",
        help="For each comparison column, render a D×D matrix of significant values or weighted effect",
    )
    p.add_argument(
        "--matrix-mode",
        choices=["sig", "weighted"],
        default="weighted",
        help="Matrix content: 'sig' plots significance mask; 'weighted' plots (1 - p) × effect",
    )
    p.add_argument(
        "--matrix-effect",
        choices=["mdiff", "cdratio"],
        default="cdratio",
        help="When matrix-mode=weighted, choose the effect to weight by",
    )
    return p.parse_args()

def load_npz(paths: dict, ws: int, lag: int, tau: int, scope: str) -> dict:
    f = (
        paths["allegiance"]
        / "cohesion_data"
        / f"cohesion_data_w{ws}_lag{lag}_tau{tau}_{scope}.npz"
    )
    return dict(np.load(f, allow_pickle=True))

# ---------------- Masks + stats helpers ----------------
BLOCK_BASE: list[tuple] = [
    ("Sex", "single", 3),
    ("Genotype", "single", 2),
    ("Sex×Genotype", "pair", 3, 2),
]


def extend_block_spec_with_phenotype(include: str) -> list[tuple]:
    spec = list(BLOCK_BASE)
    if include in {"oip", "both"}:
        spec.insert(1, ("OiP", "single", 0))
    if include in {"nor", "both"}:
        idx = 2 if include == "both" else 1
        spec.insert(idx, ("NOR", "single", 1))
    return spec


def _split_base_age(label: str) -> tuple[str, str | None]:
    parts = str(label).split()
    if len(parts) >= 2 and parts[-1] in {"2m", "4m"}:
        return " ".join(parts[:-1]), parts[-1]
    return label, None


def factor_base_indices(
    factor_idx: int, label_variables, mask_groups
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


def _wilcoxon_rows(
    X: np.ndarray, Y: np.ndarray, zero_method: str = "wilcox"
) -> np.ndarray:
    try:
        res = wilcoxon(
            X,
            Y,
            zero_method=zero_method,
            alternative="two-sided",
            axis=1,
            method="asymptotic",
        )
        return np.asarray(res.pvalue)
    except TypeError:
        p = np.empty(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            try:
                _, pv = wilcoxon(
                    X[i],
                    Y[i],
                    zero_method=zero_method,
                    alternative="two-sided",
                    method="asymptotic",
                )
            except ValueError:
                pv = 1.0
            p[i] = pv
        return p


def _ttest_rows(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    _, p = ttest_rel(X, Y, axis=1, nan_policy="propagate", alternative="two-sided")
    p = np.asarray(p, dtype=float)
    return np.where(np.isnan(p), 1.0, p)


def _mwu_rows(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    out = np.empty(n, dtype=float)
    for i in range(n):
        try:
            out[i] = float(mannwhitneyu(X[i], Y[i], alternative="two-sided").pvalue)
        except Exception:
            out[i] = 1.0
    return out


def _cohesion_diff_rows(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    eps = 1e-9
    return np.mean((Y - X) / np.maximum(Y + X, eps), axis=1)


def _cols_single(
    block: str,
    fidx: int,
    data_T: np.ndarray,
    link_labels,
    label_variables,
    mask_groups,
    value_fn,
):
    F = factor_base_indices(fidx, label_variables, mask_groups)
    keys, cols = [], []
    for base, ages in F.items():
        idx2, idx4 = ages.get("2m"), ages.get("4m")
        if (
            idx2 is None
            or idx4 is None
            or len(idx2) == 0
            or len(idx4) == 0
            or len(idx2) != len(idx4)
        ):
            continue
        X = data_T[:, idx2]
        Y = data_T[:, idx4]
        keys.append((block, base))
        cols.append(value_fn(X, Y))
    return keys, cols


def _cols_pair(
    block: str,
    fA: int,
    fB: int,
    data_T,
    link_labels,
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
            keep = np.isin(idx2A, idx2B) & np.isin(idx4A, idx4B)
            idx2 = idx2A[keep]
            idx4 = idx4A[keep]
            if len(idx2) == 0 or len(idx4) == 0 or len(idx2) != len(idx4):
                continue
            X = data_T[:, idx2]
            Y = data_T[:, idx4]
            keys.append((block, f"{a}·{b}"))
            cols.append(value_fn(X, Y))
    return keys, cols


def build_table_from_spec(
    data_T: np.ndarray, link_labels, label_variables, mask_groups, block_spec, value_fn
) -> pd.DataFrame:
    all_keys, all_cols = [], []
    for item in block_spec:
        if item[1] == "single":
            k, c = _cols_single(
                item[0],
                item[2],
                data_T,
                link_labels,
                label_variables,
                mask_groups,
                value_fn,
            )
        else:
            k, c = _cols_pair(
                item[0],
                item[2],
                item[3],
                data_T,
                link_labels,
                label_variables,
                mask_groups,
                value_fn,
            )
        all_keys += k
        all_cols += c
    if not all_cols:
        return pd.DataFrame(index=link_labels, columns=pd.MultiIndex.from_tuples([]))
    M = np.column_stack(all_cols)
    columns = pd.MultiIndex.from_tuples(all_keys, names=["Block", "Column"])
    return pd.DataFrame(M, index=link_labels, columns=columns)


def build_group_comparisons(
    data_T,
    link_labels,
    label_variables,
    mask_groups,
    *,
    factors: list[str],
    cross_age: bool,
    pooled: bool,
):
    factor_map = {"sex": ("Sex", 3), "genotype": ("Genotype", 2)}
    cols_p, cols_mdiff, cols_cdr, names = [], [], [], []
    for key in factors:
        if key not in factor_map:
            continue
        title, fidx = factor_map[key]
        F = factor_base_indices(fidx, label_variables, mask_groups)
        bases = list(F.keys())
        entries = []
        for b in bases:
            for age in ("2m", "4m"):
                idx = F[b].get(age)
                if idx is None or len(idx) == 0:
                    continue
                entries.append((f"{b}-{age}", idx))
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
                    pooled_entries.append(
                        (f"{b} (all-ages)", np.unique(np.concatenate(parts)))
                    )
        # within-age or cross-age pairs
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                name_i, idx_i = entries[i]
                name_j, idx_j = entries[j]
                if not cross_age:
                    if name_i.split("-")[-1] != name_j.split("-")[-1]:
                        continue
                if name_i.rsplit("-", 1)[0] == name_j.rsplit("-", 1)[0]:
                    continue
                X, Y = data_T[:, idx_i], data_T[:, idx_j]
                p = _mwu_rows(X, Y)
                muX, muY = np.mean(X, axis=1), np.mean(Y, axis=1)
                mdiff = muY - muX
                cdr = (muY - muX) / np.maximum(muY + muX, 1e-9)
                cols_p.append(p)
                cols_mdiff.append(mdiff)
                cols_cdr.append(cdr)
                names.append((title, f"{name_i} vs {name_j}"))
        # pooled pairs
        for i in range(len(pooled_entries)):
            for j in range(i + 1, len(pooled_entries)):
                name_i, idx_i = pooled_entries[i]
                name_j, idx_j = pooled_entries[j]
                if name_i.split(" ")[0] == name_j.split(" ")[0]:
                    continue
                X, Y = data_T[:, idx_i], data_T[:, idx_j]
                p = _mwu_rows(X, Y)
                muX, muY = np.mean(X, axis=1), np.mean(Y, axis=1)
                mdiff = muY - muX
                cdr = (muY - muX) / np.maximum(muY + muX, 1e-9)
                cols_p.append(p)
                cols_mdiff.append(mdiff)
                cols_cdr.append(cdr)
                names.append((title, f"{name_i} vs {name_j}"))

    # Combined Sex×Genotype intersections (Female/Male × wt/dKI × 2m/4m)
    if "sex_genotype" in factors:
        title = "Sex×Genotype"
        F_sex = factor_base_indices(3, label_variables, mask_groups)
        F_geno = factor_base_indices(2, label_variables, mask_groups)

        entries = []
        for sex_base, agesS in F_sex.items():
            for geno_base, agesG in F_geno.items():
                for age in ("2m", "4m"):
                    idxS = agesS.get(age)
                    idxG = agesG.get(age)
                    if idxS is None or idxG is None or len(idxS) == 0 or len(idxG) == 0:
                        continue
                    idx = np.intersect1d(idxS, idxG, assume_unique=False)
                    if idx.size:
                        entries.append((f"{sex_base} {geno_base}-{age}", idx))

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
                        pooled_entries.append(
                            (
                                f"{sex_base} {geno_base} (all-ages)",
                                np.unique(np.concatenate(parts)),
                            )
                        )

        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                name_i, idx_i = entries[i]
                name_j, idx_j = entries[j]
                if not cross_age:
                    if name_i.split("-")[-1] != name_j.split("-")[-1]:
                        continue
                base_i = name_i.rsplit("-", 1)[0]
                base_j = name_j.rsplit("-", 1)[0]
                if base_i == base_j:
                    continue
                X, Y = data_T[:, idx_i], data_T[:, idx_j]
                p = _mwu_rows(X, Y)
                muX, muY = np.mean(X, axis=1), np.mean(Y, axis=1)
                mdiff = muY - muX
                cdr = (muY - muX) / np.maximum(muY + muX, 1e-9)
                cols_p.append(p)
                cols_mdiff.append(mdiff)
                cols_cdr.append(cdr)
                names.append((title, f"{name_i} vs {name_j}"))

        for i in range(len(pooled_entries)):
            for j in range(i + 1, len(pooled_entries)):
                name_i, idx_i = pooled_entries[i]
                name_j, idx_j = pooled_entries[j]
                base_i = name_i.split(" (all-ages)")[0]
                base_j = name_j.split(" (all-ages)")[0]
                if base_i == base_j:
                    continue
                X, Y = data_T[:, idx_i], data_T[:, idx_j]
                p = _mwu_rows(X, Y)
                muX, muY = np.mean(X, axis=1), np.mean(Y, axis=1)
                mdiff = muY - muX
                cdr = (muY - muX) / np.maximum(muY + muX, 1e-9)
                cols_p.append(p)
                cols_mdiff.append(mdiff)
                cols_cdr.append(cdr)
                names.append((title, f"{name_i} vs {name_j}"))

        # POOLED ACROSS BASES PER AGE (to match curve plots)
        # Build four cohorts per age (2m/4m): Female wt, Female dKI, Male wt, Male dKI
        title_pooled = "Sex×Genotype (pooled)"
        for age in ("2m", "4m"):
            # unions for sex
            sex_union = {"Female": [], "Male": []}
            for sex_base, agesS in F_sex.items():
                if agesS.get(age) is None:
                    continue
                low = sex_base.lower()
                if "female" in low:
                    sex_union["Female"].append(agesS[age])
                elif "male" in low:
                    sex_union["Male"].append(agesS[age])
            sex_union = {
                k: (np.unique(np.concatenate(v)) if v else np.array([], dtype=int))
                for k, v in sex_union.items()
            }
            # unions for genotype
            geno_union = {"wt": [], "dKI": []}
            for geno_base, agesG in F_geno.items():
                if agesG.get(age) is None:
                    continue
                low = geno_base.lower()
                if "wt" in low:
                    geno_union["wt"].append(agesG[age])
                elif "dki" in low:
                    geno_union["dKI"].append(agesG[age])
            geno_union = {
                k: (np.unique(np.concatenate(v)) if v else np.array([], dtype=int))
                for k, v in geno_union.items()
            }

            cohorts = []  # list[(label, idx)]
            for sex_name in ("Female", "Male"):
                for geno_key in ("wt", "dKI"):
                    idx = np.intersect1d(
                        sex_union[sex_name], geno_union[geno_key], assume_unique=False
                    )
                    if idx.size:
                        cohorts.append((f"{sex_name} {geno_key}-{age}", idx))
            # pairwise across cohorts within this age
            for i in range(len(cohorts)):
                for j in range(i + 1, len(cohorts)):
                    name_i, idx_i = cohorts[i]
                    name_j, idx_j = cohorts[j]
                    X, Y = data_T[:, idx_i], data_T[:, idx_j]
                    p = _mwu_rows(X, Y)
                    muX, muY = np.mean(X, axis=1), np.mean(Y, axis=1)
                    mdiff = muY - muX
                    cdr = (muY - muX) / np.maximum(muY + muX, 1e-9)
                    cols_p.append(p)
                    cols_mdiff.append(mdiff)
                    cols_cdr.append(cdr)
                    names.append((title_pooled, f"{name_i} vs {name_j}"))
    if not cols_p:
        empty = pd.DataFrame(index=link_labels, columns=pd.MultiIndex.from_tuples([]))
        return empty, empty, empty
    P = np.column_stack(cols_p)
    MD = np.column_stack(cols_mdiff)
    CDR = np.column_stack(cols_cdr)
    cols = pd.MultiIndex.from_tuples(names, names=["Block", "Column"])
    return (
        pd.DataFrame(P, index=link_labels, columns=cols),
        pd.DataFrame(MD, index=link_labels, columns=cols),
        pd.DataFrame(CDR, index=link_labels, columns=cols),
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
    *,
    save: bool,
    out_path: Path | None,
):
    data = pvals_df.values
    mask = np.where(data <= alpha, data, np.nan)
    n_rows, n_cols = mask.shape if mask.size else (0, 0)
    fig, ax = plt.subplots(
        figsize=(max(15, 0.22 * n_cols), max(2, 0.16 * max(n_rows, 1)))
    )
    im = ax.imshow(
        mask, aspect="auto", interpolation="none", cmap="viridis_r", vmin=0, vmax=alpha
    )
    fig.colorbar(im, ax=ax).set_label("p-value")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(pvals_df.index, fontsize=6)
    col_labels = [f"{b} | {c}" for b, c in pvals_df.columns.to_list()]
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_labels, rotation=120, fontsize=8)
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
    *,
    save: bool,
    out_path: Path | None,
):
    assert tuple(pvals_df.columns) == tuple(weights_df.columns)
    assert list(pvals_df.index) == list(weights_df.index)
    p, w = pvals_df.values, weights_df.values
    Z = np.where(p <= alpha, 1 - p, np.nan) * w
    n_rows, n_cols = Z.shape if Z.size else (0, 0)
    fig, ax = plt.subplots(
        figsize=(max(15, 0.22 * n_cols), max(2, 0.16 * max(n_rows, 1)))
    )
    im = ax.imshow(
        Z, aspect="auto", interpolation="none", cmap="RdBu", vmin=-0.1, vmax=0.1
    )
    fig.colorbar(im, ax=ax).set_label("(1 - p) × effect")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(pvals_df.index, fontsize=6)
    col_labels = [f"{b} | {c}" for b, c in pvals_df.columns.to_list()]
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_labels, rotation=120, fontsize=8)
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


# -------- Per-comparison matrix helpers --------


def _infer_roi_order_from_pairs(pair_labels: np.ndarray) -> list[str]:
    # pair_labels: (L, 2) of strings
    firsts = [str(a) for a, _ in pair_labels]
    order: list[str] = []
    for a in firsts:
        if not order or a != order[-1]:
            if a not in order:
                order.append(a)
    last_b = str(pair_labels[-1][1]) if len(pair_labels) else None
    if last_b and last_b not in order:
        order.append(last_b)
    # Fallback: ensure all unique labels are included
    uniq = []
    for a, b in pair_labels:
        for z in (str(a), str(b)):
            if z not in uniq:
                uniq.append(z)
    for z in uniq:
        if z not in order:
            order.append(z)
    return order


def _vec_to_sym_matrix(
    vec: np.ndarray, pair_labels: np.ndarray, roi_order: list[str]
) -> np.ndarray:
    D = len(roi_order)
    M = np.full((D, D), np.nan, dtype=float)
    idx_map = {name: i for i, name in enumerate(roi_order)}
    for k, (a, b) in enumerate(pair_labels):
        i, j = idx_map[str(a)], idx_map[str(b)]
        val = float(vec[k]) if np.isscalar(vec[k]) or vec[k] is not None else np.nan
        M[i, j] = val
        M[j, i] = val
    np.fill_diagonal(M, 0.0)
    return M

def _sanitize_fname(s: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:200]

def plot_matrices_per_column_group(
    pvals_df: pd.DataFrame,
    *,
    alpha: float,
    mode: str,  # 'sig' or 'weighted'
    effect_df: pd.DataFrame | None,
    pair_labels: np.ndarray,
    roi_order: list[str],
    out_root: Path,
    tag_prefix: str,
    tag: str,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | None = None,
    save: bool = True,
):
    if pvals_df.empty:
        return []
    if mode == "weighted" and effect_df is None:
        raise ValueError("effect_df is required when mode='weighted'")
    assert (
        list(pvals_df.index) == list(effect_df.index) if effect_df is not None else True
    )

    out_paths = []
    # Group by first level (Block)
    for block in pvals_df.columns.get_level_values(0).unique():
        sub_cols = [c for c in pvals_df.columns if c[0] == block]
        P = pvals_df.loc[:, sub_cols].values  # (L, K)
        if effect_df is not None:
            W = effect_df.loc[:, sub_cols].values  # aligned
        labels = [c[1] for c in sub_cols]

        # Directory per block
        block_dir = out_root / _sanitize_fname(f"{tag_prefix}_{block}")
        block_dir.mkdir(parents=True, exist_ok=True)

        for k, label in enumerate(labels):
            if mode == "sig":
                vec = np.where(P[:, k] <= alpha, 1.0, np.nan)
                _vmin = 0.0 if vmin is None else vmin
                _vmax = 1.0 if vmax is None else vmax
                _cmap = "Greens" if cmap is None else cmap
                title = f"{block} — {label} (significant)"
                fname = (
                    block_dir
                    / f"{_sanitize_fname(label)}_sig_{_sanitize_fname(tag)}.png"
                )
            else:  # weighted
                vec = np.where(P[:, k] <= alpha, 1 - P[:, k], np.nan) * W[:, k]
                _vmin = -0.1 if vmin is None else vmin
                _vmax = 0.1 if vmax is None else vmax
                _cmap = "RdBu" if cmap is None else cmap
                title = f"{block} - {label} - Cohesion difference ratio"
                fname = (
                    block_dir
                    / f"{_sanitize_fname(label)}_weighted_{_sanitize_fname(tag)}.png"
                )

            M = _vec_to_sym_matrix(vec, pair_labels, roi_order)
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(M, interpolation="none", cmap=_cmap, vmin=_vmin, vmax=_vmax)
            ax.set_title(title)
            ax.set_xticks(np.arange(len(roi_order)))
            ax.set_yticks(np.arange(len(roi_order)))
            ax.set_xticklabels(roi_order, rotation=90, fontsize=7)
            ax.set_yticklabels(roi_order, fontsize=7)
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            if save:
                fig.savefig(fname, dpi=300, bbox_inches="tight")
                out_paths.append(str(fname))
            plt.close(fig)
    return out_paths

def _bonferroni_adjust(pvals_df: pd.DataFrame) -> pd.DataFrame:
    if pvals_df.empty:
        return pvals_df.copy()
    m = pvals_df.shape[0]  # number of links per column
    arr = np.minimum(1.0, np.asarray(pvals_df.values, dtype=float) * float(m))
    return pd.DataFrame(arr, index=pvals_df.index, columns=pvals_df.columns)

def _bonferroni_by_age_in_columns(pvals_df: pd.DataFrame) -> pd.DataFrame:
    """Apply Bonferroni across columns grouped by age (2m/4m) per link.

    Only columns whose label clearly ends with '-2m' or '-4m' on both sides are adjusted.
    Cross-age and pooled ('all-ages') columns are left unchanged.
    """
    if pvals_df.empty:
        return pvals_df.copy()
    cols = list(pvals_df.columns)
    # Build age groups
    group_map: dict[str, list[int]] = {"2m": [], "4m": []}
    for j, col in enumerate(cols):
        # col is (Block, Label)
        label = str(col[1])
        # Detect forms like 'Female-2m vs Male-2m'
        try:
            lhs, rhs = label.split(" vs ")
            age_l = lhs.split("-")[-1]
            age_r = rhs.split("-")[-1]
            if age_l in {"2m", "4m"} and age_l == age_r:
                group_map[age_l].append(j)
        except Exception:
            continue
    # Apply per group
    A = np.asarray(pvals_df.values, dtype=float)
    for age, idxs in group_map.items():
        if not idxs:
            continue
        k = float(len(idxs))
        A[:, idxs] = np.minimum(1.0, A[:, idxs] * k)
    return pd.DataFrame(A, index=pvals_df.index, columns=pvals_df.columns)

def main() -> int:
    setup_logging()
    args = parse_args()
    if args.no_show:
        matplotlib.use("Agg", force=True)

    # Paths
    paths = get_paths(timecourse_folder=args.timecourse_folder)
    stats_fig_dir = (paths["f_cohesion"] / "stats").expanduser()
    stats_fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir = (paths["allegiance"] / "out").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scope (file suffix) — respect explicit choice
    scope = args.roi_scope
    data = load_npz(paths, args.window_size, args.lag, args.tau, scope)
    time_ratio = np.asarray(data["time_ratio"]).astype(float)  # (A, L)
    pair_labels = np.asarray(data["pair_labels"])  # (L, 2)
    anat_labels_sorted = np.asarray(data.get("anat_labels_sorted", []))
    # ROI order used in computing; infer from pairs if not present
    roi_order = _infer_roi_order_from_pairs(pair_labels)
    link_labels = [f"{a}–{b}" for a, b in pair_labels]

    # Table input
    data_T = time_ratio.T  # (L, A)
    tag = f"w{args.window_size}_lag{args.lag}_tau{args.tau}_{scope}"

    # Load grouping
    with open(paths["preprocessed"] / "grouping_data_oip.pkl", "rb") as f:
        mask_groups, label_variables = pickle.load(f)

    save_stats = args.emit in {"stats", "both"}
    do_plots = args.emit in {"plots", "both"}

    if args.with_stats:
        # Age-paired
        if args.stats_mode in {"age", "all"}:
            spec = extend_block_spec_with_phenotype(args.include_phenotype)
            p_wil = build_table_from_spec(
                data_T, link_labels, label_variables, mask_groups, spec, _wilcoxon_rows
            )
            p_t = build_table_from_spec(
                data_T, link_labels, label_variables, mask_groups, spec, _ttest_rows
            )
            eff_cdr = build_table_from_spec(
                data_T,
                link_labels,
                label_variables,
                mask_groups,
                spec,
                _cohesion_diff_rows,
            )
            eff_mdiff = build_table_from_spec(
                data_T,
                link_labels,
                label_variables,
                mask_groups,
                spec,
                lambda X, Y: np.mean(Y, axis=1) - np.mean(X, axis=1),
            )

            # Save raw p-values
            if save_stats:
                p_wil.to_csv(out_dir / f"pvals_age_wilcoxon_{tag}.csv")
                p_t.to_csv(out_dir / f"pvals_age_ttest_{tag}.csv")
                eff_cdr.to_csv(out_dir / f"effects_age_cdratio_{tag}.csv")
                eff_mdiff.to_csv(out_dir / f"effects_age_mdiff_{tag}.csv")

            # Adjusted p-values (optional)
            title_suffix = ""
            p_wil_plot = p_wil
            if args.p_adjust == "bonferroni":
                p_wil_b = _bonferroni_adjust(p_wil)
                if save_stats:
                    p_wil_b.to_csv(out_dir / f"pvals_age_wilcoxon_bonferroni_{tag}.csv")
                p_wil_plot = p_wil_b
                title_suffix = " (Bonferroni)"

            if do_plots:
                fig1 = plot_sig_pvals_multi(
                    p_wil_plot,
                    args.alpha,
                    f"Age (2m vs 4m) — Wilcoxon significant only{title_suffix}",
                    save=args.save_plots,
                    out_path=stats_fig_dir / f"pvals_age_wilcoxon_sig_{tag}.png",
                )
                if not args.no_show:
                    plt.show()
                else:
                    plt.close(fig1)
                fig2 = plot_weighted_multi(
                    p_wil_plot,
                    eff_cdr,
                    args.alpha,
                    f"Age (2m vs 4m) — (1 - p) × cohesion-diff ratio{title_suffix}",
                    save=args.save_plots,
                    out_path=stats_fig_dir / f"weighted_age_wilcoxon_cdratio_{tag}.png",
                )
                if not args.no_show:
                    plt.show()
                else:
                    plt.close(fig2)

            # Optional per-comparison matrices for AGE
            if do_plots and args.matrix_per_comparison:
                if args.matrix_mode == "weighted":
                    eff_choice = (
                        eff_cdr if args.matrix_effect == "cdratio" else eff_mdiff
                    )
                else:
                    eff_choice = None
                out_root = stats_fig_dir / f"matrices_{tag}"
                plot_matrices_per_column_group(
                    p_wil,
                    alpha=args.alpha,
                    mode=args.matrix_mode,
                    effect_df=eff_choice,
                    pair_labels=pair_labels,
                    roi_order=roi_order,
                    out_root=out_root,
                    tag_prefix="Age",
                    tag=tag,
                    save=args.save_plots,
                )

        # Group-based (MWU)
        if args.stats_mode in {"group", "all"}:
            dims = []
            if args.group_compare in {"sex", "both"}:
                dims.append("sex")
            if args.group_compare in {"genotype", "both"}:
                dims.append("genotype")
            if args.group_compare == "sex_genotype":
                dims.append("sex_genotype")
            p_grp, eff_mdiff_g, eff_cdr_g = build_group_comparisons(
                data_T,
                link_labels,
                label_variables,
                mask_groups,
                factors=dims,
                cross_age=args.cross_age,
                pooled=args.pool_ages,
            )
            if save_stats:
                p_grp.to_csv(out_dir / f"pvals_group_mwu_{tag}.csv")
                eff_mdiff_g.to_csv(out_dir / f"effects_group_mdiff_{tag}.csv")
                eff_cdr_g.to_csv(out_dir / f"effects_group_cdratio_{tag}.csv")

            title_suffix = ""
            p_grp_plot = p_grp
            if args.p_adjust == "bonferroni":
                p_grp_b = _bonferroni_adjust(p_grp)
                if save_stats:
                    p_grp_b.to_csv(out_dir / f"pvals_group_mwu_bonferroni_{tag}.csv")
                p_grp_plot = p_grp_b
                title_suffix = " (Bonferroni)"
            elif args.p_adjust == "bonferroni-age":
                p_grp_ba = _bonferroni_by_age_in_columns(p_grp)
                if save_stats:
                    p_grp_ba.to_csv(
                        out_dir / f"pvals_group_mwu_bonferroni_age_{tag}.csv"
                    )
                p_grp_plot = p_grp_ba
                title_suffix = " (Bonferroni by age group)"

            if do_plots:
                figg1 = plot_sig_pvals_multi(
                    p_grp_plot,
                    args.alpha,
                    f"Group (MWU) — significant only{title_suffix}",
                    save=args.save_plots,
                    out_path=stats_fig_dir / f"pvals_group_mwu_sig_{tag}.png",
                )
                if not args.no_show:
                    plt.show()
                else:
                    plt.close(figg1)
                figg2 = plot_weighted_multi(
                    p_grp_plot,
                    eff_mdiff_g,
                    args.alpha,
                    f"Group (MWU) — (1 - p) × mean difference{title_suffix}",
                    save=args.save_plots,
                    out_path=stats_fig_dir / f"weighted_group_mwu_mdiff_{tag}.png",
                )
                if not args.no_show:
                    plt.show()
                else:
                    plt.close(figg2)
                figg3 = plot_weighted_multi(
                    p_grp_plot,
                    eff_cdr_g,
                    args.alpha,
                    f"Group (MWU) — (1 - p) × cohesion-diff ratio{title_suffix}",
                    save=args.save_plots,
                    out_path=stats_fig_dir / f"weighted_group_mwu_cdratio_{tag}.png",
                )
                if not args.no_show:
                    plt.show()
                else:
                    plt.close(figg3)

            # Optional per-comparison matrices for GROUP
            if do_plots and args.matrix_per_comparison:
                if args.matrix_mode == "weighted":
                    eff_choice = (
                        eff_cdr_g if args.matrix_effect == "cdratio" else eff_mdiff_g
                    )
                else:
                    eff_choice = None
                out_root = stats_fig_dir / f"matrices_{tag}"
                plot_matrices_per_column_group(
                    p_grp,
                    alpha=args.alpha,
                    mode=args.matrix_mode,
                    effect_df=eff_choice,
                    pair_labels=pair_labels,
                    roi_order=roi_order,
                    out_root=out_root,
                    tag_prefix="Group",
                    tag=tag,
                    save=args.save_plots,
                )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
