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
import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu

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
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--save-plots", action="store_true")
    p.add_argument("--no-show", action="store_true")

    p.add_argument("--with-stats", action="store_true", help="run stats and export tables + heatmaps")
    p.add_argument("--stats-mode", choices=["age", "group", "all"], default="all")
    p.add_argument("--group-compare", choices=["sex", "genotype", "both"], default="both")
    p.add_argument("--cross-age", action="store_true")
    p.add_argument("--pool-ages", action="store_true")
    p.add_argument("--include-phenotype", choices=["none", "oip", "nor", "both"], default="none")
    return p.parse_args()


def load_npz(paths: dict, ws: int, lag: int, tau: int, scope: str) -> dict:
    f = paths["allegiance"] / "cohesion_data" / f"cohesion_data_w{ws}_lag{lag}_tau{tau}_{scope}.npz"
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


def factor_base_indices(factor_idx: int, label_variables, mask_groups) -> dict[str, dict[str, np.ndarray | None]]:
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


def _cols_single(block: str, fidx: int, data_T: np.ndarray, link_labels, label_variables, mask_groups, value_fn):
    F = factor_base_indices(fidx, label_variables, mask_groups)
    keys, cols = [], []
    for base, ages in F.items():
        idx2, idx4 = ages.get("2m"), ages.get("4m")
        if idx2 is None or idx4 is None or len(idx2) == 0 or len(idx4) == 0 or len(idx2) != len(idx4):
            continue
        X = data_T[:, idx2]
        Y = data_T[:, idx4]
        keys.append((block, base))
        cols.append(value_fn(X, Y))
    return keys, cols


def _cols_pair(block: str, fA: int, fB: int, data_T, link_labels, label_variables, mask_groups, value_fn):
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


def build_table_from_spec(data_T: np.ndarray, link_labels, label_variables, mask_groups, block_spec, value_fn) -> pd.DataFrame:
    all_keys, all_cols = [], []
    for item in block_spec:
        if item[1] == "single":
            k, c = _cols_single(item[0], item[2], data_T, link_labels, label_variables, mask_groups, value_fn)
        else:
            k, c = _cols_pair(item[0], item[2], item[3], data_T, link_labels, label_variables, mask_groups, value_fn)
        all_keys += k
        all_cols += c
    if not all_cols:
        return pd.DataFrame(index=link_labels, columns=pd.MultiIndex.from_tuples([]))
    M = np.column_stack(all_cols)
    columns = pd.MultiIndex.from_tuples(all_keys, names=["Block", "Column"])
    return pd.DataFrame(M, index=link_labels, columns=columns)


def build_group_comparisons(data_T, link_labels, label_variables, mask_groups, *, factors: list[str], cross_age: bool, pooled: bool):
    factor_map = {"sex": ("Sex", 3), "genotype": ("Genotype", 2)}
    cols_p, cols_mdiff, cols_cdr, names = [], [], [], []
    for key in factors:
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
                    pooled_entries.append((f"{b} (all-ages)", np.unique(np.concatenate(parts))))
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
                cols_p.append(p); cols_mdiff.append(mdiff); cols_cdr.append(cdr)
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
                cols_p.append(p); cols_mdiff.append(mdiff); cols_cdr.append(cdr)
                names.append((title, f"{name_i} vs {name_j}"))
    if not cols_p:
        empty = pd.DataFrame(index=link_labels, columns=pd.MultiIndex.from_tuples([]))
        return empty, empty, empty
    P = np.column_stack(cols_p)
    MD = np.column_stack(cols_mdiff)
    CDR = np.column_stack(cols_cdr)
    cols = pd.MultiIndex.from_tuples(names, names=["Block", "Column"])
    return pd.DataFrame(P, index=link_labels, columns=cols), pd.DataFrame(MD, index=link_labels, columns=cols), pd.DataFrame(CDR, index=link_labels, columns=cols)


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


def plot_sig_pvals_multi(pvals_df: pd.DataFrame, alpha: float, title: str, *, save: bool, out_path: Path | None):
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


def plot_weighted_multi(pvals_df: pd.DataFrame, weights_df: pd.DataFrame, alpha: float, title: str, *, save: bool, out_path: Path | None):
    assert tuple(pvals_df.columns) == tuple(weights_df.columns)
    assert list(pvals_df.index) == list(weights_df.index)
    p, w = pvals_df.values, weights_df.values
    Z = np.where(p <= alpha, 1 - p, np.nan) * w
    n_rows, n_cols = Z.shape if Z.size else (0, 0)
    fig, ax = plt.subplots(figsize=(max(15, 0.22 * n_cols), max(2, 0.16 * max(n_rows, 1))))
    im = ax.imshow(Z, aspect="auto", interpolation="none", cmap="RdBu", vmin=-0.1, vmax=0.1)
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


def main() -> int:
    setup_logging()
    args = parse_args()
    if args.no_show:
        matplotlib.use("Agg", force=True)

    # Paths
    paths = get_paths(timecourse_folder=args.timecourse_folder)
    stats_fig_dir = (paths["f_cohesion"] / "stats").expanduser(); stats_fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir = (paths["allegiance"] / "out").expanduser(); out_dir.mkdir(parents=True, exist_ok=True)

    # Scope (file suffix)
    scope = "dmn" if args.dmn_index.strip() else "all"
    data = load_npz(paths, args.window_size, args.lag, args.tau, scope)
    time_ratio = np.asarray(data["time_ratio"]).astype(float)  # (A, L)
    pair_labels = np.asarray(data["pair_labels"])  # (L, 2)
    link_labels = [f"{a}–{b}" for a, b in pair_labels]

    # Table input
    data_T = time_ratio.T  # (L, A)
    tag = f"w{args.window_size}_lag{args.lag}_tau{args.tau}_{scope}"

    # Load grouping
    with open(paths["preprocessed"] / "grouping_data_oip.pkl", "rb") as f:
        mask_groups, label_variables = pickle.load(f)

    if args.with_stats:
        # Age-paired
        if args.stats_mode in {"age", "all"}:
            spec = extend_block_spec_with_phenotype(args.include_phenotype)
            p_wil = build_table_from_spec(data_T, link_labels, label_variables, mask_groups, spec, _wilcoxon_rows)
            p_t = build_table_from_spec(data_T, link_labels, label_variables, mask_groups, spec, _ttest_rows)
            eff_cdr = build_table_from_spec(data_T, link_labels, label_variables, mask_groups, spec, _cohesion_diff_rows)
            eff_mdiff = build_table_from_spec(data_T, link_labels, label_variables, mask_groups, spec, lambda X, Y: np.mean(Y, axis=1) - np.mean(X, axis=1))

            p_wil.to_csv(out_dir / f"pvals_age_wilcoxon_{tag}.csv")
            p_t.to_csv(out_dir / f"pvals_age_ttest_{tag}.csv")
            eff_cdr.to_csv(out_dir / f"effects_age_cdratio_{tag}.csv")
            eff_mdiff.to_csv(out_dir / f"effects_age_mdiff_{tag}.csv")

            fig1 = plot_sig_pvals_multi(p_wil, args.alpha, "Age (2m vs 4m) — Wilcoxon significant only", save=args.save_plots, out_path=stats_fig_dir / f"pvals_age_wilcoxon_sig_{tag}.png")
            if not args.no_show: plt.show()
            else: plt.close(fig1)
            fig2 = plot_weighted_multi(p_wil, eff_cdr, args.alpha, "Age (2m vs 4m) — (1 - p) × cohesion-diff ratio", save=args.save_plots, out_path=stats_fig_dir / f"weighted_age_wilcoxon_cdratio_{tag}.png")
            if not args.no_show: plt.show()
            else: plt.close(fig2)

        # Group-based (MWU)
        if args.stats_mode in {"group", "all"}:
            dims = []
            if args.group_compare in {"sex", "both"}: dims.append("sex")
            if args.group_compare in {"genotype", "both"}: dims.append("genotype")
            p_grp, eff_mdiff_g, eff_cdr_g = build_group_comparisons(data_T, link_labels, label_variables, mask_groups, factors=dims, cross_age=args.cross_age, pooled=args.pool_ages)
            p_grp.to_csv(out_dir / f"pvals_group_mwu_{tag}.csv")
            eff_mdiff_g.to_csv(out_dir / f"effects_group_mdiff_{tag}.csv")
            eff_cdr_g.to_csv(out_dir / f"effects_group_cdratio_{tag}.csv")

            figg1 = plot_sig_pvals_multi(p_grp, args.alpha, "Group (MWU) — significant only", save=args.save_plots, out_path=stats_fig_dir / f"pvals_group_mwu_sig_{tag}.png")
            if not args.no_show: plt.show()
            else: plt.close(figg1)
            figg2 = plot_weighted_multi(p_grp, eff_mdiff_g, args.alpha, "Group (MWU) — (1 - p) × mean difference", save=args.save_plots, out_path=stats_fig_dir / f"weighted_group_mwu_mdiff_{tag}.png")
            if not args.no_show: plt.show()
            else: plt.close(figg2)
            figg3 = plot_weighted_multi(p_grp, eff_cdr_g, args.alpha, "Group (MWU) — (1 - p) × cohesion-diff ratio", save=args.save_plots, out_path=stats_fig_dir / f"weighted_group_mwu_cdratio_{tag}.png")
            if not args.no_show: plt.show()
            else: plt.close(figg3)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

