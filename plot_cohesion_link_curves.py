#!/usr/bin/env python3
"""
Plot cohesion curves for selected links and annotate p-values.

- Loads cohesion summaries from cohesion_compute.py NPZ.
- Filters links whose ROI labels contain any of the provided substrings.
- Default comparison: Age (2m vs 4m), unpaired Mann-Whitney U test.
  (Paired line plots can be added later if strict pairing indices are provided.)

Figures are saved under fig/<dataset>/cohesion/link_curves/.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory
from scipy.stats import mannwhitneyu

from shared_code.fun_paths import get_paths

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    if logger.handlers:
        return
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot cohesion curves for selected links")
    parser.add_argument("--window-size", type=int, default=9, dest="window_size")
    parser.add_argument("--lag", type=int, default=1, dest="lag")
    parser.add_argument("--tau", type=int, default=3, dest="tau")
    parser.add_argument(
        "--timecourse-folder",
        type=str,
        default="Timecourses_updated_03052024",
        dest="timecourse_folder",
    )
    parser.add_argument(
        "--roi-scope",
        choices=["all", "dmn", "memory", "custom"],
        default="all",
        help="Scope used when computing cohesion (affects filename suffix)",
    )
    parser.add_argument(
        "--roi-substrings",
        type=str,
        default="d HIP,v HIP,RSP",
        help="Comma-separated substrings to match in ROI labels (case-insensitive)",
    )
    parser.add_argument("--tag", type=str, default="", help="Optional tag appended in NPZ filename")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for annotation")
    parser.add_argument(
        "--paired-age",
        action="store_true",
        help="Use paired 2m–4m animals (Wilcoxon) and draw per-animal lines",
    )
    parser.add_argument(
        "--color-by",
        choices=["age", "sex", "genotype", "both", "sex_genotype"],
        default="age",
        help="Color coding: age, sex, genotype, both (genotype), or sex_genotype (4 groups)",
    )
    parser.add_argument(
        "--errorbar",
        choices=["sd", "sem", "var"],
        default="sd",
        help="Errorbar type for aggregates: standard deviation (sd), standard error (sem), or variance (var)",
    )
    parser.add_argument("--no-stats", action="store_true", help="Do not compute or display any p-values on the plots")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Plot only mean ± SEM per group (no individual points or paired lines)",
    )
    parser.add_argument(
        "--annotate-stats",
        choices=["none", "age", "group", "both"],
        default="none",
        help="Annotate significance from stats CSVs: age (Wilcoxon), group (MWU), or both",
    )
    parser.add_argument("--save-plots", action="store_true", help="Save figures to disk")
    parser.add_argument("--no-show", action="store_true", help="Headless mode (no display)")
    return parser.parse_known_args()[0]


def load_npz(paths: dict, ws: int, lag: int, tau: int, scope: str, tag: str = "") -> dict:
    suffix = f"_{tag.strip()}" if tag and tag.strip() else ""
    fpath = (
        paths["allegiance"]
        / "cohesion_data"
        / f"cohesion_data_w{ws}_lag{lag}_tau{tau}_{scope}{suffix}.npz"
    )
    return dict(np.load(fpath, allow_pickle=True))


@dataclass
class GroupConfig:
    name: str
    color: str
    offset: float
    selector: Callable[[str], bool]


@dataclass
class GroupValues:
    name: str
    color: str
    offset: float
    values_2m: np.ndarray
    values_4m: np.ndarray
    p_age: float | None


# ---------------------------------------------------------------------------
# Pattern helpers
# ---------------------------------------------------------------------------

def _compile_patterns(spec: str) -> list[re.Pattern[str]]:
    tokens = [token.strip() for token in spec.split(",") if token.strip()]
    return [re.compile(re.escape(token), flags=re.IGNORECASE) for token in tokens]


def _link_matches(pair: tuple[str, str], patterns: Iterable[re.Pattern[str]]) -> bool:
    a, b = pair
    return any(p.search(str(a)) or p.search(str(b)) for p in patterns)


def _errbar(arr: np.ndarray, mode: str) -> float:
    if arr.size == 0:
        return np.nan
    if arr.size == 1:
        return 0.0
    if mode == "sd":
        return float(np.std(arr, ddof=1))
    if mode == "sem":
        sd = float(np.std(arr, ddof=1))
        return sd / float(np.sqrt(arr.size))
    return float(np.var(arr, ddof=1))


def _annotate_inset(ax: plt.Axes, lines: list[str]) -> None:
    if not lines:
        return
    text = "\n".join(lines)
    ax.text(
        0.98,
        0.02,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"),
    )


def _add_sig_bar_axes(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y_axes: float,
    text: str,
    h_axes: float = 0.02,
    lw: float = 1.0,
    fontsize: int = 8,
) -> None:
    """Draw a significance bar in axes coordinates (x=data, y=axes)."""

    transform = blended_transform_factory(ax.transData, ax.transAxes)
    match = re.search(r"p\s*=\s*([0-9]*\.?[0-9]+(?:e-?\d+)?)", text, re.I)
    p_val = float(match.group(1)) if match else None
    label_color = "red" if (p_val is not None and p_val < 0.05) else "black"

    y0, y1 = y_axes, y_axes + h_axes
    ax.plot([x1, x1, x2, x2], [y0, y1, y1, y0], transform=transform, color="black", lw=lw, clip_on=False)

    va = "bottom" if h_axes >= 0 else "top"
    ax.text(
        (x1 + x2) * 0.5,
        y1,
        text,
        transform=transform,
        ha="center",
        va=va,
        fontsize=fontsize,
        color=label_color,
        clip_on=False,
    )


def _safe_concat(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.array([], dtype=int)
    return np.unique(np.concatenate(chunks))


def _mannwhitney_safe(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size == 0 or right.size == 0:
        return None
    try:
        return float(mannwhitneyu(left, right, alternative="two-sided").pvalue)
    except ValueError:
        return None


def _wilcoxon_safe(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size == 0 or right.size == 0:
        return None
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(left, right, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return None


def _paired_from_F(
    fmap: dict[str, dict[str, np.ndarray | None]],
    link_idx: int,
    time_ratio: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    v2_list: list[np.ndarray] = []
    v4_list: list[np.ndarray] = []
    for ages in fmap.values():
        i2, i4 = ages.get("2m"), ages.get("4m")
        if (
            i2 is None
            or i4 is None
            or len(i2) == 0
            or len(i4) == 0
            or len(i2) != len(i4)
        ):
            continue
        v2_list.append(time_ratio[np.asarray(i2, dtype=int), link_idx])
        v4_list.append(time_ratio[np.asarray(i4, dtype=int), link_idx])
    if v2_list and v4_list:
        return np.concatenate(v2_list), np.concatenate(v4_list)
    return np.array([]), np.array([])


# ---------------------------------------------------------------------------
# Group builders
# ---------------------------------------------------------------------------

def _factor_base_indices(
    mask_groups: list,
    label_variables: list,
    factor_idx: int,
) -> dict[str, dict[str, np.ndarray | None]]:
    bases: dict[str, dict[str, np.ndarray | None]] = {}
    labels = label_variables[factor_idx]
    masks = mask_groups[factor_idx]
    for label, mask in zip(labels, masks, strict=False):
        parts = str(label).split()
        age = parts[-1] if parts and parts[-1] in {"2m", "4m"} else None
        base = " ".join(parts[:-1]) if age else str(label)
        if age not in {"2m", "4m"}:
            continue
        idx = np.flatnonzero(np.asarray(mask, dtype=bool))
        entry = bases.setdefault(base, {"2m": None, "4m": None})
        entry[age] = idx
    return bases


def _build_sex_groups(
    F: dict[str, dict[str, np.ndarray | None]],
    link_idx: int,
    time_ratio: np.ndarray,
    paired: bool,
) -> list[GroupValues]:
    definitions = [
        GroupConfig("Female", "tab:purple", -0.08, lambda base: "female" in base.lower()),
        GroupConfig("Male", "tab:green", 0.08, lambda base: "male" in base.lower()),
    ]
    results: list[GroupValues] = []
    for config in definitions:
        idx2_chunks: list[np.ndarray] = []
        idx4_chunks: list[np.ndarray] = []
        restricted: dict[str, dict[str, np.ndarray | None]] = {}
        for base, ages in F.items():
            if not config.selector(base):
                continue
            restricted[base] = ages
            if ages.get("2m") is not None:
                idx2_chunks.append(np.asarray(ages["2m"], dtype=int))
            if ages.get("4m") is not None:
                idx4_chunks.append(np.asarray(ages["4m"], dtype=int))
        idx2 = _safe_concat(idx2_chunks)
        idx4 = _safe_concat(idx4_chunks)
        if paired:
            v2, v4 = _paired_from_F(restricted, link_idx, time_ratio)
            p_age = _wilcoxon_safe(v2, v4)
        else:
            v2 = time_ratio[idx2, link_idx]
            v4 = time_ratio[idx4, link_idx]
            p_age = _mannwhitney_safe(v2, v4)
        results.append(GroupValues(config.name, config.color, config.offset, v2, v4, p_age))
    return [g for g in results if g.values_2m.size or g.values_4m.size]


def _build_genotype_groups(
    F_geno: dict[str, dict[str, np.ndarray | None]],
    link_idx: int,
    time_ratio: np.ndarray,
    paired: bool,
) -> list[GroupValues]:
    definitions = [
        GroupConfig("wt", "tab:blue", -0.08, lambda base: "wt" in base.lower()),
        GroupConfig("dKI", "tab:red", 0.08, lambda base: "dki" in base.lower()),
    ]
    results: list[GroupValues] = []
    for config in definitions:
        idx2_chunks: list[np.ndarray] = []
        idx4_chunks: list[np.ndarray] = []
        restricted: dict[str, dict[str, np.ndarray | None]] = {}
        for base, ages in F_geno.items():
            if not config.selector(base):
                continue
            restricted[base] = ages
            if ages.get("2m") is not None:
                idx2_chunks.append(np.asarray(ages["2m"], dtype=int))
            if ages.get("4m") is not None:
                idx4_chunks.append(np.asarray(ages["4m"], dtype=int))
        idx2 = _safe_concat(idx2_chunks)
        idx4 = _safe_concat(idx4_chunks)
        if paired:
            present2 = np.zeros(time_ratio.shape[0], dtype=bool)
            present4 = np.zeros(time_ratio.shape[0], dtype=bool)
            present2[idx2] = True
            present4[idx4] = True
            paired_idx = np.flatnonzero(present2 & present4)
            v2 = time_ratio[paired_idx, link_idx]
            v4 = time_ratio[paired_idx, link_idx]
            p_age = _wilcoxon_safe(v2, v4)
        else:
            v2 = time_ratio[idx2, link_idx]
            v4 = time_ratio[idx4, link_idx]
            p_age = _mannwhitney_safe(v2, v4)
        results.append(GroupValues(config.name, config.color, config.offset, v2, v4, p_age))
    return [g for g in results if g.values_2m.size or g.values_4m.size]


def _build_sex_genotype_groups(
    F: dict[str, dict[str, np.ndarray | None]],
    F_geno: dict[str, dict[str, np.ndarray | None]],
    link_idx: int,
    time_ratio: np.ndarray,
    paired: bool,
) -> list[GroupValues]:
    combos = [
        ("Female", "wt", "tab:blue", -0.12),
        ("Female", "dKI", "tab:red", -0.04),
        ("Male", "wt", "tab:green", 0.04),
        ("Male", "dKI", "tab:orange", 0.12),
    ]
    results: list[GroupValues] = []
    for sex_name, geno_key, color, offset in combos:
        sex_idx2: list[np.ndarray] = []
        sex_idx4: list[np.ndarray] = []
        for base, ages in F.items():
            if sex_name.lower() in base.lower():
                if ages.get("2m") is not None:
                    sex_idx2.append(np.asarray(ages["2m"], dtype=int))
                if ages.get("4m") is not None:
                    sex_idx4.append(np.asarray(ages["4m"], dtype=int))
        geno_idx2: list[np.ndarray] = []
        geno_idx4: list[np.ndarray] = []
        for base, ages in F_geno.items():
            low = base.lower()
            if (geno_key == "wt" and "wt" in low) or (geno_key == "dKI" and "dki" in low):
                if ages.get("2m") is not None:
                    geno_idx2.append(np.asarray(ages["2m"], dtype=int))
                if ages.get("4m") is not None:
                    geno_idx4.append(np.asarray(ages["4m"], dtype=int))
        idx2 = np.intersect1d(_safe_concat(sex_idx2), _safe_concat(geno_idx2), assume_unique=False)
        idx4 = np.intersect1d(_safe_concat(sex_idx4), _safe_concat(geno_idx4), assume_unique=False)
        if paired:
            L = min(len(idx2), len(idx4))
            v2 = time_ratio[idx2[:L], link_idx]
            v4 = time_ratio[idx4[:L], link_idx]
            p_age = _wilcoxon_safe(v2, v4)
        else:
            v2 = time_ratio[idx2, link_idx]
            v4 = time_ratio[idx4, link_idx]
            p_age = _mannwhitney_safe(v2, v4)
        results.append(GroupValues(f"{sex_name} {geno_key}", color, offset, v2, v4, p_age))
    return [g for g in results if g.values_2m.size or g.values_4m.size]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_group_values(
    ax: plt.Axes,
    group: GroupValues,
    error_mode: str,
    aggregate_only: bool,
    paired_age: bool,
) -> None:
    mu2 = float(np.mean(group.values_2m)) if group.values_2m.size else np.nan
    mu4 = float(np.mean(group.values_4m)) if group.values_4m.size else np.nan
    eb2 = _errbar(group.values_2m, error_mode)
    eb4 = _errbar(group.values_4m, error_mode)
    ax.errorbar(
        [0 + group.offset, 1 + group.offset],
        [mu2, mu4],
        yerr=[eb2, eb4],
        fmt="-o",
        color=group.color,
        capsize=3,
        lw=1.5,
        label=group.name,
    )
    if aggregate_only:
        return
    jitter = 0.05
    if group.values_2m.size:
        ax.scatter(
            np.zeros_like(group.values_2m) + group.offset + (np.random.rand(len(group.values_2m)) - 0.5) * jitter,
            group.values_2m,
            s=12,
            alpha=0.6,
            color=group.color,
        )
    if group.values_4m.size:
        ax.scatter(
            np.ones_like(group.values_4m) + group.offset + (np.random.rand(len(group.values_4m)) - 0.5) * jitter,
            group.values_4m,
            s=12,
            alpha=0.6,
            color=group.color,
        )
    if paired_age:
        count = min(len(group.values_2m), len(group.values_4m))
        for i in range(count):
            ax.plot(
                [0 + group.offset, 1 + group.offset],
                [group.values_2m[i], group.values_4m[i]],
                color=group.color,
                lw=0.6,
                alpha=0.2,
            )


def _draw_sig_bars(ax: plt.Axes, bars: list[tuple[float, float, str]], start: float = 0.94) -> None:
    if not bars:
        return
    gap = 0.06
    height = -0.015
    y_here = start
    for x1, x2, label in bars:
        _add_sig_bar_axes(ax, x1, x2, y_here, text=label, h_axes=height)
        y_here -= gap


def _group_pairwise_stats(
    groups: list[GroupValues],
    age: str,
) -> list[tuple[int, int, float | None]]:
    data: list[tuple[int, int, float | None]] = []
    if age not in {"2m", "4m"}:
        return data
    values_attr = "values_2m" if age == "2m" else "values_4m"
    for i, j in combinations(range(len(groups)), 2):
        left = getattr(groups[i], values_attr)
        right = getattr(groups[j], values_attr)
        p_val = _mannwhitney_safe(left, right)
        data.append((i, j, p_val))
    return data


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> int:
    setup_logging()
    args = parse_args()

    if args.no_show:
        matplotlib.use("Agg", force=True)

    paths = get_paths(timecourse_folder=args.timecourse_folder)
    out_dir = (paths["f_cohesion"] / "link_curves").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    scope = args.roi_scope
    data = load_npz(paths, args.window_size, args.lag, args.tau, scope, args.tag)
    time_ratio = np.asarray(data["time_ratio"], dtype=float)
    pair_labels = np.asarray(data["pair_labels"], dtype=object)

    with open(paths["preprocessed"] / "grouping_data_oip.pkl", "rb") as f:
        mask_groups, label_variables = pd.read_pickle(f)

    try:
        F = _factor_base_indices(mask_groups, label_variables, 3)
    except Exception:
        F = _factor_base_indices(mask_groups, label_variables, 0)

    try:
        F_geno = _factor_base_indices(mask_groups, label_variables, 2)
    except Exception:
        F_geno = {}

    idx2_parts: list[np.ndarray] = []
    idx4_parts: list[np.ndarray] = []
    for ages in F.values():
        if ages.get("2m") is not None:
            idx2_parts.append(np.asarray(ages["2m"], dtype=int))
        if ages.get("4m") is not None:
            idx4_parts.append(np.asarray(ages["4m"], dtype=int))
    idx2 = _safe_concat(idx2_parts)
    idx4 = _safe_concat(idx4_parts)

    if idx2.size == 0 or idx4.size == 0:
        logger.error("Could not resolve age group indices (2m/4m). Abort.")
        return 2

    patterns = _compile_patterns(args.roi_substrings)
    keep_links = [i for i, pair in enumerate(map(tuple, pair_labels)) if _link_matches(pair, patterns)]
    if not keep_links:
        logger.warning("No links matched patterns: %s", args.roi_substrings)
        return 0

    extra = f"_{args.tag.strip()}" if args.tag and args.tag.strip() else ""
    tag = f"w{args.window_size}_lag{args.lag}_tau{args.tau}_{scope}{extra}"

    age_pvals = None
    group_pvals = None
    stats_dir = (paths["allegiance"] / "out").expanduser()
    if args.annotate_stats in {"age", "both"}:
        csv_age = stats_dir / f"pvals_age_wilcoxon_{tag}.csv"
        try:
            age_pvals = pd.read_csv(csv_age, index_col=0, header=[0, 1])
        except Exception as exc:
            logger.warning("Could not load age stats CSV: %s (%s)", csv_age, exc)
    if args.annotate_stats in {"group", "both"}:
        csv_group = stats_dir / f"pvals_group_mwu_{tag}.csv"
        try:
            group_pvals = pd.read_csv(csv_group, index_col=0, header=[0, 1])
        except Exception as exc:
            logger.warning("Could not load group stats CSV: %s (%s)", csv_group, exc)

    def _get_group_p(df: pd.DataFrame | None, block: str, left: str, right: str, label: str) -> float | None:
        if df is None or label not in df.index:
            return None
        key = (block, f"{left} vs {right}")
        alt = (block, f"{right} vs {left}")
        try:
            if key in df.columns:
                return float(df.loc[label, key])
            if alt in df.columns:
                return float(df.loc[label, alt])
        except Exception:
            return None
        return None

    for link_idx in keep_links:
        node_a, node_b = map(str, pair_labels[link_idx])
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["2m", "4m"])
        ax.set_ylabel("Cohesion (time ratio)")

        inset_lines: list[str] = []
        bars_cross_age: list[tuple[float, float, str]] = []
        bars_between_groups: list[tuple[float, float, str]] = []

        if args.color_by == "age":
            values_2m = time_ratio[idx2, link_idx]
            values_4m = time_ratio[idx4, link_idx]
            mu2 = float(np.mean(values_2m)) if values_2m.size else np.nan
            mu4 = float(np.mean(values_4m)) if values_4m.size else np.nan
            eb2 = _errbar(values_2m, args.errorbar)
            eb4 = _errbar(values_4m, args.errorbar)
            ax.errorbar([0, 1], [mu2, mu4], yerr=[eb2, eb4], fmt="-o", color="tab:blue", capsize=3, lw=1.5)
            if not args.aggregate_only:
                jitter = 0.06
                ax.scatter(
                    np.zeros_like(values_2m) + (np.random.rand(len(values_2m)) - 0.5) * jitter,
                    values_2m,
                    s=12,
                    alpha=0.6,
                    color="tab:blue",
                    label="2m",
                )
                ax.scatter(
                    np.ones_like(values_4m) + (np.random.rand(len(values_4m)) - 0.5) * jitter,
                    values_4m,
                    s=12,
                    alpha=0.6,
                    color="tab:orange",
                    label="4m",
                )
            if not args.no_stats:
                if args.paired_age:
                    v2_all, v4_all = _paired_from_F(F, link_idx, time_ratio)
                    p_overall = _wilcoxon_safe(v2_all, v4_all)
                else:
                    p_overall = _mannwhitney_safe(values_2m, values_4m)
                if p_overall is not None:
                    bars_cross_age.append((0.0, 1.0, f"p={p_overall:.3g}"))
            ax.legend(frameon=False, fontsize=8)
        else:
            if args.color_by == "sex":
                groups = _build_sex_groups(F, link_idx, time_ratio, args.paired_age)
            elif args.color_by in {"genotype", "both"}:
                groups = _build_genotype_groups(F_geno, link_idx, time_ratio, args.paired_age)
            elif args.color_by == "sex_genotype":
                groups = _build_sex_genotype_groups(F, F_geno, link_idx, time_ratio, args.paired_age)
            else:
                groups = []

            if not groups:
                logger.warning("No data available for link %s–%s with color mode %s", node_a, node_b, args.color_by)
                plt.close(fig)
                continue

            for group in groups:
                _plot_group_values(ax, group, args.errorbar, args.aggregate_only, args.paired_age)
                if not args.no_stats and group.p_age is not None:
                    bars_cross_age.append((0.0 + group.offset, 1.0 + group.offset, f"{group.name}: p={group.p_age:.3g}"))

            ax.legend(frameon=False, fontsize=8)

            if not args.no_stats and len(groups) >= 2:
                for age_label, xpos in [("2m", 0.0), ("4m", 1.0)]:
                    for i, j, p_val in _group_pairwise_stats(groups, age_label):
                        if p_val is None:
                            continue
                        label = f"{age_label}: p={p_val:.3g}"
                        bars_between_groups.append(
                            (xpos + groups[i].offset, xpos + groups[j].offset, label)
                        )

        ax.set_title(f"{node_a}\u2013{node_b}")

        link_label = f"{node_a}\u2013{node_b}"
        if group_pvals is not None and args.annotate_stats in {"group", "both"}:
            sex_lines: list[str] = []
            geno_lines: list[str] = []
            p_val = _get_group_p(group_pvals, "Sex", "Female-2m", "Male-2m", link_label)
            if p_val is not None:
                sex_lines.append(f"Sex 2m: p={p_val:.3g}{' *' if p_val <= args.alpha else ''}")
            p_val = _get_group_p(group_pvals, "Sex", "Female-4m", "Male-4m", link_label)
            if p_val is not None:
                sex_lines.append(f"Sex 4m: p={p_val:.3g}{' *' if p_val <= args.alpha else ''}")
            p_val = _get_group_p(group_pvals, "Sex", "Female (all-ages)", "Male (all-ages)", link_label)
            if p_val is not None:
                sex_lines.append(f"Sex pooled: p={p_val:.3g}{' *' if p_val <= args.alpha else ''}")
            p_val = _get_group_p(group_pvals, "Sex", "Female-2m", "Male-4m", link_label)
            if p_val is not None:
                sex_lines.append(f"Sex cross F2m vs M4m: p={p_val:.3g}{' *' if p_val <= args.alpha else ''}")
            p_val = _get_group_p(group_pvals, "Sex", "Male-2m", "Female-4m", link_label)
            if p_val is not None:
                sex_lines.append(f"Sex cross M2m vs F4m: p={p_val:.3g}{' *' if p_val <= args.alpha else ''}")
            inset_lines.extend(sex_lines)

            p_val = _get_group_p(group_pvals, "Genotype", "wt-2m", "dKI-2m", link_label)
            if p_val is not None:
                geno_lines.append(f"Genotype 2m: p={p_val:.3g}{' *' if p_val <= args.alpha else ''}")
            p_val = _get_group_p(group_pvals, "Genotype", "wt-4m", "dKI-4m", link_label)
            if p_val is not None:
                geno_lines.append(f"Genotype 4m: p={p_val:.3g}{' *' if p_val <= args.alpha else ''}")
            p_val = _get_group_p(group_pvals, "Genotype", "wt (all-ages)", "dKI (all-ages)", link_label)
            if p_val is not None:
                geno_lines.append(f"Genotype pooled: p={p_val:.3g}{' *' if p_val <= args.alpha else ''}")
            p_val = _get_group_p(group_pvals, "Genotype", "wt-2m", "dKI-4m", link_label)
            if p_val is not None:
                geno_lines.append(f"Genotype cross wt2m vs dKI4m: p={p_val:.3g}{' *' if p_val <= args.alpha else ''}")
            p_val = _get_group_p(group_pvals, "Genotype", "dKI-2m", "wt-4m", link_label)
            if p_val is not None:
                geno_lines.append(f"Genotype cross dKI2m vs wt4m: p={p_val:.3g}{' *' if p_val <= args.alpha else ''}")
            inset_lines.extend(geno_lines)

        if age_pvals is not None and args.annotate_stats in {"age", "both"}:
            try:
                row = age_pvals.loc[link_label]
            except KeyError:
                row = None
            if row is not None:
                for (block, column), p_val in row.items():
                    try:
                        p_val = float(p_val)
                    except (TypeError, ValueError):
                        continue
                    mark = " *" if p_val <= args.alpha else ""
                    inset_lines.append(f"{block} {column}: p={p_val:.3g}{mark}")

        _annotate_inset(ax, inset_lines)
        _draw_sig_bars(ax, bars_cross_age)
        _draw_sig_bars(ax, bars_between_groups, start=0.70)

        fig.tight_layout()

        safe_a = re.sub(r"[^A-Za-z0-9]+", "", node_a)
        safe_b = re.sub(r"[^A-Za-z0-9]+", "", node_b)
        fname = f"curve_{safe_a}-{safe_b}_{tag}.png"
        fpath = out_dir / fname
        if args.save_plots:
            fig.savefig(fpath, dpi=300, bbox_inches="tight")
            logger.info("Saved: %s", fpath)
        if args.no_show:
            plt.close(fig)
        else:
            plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
