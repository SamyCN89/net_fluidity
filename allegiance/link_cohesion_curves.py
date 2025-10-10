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
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory
from scipy.stats import mannwhitneyu

from shared_code.fun_paths import get_paths

logger = logging.getLogger(__name__)


@dataclass
class GroupData:
    """Container for plot data of a single subgroup."""

    name: str
    color: str
    offset: float
    values_2m: np.ndarray
    values_4m: np.ndarray
    paired: bool = False

    def values_for_age(self, age_index: int) -> np.ndarray:
        return self.values_2m if age_index == 0 else self.values_4m


AGE_LABELS = {0: "2m", 1: "4m"}


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
    """Significance bar in blended coords: x=data, y=axes."""

    trans = blended_transform_factory(ax.transData, ax.transAxes)

    match = re.search(r"p\s*=\s*([0-9]*\.?[0-9]+(?:e-?\d+)?)", text, re.I)
    pvalue = float(match.group(1)) if match else None
    label_color = "red" if (pvalue is not None and pvalue < 0.05) else "black"

    y0, y1 = y_axes, y_axes + h_axes
    ax.plot(
        [x1, x1, x2, x2],
        [y0, y1, y1, y0],
        transform=trans,
        color="black",
        lw=lw,
        clip_on=False,
    )

    va = "bottom" if h_axes >= 0 else "top"
    ax.text(
        (x1 + x2) * 0.5,
        y1,
        text,
        transform=trans,
        ha="center",
        va=va,
        fontsize=fontsize,
        color=label_color,
        clip_on=False,
    )


def setup_logging() -> None:
    if logger.handlers:
        return
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cohesion curves for selected links",
    )
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
        help="Errorbar type for aggregates: sd, sem, or variance",
    )
    parser.add_argument("--no-stats", action="store_true", help="Do not compute or display any p-values")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Plot only mean ± SEM per group (no individual points or paired lines)",
    )
    parser.add_argument(
        "--annotate-stats",
        choices=["none", "age", "group", "both"],
        default="none",
        help="Annotate significance from stats CSVs",
    )
    parser.add_argument("--save-plots", action="store_true", help="Save figures to disk")
    parser.add_argument("--no-show", action="store_true", help="Headless mode (no display)")
    return parser.parse_known_args()[0]


def load_npz(paths: dict, ws: int, lag: int, tau: int, scope: str, tag: str = "") -> dict:
    suffix = f"_{tag.strip()}" if tag and tag.strip() else ""
    file_path = (
        paths["allegiance"]
        / "cohesion_data"
        / f"cohesion_data_w{ws}_lag{lag}_tau{tau}_{scope}{suffix}.npz"
    )
    return dict(np.load(file_path, allow_pickle=True))


def _compile_patterns(spec: str) -> list[re.Pattern[str]]:
    tokens = [token.strip() for token in spec.split(",") if token.strip()]
    return [re.compile(re.escape(token), flags=re.IGNORECASE) for token in tokens]


def _link_matches(pair: tuple[str, str], patterns: Iterable[re.Pattern[str]]) -> bool:
    a, b = pair
    return any(pattern.search(str(a)) or pattern.search(str(b)) for pattern in patterns)


def _errbar(values: np.ndarray, mode: str) -> float:
    count = int(values.size)
    if count == 0:
        return np.nan
    if count == 1:
        return 0.0
    if mode == "sd":
        return float(np.std(values, ddof=1))
    if mode == "sem":
        return float(np.std(values, ddof=1)) / float(np.sqrt(count))
    return float(np.var(values, ddof=1))


def _annotate_inset(ax: plt.Axes, lines: Iterable[str]) -> None:
    text = "\n".join(line for line in lines if line)
    if not text:
        return
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


def _mannwhitney_safe(values_a: np.ndarray, values_b: np.ndarray) -> float:
    if values_a.size == 0 or values_b.size == 0:
        return float("nan")
    try:
        return float(mannwhitneyu(values_a, values_b, alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def _wilcoxon_safe(values_a: np.ndarray, values_b: np.ndarray) -> float:
    if values_a.size == 0 or values_b.size == 0 or values_a.size != values_b.size:
        return float("nan")
    try:
        from scipy.stats import wilcoxon

        return float(
            wilcoxon(values_a, values_b, zero_method="wilcox", alternative="two-sided").pvalue
        )
    except Exception:
        return float("nan")


def _draw_sig_bars(ax: plt.Axes, entries: list[tuple[float, float, str, int | None]]) -> None:
    if not entries:
        return

    gap = 0.07
    height = 0.02

    grouped: dict[int | None, list[tuple[float, float, str]]] = {}
    for x1, x2, label, slot in entries:
        grouped.setdefault(slot, []).append((x1, x2, label))

    counts = {slot: len(labels) for slot, labels in grouped.items()}

    start_general = 0.95
    start_age0 = start_general - counts.get(None, 0) * gap - (0.05 if counts.get(None, 0) else 0.0)
    start_age1 = start_age0 - counts.get(0, 0) * gap - (0.05 if counts.get(0, 0) else 0.0)
    start_positions = {None: start_general, 0: start_age0, 1: start_age1}

    for slot in (None, 0, 1):
        slot_entries = grouped.get(slot, [])
        y_current = start_positions.get(slot, start_age1)
        for x1, x2, label in slot_entries:
            _add_sig_bar_axes(ax, x1, x2, y_current, text=label, h_axes=height)
            y_current -= gap


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

    def factor_base_indices(factor_idx: int) -> dict[str, dict[str, np.ndarray | None]]:
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

    try:
        bases_sex = factor_base_indices(3)
    except Exception:
        bases_sex = factor_base_indices(0)

    try:
        bases_genotype = factor_base_indices(2)
    except Exception:
        bases_genotype = {}

    idx2_parts, idx4_parts = [], []
    for entry in bases_sex.values():
        if entry.get("2m") is not None:
            idx2_parts.append(entry["2m"])
        if entry.get("4m") is not None:
            idx4_parts.append(entry["4m"])
    idx2 = np.unique(np.concatenate(idx2_parts)) if idx2_parts else np.array([], dtype=int)
    idx4 = np.unique(np.concatenate(idx4_parts)) if idx4_parts else np.array([], dtype=int)

    if idx2.size == 0 or idx4.size == 0:
        logger.error("Could not resolve age group indices (2m/4m). Abort.")
        return 2

    patterns = _compile_patterns(args.roi_substrings)
    keep_links = [i for i, labels in enumerate(pair_labels) if _link_matches(tuple(map(str, labels)), patterns)]
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

    def get_group_p(df: pd.DataFrame | None, block: str, left: str, right: str) -> float | None:
        if df is None:
            return None
        key_lr = (block, f"{left} vs {right}")
        key_rl = (block, f"{right} vs {left}")
        try:
            if key_lr in df.columns:
                return float(df.loc[link_label, key_lr])
            if key_rl in df.columns:
                return float(df.loc[link_label, key_rl])
        except Exception:
            return None
        return None

    def paired_values_from_map(
        mapping: dict[str, dict[str, np.ndarray | None]], link_idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        values_2m, values_4m = [], []
        for ages in mapping.values():
            idx_2m, idx_4m = ages.get("2m"), ages.get("4m")
            if idx_2m is None or idx_4m is None:
                continue
            if len(idx_2m) == 0 or len(idx_4m) == 0 or len(idx_2m) != len(idx_4m):
                continue
            values_2m.append(time_ratio[idx_2m, link_idx])
            values_4m.append(time_ratio[idx_4m, link_idx])
        if values_2m and values_4m:
            return np.concatenate(values_2m), np.concatenate(values_4m)
        return np.array([]), np.array([])

    for link_idx in keep_links:
        a_label, b_label = map(str, pair_labels[link_idx])
        fig, ax = plt.subplots(figsize=(5.2, 3.6))

        bars: list[tuple[float, float, str, int | None]] = []
        inset_lines: list[str] = []

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["2m", "4m"])
        ax.set_ylabel("Cohesion (time ratio)")

        if args.paired_age:
            values_2m_all, values_4m_all = paired_values_from_map(bases_sex, link_idx)
            overall_p = None if args.no_stats else _wilcoxon_safe(values_2m_all, values_4m_all)
        else:
            values_2m_all = time_ratio[idx2, link_idx]
            values_4m_all = time_ratio[idx4, link_idx]
            overall_p = None if args.no_stats else _mannwhitney_safe(values_2m_all, values_4m_all)

        if args.color_by == "age":
            mean_2m, mean_4m = np.mean(values_2m_all), np.mean(values_4m_all)
            err_2m = _errbar(values_2m_all, args.errorbar)
            err_4m = _errbar(values_4m_all, args.errorbar)
            ax.errorbar([0, 1], [mean_2m, mean_4m], yerr=[err_2m, err_4m], fmt="-o", color="tab:blue", capsize=3, lw=1.5)
            if not args.aggregate_only:
                jitter = 0.06
                ax.scatter(
                    np.zeros_like(values_2m_all) + (np.random.rand(len(values_2m_all)) - 0.5) * jitter,
                    values_2m_all,
                    s=12,
                    alpha=0.6,
                    color="tab:blue",
                    label="2m",
                )
                ax.scatter(
                    np.ones_like(values_4m_all) + (np.random.rand(len(values_4m_all)) - 0.5) * jitter,
                    values_4m_all,
                    s=12,
                    alpha=0.6,
                    color="tab:orange",
                    label="4m",
                )
                if args.paired_age:
                    for idx_point in range(min(len(values_2m_all), len(values_4m_all))):
                        ax.plot([0, 1], [values_2m_all[idx_point], values_4m_all[idx_point]], color="0.7", lw=0.6, alpha=0.2)
                ax.legend(frameon=False, fontsize=8)
            if overall_p is not None and not np.isnan(overall_p):
                bars.append((0.0, 1.0, f"p={overall_p:.3g}", None))
        else:
            group_series: list[GroupData] = []

            if args.color_by == "sex":
                mapping = {"Female": "tab:purple", "Male": "tab:green"}
                offsets = {"Female": -0.08, "Male": 0.08}
                for sex_name in ("Female", "Male"):
                    groups_indices = {"2m": [], "4m": []}
                    for base_name, ages in bases_sex.items():
                        if sex_name.lower() not in base_name.lower():
                            continue
                        if ages.get("2m") is not None:
                            groups_indices["2m"].append(ages["2m"])
                        if ages.get("4m") is not None:
                            groups_indices["4m"].append(ages["4m"])
                    if args.paired_age:
                        restricted = {
                            base_name: ages
                            for base_name, ages in bases_sex.items()
                            if sex_name.lower() in base_name.lower()
                        }
                        v2, v4 = paired_values_from_map(restricted, link_idx)
                        paired_flag = True
                    else:
                        idx2_sex = (
                            np.unique(np.concatenate(groups_indices["2m"]))
                            if groups_indices["2m"]
                            else np.array([], dtype=int)
                        )
                        idx4_sex = (
                            np.unique(np.concatenate(groups_indices["4m"]))
                            if groups_indices["4m"]
                            else np.array([], dtype=int)
                        )
                        v2 = time_ratio[idx2_sex, link_idx]
                        v4 = time_ratio[idx4_sex, link_idx]
                        paired_flag = False
                    group_series.append(
                        GroupData(
                            name=sex_name,
                            color=mapping[sex_name],
                            offset=offsets[sex_name],
                            values_2m=v2,
                            values_4m=v4,
                            paired=paired_flag,
                        )
                    )
            elif args.color_by == "sex_genotype":
                combos = [
                    ("Female", "wt", "tab:blue", -0.12),
                    ("Female", "dKI", "tab:red", -0.04),
                    ("Male", "wt", "tab:green", 0.04),
                    ("Male", "dKI", "tab:orange", 0.12),
                ]

                def collect_indices(sex_name: str, geno_key: str) -> tuple[np.ndarray, np.ndarray]:
                    sex2, sex4 = [], []
                    for base_name, ages in bases_sex.items():
                        if sex_name.lower() in base_name.lower():
                            if ages.get("2m") is not None:
                                sex2.append(ages["2m"])
                            if ages.get("4m") is not None:
                                sex4.append(ages["4m"])
                    geno2, geno4 = [], []
                    for base_name, ages in bases_genotype.items():
                        label = base_name.lower()
                        if (geno_key == "wt" and "wt" in label) or (geno_key == "dKI" and "dki" in label):
                            if ages.get("2m") is not None:
                                geno2.append(ages["2m"])
                            if ages.get("4m") is not None:
                                geno4.append(ages["4m"])
                    idx2_sex = np.unique(np.concatenate(sex2)) if sex2 else np.array([], dtype=int)
                    idx4_sex = np.unique(np.concatenate(sex4)) if sex4 else np.array([], dtype=int)
                    idx2_geno = np.unique(np.concatenate(geno2)) if geno2 else np.array([], dtype=int)
                    idx4_geno = np.unique(np.concatenate(geno4)) if geno4 else np.array([], dtype=int)
                    return (
                        np.intersect1d(idx2_sex, idx2_geno, assume_unique=False),
                        np.intersect1d(idx4_sex, idx4_geno, assume_unique=False),
                    )

                for sex_name, geno_key, color, offset in combos:
                    idx2_group, idx4_group = collect_indices(sex_name, geno_key)
                    if args.paired_age:
                        length = min(len(idx2_group), len(idx4_group))
                        values_2m_group = time_ratio[idx2_group[:length], link_idx]
                        values_4m_group = time_ratio[idx4_group[:length], link_idx]
                        paired_flag = length > 0
                    else:
                        values_2m_group = time_ratio[idx2_group, link_idx]
                        values_4m_group = time_ratio[idx4_group, link_idx]
                        paired_flag = False
                    group_series.append(
                        GroupData(
                            name=f"{sex_name} {geno_key}",
                            color=color,
                            offset=offset,
                            values_2m=values_2m_group,
                            values_4m=values_4m_group,
                            paired=paired_flag,
                        )
                    )
            else:  # genotype or both
                mapping = {"wt": ("tab:blue", -0.08), "dKI": ("tab:red", 0.08)}
                for geno_key, (color, offset) in mapping.items():
                    indices = {"2m": [], "4m": []}
                    for base_name, ages in bases_genotype.items():
                        label = base_name.lower()
                        if (geno_key == "wt" and "wt" in label) or (geno_key == "dKI" and "dki" in label):
                            if ages.get("2m") is not None:
                                indices["2m"].append(ages["2m"])
                            if ages.get("4m") is not None:
                                indices["4m"].append(ages["4m"])
                    if args.paired_age:
                        presence_2m = np.zeros(time_ratio.shape[0], dtype=bool)
                        presence_4m = np.zeros(time_ratio.shape[0], dtype=bool)
                        if indices["2m"]:
                            presence_2m[np.unique(np.concatenate(indices["2m"]))] = True
                        if indices["4m"]:
                            presence_4m[np.unique(np.concatenate(indices["4m"]))] = True
                        pair_idx = np.flatnonzero(presence_2m & presence_4m)
                        values_2m_group = time_ratio[pair_idx, link_idx]
                        values_4m_group = time_ratio[pair_idx, link_idx]
                        paired_flag = pair_idx.size > 0
                    else:
                        idx2_group = (
                            np.unique(np.concatenate(indices["2m"]))
                            if indices["2m"]
                            else np.array([], dtype=int)
                        )
                        idx4_group = (
                            np.unique(np.concatenate(indices["4m"]))
                            if indices["4m"]
                            else np.array([], dtype=int)
                        )
                        values_2m_group = time_ratio[idx2_group, link_idx]
                        values_4m_group = time_ratio[idx4_group, link_idx]
                        paired_flag = False
                    group_series.append(
                        GroupData(
                            name=geno_key,
                            color=color,
                            offset=offset,
                            values_2m=values_2m_group,
                            values_4m=values_4m_group,
                            paired=paired_flag,
                        )
                    )

            for group in group_series:
                mean_2m = np.mean(group.values_2m) if group.values_2m.size else np.nan
                mean_4m = np.mean(group.values_4m) if group.values_4m.size else np.nan
                err_2m = _errbar(group.values_2m, args.errorbar) if group.values_2m.size else np.nan
                err_4m = _errbar(group.values_4m, args.errorbar) if group.values_4m.size else np.nan
                ax.errorbar(
                    [0 + group.offset, 1 + group.offset],
                    [mean_2m, mean_4m],
                    yerr=[err_2m, err_4m],
                    fmt="-o",
                    color=group.color,
                    capsize=3,
                    lw=1.5,
                    label=group.name,
                )
                if not args.aggregate_only:
                    jitter = 0.05
                    if group.values_2m.size:
                        ax.scatter(
                            np.zeros_like(group.values_2m)
                            + group.offset
                            + (np.random.rand(len(group.values_2m)) - 0.5) * jitter,
                            group.values_2m,
                            s=12,
                            alpha=0.6,
                            color=group.color,
                        )
                    if group.values_4m.size:
                        ax.scatter(
                            np.ones_like(group.values_4m)
                            + group.offset
                            + (np.random.rand(len(group.values_4m)) - 0.5) * jitter,
                            group.values_4m,
                            s=12,
                            alpha=0.6,
                            color=group.color,
                        )
                    if args.paired_age and group.paired:
                        pair_count = min(len(group.values_2m), len(group.values_4m))
                        for idx_point in range(pair_count):
                            ax.plot(
                                [0 + group.offset, 1 + group.offset],
                                [group.values_2m[idx_point], group.values_4m[idx_point]],
                                color=group.color,
                                lw=0.6,
                                alpha=0.2,
                            )
                if not args.no_stats:
                    if group.paired and args.paired_age:
                        p_value = _wilcoxon_safe(group.values_2m, group.values_4m)
                    else:
                        p_value = _mannwhitney_safe(group.values_2m, group.values_4m)
                    if not np.isnan(p_value):
                        bars.append((0.0 + group.offset, 1.0 + group.offset, f"p={p_value:.3g}", None))
                        inset_lines.append(f"{group.name}: p={p_value:.3g}")

            if group_series:
                ax.legend(frameon=False, fontsize=8)

            if not args.no_stats:
                for age_index in (0, 1):
                    age_label = AGE_LABELS[age_index]
                    for group_a, group_b in combinations(group_series, 2):
                        values_a = group_a.values_for_age(age_index)
                        values_b = group_b.values_for_age(age_index)
                        p_value = _mannwhitney_safe(values_a, values_b)
                        if np.isnan(p_value):
                            continue
                        label = f"{group_a.name} vs {group_b.name} ({age_label}): p={p_value:.3g}"
                        bars.append((age_index + group_a.offset, age_index + group_b.offset, label, age_index))

        ax.set_title(f"{a_label}\u2013{b_label}")

        link_label = f"{a_label}\u2013{b_label}"

        if group_pvals is not None and args.annotate_stats in {"group", "both"}:
            sex_lines: list[str] = []
            geno_lines: list[str] = []
            p_val = get_group_p(group_pvals, "Sex", "Female-2m", "Male-2m")
            if p_val is not None:
                sex_lines.append(f"Sex 2m: p={p_val:.3g}{' *' if p_val <= args.alpha else ''}")
            p_val = get_group_p(group_pvals, "Sex", "Female-4m", "Male-4m")
            if p_val is not None:
                sex_lines.append(f"Sex 4m: p={p_val:.3g}{' *' if p_val <= args.alpha else ''}")
            p_val = get_group_p(group_pvals, "Sex", "Female (all-ages)", "Male (all-ages)")
            if p_val is not None:
                sex_lines.append(f"Sex pooled: p={p_val:.3g}{' *' if p_val <= args.alpha else ''}")
            p_val = get_group_p(group_pvals, "Sex", "Female-2m", "Male-4m")
            if p_val is not None:
                sex_lines.append(f"Sex cross F2m vs M4m: p={p_val:.3g}{' *' if p_val <= args.alpha else ''}")
            p_val = get_group_p(group_pvals, "Sex", "Male-2m", "Female-4m")
            if p_val is not None:
                sex_lines.append(f"Sex cross M2m vs F4m: p={p_val:.3g}{' *' if p_val <= args.alpha else ''}")
            p_val = get_group_p(group_pvals, "Genotype", "wt-2m", "dKI-2m")
            if p_val is not None:
                geno_lines.append(f"Genotype 2m: p={p_val:.3g}{' *' if p_val <= args.alpha else ''}")
            p_val = get_group_p(group_pvals, "Genotype", "wt-4m", "dKI-4m")
            if p_val is not None:
                geno_lines.append(f"Genotype 4m: p={p_val:.3g}{' *' if p_val <= args.alpha else ''}")
            p_val = get_group_p(group_pvals, "Genotype", "wt (all-ages)", "dKI (all-ages)")
            if p_val is not None:
                geno_lines.append(f"Genotype pooled: p={p_val:.3g}{' *' if p_val <= args.alpha else ''}")
            p_val = get_group_p(group_pvals, "Genotype", "wt-2m", "dKI-4m")
            if p_val is not None:
                geno_lines.append(f"Genotype cross wt2m vs dKI4m: p={p_val:.3g}{' *' if p_val <= args.alpha else ''}")
            p_val = get_group_p(group_pvals, "Genotype", "dKI-2m", "wt-4m")
            if p_val is not None:
                geno_lines.append(f"Genotype cross dKI2m vs wt4m: p={p_val:.3g}{' *' if p_val <= args.alpha else ''}")
            inset_lines.extend(sex_lines)
            inset_lines.extend(geno_lines)

        _annotate_inset(ax, inset_lines)
        _draw_sig_bars(ax, bars)

        fig.tight_layout()

        safe_a = re.sub(r"[^A-Za-z0-9]+", "", a_label)
        safe_b = re.sub(r"[^A-Za-z0-9]+", "", b_label)
        filename = f"curve_{safe_a}-{safe_b}_{tag}.png"
        output_path = out_dir / filename
        if args.save_plots:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info("Saved: %s", output_path)
        if not args.no_show:
            plt.show()
        else:
            plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
