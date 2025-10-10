#!/usr/bin/env python3
"""Plot cohesion curves for selected links and annotate p-values."""

from __future__ import annotations

import argparse
import logging
import pathlib
import re
from dataclasses import dataclass
from typing import Callable, Iterable, NamedTuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory
from scipy.stats import mannwhitneyu

from shared_code.fun_paths import get_paths

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _setup_logging() -> None:
    if logger.handlers:
        return
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


class AgeArrays(NamedTuple):
    two_month: np.ndarray
    four_month: np.ndarray


@dataclass
class GroupSpec:
    """Container describing the samples that belong to a plotting group."""

    key: str
    color: str
    offset: float
    idx_2m: np.ndarray
    idx_4m: np.ndarray
    paired_getter: Callable[[int], AgeArrays] | None = None


def _compute_error(arr: np.ndarray, mode: str) -> float:
    if arr.size == 0:
        return np.nan
    if arr.size == 1:
        return 0.0
    if mode == "sd":
        return float(np.std(arr, ddof=1))
    if mode == "sem":
        return float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    return float(np.var(arr, ddof=1))


def _scatter_individuals(
    ax: matplotlib.axes.Axes,
    x: float,
    values: np.ndarray,
    color: str,
    jitter: float,
    aggregate_only: bool,
) -> None:
    if aggregate_only or values.size == 0:
        return
    jittered = x + (np.random.rand(values.size) - 0.5) * jitter
    ax.scatter(jittered, values, s=12, alpha=0.6, color=color)


def _draw_pair_lines(
    ax: matplotlib.axes.Axes, x0: float, x1: float, values_2m: np.ndarray, values_4m: np.ndarray, color: str
) -> None:
    n = min(values_2m.size, values_4m.size)
    for i in range(n):
        ax.plot([x0, x1], [values_2m[i], values_4m[i]], color=color, lw=0.6, alpha=0.2)


def _compute_pvalue(arr1: np.ndarray, arr2: np.ndarray, paired: bool) -> float | None:
    if arr1.size == 0 or arr2.size == 0:
        return None
    try:
        if paired:
            from scipy.stats import wilcoxon

            if arr1.size != arr2.size:
                n = min(arr1.size, arr2.size)
                arr1 = arr1[:n]
                arr2 = arr2[:n]
            res = wilcoxon(arr1, arr2, zero_method="wilcox", alternative="two-sided")
            return float(res.pvalue)
        res = mannwhitneyu(arr1, arr2, alternative="two-sided")
        return float(res.pvalue)
    except ValueError:
        return None


def _add_sig_bar_axes(
    ax: matplotlib.axes.Axes,
    x1: float,
    x2: float,
    y_axes: float,
    text: str,
    h_axes: float = -0.02,
    lw: float = 1.0,
    fontsize: int = 8,
) -> None:
    """Draw a significance bar expressed in axes coordinates."""

    transform = blended_transform_factory(ax.transData, ax.transAxes)
    match = re.search(r"p\s*=\s*([0-9]*\.?[0-9]+(?:e-?\d+)?)", text, re.I)
    p_value = float(match.group(1)) if match else None
    label_color = "red" if (p_value is not None and p_value < 0.05) else "black"

    y0, y1 = y_axes, y_axes + h_axes
    ax.plot([x1, x1, x2, x2], [y0, y1, y1, y0], transform=transform, color="black", lw=lw, clip_on=False)

    vertical_alignment = "bottom" if h_axes >= 0 else "top"
    ax.text(
        0.5 * (x1 + x2),
        y1,
        text,
        transform=transform,
        ha="center",
        va=vertical_alignment,
        fontsize=fontsize,
        color=label_color,
        clip_on=False,
    )


def _annotate_inset(ax: matplotlib.axes.Axes, lines: Iterable[str]) -> None:
    content = "\n".join(line for line in lines if line)
    if not content:
        return
    ax.text(
        0.98,
        0.02,
        content,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"),
    )


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
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
        help="Errorbar type for aggregates: sd, sem, or variance",
    )
    parser.add_argument("--no-stats", action="store_true", help="Do not compute or display p-values")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Plot only mean ± error per group (no individual points or paired lines)",
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


def _load_npz(paths: dict, ws: int, lag: int, tau: int, scope: str, tag: str = "") -> dict:
    suffix = f"_{tag.strip()}" if tag and tag.strip() else ""
    fpath = paths["allegiance"] / "cohesion_data" / f"cohesion_data_w{ws}_lag{lag}_tau{tau}_{scope}{suffix}.npz"
    return dict(np.load(fpath, allow_pickle=True))


def _compile_patterns(spec: str) -> list[re.Pattern[str]]:
    return [re.compile(re.escape(tok.strip()), flags=re.IGNORECASE) for tok in spec.split(",") if tok.strip()]


def _link_matches(pair: tuple[str, str], patterns: Iterable[re.Pattern[str]]) -> bool:
    a, b = pair
    return any(pat.search(str(a)) or pat.search(str(b)) for pat in patterns)


def _factor_base_indices(mask_groups: list, label_variables: list, factor_idx: int) -> dict[str, dict[str, np.ndarray | None]]:
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


def _paired_from_F(
    Fmap: dict[str, dict[str, np.ndarray | None]], time_ratio: np.ndarray, link_idx: int
) -> AgeArrays:
    two_month, four_month = [], []
    for ages in Fmap.values():
        idx2, idx4 = ages.get("2m"), ages.get("4m")
        if idx2 is None or idx4 is None or idx2.size == 0 or idx4.size == 0:
            continue
        if idx2.size != idx4.size:
            continue
        two_month.append(time_ratio[idx2, link_idx])
        four_month.append(time_ratio[idx4, link_idx])
    if two_month and four_month:
        return AgeArrays(np.concatenate(two_month), np.concatenate(four_month))
    return AgeArrays(np.array([]), np.array([]))


# ---------------------------------------------------------------------------
# Group builders
# ---------------------------------------------------------------------------


def _collect_sex_groups(F: dict[str, dict[str, np.ndarray | None]]) -> dict[str, dict[str, np.ndarray]]:
    groups: dict[str, dict[str, np.ndarray]] = {
        "Female": {"2m": np.array([], dtype=int), "4m": np.array([], dtype=int)},
        "Male": {"2m": np.array([], dtype=int), "4m": np.array([], dtype=int)},
    }
    for base, ages in F.items():
        base_lower = base.lower()
        key = "Female" if "female" in base_lower else ("Male" if "male" in base_lower else None)
        if key is None:
            continue
        for age in ("2m", "4m"):
            idx = ages.get(age)
            if idx is None:
                continue
            groups[key][age] = np.unique(np.concatenate([groups[key][age], idx]))
    return groups


def _collect_genotype_groups(
    F_geno: dict[str, dict[str, np.ndarray | None]]
) -> dict[str, dict[str, np.ndarray]]:
    groups: dict[str, dict[str, np.ndarray]] = {}
    for base, ages in F_geno.items():
        low = base.lower()
        if "wt" in low:
            key = "wt"
        elif "dki" in low:
            key = "dKI"
        else:
            continue
        entry = groups.setdefault(key, {"2m": np.array([], dtype=int), "4m": np.array([], dtype=int)})
        for age in ("2m", "4m"):
            idx = ages.get(age)
            if idx is None:
                continue
            entry[age] = np.unique(np.concatenate([entry[age], idx]))
    return groups


def _collect_sex_genotype_indices(
    F: dict[str, dict[str, np.ndarray | None]],
    F_geno: dict[str, dict[str, np.ndarray | None]],
    sex: str,
    genotype: str,
) -> tuple[np.ndarray, np.ndarray]:
    sex_indices = {"2m": [], "4m": []}
    geno_indices = {"2m": [], "4m": []}

    for base, ages in F.items():
        if sex.lower() not in base.lower():
            continue
        for age in ("2m", "4m"):
            idx = ages.get(age)
            if idx is None:
                continue
            sex_indices[age].append(idx)

    for base, ages in F_geno.items():
        low = base.lower()
        if (genotype == "wt" and "wt" not in low) or (genotype == "dKI" and "dki" not in low):
            continue
        for age in ("2m", "4m"):
            idx = ages.get(age)
            if idx is None:
                continue
            geno_indices[age].append(idx)

    def intersect(age: str) -> np.ndarray:
        if not sex_indices[age] or not geno_indices[age]:
            return np.array([], dtype=int)
        sex_union = np.unique(np.concatenate(sex_indices[age]))
        geno_union = np.unique(np.concatenate(geno_indices[age]))
        return np.intersect1d(sex_union, geno_union, assume_unique=False)

    return intersect("2m"), intersect("4m")


# ---------------------------------------------------------------------------
# Plotting logic
# ---------------------------------------------------------------------------


def _build_group_specs(
    args: argparse.Namespace,
    color_mode: str,
    link_idx: int,
    time_ratio: np.ndarray,
    F: dict[str, dict[str, np.ndarray | None]],
    F_geno: dict[str, dict[str, np.ndarray | None]],
    idx2: np.ndarray,
    idx4: np.ndarray,
) -> tuple[list[GroupSpec], dict[str, str]]:
    """Return group specifications and human-readable labels."""

    labels: dict[str, str] = {}

    if color_mode == "age":
        spec = GroupSpec(
            key="Age",
            color="tab:blue",
            offset=0.0,
            idx_2m=idx2,
            idx_4m=idx4,
            paired_getter=(lambda li: AgeArrays(time_ratio[idx2, li], time_ratio[idx4, li]))
            if args.paired_age
            else None,
        )
        labels["Age"] = "Age"
        return [spec], labels

    if color_mode == "sex":
        offsets = {"Female": -0.08, "Male": 0.08}
        colors = {"Female": "tab:purple", "Male": "tab:green"}
        groups = _collect_sex_groups(F)
        specs: list[GroupSpec] = []

        for sex_key, idx_map in groups.items():
            if idx_map["2m"].size == 0 and idx_map["4m"].size == 0:
                continue

            def paired(li: int, sex_key=sex_key) -> AgeArrays:
                restricted = {
                    base: ages
                    for base, ages in F.items()
                    if sex_key.lower() in base.lower()
                }
                return _paired_from_F(restricted, time_ratio, li)

            specs.append(
                GroupSpec(
                    key=sex_key,
                    color=colors.get(sex_key, "tab:blue"),
                    offset=offsets.get(sex_key, 0.0),
                    idx_2m=idx_map["2m"],
                    idx_4m=idx_map["4m"],
                    paired_getter=paired if args.paired_age else None,
                )
            )
            labels[sex_key] = sex_key
        return specs, labels

    if color_mode in {"genotype", "both"}:
        offsets = {"wt": -0.08, "dKI": 0.08}
        colors = {"wt": "tab:blue", "dKI": "tab:red"}
        groups = _collect_genotype_groups(F_geno)
        specs = []
        for geno_key, idx_map in groups.items():
            if idx_map["2m"].size == 0 and idx_map["4m"].size == 0:
                continue

            def paired(li: int, geno_key=geno_key) -> AgeArrays:
                ages_map = {
                    base: ages
                    for base, ages in F_geno.items()
                    if geno_key.lower() in base.lower()
                }
                two_month = []
                four_month = []
                for ages in ages_map.values():
                    idx2, idx4 = ages.get("2m"), ages.get("4m")
                    if idx2 is None or idx4 is None:
                        continue
                    inter = np.intersect1d(idx2, idx4, assume_unique=False)
                    if inter.size == 0:
                        continue
                    two_month.append(time_ratio[inter, li])
                    four_month.append(time_ratio[inter, li])
                if two_month and four_month:
                    return AgeArrays(np.concatenate(two_month), np.concatenate(four_month))
                return AgeArrays(np.array([]), np.array([]))

            specs.append(
                GroupSpec(
                    key=geno_key,
                    color=colors.get(geno_key, "tab:blue"),
                    offset=offsets.get(geno_key, 0.0),
                    idx_2m=idx_map["2m"],
                    idx_4m=idx_map["4m"],
                    paired_getter=paired if args.paired_age else None,
                )
            )
            labels[geno_key] = geno_key
        return specs, labels

    if color_mode == "sex_genotype":
        combos = [
            ("Female", "wt", "tab:blue", -0.12),
            ("Female", "dKI", "tab:red", -0.04),
            ("Male", "wt", "tab:green", 0.04),
            ("Male", "dKI", "tab:orange", 0.12),
        ]
        specs = []
        for sex_name, geno_key, color, offset in combos:
            idx2, idx4 = _collect_sex_genotype_indices(F, F_geno, sex_name, geno_key)
            if idx2.size == 0 and idx4.size == 0:
                continue

            def paired(li: int, idx2=idx2, idx4=idx4) -> AgeArrays:
                if idx2.size == 0 or idx4.size == 0:
                    return AgeArrays(np.array([]), np.array([]))
                length = min(idx2.size, idx4.size)
                return AgeArrays(time_ratio[idx2[:length], li], time_ratio[idx4[:length], li])

            key = f"{sex_name} {geno_key}"
            specs.append(
                GroupSpec(
                    key=key,
                    color=color,
                    offset=offset,
                    idx_2m=idx2,
                    idx_4m=idx4,
                    paired_getter=paired if args.paired_age else None,
                )
            )
            labels[key] = key
        return specs, labels

    raise ValueError(f"Unsupported color mode: {color_mode}")


def _plot_groups(
    ax: matplotlib.axes.Axes,
    args: argparse.Namespace,
    link_idx: int,
    specs: list[GroupSpec],
    time_ratio: np.ndarray,
) -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
    inset_lines: list[str] = []
    values_2m: dict[str, np.ndarray] = {}
    values_4m: dict[str, np.ndarray] = {}

    for spec in specs:
        raw2 = time_ratio[spec.idx_2m, link_idx]
        raw4 = time_ratio[spec.idx_4m, link_idx]
        values_2m[spec.key] = raw2
        values_4m[spec.key] = raw4

        if args.paired_age and spec.paired_getter is not None:
            paired = spec.paired_getter(link_idx)
            v2, v4 = paired.two_month, paired.four_month
        else:
            v2, v4 = raw2, raw4

        mu2, mu4 = np.mean(v2) if v2.size else np.nan, np.mean(v4) if v4.size else np.nan
        eb2, eb4 = _compute_error(v2, args.errorbar), _compute_error(v4, args.errorbar)
        ax.errorbar(
            [0 + spec.offset, 1 + spec.offset],
            [mu2, mu4],
            yerr=[eb2, eb4],
            fmt="-o",
            color=spec.color,
            capsize=3,
            lw=1.5,
            label=spec.key,
        )
        _scatter_individuals(ax, 0 + spec.offset, raw2, spec.color, 0.06, args.aggregate_only)
        _scatter_individuals(ax, 1 + spec.offset, raw4, spec.color, 0.06, args.aggregate_only)
        if args.paired_age and not args.aggregate_only and v2.size and v4.size:
            _draw_pair_lines(ax, 0 + spec.offset, 1 + spec.offset, v2, v4, spec.color)

        if not args.no_stats:
            p_val = _compute_pvalue(v2, v4, paired=args.paired_age)
            if p_val is not None:
                inset_lines.append(f"{spec.key}: p={p_val:.3g}")

    return inset_lines, values_2m, values_4m


def _reserve_axes_bars(
    ax: matplotlib.axes.Axes, bars: list[tuple[float, float, str]], *, start: float = 0.92, gap: float = 0.07
) -> None:
    if not bars:
        return
    y = start
    for x1, x2, label in bars:
        _add_sig_bar_axes(ax, x1, x2, y, text=label, h_axes=-0.015)
        y -= gap


def _format_link_label(label_a: str, label_b: str) -> str:
    return f"{label_a}\u2013{label_b}"


# ---------------------------------------------------------------------------
# CSV annotation helpers
# ---------------------------------------------------------------------------


def _load_stats_csv(path: pathlib.Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, index_col=0, header=[0, 1])
    except Exception as exc:
        logger.warning("Could not load stats CSV %s: %s", path, exc)
        return None


def _lookup_group_p(
    df: pd.DataFrame | None, link_label: str, block: str, left: str, right: str
) -> float | None:
    if df is None:
        return None
    for key in (f"{left} vs {right}", f"{right} vs {left}"):
        col = (block, key)
        try:
            if col in df.columns:
                value = df.loc[link_label, col]
                return float(value)
        except KeyError:
            continue
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    _setup_logging()
    args = _parse_args()

    if args.no_show:
        matplotlib.use("Agg", force=True)

    paths = get_paths(timecourse_folder=args.timecourse_folder)
    out_dir = (paths["f_cohesion"] / "link_curves").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_npz(paths, args.window_size, args.lag, args.tau, args.roi_scope, args.tag)
    time_ratio = np.asarray(data["time_ratio"], dtype=float)
    pair_labels = np.asarray(data["pair_labels"], dtype=object)

    with open(paths["preprocessed"] / "grouping_data_oip.pkl", "rb") as file:
        mask_groups, label_variables = pd.read_pickle(file)

    try:
        F = _factor_base_indices(mask_groups, label_variables, 3)
    except Exception:
        F = _factor_base_indices(mask_groups, label_variables, 0)

    try:
        F_geno = _factor_base_indices(mask_groups, label_variables, 2)
    except Exception:
        F_geno = {}

    idx2_parts, idx4_parts = [], []
    for ages in F.values():
        if ages.get("2m") is not None:
            idx2_parts.append(ages["2m"])
        if ages.get("4m") is not None:
            idx4_parts.append(ages["4m"])
    idx2 = np.unique(np.concatenate(idx2_parts)) if idx2_parts else np.array([], dtype=int)
    idx4 = np.unique(np.concatenate(idx4_parts)) if idx4_parts else np.array([], dtype=int)

    if idx2.size == 0 or idx4.size == 0:
        logger.error("Could not resolve age group indices (2m/4m). Abort.")
        return 2

    patterns = _compile_patterns(args.roi_substrings)
    selected_links = [i for i, pair in enumerate(pair_labels) if _link_matches(tuple(map(str, pair)), patterns)]
    if not selected_links:
        logger.warning("No links matched patterns: %s", args.roi_substrings)
        return 0

    extra = f"_{args.tag.strip()}" if args.tag and args.tag.strip() else ""
    tag = f"w{args.window_size}_lag{args.lag}_tau{args.tau}_{args.roi_scope}{extra}"

    out_dir_stats = (paths["allegiance"] / "out").expanduser()
    age_pvals = None
    group_pvals = None
    if args.annotate_stats in {"age", "both"}:
        age_pvals = _load_stats_csv(out_dir_stats / f"pvals_age_wilcoxon_{tag}.csv")
    if args.annotate_stats in {"group", "both"}:
        group_pvals = _load_stats_csv(out_dir_stats / f"pvals_group_mwu_{tag}.csv")

    for link_idx in selected_links:
        label_a, label_b = map(str, pair_labels[link_idx])
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["2m", "4m"])
        ax.set_ylabel("Cohesion (time ratio)")

        specs, labels = _build_group_specs(args, args.color_by, link_idx, time_ratio, F, F_geno, idx2, idx4)

        default_paired = AgeArrays(time_ratio[idx2, link_idx], time_ratio[idx4, link_idx])
        inset_lines, values_2m, values_4m = _plot_groups(
            ax,
            args,
            link_idx,
            specs,
            time_ratio,
        )

        bars: list[tuple[float, float, str]] = []

        if not args.no_stats and args.color_by == "age":
            p_val = _compute_pvalue(default_paired.two_month, default_paired.four_month, paired=args.paired_age)
            if p_val is not None:
                bars.append((0.0, 1.0, f"p={p_val:.3g}"))

        if not args.no_stats and args.color_by == "sex":
            offsets = {spec.key: spec.offset for spec in specs}
            # Between sexes for each age
            if {"Female", "Male"}.issubset(values_2m.keys()):
                for age, values in (("2m", values_2m), ("4m", values_4m)):
                    p_val = _compute_pvalue(values["Female"], values["Male"], paired=False)
                    if p_val is not None:
                        x = 0.0 if age == "2m" else 1.0
                        bars.append((x + offsets["Female"], x + offsets["Male"], f"{age}: p={p_val:.3g}"))

        if not args.no_stats and args.color_by in {"genotype", "both"}:
            offsets = {spec.key: spec.offset for spec in specs}
            if {"wt", "dKI"}.issubset(values_2m.keys()):
                for age, values in (("2m", values_2m), ("4m", values_4m)):
                    p_val = _compute_pvalue(values["wt"], values["dKI"], paired=False)
                    if p_val is not None:
                        x = 0.0 if age == "2m" else 1.0
                        bars.append((x + offsets["wt"], x + offsets["dKI"], f"{age}: p={p_val:.3g}"))

        if not args.no_stats and args.color_by == "sex_genotype":
            bars.extend(_sex_genotype_bars(values_2m, values_4m, specs))

        link_label = _format_link_label(label_a, label_b)
        inset_lines.extend(
            _collect_csv_annotations(
                args,
                link_label,
                group_pvals,
                age_pvals,
                labels,
                args.alpha,
            )
        )

        ax.set_title(link_label)
        _annotate_inset(ax, inset_lines)
        _reserve_axes_bars(ax, bars)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()

        safe_a = re.sub(r"[^A-Za-z0-9]+", "", label_a)
        safe_b = re.sub(r"[^A-Za-z0-9]+", "", label_b)
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


# ---------------------------------------------------------------------------
# Sex × genotype helpers
# ---------------------------------------------------------------------------


def _sex_genotype_bars(
    values_2m: dict[str, np.ndarray],
    values_4m: dict[str, np.ndarray],
    specs: list[GroupSpec],
) -> list[tuple[float, float, str]]:
    bars: list[tuple[float, float, str]] = []
    offsets = {spec.key: spec.offset for spec in specs}

    def add_bar(age: str, left: str, right: str) -> None:
        values = values_2m if age == "2m" else values_4m
        if left not in values or right not in values:
            return
        p_val = _compute_pvalue(values[left], values[right], paired=False)
        if p_val is None:
            return
        x = 0.0 if age == "2m" else 1.0
        bars.append((x + offsets[left], x + offsets[right], f"{age}: p={p_val:.3g}"))

    # Compare genotypes within each sex
    for sex_name in ("Female", "Male"):
        left = f"{sex_name} wt"
        right = f"{sex_name} dKI"
        add_bar("2m", left, right)
        add_bar("4m", left, right)

    # Compare sexes within each genotype
    for geno_key in ("wt", "dKI"):
        left = f"Female {geno_key}"
        right = f"Male {geno_key}"
        add_bar("2m", left, right)
        add_bar("4m", left, right)

    return bars


def _collect_csv_annotations(
    args: argparse.Namespace,
    link_label: str,
    group_pvals: pd.DataFrame | None,
    age_pvals: pd.DataFrame | None,
    labels: dict[str, str],
    alpha: float,
) -> list[str]:
    inset_lines: list[str] = []

    if group_pvals is not None and args.annotate_stats in {"group", "both"}:
        def fmt_line(prefix: str, p_value: float | None) -> str | None:
            if p_value is None or np.isnan(p_value):
                return None
            star = " *" if p_value <= alpha else ""
            return f"{prefix}: p={p_value:.3g}{star}"

        sex_lines: list[str] = []
        p_val = _lookup_group_p(group_pvals, link_label, "Sex", "Female-2m", "Male-2m")
        line = fmt_line("Sex 2m", p_val)
        if line:
            sex_lines.append(line)
        p_val = _lookup_group_p(group_pvals, link_label, "Sex", "Female-4m", "Male-4m")
        line = fmt_line("Sex 4m", p_val)
        if line:
            sex_lines.append(line)
        p_val = _lookup_group_p(group_pvals, link_label, "Sex", "Female (all-ages)", "Male (all-ages)")
        line = fmt_line("Sex pooled", p_val)
        if line:
            sex_lines.append(line)
        inset_lines.extend(sex_lines)

        geno_lines: list[str] = []
        p_val = _lookup_group_p(group_pvals, link_label, "Genotype", "wt-2m", "dKI-2m")
        line = fmt_line("Genotype 2m", p_val)
        if line:
            geno_lines.append(line)
        p_val = _lookup_group_p(group_pvals, link_label, "Genotype", "wt-4m", "dKI-4m")
        line = fmt_line("Genotype 4m", p_val)
        if line:
            geno_lines.append(line)
        p_val = _lookup_group_p(group_pvals, link_label, "Genotype", "wt (all-ages)", "dKI (all-ages)")
        line = fmt_line("Genotype pooled", p_val)
        if line:
            geno_lines.append(line)
        inset_lines.extend(geno_lines)

    if age_pvals is not None and args.annotate_stats in {"age", "both"}:
        try:
            for key in labels:
                if key in age_pvals.columns.get_level_values(1):
                    p_val = float(age_pvals.loc[link_label, ("Sex×Genotype", key)])
                    star = " *" if p_val <= alpha else ""
                    inset_lines.append(f"{key}: p={p_val:.3g}{star}")
        except Exception:
            pass

    return inset_lines


if __name__ == "__main__":
    raise SystemExit(main())
