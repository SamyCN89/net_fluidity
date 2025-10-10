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
from dataclasses import dataclass
import logging
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Support both installed package and in-repo execution without install
try:
    from shared_code.fun_paths import get_paths  # type: ignore
    from shared_code.fun_plot import (  # type: ignore
        add_sig_bar_axes,
        annotate_inset,
        compute_pvalue,
        errbar,
    )
except Exception:  # pragma: no cover - dev fallback
    from shared_code.shared_code.fun_paths import get_paths  # type: ignore
    from shared_code.shared_code.fun_plot import (  # type: ignore
        add_sig_bar_axes,
        annotate_inset,
        compute_pvalue,
        errbar,
    )

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GroupSeries:
    """Container describing one cohort with measurements at both ages."""

    label: str
    color: str
    x2: float
    x4: float
    values_2m: np.ndarray
    values_4m: np.ndarray


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _compile_patterns(spec: str) -> list[re.Pattern[str]]:
    tokens = [token.strip() for token in spec.split(",") if token.strip()]
    return [re.compile(re.escape(token), flags=re.IGNORECASE) for token in tokens]


def _link_matches(pair: tuple[str, str], patterns: list[re.Pattern[str]]) -> bool:
    a, b = pair
    return any(pattern.search(str(a)) or pattern.search(str(b)) for pattern in patterns)


# ---------------------------------------------------------------------------
# CLI utilities
# ---------------------------------------------------------------------------


def setup_logging() -> None:
    if logger.handlers:
        return
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cohesion curves for selected links"
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
        default="d HIP,v HIP,RSP,PIR",
        help="Comma-separated substrings to match in ROI labels (case-insensitive)",
    )
    parser.add_argument(
        "--tag", type=str, default="", help="Optional tag appended in NPZ filename"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="Significance level for annotation"
    )
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
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Do not compute or display any p-values on the plots",
    )
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
    parser.add_argument(
        "--bars-from-csv",
        action="store_true",
        help=(
            "When available, use group CSV p-values for within-age bars in 'sex' and 'genotype' modes"
        ),
    )
    parser.add_argument(
        "--save-plots", action="store_true", help="Save figures to disk"
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Headless mode (no display)"
    )
    return parser.parse_known_args()[0]


def load_npz(
    paths: dict, ws: int, lag: int, tau: int, scope: str, tag: str = ""
) -> dict:
    suffix = f"_{tag.strip()}" if tag and tag.strip() else ""
    npz_path = (
        paths["allegiance"]
        / "cohesion_data"
        / f"cohesion_data_w{ws}_lag{lag}_tau{tau}_{scope}{suffix}.npz"
    )
    return dict(np.load(npz_path, allow_pickle=True))


def _load_group_maps(mask_groups, label_variables):
    """Return helper dictionaries describing the available cohort indices."""

    def factor_base_indices(factor_idx: int):
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
        by_sex = factor_base_indices(3)
    except Exception:
        by_sex = factor_base_indices(0)

    try:
        by_genotype = factor_base_indices(2)
    except Exception:
        by_genotype = {}

    return by_sex, by_genotype


def _paired_from_map(
    mapping: dict[str, dict[str, np.ndarray | None]],
    time_ratio: np.ndarray,
    link_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return paired arrays for 2m and 4m using the provided mapping."""

    values_2m, values_4m = [], []
    for ages in mapping.values():
        idx2, idx4 = ages.get("2m"), ages.get("4m")
        if (
            idx2 is None
            or idx4 is None
            or len(idx2) == 0
            or len(idx4) == 0
            or len(idx2) != len(idx4)
        ):
            continue
        values_2m.append(time_ratio[idx2, link_idx])
        values_4m.append(time_ratio[idx4, link_idx])
    if values_2m and values_4m:
        return np.concatenate(values_2m), np.concatenate(values_4m)
    return np.array([]), np.array([])


# ---------------------------------------------------------------------------
# Plotting logic
# ---------------------------------------------------------------------------


def _scatter_points(
    ax: matplotlib.axes.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    jitter: float = 0.05,
    size: float = 12.0,
    alpha: float = 0.6,
) -> None:
    if y.size == 0:
        return
    noise = (np.random.rand(y.size) - 0.5) * jitter
    ax.scatter(x + noise, y, s=size, alpha=alpha, color=color)


def _plot_group_series(
    ax: matplotlib.axes.Axes,
    series: GroupSeries,
    error_mode: str,
    *,
    aggregate_only: bool,
    paired: bool,
) -> None:
    """Plot mean ± error for ``series`` and optional paired lines."""

    means = [
        np.mean(series.values_2m) if series.values_2m.size else np.nan,
        np.mean(series.values_4m) if series.values_4m.size else np.nan,
    ]
    errors = [
        errbar(series.values_2m, error_mode) if series.values_2m.size else np.nan,
        errbar(series.values_4m, error_mode) if series.values_4m.size else np.nan,
    ]
    ax.errorbar(
        [series.x2, series.x4],
        means,
        yerr=errors,
        fmt="-o",
        color=series.color,
        capsize=3,
        lw=1.5,
        label=series.label,
    )
    if aggregate_only:
        return

    if series.values_2m.size:
        _scatter_points(
            ax,
            np.full(series.values_2m.size, series.x2, dtype=float),
            series.values_2m,
            color=series.color,
        )
    if series.values_4m.size:
        _scatter_points(
            ax,
            np.full(series.values_4m.size, series.x4, dtype=float),
            series.values_4m,
            color=series.color,
        )

    if paired and series.values_2m.size and series.values_4m.size:
        count = min(series.values_2m.size, series.values_4m.size)
        for idx in range(count):
            ax.plot(
                [series.x2, series.x4],
                [series.values_2m[idx], series.values_4m[idx]],
                color="0.7",
                lw=0.6,
                alpha=0.2,
            )


def _finalize_sig_bars(
    ax: matplotlib.axes.Axes,
    comparisons: list[tuple[float, float, str]],
    *,
    start: float = 0.93,
    gap: float = 0.06,
    tick: float = 0.01,
) -> None:
    if not comparisons:
        return
    y_here = start
    for x1, x2, label in comparisons:
        add_sig_bar_axes(ax, x1, x2, y_here, text=label, h_axes=tick)
        y_here -= gap


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------


def main() -> int:
    setup_logging()
    args = parse_args()

    if args.no_show:
        matplotlib.use("Agg", force=True)

    paths = get_paths(timecourse_folder=args.timecourse_folder)
    out_dir = (paths["f_cohesion"] / "link_curves").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_npz(
        paths, args.window_size, args.lag, args.tau, args.roi_scope, args.tag
    )
    time_ratio = np.asarray(data["time_ratio"], dtype=float)
    pair_labels = np.asarray(data["pair_labels"], dtype=object)

    with open(paths["preprocessed"] / "grouping_data_oip.pkl", "rb") as handle:
        mask_groups, label_variables = pd.read_pickle(handle)

    map_sex, map_genotype = _load_group_maps(mask_groups, label_variables)

    idx2 = [ages["2m"] for ages in map_sex.values() if ages.get("2m") is not None]
    idx4 = [ages["4m"] for ages in map_sex.values() if ages.get("4m") is not None]
    idx2 = np.unique(np.concatenate(idx2)) if idx2 else np.array([], dtype=int)
    idx4 = np.unique(np.concatenate(idx4)) if idx4 else np.array([], dtype=int)

    if idx2.size == 0 or idx4.size == 0:
        logger.error("Could not resolve age group indices (2m/4m). Abort.")
        return 2

    patterns = _compile_patterns(args.roi_substrings)
    keep_links = [
        idx
        for idx, pair in enumerate(pair_labels)
        if _link_matches(tuple(pair), patterns)
    ]

    if not keep_links:
        logger.warning("No links matched patterns: %s", args.roi_substrings)
        return 0

    extra = f"_{args.tag.strip()}" if args.tag and args.tag.strip() else ""
    tag = f"w{args.window_size}_lag{args.lag}_tau{args.tau}_{args.roi_scope}{extra}"

    age_pvals = None
    group_pvals = None
    stats_dir = (paths["allegiance"] / "out").expanduser()
    if args.annotate_stats in {"age", "both"}:
        csv_age = stats_dir / f"pvals_age_wilcoxon_{tag}.csv"
        try:
            age_pvals = pd.read_csv(csv_age, index_col=0, header=[0, 1])
        except Exception as exc:  # pragma: no cover - informational
            logger.warning("Could not load age stats CSV: %s (%s)", csv_age, exc)
    if args.annotate_stats in {"group", "both"}:
        csv_group = stats_dir / f"pvals_group_mwu_{tag}.csv"
        try:
            group_pvals = pd.read_csv(csv_group, index_col=0, header=[0, 1])
        except Exception as exc:  # pragma: no cover - informational
            logger.warning("Could not load group stats CSV: %s (%s)", csv_group, exc)

    def _lookup_group_p(
        df: pd.DataFrame | None,
        block: str,
        left: str,
        right: str,
        *,
        link_label: str,
    ) -> float | None:
        if df is None:
            return None
        key1 = (block, f"{left} vs {right}")
        key2 = (block, f"{right} vs {left}")
        try:
            if key1 in df.columns:
                return float(df.loc[link_label, key1])
            if key2 in df.columns:
                return float(df.loc[link_label, key2])
        except Exception:
            return None
        return None

    for link_idx in keep_links:
        roi_a, roi_b = map(str, pair_labels[link_idx])
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["2m", "4m"])
        ax.set_ylabel("Cohesion (time ratio)")

        group_series: list[GroupSeries] = []
        comparisons: list[tuple[float, float, str]] = []
        inset_lines: list[str] = []

        if args.paired_age:
            v2_all, v4_all = _paired_from_map(map_sex, time_ratio, link_idx)
        else:
            v2_all = time_ratio[idx2, link_idx]
            v4_all = time_ratio[idx4, link_idx]

        def add_overall_age_bar() -> None:
            # Do not compute p-values live; no overall age bar without CSV source
            return

        same_age_groups: dict[str, list[tuple[GroupSeries, np.ndarray, float]]] = {
            "2m": [],
            "4m": [],
        }

        def _register_group(series: GroupSeries) -> None:
            group_series.append(series)
            if series.values_2m.size:
                same_age_groups["2m"].append((series, series.values_2m, series.x2))
            if series.values_4m.size:
                same_age_groups["4m"].append((series, series.values_4m, series.x4))

        # ----------------------------- AGE ---------------------------------
        if args.color_by == "age":
            add_overall_age_bar()
            means = [
                np.mean(v2_all) if v2_all.size else np.nan,
                np.mean(v4_all) if v4_all.size else np.nan,
            ]
            errors = [
                errbar(v2_all, args.errorbar) if v2_all.size else np.nan,
                errbar(v4_all, args.errorbar) if v4_all.size else np.nan,
            ]
            ax.errorbar(
                [0.0, 1.0],
                means,
                yerr=errors,
                fmt="-o",
                color="tab:blue",
                capsize=3,
                lw=1.5,
                label="Mean",
            )
            if not args.aggregate_only:
                if v2_all.size:
                    _scatter_points(ax, np.zeros_like(v2_all), v2_all, color="tab:blue")
                if v4_all.size:
                    _scatter_points(
                        ax, np.ones_like(v4_all), v4_all, color="tab:orange"
                    )
                if args.paired_age:
                    count = min(v2_all.size, v4_all.size)
                    for idx in range(count):
                        ax.plot(
                            [0.0, 1.0],
                            [v2_all[idx], v4_all[idx]],
                            color="0.7",
                            lw=0.6,
                            alpha=0.2,
                        )
                ax.legend(frameon=False, fontsize=8)

        # ----------------------------- SEX ---------------------------------
        elif args.color_by == "sex":
            sex_offsets = {"Female": -0.08, "Male": 0.08}
            sex_colors = {"Female": "tab:purple", "Male": "tab:green"}

            for sex_label in ("Female", "Male"):
                relevant_bases = {
                    base: ages
                    for base, ages in map_sex.items()
                    if sex_label.lower() in base.lower()
                }
                if args.paired_age:
                    v2_sex, v4_sex = _paired_from_map(
                        relevant_bases, time_ratio, link_idx
                    )
                else:
                    idx2_sex = [
                        ages["2m"]
                        for ages in relevant_bases.values()
                        if ages.get("2m") is not None
                    ]
                    idx4_sex = [
                        ages["4m"]
                        for ages in relevant_bases.values()
                        if ages.get("4m") is not None
                    ]
                    v2_sex = (
                        time_ratio[np.unique(np.concatenate(idx2_sex)), link_idx]
                        if idx2_sex
                        else np.array([])
                    )
                    v4_sex = (
                        time_ratio[np.unique(np.concatenate(idx4_sex)), link_idx]
                        if idx4_sex
                        else np.array([])
                    )

                series = GroupSeries(
                    label=sex_label,
                    color=sex_colors[sex_label],
                    x2=0.0 + sex_offsets[sex_label],
                    x4=1.0 + sex_offsets[sex_label],
                    values_2m=v2_sex,
                    values_4m=v4_sex,
                )
                _register_group(series)

                if not args.no_stats and age_pvals is not None:
                    link_label = f"{roi_a}\u2013{roi_b}"
                    try:
                        pv = float(age_pvals.loc[link_label, ("Sex", sex_label)])
                        comparisons.append((series.x2, series.x4, f"p={pv:.3g}"))
                    except Exception:
                        pass

        # -------------------------- SEX × GENOTYPE --------------------------
        elif args.color_by == "sex_genotype":
            combos = [
                ("Female", "wt", "tab:blue", -0.12),
                ("Female", "dKI", "tab:red", -0.04),
                ("Male", "wt", "tab:green", 0.04),
                ("Male", "dKI", "tab:orange", 0.12),
            ]

            def collect_indices(
                sex_label: str, geno_key: str, age_key: str
            ) -> np.ndarray:
                sex_indices = []
                for base, ages in map_sex.items():
                    if (
                        sex_label.lower() in base.lower()
                        and ages.get(age_key) is not None
                    ):
                        sex_indices.append(ages[age_key])
                geno_indices = []
                for base, ages in map_genotype.items():
                    low = base.lower()
                    if (
                        (geno_key == "wt" and "wt" in low)
                        or (geno_key == "dKI" and "dki" in low)
                    ) and ages.get(age_key) is not None:
                        geno_indices.append(ages[age_key])
                if not sex_indices or not geno_indices:
                    return np.array([], dtype=int)
                sex_idx = np.unique(np.concatenate(sex_indices))
                geno_idx = np.unique(np.concatenate(geno_indices))
                return np.intersect1d(sex_idx, geno_idx, assume_unique=False)

            for sex_label, geno_key, color, offset in combos:
                idx2_combo = collect_indices(sex_label, geno_key, "2m")
                idx4_combo = collect_indices(sex_label, geno_key, "4m")
                if args.paired_age:
                    length = min(idx2_combo.size, idx4_combo.size)
                    v2_combo = time_ratio[idx2_combo[:length], link_idx]
                    v4_combo = time_ratio[idx4_combo[:length], link_idx]
                else:
                    v2_combo = time_ratio[idx2_combo, link_idx]
                    v4_combo = time_ratio[idx4_combo, link_idx]

                series = GroupSeries(
                    label=f"{sex_label} {geno_key}",
                    color=color,
                    x2=0.0 + offset,
                    x4=1.0 + offset,
                    values_2m=v2_combo,
                    values_4m=v4_combo,
                )
                _register_group(series)

                if not args.no_stats and age_pvals is not None:
                    link_label = f"{roi_a}\u2013{roi_b}"
                    col = f"{sex_label}\u00b7{geno_key}"
                    try:
                        pv = float(age_pvals.loc[link_label, ("Sex×Genotype", col)])
                        comparisons.append((series.x2, series.x4, f"p={pv:.3g}"))
                    except Exception:
                        pass

        # ---------------------- GENOTYPE / BOTH ------------------------------
        else:  # genotype or both
            geno_offsets = {"wt": -0.08, "dKI": 0.08}
            geno_colors = {"wt": "tab:blue", "dKI": "tab:red"}

            for geno_key in ("wt", "dKI"):
                relevant_bases = {
                    base: ages
                    for base, ages in map_genotype.items()
                    if (geno_key == "wt" and "wt" in base.lower())
                    or (geno_key == "dKI" and "dki" in base.lower())
                }
                if not relevant_bases:
                    continue
                if args.paired_age:
                    v2_geno, v4_geno = _paired_from_map(
                        relevant_bases, time_ratio, link_idx
                    )
                else:
                    idx2_geno = [
                        ages["2m"]
                        for ages in relevant_bases.values()
                        if ages.get("2m") is not None
                    ]
                    idx4_geno = [
                        ages["4m"]
                        for ages in relevant_bases.values()
                        if ages.get("4m") is not None
                    ]
                    v2_geno = (
                        time_ratio[np.unique(np.concatenate(idx2_geno)), link_idx]
                        if idx2_geno
                        else np.array([])
                    )
                    v4_geno = (
                        time_ratio[np.unique(np.concatenate(idx4_geno)), link_idx]
                        if idx4_geno
                        else np.array([])
                    )

                series = GroupSeries(
                    label=geno_key,
                    color=geno_colors.get(geno_key, "tab:green"),
                    x2=0.0 + geno_offsets[geno_key],
                    x4=1.0 + geno_offsets[geno_key],
                    values_2m=v2_geno,
                    values_4m=v4_geno,
                )
                _register_group(series)

                if not args.no_stats and age_pvals is not None:
                    link_label = f"{roi_a}\u2013{roi_b}"
                    try:
                        pv = float(age_pvals.loc[link_label, ("Genotype", geno_key)])
                        comparisons.append((series.x2, series.x4, f"p={pv:.3g}"))
                    except Exception:
                        pass

        # Between-group comparisons at each age (pairwise across all cohorts at that age)
        if args.color_by != "age" and not args.no_stats:
            for age_label, entries in same_age_groups.items():
                n = len(entries)
                if n < 2:
                    continue
                # If exactly two cohorts and CSV available, use CSV-derived p-value for sex/genotype modes
                if (
                    args.bars_from_csv
                    and group_pvals is not None
                    and n == 2
                    and args.color_by in {"sex", "genotype", "both"}
                ):
                    block = "Sex" if args.color_by == "sex" else "Genotype"
                    left_lab = "Female" if block == "Sex" else "wt"
                    right_lab = "Male" if block == "Sex" else "dKI"
                    link_label = f"{roi_a}\u2013{roi_b}"
                    p_csv = _lookup_group_p(
                        group_pvals,
                        block,
                        f"{left_lab}-{age_label}",
                        f"{right_lab}-{age_label}",
                        link_label=link_label,
                    )
                    if p_csv is not None and not np.isnan(p_csv):
                        comparisons.append((entries[0][2], entries[1][2], f"{age_label}: p={p_csv:.3g}"))
                        continue
                # If sex_genotype mode and CSV available, try pooled sex×genotype block lookups per pair
                if args.bars_from_csv and group_pvals is not None and args.color_by == "sex_genotype":
                    used_csv = False
                    link_label = f"{roi_a}\u2013{roi_b}"
                    for i in range(n - 1):
                        for j in range(i + 1, n):
                            s_i, _, x_i = entries[i]
                            s_j, _, x_j = entries[j]
                            left = f"{s_i.label}-{age_label}"
                            right = f"{s_j.label}-{age_label}"
                            p_csv = _lookup_group_p(
                                group_pvals,
                                "Sex×Genotype (pooled)",
                                left,
                                right,
                                link_label=link_label,
                            )
                            if p_csv is not None and not np.isnan(p_csv):
                                comparisons.append((x_i, x_j, f"{age_label}: p={p_csv:.3g}"))
                                used_csv = True
                    if used_csv:
                        continue
                # Do not compute p-values live; if CSV is missing, skip

        # ------------------------------------------------------------------
        # Plot traces
        # ------------------------------------------------------------------
        if args.color_by != "age":
            for series in group_series:
                _plot_group_series(
                    ax,
                    series,
                    args.errorbar,
                    aggregate_only=args.aggregate_only,
                    paired=args.paired_age,
                )
            if group_series:
                ax.legend(frameon=False, fontsize=8)

        ax.set_title(f"{roi_a}\u2013{roi_b}")

        if age_pvals is not None and args.annotate_stats in {"age", "both"}:
            link_label = f"{roi_a}\u2013{roi_b}"
            try:
                pv = float(age_pvals.loc[link_label, ("Age", "2m vs 4m")])
                inset_lines.append(f"p={pv:.3g}{' *' if pv <= args.alpha else ''}")
            except Exception:
                pass

        if group_pvals is not None and args.annotate_stats in {"group", "both"}:
            link_label = f"{roi_a}\u2013{roi_b}"
            sex_lines = []
            geno_lines = []
            p_val = _lookup_group_p(
                group_pvals, "Sex", "Female-2m", "Male-2m", link_label=link_label
            )
            if p_val is not None:
                sex_lines.append(f"p={p_val:.3g}{' *' if p_val <= args.alpha else ''}")
            p_val = _lookup_group_p(
                group_pvals, "Sex", "Female-4m", "Male-4m", link_label=link_label
            )
            if p_val is not None:
                sex_lines.append(f"p={p_val:.3g}{' *' if p_val <= args.alpha else ''}")
            p_val = _lookup_group_p(
                group_pvals, "Genotype", "wt-2m", "dKI-2m", link_label=link_label
            )
            if p_val is not None:
                geno_lines.append(f"p={p_val:.3g}{' *' if p_val <= args.alpha else ''}")
            p_val = _lookup_group_p(
                group_pvals, "Genotype", "wt-4m", "dKI-4m", link_label=link_label
            )
            if p_val is not None:
                geno_lines.append(f"p={p_val:.3g}{' *' if p_val <= args.alpha else ''}")
            inset_lines.extend(sex_lines + geno_lines)

        annotate_inset(ax, inset_lines)
        _finalize_sig_bars(ax, comparisons)
        fig.tight_layout()

        safe_a = re.sub(r"[^A-Za-z0-9]+", "", roi_a)
        safe_b = re.sub(r"[^A-Za-z0-9]+", "", roi_b)
        fname = f"curve_{safe_a}-{safe_b}_{tag}.png"
        if args.save_plots:
            fpath = out_dir / fname
            fig.savefig(fpath, dpi=300, bbox_inches="tight")
            logger.info("Saved: %s", fpath)
        if args.no_show:
            plt.close(fig)
        else:
            plt.show()

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
