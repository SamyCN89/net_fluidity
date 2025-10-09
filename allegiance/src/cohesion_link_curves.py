#!/usr/bin/env python3
"""
Plot cohesion curves for selected links and annotate p-values.

- Loads cohesion summaries from cohesion_compute.py NPZ.
- Filters links whose ROI labels contain any of the provided substrings.
- Default comparison: Age (2m vs 4m), unpaired Mann-Whitney U test.
  (Paired line plots can be added later if strict pairing indices are provided.)

Figures are saved under fig/<dataset>/cohesion/link_curves/.
"""

# %%
from __future__ import annotations

import argparse
import logging
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from shared_code.fun_paths import get_paths

logger = logging.getLogger(__name__)


from matplotlib.transforms import blended_transform_factory
import re

def _add_sig_bar_axes(ax, x1, x2, y_axes, text, h_axes=0.02, lw=1., fontsize=8):
    """
    Significance bar in blended coords: x=data, y=axes.
    If h_axes < 0, the bar points downward and the label is placed above it.
    """
    trans = blended_transform_factory(ax.transData, ax.transAxes)

    # color label red if p<0.05
    m = re.search(r"p\s*=\s*([0-9]*\.?[0-9]+(?:e-?\d+)?)", text, re.I)
    p = float(m.group(1)) if m else None
    label_color = "red" if (p is not None and p < 0.05) else "black"

    y0, y1 = y_axes, y_axes + h_axes
    ax.plot([x1, x1, x2, x2], [y0, y1, y1, y0],
            transform=trans, color="black", lw=lw, clip_on=False)

    va = "bottom" if h_axes >= 0 else "top"
    ax.text((x1 + x2) * 0.5, y1, text,
            transform=trans, ha="center", va=va,
            fontsize=fontsize, color=label_color, clip_on=False)



def setup_logging() -> None:
    if logger.handlers:
        return
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot cohesion curves for selected links")
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
        "--roi-scope",
        choices=["all", "dmn", "memory", "custom"],
        default="all",
        help="Scope used when computing cohesion (affects filename suffix)",
    )
    p.add_argument(
        "--roi-substrings",
        type=str,
        default="d HIP,v HIP,RSP",
        help="Comma-separated substrings to match in ROI labels (case-insensitive)",
    )
    p.add_argument(
        "--tag", type=str, default="", help="Optional tag appended in NPZ filename"
    )
    p.add_argument(
        "--alpha", type=float, default=0.05, help="Significance level for annotation"
    )
    p.add_argument(
        "--paired-age",
        action="store_true",
        help="Use paired 2m–4m animals (Wilcoxon) and draw per-animal lines",
    )
    p.add_argument(
        "--color-by",
        choices=["age", "sex", "genotype", "both", "sex_genotype"],
        default="age",
        help="Color coding: age, sex, genotype, both (genotype), or sex_genotype (4 groups)",
    )
    p.add_argument(
        "--errorbar",
        choices=["sd", "sem", "var"],
        default="sd",
        help="Errorbar type for aggregates: standard deviation (sd), standard error (sem), or variance (var)",
    )
    p.add_argument(
        "--no-stats",
        action="store_true",
        help="Do not compute or display any p-values on the plots",
    )
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Plot only mean ± SEM per group (no individual points or paired lines)",
    )
    p.add_argument(
        "--annotate-stats",
        choices=["none", "age", "group", "both"],
        default="none",
        help="Annotate significance from stats CSVs: age (Wilcoxon), group (MWU), or both",
    )
    p.add_argument("--save-plots", action="store_true", help="Save figures to disk")
    p.add_argument("--no-show", action="store_true", help="Headless mode (no display)")
    return p.parse_known_args()[0]  # for interactive testing
    # return p.parse_args()


def load_npz(
    paths: dict, ws: int, lag: int, tau: int, scope: str, tag: str = ""
) -> dict:
    suffix = f"_{tag.strip()}" if tag and tag.strip() else ""
    f = (
        paths["allegiance"]
        / "cohesion_data"
        / f"cohesion_data_w{ws}_lag{lag}_tau{tau}_{scope}{suffix}.npz"
    )
    return dict(np.load(f, allow_pickle=True))


def _compile_patterns(spec: str) -> list[re.Pattern[str]]:
    toks = [t.strip() for t in spec.split(",") if t.strip()]
    return [re.compile(re.escape(t), flags=re.IGNORECASE) for t in toks]


def _link_matches(pair: tuple[str, str], pats: list[re.Pattern[str]]) -> bool:
    a, b = pair
    return any(p.search(str(a)) or p.search(str(b)) for p in pats)


def _errbar(arr: np.ndarray, mode: str) -> float:
    n = int(arr.size)
    if n == 0:
        return np.nan
    if n == 1:
        # No dispersion with a single sample; draw no bar
        return 0.0
    if mode == "sd":
        return float(np.std(arr, ddof=1))
    if mode == "sem":
        sd = float(np.std(arr, ddof=1))
        return sd / float(np.sqrt(n))
    # variance
    return float(np.var(arr, ddof=1))


def _annotate_inset(ax, lines: list[str]) -> None:
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
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"
        ),
    )


# def _ensure_yroom(ax, top_needed: float, pad_frac: float = 0.05) -> None:
#     """Expand ylim so that `top_needed` fits with some padding."""
#     ymin, ymax = ax.get_ylim()
#     if top_needed > ymax:
#         rng = (top_needed - ymin)
#         ax.set_ylim(ymin, top_needed + rng * pad_frac)

# def _add_sig_bar(ax, x1: float, x2: float, y: float, text: str,
#                  h: float, lw: float = 1.2, fontsize: int = 9) -> None:
#     """
#     Draw a significance bar between x1 and x2 at baseline height y.
#     The vertical tick height is h (same units as y-data).
#     """
#     ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", lw=lw, clip_on=False)
#     ax.text((x1 + x2) * 0.5, y + h, text, ha="center", va="bottom", fontsize=fontsize)


# %%
def main() -> int:
    setup_logging()
    args = parse_args()

    if args.no_show:
        matplotlib.use("Agg", force=True)

    # Paths and output dirs
    paths = get_paths(timecourse_folder=args.timecourse_folder)
    out_dir = (paths["f_cohesion"] / "link_curves").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load cohesion data
    scope = args.roi_scope
    data = load_npz(paths, args.window_size, args.lag, args.tau, scope, args.tag)
    time_ratio = np.asarray(data["time_ratio"]).astype(float)  # (A, L)
    pair_labels = np.asarray(data["pair_labels"]).astype(object)  # (L, 2)
    anat_labels_sorted = np.asarray(data.get("anat_labels_sorted", []))

    # Load grouping for age/genotype split
    # We form unpaired groups across bases by default; paired can be requested.
    with open(paths["preprocessed"] / "grouping_data_oip.pkl", "rb") as f:
        mask_groups, label_variables = pd.read_pickle(f)

    # Factor index 3 was used for Sex in stats module; use it to collect 2m/4m indices across bases
    # Fallback: use factor 0 if structure differs
    def factor_base_indices(factor_idx: int):
        bases: dict[str, dict[str, np.ndarray | None]] = {}
        labels = label_variables[factor_idx]
        masks = mask_groups[factor_idx]
        for lbl, m in zip(labels, masks, strict=False):
            parts = str(lbl).split()
            age = parts[-1] if parts and parts[-1] in {"2m", "4m"} else None
            base = " ".join(parts[:-1]) if age else str(lbl)
            if age not in {"2m", "4m"}:
                continue
            idx = np.flatnonzero(np.asarray(m, dtype=bool))
            ent = bases.setdefault(base, {"2m": None, "4m": None})
            ent[age] = idx
        return bases

    # Gather indices for 2m and 4m groups (pooled across bases)
    try:
        F = factor_base_indices(3)  # Sex factor
    except Exception:
        F = factor_base_indices(0)  # Fallback to first factor structure

    # Also try to gather genotype-specific bases (factor index 2)
    def factor_base_indices_genotype():
        bases: dict[str, dict[str, np.ndarray | None]] = {}
        labels = label_variables[2]
        masks = mask_groups[2]
        for lbl, m in zip(labels, masks, strict=False):
            parts = str(lbl).split()
            age = parts[-1] if parts and parts[-1] in {"2m", "4m"} else None
            base = " ".join(parts[:-1]) if age else str(lbl)
            if age not in {"2m", "4m"}:
                continue
            idx = np.flatnonzero(np.asarray(m, dtype=bool))
            ent = bases.setdefault(base, {"2m": None, "4m": None})
            ent[age] = idx
        return bases

    try:
        F_geno = factor_base_indices_genotype()
    except Exception:
        F_geno = {}

    # Combine indices across bases, idx2 for 2m, idx4 for 4m
    idx2_parts, idx4_parts = [], []
    for ent in F.values():
        if ent.get("2m") is not None:
            idx2_parts.append(ent["2m"])  # type: ignore[arg-type]
        if ent.get("4m") is not None:
            idx4_parts.append(ent["4m"])  # type: ignore[arg-type]
    idx2 = (
        np.unique(np.concatenate(idx2_parts)) if idx2_parts else np.array([], dtype=int)
    )
    idx4 = (
        np.unique(np.concatenate(idx4_parts)) if idx4_parts else np.array([], dtype=int)
    )

    if idx2.size == 0 or idx4.size == 0:
        logger.error("Could not resolve age group indices (2m/4m). Abort.")
        return 2

    # Filter links by ROI substrings
    pats = _compile_patterns(args.roi_substrings)
    keep_links = [
        i for i, (a, b) in enumerate(pair_labels) if _link_matches((a, b), pats)
    ]
    if not keep_links:
        logger.warning("No links matched patterns: %s", args.roi_substrings)
        return 0

    extra = f"_{args.tag.strip()}" if args.tag and args.tag.strip() else ""
    tag = f"w{args.window_size}_lag{args.lag}_tau{args.tau}_{scope}{extra}"

    # Optional: load stats CSVs for annotation
    age_pvals = None
    group_pvals = None
    out_dir_stats = (paths["allegiance"] / "out").expanduser()
    if args.annotate_stats in {"age", "both"}:
        csv_age = out_dir_stats / f"pvals_age_wilcoxon_{tag}.csv"
        try:
            age_pvals = pd.read_csv(csv_age, index_col=0, header=[0, 1])
        except Exception as e:
            logger.warning("Could not load age stats CSV: %s (%s)", csv_age, e)
            age_pvals = None
    if args.annotate_stats in {"group", "both"}:
        csv_grp = out_dir_stats / f"pvals_group_mwu_{tag}.csv"
        try:
            group_pvals = pd.read_csv(csv_grp, index_col=0, header=[0, 1])
        except Exception as e:
            logger.warning("Could not load group stats CSV: %s (%s)", csv_grp, e)
            group_pvals = None

    def _get_group_p(
        df: pd.DataFrame | None, block: str, left: str, right: str
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

    # Helpers to assemble paired arrays across bases
    def _paired_from_F(
        Fmap: dict[str, dict[str, np.ndarray | None]], link_idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        v2_list, v4_list = [], []
        for base, ages in Fmap.items():
            i2, i4 = ages.get("2m"), ages.get("4m")
            if (
                i2 is None
                or i4 is None
                or len(i2) == 0
                or len(i4) == 0
                or len(i2) != len(i4)
            ):
                continue
            v2_list.append(time_ratio[i2, link_idx])
            v4_list.append(time_ratio[i4, link_idx])
        if v2_list and v4_list:
            return np.concatenate(v2_list), np.concatenate(v4_list)
        return np.array([]), np.array([])

    # Plot curves for each selected link
    for l in keep_links:
        a, b = map(str, pair_labels[l])
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        bars_to_draw: list[tuple[float, float, str]] = []  # (x1, x2, label)
        bars_to_draw: list[tuple[float, float, str]] = []  # (x1, x2, label)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["2m", "4m"])
        ax.set_ylabel("Cohesion (time ratio)")

        # Overall values depending on paired/unpaired (skip if --no-stats)
        p_overall = None
        if not args.no_stats:
            if args.paired_age:
                # Build paired arrays by concatenating per-base matched indices (sex bases)
                v2_all, v4_all = _paired_from_F(F, l)
                try:
                    from scipy.stats import wilcoxon

                    p_overall = float(
                        wilcoxon(
                            v2_all,
                            v4_all,
                            zero_method="wilcox",
                            alternative="two-sided",
                        ).pvalue
                    )
                except Exception:
                    p_overall = np.nan
            else:
                v2_all = time_ratio[idx2, l]
                v4_all = time_ratio[idx4, l]
                try:
                    p_overall = float(
                        mannwhitneyu(v2_all, v4_all, alternative="two-sided").pvalue
                    )
                except Exception:
                    p_overall = np.nan
        else:
            # Still compute aggregates for plotting
            if args.paired_age:
                v2_all, v4_all = _paired_from_F(F, l)
            else:
                v2_all = time_ratio[idx2, l]
                v4_all = time_ratio[idx4, l]

        # # overall
        # if (not args.no_stats) and (p_overall is not None) and (not np.isnan(p_overall)):
        #     bars_to_draw.append((0.0, 1.0, f"p={p_overall:.3g}"))

        # if (
        #     (not args.no_stats)
        #     and (p_overall is not None)
        #     and (not np.isnan(p_overall))
        # ):
        #     # test_name = "Wilcoxon" if args.paired_age else "MWU"
        #     bars_to_draw.append((0.0, 1.0, f"p={p_overall:.3g}"))

        # Plot by requested color mode
        if args.color_by == "age":
            mu2, mu4 = np.mean(v2_all), np.mean(v4_all)
            eb2 = _errbar(v2_all, args.errorbar)
            eb4 = _errbar(v4_all, args.errorbar)
            ax.errorbar(
                [0, 1],
                [mu2, mu4],
                yerr=[eb2, eb4],
                fmt="-o",
                color="tab:blue",
                capsize=3,
                lw=1.5,
            )
            if not args.aggregate_only:
                jit = 0.06
                ax.scatter(
                    np.zeros_like(v2_all) + (np.random.rand(len(v2_all)) - 0.5) * jit,
                    v2_all,
                    s=12,
                    alpha=0.6,
                    color="tab:blue",
                    label="2m",
                )
                ax.scatter(
                    np.ones_like(v4_all) + (np.random.rand(len(v4_all)) - 0.5) * jit,
                    v4_all,
                    s=12,
                    alpha=0.6,
                    color="tab:orange",
                    label="4m",
                )
                if args.paired_age:
                    for i in range(len(v2_all)):
                        ax.plot(
                            [0, 1],
                            [v2_all[i], v4_all[i]],
                            color="0.7",
                            lw=0.6,
                            alpha=0.2,
                        )
                ax.legend(frameon=False, fontsize=8)
        elif args.color_by == "sex":
            # Merge bases into Female/Male groups
            sex_groups: dict[str, dict[str, np.ndarray]] = {
                "Female": {
                    "2m": np.array([], dtype=int),
                    "4m": np.array([], dtype=int),
                },
                "Male": {"2m": np.array([], dtype=int), "4m": np.array([], dtype=int)},
            }
            for base, ages in F.items():
                key = (
                    "Female"
                    if "female" in base.lower()
                    else ("Male" if "male" in base.lower() else None)
                )
                if key is None:
                    continue
                if ages.get("2m") is not None:
                    sex_groups[key]["2m"] = np.unique(np.concatenate([sex_groups[key]["2m"], ages["2m"]]))  # type: ignore[arg-type]
                if ages.get("4m") is not None:
                    sex_groups[key]["4m"] = np.unique(np.concatenate([sex_groups[key]["4m"], ages["4m"]]))  # type: ignore[arg-type]
            colors = {"Female": "tab:purple", "Male": "tab:green"}
            offsets = {"Female": -0.08, "Male": 0.08}
            p_texts = []
            for k in ["Female", "Male"]:
                grp = sex_groups.get(k, None)
                if not grp:
                    continue
                off = offsets[k]
                c = colors[k]
                if args.paired_age:
                    # Build paired arrays per sex by concatenating per-base matches within this sex
                    # Build a restricted F_sex with only bases belonging to this sex
                    F_restricted = {
                        base: ages
                        for base, ages in F.items()
                        if (k.lower() in base.lower())
                    }
                    v2 = np.array([])
                    v4 = np.array([])
                    v2, v4 = _paired_from_F(F_restricted, l)
                    try:
                        from scipy.stats import wilcoxon

                        p_g = float(
                            wilcoxon(
                                v2, v4, zero_method="wilcox", alternative="two-sided"
                            ).pvalue
                        )
                    except Exception:
                        p_g = np.nan
                else:
                    v2 = time_ratio[grp["2m"], l]
                    v4 = time_ratio[grp["4m"], l]
                    try:
                        p_g = float(
                            mannwhitneyu(v2, v4, alternative="two-sided").pvalue
                        )
                    except Exception:
                        p_g = np.nan
                # if (not args.no_stats) and ("p_g" in locals()) and (p_g is not None) and (not np.isnan(p_g)):
                #     # Connect that group's 2m and 4m positions: (0+off) and (1+off)
                #     _reserve_and_draw(0.0 + off, 1.0 + off, f"{k}: p={p_g:.3g}")

                # inside sex loop (after p_g)
                if (not args.no_stats) and (p_g is not None) and (not np.isnan(p_g)):
                    bars_to_draw.append((0.0 + off, 1.0 + off, f"p={p_g:.3g}"))

                # if (not args.no_stats) and (p_g is not None) and (not np.isnan(p_g)):
                #     bars_to_draw.append((0.0 + off, 1.0 + off, f"{k}: p={p_g:.3g}"))

                mu2 = np.mean(v2) if v2.size else np.nan
                mu4 = np.mean(v4) if v4.size else np.nan
                eb2 = _errbar(v2, args.errorbar) if v2.size else np.nan
                eb4 = _errbar(v4, args.errorbar) if v4.size else np.nan
                ax.errorbar(
                    [0 + off, 1 + off],
                    [mu2, mu4],
                    yerr=[eb2, eb4],
                    fmt="-o",
                    color=c,
                    capsize=3,
                    lw=1.5,
                    label=k,
                )
                if not args.aggregate_only:
                    jit = 0.05
                    ax.scatter(
                        np.zeros_like(v2) + off + (np.random.rand(len(v2)) - 0.5) * jit,
                        v2,
                        s=12,
                        alpha=0.6,
                        color=c,
                    )
                    ax.scatter(
                        np.ones_like(v4) + off + (np.random.rand(len(v4)) - 0.5) * jit,
                        v4,
                        s=12,
                        alpha=0.6,
                        color=c,
                    )
                    if args.paired_age:
                        for i in range(len(v2)):
                            ax.plot(
                                [0 + off, 1 + off],
                                [v2[i], v4[i]],
                                color=c,
                                lw=0.6,
                                alpha=0.2,
                            )
                if not args.no_stats:
                    p_texts.append(f"p={p_g:.3g}")
            ax.legend(frameon=False, fontsize=8)
            # Stats annotation using CSV for Sex×Genotype → accumulate into inset lines later
            if (
                not args.no_stats
                and age_pvals is not None
                and args.annotate_stats == "age"
            ):
                p_texts = []
                link_label = f"{a}\u2013{b}"
                for sex_name, geno_key, _, _ in combos:
                    col = f"{sex_name}\u00b7{geno_key}"
                    try:
                        pv = float(age_pvals.loc[link_label, ("Sex×Genotype", col)])
                        p_texts.append(
                            f"{sex_name} {geno_key}: p={pv:.3g}{' *' if pv <= args.alpha else ''}"
                        )
                    except Exception:
                        continue
        elif args.color_by == "sex":
            # Merge bases into Female/Male groups
            sex_groups: dict[str, dict[str, np.ndarray]] = {
                "Female": {
                    "2m": np.array([], dtype=int),
                    "4m": np.array([], dtype=int),
                },
                "Male": {"2m": np.array([], dtype=int), "4m": np.array([], dtype=int)},
            }
            for base, ages in F.items():
                key = (
                    "Female"
                    if "female" in base.lower()
                    else ("Male" if "male" in base.lower() else None)
                )
                if key is None:
                    continue
                if ages.get("2m") is not None:
                    sex_groups[key]["2m"] = np.unique(np.concatenate([sex_groups[key]["2m"], ages["2m"]]))  # type: ignore[arg-type]
                if ages.get("4m") is not None:
                    sex_groups[key]["4m"] = np.unique(np.concatenate([sex_groups[key]["4m"], ages["4m"]]))  # type: ignore[arg-type]
            colors = {"Female": "tab:purple", "Male": "tab:green"}
            offsets = {"Female": -0.08, "Male": 0.08}
            p_texts = []
            for k in ["Female", "Male"]:
                grp = sex_groups.get(k, None)
                if not grp:
                    continue
                off = offsets[k]
                c = colors[k]
                if args.paired_age:
                    # Build paired arrays per sex by concatenating per-base matches within this sex
                    F_restricted = {
                        base: ages
                        for base, ages in F.items()
                        if (k.lower() in base.lower())
                    }
                    v2 = np.array([])
                    v4 = np.array([])
                    v2, v4 = _paired_from_F(F_restricted, l)
                else:
                    v2 = time_ratio[grp["2m"], l]
                    v4 = time_ratio[grp["4m"], l]
                mu2 = np.mean(v2) if v2.size else np.nan
                mu4 = np.mean(v4) if v4.size else np.nan
                eb2 = _errbar(v2, args.errorbar) if v2.size else np.nan
                eb4 = _errbar(v4, args.errorbar) if v4.size else np.nan
                ax.errorbar(
                    [0 + off, 1 + off],
                    [mu2, mu4],
                    yerr=[eb2, eb4],
                    fmt="-o",
                    color=c,
                    capsize=3,
                    lw=1.5,
                    label=k,
                )
                if not args.aggregate_only:
                    jit = 0.05
                    ax.scatter(
                        np.zeros_like(v2) + off + (np.random.rand(len(v2)) - 0.5) * jit,
                        v2,
                        s=12,
                        alpha=0.6,
                        color=c,
                    )
                    ax.scatter(
                        np.ones_like(v4) + off + (np.random.rand(len(v4)) - 0.5) * jit,
                        v4,
                        s=12,
                        alpha=0.6,
                        color=c,
                    )
                    if args.paired_age:
                        for i in range(min(len(v2), len(v4))):
                            ax.plot(
                                [0 + off, 1 + off],
                                [v2[i], v4[i]],
                                color=c,
                                lw=0.6,
                                alpha=0.2,
                            )
            ax.legend(frameon=False, fontsize=8)
        elif args.color_by == "sex_genotype":
            # Four groups: Female wt, Female dKI, Male wt, Male dKI
            combos = [
                ("Female", "wt", "tab:blue", -0.12),
                ("Female", "dKI", "tab:red", -0.04),
                ("Male", "wt", "tab:green", 0.04),
                ("Male", "dKI", "tab:orange", 0.12),
            ]

            try:
                if args.paired_age:
                    from scipy.stats import wilcoxon

                    p_g = float(
                        wilcoxon(
                            v2, v4, zero_method="wilcox", alternative="two-sided"
                        ).pvalue
                    )
                else:
                    p_g = float(mannwhitneyu(v2, v4, alternative="two-sided").pvalue)
            except Exception:
                p_g = np.nan

            # if (not args.no_stats) and (p_g is not None) and (not np.isnan(p_g)):
            #     _reserve_and_draw(0.0 + off, 1.0 + off, f"{sex_name} {geno_key}: p={p_g:.3g}")

            def collect_indices_for(
                sex_name: str, geno_key: str
            ) -> tuple[np.ndarray, np.ndarray]:
                # union across bases for sex and genotype, then intersect
                sex2, sex4 = [], []
                for base, ages in F.items():
                    if sex_name.lower() in base.lower():
                        if ages.get("2m") is not None:
                            sex2.append(ages["2m"])  # type: ignore[arg-type]
                        if ages.get("4m") is not None:
                            sex4.append(ages["4m"])  # type: ignore[arg-type]
                geno2, geno4 = [], []
                for base, ages in F_geno.items():
                    low = base.lower()
                    if (geno_key == "wt" and "wt" in low) or (
                        geno_key == "dKI" and "dki" in low
                    ):
                        if ages.get("2m") is not None:
                            geno2.append(ages["2m"])  # type: ignore[arg-type]
                        if ages.get("4m") is not None:
                            geno4.append(ages["4m"])  # type: ignore[arg-type]
                i2 = (
                    np.unique(np.concatenate(sex2)) if sex2 else np.array([], dtype=int)
                )
                j2 = (
                    np.unique(np.concatenate(geno2))
                    if geno2
                    else np.array([], dtype=int)
                )
                i4 = (
                    np.unique(np.concatenate(sex4)) if sex4 else np.array([], dtype=int)
                )
                j4 = (
                    np.unique(np.concatenate(geno4))
                    if geno4
                    else np.array([], dtype=int)
                )
                return np.intersect1d(i2, j2, assume_unique=False), np.intersect1d(
                    i4, j4, assume_unique=False
                )

            for sex_name, geno_key, color, off in combos:
                idx2_g, idx4_g = collect_indices_for(sex_name, geno_key)
                if args.paired_age:
                    # naïve pairing by aligned order within group
                    L = min(len(idx2_g), len(idx4_g))
                    v2 = time_ratio[idx2_g[:L], l]
                    v4 = time_ratio[idx4_g[:L], l]
                else:
                    v2 = time_ratio[idx2_g, l]
                    v4 = time_ratio[idx4_g, l]

                try:
                    if args.paired_age:
                        from scipy.stats import wilcoxon

                        p_g = float(
                            wilcoxon(
                                v2, v4, zero_method="wilcox", alternative="two-sided"
                            ).pvalue
                        )
                    else:
                        p_g = float(
                            mannwhitneyu(v2, v4, alternative="two-sided").pvalue
                        )
                except Exception:
                    p_g = np.nan

                # inside sex_genotype loop (after computing p_g)
                if (not args.no_stats) and (p_g is not None) and (not np.isnan(p_g)):
                    bars_to_draw.append(
                        (0.0 + off, 1.0 + off, f" p={p_g:.3g}")
                    )

                # if (not args.no_stats) and (p_g is not None) and (not np.isnan(p_g)):
                #     bars_to_draw.append((0.0 + off, 1.0 + off, f"{sex_name} {geno_key}: p={p_g:.3g}"))

                mu2 = np.mean(v2) if v2.size else np.nan
                mu4 = np.mean(v4) if v4.size else np.nan
                eb2 = _errbar(v2, args.errorbar) if v2.size else np.nan
                eb4 = _errbar(v4, args.errorbar) if v4.size else np.nan
                ax.errorbar(
                    [0 + off, 1 + off],
                    [mu2, mu4],
                    yerr=[eb2, eb4],
                    fmt="-o",
                    color=color,
                    capsize=3,
                    lw=1.5,
                    label=f"{sex_name} {geno_key}",
                )
                if not args.aggregate_only:
                    jit = 0.05
                    ax.scatter(
                        np.zeros_like(v2) + off + (np.random.rand(len(v2)) - 0.5) * jit,
                        v2,
                        s=12,
                        alpha=0.6,
                        color=color,
                    )
                    ax.scatter(
                        np.ones_like(v4) + off + (np.random.rand(len(v4)) - 0.5) * jit,
                        v4,
                        s=12,
                        alpha=0.6,
                        color=color,
                    )
                    if args.paired_age:
                        for i in range(min(len(v2), len(v4))):
                            ax.plot(
                                [0 + off, 1 + off],
                                [v2[i], v4[i]],
                                color=color,
                                lw=0.6,
                                alpha=0.2,
                            )
            ax.legend(frameon=False, fontsize=8)
        else:
            # genotype or both: split by genotype keys in F_geno
            colors = {"wt": "tab:blue", "dKI": "tab:red"}
            offsets = {"wt": -0.08, "dKI": 0.08}
            # Build genotype groups: merge bases with same genotype
            geno_groups: dict[str, dict[str, np.ndarray]] = {}
            for base, ages in F_geno.items():
                k = None
                low = base.lower()
                if "wt" in low:
                    k = "wt"
                elif "dki" in low:
                    k = "dKI"
                if k is None:
                    continue
                e = geno_groups.setdefault(
                    k, {"2m": np.array([], dtype=int), "4m": np.array([], dtype=int)}
                )
                if ages.get("2m") is not None:
                    e["2m"] = np.unique(np.concatenate([e["2m"], ages["2m"]]))  # type: ignore[arg-type]
                if ages.get("4m") is not None:
                    e["4m"] = np.unique(np.concatenate([e["4m"], ages["4m"]]))  # type: ignore[arg-type]

            p_texts = []
            for k, grp in geno_groups.items():
                c = colors.get(k, "tab:green")
                off = offsets.get(k, 0.0)
                # if (not args.no_stats) and (p_g is not None) and (not np.isnan(p_g)):
                #     _reserve_and_draw(0.0 + off, 1.0 + off, f"{k}: p={p_g:.3g}")

                if (not args.no_stats) and (p_g is not None) and (not np.isnan(p_g)):
                    bars_to_draw.append((0.0 + off, 1.0 + off, f" p={p_g:.3g}"))

                if args.paired_age:
                    present2 = np.zeros(time_ratio.shape[0], dtype=bool)
                    present2[grp["2m"]] = True
                    present4 = np.zeros(time_ratio.shape[0], dtype=bool)
                    present4[grp["4m"]] = True
                    pair_idx = np.flatnonzero(present2 & present4)
                    v2 = time_ratio[pair_idx, l]
                    v4 = time_ratio[pair_idx, l]
                    try:
                        from scipy.stats import wilcoxon

                        p_g = float(
                            wilcoxon(
                                v2, v4, zero_method="wilcox", alternative="two-sided"
                            ).pvalue
                        )
                    except Exception:
                        p_g = np.nan
                else:
                    v2 = time_ratio[grp["2m"], l]
                    v4 = time_ratio[grp["4m"], l]
                    try:
                        p_g = float(
                            mannwhitneyu(v2, v4, alternative="two-sided").pvalue
                        )
                    except Exception:
                        p_g = np.nan

                # genotype/both loop (after p_g)
                if (not args.no_stats) and (p_g is not None) and (not np.isnan(p_g)):
                    bars_to_draw.append((0.0 + off, 1.0 + off, f" p={p_g:.3g}"))

                mu2 = np.mean(v2) if v2.size else np.nan
                mu4 = np.mean(v4) if v4.size else np.nan
                eb2 = _errbar(v2, args.errorbar) if v2.size else np.nan
                eb4 = _errbar(v4, args.errorbar) if v4.size else np.nan
                ax.errorbar(
                    [0 + off, 1 + off],
                    [mu2, mu4],
                    yerr=[eb2, eb4],
                    fmt="-o",
                    color=c,
                    capsize=3,
                    lw=1.5,
                    label=k,
                )
                if not args.aggregate_only:
                    jit = 0.05
                    ax.scatter(
                        np.zeros_like(v2) + off + (np.random.rand(len(v2)) - 0.5) * jit,
                        v2,
                        s=12,
                        alpha=0.6,
                        color=c,
                    )
                    ax.scatter(
                        np.ones_like(v4) + off + (np.random.rand(len(v4)) - 0.5) * jit,
                        v4,
                        s=12,
                        alpha=0.6,
                        color=c,
                    )
                    if args.paired_age:
                        for i in range(len(v2)):
                            ax.plot(
                                [0 + off, 1 + off],
                                [v2[i], v4[i]],
                                color=c,
                                lw=0.6,
                                alpha=0.2,
                            )
                if not args.no_stats:
                    p_texts.append(f"{k} p={p_g:.3g}")
            ax.legend(frameon=False, fontsize=8)

        # Title only with link label; stats are annotated inside the axes
        ax.set_title(f"{a}–{b}")
        inset_lines: list[str] = []
        # if not args.no_stats and p_overall is not None:
        # test_name = "Wilcoxon" if args.paired_age else "MWU"
        # inset_lines.append(f"Overall {test_name}: p={p_overall:.3g}")
        try:
            if not args.no_stats and "p_texts" in locals() and p_texts:
                inset_lines.extend(p_texts)
        except Exception:
            pass
        # Group CSV-based annotations: within-age, pooled, cross-age
        link_label = f"{a}\u2013{b}"
        if group_pvals is not None and args.annotate_stats in {"group", "both"}:
            # Sex
            sex_lines = []
            p = _get_group_p(group_pvals, "Sex", "Female-2m", "Male-2m")
            if p is not None:
                sex_lines.append(f"Sex 2m: p={p:.3g}{' *' if p <= args.alpha else ''}")
            p = _get_group_p(group_pvals, "Sex", "Female-4m", "Male-4m")
            if p is not None:
                sex_lines.append(f"Sex 4m: p={p:.3g}{' *' if p <= args.alpha else ''}")
            p = _get_group_p(group_pvals, "Sex", "Female (all-ages)", "Male (all-ages)")
            if p is not None:
                sex_lines.append(
                    f"Sex pooled: p={p:.3g}{' *' if p <= args.alpha else ''}"
                )
            p = _get_group_p(group_pvals, "Sex", "Female-2m", "Male-4m")
            if p is not None:
                sex_lines.append(
                    f"Sex cross F2m vs M4m: p={p:.3g}{' *' if p <= args.alpha else ''}"
                )
            p = _get_group_p(group_pvals, "Sex", "Male-2m", "Female-4m")
            if p is not None:
                sex_lines.append(
                    f"Sex cross M2m vs F4m: p={p:.3g}{' *' if p <= args.alpha else ''}"
                )
            inset_lines.extend(sex_lines)
            # Genotype
            geno_lines = []
            p = _get_group_p(group_pvals, "Genotype", "wt-2m", "dKI-2m")
            if p is not None:
                geno_lines.append(
                    f"Genotype 2m: p={p:.3g}{' *' if p <= args.alpha else ''}"
                )
            p = _get_group_p(group_pvals, "Genotype", "wt-4m", "dKI-4m")
            if p is not None:
                geno_lines.append(
                    f"Genotype 4m: p={p:.3g}{' *' if p <= args.alpha else ''}"
                )
            p = _get_group_p(group_pvals, "Genotype", "wt (all-ages)", "dKI (all-ages)")
            if p is not None:
                geno_lines.append(
                    f"Genotype pooled: p={p:.3g}{' *' if p <= args.alpha else ''}"
                )
            p = _get_group_p(group_pvals, "Genotype", "wt-2m", "dKI-4m")
            if p is not None:
                geno_lines.append(
                    f"Genotype cross wt2m vs dKI4m: p={p:.3g}{' *' if p <= args.alpha else ''}"
                )
            p = _get_group_p(group_pvals, "Genotype", "dKI-2m", "wt-4m")
            if p is not None:
                geno_lines.append(
                    f"Genotype cross dKI2m vs wt4m: p={p:.3g}{' *' if p <= args.alpha else ''}"
                )
            inset_lines.extend(geno_lines)
        # ----- Significance bars setup -----
        # Determine a reasonable vertical step from data spread
        ymin, ymax = ax.get_ylim()
        yrng = max(1e-12, ymax - ymin)
        bar_step = 0.06 * yrng  # vertical gap between stacked bars
        tick_h = 0.03 * yrng  # height of the small vertical ticks
        top_data = ymax  # current top of axis; will expand as needed

        # def _reserve_and_draw(x1, x2, text):
        #     nonlocal top_data
        #     # reserve next "row" above the current top
        #     y0 = top_data + bar_step * 0.6
        #     # make sure we have room
        #     _ensure_yroom(ax, y0 + tick_h)
        #     # draw the bar
        #     _add_sig_bar(ax, x1, x2, y0, text=text, h=tick_h)
        #     # update top
        #     top_data = y0 + tick_h
        # Overall age comparison (2m vs 4m) – connect x=0 and x=1
        # if (not args.no_stats) and (p_overall is not None) and (not np.isnan(p_overall)):
        # test_name = "Wilcoxon" if args.paired_age else "MWU"
        # _reserve_and_draw(0.0, 1.0, f"p={p_overall:.3g}")

        # Age Wilcoxon CSV (per-block) when requested already handled in p_texts for some modes
        _annotate_inset(ax, inset_lines)
        # ax.grid(True, axis="y", alpha=0.25)
        # --- draw significance bars at the end, after plotting ---

        # Draw bars above the axes using axes coords (won’t change y-scale)
        # Draw significance bars inside the axes (no title overlap, no y-scale changes)
        if bars_to_draw:
            start  = 0.93   # just below the top of the axes
            gap    = 0.06   # vertical gap between bars (axes coords)
            tick   = 0.01  # negative height = draw downward, put label above the bar
            y_here = start
            for (x1, x2, label) in bars_to_draw:
                _add_sig_bar_axes(ax, x1, x2, y_here, text=label, h_axes=tick)
                y_here -= gap  # stack downward


        # Adjust layout so title + bars fit
        fig.tight_layout()

        safe_a = re.sub(r"[^A-Za-z0-9]+", "", a)
        safe_b = re.sub(r"[^A-Za-z0-9]+", "", b)
        fname = f"curve_{safe_a}-{safe_b}_{tag}.png"
        fpath = out_dir / fname
        if args.save_plots:
            fig.savefig(fpath, dpi=300, bbox_inches="tight")
            logger.info("Saved: %s", fpath)
        if not args.no_show:
            plt.show()
        else:
            plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# %%
