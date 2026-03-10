#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_nor_speed_analysis_julien.py
================================
Full NOR ~ dFC speed correlation and regression analysis for the JULIEN dataset.

Pipeline
--------
1.  Load speed stacks for all connectivity subsets.
2.  Compute per-animal scalar metrics (median, q95, q99, width, asymmetry …)
    pooled within each window segment (short / mid / long).
    → df_metrics.parquet  (one row per animal × subset × segment)
3.  Within-group Spearman correlations (NOR vs each metric) with bootstrap CI.
    → corr_summary.parquet
4.  Between-group interaction models  NOR ~ metric * group  (OLS).
    → slopes_summary.parquet
5.  3-way interaction models  NOR ~ metric * group * segment.
    → segment_models.parquet
6.  [optional] Leave-one-out robustness for all subset × metric combinations.
    → loo_slopes.parquet
7.  Effect ranking table merging correlations + slopes + LOO.
    → effect_summary.parquet  +  console print of top 10
8.  Scatter + slope figures (saved per subset × metric).
9.  Multi-segment scatter row for key subsets (sal_within, dmn_touching).
10. QC speed-vs-window figures.
11. Quantile tensor NPZ.

All outputs go under:  <paths["f_speed"]>/analysis_julien/

Run
---
  python scripts/speed_clean/run_nor_speed_analysis_julien.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── make dfc_speed_lib importable from any working directory ─────────────────
sys.path.insert(0, str(Path(__file__).parent))

from scripts.dfc.dfc_compute import DATASET_DEFAULTS, _canonical_dataset
from shared_code.fun_loaddata import load_timeseries_bundle
from shared_code.fun_paths import get_paths
from shared_code.fun_utils import load_cognitive_data, set_figure_params

from dfc_speed_lib import (
    # I/O
    load_speed_stack,
    load_speed_stack_single_region,
    discover_per_region_descriptors,
    # Grouping
    make_long_cog,
    get_group_data,
    SPEED_SUBSETS,
    PRIMARY_METRICS,
    # Windowing
    cdf_split_indices,
    select_windows,
    flatten_windows,
    global_min_max,
    # Metrics
    build_metrics_with_segments,
    add_subset_segment_columns,
    # Statistics
    summarize_segment_group_models,
    build_effect_summary,
    get_top_effects,
    leave_one_out_slopes_all,
    run_primary_analysis_from_df,
    # Window × percentile correlation analysis
    compute_window_nor_correlations,
    plot_window_nor_correlations,
    # Quantile tensor
    compute_quantile_tensor,
    save_quantile_npz,
    # Plotting
    plot_qc_3panel,
    plot_multi_segment_scatter_row,
    configure_cache,
)


# =============================================================================
# ── CONFIG ───────────────────────────────────────────────────────────────────
# =============================================================================

DATASET_NAME = "julien"
POOL_SPLIT   = "third"    # "all" | "half" | "third"

TIME_WINDOWS_RANGE = np.arange(5, 100, 1)

SPEED_LAG    = 1
SPEED_TAU    = 2
SPEED_SUBSET = "all"      # primary subset for the quantile tensor

REF_GROUP  = "WT_VEH"     # reference group for interaction models
Q_GRID     = np.linspace(0, 100, 100)

# joblib bootstrap caching — set to a path to enable, None to disable
CACHE_DIR: str | None = None   # e.g. "/tmp/dfc_cache"

# Key subsets for the multi-segment scatter row figure
KEY_SUBSETS = ["sal_within", "dmn_touching"]

# ── Toggle sections ───────────────────────────────────────────────────────────
RUN_QUANTILE_TENSOR  = True
RUN_QC_PLOTS         = True
RUN_METRICS          = True   # step 2 — compute df_metrics
RUN_CORRELATIONS     = True   # step 3 — Spearman + bootstrap CI
RUN_SLOPES           = True   # step 4 — interaction model slopes
RUN_SEGMENT_MODELS   = True   # step 5 — 3-way interaction models
RUN_LOO              = False  # step 6 — LOO robustness (heavy, enable when ready)
RUN_EFFECT_SUMMARY   = False  # step 7 — requires RUN_LOO = True
RUN_SCATTER_PLOTS    = True   # step 8 — scatter + slope figures
RUN_SEGMENT_SCATTER  = True   # step 9 — multi-segment scatter rows
RUN_WINDOW_COR       = True   # step 10 — window × percentile correlation figures

# Percentiles to compute for the window×percentile analysis
WINDOW_COR_Q_GRID    = np.array([1, 5, 25, 50, 75, 95, 99], dtype=float)
WINDOW_COR_METRIC    = "spearman"   # "spearman" | "pearson"
WINDOW_COR_ALPHA     = 0.05
WINDOW_COR_FDR       = True         # Benjamini-Hochberg FDR per ROI


# =============================================================================
# ── HELPERS ──────────────────────────────────────────────────────────────────
# =============================================================================

def _build_paths() -> dict:
    dataset = _canonical_dataset(DATASET_NAME)
    cfg = DATASET_DEFAULTS[dataset]
    return get_paths(
        dataset_name=dataset,
        timecourse_folder=cfg["timecourse_folder"],
        cognitive_data_file=cfg["cognitive_data_file"],
        anat_labels_file=cfg["anat_labels_file"],
    )


def _load_bundle_and_cog(paths: dict):
    preprocessed_root = Path(paths["preprocessed"])
    bundle   = load_timeseries_bundle(preprocessed_root / "ts_and_meta_2m4m.npz")
    n        = int(bundle.n_animals)
    r        = int(bundle.n_regions)
    tr       = int(bundle.total_tr)
    cog_path = preprocessed_root / f"cog_data_filtered_animals_{n}_regions_{r}_tr_{tr}.csv"
    cog_data = load_cognitive_data(str(cog_path))
    return bundle, cog_data


def _speed_template(paths: dict, subset: str) -> str:
    return str(
        Path(paths["speed"])
        / f"{subset}/speed_win{{w}}_lag{SPEED_LAG}_tau{SPEED_TAU}"
          "_animals_{n_animals}_regions_{regions}.npz"
    )


def _load_or_compute_metrics(
    speeds_by_subset: dict,
    nor_index: np.ndarray,
    group_data: dict,
    cache_path: Path,
) -> pd.DataFrame:
    """Load df_metrics from cache if present, otherwise compute and save."""
    if cache_path.exists():
        print(f"[INFO] Loading cached df_metrics from {cache_path.name}")
        return pd.read_parquet(cache_path)

    print("[INFO] Computing df_metrics …")
    df = build_metrics_with_segments(
        speeds_by_subset=speeds_by_subset,
        nor_index=nor_index,
        group_data=group_data,
        pool_split=POOL_SPLIT,
    )
    df.to_parquet(cache_path, index=False)
    print(f"[INFO] df_metrics saved → {cache_path}")
    return df


# =============================================================================
# ── MAIN ─────────────────────────────────────────────────────────────────────
# =============================================================================

def main() -> None:
    set_figure_params(True)

    if CACHE_DIR:
        configure_cache(CACHE_DIR)

    # ── Load data ─────────────────────────────────────────────────────────────
    paths    = _build_paths()
    bundle, cog_data = _load_bundle_and_cog(paths)
    n_animals = int(bundle.n_animals)
    regions   = int(bundle.n_regions)

    speed_root = Path(paths["speed"])
    out_root   = Path(paths["f_speed"]) / "analysis_julien"
    out_root.mkdir(parents=True, exist_ok=True)

    dir_qc   = out_root / "qc";           dir_qc.mkdir(exist_ok=True)
    dir_corr = out_root / "correlation";  dir_corr.mkdir(exist_ok=True)
    dir_figs = dir_corr / "plots";        dir_figs.mkdir(exist_ok=True)

    # ── NOR index ─────────────────────────────────────────────────────────────
    if "index_NOR" not in cog_data.columns:
        raise ValueError(
            "'index_NOR' column missing from cog_data. "
            "Check the cognitive CSV path and column names."
        )
    nor_index = cog_data["index_NOR"].to_numpy(dtype=float)

    # ── Groups ────────────────────────────────────────────────────────────────
    group_data = get_group_data(cog_data, DATASET_NAME, "genotype_treatment")
    print(f"[INFO] Groups: {list(group_data.keys())}")

    # ── Primary speed stack (for tensor + QC) ─────────────────────────────────
    print(f"[INFO] Loading primary speed stack (subset={SPEED_SUBSET}) …")
    speeds_primary = load_speed_stack(
        _speed_template(paths, SPEED_SUBSET),
        TIME_WINDOWS_RANGE, n_animals, regions,
    )
    n_windows = len(speeds_primary)
    print(f"[INFO] {n_windows} windows × {n_animals} animals")

    # =========================================================================
    # 1. QUANTILE TENSOR
    # =========================================================================
    if RUN_QUANTILE_TENSOR:
        print("[INFO] Computing quantile tensor …")
        Q       = compute_quantile_tensor(speeds_primary, Q_GRID)
        df_long = make_long_cog(cog_data, DATASET_NAME)

        name_col     = "name" if "name" in df_long.columns else df_long.columns[0]
        session_name = df_long[name_col].to_numpy().astype(str)
        genotype     = df_long["genotype"].to_numpy().astype(str) \
            if "genotype" in df_long.columns else None
        treatment    = df_long["treatment"].to_numpy().astype(str) \
            if "treatment" in df_long.columns else None

        tensor_path = out_root / (
            f"session_window_speed_quantiles_dataset-{DATASET_NAME}"
            f"_subset-{SPEED_SUBSET}_lag{SPEED_LAG}_tau{SPEED_TAU}"
            f"_animals{n_animals}_regions{regions}"
            f"_nq{len(Q_GRID)}_w{int(TIME_WINDOWS_RANGE[0])}-{int(TIME_WINDOWS_RANGE[-1])}.npz"
        )
        save_quantile_npz(
            outpath=tensor_path,
            Q=Q, q_grid=Q_GRID,
            time_windows_range=TIME_WINDOWS_RANGE,
            session_name=session_name,
            genotype=genotype, treatment=treatment,
            extra={
                "dataset_name": np.array([DATASET_NAME]),
                "subset":       np.array([SPEED_SUBSET]),
                "lag":          np.int32(SPEED_LAG),
                "tau":          np.int32(SPEED_TAU),
            },
        )

    # =========================================================================
    # 2. QC PLOTS
    # =========================================================================
    if RUN_QC_PLOTS:
        for subset in SPEED_SUBSETS:
            print(f"[QC] subset={subset}")
            try:
                speeds_sub = load_speed_stack(
                    _speed_template(paths, subset),
                    TIME_WINDOWS_RANGE, n_animals, regions,
                )
            except FileNotFoundError:
                print(f"  -> skipping (files missing)")
                continue
            fig = plot_qc_3panel(
                speeds=speeds_sub,
                time_windows_range=TIME_WINDOWS_RANGE,
                group_data=group_data,
                subset_label=subset,
                dataset_name=DATASET_NAME,
                save_path=dir_qc / f"qc_speed_vs_window_{subset}.png",
            )
            plt.close(fig)

        per_region_dir = speed_root / "per_region"
        if per_region_dir.exists():
            try:
                region_descs = discover_per_region_descriptors(
                    per_region_dir, int(TIME_WINDOWS_RANGE[0]), n_animals, regions
                )
                for rd in region_descs:
                    speeds_reg = load_speed_stack_single_region(
                        per_region_dir, TIME_WINDOWS_RANGE, n_animals, regions, rd
                    )
                    fig = plot_qc_3panel(
                        speeds=speeds_reg,
                        time_windows_range=TIME_WINDOWS_RANGE,
                        group_data=group_data,
                        subset_label=f"per_region_{rd}",
                        dataset_name=DATASET_NAME,
                        save_path=dir_qc / f"qc_speed_vs_window_per_region_{rd}.png",
                    )
                    plt.close(fig)
            except FileNotFoundError:
                pass

    # =========================================================================
    # 3. LOAD ALL SPEED SUBSETS
    # =========================================================================
    speeds_by_subset: dict = {}
    for subset in SPEED_SUBSETS:
        try:
            speeds_by_subset[subset] = load_speed_stack(
                _speed_template(paths, subset),
                TIME_WINDOWS_RANGE, n_animals, regions,
            )
        except FileNotFoundError:
            print(f"  -> subset {subset}: files missing, skipping")

    if not speeds_by_subset:
        raise RuntimeError("No speed subsets could be loaded — check file paths.")
    print(f"[INFO] Loaded {len(speeds_by_subset)} speed subsets")

    # =========================================================================
    # 4. PER-ANIMAL METRICS
    # =========================================================================
    df_metrics = _load_or_compute_metrics(
        speeds_by_subset=speeds_by_subset,
        nor_index=nor_index,
        group_data=group_data,
        cache_path=dir_corr / "df_metrics.parquet",
    ) if RUN_METRICS else pd.read_parquet(dir_corr / "df_metrics.parquet")

    if "subset_base" not in df_metrics.columns:
        df_metrics = add_subset_segment_columns(df_metrics)

    print(f"[INFO] df_metrics: {len(df_metrics)} rows, "
          f"{df_metrics['subset'].nunique()} subsets × segments")

    # =========================================================================
    # 5. CORRELATIONS + SLOPES  (steps 3–4)
    # =========================================================================
    corr_summary   = None
    slopes_summary = None

    corr_path   = dir_corr / "corr_summary.parquet"
    slopes_path = dir_corr / "slopes_summary.parquet"

    if RUN_CORRELATIONS or RUN_SLOPES:
        if corr_path.exists() and slopes_path.exists():
            print("[INFO] Loading cached corr_summary and slopes_summary")
            corr_summary   = pd.read_parquet(corr_path)
            slopes_summary = pd.read_parquet(slopes_path)
        else:
            print("[INFO] Running primary analysis (correlations + slopes) …")
            corr_summary, slopes_summary = run_primary_analysis_from_df(
                df_metrics=df_metrics,
                primary_subsets=None,
                primary_metrics=PRIMARY_METRICS,
                ref_group=REF_GROUP,
                save_plots=RUN_SCATTER_PLOTS,
                fig_root=dir_figs,
            )
            if corr_summary is not None:
                corr_summary.to_parquet(corr_path, index=False)
                print(f"[OK] corr_summary → {corr_path.name}")
            if slopes_summary is not None:
                slopes_summary.to_parquet(slopes_path, index=False)
                print(f"[OK] slopes_summary → {slopes_path.name}")

    # =========================================================================
    # 6. SEGMENT × GROUP INTERACTION MODELS  (step 5)
    # =========================================================================
    seg_models_path = dir_corr / "segment_models.parquet"

    if RUN_SEGMENT_MODELS:
        if seg_models_path.exists():
            print("[INFO] Loading cached segment models")
            seg_models_df = pd.read_parquet(seg_models_path)
        else:
            print("[INFO] Fitting segment × group interaction models …")
            seg_models_df = summarize_segment_group_models(
                df_metrics, metrics=PRIMARY_METRICS,
                ref_group=REF_GROUP, ref_segment="mid",
            )
            seg_models_df.to_parquet(seg_models_path, index=False)
            print(f"[OK] segment_models → {seg_models_path.name}")

        print("\n=== SEGMENT × GROUP MODEL SUMMARY (head) ===")
        print(seg_models_df.head(10).to_string(index=False))

    # =========================================================================
    # 7. LOO ROBUSTNESS  (step 6, optional)
    # =========================================================================
    loo_all = None
    loo_path = dir_corr / "loo_slopes.parquet"

    if RUN_LOO:
        if loo_path.exists():
            print("[INFO] Loading cached LOO slopes")
            loo_all = pd.read_parquet(loo_path)
        else:
            print("[INFO] Computing LOO slopes (this may take a while) …")
            loo_all = leave_one_out_slopes_all(
                df_metrics, metrics=PRIMARY_METRICS, ref_group=REF_GROUP
            )
            loo_all.to_parquet(loo_path, index=False)
            print(f"[OK] loo_slopes → {loo_path.name}")

    # =========================================================================
    # 8. EFFECT SUMMARY + TOP CANDIDATES  (step 7, requires LOO)
    # =========================================================================
    if RUN_EFFECT_SUMMARY:
        if loo_all is None:
            print("[WARN] RUN_EFFECT_SUMMARY=True but RUN_LOO=False — skipping")
        elif corr_summary is None or slopes_summary is None:
            print("[WARN] RUN_EFFECT_SUMMARY=True but corr/slopes missing — skipping")
        else:
            effect_summary = build_effect_summary(corr_summary, slopes_summary, loo_all)
            effect_summary.to_parquet(dir_corr / "effect_summary.parquet", index=False)
            print(f"[OK] effect_summary saved")

            top = get_top_effects(effect_summary, n=10, min_loo_same_sign=0.2)
            print("\n=== TOP 10 EFFECT CANDIDATES ===")
            print(top[[
                "subset_base", "segment", "metric", "group",
                "corr_rho_boot", "corr_q", "slope", "slope_q",
                "loo_same_sign_rate", "score",
            ]].to_string(index=False))

    # =========================================================================
    # 9. MULTI-SEGMENT SCATTER ROWS  (step 9)
    # =========================================================================
    if RUN_SEGMENT_SCATTER:
        for base in KEY_SUBSETS:
            if "subset_base" not in df_metrics.columns:
                continue
            if base not in df_metrics["subset_base"].unique():
                print(f"[WARN] Key subset {base!r} not in df_metrics, skipping")
                continue
            fig, _ = plot_multi_segment_scatter_row(
                df_metrics, subset_base=base,
                metric="speed_q95", ref_group=REF_GROUP,
            )
            out_path = dir_figs / f"nor_vs_speed_q95_segments_{base}.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"[OK] segment scatter → {out_path.name}")

    # =========================================================================
    # 10. WINDOW × PERCENTILE NOR CORRELATION  (step 10)
    # =========================================================================
    cor_win_path = dir_corr / "window_nor_correlations.parquet"

    if RUN_WINDOW_COR:
        if cor_win_path.exists():
            print("[INFO] Loading cached window×percentile correlations")
            df_cor = pd.read_parquet(cor_win_path)
        else:
            print("[INFO] Computing window×percentile NOR correlations …")
            # Compute window splits so we can add pooled rows
            from dfc_speed_lib import cdf_split_indices, select_windows
            i_third, i_half, i_two_third = cdf_split_indices(speeds_primary)
            ranges = select_windows(POOL_SPLIT, n_windows, i_third, i_half, i_two_third)

            df_cor = compute_window_nor_correlations(
                speeds_by_subset=speeds_by_subset,
                nor_index=nor_index,
                group_data=group_data,
                time_windows_range=TIME_WINDOWS_RANGE,
                ranges=ranges,
                q_grid=WINDOW_COR_Q_GRID,
            )
            df_cor.to_parquet(cor_win_path, index=False)
            print(f"[OK] window_nor_correlations → {cor_win_path.name}")
            print(f"     {len(df_cor)} rows, "
                  f"{df_cor['roi'].nunique()} subsets, "
                  f"{df_cor['window'].nunique()} windows+segments")

        # Generate figures
        dir_cor_win = dir_figs / "window_correlations"
        dir_cor_win.mkdir(exist_ok=True)
        print("[INFO] Plotting window×percentile correlation figures …")
        plot_window_nor_correlations(
            df_cor=df_cor,
            metric=WINDOW_COR_METRIC,
            alpha=WINDOW_COR_ALPHA,
            fdr=WINDOW_COR_FDR,
            save_dir=dir_cor_win,
        )
        print(f"[OK] window correlation figures → {dir_cor_win}")

    print(f"\n[OK] All outputs written to: {out_root}")


if __name__ == "__main__":
    main()
