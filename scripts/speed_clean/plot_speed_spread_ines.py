#!/usr/bin/env python3
"""
plot_speed_spread_ines.py
==========================
Exploratory analysis of dFC speed spread for the INES dataset.

Produces three figure families:

A) ERRORBAR  (2-group groupings only)
   q95-q05 (top) and q99-q01 (bottom), group mean ± CI, * / ns + Δ.
   Output: spread/errorbar/spread_errorbar_{grouping}.png

B) MATRIX (2-group groupings only)
   3 heatmaps per metric (Δ / * / Δ masked), X = regions + all, Y = segments.
   Output: spread/matrices/spread_matrices_{grouping}_{metric}.png

C) ALL-PAIRS MATRIX  (all groupings, any number of groups)
   One figure per grouping × metric: one row per pair, Δ masked only.
   Output: spread/all_pairs/spread_pairs_{grouping}_{metric}.png

No PKL files needed — reads speed NPZs directly.

Usage
-----
  PYTHONPATH=~/Bureau/vscode/net_fluidity \\
  python scripts/speed_clean/plot_speed_spread_ines.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from dfc_speed_lib import (
    cdf_split_indices,
    compute_spread_matrix,
    compute_spread_matrix_all_pairs,
    discover_per_region_descriptors,
    get_group_data,
    load_speed_stack,
    load_speed_stack_single_region,
    plot_spread_by_segment,
    plot_spread_matrices,
    plot_spread_matrix_all_pairs,
    select_windows,
)

from scripts.dfc.dfc_compute import DATASET_DEFAULTS, _canonical_dataset
from shared_code.fun_loaddata import load_timeseries_bundle
from shared_code.fun_paths import get_paths
from shared_code.fun_utils import load_cognitive_data, set_figure_params


# =============================================================================
# CONFIG
# =============================================================================

DATASET_NAME = "ines"
POOL_SPLIT   = "third"

TIME_WINDOWS_RANGE = np.arange(5, 100, 1)
SPEED_LAG          = 1
SPEED_TAU          = 2

GROUPINGS_LIST = [
    "age",
    "sex",
    "genotype",
    "age_sex",
    "age_genotype",
    "age_phenotype_oip",
    "age_phenotype_nor",
    "age_sex_genotype",
    "age_sex_phenotype_oip",
    "age_sex_phenotype_nor",
]

METRICS  = ["q95q05", "q99q01"]
N_BOOT   = 5000

RUN_ERRORBAR  = True   # 2-group only
RUN_MATRIX    = True   # 2-group only: Δ / * / Δ masked per metric
RUN_ALL_PAIRS = True   # all groupings: Δ masked per pair per metric


# =============================================================================
# HELPERS
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
    bundle = load_timeseries_bundle(preprocessed_root / "ts_and_meta_2m4m.npz")
    n  = int(bundle.n_animals)
    r  = int(bundle.n_regions)
    tr = int(bundle.total_tr)
    cog_path = (
        preprocessed_root
        / f"cog_data_filtered_animals_{n}_regions_{r}_tr_{tr}.csv"
    )
    cog_data = load_cognitive_data(str(cog_path))
    return bundle, cog_data


def _speed_template(paths: dict) -> str:
    return str(
        Path(paths["speed"])
        / f"all/speed_win{{w}}_lag{SPEED_LAG}_tau{SPEED_TAU}"
          "_animals_{n_animals}_regions_{regions}.npz"
    )


def _get_ranges(speeds: list) -> dict:
    i3, ih, i23 = cdf_split_indices(speeds)
    return select_windows(POOL_SPLIT, len(speeds), i3, ih, i23)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    set_figure_params(True)

    paths = _build_paths()
    bundle, cog_data = _load_bundle_and_cog(paths)
    n_animals = int(bundle.n_animals)
    regions   = int(bundle.n_regions)

    speed_root = Path(paths["speed"])
    out_root   = Path(paths["f_speed"]) / "analysis_ines" / "spread"
    out_root.mkdir(parents=True, exist_ok=True)

    # ── Load subset all ───────────────────────────────────────────────────────
    print("\n[SPREAD] Loading subset=all ...")
    try:
        speeds_all = load_speed_stack(
            _speed_template(paths), TIME_WINDOWS_RANGE, n_animals, regions,
        )
    except FileNotFoundError as e:
        print(f"  [WARN] Cannot load 'all' speeds: {e}")
        return

    ranges_all = _get_ranges(speeds_all)
    print(f"  Segments: { {k: (list(v)[0], list(v)[-1]) for k, v in ranges_all.items()} }")

    # ── Load per_region ───────────────────────────────────────────────────────
    speeds_per_region: dict[str, list] = {}
    per_region_dir = speed_root / "per_region"

    if per_region_dir.exists():
        try:
            region_descs = discover_per_region_descriptors(
                per_region_dir, int(TIME_WINDOWS_RANGE[0]),
                n_animals, regions, lag=SPEED_LAG, tau_count=SPEED_TAU,
            )
            print(f"  Found {len(region_descs)} regions")
            for rd in region_descs:
                try:
                    speeds_per_region[rd] = load_speed_stack_single_region(
                        per_region_dir, TIME_WINDOWS_RANGE,
                        n_animals, regions, rd,
                        lag=SPEED_LAG, tau_count=SPEED_TAU,
                    )
                except FileNotFoundError:
                    print(f"    [WARN] {rd} — missing, skipping")
        except FileNotFoundError:
            print("  [WARN] No per_region speed files found")
    else:
        print("  [WARN] per_region directory not found")

    # =========================================================================
    # A. ERRORBAR  (2-group groupings)
    # =========================================================================
    if RUN_ERRORBAR:
        dir_eb = out_root / "errorbar"
        dir_eb.mkdir(exist_ok=True)
        for grouping in GROUPINGS_LIST:
            group_data = get_group_data(cog_data, DATASET_NAME, grouping)
            if len(group_data) != 2:
                continue
            print(f"\n[ERRORBAR] grouping={grouping}")
            fig = plot_spread_by_segment(
                speeds=speeds_all,
                group_data=group_data,
                ranges=ranges_all,
                grouping_label=grouping,
                n_boot=N_BOOT,
                save_path=dir_eb / f"spread_errorbar_{grouping}.png",
            )
            plt.close(fig)
            print(f"  -> saved")

    # =========================================================================
    # B. MATRIX  (2-group groupings, Δ / * / Δ masked, both metrics)
    # =========================================================================
    if RUN_MATRIX:
        dir_mat = out_root / "matrices"
        dir_mat.mkdir(exist_ok=True)
        for grouping in GROUPINGS_LIST:
            group_data = get_group_data(cog_data, DATASET_NAME, grouping)
            if len(group_data) != 2:
                continue
            print(f"\n[MATRIX] grouping={grouping}")
            spread_result = compute_spread_matrix(
                speeds_all=speeds_all,
                speeds_per_region=speeds_per_region,
                group_data=group_data,
                ranges=ranges_all,
                metrics=METRICS,
                n_boot=N_BOOT,
            )
            figs = plot_spread_matrices(
                spread_result=spread_result,
                grouping_label=grouping,
                save_dir=dir_mat,
                save_prefix=f"spread_matrices_{grouping}",
            )
            for fig in figs.values():
                plt.close(fig)
            print(f"  -> saved ({len(figs)} figures)")

    # =========================================================================
    # C. ALL-PAIRS MATRIX  (all groupings, Δ masked per pair)
    # =========================================================================
    if RUN_ALL_PAIRS:
        dir_pairs = out_root / "all_pairs"
        dir_pairs.mkdir(exist_ok=True)
        for grouping in GROUPINGS_LIST:
            group_data = get_group_data(cog_data, DATASET_NAME, grouping)
            n_groups   = len(group_data)
            n_pairs    = n_groups * (n_groups - 1) // 2
            print(f"\n[ALL-PAIRS] grouping={grouping}  ({n_groups} groups → {n_pairs} pairs)")
            for metric in METRICS:
                result = compute_spread_matrix_all_pairs(
                    speeds_all=speeds_all,
                    speeds_per_region=speeds_per_region,
                    group_data=group_data,
                    ranges=ranges_all,
                    metric=metric,
                    n_boot=N_BOOT,
                )
                fig = plot_spread_matrix_all_pairs(
                    result=result,
                    grouping_label=grouping,
                    save_path=dir_pairs / f"spread_pairs_{grouping}_{metric}.png",
                )
                plt.close(fig)
                print(f"  -> saved: spread_pairs_{grouping}_{metric}.png")

    print(f"\n[OK] All spread figures written to: {out_root}")


if __name__ == "__main__":
    main()
