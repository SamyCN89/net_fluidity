#!/usr/bin/env python3
"""
plot_speed_distributions_ines.py
=================================
Visualise group-wise dFC speed distributions for the INES dataset.

Reads pre-computed bootstrap PKL files and produces four families of figures:
  a) Group mean speed distributions  (linear + log scale, per segment)
  b) Bootstrapped CI bands            (percentile axis, linear + log)
  c) Age contrast bands               (4M − 2M CI difference, per sub-group)
  d) Per-animal speed histograms      (individual overlaid, per group × segment)

Also produces QC figures (speed vs window size) and saves the quantile tensor.

Prerequisites
-------------
Bootstrap PKL files must already exist under:
  <paths["speed"]>/bootstrap/
  bootstrap_downsample_repeat_group_{grouping}_nresamples_10000_...seed_42.pkl

Run
---
  python scripts/speed_clean/plot_speed_distributions_ines.py
"""

from __future__ import annotations

from pathlib import Path
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

# ── make dfc_speed_lib importable from any working directory ─────────────────
sys.path.insert(0, str(Path(__file__).parent))

from dfc_speed_lib import (
    SPEED_SUBSETS,
    # Windowing
    compute_quantile_tensor,
    discover_per_region_descriptors,
    flatten_windows,
    get_group_data,
    global_min_max,
    # I/O
    load_speed_stack,
    load_speed_stack_single_region,
    # Grouping
    make_long_cog,
    plot_age_contrasts,
    plot_ci_bands,
    # Plotting
    plot_group_speed_distributions,
    plot_per_animal_histograms,
    plot_qc_3panel,
    save_quantile_npz,
)

from scripts.dfc.dfc_compute import DATASET_DEFAULTS, _canonical_dataset
from shared_code.fun_loaddata import load_timeseries_bundle
from shared_code.fun_paths import get_paths
from shared_code.fun_utils import load_cognitive_data, set_figure_params

# =============================================================================
# ── CONFIG ───────────────────────────────────────────────────────────────────
# =============================================================================

DATASET_NAME = "ines"
POOL_SPLIT = "third"  # "all" | "half" | "third"
BINS_HIST = 200  # bins for group-mean histograms
BINS_ANIMAL = 100  # bins for per-animal histograms

TIME_WINDOWS_RANGE = np.arange(5, 100, 1)

SPEED_LAG = 1
SPEED_TAU = 2
SPEED_SUBSET = "all"  # primary subset for the quantile tensor

Q_GRID = np.linspace(0, 100, 100)

# Bootstrap PKL filename pattern — adjust if your naming differs
PKL_PATTERN = (
    "bootstrap_downsample_repeat"
    "_subset_all"
    "_group_{grouping}"
    "_nresamples_10000_downsample_factor_10_seed_42.pkl"
)

# Groupings to plot (must match PKL filenames)
GROUPS_LIST = [
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

# ── Toggle sections ───────────────────────────────────────────────────────────

RUN_QUANTILE_TENSOR = False  # ya existe
RUN_QC_PLOTS = False  # ya existe
RUN_GROUP_DISTS = True  # group mean distributions
RUN_CI_BANDS = True  # bootstrapped CI bands
RUN_AGE_CONTRASTS = True  # 4M - 2M contrast (only groupings containing "age")
RUN_ANIMAL_HISTS = True  # per-animal overlaid histograms
RUN_PER_REGION = True


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
    bundle = load_timeseries_bundle(preprocessed_root / "ts_and_meta_2m4m.npz")
    n = int(bundle.n_animals)
    r = int(bundle.n_regions)
    tr = int(bundle.total_tr)
    cog_path = (
        preprocessed_root / f"cog_data_filtered_animals_{n}_regions_{r}_tr_{tr}.csv"
    )
    cog_data = load_cognitive_data(str(cog_path))
    return bundle, cog_data


def _speed_template(paths: dict, subset: str) -> str:
    return str(
        Path(paths["speed"]) / f"{subset}/speed_win{{w}}_lag{SPEED_LAG}_tau{SPEED_TAU}"
        "_animals_{n_animals}_regions_{regions}.npz"
    )


# =============================================================================
# ── MAIN ─────────────────────────────────────────────────────────────────────
# =============================================================================


def main() -> None:
    set_figure_params(True)

    # ── Load data ─────────────────────────────────────────────────────────────
    paths = _build_paths()
    bundle, cog_data = _load_bundle_and_cog(paths)
    n_animals = int(bundle.n_animals)
    regions = int(bundle.n_regions)

    speed_root = Path(paths["speed"])
    bootstrap_folder = speed_root / "bootstrap"
    out_root = Path(paths["f_speed"]) / "analysis_ines"
    out_root.mkdir(parents=True, exist_ok=True)

    dir_qc = out_root / "qc"
    dir_qc.mkdir(exist_ok=True)
    dir_dist = out_root / "distribution"
    dir_dist.mkdir(exist_ok=True)

    # ── Primary speed stack (for tensor + QC) ─────────────────────────────────
    print(f"[INFO] Loading primary speed stack (subset={SPEED_SUBSET}) …")
    speeds = load_speed_stack(
        _speed_template(paths, SPEED_SUBSET),
        TIME_WINDOWS_RANGE,
        n_animals,
        regions,
    )
    n_windows = len(speeds)
    print(f"[INFO] {n_windows} windows × {n_animals} animals")

    # Shared histogram bin axis
    all_flat = flatten_windows(speeds, 0, n_windows)
    edges = np.linspace(*global_min_max([all_flat]), BINS_HIST + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # =========================================================================
    # 1. QUANTILE TENSOR
    # =========================================================================
    if RUN_QUANTILE_TENSOR:
        print("[INFO] Computing quantile tensor …")
        Q = compute_quantile_tensor(speeds, Q_GRID)
        df_long = make_long_cog(cog_data, DATASET_NAME)

        session_col = "Name" if "Name" in df_long.columns else "name"
        session_name = (
            df_long[session_col].to_numpy().astype(str)
            if session_col in df_long.columns
            else np.array([f"s{i:03d}" for i in range(n_animals)])
        )

        genotype = (
            df_long["Genotype"].to_numpy().astype(str)
            if "Genotype" in df_long.columns
            else None
        )

        tensor_path = out_root / (
            f"session_window_speed_quantiles_dataset-{DATASET_NAME}"
            f"_subset-{SPEED_SUBSET}_lag{SPEED_LAG}_tau{SPEED_TAU}"
            f"_animals{n_animals}_regions{regions}"
            f"_nq{len(Q_GRID)}_w{int(TIME_WINDOWS_RANGE[0])}-{int(TIME_WINDOWS_RANGE[-1])}.npz"
        )
        save_quantile_npz(
            outpath=tensor_path,
            Q=Q,
            q_grid=Q_GRID,
            time_windows_range=TIME_WINDOWS_RANGE,
            session_name=session_name,
            genotype=genotype,
            extra={
                "dataset_name": np.array([DATASET_NAME]),
                "subset": np.array([SPEED_SUBSET]),
                "lag": np.int32(SPEED_LAG),
                "tau": np.int32(SPEED_TAU),
            },
        )

    # =========================================================================
    # 2. QC PLOTS
    # =========================================================================
    if RUN_QC_PLOTS:
        gd_qc = get_group_data(cog_data, DATASET_NAME, "genotype")

        for subset in SPEED_SUBSETS:
            print(f"[QC] subset={subset}")
            try:
                speeds_sub = load_speed_stack(
                    _speed_template(paths, subset),
                    TIME_WINDOWS_RANGE,
                    n_animals,
                    regions,
                )
            except FileNotFoundError:
                print("  -> skipping (files missing)")
                continue

            fig = plot_qc_3panel(
                speeds=speeds_sub,
                time_windows_range=TIME_WINDOWS_RANGE,
                group_data=gd_qc,
                subset_label=subset,
                dataset_name=DATASET_NAME,
                save_path=dir_qc / f"qc_speed_vs_window_{subset}.png",
            )
            plt.close(fig)

        # per_region QC
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
                        group_data=gd_qc,
                        subset_label=f"per_region_{rd}",
                        dataset_name=DATASET_NAME,
                        save_path=dir_qc / f"qc_speed_vs_window_per_region_{rd}.png",
                    )
                    plt.close(fig)
            except FileNotFoundError:
                pass

    # =========================================================================
    # 3–4. DISTRIBUTION PLOTS FROM BOOTSTRAP PKLs
    # =========================================================================
    for grouping in GROUPS_LIST:
        print(f"\n[DIST] grouping={grouping}")

        pkl_path = bootstrap_folder / PKL_PATTERN.format(grouping=grouping)
        if not pkl_path.exists():
            print(f"  -> PKL missing: {pkl_path.name}")
            continue

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        group_data_bs = data["group_data"]
        ranges_bs = data["ranges"]
        percentiles_bs = data["percentiles_"]
        ci_low_bs = data["ci_low_repeat"]
        ci_high_bs = data["ci_high_repeat"]
        pool_group_hists = data.get("pooled_group_hists_by_segment", {})
        group_speed_seg = data.get("group_speed_by_segment", {})
        seg_names_bs = list(ranges_bs.keys())

        # 3a. Group mean distributions
        if RUN_GROUP_DISTS and pool_group_hists:
            fig = plot_group_speed_distributions(
                group_means=pool_group_hists,
                centers=centers,
                seg_names=seg_names_bs,
                groups_selected=grouping,
                save_path=dir_dist / f"group_dists_{grouping}_{POOL_SPLIT}.png",
            )
            plt.close(fig)
            print("  -> group_dists saved")

        # 3b. CI bands
        if RUN_CI_BANDS:
            fig = plot_ci_bands(
                ci_low=ci_low_bs,
                ci_high=ci_high_bs,
                percentiles_=percentiles_bs,
                seg_names=seg_names_bs,
                groups_selected=grouping,
                save_path=dir_dist / f"ci_bands_{grouping}_{POOL_SPLIT}.png",
            )
            plt.close(fig)
            print("  -> ci_bands saved")

        # 3c. Age contrast (only for groupings that contain 'age')
        if RUN_AGE_CONTRASTS and "age" in grouping:
            fig = plot_age_contrasts(
                ci_low=ci_low_bs,
                ci_high=ci_high_bs,
                percentiles_=percentiles_bs,
                seg_names=seg_names_bs,
                group_data=group_data_bs,
                groups_selected=grouping,
                save_path=dir_dist / f"age_contrast_{grouping}_{POOL_SPLIT}.png",
            )
            plt.close(fig)
            print("  -> age_contrast saved")

        # 3d. Per-animal histograms
        if RUN_ANIMAL_HISTS and group_speed_seg:
            plot_per_animal_histograms(
                group_speed_by_segment=group_speed_seg,
                seg_names=seg_names_bs,
                group_data=group_data_bs,
                save_dir=dir_dist,
                bins=BINS_ANIMAL,
            )
            print("  -> animal histograms saved")

    print(f"\n[OK] All outputs written to: {out_root}")

    # =========================================================================
    # 5. PER-REGION DISTRIBUTION PLOTS
    # =========================================================================
    if RUN_PER_REGION:
        per_region_dir = speed_root / "per_region"
        if not per_region_dir.exists():
            print(
                "[WARN] per_region speed directory not found — run dfc_speed_compute.py first"
            )
        else:
            try:
                region_descs = discover_per_region_descriptors(
                    per_region_dir,
                    int(TIME_WINDOWS_RANGE[0]),
                    n_animals,
                    regions,
                    lag=SPEED_LAG,
                    tau_count=SPEED_TAU,
                )
            except FileNotFoundError:
                region_descs = []
                print("[WARN] No per_region speed files found")

            dir_per_region = out_root / "distribution_per_region"
            dir_per_region.mkdir(exist_ok=True)

            for rd in region_descs:
                print(f"\n[PER_REGION] {rd}")

                # Shared histogram axis for this region
                try:
                    speeds_reg = load_speed_stack_single_region(
                        per_region_dir,
                        TIME_WINDOWS_RANGE,
                        n_animals,
                        regions,
                        rd,
                        lag=SPEED_LAG,
                        tau_count=SPEED_TAU,
                    )
                except FileNotFoundError:
                    print("  -> speeds missing, skipping")
                    continue

                all_flat_reg = flatten_windows(speeds_reg, 0, len(speeds_reg))
                edges_reg = np.linspace(*global_min_max([all_flat_reg]), BINS_HIST + 1)
                centers_reg = 0.5 * (edges_reg[:-1] + edges_reg[1:])

                for grouping in GROUPS_LIST:
                    subset_label = f"per_region_{rd}"
                    # PKL name: bootstrap_downsample_repeat_subset_{subset}_group_{grouping}_...
                    pkl_name = (
                        f"bootstrap_downsample_repeat"
                        f"_subset_{subset_label}"
                        f"_group_{grouping}"
                        f"_nresamples_10000_downsample_factor_10_seed_42.pkl"
                    )
                    pkl_path = bootstrap_folder / pkl_name

                    if not pkl_path.exists():
                        print(f"  -> [{grouping}] PKL not found: {pkl_name}")
                        continue

                    with open(pkl_path, "rb") as f:
                        data_reg = pickle.load(f)

                    group_data_bs = data_reg["group_data"]
                    ranges_bs = data_reg["ranges"]
                    percentiles_bs = data_reg["percentiles_"]
                    ci_low_bs = data_reg["ci_low_repeat"]
                    ci_high_bs = data_reg["ci_high_repeat"]
                    pool_group_hists = data_reg.get("pooled_group_hists_by_segment", {})
                    group_speed_seg = data_reg.get("group_speed_by_segment", {})
                    seg_names_bs = list(ranges_bs.keys())

                    tag = f"{rd}_{grouping}_{POOL_SPLIT}"

                    if RUN_GROUP_DISTS and pool_group_hists:
                        fig = plot_group_speed_distributions(
                            group_means=pool_group_hists,
                            centers=centers_reg,
                            seg_names=seg_names_bs,
                            groups_selected=grouping,
                            save_path=dir_per_region / f"group_dists_{tag}.png",
                        )
                        plt.close(fig)

                    if RUN_CI_BANDS:
                        fig = plot_ci_bands(
                            ci_low=ci_low_bs,
                            ci_high=ci_high_bs,
                            percentiles_=percentiles_bs,
                            seg_names=seg_names_bs,
                            groups_selected=grouping,
                            save_path=dir_per_region / f"ci_bands_{tag}.png",
                        )
                        plt.close(fig)

                    if RUN_AGE_CONTRASTS and "age" in grouping:
                        fig = plot_age_contrasts(
                            ci_low=ci_low_bs,
                            ci_high=ci_high_bs,
                            percentiles_=percentiles_bs,
                            seg_names=seg_names_bs,
                            group_data=group_data_bs,
                            groups_selected=grouping,
                            save_path=dir_per_region / f"age_contrast_{tag}.png",
                        )
                        plt.close(fig)

                    if RUN_ANIMAL_HISTS and group_speed_seg:
                        plot_per_animal_histograms(
                            group_speed_by_segment=group_speed_seg,
                            seg_names=seg_names_bs,
                            group_data=group_data_bs,
                            save_dir=dir_per_region,
                            bins=BINS_ANIMAL,
                        )

                    print(f"  -> [{grouping}] figures saved")

            print(f"\n[OK] Per-region outputs written to: {dir_per_region}")


if __name__ == "__main__":
    main()
