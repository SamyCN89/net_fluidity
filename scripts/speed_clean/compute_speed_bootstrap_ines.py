#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_speed_bootstrap_ines.py
================================
Compute bootstrapped CI bands over speed distributions for the INES dataset.

Replaces:  compute_speed_bootstrap_merged.py
           compute_speed_bootstrap_julien.py
           bootstrap_speed_groups_cli.py

What it does
------------
For every speed subset x grouping recipe:

  1. Load the speed stack (all windows, all animals).
  2. Segment windows into short / mid / long based on CDF of sample counts.
  3. For each segment:
       - Build per-animal normalised histograms  -> group_means_by_segment
       - Pool all speed samples per group        -> pooled_group_speed_by_segment
       - Keep raw per-animal arrays              -> group_speed_by_segment
  4. Run bootstrap_downsampling_repeat (N_RESAMPLES repeats, N/DOWNSAMPLE_FACTOR
     samples per repeat) to estimate CI bands over the percentile curve.
  5. Save everything to a PKL with the exact structure expected by
     plot_speed_distributions_ines.py.

PKL filename pattern
--------------------
  <speed_root>/bootstrap/
    bootstrap_downsample_repeat_group_{grouping}
    _nresamples_{N_RESAMPLES}_downsample_factor_{DOWNSAMPLE_FACTOR}_seed_{SEED}.pkl

PKL keys
--------
  subset, groups_selected, group_data, ranges, percentiles_, centers,
  ci_low_repeat, ci_high_repeat, ci_btr_downsample_repeat,
  group_means_by_segment, pooled_group_hists_by_segment,
  pooled_group_speed_by_segment, group_speed_by_segment

Usage
-----
  Edit the CONFIG section below, then:
    python scripts/speed_clean/compute_speed_bootstrap_ines.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from scripts.dfc.dfc_compute import DATASET_DEFAULTS, _canonical_dataset
from shared_code.fun_loaddata import load_timeseries_bundle
from shared_code.fun_paths import get_paths
from shared_code.fun_utils import load_cognitive_data

from joblib import Parallel, delayed

from dfc_speed_lib import (
    load_speed_stack,
    load_speed_stack_single_region,
    discover_per_region_descriptors,
    get_group_data,
    SPEED_SUBSETS,
    cdf_split_indices,
    select_windows,
    flatten_windows,
    global_min_max,
    build_per_animal_normalized_hists,
    flatten_group_animals_over_windows,
    get_group_animals_over_windows,
    compute_bootstrap_ci_bands,
)


# =============================================================================
# CONFIG  <-- edit here
# =============================================================================

DATASET_NAME       = "ines"
POOL_SPLIT         = "third"      # "all" | "half" | "third"
BINS_HIST          = 200

TIME_WINDOWS_RANGE = np.arange(5, 100, 1)
SPEED_LAG          = 1
SPEED_TAU          = 2

N_RESAMPLES        = 10_000
DOWNSAMPLE_FACTOR  = 10
SEED               = 42
N_JOBS             = 2            # parallel workers inside each bootstrap inner loop
N_REGION_JOBS      = 40           # parallel regions (outer loop) — set to n_cores // N_JOBS

PERCENTILES = np.linspace(0, 100, 100)

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

# None -> process all subsets in SPEED_SUBSETS
SUBSETS: list[str] | None = None

VERBOSE = False  # print per-(segment, group) bootstrap timing


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


def _speed_template(paths: dict, subset: str) -> str:
    return str(
        Path(paths["speed"])
        / f"{subset}/speed_win{{w}}_lag{SPEED_LAG}_tau{SPEED_TAU}"
          "_animals_{n_animals}_regions_{regions}.npz"
    )


def _pkl_path(bootstrap_folder: Path, grouping: str, subset_label: str = "all") -> Path:
    """Canonical PKL name: bootstrap_downsample_repeat_subset_{subset}_group_{grouping}_..."""
    return bootstrap_folder / (
        f"bootstrap_downsample_repeat"
        f"_subset_{subset_label}"
        f"_group_{grouping}"
        f"_nresamples_{N_RESAMPLES}"
        f"_downsample_factor_{DOWNSAMPLE_FACTOR}"
        f"_seed_{SEED}.pkl"
    )


# =============================================================================
# CORE
# =============================================================================

def run_bootstrap_for_subset(
    subset_label: str,
    speeds: list[list[np.ndarray]],
    cog_data,
    bootstrap_folder: Path,
) -> None:
    """Run bootstrap for one subset x all groupings. Skips existing PKLs."""
    n_windows = len(speeds)
    n_animals = len(speeds[0])

    i_third, i_half, i_two_third = cdf_split_indices(speeds)
    ranges = select_windows(POOL_SPLIT, n_windows, i_third, i_half, i_two_third)

    all_flat = flatten_windows(speeds, 0, n_windows)
    sp_min, sp_max = global_min_max([all_flat])
    edges   = np.linspace(sp_min, sp_max, BINS_HIST + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # animal_speeds[i][j] -> speed array for animal i at window j
    animal_speeds = [
        [speeds[j][i] for j in range(n_windows)]
        for i in range(n_animals)
    ]

    for grouping in GROUPS_LIST:
        pkl = _pkl_path(bootstrap_folder, grouping, subset_label)
        if pkl.exists():
            print(f"  [SKIP] {pkl.name}")
            continue

        print(f"  [RUN ] grouping={grouping}")
        group_data = get_group_data(cog_data, DATASET_NAME, grouping)

        # 1. Per-segment group mean histograms
        group_means_by_segment: dict = {}
        for seg_name, w_range in ranges.items():
            H = build_per_animal_normalized_hists(
                speeds, w_range, BINS_HIST, (sp_min, sp_max),
            )
            group_means_by_segment[seg_name] = {
                gt: np.mean(H[list(idxs)], axis=0) if len(idxs) else np.zeros(BINS_HIST)
                for gt, idxs in group_data.items()
            }

        # 2. Pooled group distributions per segment
        pooled_group_hists_by_segment: dict = {}
        pooled_group_speed_by_segment: dict = {}
        group_speed_by_segment:        dict = {}

        for seg_name, w_range in ranges.items():
            pool_hist:  dict = {}
            pool_speed: dict = {}
            grp_speed:  dict = {}

            for gt, idxs in group_data.items():
                grp_speed[gt] = (
                    get_group_animals_over_windows(animal_speeds, list(idxs), w_range),
                )
                flat = flatten_group_animals_over_windows(
                    animal_speeds, list(idxs), w_range,
                )
                pool_speed[gt] = flat
                h, _ = np.histogram(flat, bins=BINS_HIST, range=(sp_min, sp_max))
                total = h.sum()
                pool_hist[gt] = (h / total * 2.0) if total > 0 else h.astype(float)

            pooled_group_hists_by_segment[seg_name] = pool_hist
            pooled_group_speed_by_segment[seg_name] = pool_speed
            group_speed_by_segment[seg_name]        = grp_speed

        # 3. Bootstrap CI bands
        print(
            f"       bootstrap: {N_RESAMPLES} repeats x "
            f"{len(ranges)} segments x {len(group_data)} groups ..."
        )
        ci_low, ci_high, ci_matrix = compute_bootstrap_ci_bands(
            ranges=ranges,
            pooled_group_speed_by_segment=pooled_group_speed_by_segment,
            group_data=group_data,
            percentiles=PERCENTILES,
            repeat=N_RESAMPLES,
            downsample_factor=DOWNSAMPLE_FACTOR,
            seed=SEED,
            n_jobs=N_JOBS,
            verbose=VERBOSE,
        )

        # 4. Save PKL  (structure must match plot_speed_distributions_ines.py)
        payload = {
            "subset":                        subset_label,
            "groups_selected":               grouping,
            "group_data":                    group_data,
            "ranges":                        ranges,
            "percentiles_":                  PERCENTILES,
            "centers":                       centers,
            "ci_low_repeat":                 ci_low,
            "ci_high_repeat":                ci_high,
            "ci_btr_downsample_repeat":      ci_matrix,
            "group_means_by_segment":        group_means_by_segment,
            "pooled_group_hists_by_segment": pooled_group_hists_by_segment,
            "pooled_group_speed_by_segment": pooled_group_speed_by_segment,
            "group_speed_by_segment":        group_speed_by_segment,
        }
        with open(pkl, "wb") as f:
            pickle.dump(payload, f)
        print(f"       saved -> {pkl.name}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    paths = _build_paths()
    bundle, cog_data = _load_bundle_and_cog(paths)
    n_animals = int(bundle.n_animals)
    regions   = int(bundle.n_regions)

    speed_root       = Path(paths["speed"])
    bootstrap_folder = speed_root / "bootstrap"
    bootstrap_folder.mkdir(parents=True, exist_ok=True)

    subsets_to_run = SUBSETS or SPEED_SUBSETS

    for subset in subsets_to_run:
        print(f"\n{'='*60}")
        print(f"  Subset: {subset}")
        print(f"{'='*60}")

        if subset == "per_region":
            subset_dir = speed_root / "per_region"
            try:
                region_descs = discover_per_region_descriptors(
                    subset_dir=subset_dir,
                    w0=int(TIME_WINDOWS_RANGE[0]),
                    n_animals=n_animals,
                    regions=regions,
                    lag=SPEED_LAG,
                    tau_count=SPEED_TAU,
                )
            except FileNotFoundError as e:
                print(f"  [WARN] per_region: {e}")
                continue

            print(f"  Found {len(region_descs)} regions — launching {N_REGION_JOBS} parallel workers")

            def _process_region(rd: str) -> str:
                try:
                    speeds = load_speed_stack_single_region(
                        subset_dir=subset_dir,
                        time_windows_range=TIME_WINDOWS_RANGE,
                        n_animals=n_animals,
                        regions=regions,
                        region_desc=rd,
                        lag=SPEED_LAG,
                        tau_count=SPEED_TAU,
                    )
                except FileNotFoundError as e:
                    return f"[WARN] {rd}: {e}"

                run_bootstrap_for_subset(
                    subset_label=f"per_region_{rd}",
                    speeds=speeds,
                    cog_data=cog_data,
                    bootstrap_folder=bootstrap_folder,
                )
                return f"[OK] {rd}"

            results = Parallel(n_jobs=N_REGION_JOBS, prefer="processes", verbose=5)(
                delayed(_process_region)(rd) for rd in region_descs
            )
            for r in results:
                print(f"  {r}")

        else:
            template = _speed_template(paths, subset)
            try:
                speeds = load_speed_stack(template, TIME_WINDOWS_RANGE, n_animals, regions)
            except FileNotFoundError as e:
                print(f"  [WARN] Skipping: {e}")
                continue

            run_bootstrap_for_subset(
                subset_label=subset,
                speeds=speeds,
                cog_data=cog_data,
                bootstrap_folder=bootstrap_folder,
            )

    print(f"\n[OK] Bootstrap PKLs written to: {bootstrap_folder}")


if __name__ == "__main__":
    main()
