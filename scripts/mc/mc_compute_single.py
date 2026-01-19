#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from shared_code.fun_loaddata import load_timeseries_bundle
from shared_code.fun_metaconnectivity import compute_metaconnectivity
from shared_code.fun_paths import get_paths


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute metaconnectivity (MC) and save a frozen artifact (M1.1)."
    )
    p.add_argument("--dataset", default="ines_abdallah")
    p.add_argument("--timecourse-folder", default="Timecourses_updated_03052024")
    p.add_argument("--cognitive-data-file", default="ROIs.xlsx")
    p.add_argument("--anat-labels-file", default="41_Allen.txt")

    p.add_argument("--bundle-npz", default="ts_and_meta_2m4m.npz")
    p.add_argument("--bundle-grouping", default="grouping_data_oip.pkl")

    p.add_argument("--window-size", type=int, default=9)
    p.add_argument("--lag", type=int, default=1)
    p.add_argument("--n-jobs", type=int, default=-1)

    # Optional: compute only one animal if ts is animal-indexed
    p.add_argument("--animal-idx", type=int, default=None)

    p.add_argument("--out", default=None, help="Output .npz path. If omitted, uses results/mc/...")
    return p.parse_args()


def main():
    args = parse_args()

    paths = get_paths(
        dataset_name=args.dataset,
        timecourse_folder=args.timecourse_folder,
        cognitive_data_file=args.cognitive_data_file,
        anat_labels_file=args.anat_labels_file,
    )

    bundle = load_timeseries_bundle(
        paths["preprocessed"] / args.bundle_npz,
        paths["preprocessed"] / args.bundle_grouping,
    )

    ts = bundle.ts
    if args.animal_idx is not None:
        # This assumes ts is shaped like (n_animals, T, n_regions) or similar.
        # If your shape differs, adjust here ONCE and keep the rest unchanged.
        ts = ts[args.animal_idx]
        animal_tag = f"animal{args.animal_idx:03d}"
    else:
        animal_tag = "ALL"

    dataset_name = paths["results"].name

    # Output path
    if args.out is None:
        out_dir = Path("results") / "mc" / dataset_name / animal_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "mc_single.npz"
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # We still pass a save_path because your function expects it.
    # We isolate its internal outputs under the same folder as the .npz.
    internal_save_dir = out_path.parent / "internal"
    internal_save_dir.mkdir(parents=True, exist_ok=True)

    mc = compute_metaconnectivity(
        ts,
        window_size=args.window_size,
        lag=args.lag,
        n_jobs=args.n_jobs,
        save_path=internal_save_dir,
    )

    # Sanity checks
    mc = np.asarray(mc)
    if mc.ndim != 2 or mc.shape[0] != mc.shape[1]:
        raise ValueError(f"Expected square MC matrix, got shape {mc.shape}")
    if not np.isfinite(mc).all():
        bad = np.isnan(mc).sum() + np.isinf(mc).sum()
        raise ValueError(f"MC contains non-finite values (count={bad})")

    config = {
        "dataset": args.dataset,
        "bundle_npz": args.bundle_npz,
        "bundle_grouping": args.bundle_grouping,
        "window_size": args.window_size,
        "lag": args.lag,
        "n_jobs": args.n_jobs,
        "animal_idx": args.animal_idx,
    }

    np.savez_compressed(
        out_path,
        mc=mc,
        config_json=json.dumps(config, sort_keys=True),
        dataset_name=dataset_name,
        animal_tag=animal_tag,
    )

    print(f"[OK] Saved: {out_path}")
    print(f"[OK] Internal outputs (if any): {internal_save_dir}")


if __name__ == "__main__":
    main()
