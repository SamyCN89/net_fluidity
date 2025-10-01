#!/usr/bin/env python3
"""
Thin CLI for preprocessing (julien_data/1_preprocess_data_ts_cog.py).

Lowest-risk wrapper: imports the original module and calls its `main()`
with a user-chosen filter mode. Algorithms and outputs remain identical
to running the original script with the corresponding argument.

Usage:
  python julien_data/src/preprocess.py --filter-mode exclude_shortest

Filter modes:
  - exclude_shortest (default): drop the shortest time series
  - truncate: truncate all to the shortest length
  - none: no filtering by length
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path


def setup_logging() -> None:
    cfg_path = os.getenv("NET_FLUIDITY_LOGGING", "config/logging.yaml")
    try:
        cfg = Path(cfg_path)
        if cfg.exists():
            from logging.config import dictConfig
            import yaml

            with cfg.open("r") as f:
                dictConfig(yaml.safe_load(f))
            return
    except Exception:
        pass
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess time series and cognitive data")
    p.add_argument(
        "--filter-mode",
        default="exclude_shortest",
        choices=["exclude_shortest", "truncate", "none"],
        help="How to harmonize time series lengths",
    )
    p.add_argument(
        "--only-tr",
        type=int,
        default=None,
        help="If set, keep only animals with exactly this number of timepoints (overrides --filter-mode)",
    )
    return p.parse_args()


def _run_only_tr(only_tr: int) -> int:
    """Preprocess a subset restricted to a specific timepoint length.

    Mirrors the original script's saving conventions while filtering to ts.shape[0] == only_tr.
    """
    logger = logging.getLogger(__name__)

    # Imports from shared package (installed as 'shared_code'); fallback to local
    try:
        from shared_code.fun_loaddata import (
            extract_mouse_ids,
            load_mat_timeseries,
        )
        from shared_code.fun_paths import get_paths
    except ModuleNotFoundError:
        here = Path(__file__).resolve()
        local_pkg = here.parents[2] / "shared_code" / "shared_code"
        if local_pkg.exists():
            if str(local_pkg) not in sys.path:
                sys.path.insert(0, str(local_pkg))
            from fun_loaddata import (
                extract_mouse_ids,
                load_mat_timeseries,
            )  # type: ignore
            from fun_paths import get_paths  # type: ignore
        else:
            raise
    import numpy as np
    import pandas as pd

    paths = get_paths(
        dataset_name="julien_caillette",
        timecourse_folder="time_courses_2",
        cognitive_data_file="mice_groups_comp_index_2.xlsx",
        anat_labels_file="all_ROI_coimagine_2.txt",
    )

    # Load raw
    ts_list, ts_shapes, loaded_files = load_mat_timeseries(paths["timeseries"])
    ts_ids = extract_mouse_ids(loaded_files)

    # Load cognition and labels
    cog_data = pd.read_excel(paths["cog_data"], sheet_name="mice_groups_comp_index")
    cog_data["mouse"] = cog_data["mouse"].astype(str)
    region_labels = np.loadtxt(paths["labels"], dtype=str).tolist()
    region_labels_clean = [label.replace("Both_", "") for label in region_labels]

    # Match by cognition
    matched_ids = [mid for mid in ts_ids if mid in cog_data["mouse"].values]
    cog_data_filtered = cog_data.set_index("mouse").loc[matched_ids].reset_index()

    # Restrict to exact timepoints
    keep_mask = [ts.shape[0] == int(only_tr) for ts in ts_list]
    ts_filtered = [ts for ts, keep in zip(ts_list, keep_mask, strict=False) if keep]
    ts_ids_filtered = [id_ for id_, keep in zip(ts_ids, keep_mask, strict=False) if keep]

    # Intersect with cognition order
    ts_pairs = [(ts, id_) for ts, id_ in zip(ts_filtered, ts_ids_filtered, strict=False) if id_ in matched_ids]
    ts_filtered = [ts for ts, _ in ts_pairs]
    matched_ids_filtered = [id_ for _, id_ in ts_pairs]
    cog_data_filtered = cog_data.set_index("mouse").loc[matched_ids_filtered].reset_index()

    if not ts_filtered:
        logger.error("No time series match only-tr=%s", only_tr)
        return 2

    # Basic checks and enrich columns
    n_animals = len(ts_filtered)
    total_tr = int(only_tr)
    regions = int(ts_filtered[0].shape[1])
    split_grp = cog_data_filtered["grp"].str.split("_", expand=True)
    cog_data_filtered["genotype"] = split_grp[0]
    cog_data_filtered["treatment"] = split_grp[1]
    cog_data_filtered = pd.concat(
        [cog_data_filtered, pd.get_dummies(split_grp[0]), pd.get_dummies(split_grp[1])],
        axis=1,
    )
    cog_data_filtered["n_timepoints"] = [ts.shape[0] for ts in ts_filtered]

    # Metadata dict
    import pickle

    metadata_dict = {
        "mouse_metadata": cog_data_filtered.copy(),
        "region_labels": region_labels_clean,
        "n_animals": n_animals,
        "regions": regions,
        "total_tr": total_tr,
        "anat_labels": region_labels_clean,
        "filter_mode": f"only_tr={only_tr}",
    }

    preproc = Path(paths["preprocessed"])  # type: ignore[arg-type]
    preproc.mkdir(parents=True, exist_ok=True)

    meta_path = preproc / f"metadata_animals_{n_animals}_regions_{regions}_tr_{total_tr}.pkl"
    with meta_path.open("wb") as f:
        pickle.dump(metadata_dict, f)

    # Save TS
    import numpy as _np

    if all(ts.shape == ts_filtered[0].shape for ts in ts_filtered):
        ts_array = _np.stack(ts_filtered)
        _np.savez(
            preproc / f"ts_filtered_animals_{n_animals}_regions_{regions}_tr_{total_tr}.npz",
            ts=ts_array,
            n_animals=n_animals,
            total_tr=total_tr,
            regions=regions,
            anat_labels=region_labels_clean,
        )
    else:
        _np.savez(
            preproc / f"ts_filtered_animals_{n_animals}_regions_{regions}_tr_{total_tr}.npz",
            ts=_np.array(ts_filtered, dtype=object),
            n_animals=n_animals,
            total_tr=total_tr,
            regions=regions,
            anat_labels=region_labels_clean,
        )

    # Save cognition
    cog_data_filtered.to_csv(
        preproc / f"cog_data_filtered_animals_{n_animals}_regions_{regions}_tr_{total_tr}.csv",
        index=False,
    )

    logger.info("Saved metadata and filtered TS for only_tr=%s (n_animals=%s, regions=%s)", only_tr, n_animals, regions)
    return 0


def main() -> int:
    setup_logging()
    logger = logging.getLogger(__name__)
    args = parse_args()

    # If the user requests an exact timepoint subset, run that path
    if args.only_tr is not None:
        return _run_only_tr(int(args.only_tr))

    # Otherwise import and run the original module
    here = Path(__file__).resolve()
    julien_dir = here.parent.parent
    if str(julien_dir) not in sys.path:
        sys.path.insert(0, str(julien_dir))
    target = julien_dir / "1_preprocess_data_ts_cog.py"
    if not target.exists():
        logger.error("Underlying module not found: %s", target)
        return 2

    spec = importlib.util.spec_from_file_location("julien_preprocess_v1", target)
    if spec is None or spec.loader is None:
        logger.error("Failed to create import spec for %s", target)
        return 2

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    # Call the original main with the chosen filter mode (default matches original behavior)
    try:
        mod.main(filter_mode=args.filter_mode)
        return 0
    except SystemExit as e:
        code = int(e.code) if isinstance(e.code, int) else 0
        return code
    except Exception as e:  # pragma: no cover
        logger.exception("Preprocess failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
