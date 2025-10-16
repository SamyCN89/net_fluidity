#!/usr/bin/env python3
"""Compute dynamic functional connectivity (DFC) bundles for supported datasets.

This module is the canonical implementation shared by legacy entrypoints
(`allegiance/src/dfc_compute.py`, `julien_data/2_compute_dfc_stream.py`, etc.).
It currently mirrors the modernized allegiance CLI; future consolidation work
will add optional padding/splitting for mixed TR lengths and community-specific
processing paths drawn from the Julien scripts.
"""

from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from shared_code.fun_dfcspeed import ts2dfc_stream
from shared_code.fun_paths import get_paths
from shared_code.fun_utils import load_timeseries_data


DATASET_DEFAULTS: Dict[str, Dict[str, str]] = {
    "ines_abdullah": {
        "timecourse_folder": "Timecourses_updated_03052024",
        "cognitive_data_file": "ROIs.xlsx",
        "anat_labels_file": "41_Allen.txt",
        "bundle_name": "ts_and_meta_ines_abdullah.npz",
    },
    "julien_caillette": {
        "timecourse_folder": "time_courses_2",
        "cognitive_data_file": "mice_groups_comp_index_2.xlsx",
        "anat_labels_file": "all_ROI_coimagine_2.txt",
        "bundle_name": "ts_and_meta_julien_caillette.npz",
    },
}


def _canonical_dataset(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("julien"):
        return "julien_caillette"
    if lowered.startswith("ines"):
        return "ines_abdullah"
    raise ValueError(f"Unsupported dataset '{name}'. Expected something like 'julien' or 'ines'.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-name",
        default="ines",
        help="Dataset bundle to load (e.g. 'julien' or 'ines').",
    )
    parser.add_argument(
        "--bundle-name",
        default=None,
        help="Override the expected bundle filename (default inferred from dataset).",
    )
    parser.add_argument("--format", "-f", choices=["2D", "3D"], default="3D")
    parser.add_argument(
        "--cache",
        "-c",
        choices=["skip", "load", "verify", "overwrite"],
        default="skip",
        help="How to handle pre-existing DFC files.",
    )
    parser.add_argument("--wmin", type=int, default=5, help="Minimum window size.")
    parser.add_argument("--wmax", type=int, default=100, help="Maximum window size.")
    parser.add_argument("--wstep", type=int, default=1, help="Window size increment.")
    parser.add_argument("--lag", type=int, default=1, help="Lag between windows.")
    parser.add_argument("--tau", type=int, default=5, help="Tau value embedded in filenames.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of worker threads for per-animal DFC computation (1 = serial).",
    )
    return parser


def expected_shape(
    ts: np.ndarray,
    window_size: int,
    lag: int,
    format_data: str,
) -> Tuple[int, ...]:
    n_animals, total_tr, n_regions = ts.shape[:3]
    frames = (total_tr - window_size) // lag + 1
    if format_data == "3D":
        return (n_animals, n_regions, n_regions, frames)
    n_pairs = n_regions * (n_regions - 1) // 2
    return (n_animals, n_pairs, frames)


def load_timeseries(dataset_name: str, bundle_name: str | None) -> Tuple[dict, Path]:
    cfg = DATASET_DEFAULTS[dataset_name]
    paths = get_paths(
        dataset_name=dataset_name,
        timecourse_folder=cfg["timecourse_folder"],
        cognitive_data_file=cfg["cognitive_data_file"],
        anat_labels_file=cfg["anat_labels_file"],
    )
    bundle = Path(paths["preprocessed"]) / (bundle_name or cfg["bundle_name"])
    if not bundle.exists():
        raise FileNotFoundError(f"Expected bundle not found: {bundle}")
    data_ts = load_timeseries_data(bundle)
    if data_ts["ts"].ndim != 3:
        raise ValueError(f"Timeseries bundle must be 3D (got shape={data_ts['ts'].shape})")
    return data_ts, Path(paths["dfc"])


def main(argv: None | Tuple[str, ...] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)

    try:
        dataset = _canonical_dataset(args.dataset_name)
    except ValueError as exc:
        parser.error(str(exc))

    data_ts, dfc_dir = load_timeseries(dataset, args.bundle_name)
    ts = data_ts["ts"]
    n_animals = data_ts["n_animals"]
    n_regions = data_ts["regions"]

    win_range = np.arange(args.wmin, args.wmax + 1, args.wstep)
    dfc_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Computing DFC for %s (animals=%s, regions=%s) into %s",
        dataset,
        n_animals,
        n_regions,
        dfc_dir,
    )

    start = time.time()
    for w in win_range:
        frames = (ts.shape[1] - w) // args.lag + 1
        if frames <= 0:
            logger.warning("Window %s exceeds timeseries length; skipping.", w)
            continue
        filename = (
            f"dfc_window_size={w}_lag={args.lag}_tau={args.tau}_"
            f"animals={n_animals}_regions={n_regions}.npz"
        )
        fpath = dfc_dir / filename

        if fpath.exists():
            if args.cache == "skip":
                logger.info("[cache] Skipping existing %s", fpath.name)
                continue
            if args.cache in {"load", "verify"}:
                try:
                    arr = np.load(fpath)["dfc"]
                    if args.cache == "load":
                        logger.info("[cache] Loaded %s shape=%s", fpath.name, arr.shape)
                        continue
                    if tuple(arr.shape) == expected_shape(ts, w, args.lag, args.format):
                        logger.info("[cache] Verified %s; skipping", fpath.name)
                        continue
                    logger.warning(
                        "[cache] Shape mismatch for %s (%s != %s); recomputing",
                        fpath.name,
                        arr.shape,
                        expected_shape(ts, w, args.lag, args.format),
                    )
                except Exception as exc:  # pragma: no cover
                    logger.warning("[cache] Failed to read %s (%s); recomputing", fpath.name, exc)

        t0 = time.time()
        indices = range(n_animals)
        if args.jobs > 1:
            logger.debug("Using threaded execution with %s workers", args.jobs)

            def _compute(idx: int) -> np.ndarray:
                return ts2dfc_stream(ts[idx], window_size=w, lag=args.lag, format_data=args.format)

            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                dfc_list = list(pool.map(_compute, indices))
        else:
            dfc_list = [
                ts2dfc_stream(ts[i], window_size=w, lag=args.lag, format_data=args.format) for i in indices
            ]
        dfc_arr = np.asarray(dfc_list, dtype=np.float32)
        np.savez_compressed(fpath, dfc=dfc_arr)
        logger.info("Saved %s shape=%s in %.2fs", fpath.name, dfc_arr.shape, time.time() - t0)

    logger.info("Done in %.2fs", time.time() - start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
