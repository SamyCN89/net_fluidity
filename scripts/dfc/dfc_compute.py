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
from typing import Dict, List, Tuple

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
        metavar="NAME",
        default="ines",
        help=(
            "Dataset bundle to load. Accepted aliases: 'julien', 'julien_caillette', "
            "'ines', 'ines_abdullah'."
        ),
    )
    parser.add_argument(
        "--bundle-name",
        default=None,
        metavar="FILENAME",
        help=(
            "Override the expected bundle filename (default inferred from dataset). "
            "Example: --bundle-name ts_and_meta_julien_custom.npz"
        ),
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["2D", "3D"],
        default="3D",
        help="Output array layout (3D keeps square matrices; 2D flattens the upper triangle).",
    )
    parser.add_argument(
        "--cache",
        "-c",
        choices=["skip", "load", "verify", "overwrite"],
        default="skip",
        help=(
            "How to handle pre-existing DFC files: skip (default), load (reuse), "
            "verify (shape check), overwrite (recompute)."
        ),
    )
    parser.add_argument(
        "--wmin",
        type=int,
        default=5,
        metavar="INT",
        help="Minimum window size (inclusive). Example: --wmin 5",
    )
    parser.add_argument(
        "--wmax",
        type=int,
        default=100,
        metavar="INT",
        help="Maximum window size (inclusive). Example: --wmax 25",
    )
    parser.add_argument(
        "--wstep",
        type=int,
        default=1,
        metavar="INT",
        help="Window size increment between consecutive runs. Example: --wstep 5",
    )
    parser.add_argument(
        "--lag",
        type=int,
        default=1,
        metavar="INT",
        help="Stride between sliding windows (e.g. --lag 1 uses every TR, --lag 2 skips one).",
    )
    parser.add_argument(
        "--tau",
        default="5",
        metavar="INTS",
        help=(
            "Identifier embedded in filenames. Accepts a single integer (e.g. --tau 5) "
            "or a comma-separated list (e.g. --tau 0,5,10). Values must be >= 0."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging verbosity. Example: --log-level DEBUG",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="INT",
        help="Number of worker threads for per-animal DFC computation (1 = serial). Example: --jobs 4",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect bundle metadata and planned outputs without computing or writing DFC files.",
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
    data_ts["bundle_path"] = bundle
    ts_shape = data_ts["ts"].shape
    logging.getLogger(__name__).info(
        "Loaded %s bundle from %s (timeseries shape=%s)", dataset_name, bundle, ts_shape
    )
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

    if args.lag <= 0:
        parser.error(f"--lag must be >= 1 (got {args.lag}).")

    tau_tokens = [token.strip() for token in str(args.tau).split(",") if token.strip()]
    if not tau_tokens:
        parser.error("--tau requires at least one non-negative integer.")
    tau_values: List[int] = []
    for token in tau_tokens:
        try:
            value = int(token)
        except ValueError:
            parser.error(f"--tau expects integers (got '{token}').")
        if value < 0:
            parser.error(f"--tau values must be >= 0 (got {value}).")
        tau_values.append(value)
    tau_display = ", ".join(str(v) for v in tau_values)

    data_ts, dfc_dir = load_timeseries(dataset, args.bundle_name)
    ts = data_ts["ts"]
    n_animals = data_ts["n_animals"]
    n_regions = data_ts["regions"]
    n_tr = ts.shape[1]
    bundle_path = data_ts.get("bundle_path")

    if bundle_path is not None:
        logger.info(
            "Using bundle %s with %s animals, %s regions, %s TRs",
            bundle_path,
            n_animals,
            n_regions,
            n_tr,
        )
    else:
        logger.info(
            "Loaded timeseries (animals=%s, regions=%s, tr=%s)",
            n_animals,
            n_regions,
            n_tr,
        )

    win_range = np.arange(args.wmin, args.wmax + 1, args.wstep)
    dfc_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Computing DFC for %s (animals=%s, regions=%s) with windows=%s..%s (step=%s), "
        "lag=%s, taus=%s; writing to %s",
        dataset,
        n_animals,
        n_regions,
        args.wmin,
        args.wmax,
        args.wstep,
        args.lag,
        tau_display,
        dfc_dir,
    )
    logger.info(
        "Output files will be named like dfc_window_size=<W>_lag=%s_tau=<T>_*.npz (one file per tau).",
        args.lag,
    )

    start = time.time()
    for tau in tau_values:
        logger.info("Processing tau=%s", tau)
        for w in win_range:
            frames = (ts.shape[1] - w) // args.lag + 1
            if frames <= 0:
                logger.warning("Window %s exceeds timeseries length for lag=%s; skipping.", w, args.lag)
                continue
            filename = (
                f"dfc_window_size={w}_lag={args.lag}_tau={tau}_"
                f"animals={n_animals}_regions={n_regions}.npz"
            )
            fpath = dfc_dir / filename

            if args.dry_run:
                if fpath.exists():
                    logger.info(
                        "[dry-run] %s exists for tau=%s; cache policy '%s' would apply.", fpath, tau, args.cache
                    )
                else:
                    logger.info("[dry-run] Would compute window=%s tau=%s -> %s", w, tau, fpath)
                continue

            if fpath.exists():
                if args.cache == "skip":
                    logger.info("[cache] Skipping existing %s", fpath)
                    continue
                if args.cache in {"load", "verify"}:
                    try:
                        arr = np.load(fpath)["dfc"]
                        if args.cache == "load":
                            logger.info("[cache] Loaded %s shape=%s", fpath, arr.shape)
                            continue
                        if tuple(arr.shape) == expected_shape(ts, w, args.lag, args.format):
                            logger.info("[cache] Verified %s; skipping", fpath)
                            continue
                        logger.warning(
                            "[cache] Shape mismatch for %s (%s != %s); recomputing",
                            fpath,
                            arr.shape,
                            expected_shape(ts, w, args.lag, args.format),
                        )
                    except Exception as exc:  # pragma: no cover
                        logger.warning("[cache] Failed to read %s (%s); recomputing", fpath, exc)

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
            logger.info("Saved %s shape=%s in %.2fs", fpath, dfc_arr.shape, time.time() - t0)

    if args.dry_run:
        logger.info("Dry run complete; no files were written.")
        return 0

    logger.info("Done in %.2fs", time.time() - start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
