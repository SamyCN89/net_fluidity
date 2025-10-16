#!/usr/bin/env python3
"""Preprocessing helpers for the Julien dataset."""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from shared_code.fun_loaddata import extract_mouse_ids, load_mat_timeseries
    from shared_code.fun_paths import get_paths
except ModuleNotFoundError:  # pragma: no cover - local fallback for dev shells
    here = Path(__file__).resolve()
    local_pkg = here.parents[2] / "shared_code" / "shared_code"
    if local_pkg.exists():
        if str(local_pkg) not in sys.path:
            sys.path.insert(0, str(local_pkg))
        from fun_loaddata import extract_mouse_ids, load_mat_timeseries  # type: ignore
        from fun_paths import get_paths  # type: ignore
    else:
        raise


@dataclass
class PreprocessResult:
    ts: List[np.ndarray]
    cog_data: pd.DataFrame
    metadata: Dict[str, object]
    matched_ids: Sequence[str]
    region_labels: Sequence[str]
    paths: Mapping[str, Path]
    filter_mode: str
    only_tr: Optional[int]
    dataset_name: str


def setup_logging() -> None:
    cfg_path = os.getenv("NET_FLUIDITY_LOGGING", "config/logging.yaml")
    try:
        cfg = Path(cfg_path)
        if cfg.exists():
            from logging.config import dictConfig
            import yaml

            with cfg.open("r", encoding="utf8") as handle:
                dictConfig(yaml.safe_load(handle))
            return
    except Exception:  # pragma: no cover - fall back on basicConfig
        pass
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--filter-mode",
        default="exclude_shortest",
        choices=["exclude_shortest", "truncate", "none"],
        help="How to harmonize time series lengths when only_tr is not provided.",
    )
    parser.add_argument(
        "--only-tr",
        type=int,
        default=None,
        help="Keep only animals with exactly this number of TRs (overrides --filter-mode).",
    )
    parser.add_argument(
        "--dataset-name",
        default="julien_caillette",
        help="Dataset key recognised by shared_code.fun_paths.get_paths.",
    )
    parser.add_argument(
        "--timecourse-folder",
        default="time_courses_2",
        help="Folder name containing time series MAT files.",
    )
    parser.add_argument(
        "--cognitive-data-file",
        default="mice_groups_comp_index_2.xlsx",
        help="Excel file containing cognitive metadata.",
    )
    parser.add_argument(
        "--cognitive-sheet",
        default="mice_groups_comp_index",
        help="Sheet name within the cognitive Excel workbook.",
    )
    parser.add_argument(
        "--anat-labels-file",
        default="all_ROI_coimagine_2.txt",
        help="Text file listing anatomical labels.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the bundle without writing files to disk.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser.parse_args(argv)


def _apply_filter_mode(
    ts_list: Sequence[np.ndarray],
    ts_ids: Sequence[str],
    filter_mode: str,
) -> Tuple[List[np.ndarray], List[str]]:
    min_timepoints = min(ts.shape[0] for ts in ts_list)
    if filter_mode == "exclude_shortest":
        keep_pairs = [
            (ts, id_)
            for ts, id_ in zip(ts_list, ts_ids, strict=False)
            if ts.shape[0] > min_timepoints
        ]
        return [ts for ts, _ in keep_pairs], [id_ for _, id_ in keep_pairs]
    if filter_mode == "truncate":
        return [ts[:min_timepoints, :] for ts in ts_list], list(ts_ids)
    if filter_mode == "none":
        return list(ts_list), list(ts_ids)
    raise ValueError(f"Unsupported filter_mode={filter_mode!r}")


def _filter_only_tr(
    ts_list: Sequence[np.ndarray],
    ts_ids: Sequence[str],
    only_tr: int,
) -> Tuple[List[np.ndarray], List[str]]:
    keep_mask = [ts.shape[0] == only_tr for ts in ts_list]
    filtered_ts = [ts for ts, keep in zip(ts_list, keep_mask, strict=False) if keep]
    filtered_ids = [id_ for id_, keep in zip(ts_ids, keep_mask, strict=False) if keep]
    return filtered_ts, filtered_ids


def _build_metadata_table(cog_df: pd.DataFrame, matched_ids: Sequence[str]) -> pd.DataFrame:
    cog_sorted = cog_df.set_index("mouse").loc[matched_ids].reset_index()
    split_grp = cog_sorted["grp"].str.split("_", expand=True)
    cog_sorted["genotype"] = split_grp[0]
    cog_sorted["treatment"] = split_grp[1]
    cog_sorted = pd.concat(
        [cog_sorted, pd.get_dummies(split_grp[0]), pd.get_dummies(split_grp[1])],
        axis=1,
    )
    return cog_sorted


def prepare_dataset(
    *,
    filter_mode: str,
    only_tr: Optional[int],
    dataset_name: str = "julien_caillette",
    timecourse_folder: str = "time_courses_2",
    cognitive_data_file: str = "mice_groups_comp_index_2.xlsx",
    anat_labels_file: str = "all_ROI_coimagine_2.txt",
    cognitive_sheet: str = "mice_groups_comp_index",
) -> PreprocessResult:
    """Load, filter, and align Julien cognitive data and time series."""

    logger = logging.getLogger(__name__)
    paths = get_paths(
        dataset_name=dataset_name,
        timecourse_folder=timecourse_folder,
        cognitive_data_file=cognitive_data_file,
        anat_labels_file=anat_labels_file,
    )

    ts_list, ts_shapes, loaded_files = load_mat_timeseries(paths["timeseries"])
    if not ts_list:
        raise RuntimeError("No time series found in the configured directory.")
    ts_ids = extract_mouse_ids(loaded_files)
    logger.debug("Loaded %d time series (unique shapes: %s)", len(ts_list), sorted(set(ts_shapes)))

    if only_tr is not None:
        ts_filtered, ts_ids_filtered = _filter_only_tr(ts_list, ts_ids, only_tr)
        effective_filter_mode = f"only_tr={only_tr}"
    else:
        ts_filtered, ts_ids_filtered = _apply_filter_mode(ts_list, ts_ids, filter_mode)
        effective_filter_mode = filter_mode

    if not ts_filtered:
        raise ValueError(
            f"No time series retained after applying filter_mode='{effective_filter_mode}'."
        )
    logger.info(
        "Retained %d/%d animals after applying %s",
        len(ts_filtered),
        len(ts_list),
        effective_filter_mode,
    )

    cog_df = pd.read_excel(paths["cog_data"], sheet_name=cognitive_sheet)
    cog_df["mouse"] = cog_df["mouse"].astype(str)
    region_labels = np.loadtxt(paths["labels"], dtype=str).tolist()
    region_labels_clean = [label.replace("Both_", "") for label in region_labels]

    matched_ids = [mid for mid in ts_ids_filtered if mid in cog_df["mouse"].values]
    ts_aligned = [
        ts for ts, id_ in zip(ts_filtered, ts_ids_filtered, strict=False) if id_ in matched_ids
    ]
    if not ts_aligned:
        raise RuntimeError("No overlap between filtered time series and cognitive metadata.")

    cog_table = _build_metadata_table(cog_df, matched_ids)
    excluded_ts = set(ts_ids_filtered) - set(matched_ids)
    excluded_cog = set(cog_df["mouse"]) - set(matched_ids)
    if excluded_ts:
        logger.warning("Time series without cognition: %s", sorted(excluded_ts))
    if excluded_cog:
        logger.warning("Cognition without time series: %s", sorted(excluded_cog))

    if len(ts_aligned) != len(cog_table):
        raise ValueError(
            f"Mismatch between time series ({len(ts_aligned)}) and cognition ({len(cog_table)})."
        )

    lengths = [ts.shape[0] for ts in ts_aligned]
    cog_table["n_timepoints"] = lengths
    n_animals = len(ts_aligned)
    regions = ts_aligned[0].shape[1]
    total_tr = int(np.unique(lengths)[-1])

    metadata = {
        "mouse_metadata": cog_table.copy(),
        "region_labels": region_labels_clean,
        "anat_labels": region_labels_clean,
        "n_animals": n_animals,
        "regions": regions,
        "total_tr": total_tr,
        "filter_mode": effective_filter_mode,
        "dataset_name": np.asarray(dataset_name, dtype=str),
        "timecourse_folder": np.asarray(timecourse_folder, dtype=str),
        "cognitive_data_file": np.asarray(cognitive_data_file, dtype=str),
        "anat_labels_file": np.asarray(anat_labels_file, dtype=str),
    }

    return PreprocessResult(
        ts=list(ts_aligned),
        cog_data=cog_table,
        metadata=metadata,
        matched_ids=list(matched_ids),
        region_labels=region_labels_clean,
        paths=paths,
        filter_mode=effective_filter_mode,
        only_tr=only_tr,
        dataset_name=dataset_name,
    )


def _stack_time_series(ts_list: Sequence[np.ndarray]) -> Tuple[np.ndarray, bool]:
    first_shape = ts_list[0].shape
    if all(ts.shape == first_shape for ts in ts_list):
        return np.stack(ts_list), True
    return np.array(ts_list, dtype=object), False


def write_outputs(result: PreprocessResult, *, dry_run: bool) -> None:
    """Persist preprocessing artefacts to the configured output folder."""

    output_dir = Path(result.paths["preprocessed"])  # type: ignore[arg-type]
    if dry_run:
        logging.getLogger(__name__).info("Dry-run enabled; no files written to %s", output_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    n_animals = result.metadata["n_animals"]
    regions = result.metadata["regions"]
    total_tr = result.metadata["total_tr"]

    meta_path = output_dir / f"metadata_animals_{n_animals}_regions_{regions}_tr_{total_tr}.pkl"
    with meta_path.open("wb") as handle:
        pickle.dump(result.metadata, handle)

    stacked_ts, uniform = _stack_time_series(result.ts)
    ts_path = output_dir / f"ts_filtered_animals_{n_animals}_regions_{regions}_tr_{total_tr}.npz"
    shared_payload = {
        "ts": stacked_ts,
        "n_animals": n_animals,
        "total_tr": total_tr,
        "regions": regions,
        "anat_labels": np.asarray(result.region_labels, dtype=str),
        "filter_mode": result.filter_mode,
        "dataset_name": np.asarray(result.dataset_name, dtype=str),
    }
    np.savez(ts_path, **shared_payload)

    cog_path = output_dir / f"cog_data_filtered_animals_{n_animals}_regions_{regions}_tr_{total_tr}.csv"
    result.cog_data.to_csv(cog_path, index=False)

    if uniform:
        canonical_payload = dict(shared_payload)
        canonical_payload["mouse_ids"] = np.asarray(result.matched_ids, dtype=str)

        canonical_name = f"ts_and_meta_{result.dataset_name}.npz"
        np.savez(output_dir / canonical_name, **canonical_payload)

        for name in ("ts_and_meta_julien.npz", "ts_and_meta_2m4m.npz"):
            np.savez(output_dir / name, **canonical_payload)
    else:
        logging.getLogger(__name__).warning(
            "Canonical bundle skipped because time series lengths are inconsistent."
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        result = prepare_dataset(
            filter_mode=args.filter_mode,
            only_tr=args.only_tr,
            dataset_name=args.dataset_name,
            timecourse_folder=args.timecourse_folder,
            cognitive_data_file=args.cognitive_data_file,
            anat_labels_file=args.anat_labels_file,
            cognitive_sheet=args.cognitive_sheet,
        )
    except Exception as exc:  # pragma: no cover - CLI surface
        logger.error("Preprocessing failed: %s", exc)
        return 1

    write_outputs(result, dry_run=args.dry_run)
    logger.info(
        "Preprocessing complete for %d animals (filter=%s). Outputs in %s",
        result.metadata["n_animals"],
        result.filter_mode,
        result.paths["preprocessed"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
