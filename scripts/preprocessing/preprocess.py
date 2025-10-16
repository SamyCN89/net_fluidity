#!/usr/bin/env python3
"""Unified preprocessing entrypoint for Julien and Ines datasets."""

#%%
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import MutableMapping, Optional, Sequence

if __package__ in (None, ""):
    _PKG_ROOT = Path(__file__).resolve().parent
    if str(_PKG_ROOT) not in sys.path:
        sys.path.insert(0, str(_PKG_ROOT))
    import ines as ines_mod  # type: ignore
    import julien as julien_mod  # type: ignore
else:
    from . import ines as ines_mod
    from . import julien as julien_mod


def _parse_folder_overrides(items: Sequence[str] | None) -> MutableMapping[str, str]:
    return ines_mod.parse_folder_overrides(items or [])


def _canonical_dataset(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("julien"):
        return "julien_caillette"
    if lowered.startswith("ines"):
        return "ines_abdullah"
    raise ValueError(f"Unsupported dataset '{name}'. Expected something like 'julien' or 'ines'.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="julien", help="Dataset to preprocess (julien/ines).")
    parser.add_argument("--dry-run", action="store_true", help="Skip writing artefacts.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )

    julien = parser.add_argument_group("Julien options")
    julien.add_argument(
        "--filter-mode",
        default="exclude_shortest",
        choices=["exclude_shortest", "truncate", "none"],
        help="How to harmonise TS lengths when --only-tr is not provided.",
    )
    julien.add_argument(
        "--only-tr",
        type=int,
        default=None,
        help="Restrict to animals with exactly this TR count (overrides --filter-mode).",
    )
    julien.add_argument(
        "--julien-timecourse-folder",
        default="time_courses_2",
        help="Timecourse folder override for Julien dataset.",
    )
    julien.add_argument(
        "--julien-cognitive-data-file",
        default="mice_groups_comp_index_2.xlsx",
        help="Cognitive Excel file for the Julien dataset.",
    )
    julien.add_argument(
        "--julien-cognitive-sheet",
        default="mice_groups_comp_index",
        help="Sheet inside the Julien cognitive workbook.",
    )
    julien.add_argument(
        "--julien-anat-labels-file",
        default="all_ROI_coimagine_2.txt",
        help="Anatomical labels file for the Julien dataset.",
    )

    ines = parser.add_argument_group("Ines options")
    ines.add_argument(
        "--ines-timecourse-folder",
        default="Timecourses_updated_03052024",
        help="Folder containing Ines timecourses.",
    )
    ines.add_argument(
        "--ines-cognitive-data-file",
        default="ROIs.xlsx",
        help="Excel workbook for Ines cognitive measurements.",
    )
    ines.add_argument(
        "--ines-anat-labels-file",
        default="41_Allen.txt",
        help="Anatomical labels file recorded for Ines runs.",
    )
    ines.add_argument(
        "--ines-folder",
        action="append",
        default=[],
        metavar="PERIOD=FOLDER",
        help="Override 2mois/4mois folders (may repeat).",
    )
    ines.add_argument(
        "--ines-transient",
        type=int,
        default=50,
        help="Number of transient TRs to trim from each timeseries.",
    )
    ines.add_argument(
        "--ines-threshold",
        type=float,
        default=0.2,
        help="Phenotype threshold for Ines classification util.",
    )
    ines.add_argument(
        "--ines-no-extra-groups",
        action="store_true",
        help="Skip writing exploratory grouping_data_new.pkl bundle.",
    )
    # return parser.parse_args(argv)
    return parser.parse_known_args(argv)[0]

#%%
def preprocess_julien(args: argparse.Namespace, dataset_name: str) -> None:
    julien_mod.setup_logging()

    result = julien_mod.prepare_dataset(
        filter_mode=args.filter_mode,
        only_tr=args.only_tr,
        dataset_name=dataset_name,
        timecourse_folder=args.julien_timecourse_folder,
        cognitive_data_file=args.julien_cognitive_data_file,
        anat_labels_file=args.julien_anat_labels_file,
        cognitive_sheet=args.julien_cognitive_sheet,
    )
    julien_mod.write_outputs(result, dry_run=args.dry_run)


def preprocess_ines(args: argparse.Namespace, dataset_name: str) -> None:
    folder_overrides = _parse_folder_overrides(args.ines_folder)
    result = ines_mod.prepare_cognitive_dataset(
        dataset_name=dataset_name,
        timecourse_folder=args.ines_timecourse_folder,
        cognitive_data_file=args.ines_cognitive_data_file,
        anat_labels_file=args.ines_anat_labels_file,
        folders=dict(folder_overrides) if folder_overrides else None,
        transient=args.ines_transient,
        threshold=args.ines_threshold,
    )
    ines_mod.write_outputs(result, dry_run=args.dry_run, write_extra_groups=not args.ines_no_extra_groups)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    try:
        dataset = _canonical_dataset(args.dataset_name)
    except ValueError as exc:
        logging.getLogger(__name__).error(str(exc))
        return 2

    if dataset == "julien_caillette":
        preprocess_julien(args, dataset)
    else:
        preprocess_ines(args, dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# %%
