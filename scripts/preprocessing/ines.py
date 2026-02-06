#!/usr/bin/env python3
"""Preprocessing helpers for the Ines (meta-connectivity) dataset."""
# %%
from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
import logging
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

try:
    from shared_code.fun_loaddata import extract_hash_numbers, load_matdata
    from shared_code.fun_paths import get_paths
    from shared_code.fun_utils import (
        classify_phenotypes,
        filename_sort_mat,
        make_combination_masks,
        make_masks,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - fallback for local runs
    import sys

    shared_root = Path(__file__).resolve().parents[2] / "shared_code"
    if shared_root.exists():
        sys.path.append(str(shared_root))
        from shared_code.fun_loaddata import extract_hash_numbers, load_matdata
        from shared_code.fun_paths import get_paths
        from shared_code.fun_utils import (
            classify_phenotypes,
            filename_sort_mat,
            make_combination_masks,
            make_masks,
        )
    else:
        raise exc

LOGGER = logging.getLogger(__name__)

DEFAULT_FOLDERS = {"2mois": "TC_2months", "4mois": "TC_4months"}
COG_SHEET = "Exclusions"
ANAT_SHEET = "41_Allen"


@dataclass
class GroupingPayload:
    mask_groups: tuple[Sequence[np.ndarray], ...]
    label_groups: tuple[Sequence[str], ...]
    mask_groups_per_sex: tuple[Sequence[np.ndarray], ...]
    label_groups_per_sex: tuple[Sequence[str], ...]
    extra_group_maps: dict[str, dict[tuple[str, ...], np.ndarray]]


@dataclass
class PrepResult:
    ts: np.ndarray
    cog_data: pd.DataFrame
    metadata: dict[str, np.ndarray]
    grouping: GroupingPayload
    paths: Mapping[str, Path]


def _parse_folder_overrides(overrides: Iterable[str]) -> MutableMapping[str, str]:
    folder_map: MutableMapping[str, str] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(
                f"Folder override must be of the form key=value (got {item!r})"
            )
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(
                f"Folder override key/value cannot be empty (got {item!r})"
            )
        folder_map[key] = value
    return folder_map


parse_folder_overrides = _parse_folder_overrides


def _group_indices(
    df: pd.DataFrame, columns: Sequence[str]
) -> dict[tuple[str, ...], np.ndarray]:
    grouped = df.reset_index().groupby(list(columns))["index"]
    mapping: dict[tuple[str, ...], np.ndarray] = {}
    for raw_key, index_series in grouped:
        if isinstance(raw_key, tuple):
            key = tuple(str(value) for value in raw_key)
        else:
            key = (str(raw_key),)
        mapping[key] = index_series.to_numpy(dtype=int)
    return mapping


# Load Ines recordings, align cognition metadata, and prepare grouping masks.
def prepare_cognitive_dataset(
    *,
    dataset_name: str = "ines_abdallah",
    timecourse_folder: str = "Timecourses_updated_03052024",
    cognitive_data_file: str = "ROIs.xlsx",
    anat_labels_file: str | None = "41_Allen.txt",
    folders: Mapping[str, str] | None = None,
    transient: int = 50,
    threshold: float = 0.2,
) -> PrepResult:
    """Build the cognitive preprocessing bundle for the requested dataset."""

    # Resolve dataset-specific folders from shared configuration.
    paths = get_paths(
        dataset_name=dataset_name,
        timecourse_folder=timecourse_folder,
        cognitive_data_file=cognitive_data_file,
        anat_labels_file=anat_labels_file,
    )
    folders_map = dict(DEFAULT_FOLDERS)
    if folders:
        folders_map.update(folders)

    # Load cognitive metadata and anatomical labels.
    LOGGER.info("Loading cognitive tables from %s", paths["cog_data"])
    cog_data_df = pd.read_excel(paths["cog_data"], sheet_name=COG_SHEET)
    data_roi = pd.read_excel(paths["cog_data"], sheet_name=ANAT_SHEET).to_numpy()

    timeseries_root = Path(paths["timeseries"])
    # Collect MAT filenames per age group and keep animals recorded at both ages.
    filenames = {
        period: filename_sort_mat(str(timeseries_root / folder))
        for period, folder in folders_map.items()
    }
    hash_numbers = {
        period: extract_hash_numbers(filenames[period]) for period in filenames
    }
    common_ids, idx_2m, idx_4m = np.intersect1d(
        hash_numbers["2mois"],
        hash_numbers["4mois"],
        return_indices=True,
    )
    if len(common_ids) == 0:
        raise RuntimeError(
            "No intersection between 2m and 4m recordings; cannot proceed."
        )
    LOGGER.info("Found %d animals with both 2m and 4m recordings", len(common_ids))

    # Compute derived cognitive metrics.
    cog_data_df["oip_4m-2m"] = cog_data_df["OiP_4M"] - cog_data_df["OiP_2M"]
    cog_data_df["oip_4m+2m"] = cog_data_df["OiP_4M"] + cog_data_df["OiP_2M"]
    cog_data_df["ro24h_4m-2m"] = cog_data_df["RO24h_4M"] - cog_data_df["RO24h_2M"]
    cog_data_df["ro24h_4m+2m"] = cog_data_df["RO24h_4M"] + cog_data_df["RO24h_2M"]

    # Restrict cognition table to animals with usable recordings at both ages.
    cog_data_filtered = (
        cog_data_df[cog_data_df["Name"].isin(common_ids)].sort_values(by="Name").copy()
    )
    # Keep only animals passing quality control at both ages.
    cog_data_filtered = cog_data_filtered[
        (cog_data_filtered["TC_2M"] == "ok") & (cog_data_filtered["TC_4M"] == "ok")
    ].copy()

    # Load time-series for the filtered set of animals.
    intersection = np.intersect1d(
        common_ids, cog_data_filtered["Name"], return_indices=True
    )
    idx_int_2m = idx_2m[intersection[1]]
    idx_int_4m = idx_4m[intersection[1]]

    files_2m = np.array(filenames["2mois"])[idx_int_2m]
    files_4m = np.array(filenames["4mois"])[idx_int_4m]

    LOGGER.info(
        "Loading %d 2m time-series from %s", len(files_2m), folders_map["2mois"]
    )
    ts_2m = load_matdata(str(timeseries_root), folders_map["2mois"], files_2m)
    LOGGER.info(
        "Loading %d 4m time-series from %s", len(files_4m), folders_map["4mois"]
    )
    ts_4m = load_matdata(str(timeseries_root), folders_map["4mois"], files_4m)

    # Clip transient TRs from the start of each recording.
    ts_2m = ts_2m[:, transient:]
    ts_4m = ts_4m[:, transient:]

    n2, n4 = ts_2m.shape[0], ts_4m.shape[0]
    total_tr, regions = ts_2m.shape[1:]
    n_animals = n2 + n4
    ts = np.empty((n_animals, total_tr, regions), dtype=ts_2m.dtype)
    ts[:n2] = ts_2m
    ts[n2:] = ts_4m

    # Expand mouse IDs to match stacked TS: [2m...][4m...]
    mouse_ids = cog_data_filtered["Name"].astype(str).to_numpy()
    mouse_ids_ts = np.concatenate([mouse_ids, mouse_ids])
    age_ts = np.array(["2m"] * len(mouse_ids) + ["4m"] * len(mouse_ids), dtype=str)

    assert ts.shape[0] == len(
        mouse_ids_ts
    ), f"Mismatch: ts has {ts.shape[0]} rows but expanded mouse_ids has {len(mouse_ids_ts)}"

    # Prepare grouping masks based on cognitive phenotypes.
    anat_labels = [str(entry[1]).replace(".", " ") for entry in data_roi]
    is_2month_old = np.arange(n_animals) < n2

    # Derive phenotype labels used by downstream grouping helpers.
    cog_data_filtered = classify_phenotypes(
        cog_data_filtered, metric_prefix="OiP", threshold=threshold
    )
    cog_data_filtered = classify_phenotypes(
        cog_data_filtered, metric_prefix="RO24h", threshold=threshold
    )

    genotype = cog_data_filtered["Genotype"]
    sex = cog_data_filtered["Sexe"]
    phenotype_oip = cog_data_filtered["Phenotype_OiP"]
    phenotype_nor = cog_data_filtered["Phenotype_RO24h"]

    group_phenotype_oip = (
        phenotype_oip == "good",
        phenotype_oip == "impaired",
        phenotype_oip == "learners",
        phenotype_oip == "bad",
    )
    prelab_phenotype_oip = ("Good", "Impaired", "Learners", "Bad")

    group_phenotype_nor = (
        phenotype_nor == "good",
        phenotype_nor == "impaired",
        phenotype_nor == "learners",
        phenotype_nor == "bad",
    )
    prelab_phenotype_nor = ("Good", "Impaired", "Learners", "Bad")

    group_genotype = (genotype == "wt", genotype == "dKI")
    prelab_genotype = ("wt", "dKI")

    group_sex = (sex == "F", sex == "M")
    prelab_sex = ("Female", "Male")

    mask_groups, label_groups = make_masks(
        [
            (group_phenotype_oip, prelab_phenotype_oip),
            (group_phenotype_nor, prelab_phenotype_nor),
            (group_genotype, prelab_genotype),
            (group_sex, prelab_sex),
        ],
        is_2month_old,
    )


    phenotypes = ["good", "impaired", "learners", "bad"]
    sexes = ["F", "M"]
    mask_combo_oip, label_combo_oip = make_combination_masks(
        cog_data_filtered,
        primary_col="Phenotype_OiP",
        by_col="Sexe",
        primary_levels=phenotypes,
        by_levels=sexes,
        is_2month_old=is_2month_old,
    )
    mask_combo_nor, label_combo_nor = make_combination_masks(
        cog_data_filtered,
        primary_col="Phenotype_RO24h",
        by_col="Sexe",
        primary_levels=phenotypes,
        by_levels=sexes,
        is_2month_old=is_2month_old,
    )
    genotypes = ["wt", "dKI"]
    mask_combo_gen, label_combo_gen = make_combination_masks(
        cog_data_filtered,
        primary_col="Genotype",
        by_col="Sexe",
        primary_levels=genotypes,
        by_levels=sexes,
        is_2month_old=is_2month_old,
    )

    extra_group_maps = {
        "by_sex_genotype": _group_indices(cog_data_filtered, ["Sexe", "Genotype"]),
        "by_sex_phenotype_oip": _group_indices(
            cog_data_filtered, ["Sexe", "Phenotype_OiP"]
        ),
        "by_sex_phenotype_ro24h": _group_indices(
            cog_data_filtered, ["Sexe", "Phenotype_RO24h"]
        ),
    }

    # Expand per-mouse metadata to match stacked TS: [2m...][4m...]
    genotype_mouse = cog_data_filtered["Genotype"].astype(str).to_numpy()
    sex_mouse = cog_data_filtered["Sexe"].astype(str).to_numpy()
    pheno_oip_mouse = cog_data_filtered["Phenotype_OiP"].astype(str).to_numpy()
    pheno_ro_mouse = cog_data_filtered["Phenotype_RO24h"].astype(str).to_numpy()

    genotype_ts = np.concatenate([genotype_mouse, genotype_mouse]).astype(str)
    sex_ts = np.concatenate([sex_mouse, sex_mouse]).astype(str)
    phenotype_oip_ts = np.concatenate([pheno_oip_mouse, pheno_oip_mouse]).astype(str)
    phenotype_ro24h_ts = np.concatenate([pheno_ro_mouse, pheno_ro_mouse]).astype(str)

    grouping = GroupingPayload(
        mask_groups=mask_groups,
        label_groups=label_groups,
        mask_groups_per_sex=(mask_combo_oip, mask_combo_nor, mask_combo_gen),
        label_groups_per_sex=(label_combo_oip, label_combo_nor, label_combo_gen),
        extra_group_maps=extra_group_maps,
    )

    # Store provenance metadata for reproducibility.
    metadata = {
        "n_animals": np.array(n_animals, dtype=np.int32),
        "total_tr": np.array(total_tr, dtype=np.int32),
        "regions": np.array(regions, dtype=np.int32),
        "is_2month_old": is_2month_old.astype(bool),
        "anat_labels": np.asarray(anat_labels, dtype=str),
        "transient": np.array(transient, dtype=np.int32),
        "dataset_name": np.asarray(dataset_name, dtype=str),
        "timecourse_folder": np.asarray(timecourse_folder, dtype=str),
        "mouse_ids": mouse_ids,
        "mouse_ids_ts": mouse_ids_ts,
        "age_ts": age_ts,
        "genotype_ts": genotype_ts,
        "sex_ts": sex_ts,
        "phenotype_oip_ts": phenotype_oip_ts,
        "phenotype_ro24h_ts": phenotype_ro24h_ts,

    }

    return PrepResult(
        ts=ts,
        cog_data=cog_data_filtered.reset_index(drop=True),
        metadata=metadata,
        grouping=grouping,
        paths=paths,
    )


def write_outputs(
    result: PrepResult, *, dry_run: bool = False, write_extra_groups: bool = True
) -> PrepResult:
    """Persist preprocessing artefacts to disk."""

    output_dir = Path(result.paths["preprocessed"])
    if dry_run:
        LOGGER.info("Dry-run enabled; skipping writes to %s", output_dir)
        return result

    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing Ines preprocessing outputs to %s", output_dir)

    # Metadata summary values.
    n_animals = result.metadata["n_animals"]
    regions = result.metadata["regions"]
    total_tr = result.metadata["total_tr"]

    # Cognitive metadata.
    cog_csv = (
        output_dir
        / f"cog_data_filtered_animals_{n_animals}_regions_{regions}_tr_{total_tr}.csv"
    )
    LOGGER.info("Writing cognitive metadata to %s", cog_csv)
    result.cog_data.to_csv(cog_csv, index=False)

    # Time-series bundle.
    ts_npz = output_dir / "ts_and_meta_2m4m.npz"
    LOGGER.info("Writing time-series bundle to %s", ts_npz)
    np.savez(ts_npz, ts=result.ts, **result.metadata)

    # Canonical time-series bundle.
    canonical_npz = (
        output_dir / f"ts_and_meta_{result.metadata['dataset_name'].item()}.npz"
    )
    LOGGER.info("Writing canonical bundle to %s", canonical_npz)
    np.savez(
        canonical_npz,
        ts=result.ts,
        # mouse_ids=r   esult.cog_data["Name"].astype(str).to_numpy(),
        **result.metadata,
    )

    with (output_dir / "grouping_data_oip.pkl").open("wb") as handle:
        pickle.dump((result.grouping.mask_groups, result.grouping.label_groups), handle)

    with (output_dir / "grouping_data_per_sex(gen_phen).pkl").open("wb") as handle:
        pickle.dump(
            (
                result.grouping.mask_groups_per_sex,
                result.grouping.label_groups_per_sex,
            ),
            handle,
        )

    if write_extra_groups:
        with (output_dir / "grouping_data_new.pkl").open("wb") as handle:
            pickle.dump(result.grouping.extra_group_maps, handle)

    # -------------------------
    # B0: groups_table.csv for downstream group-wise MC distribution pipeline
    # -------------------------
    mc_dir = Path(result.paths["mc"])
    dist_dir = mc_dir / "mc_dist"
    dist_dir.mkdir(parents=True, exist_ok=True)

    A = int(result.ts.shape[0])
    df_groups = pd.DataFrame(
        {
            "a": np.arange(A, dtype=int),
            "mouse_id": result.metadata["mouse_ids_ts"].astype(str),
            "age": result.metadata["age_ts"].astype(str),
            "genotype": result.metadata.get("genotype_ts", np.array(["NA"] * A, dtype=str)).astype(str),
            "sex": result.metadata.get("sex_ts", np.array(["NA"] * A, dtype=str)).astype(str),
            "phenotype_oip": result.metadata.get("phenotype_oip_ts", np.array(["NA"] * A, dtype=str)).astype(str),
            "phenotype_ro24h": result.metadata.get("phenotype_ro24h_ts", np.array(["NA"] * A, dtype=str)).astype(str),
        }
    )

    # Default "group" used by pipeline Pathway B (edit as needed)
    df_groups["group"] = (
        "age=" + df_groups["age"]
        + "|geno=" + df_groups["genotype"]
        + "|sex=" + df_groups["sex"]
    )

    out_csv = dist_dir / "groups_table.csv"
    LOGGER.info("Writing groups table to %s", out_csv)
    df_groups.to_csv(out_csv, index=False)

    return result


# Construct a standalone CLI for the Ines preprocessing helper.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-name",
        default="ines_abdallah",
        help=(
            "Dataset key recognised by shared_code.fun_paths.get_paths "
            "(default: ines_abdallah)."
        ),
    )
    parser.add_argument(
        "--timecourse-folder",
        default="Timecourses_updated_03052024",
        help="Folder containing Ines timecourses (default: Timecourses_updated_03052024).",
    )
    parser.add_argument(
        "--cognitive-data-file",
        default="ROIs.xlsx",
        help="Excel workbook for cognitive metrics (default: ROIs.xlsx).",
    )
    parser.add_argument(
        "--anat-labels-file",
        default="41_Allen.txt",
        help="Optional text file name recorded in the bundle manifest (default: 41_Allen.txt).",
    )
    parser.add_argument(
        "--folder",
        action="append",
        default=[],
        metavar="PERIOD=FOLDER",
        help="Override default folder mapping (e.g. --folder 2mois=Lot3_2mois).",
    )
    parser.add_argument(
        "--transient",
        type=int,
        default=50,
        help="Timepoints to drop from the start (default: 50).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Phenotype classification threshold passed to shared_code.fun_utils.classify_phenotypes (default: 0.2).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Assemble artefacts but skip writing to disk.",
    )
    parser.add_argument(
        "--no-extra-groups",
        action="store_true",
        help="Skip writing grouping_data_new.pkl for exploratory group maps.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> PrepResult:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(levelname)s - %(message)s"
    )

    folder_overrides = _parse_folder_overrides(args.folder) if args.folder else None
    result = prepare_cognitive_dataset(
        dataset_name=args.dataset_name,
        timecourse_folder=args.timecourse_folder,
        cognitive_data_file=args.cognitive_data_file,
        anat_labels_file=args.anat_labels_file,
        folders=folder_overrides,
        transient=args.transient,
        threshold=args.threshold,
    )
    write_outputs(
        result, dry_run=args.dry_run, write_extra_groups=not args.no_extra_groups
    )


    LOGGER.info(
        "Preprocessing bundle ready under %s (dry_run=%s)",
        result.paths["preprocessed"],
        args.dry_run,
    )
    return result


if __name__ == "__main__":
    main()

# %%
