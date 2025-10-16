#!/usr/bin/env python3
"""Central CLI to compute dynamic FC speed bundles for canonical datasets.

The script consumes precomputed dFC streams from ``scripts/dfc/dfc_compute.py``
and writes per-window speed artefacts under ``results/<dataset>/speed/``.
It replaces dataset-specific drivers such as ``julien_data/3_dfc_speed_test_v6.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from scripts.dfc.dfc_compute import DATASET_DEFAULTS, _canonical_dataset
except ModuleNotFoundError:  # pragma: no cover - allow standalone execution
    import sys

    ROOT = Path(__file__).resolve().parents[2]
    for candidate in (ROOT, ROOT / "shared_code"):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    from scripts.dfc.dfc_compute import DATASET_DEFAULTS, _canonical_dataset  # type: ignore

from shared_code.fun_dfcspeed import dfc_speed_multi_tau
from shared_code.fun_paths import get_paths
from shared_code.fun_utils import load_timeseries_data

logger = logging.getLogger(__name__)


def _sanitize_token(text: str) -> str:
    return (
        text.replace(" ", "_")
        .replace("/", "-")
        .replace(",", "-")
        .replace("|", "-")
    )


def _parse_tau_config(tau_range: str | None, tau_max: int | None) -> np.ndarray:
    if tau_range and tau_max is not None:
        raise ValueError("Specify either --tau-range or --tau-max, not both.")
    if tau_range:
        values = [int(x) for x in tau_range.split(",") if x.strip()]
        if not values:
            raise ValueError("Parsed empty tau-range; provide comma-separated integers.")
        if min(values) < 0:
            raise ValueError("Tau values must be non-negative.")
        return np.asarray(sorted(set(values)), dtype=int)
    if tau_max is not None:
        if tau_max < 0:
            raise ValueError("--tau-max must be non-negative.")
        return np.arange(0, tau_max + 1, dtype=int)
    return np.asarray([0], dtype=int)


def _map_labels_to_indices(requested: Sequence[str], labels: Sequence[str]) -> tuple[list[int], list[str]]:
    label2idx = {str(name): idx for idx, name in enumerate(labels)}
    indices: list[int] = []
    missing: list[str] = []
    for entry in requested:
        key = entry.strip()
        if not key:
            continue
        if key in label2idx:
            indices.append(label2idx[key])
        else:
            missing.append(key)
    return sorted(set(indices)), missing


def _parse_region_selection(
    *,
    indices: str | None,
    labels: str | None,
    atlas_labels: Sequence[str],
) -> tuple[np.ndarray | None, list[str]]:
    idx_list: list[int] = []
    if indices:
        idx_list.extend(int(x) for x in indices.split(",") if x.strip())
    label_names: list[str] = []
    if labels:
        requested = [token.strip() for token in labels.split(",") if token.strip()]
        found, missing = _map_labels_to_indices(requested, atlas_labels)
        if missing:
            logger.warning("Unknown region labels ignored: %s", ", ".join(sorted(missing)))
        idx_list.extend(found)
        label_names.extend(atlas_labels[i] for i in found)
    idx_unique = sorted(set(idx_list))
    if not idx_unique:
        return None, []
    if not label_names:
        label_names = [atlas_labels[i] for i in idx_unique]
    return np.asarray(idx_unique, dtype=int), label_names


def _build_pair_mask(n_regions: int, selected: np.ndarray, mode: str) -> np.ndarray:
    if selected.size == 0:
        return np.array([], dtype=bool)
    lower_i, lower_j = np.tril_indices(n_regions, k=-1)
    if mode == "within":
        mask = np.isin(lower_i, selected) & np.isin(lower_j, selected)
    else:  # touching
        mask = np.isin(lower_i, selected) | np.isin(lower_j, selected)
    return mask


def _prepare_pair_stream(
    dfc_animal: np.ndarray,
    *,
    n_regions: int,
    pair_mask: np.ndarray | None,
) -> np.ndarray:
    if dfc_animal.ndim == 3:
        idx_i, idx_j = np.tril_indices(n_regions, k=-1)
        pair_stream = dfc_animal[idx_i, idx_j, :]
    elif dfc_animal.ndim == 2:
        pair_stream = dfc_animal
    else:
        raise ValueError(f"Unexpected dFC array shape {dfc_animal.shape}")

    if pair_mask is not None:
        if pair_mask.size == 0:
            return np.empty((0, pair_stream.shape[-1]), dtype=pair_stream.dtype)
        return pair_stream[pair_mask, :]
    return pair_stream


def _find_dfc_file(
    dfc_dir: Path,
    *,
    window: int,
    lag: int,
    tau_label: int | None,
) -> Path:
    base = f"dfc_window_size={window}_lag={lag}"
    if tau_label is None:
        for pattern in [
            f"{base}_tau=*.npz",
            f"{base}_*.npz",
            f"{base}.npz",
        ]:
            matches = sorted(dfc_dir.glob(pattern))
            if matches:
                if len(matches) > 1:
                    logger.warning(
                        "Multiple dFC files matched window=%s lag=%s pattern=%s; using last: %s",
                        window,
                        lag,
                        pattern,
                        matches[-1].name,
                    )
                return matches[-1]
        raise FileNotFoundError(
            f"No dFC file found for window={window}, lag={lag} under {dfc_dir}. "
            "Re-run scripts/dfc/dfc_compute.py to generate the required bundles."
        )

    pattern = f"{base}_tau={tau_label}_*.npz"
    matches = sorted(dfc_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No dFC file found for window={window}, lag={lag}, tau={tau_label} under {dfc_dir}. "
            "Use scripts/dfc/dfc_compute.py with matching parameters."
        )
    return matches[-1]


def _compute_speeds_for_window(
    dfc_data: np.ndarray,
    *,
    n_regions: int,
    window_size: int,
    tau_range: np.ndarray,
    method: str,
    time_offset: int,
    jobs: int,
    pair_mask: np.ndarray | None,
) -> np.ndarray:
    n_animals = dfc_data.shape[0]

    def _worker(idx: int) -> np.ndarray:
        pair_stream = _prepare_pair_stream(
            dfc_data[idx],
            n_regions=n_regions,
            pair_mask=pair_mask,
        )
        if pair_stream.shape[0] == 0:
            return np.empty((tau_range.size, 0), dtype=np.float32)
        speeds = dfc_speed_multi_tau(
            pair_stream,
            vstep=window_size,
            tau_range=tau_range,
            method=method,
            time_offset=time_offset,
        )
        return speeds.astype(np.float32, copy=False)

    if jobs > 1:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(_worker, range(n_animals)))
    else:
        results = [_worker(i) for i in range(n_animals)]

    out = np.empty(n_animals, dtype=object)
    for idx, arr in enumerate(results):
        out[idx] = arr
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-name",
        default="ines",
        help="Dataset alias (e.g. 'ines', 'julien').",
    )
    parser.add_argument(
        "--subset-name",
        default="all",
        help="Subfolder under results/.../speed/ (e.g. 'all', 'shared', 'dmn').",
    )
    parser.add_argument(
        "--window-min",
        type=int,
        default=5,
        help="Minimum window size to process.",
    )
    parser.add_argument(
        "--window-max",
        type=int,
        default=5,
        help="Maximum window size to process (inclusive).",
    )
    parser.add_argument(
        "--window-step",
        type=int,
        default=1,
        help="Window size increment.",
    )
    parser.add_argument(
        "--lag",
        type=int,
        default=1,
        help="Lag used when generating dFC streams (used for filename matching).",
    )
    parser.add_argument(
        "--dfc-tau-label",
        type=int,
        default=None,
        help="Tau value embedded in dFC filenames (optional; auto-select when omitted).",
    )
    parser.add_argument(
        "--tau-range",
        default=None,
        help="Comma-separated tau offsets (e.g. '0,1,2'). Mutually exclusive with --tau-max.",
    )
    parser.add_argument(
        "--tau-max",
        type=int,
        default=None,
        help="Generate tau range [0, tau_max].",
    )
    parser.add_argument(
        "--method",
        choices=["pearson", "spearman", "cosine"],
        default="pearson",
        help="Similarity metric for speed computation.",
    )
    parser.add_argument(
        "--time-offset",
        type=int,
        default=None,
        help="Extra offset applied to FC2 indices (defaults to window size).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of worker threads for per-animal computation.",
    )
    parser.add_argument(
        "--region-indices",
        default=None,
        help="Comma-separated ROI indices to include (applies pair filtering).",
    )
    parser.add_argument(
        "--region-labels",
        default=None,
        help="Comma-separated ROI labels to include (uses anatomical labels from bundle).",
    )
    parser.add_argument(
        "--region-mode",
        choices=["touching", "within"],
        default="touching",
        help="Edge filtering: keep edges touching any selected ROI or only those fully within the selection.",
    )
    parser.add_argument(
        "--prefix",
        default="speed",
        help="Filename prefix for saved artefacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the run without computing or writing outputs.",
    )
    parser.add_argument(
        "--cache",
        choices=["skip", "overwrite", "verify"],
        default="skip",
        help="How to handle existing outputs: skip, overwrite, or verify basic metadata.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level))
    try:
        dataset = _canonical_dataset(args.dataset_name)
    except ValueError as exc:  # align CLI error messaging
        parser.error(str(exc))
    cfg = DATASET_DEFAULTS[dataset]
    paths = get_paths(
        dataset_name=dataset,
        timecourse_folder=cfg["timecourse_folder"],
        cognitive_data_file=cfg["cognitive_data_file"],
        anat_labels_file=cfg["anat_labels_file"],
    )

    bundle_path = Path(paths["preprocessed"]) / cfg["bundle_name"]
    if not bundle_path.exists():
        parser.error(f"Preprocessed bundle not found: {bundle_path}")
    bundle = load_timeseries_data(bundle_path)
    anat_labels = [str(x) for x in np.asarray(bundle["anat_labels"]).ravel()]
    n_regions_bundle = int(bundle["regions"])
    mouse_ids = [str(x) for x in np.asarray(bundle.get("mouse_ids", []))]

    tau_range = _parse_tau_config(args.tau_range, args.tau_max)
    time_offset = int(args.time_offset) if args.time_offset is not None else None

    selected_indices, selected_labels = _parse_region_selection(
        indices=args.region_indices,
        labels=args.region_labels,
        atlas_labels=anat_labels,
    )
    pair_mask = None
    if selected_indices is not None:
        if selected_indices.size > n_regions_bundle:
            parser.error("Selected region indices exceed number of regions in bundle.")
        pair_mask = _build_pair_mask(n_regions_bundle, selected_indices, args.region_mode)
        if pair_mask.size == 0:
            logger.warning(
                "Region selection produced zero edges; outputs will contain empty arrays."
            )

    dfc_dir = Path(paths["dfc"])
    if not dfc_dir.exists():
        parser.error(f"dFC directory not found: {dfc_dir}")
    speed_root = Path(paths["speed"])
    subset_dir = speed_root / _sanitize_token(str(args.subset_name or "all"))
    subset_dir.mkdir(parents=True, exist_ok=True)

    if selected_labels:
        if len(selected_labels) == 1:
            region_dir = subset_dir / f"region-{_sanitize_token(selected_labels[0])}"
        elif len(selected_labels) <= 5:
            region_dir = subset_dir / ("regions-" + "-".join(_sanitize_token(s) for s in selected_labels))
        else:
            region_dir = subset_dir / f"nregs-{len(selected_labels)}"
    elif selected_indices is not None:
        region_dir = subset_dir / ("indices-" + "_".join(str(int(i)) for i in selected_indices[:5]))
    else:
        region_dir = subset_dir / "all"
    region_dir.mkdir(parents=True, exist_ok=True)

    windows = np.arange(args.window_min, args.window_max + 1, args.window_step, dtype=int)
    if windows.size == 0:
        parser.error("No window sizes generated; check --window-min/max/step.")

    outputs: list[Path] = []
    for window_size in windows:
        dfc_file = _find_dfc_file(
            dfc_dir,
            window=int(window_size),
            lag=int(args.lag),
            tau_label=args.dfc_tau_label,
        )
        with np.load(dfc_file) as z:
            if "dfc" not in z.files:
                parser.error(f"File {dfc_file} missing 'dfc' array.")
            dfc = z["dfc"]

        if dfc.ndim == 4:
            n_animals, n_regions, _, _ = dfc.shape
        elif dfc.ndim == 3:
            n_animals = dfc.shape[0]
            n_regions = n_regions_bundle
        else:
            parser.error(f"Unexpected dFC array dimensions {dfc.shape} in {dfc_file}")

        tau_range_arr = tau_range.copy()
        offset = time_offset if time_offset is not None else int(window_size)

        out_name = (
            f"{args.prefix}_win{window_size}_lag{args.lag}_tau{tau_range_arr.size}_"
            f"animals_{n_animals}_regions_{n_regions}.npz"
        )
        out_path = region_dir / out_name

        if out_path.exists():
            if args.cache == "skip":
                logger.info("[cache] Skipping existing %s", out_path)
                outputs.append(out_path)
                continue
            if args.cache == "verify":
                logger.info("[cache] Verifying %s", out_path)
                with np.load(out_path, allow_pickle=True) as existing:
                    if "tau_range" in existing and not np.array_equal(existing["tau_range"], tau_range_arr):
                        logger.warning(
                            "[cache] tau_range mismatch for %s; recomputing.",
                            out_path.name,
                        )
                    else:
                        logger.info("[cache] Verified %s; skipping recompute.", out_path.name)
                        outputs.append(out_path)
                        continue

        logger.info(
            "Processing window=%s (lag=%s, tau=%s) for %s animals → %s",
            window_size,
            args.lag,
            ",".join(map(str, tau_range_arr)),
            n_animals,
            out_path.relative_to(speed_root),
        )

        if args.dry_run:
            outputs.append(out_path)
            continue

        speeds = _compute_speeds_for_window(
            dfc,
            n_regions=n_regions,
            window_size=int(window_size),
            tau_range=tau_range_arr,
            method=args.method,
            time_offset=offset,
            jobs=max(1, int(args.jobs)),
            pair_mask=pair_mask,
        )

        metadata = {
            "dataset": dataset,
            "dfc_file": str(dfc_file.name),
            "window_size": int(window_size),
            "lag": int(args.lag),
            "tau_range": [int(x) for x in tau_range_arr.tolist()],
            "method": args.method,
            "time_offset": int(offset),
            "subset": str(args.subset_name or "all"),
            "region_mode": args.region_mode,
            "selected_indices": selected_indices.tolist() if selected_indices is not None else None,
            "selected_labels": selected_labels,
            "created": datetime.utcnow().isoformat() + "Z",
        }
        if mouse_ids:
            metadata["mouse_ids"] = mouse_ids

        np.savez_compressed(
            out_path,
            speeds=speeds,
            tau_range=tau_range_arr.astype(int),
            metadata=json.dumps(metadata),
        )
        outputs.append(out_path)
        logger.info("Saved %s", out_path)

    if args.dry_run:
        logger.info("Dry run complete; planned %d outputs.", len(outputs))
    else:
        logger.info("Finished speed computation for %d windows.", len(outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
