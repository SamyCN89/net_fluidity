#!/usr/bin/env python3
"""Central CLI to compute dynamic FC speed bundles for canonical datasets.

The script consumes precomputed dFC streams from ``scripts/dfc/dfc_compute.py``
and writes per-window speed artefacts under ``results/<dataset>/speed/``.
It replaces dataset-specific drivers such as ``julien_data/3_dfc_speed_test_v6.py``.

Supports per-region execution via ``--per-region`` to mirror legacy pipelines.
"""

from __future__ import annotations

import argparse
import json
import logging
import textwrap
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from scripts.dfc.dfc_compute import DATASET_DEFAULTS, _canonical_dataset, load_timeseries
except ModuleNotFoundError:  # pragma: no cover - allow standalone execution
    import sys

    ROOT = Path(__file__).resolve().parents[2]
    for candidate in (ROOT, ROOT / "shared_code"):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    from scripts.dfc.dfc_compute import DATASET_DEFAULTS, _canonical_dataset, load_timeseries  # type: ignore

from shared_code.fun_dfcspeed import dfc_speed_split
from shared_code.fun_paths import get_paths

logger = logging.getLogger(__name__)


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawTextHelpFormatter,
):
    """CLI formatter that keeps newlines and shows defaults."""


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


def _selection_descriptor(
    selected_indices: np.ndarray | None,
    selected_labels: Sequence[str],
) -> str:
    if selected_labels:
        sanitized = [_sanitize_token(str(label)) for label in selected_labels]
        if len(sanitized) == 1:
            return f"region-{sanitized[0]}"
        if len(sanitized) <= 5:
            return "regions-" + "-".join(sanitized)
        return f"nregs-{len(sanitized)}"
    if selected_indices is not None:
        preview = "_".join(str(int(i)) for i in selected_indices[:5])
        return f"indices-{preview}" if preview else "indices"
    return "all"


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
    window_step: int,
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
        speeds = dfc_speed_split(
            pair_stream,
            vstep=window_step,
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
    description = textwrap.dedent(
        """\
        Compute dynamic functional connectivity (dFC) speed artefacts from the canonical
        dFC bundles produced by scripts/dfc/dfc_compute.py. The CLI loads the dataset
        bundle, finds the matching dFC NPZ files, and writes per-window speed outputs
        under results/<dataset>/speed/.
        """
    )
    epilog = textwrap.dedent(
        """\
        Example commands
        ----------------
        # Julien dataset, windows 5-25, reuse existing outputs if present
        python scripts/speed/dfc_speed_compute.py --dataset-name julien \\
            --subset-name shared --window-min 5 --window-max 25 --window-step 5 \\
            --lag 1 --tau-range 0,5 --jobs 4

        # Ines dataset, restrict to hippocampal ROIs by label
        python scripts/speed/dfc_speed_compute.py --dataset-name ines \\
            --subset-name hippocampus --region-labels CA1_L,CA1_R,CA3_L,CA3_R \\
            --region-mode touching --tau-max 10

        # Compare windows separated by 10 TRs instead of the default window size
        python scripts/speed/dfc_speed_compute.py --dataset-name julien \\
            --window-min 9 --window-max 9 --lag 1 \\
            --tau-range 0,5 --time-offset 10

        # Dry-run to verify file discovery without computing speeds
        python scripts/speed/dfc_speed_compute.py --dataset-name julien \\
            --window-min 7 --window-max 7 --lag 1 --tau-range 0,5,10 --dry-run
        """
    )
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=_HelpFormatter,
        epilog=epilog,
    )
    parser.add_argument(
        "--dataset-name",
        metavar="NAME",
        default="ines",
        help=(
            "Dataset alias resolved via scripts/dfc/dfc_compute.py "
            "(e.g. 'julien', 'ines')."
        ),
    )
    parser.add_argument(
        "--bundle-name",
        metavar="FILENAME",
        default=None,
        help=(
            "Override the inferred preprocessed bundle (defaults follow the dataset). "
            "Example: --bundle-name ts_and_meta_julien_custom.npz"
        ),
    )
    parser.add_argument(
        "--subset-name",
        metavar="TOKEN",
        default="all",
        help=(
            "Logical bucket under results/<dataset>/speed/ used to group outputs. "
            "Example: --subset-name shared"
        ),
    )
    parser.add_argument(
        "--region-subdir",
        action="store_true",
        help="Organise outputs into region-specific subdirectories within the subset folder.",
    )
    parser.add_argument(
        "--per-region",
        action="store_true",
        help="Iterate over regions and produce one output chunk per ROI.",
    )
    parser.add_argument(
        "--per-region-mode",
        choices=["touching", "within"],
        default=None,
        help="Override edge filtering mode when running with --per-region.",
    )
    parser.add_argument(
        "--window-min",
        type=int,
        default=5,
        metavar="INT",
        help="Smallest window size to process. Example: --window-min 5",
    )
    parser.add_argument(
        "--window-max",
        type=int,
        default=5,
        metavar="INT",
        help="Largest window size to process (inclusive). Example: --window-max 25",
    )
    parser.add_argument(
        "--window-step",
        type=int,
        default=1,
        metavar="INT",
        help="Increment applied between window sizes. Example: --window-step 5",
    )
    parser.add_argument(
        "--lag",
        type=int,
        default=1,
        metavar="INT",
        help=(
            "Lag between sliding windows used during dFC generation. "
            "Must match scripts/dfc/dfc_compute.py."
        ),
    )
    parser.add_argument(
        "--dfc-tau-label",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "Optional tau identifier embedded in dFC filenames. "
            "Example: --dfc-tau-label 5"
        ),
    )
    parser.add_argument(
        "--tau-range",
        default=None,
        metavar="INTS",
        help=(
            "Comma-separated tau offsets (TRs) between FC matrices. "
            "Example: --tau-range 0,5,10"
        ),
    )
    parser.add_argument(
        "--tau-max",
        type=int,
        default=None,
        metavar="INT",
        help="Generate tau offsets [0, ..., N] automatically. Example: --tau-max 10",
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
        metavar="INT",
        help=(
            "Compare FC windows separated by this many TRs instead of the current "
            "window size (default). Example: --time-offset 10"
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="INT",
        help="Worker threads for per-animal computation. Example: --jobs 4",
    )
    parser.add_argument(
        "--region-indices",
        default=None,
        metavar="LIST",
        help=(
            "Comma-separated zero-based ROI indices from the bundle metadata. "
            "Example: --region-indices 1,4,9"
        ),
    )
    parser.add_argument(
        "--region-labels",
        default=None,
        metavar="LIST",
        help=(
            "Comma-separated anatomical labels from the bundle metadata. "
            "Example: --region-labels CA1_L,CA1_R"
        ),
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
        metavar="NAME",
        help="Filename prefix for saved artefacts (default: speed).",
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
        help="How to handle existing outputs: skip them, overwrite, or verify metadata.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity. Example: --log-level DEBUG",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s - %(message)s")
    logger.info("Resolving dataset alias '%s'", args.dataset_name)
    try:
        dataset = _canonical_dataset(args.dataset_name)
    except ValueError as exc:  # align CLI error messaging
        parser.error(str(exc))
    logger.info("Dataset resolved to '%s'", dataset)
    if args.bundle_name:
        logger.info("Using custom bundle name: %s", args.bundle_name)

    cfg = DATASET_DEFAULTS[dataset]
    paths = get_paths(
        dataset_name=dataset,
        timecourse_folder=cfg["timecourse_folder"],
        cognitive_data_file=cfg["cognitive_data_file"],
        anat_labels_file=cfg["anat_labels_file"],
    )
    bundle, dfc_dir = load_timeseries(dataset, args.bundle_name)
    fallback_bundle = Path(paths["preprocessed"]) / (args.bundle_name or cfg["bundle_name"])
    bundle_path = Path(bundle.get("bundle_path", fallback_bundle))
    logger.info("Loaded timeseries bundle from %s", bundle_path)
    anat_labels = [str(x) for x in np.asarray(bundle["anat_labels"]).ravel()]
    n_regions_bundle = int(bundle["regions"])
    mouse_ids = [str(x) for x in np.asarray(bundle.get("mouse_ids", []))]
    ts_obj = bundle.get("ts")
    ts_shape = tuple(ts_obj.shape) if hasattr(ts_obj, "shape") else ()
    raw_animals = bundle.get("n_animals")
    n_animals_bundle = int(raw_animals) if raw_animals is not None else (ts_shape[0] if ts_shape else None)
    raw_tr = bundle.get("total_tr")
    total_tr_bundle = int(raw_tr) if raw_tr is not None else (ts_shape[1] if len(ts_shape) > 1 else None)
    logger.info(
        "Bundle summary: animals=%s, regions=%s, total_tr=%s, ts_shape=%s",
        n_animals_bundle if n_animals_bundle is not None else "?",
        n_regions_bundle,
        total_tr_bundle if total_tr_bundle is not None else "?",
        ts_shape if ts_shape else "?",
    )

    tau_range = _parse_tau_config(args.tau_range, args.tau_max)
    time_offset = int(args.time_offset) if args.time_offset is not None else None
    logger.info("Speed tau offsets: %s", ", ".join(str(x) for x in tau_range))
    if time_offset is None:
        logger.info("Time offset: default to window size.")
    else:
        logger.info("Time offset: using explicit value %s.", time_offset)
    window_step = int(args.lag)
    logger.info("Window step (lag) used for speed comparisons: %s", window_step)

    selected_indices, selected_labels = _parse_region_selection(
        indices=args.region_indices,
        labels=args.region_labels,
        atlas_labels=anat_labels,
    )
    if selected_indices is not None and selected_indices.size > n_regions_bundle:
        parser.error("Selected region indices exceed number of regions in bundle.")

    n_edges_total = n_regions_bundle * (n_regions_bundle - 1) // 2
    if args.per_region:
        logger.info("Per-region mode enabled; generating individual artefacts per ROI.")
        if selected_indices is None:
            logger.info("Regions to iterate: all %s ROIs.", n_regions_bundle)
        else:
            logger.info(
                "Regions to iterate: %s ROIs from the provided selection.",
                selected_indices.size,
            )
        if selected_labels:
            preview = ", ".join(selected_labels[:5])
            if len(selected_labels) > 5:
                preview += ", ..."
            logger.info("Initial selection labels: %s", preview)
    else:
        base_mask = None
        if selected_indices is not None:
            base_mask = _build_pair_mask(n_regions_bundle, selected_indices, args.region_mode)
            if base_mask.size == 0:
                logger.warning("Region selection produced zero edges; outputs will contain empty arrays.")
            selected_edge_count = int(base_mask.sum()) if base_mask.size else 0
            logger.info(
                "Region selection: %s ROIs (%s mode) → %s edges out of %s.",
                selected_indices.size,
                args.region_mode,
                selected_edge_count,
                n_edges_total,
            )
            if selected_labels:
                preview = ", ".join(selected_labels[:5])
                if len(selected_labels) > 5:
                    preview += ", ..."
                logger.info("Selected labels: %s", preview)
        else:
            logger.info("Region selection: all %s regions (%s edges).", n_regions_bundle, n_edges_total)

    dfc_dir = Path(dfc_dir)
    if not dfc_dir.exists():
        parser.error(f"dFC directory not found: {dfc_dir}")
    logger.info("Using dFC inputs from %s", dfc_dir)
    speed_root = Path(paths["speed"])
    subset_dir = speed_root / _sanitize_token(str(args.subset_name or "all"))
    subset_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Speed output root: %s", speed_root)
    logger.info("Subset directory: %s", subset_dir)

    per_region_mode = args.per_region_mode or args.region_mode
    if args.per_region:
        if selected_indices is not None:
            region_iter = [int(i) for i in selected_indices.tolist()]
        else:
            region_iter = list(range(n_regions_bundle))
        selection_specs: list[tuple[np.ndarray | None, list[str], str]] = []
        for idx in region_iter:
            selection_specs.append(
                (np.asarray([idx], dtype=int), [anat_labels[int(idx)]], per_region_mode)
            )
    else:
        selection_specs = [(selected_indices, selected_labels, args.region_mode)]

    selection_contexts: list[dict[str, object]] = []
    total_targets = len(selection_specs)
    for target_idx, (sel_indices, sel_labels, mode) in enumerate(
        selection_specs, start=1
    ):
        labels_list = list(sel_labels)
        descriptor = _selection_descriptor(sel_indices, labels_list)
        if args.per_region:
            logger.info(
                "Per-region target %s/%s: %s", target_idx, total_targets, descriptor
            )

        if sel_indices is not None:
            mask = _build_pair_mask(n_regions_bundle, sel_indices, mode)
            if mask.size == 0:
                logger.warning(
                    "Region selection '%s' produced zero edges; outputs will contain empty arrays.",
                    descriptor,
                )
            edge_count = int(mask.sum()) if mask.size else 0
            logger.info(
                "Region selection '%s': %s ROIs (%s mode) → %s edges out of %s.",
                descriptor,
                sel_indices.size,
                mode,
                edge_count,
                n_edges_total,
            )
            if labels_list:
                preview = ", ".join(labels_list[:5])
                if len(labels_list) > 5:
                    preview += ", ..."
                logger.info("Selection labels '%s': %s", descriptor, preview)
            current_mask: np.ndarray | None = mask
        else:
            current_mask = None
            edge_count = n_edges_total
            logger.info(
                "Region selection '%s': all %s regions (%s edges).",
                descriptor,
                n_regions_bundle,
                n_edges_total,
            )

        if args.region_subdir:
            region_dir = subset_dir / descriptor
        else:
            region_dir = subset_dir
        region_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Outputs for selection '%s' will be written under %s",
            descriptor,
            region_dir,
        )

        selection_contexts.append(
            {
                "indices": sel_indices,
                "labels": labels_list,
                "mode": mode,
                "descriptor": descriptor,
                "pair_mask": current_mask,
                "edge_count": edge_count,
                "region_dir": region_dir,
                "order": target_idx,
                "suffix": f"_{descriptor}" if args.per_region else "",
            }
        )

    windows = np.arange(args.window_min, args.window_max + 1, args.window_step, dtype=int)
    if windows.size == 0:
        parser.error("No window sizes generated; check --window-min/max/step.")
    logger.info("Window sizes to process: %s", ", ".join(str(int(w)) for w in windows))

    # Main processing loop over window sizes
    outputs: list[Path] = []
    for window_size in windows:
        dfc_file = _find_dfc_file(
            dfc_dir,
            window=int(window_size),
            lag=int(args.lag),
            tau_label=args.dfc_tau_label,
        )
        try:
            dfc_rel = dfc_file.relative_to(dfc_dir)
        except ValueError:  # pragma: no cover - fallback for unexpected layouts
            dfc_rel = dfc_file.name
        logger.info("Loading dFC file: %s (relative: %s)", dfc_file, dfc_rel)
        with np.load(dfc_file) as z:
            if "dfc" not in z.files:
                parser.error(f"File {dfc_file} missing 'dfc' array.")
            dfc = z["dfc"]
        logger.info("dFC array shape: %s", dfc.shape)

        if dfc.ndim == 4:
            n_animals, n_regions, _, _ = dfc.shape
        elif dfc.ndim == 3:
            n_animals = dfc.shape[0]
            n_regions = n_regions_bundle
        else:
            parser.error(f"Unexpected dFC array dimensions {dfc.shape} in {dfc_file}")

        tau_range_arr = tau_range.copy()
        offset = time_offset if time_offset is not None else int(window_size)

        for context in selection_contexts:
            context_indices = context["indices"]
            context_labels = context["labels"]
            context_descriptor = context["descriptor"]
            context_mask = context["pair_mask"]
            context_region_dir = context["region_dir"]
            file_suffix = context["suffix"]

            out_name = (
                f"{args.prefix}_win{window_size}_lag{args.lag}_tau{tau_range_arr.size}_"
                f"animals_{n_animals}_regions_{n_regions}{file_suffix}.npz"
            )
            out_path = Path(context_region_dir) / out_name

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
                "Processing window=%s (lag=%s, tau=%s) selection=%s for %s animals → %s",
                window_size,
                args.lag,
                ",".join(map(str, tau_range_arr)),
                context_descriptor,
                n_animals,
                out_path.relative_to(speed_root),
            )

            if args.dry_run:
                logger.info(
                    "[dry-run] Planned output %s", out_path.relative_to(speed_root)
                )
                outputs.append(out_path)
                continue
            mask_arg = context_mask if isinstance(context_mask, np.ndarray) else None
            speeds = _compute_speeds_for_window(
                dfc,
                n_regions=n_regions,
                window_size=int(window_size),
                window_step=window_step,
                tau_range=tau_range_arr,
                method=args.method,
                time_offset=offset,
                jobs=max(1, int(args.jobs)),
                pair_mask=mask_arg,
            )

            selected_indices_payload = (
                context_indices.tolist() if isinstance(context_indices, np.ndarray) else None
            )
            metadata = {
                "dataset": dataset,
                "dfc_file": str(dfc_file.name),
                "bundle": bundle_path.name,
                "bundle_path": str(bundle_path),
                "window_size": int(window_size),
                "lag": int(args.lag),
                "tau_range": [int(x) for x in tau_range_arr.tolist()],
                "method": args.method,
                "time_offset": int(offset),
                "subset": str(args.subset_name or "all"),
                "region_mode": context["mode"],
                "selected_indices": selected_indices_payload,
                "selected_labels": context_labels,
                "selection_descriptor": context_descriptor,
                "per_region": bool(args.per_region),
                "created": datetime.utcnow().isoformat() + "Z",
            }
            if args.per_region:
                metadata["per_region_order"] = int(context["order"])
                metadata["per_region_total"] = total_targets
            if mouse_ids:
                metadata["mouse_ids"] = mouse_ids

            np.savez_compressed(
                out_path,
                speeds=speeds,
                tau_range=tau_range_arr.astype(int),
                metadata=json.dumps(metadata),
            )
            outputs.append(out_path)
            first_speed = speeds[0] if speeds.size and isinstance(speeds[0], np.ndarray) else None
            tau_dim = first_speed.shape[0] if first_speed is not None and first_speed.ndim >= 1 else tau_range_arr.size
            edge_dim = first_speed.shape[1] if first_speed is not None and first_speed.ndim >= 2 else 0
            logger.info(
                "Saved %s (relative=%s, animals=%s, tau=%s, edges=%s)",
                out_path,
                out_path.relative_to(speed_root),
                speeds.size,
                tau_dim,
                edge_dim,
            )

    if outputs:
        header = "Planned speed artefacts:" if args.dry_run else "Speed artefacts processed:"
        logger.info(header)
        for path in outputs:
            status = "exists" if path.exists() else "planned"
            logger.info("  %s [%s]", path, status)

    if args.dry_run:
        logger.info("Dry run complete; planned %d outputs.", len(outputs))
    else:
        logger.info("Finished speed computation for %d windows.", len(outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
