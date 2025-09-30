#!/usr/bin/env python3
"""
Created on Mon Oct  2 14:42:38 2023

@author: samy
"""

# %%
import argparse
import logging
import sys
import pickle
from pathlib import Path

from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm

try:
    from julien_data.class_dataanalysis_julien import DFCAnalysis
except ModuleNotFoundError:
    # Fallback when running as a script: import from local folder
    from class_dataanalysis_julien import DFCAnalysis

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

# %%
# Utility function to map region labels to indices
def _map_labels_to_indices(requested_labels, region_labels):
    """Map a list of requested region labels to indices.

    Returns (indices, missing, mapping_dict)
    indices: list[int]
    missing: set[str] labels not found
    mapping_dict: {label: index}
    """
    label2idx = {name: i for i, name in enumerate(region_labels)}
    indices = []
    mapping = {}
    missing = set()
    for lab in requested_labels:
        if lab in label2idx:
            idx = label2idx[lab]
            indices.append(idx)
            mapping[lab] = idx
        else:
            missing.add(lab)
    return indices, missing, mapping

# %%

# Compute the speed of dFC
def dfc_speed_split(
    dfc_stream,
    vstep=1,
    tau_range=0,
    method="pearson",
    return_fc2=False,
    triu_indices=None,
    time_offset=0,
):
    """
    Unified function to calculate the speed of variation in dynamic functional connectivity (dFC).

    ----------
    dfc_stream : numpy.ndarray
        Dynamic functional connectivity stream. Can be either: 2D array (n_pairs, n_frames) 3D array (n_rois, n_rois, n_frames): Full FC matrices over time
    vstep : int, optional
        Time step for computing FC speed (default=1). Must be positive and < n_frames.
    method : str, optional
        Correlation method to use for speed computation (default='pearson').
        Supported methods:
        - 'pearson': Pearson correlation coefficient
        - 'spearman': Spearman rank correlation
        - 'cosine': Cosine similarity
    tril_indices : tuple, optional
        Pre-computed triangular indices for 3D input (default=None).
        If None, will be computed automatically for 3D input.
    return_fc2 : bool, optional
        If True, also return the second FC matrix for each time step (default=False).

    Returns
    -------
    speed_median : float
        Median of the computed speed distribution.
    speeds : numpy.ndarray
        Time series of computed speeds with shape (n_frames - vstep,).
    fc2_stream : numpy.ndarray, optional
        Second FC matrix for each time step. Only returned if return_fc2=True.
        Shape: (n_pairs, n_frames - vstep) for vectorized output.

    References
    ----------
    Dynamic Functional Connectivity as a complex random walk: Definitions and the dFCwalk toolbox
    Lucas Arbabyazd, Diego Lombardo, Olivier Blin, Mira Didic, Demian Battaglia, Viktor Jirsa
    MethodsX 2020, doi: 10.1016/j.mex.2020.101168
    """
    from shared_code.fun_optimization import (
        cosine_speed_vectorized,
        pearson_speed_vectorized,
        spearman_speed,
    )

    # Input validation
    if not isinstance(dfc_stream, np.ndarray):
        raise TypeError("dfc_stream must be a numpy array")

    if dfc_stream.ndim not in [2, 3]:
        raise ValueError(
            "dfc_stream must be 2D (n_pairs, frames) or 3D (roi, roi, frames)"
        )

    if not isinstance(vstep, int) or vstep <= 0:
        raise TypeError("vstep must be a positive integer")

    if method not in ["pearson", "spearman", "cosine"]:
        raise ValueError(
            f"Unsupported method '{method}'. Use 'pearson', 'spearman', or 'cosine'"
        )

    # Handle input format conversion
    # 3D input: (n_rois, n_rois, n_frames)
    if dfc_stream.ndim == 3:
        n_rois = dfc_stream.shape[0]
        n_frames = dfc_stream.shape[2]

        # Generate triangular indices if not provided
        if triu_indices is None:
            triu_indices = np.triu_indices(n_rois, k=1)

        # Extract upper triangular values efficiently
        fc_stream = dfc_stream[triu_indices[0], triu_indices[1], :]
    else:
        # 2D input: (n_pairs, n_frames)
        fc_stream = dfc_stream
        n_frames = fc_stream.shape[1]

    # Validate frame count vs vstep
    if vstep >= n_frames:
        raise ValueError(
            f"vstep ({vstep}) must be less than number of frames ({n_frames})"
        )

    fc1_indices = []
    fc2_indices = []

    # Determine maximum tau shift from provided tau_range
    tau_max = int(np.max(tau_range)) if np.size(tau_range) > 0 else 0
    indices_max = n_frames - (vstep + tau_max + time_offset)
    indices = np.arange(0, indices_max, 1)

    if np.size(tau_range) > 1:
        for tau_aux in tau_range:
            fc1_indices.append(indices[:-1])  # Indices for the first FC matrix
            fc2_indices.append(
                indices[1:] + tau_aux + time_offset + vstep - 1
            )  # Indices for the second FC matrix
            # print(indices[:-1], indices[1:]+tau_aux+time_offset+vstep-1)
    else:
        tau_aux = tau_range
        fc1_indices.append(indices[:-1])
        fc2_indices.append(
            indices[1:] + tau_aux + time_offset + vstep - 1
        )  # Indices for the second FC matrix

    n_speeds = (len(indices) - 1) * np.size(tau_range)
    n_pairs = fc_stream.shape[0]

    # Pre-allocate output arrays for efficiency
    speeds = np.empty((n_speeds, np.size(tau_range)), dtype=np.float32)
    fc2_stream = None

    # Extract FC matrices for vectorized computation
    fc1_matrices = fc_stream[
        :, np.array(fc1_indices).flatten()
    ]  # Shape: (n_pairs, n_speeds)
    fc2_matrices = fc_stream[
        :, np.array(fc2_indices).flatten()
    ]  # Shape: (n_pairs, n_speeds)
    if return_fc2:
        fc2_stream_indices = np.empty(
            n_speeds, dtype=int
        )  # Pre-allocate for second FC matrix indices
        # fc2_stream[:, :] = fc2_matrices
        fc2_stream_indices[:] = (np.array(fc2_indices).flatten()).astype(int)
        return fc2_stream_indices

    # Use optimized speed computation functions for maximum performance
    if method == "pearson":
        speeds = pearson_speed_vectorized(fc1_matrices, fc2_matrices)
    elif method == "spearman":
        speeds = spearman_speed(fc1_matrices, fc2_matrices)
    elif method == "cosine":
        speeds = cosine_speed_vectorized(fc1_matrices, fc2_matrices)
    else:
        raise ValueError(
            f"Unsupported method '{method}'. Use 'pearson', 'spearman', or 'cosine'"
        )

    # Ensure speeds are within valid range [0, 2] for numerical stability
    speeds = np.clip(speeds, 0, 2.0)
    speeds_mat = speeds.reshape(len(tau_range), -1)  # Reshape to (n_pairs, n_speeds)

    return speeds_mat


# %%
# Main analysis function to run DFC speed analysis across multiple window sizes and animals
def run_dfc_speed_analysis(
    data,
    time_window_range,
    tau_range,
    lag,
    save_path,
    n_animals,
    nodes,
    load_cache=False,
    processors=1,
    **kwargs,
):
    """
    DFC speed analysis handler: saves results per animal per tau.
    """

    # Parameter extraction & checks
    min_tau_zero = kwargs.get("min_tau_zero", True)  # default True
    method = kwargs.get("method", "pearson")  # Default 'pearson'
    return_fc2 = kwargs.get(
        "return_fc2", False
    )  # Whether to return the second FC matrix
    prefix = kwargs.get("prefix", "speed")  # Prefix for the DFC speed results

    # Optional: subset speed to selected regions by slicing the pair-vector
    selected_regions = kwargs.get("selected_regions", None)  # e.g., [0, 3, 5]
    region_mode = kwargs.get(
        "region_mode", "touching"
    )  # 'touching' (edges incident to any selected region) or
    #    'within' (both endpoints in selected)
    # Always use dfc_speed_split (multi-tau legacy engine)

    dry_run = kwargs.get("dry_run", False)

    # Derive selected labels (if indices provided)
    selected_labels = None
    if selected_regions is not None:
        try:
            selected_labels = [
                data.region_labels_preprocessed[int(i)] for i in selected_regions
            ]
        except Exception:
            selected_labels = None

    # ----- Function Body for speed computation -----
    # for ws_idx, window_size in tqdm(enumerate(time_window_range), desc=f"Processing animals for window_size "):
    def _sanitize_token(s: str) -> str:
        return (
            str(s)
            .replace("/", "-")
            .replace(" ", "_")
            .replace(",", "-")
            .replace("|", "-")
        )

    def process_window(window_size, regions):

        # Window size loading from dFC data
        dfc_stream = data.load_dfc_1_window(
            lag=lag, window=window_size, regions=regions
        )  # Specify regions here
        # dfc_stream = data.dfc_stream
        logging.getLogger(__name__).info(f"Loaded DFC stream shape: {dfc_stream.shape}")

        # If a region subset is provided, reduce the pair dimension to those edges
        subset_tag = ""
        sel_pairs = None
        sel = None
        labels = None
        if selected_regions is not None and len(selected_regions) > 0:
            # Map region indices to pair indices for the vectorized lower-triangular stream
            all_i, all_j = np.tril_indices(regions, k=-1)
            sel = np.array(selected_regions, dtype=int)
            if region_mode == "within":
                mask = np.isin(all_i, sel) & np.isin(all_j, sel)
            else:  # default: 'touching'
                mask = np.isin(all_i, sel) | np.isin(all_j, sel)
            sel_pairs = np.where(mask)[0]
            if sel_pairs.size == 0:
                logging.getLogger(__name__).warning(
                    f"Selected regions {selected_regions} produced 0 edges for regions={regions}. Skipping subset."
                )
            else:
                dfc_stream = dfc_stream[:, sel_pairs, :]
                # Build a descriptive subset tag for the filename
                try:
                    labels = [data.region_labels_preprocessed[i] for i in sel]
                except Exception:
                    labels = None
                # limit to a few to avoid overly long filenames
                if len(sel) == 1 and labels:
                    tag_core = f"region-{sel[0]}-{_sanitize_token(labels[0])}"
                elif len(sel) <= 5 and labels:
                    tag_core = "lab-" + "-".join(_sanitize_token(l) for l in labels)
                elif len(sel) <= 5:
                    tag_core = "idx-" + "_".join(str(int(x)) for x in sel)
                else:
                    tag_core = f"nregs-{len(sel)}"
                subset_tag = f"_subset_mode-{region_mode}_{tag_core}"

        # Choose subfolder name:
        # 1) If user provided a custom subset name, use it (sanitized)
        subset_name = kwargs.get("subset_name")
        if subset_name:
            subdir = _sanitize_token(subset_name)
        else:
            # 2) Otherwise auto-name based on selection or 'all'
            if sel_pairs is not None and sel_pairs.size > 0:
                if labels and len(labels) <= 5:
                    subdir = "regions-" + "-".join(_sanitize_token(l) for l in labels)
                elif sel is not None and len(sel) <= 5:
                    subdir = "indices-" + "_".join(str(int(x)) for x in sel)
                else:
                    subdir = f"nregs-{len(sel)}"
            else:
                subdir = "all"

        # Ensure subfolder exists
        out_dir = save_path / subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Generate the file name for the current window size data
        window_file = out_dir / (
            f"{prefix}_{'fc_' if return_fc2 else ''}win{window_size}{subset_tag}_tau{np.size(tau_range)}_animals_{n_animals}_regions_{data.regions}.npz"
        )
        # Build a concise run summary
        if sel_pairs is not None and sel_pairs.size > 0:
            edges_selected = int(sel_pairs.size)
            if labels and len(labels) <= 5:
                sel_desc = "labels=" + ",".join(labels)
            elif sel is not None and len(sel) <= 5:
                sel_desc = "indices=" + ",".join(str(int(x)) for x in sel)
            else:
                sel_desc = f"nregs={len(sel) if sel is not None else 'NA'}"
            mode_desc = region_mode
        else:
            edges_selected = int(data.regions * (data.regions - 1) // 2)
            sel_desc = "all"
            mode_desc = "global"

        logger.info(
            f"Run: win={window_size}, mode={mode_desc}, sel={sel_desc}, edges={edges_selected}, out={window_file.name}"
        )
        # If dry-run, only print summary and return planned path
        if dry_run:
            logger.info("DRY RUN → would write: %s", window_file)
            return str(window_file)

        # Initialize lists to store results for each animal
        results = []
        for animal_idx in range(n_animals):
            logging.getLogger(__name__).info(
                f"Processing animal {animal_idx + 1}/{n_animals} for window size {window_size}"
            )
            if load_cache and window_file.exists():
                logging.getLogger(__name__).info(
                    f"Loading cached results for window {window_file}"
                )
                return str(window_file)
            try:
                logging.getLogger(__name__).info(
                    f"Computing for window {window_file} and animal {animal_idx + 1}/{n_animals}"
                )
                if return_fc2:
                    fc2 = dfc_speed_split(
                        dfc_stream[animal_idx],
                        vstep=int(window_size),
                        tau_range=0,
                        method=method,
                        return_fc2=return_fc2,
                    )
                    results.append(fc2)
                    logging.getLogger(__name__).debug(
                        f"Animal {animal_idx} window {window_size}: computed FC2"
                    )
                else:
                    speeds = dfc_speed_split(
                        dfc_stream[animal_idx],
                        vstep=int(window_size),
                        tau_range=tau_range,
                        method=method,
                        return_fc2=return_fc2,
                        time_offset=window_size,
                    )
                    results.append(speeds)
                    logging.getLogger(__name__).debug(
                        f"Animal {animal_idx} window {window_size}: computed speeds"
                    )
            except Exception as e:
                logging.getLogger(__name__).error(
                    f"Error for window {window_file}: {e}"
                )

        # Save the fc2 results if return_fc2 else the speed results
        if return_fc2:
            np.savez_compressed(
                window_file,
                fc2=np.array(
                    results, dtype=object
                ),  # Use object dtype for variable-length arrays
                window_size=window_size,
            )
            logging.getLogger(__name__).info(f"✓ Saved fc2 for window {window_file}")
        else:
            np.savez_compressed(
                window_file,
                speeds=np.array(
                    results, dtype=object
                ),  # Use object dtype for variable-length arrays
                window_size=window_size,
            )
            logging.getLogger(__name__).info(f"✓ Saved speeds for window {window_file}")

        return str(window_file)

    # Use Parallel to handle multiple windows in parallel
    output_files = Parallel(n_jobs=processors, verbose=1)(
        delayed(process_window)(ws, nodes)
        for ws in tqdm(time_window_range, desc="Processing windows for ...")
    )

    if dry_run:
        logging.getLogger(__name__).info("DRY RUN complete. No computation performed.")
    else:
        logging.getLogger(__name__).info("All windows processed successfully.")
    # Final summary of outputs
    output_files = [p for p in output_files if p]
    if output_files:
        logger.info("Saved/updated %d window files under: %s", len(output_files), save_path)
        for p in output_files:
            logger.info("  - %s", p)

        # Skip merge on dry-run
        if dry_run:
            return

        # Consolidate all per-window results into a single artifact (speeds, and fc if available)
        try:
            out_paths = [Path(p) for p in output_files]
            out_dir = out_paths[0].parent

            # Merge speeds
            merged_speeds = []
            for p in out_paths:
                with np.load(p, allow_pickle=True) as arr:
                    if "speeds" in arr:
                        merged_speeds.append(arr["speeds"])  # keep object arrays per window
            if merged_speeds:
                # Build metadata for merged artifact
                from datetime import datetime
                import subprocess
                # Attempt to capture git commit
                commit = None
                try:
                    commit = (
                        subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
                        .stdout.strip()
                        or None
                    )
                except Exception:
                    commit = None
                meta = {
                    "method": method,
                    "region_mode": region_mode,
                    "selected_regions": [int(i) for i in selected_regions]
                    if selected_regions is not None
                    else None,
                    "selected_labels": selected_labels,
                    "subset_name": kwargs.get("subset_name"),
                    "tau_range": [int(x) for x in np.array(tau_range).tolist()],
                    "n_animals": int(n_animals),
                    "regions_in_filenames": int(data.regions),
                    "regions_param": int(nodes),
                    "window_sizes": [int(w) for w in np.array(time_window_range).tolist()],
                    "save_dir": str(out_dir),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "git_commit": commit,
                }
                merged_speed_file = out_dir / (
                    f"{prefix}_windows{len(merged_speeds)}_tau{np.size(tau_range)}_animals_{n_animals}_regions_{data.regions}.pkl"
                )
                with open(merged_speed_file, "wb") as fh:
                    pickle.dump({"speeds": merged_speeds, "meta": meta}, fh)
                logger.info("✓ Merged speeds saved: %s", merged_speed_file)

            # Merge fc2 if available
            if return_fc2:
                merged_fc2 = []
                for p in out_paths:
                    with np.load(p, allow_pickle=True) as arr:
                        if "fc2" in arr:
                            merged_fc2.append(arr["fc2"])  # object arrays per window
                if merged_fc2:
                    merged_fc2_file = out_dir / (
                        f"{prefix}_fc_windows{len(merged_fc2)}_tau{np.size(tau_range)}_animals_{n_animals}_regions_{data.regions}.npz"
                    )
                    # Include metadata alongside fc2; store meta as pickled object array
                    np.savez(
                        merged_fc2_file,
                        fc2=np.array(merged_fc2, dtype=object),
                        meta=np.array(meta, dtype=object),
                        allow_pickle=True,
                    )
                    logger.info("✓ Merged FC2 saved: %s", merged_fc2_file)
            # Final outputs banner
            logger.info("Final outputs:")
            if merged_speeds:
                logger.info("  speeds: %s", merged_speed_file)
            if return_fc2 and merged_fc2:
                logger.info("  fc2:    %s", merged_fc2_file)
        except Exception as e:
            logger.warning("Failed to merge per-window outputs: %s", e)

#%%
# Command-line argument parsing
def _parse_cli_args():
    p = argparse.ArgumentParser(description="Compute dFC speed with region selection")
    p.add_argument(
        "--method",
        default="pearson",
        choices=["pearson", "spearman", "cosine"],
        help="Similarity method",
    )
    p.add_argument(
        "--processors", type=int, default=-1, help="Parallel jobs (-1 = all)"
    )
    p.add_argument(
        "--load-cache",
        action="store_true",
        help="Load cached per-window results if present",
    )
    p.add_argument(
        "--list-regions",
        action="store_true",
        help="List all region labels with indices and exit",
    )
    # Engine removed: always use multi-tau dfc_speed_split
    p.add_argument(
        "--selected-regions",
        type=str,
        default=None,
        help="Comma-separated region indices, e.g. 0,3,7",
    )
    p.add_argument(
        "--selected-region-labels",
        type=str,
        default=None,
        # help="Comma-separated region labels",
        help="Comma-separated region labels",
    )
    p.add_argument(
        "--region-mode",
        default="touching",
        choices=["touching", "within"],
        help="Edge selection mode for region subsets",
    )
    p.add_argument(
        "--per-region",
        action="store_true",
        help="Loop over all regions (one file per region)",
    )
    p.add_argument(
        "--return-fc2",
        action="store_true",
        help="Also save FC2 indices/matrices when supported",
    )
    p.add_argument(
        "--subset-name",
        type=str,
        default=None,
        help="Custom subfolder name under speed/ where outputs are saved (overrides auto-naming)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned output files and subfolder per window, then exit without computing",
    )
    args, _unknown = p.parse_known_args(sys.argv[1:])
    # return p.parse_args()
    return args
#%%


def main():
    # Parse command-line arguments
    args = _parse_cli_args()
    # Print list of parsed arguments
    print(args.__dict__)

    # Load data and parameters
    data = DFCAnalysis()
    data.get_metadata()
    data.get_ts_preprocessed()
    data.get_cogdata_preprocessed()
    data.get_temporal_parameters()

    # Handle --list-regions option
    if args.list_regions:
        # Print index → label mapping and exit
        labels = getattr(data, "region_labels_preprocessed", None)
        if labels is None:
            labels = getattr(data, "region_labels", [])
        for i, name in enumerate(labels or []):
            print(f"{i:02d}: {name}")
        return

    # Extract parameters
    tau = data.tau
    lag = data.lag
    n_animals = data.n_animals
    nodes = data.regions
    save_path = data.paths["speed"]

    min_tau_zero = True
    tau_range = np.arange(0, tau + 1) if min_tau_zero else np.arange(-tau, tau + 1)
    time_window_range = data.time_window_range

    # Resolve selected regions
    selected_regions = None
    if args.selected_region_labels:
        wanted = [s.strip() for s in args.selected_region_labels.split(",") if s.strip()]
        inds, missing, mapping = _map_labels_to_indices(wanted, data.region_labels_preprocessed)
        # Log mapping details
        if mapping:
            logger.info("Label→index mapping: %s", ", ".join(f"{k}:{v}" for k, v in mapping.items()))
        if missing:
            logger.warning("Unknown labels ignored: %s", ", ".join(sorted(missing)))
        selected_regions = inds
        logger.info("Selected region indices: %s", selected_regions)
    elif args.selected_regions:
        selected_regions = [
            int(x) for x in args.selected_regions.split(",") if x.strip()
        ]
        logger.info(f"Selected regions by indices: {selected_regions}")

    # Common kwargs for the analysis function
    common_kwargs = {
        "method": args.method,
        "prefix": "speed",
        "return_fc2": args.return_fc2,
        # "processors": args.processors,
        # "preprocessors": args.processors,
        "selected_regions": selected_regions,
        "region_mode": args.region_mode,
        # no engine parameter – using dfc_speed_split only
        "subset_name": args.subset_name,
        "dry_run": args.dry_run,
    }
    # Run the analysis
    # Run analysis: either per-region or else once for all regions
    if args.per_region:
        for ind_reg in range(nodes):
            per_reg_kwargs = dict(common_kwargs)
            per_reg_kwargs.update(
                {
                    "selected_regions": [ind_reg],
                    "region_mode": "touching",
                }
            )
            logger.info(f"Running per-region analysis for region #{ind_reg}")
            run_dfc_speed_analysis(
                data,
                time_window_range,
                tau_range,
                lag,
                save_path,
                n_animals,
                nodes,
                load_cache=args.load_cache,
                processors=args.processors,
                **per_reg_kwargs,
            )
    else:
        run_dfc_speed_analysis(
            data,
            time_window_range,
            tau_range,
            lag,
            save_path,
            n_animals,
            nodes,
            load_cache=args.load_cache,
            processors=args.processors,
            **common_kwargs,
        )


if __name__ == "__main__":
    main()

# %%
