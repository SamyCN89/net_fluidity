#!/usr/bin/env python3
"""
Compute DFC streams for all animals, similar to julien_data/2_compute_dfc_stream.py
but without splitting by timepoint length. Saves one file per window size with tau
embedded in the filename as requested.
"""

#%%
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from shared_code.fun_dfcspeed import ts2dfc_stream
from shared_code.fun_paths import get_paths
from shared_code.fun_utils import load_timeseries_data


def main() -> None:
    # Configuration (defaults; can be overridden by CLI)
    WINDOW_PARAM = (5, 100, 1)
    LAG = 1
    TAU = 5  # not used in DFC computation, included in filename per request
    FORMAT = "3D"  # '2D' or '3D'
    CACHE_BEHAVIOR = "skip"  # 'skip'|'load'|'verify'|'overwrite'

    # CLI arguments
    parser = argparse.ArgumentParser(description="Compute DFC streams and save to disk.")
    parser.add_argument("--format", "-f", choices=["2D", "3D"], default=FORMAT,
                        help="Output format: '2D' vectorized lower triangle or '3D' full FC matrices.")
    parser.add_argument("--cache", "-c", choices=["skip", "load", "verify", "overwrite"], default=CACHE_BEHAVIOR,
                        help="Cache behavior for existing output files.")
    parser.add_argument("--wmin", type=int, default=WINDOW_PARAM[0], help="Minimum window size.")
    parser.add_argument("--wmax", type=int, default=WINDOW_PARAM[1], help="Maximum window size.")
    parser.add_argument("--wstep", type=int, default=WINDOW_PARAM[2], help="Window step size.")
    parser.add_argument("--lag", type=int, default=LAG, help="Lag between windows.")
    parser.add_argument("--tau", type=int, default=TAU, help="Tau value to embed in output filenames.")
    args = parser.parse_args()

    # Apply CLI overrides
    WINDOW_PARAM = (args.wmin, args.wmax, args.wstep)
    LAG = args.lag
    TAU = args.tau
    FORMAT = args.format
    CACHE_BEHAVIOR = args.cache

    # Paths and data
    paths = get_paths(
        dataset_name="ines_abdullah",
        timecourse_folder="Timecourses_updated_03052024",
        cognitive_data_file="ROIs.xlsx",
    )
    data_ts = load_timeseries_data(paths["preprocessed"] / "ts_and_meta_2m4m.npz")
    ts = data_ts["ts"]  # shape: (n_animals, T, N)
    n_animals = int(data_ts["n_animals"]) if "n_animals" in data_ts else ts.shape[0]
    n_regions = int(data_ts["regions"]) if "regions" in data_ts else ts.shape[2]

    win_min, win_max, win_step = WINDOW_PARAM
    time_window_range = np.arange(win_min, win_max + 1, win_step)

    out_dir = paths["dfc"]
    out_dir.mkdir(parents=True, exist_ok=True)

    start_all = time.time()
    # Convenience for expected shape verification
    T = ts.shape[1]
    n_pairs = n_regions * (n_regions - 1) // 2

    for ws in time_window_range:
        # Build filename with tau included
        fname = (
            f"dfc_window_size={ws}_lag={LAG}_tau={TAU}_animals={n_animals}_regions={n_regions}.npz"
        )
        fpath = out_dir / fname

        def expected_shape() -> tuple[int, ...]:
            frames = (T - ws) // LAG + 1
            if FORMAT == "3D":
                return (n_animals, n_regions, n_regions, frames)
            return (n_animals, n_pairs, frames)

        if fpath.exists():
            if CACHE_BEHAVIOR == "skip":
                print(f"[cache] Exists, skipping: {fpath}")
                continue
            if CACHE_BEHAVIOR == "load":
                try:
                    arr = np.load(fpath)["dfc"]
                    print(f"[cache] Loaded {fpath} shape={arr.shape}")
                    continue
                except Exception as e:
                    print(f"[cache] Failed to load {fpath} (reason: {e}). Recomputing...")
            if CACHE_BEHAVIOR == "verify":
                try:
                    arr = np.load(fpath)["dfc"]
                    if tuple(arr.shape) == expected_shape():
                        print(f"[cache] Verified {fpath} shape={arr.shape}; skipping")
                        continue
                    else:
                        print(
                            f"[cache] Shape mismatch {arr.shape} != {expected_shape()} for {fpath}; recomputing..."
                        )
                except Exception as e:
                    print(f"[cache] Failed to verify {fpath} (reason: {e}). Recomputing...")
            # CACHE_BEHAVIOR == 'overwrite' will fall through to recompute

        t0 = time.time()
        # Compute DFC stream for each animal
        dfc_list = [
            ts2dfc_stream(ts[i], window_size=ws, lag=LAG, format_data=FORMAT)
            for i in range(n_animals)
        ]
        dfc_arr = np.array(dfc_list, dtype=np.float32)

        np.savez_compressed(fpath, dfc=dfc_arr)
        print(
            f"Saved DFC: {fpath}  shape={dfc_arr.shape}  time={time.time() - t0:.2f}s"
        )

    print(f"Done. Total time: {time.time() - start_all:.2f}s")


if __name__ == "__main__":
    main()

# %%
