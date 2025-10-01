#!/usr/bin/env python3
"""
Compute and persist cohesion summaries (Option 3: Hybrid).

Saves compact arrays for downstream stats and an events table in Parquet:
- time_ratio (A×L): fraction of windows a link is in the same module
- mean_duration, std_duration, burstiness (A×L) from activation events
- events (Parquet): columns [animal, link, onset, offset, duration]

Figures are not produced here. Use cohesion_stats_plot.py for stats and plots.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from shared_code.fun_metaconnectivity import load_merged_allegiance
from shared_code.fun_paths import get_paths


logger = logging.getLogger(__name__)


def setup_logging() -> None:
    if logger.handlers:
        return
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute cohesion summaries and events")
    p.add_argument("--window-size", type=int, default=9, dest="window_size")
    p.add_argument("--lag", type=int, default=1, dest="lag")
    p.add_argument("--tau", type=int, default=3, dest="tau")
    p.add_argument(
        "--timecourse-folder",
        type=str,
        default="Timecourses_updated_03052024",
        dest="timecourse_folder",
    )
    p.add_argument(
        "--dmn-index",
        type=str,
        default="0,23,13,22,2,28,34,37,39,8,35",
        help="comma-separated indices (sorted label space) for DMN; empty string for all regions",
    )
    p.add_argument("--min-duration", type=int, default=2, help="minimum duration for events")
    # Optional visualization of binary ATL (same-module map)
    p.add_argument("--plot-animal", type=int, default=-1, help="plot binary ATL for a single animal index (>=0)")
    p.add_argument("--save-all-binary", action="store_true", help="save binary ATL plots for all animals")
    p.add_argument("--save-plots", action="store_true", help="save plots under fig/<dataset>/cohesion/per_animal/binary")
    p.add_argument("--no-show", action="store_true", help="do not display figures (batch mode)")
    return p.parse_args()


def load_meta(paths: dict) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(paths["preprocessed"] / "ts_and_meta_2m4m.npz", allow_pickle=True)
    ts = data["ts"]
    anat_labels = np.asarray(data["anat_labels"])
    return ts, anat_labels


def reorder_communities(paths: dict, window_size: int, lag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dfc_communities, sort_allegiances, contingency_matrices = load_merged_allegiance(
        paths, window_size=window_size, lag=lag
    )
    dfc_sorted = np.take_along_axis(dfc_communities, sort_allegiances.astype(int), axis=2)
    return dfc_sorted, sort_allegiances, contingency_matrices


def _upper_tri_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    tri = np.triu_indices(n, k=1)
    return tri[0], tri[1]


def compute_time_ratio_and_binary(dfc_sorted: np.ndarray, region_index: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return time_ratio (A×L) and binary ATL (A×T×L)."""
    A, T, N = dfc_sorted.shape
    idx = region_index
    D = len(idx)
    pi, pj = _upper_tri_indices(D)
    L = pi.size

    time_ratio = np.empty((A, L), dtype=float)
    binary_ATL = np.empty((A, T, L), dtype=np.uint8)

    for a in range(A):
        comm = dfc_sorted[a][:, idx]  # (T, D)
        same = (comm[:, pi] == comm[:, pj])  # (T, L)
        binary_ATL[a] = same.astype(np.uint8)
        time_ratio[a] = same.mean(axis=0)
    return time_ratio, binary_ATL


def extract_events_from_binary(binary_ATL: np.ndarray, min_duration: int = 1) -> pd.DataFrame:
    """Binary ATL (A×T×L) → events DataFrame with onset/offset/duration per burst (scan-based)."""
    A, T, L = binary_ATL.shape
    rows = []
    for a in range(A):
        for l in range(L):
            z = binary_ATL[a, :, l]
            t = 0
            while t < T:
                if z[t] == 1:
                    s = t
                    while t < T and z[t] == 1:
                        t += 1
                    e = t
                    dur = e - s
                    if dur >= int(min_duration):
                        rows.append((a, l, s, e, dur))
                else:
                    t += 1
    return pd.DataFrame(rows, columns=["animal", "link", "onset", "offset", "duration"]).reset_index(drop=True)


def duration_summaries(events: pd.DataFrame, n_animals: int, n_links: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = events.groupby(["animal", "link"])['duration'].mean().unstack('link')
    s = events.groupby(["animal", "link"])['duration'].std().unstack('link')
    m = m.reindex(index=range(n_animals), columns=range(n_links))
    s = s.reindex(index=range(n_animals), columns=range(n_links))
    m = m.fillna(0.0).to_numpy()
    s = s.fillna(0.0).to_numpy()
    b = (s - m) / np.maximum(s + m, 1e-9)
    b[m == 0] = 0.0
    return m, s, b


def main() -> int:
    setup_logging()
    args = parse_args()

    # Paths
    paths = get_paths(timecourse_folder=args.timecourse_folder)
    out_dir = (paths["allegiance"] / "cohesion_data").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Figure output directory
    fig_dir = (paths["f_cohesion"] / "per_animal" / "binary").expanduser()
    if args.save_plots:
        fig_dir.mkdir(parents=True, exist_ok=True)

    # Load
    ts, anat_labels = load_meta(paths)
    n_animals = len(ts)
    dfc_sorted, sort_idx, _ = reorder_communities(paths, args.window_size, args.lag)
    anat_labels_sorted = anat_labels[sort_idx[0, 0].astype(int)]
    T = dfc_sorted.shape[1]

    # Region scope
    if args.dmn_index.strip():
        region_index = [int(x) for x in args.dmn_index.split(",") if str(x).strip() != ""]
        scope = "dmn"
    else:
        region_index = list(range(dfc_sorted.shape[2]))
        scope = "all"

    # Compute
    time_ratio, binary_ATL = compute_time_ratio_and_binary(dfc_sorted, region_index)
    events = extract_events_from_binary(binary_ATL, min_duration=args.min_duration)
    mean_dur, std_dur, burst = duration_summaries(events, n_animals=n_animals, n_links=time_ratio.shape[1])

    # Pair labels in selected scope
    D = len(region_index)
    pi, pj = _upper_tri_indices(D)
    pair_labels = np.column_stack([anat_labels_sorted[region_index][pi], anat_labels_sorted[region_index][pj]])

    # Save NPZ
    tag = f"w{args.window_size}_lag{args.lag}_tau{args.tau}_{scope}"
    npz_path = out_dir / f"cohesion_data_{tag}.npz"
    np.savez_compressed(
        npz_path,
        time_ratio=time_ratio.astype(np.float32),
        mean_duration=mean_dur.astype(np.float32),
        std_duration=std_dur.astype(np.float32),
        burstiness=burst.astype(np.float32),
        pair_labels=pair_labels.astype(object),
        anat_labels_sorted=anat_labels_sorted.astype(object),
        n_animals=n_animals,
        n_windows=T,
    )
    logger.info("Saved summaries: %s", npz_path)

    # Preview: events per animal (counts)
    if len(events) > 0:
        counts = events.groupby("animal").size().reindex(range(n_animals), fill_value=0).astype(int)
    else:
        import pandas as _pd
        counts = _pd.Series([0] * n_animals, index=range(n_animals), name="count")
    nonzero = int((counts > 0).sum())
    logger.info(
        "Events summary: total=%d, nonzero_animals=%d/%d, min=%d, median=%d, mean=%.2f, max=%d",
        int(len(events)), nonzero, n_animals, int(counts.min()), int(counts.median()), float(counts.mean()), int(counts.max()),
    )
    logger.info("Events per animal (first 10): %s", counts.head(10).to_list())

    # Save events as Parquet (fallback to CSV if pyarrow missing)
    events_count = int(len(events))
    ev_path_parquet = out_dir / f"events_{tag}.parquet"
    try:
        events.to_parquet(ev_path_parquet, index=False)
        logger.info("Saved %d events (parquet): %s", events_count, ev_path_parquet)
    except Exception as e:
        logger.warning("Parquet unavailable (%s); writing CSV fallback", e)
        ev_path_csv = out_dir / f"events_{tag}.csv"
        events.to_csv(ev_path_csv, index=False)
        logger.info("Saved %d events (csv): %s", events_count, ev_path_csv)

    # Save events-per-animal counts CSV for quick inspection
    counts_path = out_dir / f"events_count_{tag}.csv"
    counts.to_csv(counts_path, header=["count"])  # index=animal id
    logger.info("Saved events-per-animal counts: %s", counts_path)

    # Save manifest
    manifest = {
        "window_size": args.window_size,
        "lag": args.lag,
        "tau": args.tau,
        "dmn_index": region_index,
        "scope": scope,
        "outputs": {
            "npz": str(npz_path),
            "events_parquet": str(ev_path_parquet),
        },
        "events_count": events_count,
        "events_count_csv": str(counts_path),
        "shapes": {
            "time_ratio": list(time_ratio.shape),
            "mean_duration": list(mean_dur.shape),
            "std_duration": list(std_dur.shape),
            "burstiness": list(burst.shape),
        },
    }
    with open(out_dir / f"manifest_{tag}.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved manifest")

    # Optional visualization of binary ATL maps
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        if args.no_show:
            matplotlib.use("Agg", force=True)
    except Exception as e:  # pragma: no cover
        logger.warning("Matplotlib not available for plotting: %s", e)
        plt = None

    def _plot_one(animal_idx: int) -> None:
        if plt is None:
            return
        if not (0 <= animal_idx < n_animals):
            logger.warning("plot-animal index %s out of range [0,%s)", animal_idx, n_animals)
            return
        Z = binary_ATL[animal_idx].T  # (L, T)
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(Z, aspect="auto", interpolation="none", cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"Cohesion binary (same-module=1) — Animal {animal_idx}")
        ax.set_xlabel("Time windows")
        ax.set_ylabel("Links (upper-tri)")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        if args.save_plots:
            fpath = fig_dir / f"binary_cohesion_{tag}_animal{animal_idx:02d}.png"
            fig.savefig(fpath, dpi=300, bbox_inches="tight")
            logger.info("Saved figure: %s", fpath)
        if not args.no_show and not args.save_all_binary:
            plt.show()
        else:
            plt.close(fig)

    if (args.plot_animal is not None and args.plot_animal >= 0) or args.save_all_binary:
        if args.save_all_binary:
            for a in range(n_animals):
                _plot_one(a)
        else:
            _plot_one(int(args.plot_animal))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
