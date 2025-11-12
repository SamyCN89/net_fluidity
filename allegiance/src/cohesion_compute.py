#!/usr/bin/env python3
"""
Compute and persist cohesion summaries (Option 3: Hybrid).

Saves compact arrays for downstream stats and an events table in Parquet:
- time_ratio (A×L): fraction of windows a link is in the same module
- mean_duration, std_duration, burstiness (A×L) from activation events
- events (Parquet): columns [animal, link, onset, offset, duration]

Figures are not produced here. Use cohesion_stats_plot.py for stats and plots.
"""

# %%
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

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
        "--roi-scope",
        choices=["all", "dmn", "memory"],
        default="all",
        help="ROI scope: 'all' (default), 'dmn' (use --dmn-index), or 'memory' (use --memory-index)",
    )
    p.add_argument(
        "--dmn-index",
        type=str,
        default="0,1,2,3,5,6,10,27,30,31,32",
        # default="0,23,13,22,2,28,34,37,39,8,35",
        help="comma-separated indices (sorted label space) for DMN; used when --roi-scope=dmn",
    )
    p.add_argument(
        "--memory-index",
        type=str,
        default="5,6,7,8,9,10,11,13,14",
        help="comma-separated indices (sorted label space) for memory ROIs; used when --roi-scope=memory",
    )
    p.add_argument(
        "--min-duration", type=int, default=2, help="minimum duration for events"
    )
    # Optional visualization of binary ATL (same-module map)
    p.add_argument(
        "--plot-animal",
        type=int,
        default=-1,
        help="plot binary ATL for a single animal index (>=0)",
    )
    p.add_argument(
        "--save-all-binary",
        action="store_true",
        help="save binary ATL plots for all animals",
    )
    p.add_argument(
        "--save-plots",
        action="store_true",
        help="save plots under fig/<dataset>/cohesion/per_animal/binary",
    )
    p.add_argument(
        "--no-show", action="store_true", help="do not display figures (batch mode)"
    )
    # Unified/simplified options (in addition to legacy ones above)
    p.add_argument(
        "--roi",
        choices=["all", "dmn", "memory", "custom"],
        default="all",
        help="Unified ROI scope selector",
    )
    p.add_argument(
        "--roi-indices",
        type=str,
        default="",
        help="Comma-separated indices in sorted label space (for --roi custom/memory)",
    )
    p.add_argument(
        "--roi-labels",
        type=str,
        default="",
        help="Comma-separated substrings to match ROI labels (sorted space)",
    )
    p.add_argument(
        "--roi-file",
        type=str,
        default="",
        help="Text file with ROI names or indices, one per line",
    )
    p.add_argument(
        "--list-rois",
        action="store_true",
        help="List sorted ROI labels with indices and exit",
    )
    p.add_argument(
        "--emit",
        choices=["all", "npz", "events", "none"],
        default="all",
        help="Which artifacts to write",
    )
    p.add_argument(
        "--tag", type=str, default="", help="Tag to append in output filenames"
    )
    p.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing outputs if present"
    )
    p.add_argument(
        "--plot",
        choices=["none", "one", "all"],
        default="none",
        help="Render binary ATL plots (replaces --plot-animal/--save-all-binary)",
    )
    p.add_argument("--animal", type=int, default=0, help="Animal index for --plot one")
    p.add_argument(
        "--show", action="store_true", help="Display plots (omit to run headless)"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Print resolved configuration and exit"
    )
    p.add_argument(
        "--verbosity",
        choices=["info", "debug"],
        default="info",
        help="Logging verbosity",
    )
    # return p.parse_args()
    return p.parse_known_args()[0]


def load_meta(paths: dict) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(paths["preprocessed"] / "ts_and_meta_2m4m.npz", allow_pickle=True)
    ts = data["ts"]
    anat_labels = np.asarray(data["anat_labels"])
    return ts, anat_labels


def reorder_communities(
    paths: dict, window_size: int, lag: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and reorder allegiance communities. Return (A, T, N), (A, 1, N), (A, N, N)."""
    dfc_communities, sort_allegiances, contingency_matrices = load_merged_allegiance(
        paths, window_size=window_size, lag=lag
    )
    dfc_sorted = np.take_along_axis(
        dfc_communities, sort_allegiances.astype(int), axis=2
    )
    return dfc_sorted, sort_allegiances, contingency_matrices


def _upper_tri_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    tri = np.triu_indices(n, k=1)
    return tri[0], tri[1]


def compute_time_ratio_and_binary(
    dfc_sorted: np.ndarray, region_index: list[int]
) -> tuple[np.ndarray, np.ndarray]:
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
        same = comm[:, pi] == comm[:, pj]  # (T, L)
        binary_ATL[a] = same.astype(np.uint8)
        time_ratio[a] = same.mean(axis=0)
    return time_ratio, binary_ATL


def extract_events_from_binary(
    binary_ATL: np.ndarray, min_duration: int = 1
) -> pd.DataFrame:
    """Vectorized diff-based event extraction.

    Binary ATL (A×T×L) → events DataFrame with onset/offset/duration per burst.
    Matches the scan-based semantics: onset inclusive, offset exclusive, duration = offset - onset.
    """
    A, T, L = binary_ATL.shape
    X = binary_ATL.astype(np.int8, copy=False)
    z = np.zeros((A, 1, L), dtype=np.int8)
    d = np.diff(np.concatenate((z, X, z), axis=1), axis=1)
    on_idx = np.argwhere(d == 1)
    off_idx = np.argwhere(d == -1)

    on = pd.DataFrame(on_idx, columns=["animal", "time", "link"]).sort_values(
        ["animal", "link", "time"]
    )
    off = pd.DataFrame(off_idx, columns=["animal", "time", "link"]).sort_values(
        ["animal", "link", "time"]
    )
    # Pair on/off per (animal, link)
    on["idx"] = on.groupby(["animal", "link"]).cumcount()
    off["idx"] = off.groupby(["animal", "link"]).cumcount()
    ev = (
        on.merge(off, on=["animal", "link", "idx"], suffixes=("_on", "_off"))[
            ["animal", "link", "time_on", "time_off"]
        ]
        .rename(columns={"time_on": "onset", "time_off": "offset"})
        .sort_values(["animal", "link", "onset"])  # stable ordering
        .reset_index(drop=True)
    )
    ev["duration"] = (ev["offset"] - ev["onset"]).astype(int)
    if min_duration > 1:
        ev = ev[ev["duration"] >= int(min_duration)].reset_index(drop=True)
    return ev


def duration_summaries(
    events: pd.DataFrame, n_animals: int, n_links: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = events.groupby(["animal", "link"])["duration"].mean().unstack("link")
    s = events.groupby(["animal", "link"])["duration"].std().unstack("link")
    m = m.reindex(index=range(n_animals), columns=range(n_links))
    s = s.reindex(index=range(n_animals), columns=range(n_links))
    m = m.fillna(0.0).to_numpy()
    s = s.fillna(0.0).to_numpy()
    b = (s - m) / np.maximum(s + m, 1e-9)
    b[m == 0] = 0.0
    return m, s, b


# %%


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
    # anat_labels_sorted = anat_labels[sort_idx[0, 0].astype(int)].astype(str)
    anat_labels_sorted = anat_labels

    T = dfc_sorted.shape[1]

    # Option: list ROIs and exit
    if args.list_rois:
        print(f"Sorted ROI labels (N={len(anat_labels_sorted)}):")
        for i, lab in enumerate(anat_labels_sorted):
            print(f"{i:3d}: {lab}")
        return 0

    # Region index for cohesion based on scope (supports unified and legacy styles)
    N = dfc_sorted.shape[2]

    def _parse_indices(s: str) -> list[int]:
        return [int(x) for x in s.split(",") if str(x).strip() != ""]

    def _indices_from_labels(spec: str) -> list[int]:
        if not spec.strip():
            return []
        toks = [t.strip() for t in spec.split(",") if t.strip()]
        out: list[int] = []
        for tok in toks:
            for i, name in enumerate(anat_labels_sorted):
                if tok.lower() in str(name).lower():
                    out.append(i)
        return sorted(set(out))

    def _indices_from_file(path: str) -> list[int]:
        try:
            lines = Path(path).read_text().splitlines()
        except Exception as e:
            logger.error("Failed to read --roi-file %s: %s", path, e)
            return []
        out: list[int] = []
        for ln in lines:
            t = ln.strip()
            if not t:
                continue
            if t.isdigit():
                out.append(int(t))
            else:
                for i, name in enumerate(anat_labels_sorted):
                    if t.lower() in str(name).lower():
                        out.append(i)
        return sorted(set(out))

    # Default: use legacy roi-scope if unified isn't specified beyond defaults
    use_unified = (args.roi != "all") or bool(
        args.roi_indices.strip() or args.roi_labels.strip() or args.roi_file.strip()
    )
    if use_unified:
        if args.roi == "all":
            region_index = list(range(N))
            scope = "all"
        elif args.roi == "dmn":
            idx = (
                _parse_indices(args.roi_indices)
                if args.roi_indices.strip()
                else (
                    _parse_indices(args.dmn_index)
                    if args.dmn_index.strip()
                    else _parse_indices("0,23,13,22,2,28,34,37,39,8,35")
                )
            )
            region_index = idx
            scope = "dmn"
        elif args.roi in {"memory", "custom"}:
            idx: list[int] = []
            if args.roi_indices.strip():
                idx += _parse_indices(args.roi_indices)
            if args.roi_labels.strip():
                idx += _indices_from_labels(args.roi_labels)
            if args.roi_file.strip():
                idx += _indices_from_file(args.roi_file)
            region_index = sorted(set(idx))
            scope = "memory" if args.roi == "memory" else "custom"
            if not region_index:
                logger.error(
                    "%s scope requires ROI indices/labels/file. Use --list-rois to inspect labels.",
                    scope,
                )
                return 2
        else:
            region_index = list(range(N))
            scope = "all"
    else:
        if args.roi_scope == "all":
            region_index = list(range(N))
            scope = "all"
        elif args.roi_scope == "dmn":
            if args.dmn_index.strip():
                region_index = _parse_indices(args.dmn_index)
            else:
                logger.error(
                    "--roi-scope=dmn requires --dmn-index (comma-separated indices)"
                )
                return 2
            scope = "dmn"
        else:  # memory
            if args.memory_index.strip():
                region_index = _parse_indices(args.memory_index)
            else:
                preview = "\n".join(
                    [f"{i:3d}: {lab}" for i, lab in enumerate(anat_labels_sorted)]
                )
                logger.error(
                    "--roi-scope=memory requires --memory-index. Sorted labels with indices:\n%s",
                    preview,
                )
                return 2
            scope = "memory"

    # Logging verbosity and dry-run
    logging.getLogger().setLevel(
        logging.DEBUG if args.verbosity == "debug" else logging.INFO
    )
    if args.dry_run:
        logger.info(
            "dry-run: ws/lag/tau=%s/%s/%s, scope=%s D=%s emit=%s tag=%s overwrite=%s",
            args.window_size,
            args.lag,
            args.tau,
            scope,
            len(region_index),
            args.emit,
            args.tag,
            args.overwrite,
        )
        return 0

    # Compute time ratio and binary ATL
    time_ratio, binary_ATL = compute_time_ratio_and_binary(dfc_sorted, region_index)
    events = extract_events_from_binary(binary_ATL, min_duration=args.min_duration)
    mean_dur, std_dur, burst = duration_summaries(
        events, n_animals=n_animals, n_links=time_ratio.shape[1]
    )

    # Pair labels in selected scope
    D = len(region_index)
    pi, pj = _upper_tri_indices(D)
    pair_labels = np.column_stack(
        [anat_labels_sorted[region_index][pi], anat_labels_sorted[region_index][pj]]
    )

    # Save NPZ
    tag_parts = [f"w{args.window_size}_lag{args.lag}_tau{args.tau}", scope]
    if args.tag.strip():
        tag_parts.append(args.tag.strip())
    tag = "_".join(tag_parts)
    npz_path = out_dir / f"cohesion_data_{tag}.npz"
    if args.emit in {"all", "npz"}:
        if npz_path.exists() and not args.overwrite:
            logger.info("NPZ exists; use --overwrite to replace: %s", npz_path)
        else:
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
        counts = (
            events.groupby("animal")
            .size()
            .reindex(range(n_animals), fill_value=0)
            .astype(int)
        )
    else:
        import pandas as _pd

        counts = _pd.Series([0] * n_animals, index=range(n_animals), name="count")
    nonzero = int((counts > 0).sum())
    logger.info(
        "Events summary: total=%d, nonzero_animals=%d/%d, min=%d, median=%d, mean=%.2f, max=%d",
        int(len(events)),
        nonzero,
        n_animals,
        int(counts.min()),
        int(counts.median()),
        float(counts.mean()),
        int(counts.max()),
    )
    logger.info("Events per animal (first 10): %s", counts.head(10).to_list())

    # Save events as Parquet (fallback to CSV if pyarrow missing)
    events_count = int(len(events))
    if args.emit in {"all", "events"}:
        ev_path_parquet = out_dir / f"events_{tag}.parquet"
        if ev_path_parquet.exists() and not args.overwrite:
            logger.info(
                "Events parquet exists; use --overwrite to replace: %s", ev_path_parquet
            )
        else:
            try:
                events.to_parquet(ev_path_parquet, index=False)
                logger.info(
                    "Saved %d events (parquet): %s", events_count, ev_path_parquet
                )
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
        "roi_scope": args.roi_scope,
        "roi_index": region_index,
        # Back-compat fields
        "dmn_index": region_index if scope == "dmn" else [],
        "scope": scope,
        "outputs": {
            "npz": str(npz_path) if (args.emit in {"all", "npz"}) else None,
            "events_parquet": (
                str(out_dir / f"events_{tag}.parquet")
                if (args.emit in {"all", "events"})
                else None
            ),
        },
        "events_count": events_count,
        "events_count_csv": (
            str(out_dir / f"events_count_{tag}.csv")
            if (args.emit in {"all", "events"})
            else None
        ),
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

        if not args.show:
            matplotlib.use("Agg", force=True)
    except Exception as e:  # pragma: no cover
        logger.warning("Matplotlib not available for plotting: %s", e)
        plt = None

    def _plot_one(animal_idx: int) -> None:
        if plt is None:
            return
        if not (0 <= animal_idx < n_animals):
            logger.warning("animal index %s out of range [0,%s)", animal_idx, n_animals)
            return
        Z = binary_ATL[animal_idx].T  # (L, T)
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(
            Z, aspect="auto", interpolation="none", cmap="gray_r", vmin=0, vmax=1
        )
        ax.set_title(f"Cohesion binary (same-module=1) — Animal {animal_idx}")
        ax.set_xlabel("Time windows")
        ax.set_ylabel("Links (upper-tri)")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        if args.save_plots:
            fpath = fig_dir / f"binary_cohesion_{tag}_animal{animal_idx:02d}.png"
            fig.savefig(fpath, dpi=300, bbox_inches="tight")
            logger.info("Saved figure: %s", fpath)
        if args.show and args.plot == "one":
            plt.show()
        else:
            plt.close(fig)

    # Back-compat mapping of old plotting flags
    if args.plot == "none":
        if (
            getattr(args, "plot_animal", -1) is not None
            and getattr(args, "plot_animal", -1) >= 0
        ):
            args.plot = "one"
            args.animal = int(args.plot_animal)
        elif getattr(args, "save_all_binary", False):
            args.plot = "all"

    if args.plot == "all":
        for a in range(n_animals):
            _plot_one(a)
    elif args.plot == "one":
        _plot_one(int(args.animal))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# %%
