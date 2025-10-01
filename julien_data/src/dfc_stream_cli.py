#!/usr/bin/env python3
"""
Compute DFC streams for the Julien dataset with a simple, predictable CLI.

This avoids the multiple repeated passes present in the legacy script and
computes either:
- one pass using all animals padded to the longest length (mode=all), or
- separate passes per distinct length group (mode=split).

Outputs are identical to those produced by shared_code's get_tenet4window_range
for the same inputs and parameters (n_animals, regions, lag, window range).

Usage examples:
  # Single pass (padded to longest)
  python julien_data/src/dfc_stream_cli.py --mode all --processors -1 --load-cache

  # Split by lengths (e.g., 500 and 400)
  python julien_data/src/dfc_stream_cli.py --mode split --processors -1 --load-cache
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np

from shared_code.fun_dfcspeed import get_tenet4window_range


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def _load_context(tr: int | None = None):
    try:
        from net_fluidity_julien.context import DFCAnalysis
    except ModuleNotFoundError:
        try:
            from julien_data.class_dataanalysis_julien import DFCAnalysis
        except ModuleNotFoundError:
            # Fallback if run from scripts/ or elsewhere
            import sys
            here = Path(__file__).resolve().parent
            cand = here.parent
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            from class_dataanalysis_julien import DFCAnalysis  # type: ignore

    data = DFCAnalysis()
    if tr is None:
        data.get_metadata()          # metadata (n_animals, regions, total_tr, etc.)
    else:
        preproc = Path(data.paths["preprocessed"])  # type: ignore[index]
        cand = sorted(preproc.glob(f"metadata_animals_*_tr_{int(tr)}.pkl"))
        if not cand:
            raise FileNotFoundError(f"No metadata found for tr={tr} under {preproc}")
        data.get_metadata(meta_filename=cand[0].name)
    data.get_ts_preprocessed()   # loads data.ts (list-like)
    data.get_cogdata_preprocessed()
    data.get_temporal_parameters()
    return data


def _build_padded_all(ts_list: List[np.ndarray], max_tp: int, regions: int) -> np.ndarray:
    n = len(ts_list)
    out = np.zeros((n, max_tp, regions), dtype=np.float32)
    for i, ts in enumerate(ts_list):
        t = ts.shape[0]
        out[i, :t, :] = ts
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute DFC streams (simple CLI)")
    p.add_argument("--mode", choices=["all", "split"], default="all", help="all=padded single pass; split=per-length groups")
    p.add_argument("--processors", type=int, default=-1, help="Parallel jobs (-1 = all)")
    p.add_argument("--load-cache", action="store_true", help="Load cached per-window results if present")
    p.add_argument("--tr", type=int, default=None, help="Select metadata by total_tr (e.g., 400 or 500); default uses the first metadata found")
    return p.parse_args()


def main() -> int:
    setup_logging()
    logger = logging.getLogger(__name__)
    args = parse_args()

    data = _load_context(args.tr)
    paths = data.paths
    lag = data.lag
    time_window_range = data.time_window_range
    processors = args.processors
    load_cache = args.load_cache

    # Collect time series by length
    lengths = sorted({ts.shape[0] for ts in data.ts})
    max_tp = max(lengths)
    regions = data.regions

    if args.mode == "all":
        logger.info("Mode=all: padding all animals to longest length (%s)", max_tp)
        ts_all = _build_padded_all(list(data.ts), max_tp, regions)
        get_tenet4window_range(
            ts_all,
            time_window_range,
            prefix="dfc",
            paths=paths,
            lag=lag,
            n_animals=ts_all.shape[0],
            regions=regions,
            processors=processors,
            load_cache=load_cache,
        )
    else:
        logger.info("Mode=split: computing per distinct length group: %s", lengths)
        for L in lengths:
            group = [ts for ts in data.ts if ts.shape[0] == L]
            if not group:
                continue
            ts_arr = np.asarray(group)
            logger.info("Length=%s: n_animals=%s", L, ts_arr.shape[0])
            get_tenet4window_range(
                ts_arr,
                time_window_range,
                prefix="dfc",
                paths=paths,
                lag=lag,
                n_animals=ts_arr.shape[0],
                regions=regions,
                processors=processors,
                load_cache=load_cache,
            )

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
