#!/usr/bin/env python3
"""
Compare dFC speed NPZ outputs between two subfolders under paths['speed'].

Use this to verify parity between the original script and the new wrapper:

  # produce two runs with different subset names
  python julien_data/3_dfc_speed_test_v6.py --subset-name orig [args]
  python julien_data/src/speed_compute.py   --subset-name wrap [args]

  # then compare for a specific window (or default to last)
  python scripts/compare_speed_outputs.py --subset-a orig --subset-b wrap --window-size 9

Exit code 0 means arrays match within tolerance and NaN masks are identical.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np


def _load_paths_and_meta():
    try:
        from julien_data.class_dataanalysis_julien import DFCAnalysis
    except ModuleNotFoundError:
        # Fallback: try relative import if running from scripts/
        import sys
        here = Path(__file__).resolve().parent
        cand = here.parent / "julien_data"
        if str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
        from class_dataanalysis_julien import DFCAnalysis  # type: ignore

    data = DFCAnalysis()
    data.load_preprocessed_data()
    data.get_temporal_parameters()
    return data


def _find_npz(base: Path, window: int) -> Optional[Path]:
    # prefer non-FC2 files (speed_ not speed_fc_)
    cands = sorted((p for p in base.glob(f"speed_win{window}_*.npz") if p.name.startswith("speed_win")))
    if not cands:
        # Also allow subpatterns created by test_v6 (with extra tags)
        cands = sorted(base.glob(f"speed_win{window}_*.npz"))
    return cands[-1] if cands else None


def compare_npz(path_a: Path, path_b: Path, verbose: bool = False) -> bool:
    npz_a = np.load(path_a, allow_pickle=True)
    npz_b = np.load(path_b, allow_pickle=True)

    if "speeds" not in npz_a or "speeds" not in npz_b:
        raise KeyError("Missing 'speeds' key in one of the NPZ files")

    A = npz_a["speeds"]
    B = npz_b["speeds"]

    if A.shape != B.shape:
        if verbose:
            print("Shape mismatch:", A.shape, "!=", B.shape)
        return False

    # Speeds can be object arrays (per-animal variable lengths), handle robustly
    equal = True
    n_animals = A.shape[0]
    for a in range(n_animals):
        arr_a = np.asarray(A[a], dtype=float)
        arr_b = np.asarray(B[a], dtype=float)
        if arr_a.shape != arr_b.shape:
            if verbose:
                print(f"animal {a}: shape mismatch {arr_a.shape} != {arr_b.shape}")
            equal = False
            continue
        # NaN mask must match
        mask_a = np.isnan(arr_a)
        mask_b = np.isnan(arr_b)
        if not np.array_equal(mask_a, mask_b):
            if verbose:
                diffs = np.count_nonzero(mask_a != mask_b)
                print(f"animal {a}: NaN mask differs at {diffs} positions")
            equal = False
            continue
        # Compare finite entries
        both_finite = ~(mask_a | mask_b)
        if both_finite.any():
            if not np.allclose(arr_a[both_finite], arr_b[both_finite], rtol=1e-6, atol=1e-8):
                if verbose:
                    max_abs = np.max(np.abs(arr_a[both_finite] - arr_b[both_finite]))
                    print(f"animal {a}: values differ (max abs diff={max_abs:.3e})")
                equal = False
    return equal


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare dFC speed NPZ files between two subfolders")
    ap.add_argument("--subset-a", required=True, help="First subfolder under speed/")
    ap.add_argument("--subset-b", required=True, help="Second subfolder under speed/")
    ap.add_argument("--window-size", type=int, default=None, help="Window size to compare (defaults to last)")
    ap.add_argument("--verbose", action="store_true", help="Verbose diff output")
    args = ap.parse_args()

    data = _load_paths_and_meta()
    root = data.paths["speed"]
    window = int(args.window_size) if args.window_size is not None else int(data.time_window_range[-1])

    dir_a = Path(root) / args.subset_a
    dir_b = Path(root) / args.subset_b
    if not dir_a.exists() or not dir_b.exists():
        print("Missing subset directories:", dir_a, dir_b)
        return 2

    f_a = _find_npz(dir_a, window)
    f_b = _find_npz(dir_b, window)
    if not f_a or not f_b:
        print("NPZ not found for window:", window)
        print("subset-a candidates:", list(dir_a.glob(f"speed_win{window}_*.npz")))
        print("subset-b candidates:", list(dir_b.glob(f"speed_win{window}_*.npz")))
        return 2

    ok = compare_npz(f_a, f_b, verbose=args.verbose)
    if ok:
        print("Match:", f_a.name, "==", f_b.name)
        return 0
    else:
        print("Mismatch:", f_a.name, "!=", f_b.name)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

