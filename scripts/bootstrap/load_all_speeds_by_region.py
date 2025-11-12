#!/usr/bin/env python3
"""
Load per-animal dFC speed values for ALL regions and ALL time windows (TR=500 by default)
into a single pickle payload, ready for notebooks.

Requirements
- Per-region speed outputs exist under results/<dataset>/speed/regions-<label>/
  (produce them with: python julien_data/src/speed_compute.py --tr 500 --per-region)

Saved structure (pickle):
{
  'tr': int,
  'subset': str | None,
  'tau_index': int | None,
  'group_cols': list[str],
  'groups': dict,          # group_key -> list[animal_idx]
  'regions': list[str],    # region labels as folder name suffixes
  'by_region': {
      '<region-label>': {
          'windows': list[int],
          'per_animal_by_window': list[list[np.ndarray]],
      }, ...
  },
}

Usage
  python scripts/load_all_speeds_by_region.py --tr 500 --subset shared --tau-index 0
  python scripts/load_all_speeds_by_region.py --tr 500 --subset shared --tau-index -1 --out reports/all_regions.pkl
"""
#%%
#%%
from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Robust imports (repo root or scripts/ as CWD)
try:
    from scripts.speed_bootstrap_nb import (
        get_context,
        load_per_animal_from_npz,
        build_groups_from_columns,
    )
except ModuleNotFoundError:
    try:
        from speed_bootstrap_nb import get_context, load_per_animal_from_npz, build_groups_from_columns  # type: ignore
    except ModuleNotFoundError:
        import sys as _sys
        here = Path(__file__).resolve()
        repo_root = here.parents[1]
        if str(repo_root) not in _sys.path:
            _sys.path.insert(0, str(repo_root))
        from scripts.speed_bootstrap_nb import (
            get_context,
            load_per_animal_from_npz,
            build_groups_from_columns,
        )


_WIN_RE = re.compile(r"speed_win(\d+)_.*\.npz$")


def _find_region_dirs(speed_root: Path) -> List[Path]:
    return sorted([p for p in speed_root.iterdir() if p.is_dir() and p.name.startswith("regions-")])
#%%


def _list_window_files(region_dir: Path) -> List[Tuple[int, Path]]:
    files: List[Tuple[int, Path]] = []
    for p in sorted(region_dir.glob("speed_win*_*.npz")):
        m = _WIN_RE.search(p.name)
        if m:
            files.append((int(m.group(1)), p))
    return files


def main() -> None:
    import sys
    ap = argparse.ArgumentParser(description="Load per-animal speeds for all regions and all windows into a pickle payload.")
    ap.add_argument("--tr", type=int, default=500, help="Select metadata by total_tr (e.g., 500).")
    ap.add_argument("--subset", type=str, default='regions', help="Subset subfolder under speed/ (e.g., 'shared').")
    ap.add_argument("--tau-index", type=int, default=0, help="Tau index to select (-1 = pool all taus).")
    ap.add_argument("--group-cols", type=str, default="genotype,treatment", help="Grouping columns, comma-separated.")
    ap.add_argument("--out", type=str, default=None, help="Output pickle path (.pkl). Defaults to reports/speeds_all_regions.pkl")
        # --- robust parsing in notebooks/VSCode/IPython:
    if any(k in sys.modules for k in ("ipykernel", "IPython")):
        args, _unknown = ap.parse_known_args()   # ignore injected flags
    else:
        args = ap.parse_args()

    data = get_context(tr=args.tr)
    speed_root = Path(data.paths["speed"])  # type: ignore[index]
    if args.subset:
        speed_root = speed_root / args.subset

    region_dirs = _find_region_dirs(speed_root)
    if not region_dirs:
        raise FileNotFoundError(
            f"No region folders found under {speed_root}. Run per-region speed compute or choose a different subset."
        )

    ti = None if int(args.tau_index) < 0 else int(args.tau_index)
    group_cols = [s.strip() for s in args.group_cols.split(",") if s.strip()]
    groups = build_groups_from_columns(data.cog_data_filtered, group_cols)

    by_region: Dict[str, Dict[str, object]] = {}
    regions: List[str] = []

    for rdir in region_dirs:
        rlabel = rdir.name.replace("regions-", "")
        print(f"Loading region '{rlabel}' from {rdir} ...")
        win_files = _list_window_files(rdir)
        if not win_files:
            # Skip empty region folder quietly
            continue
        windows: List[int] = []
        per_animal_by_window: List[List[np.ndarray]] = []
        for win, npz in win_files:
            per_animal = load_per_animal_from_npz(npz, tau_index=ti)
            windows.append(int(win))
            per_animal_by_window.append(per_animal)
        by_region[rlabel] = {
            "windows": windows,
            "per_animal_by_window": per_animal_by_window,
        }
        regions.append(rlabel)

    payload = {
        "tr": int(args.tr),
        "subset": args.subset,
        "tau_index": ti,
        "group_cols": group_cols,
        "groups": groups,
        "regions": regions,
        "by_region": by_region,
    }

    out_path = Path(args.out) if args.out else Path("reports") / "speeds_all_regions.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Loaded speeds for {len(regions)} regions from: {speed_root}")
    print(f"Saved payload → {out_path}")


if __name__ == "__main__":
    main()


# %%
# VSCode interactive cell marker
# For interactive use in notebooks/VSCode
# payload['regions']
# payload['by_region']['ACC']['windows']
# payload['by_region']['ACC']['per_animal_by_window'][0][0].shape  # first window, first tau
# payload['by_region']['ACC']['per_animal_by_window'][0][0]  # first window, first tau
# payload['by_region']['ACC']['per_animal_by_window'][0][0][:10]  # first window, first tau, first 10 animals
# len(payload['by_region']['ACC']['per_animal_by_window'][0])  # number of animals
# payload['groups']

#%%

# aux_speed_region = payload['by_region'].keys()

# #Bootstrap example
# from scripts.speed_bootstrap_nb import bootstrap_speed_differences, compute_speed_statistics
# win_idx = aux_speed_region['windows'].index(40)
# per_animal = aux_speed_region['per_animal_by_window'][win_idx]

# per_animal = aux_speed_region['per_animal_by_window'][win_idx]
