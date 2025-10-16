#!/usr/bin/env python3
"""
Load per-animal dFC speed values for all time windows into a single file.

This script scans a speed subset folder, loads every `speed_win*_*.npz` file,
extracts per-animal 1D arrays (optionally selecting a tau index), and saves a
pickled dictionary for convenient use in notebooks.

Saved structure (pickle):
{
  'tr': int,
  'subset': str | None,
  'tau_index': int | None,
  'region_dir': str,
  'windows': list[int],
  'per_animal_by_window': list[list[np.ndarray]],  # aligned with `windows`
  'groups': dict,          # grouping mapping from cog_data (default genotype,treatment)
  'group_cols': list[str],
}

Usage
  python scripts/load_all_speeds.py --tr 500 --subset shared --tau-index 0
  python scripts/load_all_speeds.py --tr 500 --subset shared --tau-index -1 --group-cols genotype,treatment
  python scripts/load_all_speeds.py --tr 500 --subset shared --region "ACC"  # looks for regions-ACC/
"""
#%%
from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Robust imports from scripts/ or repo root
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


def _list_window_files(base: Path) -> List[Tuple[int, Path]]:
    files = []
    for p in sorted(base.glob("speed_win*_*.npz")):
        m = _WIN_RE.search(p.name)
        if m:
            files.append((int(m.group(1)), p))
    return files

#%%
def main():
    ap = argparse.ArgumentParser(description="Load per-animal dFC speeds for all windows and save as pickle.")
    ap.add_argument("--tr", type=int, default=500, help="Select metadata by total_tr (e.g., 500).")
    ap.add_argument("--subset", type=str, default='all', help="Subset subfolder under speed/ (e.g., 'shared').")
    ap.add_argument(
        "--region",
        type=str,
        default=None,
        help="Region label to use (folder 'regions-<label>'). Defaults to 'all' if present, else speed root.",
    )
    ap.add_argument("--tau-index", type=int, default=0, help="Tau index to select (-1 = pool all taus).")
    ap.add_argument("--group-cols", type=str, default="genotype,treatment", help="Grouping columns, comma-separated.")
    ap.add_argument("--out", type=str, default=None, help="Output pickle path (.pkl). Defaults to reports/auto name.")
    # args = ap.parse_args()
    # --- robust parsing in notebooks/VSCode/IPython:
    if any(k in sys.modules for k in ("ipykernel", "IPython")):
        args, _unknown = ap.parse_known_args()   # ignore injected flags
    else:
        args = ap.parse_args()

    data = get_context(tr=args.tr)
    speed_root = Path(data.paths["speed"])  # type: ignore[index]
    if args.subset:
        speed_root = speed_root / args.subset

    # Pick region directory
    if args.region:
        region_dir = speed_root / f"regions-{args.region}"
        if not region_dir.exists():
            raise FileNotFoundError(f"Region folder not found: {region_dir}")
    else:
        default_all = speed_root / "all"
        region_dir = default_all if default_all.exists() else speed_root

    win_files = _list_window_files(region_dir)
    if not win_files:
        raise FileNotFoundError(f"No speed_win*_*.npz files under {region_dir}")

    tau_index = None if int(args.tau_index) < 0 else int(args.tau_index)
    windows, per_animal_by_window = [], []
    for win, npz in win_files:
        per_animal = load_per_animal_from_npz(npz, tau_index=tau_index)
        windows.append(int(win))
        per_animal_by_window.append(per_animal)

    group_cols = [s.strip() for s in args.group_cols.split(",") if s.strip()]
    groups = build_groups_from_columns(data.cog_data_filtered, group_cols)

    payload = {
        "tr": int(args.tr),
        "subset": args.subset,
        "tau_index": tau_index,
        "region_dir": str(region_dir),
        "windows": windows,
        "per_animal_by_window": per_animal_by_window,
        "groups": groups,
        "group_cols": group_cols,
    }

    out_path = (
        Path(args.out)
        if args.out
        else Path("reports") / f"speeds_all_windows_{region_dir.name if region_dir != speed_root else 'all'}.pkl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"Loaded {len(windows)} window files from: {region_dir}")
    print(f"Saved per-animal speeds payload → {out_path}")


main()
# if __name__ == "__main__":
#     main()


# %%
# For interactive use in notebooks/VSCode

# payload['region_dir']
# payload['windows']
# payload['per_animal_by_window'][0][0]  # first window, first animal
# len(payload['per_animal_by_window'][0])  # number of animals
# payload['groups']
# payload['group_cols']
# len(payload['groups']['WT,none'])  # number of animals in group WT, none
# payload['per_animal_by_window'][0][payload['groups']['WT,none'][0]]  # first window, first animal in group WT, none
# payload['per_animal_by_window'][0][payload['groups']['WT,none'][1]]  # first window, second animal in group WT, none
# payload['per_animal_by_window'][0][payload['groups']['WT,none'][2]]  # first window, third animal in group WT, none
# %%


