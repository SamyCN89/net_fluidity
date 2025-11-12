#!/usr/bin/env python3
"""
Community-wise dFC speed plots from merged outputs.

Loads the merged speed PKL (as produced by 3_dfc_speed_test_v6.py) and a
communities file (paths['allegiance']/communities_wt_veh.pkl), then creates
one violin plot per community with group distributions.

Usage:
  python julien_data/src/community_speed_plot.py --tr 400 --subset-name all --savefig

Options:
  --tr INT             Select metadata by total_tr (e.g., 400 or 500)
  --subset-name STR    Subfolder under speed/ (e.g., 'all', 'regions-...')
  --pool {all,short,long}
  --savefig            Save figures next to merged PKL
  --no-show            Do not display figures
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def _load_context(tr: int | None = None):
    try:
        from net_fluidity_julien.context import DFCAnalysis
    except ModuleNotFoundError:
        try:
            from julien_data.class_dataanalysis_julien import DFCAnalysis
        except ModuleNotFoundError:
            # Fallback to local path when running as a script
            import sys as _sys
            here = Path(__file__).resolve()
            julien_dir = here.parent.parent  # .../julien_data
            if str(julien_dir) not in _sys.path:
                _sys.path.insert(0, str(julien_dir))
            from class_dataanalysis_julien import DFCAnalysis  # type: ignore
    data = DFCAnalysis()
    if tr is None:
        data.get_metadata()
    else:
        preproc = Path(data.paths["preprocessed"])  # type: ignore[index]
        cands = sorted(preproc.glob(f"metadata_animals_*_tr_{int(tr)}.pkl"))
        if not cands:
            raise FileNotFoundError(f"No metadata file for tr={tr} under {preproc}")
        data.get_metadata(meta_filename=cands[0].name)
    data.get_ts_preprocessed(); data.get_cogdata_preprocessed(); data.get_temporal_parameters()
    return data


def _find_merged(save_root: Path, n_animals: int, regions: int, tau_count: int, subset: str | None) -> Path:
    if subset:
        subdir = save_root / subset
        cands = sorted(subdir.glob(f"speed_windows*_tau{tau_count}_animals_{n_animals}_regions_{regions}.pkl"))
    else:
        cands = sorted(save_root.rglob(f"speed_windows*_tau{tau_count}_animals_{n_animals}_regions_{regions}.pkl"))
    if not cands:
        where = (save_root / subset) if subset else save_root
        raise FileNotFoundError(f"No merged speeds PKL found under {where}")
    return cands[-1]


def _window_indices(window_sizes: list[int], pool: str) -> list[int]:
    n = len(window_sizes)
    if pool == "all":
        return list(range(n))
    mid = n // 2
    if pool == "short":
        return list(range(mid))
    if pool == "long":
        return list(range(mid, n))
    return list(range(n))


def main() -> int:
    ap = argparse.ArgumentParser(description="Community-wise dFC speed plots", allow_abbrev=False)
    ap.add_argument("--tr", type=int, default=None)
    ap.add_argument("--subset-name", type=str, default=None)
    ap.add_argument("--pool", type=str, default="all", choices=["all","short","long"])
    ap.add_argument("--savefig", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    data = _load_context(args.tr)
    save_root = Path(data.paths["speed"])  # type: ignore[index]
    tau_count = int(data.tau + 1)
    merged_path = _find_merged(save_root, data.n_animals, data.regions, tau_count, args.subset_name)
    with open(merged_path, "rb") as fh:
        payload = pickle.load(fh)
    all_speed = payload["speeds"]
    meta = payload.get("meta", {})
    window_sizes = meta.get("window_sizes") or list(map(int, data.time_window_range))

    # Load communities
    comm_path = Path(data.paths["allegiance"]) / "communities_wt_veh.pkl"  # type: ignore[index]
    with open(comm_path, "rb") as fh:
        communities = pickle.load(fh)
    n_comms = int(np.unique(communities).size)

    # Select window indices
    w_idx = _window_indices(list(map(int, window_sizes)), pool=args.pool)

    sns.set_theme(style="white", context="talk")
    for c in range(n_comms):
        plt.figure(figsize=(9, 6))
        palette = sns.color_palette("tab10", n_colors=len(data.groups))
        for (grp, idxs), color in zip(data.groups.items(), palette, strict=False):
            pooled = []
            for w in w_idx:
                win = all_speed[w]  # (n_animals, n_taus, T_w)
                # Pool all taus for simplicity here
                for a in idxs:
                    if a >= win.shape[0]:
                        continue
                    arr3 = np.asarray(win[a], float)
                    if arr3.ndim != 2:
                        continue
                    pooled.append(arr3[~np.isnan(arr3)])
            vals = np.concatenate(pooled) if pooled else np.array([])
            if vals.size == 0:
                continue
            label = f"{grp[0]}-{grp[1]}"
            sns.violinplot(y=vals, color=color, inner="quartile")
            plt.title(f"Community C{c+1} — {args.pool} windows")
            plt.ylabel("Speed")
        plt.xticks([])
        plt.tight_layout()
        if args.savefig:
            out = merged_path.with_suffix("")
            plt.savefig(out.as_posix() + f"_communityC{c+1}_{args.pool}.png", dpi=200)
        if not args.no_show:
            plt.show()
        else:
            plt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
