#!/usr/bin/env python3
"""Plot pooled dFC speed distributions split by window ranges."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Mapping

import re

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scripts.dfc.dfc_compute import DATASET_DEFAULTS, _canonical_dataset
except ModuleNotFoundError:  # pragma: no cover - allow standalone execution
    import sys

    HERE = Path(__file__).resolve()
    ROOT = HERE.parents[2]
    for candidate in (ROOT, ROOT / "shared_code"):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    from scripts.dfc.dfc_compute import DATASET_DEFAULTS, _canonical_dataset  # type: ignore

from shared_code.fun_paths import get_paths

LOGGER = logging.getLogger(__name__)


def load_dataset_context(dataset_name: str, tr_hint: int | None) -> tuple[dict, pd.DataFrame]:
    canonical = _canonical_dataset(dataset_name)
    cfg = DATASET_DEFAULTS[canonical]
    paths = get_paths(
        dataset_name=canonical,
        timecourse_folder=cfg["timecourse_folder"],
        cognitive_data_file=cfg["cognitive_data_file"],
        anat_labels_file=cfg["anat_labels_file"],
    )
    bundle_path = Path(paths["preprocessed"]) / cfg["bundle_name"]
    if not bundle_path.exists():
        raise FileNotFoundError(f"Preprocessed bundle missing: {bundle_path}")
    with np.load(bundle_path, allow_pickle=True) as bundle:
        if "mouse_ids" not in bundle.files:
            raise KeyError(f"{bundle_path} is missing 'mouse_ids'; regenerate preprocessing bundle.")
        total_tr = int(bundle["total_tr"])
    cog_path = _pick_cog_csv(Path(paths["preprocessed"]), canonical, tr_hint, total_tr)
    cog_df = pd.read_csv(cog_path)
    return paths, cog_df


def _pick_cog_csv(root: Path, dataset_key: str, tr_hint: int | None, total_tr: int) -> Path:
    def first_match(patterns: Iterable[str]) -> Path | None:
        for patt in patterns:
            hits = sorted(root.glob(patt))
            if hits:
                return hits[0]
        return None

    if dataset_key.startswith("ines"):
        path = first_match(["cog_data_sorted_2m4m.csv", "cog_data_sorted*.csv"])
        if path:
            return path
    if dataset_key.startswith("julien"):
        patterns: list[str] = []
        if tr_hint:
            patterns.append(f"cog_data_filtered*_tr_{int(tr_hint)}.csv")
        patterns.append(f"cog_data_filtered*_tr_{int(total_tr)}.csv")
        patterns.append("cog_data_filtered*.csv")
        path = first_match(patterns)
        if path:
            return path

    path = first_match(["cog_data_filtered*.csv", "cog_data_sorted*.csv", "cog_data*.csv"])
    if path:
        return path
    raise FileNotFoundError(f"No cognitive CSV found under {root}")


def _resolve_group_columns(cog_df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    lower_map = {str(col).lower(): col for col in cog_df.columns}
    resolved: list[str] = []
    for col in columns:
        key = str(col).strip()
        if not key:
            continue
        if key in cog_df.columns:
            resolved.append(key)
            continue
        match = lower_map.get(key.lower())
        if match is None:
            raise KeyError(f"Column '{col}' not present in cognitive dataframe.")
        resolved.append(match)
    if not resolved:
        raise ValueError("No valid grouping columns provided.")
    return resolved


def build_groups_from_columns(cog_df: pd.DataFrame, columns: list[str]) -> dict[tuple, list[int]]:
    grouped = cog_df.reset_index(drop=True).groupby(columns).groups
    out: dict[tuple, list[int]] = {}
    for key, idx in grouped.items():
        tup = key if isinstance(key, tuple) else (key,)
        out[tup] = sorted(int(i) for i in idx)
    return out


def load_per_animal_from_npz(npz_path: Path, tau_index: int | None = None) -> list[np.ndarray]:
    z = np.load(npz_path, allow_pickle=True)
    if "speeds" not in z:
        raise KeyError(f"NPZ missing 'speeds': {npz_path}")
    speeds = z["speeds"]
    per_animal: list[np.ndarray] = []
    for entry in speeds:
        arr = np.asarray(entry, dtype=float)
        if arr.ndim == 0:
            per_animal.append(np.array([], float))
            continue
        if tau_index is None or tau_index < 0:
            vals = arr[~np.isnan(arr)]
        else:
            if tau_index >= arr.shape[0]:
                vals = np.array([], float)
            else:
                vals = arr[tau_index][~np.isnan(arr[tau_index])]
        per_animal.append(vals)
    return per_animal


def _find_region_folders(speed_root: Path) -> list[Path]:
    prefixed = [p for p in sorted(speed_root.glob("regions-*")) if p.is_dir()]
    if prefixed:
        return prefixed
    all_dir = speed_root / "all"
    if all_dir.exists():
        return [all_dir]
    return [d for d in sorted(speed_root.iterdir()) if d.is_dir()]


def _pool_windows(windows: list[int], threshold: int | None, include_all: bool) -> dict[str, list[int]]:
    pools: dict[str, list[int]] = {}
    if threshold is not None:
        pools["short"] = [w for w in windows if w <= threshold]
        pools["long"] = [w for w in windows if w > threshold]
    if include_all or not pools:
        pools["all"] = windows[:]
    return {name: vals for name, vals in pools.items() if vals}


def _collect_values(
    region_dir: Path,
    groups_map: Mapping[tuple, list[int]],
    tau_index: int,
    pools: Mapping[str, list[int]],
) -> dict[str, dict[tuple, np.ndarray]]:
    per_pool: dict[str, dict[tuple, np.ndarray]] = {name: {} for name in pools}
    pattern = re.compile(r"speed_win(?P<window>\d+)_lag(?P<lag>\d+)")
    files: dict[int, Path] = {}
    for p in region_dir.glob("speed_win*_*.npz"):
        match = pattern.match(p.name)
        if not match:
            LOGGER.warning("Skipping unrecognised NPZ filename %s", p.name)
            continue
        win = int(match.group("window"))
        files[win] = p
    for pool, win_list in pools.items():
        pool_vals: dict[tuple, list[np.ndarray]] = {g: [] for g in groups_map}
        for win in win_list:
            npz = files.get(win)
            if npz is None:
                LOGGER.warning("Missing NPZ for window %s in %s; skipping.", win, region_dir)
                continue
            per_animal = load_per_animal_from_npz(npz, tau_index=None if tau_index < 0 else tau_index)
            for group, indices in groups_map.items():
                vals = [per_animal[i] for i in indices if i < len(per_animal) and per_animal[i].size]
                if vals:
                    pool_vals[group].append(np.concatenate(vals))
        for group, chunks in pool_vals.items():
            if chunks:
                per_pool[pool][group] = np.concatenate(chunks)
            else:
                per_pool[pool][group] = np.array([], float)
    return per_pool


def _plot_distributions(
    distributions: Mapping[str, Mapping[tuple, np.ndarray]],
    *,
    group_labels: list[tuple],
    bins: int,
    density: bool,
    output: Path,
    title: str,
):
    pools = list(distributions.keys())
    rows = len(group_labels)
    cols = len(pools)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows), squeeze=False, sharex="col", sharey="row")
    for r, group in enumerate(group_labels):
        for c, pool in enumerate(pools):
            ax = axes[r, c]
            vals = np.asarray(distributions[pool].get(group, np.array([], float)), float)
            if vals.size == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
            else:
                ax.hist(vals, bins=bins, density=density, alpha=0.75, color="#1f77b4", edgecolor="black")
                ax.axvline(np.median(vals), color="red", linestyle="--", linewidth=1.2, label="median")
            if r == 0:
                ax.set_title(f"{pool}")
            if c == 0:
                ax.set_ylabel("Density" if density else "Count")
                ax.set_xlabel("Speed")
            else:
                ax.set_xlabel("Speed")
            if vals.size:
                ax.legend(loc="best")
            ax.grid(alpha=0.2, linestyle=":")
            ax.set_facecolor("#f8f8f8")
            ax.set_title(f"{pool}", loc="right") if r == 0 else None
            if c == 0:
                ax.set_title(f"{group}", loc="left")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot pooled speed distributions split by window threshold.")
    parser.add_argument("--dataset-name", default="ines", help="Dataset alias (e.g. 'ines', 'julien').")
    parser.add_argument("--subset", default="all", help="Speed subset folder (default: all).")
    parser.add_argument("--region", default=None, help="Specific region folder under the subset (default: auto).")
    parser.add_argument("--group-cols", default="Genotype,Sexe", help="Comma-separated grouping columns.")
    parser.add_argument("--tau-index", type=int, default=0, help="Tau index to select (-1 pools all).")
    parser.add_argument(
        "--pool-threshold",
        default="median",
        help="Window threshold for short/long split (integer or 'median'). Use 'none' to disable.",
    )
    parser.add_argument("--include-all-pool", action="store_true", help="Always include an 'all' pooled distribution.")
    parser.add_argument("--bins", type=int, default=40, help="Histogram bins.")
    parser.add_argument("--density", action="store_true", help="Plot density instead of counts.")
    parser.add_argument("--plot-format", default="png", choices=["png", "pdf", "svg"], help="Figure format.")
    parser.add_argument("--outdir", default=None, help="Override output directory under paths['f_speed'].")
    parser.add_argument("--title", default=None, help="Custom figure title.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    paths, cog_df = load_dataset_context(args.dataset_name, tr_hint=None)

    group_cols = _resolve_group_columns(cog_df, [c.strip() for c in args.group_cols.split(",") if c.strip()])
    groups_map = build_groups_from_columns(cog_df, group_cols)
    group_keys = sorted(groups_map.keys())

    speed_root = Path(paths["speed"])
    if args.subset:
        speed_root = speed_root / args.subset
    if not speed_root.exists():
        raise FileNotFoundError(f"Speed subset folder not found: {speed_root}")

    region_dirs = _find_region_folders(speed_root)
    if not region_dirs:
        raise FileNotFoundError(f"No region directories under {speed_root}")
    if args.region:
        region_dir = speed_root / args.region
        if not region_dir.exists():
            raise FileNotFoundError(f"Region folder not found: {region_dir}")
    else:
        # prefer an 'all' folder when present
        region_dir = next((p for p in region_dirs if p.name == "all"), region_dirs[0])
    LOGGER.info("Using region folder: %s", region_dir)

    window_files = sorted(region_dir.glob("speed_win*_*.npz"))
    if not window_files:
        raise FileNotFoundError(f"No per-window NPZ files under {region_dir}")
    windows = []
    for p in window_files:
        name = p.name
        try:
            win_part = name.split("speed_win", 1)[1]
            win_str = win_part.split("_", 1)[0]
            windows.append(int(win_str))
        except Exception:
            LOGGER.warning("Unrecognised filename pattern for %s; skipping.", name)
    windows = sorted(set(windows))
    LOGGER.info("Found %d window files (min=%s max=%s).", len(windows), windows[0], windows[-1])

    threshold: int | None = None
    if args.pool_threshold.lower() != "none":
        if args.pool_threshold.lower() == "median":
            threshold = int(np.median(windows))
        else:
            threshold = int(float(args.pool_threshold))
        LOGGER.info("Splitting windows by threshold=%s (short<=threshold, long>threshold).", threshold)

    pools = _pool_windows(windows, threshold, include_all=args.include_all_pool or threshold is None)
    LOGGER.info("Pools: %s", ", ".join(f"{k}({len(v)})" for k, v in pools.items()))

    distributions = _collect_values(region_dir, groups_map, args.tau_index, pools)

    fig_root = Path(paths.get("f_speed", Path(paths["speed"])))  # type: ignore[index]
    outdir = fig_root / (args.outdir or args.subset or "distributions")
    outdir.mkdir(parents=True, exist_ok=True)
    filename = f"speed_distributions_{region_dir.name}.{args.plot_format}"
    title = args.title or f"{args.dataset_name} — {region_dir.name}"
    _plot_distributions(
        distributions,
        group_labels=group_keys,
        bins=args.bins,
        density=args.density,
        output=outdir / filename,
        title=title,
    )
    LOGGER.info("Saved distribution plot → %s", outdir / filename)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
