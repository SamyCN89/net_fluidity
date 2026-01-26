#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/dfc/dfc_save_session_window_quantiles.py

Save per-session × per-window dFC speed quantiles into a single NPZ tensor.

Output artifact (NPZ) contains:
  - Q: (n_sessions, n_windows, n_q) float32
      Q[i, j, k] = speed value at percentile q_grid[k] for session i at window j
  - q_grid: (n_q,) float32
  - time_windows_range: (n_windows,) int32
  - session_name: (n_sessions,) str
  - genotype: (n_sessions,) str      (if present in cog table)
  - treatment: (n_sessions,) str     (if present in cog table)
  - dataset_name, subset, lag, tau (as small metadata strings/ints where possible)

Works for:
  - dataset_name="julien" (recommended / primary)
  - dataset_name="ines"   (will save whatever session_name column exists; may need adapting)

Assumptions:
  - Speed NPZ files exist:
      <paths["speed"]>/<subset>/speed_win{w}_lag{lag}_tau{tau}_animals_{n_animals}_regions_{regions}.npz
    and each contains key "speeds" (len = n_sessions), where each entry is a 1D array of samples.

Run:
  python scripts/dfc/dfc_save_session_window_quantiles.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

from scripts.dfc.dfc_compute import DATASET_DEFAULTS, _canonical_dataset
from shared_code.fun_loaddata import load_timeseries_bundle
from shared_code.fun_paths import get_paths
from shared_code.fun_utils import load_cognitive_data


# =============================================================================
# USER CONFIG
# =============================================================================

DATASET_NAME = "julien"   # "julien" or "ines"
SUBSET = "all"            # speed subset folder, typically "all"
LAG = 1
TAU = 2

TIME_WINDOWS_RANGE = np.arange(5, 100, 1)  # windows present on disk

# quantiles you want saved in the tensor
# Q_GRID = np.array([1, 5, 25, 50, 75, 95, 99], dtype=float)
Q_GRID = np.linspace(0, 100, 100)  # 0%,5%,10%,...,100%
# output file name (goes into <paths["speed"]>/derived/)
OUT_BASENAME = "session_window_speed_quantiles"


# =============================================================================
# HELPERS
# =============================================================================

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_percentiles(x: np.ndarray, q_grid: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.full((len(q_grid),), np.nan, dtype=float)
    return np.percentile(x, q_grid)


def load_speed_stack(
    speed_path_fmt: str,
    time_windows_range: np.ndarray,
    n_sessions: int,
    regions: int,
) -> list[list[np.ndarray]]:
    """
    Returns speeds such that speeds[j][i] is a 1D array for session i at window j.
    """
    speeds: list[list[np.ndarray]] = []
    for w in time_windows_range:
        fp = speed_path_fmt.format(w=w, n_animals=n_sessions, regions=regions)
        a = np.load(fp, allow_pickle=True)
        if "speeds" not in a.files:
            raise KeyError(f"Missing key 'speeds' in {fp}. Found keys={list(a.files)}")
        s = a["speeds"]
        if len(s) != n_sessions:
            raise ValueError(f"{fp}: len(speeds)={len(s)} but expected n_sessions={n_sessions}")
        speeds.append([np.asarray(v, dtype=float).ravel() for v in s])
    return speeds


@dataclass(frozen=True)
class SessionMeta:
    session_name: np.ndarray
    genotype: np.ndarray | None
    treatment: np.ndarray | None


def extract_session_meta(cog_data: "pd.DataFrame", dataset_name: str) -> SessionMeta:
    """
    Standardize session naming and (if available) genotype/treatment columns.
    """
    if pd is None:
        raise RuntimeError("pandas is required to extract session meta from cognitive table.")

    df = cog_data.copy()

    # session name
    if dataset_name == "julien":
        if "mouse" in df.columns and "name" not in df.columns:
            df = df.rename(columns={"mouse": "name"})
        if "name" not in df.columns:
            raise ValueError("Julien cog_data must contain 'name' (or 'mouse').")
        session_name = df["name"].astype(str).to_numpy()

        genotype = df["genotype"].astype(str).to_numpy() if "genotype" in df.columns else None
        treatment = df["treatment"].astype(str).to_numpy() if "treatment" in df.columns else None
        return SessionMeta(session_name=session_name, genotype=genotype, treatment=treatment)

    # For ines: pick a sensible identifier column if present
    # (your existing pipelines often use "Name" in the raw sheet, then map to something)
    if "Name" in df.columns:
        session_name = df["Name"].astype(str).to_numpy()
    elif "name" in df.columns:
        session_name = df["name"].astype(str).to_numpy()
    else:
        # fallback: numeric ids
        session_name = np.array([f"session_{i:03d}" for i in range(len(df))], dtype=str)

    genotype = df["Genotype"].astype(str).to_numpy() if "Genotype" in df.columns else None
    treatment = df["treatment"].astype(str).to_numpy() if "treatment" in df.columns else None
    return SessionMeta(session_name=session_name, genotype=genotype, treatment=treatment)


def compute_quantile_tensor(
    speeds: list[list[np.ndarray]],
    q_grid: np.ndarray,
) -> np.ndarray:
    """
    Compute Q tensor with shape (n_sessions, n_windows, n_q).
    """
    n_windows = len(speeds)
    n_sessions = len(speeds[0])
    n_q = len(q_grid)

    Q = np.full((n_sessions, n_windows, n_q), np.nan, dtype=np.float32)

    for j in range(n_windows):
        for i in range(n_sessions):
            Q[i, j, :] = _safe_percentiles(speeds[j][i], q_grid).astype(np.float32)

    return Q


def save_npz(
    outpath: Path,
    Q: np.ndarray,
    q_grid: np.ndarray,
    time_windows_range: np.ndarray,
    meta: SessionMeta,
    extra: dict[str, Any],
) -> None:
    payload: dict[str, Any] = {
        "Q": Q.astype(np.float32),
        "q_grid": q_grid.astype(np.float32),
        "time_windows_range": time_windows_range.astype(np.int32),
        "session_name": meta.session_name.astype(str),
    }
    if meta.genotype is not None:
        payload["genotype"] = meta.genotype.astype(str)
    if meta.treatment is not None:
        payload["treatment"] = meta.treatment.astype(str)

    # extra metadata (best effort)
    for k, v in extra.items():
        payload[k] = v

    np.savez_compressed(outpath, **payload)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    if DATASET_NAME not in ("julien", "ines"):
        raise ValueError("DATASET_NAME must be 'julien' or 'ines'.")

    dataset = _canonical_dataset(DATASET_NAME)
    cfg = DATASET_DEFAULTS[dataset]

    paths = get_paths(
        dataset_name=dataset,
        timecourse_folder=cfg["timecourse_folder"],
        cognitive_data_file=cfg["cognitive_data_file"],
        anat_labels_file=cfg["anat_labels_file"],
    )

    speed_root = Path(paths["speed"])
    preprocessed_root = Path(paths["preprocessed"])

    # --- load meta bundle for dimensions ---
    loaddir_ts_meta = preprocessed_root / "ts_and_meta_2m4m.npz"
    bundle = load_timeseries_bundle(loaddir_ts_meta)
    n_sessions = int(bundle.n_animals)
    regions = int(bundle.n_regions)
    total_tr = int(bundle.total_tr)

    # --- load cognitive data ---
    loaddir_cog = preprocessed_root / (
        f"cog_data_filtered_animals_{n_sessions}_regions_{regions}_tr_{total_tr}.csv"
    )
    cog_data = load_cognitive_data(str(loaddir_cog))

    if pd is None:
        raise RuntimeError("pandas is required (your load_cognitive_data returns a DataFrame).")

    meta = extract_session_meta(cog_data, DATASET_NAME)

    # --- load speeds ---
    speed_path_fmt = str(
        speed_root / f"{SUBSET}/speed_win{{w}}_lag{LAG}_tau{TAU}_animals_{{n_animals}}_regions_{{regions}}.npz"
    )

    # quick existence check on first/last file to fail fast with useful message
    fp0 = speed_path_fmt.format(w=int(TIME_WINDOWS_RANGE[0]), n_animals=n_sessions, regions=regions)
    fplast = speed_path_fmt.format(w=int(TIME_WINDOWS_RANGE[-1]), n_animals=n_sessions, regions=regions)
    if not Path(fp0).exists():
        raise FileNotFoundError(f"Missing first speed file: {fp0}")
    if not Path(fplast).exists():
        raise FileNotFoundError(f"Missing last speed file: {fplast}")

    print("[INFO] Loading speeds from:", speed_root / SUBSET)
    speeds = load_speed_stack(speed_path_fmt, TIME_WINDOWS_RANGE, n_sessions=n_sessions, regions=regions)

    # --- compute tensor ---
    print("[INFO] Computing quantile tensor Q with shape (sessions, windows, q)...")
    Q = compute_quantile_tensor(speeds, Q_GRID)

    # --- save ---
    outdir = speed_root / "derived"
    _ensure_dir(outdir)

    outpath = outdir / (
        f"{OUT_BASENAME}_dataset-{DATASET_NAME}_subset-{SUBSET}"
        f"_lag{LAG}_tau{TAU}_animals{n_sessions}_regions{regions}"
        f"_nq{len(Q_GRID)}_w{int(TIME_WINDOWS_RANGE[0])}-{int(TIME_WINDOWS_RANGE[-1])}.npz"
    )

    extra = {
        "dataset_name": np.array([DATASET_NAME]),
        "subset": np.array([SUBSET]),
        "lag": np.int32(LAG),
        "tau": np.int32(TAU),
    }

    save_npz(outpath, Q, Q_GRID, TIME_WINDOWS_RANGE, meta, extra)
    print("[INFO] Q shape:", Q.shape)
    print("[OK] Saved quantile tensor:")
    print("   ", outpath)
    print("[OK] Contents:")
    with np.load(outpath, allow_pickle=True) as a:
        for k in a.files:
            v = a[k]
            shape = getattr(v, "shape", None)
            print(f"   - {k:18s}  shape={shape}  dtype={getattr(v, 'dtype', type(v))}")

    # small sanity print
    print("[SANITY] Q min/max (finite):",
          float(np.nanmin(Q)), float(np.nanmax(Q)))


if __name__ == "__main__":
    main()
