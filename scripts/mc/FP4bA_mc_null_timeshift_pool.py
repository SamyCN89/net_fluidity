#!/usr/bin/env python3
# %%
"""
FP4bA — Global null MC pool via circular time-shift surrogates (ALL animals, CAPPED)

Changes vs previous:
  - Uses reservoir sampling to cap the global null pool at MAX_NULL_VALUES
  - Random replacement ensures an unbiased sample from the full stream of null MC values

Produces:
  results/<dataset>/mc_dist/null_mc_pool_timeshift_all.npz
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from shared_code.fun_metaconnectivity import compute_metaconnectivity
from shared_code.fun_paths import get_paths

# =========================
# CONFIG
# =========================
DATASET = "ines_abdallah"
TIMECOURSE_FOLDER = "Timecourses_updated_03052024"
COGNITIVE_FILE = "ROIs.xlsx"
ANAT_LABELS_FILE = "41_Allen.txt"

# Surrogates
N_SURR = 100
SEED = 0

# Subsampling per surrogate (keep moderate; reservoir will cap globally anyway)
EDGE_SUBSAMPLE = 50_000

# Global cap (reservoir size)
MAX_NULL_VALUES = 10_000_000

# MC params (must match FP1)
WINDOW_SIZE = 7
LAG = 1
N_JOBS_MC = 1  # IMPORTANT: avoid nested parallelism

# Output
OUT_SUBDIR = "mc_dist"
OVERWRITE = True


# =========================
# Helpers
# =========================
def find_latest(folder: Path, pattern: str) -> Path:
    hits = sorted(folder.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No files matching {pattern} in {folder}")
    return hits[-1]


# =========================
# Reservoir sampling update
# =========================
def reservoir_update(
    reservoir: np.ndarray,
    filled: int,
    seen: int,
    x: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """
    Reservoir sampling (Algorithm R), streaming values.

    reservoir: preallocated (K,)
    filled: how many slots are currently filled (<=K)
    seen: total number of values seen so far in the stream
    x: new batch of candidate values (1D)
    Returns updated (filled, seen).
    """
    K = reservoir.size
    x = x.ravel()
    for v in x:
        seen += 1
        if filled < K:
            reservoir[filled] = v
            filled += 1
        else:
            j = int(rng.integers(0, seen))  # uniform in [0, seen-1]
            if j < K:
                reservoir[j] = v
    return filled, seen


# =========================
# MAIN
# =========================
paths = get_paths(
    dataset_name=DATASET,
    timecourse_folder=TIMECOURSE_FOLDER,
    cognitive_data_file=COGNITIVE_FILE,
    anat_labels_file=ANAT_LABELS_FILE,
)

preproc_dir = Path(paths["preprocessed"])
mc_dir = Path(paths["mc"])

# --- Load canonical FP0 bundle ---
bundle_path = find_latest(preproc_dir, "ts_and_meta_*.npz")
d0 = np.load(bundle_path, allow_pickle=True)

ts = d0["ts"]  # (A, R, T)
A, R, T = ts.shape

print("[FP0] Loaded:", bundle_path.name)
print("      ts shape:", ts.shape)

# --- Init reservoir ---
rng = np.random.default_rng(SEED)
reservoir = np.empty(MAX_NULL_VALUES, dtype=np.float32)
filled = 0
seen = 0

# =========================
# Stream null MC values into reservoir
# =========================
for a in range(A):
    print(
        f"[FP4bA] Animal {a+1}/{A} | reservoir filled: {filled:,}/{MAX_NULL_VALUES:,} | seen: {seen:,}"
    )
    ts_a = ts[a]  # (R, T)

    for r in range(N_SURR):
        # --- Circular time-shift per ROI time series (destroys cross-ROI temporal coordination) ---
        ts_shifted = np.empty_like(ts_a)
        for i in range(R):
            k = rng.integers(0, T)
            ts_shifted[i] = np.roll(ts_a[i], k)

        # --- Compute MC (single animal) ---
        mc_surr = compute_metaconnectivity(
            ts_shifted[None, ...],
            window_size=WINDOW_SIZE,
            lag=LAG,
            n_jobs=N_JOBS_MC,
            save_path=None,
        )[
            0
        ]  # (E, E)

        # --- Extract upper triangle MC values ---
        tri = np.triu_indices(mc_surr.shape[0], k=1)
        x = mc_surr[tri]
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue

        # --- Subsample this surrogate's MC values (optional, helps speed the stream) ---
        if EDGE_SUBSAMPLE is not None and x.size > EDGE_SUBSAMPLE:
            jj = rng.integers(0, x.size, size=EDGE_SUBSAMPLE)
            x = x[jj]

        # --- Update reservoir (random replacement) ---
        filled, seen = reservoir_update(
            reservoir, filled, seen, x.astype(np.float32, copy=False), rng
        )

# finalize
if filled < MAX_NULL_VALUES:
    x_null_all = reservoir[:filled].copy()
else:
    x_null_all = reservoir  # already full

print("[FP4bA] Done.")
print("  reservoir final size:", f"{x_null_all.size:,}")
print("  total values seen:", f"{seen:,}")

# =========================
# Save artifact
# =========================
out_dir = mc_dir / OUT_SUBDIR
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / "null_mc_pool_timeshift_all.npz"
if out_path.exists() and not OVERWRITE:
    raise FileExistsError(f"{out_path} exists. Set OVERWRITE=True to replace.")

params = dict(
    dataset=DATASET,
    n_animals=int(A),
    n_surrogates=int(N_SURR),
    edge_subsample=int(EDGE_SUBSAMPLE),
    max_null_values=int(MAX_NULL_VALUES),
    window_size=WINDOW_SIZE,
    lag=LAG,
    seed=int(SEED),
    bundle_path=str(bundle_path),
    reservoir_seen_total=int(seen),
)

np.savez_compressed(
    out_path,
    x_null_all=x_null_all,
    params_json=json.dumps(params, sort_keys=True),
)

print("[OK] Saved FP4bA capped null pool:")
print(" ", out_path)
