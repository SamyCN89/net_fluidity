#!/usr/bin/env python3
# %%
"""
FP4bA (FAST) — Null metaconnectivity via temporal circular-shift surrogates
               saved as LOWER-TRIANGLE vectors (tril, k=-1), aligned with FP3.

Null model
----------
For each animal:
  - Independently circular-shift each ROI time series
  - Recompute MC for each surrogate
  - Average MC across surrogates
  - Save the LOWER-triangle MC vector (K = E*(E-1)/2)

Also builds:
  - A global reservoir-sampled pool of null MC values (for FP5 / diagnostics)

CRITICAL INVARIANT
------------------
The output ordering EXACTLY matches FP3 mc_idx_tril:
    np.tril_indices(E, k=-1)

Outputs
-------
1) Per-animal null MC vectors:
   results/<dataset>/mc/mc_dist/null_mc_timeshift_per_animal_tril/
       mc_null_tril_animal_000.npy
       ...

2) Global pooled null reservoir:
   results/<dataset>/mc/mc_dist/null_mc_timeshift_global_pool.npz

3) Sidecar alignment metadata:
   results/<dataset>/mc/mc_dist/null_mc_timeshift_per_animal_tril/null_index_map.npz
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
N_SURR = 25
SEED = 0

# Reservoir controls
EDGE_SUBSAMPLE = 50_000
MAX_NULL_VALUES = 10_000_000

# MC params (MUST match FP1 / FP3)
WINDOW_SIZE = 7
LAG = 1
N_JOBS_MC = 1  # avoid nested parallelism

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


def reservoir_update(
    reservoir: np.ndarray,
    filled: int,
    seen: int,
    x: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """
    Reservoir sampling (Algorithm R), streaming values.
    """
    K = reservoir.size
    for v in x.ravel():
        seen += 1
        if filled < K:
            reservoir[filled] = v
            filled += 1
        else:
            j = int(rng.integers(0, seen))
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

ts = np.asarray(d0["ts"])
if ts.ndim != 3:
    raise ValueError(f"Expected ts ndim=3, got shape {ts.shape}")

# Enforce (A, T, R)
A, d2, d3 = ts.shape
if d2 == 41 and d3 != 41:
    ts = np.transpose(ts, (0, 2, 1))

A, T, R = ts.shape
E = R * (R - 1) // 2

print("[FP0] Loaded:", bundle_path.name)
print("      ts used shape (A,T,R):", ts.shape)
print("      A:", A, "T:", T, "R:", R, "E:", E)

# --- Triangle indices: MUST match FP3 ---
tri = np.tril_indices(E, k=-1)
Ktri = tri[0].size
print("      MC lower-tri size Ktri:", Ktri)

# --- Output dirs ---
out_dir_tril = mc_dir / OUT_SUBDIR / "null_mc_timeshift_per_animal_tril"
out_dir_tril.mkdir(parents=True, exist_ok=True)

# --- Save alignment sidecar ---
sidecar = out_dir_tril / "null_index_map.npz"
np.savez_compressed(
    sidecar,
    dataset=DATASET,
    bundle_path=str(bundle_path),
    ts_shape_raw=np.array(d0["ts"].shape, dtype=int),
    ts_shape_used=np.array(ts.shape, dtype=int),
    mouse_ids_ts=np.asarray(d0["mouse_ids_ts"]).astype(str) if "mouse_ids_ts" in d0 else np.array([], dtype=str),
    age_ts=np.asarray(d0["age_ts"]).astype(str) if "age_ts" in d0 else np.array([], dtype=str),
)
print("[OK] Saved null sidecar:", sidecar)

# --- RNG split (IMPORTANT) ---
rng_shift = np.random.default_rng(SEED + 1)
rng_sub   = np.random.default_rng(SEED + 2)
rng_res   = np.random.default_rng(SEED + 3)

# --- Reservoir init ---
reservoir = np.empty(MAX_NULL_VALUES, dtype=np.float32)
filled = 0
seen = 0

# =========================
# Per-animal null MC
# =========================
for a in range(A):
    out_a = out_dir_tril / f"mc_null_tril_animal_{a:03d}.npy"
    if out_a.exists() and not OVERWRITE:
        print(f"[SKIP] {out_a.name} exists.")
        continue

    print(f"[FP4bA] Animal {a+1}/{A} | reservoir {filled:,}/{MAX_NULL_VALUES:,}")

    ts_a = ts[a]  # (T, R)

    acc = np.zeros(Ktri, dtype=np.float64)
    count = np.zeros(Ktri, dtype=np.int32)

    for r_surr in range(N_SURR):
        # --- Independent circular shift per ROI ---
        ts_shifted = np.empty_like(ts_a)
        for i in range(R):
            k = rng_shift.integers(0, T)
            ts_shifted[:, i] = np.roll(ts_a[:, i], k)

        # --- Compute MC ---
        mc_surr = compute_metaconnectivity(
            ts_shifted[None, ...],
            window_size=WINDOW_SIZE,
            lag=LAG,
            n_jobs=N_JOBS_MC,
            save_path=None,
        )[0]  # (E,E)

        # --- Extract aligned triangle ---
        x_full = mc_surr[tri].astype(np.float64, copy=False)
        good = np.isfinite(x_full)

        acc[good] += x_full[good]
        count[good] += 1

        # --- Reservoir feed ---
        x_pool = x_full[good].astype(np.float32, copy=False)
        if x_pool.size == 0:
            continue

        if EDGE_SUBSAMPLE is not None and x_pool.size > EDGE_SUBSAMPLE:
            jj = rng_sub.integers(0, x_pool.size, size=EDGE_SUBSAMPLE)
            x_pool = x_pool[jj]

        filled, seen = reservoir_update(reservoir, filled, seen, x_pool, rng_res)

    # --- Per-animal mean ---
    mean_tril = np.full(Ktri, np.nan, dtype=np.float32)
    goodc = count > 0
    mean_tril[goodc] = (acc[goodc] / count[goodc]).astype(np.float32)

    np.save(out_a, mean_tril)

    if a == 0:
        q = np.quantile(mean_tril[np.isfinite(mean_tril)], [0.001, 0.5, 0.999])
        print("[SANITY] animal0 mean_tril quantiles:", q)

    print(f"      saved {out_a.name} | finite frac {np.isfinite(mean_tril).mean():.3f}")

# =========================
# Finalize reservoir
# =========================
x_null_all = reservoir[:filled].copy() if filled < MAX_NULL_VALUES else reservoir

print("[FP4bA] Done.")
print("  reservoir size:", f"{x_null_all.size:,}")
print("  total values seen:", f"{seen:,}")

# =========================
# Save global pool
# =========================
out_pool = mc_dir / OUT_SUBDIR / "null_mc_timeshift_global_pool.npz"
if out_pool.exists() and not OVERWRITE:
    raise FileExistsError(out_pool)

params = dict(
    dataset=DATASET,
    n_animals=int(A),
    n_surrogates=int(N_SURR),
    window_size=WINDOW_SIZE,
    lag=LAG,
    seed=int(SEED),
    edge_subsample=int(EDGE_SUBSAMPLE),
    max_null_values=int(MAX_NULL_VALUES),
    bundle_path=str(bundle_path),
    reservoir_seen_total=int(seen),
    ts_shape_used=[int(x) for x in ts.shape],
    R=int(R),
    E=int(E),
    Ktri=int(Ktri),
)

np.savez_compressed(
    out_pool,
    x_null_all=x_null_all,
    params_json=json.dumps(params, sort_keys=True),
)

print("[OK] Saved FP4bA outputs:")
print("  per-animal nulls:", out_dir_tril)
print("  global null pool:", out_pool)
# %%
