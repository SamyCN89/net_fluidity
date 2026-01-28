#!/usr/bin/env python3
# %%
"""
FP4bA (FAST) — Null metaconnectivity via temporal (circular-shift) surrogates
              saved as UPPER-TRIANGLE VECTORS (no gigantic (E,E) arrays).

Null model:
  - For each animal: independently circularly shift each ROI time series (R=41),
    recompute MC with same params as FP1, and average across N_SURR surrogates.
  - This destroys temporal coordination while preserving per-ROI marginals.

Outputs (SAFE + FAST):
  1) Per-animal null mean MC upper-triangle vectors:
     results/<dataset>/mc/mc_dist/null_mc_timeshift_per_animal_tril/
       mc_null_tril_animal_000.npy   # shape (Ktri,) where Ktri = E*(E-1)/2 (k=1)
       ...

  2) Global pooled reservoir sample of null MC values (for FP5):
     results/<dataset>/mc/mc_dist/null_mc_timeshift_global_pool.npz
       - x_null_all (<= MAX_NULL_VALUES,)

Notes:
  - Avoids writing a huge NPZ (zip container) that can break.
  - Avoids accumulating full (E,E) matrices, which is slow and memory-heavy.
  - Precomputes triangle indices once.
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
N_SURR = 25          # 25–30 is a good default; 100 is usually overkill
SEED = 0

# Subsampling per surrogate before feeding reservoir (speed)
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

ts = d0["ts"]  # expected shape (A, T, R) or (A, R, T) depending on your bundle
ts = np.asarray(ts)

# Enforce (A, T, R) for shifting along time axis cleanly
if ts.ndim != 3:
    raise ValueError(f"Expected ts ndim=3, got shape {ts.shape}")

A, dim2, dim3 = ts.shape

# Heuristic: your log shows (126, 450, 41) meaning (A, T, R)
# If it's (A, R, T), swap.
if dim2 == 41 and dim3 != 41:
    # (A, R, T) -> (A, T, R)
    ts = np.transpose(ts, (0, 2, 1))

A, T, R = ts.shape
E = R * (R - 1) // 2

print("[FP0] Loaded:", bundle_path.name)
print("      ts raw shape:", d0["ts"].shape)
print("      ts used shape (A,T,R):", ts.shape)
print("      R:", R, "E:", E, "T:", T, "A:", A)

# Precompute MC upper triangle indices ONCE
tri = np.triu_indices(E, k=1)
Ktri = tri[0].size
print("      MC upper-tri size Ktri:", Ktri)

# Output dirs
out_dir_tril = mc_dir / OUT_SUBDIR / "null_mc_timeshift_per_animal_tril"
out_dir_tril.mkdir(parents=True, exist_ok=True)

# If not overwriting, check if some outputs already exist
if not OVERWRITE:
    existing = list(out_dir_tril.glob("mc_null_tril_animal_*.npy"))
    if existing:
        raise FileExistsError(
            f"{out_dir_tril} already contains {len(existing)} files. Set OVERWRITE=True to replace."
        )

# Init reservoir
rng = np.random.default_rng(SEED)
reservoir = np.empty(MAX_NULL_VALUES, dtype=np.float32)
filled = 0
seen = 0

# =========================
# Compute per-animal null mean (upper-triangle) + global reservoir
# =========================
for a in range(A):
    out_a = out_dir_tril / f"mc_null_tril_animal_{a:03d}.npy"
    if out_a.exists() and not OVERWRITE:
        print(f"[SKIP] {out_a.name} exists.")
        continue

    print(f"[FP4bA FAST] Animal {a+1}/{A} | reservoir {filled:,}/{MAX_NULL_VALUES:,} | seen {seen:,}")

    ts_a = ts[a]  # (T, R)

    acc = np.zeros(Ktri, dtype=np.float64)

    for r_surr in range(N_SURR):
        # Independent circular shift per ROI
        ts_shifted = np.empty_like(ts_a)
        for i in range(R):
            k = rng.integers(0, T)
            ts_shifted[:, i] = np.roll(ts_a[:, i], k)

        # Compute MC for this surrogate (single animal)
        mc_surr = compute_metaconnectivity(
            ts_shifted[None, ...],  # (1, T, R)
            window_size=WINDOW_SIZE,
            lag=LAG,
            n_jobs=N_JOBS_MC,
            save_path=None,
        )[0]  # (E, E)

        # Extract upper-triangle MC values (k=1)
        x = mc_surr[tri]
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue

        # Accumulate for per-animal mean (must be full Ktri aligned)
        # If there are NaNs, we still want consistent indexing.
        # So: re-extract WITHOUT filtering for acc (keep NaNs as 0? no).
        # Best: accumulate with nan->0 and also count finite per entry.
        # Minimal, robust approach: keep a finite mask per surrogate.
        # We'll do that with one extra array.
        if r_surr == 0:
            count = np.zeros(Ktri, dtype=np.int32)

        x_full = mc_surr[tri].astype(np.float64, copy=False)
        good = np.isfinite(x_full)
        acc[good] += x_full[good]
        count[good] += 1

        # Feed reservoir (optional subsample)
        x_pool = x
        if EDGE_SUBSAMPLE is not None and x_pool.size > EDGE_SUBSAMPLE:
            jj = rng.integers(0, x_pool.size, size=EDGE_SUBSAMPLE)
            x_pool = x_pool[jj]

        filled, seen = reservoir_update(
            reservoir, filled, seen, x_pool.astype(np.float32, copy=False), rng
        )

    # Final per-animal mean: divide elementwise where count>0, else NaN
    mean_tril = np.full(Ktri, np.nan, dtype=np.float32)
    goodc = count > 0
    mean_tril[goodc] = (acc[goodc] / count[goodc]).astype(np.float32)

    np.save(out_a, mean_tril)
    print(f"      saved {out_a.name} | finite frac: {np.isfinite(mean_tril).mean():.3f}")

# Finalize reservoir
if filled < MAX_NULL_VALUES:
    x_null_all = reservoir[:filled].copy()
else:
    x_null_all = reservoir

print("[FP4bA FAST] Done.")
print("  reservoir final size:", f"{x_null_all.size:,}")
print("  total values seen:", f"{seen:,}")

# =========================
# Save global pool artifact
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

print("[OK] Saved FP4bA FAST outputs:")
print("  per-animal tril null means:", out_dir_tril)
print("  global null pool:", out_pool)
