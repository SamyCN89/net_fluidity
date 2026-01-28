#!/usr/bin/env python3
# %%
"""
FP4bB — Bootstrap null MC distribution (value bootstrap) + histogram bootstrap CI

Consumes:
  results/<dataset>/mc_dist/null_mc_pool_timeshift_all.npz

Produces:
  results/<dataset>/mc_dist/fp4b_mc_dist_null_boot.npz

Saved keys (new additions):
  bins, h_obs, h_ci_lo, h_ci_hi
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed

from shared_code.fun_paths import get_paths

# =========================
# CONFIG
# =========================
DATASET = "ines_abdallah"
TIMECOURSE_FOLDER = "Timecourses_updated_03052024"
COGNITIVE_FILE = "ROIs.xlsx"
ANAT_LABELS_FILE = "41_Allen.txt"

# Bootstrap
N_BOOT = 2000
SEED = 0
N_JOBS = -1
CHUNK = 100
EDGE_SUBSAMPLE = 500_000

# Quantiles (must match FP4a)
P = np.linspace(0.0, 1.0, 101)

# Histogram bins (must match FP4a/FP5)
BINS = np.linspace(-0.8, 1.0, 120)

# Output
OUT_SUBDIR = "mc_dist"
OVERWRITE = True

# =========================
# Helpers
# =========================
def _hist_pdf(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
    x = x[np.isfinite(x)]
    h, _ = np.histogram(x, bins=bins, density=True)
    return h.astype(np.float32)


def _bootstrap_chunk_quantiles_and_hist(
    x_all: np.ndarray,
    p: np.ndarray,
    bins: np.ndarray,
    n_rep: int,
    subsample: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = x_all.size

    subsample = int(min(subsample, n))
    q = np.empty((n_rep, p.size), dtype=np.float64)
    h = np.empty((n_rep, bins.size - 1), dtype=np.float32)

    xbuf = np.empty(subsample, dtype=x_all.dtype)

    for i in range(n_rep):
        idx = rng.integers(0, n, size=subsample)
        xbuf[:] = x_all[idx]
        q[i] = np.quantile(xbuf, p)
        h[i] = _hist_pdf(xbuf, bins)

    return q, h


# =========================
# MAIN
# =========================
paths = get_paths(
    dataset_name=DATASET,
    timecourse_folder=TIMECOURSE_FOLDER,
    cognitive_data_file=COGNITIVE_FILE,
    anat_labels_file=ANAT_LABELS_FILE,
)
mc_dir = Path(paths["mc"])
dist_dir = mc_dir / OUT_SUBDIR

# --- Load FP4bA null pool ---
null_path = dist_dir / "null_mc_pool_timeshift_all.npz"
if not null_path.exists():
    raise FileNotFoundError(f"Missing FP4bA artifact: {null_path}")

d = np.load(null_path, allow_pickle=True)
x_null_all = d["x_null_all"].astype(np.float64)

n_vals = x_null_all.size
if n_vals == 0:
    raise RuntimeError("Null MC pool is empty.")

print("[FP4bB] Loaded null pool:", null_path.name)
print(f"[FP4bB] Null MC values: {n_vals:,}")
print(f"[FP4bB] Subsample per bootstrap: {min(EDGE_SUBSAMPLE, n_vals):,}")

# =========================
# Observed summaries (FULL data, no subsampling)
# =========================
q_null_obs = np.quantile(x_null_all, P)
h_null_obs = _hist_pdf(x_null_all, BINS)

# =========================
# Bootstrap (chunked + parallel)
# =========================
n_chunks = int(np.ceil(N_BOOT / CHUNK))
reps = [CHUNK] * n_chunks
reps[-1] = N_BOOT - CHUNK * (n_chunks - 1)

chunk_seeds = [SEED + 10_000 * j for j in range(n_chunks)]

out = Parallel(n_jobs=N_JOBS, backend="loky")(
    delayed(_bootstrap_chunk_quantiles_and_hist)(
        x_null_all, P, BINS, reps[j], EDGE_SUBSAMPLE, chunk_seeds[j]
    )
    for j in range(n_chunks)
)

q_boot = np.concatenate([qq for (qq, _) in out], axis=0)
h_boot = np.concatenate([hh for (_, hh) in out], axis=0)

q_null_lo = np.quantile(q_boot, 0.025, axis=0)
q_null_hi = np.quantile(q_boot, 0.975, axis=0)

h_lo = np.quantile(h_boot, 0.025, axis=0).astype(np.float32)
h_hi = np.quantile(h_boot, 0.975, axis=0).astype(np.float32)

print("[FP4bB] Bootstrap done.")
print("  Quantiles:", P.size)
print("  Hist bins:", BINS.size - 1)
print("  Replicates:", N_BOOT)

# =========================
# Save artifact
# =========================
out_path = dist_dir / "fp4b_mc_dist_null_boot.npz"
if out_path.exists() and not OVERWRITE:
    raise FileExistsError(f"{out_path} exists. Set OVERWRITE=True to replace.")

params = dict(
    dataset=DATASET,
    null_pool_path=str(null_path),
    n_values=int(n_vals),
    subsample=int(min(EDGE_SUBSAMPLE, n_vals)),
    n_boot=int(N_BOOT),
    seed=int(SEED),
    quantiles=P.tolist(),
    bins=BINS.tolist(),
)

np.savez_compressed(
    out_path,
    # quantiles
    p=P,
    q_null_obs=q_null_obs,
    q_null_ci_lo=q_null_lo,
    q_null_ci_hi=q_null_hi,
    # histogram + CI (NEW)
    bins=BINS,
    h_obs=h_null_obs,        # keep same naming as FP4a for plotting symmetry
    h_ci_lo=h_lo,
    h_ci_hi=h_hi,
    params_json=json.dumps(params, sort_keys=True),
)

print("[OK] Saved FP4bB artifact:")
print(" ", out_path)

# %%
