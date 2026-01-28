#!/usr/bin/env python3
# %%
"""
FP4a — Empirical MC distribution (value bootstrap) + histogram bootstrap CI

- Pools all finite MC values from the upper triangle (all animals)
- Estimates full quantile function Q(p), p in [0,1]
- ALSO estimates histogram PDF with bootstrap 95% CI (binwise)
- Bootstrap over MC values with subsampling for efficiency

Produces:
  results/<dataset>/mc_dist/fp4a_mc_dist_all_boot.npz

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

# Quantiles (include endpoints)
P = np.linspace(0.0, 1.0, 101)

# Histogram bins (must match FP5 plotting)
BINS = np.linspace(-0.8, 1.0, 120)

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
    """
    Bootstrap chunk:
      q: (n_rep, len(p))      quantiles
      h: (n_rep, n_bins-1)    histogram pdf
    """
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

# --- Load FP1 mc_raw ---
mc_raw_path = find_latest(mc_dir / "mc_raw", "mc_raw_*.npz")
d = np.load(mc_raw_path, allow_pickle=True)
mc = d["mc"]  # (A, E, E)

A, E, _ = mc.shape
print("[FP1] Loaded:", mc_raw_path.name, "| mc:", mc.shape)

# --- Pool upper-triangle MC values ---
tri = np.triu_indices(E, k=1)
vals = []
for a in range(A):
    x = mc[a][tri]
    x = x[np.isfinite(x)]
    if x.size:
        vals.append(x)

x_all = np.concatenate(vals).astype(np.float64)
n_vals = x_all.size
if n_vals == 0:
    raise RuntimeError("No finite MC values found.")

print(f"[FP4a] Pooled MC values: {n_vals:,}")
print(f"[FP4a] Subsample per bootstrap: {min(EDGE_SUBSAMPLE, n_vals):,}")

# =========================
# Observed summaries (FULL data, no subsampling)
# =========================
q_obs = np.quantile(x_all, P)
h_obs = _hist_pdf(x_all, BINS)

# =========================
# Bootstrap (chunked + parallel)
# =========================
n_chunks = int(np.ceil(N_BOOT / CHUNK))
reps = [CHUNK] * n_chunks
reps[-1] = N_BOOT - CHUNK * (n_chunks - 1)
chunk_seeds = [SEED + 10_000 * j for j in range(n_chunks)]

out = Parallel(n_jobs=N_JOBS, backend="loky")(
    delayed(_bootstrap_chunk_quantiles_and_hist)(
        x_all, P, BINS, reps[j], EDGE_SUBSAMPLE, chunk_seeds[j]
    )
    for j in range(n_chunks)
)

q_boot = np.concatenate([qq for (qq, _) in out], axis=0)  # (N_BOOT, len(P))
h_boot = np.concatenate([hh for (_, hh) in out], axis=0)  # (N_BOOT, n_bins-1)

q_lo = np.quantile(q_boot, 0.025, axis=0)
q_hi = np.quantile(q_boot, 0.975, axis=0)

h_lo = np.quantile(h_boot, 0.025, axis=0).astype(np.float32)
h_hi = np.quantile(h_boot, 0.975, axis=0).astype(np.float32)

print("[FP4a] Bootstrap done.")
print("  Quantiles:", P.size)
print("  Hist bins:", BINS.size - 1)
print("  Replicates:", N_BOOT)

# =========================
# Save artifact
# =========================
out_dir = mc_dir / OUT_SUBDIR
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / "fp4a_mc_dist_all_boot.npz"
if out_path.exists() and not OVERWRITE:
    raise FileExistsError(f"{out_path} exists. Set OVERWRITE=True to replace.")

params = dict(
    dataset=DATASET,
    mc_raw_path=str(mc_raw_path),
    n_animals=int(A),
    n_edges=int(E),
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
    q_obs=q_obs,
    q_ci_lo=q_lo,
    q_ci_hi=q_hi,
    # histogram + CI (NEW)
    bins=BINS,
    h_obs=h_obs,
    h_ci_lo=h_lo,
    h_ci_hi=h_hi,
    params_json=json.dumps(params, sort_keys=True),
)

print("[OK] Saved FP4a artifact:")
print(" ", out_path)
