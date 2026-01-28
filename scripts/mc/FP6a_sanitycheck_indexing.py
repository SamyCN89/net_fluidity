#!/usr/bin/env python3
# %%
"""
FP6a sanity — verify FP3 indexing ordering and null vector compatibility.

Checks:
  1) FP3 mc_idx_tril equals np.tril_indices(E, k=-1) ordering (your pipeline convention)
  2) Null vector length matches K
  3) Optional: null_index_map alignment (if present)
"""

from __future__ import annotations
import json
from pathlib import Path
import re
import numpy as np

FP3_PATH = Path("/media/samy/Elements2/Proyectos/LauraHarsan/results/ines_abdallah/mc/mc_indexed/mc_indexed_ref=wt_2m_animals=126_E=820.npz")
NULL_DIR = Path("/media/samy/Elements2/Proyectos/LauraHarsan/results/ines_abdallah/mc/mc_dist/null_mc_timeshift_per_animal_tril")
NULL_GLOB = "mc_null_tril_animal_*.npy"
CHECK_NULL_FINITE_FRAC = True

def fail(msg: str, code: int = 3):
    print("[FAIL]", msg)
    raise SystemExit(code)

def ok(msg: str):
    print("[OK]", msg)

# ----------------------------
# Load FP3
# ----------------------------
d3 = np.load(FP3_PATH, allow_pickle=True)
if "params_json" not in d3.files:
    fail("FP3 missing params_json (need E).")

params = json.loads(d3["params_json"].item())
E = int(params["E"])
mc_idx = d3["mc_idx_tril"].astype(np.int64)
K = mc_idx.shape[0]

print("[FP6a sanity] FP3:", FP3_PATH)
print("  mc_idx:", mc_idx.shape, "K:", K, "E:", E)

# Expected orderings
tri_l = np.tril_indices(E, k=-1)
idx_l = np.stack([tri_l[0], tri_l[1]], axis=1).astype(np.int64)

tri_u = np.triu_indices(E, k=1)
idx_u = np.stack([tri_u[0], tri_u[1]], axis=1).astype(np.int64)

matches_tril = np.array_equal(mc_idx, idx_l)
matches_triu = np.array_equal(mc_idx, idx_u)

print("  FP3 matches tril(k=-1)?", matches_tril)
print("  FP3 matches triu(k=+1)?", matches_triu)

if not matches_tril:
    # Give a useful diff
    mism = np.nonzero((mc_idx[:, 0] != idx_l[:, 0]) | (mc_idx[:, 1] != idx_l[:, 1]))[0]
    k0 = int(mism[0]) if mism.size else 0
    fail(
        "FP3 ordering differs from expected tril(k=-1) ordering.\n"
        f"  First mismatch at k={k0}\n"
        f"    FP3 mc_idx_tril[k] = {tuple(mc_idx[k0])}\n"
        f"    tril ordering[k]   = {tuple(idx_l[k0])}\n"
        f"  Mismatched pairs: {int(mism.size)}/{K}"
    )
ok("FP3 mc_idx_tril is tril(k=-1) ordering.")


# ----------------------------
# Null file list sanity
# ----------------------------
null_files = sorted(NULL_DIR.glob(NULL_GLOB))
if not null_files:
    fail(f"No null files found: {NULL_DIR}/{NULL_GLOB}")

print("[NULL] dir:", NULL_DIR)
print("  n_files:", len(null_files))
print("  first:", null_files[0].name, "| last:", null_files[-1].name)

# Check contiguous indices (optional but nice)
idx = []
for f in null_files:
    m = re.search(r"animal_(\d+)\.npy$", f.name)
    if m:
        idx.append(int(m.group(1)))
if idx:
    idx = sorted(idx)
    missing = sorted(set(range(idx[0], idx[-1] + 1)) - set(idx))
    if missing:
        fail(f"Null indices have gaps. First few missing: {missing[:20]}")
    ok(f"Null indices contiguous: {idx[0]}..{idx[-1]}")

# ----------------------------
# Null length + finite fraction (check one file + optionally a few)
# ----------------------------
x0 = np.load(null_files[0])
if x0.size != K:
    fail(f"Null length mismatch: null size={x0.size} vs FP3 K={K}")
ok(f"Null length matches K: {K}")

if CHECK_NULL_FINITE_FRAC:
    finite_frac = float(np.isfinite(x0).mean())
    print("  finite fraction (first file):", finite_frac)
    if finite_frac < 0.95:
        fail(f"Null finite fraction too low: {finite_frac:.3f} (expected near 1.0)")
    ok("Null finite fraction looks healthy.")

# ----------------------------
# Optional: null_index_map.npz consistency
# ----------------------------
map_path = NULL_DIR / "null_index_map.npz"
if map_path.exists():
    z = np.load(map_path, allow_pickle=True)
    keys = set(z.files)
    print("[NULL MAP] found:", map_path.name, "| keys:", sorted(keys))
    # If you store mouse_ids_ts or indices, you can assert them here.
    ok("null_index_map present (good for future-proof ID alignment).")
else:
    print("[NULL MAP] not found (not required if count+order is guaranteed).")

print("\n[DONE] Sanity checks passed. Your FP6a assumptions are consistent.")
