#!/usr/bin/env python3
# %%
"""
FP6a — Pool metaconnectivity by topology (trimer/tetramer) and modularity (intra/inter)

Consumes:
  - FP3 indexed MC artifact (observed):
      mc_val_tril (A,K)
      mc_mod_idx  (K,)   0=inter, >0=intra (module id)
      mc_nplets_index (K,)  >0 trimer, 0 tetramer (per your FP3)
  - FP4bA FAST null (per animal):
      results/<dataset>/mc/mc_dist/null_mc_timeshift_per_animal_tril/
        mc_null_tril_animal_*.npy   shape (K,)

Produces:
  results/<dataset>/mc/mc_dist/fp6a_mc_intra_inter_trimer_tetramer_per_animal.npz

No stats. No bootstrap. No plots.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from shared_code.fun_paths import get_paths

# =========================
# CONFIG
# =========================
DATASET = "ines_abdallah"
TIMECOURSE_FOLDER = "Timecourses_updated_03052024"
COGNITIVE_FILE = "ROIs.xlsx"
ANAT_LABELS_FILE = "41_Allen.txt"

FP3_PATTERN = "mc_indexed_*.npz"  # latest
NULL_DIRNAME = "null_mc_timeshift_per_animal_tril"
NULL_PATTERN = "mc_null_tril_animal_*.npy"

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
# MAIN
# =========================
paths = get_paths(
    dataset_name=DATASET,
    timecourse_folder=TIMECOURSE_FOLDER,
    cognitive_data_file=COGNITIVE_FILE,
    anat_labels_file=ANAT_LABELS_FILE,
)
mc_dir = Path(paths["mc"])

# ---------- Load FP3 (observed MC vectors) ----------
fp3_path = find_latest(mc_dir / "mc_indexed", FP3_PATTERN)
d3 = np.load(fp3_path, allow_pickle=True)

mc_val = d3["mc_val_tril"]                       # (A, K)
mc_mod_idx = d3["mc_mod_idx"].astype(int)        # (K,)
mc_nplets_index = d3["mc_nplets_index"]          # (K,)

A, K = mc_val.shape
if mc_mod_idx.shape != (K,):
    raise ValueError(f"mc_mod_idx shape {mc_mod_idx.shape} != (K,) with K={K}")
if mc_nplets_index.shape != (K,):
    raise ValueError(f"mc_nplets_index shape {mc_nplets_index.shape} != (K,) with K={K}")

print("[FP3] Loaded:", fp3_path.name)
print("      mc_val:", mc_val.shape)

# ---------- Load null per-animal triangle vectors ----------
null_dir = mc_dir / OUT_SUBDIR / NULL_DIRNAME
null_files = sorted(null_dir.glob(NULL_PATTERN))
if not null_files:
    raise FileNotFoundError(f"No null files in {null_dir} matching {NULL_PATTERN}")

# Load as list (avoid huge stack if you want), but we’ll sanity-check lengths
mc_null = [np.load(p) for p in null_files]
A_null = len(mc_null)
K_null = int(mc_null[0].size)

print("[NULL] Loaded:", A_null, "files from", null_dir.name)
print("       K_null:", K_null)

# Hard checks
if K_null != K:
    raise RuntimeError(
        f"Observed K={K} != Null K={K_null}. "
        f"This means FP3 indexing and FP4bA triangle extraction disagree."
    )
if A_null < A:
    print(f"[WARN] Null has fewer animals ({A_null}) than observed ({A}). Will use A={A_null}.")
    A_use = A_null
else:
    A_use = A

# Infer E from K (since K = E*(E-1)/2)
E = int((1 + np.sqrt(1 + 8*K)) / 2)
if E * (E - 1) // 2 != K:
    raise RuntimeError(f"Cannot infer integer E from K={K}. Got E={E} but E*(E-1)/2 != K.")
print("       inferred E:", E)

# ---------- Masks in triangle (vector) space ----------
is_trimer = mc_nplets_index > 0
is_tetramer = mc_nplets_index == 0

# Safer than ==1: keep all intra-module pairs across any module id
is_intra = mc_mod_idx > 0
is_inter = mc_mod_idx == 0

# =========================
# Pool per animal (variable-length vectors)
# =========================
obs_intra_trimer = []
obs_inter_trimer = []
obs_intra_tetramer = []
obs_inter_tetramer = []

null_intra_trimer = []
null_inter_trimer = []
null_intra_tetramer = []
null_inter_tetramer = []

for a in range(A_use):
    # ----- observed -----
    v_obs = mc_val[a]
    good_obs = np.isfinite(v_obs)

    obs_intra_trimer.append(v_obs[good_obs & is_intra & is_trimer])
    obs_inter_trimer.append(v_obs[good_obs & is_inter & is_trimer])
    obs_intra_tetramer.append(v_obs[good_obs & is_intra & is_tetramer])
    obs_inter_tetramer.append(v_obs[good_obs & is_inter & is_tetramer])

    # ----- null (already triangle vector) -----
    v_null = mc_null[a]
    if v_null.size != K:
        raise RuntimeError(f"Null animal {a} has size {v_null.size} != K={K}")
    good_null = np.isfinite(v_null)

    null_intra_trimer.append(v_null[good_null & is_intra & is_trimer])
    null_inter_trimer.append(v_null[good_null & is_inter & is_trimer])
    null_intra_tetramer.append(v_null[good_null & is_intra & is_tetramer])
    null_inter_tetramer.append(v_null[good_null & is_inter & is_tetramer])

print("[FP6a] pooled per-animal vectors done.")

# =========================
# Save FP6a artifact
# =========================
out_dir = mc_dir / OUT_SUBDIR
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / "fp6a_mc_intra_inter_trimer_tetramer_per_animal.npz"
if out_path.exists() and not OVERWRITE:
    raise FileExistsError(out_path)

params = dict(
    dataset=DATASET,
    fp3_path=str(fp3_path),
    null_dir=str(null_dir),
    A_obs=int(A),
    A_null=int(A_null),
    A_used=int(A_use),
    K=int(K),
    E=int(E),
    note_intra_def="mc_mod_idx>0",
    note_trimer_def="mc_nplets_index>0",
)

np.savez_compressed(
    out_path,
    obs_intra_trimer=np.array(obs_intra_trimer, dtype=object),
    obs_inter_trimer=np.array(obs_inter_trimer, dtype=object),
    obs_intra_tetramer=np.array(obs_intra_tetramer, dtype=object),
    obs_inter_tetramer=np.array(obs_inter_tetramer, dtype=object),
    null_intra_trimer=np.array(null_intra_trimer, dtype=object),
    null_inter_trimer=np.array(null_inter_trimer, dtype=object),
    null_intra_tetramer=np.array(null_intra_tetramer, dtype=object),
    null_inter_tetramer=np.array(null_inter_tetramer, dtype=object),
    params_json=json.dumps(params, sort_keys=True),
)

print("[OK] Saved FP6a:", out_path)
