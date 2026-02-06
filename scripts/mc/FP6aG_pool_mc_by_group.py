#!/usr/bin/env python3
# %%
"""
FP6aG — Group-wise pooling of MC values by topology (intra/inter × trimer/tetramer),
for both observed (FP3) and null (FP4bA), stored as per-animal object vectors per group.

Consumes:
  - FP3 indexed MC:
      mc_val_tril (A,K), mc_mod_idx (K,), mc_nplets_index (K,), params_json(E)
  - FP4bA null per-animal:
      mc_dist/null_mc_timeshift_per_animal_tril/mc_null_tril_animal_###.npy  (K,)
      + null_index_map.npz (optional but strongly recommended)
  - GROUPS CSV:
      results/<dataset>/mc/mc_dist/groups_table.csv
      required columns: a, group
      recommended: mouse_id, age, genotype, sex

Produces:
  results/<dataset>/mc/mc_dist/fp6a_groups_mc_by_topology_per_animal.npz
    - groups (G,) list of group labels
    - group_sizes (G,)
    - for each group g and each of 8 conditions:
        f"{g}__obs_intra_trimer" : object array length Ag (per-animal vectors)
        ...
        f"{g}__null_inter_tetramer"
    - params_json
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from shared_code.fun_paths import get_paths

# -------------------------
# CONFIG
# -------------------------
DATASET = "ines_abdallah"
TIMECOURSE_FOLDER = "Timecourses_updated_03052024"
COGNITIVE_FILE = "ROIs.xlsx"
ANAT_LABELS_FILE = "41_Allen.txt"

MC_DIST_DIRNAME = "mc_dist"

FP3_PATTERN = "mc_indexed_*.npz"  # latest
NULL_DIRNAME = "null_mc_timeshift_per_animal_tril"
NULL_PATTERN = "mc_null_tril_animal_*.npy"

GROUPS_TABLE = "groups_table.csv"   # placed in mc_dist/
OUT_NAME = "fp6a_groups_mc_by_topology_per_animal.npz"

OVERWRITE = True

# -------------------------
# Helpers
# -------------------------
def find_latest(folder: Path, pattern: str) -> Path:
    hits = sorted(folder.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No files matching {pattern} in {folder}")
    return hits[-1]

def _as_float_1d(x) -> np.ndarray:
    x = np.asarray(x)
    if x.size == 0:
        return np.array([], dtype=np.float32)
    x = x.astype(np.float32, copy=False).ravel()
    x = x[np.isfinite(x)]
    return x

# -------------------------
# MAIN
# -------------------------
paths = get_paths(
    dataset_name=DATASET,
    timecourse_folder=TIMECOURSE_FOLDER,
    cognitive_data_file=COGNITIVE_FILE,
    anat_labels_file=ANAT_LABELS_FILE,
)
mc_dir = Path(paths["mc"])
dist_dir = mc_dir / MC_DIST_DIRNAME
dist_dir.mkdir(parents=True, exist_ok=True)

# --- groups table ---
groups_path = dist_dir / GROUPS_TABLE
if not groups_path.exists():
    raise FileNotFoundError(
        f"Missing {groups_path}. Create it with columns: a, group (and optionally mouse_id/age/genotype/sex)."
    )
df = pd.read_csv(groups_path)
if "a" not in df.columns or "group" not in df.columns:
    raise ValueError("groups_table.csv must have columns: a, group")
df["a"] = df["a"].astype(int)
df["group"] = df["group"].astype(str)

# --- FP3 ---
fp3_path = find_latest(mc_dir / "mc_indexed", FP3_PATTERN)
d3 = np.load(fp3_path, allow_pickle=True)
mc_val = d3["mc_val_tril"]                    # (A,K)
mc_mod = d3["mc_mod_idx"].astype(np.int64)    # (K,)
mc_npl = d3["mc_nplets_index"].astype(np.int64)
A, K = mc_val.shape

# --- null files ---
null_dir = dist_dir / NULL_DIRNAME
null_files = sorted(null_dir.glob(NULL_PATTERN))
if len(null_files) != A:
    raise RuntimeError(f"Null file count {len(null_files)} != FP3 A {A}. Fix alignment first.")

# topology masks (K,)
is_trimer  = (mc_npl > 0)
is_tet     = ~is_trimer
is_intra   = (mc_mod > 0)
is_inter   = (mc_mod == 0)

# group list in stable order
groups = sorted(df["group"].unique().tolist())
G = len(groups)
print(f"[FP6aG] Found G={G} groups")
group_sizes = np.zeros(G, dtype=np.int32)

out = {}
out["groups"] = np.array(groups, dtype=object)

# create per-group object arrays
for gi, g in enumerate(groups):
    idx = df.loc[df["group"] == g, "a"].to_numpy(dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < A)]
    idx = np.unique(idx)
    group_sizes[gi] = idx.size
    print(f"  - {g}: A_g={idx.size}")

    # allocate object arrays length A_g
    def make_obj():
        return np.empty(idx.size, dtype=object)

    for key in [
        "obs_intra_trimer", "obs_inter_trimer", "obs_intra_tetramer", "obs_inter_tetramer",
        "null_intra_trimer", "null_inter_trimer", "null_intra_tetramer", "null_inter_tetramer",
    ]:
        out[f"{g}__{key}"] = make_obj()

    # fill per-animal vectors (preserve per-animal identity inside group)
    for j, a in enumerate(idx):
        x_obs = mc_val[a]
        goodo = np.isfinite(x_obs)

        out[f"{g}__obs_intra_trimer"][j]    = _as_float_1d(x_obs[goodo & is_intra & is_trimer])
        out[f"{g}__obs_inter_trimer"][j]    = _as_float_1d(x_obs[goodo & is_inter & is_trimer])
        out[f"{g}__obs_intra_tetramer"][j]  = _as_float_1d(x_obs[goodo & is_intra & is_tet])
        out[f"{g}__obs_inter_tetramer"][j]  = _as_float_1d(x_obs[goodo & is_inter & is_tet])

        x_null = np.load(null_files[a]).astype(np.float32, copy=False)
        if x_null.size != K:
            raise RuntimeError(f"Null size mismatch animal {a}: {x_null.size} != K={K}")
        goodn = np.isfinite(x_null)

        out[f"{g}__null_intra_trimer"][j]   = _as_float_1d(x_null[goodn & is_intra & is_trimer])
        out[f"{g}__null_inter_trimer"][j]   = _as_float_1d(x_null[goodn & is_inter & is_trimer])
        out[f"{g}__null_intra_tetramer"][j] = _as_float_1d(x_null[goodn & is_intra & is_tet])
        out[f"{g}__null_inter_tetramer"][j] = _as_float_1d(x_null[goodn & is_inter & is_tet])

out["group_sizes"] = group_sizes

params = dict(
    dataset=DATASET,
    fp3_path=str(fp3_path),
    null_dir=str(null_dir),
    groups_table=str(groups_path),
    A=int(A), K=int(K),
    note_intra="mc_mod_idx>0",
    note_trimer="mc_nplets_index>0",
    groups=groups,
)
out["params_json"] = json.dumps(params, sort_keys=True)

out_path = dist_dir / OUT_NAME
if out_path.exists() and not OVERWRITE:
    raise FileExistsError(out_path)

np.savez_compressed(out_path, **out)
print("[OK] Saved FP6aG:", out_path)
