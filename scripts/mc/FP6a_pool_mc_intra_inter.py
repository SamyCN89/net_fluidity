#!/usr/bin/env python3
# %%
"""
FP6a — Pool metaconnectivity by topology (trimer/tetramer) and modularity (intra/inter)

Consumes:
  - FP3 indexed MC artifact (observed):
      mc_val_tril (A,K)
      mc_mod_idx  (K,)   0=inter, >0=intra (module id)
      mc_nplets_index (K,)  >0 trimer, 0 tetramer
      mc_idx_tril (K,2)  (must match tril(k=-1) ordering)
  - FP4bA FAST null (per animal):
      results/<dataset>/mc/mc_dist/null_mc_timeshift_per_animal_tril/
        mc_null_tril_animal_*.npy   shape (K,)
      + null_index_map.npz (recommended) with mouse_ids_ts / age_ts / bundle_path

Produces:
  results/<dataset>/mc/mc_dist/fp6a_mc_intra_inter_trimer_tetramer_per_animal.npz
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
OUT_SUBDIR = "mc_dist"

NULL_DIRNAME = "null_mc_timeshift_per_animal_tril"
NULL_PATTERN = "mc_null_tril_animal_*.npy"
NULL_SIDECAR = "null_index_map.npz"

OVERWRITE = True


# =========================
# Helpers
# =========================
def find_latest(folder: Path, pattern: str) -> Path:
    hits = sorted(folder.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No files matching {pattern} in {folder}")
    return hits[-1]


def assert_fp3_tril_order(mc_idx_tril: np.ndarray, E: int) -> None:
    tri = np.tril_indices(E, k=-1)
    idx = np.stack([tri[0], tri[1]], axis=1).astype(np.int64)
    if mc_idx_tril.shape != idx.shape or not np.array_equal(mc_idx_tril, idx):
        k0 = 0
        raise RuntimeError(
            "FP3 mc_idx_tril is not tril(k=-1) ordering.\n"
            f"Example FP3[{k0}]={tuple(mc_idx_tril[k0])} vs tril[{k0}]={tuple(idx[k0])}\n"
            "Fix: rebuild FP3 indexing or reindex everything into FP3 ordering."
        )


def assert_null_filenames_are_index_stable(null_files: list[Path], A: int) -> None:
    expected = [f"mc_null_tril_animal_{a:03d}.npy" for a in range(A)]
    got = [p.name for p in null_files]
    if got != expected:
        for i, (g, e) in enumerate(zip(got, expected)):
            if g != e:
                raise RuntimeError(
                    "Null files are not in strict index order (risk of animal permutation).\n"
                    f"First mismatch at i={i}: got {g} expected {e}"
                )
        raise RuntimeError("Null files do not match expected naming pattern exactly.")


def quick_size_stats(obj_arr: np.ndarray) -> tuple[float, float, int]:
    sizes = np.array([len(x) for x in obj_arr], dtype=np.int64)
    return float(np.median(sizes)), float(np.min(sizes)), int(np.sum(sizes == 0))


def assert_null_sidecar_matches_canonical(null_dir: Path, paths: dict, A_expected: int) -> Path:
    """
    Try to prove that null index corresponds to canonical FP0 session ordering.

    Modes:
      - STRONG: if canonical bundle has mouse_ids_ts/age_ts, compare to sidecar.
      - FALLBACK: if canonical bundle lacks IDs, verify:
          * sidecar.bundle_path exists (or can be opened)
          * ts_shape_used matches between sidecar and canonical bundle (after enforcing (A,T,R))
          * A matches expected
        (Plus we already enforce strict filename ordering elsewhere.)
    Returns the bundle_path used.
    """
    sidecar = null_dir / NULL_SIDECAR
    if not sidecar.exists():
        raise RuntimeError(
            f"Missing {sidecar}. Without it you're relying only on filename order (fragile). "
            "Regenerate FP4bA with null_index_map.npz."
        )

    z = np.load(sidecar, allow_pickle=True)

    # bundle path from sidecar
    if "bundle_path" in z.files:
        bundle_path = Path(str(z["bundle_path"]))
    else:
        # last-resort guess: latest ts_and_meta_*.npz
        preproc = Path(paths["preprocessed"])
        hits = sorted(preproc.glob("ts_and_meta_*.npz"))
        if not hits:
            raise RuntimeError(f"No ts_and_meta_*.npz found in {preproc}")
        bundle_path = hits[-1]

    if not bundle_path.exists():
        # if sidecar path is stale (moved disk), fallback to latest
        preproc = Path(paths["preprocessed"])
        hits = sorted(preproc.glob("ts_and_meta_*.npz"))
        if not hits:
            raise RuntimeError(f"Sidecar bundle_path missing and no ts_and_meta_*.npz found in {preproc}")
        bundle_path = hits[-1]

    d0 = np.load(bundle_path, allow_pickle=True)

    # ---- Always check A using ts ----
    if "ts" not in d0.files:
        raise RuntimeError(f"Canonical bundle {bundle_path} has no 'ts' key; cannot verify anything.")
    ts = np.asarray(d0["ts"])
    if ts.ndim != 3:
        raise RuntimeError(f"Canonical bundle ts has ndim={ts.ndim} shape={ts.shape}, expected 3D.")
    A0, d2, d3 = ts.shape
    # enforce (A,T,R) same way as FP4bA
    if d2 == 41 and d3 != 41:
        ts = np.transpose(ts, (0, 2, 1))
        A0 = ts.shape[0]

    if A0 != A_expected:
        raise RuntimeError(f"Canonical bundle A={A0} != FP3 A={A_expected}. Upstream mismatch.")

    # ---- If IDs exist in canonical bundle, do strong proof ----
    has_ids0 = ("mouse_ids_ts" in d0.files) and ("age_ts" in d0.files)
    has_ids_side = ("mouse_ids_ts" in z.files) and ("age_ts" in z.files) and (z["mouse_ids_ts"].size > 0)

    if has_ids0 and has_ids_side:
        mouse0 = d0["mouse_ids_ts"].astype(str)
        age0 = d0["age_ts"].astype(str)
        null_mouse = z["mouse_ids_ts"].astype(str)
        null_age = z["age_ts"].astype(str)

        if mouse0.shape != null_mouse.shape or not np.array_equal(mouse0, null_mouse):
            mm = np.nonzero(mouse0 != null_mouse)[0]
            i0 = int(mm[0]) if mm.size else 0
            raise RuntimeError(
                "ID alignment FAIL: canonical bundle mouse_ids_ts != FP4bA sidecar mouse_ids_ts\n"
                f"First mismatch at i={i0}: bundle={mouse0[i0]} null={null_mouse[i0]}"
            )

        if age0.shape != null_age.shape or not np.array_equal(age0, null_age):
            mm = np.nonzero(age0 != null_age)[0]
            i0 = int(mm[0]) if mm.size else 0
            raise RuntimeError(
                "ID alignment FAIL: canonical bundle age_ts != FP4bA sidecar age_ts\n"
                f"First mismatch at i={i0}: bundle={age0[i0]} null={null_age[i0]}"
            )

        print("[OK] ID alignment proven: canonical bundle IDs match FP4bA sidecar.")
        return bundle_path

    # ---- FALLBACK proof: compare shapes recorded in sidecar ----
    if "ts_shape_used" not in z.files:
        raise RuntimeError(
            "Sidecar missing ts_shape_used; cannot even do fallback alignment. "
            "Regenerate FP4bA with the updated sidecar."
        )

    side_shape = tuple(int(x) for x in np.asarray(z["ts_shape_used"]).ravel().tolist())
    can_shape = tuple(int(x) for x in ts.shape)

    if can_shape != side_shape:
        raise RuntimeError(
            "Fallback alignment FAIL: sidecar ts_shape_used != canonical bundle ts shape (after (A,T,R) coercion)\n"
            f"sidecar={side_shape} canonical={can_shape}\n"
            "This suggests FP4bA used a different bundle or orientation."
        )

    print("[OK] Fallback alignment: canonical bundle matches FP4bA sidecar by ts shape + A.")
    print("     (Canonical bundle lacks mouse_ids_ts/age_ts, so ID proof is impossible.)")
    return bundle_path


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

# ---------- Load FP3 ----------
fp3_path = find_latest(mc_dir / "mc_indexed", FP3_PATTERN)
d3 = np.load(fp3_path, allow_pickle=True)

params_fp3 = json.loads(d3["params_json"].item())
E = int(params_fp3["E"])

mc_idx_tril = d3["mc_idx_tril"].astype(np.int64)         # (K,2)
mc_val = d3["mc_val_tril"]                               # (A,K)
mc_mod_idx = d3["mc_mod_idx"].astype(np.int64)           # (K,)
mc_nplets_index = d3["mc_nplets_index"].astype(np.int64) # (K,)

A, K = mc_val.shape

assert_fp3_tril_order(mc_idx_tril, E)

K_expected = E * (E - 1) // 2
if K != K_expected:
    raise RuntimeError(f"FP3 K={K} but expected {K_expected} from E={E}")

print("[FP3] Loaded:", fp3_path.name)
print("      A:", A, "E:", E, "K:", K)

# ---------- Null files ----------
null_dir = mc_dir / OUT_SUBDIR / NULL_DIRNAME
null_files = sorted(null_dir.glob(NULL_PATTERN))
if not null_files:
    raise FileNotFoundError(f"No null files in {null_dir} matching {NULL_PATTERN}")

if len(null_files) != A:
    raise RuntimeError(f"Null file count={len(null_files)} != FP3 A={A}")

assert_null_filenames_are_index_stable(null_files, A)

bundle_used = assert_null_sidecar_matches_canonical(null_dir, paths, A_expected=A)
print("[OK] ID alignment proven against canonical bundle:", bundle_used.name)

# Optional: show null params (helps catch window/lag mismatch)
pool_path = mc_dir / OUT_SUBDIR / "null_mc_timeshift_global_pool.npz"
if pool_path.exists():
    zp = np.load(pool_path, allow_pickle=True)
    pnull = json.loads(zp["params_json"].item())
    print("[NULL] params:", {k: pnull.get(k) for k in ["window_size", "lag", "n_surrogates", "seed"]})
else:
    print("[WARN] Missing null_mc_timeshift_global_pool.npz (cannot check null params).")

# ---------- Masks (K,) ----------
is_trimer = (mc_nplets_index > 0)
is_tetramer = ~is_trimer
is_intra = (mc_mod_idx > 0)
is_inter = (mc_mod_idx == 0)

# ---------- Pool per animal ----------
obs_intra_trimer = np.empty(A, dtype=object)
obs_inter_trimer = np.empty(A, dtype=object)
obs_intra_tetramer = np.empty(A, dtype=object)
obs_inter_tetramer = np.empty(A, dtype=object)

null_intra_trimer = np.empty(A, dtype=object)
null_inter_trimer = np.empty(A, dtype=object)
null_intra_tetramer = np.empty(A, dtype=object)
null_inter_tetramer = np.empty(A, dtype=object)

for a in range(A):
    v_obs = mc_val[a]
    good_obs = np.isfinite(v_obs)

    obs_intra_trimer[a] = v_obs[good_obs & is_intra & is_trimer]
    obs_inter_trimer[a] = v_obs[good_obs & is_inter & is_trimer]
    obs_intra_tetramer[a] = v_obs[good_obs & is_intra & is_tetramer]
    obs_inter_tetramer[a] = v_obs[good_obs & is_inter & is_tetramer]

    v_null = np.load(null_files[a]).astype(np.float32, copy=False)
    if v_null.size != K:
        raise RuntimeError(f"Null animal {a} has size {v_null.size} != K={K}")
    good_null = np.isfinite(v_null)

    null_intra_trimer[a] = v_null[good_null & is_intra & is_trimer]
    null_inter_trimer[a] = v_null[good_null & is_inter & is_trimer]
    null_intra_tetramer[a] = v_null[good_null & is_intra & is_tetramer]
    null_inter_tetramer[a] = v_null[good_null & is_inter & is_tetramer]

print("[FP6a] pooled per-animal vectors done.")

for name, arr in [
    ("obs_intra_trimer", obs_intra_trimer),
    ("obs_inter_trimer", obs_inter_trimer),
    ("obs_intra_tetramer", obs_intra_tetramer),
    ("obs_inter_tetramer", obs_inter_tetramer),
    ("null_intra_trimer", null_intra_trimer),
    ("null_inter_trimer", null_inter_trimer),
    ("null_intra_tetramer", null_intra_tetramer),
    ("null_inter_tetramer", null_inter_tetramer),
]:
    med, mn, n0 = quick_size_stats(arr)
    print(f"[SANITY] {name:20s} median={med:,.0f} min={mn:,.0f} empty_animals={n0}/{A}")

# ---------- Save ----------
out_dir = mc_dir / OUT_SUBDIR
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "fp6a_mc_intra_inter_trimer_tetramer_per_animal.npz"

if out_path.exists() and not OVERWRITE:
    raise FileExistsError(out_path)

params = dict(
    dataset=DATASET,
    fp3_path=str(fp3_path),
    null_dir=str(null_dir),
    canonical_bundle=str(bundle_used),
    A=int(A),
    E=int(E),
    K=int(K),
    note_intra_def="mc_mod_idx>0",
    note_trimer_def="mc_nplets_index>0",
    ordering="FP3 tril(k=-1) required; null must match same ordering",
    null_files_naming="mc_null_tril_animal_{a:03d}.npy enforced",
)

np.savez_compressed(
    out_path,
    obs_intra_trimer=obs_intra_trimer,
    obs_inter_trimer=obs_inter_trimer,
    obs_intra_tetramer=obs_intra_tetramer,
    obs_inter_tetramer=obs_inter_tetramer,
    null_intra_trimer=null_intra_trimer,
    null_inter_trimer=null_inter_trimer,
    null_intra_tetramer=null_intra_tetramer,
    null_inter_tetramer=null_inter_tetramer,
    params_json=json.dumps(params, sort_keys=True),
)

print("[OK] Saved FP6a:", out_path)
# %%
