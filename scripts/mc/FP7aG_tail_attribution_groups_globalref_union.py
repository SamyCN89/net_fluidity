#!/usr/bin/env python3
# %%
"""
FP7aG — Group-wise tail attribution (simple + robust)

Same as FP7a, but computed separately for each biological group:
(age × genotype × sex).

Uses:
- FP3 for MC
- FP6b for global thresholds
- grouping_data_oip.pkl for genotype/sex
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from shared_code.fun_paths import get_paths


# =========================
# CONFIG
# =========================
DATASET = "ines_abdallah"

FP3_PATH = "/media/samy/Elements2/Proyectos/LauraHarsan/results/ines_abdallah/mc/mc_indexed/mc_indexed_ref=wt_2m_animals=126_E=820.npz"
FP6B_NAME = "fp6b_bootstrap_mc_fp6_conditions_POOLED_IID.npz"

GROUP_PKL = "grouping_data_oip.pkl"

P_LOW = 0.05
P_HIGH = 0.95

OUT_NAME = "fp7aG_tail_attribution_obs_only.npz"
OVERWRITE = True


# =========================
# Helpers
# =========================
import pickle
def _py(x):
    """Convert numpy scalars/arrays to plain Python types for JSON."""
    import numpy as np
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    return x

def load_oip_group_masks(pkl_path: Path):
    """
    Parse grouping_data_oip.pkl which is NOT a table.
    Expected to contain:
      - group labels (strings), length G (often 8)
      - group masks  (bool arrays), shape (G, A)
    Returns:
      labels: np.ndarray (G,) dtype=object
      masks:  np.ndarray (G, A) dtype=bool
    """
    obj = pd.read_pickle(pkl_path)  # or pickle.load(open(...,'rb'))

    # We will search recursively for:
    # - list/tuple of strings (labels)
    # - list/tuple of bool ndarrays of length A (masks)
    found_labels = None
    found_masks = None

    def walk(x):
        nonlocal found_labels, found_masks

        if isinstance(x, np.ndarray):
            # could be a mask
            if x.dtype == bool and x.ndim == 1:
                return

        if isinstance(x, (list, tuple)):
            # candidate labels?
            if found_labels is None:
                if len(x) > 0 and all(isinstance(t, str) for t in x):
                    found_labels = np.array(list(x), dtype=object)

            # candidate masks?
            if found_masks is None:
                # allow list of ndarrays
                if len(x) > 0 and all(isinstance(t, np.ndarray) for t in x):
                    arrs = list(x)
                    if all(a.dtype == bool and a.ndim == 1 for a in arrs):
                        found_masks = np.stack(arrs, axis=0).astype(bool)

            # recurse
            for t in x:
                walk(t)

    walk(obj)

    if found_labels is None or found_masks is None:
        raise RuntimeError(
            "Could not locate group labels + boolean masks inside grouping_data_oip.pkl.\n"
            f"found_labels={found_labels is not None} found_masks={found_masks is not None}"
        )

    if found_masks.shape[0] != found_labels.shape[0]:
        raise RuntimeError(
            f"Mismatch: masks G={found_masks.shape[0]} labels G={found_labels.shape[0]}"
        )

    return found_labels, found_masks


def build_group_labels_from_masks(labels: np.ndarray, masks: np.ndarray):
    """
    Build per-animal group labels from (G,A) boolean masks.
    Each animal must belong to exactly one group.
    """
    G, A = masks.shape
    membership = masks.astype(np.int8).sum(axis=0)

    if not np.all(membership == 1):
        bad0 = np.where(membership != 1)[0][:10]
        raise RuntimeError(
            "Invalid grouping: some animals belong to 0 or >1 groups.\n"
            f"Example bad indices: {bad0.tolist()} with membership={membership[bad0].tolist()}"
        )

    g_idx = np.argmax(masks, axis=0)  # unique because membership==1
    group_labels = labels[g_idx].astype(object)
    return group_labels


def get_threshold(d6, key, p_grid, p):
    q = d6[f"{key}__q_obs"]
    j = np.argmin(np.abs(p_grid - p))
    return float(q[j])


def accum_region(counts, fc_idx, mc_idx, mask):

    kk = np.where(mask)[0]
    if kk.size == 0:
        return

    e1 = mc_idx[kk, 0]
    e2 = mc_idx[kk, 1]

    r = np.concatenate([
        fc_idx[e1].ravel(),
        fc_idx[e2].ravel()
    ])

    np.add.at(counts, r, 1)


def accum_module(counts, comm, mc_idx, mask):

    kk = np.where(mask)[0]
    if kk.size == 0:
        return

    e1 = mc_idx[kk, 0]
    e2 = mc_idx[kk, 1]

    m1 = comm[e1]
    m2 = comm[e2]

    np.add.at(counts, (m1, m2), 1)
    np.add.at(counts, (m2, m1), 1)


# =========================
# MAIN
# =========================
paths = get_paths(dataset_name=DATASET)

mc_dir = Path(paths["mc"])
dist_dir = mc_dir / "mc_dist"
prep_dir = Path(paths["preprocessed"])


# ---------- FP3 ----------
d3 = np.load(FP3_PATH, allow_pickle=True)

mc_val = d3["mc_val_tril"].astype(np.float32)
mc_idx = d3["mc_idx_tril"].astype(np.int64)
fc_idx = d3["fc_edge_idx"].astype(np.int64)
mc_nplets = d3["mc_nplets_index"]
sort_idx = d3["allegiance_sort"]

A, K = mc_val.shape
E = fc_idx.shape[0]
R = int(fc_idx.max() + 1)


# ---------- Communities ----------
fp2 = sorted((mc_dir / "allegiance_ref").glob("allegiance_ref_*.npz"))[-1]
d2 = np.load(fp2, allow_pickle=True)

comm = d2["communities"][sort_idx]

uniq = np.unique(comm)
remap = {u: i for i, u in enumerate(uniq)}
comm = np.array([remap[x] for x in comm])

M = comm.max() + 1


# ---------- Masks ----------
e1 = mc_idx[:, 0]
e2 = mc_idx[:, 1]

is_intra = comm[e1] == comm[e2]
is_trimer = mc_nplets > 0


# ---------- FP6b ----------
d6 = np.load(dist_dir / FP6B_NAME, allow_pickle=True)
p_grid = d6["p_grid"]


# ---------- Bundle ----------
bundle = np.load(sorted(Path(paths["preprocessed"]).glob("ts_and_meta_*.npz"))[-1],
                 allow_pickle=True)

mouse_ids = bundle["mouse_ids"].astype(str)


# ---------- Group masks ----------
labels, masks = load_oip_group_masks(prep_dir / GROUP_PKL)

# masks should be (G,A); if it is (A,G) transpose it
if masks.shape[1] != A and masks.shape[0] == A:
    masks = masks.T

if masks.shape[1] != A:
    raise RuntimeError(f"Mask size mismatch: masks shape={masks.shape}, expected second dim A={A}")

group_labels = build_group_labels_from_masks(labels, masks)

groups = np.unique(group_labels)
print("[FP7aG] Groups found:", groups)




# ---------- Categories ----------
CATS = {
    "intra_trimer":  (True,  True),
    "inter_trimer":  (False, True),
    "intra_tetramer":(True,  False),
    "inter_tetramer":(False, False),
}


# ---------- Groups ----------
groups = np.unique(group_labels)
print("[FP7aG] Groups:", groups)


# ---------- Output ----------
out = {}

meta = dict(
    dataset=DATASET,
    groups=groups.tolist(),
    A=A,
    K=K,
    R=R,
    M=M,
    p_low=P_LOW,
    p_high=P_HIGH,
)

out["params_json"] = json.dumps(meta, sort_keys=True, default=_py)


# =========================
# Main loop
# =========================
for g in groups:

    idx = np.where(group_labels == g)[0]
    Ag = idx.size

    print(f"[FP7aG] Group {g} | n={Ag}")

    for cat, (want_intra, want_trimer) in CATS.items():

        topo = (
            (is_intra if want_intra else ~is_intra) &
            (is_trimer if want_trimer else ~is_trimer)
        )

        key = f"{g}__{cat}"

        # thresholds from global FP6b
        thr_lo = get_threshold(d6, f"obs_{cat}", p_grid, P_LOW)
        thr_hi = get_threshold(d6, f"obs_{cat}", p_grid, P_HIGH)

        cnt_r = np.zeros(R, dtype=np.int64)
        cnt_mm = np.zeros((M, M), dtype=np.int64)

        for a in idx:

            x = mc_val[a]
            good = np.isfinite(x)

            m = topo & good

            hi = m & (x >= thr_hi)
            lo = m & (x <= thr_lo)

            accum_region(cnt_r, fc_idx, mc_idx, hi | lo)
            accum_module(cnt_mm, comm, mc_idx, hi | lo)

        out[f"{key}__count_r"] = cnt_r
        out[f"{key}__count_mm"] = cnt_mm
        out[f"{key}__n_animals"] = np.int32(Ag)
        out[f"{key}__thr_lo"] = np.float32(thr_lo)
        out[f"{key}__thr_hi"] = np.float32(thr_hi)


# ---------- Save ----------
out_path = dist_dir / OUT_NAME

if out_path.exists() and not OVERWRITE:
    raise FileExistsError(out_path)

np.savez_compressed(out_path, **out)

print("[OK] Saved FP7aG:", out_path)
