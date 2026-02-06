#!/usr/bin/env python3
# %%
"""
FP7aG — Groupwise tail attribution (OBSERVED only, identity-preserving)
Global thresholds from a reference FP6b (e.g., wt_2m) applied to all groups.

Consumes
--------
FP3 (indexed MC):
  - mc_val_tril      : (A,K)
  - mc_idx_tril      : (K,2)  FC-edge indices in SORTED edge space
  - fc_edge_idx      : (E,2)  ROI pairs per FC edge (SORTED)
  - mc_nplets_index  : (K,)
  - allegiance_sort  : (E,)

FP2 (allegiance ref):
  - communities      : (E,) module id per FC edge (UNSORTED; reordered using allegiance_sort)

FP6b (threshold source; global):
  - p_grid
  - <cond>__q_obs for cond in {obs_intra_trimer, obs_inter_trimer, obs_intra_tetramer, obs_inter_tetramer}

Bundle + external metadata:
  - bundle ts_and_meta_*.npz must have: mouse_ids, is_2month_old
  - grouping_data_oip.pkl provides: genotype, sex per mouse_id

Produces
--------
results/<dataset>/mc/mc_dist/fp7aG_tail_attribution_obs_globalthr_by_group.npz

Output schema (for each group gsafe and category cat):
  - g=<gsafe>__<cat>__thr_lo / thr_hi
  - g=<gsafe>__<cat>__n_animals
  - g=<gsafe>__<cat>__n_values_possible      (#finite topology entries across group×K)
  - g=<gsafe>__<cat>__n_events_hi / n_events_lo
  - g=<gsafe>__<cat>__event_rate_hi / event_rate_lo   (events / n_values_possible)
  - g=<gsafe>__<cat>__count_hi_r_evt, __count_lo_r_evt
  - g=<gsafe>__<cat>__count_hi_r_anim, __count_lo_r_anim
  - g=<gsafe>__<cat>__count_hi_mm_evt, __count_lo_mm_evt
  - g=<gsafe>__<cat>__count_hi_mm_anim, __count_lo_mm_anim
  - optional heavy:
      g=<gsafe>__<cat>__count_hi_k_evt, __count_lo_k_evt
      g=<gsafe>__<cat>__count_hi_k_anim, __count_lo_k_anim
"""

from __future__ import annotations

import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

from shared_code.fun_paths import get_paths

# =========================
# CONFIG
# =========================
DATASET = "ines_abdallah"
TIMECOURSE_FOLDER = "Timecourses_updated_03052024"
COGNITIVE_FILE = "ROIs.xlsx"
ANAT_LABELS_FILE = "41_Allen.txt"

# Inputs
FP3_PATH = "/media/samy/Elements2/Proyectos/LauraHarsan/results/ines_abdallah/mc/mc_indexed/mc_indexed_ref=wt_2m_animals=126_E=820.npz"
FP2_PATTERN = "allegiance_ref_wt_2m_*.npz"
FP6B_NAME = "fp6b_bootstrap_mc_fp6_conditions_POOLED_IID.npz"
MC_DIST_DIRNAME = "mc_dist"

# Tail definition (global thresholds)
P_LOW  = 0.05
P_HIGH = 0.95

# Group metadata
GROUPING_PKL_NAME = "grouping_data_oip.pkl"  # in preprocessed_data/
# Build union groups
MAKE_SEX_BOTH_UNION = True

# Output
OUT_NAME = "fp7aG_tail_attribution_obs_globalthr_by_group.npz"
OVERWRITE = True

# Save per-k arrays? (K can be huge; but you did it already in FP7a)
SAVE_K_LEVEL = False

# =========================
# Helpers
# =========================
def find_latest(folder: Path, pattern: str) -> Path:
    hits = sorted(folder.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No files matching {pattern} in {folder}")
    return hits[-1]

def _get_q_at_p(q_obs: np.ndarray, p_grid: np.ndarray, p: float) -> float:
    j = int(np.argmin(np.abs(p_grid - p)))
    return float(q_obs[j])

def _sanitize_group_name(g: str) -> str:
    # safe for filesystem + npz keys
    g = str(g)
    g = g.replace(" ", "")
    g = re.sub(r"[^\w\-\=\.]+", "_", g)  # keep alnum _ - = .
    return g[:180]  # avoid insane path/key lengths

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns exist: {candidates}\nAvailable columns: {list(df.columns)}")

def _norm_sex(x: str) -> str:
    s = str(x).strip().upper()
    if s in {"F", "FEMALE"}:
        return "F"
    if s in {"M", "MALE"}:
        return "M"
    return s

def _norm_geno(x: str) -> str:
    s = str(x).strip()
    # optional normalization rules:
    # s = s.replace("WT", "wt").replace("DKI", "dKI")
    return s

def _load_group_table(preproc_data_dir: Path) -> pd.DataFrame:
    """
    Load a grouping table from a pickle that may contain nested tuples/lists/dicts.

    Accepts:
      - pd.DataFrame directly
      - dict-like (converted to DataFrame)
      - nested tuple/list structures containing a DataFrame or dict

    Raises a helpful error if nothing usable is found.
    """
    p = preproc_data_dir / GROUPING_PKL_NAME
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Check preproc_data_dir.")

    obj = pd.read_pickle(p)

    def _iter_flat(x):
        """Yield leaf objects from arbitrarily nested tuples/lists/dicts (keys+values)."""
        stack = [x]
        while stack:
            cur = stack.pop()
            if isinstance(cur, pd.DataFrame):
                yield cur
            elif isinstance(cur, dict):
                # yield dict itself as a candidate, and also explore its contents
                yield cur
                stack.extend(list(cur.values()))
                stack.extend(list(cur.keys()))
            elif isinstance(cur, (tuple, list)):
                stack.extend(list(cur))
            else:
                yield cur

    # 1) First hit: DataFrame
    for leaf in _iter_flat(obj):
        if isinstance(leaf, pd.DataFrame):
            return leaf

    # 2) Next: dict that can be turned into a DataFrame
    for leaf in _iter_flat(obj):
        if isinstance(leaf, dict):
            try:
                df = pd.DataFrame(leaf)
                if df.shape[0] > 0 and df.shape[1] > 0:
                    return df
            except Exception:
                pass

    # 3) Nothing found: dump a compact structural summary
    # (keeps it readable without printing the whole pickle)
    def _shape_hint(x):
        try:
            import numpy as _np
            if isinstance(x, _np.ndarray):
                return f"ndarray shape={x.shape} dtype={x.dtype}"
        except Exception:
            pass
        return ""

    leaves = list(_iter_flat(obj))
    types = [type(x) for x in leaves[:50]]
    hints = [_shape_hint(x) for x in leaves[:50]]
    msg = ["grouping_data_oip.pkl structure unsupported: no DataFrame/dict found.",
           f"Top-level type: {type(obj)}",
           "First ~50 leaf types/hints:"]
    for t, h in zip(types, hints):
        msg.append(f"  - {t}{(' | ' + h) if h else ''}")
    raise TypeError("\n".join(msg))

def build_group_labels(bundle, preproc_data_dir: Path) -> np.ndarray:
    """
    Uses bundle['mouse_ids'] and bundle['is_2month_old'] + grouping_data_oip.pkl to make:
      age=2m|geno=wt|sex=F
    """
    if "mouse_ids" not in bundle.files:
        raise KeyError(f"Bundle missing mouse_ids. Keys: {bundle.files}")
    if "is_2month_old" not in bundle.files:
        raise KeyError(f"Bundle missing is_2month_old. Keys: {bundle.files}")

    mouse_ids = np.asarray(bundle["mouse_ids"]).astype(str)
    is_2m = np.asarray(bundle["is_2month_old"]).astype(bool)
    age = np.where(is_2m, "2m", "4m").astype(object)

    df = _load_group_table(preproc_data_dir).copy()

    col_id = _pick_col(df, ["mouse_id", "mouse_ids", "mouse", "id", "animal_id", "MouseID"])
    col_g  = _pick_col(df, ["genotype", "geno", "Genotype", "GENO"])
    col_s  = _pick_col(df, ["sex", "Sex", "SEX"])

    df[col_id] = df[col_id].astype(str)
    df[col_g]  = df[col_g].astype(str)
    df[col_s]  = df[col_s].astype(str)
    df = df.set_index(col_id)

    missing = [mid for mid in mouse_ids if mid not in df.index]
    if missing:
        raise RuntimeError(
            f"Metadata table does not cover all bundle mouse_ids. Missing {len(missing)}.\n"
            f"Examples: {missing[:10]}"
        )

    geno = np.array([_norm_geno(df.at[mid, col_g]) for mid in mouse_ids], dtype=object)
    sex  = np.array([_norm_sex(df.at[mid, col_s]) for mid in mouse_ids], dtype=object)

    labels = np.array([f"age={age[i]}|geno={geno[i]}|sex={sex[i]}" for i in range(mouse_ids.size)], dtype=object)
    return labels

def make_sex_both_union(labels: np.ndarray) -> np.ndarray:
    out = []
    for g in labels.astype(str):
        parts = g.split("|")
        d = {p.split("=")[0]: p.split("=")[1] for p in parts}
        d["sex"] = "both"
        out.append(f"age={d['age']}|geno={d['geno']}|sex={d['sex']}")
    return np.array(out, dtype=object)

def _accum_region_counts(counts_r: np.ndarray, fc_edge_idx: np.ndarray, mc_idx: np.ndarray, hit_k: np.ndarray):
    kk = np.where(hit_k)[0]
    if kk.size == 0:
        return
    e1 = mc_idx[kk, 0]
    e2 = mc_idx[kk, 1]
    r = np.concatenate([fc_edge_idx[e1].ravel(), fc_edge_idx[e2].ravel()], axis=0).astype(np.int64)
    np.add.at(counts_r, r, 1)

def _accum_modulepair_counts(counts_mm: np.ndarray, comm_edges: np.ndarray, mc_idx: np.ndarray, hit_k: np.ndarray):
    kk = np.where(hit_k)[0]
    if kk.size == 0:
        return
    e1 = mc_idx[kk, 0]
    e2 = mc_idx[kk, 1]
    m1 = comm_edges[e1]
    m2 = comm_edges[e2]
    np.add.at(counts_mm, (m1, m2), 1)
    np.add.at(counts_mm, (m2, m1), 1)

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
dist_dir = mc_dir / MC_DIST_DIRNAME

# Where your metadata actually is (you showed preprocessed_data/ under results/<dataset>/)
# So: results/<dataset>/preprocessed_data
# results_dataset_dir = Path(paths["results"]) / DATASET
# preproc_data_dir = results_dataset_dir / "preprocessed_data"
preproc_data_dir = Path(paths["preprocessed"])
if not preproc_data_dir.exists():
    # fallback to paths["preprocessed"] if your get_paths points there
    preproc_data_dir = Path(paths.get("preprocessed_data", "")) if "preprocessed_data" in paths else preproc_data_dir
if not preproc_data_dir.exists():
    raise FileNotFoundError(f"Cannot find preprocessed_data directory. Tried: {preproc_data_dir}")

# Load canonical bundle (must contain mouse_ids + is_2month_old)
preproc_dir = Path(paths["preprocessed"])
bundle_path = find_latest(preproc_dir, "ts_and_meta_*.npz")
bundle = np.load(bundle_path, allow_pickle=True)

# ---------- Load FP3 ----------
d3 = np.load(FP3_PATH, allow_pickle=True)
mc_val = d3["mc_val_tril"].astype(np.float32)        # (A,K)
mc_idx = d3["mc_idx_tril"].astype(np.int64)          # (K,2) SORTED edge index space
fc_edge_idx = d3["fc_edge_idx"].astype(np.int64)     # (E,2) SORTED
mc_nplets = d3["mc_nplets_index"].astype(np.int8)    # (K,)
sort_idx = d3["allegiance_sort"].astype(np.int64)    # (E,)

A, K = mc_val.shape
E = fc_edge_idx.shape[0]
R = int(fc_edge_idx.max() + 1)

# ---------- Load FP2 communities and sort into FP3 space ----------
fp2_path = find_latest(mc_dir / "allegiance_ref", FP2_PATTERN)
d2 = np.load(fp2_path, allow_pickle=True)

comm_edges = d2["communities"].astype(np.int64)  # UNSORTED
comm_edges = comm_edges[sort_idx]                # SORTED (matches mc_idx)
assert comm_edges.shape == (E,)

# Normalize module labels to 0..M-1
uniq = np.unique(comm_edges)
remap = {int(u): i for i, u in enumerate(uniq)}
comm_edges = np.array([remap[int(x)] for x in comm_edges], dtype=np.int64)
M = int(comm_edges.max() + 1)

# topology masks
e1 = mc_idx[:, 0]
e2 = mc_idx[:, 1]
is_intra = (comm_edges[e1] == comm_edges[e2])
is_inter = ~is_intra
is_trimer = (mc_nplets > 0)

print("[FP7aG] FP3 A,K:", A, K, "| E,R,M:", E, R, M)
print("[FP7aG] intra frac:", float(is_intra.mean()), "| trimer frac:", float(is_trimer.mean()))
print("[FP7aG] bundle:", bundle_path.name)
print("[FP7aG] group table dir:", preproc_data_dir)

# ---------- Build group labels ----------
group_labels = build_group_labels(bundle, preproc_data_dir)  # base 8 groups
if group_labels.shape[0] != A:
    raise RuntimeError(f"group_labels length {group_labels.shape[0]} != FP3 A {A}. Bundle/FP3 mismatch.")

if MAKE_SEX_BOTH_UNION:
    group_labels_union = make_sex_both_union(group_labels)
else:
    group_labels_union = None

groups_base = np.unique(group_labels)
print("[FP7aG] base groups:", groups_base)

if group_labels_union is not None:
    groups_union = np.unique(group_labels_union)
    print("[FP7aG] union groups:", groups_union)
else:
    groups_union = np.array([], dtype=object)

# We'll compute for: base + union
GROUP_SPECS = []
for g in groups_base:
    GROUP_SPECS.append(("base", str(g)))
if group_labels_union is not None:
    for g in groups_union:
        GROUP_SPECS.append(("union_sex_both", str(g)))

# ---------- Load FP6b thresholds (GLOBAL) ----------
fp6b_path = dist_dir / FP6B_NAME
d6 = np.load(fp6b_path, allow_pickle=True)
p_grid = d6["p_grid"].astype(np.float32)

def load_q(key: str) -> np.ndarray:
    kk = f"{key}__q_obs"
    if kk not in d6.files:
        raise KeyError(f"FP6b missing {kk}. Available keys example: {d6.files[:10]}")
    return d6[kk].astype(np.float32)

CATS = {
    "intra_trimer":   dict(obs_key="obs_intra_trimer",   topo=is_intra &  is_trimer),
    "inter_trimer":   dict(obs_key="obs_inter_trimer",   topo=is_inter &  is_trimer),
    "intra_tetramer": dict(obs_key="obs_intra_tetramer", topo=is_intra & ~is_trimer),
    "inter_tetramer": dict(obs_key="obs_inter_tetramer", topo=is_inter & ~is_trimer),
}

# Precompute thresholds once (GLOBAL)
THR = {}
for cat, meta in CATS.items():
    q = load_q(meta["obs_key"])
    THR[cat] = dict(
        thr_lo=np.float32(_get_q_at_p(q, p_grid, P_LOW)),
        thr_hi=np.float32(_get_q_at_p(q, p_grid, P_HIGH)),
    )
print("[FP7aG] Global thresholds loaded from:", fp6b_path.name)
for cat in CATS:
    print(f"  {cat:14s} lo={float(THR[cat]['thr_lo']): .4f}  hi={float(THR[cat]['thr_hi']): .4f}")

# ---------- Output dict ----------
out = {}
params = dict(
    dataset=DATASET,
    fp3_path=str(FP3_PATH),
    fp2_path=str(fp2_path),
    fp6b_path=str(fp6b_path),
    bundle_path=str(bundle_path),
    group_table=str(preproc_data_dir / GROUPING_PKL_NAME),
    p_low=float(P_LOW),
    p_high=float(P_HIGH),
    A=int(A),
    E=int(E),
    K=int(K),
    R=int(R),
    n_modules=int(M),
    categories=list(CATS.keys()),
    save_k_level=bool(SAVE_K_LEVEL),
    groups_base=[str(x) for x in groups_base.tolist()],
    groups_union=[str(x) for x in groups_union.tolist()],
    union_policy="sex=both union over F+M, keep age+geno",
    thresholds_policy="GLOBAL thresholds from FP6b (ref distribution), applied to all groups",
)
out["params_json"] = json.dumps(params, sort_keys=True)

# =========================
# Groupwise attribution
# =========================
for mode, g in GROUP_SPECS:
    if mode == "base":
        gmask = (group_labels == g)
    elif mode == "union_sex_both":
        gmask = (group_labels_union == g)
    else:
        raise RuntimeError(mode)

    idx_animals = np.where(gmask)[0]
    Ag = int(idx_animals.size)
    if Ag == 0:
        continue

    gsafe = _sanitize_group_name(g)
    print(f"[FP7aG] group {mode} | {g} | A={Ag} -> key={gsafe}")

    # For each category: count events and participation
    for cat, meta in CATS.items():
        topo = meta["topo"]  # (K,)
        thr_lo = THR[cat]["thr_lo"]
        thr_hi = THR[cat]["thr_hi"]

        # outputs
        if SAVE_K_LEVEL:
            cnt_hi_k_evt = np.zeros(K, dtype=np.int64)
            cnt_lo_k_evt = np.zeros(K, dtype=np.int64)
            cnt_hi_k_anim = np.zeros(K, dtype=np.int32)
            cnt_lo_k_anim = np.zeros(K, dtype=np.int32)

        cnt_hi_mm_evt = np.zeros((M, M), dtype=np.int64)
        cnt_lo_mm_evt = np.zeros((M, M), dtype=np.int64)
        cnt_hi_r_evt  = np.zeros(R, dtype=np.int64)
        cnt_lo_r_evt  = np.zeros(R, dtype=np.int64)

        cnt_hi_mm_anim = np.zeros((M, M), dtype=np.int32)
        cnt_lo_mm_anim = np.zeros((M, M), dtype=np.int32)
        cnt_hi_r_anim  = np.zeros(R, dtype=np.int32)
        cnt_lo_r_anim  = np.zeros(R, dtype=np.int32)

        # normalization denominator: number of finite topo entries across group×K
        n_values_possible = 0
        n_events_hi = 0
        n_events_lo = 0

        for a in idx_animals:
            x = mc_val[a]  # (K,)
            finite = np.isfinite(x)
            m = topo & finite
            n_values_possible += int(m.sum())

            hit_hi = m & (x >= thr_hi)
            hit_lo = m & (x <= thr_lo)

            # event totals (scalar)
            n_events_hi += int(hit_hi.sum())
            n_events_lo += int(hit_lo.sum())

            # per-k
            if SAVE_K_LEVEL:
                cnt_hi_k_evt += hit_hi.astype(np.int64)
                cnt_lo_k_evt += hit_lo.astype(np.int64)
                cnt_hi_k_anim[hit_hi] += 1
                cnt_lo_k_anim[hit_lo] += 1

            # event counts into ROI/module bins
            _accum_modulepair_counts(cnt_hi_mm_evt, comm_edges, mc_idx, hit_hi)
            _accum_modulepair_counts(cnt_lo_mm_evt, comm_edges, mc_idx, hit_lo)
            _accum_region_counts(cnt_hi_r_evt, fc_edge_idx, mc_idx, hit_hi)
            _accum_region_counts(cnt_lo_r_evt, fc_edge_idx, mc_idx, hit_lo)

            # animal participation into ROI/module bins:
            # define participation as ">=1 event for that bin in this animal"
            # implement by unique-ing the affected bins:
            kk = np.where(hit_hi)[0]
            if kk.size:
                e1h = mc_idx[kk, 0]; e2h = mc_idx[kk, 1]
                mh1 = comm_edges[e1h]; mh2 = comm_edges[e2h]
                pairs = np.unique(np.stack([mh1, mh2], axis=1), axis=0)
                cnt_hi_mm_anim[pairs[:, 0], pairs[:, 1]] += 1
                cnt_hi_mm_anim[pairs[:, 1], pairs[:, 0]] += 1

                rh = np.unique(np.concatenate([fc_edge_idx[e1h].ravel(), fc_edge_idx[e2h].ravel()]).astype(np.int64))
                cnt_hi_r_anim[rh] += 1

            kk = np.where(hit_lo)[0]
            if kk.size:
                e1l = mc_idx[kk, 0]; e2l = mc_idx[kk, 1]
                ml1 = comm_edges[e1l]; ml2 = comm_edges[e2l]
                pairs = np.unique(np.stack([ml1, ml2], axis=1), axis=0)
                cnt_lo_mm_anim[pairs[:, 0], pairs[:, 1]] += 1
                cnt_lo_mm_anim[pairs[:, 1], pairs[:, 0]] += 1

                rl = np.unique(np.concatenate([fc_edge_idx[e1l].ravel(), fc_edge_idx[e2l].ravel()]).astype(np.int64))
                cnt_lo_r_anim[rl] += 1

        # normalized rates
        ev_rate_hi = (n_events_hi / n_values_possible) if n_values_possible > 0 else np.nan
        ev_rate_lo = (n_events_lo / n_values_possible) if n_values_possible > 0 else np.nan

        prefix = f"g={gsafe}__{cat}"
        out[f"{prefix}__group_label"] = np.array([g], dtype=object)
        out[f"{prefix}__group_mode"]  = np.array([mode], dtype=object)

        out[f"{prefix}__thr_lo"] = np.float32(thr_lo)
        out[f"{prefix}__thr_hi"] = np.float32(thr_hi)

        out[f"{prefix}__n_animals"] = np.int32(Ag)
        out[f"{prefix}__n_values_possible"] = np.int64(n_values_possible)
        out[f"{prefix}__n_events_hi"] = np.int64(n_events_hi)
        out[f"{prefix}__n_events_lo"] = np.int64(n_events_lo)
        out[f"{prefix}__event_rate_hi"] = np.float32(ev_rate_hi)
        out[f"{prefix}__event_rate_lo"] = np.float32(ev_rate_lo)

        out[f"{prefix}__count_hi_mm_evt"]  = cnt_hi_mm_evt
        out[f"{prefix}__count_lo_mm_evt"]  = cnt_lo_mm_evt
        out[f"{prefix}__count_hi_r_evt"]   = cnt_hi_r_evt
        out[f"{prefix}__count_lo_r_evt"]   = cnt_lo_r_evt

        out[f"{prefix}__count_hi_mm_anim"] = cnt_hi_mm_anim
        out[f"{prefix}__count_lo_mm_anim"] = cnt_lo_mm_anim
        out[f"{prefix}__count_hi_r_anim"]  = cnt_hi_r_anim
        out[f"{prefix}__count_lo_r_anim"]  = cnt_lo_r_anim

        if SAVE_K_LEVEL:
            out[f"{prefix}__count_hi_k_evt"]  = cnt_hi_k_evt
            out[f"{prefix}__count_lo_k_evt"]  = cnt_lo_k_evt
            out[f"{prefix}__count_hi_k_anim"] = cnt_hi_k_anim
            out[f"{prefix}__count_lo_k_anim"] = cnt_lo_k_anim

# =========================
# Save
# =========================
out_path = dist_dir / OUT_NAME
if out_path.exists() and not OVERWRITE:
    raise FileExistsError(out_path)

np.savez_compressed(out_path, **out)
print("[OK] Saved FP7aG:", out_path)
# %%
