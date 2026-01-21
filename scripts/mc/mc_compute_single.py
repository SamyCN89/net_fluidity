#!/usr/bin/env python3
# %%
"""
MC_00_freeze_mc.py

Finish point A:
- compute MC (animals, E, E)
- sanity checks (diag/off-diag)

Finish point B:
- load allegiance cache (preferred) OR compute if missing
- build module mask + trimer index
- save ONE frozen artifact to results/<dataset>/mc_frozen/

Run as blocks in VSCode (#%%) or as a script:
    python scripts/mc/MC_00_freeze_mc.py
"""
#%%
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import joblib


from shared_code.fun_loaddata import load_timeseries_bundle
from shared_code.fun_metaconnectivity import (
    compute_metaconnectivity,
    fun_allegiance_communities,
    get_fc_mc_indices,
    get_mc_region_identities,
    intramodule_indices_mask,
    build_trimer_mask,
    compute_trimers_identity,
)
from shared_code.fun_paths import get_paths

# %%
# =========================
# CONFIG
# =========================
DATASET = "ines_abdallah"
TIMECOURSE_FOLDER = "Timecourses_updated_03052024"
COGNITIVE_FILE = "ROIs.xlsx"
ANAT_LABELS_FILE = "41_Allen.txt"

BUNDLE_NPZ = "ts_and_meta_2m4m.npz"
BUNDLE_GROUPING = "grouping_data_oip.pkl"

WINDOW_SIZE = 7
LAG = 1
N_JOBS = -1

# Allegiance (heavy)
N_RUNS_ALLEGIANCE = 1000
GAMMA_PT = 100

# Reference group selection
REF_COL = 2
REF_ROW = 0

# Gates
RUN_HEAVY = True          # set False if you only want MC sanity
SAVE_FROZEN = True        # set False to dry-run heavy steps

RUN_TAG = f"w={WINDOW_SIZE}_lag={LAG}_runs={N_RUNS_ALLEGIANCE}_gamma={GAMMA_PT}_ref={REF_COL}-{REF_ROW}"
print("RUN_TAG:", RUN_TAG)

# %%
# =========================
# LOAD DATA
# =========================
paths = get_paths(
    dataset_name=DATASET,
    timecourse_folder=TIMECOURSE_FOLDER,
    cognitive_data_file=COGNITIVE_FILE,
    anat_labels_file=ANAT_LABELS_FILE,
)

bundle = load_timeseries_bundle(
    paths["preprocessed"] / BUNDLE_NPZ,
    paths["preprocessed"] / BUNDLE_GROUPING,
)

ts = bundle.ts
n_animals = bundle.n_animals
regions = bundle.n_regions
mask_groups = bundle.mask_groups
label_variables = bundle.label_variables

if mask_groups is None or label_variables is None:
    raise ValueError("Grouping data missing: expected mask_groups and label_variables in bundle.")

# label reference is used for allegiance naming only
label_ref = label_variables[REF_COL][REF_ROW]
ind_ref = mask_groups[REF_COL][REF_ROW]

print("ts shape:", np.shape(ts))
print("n_animals:", n_animals, "regions:", regions)
print("Reference label:", label_ref)

# %%
# =========================
# FINISH POINT A — Metaconnectivity computation
# =========================
E_expected = regions * (regions - 1) // 2
print("E (FC edges) expected:", E_expected)

t0 = time.time()
mc = compute_metaconnectivity(
    ts,
    window_size=WINDOW_SIZE,
    lag=LAG,
    n_jobs=N_JOBS,
    save_path=None,     # IMPORTANT: no reports/ saving
)
t1 = time.time()

mc = np.asarray(mc)
print(f"MC shape: {mc.shape} (computed in {t1 - t0:.2f}s)")

#%%
# --- Validate MC shape ---
if mc.shape != (n_animals, E_expected, E_expected):
    raise ValueError(f"Unexpected MC shape {mc.shape}, expected {(n_animals, E_expected, E_expected)}")

finite = np.isfinite(mc)
print("finite fraction:", float(finite.mean()))

diag = np.array([np.diag(mc[a]) for a in range(n_animals)])
print("diag mean ± std:", float(np.nanmean(diag)), float(np.nanstd(diag)))

rng = np.random.default_rng(0)
a = int(rng.integers(0, n_animals))
idx = rng.choice(E_expected, size=300, replace=False)
sub = mc[a][np.ix_(idx, idx)]
off = sub[~np.eye(sub.shape[0], dtype=bool)]
print("off-diag mean ± std:", float(np.nanmean(off)), float(np.nanstd(off)))
print("off-diag min/max:", float(np.nanmin(off)), float(np.nanmax(off)))

if not RUN_HEAVY:
    raise SystemExit("Stopped after Finish point A. Set RUN_HEAVY=True to continue.")

# %%
# =========================
# FINISH POINT B — Allegiance (load-first)
# =========================
mc_ref = np.mean(mc[ind_ref], axis=0)

alleg_dir = Path("reports/metaconnectivity") / paths["results"].name / "allegiance"
candidates = [
    alleg_dir / f"allegiance_{str(label_ref).replace(' ','_')}.joblib",
    alleg_dir / f"allegiance_{str(label_ref).lower().replace(' ','_')}.joblib",
    alleg_dir / "allegiance_wt_2m.joblib",
    alleg_dir / "allegiance_wt_2m_recursive.joblib",
]
cache_path = next((p for p in candidates if p.exists()), None)

if cache_path is not None:
    obj = joblib.load(cache_path)
    if isinstance(obj, dict):
        mc_ref_allegiance_communities = obj.get("mc_ref_allegiance_communities", obj.get("communities"))
        allegiance_sort = obj.get("allegiance_sort", obj.get("sort", obj.get("mc_ref_allegiance_sort")))
        contingency_matrix = obj.get("contingency_matrix", None)
    else:
        mc_ref_allegiance_communities, allegiance_sort, contingency_matrix = obj

    mc_ref_allegiance_communities = np.asarray(mc_ref_allegiance_communities)
    allegiance_sort = np.asarray(allegiance_sort)

    print(f"[OK] Loaded cached allegiance: {cache_path}")
else:
    (mc_ref_allegiance_communities, allegiance_sort, contingency_matrix) = fun_allegiance_communities(
        mc_ref,
        n_runs=N_RUNS_ALLEGIANCE,
        gamma_pt=GAMMA_PT,
        save_path=None,   # IMPORTANT: no reports/ saving
        ref_name=str(label_ref),
        n_jobs=N_JOBS,
    )
    print("[OK] Computed allegiance (no cache found)")

print("allegiance_sort length:", len(allegiance_sort))
print("communities shape:", mc_ref_allegiance_communities.shape)

# %%
# --- Validate allegiance usefulness (cheap, mandatory) ---
def blockiness_score(M: np.ndarray, communities: np.ndarray) -> tuple[float, float]:
    M = M.copy()
    np.fill_diagonal(M, np.nan)
    same = communities[:, None] == communities[None, :]
    intra = np.abs(M[same & ~np.eye(len(M), dtype=bool)])
    inter = np.abs(M[~same])
    return float(np.nanmean(intra)), float(np.nanmean(inter))

intra0, inter0 = blockiness_score(mc_ref, mc_ref_allegiance_communities)

mc_ref_sorted = mc_ref[allegiance_sort][:, allegiance_sort]
comm_sorted = mc_ref_allegiance_communities[allegiance_sort]
intra1, inter1 = blockiness_score(mc_ref_sorted, comm_sorted)

print("Blockiness (mean |MC|):")
print(f"  Unsorted  intra={intra0:.4f}  inter={inter0:.4f}")
print(f"  Sorted    intra={intra1:.4f}  inter={inter1:.4f}")
print(f"  Δ(intra-inter): {(intra1-inter1)-(intra0-inter0):.4f}")

# %%
# =========================
# Build downstream indices (mc_val, modules, trimers)
# =========================
mc_allegiance = mc[:, allegiance_sort][:, :, allegiance_sort]
idx_diag = np.arange(E_expected)
mc_allegiance[..., idx_diag, idx_diag] = np.nan

intramodules_idx, intramodule_indices, mc_modules_mask = intramodule_indices_mask(
    mc_ref_allegiance_communities
)
mc_modules_mask = mc_modules_mask[allegiance_sort][:, allegiance_sort]

fc_idx, mc_idx = get_fc_mc_indices(regions, allegiance_sort=allegiance_sort)
mc_reg_idx, fc_reg_idx = get_mc_region_identities(fc_idx, mc_idx)

mc_val = mc_allegiance[:, mc_idx[:, 0], mc_idx[:, 1]]
mc_mod_idx = mc_modules_mask[mc_idx[:, 0], mc_idx[:, 1]].astype(int)

trimer_index, trimer_reg_id, trimer_apex = compute_trimers_identity(regions)
mc_nplets_mask = build_trimer_mask(trimer_index, trimer_apex, E_expected)
mc_nplets_mask = mc_nplets_mask[allegiance_sort][:, allegiance_sort]
mc_nplets_index = mc_nplets_mask[mc_idx[:, 0], mc_idx[:, 1]]

print("mc_val shape:", mc_val.shape)
print("mc_mod_idx shape:", mc_mod_idx.shape)
print("mc_nplets_index shape:", mc_nplets_index.shape)
print("n trimers:", int(np.sum(mc_nplets_index > 0)), "n tetramers:", int(np.sum(mc_nplets_index == 0)))

# %%
# =========================
# SAVE frozen artifact (ONE file)
# =========================
if SAVE_FROZEN:
    frozen_dir = paths["results"] / "mc_frozen"
    frozen_dir.mkdir(parents=True, exist_ok=True)

    params = dict(
        dataset=paths["results"].name,
        window_size=WINDOW_SIZE,
        lag=LAG,
        n_runs_allegiance=N_RUNS_ALLEGIANCE,
        gamma_pt=GAMMA_PT,
        ref_col=REF_COL,
        ref_row=REF_ROW,
        ref_label=str(label_ref),
        n_animals=int(n_animals),
        n_regions=int(regions),
    )

    out_path = frozen_dir / f"mc_frozen_{RUN_TAG}_animals={n_animals}_regions={regions}.npz"

    np.savez_compressed(
        out_path,
        mc_val_tril=mc_val,
        mc_idx_tril=mc_idx,
        fc_idx_tril=fc_idx,
        mc_mod_idx=mc_mod_idx,
        mc_reg_idx=mc_reg_idx,
        mc_nplets_index=mc_nplets_index,
        allegiance_sort=allegiance_sort,
        mc_ref_allegiance_communities=mc_ref_allegiance_communities,
        mc_modules_mask=mc_modules_mask,
        label_variables=np.array(label_variables, dtype=object),
        mask_groups=np.array(mask_groups, dtype=object),
        params_json=json.dumps(params, sort_keys=True),
    )

    print("[OK] Saved frozen artifact:")
    print(" ", out_path)
else:
    print("[DRY RUN] SAVE_FROZEN=False; nothing was saved.")

# %%
