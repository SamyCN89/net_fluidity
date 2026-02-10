#!/usr/bin/env python3
# %%
"""
FP6b_multi — Multi-scheme bootstrap of MC distributions.

- Animal-bootstrap for single-base groups
- Group-bootstrap for pooled groups
- Supports age / geno / sex / phenotype

Consumes:
  fp6a_groups_mc_by_topology_per_animal.npz

Produces:
  fp6b_groups__<scheme>.npz
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from shared_code.fun_paths import get_paths

# ======================================================
# CONFIG
# ======================================================

DATASET = "ines_abdallah"
TIMECOURSE_FOLDER = "Timecourses_updated_03052024"
COGNITIVE_FILE = "ROIs.xlsx"
ANAT_LABELS_FILE = "41_Allen.txt"

MC_DIST_DIRNAME = "mc_dist"
FP6AG_NAME = "fp6a_groups_mc_by_topology_per_animal.npz"

OVERWRITE = True

N_BOOT = 2000
SEED = 0
BOOT_DRAWS: Optional[int] = 300_000

P_GRID = np.linspace(0, 1, 101).astype(np.float32)

BINS_MIN = -0.8
BINS_MAX = 0.8
NBINS = 401

N_JOBS = -1

# Base conditions
CONDS = [
    "obs_intra_trimer","obs_inter_trimer","obs_intra_tetramer","obs_inter_tetramer",
    "null_intra_trimer","null_inter_trimer","null_intra_tetramer","null_inter_tetramer",
    "obs_all","null_all","obs_intra","obs_inter","obs_trimer","obs_tetramer",
    "null_intra","null_inter","null_trimer","null_tetramer"
]

# Standard schemes
SCHEMES = {
    "by_sex": ("sex",),
    "by_age": ("age",),
    "by_geno": ("geno",),
    "by_age_sex": ("age","sex"),
    "by_age_geno": ("age","geno"),
    "by_age_sex_geno": ("age","sex","geno"),
}

# Phenotype columns
PHENO_COLS = {
    "pheno_oip": "Phenotype_OiP",
    "pheno_ro24h": "Phenotype_RO24h",
}

PHENO_SCHEMES = {
    "by_pheno": ("phenotype",),
    "by_age_pheno": ("age","phenotype"),
    "by_sex_pheno": ("sex","phenotype"),
    "by_age_sex_pheno": ("age","sex","phenotype"),
}

# ======================================================
# Helpers
# ======================================================

def _as_float_1d(x):
    x = np.asarray(x)
    if x.size == 0:
        return np.array([], dtype=np.float32)
    x = x.astype(np.float32, copy=False).ravel()
    return x[np.isfinite(x)]


def _pool_concat(obj_arr):
    parts = []
    for v in obj_arr:
        x = _as_float_1d(v)
        if x.size:
            parts.append(x)
    return np.concatenate(parts) if parts else np.array([], dtype=np.float32)


def _summaries(x, bins, p):
    if x.size == 0:
        return (
            np.full(bins.size-1, np.nan, np.float32),
            np.full(p.size, np.nan, np.float32),
        )

    counts, _ = np.histogram(x, bins=bins, density=False)
    w = np.diff(bins)

    pdf = counts / (x.size * w)
    q = np.quantile(x, p)

    return pdf.astype(np.float32), q.astype(np.float32)


# ======================================================
# Bootstrap units
# ======================================================

def _boot_animals(obj_arr, rng):

    A = obj_arr.size
    pick = rng.integers(0, A, size=A)

    parts = []
    for i in pick:
        x = _as_float_1d(obj_arr[int(i)])
        if x.size:
            parts.append(x)

    if not parts:
        return np.array([], dtype=np.float32)

    x = np.concatenate(parts)

    if BOOT_DRAWS and x.size > BOOT_DRAWS:
        jj = rng.integers(0, x.size, BOOT_DRAWS)
        x = x[jj]

    return x


def _boot_groups(member_arrays, rng):

    G = len(member_arrays)
    pick = rng.integers(0, G, size=G)

    parts = []

    for gi in pick:
        obj = member_arrays[int(gi)]
        for v in obj:
            x = _as_float_1d(v)
            if x.size:
                parts.append(x)

    if not parts:
        return np.array([], dtype=np.float32)

    x = np.concatenate(parts)

    if BOOT_DRAWS and x.size > BOOT_DRAWS:
        jj = rng.integers(0, x.size, BOOT_DRAWS)
        x = x[jj]

    return x


# ======================================================
# FP6a access
# ======================================================

def _cat(a, b):

    out = np.empty(a.size, dtype=object)

    for i in range(a.size):

        xa = _as_float_1d(a[i])
        xb = _as_float_1d(b[i])

        if xa.size and xb.size:
            out[i] = np.concatenate([xa, xb])
        elif xa.size:
            out[i] = xa
        elif xb.size:
            out[i] = xb
        else:
            out[i] = np.array([], dtype=np.float32)

    return out


def get_fp6a_obj(z, g, key):

    if key in (
        "obs_all","null_all","obs_intra","obs_inter","obs_trimer","obs_tetramer",
        "null_intra","null_inter","null_trimer","null_tetramer"
    ):

        o_it = z[f"{g}__obs_intra_trimer"]
        o_et = z[f"{g}__obs_inter_trimer"]
        o_ia = z[f"{g}__obs_intra_tetramer"]
        o_ea = z[f"{g}__obs_inter_tetramer"]

        n_it = z[f"{g}__null_intra_trimer"]
        n_et = z[f"{g}__null_inter_trimer"]
        n_ia = z[f"{g}__null_intra_tetramer"]
        n_ea = z[f"{g}__null_inter_tetramer"]

        if key == "obs_all": return _cat(_cat(o_it,o_et), _cat(o_ia,o_ea))
        if key == "null_all": return _cat(_cat(n_it,n_et), _cat(n_ia,n_ea))
        if key == "obs_intra": return _cat(o_it,o_ia)
        if key == "obs_inter": return _cat(o_et,o_ea)
        if key == "obs_trimer": return _cat(o_it,o_et)
        if key == "obs_tetramer": return _cat(o_ia,o_ea)
        if key == "null_intra": return _cat(n_it,n_ia)
        if key == "null_inter": return _cat(n_et,n_ea)
        if key == "null_trimer": return _cat(n_it,n_et)
        if key == "null_tetramer": return _cat(n_ia,n_ea)

    return z[f"{g}__{key}"]


# ======================================================
# Group parsing
# ======================================================

def parse_group(g):

    kv = {}

    for p in g.split("|"):
        k,v = p.split("=")
        kv[k] = v

    return kv


def make_group(kv, keys):

    return "|".join([f"{k}={kv[k]}" for k in keys])


def pooled_groups(groups_raw, keys):

    mapping = {}

    for g in groups_raw:
        kv = parse_group(g)
        pg = make_group(kv, keys)
        mapping.setdefault(pg, []).append(g)

    return sorted(mapping), mapping



# ======================================================
# MAIN
# ======================================================
bins = np.linspace(BINS_MIN,BINS_MAX,NBINS, dtype=np.float32)

paths = get_paths(DATASET,TIMECOURSE_FOLDER,COGNITIVE_FILE,ANAT_LABELS_FILE)

mc_dir = Path(paths["mc"])
dist_dir = mc_dir / MC_DIST_DIRNAME

z = np.load(dist_dir / FP6AG_NAME, allow_pickle=True)

groups_raw = [str(g) for g in z["groups"]]


# Load CSV
cog_path = sorted(Path(paths["preprocessed"]).glob("cog_data_filtered_*.csv"))[-1]
df = pd.read_csv(cog_path)

df["sex"] = df["Sexe"].astype(str).str.upper().str[0]
df["geno"] = df["Genotype"].astype(str)

# ======================================================
# Run schemes
# ======================================================

def run_scheme(tag, keys, pheno_col=None):
    """
    tag: output tag
    keys: tuple of fields to keep in pooled group name (age/geno/sex or age/sex/phenotype)
    pheno_col: if not None, use this CSV column to define phenotype labels and filter animals
    """

    # --- build base mapping: pooled_group -> list of base FP6a groups (always age×geno×sex) ---
    groups_base = groups_raw  # FP6a truth
    groups, mapping = pooled_groups(groups_base, tuple(k for k in keys if k != "phenotype"))

    # If no phenotype involved: mapping is final
    if pheno_col is None:
        def members_for(pg):
            return mapping[pg]

        def filter_mask_for_base_group(g, target_pheno=None):
            return None  # no filtering

        pheno_levels = None

    else:
        # phenotype mapping from CSV
        if pheno_col not in df.columns:
            raise KeyError(f"CSV missing phenotype column: {pheno_col}")

        ph_map = dict(zip(df["Name"].astype(str), df[pheno_col].astype(str)))

        # discover phenotype levels present in the data (stable order)
        pheno_levels = sorted({str(v) for v in ph_map.values() if str(v) != "nan"})

        # now define pooled groups including phenotype in the label
        # for each non-phenotype pooled group pg0, split into pg0|phenotype=<level>
        groups_ph = []
        mapping_ph = {}

        for pg0, members in mapping.items():
            # only include phenotype levels that actually appear inside these members
            seen = set()
            for g in members:
                ids_key = f"{g}__animal_ids"
                if ids_key not in z.files:
                    raise KeyError(f"Missing {ids_key} in FP6a. Re-run FP6aG with animal_ids patch.")
                ids = np.asarray(z[ids_key], dtype=object)
                for mid in ids:
                    p = ph_map.get(str(mid), None)
                    if p is None or p == "nan":
                        continue
                    seen.add(str(p))

            for p in sorted(seen):
                pg = f"{pg0}|phenotype={p}" if pg0 else f"phenotype={p}"
                groups_ph.append(pg)
                mapping_ph[pg] = members  # same base members; filtering will choose phenotype within them

        groups = sorted(groups_ph)
        mapping = mapping_ph

        def members_for(pg):
            return mapping[pg]

        def filter_mask_for_base_group(g, target_pheno: str):
            ids = np.asarray(z[f"{g}__animal_ids"], dtype=object)
            m = np.array([ph_map.get(str(mid), "nan") == target_pheno for mid in ids], dtype=bool)
            return m

    # --- build jobs ---
    jobs = []
    for gi, pg in enumerate(groups):
        for ci, key in enumerate(CONDS):
            seed = SEED + 10_000*gi + 100*ci
            jobs.append((pg, key, seed))

    def summarize_one(pg: str, key: str, seed: int):
        members = members_for(pg)

        # phenotype target (if used)
        target_ph = None
        if pheno_col is not None:
            # parse from "...|phenotype=XYZ"
            parts = pg.split("|")
            for part in parts:
                if part.startswith("phenotype="):
                    target_ph = part.split("=", 1)[1]
                    break
            if target_ph is None:
                raise RuntimeError(f"Internal: phenotype group missing phenotype= in '{pg}'")

        # collect member arrays (possibly filtered by phenotype)
        member_arrays = []
        for g in members:
            obj = get_fp6a_obj(z, g, key)

            if pheno_col is not None:
                m = filter_mask_for_base_group(g, target_ph)
                # filter per-animal object array
                obj = obj[m]

            member_arrays.append(obj)

        # observed pool
        parts = []
        n_animals_total = 0
        for arr in member_arrays:
            n_animals_total += int(arr.size)
            for v in arr:
                x = _as_float_1d(v)
                if x.size:
                    parts.append(x)
        x_obs = np.concatenate(parts) if parts else np.array([], dtype=np.float32)

        pdf_obs, q_obs = _summaries(x_obs, bins, P_GRID)

        # choose bootstrap unit
        # if more than 1 member group contributes -> group-bootstrap
        # else -> animal-bootstrap
        use_group_boot = (len(members) > 1)

        nb = bins.size - 1
        nq = P_GRID.size
        pdf_boot = np.empty((N_BOOT, nb), dtype=np.float32)
        q_boot = np.empty((N_BOOT, nq), dtype=np.float32)

        rng = np.random.default_rng(seed)
        for b in range(N_BOOT):
            if use_group_boot:
                xb = _boot_groups(member_arrays, rng)
            else:
                xb = _boot_animals(member_arrays[0], rng)
            pdf_boot[b], q_boot[b] = _summaries(xb, bins, P_GRID)

        return dict(
            g=pg, key=key,
            boot_unit="group" if use_group_boot else "animal",
            n_members=len(members),
            n_animals=np.int32(n_animals_total),
            n_pool=np.int64(x_obs.size),
            pdf_obs=pdf_obs,
            pdf_ci_lo=np.quantile(pdf_boot, 0.025, axis=0).astype(np.float32),
            pdf_ci_hi=np.quantile(pdf_boot, 0.975, axis=0).astype(np.float32),
            q_obs=q_obs,
            q_ci_lo=np.quantile(q_boot, 0.025, axis=0).astype(np.float32),
            q_ci_hi=np.quantile(q_boot, 0.975, axis=0).astype(np.float32),
        )

    results = Parallel(n_jobs=N_JOBS, backend="threading")(
        delayed(summarize_one)(pg, key, seed) for (pg, key, seed) in jobs
    )

    out = {
        "bins": bins,
        "p_grid": P_GRID.astype(np.float32),
        "groups": np.array(groups, dtype=object),
    }

    boot_units = {}
    for r in results:
        g = r["g"]; key = r["key"]
        boot_units[g] = r["boot_unit"]
        for f in ["pdf_obs","pdf_ci_lo","pdf_ci_hi","q_obs","q_ci_lo","q_ci_hi","n_animals","n_pool","n_members"]:
            out[f"{g}__{key}__{f}"] = r[f]

    params = dict(
        dataset=DATASET,
        scheme=tag,
        keys=list(keys),
        pheno_col=pheno_col,
        n_boot=int(N_BOOT),
        seed=int(SEED),
        boot_draws=None if BOOT_DRAWS is None else int(BOOT_DRAWS),
        bins=[float(BINS_MIN), float(BINS_MAX), int(NBINS)],
        conditions=CONDS,
        source_fp6a=str(dist_dir / FP6AG_NAME),
        boot_units=boot_units,
        note="animal-bootstrap for single-base groups; group-bootstrap for pooled groups; phenotype uses per-animal filtering via FP6a __animal_ids + CSV",
    )
    out["params_json"] = json.dumps(params, sort_keys=True)

    out_path = dist_dir / f"fp6b_groups__{tag}.npz"
    if out_path.exists() and not OVERWRITE:
        raise FileExistsError(out_path)
    np.savez_compressed(out_path, **out)
    print("[OK] Saved:", out_path.name)


# ======================================================
# Run all
# ======================================================

# Standard
for tag,keys in SCHEMES.items():
    run_scheme(tag,keys)

# Phenotype
for ph_tag, col in PHENO_COLS.items():
    for tag,keys in PHENO_SCHEMES.items():
        run_scheme(f"{tag}_{ph_tag}", keys, col)

print("[DONE] FP6b multi")
