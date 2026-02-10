#!/usr/bin/env python3
# %%
"""
FP6d — 2D cluster permutation on TRIMER distributions at GROUP level.

Test map:
  T[g, x] computed from bootstrap replicates of ΔPDF = PDF_obs_trimer - PDF_null_trimer

Clustering domain:
  - group dimension: adjacency defined by "one-factor-away" (age/geno/sex/phenotype tokens)
  - x dimension: adjacent bins

Permutation:
  - sign-flip (one-sample) across bootstrap replicates, per group×bin
    (common cluster-permutation for one-sample effect maps)

Consumes:
  mc_dist/fp6a_groups_mc_by_topology_per_animal.npz
  mc_dist/fp6b_groups__<scheme>.npz  (only for the pooled group list + scheme keys)

Produces:
  mc_dist/fp6d_2dcluster__<scheme>__trimer_obs_minus_null.npz
  fig/<dataset>/mc/FP6d/<scheme>/
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

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
FP6BG_SCHEME = "fp6b_groups__by_age_geno.npz"  # <-- choose scheme

# bootstrap used to estimate ΔPDF distribution
N_BOOT = 2000
SEED = 0
BOOT_DRAWS: Optional[int] = 300_000  # None = full pooled

# histogram bins (must match FP6b)
BINS_MIN = -0.8
BINS_MAX = 0.8
NBINS = 401

# permutation
N_PERM = 2000          # can match N_BOOT
T_THRESH = 2.0         # cluster-forming threshold on |T|
TWO_SIDED = True       # cluster on abs(T)

# plotting
SAVE_FIG = True
DPI = 200
Y_LOG = False  # this is for T map plots, not PDFs

# groups to include (None = all groups present in scheme file)
INCLUDE_GROUPS: Optional[List[str]] = None


# ======================================================
# Helpers: FP6a pooling & bootstrap
# ======================================================
def _as_float_1d(x) -> np.ndarray:
    x = np.asarray(x)
    if x.size == 0:
        return np.array([], dtype=np.float32)
    x = x.astype(np.float32, copy=False).ravel()
    return x[np.isfinite(x)]

def _cat_obj_per_animal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Concatenate two per-animal object arrays into one per-animal object array."""
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

def get_fp6a_trimer_obj(z6a, base_group: str, which: str) -> np.ndarray:
    """
    Return per-animal object array for TRIMER marginal:
      which="obs" -> cat(obs_intra_trimer, obs_inter_trimer)
      which="null"-> cat(null_intra_trimer, null_inter_trimer)
    """
    if which == "obs":
        a = z6a[f"{base_group}__obs_intra_trimer"]
        b = z6a[f"{base_group}__obs_inter_trimer"]
        return _cat_obj_per_animal(a, b)
    if which == "null":
        a = z6a[f"{base_group}__null_intra_trimer"]
        b = z6a[f"{base_group}__null_inter_trimer"]
        return _cat_obj_per_animal(a, b)
    raise ValueError(which)

def pool_concat(obj_arr: np.ndarray) -> np.ndarray:
    parts = []
    for v in obj_arr:
        x = _as_float_1d(v)
        if x.size:
            parts.append(x)
    return np.concatenate(parts, axis=0) if parts else np.array([], dtype=np.float32)

def summaries_pdf(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.full(bins.size - 1, np.nan, np.float32)
    counts, _ = np.histogram(x, bins=bins, density=False)
    widths = np.diff(bins).astype(np.float32)
    pdf = counts.astype(np.float32) / (x.size * widths)
    return pdf.astype(np.float32)

def boot_animals(obj_arr: np.ndarray, rng: np.random.Generator, boot_draws: Optional[int]) -> np.ndarray:
    A = obj_arr.size
    pick = rng.integers(0, A, size=A)  # resample animals with replacement
    parts = []
    for i in pick:
        x = _as_float_1d(obj_arr[int(i)])
        if x.size:
            parts.append(x)
    if not parts:
        return np.array([], dtype=np.float32)
    x = np.concatenate(parts, axis=0)
    if boot_draws is not None and x.size > boot_draws:
        jj = rng.integers(0, x.size, size=boot_draws)
        x = x[jj]
    return x

def boot_groups(member_obj_arrays: List[np.ndarray], rng: np.random.Generator, boot_draws: Optional[int]) -> np.ndarray:
    """
    member_obj_arrays: list of per-animal object arrays, one per base group in the pooled group.
    Group-bootstrap: resample base groups with replacement, then pool all animals/values.
    """
    G = len(member_obj_arrays)
    pick = rng.integers(0, G, size=G)
    parts = []
    for gi in pick:
        obj = member_obj_arrays[int(gi)]
        for v in obj:
            x = _as_float_1d(v)
            if x.size:
                parts.append(x)
    if not parts:
        return np.array([], dtype=np.float32)
    x = np.concatenate(parts, axis=0)
    if boot_draws is not None and x.size > boot_draws:
        jj = rng.integers(0, x.size, size=boot_draws)
        x = x[jj]
    return x


# ======================================================
# Helpers: group parsing & adjacency
# ======================================================
def parse_group_tokens(g: str) -> Dict[str, str]:
    kv = {}
    for part in g.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            kv[k] = v
    return kv

def build_group_adjacency(groups: List[str]) -> List[List[int]]:
    """
    Neighbors if they differ by exactly ONE token value (e.g., age or geno).
    Works for groups like:
      age=2m|geno=wt
      age=4m|geno=dKI
    """
    toks = [parse_group_tokens(g) for g in groups]
    keys = sorted(set().union(*[set(t.keys()) for t in toks]))
    adj = [[] for _ in groups]

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            di = 0
            for k in keys:
                vi = toks[i].get(k, None)
                vj = toks[j].get(k, None)
                if vi != vj:
                    di += 1
            if di == 1:
                adj[i].append(j)
                adj[j].append(i)
    return adj

def find_2d_clusters(mask: np.ndarray, group_adj: List[List[int]]) -> List[List[Tuple[int, int]]]:
    """
    mask shape (G, X). Clusters are connected components using:
      - x adjacency: (x±1) within same group
      - group adjacency: neighbors in group graph at same x
    Returns clusters as list of list of (g,x) coordinates.
    """
    G, X = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    clusters = []

    for g0 in range(G):
        for x0 in range(X):
            if not mask[g0, x0] or visited[g0, x0]:
                continue
            # BFS
            stack = [(g0, x0)]
            visited[g0, x0] = True
            coords = []
            while stack:
                g, x = stack.pop()
                coords.append((g, x))

                # x neighbors
                for xn in (x - 1, x + 1):
                    if 0 <= xn < X and mask[g, xn] and not visited[g, xn]:
                        visited[g, xn] = True
                        stack.append((g, xn))

                # group neighbors (same x)
                for gn in group_adj[g]:
                    if mask[gn, x] and not visited[gn, x]:
                        visited[gn, x] = True
                        stack.append((gn, x))

            clusters.append(coords)
    return clusters

#%%
# ======================================================
# Main: build ΔPDF bootstrap cube, then cluster-permutation
# ======================================================
paths = get_paths(DATASET, TIMECOURSE_FOLDER, COGNITIVE_FILE, ANAT_LABELS_FILE)
mc_dir = Path(paths["mc"])
dist_dir = mc_dir / MC_DIST_DIRNAME

# sanity check input file
fp6a_path = dist_dir / FP6AG_NAME
fp6b_path = dist_dir / FP6BG_SCHEME

# sanity check input files
if not fp6a_path.exists():
    raise FileNotFoundError(fp6a_path)
if not fp6b_path.exists():
    raise FileNotFoundError(fp6b_path)

# Load FP6a and FP6b data for group membership & pooling
z6a = np.load(fp6a_path, allow_pickle=True)
z6b = np.load(fp6b_path, allow_pickle=True)

# Extract scheme name and pooled groups from FP6b file (we only use it for the pooled group list + scheme name)
scheme_params = json.loads(z6b["params_json"].item())
scheme_name = scheme_params.get("scheme", fp6b_path.stem.replace("fp6b_groups__", ""))

pooled_groups = [str(g) for g in z6b["groups"]]
if INCLUDE_GROUPS is not None:
    keep = set(INCLUDE_GROUPS)
    pooled_groups = [g for g in pooled_groups if g in keep]

# Build mapping pooled_group -> base FP6a groups by matching tokens
# (Because FP6b already produced pooled groups; we reconstruct membership using FP6a base groups list.)
base_groups = [str(g) for g in z6a["groups"]]
base_tokens = [parse_group_tokens(g) for g in base_groups]
pooled_tokens = [parse_group_tokens(g) for g in pooled_groups]

def members_for_pooled(pg: str) -> List[str]:
    """
    Return list of FP6a base groups that are members of the pooled group pg, by matching tokens.
    E.g ., if pg has tokens age=2m|geno=wt, then we find all base groups that have age=2m and geno=wt (regardless of other tokens
    """
    pt = parse_group_tokens(pg)
    members = []
    for bg, bt in zip(base_groups, base_tokens):
        ok = True
        for k, v in pt.items():
            if bt.get(k, None) != v:
                ok = False
                break
        if ok:
            members.append(bg)
    return members

members_map: Dict[str, List[str]] = {pg: members_for_pooled(pg) for pg in pooled_groups}

# sanity
for pg in pooled_groups:
    if len(members_map[pg]) == 0:
        raise RuntimeError(f"No FP6a base groups found for pooled group: {pg}")

bins = np.linspace(BINS_MIN, BINS_MAX, NBINS, dtype=np.float32)
X = bins.size - 1
G = len(pooled_groups)

print("[FP6d] scheme:", scheme_name)
print("[FP6d] pooled groups:", G)
for pg in pooled_groups:
    print("  ", pg, "members:", len(members_map[pg]))

# adjacency across pooled groups
group_adj = build_group_adjacency(pooled_groups)

# ------------------------------------------------------
# Build bootstrap cube of ΔPDF: shape (B, G, X)
# ------------------------------------------------------
delta_boot = np.empty((N_BOOT, G, X), dtype=np.float32)
delta_obs = np.empty((G, X), dtype=np.float32)

for gi, pg in enumerate(pooled_groups):
    members = members_map[pg]

    # build member per-animal arrays for obs/null trimers
    obs_members = [get_fp6a_trimer_obj(z6a, bg, "obs") for bg in members]
    nul_members = [get_fp6a_trimer_obj(z6a, bg, "null") for bg in members]

    # observed pooled PDFs
    x_obs = np.concatenate([pool_concat(a) for a in obs_members]) if obs_members else np.array([], np.float32)
    x_nul = np.concatenate([pool_concat(a) for a in nul_members]) if nul_members else np.array([], np.float32)
    pdf_obs = summaries_pdf(x_obs, bins)
    pdf_nul = summaries_pdf(x_nul, bins)
    delta_obs[gi] = pdf_obs - pdf_nul

    # bootstrap unit: pooled group -> group-bootstrap if >1 base member, else animal-bootstrap
    use_group_boot = (len(members) > 1)

    rng = np.random.default_rng(SEED + 1000 * gi)
    for b in range(N_BOOT):
        if use_group_boot:
            xb_obs = boot_groups(obs_members, rng, BOOT_DRAWS)
            xb_nul = boot_groups(nul_members, rng, BOOT_DRAWS)
        else:
            xb_obs = boot_animals(obs_members[0], rng, BOOT_DRAWS)
            xb_nul = boot_animals(nul_members[0], rng, BOOT_DRAWS)

        pdfb_obs = summaries_pdf(xb_obs, bins)
        pdfb_nul = summaries_pdf(xb_nul, bins)
        delta_boot[b, gi] = pdfb_obs - pdfb_nul

print("[FP6d] delta_boot:", delta_boot.shape)

# ------------------------------------------------------
# Build observed T map from bootstrap distribution
# ------------------------------------------------------
mu = np.nanmean(delta_boot, axis=0)              # (G,X)
sd = np.nanstd(delta_boot, axis=0, ddof=1)       # (G,X)
T_obs = mu / (sd + 1e-12)

# ------------------------------------------------------
# Cluster-forming threshold & clusters on observed map
# ------------------------------------------------------
if TWO_SIDED:
    supra = np.abs(T_obs) > T_THRESH
else:
    supra = T_obs > T_THRESH

clusters = find_2d_clusters(supra, group_adj)

def cluster_mass(T: np.ndarray, coords: List[Tuple[int,int]]) -> float:
    vals = np.array([T[g, x] for (g, x) in coords], dtype=np.float32)
    return float(np.nansum(np.abs(vals) if TWO_SIDED else vals))

obs_masses = np.array([cluster_mass(T_obs, c) for c in clusters], dtype=np.float32)

# ------------------------------------------------------
# Permutation via sign-flip of bootstrap replicates
# ------------------------------------------------------
# We treat delta_boot replicates as exchangeable draws around 0 under H0.
# For each perm, flip sign per replicate (same flip applied to all (g,x) for that replicate).
rngp = np.random.default_rng(SEED + 999_999)
max_masses = np.zeros(N_PERM, dtype=np.float32)

for p in range(N_PERM):
    flips = rngp.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(N_BOOT, 1, 1))
    db = delta_boot * flips
    mu_p = np.nanmean(db, axis=0)
    sd_p = np.nanstd(db, axis=0, ddof=1)
    T_p = mu_p / (sd_p + 1e-12)

    if TWO_SIDED:
        supra_p = np.abs(T_p) > T_THRESH
    else:
        supra_p = T_p > T_THRESH

    cl_p = find_2d_clusters(supra_p, group_adj)
    if cl_p:
        max_masses[p] = max(cluster_mass(T_p, c) for c in cl_p)
    else:
        max_masses[p] = 0.0

# p-values for each observed cluster (max-stat correction)

pvals = np.array(
    [(1 + np.sum(max_masses >= m)) / (N_PERM + 1) for m in obs_masses],
    dtype=np.float32
)


print("[FP6d] clusters found:", len(clusters))
for i, (m, pv) in enumerate(zip(obs_masses, pvals)):
    print(f"  - cluster#{i:02d}: mass={m:.3f}  p={pv:.4f}  size={len(clusters[i])}")

# ------------------------------------------------------
# Save
# ------------------------------------------------------
out_npz = dist_dir / f"fp6d_2dcluster__{scheme_name}__trimer_obs_minus_null.npz"

out = dict(
    bins=bins,
    groups=np.array(pooled_groups, dtype=object),
    T_obs=T_obs.astype(np.float32),
    delta_obs=delta_obs.astype(np.float32),
    t_thresh=np.float32(T_THRESH),
    two_sided=np.int8(1 if TWO_SIDED else 0),
    n_boot=np.int32(N_BOOT),
    n_perm=np.int32(N_PERM),
    max_masses=max_masses.astype(np.float32),
    n_clusters=np.int32(len(clusters)),
    cluster_masses=obs_masses.astype(np.float32),
    cluster_pvals=pvals.astype(np.float32),
)

# store clusters as ragged lists -> save as object arrays
out["clusters_coords"] = np.array([np.array(c, dtype=np.int32) for c in clusters], dtype=object)

params = dict(
    dataset=DATASET,
    scheme=scheme_name,
    fp6a=str(fp6a_path),
    fp6b_scheme=str(fp6b_path),
    note_test="2D cluster permutation on ΔPDF = obs_trimer - null_trimer (group-level pooled)",
    boot_draws=None if BOOT_DRAWS is None else int(BOOT_DRAWS),
    t_thresh=float(T_THRESH),
    n_boot=int(N_BOOT),
    n_perm=int(N_PERM),
    permutation="sign-flip over bootstrap replicates",
    adjacency="groups differ by exactly one token; bins adjacent in x",
)
out["params_json"] = json.dumps(params, sort_keys=True)

np.savez_compressed(out_npz, **out)
print("[OK] Saved:", out_npz)

# ------------------------------------------------------
# Quick diagnostic plot (optional)
# ------------------------------------------------------
if SAVE_FIG:
    fig_dir = Path(paths["f_mod"]) / "FP6d" / scheme_name
    fig_dir.mkdir(parents=True, exist_ok=True)

    # show T_obs as image: groups (rows) x bins (cols)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    im = ax.imshow(T_obs, aspect="auto", interpolation="nearest")
    ax.set_title(f"{scheme_name} - 2D cluster map (T_obs) for ΔPDF trimer (obs-null)")
    ax.set_ylabel("Group index")
    ax.set_xlabel("Bin index")
    fig.colorbar(im, ax=ax, shrink=0.85, label="T")
    fig.tight_layout()
    fig.savefig(fig_dir / "fp6d_Tobs_trimer_obs_minus_null.png", dpi=DPI)
    plt.close(fig)

    # overlay significant clusters (p<0.05)
    sig = np.zeros_like(T_obs, dtype=bool)
    for c, pv in zip(clusters, pvals):
        if pv < 0.05:
            for (g, x) in c:
                sig[g, x] = True

    fig, ax = plt.subplots(figsize=(12, 4.5))
    im = ax.imshow(T_obs, aspect="auto", interpolation="nearest")
    ax.contour(sig.astype(float), levels=[0.5], linewidths=1.0)
    ax.set_title(f"{scheme_name} - Significant clusters (p<0.05) on T_obs")
    ax.set_ylabel("Group index")
    ax.set_xlabel("Bin index")
    fig.colorbar(im, ax=ax, shrink=0.85, label="T")
    fig.tight_layout()
    fig.savefig(fig_dir / "fp6d_Tobs_trimer_sigclusters.png", dpi=DPI)
    plt.close(fig)

    print("[OK] Saved figs to:", fig_dir)

