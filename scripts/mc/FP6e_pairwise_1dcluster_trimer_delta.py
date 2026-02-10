#!/usr/bin/env python3
# %%
"""
FP6e — Pairwise 1D cluster permutation on TRIMER ΔPDF differences (obs-null), GROUP-level.

For each scheme file fp6b_groups__<scheme>.npz:
  1) Build bootstrap cube of ΔPDF_trimer = PDF_obs_trimer - PDF_null_trimer:
       delta_boot[b, g, x]
     using FP6a per-animal object arrays and the same bootstrap logic as FP6b_multi.

  2) Build ALL PAIRWISE group comparisons within the scheme, with preferred sign when possible:
       geno: wt - dKI
       age : 2m - 4m
     Otherwise deterministic fallback ordering.

  3) For each comparison c, compute:
       diff_boot[b, x] = Σ_g W[c, g] * delta_boot[b, g, x]
       T_obs[x] = mean_b(diff_boot)/std_b(diff_boot)

     Cluster-permutation (1D along x):
       - cluster-forming threshold: |T| > T_THRESH
       - cluster mass: sum(|T|) over cluster
       - permutation: sign-flip across bootstrap replicates
       - corrected p-values using max cluster mass null

Produces per scheme:
  mc_dist/fp6e_1dcluster_pairwise__<scheme>__trimer_delta_obs_minus_null.npz
  fig/<dataset>/mc/FP6e/<scheme>/  (one plot per comparison + a summary "resume" plot)

Notes:
  - T is dimensionless (standardized effect size across bootstrap replicates).
  - x-axis is MC values (bin centers), not indices.
"""

from __future__ import annotations

import json
import re
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

# Run all schemes matching this pattern:
FP6B_GLOB = "fp6b_groups__*.npz"
EXCLUDE_FP6B = {"fp6b_groups__by_age_sex_geno.npz"}  # optional; set() to run all

# bootstrap used to estimate ΔPDF distribution
N_BOOT = 2000
SEED = 0
BOOT_DRAWS: Optional[int] = 300_000  # None = full pooled

# histogram bins (MATCH FP6b_multi exactly: edges, length NBINS)
BINS_MIN = -0.8
BINS_MAX = 0.8
NBINS = 401

# permutation
N_PERM = 2000
T_THRESH = 2.0
TWO_SIDED = True
MASS_MODE = "abs"  # abs cluster-mass for two-sided tests
ALPHA = 0.05

# plotting
SAVE_FIG = True
DPI = 200
XTICK_STEP = 0.2  # MC tick spacing on x-axis
SUMMARY_MAX_YTICKS = 25  # avoid unreadable y-axis on summary

# groups to include (None = all groups present in scheme file)
INCLUDE_GROUPS: Optional[List[str]] = None


# ======================================================
# Helpers: FP6a pooling & bootstrap (aligned to FP6b_multi)
# ======================================================
def _as_float_1d(x) -> np.ndarray:
    x = np.asarray(x)
    if x.size == 0:
        return np.array([], dtype=np.float32)
    x = x.astype(np.float32, copy=False).ravel()
    x = x[np.isfinite(x)]
    return x

def _cat_obj_per_animal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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
    Per-animal object array for TRIMER marginal:
      obs  = cat(obs_intra_trimer, obs_inter_trimer)
      null = cat(null_intra_trimer, null_inter_trimer)
    """
    if which == "obs":
        return _cat_obj_per_animal(
            z6a[f"{base_group}__obs_intra_trimer"],
            z6a[f"{base_group}__obs_inter_trimer"],
        )
    if which == "null":
        return _cat_obj_per_animal(
            z6a[f"{base_group}__null_intra_trimer"],
            z6a[f"{base_group}__null_inter_trimer"],
        )
    raise ValueError(which)

def summaries_pdf(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.full(bins.size - 1, np.nan, np.float32)
    counts, _ = np.histogram(x, bins=bins, density=False)
    w = np.diff(bins).astype(np.float32)
    pdf = counts.astype(np.float32) / (x.size * w)
    return pdf.astype(np.float32)

def boot_animals(obj_arr: np.ndarray, rng: np.random.Generator, boot_draws: Optional[int]) -> np.ndarray:
    A = obj_arr.size
    pick = rng.integers(0, A, size=A)
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
    Gm = len(member_obj_arrays)
    pick = rng.integers(0, Gm, size=Gm)
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
# Helpers: group tokens, pooling membership, preferred sign
# ======================================================
def parse_group_tokens(g: str) -> Dict[str, str]:
    kv = {}
    for part in g.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            kv[k] = v
    return kv

def members_for_pooled(pg, base_groups, base_tokens):
    pt = parse_group_tokens(pg)

    # phenotype is not in FP6a groups → ignore for membership
    pt = {k: v for k, v in pt.items() if k != "phenotype"}

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

def _is_one_factor_pair(ti: Dict[str, str], tj: Dict[str, str]) -> Tuple[bool, Optional[str]]:
    if set(ti.keys()) != set(tj.keys()):
        return (False, None)
    diffs = [k for k in ti.keys() if ti.get(k) != tj.get(k)]
    if len(diffs) == 1:
        return (True, diffs[0])
    return (False, None)

def _context_tokens(t: Dict[str, str], drop: str) -> str:
    items = [(k, v) for k, v in t.items() if k != drop]
    if not items:
        return "all"
    return "|".join([f"{k}={v}" for (k, v) in sorted(items)])

def _prefer_direction(i: int, j: int, groups: List[str]) -> Tuple[int, int, str]:
    """
    Enforce user convention when possible:
      geno: wt - dKI
      age : 2m - 4m
    """
    gi, gj = groups[i], groups[j]
    ti, tj = parse_group_tokens(gi), parse_group_tokens(gj)

    ok, key = _is_one_factor_pair(ti, tj)

    if ok and key == "geno":
        if ti.get("geno") == "wt" and tj.get("geno") == "dKI":
            return i, j, f"geno@{_context_tokens(ti, drop='geno')}: wt-dKI"
        if ti.get("geno") == "dKI" and tj.get("geno") == "wt":
            return j, i, f"geno@{_context_tokens(ti, drop='geno')}: wt-dKI"

    if ok and key == "age":
        if ti.get("age") == "2m" and tj.get("age") == "4m":
            return i, j, f"age@{_context_tokens(ti, drop='age')}: 2m-4m"
        if ti.get("age") == "4m" and tj.get("age") == "2m":
            return j, i, f"age@{_context_tokens(ti, drop='age')}: 2m-4m"

    # fallback deterministic
    if gi <= gj:
        return i, j, f"pair: {gi} - {gj}"
    else:
        return j, i, f"pair: {gj} - {gi}"


# ======================================================
# 1D cluster stats
# ======================================================
def find_1d_clusters(mask: np.ndarray) -> List[np.ndarray]:
    X = mask.size
    clusters = []
    i = 0
    while i < X:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < X and mask[j]:
            j += 1
        clusters.append(np.arange(i, j, dtype=np.int32))
        i = j
    return clusters

def cluster_mass_1d(T: np.ndarray, idx: np.ndarray) -> float:
    vals = T[idx].astype(np.float32)
    if TWO_SIDED:
        return float(np.nansum(np.abs(vals)) if MASS_MODE == "abs" else np.nansum(vals))
    return float(np.nansum(vals))

def compute_T_from_boot(diff_boot: np.ndarray) -> np.ndarray:
    mu = np.nanmean(diff_boot, axis=0)
    sd = np.nanstd(diff_boot, axis=0, ddof=1)
    return mu / (sd + 1e-12)


# ======================================================
# Plot helpers
# ======================================================
def make_mc_ticks(bins: np.ndarray, step: float) -> Tuple[np.ndarray, List[str]]:
    x_mc = 0.5 * (bins[:-1] + bins[1:])
    ticks = np.arange(BINS_MIN, BINS_MAX + 1e-9, step, dtype=float)
    # map tick values to nearest bin center index
    idx = np.array([int(np.argmin(np.abs(x_mc - t))) for t in ticks], dtype=int)
    labels = [f"{t:.1f}" for t in ticks]
    return idx, labels

def safe_name(s: str, maxlen: int = 140) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)[:maxlen]


# ======================================================
# Scheme runner
# ======================================================
def run_scheme(fp6b_path: Path, z6a, dist_dir: Path, fig_root: Path) -> None:
    z6b = np.load(fp6b_path, allow_pickle=True)
    scheme_params = json.loads(z6b["params_json"].item())
    scheme_name = scheme_params.get("scheme", fp6b_path.stem.replace("fp6b_groups__", ""))

    pooled_groups = [str(g) for g in z6b["groups"]]
    if INCLUDE_GROUPS is not None:
        keep = set(INCLUDE_GROUPS)
        pooled_groups = [g for g in pooled_groups if g in keep]

    print("\n" + "=" * 70)
    print("[FP6e] scheme:", scheme_name)
    print("[FP6e] fp6b:", fp6b_path.name)
    print("[FP6b] Group order:")
    for ii, gg in enumerate(pooled_groups):
        print(f"  {ii:02d}: {gg}")

    # membership map pooled -> FP6a base groups
    base_groups = [str(g) for g in z6a["groups"]]
    base_tokens = [parse_group_tokens(g) for g in base_groups]
    members_map = {pg: members_for_pooled(pg, base_groups, base_tokens) for pg in pooled_groups}
    for pg in pooled_groups:
        if len(members_map[pg]) == 0:
            raise RuntimeError(f"[{scheme_name}] No FP6a base groups found for pooled group: {pg}")

    # bins & x-axis
    bins = np.linspace(BINS_MIN, BINS_MAX, NBINS, dtype=np.float32)
    x_mc = 0.5 * (bins[:-1] + bins[1:])
    X = bins.size - 1
    G = len(pooled_groups)

    # build delta_boot: (B,G,X)
    delta_boot = np.empty((N_BOOT, G, X), dtype=np.float32)

    rng0 = np.random.default_rng(SEED)
    boot_seeds = rng0.integers(0, 2**32 - 1, size=N_BOOT, dtype=np.uint32)

    for gi, pg in enumerate(pooled_groups):
        members = members_map[pg]
        obs_members = [get_fp6a_trimer_obj(z6a, bg, "obs") for bg in members]
        nul_members = [get_fp6a_trimer_obj(z6a, bg, "null") for bg in members]
        use_group_boot = (len(members) > 1)

        for b in range(N_BOOT):
            rngb = np.random.default_rng(int(boot_seeds[b]) + 1000 * gi)
            if use_group_boot:
                xb_obs = boot_groups(obs_members, rngb, BOOT_DRAWS)
                xb_nul = boot_groups(nul_members, rngb, BOOT_DRAWS)
            else:
                xb_obs = boot_animals(obs_members[0], rngb, BOOT_DRAWS)
                xb_nul = boot_animals(nul_members[0], rngb, BOOT_DRAWS)

            delta_boot[b, gi] = summaries_pdf(xb_obs, bins) - summaries_pdf(xb_nul, bins)

    # comparisons (ALL PAIRS) + encoded contrast matrix W
    comparisons = []
    W = []
    for i in range(G):
        for j in range(i + 1, G):
            ia, ib, lbl = _prefer_direction(i, j, pooled_groups)
            comparisons.append((ia, ib, lbl))
            w = np.zeros((G,), dtype=np.float32)
            w[ia] = 1.0
            w[ib] = -1.0
            W.append(w)

    comparisons_names = np.array([c[2] for c in comparisons], dtype=object)
    comparisons_i = np.array([c[0] for c in comparisons], dtype=np.int32)
    comparisons_j = np.array([c[1] for c in comparisons], dtype=np.int32)
    comparisons_W = np.stack(W, axis=0).astype(np.float32)
    C = comparisons_W.shape[0]

    print(f"[FP6e] comparisons: {C}")

    # permutation RNG
    rngp = np.random.default_rng(SEED + 999_123)

    # store per-comparison outputs
    T_list = []
    clusters_list = []
    masses_list = []
    pvals_list = []
    nullmax_list = []
    sig_mask_list = []

    # plotting ticks
    xt_idx, xt_lab = make_mc_ticks(bins, XTICK_STEP)

    fig_dir = fig_root / "FP6e" / scheme_name
    if SAVE_FIG:
        fig_dir.mkdir(parents=True, exist_ok=True)

    for ci in range(C):
        name = str(comparisons_names[ci])
        w = comparisons_W[ci]
        diff_boot = np.tensordot(delta_boot, w, axes=([1], [0]))  # (B,X)
        T_obs = compute_T_from_boot(diff_boot)

        supra = (np.abs(T_obs) > T_THRESH) if TWO_SIDED else (T_obs > T_THRESH)
        clusters = find_1d_clusters(supra)
        obs_masses = np.array([cluster_mass_1d(T_obs, c) for c in clusters], dtype=np.float32)

        # permutation: sign-flip across bootstrap replicates
        max_masses = np.zeros(N_PERM, dtype=np.float32)
        for p in range(N_PERM):
            flips = rngp.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(N_BOOT, 1))
            T_p = compute_T_from_boot(diff_boot * flips)
            supra_p = (np.abs(T_p) > T_THRESH) if TWO_SIDED else (T_p > T_THRESH)
            clp = find_1d_clusters(supra_p)
            max_masses[p] = max((cluster_mass_1d(T_p, c) for c in clp), default=0.0)

        # corrected p-values (+1)
        pvals = np.array([(1 + np.sum(max_masses >= m)) / (N_PERM + 1) for m in obs_masses], dtype=np.float32)

        # sig mask at ALPHA
        sig = np.zeros(X, dtype=bool)
        for c, pv in zip(clusters, pvals):
            if pv < ALPHA:
                sig[c] = True

        # collect
        T_list.append(T_obs.astype(np.float32))
        clusters_list.append(np.array([c.astype(np.int32) for c in clusters], dtype=object))
        masses_list.append(obs_masses.astype(np.float32))
        pvals_list.append(pvals.astype(np.float32))
        nullmax_list.append(max_masses.astype(np.float32))
        sig_mask_list.append(sig)

        print(f"[FP6e] {ci:02d}/{C-1:02d} {name}: clusters={len(clusters)} sig_bins={int(sig.sum())}")

        # --- per-comparison plot ---
        if SAVE_FIG:
            fig, ax = plt.subplots(figsize=(12, 3.6))
            ax.plot(x_mc, T_obs, lw=1.0)

            # threshold lines = cluster-forming threshold
            ax.axhline(T_THRESH, ls="--", lw=1.0)
            ax.axhline(-T_THRESH, ls="--", lw=1.0)

            # shade significant bins
            if np.any(sig):
                ax.fill_between(x_mc, T_obs.min(), T_obs.max(), where=sig, alpha=0.15, label=f"clusters p<{ALPHA}")

            ax.set_title(f"{scheme_name} — {name}\nT(x) on (ΔPDF_trimer obs-null) contrast; T is unitless")
            ax.set_xlabel("Metaconnectivity value (MC bin center)")
            ax.set_ylabel("T statistic (unitless)")

            ax.set_xticks(x_mc[xt_idx])
            ax.set_xticklabels(xt_lab)

            # small legend text for threshold lines
            ax.text(
                0.01, 0.02,
                f"-- lines: cluster-forming threshold |T|>{T_THRESH}",
                transform=ax.transAxes,
                fontsize=9,
                va="bottom",
            )

            fig.tight_layout()
            fig.savefig(fig_dir / f"fp6e_{ci:02d}_{safe_name(name)}.png", dpi=DPI)
            plt.close(fig)

    # --- resume / summary plot: comparisons × MC value (imshow of T) ---
    if SAVE_FIG:
        T_stack = np.vstack([t[None, :] for t in T_list])  # (C,X)
        sig_stack = np.vstack([m[None, :].astype(float) for m in sig_mask_list])  # (C,X)

        fig, ax = plt.subplots(figsize=(12, max(4.0, 0.28 * C)))
        im = ax.imshow(T_stack, aspect="auto", interpolation="nearest")

        # overlay significance contour
        if sig_stack.shape[0] >= 2 and sig_stack.shape[1] >= 2 and np.any(sig_stack):
            ax.contour(sig_stack, levels=[0.5], linewidths=1.0)


        ax.set_title(f"{scheme_name} — Summary: all pairwise comparisons (rows) × MC bins (cols)")
        ax.set_xlabel("Metaconnectivity value (MC bin center)")
        ax.set_ylabel("Comparison index")

        # x ticks as MC values
        ax.set_xticks(xt_idx)
        ax.set_xticklabels(xt_lab)

        # y tick labels (sparse if too many)
        if C <= SUMMARY_MAX_YTICKS:
            ax.set_yticks(np.arange(C))
            ax.set_yticklabels([str(n) for n in comparisons_names], fontsize=8)
        else:
            step = max(1, C // SUMMARY_MAX_YTICKS)
            yy = np.arange(0, C, step)
            ax.set_yticks(yy)
            ax.set_yticklabels([str(comparisons_names[i]) for i in yy], fontsize=8)

        fig.colorbar(im, ax=ax, shrink=0.85, label="T (unitless)")
        fig.tight_layout()
        fig.savefig(fig_dir / "fp6e_summary_Tmatrix.png", dpi=DPI)
        plt.close(fig)

    # save NPZ for this scheme
    out_npz = dist_dir / f"fp6e_1dcluster_pairwise__{scheme_name}__trimer_delta_obs_minus_null.npz"
    out = dict(
        bins=bins,
        groups=np.array(pooled_groups, dtype=object),
        scheme=scheme_name,
        n_boot=np.int32(N_BOOT),
        n_perm=np.int32(N_PERM),
        t_thresh=np.float32(T_THRESH),
        two_sided=np.int8(1 if TWO_SIDED else 0),
        mass_mode=str(MASS_MODE),
        alpha=np.float32(ALPHA),

        comparisons_names=comparisons_names,
        comparisons_i=comparisons_i,
        comparisons_j=comparisons_j,
        comparisons_W=comparisons_W,

        T_obs_list=np.array(T_list, dtype=object),
        clusters_coords_list=np.array(clusters_list, dtype=object),
        clusters_mass_list=np.array(masses_list, dtype=object),
        clusters_p_list=np.array(pvals_list, dtype=object),
        null_max_masses_list=np.array(nullmax_list, dtype=object),
        sig_mask_list=np.array([m.astype(np.uint8) for m in sig_mask_list], dtype=object),
    )

    params = dict(
        dataset=DATASET,
        scheme=scheme_name,
        fp6a=str(dist_dir / FP6AG_NAME),
        fp6b_scheme=str(fp6b_path),
        contrast="all-pairs on ΔPDF_trimer (obs-null), using W over groups",
        permutation="sign-flip over bootstrap replicates",
        direction_convention=dict(geno="wt-dKI", age="2m-4m"),
        comparisons="all_pairs",
        note="Cluster-forming threshold defines candidate bins; shaded bins belong to significant clusters (max-mass corrected).",
    )
    out["params_json"] = json.dumps(params, sort_keys=True)

    np.savez_compressed(out_npz, **out)
    print("[OK] Saved:", out_npz.name)
    if SAVE_FIG:
        print("[OK] Saved figs to:", fig_dir)


# ======================================================
# MAIN
# ======================================================
def main():
    paths = get_paths(DATASET, TIMECOURSE_FOLDER, COGNITIVE_FILE, ANAT_LABELS_FILE)
    mc_dir = Path(paths["mc"])
    dist_dir = mc_dir / MC_DIST_DIRNAME
    dist_dir.mkdir(parents=True, exist_ok=True)

    fp6a_path = dist_dir / FP6AG_NAME
    if not fp6a_path.exists():
        raise FileNotFoundError(fp6a_path)
    z6a = np.load(fp6a_path, allow_pickle=True)

    # figure root (your pipeline uses paths["f_mod"])
    fig_root = Path(paths["f_mod"]) / "mc"

    fp6b_files = sorted(dist_dir.glob(FP6B_GLOB))
    fp6b_files = [p for p in fp6b_files if p.name not in EXCLUDE_FP6B]

    if not fp6b_files:
        raise FileNotFoundError(f"No scheme files found in {dist_dir} matching {FP6B_GLOB}")

    print("[FP6e] Found schemes:", len(fp6b_files))
    for p in fp6b_files:
        print("  -", p.name)

    for fp6b_path in fp6b_files:
        run_scheme(fp6b_path, z6a, dist_dir, fig_root)

    print("[DONE] FP6e all schemes")


if __name__ == "__main__":
    main()
