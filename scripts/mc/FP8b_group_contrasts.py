#!/usr/bin/env python3
# %%
"""
FP8b — Group contrasts from FP8a only (NO recompute)

Consumes:
  - results/<dataset>/mc/mc_dist/fp8a_bootstrap_mc_tail_distributions_by_group.npz

Produces:
  - results/<dataset>/mc/mc_dist/fp8b_group_tail_contrasts.npz

Method:
  - Δ = g1 - g2 computed on q_obs / pdf_obs
  - CI for Δ via CI propagation using FP8a CI bands (normal approx):
        se ≈ (ci_hi - ci_lo) / (2*z)
        se_Δ = sqrt(se1^2 + se2^2)
        CI_Δ = Δ ± z*se_Δ

Notes:
  - This is an approximation because FP8a does not store bootstrap draws.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from shared_code.fun_paths import get_paths

# =============================================================================
# CONFIG
# =============================================================================
DATASET = "ines_abdallah"

FP8A_NAME = "fp8a_bootstrap_mc_tail_distributions_by_group.npz"
FP8B_NAME = "fp8b_group_tail_contrasts.npz"

SIZE_COND = "obs_all"

A_MIN_INTERACTIONS = 8  # gate for interactions
CI_ALPHA = 0.05
Z_95 = 1.959963984540054  # ~ scipy.stats.norm.ppf(0.975) but no scipy dependency
Z = Z_95  # assume 95% CI bands in FP8a; if you used different, change this

P_POINTS = np.array([0.01, 0.05, 0.95, 0.99], dtype=float)

SAVE_DTYPE = np.float32  # keep npz small

# =============================================================================
# Helpers
# =============================================================================

def load_npz(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}

def as_list_str(x: np.ndarray) -> list[str]:
    return [str(v) for v in np.asarray(x).reshape(-1).tolist()]

def cell_key(group: str, cond: str, field: str) -> str:
    return f"{group}::{cond}__{field}"

def safe_scalar_int(x: np.ndarray) -> int:
    return int(np.asarray(x).reshape(-1)[0])

def approx_se_from_ci(ci_lo: np.ndarray, ci_hi: np.ndarray, z: float) -> np.ndarray:
    # se ≈ (hi - lo) / (2*z)
    return (ci_hi - ci_lo) / (2.0 * z)

def ci_propagate_diff(
    x1_obs: np.ndarray, x1_lo: np.ndarray, x1_hi: np.ndarray,
    x2_obs: np.ndarray, x2_lo: np.ndarray, x2_hi: np.ndarray,
    z: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d_obs = x1_obs - x2_obs
    se1 = approx_se_from_ci(x1_lo, x1_hi, z=z)
    se2 = approx_se_from_ci(x2_lo, x2_hi, z=z)
    se_d = np.sqrt(se1**2 + se2**2)
    d_lo = d_obs - z * se_d
    d_hi = d_obs + z * se_d
    return d_obs, d_lo, d_hi

def auc_trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapezoid(y, x))

def nearest_idx(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    # for each value in values, pick nearest index in grid
    grid = np.asarray(grid)
    values = np.asarray(values)
    return np.array([int(np.argmin(np.abs(grid - v))) for v in values], dtype=int)

@dataclass(frozen=True)
class PairContrast:
    name: str
    g1: str
    g2: str
    kind: str  # "main" or "simple"

@dataclass(frozen=True)
class InteractionContrast:
    name: str
    # difference in differences:
    # (a1 - a2) - (b1 - b2)
    a1: str
    a2: str
    b1: str
    b2: str
    kind: str  # "interaction"

# =============================================================================
# Contrast builder (based on your group naming)
# =============================================================================

def build_contrasts(groups: list[str]) -> tuple[list[PairContrast], list[InteractionContrast], list[InteractionContrast]]:
    gset = set(groups)

    def has(g: str) -> bool:
        return g in gset

    pairs: list[PairContrast] = []
    inter2: list[InteractionContrast] = []
    inter3: list[InteractionContrast] = []

    # ---- mandatory main effects
    if has("age=2m") and has("age=4m"):
        pairs.append(PairContrast("age:2m-4m", "age=2m", "age=4m", "main"))
    if has("genotype=wt") and has("genotype=dKI"):
        pairs.append(PairContrast("genotype:wt-dKI", "genotype=wt", "genotype=dKI", "main"))
    if has("sex=F") and has("sex=M"):
        pairs.append(PairContrast("sex:F-M", "sex=F", "sex=M", "main"))

    # ---- simple effects (recommended)
    # genotype within age
    if has("age=2m&genotype=wt") and has("age=2m&genotype=dKI"):
        pairs.append(PairContrast("genotype@age=2m:wt-dKI", "age=2m&genotype=wt", "age=2m&genotype=dKI", "simple"))
    if has("age=4m&genotype=wt") and has("age=4m&genotype=dKI"):
        pairs.append(PairContrast("genotype@age=4m:wt-dKI", "age=4m&genotype=wt", "age=4m&genotype=dKI", "simple"))

    # sex within age
    if has("age=2m&sex=F") and has("age=2m&sex=M"):
        pairs.append(PairContrast("sex@age=2m:F-M", "age=2m&sex=F", "age=2m&sex=M", "simple"))
    if has("age=4m&sex=F") and has("age=4m&sex=M"):
        pairs.append(PairContrast("sex@age=4m:F-M", "age=4m&sex=F", "age=4m&sex=M", "simple"))

    # sex within genotype
    if has("sex=F&genotype=wt") and has("sex=M&genotype=wt"):
        pairs.append(PairContrast("sex@genotype=wt:F-M", "sex=F&genotype=wt", "sex=M&genotype=wt", "simple"))
    if has("sex=F&genotype=dKI") and has("sex=M&genotype=dKI"):
        pairs.append(PairContrast("sex@genotype=dKI:F-M", "sex=F&genotype=dKI", "sex=M&genotype=dKI", "simple"))

    # age within genotype
    if has("age=2m&genotype=wt") and has("age=4m&genotype=wt"):
        pairs.append(PairContrast("age@genotype=wt:2m-4m", "age=2m&genotype=wt", "age=4m&genotype=wt", "simple"))
    if has("age=2m&genotype=dKI") and has("age=4m&genotype=dKI"):
        pairs.append(PairContrast("age@genotype=dKI:2m-4m", "age=2m&genotype=dKI", "age=4m&genotype=dKI", "simple"))

    # age within sex
    if has("age=2m&sex=F") and has("age=4m&sex=F"):
        pairs.append(PairContrast("age@sex=F:2m-4m", "age=2m&sex=F", "age=4m&sex=F", "simple"))
    if has("age=2m&sex=M") and has("age=4m&sex=M"):
        pairs.append(PairContrast("age@sex=M:2m-4m", "age=2m&sex=M", "age=4m&sex=M", "simple"))

    # genotype within (age, sex)
    for age in ["2m", "4m"]:
        for sex in ["F", "M"]:
            g_wt = f"age={age}&sex={sex}&genotype=wt"
            g_ki = f"age={age}&sex={sex}&genotype=dKI"
            if has(g_wt) and has(g_ki):
                pairs.append(PairContrast(f"genotype@age={age}&sex={sex}:wt-dKI", g_wt, g_ki, "simple"))

    # sex within (age, genotype)
    for age in ["2m", "4m"]:
        for geno in ["wt", "dKI"]:
            g_F = f"age={age}&sex=F&genotype={geno}"
            g_M = f"age={age}&sex=M&genotype={geno}"
            if has(g_F) and has(g_M):
                pairs.append(PairContrast(f"sex@age={age}&genotype={geno}:F-M", g_F, g_M, "simple"))

    # age within (sex, genotype)
    for sex in ["F", "M"]:
        for geno in ["wt", "dKI"]:
            g_2 = f"age=2m&sex={sex}&genotype={geno}"
            g_4 = f"age=4m&sex={sex}&genotype={geno}"
            if has(g_2) and has(g_4):
                pairs.append(PairContrast(f"age@sex={sex}&genotype={geno}:2m-4m", g_2, g_4, "simple"))

    # ---- true 2-way interactions (difference-in-differences)
    # age×genotype: (2m_wt - 2m_dKI) - (4m_wt - 4m_dKI)
    a1, a2 = "age=2m&genotype=wt", "age=2m&genotype=dKI"
    b1, b2 = "age=4m&genotype=wt", "age=4m&genotype=dKI"
    if has(a1) and has(a2) and has(b1) and has(b2):
        inter2.append(InteractionContrast("age×genotype:(2m_wt-2m_dKI)-(4m_wt-4m_dKI)", a1, a2, b1, b2, "interaction"))

    # age×sex: (2m_F - 2m_M) - (4m_F - 4m_M)
    a1, a2 = "age=2m&sex=F", "age=2m&sex=M"
    b1, b2 = "age=4m&sex=F", "age=4m&sex=M"
    if has(a1) and has(a2) and has(b1) and has(b2):
        inter2.append(InteractionContrast("age×sex:(2m_F-2m_M)-(4m_F-4m_M)", a1, a2, b1, b2, "interaction"))

    # sex×genotype: (F_wt - F_dKI) - (M_wt - M_dKI)
    a1, a2 = "sex=F&genotype=wt", "sex=F&genotype=dKI"
    b1, b2 = "sex=M&genotype=wt", "sex=M&genotype=dKI"
    if has(a1) and has(a2) and has(b1) and has(b2):
        inter2.append(InteractionContrast("sex×genotype:(F_wt-F_dKI)-(M_wt-M_dKI)", a1, a2, b1, b2, "interaction"))

    # ---- true 3-way interaction (difference-in-diff-in-diff)
    # age×sex×genotype:
    #   [ (2m,F,wt - 2m,F,dKI) - (2m,M,wt - 2m,M,dKI) ] -
    #   [ (4m,F,wt - 4m,F,dKI) - (4m,M,wt - 4m,M,dKI) ]
    g_2F_wt = "age=2m&sex=F&genotype=wt"
    g_2F_ki = "age=2m&sex=F&genotype=dKI"
    g_2M_wt = "age=2m&sex=M&genotype=wt"
    g_2M_ki = "age=2m&sex=M&genotype=dKI"
    g_4F_wt = "age=4m&sex=F&genotype=wt"
    g_4F_ki = "age=4m&sex=F&genotype=dKI"
    g_4M_wt = "age=4m&sex=M&genotype=wt"
    g_4M_ki = "age=4m&sex=M&genotype=dKI"
    if all(has(g) for g in [g_2F_wt,g_2F_ki,g_2M_wt,g_2M_ki,g_4F_wt,g_4F_ki,g_4M_wt,g_4M_ki]):
        # Represent as two stacked DoDs:
        # left = (2F_wt-2F_ki) - (2M_wt-2M_ki)
        # right= (4F_wt-4F_ki) - (4M_wt-4M_ki)
        # overall = left - right
        # We'll store as a 2-level interaction contrast:
        inter3.append(InteractionContrast(
            "age×sex×genotype:([2F_wt-2F_dKI]-[2M_wt-2M_dKI]) - ([4F_wt-4F_dKI]-[4M_wt-4M_dKI])",
            # a1-a2 is left numerator; b1-b2 is left denominator for first DoD
            # We'll compute explicitly later; this dataclass isn't enough alone,
            # but we keep the name here and handle it custom.
            "NA","NA","NA","NA","interaction"
        ))

    return pairs, inter2, inter3


# =============================================================================
# Main FP8b
# =============================================================================

def main():
    paths = get_paths()
    fp8a_path = Path(paths["results"]) / "mc" / "mc_dist" / FP8A_NAME
    out_path = Path(paths["results"])  / "mc" / "mc_dist" / FP8B_NAME

    d = load_npz(fp8a_path)
    groups = as_list_str(d["groups"])
    conditions = as_list_str(d["conditions"])
    p_grid = np.asarray(d["p_grid"], dtype=float)
    bin_centers = np.asarray(d["bin_centers"], dtype=float) if "bin_centers" in d else None

    pairs, inter2, inter3 = build_contrasts(groups)

    print("[OK] FP8a:", fp8a_path)
    print("[INFO] n_groups:", len(groups), "n_conditions:", len(conditions))
    print("[INFO] n_pair_contrasts:", len(pairs), "n_2way_interactions:", len(inter2), "n_3way_interactions:", len(inter3))
    print("[INFO] Output:", out_path)

    # Precompute indices for effect-size points on p_grid
    p_idx = nearest_idx(p_grid, P_POINTS)
    p_points_snap = p_grid[p_idx]  # actual p positions used

    out: dict[str, np.ndarray] = {}

    # metadata
    out["meta__source_fp8a_path"] = np.array(str(fp8a_path), dtype=object)
    out["meta__created_unix"] = np.array(int(time.time()), dtype=np.int64)
    out["meta__ci_alpha"] = np.array(CI_ALPHA, dtype=float)
    out["meta__ci_z"] = np.array(Z, dtype=float)
    out["meta__ci_method"] = np.array("ci_propagation", dtype=object)
    out["meta__A_min_interactions"] = np.array(A_MIN_INTERACTIONS, dtype=np.int32)
    out["meta__p_points_requested"] = P_POINTS.astype(float)
    out["meta__p_points_used"] = p_points_snap.astype(float)
    out["meta__assumes_fp8a_ci_level"] = np.array(1.0 - CI_ALPHA, dtype=float)

    # store grids once
    out["p_grid"] = p_grid.astype(SAVE_DTYPE)
    if bin_centers is not None:
        out["bin_centers"] = bin_centers.astype(SAVE_DTYPE)

    # Helper: fetch per-cell arrays
    def get_cell(g: str, c: str, field: str) -> np.ndarray:
        return np.asarray(d[cell_key(g, c, field)])

    def get_counts(g: str) -> tuple[int, int]:
        A = safe_scalar_int(get_cell(g, SIZE_COND, "n_animals"))
        N = safe_scalar_int(get_cell(g, SIZE_COND, "n_obs_used"))
        return A, N

    # gate for interactions: all four groups need A>=A_min
    def ok_interaction_groups(gs: list[str]) -> bool:
        for g in gs:
            A, _ = get_counts(g)
            if A < A_MIN_INTERACTIONS:
                return False
        return True

    # Save one contrast result (q and pdf)
    def save_pair_contrast(name: str, g1: str, g2: str, c: str):
        # Q
        q1 = get_cell(g1, c, "q_obs").astype(float)
        q1lo = get_cell(g1, c, "q_ci_lo").astype(float)
        q1hi = get_cell(g1, c, "q_ci_hi").astype(float)

        q2 = get_cell(g2, c, "q_obs").astype(float)
        q2lo = get_cell(g2, c, "q_ci_lo").astype(float)
        q2hi = get_cell(g2, c, "q_ci_hi").astype(float)

        dq_obs, dq_lo, dq_hi = ci_propagate_diff(q1,q1lo,q1hi, q2,q2lo,q2hi, z=Z)

        # Effect sizes on Q
        dq_at_p = dq_obs[p_idx]
        dq_at_p_lo = dq_lo[p_idx]
        dq_at_p_hi = dq_hi[p_idx]

        auc_dq = auc_trapz(dq_obs, p_grid)
        auc_lo = auc_trapz(dq_lo, p_grid)
        auc_hi = auc_trapz(dq_hi, p_grid)

        # PDF
        pdf1 = get_cell(g1, c, "pdf_obs").astype(float)
        pdf1lo = get_cell(g1, c, "pdf_ci_lo").astype(float)
        pdf1hi = get_cell(g1, c, "pdf_ci_hi").astype(float)

        pdf2 = get_cell(g2, c, "pdf_obs").astype(float)
        pdf2lo = get_cell(g2, c, "pdf_ci_lo").astype(float)
        pdf2hi = get_cell(g2, c, "pdf_ci_hi").astype(float)

        dpdf_obs, dpdf_lo, dpdf_hi = ci_propagate_diff(pdf1,pdf1lo,pdf1hi, pdf2,pdf2lo,pdf2hi, z=Z)

        # PDF AUC over x (should be ~0 if pdfs integrate to 1 and groups comparable; still informative)
        if bin_centers is not None:
            auc_dpdf = auc_trapz(dpdf_obs, bin_centers)
            auc_dpdf_lo = auc_trapz(dpdf_lo, bin_centers)
            auc_dpdf_hi = auc_trapz(dpdf_hi, bin_centers)
        else:
            auc_dpdf = np.nan; auc_dpdf_lo = np.nan; auc_dpdf_hi = np.nan

        A1, N1 = get_counts(g1)
        A2, N2 = get_counts(g2)

        prefix = f"contrast__{name}__condition__{c}__field__"

        out[prefix + "g1"] = np.array(g1, dtype=object)
        out[prefix + "g2"] = np.array(g2, dtype=object)

        out[prefix + "dq_obs"] = dq_obs.astype(SAVE_DTYPE)
        out[prefix + "dq_ci_lo"] = dq_lo.astype(SAVE_DTYPE)
        out[prefix + "dq_ci_hi"] = dq_hi.astype(SAVE_DTYPE)

        out[prefix + "dpdf_obs"] = dpdf_obs.astype(SAVE_DTYPE)
        out[prefix + "dpdf_ci_lo"] = dpdf_lo.astype(SAVE_DTYPE)
        out[prefix + "dpdf_ci_hi"] = dpdf_hi.astype(SAVE_DTYPE)

        out[prefix + "dq_at_p_points"] = p_points_snap.astype(SAVE_DTYPE)
        out[prefix + "dq_at_p_obs"] = dq_at_p.astype(SAVE_DTYPE)
        out[prefix + "dq_at_p_ci_lo"] = dq_at_p_lo.astype(SAVE_DTYPE)
        out[prefix + "dq_at_p_ci_hi"] = dq_at_p_hi.astype(SAVE_DTYPE)

        out[prefix + "AUC_dq_obs"] = np.array(auc_dq, dtype=float)
        out[prefix + "AUC_dq_ci_lo"] = np.array(auc_lo, dtype=float)
        out[prefix + "AUC_dq_ci_hi"] = np.array(auc_hi, dtype=float)

        out[prefix + "AUC_dpdf_obs"] = np.array(auc_dpdf, dtype=float)
        out[prefix + "AUC_dpdf_ci_lo"] = np.array(auc_dpdf_lo, dtype=float)
        out[prefix + "AUC_dpdf_ci_hi"] = np.array(auc_dpdf_hi, dtype=float)

        out[prefix + "n_animals_g1"] = np.array(A1, dtype=np.int32)
        out[prefix + "n_animals_g2"] = np.array(A2, dtype=np.int32)
        out[prefix + "n_obs_used_g1"] = np.array(N1, dtype=np.int32)
        out[prefix + "n_obs_used_g2"] = np.array(N2, dtype=np.int32)

    # Run pair contrasts
    n_done = 0
    for pc in pairs:
        # always run; you already have the groups
        for c in conditions:
            save_pair_contrast(pc.name, pc.g1, pc.g2, c)
            n_done += 1
    print("[OK] Pair contrasts saved:", n_done)

    # 2-way interactions (DoD): compute as linear combination of pair contrasts,
    # with CI propagated from the four input CIs (still normal approx).
    def save_dod_contrast(name: str, a1: str, a2: str, b1: str, b2: str, c: str):
        # compute (a1-a2) - (b1-b2) = a1 - a2 - b1 + b2
        # For normal propagation: var(sum s_i X_i) = sum (s_i^2 var_i) assuming independence
        # We approximate var_i from CI bands as se_i^2

        # Q terms
        q_a1 = get_cell(a1,c,"q_obs").astype(float); q_a1lo = get_cell(a1,c,"q_ci_lo").astype(float); q_a1hi = get_cell(a1,c,"q_ci_hi").astype(float)
        q_a2 = get_cell(a2,c,"q_obs").astype(float); q_a2lo = get_cell(a2,c,"q_ci_lo").astype(float); q_a2hi = get_cell(a2,c,"q_ci_hi").astype(float)
        q_b1 = get_cell(b1,c,"q_obs").astype(float); q_b1lo = get_cell(b1,c,"q_ci_lo").astype(float); q_b1hi = get_cell(b1,c,"q_ci_hi").astype(float)
        q_b2 = get_cell(b2,c,"q_obs").astype(float); q_b2lo = get_cell(b2,c,"q_ci_lo").astype(float); q_b2hi = get_cell(b2,c,"q_ci_hi").astype(float)

        dq_obs = (q_a1 - q_a2) - (q_b1 - q_b2)

        se_a1 = approx_se_from_ci(q_a1lo, q_a1hi, z=Z)
        se_a2 = approx_se_from_ci(q_a2lo, q_a2hi, z=Z)
        se_b1 = approx_se_from_ci(q_b1lo, q_b1hi, z=Z)
        se_b2 = approx_se_from_ci(q_b2lo, q_b2hi, z=Z)

        se = np.sqrt(se_a1**2 + se_a2**2 + se_b1**2 + se_b2**2)
        dq_lo = dq_obs - Z*se
        dq_hi = dq_obs + Z*se

        # effect sizes
        dq_at_p = dq_obs[p_idx]
        dq_at_p_lo = dq_lo[p_idx]
        dq_at_p_hi = dq_hi[p_idx]

        auc_dq = auc_trapz(dq_obs, p_grid)
        auc_lo = auc_trapz(dq_lo, p_grid)
        auc_hi = auc_trapz(dq_hi, p_grid)

        # PDF terms
        p_a1 = get_cell(a1,c,"pdf_obs").astype(float); p_a1lo = get_cell(a1,c,"pdf_ci_lo").astype(float); p_a1hi = get_cell(a1,c,"pdf_ci_hi").astype(float)
        p_a2 = get_cell(a2,c,"pdf_obs").astype(float); p_a2lo = get_cell(a2,c,"pdf_ci_lo").astype(float); p_a2hi = get_cell(a2,c,"pdf_ci_hi").astype(float)
        p_b1 = get_cell(b1,c,"pdf_obs").astype(float); p_b1lo = get_cell(b1,c,"pdf_ci_lo").astype(float); p_b1hi = get_cell(b1,c,"pdf_ci_hi").astype(float)
        p_b2 = get_cell(b2,c,"pdf_obs").astype(float); p_b2lo = get_cell(b2,c,"pdf_ci_lo").astype(float); p_b2hi = get_cell(b2,c,"pdf_ci_hi").astype(float)

        dpdf_obs = (p_a1 - p_a2) - (p_b1 - p_b2)

        se_a1p = approx_se_from_ci(p_a1lo, p_a1hi, z=Z)
        se_a2p = approx_se_from_ci(p_a2lo, p_a2hi, z=Z)
        se_b1p = approx_se_from_ci(p_b1lo, p_b1hi, z=Z)
        se_b2p = approx_se_from_ci(p_b2lo, p_b2hi, z=Z)
        se_p = np.sqrt(se_a1p**2 + se_a2p**2 + se_b1p**2 + se_b2p**2)
        dpdf_lo = dpdf_obs - Z*se_p
        dpdf_hi = dpdf_obs + Z*se_p

        if bin_centers is not None:
            auc_dpdf = auc_trapz(dpdf_obs, bin_centers)
            auc_dpdf_lo = auc_trapz(dpdf_lo, bin_centers)
            auc_dpdf_hi = auc_trapz(dpdf_hi, bin_centers)
        else:
            auc_dpdf = np.nan; auc_dpdf_lo = np.nan; auc_dpdf_hi = np.nan

        A_a1, N_a1 = get_counts(a1)
        A_a2, N_a2 = get_counts(a2)
        A_b1, N_b1 = get_counts(b1)
        A_b2, N_b2 = get_counts(b2)

        prefix = f"contrast__{name}__condition__{c}__field__"
        out[prefix + "formula"] = np.array(f"({a1}-{a2})-({b1}-{b2})", dtype=object)

        out[prefix + "dq_obs"] = dq_obs.astype(SAVE_DTYPE)
        out[prefix + "dq_ci_lo"] = dq_lo.astype(SAVE_DTYPE)
        out[prefix + "dq_ci_hi"] = dq_hi.astype(SAVE_DTYPE)

        out[prefix + "dpdf_obs"] = dpdf_obs.astype(SAVE_DTYPE)
        out[prefix + "dpdf_ci_lo"] = dpdf_lo.astype(SAVE_DTYPE)
        out[prefix + "dpdf_ci_hi"] = dpdf_hi.astype(SAVE_DTYPE)

        out[prefix + "dq_at_p_points"] = p_points_snap.astype(SAVE_DTYPE)
        out[prefix + "dq_at_p_obs"] = dq_at_p.astype(SAVE_DTYPE)
        out[prefix + "dq_at_p_ci_lo"] = dq_at_p_lo.astype(SAVE_DTYPE)
        out[prefix + "dq_at_p_ci_hi"] = dq_at_p_hi.astype(SAVE_DTYPE)

        out[prefix + "AUC_dq_obs"] = np.array(auc_dq, dtype=float)
        out[prefix + "AUC_dq_ci_lo"] = np.array(auc_lo, dtype=float)
        out[prefix + "AUC_dq_ci_hi"] = np.array(auc_hi, dtype=float)

        out[prefix + "AUC_dpdf_obs"] = np.array(auc_dpdf, dtype=float)
        out[prefix + "AUC_dpdf_ci_lo"] = np.array(auc_dpdf_lo, dtype=float)
        out[prefix + "AUC_dpdf_ci_hi"] = np.array(auc_dpdf_hi, dtype=float)

        # record counts for all four groups (so you can audit interaction validity later)
        out[prefix + "n_animals_a1"] = np.array(A_a1, dtype=np.int32)
        out[prefix + "n_animals_a2"] = np.array(A_a2, dtype=np.int32)
        out[prefix + "n_animals_b1"] = np.array(A_b1, dtype=np.int32)
        out[prefix + "n_animals_b2"] = np.array(A_b2, dtype=np.int32)

        out[prefix + "n_obs_used_a1"] = np.array(N_a1, dtype=np.int32)
        out[prefix + "n_obs_used_a2"] = np.array(N_a2, dtype=np.int32)
        out[prefix + "n_obs_used_b1"] = np.array(N_b1, dtype=np.int32)
        out[prefix + "n_obs_used_b2"] = np.array(N_b2, dtype=np.int32)

    # run 2-way interactions
    n_int = 0
    for ic in inter2:
        needed = [ic.a1, ic.a2, ic.b1, ic.b2]
        if not ok_interaction_groups(needed):
            print(f"[SKIP] {ic.name} (A_min gate failed)")
            continue
        for c in conditions:
            save_dod_contrast(ic.name, ic.a1, ic.a2, ic.b1, ic.b2, c)
            n_int += 1
    print("[OK] 2-way interaction contrasts saved:", n_int)

    # run 3-way interaction (custom)
    # age×sex×genotype:
    # left  = (2F_wt-2F_dKI) - (2M_wt-2M_dKI)
    # right = (4F_wt-4F_dKI) - (4M_wt-4M_dKI)
    # overall = left - right
    name3 = "age×sex×genotype:([2F_wt-2F_dKI]-[2M_wt-2M_dKI]) - ([4F_wt-4F_dKI]-[4M_wt-4M_dKI])"
    g_2F_wt = "age=2m&sex=F&genotype=wt"
    g_2F_ki = "age=2m&sex=F&genotype=dKI"
    g_2M_wt = "age=2m&sex=M&genotype=wt"
    g_2M_ki = "age=2m&sex=M&genotype=dKI"
    g_4F_wt = "age=4m&sex=F&genotype=wt"
    g_4F_ki = "age=4m&sex=F&genotype=dKI"
    g_4M_wt = "age=4m&sex=M&genotype=wt"
    g_4M_ki = "age=4m&sex=M&genotype=dKI"

    needed3 = [g_2F_wt,g_2F_ki,g_2M_wt,g_2M_ki,g_4F_wt,g_4F_ki,g_4M_wt,g_4M_ki]
    if all(g in set(groups) for g in needed3) and ok_interaction_groups(needed3):
        n3 = 0
        for c in conditions:
            # compute overall = (2F_wt - 2F_ki - 2M_wt + 2M_ki) - (4F_wt - 4F_ki - 4M_wt + 4M_ki)
            # = 2F_wt - 2F_ki - 2M_wt + 2M_ki - 4F_wt + 4F_ki + 4M_wt - 4M_ki
            # We'll reuse save_dod_contrast twice then subtract? But we'd need intermediate SE.
            # Do directly with SE propagation across 8 groups.

            # Q
            terms = [
                (g_2F_wt, +1), (g_2F_ki, -1), (g_2M_wt, -1), (g_2M_ki, +1),
                (g_4F_wt, -1), (g_4F_ki, +1), (g_4M_wt, +1), (g_4M_ki, -1),
            ]
            q_obs = None
            se2 = None
            for g, sgn in terms:
                qg = get_cell(g,c,"q_obs").astype(float)
                qlo = get_cell(g,c,"q_ci_lo").astype(float)
                qhi = get_cell(g,c,"q_ci_hi").astype(float)
                seg = approx_se_from_ci(qlo, qhi, z=Z)
                if q_obs is None:
                    q_obs = sgn * qg
                    se2 = seg**2
                else:
                    q_obs = q_obs + sgn * qg
                    se2 = se2 + seg**2
            se = np.sqrt(se2)
            q_lo = q_obs - Z*se
            q_hi = q_obs + Z*se

            dq_at_p = q_obs[p_idx]
            dq_at_p_lo = q_lo[p_idx]
            dq_at_p_hi = q_hi[p_idx]
            auc_dq = auc_trapz(q_obs, p_grid)
            auc_lo = auc_trapz(q_lo, p_grid)
            auc_hi = auc_trapz(q_hi, p_grid)

            # PDF
            p_obs = None
            se2p = None
            for g, sgn in terms:
                pg = get_cell(g,c,"pdf_obs").astype(float)
                plo = get_cell(g,c,"pdf_ci_lo").astype(float)
                phi = get_cell(g,c,"pdf_ci_hi").astype(float)
                seg = approx_se_from_ci(plo, phi, z=Z)
                if p_obs is None:
                    p_obs = sgn * pg
                    se2p = seg**2
                else:
                    p_obs = p_obs + sgn * pg
                    se2p = se2p + seg**2
            sep = np.sqrt(se2p)
            p_lo = p_obs - Z*sep
            p_hi = p_obs + Z*sep

            if bin_centers is not None:
                auc_dpdf = auc_trapz(p_obs, bin_centers)
                auc_dpdf_lo = auc_trapz(p_lo, bin_centers)
                auc_dpdf_hi = auc_trapz(p_hi, bin_centers)
            else:
                auc_dpdf = np.nan; auc_dpdf_lo = np.nan; auc_dpdf_hi = np.nan

            prefix = f"contrast__{name3}__condition__{c}__field__"
            out[prefix + "formula"] = np.array("2F_wt-2F_dKI-2M_wt+2M_dKI-4F_wt+4F_dKI+4M_wt-4M_dKI", dtype=object)

            out[prefix + "dq_obs"] = q_obs.astype(SAVE_DTYPE)
            out[prefix + "dq_ci_lo"] = q_lo.astype(SAVE_DTYPE)
            out[prefix + "dq_ci_hi"] = q_hi.astype(SAVE_DTYPE)

            out[prefix + "dpdf_obs"] = p_obs.astype(SAVE_DTYPE)
            out[prefix + "dpdf_ci_lo"] = p_lo.astype(SAVE_DTYPE)
            out[prefix + "dpdf_ci_hi"] = p_hi.astype(SAVE_DTYPE)

            out[prefix + "dq_at_p_points"] = p_points_snap.astype(SAVE_DTYPE)
            out[prefix + "dq_at_p_obs"] = dq_at_p.astype(SAVE_DTYPE)
            out[prefix + "dq_at_p_ci_lo"] = dq_at_p_lo.astype(SAVE_DTYPE)
            out[prefix + "dq_at_p_ci_hi"] = dq_at_p_hi.astype(SAVE_DTYPE)

            out[prefix + "AUC_dq_obs"] = np.array(auc_dq, dtype=float)
            out[prefix + "AUC_dq_ci_lo"] = np.array(auc_lo, dtype=float)
            out[prefix + "AUC_dq_ci_hi"] = np.array(auc_hi, dtype=float)

            out[prefix + "AUC_dpdf_obs"] = np.array(auc_dpdf, dtype=float)
            out[prefix + "AUC_dpdf_ci_lo"] = np.array(auc_dpdf_lo, dtype=float)
            out[prefix + "AUC_dpdf_ci_hi"] = np.array(auc_dpdf_hi, dtype=float)

            # counts for audit
            for g, _ in terms:
                A, N = get_counts(g)
                out[prefix + f"n_animals__{g}"] = np.array(A, dtype=np.int32)
                out[prefix + f"n_obs_used__{g}"] = np.array(N, dtype=np.int32)

            n3 += 1
        print("[OK] 3-way interaction contrasts saved:", n3)
    else:
        print("[SKIP] 3-way interaction (missing groups or A_min gate failed)")

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    print("[OK] Saved FP8b:", out_path)
    print("[INFO] Keys saved:", len(out))


if __name__ == "__main__":
    main()
