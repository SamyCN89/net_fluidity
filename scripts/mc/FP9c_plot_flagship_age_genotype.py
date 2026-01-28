#!/usr/bin/env python3
# %%
"""
FP9c — Flagship figure
Age × Genotype interaction on ΔQ(p), intra-trimer edges only

Consumes:
  - results/<dataset>/mc/mc_dist/fp8b_group_tail_contrasts.npz

Produces:
  - fig/<dataset>/FP9c_age_genotype_intra_trimer.pdf
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from shared_code.fun_paths import get_paths
from shared_code.fun_utils import set_figure_params

# =============================================================================
# CONFIG
# =============================================================================
DATASET = "ines_abdallah"
FP8B_NAME = "fp8b_group_tail_contrasts.npz"

COND = "obs_intra_trimer"

# contrasts (must exist in FP8b)
C_WT   = "age@genotype=wt:2m-4m"
C_DKI  = "age@genotype=dKI:2m-4m"
C_INT  = "age×genotype:(2m_wt-2m_dKI)-(4m_wt-4m_dKI)"

P_LIM = (0.01, 0.99)

OUTNAME = "FP9c_age_genotype_intra_trimer.pdf"
DPI = 300
# =============================================================================


def ck(contrast: str, cond: str, field: str) -> str:
    return f"contrast__{contrast}__condition__{cond}__field__{field}"


# =============================================================================
# LOAD
# =============================================================================
paths = get_paths()
fp8b_path = Path(paths["results"]) / "mc" / "mc_dist" / FP8B_NAME
z = np.load(fp8b_path, allow_pickle=True)
d = {k: z[k] for k in z.files}

p = np.asarray(d["p_grid"], dtype=float)

def load_curve(cname):
    return (
        np.asarray(d[ck(cname, COND, "dq_obs")], dtype=float),
        np.asarray(d[ck(cname, COND, "dq_ci_lo")], dtype=float),
        np.asarray(d[ck(cname, COND, "dq_ci_hi")], dtype=float),
    )

dq_wt, lo_wt, hi_wt   = load_curve(C_WT)
dq_dk, lo_dk, hi_dk   = load_curve(C_DKI)
dq_int, lo_i, hi_i    = load_curve(C_INT)

mask = (p >= P_LIM[0]) & (p <= P_LIM[1])
p = p[mask]

dq_wt, lo_wt, hi_wt = dq_wt[mask], lo_wt[mask], hi_wt[mask]
dq_dk, lo_dk, hi_dk = dq_dk[mask], lo_dk[mask], hi_dk[mask]
dq_int, lo_i, hi_i  = dq_int[mask], lo_i[mask], hi_i[mask]

# =============================================================================
# PLOT
# =============================================================================
set_figure_params()  # your house style

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)

panels = [
    ("WT: age effect (2m − 4m)", dq_wt, lo_wt, hi_wt),
    ("dKI: age effect (2m − 4m)", dq_dk, lo_dk, hi_dk),
    ("Interaction (WT − dKI)", dq_int, lo_i, hi_i),
]

for ax, (title, y, lo, hi) in zip(axes, panels):
    ax.plot(p, y, lw=2)
    ax.fill_between(p, lo, hi, alpha=0.3)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_title(title)
    ax.set_xlabel("Quantile p")
    ax.set_xlim(P_LIM)

axes[0].set_ylabel("ΔQ(p)")

fig.suptitle("Age × Genotype interaction in intra-trimer metaconnectivity", y=1.05)
fig.tight_layout()

# =============================================================================
# SAVE
# =============================================================================
fig_dir = Path(paths["f_mod"]) / 'mc_groups'
fig_dir.mkdir(parents=True, exist_ok=True)
out = fig_dir / OUTNAME
fig.savefig(out, dpi=DPI)
plt.close(fig)

print("[OK] FP9c saved:", out)
