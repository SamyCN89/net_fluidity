#!/usr/bin/env python3
# %%
"""
FP4 — Bootstrap genotype tail differences (no CLI)

Consumes:
  - FP3 mc_indexed_*.npz
  - FP0 cog_data_filtered_*.csv (genotype/sex/behavior)
  - FP0/FP1 session labels: mouse_ids_ts + age_ts

Produces:
  - results/<dataset>/mc_bootstrap/mc_bootstrap_genotype_*.npz
  - results/<dataset>/figures/FP4_genotype_tails_*.png
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shared_code.fun_paths import get_paths


# =========================
# CONFIG
# =========================
DATASET = "ines_abdallah"
TIMECOURSE_FOLDER = "Timecourses_updated_03052024"
COGNITIVE_FILE = "ROIs.xlsx"
ANAT_LABELS_FILE = "41_Allen.txt"

# Pick FP3 explicitly (recommended)
FP3_FILE = "mc_indexed_ref=wt_2m_animals=126_E=820.npz"  # EDIT

# Comparison
AGE = "2m"                 # analyze within this age
GROUP_A = "wt"             # genotype A
GROUP_B = "dKI"            # genotype B

# What to analyze
SUBSET = "all"             # "all" | "module" | "trimer"
MODULE_ID = 1              # used if SUBSET="module"
TRIMER_ONLY = True         # used if SUBSET="trimer"

# Bootstrap
N_BOOT = 2000
SEED = 0
BINS = np.linspace(-0.8, 1.0, 120)

# Outputs
OUT_SUBDIR = "mc_bootstrap"
FIG_SUBDIR = "figures"
OVERWRITE = False
#%%

# =========================
# Helpers
# =========================
def find_latest(folder: Path, pattern: str) -> Path:
    hits = sorted(folder.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No files matching {pattern} in {folder}")
    return hits[-1]


def _tail_metrics_from_samples(x: np.ndarray) -> dict:
    """Minimal, robust tail descriptors (edit later)."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0)
    q01, q05, q50, q95, q99 = np.quantile(x, [0.01, 0.05, 0.5, 0.95, 0.99])
    return dict(
        n=int(x.size),
        q01=float(q01),
        q05=float(q05),
        q50=float(q50),
        q95=float(q95),
        q99=float(q99),
        width50=float(q95 - q05),
        width_extreme=float(q99 - q01),
        asymmetry=float((q95 - q50) - (q50 - q05)),
    )


def _pdf(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
    x = x[np.isfinite(x)]
    h, _ = np.histogram(x, bins=bins, density=True)
    return h


def _bootstrap_delta_pdf(xA: np.ndarray, xB: np.ndarray, bins: np.ndarray, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    nA, nB = xA.size, xB.size
    if nA == 0 or nB == 0:
        raise RuntimeError("Empty group after filtering; cannot bootstrap.")
    # observed
    d_obs = _pdf(xA, bins) - _pdf(xB, bins)

    # bootstrap
    boots = np.empty((n_boot, d_obs.size), dtype=np.float32)
    for b in range(n_boot):
        sA = xA[rng.integers(0, nA, size=nA)]
        sB = xB[rng.integers(0, nB, size=nB)]
        boots[b] = _pdf(sA, bins) - _pdf(sB, bins)

    lo = np.quantile(boots, 0.025, axis=0)
    hi = np.quantile(boots, 0.975, axis=0)
    return d_obs, lo, hi, boots


def _apply_subset_mask(mc_val: np.ndarray, mc_mod_idx: np.ndarray, mc_nplets_index: np.ndarray):
    """Return boolean mask over K features."""
    K = mc_val.shape[1]
    m = np.ones(K, dtype=bool)
    if SUBSET == "module":
        m &= (mc_mod_idx == MODULE_ID)
    if SUBSET == "trimer":
        m &= (mc_nplets_index > 0) if TRIMER_ONLY else (mc_nplets_index >= 0)
    return m
#%%

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
preproc_dir = Path(paths["preprocessed"])

# Load FP3
fp3_path = mc_dir / "mc_indexed" / FP3_FILE
d3 = np.load(fp3_path, allow_pickle=True)
mc_val = d3["mc_val_tril"]          # (A, K) metaconnectivity values
mc_mod_idx = d3["mc_mod_idx"]       # (K,) module identities
mc_nplets_index = d3["mc_nplets_index"]  # (K,) trimer/tetramer identities

# Load session labels from canonical bundle (FP0 output)
canon_npz = preproc_dir / "ts_and_meta_ines_abdallah.npz"
d0 = np.load(canon_npz, allow_pickle=True)
mouse_ids_ts = d0["mouse_ids_ts"].astype(str)   # (A,)
age_ts = d0["age_ts"].astype(str)               # (A,)

#
A = mc_val.shape[0]
assert mouse_ids_ts.shape[0] == A and age_ts.shape[0] == A, "Session label length mismatch."

# Load cognitive table (per-mouse)
cog_csv_path = find_latest(preproc_dir, "cog_data_filtered_*.csv")
cog = pd.read_csv(cog_csv_path)
cog["Name"] = cog["Name"].astype(str)

# Map mouse -> genotype (per-mouse) into per-session
geno_map = dict(zip(cog["Name"].astype(str), cog["Genotype"].astype(str)))
genotype_ts = np.array([geno_map.get(mid, "NA") for mid in mouse_ids_ts], dtype=str)

# Filter sessions: age + valid genotype
keep_age = (age_ts == AGE)
keep_geno = np.isin(genotype_ts, [GROUP_A, GROUP_B])
keep_sessions = keep_age & keep_geno

# Subset features (K mask)
feat_mask = _apply_subset_mask(mc_val, mc_mod_idx, mc_nplets_index)

# Flatten to samples for each group (simple FP4 baseline)
xA = mc_val[keep_sessions & (genotype_ts == GROUP_A)][:, feat_mask].ravel()
xB = mc_val[keep_sessions & (genotype_ts == GROUP_B)][:, feat_mask].ravel()

print("FP4")
print("  sessions kept:", int(keep_sessions.sum()), "/", A, "age:", AGE)
print("  group sizes (sessions):",
      int(np.sum(keep_sessions & (genotype_ts == GROUP_A))),
      int(np.sum(keep_sessions & (genotype_ts == GROUP_B))))
print("  samples:", xA.size, xB.size, "subset:", SUBSET)

#%%

# =========================
# ANALYSIS
# =========================

# Tail metrics (observed)
mA = _tail_metrics_from_samples(xA)
mB = _tail_metrics_from_samples(xB)
mDelta = {k: (mA[k] - mB[k]) for k in mA.keys() if k in mB and k != "n"}

# Bootstrap delta-PDF
d_obs, d_lo, d_hi, boots = _bootstrap_delta_pdf(xA, xB, BINS, N_BOOT, SEED)

#%%
# =========================
# SAVE + FIGURES
# =========================

# Save artifact
out_dir = mc_dir / OUT_SUBDIR
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / (
    f"mc_bootstrap_genotype_{GROUP_A}-vs-{GROUP_B}_age={AGE}"
    f"_subset={SUBSET}"
    f"_runs={N_BOOT}.npz"
)
if out_path.exists() and not OVERWRITE:
    raise FileExistsError(f"{out_path} exists. Set OVERWRITE=True to replace.")

params = dict(
    dataset=DATASET,
    fp3_path=str(fp3_path),
    canon_npz=str(canon_npz),
    cog_csv_path=str(cog_csv_path),
    age=AGE,
    group_a=GROUP_A,
    group_b=GROUP_B,
    subset=SUBSET,
    module_id=int(MODULE_ID),
    trimer_only=bool(TRIMER_ONLY),
    n_boot=int(N_BOOT),
    seed=int(SEED),
    bins=list(map(float, BINS)),
)

np.savez_compressed(
    out_path,
    bins=BINS,
    delta_pdf=d_obs,
    delta_pdf_ci_lo=d_lo,
    delta_pdf_ci_hi=d_hi,
    tail_A_json=json.dumps(mA, sort_keys=True),
    tail_B_json=json.dumps(mB, sort_keys=True),
    tail_delta_json=json.dumps(mDelta, sort_keys=True),
    params_json=json.dumps(params, sort_keys=True),
)

print("[OK] Saved FP4 artifact:", out_path)

# Figure (single)
fig_dir = Path(paths["results"]) / FIG_SUBDIR
fig_dir.mkdir(parents=True, exist_ok=True)
fig_path = fig_dir / out_path.with_suffix("").name
fig_path = fig_path.with_suffix(".png")

centers = 0.5 * (BINS[:-1] + BINS[1:])
plt.figure()
plt.plot(centers, d_obs, label=f"{GROUP_A}-{GROUP_B}")
plt.fill_between(centers, d_lo, d_hi, alpha=0.2, label="95% bootstrap CI")
plt.axhline(0, linewidth=1)
plt.xlabel("MC value")
plt.ylabel("Δ PDF")
plt.title(f"FP4 genotype tails | age={AGE} | subset={SUBSET}")
plt.legend()
plt.tight_layout()
plt.savefig(fig_path, dpi=200)
plt.close()
print("[OK] Saved figure:", fig_path)
# %%
