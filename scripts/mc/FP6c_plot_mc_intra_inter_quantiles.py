#!/usr/bin/env python3


# %%
"""
FP6c — Plot FP6 bootstrap summaries (PDF + quantiles) from FP6b FAST artifact.

Consumes:
  results/<dataset>/mc/mc_dist/fp6b_bootstrap_mc_fp6_conditions_FAST.npz

Produces (PNG + PDF):
  fig/<dataset>/mc/FP6/
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from shared_code.fun_paths import get_paths

# =========================
# CONFIG
# =========================
DATASET = "ines_abdallah"
TIMECOURSE_FOLDER = "Timecourses_updated_03052024"
COGNITIVE_FILE = "ROIs.xlsx"
ANAT_LABELS_FILE = "41_Allen.txt"

MC_DIST_DIRNAME = "mc_dist"
FP6B_NAME = "fp6b_bootstrap_mc_fp6_conditions_FAST.npz"

SAVE_PNG = True
SAVE_PDF = True
DPI = 200

# CI & lines
ALPHA_BAND = 0.20
LW = 1.2

# PDF x-limits (set None to see full [-1,1])
X_LIM = (-0.8, 0.8)

# Show extra debug prints
DEBUG_PRINT = True

# =========================
# Helpers
# =========================
Y_LOG = True
Y_LOG_FLOOR = 1e-3   # tune (1e-5 to 1e-3) depending on how sparse tails are

def set_log_pdf_axis(ax):
    if Y_LOG:
        ax.set_yscale("log")
        ax.set_ylim(bottom=Y_LOG_FLOOR)

def _bin_centers(bins: np.ndarray) -> np.ndarray:
    return (bins[:-1] + bins[1:]) * 0.5

def _maybe_xlim(ax, xlim):
    if xlim is not None:
        ax.set_xlim(*xlim)

def _load_series(d, key: str):
    """Load one condition block saved by FP6b (key__field)."""
    need = [
        f"{key}__pdf_obs", f"{key}__pdf_ci_lo", f"{key}__pdf_ci_hi",
        f"{key}__q_obs",   f"{key}__q_ci_lo",   f"{key}__q_ci_hi",
        f"{key}__n_animals",
    ]
    missing = [k for k in need if k not in d.files]
    if missing:
        raise KeyError(f"Missing keys for condition '{key}': {missing}")
    out = dict(
        pdf_obs=d[f"{key}__pdf_obs"].astype(np.float32),
        pdf_lo=d[f"{key}__pdf_ci_lo"].astype(np.float32),
        pdf_hi=d[f"{key}__pdf_ci_hi"].astype(np.float32),
        q_obs=d[f"{key}__q_obs"].astype(np.float32),
        q_lo=d[f"{key}__q_ci_lo"].astype(np.float32),
        q_hi=d[f"{key}__q_ci_hi"].astype(np.float32),
        n_animals=int(d[f"{key}__n_animals"]),
    )
    return out

def _assert_ok_pdf(bins, pdf, name):
    # NaNs allowed but should be rare; negative density is not OK
    if np.any(pdf[np.isfinite(pdf)] < -1e-8):
        raise ValueError(f"[BAD] {name}: negative density found.")
    # integral sanity (approx)
    widths = np.diff(bins)
    area = float(np.nansum(pdf * widths))
    if not (0.5 <= area <= 1.5):
        # Don’t crash, but warn loudly — bootstrap subsampling may make it imperfect
        print(f"[WARN] {name}: PDF area={area:.3f} (expected ~1).")

def _assert_ci_order(lo, hi, name):
    bad = np.any((lo > hi) & np.isfinite(lo) & np.isfinite(hi))
    if bad:
        raise ValueError(f"[BAD] {name}: CI has lo>hi somewhere.")

def _plot_pdf(ax, x, bins, s_obs, s_null, *, color, label):
    # observed
    ax.plot(x, s_obs["pdf_obs"], lw=LW, color=color, ls="-", label=f"{label} (obs)")
    ax.fill_between(x, s_obs["pdf_lo"], s_obs["pdf_hi"], color=color, alpha=ALPHA_BAND, linewidth=0)
    # null
    ax.plot(x, s_null["pdf_obs"], lw=LW, color=color, ls="--", label=f"{label} (null)")
    ax.fill_between(x, s_null["pdf_lo"], s_null["pdf_hi"], color=color, alpha=ALPHA_BAND, linewidth=0)

def _plot_q(ax, p, s_obs, s_null, *, color, label):
    ax.plot(p, s_obs["q_obs"], lw=LW, color=color, ls="-", label=f"{label} (obs)")
    ax.fill_between(p, s_obs["q_lo"], s_obs["q_hi"], color=color, alpha=ALPHA_BAND, linewidth=0)
    ax.plot(p, s_null["q_obs"], lw=LW, color=color, ls="--", label=f"{label} (null)")
    ax.fill_between(p, s_null["q_lo"], s_null["q_hi"], color=color, alpha=ALPHA_BAND, linewidth=0)

def _save(fig, outbase: Path):
    fig.tight_layout()
    if SAVE_PNG:
        fig.savefig(outbase.with_suffix(".png"), dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    if SAVE_PDF:
        fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def _print_tail_debug(p_grid, s, name):
    # report q01 q50 q99 if available
    def idx(p):
        return int(np.argmin(np.abs(p_grid - p)))
    i01, i50, i99 = idx(0.01), idx(0.5), idx(0.99)
    q = s["q_obs"]
    print(f"  {name}: q01={float(q[i01]): .4f}  q50={float(q[i50]): .4f}  q99={float(q[i99]): .4f}")

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
fp6b_path = mc_dir / MC_DIST_DIRNAME / FP6B_NAME
if not fp6b_path.exists():
    raise FileNotFoundError(fp6b_path)

d = np.load(fp6b_path, allow_pickle=True)

bins = d["bins"].astype(np.float32)
p_grid = d["p_grid"].astype(np.float32)
x = _bin_centers(bins)

params = json.loads(d["params_json"].item())
conds = params.get("conditions", [])
print("[FP6c] Loaded:", fp6b_path)
print("[FP6c] n_conditions:", len(conds))

# Output dir: fig/<dataset>/mc/FP6/
out_dir = Path(paths["f_mod"]) / "mc_dist"


out_dir.mkdir(parents=True, exist_ok=True)

# Required keys (hard fail if missing)
REQ = [
    "obs_all", "null_all",
    "obs_intra", "null_intra",
    "obs_inter", "null_inter",
    "obs_trimer", "null_trimer",
    "obs_tetramer", "null_tetramer",
]
for k in REQ:
    if not any(s.startswith(f"{k}__") for s in d.files):
        raise KeyError(f"Missing condition '{k}' in FP6b artifact. Available sample keys: {sorted(d.files)[:20]}")

# Load series
S = {k: _load_series(d, k) for k in REQ}

# Sanity checks
for k in REQ:
    _assert_ci_order(S[k]["pdf_lo"], S[k]["pdf_hi"], f"{k} pdf")
    _assert_ci_order(S[k]["q_lo"], S[k]["q_hi"], f"{k} q")
    _assert_ok_pdf(bins, S[k]["pdf_obs"], f"{k} pdf_obs")

if DEBUG_PRINT:
    print("[FP6c] Tail debug (observed):")
    _print_tail_debug(p_grid, S["obs_all"], "obs_all")
    _print_tail_debug(p_grid, S["obs_intra"], "obs_intra")
    _print_tail_debug(p_grid, S["obs_inter"], "obs_inter")
    _print_tail_debug(p_grid, S["obs_trimer"], "obs_trimer")
    _print_tail_debug(p_grid, S["obs_tetramer"], "obs_tetramer")
    print("[FP6c] Tail debug (null):")
    _print_tail_debug(p_grid, S["null_all"], "null_all")
    _print_tail_debug(p_grid, S["null_intra"], "null_intra")
    _print_tail_debug(p_grid, S["null_inter"], "null_inter")
    _print_tail_debug(p_grid, S["null_trimer"], "null_trimer")
    _print_tail_debug(p_grid, S["null_tetramer"], "null_tetramer")

# =========================
# FIG 1 — Global PDF
# =========================
fig, ax = plt.subplots(figsize=(6.6, 4.6))
_plot_pdf(ax, x, bins, S["obs_all"], S["null_all"], color="C0", label="All")
ax.set_title("Global MC distribution")
ax.set_xlabel("MC value")
ax.set_ylabel("Density")
_maybe_xlim(ax, X_LIM)
ax.legend(frameon=False)
set_log_pdf_axis(ax)
_save(fig, out_dir / "FP6c_global_pdf")

# =========================
# FIG 2 — Global quantiles
# =========================
fig, ax = plt.subplots(figsize=(6.6, 4.6))
_plot_q(ax, p_grid, S["obs_all"], S["null_all"], color="C0", label="All")
ax.set_title("Global MC quantiles")
ax.set_xlabel("p")
ax.set_ylabel("Q(p)")
ax.set_xlim(0, 1)
ax.legend(frameon=False)
_save(fig, out_dir / "FP6c_global_quantiles")

# =========================
# FIG 3 — Intra vs inter PDF
# =========================
fig, ax = plt.subplots(figsize=(6.8, 4.8))
_plot_pdf(ax, x, bins, S["obs_intra"], S["null_intra"], color="C0", label="Intra")
_plot_pdf(ax, x, bins, S["obs_inter"], S["null_inter"], color="C1", label="Inter")
ax.set_title("Intra vs inter-module MC (PDF)")
ax.set_xlabel("MC value")
ax.set_ylabel("Density")
_maybe_xlim(ax, X_LIM)
ax.legend(frameon=False, ncols=2)
set_log_pdf_axis(ax)
_save(fig, out_dir / "FP6c_intra_inter_pdf")

# =========================
# FIG 4 — Intra vs inter quantiles
# =========================
fig, ax = plt.subplots(figsize=(6.8, 4.8))
_plot_q(ax, p_grid, S["obs_intra"], S["null_intra"], color="C0", label="Intra")
_plot_q(ax, p_grid, S["obs_inter"], S["null_inter"], color="C1", label="Inter")
ax.set_title("Intra vs inter-module MC (quantiles)")
ax.set_xlabel("p")
ax.set_ylabel("Q(p)")
ax.set_xlim(0, 1)
ax.legend(frameon=False, ncols=2)
_save(fig, out_dir / "FP6c_intra_inter_quantiles")

# =========================
# FIG 5 — Trimer vs tetramer PDF
# =========================
fig, ax = plt.subplots(figsize=(6.8, 4.8))
_plot_pdf(ax, x, bins, S["obs_trimer"], S["null_trimer"], color="C2", label="Trimer")
_plot_pdf(ax, x, bins, S["obs_tetramer"], S["null_tetramer"], color="C3", label="Tetramer")
ax.set_title("Trimer vs tetramer MC (PDF)")
ax.set_xlabel("MC value")
ax.set_ylabel("Density")
_maybe_xlim(ax, X_LIM)
ax.legend(frameon=False, ncols=2)
set_log_pdf_axis(ax)
_save(fig, out_dir / "FP6c_trimer_tetramer_pdf")

# =========================
# FIG 6 — Trimer vs tetramer quantiles
# =========================
fig, ax = plt.subplots(figsize=(6.8, 4.8))
_plot_q(ax, p_grid, S["obs_trimer"], S["null_trimer"], color="C2", label="Trimer")
_plot_q(ax, p_grid, S["obs_tetramer"], S["null_tetramer"], color="C3", label="Tetramer")
ax.set_title("Trimer vs tetramer MC (quantiles)")
ax.set_xlabel("p")
ax.set_ylabel("Q(p)")
ax.set_xlim(0, 1)
ax.legend(frameon=False, ncols=2)
_save(fig, out_dir / "FP6c_trimer_tetramer_quantiles")

# =========================
# Optional FIG 7 — 2×2 PDFs for atomic conditions
# =========================
atomic = [
    ("obs_intra_trimer", "null_intra_trimer", "Intra × Trimer", "C0"),
    ("obs_intra_tetramer", "null_intra_tetramer", "Intra × Tetramer", "C0"),
    ("obs_inter_trimer", "null_inter_trimer", "Inter × Trimer", "C1"),
    ("obs_inter_tetramer", "null_inter_tetramer", "Inter × Tetramer", "C1"),
]

# only plot if these exist
if all(any(s.startswith(f"{k}__") for s in d.files) for k, _, _, _ in atomic):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (k_obs, k_null, title, col) in zip(axes, atomic):
        so = _load_series(d, k_obs)
        sn = _load_series(d, k_null)
        _plot_pdf(ax, x, bins, so, sn, color=col, label=title.replace(" × ", "-"))
        ax.set_title(title)
        ax.set_xlabel("MC value")
        ax.set_ylabel("Density")
        _maybe_xlim(ax, X_LIM)
        ax.legend(frameon=False, fontsize=9)
    set_log_pdf_axis(ax)
    _save(fig, out_dir / "FP6c_2x2_atomic_pdfs")

#
# =========================
# FIG 8 — Observed only: intra/inter × trimer/tetramer (single plot)
# =========================
obs_it_tt = [
    ("obs_intra_trimer",   "Intra–Trimer",   "C0", "-"),
    ("obs_intra_tetramer", "Intra–Tetramer", "C1", "-"),
    ("obs_inter_trimer",   "Inter–Trimer",   "C2", "-"),
    ("obs_inter_tetramer", "Inter–Tetramer", "C3", "-"),
]


fig, ax = plt.subplots(figsize=(7.2, 5.0))
for key, label, color, ls in obs_it_tt:
    s = _load_series(d, key)
    ax.plot(x, s["pdf_obs"], lw=LW, color=color, ls=ls, label=label)
    ax.fill_between(x, s["pdf_lo"], s["pdf_hi"], color=color, alpha=ALPHA_BAND, linewidth=0)
ax.set_title("Observed MC — intra/inter × trimer/tetramer")
ax.set_xlabel("MC value")
ax.set_ylabel("Density")
_maybe_xlim(ax, X_LIM)
set_log_pdf_axis(ax)  # <-- if you want log here too
ax.legend(frameon=False, ncols=2)

_save(fig, out_dir / "FP6c_obs_only_intra_inter_trimer_tetramer_pdf")

# =========================
# FIG 9 — Observed only: intra/inter × trimer/tetramer quantiles
# =========================

fig, ax = plt.subplots(figsize=(7.2, 5.0))
for key, label, color, ls in obs_it_tt:
    s = _load_series(d, key)
    ax.plot(p_grid, s["q_obs"], lw=LW, color=color, ls=ls, label=label)

ax.set_title("Observed MC quantiles — intra/inter × trimer/tetramer")
ax.set_xlabel("p")
ax.set_ylabel("Q(p)")
ax.set_xlim(0, 1)
ax.legend(frameon=False, ncols=2)
_save(fig, out_dir / "FP6c_obs_only_intra_inter_trimer_tetramer_quantiles")


print("[DONE] FP6c saved figures to:", out_dir)

#

# %%
