#!/usr/bin/env python3
# %%
"""
FP6cG — Plot group-wise MC distribution profiles (PDF + quantiles) from FP6bG animal-bootstrap artifact.

Consumes:
  results/<dataset>/mc/mc_dist/fp6b_groups_bootstrap_mc_conditions_ANIMALBOOT.npz

Produces:
  fig/<dataset>/mc/FP6G/<group>/
    - global PDF + quantiles (obs vs null)
    - intra vs inter PDF + quantiles (obs vs null)
    - trimer vs tetramer PDF + quantiles (obs vs null)
    - 2×2 atomic PDFs (obs vs null) if present
    - obs-only 4-condition PDF + quantiles

Notes:
- Uses log-y for PDFs by default to show tails.
- No seaborn; no forced color palettes beyond matplotlib defaults.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from shared_code.fun_paths import get_paths

# -------------------------
# GLOBAL STYLE
# -------------------------
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False

# =========================
# CONFIG
# =========================
DATASET = "ines_abdallah"
TIMECOURSE_FOLDER = "Timecourses_updated_03052024"
COGNITIVE_FILE = "ROIs.xlsx"
ANAT_LABELS_FILE = "41_Allen.txt"

MC_DIST_DIRNAME = "mc_dist"
FP6BG_NAME = "fp6b_groups_bootstrap_mc_conditions_ANIMALBOOT.npz"

SAVE_PNG = True
SAVE_PDF = True
DPI = 200

# Appearance
ALPHA_BAND = 0.20
LW = 1.2

Y_LOG = True
Y_LOG_FLOOR = 1e-6  # tails visibility; tune if needed

FILL_NULL_CI = True

X_LIM = (-0.8, 0.8)

DEBUG_PRINT = True

# If True: only plot some groups by regex
GROUP_REGEX = None  # e.g. r"wt_2m" or r"geno=WT" etc.


# =========================
# Helpers
# =========================
def _sanitize_group_name(g: str, maxlen: int = 140) -> str:
    """
    Make group string safe for Windows-like filesystems (exFAT/NTFS).
    Removes <>:"/\\|?* and control chars, trims trailing dots/spaces.
    Adds a short hash to avoid collisions after sanitization.
    """
    g = str(g)

    # replace forbidden chars (Windows set, safe everywhere)
    g_clean = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", g)

    # collapse whitespace
    g_clean = re.sub(r"\s+", "_", g_clean).strip(" .")

    # avoid empty
    if not g_clean:
        g_clean = "group"

    # add stable short hash so two different groups don't collide after cleaning
    h = hashlib.sha1(g.encode("utf-8")).hexdigest()[:8]

    # limit length
    g_clean = g_clean[:maxlen].rstrip(" ._")

    return f"{g_clean}__{h}"


def _bin_centers(bins: np.ndarray) -> np.ndarray:
    return (bins[:-1] + bins[1:]) * 0.5


def _save(fig, outbase: Path):
    fig.tight_layout()
    if SAVE_PNG:
        fig.savefig(
            outbase.with_suffix(".png"), dpi=DPI, bbox_inches="tight", pad_inches=0.02
        )
    if SAVE_PDF:
        fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def set_log_pdf_axis(ax):
    if Y_LOG:
        ax.set_yscale("log")
        ax.set_ylim(bottom=Y_LOG_FLOOR)


def _maybe_xlim(ax, xlim):
    if xlim is not None:
        ax.set_xlim(*xlim)


def _assert_ci_order(lo, hi, name):
    bad = np.any((lo > hi) & np.isfinite(lo) & np.isfinite(hi))
    if bad:
        raise ValueError(f"[BAD] {name}: CI has lo>hi somewhere.")


def _assert_ok_pdf(bins, pdf, name, debug=False):
    if np.any(pdf[np.isfinite(pdf)] < -1e-8):
        raise ValueError(f"[BAD] {name}: negative density found.")
    widths = np.diff(bins)
    area = float(np.nansum(pdf * widths))
    if debug and not (0.3 <= area <= 1.7):
        print(
            f"[WARN] {name}: PDF area={area:.3f} (expected ~1-ish; boot subsampling can move it)."
        )


def _print_tail_debug(p_grid, s, name):
    def idx(p):
        return int(np.argmin(np.abs(p_grid - p)))

    i01, i50, i99 = idx(0.01), idx(0.5), idx(0.99)
    q = s["q_obs"]
    print(
        f"  {name}: q01={float(q[i01]): .4f}  q50={float(q[i50]): .4f}  q99={float(q[i99]): .4f}"
    )


def _load_series(d, g: str, key: str):
    need = [
        f"{g}__{key}__pdf_obs",
        f"{g}__{key}__pdf_ci_lo",
        f"{g}__{key}__pdf_ci_hi",
        f"{g}__{key}__q_obs",
        f"{g}__{key}__q_ci_lo",
        f"{g}__{key}__q_ci_hi",
        f"{g}__{key}__n_animals",
        f"{g}__{key}__n_pool",
    ]
    missing = [k for k in need if k not in d.files]
    if missing:
        raise KeyError(f"Missing keys for group '{g}' condition '{key}': {missing}")
    return dict(
        pdf_obs=d[need[0]].astype(np.float32),
        pdf_lo=d[need[1]].astype(np.float32),
        pdf_hi=d[need[2]].astype(np.float32),
        q_obs=d[need[3]].astype(np.float32),
        q_lo=d[need[4]].astype(np.float32),
        q_hi=d[need[5]].astype(np.float32),
        n_animals=int(d[need[6]]),
        n_pool=int(d[need[7]]),
    )


def _plot_pdf(ax, x, s_obs, s_null, *, label, style_idx=0):
    # observed
    ax.plot(x, s_obs["pdf_obs"], lw=LW, ls="-", label=f"{label} (obs)")
    ax.fill_between(x, s_obs["pdf_lo"], s_obs["pdf_hi"], alpha=ALPHA_BAND, linewidth=0)

    # null
    ax.plot(x, s_null["pdf_obs"], lw=LW, ls="--", label=f"{label} (null)")
    if FILL_NULL_CI:
        ax.fill_between(
            x, s_null["pdf_lo"], s_null["pdf_hi"], alpha=ALPHA_BAND * 0.6, linewidth=0
        )


def _plot_q(ax, p, s_obs, s_null, *, label):
    ax.plot(p, s_obs["q_obs"], lw=LW, ls="-", label=f"{label} (obs)")
    ax.fill_between(p, s_obs["q_lo"], s_obs["q_hi"], alpha=ALPHA_BAND, linewidth=0)

    ax.plot(p, s_null["q_obs"], lw=LW, ls="--", label=f"{label} (null)")
    if FILL_NULL_CI:
        ax.fill_between(
            p, s_null["q_lo"], s_null["q_hi"], alpha=ALPHA_BAND * 0.6, linewidth=0
        )


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

fp6bg_path = dist_dir / FP6BG_NAME
if not fp6bg_path.exists():
    raise FileNotFoundError(fp6bg_path)

d = np.load(fp6bg_path, allow_pickle=True)

bins = d["bins"].astype(np.float32)
p_grid = d["p_grid"].astype(np.float32)
x = _bin_centers(bins)

params = json.loads(d["params_json"].item())
groups = list(d["groups"])
conds = params.get("conditions", [])
boot_draws = params.get("boot_draws", None)

print("[FP6cG] Loaded:", fp6bg_path.name)
print(
    "[FP6cG] groups:",
    len(groups),
    "| conditions:",
    len(conds),
    "| boot_draws:",
    boot_draws,
)

# output base dir
out_base = Path(paths["f_mod"]) / "FP6G"
out_base.mkdir(parents=True, exist_ok=True)

REQ = [
    "obs_all",
    "null_all",
    "obs_intra",
    "null_intra",
    "obs_inter",
    "null_inter",
    "obs_trimer",
    "null_trimer",
    "obs_tetramer",
    "null_tetramer",
]

atomic = [
    ("obs_intra_trimer", "null_intra_trimer", "Intra × Trimer"),
    ("obs_intra_tetramer", "null_intra_tetramer", "Intra × Tetramer"),
    ("obs_inter_trimer", "null_inter_trimer", "Inter × Trimer"),
    ("obs_inter_tetramer", "null_inter_tetramer", "Inter × Tetramer"),
]

obs_it_tt = [
    ("obs_intra_trimer", "Intra–Trimer"),
    ("obs_intra_tetramer", "Intra–Tetramer"),
    ("obs_inter_trimer", "Inter–Trimer"),
    ("obs_inter_tetramer", "Inter–Tetramer"),
]

# optional filter
if GROUP_REGEX is not None:
    rx = re.compile(GROUP_REGEX)
    groups = [g for g in groups if rx.search(g)]
    print("[FP6cG] filtered groups:", len(groups))

for g in groups:
    gsafe = _sanitize_group_name(g)
    out_dir = out_base / gsafe
    out_dir.mkdir(parents=True, exist_ok=True)

    # load required series
    S = {k: _load_series(d, g, k) for k in REQ}

    # checks
    for k in REQ:
        _assert_ci_order(S[k]["pdf_lo"], S[k]["pdf_hi"], f"{g}:{k} pdf")
        _assert_ci_order(S[k]["q_lo"], S[k]["q_hi"], f"{g}:{k} q")
        _assert_ok_pdf(bins, S[k]["pdf_obs"], f"{g}:{k} pdf_obs", debug=DEBUG_PRINT)

    if DEBUG_PRINT:
        print(
            f"\n[FP6cG] Group: {g}  (A={S['obs_all']['n_animals']}, pool={S['obs_all']['n_pool']:,})"
        )
        print("  Tail debug (obs):")
        for kk in ["obs_all", "obs_intra", "obs_inter", "obs_trimer", "obs_tetramer"]:
            _print_tail_debug(p_grid, S[kk], kk)

    # -------------------------
    # FIG 1 — Global PDF
    # -------------------------
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    _plot_pdf(ax, x, S["obs_all"], S["null_all"], label="All")
    ax.set_title(f"{g} — Global MC distribution")
    ax.set_xlabel("MC value")
    ax.set_ylabel("Density")
    _maybe_xlim(ax, X_LIM)
    ax.legend(frameon=False)
    set_log_pdf_axis(ax)
    _save(fig, out_dir / "FP6cG_global_pdf")

    # -------------------------
    # FIG 2 — Global quantiles
    # -------------------------
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    _plot_q(ax, p_grid, S["obs_all"], S["null_all"], label="All")
    ax.set_title(f"{g} — Global MC quantiles")
    ax.set_xlabel("p")
    ax.set_ylabel("Q(p)")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False)
    _save(fig, out_dir / "FP6cG_global_quantiles")

    # -------------------------
    # FIG 3 — Intra vs inter PDF
    # -------------------------
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    _plot_pdf(ax, x, S["obs_intra"], S["null_intra"], label="Intra")
    _plot_pdf(ax, x, S["obs_inter"], S["null_inter"], label="Inter")
    ax.set_title(f"{g} — Intra vs inter-module MC (PDF)")
    ax.set_xlabel("MC value")
    ax.set_ylabel("Density")
    _maybe_xlim(ax, X_LIM)
    ax.legend(frameon=False, ncols=2)
    set_log_pdf_axis(ax)
    _save(fig, out_dir / "FP6cG_intra_inter_pdf")

    # -------------------------
    # FIG 4 — Intra vs inter quantiles
    # -------------------------
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    _plot_q(ax, p_grid, S["obs_intra"], S["null_intra"], label="Intra")
    _plot_q(ax, p_grid, S["obs_inter"], S["null_inter"], label="Inter")
    ax.set_title(f"{g} — Intra vs inter-module MC (quantiles)")
    ax.set_xlabel("p")
    ax.set_ylabel("Q(p)")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, ncols=2)
    _save(fig, out_dir / "FP6cG_intra_inter_quantiles")

    # -------------------------
    # FIG 5 — Trimer vs tetramer PDF
    # -------------------------
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    _plot_pdf(ax, x, S["obs_trimer"], S["null_trimer"], label="Trimer")
    _plot_pdf(ax, x, S["obs_tetramer"], S["null_tetramer"], label="Tetramer")
    ax.set_title(f"{g} — Trimer vs tetramer MC (PDF)")
    ax.set_xlabel("MC value")
    ax.set_ylabel("Density")
    _maybe_xlim(ax, X_LIM)
    ax.legend(frameon=False, ncols=2)
    set_log_pdf_axis(ax)
    _save(fig, out_dir / "FP6cG_trimer_tetramer_pdf")

    # -------------------------
    # FIG 6 — Trimer vs tetramer quantiles
    # -------------------------
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    _plot_q(ax, p_grid, S["obs_trimer"], S["null_trimer"], label="Trimer")
    _plot_q(ax, p_grid, S["obs_tetramer"], S["null_tetramer"], label="Tetramer")
    ax.set_title(f"{g} — Trimer vs tetramer MC (quantiles)")
    ax.set_xlabel("p")
    ax.set_ylabel("Q(p)")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, ncols=2)
    _save(fig, out_dir / "FP6cG_trimer_tetramer_quantiles")

    # -------------------------
    # FIG 7 — 2×2 atomic PDFs (if present)
    # -------------------------
    has_atomic = True
    for k_obs, k_null, _ in atomic:
        for suf in ["__pdf_obs", "__pdf_ci_lo", "__pdf_ci_hi"]:
            if (
                f"{g}__{k_obs}{suf}" not in d.files
                or f"{g}__{k_null}{suf}" not in d.files
            ):
                has_atomic = False
                break
    if has_atomic:
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), sharex=True, sharey=True)
        axes = axes.ravel()
        for ax, (k_obs, k_null, title) in zip(axes, atomic, strict=False):
            so = _load_series(d, g, k_obs)
            sn = _load_series(d, g, k_null)
            _plot_pdf(ax, x, so, sn, label=title.replace(" × ", "-"))
            ax.set_title(title)
            ax.set_xlabel("MC value")
            ax.set_ylabel("Density")
            _maybe_xlim(ax, X_LIM)
            ax.legend(frameon=False, fontsize=9)
            set_log_pdf_axis(ax)
        _save(fig, out_dir / "FP6cG_2x2_atomic_pdfs")

    # -------------------------
    # FIG 8 — Observed only: intra/inter × trimer/tetramer (single plot, PDFs)
    # -------------------------
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for key, label in obs_it_tt:
        s = _load_series(d, g, key)
        ax.plot(x, s["pdf_obs"], lw=LW, ls="-", label=label)
        ax.fill_between(x, s["pdf_lo"], s["pdf_hi"], alpha=ALPHA_BAND, linewidth=0)
    ax.set_title(f"{g} — Observed MC (intra/inter × trimer/tetramer)")
    ax.set_xlabel("MC value")
    ax.set_ylabel("Density")
    _maybe_xlim(ax, X_LIM)
    set_log_pdf_axis(ax)
    ax.legend(frameon=False, ncols=2)
    _save(fig, out_dir / "FP6cG_obs_only_intra_inter_trimer_tetramer_pdf")

    # -------------------------
    # FIG 9 — Observed only quantiles
    # -------------------------
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for key, label in obs_it_tt:
        s = _load_series(d, g, key)
        ax.plot(p_grid, s["q_obs"], lw=LW, ls="-", label=label)
        # (optional) you can add CI bands here too, but it gets busy
    ax.set_title(f"{g} — Observed MC quantiles (intra/inter × trimer/tetramer)")
    ax.set_xlabel("p")
    ax.set_ylabel("Q(p)")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, ncols=2)
    _save(fig, out_dir / "FP6cG_obs_only_intra_inter_trimer_tetramer_quantiles")

print("[DONE] FP6cG saved figures to:", out_base)
# %%
