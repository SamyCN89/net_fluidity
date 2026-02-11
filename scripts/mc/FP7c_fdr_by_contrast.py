#!/usr/bin/env python3
"""
FP7c — FDR correction per contrast (row-wise).

Input:
  *_raw.npz from FP7b

Output:
  *_fdr.npz with corrected p-values
"""

from pathlib import Path
import numpy as np
from statsmodels.stats.multitest import multipletests

# =====================
# CONFIG
# =====================

# Folder where FP7b saved raw matrices
RAW_DIR = Path("/media/samy/Elements2/Proyectos/LauraHarsan/results/ines_abdallah/trimers")

ALPHA = 0.05
METHOD = "fdr_bh"   # Benjamini-Hochberg


# =====================
# CORE
# =====================

def fdr_per_contrast(P, alpha=0.05, method="fdr_bh"):
    """
    Apply FDR row-wise (per contrast).
    """
    Pc = np.full_like(P, np.nan)

    for i in range(P.shape[0]):

        p = P[i]
        ok = np.isfinite(p)

        if ok.sum() == 0:
            continue

        _, p_corr, _, _ = multipletests(
            p[ok],
            alpha=alpha,
            method=method,
        )

        Pc[i, ok] = p_corr

    return Pc


# =====================
# MAIN
# =====================

def process_file(path: Path):

    z = np.load(path, allow_pickle=True)

    P = z["P"]
    pair_labels = z["pair_labels"]
    region_labels = z["region_labels"]

    Pc = fdr_per_contrast(P, ALPHA, METHOD)

    sig = Pc < ALPHA

    out = path.with_name(path.stem + "__fdr.npz")

    np.savez_compressed(
        out,
        P_raw=P,
        P_fdr=Pc,
        sig=sig,
        pair_labels=pair_labels,
        region_labels=region_labels,
        alpha=ALPHA,
        method=METHOD,
        source=str(path),
    )

    print("[OK]", out.name)


def main():

    files = sorted(RAW_DIR.rglob("*__raw.npz"))

    print(f"[FOUND] {len(files)} raw files")

    for f in files:
        process_file(f)


if __name__ == "__main__":
    main()
