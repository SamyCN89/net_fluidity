#!/usr/bin/env python3
"""
FP7b — Pairwise group comparisons for trimer-root strength biomarkers.

Loads:
  - mc_dist/groups_table.csv  (design matrix; must contain column 'a')
  - mc_dist/fp7a1_trimer_root_strength_per_animal__all_intra_inter.npz

Builds schemes:
  - g_age_geno
  - g_age_sex
  - g_age_geno_sex
  - g_age_pheno_oip
  - g_age_pheno_ro24h

Produces (figures):
  - P-value heatmaps (pairwise comparisons × region)
  - Effect heatmaps masked by p<alpha
    * Δmean (MWU)
    * Cliff’s delta (MWU)
    * Δ tail proportion (Fisher exact)
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, fisher_exact

from shared_code.fun_paths import get_paths

# =====================
# CONFIG
# =====================
DATASET = "ines_abdallah"
TIMECOURSE_FOLDER = "Timecourses_updated_03052024"
COGNITIVE_FILE = "ROIs.xlsx"
ANAT_LABELS_FILE = "41_Allen.txt"

paths = get_paths(
    dataset_name=DATASET,
    timecourse_folder=TIMECOURSE_FOLDER,
    cognitive_data_file=COGNITIVE_FILE,
    anat_labels_file=ANAT_LABELS_FILE,
)

DIST_DIR = Path(paths["mc"]) / "mc_dist"
FP7A1_NAME = "fp7a1_trimer_root_strength_per_animal__all_intra_inter.npz"
GROUPS_TABLE_NAME = "groups_table.csv"

DROP_PHENO_BAD = False

# Output folders
FIG_DIR_DMEAN = Path(paths["f_trimers"]) / "root_comparison_dmean"
FIG_DIR_TAILMEAN = Path(paths["f_trimers"]) / "root_comparison_tailmean"
FIG_DIR_CLIFF = Path(paths["f_trimers"]) / "root_comparison_cliffs"

SAVE = True
DPI = 200
TICK_STEP = 1
ALPHA = 0.05
VMAX_P = 0.1

# Tail settings (use true extremes)
Q_UPPER_LIST = [0.90, 0.95, 0.99]
Q_LOWER_LIST = [0.10, 0.05, 0.01]
MIN_TAIL_N = 3


# =====================
# UTIL: region labels
# =====================
def load_region_labels(paths) -> list[str]:
    npz_path = Path(paths["preprocessed"]) / "ts_and_meta_ines_abdallah.npz"
    dmeta = np.load(npz_path, allow_pickle=True)
    return [str(x) for x in dmeta["anat_labels"].tolist()]


def set_region_xticks(R: int, region_labels: list[str], tick_step: int):
    xt = np.arange(0, R, tick_step)
    plt.xticks(xt, [region_labels[i] for i in xt], rotation=90)

def save_matrices_npz(out_path: Path, **kw):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # store strings safely
    for k in ["pair_labels", "region_labels"]:
        if k in kw and isinstance(kw[k], list):
            kw[k] = np.array(kw[k], dtype=object)
    np.savez_compressed(out_path, **kw)

# =====================
# LOAD + ALIGN
# =====================
def load_aligned(dist_dir: Path):
    df = pd.read_csv(dist_dir / GROUPS_TABLE_NAME)
    if "a" not in df.columns:
        raise KeyError("groups_table.csv must contain column 'a' (animal index).")
    df["a"] = df["a"].astype(int)

    zT = np.load(dist_dir / FP7A1_NAME, allow_pickle=True)
    T_all = zT["T_all"]
    T_intra = zT["T_intra"]
    T_inter = zT["T_inter"]

    A = T_all.shape[0]

    df = (
        df[(df["a"] >= 0) & (df["a"] < A)]
        .drop_duplicates("a")
        .sort_values("a")
        .reset_index(drop=True)
    )

    idx_a = df["a"].to_numpy()
    Xa = T_all[idx_a]
    Xi = T_intra[idx_a]
    Xe = T_inter[idx_a]

    ok = np.allclose(Xa, Xi + Xe, atol=1e-6)
    print("[SANITY] allclose(all, intra+inter):", ok)
    if not ok:
        print("[WARN] Xa != Xi+Xe; check FP3 mc_mod_idx / FP7a1 aggregation.")

    return df, Xa, Xi, Xe


# =====================
# SCHEMES
# =====================
def add_schemes(df: pd.DataFrame) -> pd.DataFrame:
    required = ["age", "genotype", "sex", "phenotype_oip", "phenotype_ro24h"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"groups_table.csv missing required columns: {missing}")

    for c in required:
        df[c] = df[c].astype(str)

    if DROP_PHENO_BAD:
        bad_tokens = {"bad", "nan", "None", ""}
        df.loc[df["phenotype_oip"].isin(bad_tokens), "phenotype_oip"] = "NA"
        df.loc[df["phenotype_ro24h"].isin(bad_tokens), "phenotype_ro24h"] = "NA"

    df["g_age_geno"] = df[["age", "genotype"]].agg("|".join, axis=1)
    df["g_age_sex"] = df[["age", "sex"]].agg("|".join, axis=1)
    df["g_age_geno_sex"] = df[["age", "genotype", "sex"]].agg("|".join, axis=1)
    df["g_age_pheno_oip"] = df[["age", "phenotype_oip"]].agg("|".join, axis=1)
    df["g_age_pheno_ro24h"] = df[["age", "phenotype_ro24h"]].agg("|".join, axis=1)

    return df


# =====================
# PLOTTING
# =====================
def plot_p_heatmap(Pmat, pair_labels, title, region_labels, vmax, tick_step, dpi, save_path=None):
    plt.figure(figsize=(12, max(3, 0.35 * len(pair_labels))))
    plt.imshow(Pmat, aspect="auto", interpolation="none", vmin=0, vmax=vmax, cmap="viridis_r")

    R = Pmat.shape[1]
    set_region_xticks(R, region_labels, tick_step=tick_step)

    plt.yticks(range(len(pair_labels)), pair_labels)
    plt.xlabel("Region")
    plt.ylabel("Pairwise comparison")
    plt.title(title)
    plt.colorbar(label=f"p-value (clipped 0–{vmax})")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi)

    plt.show()


def plot_effect_masked(E, P, pair_labels, title, region_labels, alpha, tick_step, dpi=200, save_path=None, cbar_label="effect"):
    M = np.array(E, copy=True)
    M[~np.isfinite(P)] = np.nan
    M[P >= alpha] = np.nan

    plt.figure(figsize=(12, max(3, 0.35 * len(pair_labels))))

    vmax = np.nanmax(np.abs(M))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    im = plt.imshow(M, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)

    R = P.shape[1]
    set_region_xticks(R, region_labels, tick_step=tick_step)

    plt.yticks(range(len(pair_labels)), pair_labels)
    plt.xlabel("Region")
    plt.ylabel("Pairwise comparison")
    plt.title(title + f" (masked p<{alpha})")
    plt.colorbar(im, label=cbar_label)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi)

    plt.show()


# =====================
# METRICS
# =====================
def _collect_groups(df, X, scheme_col):
    groups, data = [], []
    for g, dfg in df.groupby(scheme_col, sort=True):
        idx = dfg.index.to_numpy()
        groups.append(str(g))
        data.append(X[idx])
    return groups, data


def pairwise_p_and_dmean(df, X, scheme_col, alternative="two-sided"):
    groups, data = _collect_groups(df, X, scheme_col)
    G, R = len(groups), X.shape[1]

    pair_labels, P_rows, D_rows = [], [], []

    for i in range(G):
        Xi = data[i]
        for j in range(i + 1, G):
            Xj = data[j]
            pair_labels.append(f"{groups[i]}  vs  {groups[j]}")

            p_row = np.full(R, np.nan)
            d_row = np.nanmean(Xi, axis=0) - np.nanmean(Xj, axis=0)

            for r in range(R):
                a = Xi[:, r]; b = Xj[:, r]
                a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
                if a.size < 2 or b.size < 2:
                    continue
                p_row[r] = mannwhitneyu(a, b, alternative=alternative).pvalue

            P_rows.append(p_row)
            D_rows.append(d_row)

    return pair_labels, np.vstack(P_rows), np.vstack(D_rows)


def cliffs_delta(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    na, nb = a.size, b.size
    if na == 0 or nb == 0:
        return np.nan

    gt = 0
    lt = 0
    for x in a:
        gt += np.sum(x > b)
        lt += np.sum(x < b)
    return (gt - lt) / (na * nb)


def pairwise_p_and_cliff(df, X, scheme_col, dpi=200, alternative="two-sided"):
    groups, data = _collect_groups(df, X, scheme_col)
    G, R = len(groups), X.shape[1]

    pair_labels, P_rows, C_rows = [], [], []

    for i in range(G):
        Xi = data[i]
        for j in range(i + 1, G):
            Xj = data[j]
            pair_labels.append(f"{groups[i]}  vs  {groups[j]}")

            p_row = np.full(R, np.nan)
            c_row = np.full(R, np.nan)

            for r in range(R):
                a = Xi[:, r]; b = Xj[:, r]
                a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
                if a.size < 2 or b.size < 2:
                    continue
                p_row[r] = mannwhitneyu(a, b, alternative=alternative).pvalue
                c_row[r] = cliffs_delta(a, b)

            P_rows.append(p_row)
            C_rows.append(c_row)

    return pair_labels, np.vstack(P_rows), np.vstack(C_rows)


def pairwise_p_and_tailmean(df, X, scheme_col, q=0.95, tail="upper", alternative="two-sided", min_tail_n=3):
    """
    Tail-mean test:
      - threshold per region from pooled X (all animals in df, already aligned)
      - restrict each group to values in chosen tail
      - compare tail distributions with MWU
      - effect = mean(tail_A) - mean(tail_B)

    Returns:
      pair_labels, Pmat, Emat
    """
    groups, data = _collect_groups(df, X, scheme_col)
    G, R = len(groups), X.shape[1]

    thr = np.nanquantile(X, q, axis=0)

    pair_labels, P_rows, E_rows = [], [], []

    for i in range(G):
        Xi = data[i]
        for j in range(i + 1, G):
            Xj = data[j]
            pair_labels.append(f"{groups[i]}  vs  {groups[j]}")

            p_row = np.full(R, np.nan)
            e_row = np.full(R, np.nan)

            for r in range(R):
                a = Xi[:, r]; b = Xj[:, r]
                a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
                if a.size < 2 or b.size < 2:
                    continue

                if tail == "upper":
                    a_tail = a[a > thr[r]]
                    b_tail = b[b > thr[r]]
                elif tail == "lower":
                    a_tail = a[a < thr[r]]
                    b_tail = b[b < thr[r]]
                else:
                    raise ValueError(tail)

                if a_tail.size < min_tail_n or b_tail.size < min_tail_n:
                    continue

                p_row[r] = mannwhitneyu(a_tail, b_tail, alternative=alternative).pvalue
                e_row[r] = float(np.mean(a_tail) - np.mean(b_tail))

            P_rows.append(p_row)
            E_rows.append(e_row)

    return pair_labels, np.vstack(P_rows), np.vstack(E_rows)


# =====================
# RUNNERS
# =====================
def run_pairwise_and_plot(
    df, X, scheme_name, topo_name,
    region_labels,
    fig_dir,
    alpha=0.05,
    vmax_p=0.1,
    tick_step=2,
    save=True,
    dpi=200,
    mode="dmean",
    q=None,
    tail=None,
):
    if mode == "dmean":
        pair_labels, Pmat, Emat = pairwise_p_and_dmean(df, X, scheme_name)
        tag = "dmean"
        effect_label = "dmean"
        cbar = "Δmean"
    elif mode == "cliff":
        pair_labels, Pmat, Emat = pairwise_p_and_cliff(df, X, scheme_name)
        tag = "cliff"
        effect_label = "cliff"
        cbar = "Cliff δ"
    elif mode == "tailmean":
        assert q is not None and tail is not None
        pair_labels, Pmat, Emat = pairwise_p_and_tailmean(df, X, scheme_name, q=q, tail=tail, min_tail_n=MIN_TAIL_N)
        tag = f"tailmean_q{q:.2f}_{tail}"
        effect_label = "tailmean"
        cbar = "Δ tail-mean"
    else:
        raise ValueError(mode)

    out_dir = fig_dir / scheme_name
    p_path = out_dir / f"{topo_name}__{tag}__pvals.png" if save else None
    e_path = out_dir / f"{topo_name}__{tag}__effect_masked_p{alpha}.png" if save else None

    # Save matrices for later use (e.g. FP7c)
    oud_dir_npz = Path(paths["trimers"]) / scheme_name
    z_path = oud_dir_npz / f"{topo_name}__{tag}__raw.npz" if save else None

    plot_p_heatmap(
        Pmat, pair_labels,
        title=f"{topo_name} | {tag} p-values | {scheme_name}",
        region_labels=region_labels,
        vmax=vmax_p,
        tick_step=tick_step,
        save_path=p_path,
        dpi=dpi,
    )

    plot_effect_masked(
        Emat, Pmat, pair_labels,
        title=f"{topo_name} | {tag} | {scheme_name}",
        region_labels=region_labels,
        alpha=alpha,
        tick_step=tick_step,
        save_path=e_path,
        dpi=dpi,
    )

    if z_path is not None:
        save_matrices_npz(
            z_path,
            P=Pmat,
            E=Emat,
            pair_labels=pair_labels,
            region_labels=region_labels,
            scheme=scheme_name,
            topo=topo_name,
            mode=mode,
            q=q if q is not None else np.nan,
            tail=tail if tail is not None else "",
            alpha=alpha,
        )



def main():
    df, Xa, Xi, Xe = load_aligned(DIST_DIR)
    df = add_schemes(df)

    region_labels = load_region_labels(paths)
    assert len(region_labels) == Xa.shape[1], (len(region_labels), Xa.shape[1])

    schemes = ["g_age_geno", "g_age_sex", "g_age_geno_sex", "g_age_pheno_oip", "g_age_pheno_ro24h"]
    topologies = [("ALL", Xa), ("INTRA", Xi), ("INTER", Xe)]

    # Main exploration outputs
    for scheme_name in schemes:
        for topo_name, X in [("ALL", Xa), ("INTRA", Xi), ("INTER", Xe)]:

            # dmean outputs
            run_pairwise_and_plot(
                df, X, scheme_name, topo_name,
                region_labels, FIG_DIR_DMEAN,
                alpha=ALPHA, vmax_p=0.1, tick_step=TICK_STEP,
                save=SAVE, dpi=DPI,
                mode="dmean",
            )

            # cliff outputs
            run_pairwise_and_plot(
                df, X, scheme_name, topo_name,
                region_labels, FIG_DIR_CLIFF,
                alpha=ALPHA, vmax_p=0.1, tick_step=TICK_STEP,
                save=SAVE, dpi=DPI,
                mode="cliff",
            )

            # tail-mean upper
            for q in Q_UPPER_LIST:
                run_pairwise_and_plot(
                    df, X, scheme_name, topo_name,
                    region_labels, FIG_DIR_TAILMEAN,
                    alpha=ALPHA, vmax_p=0.1, tick_step=TICK_STEP,
                    save=SAVE, dpi=DPI,
                    mode="tailmean", q=q, tail="upper",
                )

            # tail-mean lower
            for q in Q_LOWER_LIST:
                run_pairwise_and_plot(
                    df, X, scheme_name, topo_name,
                    region_labels, FIG_DIR_TAILMEAN,
                    alpha=ALPHA, vmax_p=0.1, tick_step=TICK_STEP,
                    save=SAVE, dpi=DPI,
                    mode="tailmean", q=q, tail="lower",
                )


if __name__ == "__main__":
    main()
