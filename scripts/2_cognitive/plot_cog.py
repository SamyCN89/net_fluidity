#!/usr/bin/env python3
"""
Created on Wed Apr  2 02:59:41 2025

@author: samy
"""

# %%
import json

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import tqdm

from shared_code.fun_loaddata import load_timeseries_bundle

# from fun_utils import get_paths, set_figure_params
from shared_code.fun_paths import get_paths

# from shared_code.fun_utils import set_figure_params
from shared_code.fun_utils import load_cognitive_data, set_figure_params

save_fig = set_figure_params(False)

timecourse_folder = "Timecourses_updated_03052024"
paths = get_paths(
    dataset_name="ines_abdallah",
    timecourse_folder=timecourse_folder,
    cognitive_data_file="ROIs.xlsx",
    anat_labels_file="41_Allen.txt",
)

bundle = load_timeseries_bundle(
    paths["preprocessed"] / "ts_and_meta_2m4m.npz",
    paths["preprocessed"] / "grouping_data_oip.pkl",
)
ts = bundle.ts
n_animals = bundle.n_animals
total_tr = bundle.total_tr
anat_labels = bundle.anat_labels
regions = bundle.n_regions
mask_groups = bundle.mask_groups
label_variables = bundle.label_variables

# %%
# ========================== Prepare cognitive data =========================
# Load cognitive data

cog_data = load_cognitive_data(paths["preprocessed"] / "cog_data_sorted_2m4m.csv")
# import pickle
# with open(paths["preprocessed"] / "grouping_data_oip.pkl", "rb") as f:
#     cognitive_data = pickle.load(f) # Dictionary with cognitive data
# %%
# ========================== Figure parameters ================================
# Set figure parameters globally
# save_fig = set_figure_params(True)

# # =================== Paths and folders =======================================
# # paths = get_paths()
# paths = get_paths(
#     dataset_name="ines_abdallah",
#     timecourse_folder=timecourse_folder,
#     cognitive_data_file="ROIs.xlsx",
#     anat_labels_file="41_Allen.txt",
# )
# data_ts = load_timeseries_data(paths["preprocessed"] / "ts_and_meta_2m4m.npz")
# is_2month_old = data_ts["is_2month_old"]

# %%
# ========================== Load data =========================

# Parameters and indices of variables
# ts          = data_ts['ts']
# n_animals   = int(data_ts['n_animals'])
# regions     = data_ts['regions']
# anat_labels = data_ts['anat_labels']

# %%
# Example: Plotting all time series stacked with offset
plt.figure(figsize=(12, 8))
offset = 0.07  # vertical offset between time series
for i, ts1 in enumerate(ts[0].T):
    plt.plot(ts1 + i * offset, label=f"TS {i+1}")
plt.ylim(-0.1, len(anat_labels) * offset + offset)
plt.yticks(np.arange(len(anat_labels)) * offset, anat_labels)
# plt.title("Stacked Time Series")
plt.xlabel("TR")
plt.xlim(0, 300)
# plt.ylabel("Signal + Offset")
plt.tight_layout()
plt.show()
plt.savefig(paths["figures"] / f"ts/ts_extract_{timecourse_folder}.png")

# %%

# Example: Plotting a histogram of a cognitive score for male_wt_data
plt.figure(1, figsize=(10, 6))
plt.clf()
plt.subplot(211)

male_ind = cog_data["Sexe"] == "M"
female_ind = cog_data["Sexe"] == "F"
wt_ind = cog_data["Genotype"] == "wt"
dki_ind = cog_data["Genotype"] == "dKI"

# plt.hist((male_wt_data['OiP_2M'], male_wt_data['OiP_4M'], male_dki_data['OiP_2M'], male_dki_data['OiP_4M']), bins=4,
#          alpha=0.7,
#           histtype='step',
#          label=('Male WT 2M', 'Male WT 4M', 'Male dKI 2M', 'Male dKI 4M'))
plt.violinplot(
    (
        cog_data["OiP_2M"][male_ind & wt_ind],
        cog_data["OiP_4M"][male_ind & wt_ind],
        cog_data["OiP_2M"][male_ind & dki_ind],
        cog_data["OiP_4M"][male_ind & dki_ind],
    )
)
plt.violinplot(
    (
        cog_data["OiP_2M"][female_ind & wt_ind],
        cog_data["OiP_4M"][female_ind & wt_ind],
        cog_data["OiP_2M"][female_ind & dki_ind],
        cog_data["OiP_4M"][female_ind & dki_ind],
    )
)

# labels=('Male WT 2M', 'Male WT 4M', 'Male dKI 2M', 'Male dKI 4M'))
plt.xticks([1, 2, 3, 4], ["Male WT 2M", "Male WT 4M", "Male dKI 2M", "Male dKI 4M"])
plt.axhline(0, c="k")
plt.ylabel("OiP score")
plt.title("OiP task scores")
plt.subplot(212)

plt.violinplot(
    (
        cog_data["RO24h_2M"][male_ind & wt_ind],
        cog_data["RO24h_4M"][male_ind & wt_ind],
        cog_data["RO24h_2M"][male_ind & dki_ind],
        cog_data["RO24h_4M"][male_ind & dki_ind],
    )
)
plt.xticks([1, 2, 3, 4], ["Male WT 2M", "Male WT 4M", "Male dKI 2M", "Male dKI 4M"])
plt.axhline(0, c="k")
plt.ylabel("RO24h score")
plt.title("RO24h Task")
plt.tight_layout()
plt.legend()
plt.subplot(212)
plt.violinplot(
    (
        male_wt_data["RO24h_2M"],
        male_wt_data["RO24h_4M"],
        male_dki_data["RO24h_2M"],
        male_dki_data["RO24h_4M"],
    )
)
plt.xticks([1, 2, 3, 4], ["Male WT 2M", "Male WT 4M", "Male dKI 2M", "Male dKI 4M"])
plt.axhline(0, c="k")
plt.ylabel("RO24h score")
plt.title("Distribution of RO24h for Male")
plt.legend()


# %%
# %%


# %%
# %%
import numpy as np
import pandas as pd
from scipy import stats


# ---------- build tidy data (robust) ----------
def build_df(cog, score_key):
    # normalize labels (handles 'M'/'F', 'wt'/'dki', mixed case)
    sex_raw = np.asarray(cog["Sexe"]).astype(str)
    sex = np.where(np.char.upper(sex_raw) == "M", "Male", "Female")

    geno_raw = np.asarray(cog["Genotype"]).astype(str)
    geno_low = np.char.lower(geno_raw)
    geno = np.where(geno_low == "wt", "WT", "dKI")

    rows = []
    for age_key, age in [("2M", "2M"), ("4M", "4M")]:
        y = np.asarray(cog[f"{score_key}_{age_key}"], float)
        for i in range(y.size):
            if not np.isnan(y[i]):
                rows.append(
                    {
                        "score": float(y[i]),
                        "Sex": sex[i],
                        "Genotype": geno[i],
                        "Age": age,
                    }
                )
    return pd.DataFrame(rows)


# ---------- effect sizes ----------
def cliffs_delta(x, y):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    y = np.asarray(y, float)
    y = y[~np.isnan(y)]
    if x.size == 0 or y.size == 0:
        return np.nan
    # δ = P(X>Y) - P(X<Y)
    diff = x[:, None] - y[None, :]
    return (np.sum(diff > 0) - np.sum(diff < 0)) / (x.size * y.size)


def hodges_lehmann(x, y):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    y = np.asarray(y, float)
    y = y[~np.isnan(y)]
    if x.size == 0 or y.size == 0:
        return np.nan
    diffs = x[:, None] - y[None, :]
    return np.median(diffs)


def bootstrap_ci(a, b=None, stat="median_diff", n_boot=5000, rng=0, ci=95):
    r = np.random.default_rng(rng)
    if b is None:  # one-sample stat
        a = np.asarray(a, float)
        a = a[~np.isnan(a)]
        if a.size == 0:
            return np.nan, (np.nan, np.nan)
        boots = np.median(r.choice(a, (n_boot, a.size), replace=True), axis=1)
    else:
        a = np.asarray(a, float)
        a = a[~np.isnan(a)]
        b = np.asarray(b, float)
        b = b[~np.isnan(b)]
        if a.size == 0 or b.size == 0:
            return np.nan, (np.nan, np.nan)
        boots = []
        for _ in range(n_boot):
            aa = r.choice(a, a.size, replace=True)
            bb = r.choice(b, b.size, replace=True)
            boots.append(np.median(aa) - np.median(bb))
        boots = np.asarray(boots)
    lo = (100 - ci) / 2
    hi = 100 - lo
    return np.median(boots), (np.percentile(boots, lo), np.percentile(boots, hi))


# ---------- permutation omnibus (nonparametric, design-aware) ----------
def _cell_median(df, sex, geno, age):
    v = df.loc[
        (df.Sex == sex) & (df.Genotype == geno) & (df.Age == age), "score"
    ].values
    # print(df.loc[(df.Sex==sex)&(df.Genotype==geno)&(df.Age==age),'score'])
    v = v[~np.isnan(v)]
    return np.median(v) if v.size else np.nan


# ---------- robust tidy builder ----------


# ---------- permutation omnibus (weights fixed) ----------
def omnibus_permutation(df, effect="Sex", B=10000, rng=1):
    rng = np.random.default_rng(rng)
    levels = {"Sex": ["Male", "Female"], "Genotype": ["WT", "dKI"], "Age": ["2M", "4M"]}

    def med_and_n(df_):
        med, n = {}, {}
        for s in levels["Sex"]:
            for g in levels["Genotype"]:
                for a in levels["Age"]:
                    vals = df_.loc[
                        (df_.Sex == s) & (df_.Genotype == g) & (df_.Age == a), "score"
                    ].to_numpy()
                    vals = vals[~np.isnan(vals)]
                    med[(s, g, a)] = np.median(vals) if vals.size else np.nan
                    n[(s, g, a)] = int(vals.size)
        return med, n

    def weighted_avg(pairs):
        """pairs: list of (value, weight). Skip NaNs/zero weights."""
        vals = [(v, w) for (v, w) in pairs if np.isfinite(v) and w > 0]
        if not vals:
            return np.nan
        v, w = zip(*vals, strict=False)
        return np.average(v, weights=w)

    def contrast(df_):
        med, n = med_and_n(df_)

        if effect == "Sex":
            # across Genotype×Age: median(F) - median(M)
            pairs = []
            for g in levels["Genotype"]:
                for a in levels["Age"]:
                    F = ("Female", g, a)
                    M = ("Male", g, a)
                    val = med[F] - med[M]
                    wt = n[F] + n[M]
                    pairs.append((val, wt))
            return weighted_avg(pairs)

        if effect == "Age":
            # across Sex×Genotype: median(4M) - median(2M)
            pairs = []
            for s in levels["Sex"]:
                for g in levels["Genotype"]:
                    A4 = (s, g, "4M")
                    A2 = (s, g, "2M")
                    val = med[A4] - med[A2]
                    wt = n[A4] + n[A2]
                    pairs.append((val, wt))
            return weighted_avg(pairs)

        if effect == "Genotype":
            # across Sex×Age: median(dKI) - median(WT)
            pairs = []
            for s in levels["Sex"]:
                for a in levels["Age"]:
                    D = (s, "dKI", a)
                    W = (s, "WT", a)
                    val = med[D] - med[W]
                    wt = n[D] + n[W]
                    pairs.append((val, wt))
            return weighted_avg(pairs)

        if effect == "Sex:Age":
            # per Genotype: (F4-F2) - (M4-M2); average over genotype, weight by the 4 cells used
            pairs = []
            for g in levels["Genotype"]:
                F4, F2 = ("Female", g, "4M"), ("Female", g, "2M")
                M4, M2 = ("Male", g, "4M"), ("Male", g, "2M")
                val = (med[F4] - med[F2]) - (med[M4] - med[M2])
                wt = n[F4] + n[F2] + n[M4] + n[M2]
                pairs.append((val, wt))
            return weighted_avg(pairs)

        if effect == "Sex:Genotype":
            # per Age: (F_dKI - F_WT) - (M_dKI - M_WT); average over age
            pairs = []
            for a in levels["Age"]:
                Fd, Fw = ("Female", "dKI", a), ("Female", "WT", a)
                Md, Mw = ("Male", "dKI", a), ("Male", "WT", a)
                val = (med[Fd] - med[Fw]) - (med[Md] - med[Mw])
                wt = n[Fd] + n[Fw] + n[Md] + n[Mw]
                pairs.append((val, wt))
            return weighted_avg(pairs)

        if effect == "Age:Genotype":
            # per Sex: (dKI_4 - dKI_2) - (WT_4 - WT_2); average over sex
            pairs = []
            for s in levels["Sex"]:
                d4, d2 = (s, "dKI", "4M"), (s, "dKI", "2M")
                w4, w2 = (s, "WT", "4M"), (s, "WT", "2M")
                val = (med[d4] - med[d2]) - (med[w4] - med[w2])
                wt = n[d4] + n[d2] + n[w4] + n[w2]
                pairs.append((val, wt))
            return weighted_avg(pairs)

        if effect == "Sex:Age:Genotype":
            # difference of genotype-specific Sex×Age interactions
            Fd4, Fd2, Md4, Md2 = (
                ("Female", "dKI", "4M"),
                ("Female", "dKI", "2M"),
                ("Male", "dKI", "4M"),
                ("Male", "dKI", "2M"),
            )
            Fw4, Fw2, Mw4, Mw2 = (
                ("Female", "WT", "4M"),
                ("Female", "WT", "2M"),
                ("Male", "WT", "4M"),
                ("Male", "WT", "2M"),
            )
            val = ((med[Fd4] - med[Fd2]) - (med[Md4] - med[Md2])) - (
                (med[Fw4] - med[Fw2]) - (med[Mw4] - med[Mw2])
            )
            return val

        raise ValueError("Unknown effect.")

    # observed statistic
    T_obs = contrast(df)
    if not np.isfinite(T_obs):
        return np.nan, np.nan

    # permutation: shuffle only the label tied to the effect within the right strata
    p_extreme = 0
    for _ in tqdm.tqdm(range(B), desc=f"Permutations for {effect}", ncols=80):
        dfp = df.copy()
        if effect in ("Sex", "Age", "Genotype"):
            strata = {
                "Sex": ["Genotype", "Age"],
                "Age": ["Sex", "Genotype"],
                "Genotype": ["Sex", "Age"],
            }[effect]
            within = dfp.groupby(strata, group_keys=False)
            dfp[effect] = within[effect].transform(np.random.permutation)
        elif effect == "Sex:Age":
            within = dfp.groupby(["Sex", "Genotype"], group_keys=False)
            dfp["Age"] = within["Age"].transform(np.random.permutation)
        elif effect == "Sex:Genotype":
            within = dfp.groupby(["Sex", "Age"], group_keys=False)
            dfp["Genotype"] = within["Genotype"].transform(np.random.permutation)
        elif effect == "Age:Genotype":
            within = dfp.groupby(["Genotype", "Sex"], group_keys=False)
            dfp["Age"] = within["Age"].transform(np.random.permutation)
        elif effect == "Sex:Age:Genotype":
            within = dfp.groupby(["Sex", "Age"], group_keys=False)
            dfp["Genotype"] = within["Genotype"].transform(np.random.permutation)

        T_perm = contrast(dfp)
        if np.isfinite(T_perm) and (np.abs(T_perm) >= np.abs(T_obs)):
            p_extreme += 1

    p = (p_extreme + 1) / (B + 1)  # add-one smoothing
    return T_obs, p


# ---------- pairwise BM/MWU with Holm–Šidák ----------
def holm_sidak(pvals):
    p = np.array(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty_like(p)
    for k, idx in enumerate(order):
        alpha_k = 1 - (1 - 0.05) ** (1 / (m - k))
        adj[idx] = min(
            p[idx] * ((1 - (1 - 0.05)) / alpha_k), 1.0
        )  # display-friendly; decision uses step-down
    return adj


def planned_pairwise(df):
    rows = []
    # 4 Sex contrasts within Genotype×Age
    for g in ["WT", "dKI"]:
        for a in ["2M", "4M"]:
            x = df.query("Sex=='Male' & Genotype==@g & Age==@a")["score"].values
            y = df.query("Sex=='Female' & Genotype==@g & Age==@a")["score"].values
            if x.size and y.size:
                stat, p = stats.brunnermunzel(x, y, alternative="two-sided")
                delta = cliffs_delta(y, x)  # Female − Male
                hl, (lo, hi) = bootstrap_ci(y, x)
                rows.append(["Sex", g, a, p, delta, hl, lo, hi, x.size, y.size])
    # 4 Age contrasts within Sex×Genotype
    for s in ["Male", "Female"]:
        for g in ["WT", "dKI"]:
            x = df.query("Sex==@s & Genotype==@g & Age=='2M'")["score"].values
            y = df.query("Sex==@s & Genotype==@g & Age=='4M'")["score"].values
            if x.size and y.size:
                stat, p = stats.brunnermunzel(x, y, alternative="two-sided")
                delta = cliffs_delta(y, x)  # 4M − 2M
                hl, (lo, hi) = bootstrap_ci(y, x)
                rows.append(["Age", g, s, p, delta, hl, lo, hi, x.size, y.size])
    # 4 Genotype contrasts within Sex×Age
    for s in ["Male", "Female"]:
        for a in ["2M", "4M"]:
            x = df.query("Sex==@s & Age==@a & Genotype=='WT'")["score"].values
            y = df.query("Sex==@s & Age==@a & Genotype=='dKI'")["score"].values
            if x.size and y.size:
                stat, p = stats.brunnermunzel(x, y, alternative="two-sided")
                delta = cliffs_delta(y, x)  # dKI − WT
                hl, (lo, hi) = bootstrap_ci(y, x)
                rows.append(["Genotype", s, a, p, delta, hl, lo, hi, x.size, y.size])
    out = pd.DataFrame(
        rows,
        columns=[
            "Family",
            "Stratum1",
            "Stratum2",
            "p",
            "Cliffs_delta",
            "HL_diff",
            "HL_lo",
            "HL_hi",
            "n1",
            "n2",
        ],
    )
    # Holm–Šidák within each family
    out["p_adj"] = out.groupby("Family")["p"].transform(
        lambda s: pd.Series(holm_sidak(s.values), index=s.index)
    )
    return out.sort_values(["Family", "Stratum1", "Stratum2"]).reset_index(drop=True)


# ---------- run for a task ----------
def analyze_task(cog, score_key, B=10000):
    df = build_df(cog, score_key)
    # omnibus
    effects = [
        "Sex",
        "Age",
        "Genotype",
        "Sex:Age",
        "Sex:Genotype",
        "Age:Genotype",
        "Sex:Age:Genotype",
    ]
    om = []
    for eff in effects:
        print(f"Computing omnibus for effect {eff} ...")
        T, p = omnibus_permutation(df, effect=eff, B=B, rng=42)
        om.append([eff, T, p])
    omnibus_tbl = pd.DataFrame(
        om, columns=["Effect", "Contrast_value", "p_perm"]
    ).sort_values("Effect")
    # pairwise
    pairwise_tbl = planned_pairwise(df)
    return omnibus_tbl, pairwise_tbl


# Example usage:
# om_OiP, pw_OiP = analyze_task(cog_data, 'OiP', B=10)
om_OiP, pw_OiP = analyze_task(cog_data, "OiP", B=10_000)
om_RO, pw_RO = analyze_task(cog_data, "RO24h", B=10_000)
print(om_OiP)
print(pw_OiP)
print(om_RO)
print(pw_RO)


# %%
def save_stats(paths, score_key, B, omnibus_tbl, pairwise_tbl, rng=42):
    import json

    outdir = paths
    base = f"{score_key}_B{B}_seed{rng}"
    omnibus_tbl.to_csv(outdir / f"{base}_omnibus.csv", index=False)
    pairwise_tbl.to_csv(outdir / f"{base}_pairwise.csv", index=False)

    def row2dict(r):
        return {
            "family": r["Family"],
            "s1": r["Stratum1"],
            "s2": r["Stratum2"],
            "p": float(r["p"]),
            "p_adj": float(r["p_adj"]),
            "delta": float(r["Cliffs_delta"]),
            "HL": float(r["HL_diff"]),
            "lo": float(r["HL_lo"]),
            "hi": float(r["HL_hi"]),
        }

    summary = {
        "score": score_key,
        "B": int(B),
        "seed": int(rng),
        "omnibus": [
            {
                "effect": r["Effect"],
                "contrast": float(r["Contrast_value"]),
                "p_perm": float(r["p_perm"]),
            }
            for _, r in omnibus_tbl.iterrows()
        ],
        "pairwise_all": [row2dict(r) for _, r in pairwise_tbl.iterrows()],
        "pairwise_sig": [
            row2dict(r) for _, r in pairwise_tbl.iterrows() if r["p_adj"] < 0.05
        ],
    }
    with open(outdir / f"{base}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


# %%
def load_stats(paths, score_key, B, rng=42):
    outdir = paths
    base = f"{score_key}_B{B}_seed{rng}"
    with open(outdir / f"{base}_summary.json") as f:
        return json.load(f)


# save_stats
save_stats(paths["preprocessed"], "RO24h", 10000, om_RO, pw_RO, rng=42)
save_stats(paths["preprocessed"], "OiP", 10000, om_OiP, pw_OiP, rng=42)


# %%
# ========================================================================================
# ===================== Violin panels: Male (left) vs Female (right) =====================
# ========================================================================================

# -------------------- CLEAN ANNOTATION HELPERS --------------------
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import numpy as np


def _stagger_heights(base, step, n):
    # positions: base, base-step, base-2*step, ... (stack up toward the top edge)
    return [base - i * step for i in range(n)]


def _p_to_stars(p):
    return (
        "****"
        if p < 1e-4
        else "***" if p < 1e-3 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    )


def _anchored_badge(ax, lines, colors=None, loc="upper left"):
    if not lines:
        return
    txt = "\n".join(lines)
    box = AnchoredText(
        txt,
        loc=2 if loc == "upper left" else 1,
        prop=dict(size=9),
        frameon=True,
        bbox_to_anchor=None,
        bbox_transform=ax.transAxes,
    )
    box.patch.set_alpha(0.85)
    box.patch.set_edgecolor("0.6")
    ax.add_artist(box)

    if colors:
        for t, c in zip(box.txt._text.split("\n"), colors, strict=False):
            # Matplotlib doesn’t expose per-line styling easily;
            # simplest is color whole box to darkest common color OR keep single color.
            pass  # keep single color; we encode sig by stars (not numbers)
    # We keep a single color and encode significance by stars + 'ns'
    # (no numeric Δ or p)
    return


def _draw_bracket(ax, x1, x2, y, stars, color="0.2", lw=1.2, frac=0.02, ylim=None):
    """
    Draw a bracket between x1 and x2 with a small vertical rise = frac * (yspan).
    'y' is the bottom of the bracket (in data coords). If stars is '',
    the bracket is still drawn (but we only call this for significant pairs).
    """
    # if ylim is None:
    #     ylim = ax.get_ylim()
    # ymin, ymax = ylim
    # dh = frac * (ymax - ymin)
    # ax.plot([x1, x1, x2, x2], [y, y+dh, y+dh, y], color=color, lw=lw, clip_on=False)
    # if stars:
    #     ax.text((x1+x2)/2, y+dh*1.3, stars, ha="center", va="bottom", fontsize=10, color=color)
    ax.plot(
        [x1, x1, x2, x2], [y, y + 0.02, y + 0.02, y], color=color, lw=lw, clip_on=False
    )
    if stars and stars != "ns":
        ax.text(
            (x1 + x2) / 2,
            y + 0.024,
            stars,
            ha="center",
            va="bottom",
            fontsize=10,
            color=color,
        )
    elif stars == "ns":
        ax.text(
            (x1 + x2) / 2,
            y + 0.024,
            "ns",
            ha="center",
            va="bottom",
            fontsize=9,
            color="0.5",
        )


def _collect_sig(summary, family):
    return [r for r in summary["pairwise_sig"] if r["family"] == family]


def apply_stats_annotations(
    ax, summary, ylim=None, show_families=("Age", "Genotype", "Sex"), show_non_sig=True
):
    if ylim is not None:
        ax.set_ylim(*ylim)

    # x positions in your plot
    age_map = {
        ("WT", "Male"): (1, 2),
        ("dKI", "Male"): (3, 4),
        ("WT", "Female"): (6, 7),
        ("dKI", "Female"): (8, 9),
    }
    gen_map = {
        ("Male", "2M"): (1, 3),
        ("Male", "4M"): (2, 4),
        ("Female", "2M"): (6, 8),
        ("Female", "4M"): (7, 9),
    }
    sex_map = {
        ("WT", "2M"): (1, 6),
        ("WT", "4M"): (2, 7),
        ("dKI", "2M"): (3, 8),
        ("dKI", "4M"): (4, 9),
    }

    rows = summary.get("pairwise_all", summary.get("pairwise_sig", []))

    def stars(p):
        return (
            "****"
            if p < 1e-4
            else (
                "***"
                if p < 1e-3
                else (
                    "**"
                    if p < 0.01
                    else "*" if p < 0.05 else ("ns" if show_non_sig else "")
                )
            )
        )

    def collect(family, fmap):
        out = []
        for r in rows:
            if r["family"] != family:
                continue
            key = (r["s1"], r["s2"])
            if key not in fmap:
                continue
            s = stars(r["p_adj"])
            if s:
                out.append((fmap[key], s))
        return out

    age_pairs = collect("Age", age_map) if "Age" in show_families else []
    gen_pairs = collect("Genotype", gen_map) if "Genotype" in show_families else []
    sex_pairs = collect("Sex", sex_map) if "Sex" in show_families else []

    # Place bands high in axes-y to avoid Male/Female labels (~0.09 axes-y):
    # draw Sex highest, then Genotype, then Age (each staggered slightly)
    y0_sex = 0.94
    y0_gen = 0.88
    y0_age = 0.82
    dy = 0.055  # vertical spacing between stacked brackets within a band

    for i, ((x1, x2), s) in enumerate(sex_pairs):
        _draw_bracket_axesy(ax, x1, x2, y0_sex - i * dy, s)
    for i, ((x1, x2), s) in enumerate(gen_pairs):
        _draw_bracket_axesy(ax, x1, x2, y0_gen - i * dy, s)
    for i, ((x1, x2), s) in enumerate(age_pairs):
        _draw_bracket_axesy(ax, x1, x2, y0_age - i * dy, s)


def _fmt_p(p):
    if p < 1e-4:
        return "<1e-4"
    if p < 1e-3:
        return f"{p:.1e}"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.2f}"


def _add_omnibus_box(ax, summary, ypad=0.04):
    """Put omnibus effects (significant only) above the panel."""
    sig = [o for o in summary["omnibus"] if o["p_perm"] < 0.05]
    if not sig:
        return
    lines = [
        f"{o['effect']} Δ={o['contrast']:+.3f}, p_perm={_fmt_p(o['p_perm'])}"
        for o in sig
    ]
    txt = "Omnibus: " + "; ".join(lines)
    ax.text(
        0.5,
        1.02 + ypad,
        txt,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        weight="bold",
    )


def _add_pairwise_summary(ax, summary, family, color="k", where="top", max_items=4):
    """One concise line listing significant pairwise results for a given family."""
    sig = [p for p in summary["pairwise_sig"] if p["family"] == family]
    if not sig:
        return
    # sort by adjusted p
    sig = sorted(sig, key=lambda r: r["p_adj"])[:max_items]
    if family == "Age":
        # items like ('WT','Female') or ('dKI','Male')
        parts = [
            f"{r['s1']}-{r['s2']}: Δ={r['HL']:+.2f} [{r['lo']:+.2f},{r['hi']:+.2f}], p_adj={_fmt_p(r['p_adj'])}"
            for r in sig
        ]
        label = "Age (4M−2M)"
    elif family == "Genotype":
        # items like ('Female','2M') or ('Male','4M')
        parts = [
            f"{r['s1']} {r['s2']}: Δ={r['HL']:+.2f}, p_adj={_fmt_p(r['p_adj'])}"
            for r in sig
        ]
        label = "Genotype (dKI−WT)"
    elif family == "Sex":
        # items like ('WT','2M') or ('dKI','4M')
        parts = [
            f"{r['s1']} {r['s2']}: Δ={r['HL']:+.2f}, p_adj={_fmt_p(r['p_adj'])}"
            for r in sig
        ]
        label = "Sex (F−M)"
    else:
        return

    txt = f"{label}: " + "; ".join(parts)
    y = 1.02 if where == "top" else -0.20
    va = "bottom" if where == "top" else "top"
    ax.text(
        0.5, y, txt, transform=ax.transAxes, ha="center", va=va, fontsize=9, color=color
    )


# Optional: minimal bracket between 2M and 4M (Age family) and WT vs dKI (Genotype family)
def _bracket(ax, x1, x2, y, text=None, color="k"):
    ax.plot([x1, x1, x2, x2], [y, y + 0.02, y + 0.02, y], color=color, lw=1)
    if text:
        ax.text(
            (x1 + x2) / 2,
            y + 0.025,
            text,
            ha="center",
            va="bottom",
            fontsize=9,
            color=color,
        )


def annotate_panel_from_stats(ax, summary, panel="OiP", ylim=None):
    # 1) Omnibus (significant only)
    _add_omnibus_box(ax, summary, ypad=0.00)

    # 2) Concise pairwise summaries (top). Show Age and Genotype (most interpretable with your layout)
    _add_pairwise_summary(ax, summary, family="Age", color="dimgray", where="top")
    _add_pairwise_summary(ax, summary, family="Genotype", color="dimgray", where="top")

    # 3) Optional small brackets on the x-axis:
    # positions used earlier: Male block = [1,2,3,4], Female block = [6,7,8,9]
    # Age: (2M vs 4M) -> (1,2), (3,4), (6,7), (8,9)
    # Genotype: (WT vs dKI) -> (1,3), (2,4), (6,8), (7,9)
    if ylim is None:
        ylim = ax.get_ylim()
    yb = ylim[0] + 0.02 * (ylim[1] - ylim[0])
    yg = ylim[0] + 0.09 * (ylim[1] - ylim[0])

    # Draw only if that contrast appears significant in summary
    sig_age = {
        (r["s1"], r["s2"])
        for r in summary["pairwise_sig"]
        if r["family"] == "Age" and r["p_adj"] < 0.05
    }
    sig_gen = {
        (r["s1"], r["s2"])
        for r in summary["pairwise_sig"]
        if r["family"] == "Genotype" and r["p_adj"] < 0.05
    }

    # map helpers for brackets
    # Age family key: (Genotype, Sex)
    age_pos = {
        ("WT", "Male"): (1, 2),
        ("dKI", "Male"): (3, 4),
        ("WT", "Female"): (6, 7),
        ("dKI", "Female"): (8, 9),
    }
    for (g, s), (x1, x2) in age_pos.items():
        if (g, s) in sig_age:
            _bracket(ax, x1, x2, yb, text="*", color="black")

    # Genotype family key: (Sex, Age)
    gen_pos = {
        ("Male", "2M"): (1, 3),
        ("Male", "4M"): (2, 4),
        ("Female", "2M"): (6, 8),
        ("Female", "4M"): (7, 9),
    }
    for (s, a), (x1, x2) in gen_pos.items():
        if (s, a) in sig_gen:
            _bracket(ax, x1, x2, yg, text="*", color="black")


def _draw_bracket_axesy(ax, x1, x2, y_ax, stars, lw=1.3, color="0.2"):
    """
    Draw a bracket between data x1..x2, at a fixed *axes* y position y_ax (0..1).
    Keeps brackets away from data and from 'Male/Female' labels.
    """
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot(
        [x1, x1, x2, x2],
        [y_ax, y_ax + 0.02, y_ax + 0.02, y_ax],
        transform=trans,
        color=color,
        lw=lw,
        clip_on=False,
    )
    if stars:
        ax.text(
            (x1 + x2) / 2,
            y_ax + 0.025,
            stars if stars != "ns" else "ns",
            transform=trans,
            ha="center",
            va="bottom",
            fontsize=10 if stars != "ns" else 9,
            color=color,
        )


def _main_effects_badge(ax, summary):
    # Expect summary["omnibus"] as list of dicts: {"effect": "...", "p_perm": ...}
    # Always show Sex, Age, Genotype with stars only.
    want = {"Sex": "Sex", "Age": "Age", "Genotype": "Genotype"}
    have = {
        o["effect"]: o["p_perm"]
        for o in summary.get("omnibus", [])
        if o["effect"] in want
    }
    lines = []
    for eff in ("Sex", "Age", "Genotype"):
        p = have.get(eff, 1.0)
        lines.append(f"{eff}: {_p_to_stars(p)}")
    _anchored_badge(ax, lines, loc="upper left")


def _bootstrap_ci(a, n_boot=2000, func=np.median, ci=95, rng=None):
    a = np.asarray(a).astype(float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return np.nan, (np.nan, np.nan)
    rng = np.random.default_rng(None if rng is None else rng)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boots[i] = func(rng.choice(a, size=a.size, replace=True))
    lo = (100 - ci) / 2
    hi = 100 - lo
    return func(a), (np.percentile(boots, lo), np.percentile(boots, hi))


def _extract_groups(cog, score_key, male_idx, female_idx, wt_idx, dki_idx):
    # order: WT 2M, WT 4M, dKI 2M, dKI 4M
    g_male = [
        cog[f"{score_key}_2M"][male_idx & wt_idx],
        cog[f"{score_key}_4M"][male_idx & wt_idx],
        cog[f"{score_key}_2M"][male_idx & dki_idx],
        cog[f"{score_key}_4M"][male_idx & dki_idx],
    ]
    g_fem = [
        cog[f"{score_key}_2M"][female_idx & wt_idx],
        cog[f"{score_key}_4M"][female_idx & wt_idx],
        cog[f"{score_key}_2M"][female_idx & dki_idx],
        cog[f"{score_key}_4M"][female_idx & dki_idx],
    ]
    return g_male, g_fem


def _plot_panel(
    ax,
    groups_male,
    groups_fem,
    title,
    ylabel,
    y0=0.0,
    ylim=None,
    jitter=0.2,
    seed=4,
    show_counts=False,
):
    # positions (Male block left: WT {2M,4M}, dKI {2M,4M}; Female block right: same)
    # This guarantees 2M (left) and 4M (right) within each genotype.
    pos_m = np.array([1, 2, 3, 4])  # WT 2M, WT 4M, dKI 2M, dKI 4M
    pos_f = np.array([6, 7, 8, 9])  # WT 2M, WT 4M, dKI 2M, dKI 4M

    # draw violins
    v1 = ax.violinplot(
        groups_male,
        positions=pos_m,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    v2 = ax.violinplot(
        groups_fem,
        positions=pos_f,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    # colors
    col_m = "C1"  # male
    col_f = "C2"  # female
    for b in v1["bodies"]:
        b.set_alpha(0.1)
        b.set_facecolor(col_m)
        b.set_edgecolor("black")
        b.set_linewidth(0.8)
    for b in v2["bodies"]:
        b.set_alpha(0.1)
        b.set_facecolor(col_f)
        b.set_edgecolor("black")
        b.set_linewidth(0.8)

    # helper: jittered raw points
    rng = np.random.default_rng(seed)

    def scatter_points(groups, positions, color):
        for g, x in zip(groups, positions, strict=False):
            y = np.asarray(g, float)
            y = y[~np.isnan(y)]
            if y.size == 0:
                continue
            xj = x + rng.uniform(-jitter, jitter, size=y.size)
            ax.scatter(
                xj,
                y,
                s=16,
                alpha=0.6,
                color=color,
                facecolors="none",
                linewidths=1,
                zorder=3,
            )

    # draw points
    scatter_points(groups_male, pos_m, col_m)
    scatter_points(groups_fem, pos_f, col_f)

    # medians + 95% CI whiskers + n labels
    def annotate(groups, positions, color):
        for g, x in zip(groups, positions, strict=False):
            med, (lo, hi) = _bootstrap_ci(g, func=np.median)
            n = int(np.sum(~np.isnan(g)))
            # CI line
            ax.plot([x, x], [lo, hi], linewidth=2, color=color, zorder=4)
            # median marker
            ax.plot(x, med, marker="o", markersize=5, color=color, zorder=5)
            if show_counts:
                n = int(np.sum(~np.isnan(g)))
                ax.text(
                    x,
                    ax.get_ylim()[0] - 0.3 if ylim is None else ylim[0],
                    f"n={n}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=color,
                )
                # n label at bottom
            # yb = (ax.get_ylim()[0] if ylim is None else ylim[0])
            # ax.text(x, yb, f"n={n}", ha='center', va='top', fontsize=9, color=color)
            # ax.text(x, ax.get_ylim()[0] - 0.3 if ylim is None else ylim[0], f"n={n}", ha='center', va='bottom', fontsize=9, rotation=0, color=color)

    # set temporary ylim to place n labels, then annotate
    if ylim is None:
        ax.set_ylim(auto=True)
    annotate(groups_male, pos_m, "C1")
    annotate(groups_fem, pos_f, "C2")

    # x ticks and labels (repeat per block)
    ax.set_xticks(np.r_[pos_m, pos_f])
    ax.set_xticklabels(
        ["WT 2M", "WT 4M", "dKI 2M", "dKI 4M"] * 2, rotation=0, fontsize=12
    )

    # headers for sex blocks
    topy = ax.get_ylim()[1] if ylim is None else ylim[1]
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        pos_m.mean(),
        0.09,
        "Male",
        transform=trans,
        ha="center",
        va="bottom",
        fontsize=13,
        weight="bold",
        fontstyle="italic",
        zorder=6,
        color="C1",
    )
    ax.text(
        pos_f.mean(),
        0.09,
        "Female",
        transform=trans,
        ha="center",
        va="bottom",
        fontsize=13,
        weight="bold",
        fontstyle="italic",
        zorder=6,
        color="C2",
    )

    # cosmetics
    ax.axhline(y0, color="k", linewidth=1.0)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=14, weight="bold")


# ---- build indices
male_ind = cog_data["Sexe"] == "M"
female_ind = cog_data["Sexe"] == "F"
wt_ind = cog_data["Genotype"] == "wt"
dki_ind = cog_data["Genotype"] == "dKI"

# ---- extract groups
m_OiP, f_OiP = _extract_groups(cog_data, "OiP", male_ind, female_ind, wt_ind, dki_ind)
m_RO, f_RO = _extract_groups(cog_data, "RO24h", male_ind, female_ind, wt_ind, dki_ind)

# ---- figure
plt.figure(1, figsize=(10.5, 7.5))
plt.clf()
ax1 = plt.subplot(2, 1, 1)
_plot_panel(
    ax1, m_OiP, f_OiP, title="OiP", ylabel="OiP score", y0=0.0, show_counts=False
)

ax1.axhline(0.2, color="gray", linestyle="--", linewidth=1.0)
ax1.axhline(-0.2, color="gray", linestyle="--", linewidth=1.0)
ax1.set_ylim(-0.5, 1.1)
ax1.set_yticks([-0.2, 0.2, 1])
ax1.tick_params(axis="y", labelsize=12)


ax2 = plt.subplot(2, 1, 2)
_plot_panel(
    ax2, m_RO, f_RO, title="RO24h", ylabel="RO24h score", y0=0.0, show_counts=False
)
ax2.axhline(0.2, color="gray", linestyle="--", linewidth=1.0)
ax2.axhline(-0.2, color="gray", linestyle="--", linewidth=1.0)
ax2.set_ylim(-0.5, 1.1)
ax2.set_yticks([-0.2, 0.2, 1])
ax2.tick_params(axis="y", labelsize=12)
for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Load summaries you saved earlier
sum_OiP = load_stats(paths["preprocessed"], "OiP", B=10_000, rng=42)
sum_RO = load_stats(paths["preprocessed"], "RO24h", B=10_000, rng=42)

# Clean annotations
apply_stats_annotations(ax1, sum_OiP, ylim=(-0.6, 1.1), show_non_sig=False)
apply_stats_annotations(ax2, sum_RO, ylim=(-0.6, 1.1), show_non_sig=False)


plt.tight_layout()
plt.savefig(
    paths["f_cog"] / f"violin_OiP_RO24h_{timecourse_folder}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    paths["f_cog"] / f"violin_OiP_RO24h_{timecourse_folder}.svg",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# %%
