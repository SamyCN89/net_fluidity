#!/usr/bin/env python3
"""
Created on Mon Sep 23 13:26:30 2024

@author: samy
"""
# %%


from tkinter import font
from turtle import title
import matplotlib.pyplot as plt
import numpy as np

# from functions_analysis import *
from shared_code.fun_loaddata import load_timeseries_bundle
from shared_code.fun_paths import get_paths
from shared_code.fun_utils import set_figure_params

# =============================================================================
# This code uses
#   cog_data_filtered, data_ts, grouping_data, and data_mc_mod

# Plot
# PLOT MC using allegiance reference
# Plot MC for each individual
# Plot MC intra/inter modular per group
# Plot MC intra/inter for each individual
# Plot MC modules for each individual
# plot MC modules per group
# =============================================================================
# %%
# ========================== Figure parameters ================================
# Save options
plt.rcParams.update(
    {
        "axes.labelsize": 15,
        "axes.titlesize": 13,
        # 'axes.spines.left': False, 'axes.spines.bottom': False,
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
)

title_groups = ('oip', 'nor', 'genotype', 'sex')

save_fig = set_figure_params(True)
bins_parameter = 200

# %%Load and plot parameters
# =================== Paths and folders =======================================
# ----------------- Paths -----------------
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

# ========================== Load data =========================
# ts data and metadata
ts = bundle.ts
n_animals = bundle.n_animals
regions = bundle.n_regions
anat_labels = bundle.anat_labels
is_2month_old = bundle.is_2month_old
total_tr = bundle.total_tr
# Grouping data
mask_groups = bundle.mask_groups
label_variables = bundle.label_variables

# %%
# ========================= Create folders ================================
dataset_name = paths["results"].name
mc_mod_dir = paths["mc_mod"]  # same as in the other script
trimers_dir = paths["trimers"]
motif_fig_dir = paths["f_motif"]

for directory in (mc_mod_dir, trimers_dir, motif_fig_dir):
    directory.mkdir(parents=True, exist_ok=True)
folders = {"2mois": "TC_2months", "4mois": "TC_4months"}
# %%

# Metaconnectivity
# =============================================================================
# Parameters speed

PROCESSORS = -1

lag = 1
tau = 5
window_size = 7
window_parameter = (5, 100, 1)

# Parameters allegiance analysis
n_runs_allegiance = 1000
gamma_pt_allegiance = 100
ind_ref = mask_groups[0][0]  # the mask of the reference matrix
# label_ref = label_variables[0][0] #The label of the reference matrix
# label_ref = "Good2m"  # The label of the reference matrix
label_ref = 'wt2m' #The label of the reference matrix

tau_array = np.append(np.arange(0, tau), tau)
lentau = len(tau_array)

time_window_min, time_window_max, time_window_step = window_parameter
time_window_range = np.arange(time_window_min, time_window_max + 1, time_window_step)
# %%
# ----------------- Load MC (same naming as in MC script) -----------------
mc_data_filename = (
    f"mc_allegiance_ref(runs={label_ref}_gammaval={n_runs_allegiance})="
    f"{gamma_pt_allegiance}_lag={lag}_windowsize={window_size}_animals={n_animals}_regions={regions}.npz"
).replace(" ", "")

data_mc_mod_filename = mc_mod_dir / mc_data_filename
data_mc_mod = np.load(data_mc_mod_filename, allow_pickle=True)

mc_allegiance = data_mc_mod["mc"]
mc_ref_allegiance_communities = data_mc_mod["mc_ref_allegiance_communities"]
mc_val = data_mc_mod["mc_val_tril"]
mc_mod_idx = np.squeeze(data_mc_mod["mc_mod_idx"])
mc_reg_idx = data_mc_mod["mc_reg_idx"]
mc_idx = data_mc_mod["mc_idx_tril"]  # <-- add this line

# ----------------- Load trimers indices -----------------
trimers_filename = (
    f"trimers_allegiance_ref(runs={label_ref}_gammaval={n_runs_allegiance})="
    f"{gamma_pt_allegiance}_lag={lag}_windowsize={window_size}_animals={n_animals}_regions={regions}.npz"
).replace(" ", "")

trimers_path = trimers_dir / trimers_filename
trimers_load = np.load(trimers_path)
mc_nplets_index = trimers_load["nplets_index"]

# %% Figures
# =========================== Labels figures ==================================
label_mclinks = r"Inter-regional links"
label_mc_formula = r"MC$_{[ij, kl]} = CC[FC_{ij}(t), FC_{kl}(t)]$"
label_yhist = "Probability density"
label_mc_trimers = r"Trimer Meta-strengths $MC(i) = \sum_{jk} MC_{[ij, il]}$"

label_trimer = r"Trimer = MC$_{[ir, jr]}$"


# %%
# =============================================================================
# Plot MC trimer intra/inter per group
# =============================================================================
n_groups = len(mask_groups)

# plt.figure(1, figsize=(5 * n_groups, 13))
plt.figure(1, figsize=(13, 8))
plt.clf()

for idx, setb in enumerate(mask_groups):
    aux_label = label_variables[idx]

    # column index
    col = idx

    # --- Row 1: all trimers ---
    ax1 = plt.subplot(3, n_groups, 1 + col)
    ax1.set_title("Trimer Motif", fontsize=11)

    ax1.hist(
        [mc_val[mod][:, mc_nplets_index > 0].ravel() for mod in setb],
        histtype="step",
        bins=bins_parameter,
        density=True,
        label=aux_label,
        alpha=0.5,
    )
    ax1.set_yscale("log")
    ax1.set_ylabel(label_yhist, fontsize=11)
    ax1.set_xlabel(label_trimer, fontsize=11)
    ax1.set_xticks([-0.8, 0, 0.8])
    # ax1.set_ylim(1e-5, 1e0)

    # --- Row 2: intra-module trimers ---
    ax2 = plt.subplot(3, n_groups, 1 + n_groups + col)
    ax2.set_title("Trimers Intra-module", fontsize=11)

    ax2.hist(
        [
            mc_val[mod][:, (mc_nplets_index > 0) * (mc_mod_idx > 0)].ravel()
            for mod in setb
        ],
        histtype="step",
        bins=bins_parameter,
        density=True,
        label=aux_label,
        alpha=0.5,
    )
    ax2.set_yscale("log")
    ax2.set_ylabel(label_yhist, fontsize=11)
    ax2.set_xlabel(label_trimer, fontsize=11)
    ax2.set_xticks([-0.8, 0, 0.8])
    # ax2.set_ylim(1e-5, 1e0)

    # --- Row 3: inter-module trimers ---
    ax3 = plt.subplot(3, n_groups, 1 + 2 * n_groups + col)
    ax3.set_title("Trimers inter-module", fontsize=11)

    ax3.hist(
        [
            mc_val[mod][:, (mc_nplets_index > 0) * (mc_mod_idx == 0)].ravel()
            for mod in setb
        ],
        histtype="step",
        bins=bins_parameter,
        density=True,
        label=aux_label,
        alpha=0.5,
    )
    ax3.set_yscale("log")
    ax3.set_ylabel(label_yhist, fontsize=11)
    ax3.set_xlabel(label_trimer, fontsize=11)
    ax3.set_xticks([-0.8, 0, 0.8])
    # ax3.set_ylim(1e-5, 1e0)

# only one legend to avoid over-plotting
plt.subplot(3, n_groups, 1 + 2 * n_groups)  # bottom-left panel
plt.legend(fontsize=8)

plt.tight_layout()
if save_fig:
    plt.savefig(motif_fig_dir / "hist_trimer_mcvalues_mod(intra_inter).png")

# %%
# =============================================================================
# Plot MC trimer per group and per module
# =============================================================================
unique_modules = np.unique(mc_ref_allegiance_communities)


n_modules = len(unique_modules)
n_groups = len(mask_groups)

fig_mod, axes = plt.subplots(
    n_modules,
    n_groups,
    figsize=(4 * n_groups, 3 * n_modules),
    sharex=True,
    sharey=True,
)

# If only one group or one module, axes may be 1D → force 2D
if n_modules == 1 and n_groups == 1:
    axes = np.array([[axes]])
elif n_modules == 1:
    axes = axes.reshape(1, n_groups)
elif n_groups == 1:
    axes = axes.reshape(n_modules, 1)

# -------- one legend per column, at the top but not on top of title --------


for row_mod, mod_id in enumerate(unique_modules):
    for col_grp, setb in enumerate(mask_groups):
        aux_label = label_variables[col_grp]  # list of group names for this column
        ax = axes[row_mod, col_grp]

        # Row labels
        if col_grp == 0:
            ax.set_ylabel(f"Module {mod_id}\n{label_yhist}", fontsize=10)

        # Only bottom row gets x-labels
        if row_mod == n_modules - 1:
            ax.set_xlabel(label_trimer, fontsize=12)
            ax.set_xticks([-0.7, 0, 0.8])
        else:
            ax.set_xticks([])

        # Column titles (top row)
        if row_mod == 0:
            ax.set_title(title_groups[col_grp], fontsize=10, pad=20)

        # Histogram for this module × grouping
        ax.hist(
            [mc_val[grp_idx][:, mc_mod_idx == mod_id].ravel() for grp_idx in setb],
            histtype="step",
            bins=50,
            density=True,
            label=aux_label,  # 1 label per group
        )
        ax.set_yscale("log")
        ax.set_xlim(-0.9, 0.9)

# -------- one legend per column, at the top --------
for col_grp in range(n_groups):
    ax_top = axes[0, col_grp]

    # Column titles – centered, close to the axes
    ax_top.set_title(title_groups[col_grp], fontsize=10, pad=10)

    handles, labels = ax_top.get_legend_handles_labels()
    ax_top.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.4),  # above and to the left of the axes
        ncol=2,
        fontsize=8,
        frameon=False,
    )

fig_mod.tight_layout(rect=[0, 0, 0.7, 1.1])  # leave a bit of space for legends
if save_fig:
    fig_mod.savefig(motif_fig_dir / "hist_trimer_mcvalues_mod(intra_num).png")

# %%
# =============================================================================
# Trimers per intramodule per region
# Plot MC trimer intra/inter per group and per module
# =============================================================================


for idx, setb in enumerate(mask_groups):
    # setb= mask_groups[0]
    # idx=0
    plt.figure(18 + idx, figsize=(15, 12))
    plt.clf()
    aux_label = label_variables[idx]
    for tr in range(regions):
        plt.subplot(6, 7, 1 + tr)
        plt.title(anat_labels[tr], fontsize=10)
        # plt.hist(([trimer_per_region[tr,mod].ravel() for mod in setb]),
        # plt.hist(([mc_val[mod][:, (mc_nplets_index>0) * (mc_mod_idx==0)].ravel()     for mod in setb]),
        plt.hist(
            (
                [
                    mc_val[mod][
                        :,
                        (mc_nplets_index > 0)
                        * (mc_mod_idx > 0)
                        * ((mc_nplets_index - 1) == tr),
                    ].ravel()
                    for mod in setb
                ]
            ),
            histtype="step",
            bins=50,
            density=True,
            label=aux_label,
        )
        plt.yscale("log")
        plt.ylim(10e-4, 10e0)
        plt.yticks([])
        if tr + 1 > (5 * 7):
            plt.xticks([-0.8, 0, 0.8], fontsize=13)
        else:
            plt.xticks([])
    plt.legend()
    plt.subplot(6, 7, 1 + tr + 1)
    plt.title("trimer per region")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if save_fig == True:
        # plt.savefig(path_figures + 'trimers/hist_trimer_intramodule_per_region_%s_%s.png'%(aux_label[0],aux_label[2]))
        plt.savefig(
            motif_fig_dir
            / f"hist_trimer_intramodule_per_region_{aux_label[0]}_{aux_label[2]}.png"
        )
# %%
# =============================================================================
# Trimers per intra-module per region
# =============================================================================
# subplotindex = (0,3,6,9)
for idx, setb in enumerate(mask_groups):
    # setb= mask_groups[0]
    # idx=0
    plt.figure(18 + idx, figsize=(15, 12))
    plt.clf()
    aux_label = label_variables[idx]

    for tr in range(regions):
        plt.subplot(6, 7, 1 + tr)
        plt.title(anat_labels[tr], fontsize=10)
        plt.hist(
            (
                [
                    mc_val[mod][
                        :,
                        (mc_nplets_index > 0)
                        * (mc_mod_idx > 0)
                        * (mc_nplets_index == tr + 1),
                    ].ravel()
                    for mod in setb
                ]
            ),
            histtype="step",
            bins=50,
            density=True,
            label=aux_label,
        )
        plt.yscale("log")
        plt.ylim(10e-4, 10e0)
        plt.yticks([])
        if tr + 1 > (5 * 7):
            plt.xticks([-0.8, 0, 0.8], fontsize=13)
        else:
            plt.xticks([])
    plt.legend()
    plt.subplot(6, 7, 1 + tr + 1)
    plt.title("trimer per region")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if save_fig==True:
        plt.savefig(motif_fig_dir / f"hist_trimer_intramodule_per_region_{aux_label[0]}_{aux_label[2]}.png")
# %%
# =============================================================================
# Trimers per inter-module per region
# =============================================================================
for idx, setb in enumerate(mask_groups):
    # setb= mask_groups[0]
    # idx=0
    plt.figure(21 + idx, figsize=(15, 12))
    plt.clf()
    aux_label = label_variables[idx]
    for tr in range(regions):
        print(tr)
        plt.subplot(6, 7, 1 + tr)
        plt.title(anat_labels[tr], fontsize=10)
        # plt.hist(([trimer_per_region[tr,mod].ravel() for mod in setb]),
        plt.hist(
            (
                [
                    mc_val[mod][
                        :,
                        (mc_nplets_index > 0)
                        * (mc_mod_idx == 0)
                        * (mc_nplets_index == tr + 1),
                    ].ravel()
                    for mod in setb
                ]
            ),
            # plt.hist(([mc_val[mod][:, (mc_nplets_index>0) * (mc_mod_idx>0) * (mc_nplets_index==tr+1)].ravel()     for mod in setb]),
            histtype="step",
            bins=50,
            density=True,
            label=aux_label,
        )
        plt.yscale("log")
        plt.ylim(10e-4, 10e0)
        plt.yticks([])
        if tr + 1 > (5 * 7):
            plt.xticks([-0.8, 0, 0.8], fontsize=13)
        else:
            plt.xticks([])
    plt.legend()
    # plt.subplot(6,7,1+tr+1)
    plt.tight_layout()
    if save_fig==True:
        plt.savefig(motif_fig_dir / f"hist_trimer_intermodule_per_region_{aux_label[0]}_{aux_label[2]}.png")



# %%
# =============================================================================
# Search for the top values
# =============================================================================


# mc_allegiance_10p = np.nanpercentile(mc_allegiance,q=90,axis=(1,2))
# mc_top10_mask = mc_allegiance>mc_allegiance_10p[:,None,None]
mc_allegiance_thr_10pos = np.nanpercentile(mc_allegiance, q=90, axis=(1, 2))
mc_top10_mask = mc_allegiance > mc_allegiance_thr_10pos[:, None, None]

mc_allegiance_thr_10neg = np.nanpercentile(mc_allegiance, q=10, axis=(1, 2))
mc_bottom10_mask = mc_allegiance < mc_allegiance_thr_10neg[:, None, None]

mc_top10_idx = mc_top10_mask[:, mc_idx[:, 0], mc_idx[:, 1]]
mc_bottom10_idx = mc_bottom10_mask[:, mc_idx[:, 0], mc_idx[:, 1]]

# %%
top10_mask   = mc_top10_idx                  # shape (126, 335790)
inter_mask   = (mc_mod_idx == 0)[None, :]    # broadcast to (1, n_edges)
intra_mask   = (mc_mod_idx > 0)[None, :]

plets4_mask  = (mc_nplets_index == 0)[None, :]
plets3_mask  = (mc_nplets_index > 0)[None, :]


#%%
# =============================================================================
# QC - Plot top 10% and trimers intra/inter per group
# =============================================================================

def extract_mc(mc_val, mask):
    """
    mc_val : (n_animals, n_edges)
    mask   : boolean mask broadcastable to mc_val
    """
    return mc_val[mask]



plt.figure(60, figsize=(8, 10))
plt.clf()

# ================= Panel 1: intra vs inter =================
plt.subplot(311)
plt.hist(
    [
        extract_mc(mc_val, top10_mask),
        extract_mc(mc_val, top10_mask & inter_mask),
        extract_mc(mc_val, top10_mask & intra_mask),
    ],
    bins=70,
    density=True,
    histtype="step",
    label=("top 10%", "top10 inter-module", "top10 intra-module"),
    linewidth=2,
    alpha=0.5
)
plt.legend()
plt.xlabel(r"MC$_{[ij, kl]}$")
plt.ylabel("Probability density")

# ================= Panel 2: 3- vs 4-plets =================
plt.subplot(312)
plt.hist(
    [
        extract_mc(mc_val, top10_mask),
        extract_mc(mc_val, top10_mask & plets4_mask),
        extract_mc(mc_val, top10_mask & plets3_mask),
    ],
    bins=70,
    density=True,
    histtype="step",
    label=("top 10%", "top10 4-plets", "top10 3-plets"),
    linewidth=2,
    alpha=0.5
)
plt.legend()
plt.xlabel(r"MC$_{[ij, kl]}$")
plt.ylabel("Probability density")

# ================= Panel 3: full decomposition =================
plt.subplot(313)
plt.hist(
    [
        extract_mc(mc_val, top10_mask),
        extract_mc(mc_val, top10_mask & plets4_mask & inter_mask),
        extract_mc(mc_val, top10_mask & plets3_mask & inter_mask),
        extract_mc(mc_val, top10_mask & plets4_mask & intra_mask),
        extract_mc(mc_val, top10_mask & plets3_mask & intra_mask),
    ],
    bins=70,
    density=True,
    histtype="step",
    label=(
        "top 10%",
        "4-plets inter",
        "3-plets inter",
        "4-plets intra",
        "3-plets intra",
    ),
    linewidth=2,
    alpha=0.5
)
plt.legend()
plt.xlabel(r"MC$_{[ij, kl]}$")
plt.ylabel("Probability density")

plt.tight_layout()

if save_fig == True:
    plt.savefig(
        motif_fig_dir / f"top10_trimer_intermodule_{aux_label[0]}_{aux_label[2]}.png"
    )

#%%
# =============================================================================
# ECDF of top 10% MC values
# =============================================================================
def ecdf(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    x = np.sort(x)
    n = x.size
    y = np.arange(1, n + 1) / (n + 1.0)
    return x, y

x_all   = extract_mc(mc_val, top10_mask)
x_inter = extract_mc(mc_val, top10_mask & inter_mask)
x_intra = extract_mc(mc_val, top10_mask & intra_mask)

fig, ax = plt.subplots(figsize=(6, 4))

for x, lab in [
    (x_all,   "top 10%"),
    (x_inter, "top10 inter-module"),
    (x_intra, "top10 intra-module"),
]:
    xs, ys = ecdf(x)
    ax.plot(xs, ys, label=lab)

ax.set_xlabel(r"MC$_{[ij, kl]}$")
ax.set_ylabel("ECDF")
ax.set_title("ECDF of top-10% MC values")
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()
#%%
# %%
# --------------------------------------------------
# Top-10% threshold
# --------------------------------------------------
mc_all = mc_val.ravel()
thr_90 = np.percentile(mc_all, 90)

top_mask = mc_val >= thr_90    # shape (n_animals, n_edges)

inter_mask  = (mc_mod_idx == 0)[None, :]
intra_mask  = (mc_mod_idx > 0)[None, :]

plets3_mask = (mc_nplets_index > 0)[None, :]
plets4_mask = (mc_nplets_index == 0)[None, :]

# Baseline: all top-10% values
x_base = mc_val[top_mask]

# Combinatory subsets
x_3_intra = mc_val[top_mask & plets3_mask & intra_mask]
x_4_intra = mc_val[top_mask & plets4_mask & intra_mask]

x_3_inter = mc_val[top_mask & plets3_mask & inter_mask]
x_4_inter = mc_val[top_mask & plets4_mask & inter_mask]

def shift_function(x_ref, x_cmp, pcts):
    q_ref = np.percentile(x_ref, pcts)
    q_cmp = np.percentile(x_cmp, pcts)
    return q_cmp - q_ref

PCTS = np.linspace(5, 95, 19)

shifts_top = {
    "3-plets intra": shift_function(x_base, x_3_intra, PCTS),
    "4-plets intra": shift_function(x_base, x_4_intra, PCTS),
    "3-plets inter": shift_function(x_base, x_3_inter, PCTS),
    "4-plets inter": shift_function(x_base, x_4_inter, PCTS),
}

fig, ax = plt.subplots(figsize=(7, 4))

ax.axhline(0, color="gray", linestyle="--", lw=1)

for label, shift in shifts_top.items():
    ax.plot(PCTS, shift, marker="o", label=label)

ax.set_xlabel("Percentile (within top-10% MC tail)")
ax.set_ylabel("Δ MC (relative to top-10% baseline)")
ax.set_title("Shift functions within the top-10% MC tail")

ax.grid(True, alpha=0.3)
ax.legend()

ax.text(
    0.02, 0.02,
    "Positive shift = stronger coupling\nNegative shift = weaker coupling",
    transform=ax.transAxes,
    fontsize=9,
)

fig.tight_layout()



# %%
# =============================================================================
# QC - Plot bottom 10% and trimers intra/inter per group
# =============================================================================
bottom10_mask = mc_bottom10_idx                    # (n_animals, n_edges)

inter_mask  = (mc_mod_idx == 0)[None, :]           # broadcastable
intra_mask  = (mc_mod_idx > 0)[None, :]

plets4_mask = (mc_nplets_index == 0)[None, :]
plets3_mask = (mc_nplets_index > 0)[None, :]

plt.figure(61, figsize=(8, 10))
plt.clf()

# =========================================================
# Panel 1 — Bottom 10%: intra vs inter
# =========================================================
plt.subplot(311)

plt.hist(
    [
        extract_mc(mc_val, bottom10_mask),
        extract_mc(mc_val, bottom10_mask & inter_mask),
        extract_mc(mc_val, bottom10_mask & intra_mask),
    ],
    bins=70,
    density=True,
    histtype="step",
    label=("bottom 10%", "b10 inter-module", "b10 intra-module"),
)

plt.yscale("log")
plt.legend()
plt.xlabel(r"MC$_{[ij, kl]}$")
plt.ylabel("Probability density")

# =========================================================
# Panel 2 — Bottom 10%: 3-plets vs 4-plets
# =========================================================
plt.subplot(312)

plt.hist(
    [
        extract_mc(mc_val, bottom10_mask),
        extract_mc(mc_val, bottom10_mask & plets4_mask),
        extract_mc(mc_val, bottom10_mask & plets3_mask),
    ],
    bins=70,
    density=True,
    histtype="step",
    label=("bottom 10%", "b10 4-plets", "b10 3-plets"),
)

plt.yscale("log")
plt.legend()
plt.xlabel(r"MC$_{[ij, kl]}$")
plt.ylabel("Probability density")

# =========================================================
# Panel 3 — Bottom 10%: full decomposition
# =========================================================
plt.subplot(313)

plt.hist(
    [
        extract_mc(mc_val, bottom10_mask),
        extract_mc(mc_val, bottom10_mask & plets4_mask & inter_mask),
        extract_mc(mc_val, bottom10_mask & plets3_mask & inter_mask),
        extract_mc(mc_val, bottom10_mask & plets4_mask & intra_mask),
        extract_mc(mc_val, bottom10_mask & plets3_mask & intra_mask),
    ],
    bins=70,
    density=True,
    histtype="step",
    label=(
        "bottom 10%",
        "4-plets inter-module",
        "3-plets inter-module",
        "4-plets intra-module",
        "3-plets intra-module",
    ),
)

plt.yscale("log")
plt.legend()
plt.xlabel(r"MC$_{[ij, kl]}$")
plt.ylabel("Probability density")

plt.tight_layout()

if save_fig:
    plt.savefig(
        motif_fig_dir / f"bottom10_trimer_decomposition_{aux_label[0]}_{aux_label[2]}.png"
    )
#%%
# --------------------------------------------------
# Bottom-10% threshold
# --------------------------------------------------
mc_all = mc_val.ravel()
thr_10 = np.percentile(mc_all, 10)

bottom_mask = mc_val <= thr_10    # shape (n_animals, n_edges)

inter_mask  = (mc_mod_idx == 0)[None, :]
intra_mask  = (mc_mod_idx > 0)[None, :]

plets3_mask = (mc_nplets_index > 0)[None, :]
plets4_mask = (mc_nplets_index == 0)[None, :]
# Baseline: all bottom-10% values
x_base = mc_val[bottom_mask]

# Combinatory subsets
x_3_intra = mc_val[bottom_mask & plets3_mask & intra_mask]
x_4_intra = mc_val[bottom_mask & plets4_mask & intra_mask]

x_3_inter = mc_val[bottom_mask & plets3_mask & inter_mask]
x_4_inter = mc_val[bottom_mask & plets4_mask & inter_mask]

def shift_function(x_ref, x_cmp, pcts):
    """
    Shift function: quantile(x_cmp) - quantile(x_ref)
    """
    q_ref = np.percentile(x_ref, pcts)
    q_cmp = np.percentile(x_cmp, pcts)
    return q_cmp - q_ref

PCTS = np.linspace(5, 95, 19)   # within bottom-10% distribution

shifts = {
    "3-plets intra": shift_function(x_base, x_3_intra, PCTS),
    "4-plets intra": shift_function(x_base, x_4_intra, PCTS),
    "3-plets inter": shift_function(x_base, x_3_inter, PCTS),
    "4-plets inter": shift_function(x_base, x_4_inter, PCTS),
}

fig, ax = plt.subplots(figsize=(7, 4))

ax.axhline(0, color="gray", linestyle="--", lw=1)

for label, shift in shifts.items():
    # ax.plot(PCTS,   , marker="o", label=label)
    ax.plot(PCTS, shift, marker="o", label=label)
ax.set_xlabel("Percentile (within bottom-10% MC tail)")
ax.set_ylabel("Δ MC (relative to bottom-10% baseline)")
ax.set_title("Shift functions within the bottom-10% MC tail")

ax.grid(True, alpha=0.3)
ax.legend()

ax.text(
    0.02, 0.02,
    "Negative shift = stronger anti-correlation\nPositive shift = weaker anti-correlation",
    transform=ax.transAxes,
    fontsize=9,
)

fig.tight_layout()


#%%
# %%

for idx, setb in enumerate(mask_groups):
    # idx=0
    # setb = mask_groups[0]
    aux_label = label_variables[idx]
    num_group = int(len(aux_label) / 2)

    plt.figure(62 + idx, figsize=(13, 10))
    plt.clf()

    # plt.hist(mc_val[mc_top10_idx].ravel(),

    plt.subplot(321)
    plt.title("top 10")
    plt.hist(
        (
            [mc_val[mc_top10_idx * mod[:, None]].ravel() for mod in setb]
            # mc_val[mc_top10_idx *(mc_mod_idx==0)[None,:] ],
            # mc_val[mc_top10_idx *(mc_mod_idx>0)[None,:] ])
            # for mod in setb]
        ),
        histtype="step",
        bins=70,
        density=True,
        # label=('top 10', 't10 intermod', 't10 intramod'),
        label=aux_label,
    )
    # histtype='step',bins=50, density=True)
    # plt.boxplot(mc_allegiance_10p)
    plt.yscale("log")
    plt.legend()
    plt.xlabel(r"MC$_{[ij, kl]} $", fontsize=12)
    plt.ylabel("Probability density", fontsize=10)

    plt.subplot(322)
    plt.title("Inter-modularity")
    plt.hist(
        (
            [
                mc_val[mc_top10_idx * mod[:, None] * (mc_mod_idx == 0)[None, :]].ravel()
                for mod in setb
            ]
        ),
        histtype="step",
        bins=70,
        density=True,
        # label=('top 10', 't10 4-plets', 't10 3-plets')#, 't10 4-plets intramod', 't10 3-plets intramod'),
        label=aux_label,
    )
    # plt.boxplot(mc_allegiance_10p)
    plt.yscale("log")
    plt.xlabel(r"MC$_{[ij, kl]} $", fontsize=12)
    plt.ylabel("Probability density", fontsize=10)

    plt.legend()

    plt.subplot(323)
    plt.title("Intra-modularity")
    plt.hist(
        (
            [
                mc_val[mc_top10_idx * mod[:, None] * (mc_mod_idx > 0)[None, :]].ravel()
                for mod in setb
            ]
        ),
        histtype="step",
        bins=70,
        density=True,
        # label=('top 10', 't10 4-plets', 't10 3-plets')#, 't10 4-plets intramod', 't10 3-plets intramod'),
        label=aux_label,
    )
    # plt.boxplot(mc_allegiance_10p)
    plt.yscale("log")
    plt.xlabel(r"MC$_{[ij, kl]} $", fontsize=12)
    plt.ylabel("Probability density", fontsize=10)

    plt.legend()

    plt.subplot(324)
    plt.title("Intra-modularity 4plets")
    plt.hist(
        (
            [
                mc_val[
                    mc_top10_idx
                    * mod[:, None]
                    * (mc_mod_idx > 0)[None, :]
                    * (mc_nplets_index == 0)[None, :]
                ].ravel()
                for mod in setb
            ]
        ),
        histtype="step",
        bins=70,
        density=True,
        # label=('top 10', 't10 4-plets', 't10 3-plets')#, 't10 4-plets intramod', 't10 3-plets intramod'),
        label=aux_label,
    )
    # plt.boxplot(mc_allegiance_10p)
    plt.yscale("log")
    plt.xlabel(r"MC$_{[ij, kl]} $", fontsize=12)
    plt.ylabel("Probability density", fontsize=10)

    plt.legend()
    plt.subplot(325)
    plt.title("Intra-modularity 3plets")
    plt.hist(
        (
            [
                mc_val[
                    mc_top10_idx
                    * mod[:, None]
                    * (mc_mod_idx > 0)[None, :]
                    * (mc_nplets_index > 0)[None, :]
                ].ravel()
                for mod in setb
            ]
        ),
        histtype="step",
        bins=70,
        density=True,
        # label=('top 10', 't10 4-plets', 't10 3-plets')#, 't10 4-plets intramod', 't10 3-plets intramod'),
        label=aux_label,
    )
    # plt.boxplot(mc_allegiance_10p)
    plt.yscale("log")
    plt.xlabel(r"MC$_{[ij, kl]} $", fontsize=12)
    plt.ylabel("Probability density", fontsize=10)

    plt.subplot(326)
    plt.title("3plets")
    plt.hist(
        (
            [
                mc_val[
                    mc_top10_idx * mod[:, None] * (mc_nplets_index > 0)[None, :]
                ].ravel()
                for mod in setb
            ]
        ),
        histtype="step",
        bins=70,
        density=True,
        # label=('top 10', 't10 4-plets', 't10 3-plets')#, 't10 4-plets intramod', 't10 3-plets intramod'),
        label=aux_label,
    )
    # plt.boxplot(mc_allegiance_10p)
    plt.yscale("log")
    plt.xlabel(r"MC$_{[ij, kl]} $", fontsize=12)
    plt.ylabel("Probability density", fontsize=10)

    plt.legend()

    # plt.subplot(313)
    # plt.hist((mc_val[mc_top10_idx].ravel(),
    #           mc_val[mc_top10_idx *(mc_nplets_index==0)[None,:] *(mc_mod_idx==0)[None,:] ],
    #           mc_val[mc_top10_idx *(mc_nplets_index>0)[None,:] * (mc_mod_idx==0)[None,:] ],

    #           mc_val[mc_top10_idx *(mc_nplets_index==0)[None,:] *(mc_mod_idx>0)[None,:] ],
    #           mc_val[mc_top10_idx *(mc_nplets_index>0)[None,:] * (mc_mod_idx>0)[None,:] ]),

    #          histtype='step',
    #           bins=70,
    #          density=True,
    #          label=aux_label,
    #          # label=('top 10', 't10 4-plets intermod', 't10 3-plets intermod', 't10 4-plets intramod', 't10 3-plets intramod'),
    #          )
    # # plt.boxplot(mc_allegiance_10p)
    # plt.yscale('log')
    # plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)
    # plt.ylabel('Probability density', fontsize=10)
    # plt.ylim(10e-4,10e0)
    # plt.legend()
    # plt.subplot(6,7,1+tr+1)
    plt.tight_layout()
    if save_fig == True:
        plt.savefig(
            motif_fig_dir
            / f"top10_trimer_intermodule_{aux_label[0]}_{aux_label[2]}.png"
        )
# %%
for idx, setb in enumerate(mask_groups):
    # idx=0
    # setb = mask_groups[0]
    aux_label = label_variables[idx]
    num_group = int(len(aux_label) / 2)

    plt.figure(66 + idx, figsize=(13, 10))
    plt.clf()

    # plt.hist(mc_val[mc_top10_idx].ravel(),

    plt.subplot(321)
    plt.title("bottom 10")
    plt.hist(
        (
            [mc_val[mc_bottom10_idx * mod[:, None]].ravel() for mod in setb]
            # mc_val[mc_bottom10_idx*(mc_mod_idx==0)[None,:] ],
            # mc_val[mc_bottom10_idx*(mc_mod_idx>0)[None,:] ])
            # for mod in setb]
        ),
        histtype="step",
        bins=70,
        density=True,
        # label=('top 10', 't10 intermod', 't10 intramod'),
        label=aux_label,
    )
    # histtype='step',bins=50, density=True)
    # plt.boxplot(mc_allegiance_10p)
    plt.yscale("log")
    plt.legend()
    plt.xlabel(r"MC$_{[ij, kl]} $", fontsize=12)
    plt.ylabel("Probability density", fontsize=10)

    plt.subplot(322)
    plt.title("Inter-modularity")
    plt.hist(
        (
            [
                mc_val[
                    mc_bottom10_idx * mod[:, None] * (mc_mod_idx == 0)[None, :]
                ].ravel()
                for mod in setb
            ]
        ),
        histtype="step",
        bins=70,
        density=True,
        # label=('top 10', 't10 4-plets', 't10 3-plets')#, 't10 4-plets intramod', 't10 3-plets intramod'),
        label=aux_label,
    )
    # plt.boxplot(mc_allegiance_10p)
    plt.yscale("log")
    plt.xlabel(r"MC$_{[ij, kl]} $", fontsize=12)
    plt.ylabel("Probability density", fontsize=10)

    plt.legend()

    plt.subplot(323)
    plt.title("Intra-modularity")
    plt.hist(
        (
            [
                mc_val[
                    mc_bottom10_idx * mod[:, None] * (mc_mod_idx > 0)[None, :]
                ].ravel()
                for mod in setb
            ]
        ),
        histtype="step",
        bins=70,
        density=True,
        # label=('top 10', 't10 4-plets', 't10 3-plets')#, 't10 4-plets intramod', 't10 3-plets intramod'),
        label=aux_label,
    )
    # plt.boxplot(mc_allegiance_10p)
    plt.yscale("log")
    plt.xlabel(r"MC$_{[ij, kl]} $", fontsize=12)
    plt.ylabel("Probability density", fontsize=10)

    plt.legend()

    plt.subplot(324)
    plt.title("Intra-modularity 4plets")
    plt.hist(
        (
            [
                mc_val[
                    mc_bottom10_idx
                    * mod[:, None]
                    * (mc_mod_idx > 0)[None, :]
                    * (mc_nplets_index == 0)[None, :]
                ].ravel()
                for mod in setb
            ]
        ),
        histtype="step",
        bins=70,
        density=True,
        # label=('top 10', 't10 4-plets', 't10 3-plets')#, 't10 4-plets intramod', 't10 3-plets intramod'),
        label=aux_label,
    )
    # plt.boxplot(mc_allegiance_10p)
    plt.yscale("log")
    plt.xlabel(r"MC$_{[ij, kl]} $", fontsize=12)
    plt.ylabel("Probability density", fontsize=10)

    plt.legend()
    plt.subplot(325)
    plt.title("Intra-modularity 3plets")
    plt.hist(
        (
            [
                mc_val[
                    mc_bottom10_idx
                    * mod[:, None]
                    * (mc_mod_idx > 0)[None, :]
                    * (mc_nplets_index > 0)[None, :]
                ].ravel()
                for mod in setb
            ]
        ),
        histtype="step",
        bins=70,
        density=True,
        # label=('top 10', 't10 4-plets', 't10 3-plets')#, 't10 4-plets intramod', 't10 3-plets intramod'),
        label=aux_label,
    )
    # plt.boxplot(mc_allegiance_10p)
    plt.yscale("log")
    plt.xlabel(r"MC$_{[ij, kl]} $", fontsize=12)
    plt.ylabel("Probability density", fontsize=10)

    plt.subplot(326)
    plt.title("3plets")
    plt.hist(
        (
            [
                mc_val[
                    mc_bottom10_idx * mod[:, None] * (mc_nplets_index > 0)[None, :]
                ].ravel()
                for mod in setb
            ]
        ),
        histtype="step",
        bins=70,
        density=True,
        # label=('top 10', 't10 4-plets', 't10 3-plets')#, 't10 4-plets intramod', 't10 3-plets intramod'),
        label=aux_label,
    )
    # plt.boxplot(mc_allegiance_10p)
    plt.yscale("log")
    plt.xlabel(r"MC$_{[ij, kl]} $", fontsize=12)
    plt.ylabel("Probability density", fontsize=10)

    plt.legend()

    # plt.subplot(313)
    # plt.hist((mc_val[mc_top10_idx].ravel(),
    #           mc_val[mc_bottom10_idx*(mc_nplets_index==0)[None,:] *(mc_mod_idx==0)[None,:] ],
    #           mc_val[mc_bottom10_idx*(mc_nplets_index>0)[None,:] * (mc_mod_idx==0)[None,:] ],

    #           mc_val[mc_bottom10_idx*(mc_nplets_index==0)[None,:] *(mc_mod_idx>0)[None,:] ],
    #           mc_val[mc_bottom10_idx*(mc_nplets_index>0)[None,:] * (mc_mod_idx>0)[None,:] ]),

    #          histtype='step',
    #           bins=70,
    #          density=True,
    #          label=aux_label,
    #          # label=('top 10', 't10 4-plets intermod', 't10 3-plets intermod', 't10 4-plets intramod', 't10 3-plets intramod'),
    #          )
    # # plt.boxplot(mc_allegiance_10p)
    # plt.yscale('log')
    # plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)
    # plt.ylabel('Probability density', fontsize=10)
    # plt.ylim(10e-4,10e0)
    # plt.legend()
    # plt.subplot(6,7,1+tr+1)
    plt.tight_layout()
    if save_fig == True:
        plt.savefig(
            motif_fig_dir
            / f"top10_trimer_intermodule_{aux_label[0]}_{aux_label[2]}.png"
        )
# %%

from statsmodels.distributions.empirical_distribution import ECDF


# --- Define ECDF Function ---
def fun_ecdf(data, side="right"):
    ecdf = ECDF(data)
    x = np.sort(data)
    if side == "right":
        y = 1 - ecdf(x)
    elif side == "left":
        y = ecdf(x)
    return x, y


# %% ecdf top

for idx, setb in enumerate(mask_groups):
    # idx=0
    # setb = mask_groups[0]
    aux_label = label_variables[idx]
    num_group = int(len(aux_label) / 2)

    plt.figure(76 + idx, figsize=(13, 10))
    plt.clf()

    # --- Plot ---

    plt.subplot(331)
    plt.title("Top 10")
    aux_x = [fun_ecdf(mc_val[mc_top10_idx * mod[:, None]].ravel()) for mod in setb]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label="%s" % aux_label[i])
    plt.ylabel(" ECDF", fontsize=10)
    plt.yscale("log")
    plt.xlabel(r"MC$_{[ij, kl]} $", fontsize=12)
    aux_x = None
    # plt.legend()

    plt.subplot(332)
    plt.title("Inter-modularity")
    aux_x = [
        fun_ecdf(
            mc_val[mc_top10_idx * mod[:, None] * (mc_mod_idx == 0)[None, :]].ravel()
        )
        for mod in setb
    ]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label="%s" % aux_label[i])
    plt.ylabel(" ECDF", fontsize=10)
    plt.yscale("log")
    plt.xlabel(r"MC$_{[ij, kl]} $", fontsize=12)

    plt.subplot(333)
    plt.title("Intra-modularity")
    aux_x = [
        fun_ecdf(
            mc_val[mc_top10_idx * mod[:, None] * (mc_mod_idx > 0)[None, :]].ravel()
        )
        for mod in setb
    ]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label="%s" % aux_label[i])
    plt.ylabel(" ECDF", fontsize=10)
    plt.yscale("log")
    plt.xlabel(r"MC$_{[ij, kl]} $", fontsize=12)

    plt.subplot(334)
    plt.title("Tetramers")
    aux_x = [
        fun_ecdf(
            mc_val[
                mc_top10_idx * mod[:, None] * (mc_nplets_index == 0)[None, :]
            ].ravel()
        )
        for mod in setb
    ]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label="%s" % aux_label[i])
    plt.ylabel(" ECDF", fontsize=10)
    plt.yscale("log")
    plt.xlabel(r"MC$_{[ij, kl]} $", fontsize=12)

    plt.subplot(335)
    plt.title("Inter-modularity tetramers")
    aux_x = [
        fun_ecdf(
            mc_val[
                mc_top10_idx
                * mod[:, None]
                * (mc_mod_idx > 0)[None, :]
                * (mc_nplets_index == 0)[None, :]
            ].ravel()
        )
        for mod in setb
    ]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label="%s" % aux_label[i])
    plt.ylabel(" ECDF", fontsize=10)
    plt.yscale("log")
    plt.xlabel(r"MC$_{[ij, kl]} $", fontsize=12)

    plt.subplot(336)
    plt.title("Intra-modularity tetramers")
    aux_x = [
        fun_ecdf(
            mc_val[
                mc_top10_idx
                * mod[:, None]
                * (mc_mod_idx == 0)[None, :]
                * (mc_nplets_index == 0)[None, :]
            ].ravel()
        )
        for mod in setb
    ]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label="%s" % aux_label[i])
    plt.ylabel(" ECDF", fontsize=10)
    plt.yscale("log")
    plt.xlabel(r"MC$_{[ij, kl]} $", fontsize=12)

    plt.subplot(337)
    plt.title("Trimers")
    aux_x = [
        fun_ecdf(
            mc_val[mc_top10_idx * mod[:, None] * (mc_nplets_index > 0)[None, :]].ravel()
        )
        for mod in setb
    ]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label="%s" % aux_label[i])
    plt.ylabel(" ECDF", fontsize=10)
    plt.yscale("log")
    plt.xlabel(r"MC$_{[ij, kl]} $", fontsize=12)

    plt.subplot(338)
    plt.title("Inter-modularity Trimers")
    aux_x = [
        fun_ecdf(
            mc_val[
                mc_top10_idx
                * mod[:, None]
                * (mc_mod_idx == 0)[None, :]
                * (mc_nplets_index > 0)[None, :]
            ].ravel()
        )
        for mod in setb
    ]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label="%s" % aux_label[i])
    plt.ylabel(" ECDF", fontsize=10)
    plt.yscale("log")
    plt.xlabel(r"MC$_{[ij, kl]} $", fontsize=12)

    plt.subplot(339)
    plt.title("Intra-modularity Trimers")
    aux_x = [
        fun_ecdf(
            mc_val[
                mc_top10_idx
                * mod[:, None]
                * (mc_mod_idx > 0)[None, :]
                * (mc_nplets_index > 0)[None, :]
            ].ravel()
        )
        for mod in setb
    ]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label="%s" % aux_label[i])
    plt.ylabel(" ECDF", fontsize=10)
    plt.yscale("log")
    plt.xlabel(r"MC$_{[ij, kl]} $", fontsize=12)

    plt.legend()
    plt.tight_layout()
    # if save_fig==True:
    #     plt.savefig(path_figures + 'trimers/ecdf_top10_trimer_intermodule_%s_%s.png'%(aux_label[0],aux_label[2]))


# %% ecdf bottom


for idx, setb in enumerate(mask_groups):
    # idx=0
    # setb = mask_groups[0]
    aux_label = label_variables[idx]
    num_group = int(len(aux_label) / 2)

    plt.figure(86 + idx, figsize=(13, 10))
    plt.clf()

    # --- Plot ---

    plt.subplot(331)
    plt.title("bottom 10")
    aux_x = [
        fun_ecdf(mc_val[mc_bottom10_idx * mod[:, None]].ravel(), side="left")
        for mod in setb
    ]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label="%s" % aux_label[i])
    plt.ylabel(" ECDF", fontsize=10)
    plt.yscale("log")
    plt.xlabel(r"MC$_{[ij, kl]} $", fontsize=12)

    plt.subplot(332)
    plt.title("Inter-modularity")
    aux_x = [
        fun_ecdf(
            mc_val[mc_bottom10_idx * mod[:, None] * (mc_mod_idx == 0)[None, :]].ravel(),
            side="left",
        )
        for mod in setb
    ]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label="%s" % aux_label[i])
    plt.ylabel(" ECDF", fontsize=10)
    # plt.yscale('log')
    plt.xlabel(r"MC$_{[ij, kl]} $", fontsize=12)

    plt.subplot(333)
    plt.title("Intra-modularity")
    aux_x = [
        fun_ecdf(
            mc_val[mc_bottom10_idx * mod[:, None] * (mc_mod_idx > 0)[None, :]].ravel(),
            side="left",
        )
        for mod in setb
    ]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label="%s" % aux_label[i])
    plt.ylabel(" ECDF", fontsize=10)
    # plt.yscale('log')
    plt.xlabel(r"MC$_{[ij, kl]} $", fontsize=12)

    plt.subplot(334)
    plt.title("Tetramers")
    aux_x = [
        fun_ecdf(
            mc_val[
                mc_bottom10_idx * mod[:, None] * (mc_nplets_index == 0)[None, :]
            ].ravel(),
            side="left",
        )
        for mod in setb
    ]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label="%s" % aux_label[i])
    plt.ylabel(" ECDF", fontsize=10)
    # plt.yscale('log')
    plt.xlabel(r"MC$_{[ij, kl]} $", fontsize=12)

    plt.subplot(335)
    plt.title("Inter-modularity tetramers")
    aux_x = [
        fun_ecdf(
            mc_val[
                mc_bottom10_idx
                * mod[:, None]
                * (mc_mod_idx > 0)[None, :]
                * (mc_nplets_index == 0)[None, :]
            ].ravel(),
            side="left",
        )
        for mod in setb
    ]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label="%s" % aux_label[i])
    plt.ylabel(" ECDF", fontsize=10)
    # plt.yscale('log')
    plt.xlabel(r"MC$_{[ij, kl]} $", fontsize=12)

    plt.subplot(336)
    plt.title("Intra-modularity tetramers")
    aux_x = [
        fun_ecdf(
            mc_val[
                mc_bottom10_idx
                * mod[:, None]
                * (mc_mod_idx == 0)[None, :]
                * (mc_nplets_index == 0)[None, :]
            ].ravel(),
            side="left",
        )
        for mod in setb
    ]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label="%s" % aux_label[i])
    plt.ylabel(" ECDF", fontsize=10)
    # plt.yscale('log')
    plt.xlabel(r"MC$_{[ij, kl]} $", fontsize=12)

    plt.subplot(337)
    plt.title("Trimers")
    aux_x = [
        fun_ecdf(
            mc_val[
                mc_bottom10_idx * mod[:, None] * (mc_nplets_index > 0)[None, :]
            ].ravel(),
            side="left",
        )
        for mod in setb
    ]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label="%s" % aux_label[i])
    plt.ylabel(" ECDF", fontsize=10)
    # plt.yscale('log')
    plt.xlabel(r"MC$_{[ij, kl]} $", fontsize=12)

    plt.subplot(338)
    plt.title("Inter-modularity Trimers")
    aux_x = [
        fun_ecdf(
            mc_val[
                mc_bottom10_idx
                * mod[:, None]
                * (mc_mod_idx == 0)[None, :]
                * (mc_nplets_index > 0)[None, :]
            ].ravel(),
            side="left",
        )
        for mod in setb
    ]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label="%s" % aux_label[i])
    plt.ylabel(" ECDF", fontsize=10)
    # plt.yscale('log')
    plt.xlabel(r"MC$_{[ij, kl]} $", fontsize=12)

    plt.subplot(339)
    plt.title("Intra-modularity Trimers")
    aux_x = [
        fun_ecdf(
            mc_val[
                mc_bottom10_idx
                * mod[:, None]
                * (mc_mod_idx > 0)[None, :]
                * (mc_nplets_index > 0)[None, :]
            ].ravel(),
            side="left",
        )
        for mod in setb
    ]
    for i, (x, y) in enumerate(aux_x):
        plt.plot(x, y, label="%s" % aux_label[i])
    plt.ylabel(" ECDF", fontsize=10)
    # plt.yscale('log')
    plt.xlabel(r"MC$_{[ij, kl]} $", fontsize=12)

    plt.legend()
    plt.tight_layout()
    # if save_fig==True:
    #     plt.savefig(path_figures + 'trimers/ecdf_bottom10_trimer_intermodule_%s_%s.png'%(aux_label[0],aux_label[2]))

# %%
import seaborn as sns

# original, subset = mc_val[mc_top10_idx].ravel(), mc_val[mc_top10_idx*(mc_mod_idx>0)[None,:]].ravel()
original, subset = (
    mc_val[mc_top10_idx].ravel(),
    mc_val[mc_top10_idx * (mc_nplets_index > 0)[None, :]].ravel(),
)

plt.figure(91)
plt.clf()
sns.boxplot(data=[original, subset])
plt.xticks([0, 1], ["Original", "Subset"])
plt.title("Boxplot")

plt.tight_layout()
plt.show()

from scipy.stats import ks_2samp

ks_stat, ks_p = ks_2samp(original, subset)
print(f"KS test: statistic={ks_stat:.3f}, p-value={ks_p:.3f}")

# # Cohen's d
# cohen_d = (original.mean() - subset.mean()) / ((original.std()**2 + subset.std()**2)/2)**0.5
# print(f"Cohen's d: {cohen_d:.3f}")
# #%%

# import scipy.stats as stats

# # Make sure both datasets are sorted and of the same length
# # If not, you can interpolate to match sizes
# min_len = min(len(original), len(subset))
# orig_sorted = np.sort(original)
# sub_sorted = np.sort(subset)

# # Interpolate if needed
# if len(original) > len(subset):
#     orig_interp = np.percentile(orig_sorted, np.linspace(0, 100, min_len))
#     sub_interp = sub_sorted
# elif len(subset) > len(original):
#     orig_interp = orig_sorted
#     sub_interp = np.percentile(sub_sorted, np.linspace(0, 100, min_len))
# else:
#     orig_interp = orig_sorted
#     sub_interp = sub_sorted

# # Q-Q Plot
# plt.figure(8890,figsize=(6,6))
# # plt.clf()
# plt.plot(orig_interp, sub_interp, 'o')
# min_val = min(orig_interp.min(), sub_interp.min())
# max_val = max(orig_interp.max(), sub_interp.max())
# plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
# plt.xlabel('Original Quantiles')
# plt.ylabel('Subset Quantiles')
# plt.title('Q-Q Plot: Original vs Subset (Raw Values)')
# plt.legend()
# plt.grid(True)
# plt.show()

# %%

# ...existing code...

_mask = mask_groups[0]
tr = 1

animal_mask = _mask[0]  # boolean mask over animals OR indices selecting animals

# mc_reg_idx is typically (4, n_edges) or (n_edges, 4) → reduce to (n_edges,)
if mc_reg_idx.ndim != 2:
    raise ValueError(f"Expected mc_reg_idx to be 2D, got shape={mc_reg_idx.shape}")

if mc_reg_idx.shape[0] == 4:
    reg_edge_mask = np.any(mc_reg_idx == (tr + 1), axis=0)
elif mc_reg_idx.shape[1] == 4:
    reg_edge_mask = np.any(mc_reg_idx == (tr + 1), axis=1)
else:
    raise ValueError(f"Unexpected mc_reg_idx shape={mc_reg_idx.shape} (expected one dim == 4)")

edge_mask = (mc_nplets_index > 0) & (mc_mod_idx > 0) & reg_edge_mask

test = mc_val[animal_mask][:, edge_mask]
print(np.shape(test))

# ...existing code...
# %%
# for idx, setb in enumerate(mask_groups):
setb = mask_groups[1]
idx = 1
plt.figure(180 + idx, figsize=(15, 12))
plt.clf()
aux_label = label_variables[idx]
for tr in range(regions):
    plt.subplot(6, 7, 1 + tr)
    plt.title(anat_labels[tr], fontsize=10)
    # plt.hist(([trimer_per_region[tr,mod].ravel() for mod in setb]),
    # plt.hist(([mc_val[mod][:, (mc_nplets_index>0) * (mc_mod_idx==0)].ravel()     for mod in setb]),
    plt.bar(
        aux_label,
        np.array(
            [
                np.nanpercentile(
                    mc_val[mod][
                        :,
                        (mc_nplets_index > 0)
                        * (mc_mod_idx > 0)
                        * (mc_nplets_index == tr + 1),
                    ].ravel(),
                    90,
                )
                for mod in setb
            ]
        ).ravel(),
        # np.array([np.nanpercentile(mc_val[mod][:, (mc_mod_idx>0) * (mc_nplets_index==tr+1)].ravel(),90) for mod in setb]).ravel()).flatten(),
    )
    # plt.yscale('log')
    # plt.ylim(10e-4,10e0)
    plt.yticks([])
    if tr + 1 > (5 * 7):
        plt.xticks([-0.8, 0, 0.8], fontsize=13)
    else:
        plt.xticks([])
# plt.legend()
plt.subplot(6, 7, 1 + tr + 1)
plt.title("trimer per region")
plt.xticks([])
plt.yticks([])
plt.tight_layout()


# %%
# aux_label= label_variables[idx]
# num_group = int(len(aux_label)/2)

# plt.figure(76+idx, figsize=(13,10))
# plt.clf()

# # --- Plot ---

# plt.subplot(331)
# plt.title('Top 10')
# aux_x = [fun_ecdf(mc_val[mc_top10_idx * mod[:, None]].ravel()) for mod in setb]
# for i, (x, y) in enumerate(aux_x):
#     plt.plot(x, y, label='%s'%aux_label[i])
# plt.ylabel(' ECDF', fontsize=10)
# plt.yscale('log')
# plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)
# aux_x=None
# # plt.legend()

# plt.subplot(332)
# plt.title('Inter-modularity')
# aux_x = [fun_ecdf(mc_val[mc_top10_idx * mod[:, None] * (mc_mod_idx==0)[None,:] ].ravel()) for mod in setb]
# for i, (x, y) in enumerate(aux_x):
#     plt.plot(x, y, label='%s'%aux_label[i])
# plt.ylabel(' ECDF', fontsize=10)
# plt.yscale('log')
# plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)


# plt.subplot(333)
# plt.title('Intra-modularity')
# aux_x = [fun_ecdf(mc_val[mc_top10_idx * mod[:, None] * (mc_mod_idx>0)[None,:] ].ravel()) for mod in setb]
# for i, (x, y) in enumerate(aux_x):
#     plt.plot(x, y, label='%s'%aux_label[i])
# plt.ylabel(' ECDF', fontsize=10)
# plt.yscale('log')
# plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)

# plt.subplot(334)
# plt.title('Tetramers')
# aux_x = [fun_ecdf(mc_val[mc_top10_idx * mod[:, None] * (mc_nplets_index==0)[None,:]].ravel()) for mod in setb]
# for i, (x, y) in enumerate(aux_x):
#     plt.plot(x, y, label='%s'%aux_label[i])
# plt.ylabel(' ECDF', fontsize=10)
# plt.yscale('log')
# plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)

# plt.subplot(335)
# plt.title('Inter-modularity tetramers')
# aux_x = [fun_ecdf(mc_val[mc_top10_idx * mod[:, None] * (mc_mod_idx>0)[None,:] * (mc_nplets_index==0)[None,:]].ravel()) for mod in setb]
# for i, (x, y) in enumerate(aux_x):
#     plt.plot(x, y, label='%s'%aux_label[i])
# plt.ylabel(' ECDF', fontsize=10)
# plt.yscale('log')
# plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)

# plt.subplot(336)
# plt.title('Intra-modularity tetramers')
# aux_x = [fun_ecdf(mc_val[mc_top10_idx * mod[:, None] * (mc_mod_idx==0)[None,:] * (mc_nplets_index==0)[None,:]].ravel()) for mod in setb]
# for i, (x, y) in enumerate(aux_x):
#     plt.plot(x, y, label='%s'%aux_label[i])
# plt.ylabel(' ECDF', fontsize=10)
# plt.yscale('log')
# plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)

# plt.subplot(337)
# plt.title('Trimers')
# aux_x = [fun_ecdf(mc_val[mc_top10_idx * mod[:, None] * (mc_nplets_index>0)[None,:]].ravel()) for mod in setb]
# for i, (x, y) in enumerate(aux_x):
#     plt.plot(x, y, label='%s'%aux_label[i])
# plt.ylabel(' ECDF', fontsize=10)
# plt.yscale('log')
# plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)

# plt.subplot(338)
# plt.title('Inter-modularity Trimers')
# aux_x = [fun_ecdf(mc_val[mc_top10_idx * mod[:, None] * (mc_mod_idx==0)[None,:] * (mc_nplets_index>0)[None,:]].ravel()) for mod in setb]
# for i, (x, y) in enumerate(aux_x):
#     plt.plot(x, y, label='%s'%aux_label[i])
# plt.ylabel(' ECDF', fontsize=10)
# plt.yscale('log')
# plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)

# plt.subplot(339)
# plt.title('Intra-modularity Trimers')
# aux_x = [fun_ecdf(mc_val[mc_top10_idx * mod[:, None] * (mc_mod_idx>0)[None,:] * (mc_nplets_index>0)[None,:]].ravel()) for mod in setb]
# for i, (x, y) in enumerate(aux_x):
#     plt.plot(x, y, label='%s'%aux_label[i])
# plt.ylabel(' ECDF', fontsize=10)
# plt.yscale('log')
# plt.xlabel(r'MC$_{[ij, kl]} $', fontsize=12)


# plt.legend()
# plt.tight_layout()
# =============================================================================
# Index of regions, modularity and nplets
# =============================================================================
# mc indices tril
# fc_indx = np.array(np.tril_indices(regions,k=-1)).T
# mc_idx = np.array(np.tril_indices(fc_indx.shape[0], k=-1)).T
# aux_indentity_region_mc =fc_indx[mc_idx]
# mc_reg_idx = np.transpose(np.reshape(aux_indentity_region_mc,
#                           (aux_indentity_region_mc.shape[0], 4))
#                           ,(1,0)) #identity of the regios in the MC_i

# #mc values tril
# mc_idx = np.transpose(mc_idx,(1,0))
# mc_val  = mc_allegiance[:,mc_idx[0],mc_idx[1]]

# #mc modules indices and mask
# mc_mod_idx = (mc_modules_mask[mc_idx[0], mc_idx[1]])
# mc_mod_idx = mc_mod_idx.astype(int)

## %% Trimers

# =============================================================================intramodules_idx
# trimer - values and identity
# trimer_index, trimer_reg, trimer_reg_apex, trimer_values
# =============================================================================

# %%
# --------------------------------------------------
# Bottom-10% threshold
# --------------------------------------------------
mc_all = mc_val.ravel()
thr_10 = np.percentile(mc_all, 10)

bottom_mask = mc_val <= thr_10    # shape (n_animals, n_edges)

intra_mask  = (mc_mod_idx > 0)[None, :]
inter_mask  = (mc_mod_idx == 0)[None, :]

plets3_mask = (mc_nplets_index > 0)[None, :]
plets4_mask = (mc_nplets_index == 0)[None, :]

# Baseline: all bottom-10% values
x_base = mc_val[bottom_mask]

bottom_mask_aux = bottom_mask[0]  # for a single animal

x_3_intra = mc_val[bottom_mask_aux & plets3_mask & intra_mask]
#%%
for idx, setb in enumerate(mask_groups):
    # idx=0
    # setb = mask_groups[0]
    aux_label = label_variables[idx]
    num_group = int(len(aux_label) / 2)
    print(aux_label, idx, num_group, setb)


counts, edges = np.histogram(x_base, bins=bins)

bottom_mask_aux = bottom_mask[setb[0]]  # for a single animal
mc_value_subset = mc_val[bottom_mask_aux & intra_mask]



plt.figure(200+idx, figsize=(7,5))
plt.clf()
plt.title(f"Bottom-10% MC distribution - {aux_label[0]} vs {aux_label[2]}")
plt.plot(edges[:-1], counts/counts.sum(), label="Baseline (all bottom-10%)", marker="o")
for i, mod in enumerate(setb):
    bottom_mask_aux = bottom_mask[setb[i]]  # for a single animal
    x_subset = mc_val[bottom_mask_aux & mod[:, None] & intra_mask]
    counts_subset, _ = np.histogram(x_subset, bins=bins)
    plt.plot(edges[:-1], counts_subset/counts_subset.sum(), label=f"Subset: {aux_label[i]}", marker="o")
plt.xlabel("MC values")
plt.ylabel("Probability density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
#%%

# # Combinatory subsets
# x_3_intra = mc_val[bottom_mask & plets3_mask & intra_mask]
# x_4_intra = mc_val[bottom_mask & plets4_mask & intra_mask]

# x_3_inter = mc_val[bottom_mask & plets3_mask & inter_mask]
# x_4_inter = mc_val[bottom_mask & plets4_mask & inter_mask]

# def shift_function(x_ref, x_cmp, pcts):
#     """
#     Shift function: quantile(x_cmp) - quantile(x_ref)
#     """
#     q_ref = np.percentile(x_ref, pcts)
#     q_cmp = np.percentile(x_cmp, pcts)
#     return q_cmp - q_ref

# PCTS = np.linspace(5, 95, 19)   # within bottom-10% distribution

# shifts = {
#     "3-plets intra": shift_function(x_base, x_3_intra, PCTS),
#     "4-plets intra": shift_function(x_base, x_4_intra, PCTS),
#     "3-plets inter": shift_function(x_base, x_3_inter, PCTS),
#     "4-plets inter": shift_function(x_base, x_4_inter, PCTS),
# }
#%%



# fig, ax = plt.subplots(figsize=(7, 4))

# ax.axhline(0, color="gray", linestyle="--", lw=1)

# for label, shift in shifts.items():
#     # ax.plot(PCTS,   , marker="o", label=label)
#     ax.plot(PCTS, shift, marker="o", label=label)
# ax.set_xlabel("Percentile (within bottom-10% MC tail)")
# ax.set_ylabel("Δ MC (relative to bottom-10% baseline)")
# ax.set_title("Shift functions within the bottom-10% MC tail")

# ax.grid(True, alpha=0.3)
# ax.legend()

# ax.text(
#     0.02, 0.02,
#     "Negative shift = stronger anti-correlation\nPositive shift = weaker anti-correlation",
#     transform=ax.transAxes,
#     fontsize=9,
# )

# fig.tight_layout()
#%%
animal_idx = setb[i]            # boolean mask, shape (126,)
mc_group = mc_val[animal_idx]   # shape (n_group_animals, 335790)

edge_mask = (
    (mc_group <= thr_10) &      # bottom-10% PER ANIMAL
    (mc_mod_idx > 0)[None, :] &  # intra-module
    (mc_nplets_index == 0)[None, :] # tetramers
)

# %%
x_subset = mc_group[edge_mask]

# %%
plt.figure(200 + idx, figsize=(7, 5))
plt.clf()

plt.title(f"Bottom-10% MC distribution - {aux_label[0]} vs {aux_label[2]}")

# Baseline (all animals, bottom-10%)

bins= np.linspace(-0.7, 0, 100)
counts, edges = np.histogram(x_base, bins=bins)
plt.plot(
    edges[:-1],
    counts / counts.sum(),
    label="Baseline (all bottom-10%)",
    # marker="o",
)

# Group-wise curves
for i, animal_mask in enumerate(setb):
    mc_group = mc_val[animal_mask]  # (n_group_animals, n_edges)

    edge_mask = (
        (mc_group <= thr_10) &
        (mc_mod_idx == 0)[None, :] &   # intra-module
        (mc_nplets_index > 0)[None, :] # tetramers
    )

    x_subset = mc_group[edge_mask]

    counts_subset, _ = np.histogram(x_subset, bins=bins)

    plt.plot(
        edges[:-1],
        counts_subset / counts_subset.sum(),
        label=f"{aux_label[i]}",
        # marker="o",
    )
plt.yscale("log")
plt.xlabel("MC values")
plt.ylabel("Probability density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

#%%

plt.figure(200 + idx, figsize=(7, 5))
plt.clf()

plt.title(f"Bottom-10% MC distribution - {aux_label[0]} vs {aux_label[2]}")

# Baseline (all animals, bottom-10%)

bins= np.linspace(-0.7, 0, 100)
counts, edges = np.histogram(x_base, bins=bins)

x_ref = counts / counts.sum()
# Group-wise curves
for i, animal_mask in enumerate(setb):
    mc_group = mc_val[animal_mask]  # (n_group_animals, n_edges)

    edge_mask = (
        (mc_group <= thr_10) &
        (mc_mod_idx == 0)[None, :] &   # intra-module
        (mc_nplets_index > 0)[None, :] # tetramers
    )

    x_subset = mc_group[edge_mask]

    counts_subset, _ = np.histogram(x_subset, bins=bins)

    plt.plot(
        edges[:-1],
        x_ref - counts_subset / counts_subset.sum(),
        label=f"{aux_label[i]}",
        # marker="o",
    )
# plt.yscale("log")
plt.xlabel("MC values")
plt.ylabel("Probability density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

#%%

# %%
# ============================================================
# MC tail comparison – three visual metrics (REAL DATA)
# Self-contained, clean-notebook safe
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from shared_code.fun_loaddata import load_timeseries_bundle
from shared_code.fun_paths import get_paths
from shared_code.fun_utils import set_figure_params

# ============================================================
# CONFIG
# ============================================================

REF_GROUP = "wt 2m"
REF_BINS = np.linspace(-0.8, 1.0, 100)

GROUP_COLORS = {
    "wt 2m": "#1f77b4",
    "wt 4m": "#ff7f0e",
    "dkl 2m": "#2ca02c",
    "dkl 4m": "#d62728",
}

plt.rcParams.update({
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.labelsize": 13,
    "axes.titlesize": 12,
})

save_fig = set_figure_params(False)

# ============================================================
# LOAD DATA
# ============================================================

paths = get_paths(
    dataset_name="ines_abdallah",
    timecourse_folder="Timecourses_updated_03052024",
    cognitive_data_file="ROIs.xlsx",
    anat_labels_file="41_Allen.txt",
)

bundle = load_timeseries_bundle(
    paths["preprocessed"] / "ts_and_meta_2m4m.npz",
    paths["preprocessed"] / "grouping_data_oip.pkl",
)

mc = np.load(
    paths["mc_mod"]
    / f"mc_allegiance_ref(runs=wt2m_gammaval=1000)=100_"
      f"lag=1_windowsize=7_animals={bundle.n_animals}_regions={bundle.n_regions}.npz",
    allow_pickle=True,
)

mc_val = mc["mc_val_tril"]   # (animals, edges)

#%%
# ============================================================
# HELPERS
# ============================================================

def normalize_label(label):
    return label.strip().lower().replace("dki", "dkl")

def true_indices_nested(mask_groups):
    return [[np.flatnonzero(arr) for arr in group] for group in mask_groups]

def prob_hist(x, bins):
    h, _ = np.histogram(x, bins=bins)
    return h / h.sum() if h.sum() > 0 else np.zeros(len(bins) - 1)

# --- three comparison metrics --------------------------------

def delta_p_weighted(p_ref, p_group):
    support = p_ref + p_group
    out = np.zeros_like(p_ref)
    valid = support > 0
    out[valid] = (p_ref[valid] - p_group[valid]) / support[valid]
    return out * support

def delta_p_excess(p_ref, p_group):
    return p_ref - p_group

def cumulative_tail_excess(p_ref, p_group, direction="right"):
    diff = p_ref - p_group
    if direction == "right":
        return np.cumsum(diff[::-1])[::-1]
    else:
        return np.cumsum(diff)
#%%
# ============================================================
# BUILD GROUP HISTOGRAMS (REAL DATA)
# ============================================================

indices_mask = true_indices_nested(bundle.mask_groups)[2]
labels_raw = bundle.label_variables[2]
labels = [normalize_label(l) for l in labels_raw]

p_groups = {}
for inds, label in zip(indices_mask, labels):
    mc_subset = mc_val[inds, :].ravel()
    p_groups[label] = prob_hist(mc_subset, REF_BINS)

assert REF_GROUP in p_groups, f"Reference group '{REF_GROUP}' not found"

# ============================================================
# PLOT: THREE OPTIONS SIDE-BY-SIDE
# ============================================================

x = REF_BINS[:-1]
p_ref = p_groups[REF_GROUP]

fig, axes = plt.subplots(1, 1, figsize=(8, 7), sharex=True)


# ---- 1) Weighted ΔP ----------------------------------------
ax = axes
ax.axhline(0, color="black", lw=1)
for label, p in p_groups.items():
    if label == REF_GROUP:
        continue
    ax.plot(
        x,
        delta_p_weighted(p_ref, p),
        color=GROUP_COLORS[label],
        label=label,
    )
ax.set_ylabel("Weighted ΔP(2m wt-group)")
ax.set_title("Weighted ΔP (bin-wise, support-aware)")
ax.legend()

plt.show()

#%%
# ---- 2) Signed excess probability --------------------------
fig, axes = plt.subplots(1, 1, figsize=(8, 7), sharex=True)
ax = axes
ax.axhline(0, color="black", lw=1)
for label, p in p_groups.items():
    if label == REF_GROUP:
        continue
    ax.plot(
        x,
        delta_p_excess(p_ref, p),
        color=GROUP_COLORS[label],
    )
ax.set_ylabel("P(wt2m) − P(group)")
ax.set_title("Signed excess probability")
plt.show()
#%%
# ---- 3) Cumulative right-tail excess -----------------------
fig, axes = plt.subplots(1, 1, figsize=(8, 7), sharex=True)
ax = axes
ax.axhline(0, color="black", lw=1)
for label, p in p_groups.items():
    if label == REF_GROUP:
        continue
    ax.plot(
        x,
        cumulative_tail_excess(p_ref, p, direction="right"),
        color=GROUP_COLORS[label],
        label=label,
    )
ax.set_ylabel("Cumulative excess")
ax.set_xlabel("MC values")
ax.set_title("Cumulative excess (right tail → inward)")
ax.legend()
fig.tight_layout()
plt.show()


