#!/usr/bin/env python3
"""
Created on Mon Sep 23 13:26:30 2024

@author: samy
"""
# %%
from pathlib import Path

# from fun_dfcspeed import *
import matplotlib.pyplot as plt
import numpy as np

from shared_code.fun_loaddata import load_timeseries_bundle

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

# %%Load and plot parameters

bins_parameter = 200

from shared_code.fun_paths import get_paths
from shared_code.fun_utils import set_figure_params

# ========================== Figure parameters ================================
# Set figure parameters globally
save_fig = set_figure_params(True)

# =================== Paths and folders =======================================
timecourse_folder = "Timecourses_updated_03052024"
# paths = get_paths()
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
regions = bundle.n_regions
mask_groups = bundle.mask_groups
label_variables = bundle.label_variables

if mask_groups is None or label_variables is None:
    raise ValueError(
        "Grouping data missing from the preprocessed bundle; expected masks and labels."
    )

dataset_name = paths["results"].name
# report_root = Path("reports/metaconnectivity") / dataset_name
report_root = paths['root']
mc_mod_dir = paths['mc_mod']
fig_root = paths['f_mod']
allegiance_fig_dir = paths['f_allegiance']
modularity_fig_dir = paths['f_mod']
for directory in (mc_mod_dir, allegiance_fig_dir, modularity_fig_dir):
    directory.mkdir(parents=True, exist_ok=True)
# ========================== Load data =========================
# cog_data_filtered = pd.read_csv(paths['sorted'] / 'cog_data_sorted_2m4m.csv')

# # ts=data_ts['ts']
# data_ts = np.load(paths['sorted'] / 'ts_and_meta_2m4m.npz')
# n_animals   = int(data_ts['n_animals'])
# total_tp = data_ts['total_tp']
# regions = data_ts['regions']
# is_2month_old = data_ts['is_2month_old']
# anat_labels= data_ts['anat_labels']

# results_path = paths['results'] / "grouping_data_oip.pkl"
# with results_path.open("rb") as f:
#     mask_groups, label_variables = pickle.load(f)
# %% Metaconnectivity computing
# =============================================================================
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
label_ref = "Good2m"  # The label of the reference matrix
# label_ref = 'wt2M_recurrecy' #The label of the reference matrix

tau_array = np.append(np.arange(0, tau), tau)
lentau = len(tau_array)

time_window_min, time_window_max, time_window_step = window_parameter
time_window_range = np.arange(time_window_min, time_window_max + 1, time_window_step)

mc_data_filename = f"mc_allegiance_ref(runs={label_ref}_gammaval={n_runs_allegiance})={gamma_pt_allegiance}_lag={lag}_windowsize={window_size}_animals={n_animals}_regions={regions}.npz".replace(
    " ", ""
)
# %% Load data metaconnectivity, and modularity
# =============================================================================
# Load data
# =============================================================================
data_mc_mod_filename = mc_mod_dir / mc_data_filename
data_mc_mod = np.load(data_mc_mod_filename, allow_pickle=True)

mc_allegiance = data_mc_mod["mc"]
mc_ref_allegiance_communities = data_mc_mod["mc_ref_allegiance_communities"]
# mc_ref_allegiance_sort   = data_mc_mod['mc_ref_allegiance_sort']

# mc_modules_mask                 = data_mc_mod['mc_modules_mask']
# mc_idx = data_mc_mod['mc_idx_tril']

mc_val = data_mc_mod["mc_val_tril"]
# mc_reg_idx             = data_mc_mod['mc_reg_idx']
mc_mod_idx = (data_mc_mod["mc_mod_idx"],)
mc_mod_idx = np.squeeze(mc_mod_idx)

# %%Figures
# =========================== Labels figures ==================================
label_mclinks = r"Inter-regional links"
label_mc_formula = r"MC$_{[ij, kl]} = CC[FC_{ij}(t), FC_{kl}(t)]$"
label_yhist = "Probability density"
# %%Metaconnectivity
# =============================================================================
# MC allegiance template (using now template as good ones)

# =============================================================================
for idx, setb in enumerate(mask_groups):
    # idx=0
    # setb = mask_groups[0]
    aux_label = label_variables[idx]
    num_group = int(len(aux_label) / 2)

    plt.figure(1 + idx, figsize=(13, 10))
    plt.clf()
    for xx in range(num_group * 2):
        plt.subplot(num_group, 2, 1 + xx)
        plt.title("MC ref allegiance sorted %s " % aux_label[xx])
        plt.imshow(
            np.mean(mc_allegiance[setb[idx]], axis=0),
            interpolation="none",
            aspect="auto",
            cmap="coolwarm",
        )
        plt.xticks((25, 700), labels=["1 ...", r"... $N^2-N$"], fontsize=12)
        plt.yticks(
            (25, 150, 580, 779),
            labels=["1", " .\n.\n.", " .\n.\n.", r"$N^2-N$"],
            fontsize=12,
        )
        plt.xlabel(label_mclinks, labelpad=-3, fontsize=11)
        plt.ylabel(label_mclinks, labelpad=-27, fontsize=11)

        cbar = plt.colorbar()
        cbar.set_label(
            label_mc_formula, rotation=270, labelpad=25, fontsize=11
        )  # <- your colorbar label
        plt.clim((-0.1, 0.1))
        cbar.set_ticks([-0.1, 0, 0.1])
        cbar.ax.tick_params(labelsize=15)
    plt.tight_layout()
    if save_fig == True:
        plt.savefig(allegiance_fig_dir / f"Allegiance_consensus_{aux_label}.png")
# %%

# =============================================================================
# MC for each individual
# =============================================================================

for idx, setb in enumerate(mask_groups):
    aux_label = label_variables[idx]
    num_group = int(len(aux_label) / 2)
    for ii, ind in enumerate(setb):
        # ind= mask_groups_good_impaired[0]
        plt.figure(4 + ii + (4 * idx), figsize=(13, 10))
        plt.clf()

        aux_numplot_ = mc_allegiance[ind].shape[0]
        if np.sqrt(aux_numplot_) / np.round(np.sqrt(aux_numplot_)) == 1:
            aux2 = int(np.sqrt(aux_numplot_))
        else:
            aux2 = int(np.ceil(np.sqrt(aux_numplot_)))
        # print(aux_numplot_, aux2)
        plt.title("Consensus module MC %s " % aux_label[ii])
        for xx in range(aux_numplot_):
            plt.subplot(aux2, aux2, 1 + xx)
            plt.imshow(
                mc_allegiance[xx], interpolation="none", aspect="auto", cmap="coolwarm"
            )
            plt.xticks([])
            plt.yticks([])
            plt.clim((-0.25, 0.25))
        plt.tight_layout()
        if save_fig == True:
            plt.savefig(
                allegiance_fig_dir / f"Allegiance_consensus_{aux_label[ii]}.png"
            )

# %%Modularity
# =============================================================================
# MC intra/inter modular per group
# =============================================================================
# ----------------- GLOBAL POOLED DISTRIBUTIONS -----------------
fig_global, ax = plt.subplots(figsize=(5, 4))

ax.hist(
    (mc_val[:, mc_mod_idx > 0].ravel(), mc_val[:, mc_mod_idx == 0].ravel()),
    bins=bins_parameter,
    density=True,
    histtype="step",
    label=("Intra-module", "Inter-module"),
)

ax.set_yscale("log")
ax.set_xlabel(label_mc_formula)
ax.set_ylabel(label_yhist)
ax.set_xticks([-0.7, 0, 0.7])
ax.set_ylim(1e-5, 1e1)
ax.legend()

fig_global.tight_layout()
if save_fig:
    fig_global.savefig(modularity_fig_dir / "MC_global_intra_vs_inter.png")

# ----------------- PER-GROUP DISTRIBUTIONS -----------------
n_groups = len(mask_groups)
fig_groups, axes = plt.subplots(
    n_groups, 2, figsize=(10, 3 * n_groups), sharex=True, sharey=True
)

# If there is only one group, axes is 1D → make it 2D
if n_groups == 1:
    axes = axes.reshape(1, 2)

for idx, setb in enumerate(mask_groups):
    aux_label = label_variables[idx]
    ax_intra = axes[idx, 0]
    ax_inter = axes[idx, 1]

    # Intra-module
    ax_intra.set_title("Intra-module")
    ax_intra.hist(
        [mc_val[xx][:, mc_mod_idx > 0].ravel() for xx in setb],
        bins=bins_parameter,
        density=True,
        histtype="step",
        label=aux_label,
    )
    ax_intra.set_yscale("log")
    ax_intra.set_ylim(1e-5, 1e1)
    ax_intra.set_xticks([-0.7, 0, 0.7])
    if idx == n_groups - 1:
        ax_intra.set_xlabel(label_mc_formula)
    ax_intra.set_ylabel(label_yhist)

    # Inter-module
    ax_inter.set_title("Inter-module")
    ax_inter.hist(
        [mc_val[xx][:, mc_mod_idx == 0].ravel() for xx in setb],
        bins=bins_parameter,
        density=True,
        histtype="step",
        label=aux_label,
    )
    ax_inter.set_yscale("log")
    ax_inter.set_ylim(1e-5, 1e1)
    ax_inter.set_xticks([-0.7, 0, 0.7])
    if idx == n_groups - 1:
        ax_inter.set_xlabel(label_mc_formula)
    # show legend only once per column to avoid clutter
    if idx == 0:
        ax_inter.legend(loc="upper right", fontsize=8)

fig_groups.tight_layout()
if save_fig:
    fig_groups.savefig(modularity_fig_dir / "MC_intra_inter_per_group.png")

# %%
# =============================================================================
# Plot MC intra/inter values of each individual
# =============================================================================
plt.figure(9)
plt.clf()

plt.subplot(211)
plt.title("Individual Intra-module values")
plt.hist(
    [mc_val[ind, mc_mod_idx > 0].ravel() for ind in range(n_animals)],
    bins=bins_parameter,
    density=True,
    histtype="step",
    alpha=0.2,
    color=np.full(n_animals, "Gray"),
)
plt.hist(
    (mc_val[:, mc_mod_idx > 0].ravel()),
    bins=bins_parameter,
    density=True,
    histtype="step",
    linewidth=1.5,
    color="k",
)
plt.yscale("log")
plt.ylabel(label_yhist)
plt.xticks([-0.7, 0, 0.7], fontsize=13)

plt.subplot(212)
plt.title("Individual Inter-module values")
plt.hist(
    [mc_val[ind, mc_mod_idx == 0].ravel() for ind in range(n_animals)],
    bins=bins_parameter,
    density=True,
    histtype="step",
    alpha=0.2,
    color=np.full(n_animals, "Gray"),
)

plt.hist(
    (mc_val[:, mc_mod_idx == 0].ravel()),
    bins=bins_parameter,
    density=True,
    histtype="step",
    linewidth=1.5,
    color="k",
)
plt.xticks([-0.7, 0, 0.7], fontsize=13)
plt.yscale("log")
plt.ylabel(label_mc_formula)

# plt.legend()
plt.xlabel(label_mc_formula)
plt.ylabel(label_yhist)
plt.tight_layout()
if save_fig == True:
    # plt.savefig(paths['figures'] + 'modularity/Allmice_Intra_intermodularity_mc_values.png')
    plt.savefig(modularity_fig_dir / "Allmice_Intra_intermodularity_mc_values.png")

# %%
# =============================================================================
# Q–Q plot: Intra vs Inter MC
# =============================================================================
# common probability grid, avoid extreme 0 and 1
p_grid = np.linspace(0.001, 0.999, 199)

q_intra = np.quantile(mc_intra, p_grid)
q_inter = np.quantile(mc_inter, p_grid)

fig_qq, ax = plt.subplots(figsize=(8, 8))



ax.plot(q_inter, q_intra, ".", markersize=3, alpha=0.7, color='C1')
# identity line
xy_min = min(q_inter.min(), q_intra.min())
xy_max = max(q_inter.max(), q_intra.max())
ax.plot([xy_min, xy_max], [xy_min, xy_max], "k--", linewidth=1)

ax.set_xlabel("Inter-module MC quantiles")
ax.set_ylabel("Intra-module MC quantiles")
ax.set_title("Q–Q plot: Intra vs Inter MC")
ax.axhline(0, linestyle="--", color="gray", alpha=0.5)
ax.axvline(0, linestyle="--", color="gray", alpha=0.5)


fig_qq.tight_layout()
if save_fig:
    fig_qq.savefig(modularity_fig_dir / "MC_global_QQ_intra_vs_inter.png")

#%%
# %%
# =============================================================================
# ECDF plot: Intra vs Inter MC
# =============================================================================
mc_intra = mc_val[:, mc_mod_idx > 0].ravel()
mc_inter = mc_val[:, mc_mod_idx == 0].ravel()

# ECDF plot: Intra vs Inter MC
def ecdf(x):
    x_sorted = np.sort(x)
    n = x_sorted.size
    y = np.arange(1, n + 1) / (n + 1.0)
    return x_sorted, y

x_intra, y_intra = ecdf(mc_intra)
x_inter, y_inter = ecdf(mc_inter)

fig_ecdf, ax = plt.subplots(figsize=(8, 6))

ax.plot(x_intra, y_intra, label="Intra-module", linewidth=1.5)
ax.plot(x_inter, y_inter, label="Inter-module", linewidth=1.5)

ax.set_xlabel(label_mc_formula)
ax.set_ylabel("ECDF")
ax.set_title("ECDF - Intra vs Inter MC")
ax.set_xticks([-0.7, 0, 0.7])
# ax.set_yscale('log')
ax.legend()

fig_ecdf.tight_layout()
if save_fig:
    fig_ecdf.savefig(modularity_fig_dir / "MC_global_ECDF_intra_vs_inter.png")




#%%
# =============================================================================
# Plot MC modules values of each individual
# =============================================================================


plt.figure(10)
plt.clf()

aux_numplot_ = len(np.unique(mc_ref_allegiance_communities))
if np.sqrt(aux_numplot_) / np.round(np.sqrt(aux_numplot_)) == 1:
    aux2 = int(np.sqrt(aux_numplot_))
else:
    aux2 = int(np.ceil(np.sqrt(aux_numplot_)))
# print(aux_numplot_, aux2)

for xx in range(len(np.unique(mc_ref_allegiance_communities))):
    # plt.subplot(3,3, 1+xx)
    plt.subplot(aux2, 2, 1 + xx)
    plt.title("Module %s" % (xx + 1))
    plt.hist(
        mc_val[:, mc_mod_idx == xx + 1].T,
        bins=bins_parameter,
        density=True,
        histtype="step",
        alpha=0.2,
        # color=np.full(n_animals, 'Gray')
    )
    plt.yscale("log")

    plt.hist(
        (mc_val[:, mc_mod_idx == xx + 1]).ravel(),
        bins=bins_parameter,
        density=True,
        histtype="step",
        linewidth=1.1,
        color="k",
    )
    plt.yscale("log")
    plt.xticks([-0.7, 0, 0.7], fontsize=13)
    plt.ylabel(label_yhist)
    plt.xlabel(label_mc_formula)
plt.tight_layout()
if save_fig == True:
    plt.savefig(modularity_fig_dir / "Allmice_Intramodularity_mc_values.png")
    # plt.savefig(paths['figures'] + 'modularity/Allmice_Intramodularity_alone_mc_values.png')
    # plt.savefig(paths['figures'] + 'modularity/Allmice_Intramodularity_alone_mc_values.png')

# %%
# =============================================================================
# plot MC modules values for each group
# =============================================================================

for idx, setb in enumerate(mask_groups):
    aux_label = label_variables[idx]

    plt.figure(11 + idx, figsize=(13, 8))
    plt.clf()
    for xx in np.unique(mc_ref_allegiance_communities):
        plt.subplot(aux2, 2, xx)
        # plt.subplot(2,2, xx)
        plt.title("Module %s" % (xx))
        plt.hist(
            ([mc_val[ind][:, mc_mod_idx == xx].ravel() for ind in setb]),
            bins=70,
            density=True,
            histtype="step",
            label=aux_label,
            # color=np.full(124, 'Gray')
        )
        plt.yscale("log")
        plt.ylabel(label_yhist, fontsize=12)
        plt.xlabel(label_mc_formula, fontsize=12)
        plt.xticks([-0.7, 0, 0.7], fontsize=13)
    plt.legend()
    plt.tight_layout()
    if save_fig == True:
        plt.savefig(
            modularity_fig_dir
            / f"Intramodularity_mc_values_{aux_label[0]}_{aux_label[2]}.png"
        )
        # plt.savefig(paths['figures'] + 'modularity/Intramodularity_mc_values_%s_%s.png'%(aux_label[0], aux_label[2]))

# %%
