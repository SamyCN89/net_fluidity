#!/usr/bin/env python3
"""
Created on Wed Apr  2 02:59:41 2025

@author: samy
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from shared_code.fun_dfcspeed import dfc_stream2fcd, ts2dfc_stream
from shared_code.fun_loaddata import load_timeseries_bundle

# from fun_utils import get_paths, set_figure_params
from shared_code.fun_paths import get_paths

# from shared_code.fun_utils import set_figure_params
from shared_code.fun_utils import load_cognitive_data, set_figure_params

# %%
# ========================= Figure parameters ================================
save_fig = set_figure_params(True)

# =================== Paths and folders =======================================
timecourse_folder = "Timecourses_updated_03052024"
paths = get_paths(
    dataset_name="ines_abdallah",
    timecourse_folder=timecourse_folder,
    cognitive_data_file="ROIs.xlsx",
    anat_labels_file="41_Allen.txt",
)

# =================== Load time series data ===================================
bundle = load_timeseries_bundle(
    paths["preprocessed"] / "ts_and_meta_2m4m.npz",
    paths["preprocessed"] / "grouping_data_oip.pkl",
)
ts = bundle.ts
n_animals = bundle.n_animals
total_tr = bundle.total_tr
anat_labels = bundle.anat_labels
regions = bundle.n_regions

# ========================== Mask groups and label variables =========================
mask_groups = bundle.mask_groups
label_variables = bundle.label_variables

# ========================== Prepare cognitive data =========================
# Load cognitive data
cog_data = load_cognitive_data(paths["preprocessed"] / "cog_data_sorted_2m4m.csv")
# %%
# Time series checking and plotting

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
# Cognitive data checking and plotting
male_ind = cog_data["Sexe"] == "M"
female_ind = cog_data["Sexe"] == "F"
wt_ind = cog_data["Genotype"] == "wt"
dki_ind = cog_data["Genotype"] == "dKI"

mouse_hash_cog = cog_data["Name"].to_numpy()

sex_label = cog_data["Sexe"].to_numpy()
gen_label = cog_data["Genotype"].to_numpy()

# %%

# =============================================================================
# FCD stream
# =============================================================================


# %%


def plot_fcdxsubjects(fcd, mouse_hash_cog, gen_label):
    """
    Plot FCD matrices for each subject in a grid layout.
    Each FCD matrix is displayed in a subplot with appropriate titles and colorbars.
    """
    for idx_mice in tqdm.tqdm(range(n_animals // 2), desc="Plotting FCDs"):
        # for idx_mice in range(2):
        plt.figure(figsize=(6, 10))
        # plt.clf()

        plt.title(
            f" FCD (window={windows_size}, lag={lag}) mouse {mouse_hash_cog[idx_mice]} {gen_label[idx_mice]} {sex_label[idx_mice]}",
            y=1.05,
            loc="left",
            fontsize=13,
        )
        # remove the lines at bottom and left of the plot
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(211)
        plt.title("2m")
        plt.imshow(
            fcd[idx_mice],
            aspect="auto",
            interpolation="none",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
        )
        # plt.colorbar()

        plt.xticks([])
        plt.yticks([])
        plt.xlabel(r"$tw_{i}$")
        plt.ylabel(r"$tw_{j}$")
        # colorbar label
        cbar = plt.colorbar(label=r"CC(FC($tw_{i}$), FC($tw_{j}$))")

        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])  # positions
        cbar.set_ticklabels(
            ["-1", "-0.5", "0", "0.5", "1"], fontsize=12
        )  # optional labels

        # plt.clim(-1, 1)
        # plt.title("Functional Connectivity Dynamics")
        # plt.title(f" FCD (window={windows_size}, lag={lag}) mouse
        # plt.ylabel("Windowed time (TR)")

        plt.subplot(212)
        plt.title("4m")
        plt.imshow(
            fcd[idx_mice * 2],
            aspect="auto",
            interpolation="none",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
        )
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(r"$tw_{i}$")
        plt.ylabel(r"$tw_{j}$")
        # colorbar label
        cbar = plt.colorbar(label=r"CC(FC($tw_{i}$), FC($tw_{j}$))")

        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])  # positions
        cbar.set_ticklabels(
            ["-1", "-0.5", "0", "0.5", "1"], fontsize=12
        )  # optional labels

        # plt.show()
        plt.tight_layout()
        if save_fig:
            plt.savefig(
                paths["f_dfc"]
                / f"fcd_window_{windows_size}_lag_{lag}_mouse_#{mouse_hash_cog[idx_mice]}_{gen_label[idx_mice]}_{sex_label[idx_mice]}.pdf"
            )
            plt.savefig(
                paths["f_dfc"]
                / f"fcd_window_{windows_size}_lag_{lag}_mouse_#{mouse_hash_cog[idx_mice]}_{gen_label[idx_mice]}_{sex_label[idx_mice]}.png"
            )
        plt.close()


# %%

lag = 1
windows_size_range = np.arange(73, 100, 1)

# windows_size=25
for windows_size in windows_size_range:
    print(f"Processing window size: {windows_size}")
    dfc_stream = np.array(
        [
            ts2dfc_stream(ts[aa], windows_size, lag, format_data="2D")
            for aa in range(n_animals)
        ]
    )
    fcd = np.array([dfc_stream2fcd(dfc_stream[aa]) for aa in range(n_animals)])
    plot_fcdxsubjects(fcd, mouse_hash_cog, gen_label)

# %%
