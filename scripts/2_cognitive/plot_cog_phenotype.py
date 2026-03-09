#!/usr/bin/env python3
"""
Created on Wed Apr  2 02:59:41 2025

@author: samy
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

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
# plt.violinplot(
    # (
        # male_wt_data["RO24h_2M"],
        # male_wt_data["RO24h_4M"],
        # male_dki_data["RO24h_2M"],
        # male_dki_data["RO24h_4M"],
#     )
# )
plt.xticks([1, 2, 3, 4], ["Male WT 2M", "Male WT 4M", "Male dKI 2M", "Male dKI 4M"])
plt.axhline(0, c="k")
plt.ylabel("RO24h score")
plt.title("Distribution of RO24h for Male")
# plt.legend()


# %%
# %%
