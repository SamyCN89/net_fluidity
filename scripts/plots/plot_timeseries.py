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
from shared_code.fun_utils import (
    load_cognitive_data,
    set_figure_params,
)

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

# %%
# ========================== Plot grouped, colored, upside-down ==========================
import os

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

# --- Networks definitions (as before) ---
dmn_spec = ["PL ILA", "PFC", "ACA", "RSP", 0, 1, 2, 3]
mem_spec = [
    "d HIP",
    "v HIP",
    "d DG",
    "v DG",
    "PERI",
    "ENT",
    "SUB",
    "ReRh",
    "THAL memory",
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
]


def _normalize(s):
    return " ".join(str(s).split()).lower()


def resolve_indices(spec, labels):
    idxs, seen = [], set()
    norm_labels = [_normalize(l) for l in labels]
    for item in spec:
        if isinstance(item, int):
            if 0 <= item < len(labels) and item not in seen:
                idxs.append(item)
                seen.add(item)
            continue
        target = _normalize(item)
        found = [i for i, nl in enumerate(norm_labels) if nl == target]
        if not found:
            found = [
                i
                for i, nl in enumerate(norm_labels)
                if nl.startswith(target) or target in nl
            ]
        for i in found:
            if i not in seen:
                idxs.append(i)
                seen.add(i)
        if not found:
            print(f"[warn] Label not found for spec item: {item!r}")
    return idxs


dmn_idx = resolve_indices(dmn_spec, anat_labels)
mem_idx = resolve_indices(mem_spec, anat_labels)
mem_idx = [i for i in mem_idx if i not in set(dmn_idx)]

all_idx = list(range(len(anat_labels)))
other_idx = [i for i in all_idx if i not in set(dmn_idx) | set(mem_idx)]

# Group order (DMN first, then Memory, then Others)
plot_order = dmn_idx + mem_idx + other_idx

# --- Colors ---
clr_dmn, clr_mem, clr_other = "tab:blue", "tab:red", "tab:gray"
color_by_index = {
    i: clr_dmn if i in dmn_idx else clr_mem if i in mem_idx else clr_other
    for i in all_idx
}

# ========================== Plot (upside-down + colored labels) ==========================
plt.figure(figsize=(12, 10))
offset = 0.06

# ts[0] assumed shape (T, R)
T = ts[0].shape[0]
# Flip vertically: plot from top to bottom reversed order
plot_order_reversed = plot_order[::-1]

for row, ridx in enumerate(plot_order_reversed):
    plt.plot(ts[0][:, ridx] + row * offset, lw=1.0, color=color_by_index[ridx])

# Flip y-axis so top label corresponds to first in reversed order
plt.ylim(-offset, len(anat_labels) * offset)
plt.yticks(
    np.arange(len(plot_order_reversed)) * offset,
    [anat_labels[i] for i in plot_order_reversed],
)

# --- Color the ytick labels to match their network ---
ax = plt.gca()
for ticklabel, ridx in zip(ax.get_yticklabels(), plot_order_reversed, strict=False):
    ticklabel.set_color(color_by_index[ridx])

plt.xlabel("TR")
plt.xlim(0, min(200, T))
plt.xticks(np.arange(0, min(200, T) + 1, 50), fontsize=12)
plt.tight_layout()

# Legend
handles = [
    Line2D([0], [0], color=clr_dmn, lw=2, label="DMN"),
    Line2D([0], [0], color=clr_mem, lw=2, label="Memory"),
    Line2D([0], [0], color=clr_other, lw=2, label="Other"),
]
plt.legend(
    handles=handles,
    loc="upper right",
)

# Save and show
outdir = paths["figures"] / "ts"
os.makedirs(outdir, exist_ok=True)
plt.savefig(outdir / f"ts_extract_{timecourse_folder}_grouped_inverted.png", dpi=200)
plt.show()
