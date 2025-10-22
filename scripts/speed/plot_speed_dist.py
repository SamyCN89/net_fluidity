

#%%

from cProfile import label
from multiprocessing import pool
from os import path
import time
import numpy as np
import matplotlib.pyplot as plt
from scripts import speed
from shared_code.fun_utils import load_cognitive_data, set_figure_params
from shared_code.fun_paths import get_paths
from shared_code.fun_loaddata import load_timeseries_bundle

#%%
# --------- Load data ---------

save_fig = set_figure_params(False)
timecourse_folder = "Timecourses_updated_03052024"

# Get paths for data loading ines_abdullah dataset
paths = get_paths(
    dataset_name="ines_abdullah",
    timecourse_folder=timecourse_folder,
    cognitive_data_file="ROIs.xlsx",
    anat_labels_file="41_Allen.txt",
)


# Load timeseries bundle and grouping data
bundle = load_timeseries_bundle(
    paths["preprocessed"] / "ts_and_meta_2m4m.npz",
    paths["preprocessed"] / "grouping_data_oip.pkl",
)
# Load cognitive data
cog_data = load_cognitive_data(paths["preprocessed"] / "cog_data_sorted_2m4m.csv")

# Extract relevant data from bundle
n_animals = bundle.n_animals
total_tr = bundle.total_tr
anat_labels = bundle.anat_labels
regions = bundle.n_regions
# Create masks for each region group
mask_groups = bundle.mask_groups
label_variables = bundle.label_variables
#%%

# -------- speed loading --------

time_windows_range = np.arange(5,100,1)

speeds = []
for w in time_windows_range:
    filepath = f'/media/samy/Elements2/Proyectos/LauraHarsan/results/ines_abdullah/speed/all/all/speed_win{w}_lag1_tau4_animals_126_regions_41.npz'
    a = np.load(filepath, allow_pickle=True)
    s = a['speeds']
    # Flatten each animal’s array from (1, N) → (N,)
    s_flat = np.array([x.ravel() for x in s], dtype=object)
    speeds.append(s_flat)

    print(f"window {w}: n_animals={len(s_flat)}, len(speeds[0])={len(s_flat[0])}")

#%%

# Plotting mean speeds vs window size for each animal
plt.figure(figsize=(8,6))
for i in range(n_animals):
    animal_speeds = [speeds[j][i] for j in range(len(time_windows_range))]
    mean_speeds = [np.mean(animal_speeds[j]) for j in range(len(time_windows_range))]
    plt.plot(time_windows_range, mean_speeds, '.-', label=f'Animal {i+1}')
plt.xlabel('Time Window Size')
plt.ylabel('Mean dFC Speed')
plt.title('dFC Speed vs Time Window Size for Each Animal')
# %%
# Pooling all speeds across windows and animals
# Pool all speed values from all windows:



#%%

# -------- Pooled speed distribution per equal windows range--------

# Create a new figure for the pooled speed distribution
pooled_speeds = np.array([np.shape([s for s in speed])[1] for speed in speeds])
pooled_speeds_cdf = np.cumsum(pooled_speeds) / np.sum(pooled_speeds)

# Find indices for one-third and two-thirds of the CDF
indice_half = np.where(pooled_speeds_cdf >= 0.5)[0][0]
indice_third = np.where(pooled_speeds_cdf >= 1/3)[0][0]
indice_two_third = np.where(pooled_speeds_cdf >= 2/3)[0][0]

# cumulative distribution function (CDF)
plt.figure(figsize=(7, 5))
plt.title("Cumulative Distribution of dFC Speeds across Time Windows")
plt.axvline(x=time_windows_range[indice_half], color='red', linestyle='--', label='Median Window Size', alpha=0.5)
plt.axvline(x=time_windows_range[indice_third], color='green', linestyle='--', label='1/3 Window Size', alpha=0.5)
plt.axvline(x=time_windows_range[indice_two_third], color='blue', linestyle='--', label='2/3 Window Size', alpha=0.5)

plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
plt.axhline(y=1/3, color='green', linestyle='--', alpha=0.5)
plt.axhline(y=2/3, color='blue', linestyle='--', alpha=0.5)
plt.plot(time_windows_range, pooled_speeds_cdf, color='orange', lw=2, alpha=0.8)

plt.xlabel("Time Window Size")
plt.ylabel("Cumulative Frequency")
plt.xticks(time_windows_range[::5])
plt.legend()
plt.tight_layout()


#%%
pool_split = 'third'  # Options: 'half' or 'third'
# Flatten all speeds for overall histogram
all_speeds_flat = np.concatenate([np.concatenate([s.flatten() for s in speed])
                                  for speed in speeds])

if pool_split=='half':
    short_speeds_flat = np.concatenate([np.concatenate([s.flatten() for s in speed])
                                    for speed in speeds[:indice_half]])

    long_speeds_flat = np.concatenate([np.concatenate([s.flatten() for s in speed])
                                    for speed in speeds[indice_half:]])
elif pool_split=='third':
    short_speeds_flat = np.concatenate([np.concatenate([s.flatten() for s in speed])
                                    for speed in speeds[:indice_third]])

    mid_speeds_flat = np.concatenate([np.concatenate([s.flatten() for s in speed])
                                    for speed in speeds[indice_third:indice_two_third]])

    long_speeds_flat = np.concatenate([np.concatenate([s.flatten() for s in speed])
                                    for speed in speeds[indice_two_third:]])

# np.concatenate([speed for speed in speeds[indice_half:]])


#%%

#%%


#%%
# --------- Distribution by region groups ---------
# group_dict = {}
# for i in range(len(mask_groups)):
#     for lb, idx in zip(label_variables[i], mask_groups[i]):
#         indice_mask = np.where(idx==True)[0]
#         label_mask = lb
#         if i==0:
#             label_mask += ' OiP'
#         if i==1:
#             label_mask += ' NOR'
#         group_dict[label_mask] = indice_mask
#         print(group_dict[label_mask], label_mask)
# #%%
# label_var_updated = []
# for i in range(len(label_variables)):
#     aux_lab = []
#     for lbl in label_variables[i]:
#         if i==0:
#             lbl += ' OiP'
#         if i==1:
#             lbl += ' NOR'
#         print(lbl)
#         aux_lab.append(lbl)
#     label_var_updated.append(aux_lab)
#%%

# Create group dictionary for region groups with suffixes
group_dict: dict[str, list[int]] = {}
label_sets: list[list[str]] = []
total_sets = len(label_variables)
for i, labels in enumerate(label_variables):
    print(f"Processing label group {i+1}/{total_sets}"  )
    if i < 2:
        suffix = " OiP" if i == 0 else " NOR"
    else:
        suffix = ""
    print(f"Suffix for this group: '{suffix}'")
    label_group: list[str] = []
    for lbl, mask in zip(labels, mask_groups[i], strict=False):
        mask = np.asarray(mask, dtype=bool)
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            continue
        name = f"{lbl}{suffix}"
        group_dict[name] = indices.tolist()
        label_group.append(name)
    if label_group:
        label_sets.append(label_group)
if not label_sets:
    label_sets = [list(group_dict.keys())]
print("Final group dictionary keys:", list(group_dict.values()), 'label_sets:', label_sets)
#%%
animal_speeds = []
for i in range(n_animals):
    animal_s = [speeds[j][i] for j in range(len(time_windows_range))]
    animal_speeds.append(animal_s)
    # print(f"Animal {i+1}: n_speeds={np.shape(animal_s)}")
#%%
# Flatten speeds for each group across all animals and windows

speed_grp_all_flat = [np.array(
    [animal_speeds[g] for g in grp], object
    ).flatten()
    for grp in group_dict.values()]


#%%
# --------- Histograms ---------

bins_hist = 200


all_speeds_min = np.min(all_speeds_flat)
all_speeds_max = np.max(all_speeds_flat)

all_speeds_hist, bin_edge = np.histogram(all_speeds_flat,
                               bins=bins_hist,
                               range=(all_speeds_min, all_speeds_max),
                               density=True)
all_speeds_hist = all_speeds_hist / np.sum(all_speeds_hist)

if pool_split=='half':
    short_speeds_hist, _ = np.histogram(short_speeds_flat,
                                     bins=bins_hist,
                                     range=(all_speeds_min, all_speeds_max),
                                     density=True)
    long_speeds_hist, _ = np.histogram(long_speeds_flat,
                                    bins=bins_hist,
                                    range=(all_speeds_min, all_speeds_max),
                                    density=True)
    long_speeds_hist = long_speeds_hist / np.sum(long_speeds_hist)

elif pool_split=='third':
    short_speeds_hist, _ = np.histogram(short_speeds_flat,
                                     bins=bins_hist,
                                     range=(all_speeds_min, all_speeds_max),
                                     density=True)
    short_speeds_hist = short_speeds_hist / np.sum(short_speeds_hist)

    mid_speeds_hist, _ = np.histogram(mid_speeds_flat,
                                   bins=bins_hist,
                                   range=(all_speeds_min, all_speeds_max),
                                   density=True)
    mid_speeds_hist = mid_speeds_hist / np.sum(mid_speeds_hist)

    long_speeds_hist, _ = np.histogram(long_speeds_flat,
                                    bins=bins_hist,
                                    range=(all_speeds_min, all_speeds_max),
                                density=True)
    long_speeds_hist = long_speeds_hist / np.sum(long_speeds_hist)

#%%
# Group speeds by label
all_speeds_grp_hist = {}
for label, indices in group_dict.items():
    group_speeds = []
    for animal_s in animal_speeds:
        group_vals = np.concatenate([animal_s[j][indices] for j in range(len(time_windows_range))])
        group_speeds.append(group_vals)
    group_speeds_flat = np.concatenate(group_speeds)
    group_hist, _ = np.histogram(group_speeds_flat,
                              bins=bins_hist,
                              range=(all_speeds_min, all_speeds_max),
                              density=True)
    all_speeds_grp_hist[label] = group_hist / np.sum(group_hist)


short_speeds_grp_hist = {}
mid_speeds_grp_hist = {}
long_speeds_grp_hist = {}

for label, indices in group_dict.items():

    # Short window speeds
    group_speeds_short = []
    for animal_s in animal_speeds:
        # group_vals_short = np.concatenate([animal_s[j][indices] for j in range(indice_half  if pool_split=='half' else indice_mid)])
        group_vals_short = np.concatenate([animal_s[j][indices] for j in range(0,indice_third)] if pool_split=='third' else
                                        [animal_s[j][indices] for j in range(0,indice_half)])
        group_speeds_short.append(group_vals_short)
    group_speeds_short_flat = np.concatenate(group_speeds_short)
    group_hist_short, _ = np.histogram(group_speeds_short_flat,
                                    bins=bins_hist,
                                    range=(all_speeds_min, all_speeds_max),
                                    density=True)
    short_speeds_grp_hist[label] = group_hist_short / np.sum(group_hist_short)

    # Mid window speeds
    group_speeds_mid = []
    for animal_s in animal_speeds:
        group_vals_mid = np.concatenate([animal_s[j][indices] for j in range(indice_third, indice_two_third)]) if pool_split=='third' else[animal_s[j][indices] for j in range(indice_half, len(time_windows_range) - indice_two_third)]
        group_speeds_mid.append(group_vals_mid)
    group_speeds_mid_flat = np.concatenate(group_speeds_mid)
    group_hist_mid, _ = np.histogram(group_speeds_mid_flat,
                                  bins=bins_hist,
                                  range=(all_speeds_min, all_speeds_max),
                                  density=True)
    mid_speeds_grp_hist[label] = group_hist_mid / np.sum(group_hist_mid)

    # Long window speeds
    group_speeds_long = []
    for animal_s in animal_speeds:
        group_vals_long = np.concatenate([animal_s[j][indices] for j in range(indice_two_third, len(time_windows_range))])
        group_speeds_long.append(group_vals_long)
    group_speeds_long_flat = np.concatenate(group_speeds_long)
    group_hist_long, _ = np.histogram(group_speeds_long_flat,
                                   bins=bins_hist,
                                   range=(all_speeds_min, all_speeds_max),
                                   density=True)
    long_speeds_grp_hist[label] = group_hist_long / np.sum(group_hist_long)
#%%
# Plot overall distribution of speeds

plt.figure(figsize=(7, 5))
# plt.subplot(3, 1, 1)
plt.title("Pooled Speed (all windows pooled)")
plt.plot(bin_edge[:-1], all_speeds_hist, color='dodgerblue', lw=2, alpha=0.8, label='all animals')
if pool_split=='half':
    plt.plot(bin_edge[:-1], short_speeds_hist, color='orange', lw=2, alpha=0.8, label='short windows')
    plt.plot(bin_edge[:-1], long_speeds_hist, color='green', lw=2, alpha=0.8, label='long windows')
elif pool_split=='third':
    plt.plot(bin_edge[:-1], short_speeds_hist, color='orange', lw=2, alpha=0.8, label='short windows')
    plt.plot(bin_edge[:-1], mid_speeds_hist, color='purple', lw=2, alpha=0.8, label='mid windows')
    plt.plot(bin_edge[:-1], long_speeds_hist, color='green', lw=2, alpha=0.8, label='long windows')
# # Alternative plotting method
# plt.plot(all_speeds_hist[1][:-1], all_speeds_hist[0], color='dodgerblue', lw=2, alpha=0.8, label='all animals')
# plt.plot(short_speeds_hist[1][:-1], short_speeds_hist[0], color='orange', lw=2, alpha=0.8, label='short windows')
# plt.plot(mid_speeds_hist[1][:-1], mid_speeds_hist[0], color='purple', lw=2, alpha=0.8, label='mid windows')
# plt.plot(long_speeds_hist[1][:-1], long_speeds_hist[0], color='green', lw=2, alpha=0.8, label='long windows')
plt.legend()
plt.xlabel("Speed")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
#%%
# Plot distribution by region groups
# for label, hist in all_speeds_grp_hist.items():
for label_big in label_sets:
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    for label in label_big:
        hist = all_speeds_grp_hist[label]
        plt.plot(bin_edge[:-1], hist,
                lw=1, alpha=0.4,
                label=label)
    plt.title(f"Distribution of dFC Speed")
    plt.xlabel("Speed")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    for label in label_big:
        hist = all_speeds_grp_hist[label]
        plt.plot(bin_edge[:-1], hist,
                # Alternative plotting method
                # plt.plot(hist[1][:-1], hist[0],
                lw=1, alpha=0.4,
                label=label)
    plt.title(f"Distribution of dFC Speed (Log Scale)")
    plt.xlabel("Speed")
    plt.ylabel("Frequency")
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()

#%%
# Plot distribution by region groups for short and long windows
for label_big in label_sets:
    plt.figure(figsize=(12, 10))
    # Short windows
    plt.subplot(3, 2, 1)
    for label in label_big:
        hist_short = short_speeds_grp_hist[label]
        plt.plot(bin_edge[:-1], hist_short,
                lw=1, alpha=0.8,
                label=label)
    plt.title(f"Short Windows {time_windows_range[0]}-{time_windows_range[indice_third-1] if pool_split=='third' else time_windows_range[indice_half-1]} TR")
    plt.ylabel("Frequency")
    plt.xticks([])
    plt.yticks([])
    # Add grid
    plt.grid(alpha=0.3)

    plt.subplot(3, 2, 2)
    for label in label_big:
        hist_short = short_speeds_grp_hist[label]
        plt.plot(bin_edge[:-1], hist_short,
                lw=1, alpha=0.4,
                label=label)
    plt.grid()
    plt.title(f"Short Windows {time_windows_range[0]}-{time_windows_range[indice_third-1] if pool_split=='third' else time_windows_range[indice_half-1]} TR")
    plt.ylabel("Frequency")
    plt.yscale('log')
    plt.legend()
    plt.xticks([])
    plt.yticks([])

    # Mid windows
    plt.subplot(3, 2, 3)
    for label in label_big:
        hist_mid = mid_speeds_grp_hist[label]
        plt.plot(bin_edge[:-1], hist_mid,
                lw=1, alpha=0.4,
                label=label)
    plt.title(f"Mid Windows {time_windows_range[indice_third]}-{time_windows_range[indice_two_third] if pool_split=='third' else time_windows_range[indice_half-1]} TR")
    plt.ylabel("Frequency")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3, 2, 4)
    for label in label_big:
        hist_mid = mid_speeds_grp_hist[label]
        plt.plot(bin_edge[:-1], hist_mid,
                lw=1, alpha=0.4,
                label=label)
    plt.title(f"Mid Windows {time_windows_range[indice_third]}-{time_windows_range[indice_two_third] if pool_split=='third' else time_windows_range[indice_half-1]} TR")
    plt.ylabel("Frequency")
    plt.yscale('log')
    plt.legend()
    plt.xticks([])
    plt.yticks([])

    # Long windows
    plt.subplot(3, 2, 5)
    for label in label_big:
        hist_long = long_speeds_grp_hist[label]
        plt.plot(bin_edge[:-1], hist_long,
                lw=1, alpha=0.4,
                label=label)
    plt.title(f"Long Windows {time_windows_range[indice_two_third]}-{time_windows_range[-1] if pool_split=='third' else time_windows_range[indice_half-1]} TR")
    plt.xlabel("Speed")
    plt.ylabel("Frequency")
    plt.legend()
    plt.yticks([])
    plt.xticks([0.2, 0.6, 1.0])

    plt.subplot(3, 2, 6)
    for label in label_big:
        hist_long = long_speeds_grp_hist[label]
        plt.plot(bin_edge[:-1], hist_long,
                lw=1, alpha=0.4,
                label=label)
    plt.title(f"Long Windows {time_windows_range[indice_two_third]}-{time_windows_range[-1] if pool_split=='third' else time_windows_range[indice_half-1]} TR")
    plt.xlabel("Speed")
    plt.ylabel("Frequency")
    plt.yscale('log')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()

    plt.savefig(paths['f_speed'] / f'speed_distribution_by_region_groups_{label_big}_windows_ines.png', dpi=300)
# %%

# %%
