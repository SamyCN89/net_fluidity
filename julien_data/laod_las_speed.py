
#%%
import pickle
from pathlib import Path
from networkx import density
import numpy as np
from class_dataanalysis_julien import DFCAnalysis

data = DFCAnalysis()

# Load raw data
data.load_raw_timeseries()
data.load_raw_cognitive_data()
data.load_raw_region_labels()

# Load preprocessed data
data.load_preprocessed_data()


data.get_temporal_parameters()
#%%
# Match these variables to your last run:
prefix = "speed"
save_path = data.paths['speed']  # <-- update this!
time_window_range = data.time_window_range           # <-- list of window sizes, same as in your analysis
tau_range = np.arange(0,data.tau+ 1)                   # <-- as above
n_animals = data.n_animals                # <-- as above

window_file_total = save_path / f"{prefix}_windows{len(time_window_range)}_tau{len(tau_range)}_animals_{n_animals}.pkl"
#%%
with open(window_file_total, 'rb') as f:
    all_speed = pickle.load(f)

# Now all_speed is a list (or similar) with each entry for one window_size.
# The last one:
last_speed = all_speed[-1]  # This is the speed array for the last window size

# Example: print shape/info
print(f"Loaded speed for window {time_window_range[-1]}: shape = {last_speed.shape}")

# %%
#print the shape of each time windows
for i, speed in enumerate(all_speed):
    print(f"Window size {time_window_range[i]}: shape = {speed.shape}")
# %%

# Plot a hist distribution that pools (ravel or flatten) all the speed together

import matplotlib.pyplot as plt
# Pool all speed values from all windows:
all_speeds_flat = np.concatenate([np.concatenate([s.flatten() for s in speed]) 
                                  for speed in all_speed])
([np.shape([s.flatten() for s in speed]) 
for speed in all_speed])


np.shape(all_speeds_flat)

plt.figure(figsize=(7, 5))
plt.hist((all_speeds_flat), 
         bins=150, alpha=0.8,
         histtype='step',
         density=True,)
plt.title("Overall Distribution of dFC Speed (all windows pooled)")
plt.xlabel("Speed")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

window_medians = []
window_p25 = []
window_p75 = []

for win_list in all_speed:  # Iterate windows
    window_speeds = []
    for animal_list in win_list:  # Iterate animals
        for tau_speed in animal_list:  # Iterate tau (pool all)
            speed_flat = np.array(tau_speed, dtype='float').flatten()
            window_speeds.append(speed_flat)
    # Concatenate all animals and all tau for this window
    window_speeds_all = np.concatenate(window_speeds)
    # Compute stats
    window_medians.append(np.nanmedian(window_speeds_all))
    window_p25.append(np.nanpercentile(window_speeds_all, 25))
    window_p75.append(np.nanpercentile(window_speeds_all, 75))

window_medians = np.array(window_medians)
window_p25 = np.array(window_p25)
window_p75 = np.array(window_p75)

# Plot
plt.figure(figsize=(9,5))
plt.plot(time_window_range, window_medians, label='Median', color='navy')
plt.fill_between(time_window_range, window_p25, window_p75, color='lightblue', alpha=0.5, label='25th–75th percentile')
plt.xlabel("Window size")
plt.ylabel("Speed (all tau pooled)")
plt.title("dFC Speed: Median and Interquartile Range per Window Size (all tau pooled)")
plt.legend()
plt.tight_layout()
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

# --- Setup
# Get tau=0 index
tau_array = np.array(tau_range)
tau_idx = np.where(tau_array == 0)[0][0]

group_stats = {}

for group_name, animal_indices in data.groups.items():
    medians = []
    p25 = []
    p75 = []
    for i_win, win_list in enumerate(all_speed):
        window_speeds = []
        for idx in animal_indices:
            # Check for out-of-bounds
            if idx >= len(win_list):
                continue
            speed_arr = win_list[idx][tau_idx]
            speed_flat = np.array(speed_arr, dtype='float').flatten()
            window_speeds.append(speed_flat)
        if len(window_speeds) == 0:
            medians.append(np.nan)
            p25.append(np.nan)
            p75.append(np.nan)
            continue
        window_speeds_all = np.concatenate(window_speeds)
        medians.append(np.nanmedian(window_speeds_all))
        p25.append(np.nanpercentile(window_speeds_all, 25))
        p75.append(np.nanpercentile(window_speeds_all, 75))
    group_stats[group_name] = {
        'medians': np.array(medians),
        'p25': np.array(p25),
        'p75': np.array(p75),
    }
#//
# --- Plotting
plt.figure(figsize=(10,6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # One color per group
for i, (group_name, stats) in enumerate(group_stats.items()):
    label = f"{group_name[0]} - {group_name[1]}"
    plt.plot(time_window_range, stats['medians'], label=label, color=colors[i])
    # plt.fill_between(time_window_range, stats['p25'], stats['p75'], alpha=0.25, color=colors[i])

plt.xlabel("Window size")
plt.ylabel("Speed (tau=0)")
plt.title("dFC Speed by Group: Median and Interquartile Range per Window Size (tau=0)")
plt.legend()
plt.tight_layout()
plt.show()
# %%
# Count points per window, using ONLY tau=0
points_per_window = []
for win_list in all_speed:
    n_points = 0
    for animal_list in win_list:
        speed_arr = animal_list[tau_idx]  # Only tau=0
        n_points += np.array(speed_arr, dtype='float').size
    points_per_window.append(n_points)

points_per_window = np.array(points_per_window)
cumsum_points = np.cumsum(points_per_window)
total_points = cumsum_points[-1]
third_points = total_points / 3

# Find split indices
split_idx1 = np.searchsorted(cumsum_points, third_points)
split_idx2 = np.searchsorted(cumsum_points, 2 * third_points)

windows_pool1 = np.array(time_window_range[:split_idx1+1])
windows_pool2 = np.array(time_window_range[split_idx1+1:split_idx2+1])
windows_pool3 = np.array(time_window_range[split_idx2+1:])

print(f"Pool 1 windows: {windows_pool1}")
print(f"Pool 2 windows: {windows_pool2}")
print(f"Pool 3 windows: {windows_pool3}")

# %%
group_pool_speeds = {k: [[], [], []] for k in data.groups}  # [pool1, pool2, pool3]

for group_name, animal_indices in data.groups.items():
    for i_win, win_list in enumerate(all_speed):
        if i_win <= split_idx1:
            pool_idx = 0
        elif i_win <= split_idx2:
            pool_idx = 1
        else:
            pool_idx = 2
        for idx in animal_indices:
            if idx >= len(win_list):
                continue
            speed_arr = win_list[idx][0]  # Only tau=0
            group_pool_speeds[group_name][pool_idx].append(np.array(speed_arr, dtype='float').flatten())
    # Concatenate for each pool
    for pool_idx in [0, 1, 2]:
        if group_pool_speeds[group_name][pool_idx]:
            group_pool_speeds[group_name][pool_idx] = np.concatenate(group_pool_speeds[group_name][pool_idx])
        else:
            group_pool_speeds[group_name][pool_idx] = np.array([])

# %%
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
group_names = list(data.groups.keys())
pool_labels = [
    f"Pool 1 (windows {windows_pool1[0]}–{windows_pool1[-1]})",
    f"Pool 2 (windows {windows_pool2[0]}–{windows_pool2[-1]})",
    f"Pool 3 (windows {windows_pool3[0]}–{windows_pool3[-1]})"
]

fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)

for pool_idx, ax in enumerate(axes):
    for i, group_name in enumerate(group_names):
        arr = group_pool_speeds[group_name][pool_idx]
        if arr.size > 0:
            ax.hist(arr, bins=100, histtype='step',
                    density=True,
                    label=f"{group_name[0]} - {group_name[1]}", linewidth=1.5)
    ax.set_ylabel("Density")
    ax.set_title(pool_labels[pool_idx])
    ax.legend()
    # ax.grid(alpha=0.3)

axes[-1].set_xlabel("Speed (tau=0)")
plt.suptitle("dFC Speed Distribution (KDE): Three Window Pools (tau=0 only), Per Group", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()

# %%
from itertools import combinations
# group_names = list(data.groups.keys())  # List of group keys (tuples)
# group_pairs = list(combinations(group_names, 2))  # All unique pairs
group_names = list(data.groups.keys())[:3]  # Only first 3 groups
group_pairs = list(combinations(group_names, 2))  # 3 pairs: (0,1), (0,2), (1,2)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import probplot

# Choose pool index (0, 1, or 2)
for pool_idx in range(3):
    pool_label = pool_labels[pool_idx]
    fig, axes = plt.subplots(3, 3, figsize=(14, 14), sharex=True, sharey=True)
    fig.suptitle(f"QQ plots for {pool_label} (tau=0 only)", fontsize=18)
    axes = axes.flatten()
    plot_idx = 0
    for i, g1 in enumerate(group_names):
        for j, g2 in enumerate(group_names):
            ax = axes[plot_idx]
            plot_idx += 1
            if i >= j:
                ax.axis('off')  # Only plot each pair once, upper triangle
                continue
            arr1 = group_pool_speeds[g1][pool_idx]
            arr2 = group_pool_speeds[g2][pool_idx]
            if arr1.size == 0 or arr2.size == 0:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=12)
                ax.set_axis_off()
                continue
            # QQ plot (empirical quantiles)
            q1 = np.quantile(arr1, np.linspace(0, 1, min(len(arr1), len(arr2))))
            q2 = np.quantile(arr2, np.linspace(0, 1, min(len(arr1), len(arr2))))
            ax.plot(q1, q2, 'o', alpha=0.6)
            # Diagonal reference
            lims = [min(q1.min(), q2.min()), max(q1.max(), q2.max())]
            ax.plot(lims, lims, 'k--', alpha=0.7)
            ax.set_title(f"{g1[0]}-{g1[1]} vs {g2[0]}-{g2[1]}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

# %%
from itertools import combinations
import matplotlib.pyplot as plt

group_names = list(data.groups.keys())
group_pairs = list(combinations(group_names, 2))

for pool_idx in range(3):
    pool_label = pool_labels[pool_idx]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    fig.suptitle(f"QQ plots for {pool_label} (tau=0 only)", fontsize=18)
    for k, (g1, g2) in enumerate(group_pairs):
        arr1 = group_pool_speeds[g1][pool_idx]
        arr2 = group_pool_speeds[g2][pool_idx]
        ax = axes[k]
        if arr1.size == 0 or arr2.size == 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=12)
            ax.set_axis_off()
            continue
        # QQ
        q1 = np.quantile(arr1, np.linspace(0, 1, min(len(arr1), len(arr2))))
        q2 = np.quantile(arr2, np.linspace(0, 1, min(len(arr1), len(arr2))))
        ax.plot(q1, q2, 'o', alpha=0.6)
        lims = [min(q1.min(), q2.min()), max(q1.max(), q2.max())]
        ax.plot(lims, lims, 'k--', alpha=0.7)
        ax.set_title(f"{g1[0]}-{g1[1]} vs {g2[0]}-{g2[1]}")
    for ax in axes[len(group_pairs):]:
        ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

# %%
