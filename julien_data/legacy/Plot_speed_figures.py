#%%
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu, kruskal

# Load analysis class and preprocessed data
from class_dataanalysis_julien import DFCAnalysis
data = DFCAnalysis()
data.load_preprocessed_data()
cog_data_filtered = data.cog_data_filtered
df_cog = cog_data_filtered.copy()
data.get_temporal_parameters()

# Load all_speed
prefix = "speed"
save_path = data.paths['speed']
time_window_range = data.time_window_range
tau_range = np.arange(0, data.tau + 1)
n_animals = data.n_animals
window_file_total = save_path / f"{prefix}_windows{len(time_window_range)}_tau{len(tau_range)}_animals_{n_animals}.pkl"
with open(window_file_total, 'rb') as f:
    all_speed = pickle.load(f)

# %%
with open(Path(data.paths['allegiance']) / 'communities_wt_veh.pkl', 'rb') as f:
    communities = pickle.load(f)

communities_speed = []
for c in np.unique(communities):
    n_regions = np.sum(communities == c)
    print(n_regions)
    comm_file = save_path / f"{prefix}_windows{len(time_window_range)}_tau{len(tau_range)}_animals_{n_animals}_regions_{n_regions}.pkl"
    print(comm_file)
    with open(comm_file, 'rb') as f:
        comm_speed = pickle.load(f)
    communities_speed.append(comm_speed)
n_communities = len(communities_speed)

#%%
# Define window indices for different time windows
n_windows = len(communities_speed[0])
mid = n_windows // 2
window_idx_all   = np.arange(n_windows)
window_idx_short = np.arange(mid)
window_idx_long  = np.arange(mid, n_windows)

# %%
def pooled_community_speeds(community, animal_idxs, window_idx_list):
    pooled = []
    for animal_idx in animal_idxs:
        arrs = []
        for win_idx in window_idx_list:
            win_arr = communities_speed[community][win_idx]
            for tau in range(win_arr.shape[1]):
                vals = win_arr[animal_idx, tau, :].astype(float)
                vals = vals[~np.isnan(vals)]
                if vals.size > 0:
                    arrs.append(vals)
        pooled.append(np.concatenate(arrs) if arrs else np.array([]))
    return pooled

# %%

# %%
def build_community_speed_df(window_idx_list, window_label):
    plot_data = []
    for community in range(n_communities):
        for group, animal_idxs in data.groups.items():
            pooled = pooled_community_speeds(community, animal_idxs, window_idx_list)
            group_label = f"{group[0]}/{group[1]}"
            for arr in pooled:
                for val in arr:
                    plot_data.append({
                        'Community': f"C{community+1}",
                        'Group': group_label,
                        'Speed': val,
                        'Window': window_label
                    })
    return pd.DataFrame(plot_data)

df_all   = build_community_speed_df(window_idx_all,   'All windows')
df_short = build_community_speed_df(window_idx_short, 'Short windows')
df_long  = build_community_speed_df(window_idx_long,  'Long windows')
df_plot = pd.concat([df_all, df_short, df_long], ignore_index=True)

# %%
sns.set_theme(style='white', context='notebook')
g = sns.catplot(
    data=df_plot, x='Group', y='Speed',
    col='Community', row='Window',
    kind='violin', inner='quartile',
    sharey=False, height=3.5, aspect=1.2
)
g.set_titles("{row_name} | {col_name}")
g.set_axis_labels("Group", "Community dFC Speed")
plt.subplots_adjust(top=0.9)
plt.suptitle("Distribution of dFC Speeds per Community\nby Group and Window Pool")
plt.show()

# %%
for community in range(n_communities):
    for win_idx_list, pool_label in [(window_idx_all, "All windows"), (window_idx_short, "Short"), (window_idx_long, "Long")]:
        print(f"\nCommunity {community+1}, {pool_label}")
        group_data = []
        group_labels = []
        for group, animal_idxs in data.groups.items():
            pooled = pooled_community_speeds(community, animal_idxs, win_idx_list)
            group_arr = np.concatenate([arr for arr in pooled if arr.size > 0])
            group_data.append(group_arr)
            group_labels.append(f"{group[0]}-{group[1]}")
        # Kruskal–Wallis test
        if all(len(arr) > 0 for arr in group_data):
            stat, pval = kruskal(*group_data)
            print(f"  Kruskal–Wallis H = {stat:.2f}, p = {pval:.2g}")
        # Pairwise Mann–Whitney
        from itertools import combinations
        for (i, j) in combinations(range(len(group_data)), 2):
            u, p = mannwhitneyu(group_data[i], group_data[j])
            print(f"    {group_labels[i]} vs {group_labels[j]}: U={u:.1f}, p={p:.4g}")

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_community_histograms(communities_speed, groups, 
                              window_idx_list, pool_label="All windows", 
                              bins=60, kind='hist', kde=True, alpha=0.55):
    n_communities = len(communities_speed)
    palette = sns.color_palette('tab10', n_colors=len(groups))
    group_labels = [f"{g[0]}-{g[1]}" for g in groups.keys()]

    for comm in range(n_communities):
        plt.figure(figsize=(9,5))
        for idx, (group, animal_idxs) in enumerate(groups.items()):
            pooled = []
            for animal_idx in animal_idxs:
                arrs = []
                for win_idx in window_idx_list:
                    win_arr = communities_speed[comm][win_idx]
                    for tau in range(win_arr.shape[1]):
                        arr = win_arr[animal_idx, tau, :].astype(float)
                        arr = arr[~np.isnan(arr)]
                        if arr.size > 0:
                            arrs.append(arr)
                if arrs:
                    pooled.append(np.concatenate(arrs))
            comm_speeds = np.concatenate(pooled) if pooled else np.array([])
            if comm_speeds.size > 0:
                color = palette[idx]
                if kind == 'hist':
                    sns.histplot(comm_speeds, bins=bins, stat='density', element='step',
                                 fill=False, color=color, label=group_labels[idx], alpha=alpha, linewidth=1.7)
                if kde and comm_speeds.size > 10:
                    sns.kdeplot(comm_speeds, color=color, lw=2, label=None)
                plt.axvline(np.median(comm_speeds), color=color, linestyle='--', lw=1)
        plt.xlabel("Community dFC Speed")
        plt.ylabel("Density")
        plt.title(f"Community {comm+1} | {pool_label}")
        plt.legend(title="Group", fontsize=10)
        plt.tight_layout()
        plt.show()

plot_community_histograms(communities_speed, data.groups, window_idx_all, pool_label="All windows")


# %%
