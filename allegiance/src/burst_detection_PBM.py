#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 02:59:41 2025

@author: samy
"""
#%%
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import brainconn as bct


from scipy.stats import zscore, pearsonr
#Compute k-means clustering on the z-scored time series
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from shared_code.fun_utils import get_paths, set_figure_params
from shared_code.fun_dfcspeed import ts2fc

# ========================== Figure parameters ================================
# Set figure parameters globally
save_fig = set_figure_params(True)

# =================== Paths and folders =======================================

paths = get_paths()
data_ts = np.load(paths['sorted'] /  'ts_and_meta_2m4m.npz')

# ========================== Load data =========================

#Parameters and indices of variables
ts          = data_ts['ts']
n_animals   = int(data_ts['n_animals'])
total_tp    = data_ts['total_tp']
regions     = data_ts['regions']
is_2month_old = data_ts['is_2month_old']
anat_labels = data_ts['anat_labels']

#%%

plt.figure(figsize=(12, 8))
offset = 0.07  # vertical offset between time series
for i, ts1 in enumerate(ts[0].T):
    plt.plot(ts1 + i * offset, label=f"TS {i+1}")
# plt.ylim(-0.1,0.75)
plt.title("Time Series")
plt.xlabel("Time Points")
plt.ylabel("Signal + Offset")
plt.tight_layout()
plt.savefig(paths['figures'] / 'ts/ts_extract.png')
plt.show()
# %%


# Z-score ts_df by columns and maintain as ts_df

#Concatenate all time series
ts_concat = np.concatenate(ts, axis=0)
# Z-score the concatenated time series
ts_zscore = zscore(ts_concat, axis=0)

#%%
#Create a dataframe with the zscore time series
ts_df = pd.DataFrame(ts_zscore, columns=[f"{anat_labels[i]}" for i in range(ts_zscore.shape[1])])
# Create a new column for the animal number
ts_df['animal'] = np.repeat(np.arange(1, n_animals + 1), total_tp)

# %%
# Plot the z-scored time series for the anatomical regions
plt.figure(figsize=(12, 8))
offset = 0.7
# plt.plot(ts_df_zscore, offset=offset, label=f"Z-scored TS")
for i, ts_z in enumerate(ts_zscore.T):
    plt.plot(ts_z + i * offset * 10, label=f"Z-scored TS {i+1}")
# plt.ylim(-0.1,0.75)
plt.title("Z-Scored Concatenated Time Series")
plt.xlabel("TRs (seconds)")
plt.yticks(np.arange(len(anat_labels))*offset*10, anat_labels)
plt.xlim(0, ts_zscore.shape[0])
plt.xticks(np.arange(0, ts_zscore.shape[0], step=10000), rotation=45)
plt.tight_layout()
plt.savefig(paths['figures'] / 'ts/ts_zscore_concatenated_all_animals.png')
plt.show()

# %%


#%%
#Test for one animal

# ================= Kmeans clustering ========================

# methjod to determine the number of clusters: Elbow method and Silhouette method
# Elbow method: plot the inertia (sum of squared distances to closest cluster center) as a function of the number of clusters
# Silhouette method: plot the silhouette score as a function of the number of clusters
# Define the range of clusters to test

n_clusters_range = range(2, 10)

start = time.time()
# Define the number of clusters
n_clusters = 4

# Create a KMeans instance
kmeans = KMeans(n_clusters=n_clusters, random_state=42, max_iter=100, n_init=100)
# Fit the KMeans model to the z-scored time series
kmeans.fit(ts_zscore)

# Print the inertia (sum of squared distances to closest cluster center)
print(f"Inertia (sum of squared distances to closest cluster center): {kmeans.inertia_}")
stop = time.time()
print(f"Time taken to create KMeans instance: {stop - start:.2f} seconds")
#%%


kmeans_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

#Plot the k-means clustering results 
plt.figure(figsize=(12, 8))
plt.title(f"KMeans Cluster Centers (n_clusters={n_clusters})")
plt.imshow(cluster_centers.T, aspect='auto', cmap='RdBu_r', interpolation='none')
plt.colorbar(label='Cluster Center Value')
plt.xticks(np.arange(n_clusters), [f"Cluster {i+1}" for i in range(n_clusters)])
plt.yticks(np.arange(len(anat_labels)), anat_labels)
plt.clim(-1.5,1.5)



#%%
#FC cluster, taking the respective time points of each cluster
# Compute the functional connectivity (FC) for each cluster
# Compute the Pearson correlation between the z-scored time series and the cluster centers


# FC_lambda (ts_zscore, kmeans_labels)

ts_cluster = np.array([ts_zscore[kmeans_labels == k, :] for k in range(n_clusters)], object)
print('dimension of each cluster', [np.shape(ts_cluster[k]) for k in range(n_clusters)])

fc_cluster = []
for k in range(n_clusters):
    if ts_cluster[k].shape[0] == 0:
        print(f"Warning: Cluster {k} is empty")
        fc_cluster.append(np.zeros((ts_zscore.shape[1], ts_zscore.shape[1])))
    else:
        fc_cluster.append(ts2fc(ts_cluster[k], method='pearson'))
fc_cluster = np.array(fc_cluster)
# fc_cluster2 = np.array([ts2fc(ts_cluster[k], method='pearson') for k in range(n_clusters)])

#Plot matrix of each fc_cluster as a subplot squared
plt.figure(figsize=(12, 12))
plt.title(f"Functional Connectivity Matrix for All Clusters (n_clusters={n_clusters})")  
for i in range(n_clusters):

    n_rows = int(np.ceil(np.sqrt(n_clusters)))
    n_cols = int(np.ceil(n_clusters / n_rows))
    plt.subplot(n_rows, n_cols, i + 1)

    plt.imshow(fc_cluster[i], aspect='auto', cmap='RdBu_r', interpolation='none')
    plt.colorbar(label='Functional Connectivity Value')
    # plt.xticks(np.arange(len(anat_labels)), anat_labels, rotation=90)
    # plt.yticks(np.arange(len(anat_labels)), anat_labels)

    plt.clim(-0.25,0.25)
    # plt.savefig(paths['figures'] / f'ts/cluster_{n_clusters}_fc_matrix.png')
plt.tight_layout()
plt.show()

# FC_lambda = pearsonr(ts_zscore))



#%%
# ================== Network analysis: Global and Local efficiency  =========================
# Compute the global efficiency of the functional connectivity matrix

global_efficiency = np.array([bct.distance.efficiency_wei(abs(fc_cluster[nn]), local=False) for nn in range(n_clusters)])
local_efficiency = np.array([bct.distance.efficiency_wei(abs(fc_cluster[nn]), local=True) for nn in range(n_clusters)])
#%%
#Plot a barplot of the global efficiency and local efficiency
plt.figure(figsize=(12, 6))
bar_width = 0.35
x = np.arange(n_clusters)
plt.subplot(121)
plt.title('Global Efficiency of Clusters')
plt.bar(x - bar_width/2, global_efficiency, width=bar_width, label='Global Efficiency')
plt.ylim(0,0.12)
plt.ylabel('Efficiency')
plt.xticks(x, [f'State {i+1}' for i in range(n_clusters)])

plt.subplot(122)
plt.bar(x + bar_width/2, np.mean(local_efficiency, axis=1), width=bar_width, label='Local Efficiency')
plt.ylim(0,0.12)
plt.ylabel('Efficiency')
plt.title('Local Efficiency of Clusters')
plt.xticks(x, [f'State {i+1}' for i in range(n_clusters)])
plt.legend()
plt.tight_layout()
plt.show()

#%%
# ================================= T-Graphlet =======================

# Reorganize the kmeans_labels to 
# Reconstruct per-animal kmeans labels
kmeans_labels_per_animal = np.array([
    kmeans_labels[i * total_tp : (i + 1) * total_tp] for i in range(n_animals)
])


#Assign a fc_cluster to each label in each animal
fc_cluster_per_animal = np.array([
    fc_cluster[kmeans_labels_per_animal[i]] for i in range(n_animals)
    ])

region_triu_index = np.triu_indices(regions, k=1)
fc_cluster_per_animal= fc_cluster_per_animal[:,:,region_triu_index[0], region_triu_index[1]]
#%%
thr_=0.01
threshold_links = np.max(fc_cluster_per_animal, axis=1)*thr_
binary_fc_cluster = (fc_cluster_per_animal>threshold_links[:,None,:]).astype(int)
#Plot one link of fc_cluster_per_animal--

plt.figure(7, figsize=(14,5))
plt.clf()
# plt.plot(fc_cluster_per_animal[0,0],'.-')
plt.imshow(binary_fc_cluster[0].T, interpolation='none', aspect='auto', cmap= 'Greys')


#%%
def extract_link_activations(binary_fc_data):
    """
    Extract onset, offset, and duration of active (1-valued) FC links over time.

    Parameters:
    -----------
    binary_fc_data : np.ndarray
        Shape (n_animals, time_points, n_links), binary values (0 or 1).

    Returns:
    --------
    all_events : list of lists of dicts
        all_events[animal][link] is a list of event dicts with keys: onset, offset, duration
    """
    n_animals, n_timepoints, n_links = binary_fc_data.shape
    all_events = []

    for animal in range(n_animals):
        animal_events = []
        for link in range(n_links):
            signal = binary_fc_data[animal, :, link]
            diff = np.diff(np.concatenate([[0], signal, [0]]))
            onsets = np.where(diff == 1)[0]
            offsets = np.where(diff == -1)[0]
            durations = offsets - onsets

            events = [
                {"onset": int(o), "offset": int(f), "duration": int(d)}
                for o, f, d in zip(onsets, offsets, durations)
            ]
            animal_events.append(events)
        all_events.append(animal_events)

    return all_events

link_events = extract_link_activations(binary_fc_cluster)

# Example: print first 5 events for link 10 in animal 0
for event in link_events[0][10][:5]:
    print(event)
# link_events[animal][link] = list of {'onset', 'offset', 'duration'}
n_animals = len(link_events)
n_links = len(link_events[0])

burst_counts = np.zeros((n_animals, n_links), dtype=int)
durations_by_link = [[] for _ in range(n_links)]

for animal in range(n_animals):
    for link in range(n_links):
        for ee in link_events[animal][link]:
            durations_by_link[link].append(ee['duration'])
        
        # burst_counts[animal, link] = len(link_events[animal][link])


#%%

# Flatten into rows: each row = (link, duration)
link_ids = []
durations = []
for link, dur_list in enumerate(durations_by_link):
    link_ids.extend([link] * len(dur_list))
    durations.extend(dur_list)

df_durations = pd.DataFrame({'link': link_ids, 'duration': durations})


df_durations.groupby('link')['duration'].describe()



plt.figure(figsize=(12, 5))
sns.violinplot(data=df_durations, x='link', y='duration', cut=0)
plt.xticks([], [])  # hide labels if too many links
plt.title("Burst Duration Distributions per Link")
plt.tight_layout()
plt.show()
#%%

#New proposiiton


def analyze_link_dynamics(kmeans_labels, fc_clusters, meta, save_fig, fig_path):
    n_animals = meta['n_animals']
    total_tp = meta['total_tp']
    regions = meta['regions']

    kmeans_labels_per_animal = np.array([
        kmeans_labels[i * total_tp : (i + 1) * total_tp] for i in range(n_animals)
    ])
    fc_cluster_per_animal = np.array([
        fc_clusters[kmeans_labels_per_animal[i]] for i in range(n_animals)
    ])
    region_triu_index = np.triu_indices(regions, k=1)
    fc_cluster_per_animal = fc_cluster_per_animal[:,:,region_triu_index[0], region_triu_index[1]]

    thr_ = 0.01
    threshold_links = np.max(fc_cluster_per_animal, axis=1)*thr_
    binary_fc_cluster = (fc_cluster_per_animal > threshold_links[:, None, :]).astype(int)

    link_events = extract_link_activations(binary_fc_cluster)

    durations_by_link = [[] for _ in range(binary_fc_cluster.shape[2])]
    for animal in range(n_animals):
        for link in range(len(link_events[animal])):
            durations_by_link[link].extend([e['duration'] for e in link_events[animal][link]])

    link_ids = []
    durations = []
    for link, dur_list in enumerate(durations_by_link):
        link_ids.extend([link] * len(dur_list))
        durations.extend(dur_list)

    df_durations = pd.DataFrame({'link': link_ids, 'duration': durations})

    plt.figure(figsize=(12, 5))
    sns.violinplot(data=df_durations, x='link', y='duration', cut=0)
    plt.xticks([], [])
    plt.title("Burst Duration Distributions per Link")
    plt.tight_layout()
    save_figure(fig_path / 'link_burst_durations.png', save_fig)
    plt.show()

analyze_link_dynamics(kmeans_labels, fc_clusters, meta, save_fig, fig_path)

# aux_plot = []

# range_thr = np.linspace(0.01,0.9)
# for thr_ in range_thr: 
#     threshold_links = np.max(fc_cluster_per_animal, axis=1)*thr_
#     thr_num = np.sum(fc_cluster_per_animal>threshold_links[:,None,:])
#     aux_plot.append(thr_num/n_animals)
# #%%
# plt.figure(10)
# plt.plot(range_thr, np.array(aux_plot)/(450*820))
# # plt.yscale('log')















#%%


def compute_dwell_times_np(labels):
    """
    Compute dwell times (consecutive durations) for each cluster using NumPy only.
    
    Parameters:
    -----------
    labels : 1D np.ndarray of shape (n_timepoints,)
        Cluster labels over time (e.g., from k-means).
    
    Returns:
    --------
    dwell_cluster_ids : np.ndarray of shape (n_dwell,)
        Cluster label for each dwell segment.
    
    dwell_lengths : np.ndarray of shape (n_dwell,)
        Duration of each dwell segment (i.e., how many time points).
    
    dwell_starts : np.ndarray of shape (n_dwell,)
        Starting index of each dwell segment.
    """
    labels = np.asarray(labels)
    
    # Find where the label changes
    change_points = np.where(np.diff(labels) != 0)[0] + 1
    
    # Start indices of each dwell
    dwell_starts = np.insert(change_points, 0, 0)
    
    # End indices (exclusive)
    dwell_ends = np.append(change_points, len(labels))
    
    # Length of each dwell period
    dwell_lengths = dwell_ends - dwell_starts
    
    # Label for each dwell period
    dwell_cluster_ids = labels[dwell_starts]
    
    return dwell_cluster_ids, dwell_lengths, dwell_starts

dwell_cluster_ids, dwell_lengths, dwell_starts = compute_dwell_times_np(kmeans_labels)
# Dwell time of each cluster in time series
# Compute the dwell time of each cluster
# Dwell time is the number of time points in each cluster
# Dwell time is the number of time points in each cluster
#%%
# Plot the dwell times
plt.figure(figsize=(12, 6))
plt.title('Dwell Times of Clusters')

dwell_time_per_cluster = [dwell_lengths[dwell_cluster_ids==xx] for xx in range(n_clusters)]

for i in range(n_clusters):
    plt.subplot(2, 2, i+1)
    plt.hist(dwell_time_per_cluster[i], bins=100, label=f"Cluster {i+1}", histtype='step')
    plt.xlabel('Dwell Time (Time Points)')
    plt.ylabel('Frequency')
    plt.title(f'Dwell Time Distribution for Cluster {i+1}')
    plt.legend()
# plt.hist(dwell_time_per_cluster[3], bins=100, label=[f"Cluster {i+1}" for i in range(n_clusters)], histtype='step')
plt.xlabel('Dwell Time (Time Points)')

# plt.bar(np.arange(len(dwell_lengths)), dwell_lengths, tick_label=[f"Cluster {i+1}" for i in range(len(dwell_lengths))])
# plt.xlabel('Cluster')
# plt.ylabel('Dwell Time (Time Points)')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# np.argpartition((kmeans_labels == 1)

# dwell_time = np.array([np.sum(kmeans_labels == k) for k in range(n_clusters)])

#%%




















