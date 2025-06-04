#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:26:30 2024

@author: samy
"""
#%%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# import sys
# sys.path.append('../shared_code')

from shared_code.fun_loaddata import extract_hash_numbers, load_mat_timeseries, extract_mouse_ids
from shared_code.fun_utils import filename_sort_mat, load_matdata, classify_phenotypes, make_combination_masks, make_masks
from shared_code.fun_paths import get_paths
import time
from scipy.io import loadmat
import re
#%%

# =============================================================================
# This code loads the time series data and cognitive data from files
# =============================================================================

# ------------------------Configuration------------------------

paths = get_paths(dataset_name='julien_caillette', 
                  timecourse_folder='time_courses',
                  cognitive_data_file='mice_groups_comp_index.xlsx',
                  anat_labels_file='all_ROI_coimagine.txt',)

# %%
# -----------------------------------------------------------------------------

def main():
    # Load all time series data
    ts_list, ts_shapes, loaded_files = load_mat_timeseries(paths['timeseries'])
    # Check if all loaded files have the same shape
    if len(set(ts_shapes)) > 1:
        print("Warning: Not all loaded files have the same shape.")
    # Extract mouse IDs from filenames
    ts_ids = extract_mouse_ids(loaded_files)

    # =============================================================================
    # Load cognitive data from .xlsx document
    # =============================================================================
    #Load cognitive data
    cog_data     = pd.read_excel(paths['cog_data'], sheet_name='mice_groups_comp_index')
    cog_data['mouse'] = cog_data['mouse'].astype(str) #Ensure mouse IDs are strings

    #Region labels are loaded from a text file
    region_labels        = np.loadtxt(paths['labels'], dtype=str).tolist()
    region_labels_clean = [label.replace("Both_", "") for label in region_labels]

    # =============================================================================
    # Filter time series data to include only those with mouse IDs present in cognitive data   
    # =============================================================================
    matched_ids = [mid for mid in ts_ids 
                   if mid in cog_data['mouse'].values]

    # Filter cognitive data to include only matched mouse IDs and sorted by mouse IDs
    cog_data_filtered = cog_data.set_index('mouse').loc[matched_ids].reset_index()

    #List of time series that match the mouse IDs in the cognitive data, preserving the order
    ts_filtered = [ts for ts, id_ in zip(ts_list, ts_ids) if id_ in matched_ids]

    # Print excluded mice
    excluded_ts_ids = [mid for mid in ts_ids if mid not in cog_data['mouse'].values]
    excluded_cog_ids = [mid for mid in cog_data['mouse'].astype(str).values if mid not in ts_ids]

    print("Mice with time series but NO cognitive data:", excluded_ts_ids)
    print("Mice with cognitive data but NO time series:", excluded_cog_ids)

    # Check if the number of time series matches the number of cognitive data entries
    if len(ts_filtered) != len(cog_data_filtered):
        raise ValueError("Mismatch in time series and cognitive data entries.")

    # Animals and cognitive data paramaters    
    n_animals = len(ts_filtered)
    total_tp, regions = ts_filtered[0].shape

    # Extract group features
    split_grp = cog_data_filtered["grp"].str.split("_", expand=True)
    cog_data_filtered["genotype"] = split_grp[0]
    cog_data_filtered["treatment"] = split_grp[1]
    cog_data_filtered = pd.concat([cog_data_filtered,
                                    pd.get_dummies(split_grp[0]),
                                    pd.get_dummies(split_grp[1])], axis=1)

    print(f"Matched {len(matched_ids)} mice")
    print(f"Region labels loaded: {len(region_labels_clean)}")
    print(f"Filtered cognitive data shape: {cog_data_filtered.shape}")

    #add the number of time points of each animal
    cog_data_filtered['n_timepoints'] = [ts.shape[0] for ts in ts_filtered]

    # Save processed data
    if all(ts.shape == ts_filtered[0].shape for ts in ts_filtered):
        ts_array = np.stack(ts_filtered)
        np.savez(paths['preprocessed'] / "ts_filtered.npz", 
                 ts=ts_array,
                 n_animals=n_animals,
                 total_tp=total_tp,
                 regions=regions,
                 anat_labels = region_labels_clean)
        print(f"Saved: ts_filtered.npz with shape {ts_array.shape}")
    else:
        np.savez(paths['preprocessed'] / "ts_filtered_unstacked.npz", 
                                ts=np.array(ts_filtered, dtype=object),
                                n_animals=n_animals,
                                total_tp=total_tp,
                                regions=regions,
                                anat_labels = region_labels_clean)
        print("Saved: ts_filtered_unstacked.npz")

    cog_data_filtered.to_csv(paths['preprocessed'] / "cog_data_filtered.csv", index=False)
    print("Saved: cog_data_filtered.csv")
    return ts_filtered, cog_data_filtered

if __name__ == "__main__":
    ts_filtered, cog_data_filtered = main()

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Group by genotype and treatment, calculate mean and SEM of NOR
grouped = cog_data_filtered.groupby(['genotype', 'treatment'])['index_NOR'].agg(['mean', 'sem', 'count']).reset_index()
print(grouped)

# %%

from statannotations.Annotator import Annotator

plt.figure(figsize=(8, 6))
ax =sns.violinplot(
    data=cog_data_filtered,
    x='genotype',
    y='index_NOR',
    hue='treatment',
    split=True,        # <--- split by hue, only works for two treatments!
    inner='quartile',  # shows median and quartiles inside
    linewidth=1.2,
    palette='pastel'
)
plt.title('NOR values by Genotype, split by Treatment')
plt.ylabel('NOR')
plt.xlabel('Genotype')
plt.legend(title='Treatment')
plt.tight_layout()

sns.stripplot(
    data=cog_data_filtered,
    x='genotype',
    y='index_NOR',
    hue='treatment',
    dodge=True,
    color='k',
    alpha=0.4,
    linewidth=0
)
handles, labels = plt.gca().get_legend_handles_labels()
n = len(set(cog_data_filtered['treatment']))
plt.legend(handles[:n], labels[:n], title='Treatment')

# === Statistical annotation code goes here! ===

# Define the pairs you want to compare:
# pairs = [
#     (('WT', 'VEH'), ('WT', 'LCTB92')),         # Compare VEH vs LCTB92 within WT
#     (('Dp1Yey', 'VEH'), ('Dp1Yey', 'LCTB92'))  # Compare VEH vs LCTB92 within Dp1Yey
# ]

# pairs = [
#     (('WT', 'VEH'), ('Dp1Yey', 'VEH')),       # VEH: WT vs Dp1Yey
#     (('WT', 'LCTB92'), ('Dp1Yey', 'LCTB92')) # LCTB92: WT vs Dp1Yey
# ]

pairs = [
    (('WT', 'VEH'), ('WT', 'LCTB92')),         # Compare VEH vs LCTB92 within WT
    (('Dp1Yey', 'VEH'), ('Dp1Yey', 'LCTB92')), # Compare VEH vs LCTB92 within Dp1Yey
    (('WT', 'VEH'), ('Dp1Yey', 'VEH')),        # Compare WT VEH vs Dp1Yey VEH
    (('WT', 'LCTB92'), ('Dp1Yey', 'LCTB92')),  # Compare WT LCTB92 vs Dp1Yey LCTB92
    (('WT', 'LCTB92'), ('Dp1Yey', 'VEH'))     # Compare WT LCTB92 vs Dp1Yey VEH
]

# Create the Annotator object
annotator = Annotator(
    ax,
    pairs,
    data=cog_data_filtered,
    x='genotype',
    y='index_NOR',
    hue='treatment'
)

# Configure the test you want to use (t-test or Mann-Whitney), and how to display
annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose=2)

# Apply the annotation (calculates and adds stars/lines)
annotator.apply_and_annotate()

# === End of statistical annotation code ===

plt.show()

# %%


# =============================================================================
# Kruskal-Wallis test
# =============================================================================
# Filter the cognitive data to include only relevant columns
cog_data_filtered['group'] = cog_data_filtered['genotype'] + "_" + cog_data_filtered['treatment']

from scipy.stats import kruskal

# Get the list of unique groups
groups = cog_data_filtered['group'].unique()

# Gather the index_NOR values for each group into a list
group_values = [cog_data_filtered[cog_data_filtered['group'] == g]['index_NOR'].values for g in groups]

# Run Kruskal–Wallis test
stat, p = kruskal(*group_values)
print("Kruskal-Wallis H-test result:")
print(f"Statistic: {stat:.4f}, p-value: {p:.4g}")

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import itertools

pairs = list(itertools.combinations(groups, 2))
pvals = []
for g1, g2 in pairs:
    vals1 = cog_data_filtered[cog_data_filtered['group'] == g1]['index_NOR']
    vals2 = cog_data_filtered[cog_data_filtered['group'] == g2]['index_NOR']
    stat, p = mannwhitneyu(vals1, vals2, alternative='two-sided')
    pvals.append(p)
    print(f"{g1} vs {g2}: p={p:.4g}")

# FDR-correct the p-values for multiple comparisons
reject, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')
print("\nFDR-corrected p-values:")
for (g1, g2), p_corr, sig in zip(pairs, pvals_corrected, reject):
    print(f"{g1} vs {g2}: corrected p={p_corr:.4g}, {'significant' if sig else 'ns'}")

# %%
# Helper to map combined group name back to (genotype, treatment)
def group_to_tuple(group):
    genotype, treatment = group.split('_', 1)
    return (genotype, treatment)

pairs_statann = [(group_to_tuple(g1), group_to_tuple(g2)) for g1, g2 in pairs]

import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator

plt.figure(figsize=(10, 7))
ax = sns.violinplot(
    data=cog_data_filtered,
    x='genotype',
    y='index_NOR',
    hue='treatment',
    split=True,
    inner='quartile',
    linewidth=1.2,
    palette='pastel'
)
sns.stripplot(
    data=cog_data_filtered,
    x='genotype',
    y='index_NOR',
    hue='treatment',
    dodge=True,
    color='k',
    alpha=0.4,
    linewidth=0
)

# Fix duplicate legends
handles, labels = ax.get_legend_handles_labels()
n = len(cog_data_filtered['treatment'].unique())
ax.legend(handles[:n], labels[:n], title='Treatment')

# Prepare annotation labels for the plot
star_labels = []
for sig, p_corr in zip(reject, pvals_corrected):
    if p_corr < 0.001:
        star = '***'
    elif p_corr < 0.01:
        star = '**'
    elif p_corr < 0.05:
        star = '*'
    else:
        star = 'ns'
    star_labels.append(star)

# Annotate all pairwise group comparisons
annotator = Annotator(
    ax,
    pairs_statann,
    data=cog_data_filtered,
    x='genotype',
    y='index_NOR',
    hue='treatment'
)
annotator.set_pvalues_and_annotate(pvals_corrected)
plt.title('NOR values by Genotype and Treatment\n(Kruskal-Wallis: p={:.3g})'.format(p))
plt.tight_layout()
plt.show()

# %%
