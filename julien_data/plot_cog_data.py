

#%%
from os import path
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from statannotations.Annotator import Annotator
import seaborn as sns
from scipy.stats import kruskal

from shared_code.fun_utils import set_figure_params
from shared_code.fun_paths import get_paths



paths = get_paths(dataset_name='julien_caillette', 
                  timecourse_folder='time_courses',
                  cognitive_data_file='mice_groups_comp_index.xlsx',
                  anat_labels_file='all_ROI_coimagine.txt')

# Load cognitive data
cog_data_filtered = pd.read_csv(paths['preprocessed'] / Path("cog_data_filtered.csv"))

savefig = set_figure_params(True)
(paths['f_cog'] / 'NOR').mkdir(parents=True, exist_ok=True)
#%%
# Group by genotype and treatment, calculate mean and SEM of NOR
grouped = cog_data_filtered.groupby(['genotype', 'treatment'])['index_NOR'].agg(['mean', 'sem', 'count']).reset_index()
print(grouped)

# %%

# Violinplot of the NOR index by genotype and treatment

plt.figure(1, figsize=(8, 6))
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

plt.savefig(paths['f_cog'] / 'NOR' / 'cog_data_violinplot.png', dpi=300, bbox_inches='tight') if savefig else None
plt.savefig(paths['f_cog'] / 'NOR' / 'cog_data_violinplot.pdf', dpi=300, bbox_inches='tight') if savefig else None

# === Statistical annotation code goes here! ===

# Define the pairs you want to compare:

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


plt.title('NOR values by Genotype and Treatment\n(Mann-Whitney U test, p-values < 0.05)')
plt.tight_layout()
plt.savefig(paths['f_cog'] / 'NOR' / 'cog_data_violinplot_Mann_Whitney.png', dpi=300, bbox_inches='tight') if savefig else None
plt.savefig(paths['f_cog'] / 'NOR' / 'cog_data_violinplot_Mann_Whitney.pdf', dpi=300, bbox_inches='tight') if savefig else None

plt.show()

# %%


# =============================================================================
# Kruskal-Wallis test
# =============================================================================
# Filter the cognitive data to include only relevant columns
cog_data_filtered['group'] = cog_data_filtered['genotype'] + "_" + cog_data_filtered['treatment']


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
plt.savefig(paths['f_cog'] / 'NOR' / 'cog_data_violinplot_Kruskal_Wallis.png', dpi=300, bbox_inches='tight') if savefig else None
plt.savefig(paths['f_cog'] / 'NOR' / 'cog_data_violinplot_Kruskal_Wallis.pdf', dpi=300, bbox_inches='tight') if savefig else None
plt.show()

# %%
