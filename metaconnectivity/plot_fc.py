#!/usr/bin/env python3
"""
Created on Wed Apr  2 02:59:41 2025

@author: samy
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from shared_code.fun_dfcspeed import ts2fc
from shared_code.fun_loaddata import load_timeseries_bundle

# from fun_utils import get_paths, set_figure_params
from shared_code.fun_paths import get_paths

# from shared_code.fun_utils import set_figure_params
from shared_code.fun_utils import load_cognitive_data, set_figure_params

# %%
# ========================= Figure parameters ================================
save_fig = set_figure_params(False)

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


# plt.hist((male_wt_data['OiP_2M'], male_wt_data['OiP_4M'], male_dki_data['OiP_2M'], male_dki_data['OiP_4M']), bins=4,
#          alpha=0.7,
#           histtype='step',
#          label=('Male WT 2M', 'Male WT 4M', 'Male dKI 2M', 'Male dKI 4M'))
plt.figure(1, figsize=(10, 6))
plt.clf()
plt.subplot(211)
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
plt.xticks([1, 2, 3, 4], ["WT 2M", "WT 4M", "dKI 2M", "dKI 4M"])
plt.axhline(0, c="k")
plt.axhline(0.2, c="gray", ls="--")
plt.axhline(-0.2, c="gray", ls="--")
plt.ylabel("OiP score")
plt.title("OiP task scores")
# label with legend male and female and color code C0 and C1

plt.legend(
    ["Male", "Female"],
    #    title='Group',
    #    title_fontsize='13',
    facecolor=("C0", "C1"),
    #    edgecolor='black',
    loc="upper right",
)
# plt.legend(label=['Male','Female'],loc='upper right')


plt.subplot(212)

plt.violinplot(
    (
        cog_data["RO24h_2M"][wt_ind],
        cog_data["RO24h_4M"][wt_ind],
        cog_data["RO24h_2M"][dki_ind],
        cog_data["RO24h_4M"][dki_ind],
    )
)
plt.xticks([1, 2, 3, 4], ["Male WT 2M", "Male WT 4M", "Male dKI 2M", "Male dKI 4M"])
plt.axhline(0, c="k")
plt.axhline(0.2, c="gray", ls="--")
plt.axhline(-0.2, c="gray", ls="--")

plt.ylabel("RO24h score")
plt.title("RO24h Task")
plt.legend()
plt.tight_layout()


# %%
# %%
# =============================================================================
# FC and modularity
# -There is another modularity algorithm which claims to be better than Louvain. It's called leiden algorithm
# https://www.nature.com/articles/s41598-019-41695-z
# =============================================================================


# %%
def sort_modularity(fc):
    import brainconn as bct

    # Modularity of Louvain
    # modules, louvain = bct.modularity.modularity_louvain_dir(fc)
    # modules, louvain = bct.modularity.modularity_louvain_dir(fc)
    modules, louvain = bct.modularity.modularity_louvain_und_sign(fc, gamma=1.2)
    # print(np.unique(modules),louvain)

    # sort accord the modularity
    sort_modules = np.argsort(modules)
    # print(sort_modules)
    fc_mod = fc[:, sort_modules][sort_modules, :]  # fc sorted by modularity

    return fc_mod


# Functional connectivity
fc = np.array([ts2fc(ts[xx]) for xx in range(n_animals)])  # (Animals, regions, regions)
# fc_4m = np.array([ts2fc(ts4m[xx]) for xx in range(n_animals)])

# Modularity
fc_mod = np.array([sort_modularity(fc[xx]) for xx in range(n_animals)])
# fc_4m_mod = np.array([sort_modularity(fc_4m[xx]) for xx in range(n_animals)])

# superior triangular (maybe fucntion)
ind_fctri = np.triu_indices(fc.shape[2], 1)
# ind_fctri_4m = np.triu_indices(fc_4m.shape[2],1)

tri = np.array([fc[tt, ind_fctri[0], ind_fctri[1]] for tt in range(n_animals)])
# tri_4m = np.array([fc_4m[tt, ind_fctri_4m[0], ind_fctri_4m[1]] for tt in range(n_animals)])

# %%


for idx_mice in range(n_animals // 2):
    # for idx_mice in range(2):

    aux_ts2m = ts[idx_mice]
    aux_ts4m = ts[idx_mice * 2]

    plt.figure(2, figsize=(10, 10))
    plt.clf()
    plt.subplot(321)
    plt.title(
        "2m mouse #%s %s %s"
        % (mouse_hash_cog[idx_mice], gen_label[idx_mice], sex_label[idx_mice])
    )
    plt.plot(aux_ts2m)
    plt.ylabel("Bold")
    plt.xlabel("time")

    plt.subplot(322)
    plt.title(
        "4m mouse #%s %s %s"
        % (mouse_hash_cog[idx_mice], gen_label[idx_mice], sex_label[idx_mice])
    )
    plt.plot(aux_ts4m)
    plt.ylabel("Bold")
    plt.xlabel("time")

    plt.subplot(323)
    plt.title("FC")
    plt.imshow(
        fc[idx_mice],
        aspect="auto",
        interpolation="none",
        cmap="RdBu_r",
        vmin=-0.5,
        vmax=0.5,
    )
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("regions")
    plt.ylabel("regions")

    plt.subplot(324)
    plt.title("FC")
    plt.imshow(
        fc[idx_mice * 2],
        aspect="auto",
        interpolation="none",
        cmap="RdBu_r",
        vmin=-0.5,
        vmax=0.5,
    )
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("regions")
    plt.ylabel("regions")

    plt.subplot(325)
    # Fit linear regression via least squares with numpy.polyfit
    # It returns an slope (b) and intercept (a)
    # deg=1 means linear fit (i.e. polynomial of degree 1)
    # Create sequence of 100 numbers from 0 to 100
    b, a = np.polyfit(tri[idx_mice], tri[idx_mice * 2], deg=1)
    xseq = np.linspace(-1, 1, num=100)

    plt.title("slope:%s" % np.round(b, 3))

    x = np.linspace(-1, 1, 1000)
    plt.plot(x, x, color="gray", lw=1, ls="--")  # 45 degree line
    plt.axhline(0, c="k", ls="--")
    plt.axvline(0, c="k", ls="--")

    plt.scatter(
        tri[idx_mice],
        tri[idx_mice * 2],
        facecolors="none",
        edgecolors="C3",
        marker="o",
        s=4,
        alpha=0.7,
    )
    # Plot regression line
    plt.plot(xseq, a + b * xseq, color="k", lw=2)

    plt.xlabel("2m", c="C1")
    plt.ylabel("4m", c="C0")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xticks(np.arange(-1, 1.1, 0.5), fontsize=12)
    plt.yticks(np.arange(-1, 1.1, 0.5), fontsize=12)

    plt.subplot(326)

    plt.hist((tri[idx_mice], tri[idx_mice * 2]), histtype="step", bins=30, density=True)
    plt.legend(("2m", "4m"))
    plt.xlabel("CC")
    plt.ylabel("Counts #")
    plt.tight_layout()

    # for idx_mice in range(n_animals):
    if save_fig == True:
        # plt.title('2m mouse #%s %s %s'%(mouse_hash_cog[idx_mice], gen_label[idx_mice], sex_label[idx_mice]))
        plt.savefig(
            paths["f_fc"]
            + f"mouse_#{mouse_hash_cog[idx_mice]}_{gen_label[idx_mice]}_{sex_label[idx_mice]}.pdf"
        )
        plt.savefig(
            paths["f_fc"]
            + f"mouse_#{mouse_hash_cog[idx_mice]}_{gen_label[idx_mice]}_{sex_label[idx_mice]}.png"
        )
        # plt.savefig('fig/fc/mouse_#%s.pdf'%mouse_hash_cog[idx_mice])
# %%

plt.figure(3)
plt.clf()

dki_index = np.concatenate((dki_ind, dki_ind))


plt.subplot(211)
b1, a1 = np.polyfit(
    (tri[dki_index[: (n_animals // 2)]]).flatten(),
    (tri[dki_index[n_animals // 2 :]].flatten()),
    deg=1,
)
b2, a2 = np.polyfit(
    (tri_2m[ctrl_index]).flatten(), (tri_4m[ctrl_index].flatten()), deg=1
)
xseq = np.linspace(-1, 1, num=100)


# plt.title('dKi vs ctrl slope:%s'%np.round(b,3))
plt.title("dKi vs ctrl ")
plt.scatter(
    tri_2m[dki_index],
    tri_4m[dki_index],
    marker=".",
    label="dki slope=%s" % np.round(b1, 3),
)
plt.scatter(
    tri_2m[ctrl_index],
    tri_4m[ctrl_index],
    marker=".",
    alpha=0.5,
    label="ctrl slope=%s" % np.round(b2, 3),
)

plt.plot(xseq, a1 + b1 * xseq, color="C0", lw=2.5)
plt.plot(xseq, a2 + b2 * xseq, color="C1", lw=2.5)
plt.xlabel("2m")
plt.ylabel("4m")
plt.xlim(-1, 1)
plt.ylim(-1, 1)

plt.legend()

plt.subplot(212)
plt.hist(
    (
        tri_2m[dki_index].flatten(),
        tri_2m[ctrl_index].flatten(),
        tri_4m[dki_index].flatten(),
        tri_4m[ctrl_index].flatten(),
    ),
    histtype="step",
    bins=50,
)
# plt.hist((tri_2m[dki_index], tri_2m[ctrl_index], tri_4m[dki_index], tri_4m[ctrl_index]),histtype='step',bins=50)
plt.legend(("2m_dki", "2m_ctrl", "4m_dki", "4m_ctrl"))
plt.xlabel("CC")
plt.ylabel("Counts #")

if save_fig == True:
    plt.savefig("fig/fc/fc_all_dki_vs_ctrl.png")
    plt.savefig("fig/fc/fc_all_dki_vs_ctrl.pdf")
