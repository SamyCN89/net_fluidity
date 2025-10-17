

#%%

from cProfile import label
import time
import numpy as np
import matplotlib.pyplot as plt
from scripts import speed
from shared_code.fun_utils import load_cognitive_data, set_figure_params
from shared_code.fun_paths import get_paths
from shared_code.fun_loaddata import load_timeseries_bundle

save_fig = set_figure_params(False)

timecourse_folder = "Timecourses_updated_03052024"
paths = get_paths(
    dataset_name="ines_abdullah",
    timecourse_folder=timecourse_folder,
    cognitive_data_file="ROIs.xlsx",
    anat_labels_file="41_Allen.txt",
)

bundle = load_timeseries_bundle(
    paths["preprocessed"] / "ts_and_meta_2m4m.npz",
    paths["preprocessed"] / "grouping_data_oip.pkl",
)
cog_data = load_cognitive_data(paths["preprocessed"] / "cog_data_sorted_2m4m.csv")

n_animals = bundle.n_animals
total_tr = bundle.total_tr
anat_labels = bundle.anat_labels
regions = bundle.n_regions
mask_groups = bundle.mask_groups
label_variables = bundle.label_variables


#%%
time_windows_range = np.arange(5,100,1)

# speed loading

speeds = []

for w in time_windows_range:
    filepath = f'/media/samy/Elements2/Proyectos/LauraHarsan/results/ines_abdullah/speed/all/all/speed_win{w}_lag1_tau1_animals_126_regions_41.npz'
    a = np.load(filepath, allow_pickle=True)
    s = a['speeds']
    # Flatten each animal’s array from (1, N) → (N,)
    s_flat = np.array([x.ravel() for x in s], dtype=object)
    speeds.append(s_flat)

    print(f"window {w}: n_animals={len(s_flat)}, len(speeds[0])={len(s_flat[0])}")

#%%
# Plotting mean speeds for each animal across different time windows
plt.figure(figsize=(8,6))
for i in range(n_animals):
    animal_speeds = [speeds[j][i] for j in range(len(time_windows_range))]
    mean_speeds = [np.mean(animal_speeds[j]) for j in range(len(time_windows_range))]
    plt.plot(time_windows_range, mean_speeds, '.-', label=f'Animal {i+1}')
plt.xlabel('Time Window Size')
plt.ylabel('Mean dFC Speed')
plt.title('dFC Speed vs Time Window Size for Each Animal')
# %%
# Pool all speed values from all windows:
all_speeds_flat = np.concatenate([np.concatenate([s.flatten() for s in speed])
                                  for speed in speeds])
([np.shape([s.flatten() for s in speed])
for speed in speeds])

animal_speeds = []
for i in range(n_animals):
    animal_s = [speeds[j][i] for j in range(len(time_windows_range))]
    animal_speeds.append(animal_s)
    # print(f"Animal {i+1}: n_speeds={np.shape(animal_s)}")


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

#%%
group_dict = {}
for i in range(len(mask_groups)):
    for lb, idx in zip(label_variables[i], mask_groups[i]):
        indice_mask = np.where(idx==True)[0]
        label_mask = lb
        if i==0:
            label_mask += ' OiP'
        group_dict[label_mask] = indice_mask
        print(group_dict[label_mask], label_mask)
#%%
label_var_updated = []
for i in range(len(label_variables)):
    aux_lab = []
    for lbl in label_variables[i]:
        if i==0:
            lbl += ' OiP'
        if i==1:
            lbl += ' NOR'
        print(lbl)
        aux_lab.append(lbl)
    label_var_updated.append(aux_lab)
#%%


plt.figure(figsize=(10, 6))
for label, indices in group_dict.items():
    group_speeds = []
    for animal_s in animal_speeds:
        group_vals = np.concatenate([animal_s[j][indices] for j in range(len(time_windows_range))])
        group_speeds.append(group_vals)
    group_speeds_flat = np.concatenate(group_speeds)
    plt.hist(group_speeds_flat,
             bins=100, alpha=0.6,
             histtype='step',
             density=True,
             label=label)
plt.title("Distribution of dFC Speed by Brain Region Group")
plt.xlabel("Speed")
plt.ylabel("Frequency")
plt.legend()


# %%



#%%
# import necessary functions
from pathlib import Path

from scripts.bootstrap.compute_speed_bootstrap import (
         BootstrapConfig,
         load_dataset_context,
         load_per_animal_from_npz,
         _list_window_files,
         _find_region_folders,
         _pool_windows_indices,
     )

cfg = BootstrapConfig(dataset_name="ines", subset="all", tau_index=3)
ctx = load_dataset_context(cfg.dataset_name, tr_hint=cfg.tr)
region_dirs = _find_region_folders(Path(ctx.paths["speed"]) / (cfg.subset or ""))
#%%
#   3. Grab the per–animal series you want to visualise

win_files = _list_window_files(region_dirs[0])  # choose a region, e.g. “regions-all”
window, npz_path = win_files[0]                 # pick a window size, e.g. 10 TR
per_animal = load_per_animal_from_npz(npz_path, tau_index=cfg.tau_index)

#%%
#   4. Pool values and draw the distribution

import matplotlib.pyplot as plt
from shared_code.fun_bootstrap import pool_per_animal

group_cols = cfg.group_cols_resolved or ["Genotype", "Sexe"]   # whatever you passed to the bootstrap
groups = build_groups_from_columns(ctx.cog_df, group_cols)

fig, ax = plt.subplots(figsize=(8, 5))
for grp_key, idxs in groups.items():
    vals = pool_per_animal(per_animal, idxs)
    if vals.size == 0:
        continue
    ax.hist(vals, bins=100, density=True, alpha=0.4, label=str(grp_key))
ax.set(title=f"dFC speed W={window} (tau={cfg.tau_index})", xlabel="Speed", ylabel="Density")
ax.legend()
#%%
  5. Optionally plot pooled windows

     pools = _pool_windows_indices([w for w, _ in win_files], threshold="median")
     # Load per-window arrays with load_per_animal_from_npz and concatenate with _concat_per_animal before plotting.
  6. Wrap it in a helper
     Add a function such as plot_ines_speed_distribution() to julien_data/speed_plots.py (or a new ines_speed_plots.py) that follows the steps above. Because
     julien_data is already full of quick scripts, that’s the best place to keep it lightweight. You can then call it from a notebook:

     from julien_data.speed_plots import plot_ines_speed_distribution
     plot_ines_speed_distribution(window_size=10, tau_index=3, subset="all")

     Inside that helper you can parameterise:
      - subset (None, "all", "regions-ACC-THAL", …)
      - window_size (choose from _list_window_files)
      - tau_index (-1 to pool all taus)
      - choice of histogram vs KDE (you already have plot_group_distributions that wraps seaborn/statsmodels if you prefer KDEs).
  7. If you prefer metaconnectivity
     Copy the same helper into a new file such as metaconnectivity/plot_speed_ines.py, but remember you still need to import shared_code.fun_paths and the helpers
     above. Nothing else changes; you just execute it f
