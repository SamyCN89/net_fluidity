
#%%
"""
This script is part of the fluidity analysis for the Julien Caillette dataset.
It loads raw timeseries data, cognitive data, and region labels,
and processes the data to compute DFC streams and speeds.
"""
from importlib import metadata
from math import e
from pathlib import Path
import numpy as np
import pandas as pd
import scipy as sp
import pickle

from shared_code.fun_paths import get_paths
from shared_code.fun_loaddata import (
    load_mat_timeseries, extract_mouse_ids, load_npz_dict, make_file_path, load_pickle, load_fc2_npz
)
from shared_code.shared_code.fun_loaddata import load_pickle
#%%
class DFCAnalysis:
    def __init__(self, dataset_name='julien_caillette'):
        self.paths = get_paths(
            dataset_name=dataset_name,
            timecourse_folder='time_courses',
            cognitive_data_file='mice_groups_comp_index.xlsx',
            anat_labels_file='all_ROI_coimagine.txt'
        )


        self.metadata = None
        self.ts_list = None
        self.ts_shapes = None
        self.ts_ids = None
        self.cog_data = None
        self.region_labels = None
        self.ts = None
        self.cog_data_filtered = None
        self.groups = None
        self.anat_labels = None
        self.n_animals = None
        self.total_tr = None
        self.regions = None

    # 1.1 Raw data loading
    def load_raw_timeseries(self):
        self.ts_list, self.ts_shapes, loaded_files = load_mat_timeseries(self.paths['timeseries'])
        self.ts_ids = extract_mouse_ids(loaded_files)
    
    # 1.2 Load raw cognitive data
    def load_raw_cognitive_data(self):
        self.cog_data = pd.read_excel(self.paths['cog_data'], sheet_name='mice_groups_comp_index')

    # 1.3 Load raw region labels
    def load_raw_region_labels(self):
        self.region_labels = np.loadtxt(self.paths['labels'], dtype=str).tolist()

    #-----------------------------------------------------------------------------------------------------
    # 2. Preprocessed data loading
    # 2.1 Metadata loading
    def get_metadata(self, meta_filename=None):
        preproc = Path(self.paths['preprocessed'])
        
        # Find the metadata pickle file
        if meta_filename is None:
            files = list(preproc.glob("metadata_animals_*.pkl"))
            if not files:
                raise FileNotFoundError("No metadata pickle file found in preprocessed directory.")
            meta_file = files[0]
        else:
            meta_file = preproc / meta_filename
        with open(meta_file, "rb") as f:
            metadata_dict = pickle.load(f)
        
        # Set attributes from dict
        self.metadata = metadata_dict
        self.cog_data_filtered = metadata_dict.get('mouse_metadata', None)
        self.region_labels_preprocessed = metadata_dict.get('region_labels', None)
        self.n_animals = metadata_dict.get('n_animals', None)
        self.regions = metadata_dict.get('regions', None)
        self.total_tr = metadata_dict.get('total_tr', None)
        self.filter_mode = metadata_dict.get('filter_mode', "unknown")
        print(f"Loaded metadata for {self.n_animals} animals from {meta_file.name}.")

    # 2.2 Preprocessed timeseries data loading
    def get_ts_preprocessed(self):
        data_ts_preprocessed = load_npz_dict(self.paths['preprocessed'] / Path(f'ts_filtered_animals_{self.n_animals}_regions_{self.regions}_tr_{self.total_tr}.npz'))
        self.ts = data_ts_preprocessed['ts']

    # 2.3 Cognitive data preprocessed loading
    def get_cogdata_preprocessed(self):
        self.cog_data_filtered = pd.read_csv(self.paths['preprocessed'] / Path(f"cog_data_filtered_animals_{self.n_animals}_regions_{self.regions}_tr_{self.total_tr}.csv"))
        self.groups = self.cog_data_filtered.groupby(['genotype', 'treatment']).groups

    # 2.4 Get preprocessed data
    def load_preprocessed_data(self):
        self.get_metadata()  # Load metadata first
        self.get_ts_preprocessed()  # Load preprocessed timeseries
        self.get_cogdata_preprocessed()  # Load preprocessed cognitive data

    #-----------------------------------------------------------------------------------------------------
    # 3.1 Analyzed data
    def get_temporal_parameters(self):
        """
        This method is a placeholder for loading or computing temporal parameters.
        It can be customized based on the specific requirements of the analysis.
        """
        # Example parameters, adjust as needed
        self.lag = self.metadata.get('lag', 1)  # Default to 1 if not found
        self.tau = self.metadata.get('tau', 3)  # Default to 3 if not found
        self.window_parameter = self.metadata.get('window_range', (5, 50, 1))  # Default to (5, 50, 1) if not found

        self.time_window_min, self.time_window_max, self.time_window_step = self.window_parameter
        self.time_window_range = np.arange(self.time_window_min,
                                           self.time_window_max + 1,
                                           self.time_window_step)
    # 3.2 Load 1 dfc window
    def load_dfc_1_window(self, lag=1, window=9):
        prefix = 'dfc'
        self.dfc_file_path = make_file_path(self.paths['dfc'], prefix, window, lag, self.n_animals, self.regions)
        results = load_npz_dict(self.dfc_file_path)
        # print(results.keys())
        self.dfc_stream = results[prefix]
        print(f"Loaded dfc stream for window size {window} with lag {lag} from {self.dfc_file_path.name}.")
        return self.dfc_stream

    # 3.3 Load dfc stream
    def load_dfc_stream(self, lag=1, tau=3, window_range=(5, 50, 1)):
        self.dfc_streams = {}
        time_window_range = np.arange(*window_range)
        for window_size in time_window_range:
            prefix = 'dfc'
            file_path = make_file_path(self.paths['dfc'], prefix, window_size, lag, self.n_animals, self.regions)
            results = load_npz_dict(file_path)
            self.dfc_streams[window_size] = self.load_dfc_1_window( lag, window_size)

    # 3.4 Load speed analysis
    def get_speed_analysis(self, tau_arange=np.arange(4), time_window_range=np.arange(5, 50+1, 1)):
        prefix='speed'
        file_path =self.paths['speed'] /  f"{prefix}_windows{len(time_window_range)}_tau{np.size(tau_arange)}_animals_{self.n_animals}.pkl"
        self.speed = load_pickle(file_path)
        # self.speeds_all = self.speed['speeds_all']

    # 3.5 Load speed fc analysis
    def get_speed_fc_analysis(self, tau_arange=np.arange(4), time_window_range = np.arange(5, 50+1, 1)):
        prefix = 'speed_fc'
        file_path = self.paths['speed'] / f"{prefix}_windows{len(time_window_range)}_tau{np.size(tau_arange)}_animals_{self.n_animals}.npz"
        self.speed_fc = load_fc2_npz(file_path)

    # def load_speed_fc_analysis(self, tau=3, window_range=(5, 50, 1)):
    #     time_window_range = np.arange(*window_range)
    #     for window_size in time_window_range:
    #         prefix = 'speed'
    #         file_path = make_file_path(self.paths['speed'], prefix, window_size, tau, self.n_animals, self.regions)
    #         results = load_npz_dict(file_path)

#%%
        #     self.speeds.append(results['speed'])
        #     self.fc_speeds.append(results['fc2'])
        # # Example post-processing:
        # self.speeds_all = [self.speeds[ws] for ws in np.arange(len(time_window_range))]
        # self.n_windows = len(self.speeds_all)
        # self.n_animals = self.speeds_all[0].shape[0]
        # ...you can add more post-processing as in your script...
        
    # def load_speed_analysis(self, tau=3, window_range=(5, 50, 1)):
    # 3.5 Load fc stream
    # def load_fc_stream(self, lag=1, tau=3, window_range=(5, 50, 1)):
    
    # Add more methods as needed for other processing steps*
    # 3.5 Load Metaconnectivity
    # 3.6 Load allegiance
    # 3.7 Load trimers
    
    # def load_metaconnectivity(self, lag=1, window=9):
    #     prefix = 'metaconnectivity'
    #     self.metaconnectivity_file_path = make_file_path(self.paths['metaconnectivity'], prefix, window, lag, self.n_animals, self.regions)
    #     results = load_npz_dict(self.metaconnectivity_file_path)
    #     self.metaconnectivity = results[prefix]

#-----------------------------------------------------------------------------------------------------
#%%
def example_usage():
    """
    Example usage of the DFCAnalysis class.
    This function demonstrates how to load and process the data.
    """
    analysis = DFCAnalysis()

    # Load raw data
    analysis.load_raw_timeseries()
    analysis.load_raw_cognitive_data()
    analysis.load_raw_region_labels()

    # Load preprocessed data
    analysis.load_preprocessed_data()

    # Load analyzed data
    analysis.get_temporal_parameters()
    analysis.load_dfc_1_window()
    analysis.load_dfc_stream()
    analysis.get_speed_analysis()
    analysis.get_speed_fc_analysis()
#-----------------------------------------------------------------------------------------------------

# example_usage()

# # # Example usage:
# analysis = DFCAnalysis()

# # #Raw data loading
# analysis.load_preprocessed_data()
# # analysis.load_raw_timeseries()
# # analysis.load_raw_cognitive_data()
# # analysis.load_raw_region_labels()

# # # Preprocessing data loading
# analysis.get_metadata()  # Load metadata only if needed
# analysis.get_ts_preprocessed()
# analysis.get_cogdata_preprocessed()


# # # Load analyzed data
# analysis.get_temporal_parameters()
# analysis.load_dfc_1_window()
# analysis.load_dfc_stream()
# analysis.load_speed_analysis()
# #-----------------------------------------------------------------------------------------------------

#%%


#%%








# #%% 1 F
# import numpy as np
# import matplotlib.pyplot as plt

# filtered_df = cog_data_filtered[cog_data_filtered['n_timepoints'] >= 500].reset_index(drop=True)
# groups = filtered_df.groupby(['genotype', 'treatment']).groups

# tau_count = 3  # adjust as needed
# animal_count = len(filtered_df)
# time_window_count = len(speeds_all)  # now using len instead of shape

# plt.figure(figsize=(8, 5))

# for group, animal_idxs in groups.items():
#     pooled = []
#     for animal_idx in animal_idxs:
#         for tau in range(tau_count):
#             for time_win in range(time_window_count):
#                 # speeds_all is a list: speeds_all[time_win][animal_idx * tau_count + tau, :]
#                 print(animal_idx, animal_idx * tau_count + tau)
#                 arr = speeds_all[time_win][animal_idx * tau_count + tau]
#                 arr = np.asarray(arr, dtype=float)
#                 arr = arr[~np.isnan(arr)]
#                 if arr.size > 0:
#                     pooled.append(arr)
#     if pooled:
#         group_speeds = np.concatenate(pooled)
#         if group_speeds.size > 0:
#             plt.hist(group_speeds, bins=100, alpha=0.5, 
#                      label=f"{group}", histtype='step', linewidth=1.7, density=True)

# plt.xlabel("DFC Speed")
# plt.ylabel("Density")
# plt.title("Histogram of DFC Speeds by Group (all tau pooled)")
# plt.legend()
# plt.tight_layout()
# plt.show()

# #%%
# import matplotlib.pyplot as plt


# filtered_df = cog_data_filtered[cog_data_filtered['n_timepoints'] >= 500]
# grouped = filtered_df.groupby(['genotype', 'treatment'])['index_NOR'].agg(['mean', 'sem', 'count']).reset_index()
# print(grouped)




# all_speeds = np.concatenate([speed for animal in speeds_all_T for speed in animal]).astype(np.float32)
# all_speeds = all_speeds[~np.isnan(all_speeds)]


# # Pool all tau for each animal (removing NaNs)
# pooled_speeds_per_animal = []
# for animal_speeds in speeds_all_T:
#     # animal_speeds is a list of arrays, one per tau
#     animal_pool = np.concatenate([
#         speed_arr[~np.isnan(speed_arr)] for speed_arr in animal_speeds if speed_arr is not None
#     ])
#     pooled_speeds_per_animal.append(animal_pool)

# # If you want all speeds across all animals (for a grand histogram or summary)
# all_speeds = np.concatenate(pooled_speeds_per_animal)

# plt.figure(figsize=(8,5))
# plt.hist(all_speeds, bins=175, color='skyblue', histtype='step', linewidth=1.5, density=True)
# plt.xlabel("Speed")
# plt.ylabel("Count")
# plt.title("Histogram of pooled DFC speeds (all animals, all windows)")
# plt.tight_layout()
# plt.show()

# # %% 2 F
# # =========  Histogram per group, all animals, all windows 



# import numpy as np
# import matplotlib.pyplot as plt

# filtered_df = cog_data_filtered[cog_data_filtered['n_timepoints'] >= 500].reset_index(drop=True)
# groups = filtered_df.groupby(['genotype', 'treatment']).groups

# plt.figure(figsize=(8, 5))

# for group, animal_idxs in groups.items():
#     pooled = []
#     for animal_idx in animal_idxs:
#         # For each tau for this animal, concatenate all non-nan speeds
#         animal_tau_speeds = [
#             speed_arr.astype(float)[~np.isnan(speed_arr)] 
#             for speed_arr in speeds_all_T[animal_idx] if speed_arr is not None
#         ]
#         if animal_tau_speeds:
#             pooled.append(np.concatenate(animal_tau_speeds))
#     if pooled:
#         group_speeds = np.concatenate(pooled)
#         if len(group_speeds) > 0:
#             plt.hist(group_speeds, bins=75, alpha=0.5, 
#                      label=f"{group}", histtype='step', linewidth=1.7, density=True)

# plt.xlabel("DFC Speed")
# plt.ylabel("Density")
# plt.title("Histogram of DFC Speeds by Group (all tau pooled)")
# plt.legend()
# plt.tight_layout()
# plt.show()


# #%%
# filtered_df = cog_data_filtered[cog_data_filtered['n_timepoints'] >= 500]
# groups = filtered_df.groupby(['genotype', 'treatment']).groups

# for group, animal_idxs in groups.items():
#     group_speeds = np.concatenate([
#         speed.astype(float)
#         for animal_idx in animal_idxs
#         for speed in speeds_all_T[animal_idx]
#     ])
#     group_speeds = group_speeds[~np.isnan(group_speeds)]
#     plt.hist(group_speeds, bins=50, alpha=0.5, label=f"{group}", histtype='step', linewidth=1.7, density=True)

# plt.xlabel("Speed")
# plt.ylabel("Count")
# plt.title("Histogram of DFC speeds by group")
# # plt.yscale('log')  # Log scale for better visibility
# plt.legend()
# plt.tight_layout()
# plt.show()

# # %%
# import matplotlib.pyplot as plt
# import seaborn as sns

# filtered_df = cog_data_filtered[cog_data_filtered['n_timepoints'] >= 500]
# groups = filtered_df.groupby(['genotype', 'treatment']).groups

# plt.figure(figsize=(10, 6))
# palette = sns.color_palette('tab10', n_colors=len(groups))

# for idx, (group, animal_idxs) in enumerate(groups.items()):
#     group_speeds = np.concatenate([
#         speed.astype(float)
#         for animal_idx in animal_idxs
#         for speed in speeds_all_T[animal_idx]
#     ])
#     group_speeds = group_speeds[~np.isnan(group_speeds)]
    
#     color = palette[idx]
#     plt.hist(group_speeds, bins=100, alpha=0.5, label=f"{group}", 
#              histtype='step', linewidth=1.7, density=True, color=color)
    
#     # Stats
#     median = np.median(group_speeds)
#     q05 = np.quantile(group_speeds, 0.05)
#     q95 = np.quantile(group_speeds, 0.95)
    
#     plt.axvline(median, color=color, linestyle='-', linewidth=1, 
#                 label=f"{group} median")
#     plt.axvline(q05, color=color, linestyle='--', linewidth=1, 
#                 label=f"{group} q=0.05/0.95")
#     plt.axvline(q95, color=color, linestyle='--', linewidth=1)

# plt.xlabel("Speed")
# plt.ylabel("Density")
# plt.title("Histogram of DFC speeds by group")
# plt.legend()
# plt.tight_layout()
# plt.show()

# # %%
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# filtered_df = cog_data_filtered[cog_data_filtered['n_timepoints'] >= 500]
# groups = filtered_df.groupby(['genotype', 'treatment']).groups

# plt.figure(figsize=(10, 6))
# palette = sns.color_palette('tab10', n_colors=len(groups))

# for idx, (group, animal_idxs) in enumerate(groups.items()):
#     group_speeds = np.concatenate([
#         speed.astype(float)
#         for animal_idx in animal_idxs
#         for speed in speeds_all_T[animal_idx]
#     ])
#     group_speeds = group_speeds[~np.isnan(group_speeds)]
    
#     color = palette[idx]
#     # Plot histogram
#     plt.hist(group_speeds, bins=100, alpha=0.5, label=f"{group}", 
#              histtype='step', linewidth=1.7, density=True, color=color)
#     # Median and quantiles
#     median = np.median(group_speeds)
#     q05 = np.quantile(group_speeds, 0.05)
#     q95 = np.quantile(group_speeds, 0.95)
#     # Vertical lines
#     plt.axvline(median, color=color, linestyle='-', linewidth=1.2, label=f"{group} median")
#     plt.axvline(q05, color=color, linestyle='--', linewidth=1, label=f"{group} q=0.05")
#     plt.axvline(q95, color=color, linestyle='--', linewidth=1, label=f"{group} q=0.95")

# plt.xlabel("Speed")
# plt.ylabel("Density")
# plt.title("Histogram of DFC speeds by group")
# plt.legend(fontsize=10, ncol=2)  # You can adjust ncol if legend is long
# plt.tight_layout()
# plt.show()

# # %%
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# filtered_df = cog_data_filtered[cog_data_filtered['n_timepoints'] >= 500]
# groups = filtered_df.groupby(['genotype', 'treatment']).groups

# plt.figure(figsize=(10, 6))
# palette = sns.color_palette('tab10', n_colors=len(groups))

# for idx, (group, animal_idxs) in enumerate(groups.items()):
#     group_speeds = np.concatenate([
#         speed.astype(float)
#         for animal_idx in animal_idxs
#         for speed in speeds_all_T[animal_idx]
#     ])
#     group_speeds = group_speeds[~np.isnan(group_speeds)]
    
#     color = palette[idx]
#     # Plot histogram (label for legend)
#     plt.hist(group_speeds, bins=75, alpha=0.5, label=f"{group}", 
#              histtype='step', linewidth=1.7, density=True, color=color)
#     # Median and quantiles (no label)
#     median = np.median(group_speeds)
#     q05 = np.quantile(group_speeds, 0.05)
#     q95 = np.quantile(group_speeds, 0.95)
#     plt.axvline(median, color=color, linestyle='-', linewidth=1.2)
#     plt.axvline(q05, color=color, linestyle='--', linewidth=1)
#     plt.axvline(q95, color=color, linestyle='--', linewidth=1)

# plt.xlabel("Speed")
# plt.ylabel("Density")
# plt.title("Histogram of DFC speeds by group")
# plt.legend(fontsize=10, ncol=2)
# plt.tight_layout()
# plt.show()

# # %%
# # window size/ median speed

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# filtered_df = cog_data_filtered[cog_data_filtered['n_timepoints'] >= 500]
# groups = filtered_df.groupby(['genotype', 'treatment']).groups

# window_sizes = time_window_range  # fill with your actual window sizes
# n_windows = len(speeds_all)  # Or len(window_sizes)


# palette = sns.color_palette('tab10', n_colors=len(groups))

# plt.figure(figsize=(10,6))

# for idx, (group, animal_idxs) in enumerate(groups.items()):
#     medians_per_window = []
#     for win_idx in range(n_windows):
#         # Pool all animals' speeds for this window and group
#         speeds_this_window = [
#             speeds_all[win_idx][animal_idx].astype(float)
#             for animal_idx in animal_idxs
#             if len(speeds_all[win_idx][animal_idx]) > 0  # skip empty arrays
#         ]
#         if speeds_this_window:
#             flat_speeds = np.concatenate(speeds_this_window)
#             flat_speeds = flat_speeds[~np.isnan(flat_speeds)]
#             median_speed = np.median(flat_speeds)
#         else:
#             median_speed = np.nan
#         medians_per_window.append(median_speed)
#     plt.plot(window_sizes, medians_per_window, marker='.', label=str(group), color=palette[idx])

# plt.xlabel("Time Window Size")
# plt.ylabel("Median Speed (all animals, pooled)")
# plt.title("Median DFC Speed vs. Window Size by Group")
# plt.legend(fontsize=10, ncol=2)
# plt.tight_layout()
# plt.show()

# # %%

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# filtered_df = cog_data_filtered[cog_data_filtered['n_timepoints'] >= 500]
# groups = filtered_df.groupby(['genotype', 'treatment']).groups

# n_windows = len(speeds_all)
# palette = sns.color_palette('tab10', n_colors=len(groups))

# plt.figure(figsize=(10,6))

# for idx, (group, animal_idxs) in enumerate(groups.items()):
#     medians_per_window = []
#     q25_per_window = []
#     q75_per_window = []
#     for win_idx in range(n_windows):
#         # Pool all animals' speeds for this window and group
#         speeds_this_window = [
#             speeds_all[win_idx][animal_idx].astype(float)
#             for animal_idx in animal_idxs
#             if len(speeds_all[win_idx][animal_idx]) > 0  # skip empty
#         ]
#         if speeds_this_window:
#             flat_speeds = np.concatenate(speeds_this_window)
#             flat_speeds = flat_speeds[~np.isnan(flat_speeds)]
#             median = np.median(flat_speeds)
#             q25 = np.quantile(flat_speeds, 0.25)
#             q75 = np.quantile(flat_speeds, 0.75)
#         else:
#             median = np.nan
#             q25 = np.nan
#             q75 = np.nan
#         medians_per_window.append(median)
#         q25_per_window.append(q25)
#         q75_per_window.append(q75)
#     color = palette[idx]
#     plt.plot(window_sizes, medians_per_window, marker='.', label=str(group), color=color)
#     plt.fill_between(window_sizes, q25_per_window, q75_per_window, color=color, alpha=0.1)

# plt.xlabel("Time Window Size")
# plt.ylabel("Median DFC Speed (all animals pooled)")
# plt.title("Median DFC Speed vs. Window Size by Group\nShaded = 25–75% quantile")
# plt.legend(fontsize=10, ncol=2)
# plt.tight_layout()
# plt.show()

# # %%

# quantile_levels = np.linspace(0.05, 0.95, 19)  # e.g. 0.05, 0.1, ..., 0.95
# window_sizes = time_window_range  # fill with your actual window sizes
# n_windows = len(window_sizes)
# n_q = len(quantile_levels)

# # Get group indices
# animal_idxs = list(groups.values())[0]  # Or select a specific group

# speed_matrix = np.zeros((n_q, n_windows))

# for win_idx in range(n_windows):
#     # Pool all speeds for this window and group
#     speeds_this_window = [
#         speeds_all[win_idx][animal_idx].astype(float)
#         for animal_idx in animal_idxs
#         if len(speeds_all[win_idx][animal_idx]) > 0
#     ]
#     if speeds_this_window:
#         flat_speeds = np.concatenate(speeds_this_window)
#         flat_speeds = flat_speeds[~np.isnan(flat_speeds)]
#         speed_matrix[:, win_idx] = [np.quantile(flat_speeds, q) for q in quantile_levels]
#     else:
#         speed_matrix[:, win_idx] = np.nan

# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 5))
# im = plt.imshow(
#     np.log(speed_matrix), 
#     aspect='auto', 
#     origin='lower', 
#     extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],
#     cmap='viridis'
# )
# plt.colorbar(im, label='DFC Speed')
# plt.clim(np.nanmin(np.log(speed_matrix)), np.nanmax(np.log(speed_matrix)))  # Set color limits to avoid NaNs affecting the colormap
# plt.xlabel('Window Size')
# plt.ylabel('Quantile')
# plt.title('Speed quantile matrix\n(Group: {})'.format(list(groups.keys())[0]))
# plt.tight_layout()
# plt.show()

# # %%

# import matplotlib.colors as mcolors
# speed_matrix_clipped = np.clip(speed_matrix, 1e-10, None)
# log_speed_matrix = np.log(speed_matrix_clipped)

# plt.figure(figsize=(10, 5))
# im = plt.imshow(
#     speed_matrix, 
#     aspect='auto', 
#     origin='lower', 
#     extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],
#     cmap='viridis',
#     norm=mcolors.LogNorm(vmin=np.nanmin(speed_matrix[speed_matrix > 0]), vmax=np.nanmax(speed_matrix))
# )
# plt.colorbar(im, label='DFC Speed (log scale)')
# plt.xlabel('Window Size')
# plt.ylabel('Quantile')
# plt.title('Speed quantile matrix\n(Group: {})'.format(list(groups.keys())[0]))
# plt.tight_layout()
# plt.show()


# # %%

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib.colors as mcolors

# filtered_df = cog_data_filtered[cog_data_filtered['n_timepoints'] >= 500]
# groups = filtered_df.groupby(['genotype', 'treatment']).groups

# quantile_levels = np.linspace(0.05, 0.95, 19)      # or your preferred quantiles
# window_sizes = time_window_range  # fill with your actual window sizes
# n_windows = len(window_sizes)
# n_q = len(quantile_levels)

# for group, animal_idxs in groups.items():
#     speed_matrix = np.zeros((n_q, n_windows))
#     for win_idx in range(n_windows):
#         speeds_this_window = [
#             speeds_all[win_idx][animal_idx].astype(float)
#             for animal_idx in animal_idxs
#             if len(speeds_all[win_idx][animal_idx]) > 0
#         ]
#         if speeds_this_window:
#             flat_speeds = np.concatenate(speeds_this_window)
#             flat_speeds = flat_speeds[~np.isnan(flat_speeds)]
#             # Ensure only positive values (lognorm needs >0)
#             flat_speeds = flat_speeds[flat_speeds > 0]
#             if flat_speeds.size > 0:
#                 speed_matrix[:, win_idx] = [np.quantile(flat_speeds, q) for q in quantile_levels]
#             else:
#                 speed_matrix[:, win_idx] = np.nan
#         else:
#             speed_matrix[:, win_idx] = np.nan

#     # Plot
#     plt.figure(figsize=(10, 5))
#     # Avoid zeros/NaNs for lognorm
#     valid = ~np.isnan(speed_matrix) & (speed_matrix > 0)
#     vmin = np.nanmin(speed_matrix[valid]) if np.any(valid) else 1e-6
#     vmax = np.nanmax(speed_matrix[valid]) if np.any(valid) else 1
#     im = plt.imshow(
#         speed_matrix, 
#         aspect='auto', 
#         origin='lower', 
#         extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],
#         cmap='viridis',
#         norm=mcolors.LogNorm(vmin=vmin, vmax=vmax)
#     )
#     plt.colorbar(im, label='DFC Speed (log scale)')
#     plt.xlabel('Window Size')
#     plt.ylabel('Quantile')
#     plt.title(f'Speed quantile matrix\nGroup: {group}')
#     plt.tight_layout()
#     plt.show()

# # %%

# import numpy as np
# import matplotlib.pyplot as plt

# filtered_df = cog_data_filtered[cog_data_filtered['n_timepoints'] >= 500]
# groups = filtered_df.groupby(['genotype', 'treatment']).groups

# quantile_levels = np.linspace(0, 1, 100)      # e.g. 19 quantiles
# window_sizes = time_window_range  # fill with your actual window sizes
# n_windows = len(window_sizes)
# n_q = len(quantile_levels)
# group_names = list(groups.keys())
# n_groups = len(group_names)

# speed_matrices = []

# for group in group_names:
#     animal_idxs = groups[group]
#     speed_matrix = np.zeros((n_q, n_windows))
#     for win_idx in range(n_windows):
#         speeds_this_window = [
#             speeds_all[win_idx][animal_idx].astype(float)
#             for animal_idx in animal_idxs
#             if len(speeds_all[win_idx][animal_idx]) > 0
#         ]
#         if speeds_this_window:
#             flat_speeds = np.concatenate(speeds_this_window)
#             flat_speeds = flat_speeds[~np.isnan(flat_speeds)]
#             if flat_speeds.size > 0:
#                 speed_matrix[:, win_idx] = [np.quantile(flat_speeds, q) for q in quantile_levels]
#             else:
#                 speed_matrix[:, win_idx] = np.nan
#         else:
#             speed_matrix[:, win_idx] = np.nan
#     speed_matrices.append(speed_matrix)

# # %%
# # Insert a row of nans between groups for clarity
# nan_gap = np.full((1, n_windows), np.nan)
# full_matrix = speed_matrices[0]
# yticks = []
# yticklabels = []

# for i, mat in enumerate(speed_matrices):
#     if i > 0:
#         full_matrix = np.vstack([full_matrix, nan_gap, mat])
#     # For labeling, put yticks at the center of each group's block
#     yticks.append(full_matrix.shape[0] - mat.shape[0]//2)
#     yticklabels.append(str(group_names[i]))

# # %%
# plt.figure(figsize=(12, 2.5 * n_groups))
# im = plt.imshow(
#     full_matrix,
#     aspect='auto',
#     origin='lower',
#     extent=[window_sizes[0], window_sizes[-1], 0, full_matrix.shape[0]],
#     cmap='viridis'
# )
# plt.colorbar(im, label='DFC Speed')
# plt.xlabel('Window Size')
# plt.ylabel('Quantile / Group')
# # Add group labels on y axis
# plt.yticks(yticks, yticklabels)
# plt.title('Speed quantile matrices, all groups stacked')
# plt.tight_layout()
# plt.show()

# # %%
# import numpy as np
# import matplotlib.pyplot as plt

# filtered_df = cog_data_filtered[cog_data_filtered['n_timepoints'] >= 500]
# groups = filtered_df.groupby(['genotype', 'treatment']).groups

# quantile_levels = np.linspace(0, 1, 100)
# n_windows = len(window_sizes)
# n_q = len(quantile_levels)

# group_names = list(groups.keys())
# n_groups = len(group_names)
# n_rows = n_cols = 2  # For 2x2 grid, adjust as needed

# fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 10), sharex=True, sharey=True)
# axes = axes.flatten()

# vmin = np.inf
# vmax = -np.inf

# # First: compute all matrices and global min/max for consistent color scale
# speed_matrices = []
# for group in group_names:
#     animal_idxs = groups[group]
#     speed_matrix = np.zeros((n_q, n_windows))
#     for win_idx in range(n_windows):
#         speeds_this_window = [
#             speeds_all[win_idx][animal_idx].astype(float)
#             for animal_idx in animal_idxs
#             if len(speeds_all[win_idx][animal_idx]) > 0
#         ]
#         if speeds_this_window:
#             flat_speeds = np.concatenate(speeds_this_window)
#             flat_speeds = flat_speeds[~np.isnan(flat_speeds)]
#             if flat_speeds.size > 0:
#                 speed_matrix[:, win_idx] = [np.quantile(flat_speeds, q) for q in quantile_levels]
#             else:
#                 speed_matrix[:, win_idx] = np.nan
#         else:
#             speed_matrix[:, win_idx] = np.nan
#     vmin = min(vmin, np.nanmin(speed_matrix))
#     vmax = max(vmax, np.nanmax(speed_matrix))
#     speed_matrices.append(speed_matrix)

# # Plot
# for idx, group in enumerate(group_names):
#     ax = axes[idx]
#     im = ax.imshow(
#         speed_matrices[idx],
#         aspect='auto',
#         origin='lower',
#         extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],
#         cmap='magma',
#         vmin=vmin, vmax=vmax
#     )
#     ax.set_title(f'Group: {group}')
#     ax.set_xlabel('Window Size')
#     ax.set_ylabel('Quantile')
#     ax.label_outer()  # Only show outer labels

# fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04, label='DFC Speed')
# plt.tight_layout()
# plt.show()

# # %%

# import numpy as np
# import matplotlib.pyplot as plt

# filtered_df = cog_data_filtered[cog_data_filtered['n_timepoints'] >= 500]
# groups = filtered_df.groupby(['genotype', 'treatment']).groups

# quantile_levels = np.linspace(0, 1, 20)
# n_windows = len(window_sizes)
# n_q = len(quantile_levels)

# group_names = list(groups.keys())[::-1]  # Reverse order here!
# n_groups = len(group_names)
# n_rows = n_cols = 2  # For 2x2 grid

# fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 10), sharex=True, sharey=True)
# axes = axes.flatten()

# vmin = np.inf
# vmax = -np.inf

# # Compute matrices and vmin/vmax
# speed_matrices = []
# for group in group_names:
#     animal_idxs = groups[group]
#     speed_matrix = np.zeros((n_q, n_windows))
#     for win_idx in range(n_windows):
#         speeds_this_window = [
#             speeds_all[win_idx][animal_idx].astype(float)
#             for animal_idx in animal_idxs
#             if len(speeds_all[win_idx][animal_idx]) > 0
#         ]
#         if speeds_this_window:
#             flat_speeds = np.concatenate(speeds_this_window)
#             flat_speeds = flat_speeds[~np.isnan(flat_speeds)]
#             if flat_speeds.size > 0:
#                 speed_matrix[:, win_idx] = [np.quantile(flat_speeds, q) for q in quantile_levels]
#             else:
#                 speed_matrix[:, win_idx] = np.nan
#         else:
#             speed_matrix[:, win_idx] = np.nan
#     vmin = min(vmin, np.nanmin(speed_matrix))
#     vmax = max(vmax, np.nanmax(speed_matrix))
#     speed_matrices.append(speed_matrix)

# # Plot
# for idx, group in enumerate(group_names):
#     ax = axes[idx]
#     im = ax.imshow(
#         speed_matrices[idx],
#         aspect='auto',
#         origin='lower',
#         extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],
#         cmap='magma',
#         vmin=vmin, vmax=vmax
#     )
#     ax.set_title(f'Group: {group}')
#     ax.set_xlabel('Window Size')
#     ax.set_ylabel('Quantile')
#     ax.label_outer()

# # Place colorbar on the left
# fig.subplots_adjust(left=0.15, right=0.95)  # Give space for colorbar
# cbar_ax = fig.add_axes([0.05, 0.25, 0.02, 0.5])  # [left, bottom, width, height]
# fig.colorbar(im, cax=cbar_ax, orientation='vertical', label='DFC Speed')

# plt.tight_layout(rect=[0.15, 0, 1, 1])  # Leave space on left for colorbar
# plt.show()


# # %%
# import numpy as np
# import matplotlib.pyplot as plt

# diff_AB = speed_matrices[0] - speed_matrices[1]   # A - B
# diff_AC = speed_matrices[0] - speed_matrices[2]   # A - C
# diff_AD = speed_matrices[0] - speed_matrices[3]   # A - D   
# diff_BC = speed_matrices[1] - speed_matrices[2]   # B - C
# diff_BD = speed_matrices[1] - speed_matrices[3]   # B - D
# diff_CD = speed_matrices[2] - speed_matrices[3]   # C - D

# fig = plt.figure(figsize=(20, 18))
# gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1])  # 3 rows, 3 cols

# diff_vmax = np.nanmax(np.abs([diff_AB, diff_CD, diff_AC, diff_BC, diff_BD, diff_AD]))
# diff_cmap = 'bwr'
# # Row 1: Groups A and B and their difference
# ax1 = fig.add_subplot(gs[0, 0])
# im1 = ax1.imshow(speed_matrices[0], aspect='auto', origin='lower',
#                  extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],
#                  cmap='magma', vmin=vmin, vmax=vmax)
# ax1.set_title(f'Group: {group_names[0]}')
# ax1.set_ylabel('Quantile')

# ax2 = fig.add_subplot(gs[0, 1])
# im2 = ax2.imshow(speed_matrices[1], aspect='auto', origin='lower',
#                  extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],
#                  cmap='magma', vmin=vmin, vmax=vmax)
# ax2.set_title(f'Group: {group_names[1]}')

# ax3 = fig.add_subplot(gs[0, 2])
# im3 = ax3.imshow(diff_AB, aspect='auto', origin='lower',
#                  extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],
#                  cmap=diff_cmap, vmin=-diff_vmax, vmax=diff_vmax)
# ax3.set_title(f'Diff: {group_names[0]} - {group_names[1]}')
# ax3.set_ylabel('Quantile')

# # Row 2: Difference A-B | Difference C-D
# ax5 = fig.add_subplot(gs[1, 0])
# im5 = ax5.imshow(speed_matrices[2], aspect='auto', origin='lower',
#                  extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],
#                  cmap='magma', vmin=vmin, vmax=vmax)
# ax5.set_title(f'Group: {group_names[2]}')
# ax5.set_ylabel('Quantile')

# ax6 = fig.add_subplot(gs[1, 1])
# im6 = ax6.imshow(speed_matrices[3], aspect='auto', origin='lower',
#                  extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],
#                  cmap='magma', vmin=vmin, vmax=vmax)
# ax6.set_title(f'Group: {group_names[3]}')

# ax4 = fig.add_subplot(gs[1, 2])
# im4 = ax4.imshow(diff_CD, aspect='auto', origin='lower',
#                  extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],
#                  cmap=diff_cmap, vmin=-diff_vmax, vmax=diff_vmax)
# ax4.set_title(f'Diff: {group_names[2]} - {group_names[3]}')


# # Row 3: Groups C and D
# # Row 4: Difference A-C (spans both columns)
# ax7 = fig.add_subplot(gs[2, 0])
# im7 = ax7.imshow(diff_AC, aspect='auto', origin='lower',
#                  extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],
#                  cmap=diff_cmap, vmin=-diff_vmax, vmax=diff_vmax)
# ax7.set_title(f'Diff: {group_names[0]} - {group_names[2]}')
# ax7.set_xlabel('Window Size')
# ax7.set_ylabel('Quantile')

# ax8 = fig.add_subplot(gs[2, 1])
# im8 = ax8.imshow(diff_BD, aspect='auto', origin='lower',
#                  extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],
#                  cmap=diff_cmap, vmin=-diff_vmax, vmax=diff_vmax)
# ax8.set_title(f'Diff: {group_names[1]} - {group_names[3]}')
# ax8.set_xlabel('Window Size')

# ax9 = fig.add_subplot(gs[2, 2])
# im9 = ax9.imshow(diff_AD, aspect='auto', origin='lower',
#                  extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],
#                  cmap=diff_cmap, vmin=-diff_vmax, vmax=diff_vmax)
# ax9.set_title(f'Diff: {group_names[0]} - {group_names[3]}')
# ax9.set_xlabel('Window Size')
# # Adjust layout
# # fig.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbars

# # Shared colorbars
# cbar_ax1 = fig.add_axes([0.92, 0.65, 0.015, 0.27])  # [left, bottom, width, height]
# fig.colorbar(im1, cax=cbar_ax1, orientation='vertical', label='DFC Speed')

# cbar_ax2 = fig.add_axes([0.92, 0.12, 0.015, 0.35])
# fig.colorbar(im3, cax=cbar_ax2, orientation='vertical', label='DFC Speed Diff')

# plt.tight_layout(rect=[0, 0, 0.91, 1])
# plt.show()

# # %%

# %%
