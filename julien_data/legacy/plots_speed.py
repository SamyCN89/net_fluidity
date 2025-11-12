# %% [markdown]
# jupytext: formats=ipynb,py:percent
# jupytext: text_representation={extension:".py",format_name:"percent",format_version:"1.3",jupytext_version:"1.16.0"}
# %%
#%%



import pickle

from pathlib import Path

from networkx import density

import numpy as np

from class_dataanalysis_julien import DFCAnalysis



data = DFCAnalysis()
# %%
#%%



# Load raw data

# data.load_raw_timeseries()

# data.load_raw_cognitive_data()

# data.load_raw_region_labels()



# Load preprocessed data

data.load_preprocessed_data()





data.get_temporal_parameters()
# %%
#%%



# Match these variables to your last run:

prefix = "speed"

save_path = data.paths['speed']  # <-- update this!

time_window_range = data.time_window_range           # <-- list of window sizes, same as in your analysis

tau_range = np.arange(0,data.tau+ 1)                   # <-- as above

n_animals = data.n_animals                # <-- as above

data.load_preprocessed_data()



window_file_total = save_path / f"{prefix}_windows{len(time_window_range)}_tau{len(tau_range)}_animals_{n_animals}.pkl"
# %%
#%%



with open(window_file_total, 'rb') as f:

    all_speed = pickle.load(f)



# Now all_speed is a list (or similar) with each entry for one window_size.

# The last one:

last_speed = all_speed[-1]  # This is the speed array for the last window size



# Example: print shape/info

print(f"Loaded speed for window {time_window_range[-1]}: shape = {last_speed.shape}")
# %%
# %%



#print the shape of each time windows

for i, speed in enumerate(all_speed):

    print(f"Window size {time_window_range[i]}: shape = {speed.shape}")
# %%
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
#%%



import numpy as np



n_taus = all_speed[0].shape[1]  # should be 4



# Prepare to collect all speeds per tau, across all animals, all windows, all timepoints

all_speeds_by_tau = [[] for _ in range(n_taus)]



for speed in all_speed:

    # speed.shape = (48, 4, N)

    for tau in range(n_taus):

        # Collect all speeds for this tau across all animals and timepoints for this window size

        speeds_tau = speed[:, tau, :].flatten()

        all_speeds_by_tau[tau].append(speeds_tau)



# Concatenate across windows

all_speeds_by_tau = [np.concatenate(speeds) for speeds in all_speeds_by_tau]



import matplotlib.pyplot as plt



plt.figure(figsize=(10, 7))

for tau, speeds in enumerate(all_speeds_by_tau):

    plt.hist(speeds, bins=120, alpha=0.4, histtype='stepfilled', 

             density=True, label=f"tau {tau}")



plt.title("Distribution of dFC Speed per tau (all windows pooled)")

plt.xlabel("Speed")

plt.ylabel("Density")

plt.legend()

plt.tight_layout()

plt.show()
# %%
#%%



import numpy as np



n_taus = all_speed[0].shape[1]  # should be 4



# Prepare to collect all speeds per tau, across all animals, all windows, all timepoints

all_speeds_by_tau = [[] for _ in range(n_taus)]



for speed in all_speed:

    # speed.shape = (48, 4, N)

    for tau in range(n_taus):

        # Collect all speeds for this tau across all animals and timepoints for this window size

        speeds_tau = speed[:, tau, :].flatten()

        all_speeds_by_tau[tau].append(speeds_tau)



# Concatenate across windows

all_speeds_by_tau = [np.concatenate(speeds) for speeds in all_speeds_by_tau]



import matplotlib.pyplot as plt



plt.figure(figsize=(10, 7))

for tau, speeds in enumerate(all_speeds_by_tau):

    plt.hist(speeds, bins=120, alpha=0.4, histtype='step', 

             density=True, label=f"tau {tau}")



plt.title("Distribution of dFC Speed per tau (all windows pooled)")

plt.xlabel("Speed")

plt.ylabel("Density")

plt.legend()

plt.tight_layout()

plt.show()
# %%
#%%



import numpy as np



n_taus = all_speed[0].shape[1]  # should be 4



# Prepare to collect all speeds per tau, across all animals, all windows, all timepoints

all_speeds_by_tau = [[] for _ in range(n_taus)]



for speed in all_speed:

    # speed.shape = (48, 4, N)

    for tau in range(n_taus):

        # Collect all speeds for this tau across all animals and timepoints for this window size

        speeds_tau = speed[:, tau, :].flatten()

        all_speeds_by_tau[tau].append(speeds_tau)



# Concatenate across windows

all_speeds_by_tau = [np.concatenate(speeds) for speeds in all_speeds_by_tau]



import matplotlib.pyplot as plt



plt.figure(figsize=(10, 7))

for tau, speeds in enumerate(all_speeds_by_tau):

    plt.hist(speeds, bins=120, histtype='step', 

             density=True, label=f"tau {tau}")



plt.title("Distribution of dFC Speed per tau (all windows pooled)")

plt.xlabel("Speed")

plt.ylabel("Density")

plt.legend()

plt.tight_layout()

plt.show()
# %%
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
#%%

import pickle

from pathlib import Path

from networkx import density

import numpy as np

from class_dataanalysis_julien import DFCAnalysis



data = DFCAnalysis()
# %%
#%%



# Load raw data

# data.load_raw_timeseries()

# data.load_raw_cognitive_data()

# data.load_raw_region_labels()



# Load preprocessed data

data.load_preprocessed_data()





data.get_temporal_parameters()
# %%
#%%



# Match these variables to your last run:

prefix = "speed"

save_path = data.paths['speed']  # <-- update this!

time_window_range = data.time_window_range           # <-- list of window sizes, same as in your analysis

tau_range = np.arange(0,data.tau+ 1)                   # <-- as above

n_animals = data.n_animals                # <-- as above

data.load_preprocessed_data()



window_file_total = save_path / f"{prefix}_windows{len(time_window_range)}_tau{len(tau_range)}_animals_{n_animals}.pkl"
# %%
#%%



with open(window_file_total, 'rb') as f:

    all_speed = pickle.load(f)



# Now all_speed is a list (or similar) with each entry for one window_size.

# The last one:

last_speed = all_speed[-1]  # This is the speed array for the last window size



# Example: print shape/info

print(f"Loaded speed for window {time_window_range[-1]}: shape = {last_speed.shape}")
# %%
# %%



#print the shape of each time windows

for i, speed in enumerate(all_speed):

    print(f"Window size {time_window_range[i]}: shape = {speed.shape}")
# %%
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
#%%



import numpy as np



n_taus = all_speed[0].shape[1]  # should be 4



# Prepare to collect all speeds per tau, across all animals, all windows, all timepoints

all_speeds_by_tau = [[] for _ in range(n_taus)]



for speed in all_speed:

    # speed.shape = (48, 4, N)

    for tau in range(n_taus):

        # Collect all speeds for this tau across all animals and timepoints for this window size

        speeds_tau = speed[:, tau, :].flatten()

        all_speeds_by_tau[tau].append(speeds_tau)



# Concatenate across windows

all_speeds_by_tau = [np.concatenate(speeds) for speeds in all_speeds_by_tau]



import matplotlib.pyplot as plt



plt.figure(figsize=(10, 7))

for tau, speeds in enumerate(all_speeds_by_tau):

    plt.hist(speeds, bins=120, histtype='step', 

             density=True, label=f"tau {tau}")



plt.title("Distribution of dFC Speed per tau (all windows pooled)")

plt.xlabel("Speed")

plt.ylabel("Density")

plt.legend()

plt.tight_layout()

plt.show()
# %%
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
#%%



n_tau = all_speed[0].shape[1]  # should be 4



window_medians = [[] for _ in range(n_tau)]

window_p25 = [[] for _ in range(n_tau)]

window_p75 = [[] for _ in range(n_tau)]



for win_list in all_speed:  # Iterate over windows

    # win_list shape: (48, 4, N)

    for tau in range(n_tau):

        # Extract and flatten all speeds for this tau, across all animals and timepoints

        tau_speeds = win_list[:, tau, :].flatten()

        # Compute statistics (handle NaNs as before)

        window_medians[tau].append(np.nanmedian(tau_speeds))

        window_p25[tau].append(np.nanpercentile(tau_speeds, 25))

        window_p75[tau].append(np.nanpercentile(tau_speeds, 75))



import matplotlib.pyplot as plt



plt.figure(figsize=(10, 6))

colors = ['navy', 'crimson', 'darkgreen', 'goldenrod']  # One color per tau



for tau in range(n_tau):

    plt.plot(time_window_range, window_medians[tau], label=f'tau {tau}', color=colors[tau])

    plt.fill_between(

        time_window_range,

        window_p25[tau],

        window_p75[tau],

        color=colors[tau],

        alpha=0.2

    )



plt.xlabel("Window size")

plt.ylabel("Speed")

plt.title("dFC Speed: Median and Interquartile Range per Window Size (per tau)")

plt.legend()

plt.tight_layout()

plt.show()
# %%
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
#%%%



import numpy as np

import matplotlib.pyplot as plt



group_name = ("WT", "VEH")  # <-- Replace with your desired group (tuple, as in your data.groups)

animal_indices = data.groups[group_name]



n_tau = all_speed[0].shape[1]



group_stats = {

    'medians': [[] for _ in range(n_tau)],

    'p25':     [[] for _ in range(n_tau)],

    'p75':     [[] for _ in range(n_tau)],

}



for i_win, win_list in enumerate(all_speed):

    # win_list: shape (48, 4, N)

    for tau in range(n_tau):

        window_speeds = []

        for idx in animal_indices:

            if idx >= len(win_list):

                continue

            speed_arr = win_list[idx][tau]

            speed_flat = np.array(speed_arr, dtype='float').flatten()

            window_speeds.append(speed_flat)

        if len(window_speeds) == 0:

            group_stats['medians'][tau].append(np.nan)

            group_stats['p25'][tau].append(np.nan)

            group_stats['p75'][tau].append(np.nan)

            continue

        window_speeds_all = np.concatenate(window_speeds)

        group_stats['medians'][tau].append(np.nanmedian(window_speeds_all))

        group_stats['p25'][tau].append(np.nanpercentile(window_speeds_all, 25))

        group_stats['p75'][tau].append(np.nanpercentile(window_speeds_all, 75))



colors = ['navy', 'crimson', 'darkgreen', 'goldenrod']

plt.figure(figsize=(10, 6))



for tau in range(n_tau):

    plt.plot(time_window_range, group_stats['medians'][tau], label=f"tau {tau}", color=colors[tau])

    plt.fill_between(

        time_window_range,

        group_stats['p25'][tau],

        group_stats['p75'][tau],

        color=colors[tau],

        alpha=0.2

    )



plt.xlabel("Window size")

plt.ylabel("Speed")

plt.title(f"dFC Speed: Median/IQR per Window Size (group: {group_name[0]}-{group_name[1]})")

plt.legend()

plt.tight_layout()

plt.show()
# %%
#%%



import pickle

from pathlib import Path

from networkx import density

import numpy as np

from class_dataanalysis_julien import DFCAnalysis



data = DFCAnalysis()
# %%
#%%



# Load raw data

# data.load_raw_timeseries()

# data.load_raw_cognitive_data()

# data.load_raw_region_labels()



# Load preprocessed data

data.load_preprocessed_data()





data.get_temporal_parameters()
# %%
#%%



# Match these variables to your last run:

prefix = "speed"

save_path = data.paths['speed']  # <-- update this!

time_window_range = data.time_window_range           # <-- list of window sizes, same as in your analysis

tau_range = np.arange(0,data.tau+ 1)                   # <-- as above

n_animals = data.n_animals                # <-- as above

data.load_preprocessed_data()



window_file_total = save_path / f"{prefix}_windows{len(time_window_range)}_tau{len(tau_range)}_animals_{n_animals}.pkl"
# %%
#%%



with open(window_file_total, 'rb') as f:

    all_speed = pickle.load(f)



# Now all_speed is a list (or similar) with each entry for one window_size.

# The last one:

last_speed = all_speed[-1]  # This is the speed array for the last window size



# Example: print shape/info

print(f"Loaded speed for window {time_window_range[-1]}: shape = {last_speed.shape}")
# %%
# %%



#print the shape of each time windows

for i, speed in enumerate(all_speed):

    print(f"Window size {time_window_range[i]}: shape = {speed.shape}")
# %%
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
#%%



import numpy as np



n_taus = all_speed[0].shape[1]  # should be 4



# Prepare to collect all speeds per tau, across all animals, all windows, all timepoints

all_speeds_by_tau = [[] for _ in range(n_taus)]



for speed in all_speed:

    # speed.shape = (48, 4, N)

    for tau in range(n_taus):

        # Collect all speeds for this tau across all animals and timepoints for this window size

        speeds_tau = speed[:, tau, :].flatten()

        all_speeds_by_tau[tau].append(speeds_tau)



# Concatenate across windows

all_speeds_by_tau = [np.concatenate(speeds) for speeds in all_speeds_by_tau]



import matplotlib.pyplot as plt



plt.figure(figsize=(10, 7))

for tau, speeds in enumerate(all_speeds_by_tau):

    plt.hist(speeds, bins=120, histtype='step', 

             density=True, label=f"tau {tau}")



plt.title("Distribution of dFC Speed per tau (all windows pooled)")

plt.xlabel("Speed")

plt.ylabel("Density")

plt.legend()

plt.tight_layout()

plt.show()
# %%
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
#%%



n_tau = all_speed[0].shape[1]  # should be 4



window_medians = [[] for _ in range(n_tau)]

window_p25 = [[] for _ in range(n_tau)]

window_p75 = [[] for _ in range(n_tau)]



for win_list in all_speed:  # Iterate over windows

    # win_list shape: (48, 4, N)

    for tau in range(n_tau):

        # Extract and flatten all speeds for this tau, across all animals and timepoints

        tau_speeds = win_list[:, tau, :].flatten()

        # Compute statistics (handle NaNs as before)

        window_medians[tau].append(np.nanmedian(tau_speeds))

        window_p25[tau].append(np.nanpercentile(tau_speeds, 25))

        window_p75[tau].append(np.nanpercentile(tau_speeds, 75))



import matplotlib.pyplot as plt



plt.figure(figsize=(10, 6))

colors = ['navy', 'crimson', 'darkgreen', 'goldenrod']  # One color per tau



for tau in range(n_tau):

    plt.plot(time_window_range, window_medians[tau], label=f'tau {tau}', color=colors[tau])

    plt.fill_between(

        time_window_range,

        window_p25[tau],

        window_p75[tau],

        color=colors[tau],

        alpha=0.2

    )



plt.xlabel("Window size")

plt.ylabel("Speed")

plt.title("dFC Speed: Median and Interquartile Range per Window Size (per tau)")

plt.legend()

plt.tight_layout()

plt.show()
# %%
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
#%%%



import numpy as np

import matplotlib.pyplot as plt



group_name = ("WT", "VEH")  # <-- Replace with your desired group (tuple, as in your data.groups)

animal_indices = data.groups[group_name]



n_tau = all_speed[0].shape[1]



group_stats = {

    'medians': [[] for _ in range(n_tau)],

    'p25':     [[] for _ in range(n_tau)],

    'p75':     [[] for _ in range(n_tau)],

}



for i_win, win_list in enumerate(all_speed):

    # win_list: shape (48, 4, N)

    for tau in range(n_tau):

        window_speeds = []

        for idx in animal_indices:

            if idx >= len(win_list):

                continue

            speed_arr = win_list[idx][tau]

            speed_flat = np.array(speed_arr, dtype='float').flatten()

            window_speeds.append(speed_flat)

        if len(window_speeds) == 0:

            group_stats['medians'][tau].append(np.nan)

            group_stats['p25'][tau].append(np.nan)

            group_stats['p75'][tau].append(np.nan)

            continue

        window_speeds_all = np.concatenate(window_speeds)

        group_stats['medians'][tau].append(np.nanmedian(window_speeds_all))

        group_stats['p25'][tau].append(np.nanpercentile(window_speeds_all, 25))

        group_stats['p75'][tau].append(np.nanpercentile(window_speeds_all, 75))



colors = ['navy', 'crimson', 'darkgreen', 'goldenrod']

plt.figure(figsize=(10, 6))



for tau in range(n_tau):

    plt.plot(time_window_range, group_stats['medians'][tau], label=f"tau {tau}", color=colors[tau])

    plt.fill_between(

        time_window_range,

        group_stats['p25'][tau],

        group_stats['p75'][tau],

        color=colors[tau],

        alpha=0.2

    )



plt.xlabel("Window size")

plt.ylabel("Speed")

plt.title(f"dFC Speed: Median/IQR per Window Size (group: {group_name[0]}-{group_name[1]})")

plt.legend()

plt.tight_layout()

plt.show()
# %%
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
#%%



import numpy as np



n_taus = all_speed[0].shape[1]  # should be 4



# Prepare to collect all speeds per tau, across all animals, all windows, all timepoints

all_speeds_by_tau = [[] for _ in range(n_taus)]



for speed in all_speed:

    # speed.shape = (48, 4, N)

    for tau in range(n_taus):

        # Collect all speeds for this tau across all animals and timepoints for this window size

        speeds_tau = speed[:, tau, :].flatten()

        all_speeds_by_tau[tau].append(speeds_tau)



# Concatenate across windows

all_speeds_by_tau = [np.concatenate(speeds) for speeds in all_speeds_by_tau]



import matplotlib.pyplot as plt



plt.figure(figsize=(10, 7))

for tau, speeds in enumerate(all_speeds_by_tau):

    plt.hist(speeds, bins=120, histtype='step', 

             density=True, label=f"tau {tau}")



plt.title("Distribution of dFC Speed per tau (all windows pooled)")

plt.xlabel("Speed")

plt.ylabel("Density")

plt.legend()

plt.tight_layout()

plt.show()
# %%
#%%



import numpy as np



n_taus = all_speed[0].shape[1]  # should be 4



# Prepare to collect all speeds per tau, across all animals, all windows, all timepoints

all_speeds_by_tau = [[] for _ in range(n_taus)]



for speed in all_speed:

    # speed.shape = (48, 4, N)

    for tau in range(n_taus):

        # Collect all speeds for this tau across all animals and timepoints for this window size

        speeds_tau = speed[:, tau, :].flatten()

        all_speeds_by_tau[tau].append(speeds_tau)



# Concatenate across windows

all_speeds_by_tau = [np.concatenate(speeds) for speeds in all_speeds_by_tau]



import matplotlib.pyplot as plt



plt.figure(figsize=(10, 7))

for tau, speeds in enumerate(all_speeds_by_tau):

    plt.hist(speeds, bins=120, histtype='step', 

             density=True, label=f"tau {tau}")



plt.title("Distribution of dFC Speed per tau (all windows pooled)")

plt.xlabel("Speed")

plt.ylabel("Density")

plt.legend()

plt.tight_layout()

plt.show()
# %%
import numpy as np

import matplotlib.pyplot as plt



filtered_df = cog_data_filtered[cog_data_filtered['n_timepoints'] >= 500].reset_index(drop=True)

groups = filtered_df.groupby(['genotype', 'treatment']).groups



tau_count = 3  # adjust as needed

animal_count = len(filtered_df)

time_window_count = len(speeds_all)  # now using len instead of shape



plt.figure(figsize=(8, 5))



for group, animal_idxs in groups.items():

    pooled = []

    for animal_idx in animal_idxs:

        for tau in range(tau_count):

            for time_win in range(time_window_count):

                # speeds_all is a list: speeds_all[time_win][animal_idx * tau_count + tau, :]

                print(animal_idx, animal_idx * tau_count + tau)

                arr = speeds_all[time_win][animal_idx * tau_count + tau]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            plt.hist(group_speeds, bins=100, alpha=0.5, 

                     label=f"{group}", histtype='step', linewidth=1.7, density=True)



plt.xlabel("DFC Speed")

plt.ylabel("Density")

plt.title("Histogram of DFC Speeds by Group (all tau pooled)")

plt.legend()

plt.tight_layout()

plt.show()



# %%
#%%



import numpy as np



n_taus = all_speed[0].shape[1]  # should be 4



# Prepare to collect all speeds per tau, across all animals, all windows, all timepoints

all_speeds_by_tau = [[] for _ in range(n_taus)]



for speed in all_speed:

    # speed.shape = (48, 4, N)

    for tau in range(n_taus):

        # Collect all speeds for this tau across all animals and timepoints for this window size

        speeds_tau = speed[:, tau, :].flatten()

        all_speeds_by_tau[tau].append(speeds_tau)



# Concatenate across windows

all_speeds_by_tau = [np.concatenate(speeds) for speeds in all_speeds_by_tau]



import matplotlib.pyplot as plt



plt.figure(figsize=(10, 7))

for tau, speeds in enumerate(all_speeds_by_tau):

    plt.hist(speeds, bins=120, histtype='step', 

             density=True, label=f"tau {tau}")



plt.title("Distribution of dFC Speed per tau (all windows pooled)")

plt.xlabel("Speed")

plt.ylabel("Density")

plt.legend()

plt.tight_layout()

plt.show()
# %%
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



import numpy as np

import matplotlib.pyplot as plt



group_name = ("WT", "VEH")  # Change to desired group

animal_indices = data.groups[group_name]



all_group_speeds = []



for win_list in all_speed:  # Iterate over windows

    # win_list shape: (48, 4, N)

    for idx in animal_indices:

        if idx >= len(win_list):

            continue

        # Pool across all taus for this animal and window

        animal_all_tau = win_list[idx]  # shape: (4, N)

        # Flatten all tau and timepoints

        all_group_speeds.append(np.array(animal_all_tau, dtype='float').flatten())



# Concatenate all

all_group_speeds_flat = np.concatenate(all_group_speeds)
# %%
#%%



group_name = ("WT", "VEH")  # Change to desired group

animal_indices = data.groups[group_name]



all_group_speeds = []



for win_list in all_speed:  # Iterate over windows

    # win_list shape: (48, 4, N)

    for idx in animal_indices:

        if idx >= len(win_list):

            continue

        # Pool across all taus for this animal and window

        animal_all_tau = win_list[idx]  # shape: (4, N)

        # Flatten all tau and timepoints

        all_group_speeds.append(np.array(animal_all_tau, dtype='float').flatten())



# Concatenate all

all_group_speeds_flat = np.concatenate(all_group_speeds)
# %%
#%%



group_name = ("WT", "VEH")  # Change to desired group

animal_indices = data.groups[group_name]



all_group_speeds = []



for win_list in all_speed:  # Iterate over windows

    # win_list shape: (48, 4, N)

    for idx in animal_indices:

        if idx >= len(win_list):

            continue

        # Pool across all taus for this animal and window

        animal_all_tau = win_list[idx]  # shape: (4, N)

        # Flatten all tau and timepoints

        all_group_speeds.append(np.array(animal_all_tau, dtype='float').flatten())



# Concatenate all

all_group_speeds_flat = np.concatenate(all_group_speeds)







plt.figure(figsize=(8, 5))

plt.hist(all_group_speeds_flat, bins=150, alpha=0.8, histtype='step', density=True)

plt.title(f"Distribution of dFC Speed (all taus, all windows, group: {group_name[0]}-{group_name[1]})")

plt.xlabel("Speed")

plt.ylabel("Density")

plt.tight_layout()

plt.show()
# %%
data.groups
# %%
# %%



# Plot a hist distribution that pools (ravel or flatten) all the speed together



# -----------------  Pool all speed values from all windows -----------------

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
#%%



import numpy as np

import matplotlib.pyplot as plt



plt.figure(figsize=(9, 6))



for group_name, animal_indices in data.groups.items():

    pooled = []

    for win_list in all_speed:  # Iterate over window sizes (len=50)

        # win_list shape: (48, 4, variable)

        for animal_idx in animal_indices:

            for tau in range(win_list.shape[1]):  # Usually 4

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            plt.hist(group_speeds, bins=100, alpha=0.5, 

                     label=f"{group_name[0]}-{group_name[1]}", 

                     histtype='step', linewidth=1.7, density=True)



plt.xlabel("dFC Speed")

plt.ylabel("Density")

plt.title("Histogram of dFC Speeds by Group (all tau, all windows pooled)")

plt.legend()

plt.tight_layout()

plt.show()
# %%
#%%



# --------------------------------------------------------------------------

# Plot a histogram of dFC speed for each group, pooling all windows and taus

# --------------------------------------------------------------------------



plt.figure(figsize=(9, 6))



for group_name, animal_indices in data.groups.items():

    pooled = []

    for win_list in all_speed:  # Iterate over window sizes (len=50)

        # win_list shape: (48, 4, variable)

        for animal_idx in animal_indices:

            for tau in range(win_list.shape[1]):  # Usually 4

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            plt.hist(group_speeds, bins=100, alpha=0.7, 

                     label=f"{group_name[0]}-{group_name[1]}", 

                     histtype='step', linewidth=1.7, density=True)



plt.xlabel("dFC Speed")

plt.ylabel("Density")

plt.title("Histogram of dFC Speeds by Group (all tau, all windows pooled)")

plt.legend()

plt.tight_layout()

plt.show()
# %%
#%%



# --------------------------------------------------------------------------

# Plot a histogram of dFC speed for each group, pooling all windows and taus

# --------------------------------------------------------------------------



plt.figure(figsize=(9, 6))



for group_name, animal_indices in data.groups.items():

    pooled = []

    for win_list in all_speed:  # Iterate over window sizes (len=50)

        # win_list shape: (48, 4, variable)

        for animal_idx in animal_indices:

            for tau in range(win_list.shape[1]):  # Usually 4

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            plt.hist(group_speeds, bins=100, alpha=0., 

                     label=f"{group_name[0]}-{group_name[1]}", 

                     histtype='step', linewidth=1.7, density=True)



plt.xlabel("dFC Speed")

plt.ylabel("Density")

plt.title("Histogram of dFC Speeds by Group (all tau, all windows pooled)")

plt.legend()

plt.tight_layout()

plt.show()
# %%
#%%



# --------------------------------------------------------------------------

# Plot a histogram of dFC speed for each group, pooling all windows and taus

# --------------------------------------------------------------------------



plt.figure(figsize=(9, 6))



for group_name, animal_indices in data.groups.items():

    pooled = []

    for win_list in all_speed:  # Iterate over window sizes (len=50)

        # win_list shape: (48, 4, variable)

        for animal_idx in animal_indices:

            for tau in range(win_list.shape[1]):  # Usually 4

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            plt.hist(group_speeds, bins=100, alpha=0.6, 

                     label=f"{group_name[0]}-{group_name[1]}", 

                     histtype='step', linewidth=1.7, density=True)



plt.xlabel("dFC Speed")

plt.ylabel("Density")

plt.title("Histogram of dFC Speeds by Group (all tau, all windows pooled)")

plt.legend()

plt.tight_layout()

plt.show()
# %%
#%%



# --------------------------------------------------------------------------

# Plot a histogram of dFC speed for each group, pooling all windows and taus

# --------------------------------------------------------------------------



plt.figure(figsize=(9, 6))



for group_name, animal_indices in data.groups.items():

    pooled = []

    for win_list in all_speed:  # Iterate over window sizes (len=50)

        # win_list shape: (48, 4, variable)

        for animal_idx in animal_indices:

            for tau in range(win_list.shape[1]):  # Usually 4

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            plt.hist(group_speeds, bins=150, alpha=0.6, 

                     label=f"{group_name[0]}-{group_name[1]}", 

                     histtype='step', linewidth=1.7, density=True)



plt.xlabel("dFC Speed")

plt.ylabel("Density")

plt.title("Histogram of dFC Speeds by Group (all tau, all windows pooled)")

plt.legend()

plt.tight_layout()

plt.show()
# %%
#%%



# --------------------------------------------------------------------------

# Plot a histogram of dFC speed for each group, pooling all windows and taus

# --------------------------------------------------------------------------



plt.figure(figsize=(9, 6))



for group_name, animal_indices in data.groups.items():

    pooled = []

    for win_list in all_speed:  # Iterate over window sizes (len=50)

        # win_list shape: (48, 4, variable)

        for animal_idx in animal_indices:

            for tau in range(win_list.shape[1]):  # Usually 4

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            plt.hist(group_speeds, bins=(0, alpha=0.6, 

                     label=f"{group_name[0]}-{group_name[1]}", 

                     histtype='step', linewidth=1.7, density=True)



plt.xlabel("dFC Speed")

plt.ylabel("Density")

plt.title("Histogram of dFC Speeds by Group (all tau, all windows pooled)")

plt.legend()

plt.tight_layout()

plt.show()
# %%
#%%



# --------------------------------------------------------------------------

# Plot a histogram of dFC speed for each group, pooling all windows and taus

# --------------------------------------------------------------------------



plt.figure(figsize=(9, 6))



for group_name, animal_indices in data.groups.items():

    pooled = []

    for win_list in all_speed:  # Iterate over window sizes (len=50)

        # win_list shape: (48, 4, variable)

        for animal_idx in animal_indices:

            for tau in range(win_list.shape[1]):  # Usually 4

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            plt.hist(group_speeds, bins=50, alpha=0.6, 

                     label=f"{group_name[0]}-{group_name[1]}", 

                     histtype='step', linewidth=1.7, density=True)



plt.xlabel("dFC Speed")

plt.ylabel("Density")

plt.title("Histogram of dFC Speeds by Group (all tau, all windows pooled)")

plt.legend()

plt.tight_layout()

plt.show()
# %%
#%%



# --------------------------------------------------------------------------

# Plot a histogram of dFC speed for each group, pooling all windows and taus

# --------------------------------------------------------------------------



plt.figure(figsize=(9, 6))



for group_name, animal_indices in data.groups.items():

    pooled = []

    for win_list in all_speed:  # Iterate over window sizes (len=50)

        # win_list shape: (48, 4, variable)

        for animal_idx in animal_indices:

            for tau in range(win_list.shape[1]):  # Usually 4

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            plt.hist(group_speeds, bins=150, alpha=0.6, 

                     label=f"{group_name[0]}-{group_name[1]}", 

                     histtype='step', linewidth=1.7, density=True)



plt.xlabel("dFC Speed")

plt.ylabel("Density")

plt.title("Histogram of dFC Speeds by Group (all tau, all windows pooled)")

plt.legend()

plt.tight_layout()

plt.show()
# %%
#%%



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# Set publication style (can customize further)

plt.style.use('seaborn-whitegrid')

plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 20, 'legend.fontsize': 14})



# Use a color palette with distinct colors for groups

palette = sns.color_palette('Set2', n_colors=len(data.groups))



plt.figure(figsize=(10, 6))



for (group_name, animal_indices), color in zip(data.groups.items(), palette):

    pooled = []

    for win_list in all_speed:  # Each window size

        for animal_idx in animal_indices:

            for tau in range(win_list.shape[1]):

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            # KDE plot for smooth, publication-ready curves

            sns.kdeplot(group_speeds, bw_adjust=1.2, 

                        label=f"{group_name[0]}-{group_name[1]}",

                        color=color, linewidth=2.5, clip=(0, np.percentile(group_speeds, 99.5)))

            

plt.xlabel("dFC Speed", labelpad=10)

plt.ylabel("Density", labelpad=10)

plt.title("Distribution of dFC Speeds by Group\n(All taus, all windows pooled)", pad=15)

plt.legend(frameon=True, loc='best', title='Group')

plt.tight_layout()



# Remove top/right spines for a cleaner look

sns.despine(trim=True)



plt.show()
# %%
#%%



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# Set publication style (can customize further)

plt.style.use('seaborn-whitegrid')

plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 20, 'legend.fontsize': 14})



# Use a color palette with distinct colors for groups

palette = sns.color_palette('Set2', n_colors=len(data.groups))



plt.figure(figsize=(10, 6))



for (group_name, animal_indices), color in zip(data.groups.items(), palette):

    pooled = []

    for win_list in all_speed:  # Each window size

        for animal_idx in animal_indices:

            for tau in range(win_list.shape[1]):

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            # KDE plot for smooth, publication-ready curves

            sns.kdeplot(group_speeds, bw_adjust=1.2, 

                        label=f"{group_name[0]}-{group_name[1]}",

                        color=color, linewidth=2.5, clip=(0, np.percentile(group_speeds, 99.5)))

            

plt.xlabel("dFC Speed", labelpad=10)

plt.ylabel("Density", labelpad=10)

plt.title("Distribution of dFC Speeds by Group\n(All taus, all windows pooled)", pad=15)

plt.legend(frameon=True, loc='best', title='Group')

plt.tight_layout()



# Remove top/right spines for a cleaner look

sns.despine(trim=True)



plt.show()
# %%
#%%



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# Set publication style (can customize further)

plt.style.use('whitegrid')

plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 20, 'legend.fontsize': 14})



# Use a color palette with distinct colors for groups

palette = sns.color_palette('Set2', n_colors=len(data.groups))



plt.figure(figsize=(10, 6))



for (group_name, animal_indices), color in zip(data.groups.items(), palette):

    pooled = []

    for win_list in all_speed:  # Each window size

        for animal_idx in animal_indices:

            for tau in range(win_list.shape[1]):

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            # KDE plot for smooth, publication-ready curves

            sns.kdeplot(group_speeds, bw_adjust=1.2, 

                        label=f"{group_name[0]}-{group_name[1]}",

                        color=color, linewidth=2.5, clip=(0, np.percentile(group_speeds, 99.5)))

            

plt.xlabel("dFC Speed", labelpad=10)

plt.ylabel("Density", labelpad=10)

plt.title("Distribution of dFC Speeds by Group\n(All taus, all windows pooled)", pad=15)

plt.legend(frameon=True, loc='best', title='Group')

plt.tight_layout()



# Remove top/right spines for a cleaner look

sns.despine(trim=True)



plt.show()
# %%
#%%



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# Set publication style (can customize further)

plt.set_theme('whitegrid')

plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 20, 'legend.fontsize': 14})



# Use a color palette with distinct colors for groups

palette = sns.color_palette('Set2', n_colors=len(data.groups))



plt.figure(figsize=(10, 6))



for (group_name, animal_indices), color in zip(data.groups.items(), palette):

    pooled = []

    for win_list in all_speed:  # Each window size

        for animal_idx in animal_indices:

            for tau in range(win_list.shape[1]):

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            # KDE plot for smooth, publication-ready curves

            sns.kdeplot(group_speeds, bw_adjust=1.2, 

                        label=f"{group_name[0]}-{group_name[1]}",

                        color=color, linewidth=2.5, clip=(0, np.percentile(group_speeds, 99.5)))

            

plt.xlabel("dFC Speed", labelpad=10)

plt.ylabel("Density", labelpad=10)

plt.title("Distribution of dFC Speeds by Group\n(All taus, all windows pooled)", pad=15)

plt.legend(frameon=True, loc='best', title='Group')

plt.tight_layout()



# Remove top/right spines for a cleaner look

sns.despine(trim=True)



plt.show()
# %%
#%%



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# Set publication style (can customize further)

sns.set_theme('whitegrid')

plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 20, 'legend.fontsize': 14})



# Use a color palette with distinct colors for groups

palette = sns.color_palette('Set2', n_colors=len(data.groups))



plt.figure(figsize=(10, 6))



for (group_name, animal_indices), color in zip(data.groups.items(), palette):

    pooled = []

    for win_list in all_speed:  # Each window size

        for animal_idx in animal_indices:

            for tau in range(win_list.shape[1]):

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            # KDE plot for smooth, publication-ready curves

            sns.kdeplot(group_speeds, bw_adjust=1.2, 

                        label=f"{group_name[0]}-{group_name[1]}",

                        color=color, linewidth=2.5, clip=(0, np.percentile(group_speeds, 99.5)))

            

plt.xlabel("dFC Speed", labelpad=10)

plt.ylabel("Density", labelpad=10)

plt.title("Distribution of dFC Speeds by Group\n(All taus, all windows pooled)", pad=15)

plt.legend(frameon=True, loc='best', title='Group')

plt.tight_layout()



# Remove top/right spines for a cleaner look

sns.despine(trim=True)



plt.show()
# %%
#%%



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# Set publication style (can customize further)

sns.set_theme('whitegrid')

plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 20, 'legend.fontsize': 14})



# Use a color palette with distinct colors for groups

palette = sns.color_palette('Set2', n_colors=len(data.groups))



plt.figure(figsize=(10, 6))



for (group_name, animal_indices), color in zip(data.groups.items(), palette):

    pooled = []

    for win_list in all_speed:  # Each window size

        for animal_idx in animal_indices:

            for tau in range(win_list.shape[1]):

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            # KDE plot for smooth, publication-ready curves

            sns.kdeplot(group_speeds, bw_adjust=1.2, 

                        label=f"{group_name[0]}-{group_name[1]}",

                        color=color, linewidth=2.5, clip=(0, np.percentile(group_speeds, 99.5)))

            

plt.xlabel("dFC Speed", labelpad=10)

plt.ylabel("Density", labelpad=10)

plt.title("Distribution of dFC Speeds by Group\n(All taus, all windows pooled)", pad=15)

plt.legend(frameon=True, loc='best', title='Group')

plt.tight_layout()



# Remove top/right spines for a cleaner look

sns.despine(trim=True)



plt.show()
# %%
#%%



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# Set publication style (can customize further)

sns.set_theme(style='whitegrid')



plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 20, 'legend.fontsize': 14})



# Use a color palette with distinct colors for groups

palette = sns.color_palette('Set2', n_colors=len(data.groups))



plt.figure(figsize=(10, 6))



for (group_name, animal_indices), color in zip(data.groups.items(), palette):

    pooled = []

    for win_list in all_speed:  # Each window size

        for animal_idx in animal_indices:

            for tau in range(win_list.shape[1]):

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            # KDE plot for smooth, publication-ready curves

            sns.kdeplot(group_speeds, bw_adjust=1.2, 

                        label=f"{group_name[0]}-{group_name[1]}",

                        color=color, linewidth=2.5, clip=(0, np.percentile(group_speeds, 99.5)))

            

plt.xlabel("dFC Speed", labelpad=10)

plt.ylabel("Density", labelpad=10)

plt.title("Distribution of dFC Speeds by Group\n(All taus, all windows pooled)", pad=15)

plt.legend(frameon=True, loc='best', title='Group')

plt.tight_layout()



# Remove top/right spines for a cleaner look

sns.despine(trim=True)



plt.show()
# %%
#%%



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# Set publication style (can customize further)

sns.set_theme(style='whitegrid')



plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 20, 'legend.fontsize': 10})



# Use a color palette with distinct colors for groups

palette = sns.color_palette('Set2', n_colors=len(data.groups))



plt.figure(figsize=(10, 6))



for (group_name, animal_indices), color in zip(data.groups.items(), palette):

    pooled = []

    for win_list in all_speed:  # Each window size

        for animal_idx in animal_indices:

            for tau in range(win_list.shape[1]):

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            # KDE plot for smooth, publication-ready curves

            sns.kdeplot(group_speeds, bw_adjust=1.2, 

                        label=f"{group_name[0]}-{group_name[1]}",

                        color=color, linewidth=2.5, clip=(0, np.percentile(group_speeds, 99.5)))

            

plt.xlabel("dFC Speed", labelpad=10)

plt.ylabel("Density", labelpad=10)

plt.title("Distribution of dFC Speeds by Group\n(All taus, all windows pooled)", pad=15)

plt.legend(frameon=True, loc='best', title='Group')

plt.tight_layout()



# Remove top/right spines for a cleaner look

sns.despine(trim=True)



plt.show()
# %%
#%%



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# Set publication style (can customize further)

sns.set_theme(style='white', palette='deep', context='talk')



plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 20, 'legend.fontsize': 10})



# Use a color palette with distinct colors for groups

palette = sns.color_palette('Set2', n_colors=len(data.groups))



plt.figure(figsize=(10, 6))



for (group_name, animal_indices), color in zip(data.groups.items(), palette):

    pooled = []

    for win_list in all_speed:  # Each window size

        for animal_idx in animal_indices:

            for tau in range(win_list.shape[1]):

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            # KDE plot for smooth, publication-ready curves

            sns.kdeplot(group_speeds, bw_adjust=1.2, 

                        label=f"{group_name[0]}-{group_name[1]}",

                        color=color, linewidth=2.5, clip=(0, np.percentile(group_speeds, 99.5)))

            

plt.xlabel("dFC Speed", labelpad=10)

plt.ylabel("Density", labelpad=10)

plt.title("Distribution of dFC Speeds by Group\n(All taus, all windows pooled)", pad=15)

plt.legend(frameon=True, loc='best', title='Group')

plt.tight_layout()



# Remove top/right spines for a cleaner look

sns.despine(trim=True)



plt.show()
# %%
#%%



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# Set publication style (can customize further)

sns.set_theme(style='white', palette='deep', context='talk')



plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 20, 'legend.fontsize': 10})



# Use a color palette with distinct colors for groups

palette = sns.color_palette('Set2', n_colors=len(data.groups))



plt.figure(figsize=(10, 6))



for (group_name, animal_indices), color in zip(data.groups.items(), palette):

    pooled = []

    for win_list in all_speed:  # Each window size

        for animal_idx in animal_indices:

            for tau in range(win_list.shape[1]):

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            # KDE plot for smooth, publication-ready curves

            sns.kdeplot(group_speeds, 

                        bw_adjust=1.2, 

                        label=f"{group_name[0]}-{group_name[1]}".lower(),

                        color=color, linewidth=2.5, clip=(0, np.percentile(group_speeds, 99.5)))

            

plt.xlabel("dFC Speed", labelpad=10)

plt.ylabel("Density", labelpad=10)

plt.title("Distribution of dFC Speeds by Group\n(All taus, all windows pooled)", pad=15)

plt.legend(frameon=True, loc='best', title='Group')

plt.tight_layout()



# Remove top/right spines for a cleaner look

sns.despine(trim=True)



plt.show()
# %%
#%%



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# Set publication style (can customize further)

sns.set_theme(style='white', palette='deep', context='talk')



plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 20, 'legend.fontsize': 14})



# Use a color palette with distinct colors for groups

palette = sns.color_palette('Set2', n_colors=len(data.groups))



plt.figure(figsize=(10, 6))



for (group_name, animal_indices), color in zip(data.groups.items(), palette):

    pooled = []

    for win_list in all_speed:  # Each window size

        for animal_idx in animal_indices:

            for tau in range(win_list.shape[1]):

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            # KDE plot for smooth, publication-ready curves

            sns.kdeplot(group_speeds, 

                        bw_adjust=1.2, 

                        label=f"{group_name[0]}-{group_name[1]}".lower(),

                        color=color, linewidth=2.5, clip=(0, np.percentile(group_speeds, 99.5)))

            

plt.xlabel("dFC Speed", labelpad=10)

plt.ylabel("Density", labelpad=10)

plt.title("Distribution of dFC Speeds by Group\n(All taus, all windows pooled)", pad=15)

plt.legend(frameon=True, loc='best', title='Group')

plt.tight_layout()



# Remove top/right spines for a cleaner look

sns.despine(trim=True)



plt.show()
# %%
#%%



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# Set publication style (can customize further)

sns.set_theme(style='white', palette='deep', context='talk')



plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 20, 'legend.fontsize': 14})



# Use a color palette with distinct colors for groups

palette = sns.color_palette('Set2', n_colors=len(data.groups))



plt.figure(figsize=(10, 6))



for (group_name, animal_indices), color in zip(data.groups.items(), palette):

    pooled = []

    for win_list in all_speed:  # Each window size

        for animal_idx in animal_indices:

            for tau in range(win_list.shape[1]):

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            # KDE plot for smooth, publication-ready curves

            sns.kdeplot(group_speeds, 

                        bw_adjust=1.2, 

                        label=f"{group_name[0]}-{group_name[1]}".lower(),

                        color=color, linewidth=2.5, clip=(0, np.percentile(group_speeds, 99.5)))

            

plt.xlabel("dFC Speed", labelpad=10)

plt.ylabel("Density", labelpad=10)

plt.yscale('log')  # Log scale for better visibility of tails

plt.title("Distribution of dFC Speeds by Group\n(All taus, all windows pooled)", pad=15)

plt.legend(frameon=True, loc='best', title='Group')

plt.tight_layout()



# Remove top/right spines for a cleaner look

sns.despine(trim=True)



plt.show()
# %%
#%%



# --------------------------------------------------------------------------

# Plot a histogram of dFC speed for each group, pooling all windows and taus

# --------------------------------------------------------------------------



plt.figure(figsize=(9, 6))



for group_name, animal_indices in data.groups.items():

    pooled = []

    for win_list in all_speed:  # Iterate over window sizes (len=50)

        # win_list shape: (48, 4, variable)

        for animal_idx in animal_indices:

            for tau in range(win_list.shape[1]):  # Usually 4

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            plt.hist(group_speeds, bins=150, alpha=0.6, 

                     label=f"{group_name[0]}-{group_name[1]}".lower(), 

                     histtype='step', linewidth=1.7, density=True)



plt.xlabel("dFC Speed")

plt.ylabel("Density")

plt.title("Histogram of dFC Speeds by Group (all tau, all windows pooled)")

plt.legend()

plt.tight_layout()

plt.show()
# %%
#%%



# --------------------------------------------------------------------------

# Plot a histogram of dFC speed for each group, pooling all windows and taus

# --------------------------------------------------------------------------



plt.figure(figsize=(9, 6))



for group_name, animal_indices in data.groups.items():

    pooled = []

    for win_list in all_speed:  # Iterate over window sizes (len=50)

        # win_list shape: (48, 4, variable)

        for animal_idx in animal_indices:

            for tau in range(win_list.shape[1]):  # Usually 4

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            plt.hist(group_speeds, bins=150, alpha=0.6, 

                     label=f"{group_name[0]}-{group_name[1]}".lower(), 

                     histtype='step', linewidth=1.7, density=True)



plt.xlabel("dFC Speed")

plt.ylabel("Density")

plt.title("Histogram of dFC Speeds by Group (all tau, all windows pooled)")

plt.despine(trim=True)

plt.legend()

plt.tight_layout()

plt.show()
# %%
#%%



# --------------------------------------------------------------------------

# Plot a histogram of dFC speed for each group, pooling all windows and taus

# --------------------------------------------------------------------------



plt.figure(figsize=(9, 6))



for group_name, animal_indices in data.groups.items():

    pooled = []

    for win_list in all_speed:  # Iterate over window sizes (len=50)

        # win_list shape: (48, 4, variable)

        for animal_idx in animal_indices:

            for tau in range(win_list.shape[1]):  # Usually 4

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            plt.hist(group_speeds, bins=150, alpha=0.6, 

                     label=f"{group_name[0]}-{group_name[1]}".lower(), 

                     histtype='step', linewidth=1.7, density=True)



plt.xlabel("dFC Speed")

plt.ylabel("Density")

plt.title("Histogram of dFC Speeds by Group (all tau, all windows pooled)")

plt.legend()

plt.yscale('log')  # Log scale for better visibility of tails

plt.tight_layout()

plt.show()
# %%
#%%



# Set publication style (can customize further)

sns.set_theme(style='white', palette='deep', context='talk')



plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 20, 'legend.fontsize': 14})



# Use a color palette with distinct colors for groups

palette = sns.color_palette('Set2', n_colors=len(data.groups))



plt.figure(figsize=(10, 6))



for (group_name, animal_indices), color in zip(data.groups.items(), palette):

    pooled = []

    for win_list in all_speed:  # Each window size

        for animal_idx in animal_indices:

            for tau in range(win_list.shape[1]):

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            # KDE plot for smooth, publication-ready curves

            sns.kdeplot(group_speeds, 

                        bw_adjust=1.2, 

                        label=f"{group_name[0]}-{group_name[1]}".lower(),

                        color=color, linewidth=2.5, clip=(0, np.percentile(group_speeds, 99.5)))

            

plt.xlabel("dFC Speed", labelpad=10)

plt.ylabel("Density", labelpad=10)

plt.yscale('log')  # Log scale for better visibility of tails

plt.title("Distribution of dFC Speeds by Group\n(All taus, all windows pooled)", pad=15)

plt.legend(frameon=True, loc='best', title='Group')

plt.tight_layout()



# Remove top/right spines for a cleaner look

sns.despine(trim=True)



plt.show()
# %%
#%%



# Set publication style (can customize further)

sns.set_theme(style='white', palette='deep', context='talk')



plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 20, 'legend.fontsize': 14})



# Use a color palette with distinct colors for groups

palette = sns.color_palette('Set2', n_colors=len(data.groups))



plt.figure(figsize=(10, 6))



for (group_name, animal_indices), color in zip(data.groups.items(), palette):

    pooled = []

    for win_list in all_speed:  # Each window size

        for animal_idx in animal_indices:

            for tau in range(win_list.shape[1]):

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            # KDE plot for smooth, publication-ready curves

            sns.kdeplot(group_speeds, 

                        bw_adjust=1.2, 

                        label=f"{group_name[0]}-{group_name[1]}".lower(),

                        color=color, linewidth=2.5, clip=(0, -2))

            

plt.xlabel("dFC Speed", labelpad=10)

plt.ylabel("Density", labelpad=10)

plt.yscale('log')  # Log scale for better visibility of tails

plt.title("Distribution of dFC Speeds by Group\n(All taus, all windows pooled)", pad=15)

plt.legend(frameon=True, loc='best', title='Group')

plt.tight_layout()



# Remove top/right spines for a cleaner look

sns.despine(trim=True)



plt.show()
# %%
#%%



# Set publication style (can customize further)

sns.set_theme(style='white', palette='deep', context='talk')



plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 20, 'legend.fontsize': 14})



# Use a color palette with distinct colors for groups

palette = sns.color_palette('Set2', n_colors=len(data.groups))



plt.figure(figsize=(10, 6))



for (group_name, animal_indices), color in zip(data.groups.items(), palette):

    pooled = []

    for win_list in all_speed:  # Each window size

        for animal_idx in animal_indices:

            for tau in range(win_list.shape[1]):

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            # KDE plot for smooth, publication-ready curves

            sns.kdeplot(group_speeds, 

                        bw_adjust=1.2, 

                        label=f"{group_name[0]}-{group_name[1]}".lower(),

                        color=color, linewidth=2.5, clip=(0, 2))

            

plt.xlabel("dFC Speed", labelpad=10)

plt.ylabel("Density", labelpad=10)

plt.yscale('log')  # Log scale for better visibility of tails

plt.title("Distribution of dFC Speeds by Group\n(All taus, all windows pooled)", pad=15)

plt.legend(frameon=True, loc='best', title='Group')

plt.tight_layout()



# Remove top/right spines for a cleaner look

sns.despine(trim=True)



plt.show()
# %%
#%%



# Set publication style (can customize further)

sns.set_theme(style='white', palette='deep', context='talk')



plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 20, 'legend.fontsize': 14})



# Use a color palette with distinct colors for groups

palette = sns.color_palette('Set2', n_colors=len(data.groups))



plt.figure(figsize=(10, 6))



for (group_name, animal_indices), color in zip(data.groups.items(), palette):

    pooled = []

    for win_list in all_speed:  # Each window size

        for animal_idx in animal_indices:

            for tau in range(win_list.shape[1]):

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            # KDE plot for smooth, publication-ready curves

            sns.kdeplot(group_speeds, 

                        bw_adjust=1.2, 

                        label=f"{group_name[0]}-{group_name[1]}".lower(),

                        color=color, linewidth=2.5, clip=(0, 2))

            

plt.xlabel("dFC Speed", labelpad=10)

plt.ylabel("Density", labelpad=10)

# plt.yscale('log')  # Log scale for better visibility of tails

plt.title("Distribution of dFC Speeds by Group\n(All taus, all windows pooled)", pad=15)

plt.legend(frameon=True, loc='best', title='Group')

plt.tight_layout()



# Remove top/right spines for a cleaner look

sns.despine(trim=True)



plt.show()
# %%
#%%



import numpy as np

import matplotlib.pyplot as plt



# Pool ALL speeds (all windows, all animals, all taus)

pooled = []

for win_list in all_speed:              # Each window (len=50)

    # win_list shape: (48, 4, variable)

    for animal_idx in range(win_list.shape[0]):

        for tau in range(win_list.shape[1]):

            arr = win_list[animal_idx, tau, :]

            arr = np.asarray(arr, dtype=float)

            arr = arr[~np.isnan(arr)]

            if arr.size > 0:

                pooled.append(arr)

# Combine everything

all_speeds = np.concatenate(pooled)



plt.figure(figsize=(8,5))

plt.hist(all_speeds, bins=175, color='skyblue', histtype='step', linewidth=1.5, density=True)

plt.xlabel("dFC Speed")

plt.ylabel("Density")

plt.title("Histogram of pooled dFC speeds (all animals, all windows, all taus)")

plt.tight_layout()

plt.show()
# %%
#%%



pooled_speeds_per_animal = []

for animal_idx in range(all_speed[0].shape[0]):  # 48 animals

    animal_pooled = []

    for win_list in all_speed:  # each window

        for tau in range(win_list.shape[1]):

            arr = win_list[animal_idx, tau, :]

            arr = np.asarray(arr, dtype=float)

            arr = arr[~np.isnan(arr)]

            if arr.size > 0:

                animal_pooled.append(arr)

    # Pool for this animal across all windows and taus

    if animal_pooled:

        pooled_speeds_per_animal.append(np.concatenate(animal_pooled))

    else:

        pooled_speeds_per_animal.append(np.array([]))
# %%
#%%



for group_name, animal_indices in data.groups.items():

    pooled = []

    for win_list in all_speed:

        for animal_idx in animal_indices:

            for tau in range(win_list.shape[1]):

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        plt.hist(group_speeds, bins=150, alpha=0.5, 

                 label=f"{group_name[0]}-{group_name[1]}",

                 histtype='step', linewidth=1.7, density=True)



plt.xlabel("dFC Speed")

plt.ylabel("Density")

plt.title("Histogram of dFC Speeds by Group (all tau, all windows pooled)")

plt.legend()

plt.tight_layout()

plt.show()
# %%
#%%



import numpy as np

import matplotlib.pyplot as plt



plt.figure(figsize=(10, 6))



for group_name, animal_indices in data.groups.items():

    pooled = []

    for animal_idx in animal_indices:

        animal_tau_speeds = []

        for win_list in all_speed:  # 50 windows

            for tau in range(win_list.shape[1]):  # usually 4

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    animal_tau_speeds.append(arr)

        if animal_tau_speeds:

            pooled.append(np.concatenate(animal_tau_speeds))

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            plt.hist(group_speeds, bins=75, alpha=0.5, 

                     label=f"{group_name[0]}-{group_name[1]}", 

                     histtype='step', linewidth=1.7, density=True)



plt.xlabel("dFC Speed")

plt.ylabel("Density")

plt.title("Histogram of dFC Speeds by Group (all tau, all windows pooled)")

plt.legend()

plt.tight_layout()

plt.show()
# %%
# %%



import numpy as np

import matplotlib.pyplot as plt



plt.figure(figsize=(10, 6))



for group_name, animal_indices in data.groups.items():

    pooled = []

    for animal_idx in animal_indices:

        animal_tau_speeds = []

        for win_list in all_speed:  # 50 windows

            for tau in range(win_list.shape[1]):  # usually 4

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    animal_tau_speeds.append(arr)

        if animal_tau_speeds:

            pooled.append(np.concatenate(animal_tau_speeds))

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            plt.hist(group_speeds, bins=75, alpha=0.5, 

                     label=f"{group_name[0]}-{group_name[1]}", 

                     histtype='step', linewidth=1.7, density=True)



plt.xlabel("dFC Speed")

plt.ylabel("Density")

plt.title("Histogram of dFC Speeds by Group (all tau, all windows pooled)")

plt.legend()

plt.tight_layout()

plt.show()
# %%
import numpy as np

import matplotlib.pyplot as plt



plt.figure(figsize=(10, 6))



for group_name, animal_indices in data.groups.items():

    pooled_speeds = []

    for animal_idx in animal_indices:

        for win_list in all_speed:  # Iterate over windows

            for tau in range(win_list.shape[1]):  # 4 taus typically

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled_speeds.append(arr)

    if pooled_speeds:

        group_speeds = np.concatenate(pooled_speeds)

        if group_speeds.size > 0:

            plt.hist(group_speeds, bins=50, alpha=0.5, 

                     label=f"{group_name[0]}-{group_name[1]}", 

                     histtype='step', linewidth=1.7, density=True)



plt.xlabel("dFC Speed")

plt.ylabel("Density")

plt.title("Histogram of dFC speeds by group (all taus, all windows pooled)")

plt.legend()

plt.tight_layout()

plt.show()



# %%
import matplotlib.pyplot as plt

import seaborn as sns



filtered_df = cog_data_filtered[cog_data_filtered['n_timepoints'] >= 500]

groups = filtered_df.groupby(['genotype', 'treatment']).groups



plt.figure(figsize=(10, 6))

palette = sns.color_palette('tab10', n_colors=len(groups))



for idx, (group, animal_idxs) in enumerate(groups.items()):

    group_speeds = np.concatenate([

        speed.astype(float)

        for animal_idx in animal_idxs

        for speed in speeds_all_T[animal_idx]

    ])

    group_speeds = group_speeds[~np.isnan(group_speeds)]

    

    color = palette[idx]

    plt.hist(group_speeds, bins=100, alpha=0.5, label=f"{group}", 

             histtype='step', linewidth=1.7, density=True, color=color)

    

    # Stats

    median = np.median(group_speeds)

    q05 = np.quantile(group_speeds, 0.05)

    q95 = np.quantile(group_speeds, 0.95)

    

    plt.axvline(median, color=color, linestyle='-', linewidth=1, 

                label=f"{group} median")

    plt.axvline(q05, color=color, linestyle='--', linewidth=1, 

                label=f"{group} q=0.05/0.95")

    plt.axvline(q95, color=color, linestyle='--', linewidth=1)



plt.xlabel("Speed")

plt.ylabel("Density")

plt.title("Histogram of DFC speeds by group")

plt.legend()

plt.tight_layout()

plt.show()



# %%
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize=(10, 6))

palette = sns.color_palette('tab10', n_colors=len(data.groups))



for idx, (group_name, animal_indices) in enumerate(data.groups.items()):

    pooled_speeds = []

    for animal_idx in animal_indices:

        for win_list in all_speed:  # 50 windows

            for tau in range(win_list.shape[1]):  # usually 4 taus

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled_speeds.append(arr)

    if pooled_speeds:

        group_speeds = np.concatenate(pooled_speeds)

        if group_speeds.size > 0:

            color = palette[idx]

            plt.hist(group_speeds, bins=100, alpha=0.5, label=f"{group_name[0]}-{group_name[1]}", 

                     histtype='step', linewidth=1.7, density=True, color=color)

            

            # Stats

            median = np.median(group_speeds)

            q05 = np.quantile(group_speeds, 0.05)

            q95 = np.quantile(group_speeds, 0.95)

            

            plt.axvline(median, color=color, linestyle='-', linewidth=1, 

                        label=f"{group_name[0]}-{group_name[1]} median")

            plt.axvline(q05, color=color, linestyle='--', linewidth=1, 

                        label=f"{group_name[0]}-{group_name[1]} q=0.05/0.95")

            plt.axvline(q95, color=color, linestyle='--', linewidth=1)



plt.xlabel("dFC Speed")

plt.ylabel("Density")

plt.title("Histogram of dFC Speeds by Group (all taus, all windows pooled)")

plt.legend()

plt.tight_layout()

plt.show()



# %%
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize=(10, 6))

palette = sns.color_palette('tab10', n_colors=len(data.groups))



for idx, (group_name, animal_indices) in enumerate(data.groups.items()):

    pooled_speeds = []

    for animal_idx in animal_indices:

        for win_list in all_speed:  # 50 windows

            for tau in range(win_list.shape[1]):  # 4 taus

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled_speeds.append(arr)

    if pooled_speeds:

        group_speeds = np.concatenate(pooled_speeds)

        if group_speeds.size > 0:

            color = palette[idx]

            label = f"{group_name[0]}-{group_name[1]}".lower()  # lowercase

            

            # KDE plot for smooth distribution

            sns.kdeplot(group_speeds, bw_adjust=1.2, 

                        label=label, color=color, linewidth=2.5)

            

            # Stats lines: not in legend (set label to "_nolegend_")

            median = np.median(group_speeds)

            q05 = np.quantile(group_speeds, 0.05)

            q95 = np.quantile(group_speeds, 0.95)

            plt.axvline(median, color=color, linestyle='-', linewidth=1, alpha=0.8, label='_nolegend_')

            plt.axvline(q05, color=color, linestyle='--', linewidth=1, alpha=0.6, label='_nolegend_')

            plt.axvline(q95, color=color, linestyle='--', linewidth=1, alpha=0.6, label='_nolegend_')



plt.xlabel("dFC speed")

plt.ylabel("Density")

plt.title("Distribution of dFC speeds by group\n(all tau, all windows pooled)")

plt.legend(title='group', frameon=True)

plt.tight_layout()

sns.despine(trim=True)

plt.show()



# %%
# %%



import numpy as np

import matplotlib.pyplot as plt



plt.figure(figsize=(10, 6))



for group_name, animal_indices in data.groups.items():

    pooled = []

    for animal_idx in animal_indices:

        animal_tau_speeds = []

        for win_list in all_speed:  # 50 windows

            for tau in range(win_list.shape[1]):  # usually 4

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    animal_tau_speeds.append(arr)

        if animal_tau_speeds:

            pooled.append(np.concatenate(animal_tau_speeds))

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            plt.hist(group_speeds, bins=75, alpha=0.5, 

                     label=f"{group_name[0]}-{group_name[1]}", 

                     histtype='step', linewidth=1.7, density=True)



plt.xlabel("dFC Speed")

plt.ylabel("Density")

plt.title("Histogram of dFC Speeds by Group (all tau, all windows pooled)")

plt.legend()

plt.tight_layout()

plt.show()
# %%
#%%



import numpy as np

import matplotlib.pyplot as plt



plt.figure(figsize=(10, 6))



for group_name, animal_indices in data.groups.items():

    pooled = []

    for animal_idx in animal_indices:

        animal_tau_speeds = []

        for win_list in all_speed:  # 50 windows

            for tau in range(win_list.shape[1]):  # usually 4

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    animal_tau_speeds.append(arr)

        if animal_tau_speeds:

            pooled.append(np.concatenate(animal_tau_speeds))

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            plt.hist(group_speeds, bins=75, alpha=0.5, 

                     label=f"{group_name[0]}-{group_name[1]}", 

                     histtype='step', linewidth=1.7, density=True)



plt.xlabel("dFC Speed")

plt.ylabel("Density")

plt.title("Histogram of dFC Speeds by Group (all tau, all windows pooled)")

plt.legend()

plt.tight_layout()

plt.show()
# %%
#%%



# Set publication style (can customize further)

sns.set_theme(style='white', palette='deep', context='talk')



plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 20, 'legend.fontsize': 14})



# Use a color palette with distinct colors for groups

palette = sns.color_palette('Set2', n_colors=len(data.groups))



plt.figure(figsize=(10, 6))



for (group_name, animal_indices), color in zip(data.groups.items(), palette):

    pooled = []

    for win_list in all_speed:  # Each window size

        for animal_idx in animal_indices:

            for tau in range(win_list.shape[1]):

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            # KDE plot for smooth, publication-ready curves

            sns.kdeplot(group_speeds, 

                        bw_adjust=1.2, 

                        label=f"{group_name[0]}-{group_name[1]}".lower(),

                        color=color, linewidth=2.5, clip=(0, 2))

            

plt.xlabel("dFC Speed", labelpad=10)

plt.ylabel("Density", labelpad=10)

# plt.yscale('log')  # Log scale for better visibility of tails

plt.title("Distribution of dFC Speeds by Group\n(All taus, all windows pooled)", pad=15)

plt.legend(frameon=True, loc='best', title='Group')

plt.tight_layout()



# Remove top/right spines for a cleaner look

sns.despine(trim=True)



plt.show()
# %%
#%%



# Set publication style (can customize further)

sns.set_theme(style='white', palette='deep', context='talk')



plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 20, 'legend.fontsize': 14})



# Use a color palette with distinct colors for groups

palette = sns.color_palette('tab10', n_colors=len(data.groups))



plt.figure(figsize=(10, 6))



for (group_name, animal_indices), color in zip(data.groups.items(), palette):

    pooled = []

    for win_list in all_speed:  # Each window size

        for animal_idx in animal_indices:

            for tau in range(win_list.shape[1]):

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            # KDE plot for smooth, publication-ready curves

            sns.kdeplot(group_speeds, 

                        bw_adjust=1.2, 

                        label=f"{group_name[0]}-{group_name[1]}".lower(),

                        color=color, linewidth=2.5, clip=(0, 2))

            

plt.xlabel("dFC Speed", labelpad=10)

plt.ylabel("Density", labelpad=10)

# plt.yscale('log')  # Log scale for better visibility of tails

plt.title("Distribution of dFC Speeds by Group\n(All taus, all windows pooled)", pad=15)

plt.legend(frameon=True, loc='best', title='Group')

plt.tight_layout()



# Remove top/right spines for a cleaner look

sns.despine(trim=True)



plt.show()
# %%
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



plt.figure(figsize=(10, 6))

palette = sns.color_palette('tab10', n_colors=len(data.groups))



for idx, (group_name, animal_indices) in enumerate(data.groups.items()):

    # Pool all windows and taus for all animals in group

    pooled_speeds = []

    for animal_idx in animal_indices:

        for win_list in all_speed:  # all windows

            for tau in range(win_list.shape[1]):

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled_speeds.append(arr)

    if pooled_speeds:

        group_speeds = np.concatenate(pooled_speeds)

        color = palette[idx]

        label = f"{group_name[0]}-{group_name[1]}".lower()

        # KDE plot for publication quality

        sns.kdeplot(group_speeds, bw_adjust=1.2, 

                    label=label, color=color, linewidth=2.5)

        # Median and quantiles, not added to legend

        median = np.median(group_speeds)

        q05 = np.quantile(group_speeds, 0.05)

        q95 = np.quantile(group_speeds, 0.95)

        plt.axvline(median, color=color, linestyle='-', linewidth=1.2, alpha=0.8, label='_nolegend_')

        plt.axvline(q05, color=color, linestyle='--', linewidth=1, alpha=0.6, label='_nolegend_')

        plt.axvline(q95, color=color, linestyle='--', linewidth=1, alpha=0.6, label='_nolegend_')



plt.xlabel("dFC speed")

plt.ylabel("Density")

plt.title("Distribution of dFC speeds by group\n(all tau, all windows pooled)")

plt.legend(title='group', frameon=True, fontsize=11, ncol=2)

plt.tight_layout()

sns.despine(trim=True)

plt.show()



# %%
#%%



# Set publication style (can customize further)

sns.set_theme(style='white', palette='deep', context='talk')



plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 20, 'legend.fontsize': 14})



# Use a color palette with distinct colors for groups

palette = sns.color_palette('tab10', n_colors=len(data.groups))



plt.figure(figsize=(10, 6))



for (group_name, animal_indices), color in zip(data.groups.items(), palette):

    pooled = []

    for win_list in all_speed:  # Each window size

        for animal_idx in animal_indices:

            for tau in range(win_list.shape[1]):

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            # KDE plot for smooth, publication-ready curves

            sns.kdeplot(group_speeds, 

                        bw_adjust=.2, 

                        label=f"{group_name[0]}-{group_name[1]}".lower(),

                        color=color, linewidth=2.5, clip=(0, 2))

            

plt.xlabel("dFC Speed", labelpad=10)

plt.ylabel("Density", labelpad=10)

# plt.yscale('log')  # Log scale for better visibility of tails

plt.title("Distribution of dFC Speeds by Group\n(All taus, all windows pooled)", pad=15)

plt.legend(frameon=True, loc='best', title='Group')

plt.tight_layout()



# Remove top/right spines for a cleaner look

sns.despine(trim=True)



plt.show()
# %%
#%%



# Set publication style (can customize further)

sns.set_theme(style='white', palette='deep', context='talk')



plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'axes.titlesize': 20, 'legend.fontsize': 14})



# Use a color palette with distinct colors for groups

palette = sns.color_palette('tab10', n_colors=len(data.groups))



plt.figure(figsize=(10, 6))



for (group_name, animal_indices), color in zip(data.groups.items(), palette):

    pooled = []

    for win_list in all_speed:  # Each window size

        for animal_idx in animal_indices:

            for tau in range(win_list.shape[1]):

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled.append(arr)

    if pooled:

        group_speeds = np.concatenate(pooled)

        if group_speeds.size > 0:

            # KDE plot for smooth, publication-ready curves

            sns.kdeplot(group_speeds, 

                        bw_adjust=.5, 

                        label=f"{group_name[0]}-{group_name[1]}".lower(),

                        color=color, linewidth=2.5, clip=(0, 2))

            

plt.xlabel("dFC Speed", labelpad=10)

plt.ylabel("Density", labelpad=10)

# plt.yscale('log')  # Log scale for better visibility of tails

plt.title("Distribution of dFC Speeds by Group\n(All taus, all windows pooled)", pad=15)

plt.legend(frameon=True, loc='best', title='Group')

plt.tight_layout()



# Remove top/right spines for a cleaner look

sns.despine(trim=True)



plt.show()
# %%
# %%



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize=(10, 6))

palette = sns.color_palette('tab10', n_colors=len(data.groups))



for idx, (group_name, animal_indices) in enumerate(data.groups.items()):

    pooled_speeds = []

    for animal_idx in animal_indices:

        for win_list in all_speed:  # 50 windows

            for tau in range(win_list.shape[1]):  # 4 taus

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled_speeds.append(arr)

    if pooled_speeds:

        group_speeds = np.concatenate(pooled_speeds)

        if group_speeds.size > 0:

            color = palette[idx]

            label = f"{group_name[0]}-{group_name[1]}".lower()  # lowercase

            

            # KDE plot for smooth distribution

            sns.kdeplot(group_speeds, bw_adjust=.5, 

                        label=label, color=color, linewidth=2.5)

            

            # Stats lines: not in legend (set label to "_nolegend_")

            median = np.median(group_speeds)

            q05 = np.quantile(group_speeds, 0.05)

            q95 = np.quantile(group_speeds, 0.95)

            plt.axvline(median, color=color, linestyle='-', linewidth=1, alpha=0.8, label='_nolegend_')

            plt.axvline(q05, color=color, linestyle='--', linewidth=1, alpha=0.6, label='_nolegend_')

            plt.axvline(q95, color=color, linestyle='--', linewidth=1, alpha=0.6, label='_nolegend_')



plt.xlabel("dFC speed")

plt.ylabel("Density")

plt.title("Distribution of dFC speeds by group\n(all tau, all windows pooled)")

plt.legend(title='group', frameon=True)

plt.tight_layout()

sns.despine(trim=True)

plt.show()
# %%
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



plt.figure(figsize=(10, 6))

palette = sns.color_palette('tab10', n_colors=len(groups))



for idx, (group, animal_idxs) in enumerate(groups.items()):

    group_speeds = np.concatenate([

        speed.astype(float)

        for animal_idx in animal_idxs

        for speed in speeds_all_T[animal_idx]

    ])

    group_speeds = group_speeds[~np.isnan(group_speeds)]



    color = palette[idx]

    label = f"{group[0]}-{group[1]}".lower()



    # KDE plot instead of histogram for clarity

    sns.kdeplot(group_speeds, bw_adjust=1.2, 

                label=label, color=color, linewidth=2.2)

    # Median and quantiles (not in legend, only visual markers)

    median = np.median(group_speeds)

    q05 = np.quantile(group_speeds, 0.05)

    q95 = np.quantile(group_speeds, 0.95)

    plt.axvline(median, color=color, linestyle='-', linewidth=1.1, alpha=0.8, label='_nolegend_')

    plt.axvline(q05, color=color, linestyle='--', linewidth=1, alpha=0.5, label='_nolegend_')

    plt.axvline(q95, color=color, linestyle='--', linewidth=1, alpha=0.5, label='_nolegend_')



plt.xlabel("dFC speed")

plt.ylabel("Density")

plt.title("Distribution of dFC speeds by group\n(all tau, all windows pooled)")

plt.legend(title='group', frameon=True, fontsize=10, ncol=2)

plt.tight_layout()

sns.despine(trim=True)

plt.show()



# %%
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



plt.figure(figsize=(10, 6))

palette = sns.color_palette('tab10', n_colors=len(data.groups))







for idx, (group, animal_idxs) in enumerate(groups.items()):

    group_speeds = np.concatenate([

        speed.astype(float)

        for animal_idx in animal_idxs

        for speed in speeds_all_T[animal_idx]

    ])

    group_speeds = group_speeds[~np.isnan(group_speeds)]



    color = palette[idx]

    label = f"{group[0]}-{group[1]}".lower()



    # KDE plot instead of histogram for clarity

    sns.kdeplot(group_speeds, bw_adjust=1.2, 

                label=label, color=color, linewidth=2.2)

    # Median and quantiles (not in legend, only visual markers)

    median = np.median(group_speeds)

    q05 = np.quantile(group_speeds, 0.05)

    q95 = np.quantile(group_speeds, 0.95)

    plt.axvline(median, color=color, linestyle='-', linewidth=1.1, alpha=0.8, label='_nolegend_')

    plt.axvline(q05, color=color, linestyle='--', linewidth=1, alpha=0.5, label='_nolegend_')

    plt.axvline(q95, color=color, linestyle='--', linewidth=1, alpha=0.5, label='_nolegend_')



plt.xlabel("dFC speed")

plt.ylabel("Density")

plt.title("Distribution of dFC speeds by group\n(all tau, all windows pooled)")

plt.legend(title='group', frameon=True, fontsize=10, ncol=2)

plt.tight_layout()

sns.despine(trim=True)

plt.show()



# %%
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



plt.figure(figsize=(10, 6))

palette = sns.color_palette('tab10', n_colors=len(data.groups))







for idx, (group, animal_idxs) in enumerate(data.groups.items()):





    group_speeds = np.concatenate([

        speed.astype(float)

        for animal_idx in animal_idxs

        for speed in speeds_all_T[animal_idx]

    ])

    group_speeds = group_speeds[~np.isnan(group_speeds)]



    color = palette[idx]

    label = f"{group[0]}-{group[1]}".lower()



    # KDE plot instead of histogram for clarity

    sns.kdeplot(group_speeds, bw_adjust=1.2, 

                label=label, color=color, linewidth=2.2)

    # Median and quantiles (not in legend, only visual markers)

    median = np.median(group_speeds)

    q05 = np.quantile(group_speeds, 0.05)

    q95 = np.quantile(group_speeds, 0.95)

    plt.axvline(median, color=color, linestyle='-', linewidth=1.1, alpha=0.8, label='_nolegend_')

    plt.axvline(q05, color=color, linestyle='--', linewidth=1, alpha=0.5, label='_nolegend_')

    plt.axvline(q95, color=color, linestyle='--', linewidth=1, alpha=0.5, label='_nolegend_')



plt.xlabel("dFC speed")

plt.ylabel("Density")

plt.title("Distribution of dFC speeds by group\n(all tau, all windows pooled)")

plt.legend(title='group', frameon=True, fontsize=10, ncol=2)

plt.tight_layout()

sns.despine(trim=True)

plt.show()



# %%
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



plt.figure(figsize=(10, 6))

palette = sns.color_palette('tab10', n_colors=len(data.groups))



for idx, (group_name, animal_indices) in enumerate(data.groups.items()):

    pooled_speeds = []

    for animal_idx in animal_indices:

        for win_list in all_speed:  # Iterate windows

            for tau in range(win_list.shape[1]):  # Iterate taus (typically 4)

                arr = win_list[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled_speeds.append(arr)

    if pooled_speeds:

        group_speeds = np.concatenate(pooled_speeds)

        color = palette[idx]

        label = f"{group_name[0]}-{group_name[1]}".lower()

        # KDE plot for publication

        sns.kdeplot(group_speeds, bw_adjust=1.2, 

                    label=label, color=color, linewidth=2.2)

        # Median and quantiles (not in legend)

        median = np.median(group_speeds)

        q05 = np.quantile(group_speeds, 0.05)

        q95 = np.quantile(group_speeds, 0.95)

        plt.axvline(median, color=color, linestyle='-', linewidth=1.1, alpha=0.8, label='_nolegend_')

        plt.axvline(q05, color=color, linestyle='--', linewidth=1, alpha=0.5, label='_nolegend_')

        plt.axvline(q95, color=color, linestyle='--', linewidth=1, alpha=0.5, label='_nolegend_')



plt.xlabel("dFC speed")

plt.ylabel("Density")

plt.title("Distribution of dFC speeds by group\n(all tau, all windows pooled)")

plt.legend(title='group', frameon=True, fontsize=10, ncol=2)

plt.tight_layout()

sns.despine(trim=True)

plt.show()



# %%
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



window_sizes = time_window_range  # your list/array of window sizes

n_windows = len(all_speed)        # should match window_sizes



palette = sns.color_palette('tab10', n_colors=len(data.groups))



plt.figure(figsize=(10,6))



for idx, (group_name, animal_indices) in enumerate(data.groups.items()):

    medians_per_window = []

    for win_idx in range(n_windows):

        win_arr = all_speed[win_idx]  # shape: (n_animals, n_taus, n_timepoints)

        # Pool all animals, all taus for this group at this window

        speeds_this_window = []

        for animal_idx in animal_indices:

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    speeds_this_window.append(arr)

        if speeds_this_window:

            flat_speeds = np.concatenate(speeds_this_window)

            median_speed = np.median(flat_speeds)

        else:

            median_speed = np.nan

        medians_per_window.append(median_speed)

    label = f"{group_name[0]}-{group_name[1]}".lower()

    plt.plot(window_sizes, medians_per_window, marker='o', label=label, color=palette[idx], linewidth=2)



plt.xlabel("Time Window Size")

plt.ylabel("Median dFC speed (group, all tau pooled)")

plt.title("Median dFC speed vs. window size by group")

plt.legend(title='group', fontsize=10, ncol=2)

plt.tight_layout()

sns.despine(trim=True)

plt.show()



# %%
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



window_sizes = time_window_range  # Your array/list of window sizes

n_windows = len(all_speed)

palette = sns.color_palette('tab10', n_colors=len(data.groups))



plt.figure(figsize=(10,6))



for idx, (group_name, animal_indices) in enumerate(data.groups.items()):

    medians_per_window = []

    q25_per_window = []

    q75_per_window = []

    for win_idx in range(n_windows):

        win_arr = all_speed[win_idx]  # shape: (n_animals, n_taus, n_timepoints)

        # Pool all animals and all taus for this group and window

        speeds_this_window = []

        for animal_idx in animal_indices:

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    speeds_this_window.append(arr)

        if speeds_this_window:

            flat_speeds = np.concatenate(speeds_this_window)

            median = np.median(flat_speeds)

            q25 = np.quantile(flat_speeds, 0.25)

            q75 = np.quantile(flat_speeds, 0.75)

        else:

            median = np.nan

            q25 = np.nan

            q75 = np.nan

        medians_per_window.append(median)

        q25_per_window.append(q25)

        q75_per_window.append(q75)

    color = palette[idx]

    label = f"{group_name[0]}-{group_name[1]}".lower()

    plt.plot(window_sizes, medians_per_window, marker='o', label=label, color=color, linewidth=2)

    plt.fill_between(window_sizes, q25_per_window, q75_per_window, color=color, alpha=0.18)



plt.xlabel("Time Window Size")

plt.ylabel("Median dFC Speed (group, all tau pooled)")

plt.title("Median dFC Speed vs. Window Size by Group\nShading = 25–75% quantile")

plt.legend(title='group', fontsize=10, ncol=2)

plt.tight_layout()

sns.despine(trim=True)

plt.show()



# %%
#%%



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



window_sizes = time_window_range  # Your array/list of window sizes

n_windows = len(all_speed)

palette = sns.color_palette('tab10', n_colors=len(data.groups))



plt.figure(figsize=(10,6))



for idx, (group_name, animal_indices) in enumerate(data.groups.items()):

    medians_per_window = []

    q25_per_window = []

    q75_per_window = []

    for win_idx in range(n_windows):

        win_arr = all_speed[win_idx]  # shape: (n_animals, n_taus, n_timepoints)

        # Pool all animals and all taus for this group and window

        speeds_this_window = []

        for animal_idx in animal_indices:

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    speeds_this_window.append(arr)

        if speeds_this_window:

            flat_speeds = np.concatenate(speeds_this_window)

            median = np.median(flat_speeds)

            q25 = np.quantile(flat_speeds, 0.25)

            q75 = np.quantile(flat_speeds, 0.75)

        else:

            median = np.nan

            q25 = np.nan

            q75 = np.nan

        medians_per_window.append(median)

        q25_per_window.append(q25)

        q75_per_window.append(q75)

    color = palette[idx]

    label = f"{group_name[0]}-{group_name[1]}".lower()

    plt.plot(window_sizes, medians_per_window, marker='o', label=label, color=color, linewidth=2)

    plt.fill_between(window_sizes, q25_per_window, q75_per_window, color=color, alpha=0.18)



plt.xlabel("Time Window Size")

plt.ylabel("Median dFC Speed (group, all tau pooled)")

plt.title("Median dFC Speed vs. Window Size by Group\nShading = 25–75% quantile")

plt.legend(title='group', fontsize=10, ncol=2)

plt.tight_layout()

sns.despine(trim=True)

plt.show()
# %%
#%%



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



window_sizes = time_window_range  # Your array/list of window sizes

n_windows = len(all_speed)

palette = sns.color_palette('tab10', n_colors=len(data.groups))



plt.figure(figsize=(15,6))



for idx, (group_name, animal_indices) in enumerate(data.groups.items()):

    medians_per_window = []

    q25_per_window = []

    q75_per_window = []

    for win_idx in range(n_windows):

        win_arr = all_speed[win_idx]  # shape: (n_animals, n_taus, n_timepoints)

        # Pool all animals and all taus for this group and window

        speeds_this_window = []

        for animal_idx in animal_indices:

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    speeds_this_window.append(arr)

        if speeds_this_window:

            flat_speeds = np.concatenate(speeds_this_window)

            median = np.median(flat_speeds)

            q25 = np.quantile(flat_speeds, 0.25)

            q75 = np.quantile(flat_speeds, 0.75)

        else:

            median = np.nan

            q25 = np.nan

            q75 = np.nan

        medians_per_window.append(median)

        q25_per_window.append(q25)

        q75_per_window.append(q75)

    color = palette[idx]

    label = f"{group_name[0]}-{group_name[1]}".lower()

    plt.plot(window_sizes, medians_per_window, marker='o', label=label, color=color, linewidth=2)

    plt.fill_between(window_sizes, q25_per_window, q75_per_window, color=color, alpha=0.18)



plt.xlabel("Time Window Size")

plt.ylabel("Median dFC Speed (group, all tau pooled)")

plt.title("Median dFC Speed vs. Window Size by Group\nShading = 25–75% quantile")

plt.legend(title='group', fontsize=10, ncol=2)

plt.tight_layout()

sns.despine(trim=True)

plt.show()
# %%
#%%



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



window_sizes = time_window_range  # Your array/list of window sizes

n_windows = len(all_speed)

palette = sns.color_palette('tab10', n_colors=len(data.groups))



plt.figure(figsize=(13,6))



for idx, (group_name, animal_indices) in enumerate(data.groups.items()):

    medians_per_window = []

    q25_per_window = []

    q75_per_window = []

    for win_idx in range(n_windows):

        win_arr = all_speed[win_idx]  # shape: (n_animals, n_taus, n_timepoints)

        # Pool all animals and all taus for this group and window

        speeds_this_window = []

        for animal_idx in animal_indices:

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    speeds_this_window.append(arr)

        if speeds_this_window:

            flat_speeds = np.concatenate(speeds_this_window)

            median = np.median(flat_speeds)

            q25 = np.quantile(flat_speeds, 0.25)

            q75 = np.quantile(flat_speeds, 0.75)

        else:

            median = np.nan

            q25 = np.nan

            q75 = np.nan

        medians_per_window.append(median)

        q25_per_window.append(q25)

        q75_per_window.append(q75)

    color = palette[idx]

    label = f"{group_name[0]}-{group_name[1]}".lower()

    plt.plot(window_sizes, medians_per_window, marker='o', label=label, color=color, linewidth=2)

    plt.fill_between(window_sizes, q25_per_window, q75_per_window, color=color, alpha=0.18)



plt.xlabel("Time Window Size")

plt.ylabel("Median dFC Speed (group, all tau pooled)")

plt.title("Median dFC Speed vs. Window Size by Group\nShading = 25–75% quantile")

plt.legend(title='group', fontsize=10, ncol=2)

plt.tight_layout()

sns.despine(trim=True)

plt.show()
# %%
#%%



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



window_sizes = time_window_range  # Your array/list of window sizes

n_windows = len(all_speed)

palette = sns.color_palette('tab10', n_colors=len(data.groups))



plt.figure(figsize=(13,6))



for idx, (group_name, animal_indices) in enumerate(data.groups.items()):

    medians_per_window = []

    q25_per_window = []

    q75_per_window = []

    for win_idx in range(n_windows):

        win_arr = all_speed[win_idx]  # shape: (n_animals, n_taus, n_timepoints)

        # Pool all animals and all taus for this group and window

        speeds_this_window = []

        for animal_idx in animal_indices:

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    speeds_this_window.append(arr)

        if speeds_this_window:

            flat_speeds = np.concatenate(speeds_this_window)

            median = np.median(flat_speeds)

            q25 = np.quantile(flat_speeds, 0.25)

            q75 = np.quantile(flat_speeds, 0.75)

        else:

            median = np.nan

            q25 = np.nan

            q75 = np.nan

        medians_per_window.append(median)

        q25_per_window.append(q25)

        q75_per_window.append(q75)

    color = palette[idx]

    label = f"{group_name[0]}-{group_name[1]}".lower()

    plt.plot(window_sizes, medians_per_window, marker='.', label=label, color=color, linewidth=2)

    plt.fill_between(window_sizes, q25_per_window, q75_per_window, color=color, alpha=0.18)



plt.xlabel("Time Window Size")

plt.ylabel("Median dFC Speed (group, all tau pooled)")

plt.title("Median dFC Speed vs. Window Size by Group\nShading = 25–75% quantile")

plt.legend(title='group', fontsize=10, ncol=2)

plt.tight_layout()

sns.despine(trim=True)

plt.show()
# %%
#%%



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



window_sizes = time_window_range  # Your array/list of window sizes

n_windows = len(all_speed)

palette = sns.color_palette('tab10', n_colors=len(data.groups))



plt.figure(figsize=(13,6))



for idx, (group_name, animal_indices) in enumerate(data.groups.items()):

    medians_per_window = []

    q25_per_window = []

    q75_per_window = []

    for win_idx in range(n_windows):

        win_arr = all_speed[win_idx]  # shape: (n_animals, n_taus, n_timepoints)

        # Pool all animals and all taus for this group and window

        speeds_this_window = []

        for animal_idx in animal_indices:

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    speeds_this_window.append(arr)

        if speeds_this_window:

            flat_speeds = np.concatenate(speeds_this_window)

            median = np.median(flat_speeds)

            q25 = np.quantile(flat_speeds, 0.25)

            q75 = np.quantile(flat_speeds, 0.75)

        else:

            median = np.nan

            q25 = np.nan

            q75 = np.nan

        medians_per_window.append(median)

        q25_per_window.append(q25)

        q75_per_window.append(q75)

    color = palette[idx]

    label = f"{group_name[0]}-{group_name[1]}".lower()

    plt.plot(window_sizes, medians_per_window, marker='0', label=label, color=color, linewidth=2)

    plt.fill_between(window_sizes, q25_per_window, q75_per_window, color=color, alpha=0.18)



plt.xlabel("Time Window Size")

plt.ylabel("Median dFC Speed (group, all tau pooled)")

plt.title("Median dFC Speed vs. Window Size by Group\nShading = 25–75% quantile")

plt.legend(title='group', fontsize=10, ncol=2)

plt.tight_layout()

sns.despine(trim=True)

plt.show()
# %%
#%%



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



window_sizes = time_window_range  # Your array/list of window sizes

n_windows = len(all_speed)

palette = sns.color_palette('tab10', n_colors=len(data.groups))



plt.figure(figsize=(13,6))



for idx, (group_name, animal_indices) in enumerate(data.groups.items()):

    medians_per_window = []

    q25_per_window = []

    q75_per_window = []

    for win_idx in range(n_windows):

        win_arr = all_speed[win_idx]  # shape: (n_animals, n_taus, n_timepoints)

        # Pool all animals and all taus for this group and window

        speeds_this_window = []

        for animal_idx in animal_indices:

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    speeds_this_window.append(arr)

        if speeds_this_window:

            flat_speeds = np.concatenate(speeds_this_window)

            median = np.median(flat_speeds)

            q25 = np.quantile(flat_speeds, 0.25)

            q75 = np.quantile(flat_speeds, 0.75)

        else:

            median = np.nan

            q25 = np.nan

            q75 = np.nan

        medians_per_window.append(median)

        q25_per_window.append(q25)

        q75_per_window.append(q75)

    color = palette[idx]

    label = f"{group_name[0]}-{group_name[1]}".lower()

    plt.plot(window_sizes, medians_per_window, marker='o', label=label, color=color, linewidth=2)

    plt.fill_between(window_sizes, q25_per_window, q75_per_window, color=color, alpha=0.18)



plt.xlabel("Time Window Size")

plt.ylabel("Median dFC Speed (group, all tau pooled)")

plt.title("Median dFC Speed vs. Window Size by Group\nShading = 25–75% quantile")

plt.legend(title='group', fontsize=10, ncol=2)

plt.tight_layout()

sns.despine(trim=True)

plt.show()
# %%
#%%



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



window_sizes = time_window_range  # Your array/list of window sizes

n_windows = len(all_speed)

palette = sns.color_palette('tab10', n_colors=len(data.groups))



plt.figure(figsize=(13,6))



for idx, (group_name, animal_indices) in enumerate(data.groups.items()):

    medians_per_window = []

    q25_per_window = []

    q75_per_window = []

    for win_idx in range(n_windows):

        win_arr = all_speed[win_idx]  # shape: (n_animals, n_taus, n_timepoints)

        # Pool all animals and all taus for this group and window

        speeds_this_window = []

        for animal_idx in animal_indices:

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    speeds_this_window.append(arr)

        if speeds_this_window:

            flat_speeds = np.concatenate(speeds_this_window)

            median = np.median(flat_speeds)

            q25 = np.quantile(flat_speeds, 0.25)

            q75 = np.quantile(flat_speeds, 0.75)

        else:

            median = np.nan

            q25 = np.nan

            q75 = np.nan

        medians_per_window.append(median)

        q25_per_window.append(q25)

        q75_per_window.append(q75)

    color = palette[idx]

    label = f"{group_name[0]}-{group_name[1]}".lower()

    plt.plot(window_sizes, medians_per_window, marker='o', label=label, color=color, linewidth=2)

    plt.fill_between(window_sizes, q25_per_window, q75_per_window, color=color, alpha=0.1)



plt.xlabel("Time Window Size")

plt.ylabel("Median dFC Speed (group, all tau pooled)")

plt.title("Median dFC Speed vs. Window Size by Group\nShading = 25–75% quantile")

plt.legend(title='group', fontsize=10, ncol=2)

plt.tight_layout()

sns.despine(trim=True)

plt.show()
# %%
#%%



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



window_sizes = time_window_range  # Your array/list of window sizes

n_windows = len(all_speed)

palette = sns.color_palette('tab10', n_colors=len(data.groups))



plt.figure(figsize=(13,6))



for idx, (group_name, animal_indices) in enumerate(data.groups.items()):

    medians_per_window = []

    q25_per_window = []

    q75_per_window = []

    for win_idx in range(n_windows):

        win_arr = all_speed[win_idx]  # shape: (n_animals, n_taus, n_timepoints)

        # Pool all animals and all taus for this group and window

        speeds_this_window = []

        for animal_idx in animal_indices:

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    speeds_this_window.append(arr)

        if speeds_this_window:

            flat_speeds = np.concatenate(speeds_this_window)

            median = np.median(flat_speeds)

            q25 = np.quantile(flat_speeds, 0.25)

            q75 = np.quantile(flat_speeds, 0.75)

        else:

            median = np.nan

            q25 = np.nan

            q75 = np.nan

        medians_per_window.append(median)

        q25_per_window.append(q25)

        q75_per_window.append(q75)

    color = palette[idx]

    label = f"{group_name[0]}-{group_name[1]}".lower()

    plt.plot(window_sizes, medians_per_window, marker='o', label=label, color=color, linewidth=2)

    plt.fill_between(window_sizes, q25_per_window, q75_per_window, color=color, alpha=0.2)



plt.xlabel("Time Window Size")

plt.ylabel("Median dFC Speed (group, all tau pooled)")

plt.title("Median dFC Speed vs. Window Size by Group\nShading = 25–75% quantile")

plt.legend(title='group', fontsize=10, ncol=2)

plt.tight_layout()

sns.despine(trim=True)

plt.show()
# %%
#%%



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



window_sizes = time_window_range  # Your array/list of window sizes

n_windows = len(all_speed)

palette = sns.color_palette('tab10', n_colors=len(data.groups))



plt.figure(figsize=(13,6))



for idx, (group_name, animal_indices) in enumerate(data.groups.items()):

    medians_per_window = []

    q25_per_window = []

    q75_per_window = []

    for win_idx in range(n_windows):

        win_arr = all_speed[win_idx]  # shape: (n_animals, n_taus, n_timepoints)

        # Pool all animals and all taus for this group and window

        speeds_this_window = []

        for animal_idx in animal_indices:

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    speeds_this_window.append(arr)

        if speeds_this_window:

            flat_speeds = np.concatenate(speeds_this_window)

            median = np.median(flat_speeds)

            q25 = np.quantile(flat_speeds, 0.25)

            q75 = np.quantile(flat_speeds, 0.75)

        else:

            median = np.nan

            q25 = np.nan

            q75 = np.nan

        medians_per_window.append(median)

        q25_per_window.append(q25)

        q75_per_window.append(q75)

    color = palette[idx]

    label = f"{group_name[0]}-{group_name[1]}".lower()

    plt.plot(window_sizes, medians_per_window, marker='o', label=label, color=color, linewidth=2)

    plt.fill_between(window_sizes, q25_per_window, q75_per_window, color=color, alpha=0.05)



plt.xlabel("Time Window Size")

plt.ylabel("Median dFC Speed (group, all tau pooled)")

plt.title("Median dFC Speed vs. Window Size by Group\nShading = 25–75% quantile")

plt.legend(title='group', fontsize=10, ncol=2)

plt.tight_layout()

sns.despine(trim=True)

plt.show()
# %%
#%%



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



window_sizes = time_window_range  # Your array/list of window sizes

n_windows = len(all_speed)

palette = sns.color_palette('tab10', n_colors=len(data.groups))



plt.figure(figsize=(13,6))



for idx, (group_name, animal_indices) in enumerate(data.groups.items()):

    medians_per_window = []

    q25_per_window = []

    q75_per_window = []

    for win_idx in range(n_windows):

        win_arr = all_speed[win_idx]  # shape: (n_animals, n_taus, n_timepoints)

        # Pool all animals and all taus for this group and window

        speeds_this_window = []

        for animal_idx in animal_indices:

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    speeds_this_window.append(arr)

        if speeds_this_window:

            flat_speeds = np.concatenate(speeds_this_window)

            median = np.median(flat_speeds)

            q25 = np.quantile(flat_speeds, 0.25)

            q75 = np.quantile(flat_speeds, 0.75)

        else:

            median = np.nan

            q25 = np.nan

            q75 = np.nan

        medians_per_window.append(median)

        q25_per_window.append(q25)

        q75_per_window.append(q75)

    color = palette[idx]

    label = f"{group_name[0]}-{group_name[1]}".lower()

    plt.plot(window_sizes, medians_per_window, marker='o', label=label, color=color, linewidth=2)

    plt.fill_between(window_sizes, q25_per_window, q75_per_window, color=color, alpha=0.1)



plt.xlabel("Time Window Size")

plt.ylabel("Median dFC Speed (group, all tau pooled)")

plt.title("Median dFC Speed vs. Window Size by Group\nShading = 25–75% quantile")

plt.legend(title='group', fontsize=10, ncol=2)

plt.tight_layout()

sns.despine(trim=True)

plt.show()
# %%
import numpy as np

import matplotlib.pyplot as plt



quantile_levels = np.linspace(0.05, 0.95, 19)  # 0.05, 0.1, ..., 0.95

window_sizes = time_window_range

n_windows = len(window_sizes)

n_q = len(quantile_levels)



# Choose group (replace [0] with desired index or group key)

group_name = list(data.groups.keys())[0]

animal_indices = list(data.groups.values())[0]

# Or: group_name, animal_indices = list(data.groups.items())[0]



speed_matrix = np.full((n_q, n_windows), np.nan)



for win_idx in range(n_windows):

    win_arr = all_speed[win_idx]  # (n_animals, n_taus, n_timepoints)

    speeds_this_window = []

    for animal_idx in animal_indices:

        for tau in range(win_arr.shape[1]):

            arr = win_arr[animal_idx, tau, :]

            arr = np.asarray(arr, dtype=float)

            arr = arr[~np.isnan(arr)]

            if arr.size > 0:

                speeds_this_window.append(arr)

    if speeds_this_window:

        flat_speeds = np.concatenate(speeds_this_window)

        speed_matrix[:, win_idx] = [np.quantile(flat_speeds, q) for q in quantile_levels]

    # else: speed_matrix[:, win_idx] is already np.nan



plt.figure(figsize=(10, 5))

im = plt.imshow(

    np.log(speed_matrix), 

    aspect='auto', 

    origin='lower', 

    extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

    cmap='viridis'

)

plt.colorbar(im, label='log(dFC Speed)')

plt.clim(np.nanmin(np.log(speed_matrix)), np.nanmax(np.log(speed_matrix)))

plt.xlabel('Window Size')

plt.ylabel('Quantile')

plt.title(f'Speed quantile matrix\n(Group: {group_name[0]}-{group_name[1]})')

plt.tight_layout()

plt.show()



# %%
import numpy as np

import matplotlib.pyplot as plt



quantile_levels = np.linspace(0.05, 0.95, 19)  # 0.05, 0.1, ..., 0.95

window_sizes = time_window_range

n_windows = len(window_sizes)

n_q = len(quantile_levels)



# Choose group (replace [0] with desired index or group key)

group_name = list(data.groups.keys())[0]

animal_indices = list(data.groups.values())[0]

# Or: group_name, animal_indices = list(data.groups.items())[0]



speed_matrix = np.full((n_q, n_windows), np.nan)



for win_idx in range(n_windows):

    win_arr = all_speed[win_idx]  # (n_animals, n_taus, n_timepoints)

    speeds_this_window = []

    for animal_idx in animal_indices:

        for tau in range(win_arr.shape[1]):

            arr = win_arr[animal_idx, tau, :]

            arr = np.asarray(arr, dtype=float)

            arr = arr[~np.isnan(arr)]

            if arr.size > 0:

                speeds_this_window.append(arr)

    if speeds_this_window:

        flat_speeds = np.concatenate(speeds_this_window)

        speed_matrix[:, win_idx] = [np.quantile(flat_speeds, q) for q in quantile_levels]

    # else: speed_matrix[:, win_idx] is already np.nan



plt.figure(figsize=(10, 5))

im = plt.imshow(

    speed_matrix, 

    aspect='auto', 

    origin='lower', 

    extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

    cmap='viridis'

)

plt.colorbar(im, label='log(dFC Speed)')

plt.clim(np.nanmin(speed_matrix), np.nanmax(speed_matrix))





plt.xlabel('Window Size')

plt.ylabel('Quantile')

plt.title(f'Speed quantile matrix\n(Group: {group_name[0]}-{group_name[1]})')

plt.tight_layout()

plt.show()



# %%
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.colors as mcolors



quantile_levels = np.linspace(0.05, 0.95, 19)

window_sizes = time_window_range

n_windows = len(window_sizes)

n_q = len(quantile_levels)



for group_name, animal_indices in data.groups.items():

    speed_matrix = np.full((n_q, n_windows), np.nan)

    for win_idx in range(n_windows):

        win_arr = all_speed[win_idx]  # (n_animals, n_taus, n_timepoints)

        speeds_this_window = []

        for animal_idx in animal_indices:

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                arr = arr[arr > 0]  # Only positive for log scale

                if arr.size > 0:

                    speeds_this_window.append(arr)

        if speeds_this_window:

            flat_speeds = np.concatenate(speeds_this_window)

            if flat_speeds.size > 0:

                speed_matrix[:, win_idx] = [np.quantile(flat_speeds, q) for q in quantile_levels]



    # Plot only if there is at least one valid value

    if np.nansum(speed_matrix) > 0:

        plt.figure(figsize=(10, 5))

        # Mask zeros/NaNs for lognorm

        valid = ~np.isnan(speed_matrix) & (speed_matrix > 0)

        vmin = np.nanmin(speed_matrix[valid]) if np.any(valid) else 1e-6

        vmax = np.nanmax(speed_matrix[valid]) if np.any(valid) else 1



        im = plt.imshow(

            speed_matrix,

            aspect='auto',

            origin='lower',

            extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

            cmap='viridis',

            norm=mcolors.LogNorm(vmin=vmin, vmax=vmax)

        )

        plt.colorbar(im, label='dFC Speed (log scale)')

        plt.xlabel('Window Size')

        plt.ylabel('Quantile')

        plt.title(f'Speed quantile matrix\nGroup: {group_name[0]}-{group_name[1]}')

        plt.tight_layout()

        plt.show()

    else:

        print(f"Skipping group {group_name}: no valid speeds for plotting.")



# %%
import numpy as np



quantile_levels = np.linspace(0, 1, 100)

window_sizes = time_window_range

n_windows = len(window_sizes)

n_q = len(quantile_levels)

group_names = list(data.groups.keys())

n_groups = len(group_names)



speed_matrices = []



for group_name in group_names:

    animal_indices = data.groups[group_name]

    speed_matrix = np.full((n_q, n_windows), np.nan)

    for win_idx in range(n_windows):

        win_arr = all_speed[win_idx]  # (n_animals, n_taus, n_timepoints)

        speeds_this_window = []

        for animal_idx in animal_indices:

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    speeds_this_window.append(arr)

        if speeds_this_window:

            flat_speeds = np.concatenate(speeds_this_window)

            if flat_speeds.size > 0:

                speed_matrix[:, win_idx] = [np.quantile(flat_speeds, q) for q in quantile_levels]

            # else: already nan

        # else: already nan

    speed_matrices.append(speed_matrix)



# %%
import numpy as np

import matplotlib.pyplot as plt



nan_gap = np.full((1, n_windows), np.nan)

full_matrix = speed_matrices[0].copy()

yticks = []

yticklabels = []



for i, mat in enumerate(speed_matrices):

    if i > 0:

        full_matrix = np.vstack([full_matrix, nan_gap, mat])

    # Center of each group block for yticks

    start = full_matrix.shape[0] - mat.shape[0]

    yticks.append(start + mat.shape[0] // 2)

    label = f"{group_names[i][0]}-{group_names[i][1]}".lower()

    yticklabels.append(label)



plt.figure(figsize=(12, 2.5 * len(speed_matrices)))

im = plt.imshow(

    full_matrix,

    aspect='auto',

    origin='lower',

    extent=[window_sizes[0], window_sizes[-1], 0, full_matrix.shape[0]],

    cmap='viridis'

)

plt.colorbar(im, label='dFC Speed')

plt.xlabel('Window Size')

plt.ylabel('Quantile / Group')

plt.yticks(yticks, yticklabels)

plt.title('Speed quantile matrices, all groups stacked')

plt.tight_layout()

plt.show()



# %%
import numpy as np

import matplotlib.pyplot as plt



quantile_levels = np.linspace(0, 1, 100)

n_windows = len(window_sizes)

n_q = len(quantile_levels)



group_names = list(data.groups.keys())

n_groups = len(group_names)

# Auto-arrange grid (e.g., 2x2 for 4 groups)

n_rows = int(np.ceil(np.sqrt(n_groups)))

n_cols = int(np.ceil(n_groups / n_rows))



fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows), sharex=True, sharey=True)

axes = axes.flatten()



vmin = np.inf

vmax = -np.inf

speed_matrices = []



# First pass: collect quantile matrices and color scale range

for group_name in group_names:

    animal_indices = data.groups[group_name]

    speed_matrix = np.full((n_q, n_windows), np.nan)

    for win_idx in range(n_windows):

        win_arr = all_speed[win_idx]  # (n_animals, n_taus, n_timepoints)

        speeds_this_window = []

        for animal_idx in animal_indices:

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    speeds_this_window.append(arr)

        if speeds_this_window:

            flat_speeds = np.concatenate(speeds_this_window)

            if flat_speeds.size > 0:

                speed_matrix[:, win_idx] = [np.quantile(flat_speeds, q) for q in quantile_levels]

        # else already nan

    valid = ~np.isnan(speed_matrix)

    if np.any(valid):

        vmin = min(vmin, np.nanmin(speed_matrix))

        vmax = max(vmax, np.nanmax(speed_matrix))

    speed_matrices.append(speed_matrix)



# Second pass: plotting

for idx, group_name in enumerate(group_names):

    ax = axes[idx]

    im = ax.imshow(

        speed_matrices[idx],

        aspect='auto',

        origin='lower',

        extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

        cmap='magma',

        vmin=vmin, vmax=vmax

    )

    ax.set_title(f"{group_name[0]}-{group_name[1]}".lower())

    ax.set_xlabel('Window Size')

    ax.set_ylabel('Quantile')

    ax.label_outer()  # Only show outer labels



# Hide unused axes (if any)

for j in range(idx+1, len(axes)):

    axes[j].axis('off')



fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.03, pad=0.04, label='dFC Speed')

plt.tight_layout()

plt.show()



# %%
for group_idx, group_name in enumerate(group_names):

    mat = speed_matrices[group_idx]  # (n_q, n_windows)

    plt.plot(window_sizes, mat[50, :], label=f'{group_name} median')  # 50th percentile (median)

    plt.plot(window_sizes, mat[25, :], '--', label=f'{group_name} q25')  # 25th percentile

    plt.plot(window_sizes, mat[75, :], '--', label=f'{group_name} q75')  # 75th percentile

plt.xlabel('Window size')

plt.ylabel('dFC speed')

plt.legend()

plt.show()



# %%
for group_idx, group_name in enumerate(group_names):

    mat = speed_matrices[group_idx]

    plt.imshow(mat, aspect='auto', origin='lower', ...)

    # Overlay the median curve (and maybe IQR)

    plt.plot(window_sizes, mat[50, :], color='w', lw=2, label='median')

    plt.plot(window_sizes, mat[25, :], color='w', ls='--', lw=1, label='q25')

    plt.plot(window_sizes, mat[75, :], color='w', ls='--', lw=1, label='q75')

    plt.legend()

    plt.show()



# %%
for group_idx, group_name in enumerate(group_names):

    mat = speed_matrices[group_idx]

    plt.imshow(mat, aspect='auto', origin='lower')





    # Overlay the median curve (and maybe IQR)

    plt.plot(window_sizes, mat[50, :], color='w', lw=2, label='median')

    plt.plot(window_sizes, mat[25, :], color='w', ls='--', lw=1, label='q25')

    plt.plot(window_sizes, mat[75, :], color='w', ls='--', lw=1, label='q75')

    plt.legend()

    plt.show()

# %%
import matplotlib.pyplot as plt

import numpy as np



# Assume: speed_matrices, window_sizes, quantile_levels, group_names as defined above



q25_idx = np.searchsorted(quantile_levels, 0.25)

q50_idx = np.searchsorted(quantile_levels, 0.5)

q75_idx = np.searchsorted(quantile_levels, 0.75)



for group_idx, group_name in enumerate(group_names):

    mat = speed_matrices[group_idx]  # (n_q, n_windows)

    plt.figure(figsize=(12, 5))

    im = plt.imshow(

        mat, aspect='auto', origin='lower',

        extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

        cmap='magma'

    )

    # Overlay quantile curves

    plt.plot(window_sizes, mat[q50_idx, :], color='w', linewidth=2.2, label='Median')

    plt.plot(window_sizes, mat[q25_idx, :], color='w', linestyle='--', linewidth=1.2, label='Q25/Q75')

    plt.plot(window_sizes, mat[q75_idx, :], color='w', linestyle='--', linewidth=1.2)

    plt.colorbar(im, label='dFC Speed')

    plt.xlabel('Window Size')

    plt.ylabel('Quantile')

    plt.title(f'Quantile matrix with overlays\nGroup: {group_name[0]}-{group_name[1]}')

    plt.legend()

    plt.tight_layout()

    plt.show()



# %%
for group_idx, group_name in enumerate(group_names):

    mat = speed_matrices[group_idx]

    plt.figure(figsize=(10, 6))

    for qidx, q in enumerate(quantile_levels):

        plt.plot(window_sizes, mat[qidx, :], color='gray', alpha=0.2)

    plt.plot(window_sizes, mat[q50_idx, :], color='k', linewidth=2.5, label='Median')

    plt.xlabel('Window Size')

    plt.ylabel('dFC Speed')

    plt.title(f'Quantile curves\nGroup: {group_name[0]}-{group_name[1]}')

    plt.legend()

    plt.tight_layout()

    plt.show()



# %%
import numpy as np

import matplotlib.pyplot as plt



quantile_levels = np.linspace(0, 1, 20)

n_windows = len(window_sizes)

n_q = len(quantile_levels)



group_names = list(data.groups.keys())[::-1]  # Reverse order if you want!

n_groups = len(group_names)

n_rows = int(np.ceil(np.sqrt(n_groups)))

n_cols = int(np.ceil(n_groups / n_rows))



fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 10), sharex=True, sharey=True)

axes = axes.flatten()



vmin = np.inf

vmax = -np.inf

speed_matrices = []



# Compute quantile matrices and color scale

for group_name in group_names:

    animal_indices = data.groups[group_name]

    speed_matrix = np.full((n_q, n_windows), np.nan)

    for win_idx in range(n_windows):

        win_arr = all_speed[win_idx]  # (n_animals, n_taus, n_timepoints)

        speeds_this_window = []

        for animal_idx in animal_indices:

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :]

                arr = np.asarray(arr, dtype=float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    speeds_this_window.append(arr)

        if speeds_this_window:

            flat_speeds = np.concatenate(speeds_this_window)

            if flat_speeds.size > 0:

                speed_matrix[:, win_idx] = [np.quantile(flat_speeds, q) for q in quantile_levels]

        # else: already nan

    valid = ~np.isnan(speed_matrix)

    if np.any(valid):

        vmin = min(vmin, np.nanmin(speed_matrix))

        vmax = max(vmax, np.nanmax(speed_matrix))

    speed_matrices.append(speed_matrix)



# Plot each group's quantile fan matrix

for idx, group_name in enumerate(group_names):

    ax = axes[idx]

    mat = speed_matrices[idx]

    im = ax.imshow(

        mat,

        aspect='auto',

        origin='lower',

        extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

        cmap='magma',

        vmin=vmin, vmax=vmax

    )

    label = f"{group_name[0]}-{group_name[1]}".lower()

    ax.set_title(label)

    ax.set_xlabel('Window Size')

    ax.set_ylabel('Quantile')

    ax.label_outer()



# Place colorbar on the left

fig.subplots_adjust(left=0.15, right=0.95)

cbar_ax = fig.add_axes([0.05, 0.25, 0.02, 0.5])

fig.colorbar(im, cax=cbar_ax, orientation='vertical', label='dFC Speed')



plt.tight_layout(rect=[0.15, 0, 1, 1])

plt.show()



# %%
import numpy as np

import matplotlib.pyplot as plt



quantile_levels = np.linspace(0, 1, 20)

n_windows = len(window_sizes)

n_q = len(quantile_levels)



group_names = list(data.groups.keys())[::-1]

n_groups = len(group_names)

n_rows = int(np.ceil(np.sqrt(n_groups)))

n_cols = int(np.ceil(n_groups / n_rows))



fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 10), sharex=True, sharey=True)

axes = axes.flatten()



vmin = np.inf

vmax = -np.inf

speed_matrices = []



for group_name in group_names:

    animal_indices = data.groups[group_name]

    speed_matrix = np.full((n_q, n_windows), np.nan)

    for win_idx in range(n_windows):

        win_arr = all_speed[win_idx]

        speeds_this_window = []

        for animal_idx in animal_indices:

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :]

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    speeds_this_window.append(arr)

        if speeds_this_window:

            flat_speeds = np.concatenate(speeds_this_window)

            if flat_speeds.size > 0:

                speed_matrix[:, win_idx] = [np.quantile(flat_speeds, q) for q in quantile_levels]

    valid = ~np.isnan(speed_matrix)

    if np.any(valid):

        vmin = min(vmin, np.nanmin(speed_matrix))

        vmax = max(vmax, np.nanmax(speed_matrix))

    speed_matrices.append(speed_matrix)



# --- Overlay IQR and median curves ---

q25_idx = np.argmin(np.abs(quantile_levels - 0.25))

q50_idx = np.argmin(np.abs(quantile_levels - 0.5))

q75_idx = np.argmin(np.abs(quantile_levels - 0.75))



for idx, group_name in enumerate(group_names):

    ax = axes[idx]

    mat = speed_matrices[idx]

    im = ax.imshow(

        mat,

        aspect='auto',

        origin='lower',

        extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

        cmap='magma',

        vmin=vmin, vmax=vmax

    )

    label = f"{group_name[0]}-{group_name[1]}".lower()

    ax.set_title(label)

    ax.set_xlabel('Window Size')

    ax.set_ylabel('Quantile')

    ax.label_outer()

    # Overlay median and IQR

    ax.plot(window_sizes, mat[q50_idx, :], color='w', lw=2.2, label='Median')

    ax.plot(window_sizes, mat[q25_idx, :], color='w', lw=1.3, ls='--', label='IQR')

    ax.plot(window_sizes, mat[q75_idx, :], color='w', lw=1.3, ls='--')



# Place colorbar on the left

fig.subplots_adjust(left=0.15, right=0.95)

cbar_ax = fig.add_axes([0.05, 0.25, 0.02, 0.5])

fig.colorbar(im, cax=cbar_ax, orientation='vertical', label='dFC Speed')



# Add legend to the first panel (remove duplicate labels)

handles, labels = axes[0].get_legend_handles_labels()

axes[0].legend(handles[:2], ['Median', 'IQR'], loc='upper right', frameon=True)



plt.tight_layout(rect=[0.15, 0, 1, 1])

plt.show()



# %%
import numpy as np

import matplotlib.pyplot as plt



quantile_levels = np.linspace(0, 1, 20)

n_windows = len(window_sizes)

n_q = len(quantile_levels)



group_names = list(data.groups.keys())[::-1]

n_groups = len(group_names)

n_rows = int(np.ceil(np.sqrt(n_groups)))

n_cols = int(np.ceil(n_groups / n_rows))



fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 10), sharex=True, sharey=True)

axes = axes.flatten()



vmin = np.inf

vmax = -np.inf

speed_matrices = []



for group_name in group_names:

    animal_indices = data.groups[group_name]

    speed_matrix = np.full((n_q, n_windows), np.nan)

    for win_idx in range(n_windows):

        win_arr = all_speed[win_idx]

        speeds_this_window = []

        for animal_idx in animal_indices:

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :].astype(float)





                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    speeds_this_window.append(arr)

        if speeds_this_window:

            flat_speeds = np.concatenate(speeds_this_window)

            if flat_speeds.size > 0:

                speed_matrix[:, win_idx] = [np.quantile(flat_speeds, q) for q in quantile_levels]

    valid = ~np.isnan(speed_matrix)

    if np.any(valid):

        vmin = min(vmin, np.nanmin(speed_matrix))

        vmax = max(vmax, np.nanmax(speed_matrix))

    speed_matrices.append(speed_matrix)



# --- Overlay IQR and median curves ---

q25_idx = np.argmin(np.abs(quantile_levels - 0.25))

q50_idx = np.argmin(np.abs(quantile_levels - 0.5))

q75_idx = np.argmin(np.abs(quantile_levels - 0.75))



for idx, group_name in enumerate(group_names):

    ax = axes[idx]

    mat = speed_matrices[idx]

    im = ax.imshow(

        mat,

        aspect='auto',

        origin='lower',

        extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

        cmap='magma',

        vmin=vmin, vmax=vmax

    )

    label = f"{group_name[0]}-{group_name[1]}".lower()

    ax.set_title(label)

    ax.set_xlabel('Window Size')

    ax.set_ylabel('Quantile')

    ax.label_outer()

    # Overlay median and IQR

    ax.plot(window_sizes, mat[q50_idx, :], color='w', lw=2.2, label='Median')

    ax.plot(window_sizes, mat[q25_idx, :], color='w', lw=1.3, ls='--', label='IQR')

    ax.plot(window_sizes, mat[q75_idx, :], color='w', lw=1.3, ls='--')



# Place colorbar on the left

fig.subplots_adjust(left=0.15, right=0.95)

cbar_ax = fig.add_axes([0.05, 0.25, 0.02, 0.5])

fig.colorbar(im, cax=cbar_ax, orientation='vertical', label='dFC Speed')



# Add legend to the first panel (remove duplicate labels)

handles, labels = axes[0].get_legend_handles_labels()

axes[0].legend(handles[:2], ['Median', 'IQR'], loc='upper right', frameon=True)



plt.tight_layout(rect=[0.15, 0, 1, 1])

plt.show()



# %%
import numpy as np

import matplotlib.pyplot as plt



# Calculate pairwise differences (assumes 4 groups: A, B, C, D)

diff_AB = speed_matrices[0] - speed_matrices[1]

diff_AC = speed_matrices[0] - speed_matrices[2]

diff_AD = speed_matrices[0] - speed_matrices[3]

diff_BC = speed_matrices[1] - speed_matrices[2]

diff_BD = speed_matrices[1] - speed_matrices[3]

diff_CD = speed_matrices[2] - speed_matrices[3]



diff_vmax = np.nanmax(np.abs([diff_AB, diff_AC, diff_AD, diff_BC, diff_BD, diff_CD]))

diff_cmap = 'bwr'



fig = plt.figure(figsize=(20, 18))

gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1])



# Shared color limits for all original matrices

vmin = min(np.nanmin(m) for m in speed_matrices)

vmax = max(np.nanmax(m) for m in speed_matrices)



# Row 1: A, B, A-B

ax1 = fig.add_subplot(gs[0, 0])

im1 = ax1.imshow(speed_matrices[0], aspect='auto', origin='lower',

                 extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

                 cmap='magma', vmin=vmin, vmax=vmax)

ax1.set_title(f'{group_names[0][0]}-{group_names[0][1]}'.lower())

ax1.set_ylabel('Quantile')



ax2 = fig.add_subplot(gs[0, 1])

im2 = ax2.imshow(speed_matrices[1], aspect='auto', origin='lower',

                 extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

                 cmap='magma', vmin=vmin, vmax=vmax)

ax2.set_title(f'{group_names[1][0]}-{group_names[1][1]}'.lower())



ax3 = fig.add_subplot(gs[0, 2])

im3 = ax3.imshow(diff_AB, aspect='auto', origin='lower',

                 extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

                 cmap=diff_cmap, vmin=-diff_vmax, vmax=diff_vmax)

ax3.set_title(f'Diff: {group_names[0][0]}-{group_names[0][1]} - {group_names[1][0]}-{group_names[1][1]}')

ax3.set_ylabel('Quantile')



# Row 2: C, D, C-D

ax4 = fig.add_subplot(gs[1, 0])

im4 = ax4.imshow(speed_matrices[2], aspect='auto', origin='lower',

                 extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

                 cmap='magma', vmin=vmin, vmax=vmax)

ax4.set_title(f'{group_names[2][0]}-{group_names[2][1]}'.lower())

ax4.set_ylabel('Quantile')



ax5 = fig.add_subplot(gs[1, 1])

im5 = ax5.imshow(speed_matrices[3], aspect='auto', origin='lower',

                 extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

                 cmap='magma', vmin=vmin, vmax=vmax)

ax5.set_title(f'{group_names[3][0]}-{group_names[3][1]}'.lower())



ax6 = fig.add_subplot(gs[1, 2])

im6 = ax6.imshow(diff_CD, aspect='auto', origin='lower',

                 extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

                 cmap=diff_cmap, vmin=-diff_vmax, vmax=diff_vmax)

ax6.set_title(f'Diff: {group_names[2][0]}-{group_names[2][1]} - {group_names[3][0]}-{group_names[3][1]}')



# Row 3: A-C, B-D, A-D

ax7 = fig.add_subplot(gs[2, 0])

im7 = ax7.imshow(diff_AC, aspect='auto', origin='lower',

                 extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

                 cmap=diff_cmap, vmin=-diff_vmax, vmax=diff_vmax)

ax7.set_title(f'Diff: {group_names[0][0]}-{group_names[0][1]} - {group_names[2][0]}-{group_names[2][1]}')

ax7.set_xlabel('Window Size')

ax7.set_ylabel('Quantile')



ax8 = fig.add_subplot(gs[2, 1])

im8 = ax8.imshow(diff_BD, aspect='auto', origin='lower',

                 extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

                 cmap=diff_cmap, vmin=-diff_vmax, vmax=diff_vmax)

ax8.set_title(f'Diff: {group_names[1][0]}-{group_names[1][1]} - {group_names[3][0]}-{group_names[3][1]}')

ax8.set_xlabel('Window Size')



ax9 = fig.add_subplot(gs[2, 2])

im9 = ax9.imshow(diff_AD, aspect='auto', origin='lower',

                 extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

                 cmap=diff_cmap, vmin=-diff_vmax, vmax=diff_vmax)

ax9.set_title(f'Diff: {group_names[0][0]}-{group_names[0][1]} - {group_names[3][0]}-{group_names[3][1]}')

ax9.set_xlabel('Window Size')



# Shared colorbars

fig.subplots_adjust(left=0.07, right=0.91, wspace=0.27, hspace=0.23)

cbar_ax1 = fig.add_axes([0.93, 0.65, 0.015, 0.27])

fig.colorbar(im1, cax=cbar_ax1, orientation='vertical', label='dFC Speed')



cbar_ax2 = fig.add_axes([0.93, 0.12, 0.015, 0.35])

fig.colorbar(im3, cax=cbar_ax2, orientation='vertical', label='dFC Speed Diff')



plt.show()



# %%
#%%



import numpy as np

import matplotlib.pyplot as plt



quantile_levels = np.linspace(0, 1, 100)

n_windows = len(window_sizes)

n_q = len(quantile_levels)



group_names = list(data.groups.keys())[::-1]

n_groups = len(group_names)

n_rows = int(np.ceil(np.sqrt(n_groups)))

n_cols = int(np.ceil(n_groups / n_rows))



fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 10), sharex=True, sharey=True)

axes = axes.flatten()



vmin = np.inf

vmax = -np.inf

speed_matrices = []



for group_name in group_names:

    animal_indices = data.groups[group_name]

    speed_matrix = np.full((n_q, n_windows), np.nan)

    for win_idx in range(n_windows):

        win_arr = all_speed[win_idx]

        speeds_this_window = []

        for animal_idx in animal_indices:

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :].astype(float)





                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    speeds_this_window.append(arr)

        if speeds_this_window:

            flat_speeds = np.concatenate(speeds_this_window)

            if flat_speeds.size > 0:

                speed_matrix[:, win_idx] = [np.quantile(flat_speeds, q) for q in quantile_levels]

    valid = ~np.isnan(speed_matrix)

    if np.any(valid):

        vmin = min(vmin, np.nanmin(speed_matrix))

        vmax = max(vmax, np.nanmax(speed_matrix))

    speed_matrices.append(speed_matrix)



# --- Overlay IQR and median curves ---

q25_idx = np.argmin(np.abs(quantile_levels - 0.25))

q50_idx = np.argmin(np.abs(quantile_levels - 0.5))

q75_idx = np.argmin(np.abs(quantile_levels - 0.75))



for idx, group_name in enumerate(group_names):

    ax = axes[idx]

    mat = speed_matrices[idx]

    im = ax.imshow(

        mat,

        aspect='auto',

        origin='lower',

        extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

        cmap='magma',

        vmin=vmin, vmax=vmax

    )

    label = f"{group_name[0]}-{group_name[1]}".lower()

    ax.set_title(label)

    ax.set_xlabel('Window Size')

    ax.set_ylabel('Quantile')

    ax.label_outer()

    # Overlay median and IQR

    ax.plot(window_sizes, mat[q50_idx, :], color='w', lw=2.2, label='Median')

    ax.plot(window_sizes, mat[q25_idx, :], color='w', lw=1.3, ls='--', label='IQR')

    ax.plot(window_sizes, mat[q75_idx, :], color='w', lw=1.3, ls='--')



# Place colorbar on the left

fig.subplots_adjust(left=0.15, right=0.95)

cbar_ax = fig.add_axes([0.05, 0.25, 0.02, 0.5])

fig.colorbar(im, cax=cbar_ax, orientation='vertical', label='dFC Speed')



# Add legend to the first panel (remove duplicate labels)

handles, labels = axes[0].get_legend_handles_labels()

axes[0].legend(handles[:2], ['Median', 'IQR'], loc='upper right', frameon=True)



plt.tight_layout(rect=[0.15, 0, 1, 1])

plt.show()
# %%
import numpy as np

import matplotlib.pyplot as plt





# Calculate pairwise differences (assumes 4 groups: A, B, C, D)

diff_AB = speed_matrices[0] - speed_matrices[1]

diff_AC = speed_matrices[0] - speed_matrices[2]

diff_AD = speed_matrices[0] - speed_matrices[3]

diff_BC = speed_matrices[1] - speed_matrices[2]

diff_BD = speed_matrices[1] - speed_matrices[3]

diff_CD = speed_matrices[2] - speed_matrices[3]



diff_vmax = np.nanmax(np.abs([diff_AB, diff_AC, diff_AD, diff_BC, diff_BD, diff_CD]))

diff_cmap = 'bwr'



fig = plt.figure(figsize=(20, 18))

gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1])



# Shared color limits for all original matrices

vmin = min(np.nanmin(m) for m in speed_matrices)

vmax = max(np.nanmax(m) for m in speed_matrices)



# Row 1: A, B, A-B

ax1 = fig.add_subplot(gs[0, 0])

im1 = ax1.imshow(speed_matrices[0], aspect='auto', origin='lower',

                 extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

                 cmap='magma', vmin=vmin, vmax=vmax)

ax1.set_title(f'{group_names[0][0]}-{group_names[0][1]}'.lower())

ax1.set_ylabel('Quantile')



ax2 = fig.add_subplot(gs[0, 1])

im2 = ax2.imshow(speed_matrices[1], aspect='auto', origin='lower',

                 extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

                 cmap='magma', vmin=vmin, vmax=vmax)

ax2.set_title(f'{group_names[1][0]}-{group_names[1][1]}'.lower())



ax3 = fig.add_subplot(gs[0, 2])

im3 = ax3.imshow(diff_AB, aspect='auto', origin='lower',

                 extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

                 cmap=diff_cmap, vmin=-diff_vmax, vmax=diff_vmax)

ax3.set_title(f'Diff: {group_names[0][0]}-{group_names[0][1]} - {group_names[1][0]}-{group_names[1][1]}')

ax3.set_ylabel('Quantile')



# Row 2: C, D, C-D

ax4 = fig.add_subplot(gs[1, 0])

im4 = ax4.imshow(speed_matrices[2], aspect='auto', origin='lower',

                 extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

                 cmap='magma', vmin=vmin, vmax=vmax)

ax4.set_title(f'{group_names[2][0]}-{group_names[2][1]}'.lower())

ax4.set_ylabel('Quantile')



ax5 = fig.add_subplot(gs[1, 1])

im5 = ax5.imshow(speed_matrices[3], aspect='auto', origin='lower',

                 extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

                 cmap='magma', vmin=vmin, vmax=vmax)

ax5.set_title(f'{group_names[3][0]}-{group_names[3][1]}'.lower())



ax6 = fig.add_subplot(gs[1, 2])

im6 = ax6.imshow(diff_CD, aspect='auto', origin='lower',

                 extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

                 cmap=diff_cmap, vmin=-diff_vmax, vmax=diff_vmax)

ax6.set_title(f'Diff: {group_names[2][0]}-{group_names[2][1]} - {group_names[3][0]}-{group_names[3][1]}')



# Row 3: A-C, B-D, A-D

ax7 = fig.add_subplot(gs[2, 0])

im7 = ax7.imshow(diff_AC, aspect='auto', origin='lower',

                 extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

                 cmap=diff_cmap, vmin=-diff_vmax, vmax=diff_vmax)

ax7.set_title(f'Diff: {group_names[0][0]}-{group_names[0][1]} - {group_names[2][0]}-{group_names[2][1]}')

ax7.set_xlabel('Window Size')

ax7.set_ylabel('Quantile')



ax8 = fig.add_subplot(gs[2, 1])

im8 = ax8.imshow(diff_BD, aspect='auto', origin='lower',

                 extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

                 cmap=diff_cmap, vmin=-diff_vmax, vmax=diff_vmax)

ax8.set_title(f'Diff: {group_names[1][0]}-{group_names[1][1]} - {group_names[3][0]}-{group_names[3][1]}')

ax8.set_xlabel('Window Size')



ax9 = fig.add_subplot(gs[2, 2])

im9 = ax9.imshow(diff_AD, aspect='auto', origin='lower',

                 extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

                 cmap=diff_cmap, vmin=-diff_vmax, vmax=diff_vmax)

ax9.set_title(f'Diff: {group_names[0][0]}-{group_names[0][1]} - {group_names[3][0]}-{group_names[3][1]}')

ax9.set_xlabel('Window Size')



# Shared colorbars

fig.subplots_adjust(left=0.07, right=0.91, wspace=0.27, hspace=0.23)

cbar_ax1 = fig.add_axes([0.93, 0.65, 0.015, 0.27])

fig.colorbar(im1, cax=cbar_ax1, orientation='vertical', label='dFC Speed')



cbar_ax2 = fig.add_axes([0.93, 0.12, 0.015, 0.35])

fig.colorbar(im3, cax=cbar_ax2, orientation='vertical', label='dFC Speed Diff')



plt.show()



# %%
import numpy as np

import matplotlib.pyplot as plt

import itertools

import math



# Assume speed_matrices, group_names, window_sizes, quantile_levels are defined

N = len(speed_matrices)

diff_pairs = list(itertools.combinations(range(N), 2))  # All unique pairs

n_diffs = len(diff_pairs)



ncols = N  # One column per group

nrows = 1 + math.ceil(n_diffs / ncols)  # 1 row for originals, rest for differences



fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), sharex=True, sharey=True)

axes = axes.flatten()



# Color scaling for original and difference matrices

vmin = min(np.nanmin(m) for m in speed_matrices)

vmax = max(np.nanmax(m) for m in speed_matrices)

diff_matrices = []

for i, j in diff_pairs:

    diff_matrices.append(speed_matrices[i] - speed_matrices[j])

diff_vmax = np.nanmax(np.abs(diff_matrices))



# Row 1: original groups

for idx, mat in enumerate(speed_matrices):

    ax = axes[idx]

    im = ax.imshow(mat, aspect='auto', origin='lower',

                   extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

                   cmap='magma', vmin=vmin, vmax=vmax)

    label = f"{group_names[idx][0]}-{group_names[idx][1]}".lower()

    ax.set_title(label)

    ax.set_ylabel('Quantile')

    ax.set_xlabel('Window Size')

    ax.label_outer()



# Next rows: all pairwise differences

for d_idx, (i, j) in enumerate(diff_pairs):

    ax_idx = N + d_idx

    ax = axes[ax_idx]

    im_diff = ax.imshow(

        speed_matrices[i] - speed_matrices[j],

        aspect='auto', origin='lower',

        extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

        cmap='bwr', vmin=-diff_vmax, vmax=diff_vmax

    )

    label = f"Diff: {group_names[i][0]}-{group_names[i][1]} - {group_names[j][0]}-{group_names[j][1]}"

    ax.set_title(label.lower())

    ax.set_ylabel('Quantile')

    ax.set_xlabel('Window Size')

    ax.label_outer()



# Hide unused axes

for ax in axes[N + n_diffs:]:

    ax.axis('off')



# Colorbars

fig.subplots_adjust(right=0.92, hspace=0.38, wspace=0.18)

cbar_ax1 = fig.add_axes([0.93, 0.77, 0.015, 0.17])

fig.colorbar(im, cax=cbar_ax1, orientation='vertical', label='dFC Speed')

cbar_ax2 = fig.add_axes([0.93, 0.15, 0.015, 0.57])

fig.colorbar(im_diff, cax=cbar_ax2, orientation='vertical', label='dFC Speed Diff')



plt.tight_layout(rect=[0, 0, 0.92, 1])

plt.show()



# %%
# %%



import numpy as np

import matplotlib.pyplot as plt





# Calculate pairwise differences (assumes 4 groups: A, B, C, D)

diff_AB = speed_matrices[0] - speed_matrices[1]

diff_AC = speed_matrices[0] - speed_matrices[2]

diff_AD = speed_matrices[0] - speed_matrices[3]

diff_BC = speed_matrices[1] - speed_matrices[2]

diff_BD = speed_matrices[1] - speed_matrices[3]

diff_CD = speed_matrices[2] - speed_matrices[3]



diff_vmax = np.nanmax(np.abs([diff_AB, diff_AC, diff_AD, diff_BC, diff_BD, diff_CD]))

diff_cmap = 'bwr'



fig = plt.figure(figsize=(20, 18))

gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1])



# Shared color limits for all original matrices

vmin = min(np.nanmin(m) for m in speed_matrices)

vmax = max(np.nanmax(m) for m in speed_matrices)



# Row 1: A, B, A-B

ax1 = fig.add_subplot(gs[0, 0])

im1 = ax1.imshow(speed_matrices[0], aspect='auto', origin='lower',

                 extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

                 cmap='magma', vmin=vmin, vmax=vmax)

ax1.set_title(f'{group_names[0][0]}-{group_names[0][1]}'.lower())

ax1.set_ylabel('Quantile')



ax2 = fig.add_subplot(gs[0, 1])

im2 = ax2.imshow(speed_matrices[1], aspect='auto', origin='lower',

                 extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

                 cmap='magma', vmin=vmin, vmax=vmax)

ax2.set_title(f'{group_names[1][0]}-{group_names[1][1]}'.lower())



ax3 = fig.add_subplot(gs[0, 2])

im3 = ax3.imshow(diff_AB, aspect='auto', origin='lower',

                 extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

                 cmap=diff_cmap, vmin=-diff_vmax, vmax=diff_vmax)

ax3.set_title(f'Diff: {group_names[0][0]}-{group_names[0][1]} - {group_names[1][0]}-{group_names[1][1]}')

ax3.set_ylabel('Quantile')



# Row 2: C, D, C-D

ax4 = fig.add_subplot(gs[1, 0])

im4 = ax4.imshow(speed_matrices[2], aspect='auto', origin='lower',

                 extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

                 cmap='magma', vmin=vmin, vmax=vmax)

ax4.set_title(f'{group_names[2][0]}-{group_names[2][1]}'.lower())

ax4.set_ylabel('Quantile')



ax5 = fig.add_subplot(gs[1, 1])

im5 = ax5.imshow(speed_matrices[3], aspect='auto', origin='lower',

                 extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

                 cmap='magma', vmin=vmin, vmax=vmax)

ax5.set_title(f'{group_names[3][0]}-{group_names[3][1]}'.lower())



ax6 = fig.add_subplot(gs[1, 2])

im6 = ax6.imshow(diff_CD, aspect='auto', origin='lower',

                 extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

                 cmap=diff_cmap, vmin=-diff_vmax, vmax=diff_vmax)

ax6.set_title(f'Diff: {group_names[2][0]}-{group_names[2][1]} - {group_names[3][0]}-{group_names[3][1]}')



# Row 3: A-C, B-D, A-D

ax7 = fig.add_subplot(gs[2, 0])

im7 = ax7.imshow(diff_AC, aspect='auto', origin='lower',

                 extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

                 cmap=diff_cmap, vmin=-diff_vmax, vmax=diff_vmax)

ax7.set_title(f'Diff: {group_names[0][0]}-{group_names[0][1]} - {group_names[2][0]}-{group_names[2][1]}')

ax7.set_xlabel('Window Size')

ax7.set_ylabel('Quantile')



ax8 = fig.add_subplot(gs[2, 1])

im8 = ax8.imshow(diff_BD, aspect='auto', origin='lower',

                 extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

                 cmap=diff_cmap, vmin=-diff_vmax, vmax=diff_vmax)

ax8.set_title(f'Diff: {group_names[1][0]}-{group_names[1][1]} - {group_names[3][0]}-{group_names[3][1]}')

ax8.set_xlabel('Window Size')



ax9 = fig.add_subplot(gs[2, 2])

im9 = ax9.imshow(diff_AD, aspect='auto', origin='lower',

                 extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

                 cmap=diff_cmap, vmin=-diff_vmax, vmax=diff_vmax)

ax9.set_title(f'Diff: {group_names[0][0]}-{group_names[0][1]} - {group_names[3][0]}-{group_names[3][1]}')

ax9.set_xlabel('Window Size')



# Shared colorbars

fig.subplots_adjust(left=0.07, right=0.91, wspace=0.27, hspace=0.23)

cbar_ax1 = fig.add_axes([0.93, 0.65, 0.015, 0.27])

fig.colorbar(im1, cax=cbar_ax1, orientation='vertical', label='dFC Speed')



cbar_ax2 = fig.add_axes([0.93, 0.12, 0.015, 0.35])

fig.colorbar(im3, cax=cbar_ax2, orientation='vertical', label='dFC Speed Diff')



plt.show()
# %%
# %%



import numpy as np

import matplotlib.pyplot as plt

import itertools

import math



# Assume speed_matrices, group_names, window_sizes, quantile_levels are defined

N = len(speed_matrices)

diff_pairs = list(itertools.combinations(range(N), 2))  # All unique pairs

n_diffs = len(diff_pairs)



ncols = N  # One column per group

nrows = 1 + math.ceil(n_diffs / ncols)  # 1 row for originals, rest for differences



fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), sharex=True, sharey=True)

axes = axes.flatten()



# Color scaling for original and difference matrices

vmin = min(np.nanmin(m) for m in speed_matrices)

vmax = max(np.nanmax(m) for m in speed_matrices)

diff_matrices = []

for i, j in diff_pairs:

    diff_matrices.append(speed_matrices[i] - speed_matrices[j])

diff_vmax = np.nanmax(np.abs(diff_matrices))



# Row 1: original groups

for idx, mat in enumerate(speed_matrices):

    ax = axes[idx]

    im = ax.imshow(mat, aspect='auto', origin='lower',

                   extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

                   cmap='magma', vmin=vmin, vmax=vmax)

    label = f"{group_names[idx][0]}-{group_names[idx][1]}".lower()

    ax.set_title(label)

    ax.set_ylabel('Quantile')

    ax.set_xlabel('Window Size')

    ax.label_outer()



# Next rows: all pairwise differences

for d_idx, (i, j) in enumerate(diff_pairs):

    ax_idx = N + d_idx

    ax = axes[ax_idx]

    im_diff = ax.imshow(

        speed_matrices[i] - speed_matrices[j],

        aspect='auto', origin='lower',

        extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

        cmap='bwr', vmin=-diff_vmax, vmax=diff_vmax

    )

    label = f"Diff: {group_names[i][0]}-{group_names[i][1]} - {group_names[j][0]}-{group_names[j][1]}"

    ax.set_title(label.lower())

    ax.set_ylabel('Quantile')

    ax.set_xlabel('Window Size')

    ax.label_outer()



# Hide unused axes

for ax in axes[N + n_diffs:]:

    ax.axis('off')



# Colorbars

fig.subplots_adjust(right=0.92, hspace=0.38, wspace=0.18)

cbar_ax1 = fig.add_axes([0.93, 0.77, 0.015, 0.17])

fig.colorbar(im, cax=cbar_ax1, orientation='vertical', label='dFC Speed')

cbar_ax2 = fig.add_axes([0.93, 0.15, 0.015, 0.57])

fig.colorbar(im_diff, cax=cbar_ax2, orientation='vertical', label='dFC Speed Diff')



plt.tight_layout(rect=[0, 0, 0.92, 1])

plt.show()
# %%
data.cog_data_filtered
# %%
data.cog_data_filtered['index_NOR']
# %%
#%%



#Cognitive scores

NOR_score = data.cog_data_filtered['index_NOR'] 
# %%
n_animals
# %%
#%%



import pandas as pd

from scipy.stats import spearmanr



# ------------------------ NOR scores vs dFC speed ------------------------

# Load cognitive data

NOR_score = data.cog_data_filtered['index_NOR'] 



# 1. Compute per-animal dFC speed median

# Assume all_speed: list of window arrays (n_animals, n_taus, n_timepoints)

per_animal_speeds = []

for animal_idx in range(n_animals):

    pooled = []

    for win_arr in all_speed:

        for tau in range(win_arr.shape[1]):

            arr = win_arr[animal_idx, tau, :].astype(float)

            arr = arr[~np.isnan(arr)]

            pooled.append(arr)

    if pooled:

        flat = np.concatenate(pooled)

        per_animal_speeds.append(np.median(flat))

    else:

        per_animal_speeds.append(np.nan)



per_animal_speeds = np.array(per_animal_speeds)



# 2. Extract cognitive scores, aligned to animals in the same order!

cog_scores = cog_data_filtered.loc[:n_animals-1, 'index_NOR'].values  # adjust column as needed



# 3. Scatter plot

plt.figure(figsize=(7,5))

plt.scatter(per_animal_speeds, cog_scores, c='k', alpha=0.8)

plt.xlabel('Median dFC Speed per animal')

plt.ylabel('Cognitive score (e.g., NOR index)')

plt.title('Relationship between dFC speed and cognitive score')



# 4. Correlation

mask = ~np.isnan(per_animal_speeds) & ~np.isnan(cog_scores)

rho, pval = spearmanr(per_animal_speeds[mask], cog_scores[mask])

plt.text(0.05, 0.95, f"Spearman r={rho:.2f}, p={pval:.3g}",

         transform=plt.gca().transAxes, va='top', ha='left', fontsize=12)



plt.tight_layout()

plt.show()
# %%
#%%



import pandas as pd

from scipy.stats import spearmanr



# ------------------------ NOR scores vs dFC speed ------------------------

# Load cognitive data

NOR_score = data.cog_data_filtered['index_NOR'] 



# 1. Compute per-animal dFC speed median

# Assume all_speed: list of window arrays (n_animals, n_taus, n_timepoints)

per_animal_speeds = []

for animal_idx in range(n_animals):

    pooled = []

    for win_arr in all_speed:

        for tau in range(win_arr.shape[1]):

            arr = win_arr[animal_idx, tau, :].astype(float)

            arr = arr[~np.isnan(arr)]

            pooled.append(arr)

    if pooled:

        flat = np.concatenate(pooled)

        per_animal_speeds.append(np.median(flat))

    else:

        per_animal_speeds.append(np.nan)



per_animal_speeds = np.array(per_animal_speeds)



# 2. Extract cognitive scores, aligned to animals in the same order!

cog_scores = data.cog_data_filtered.loc[:n_animals-1, 'index_NOR'].values  # adjust column as needed



# 3. Scatter plot

plt.figure(figsize=(7,5))

plt.scatter(per_animal_speeds, cog_scores, c='k', alpha=0.8)

plt.xlabel('Median dFC Speed per animal')

plt.ylabel('Cognitive score (e.g., NOR index)')

plt.title('Relationship between dFC speed and cognitive score')



# 4. Correlation

mask = ~np.isnan(per_animal_speeds) & ~np.isnan(cog_scores)

rho, pval = spearmanr(per_animal_speeds[mask], cog_scores[mask])

plt.text(0.05, 0.95, f"Spearman r={rho:.2f}, p={pval:.3g}",

         transform=plt.gca().transAxes, va='top', ha='left', fontsize=12)



plt.tight_layout()

plt.show()
# %%
#%%



import pandas as pd

from scipy.stats import spearmanr



# ------------------------ NOR scores vs dFC speed ------------------------

# Load cognitive data

cog_scores = data.cog_data_filtered['index_NOR'] 



# 1. Compute per-animal dFC speed median

# Assume all_speed: list of window arrays (n_animals, n_taus, n_timepoints)

per_animal_speeds = []

for animal_idx in range(n_animals):

    pooled = []

    for win_arr in all_speed:

        for tau in range(win_arr.shape[1]):

            arr = win_arr[animal_idx, tau, :].astype(float)

            arr = arr[~np.isnan(arr)]

            pooled.append(arr)

    if pooled:

        flat = np.concatenate(pooled)

        per_animal_speeds.append(np.median(flat))

    else:

        per_animal_speeds.append(np.nan)



per_animal_speeds = np.array(per_animal_speeds)



# 2. Extract cognitive scores, aligned to animals in the same order!

# cog_scores = data.cog_data_filtered.loc[:n_animals-1, 'index_NOR'].values  # adjust column as needed



# 3. Scatter plot

plt.figure(figsize=(7,5))

plt.scatter(per_animal_speeds, cog_scores, c='k', alpha=0.8)

plt.xlabel('Median dFC Speed per animal')

plt.ylabel('Cognitive score (e.g., NOR index)')

plt.title('Relationship between dFC speed and cognitive score')



# 4. Correlation

mask = ~np.isnan(per_animal_speeds) & ~np.isnan(cog_scores)

rho, pval = spearmanr(per_animal_speeds[mask], cog_scores[mask])

plt.text(0.05, 0.95, f"Spearman r={rho:.2f}, p={pval:.3g}",

         transform=plt.gca().transAxes, va='top', ha='left', fontsize=12)



plt.tight_layout()

plt.show()
# %%
#%%



import pandas as pd

from scipy.stats import spearmanr



# ------------------------ NOR scores vs dFC speed ------------------------

# Load cognitive data

cog_scores = data.cog_data_filtered['index_NOR'] 



# 1. Compute per-animal dFC speed median

# Assume all_speed: list of window arrays (n_animals, n_taus, n_timepoints)

per_animal_speeds = []

for animal_idx in range(n_animals):

    pooled = []

    for win_arr in all_speed:

        for tau in range(win_arr.shape[1]):

            arr = win_arr[animal_idx, tau, :].astype(float)

            arr = arr[~np.isnan(arr)]

            pooled.append(arr)

    if pooled:

        flat = np.concatenate(pooled)

        per_animal_speeds.append(np.median(flat))

    else:

        per_animal_speeds.append(np.nan)



per_animal_speeds = np.array(per_animal_speeds)



# 2. Extract cognitive scores, aligned to animals in the same order!

# cog_scores = data.cog_data_filtered.loc[:n_animals-1, 'index_NOR'].values  # adjust column as needed



# 3. Scatter plot

plt.figure(figsize=(7,5))

plt.scatter(per_animal_speeds, cog_scores, c='k', alpha=0.8)

plt.xlabel('Median dFC Speed per animal')

plt.ylabel('Cognitive score (e.g., NOR index)')

plt.title('Relationship between dFC speed and cognitive score')



# 4. Correlation

mask = ~np.isnan(per_animal_speeds) & ~np.isnan(cog_scores)

rho, pval = spearmanr(per_animal_speeds[mask], cog_scores[mask])

plt.text(0.05, 0.95, f"Spearman r={rho:.2f}, p={pval:.3g}",

         transform=plt.gca().transAxes, va='top', ha='left', fontsize=12)



plt.tight_layout()

plt.show()
# %%
#%%



import pandas as pd

from scipy.stats import spearmanr



# ------------------------ NOR scores vs dFC speed ------------------------

# Load cognitive data

cog_scores = data.cog_data_filtered['index_NOR'].values



# 1. Compute per-animal dFC speed median

# Assume all_speed: list of window arrays (n_animals, n_taus, n_timepoints)

per_animal_speeds = []

for animal_idx in range(n_animals):

    pooled = []

    for win_arr in all_speed:

        for tau in range(win_arr.shape[1]):

            arr = win_arr[animal_idx, tau, :].astype(float)

            arr = arr[~np.isnan(arr)]

            pooled.append(arr)

    if pooled:

        flat = np.concatenate(pooled)

        per_animal_speeds.append(np.median(flat))

    else:

        per_animal_speeds.append(np.nan)



per_animal_speeds = np.array(per_animal_speeds)



# 2. Extract cognitive scores, aligned to animals in the same order!

# cog_scores = data.cog_data_filtered.loc[:n_animals-1, 'index_NOR'].values  # adjust column as needed



# 3. Scatter plot

plt.figure(figsize=(7,5))

plt.scatter(per_animal_speeds, cog_scores, c='k', alpha=0.8)

plt.xlabel('Median dFC Speed per animal')

plt.ylabel('Cognitive score (e.g., NOR index)')

plt.title('Relationship between dFC speed and cognitive score')



# 4. Correlation

mask = ~np.isnan(per_animal_speeds) & ~np.isnan(cog_scores)

rho, pval = spearmanr(per_animal_speeds[mask], cog_scores[mask])

plt.text(0.05, 0.95, f"Spearman r={rho:.2f}, p={pval:.3g}",

         transform=plt.gca().transAxes, va='top', ha='left', fontsize=12)



plt.tight_layout()

plt.show()
# %%
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from scipy.stats import spearmanr, linregress



# 1. Compute median dFC speed per animal (as before)

n_animals = all_speed[0].shape[0]

per_animal_speeds = []

for animal_idx in range(n_animals):

    pooled = []

    for win_arr in all_speed:

        for tau in range(win_arr.shape[1]):

            arr = win_arr[animal_idx, tau, :].astype(float)

            arr = arr[~np.isnan(arr)]

            pooled.append(arr)

    if pooled:

        flat = np.concatenate(pooled)

        per_animal_speeds.append(np.median(flat))

    else:

        per_animal_speeds.append(np.nan)

per_animal_speeds = np.array(per_animal_speeds)



# 2. Cognitive scores and group labels

cog_df = cog_data_filtered.reset_index(drop=True)

cog_scores = cog_df['index_NOR'].values  # adjust if you want a different score

group_labels = list(zip(cog_df['genotype'], cog_df['treatment']))



# 3. Assign a unique color/marker to each group

import seaborn as sns

groups = sorted(set(group_labels))

palette = sns.color_palette('tab10', n_colors=len(groups))

group2color = {g: palette[i] for i, g in enumerate(groups)}

markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '+', 'x']

group2marker = {g: markers[i % len(markers)] for i, g in enumerate(groups)}



plt.figure(figsize=(8, 6))

for i, group in enumerate(groups):

    idxs = [j for j, g in enumerate(group_labels) if g == group]

    speeds = per_animal_speeds[idxs]

    scores = cog_scores[idxs]

    plt.scatter(

        speeds, scores,

        color=group2color[group], marker=group2marker[group],

        label=f"{group[0]}-{group[1]}", s=70, alpha=0.8

    )

    # Robust regression (within group), skip if too few points

    mask = ~np.isnan(speeds) & ~np.isnan(scores)

    if np.sum(mask) > 2:

        # Linear fit for simplicity (use Theil-Sen or quantile regression if desired)

        slope, intercept, r, p, _ = linregress(speeds[mask], scores[mask])

        xfit = np.linspace(np.nanmin(speeds[mask]), np.nanmax(speeds[mask]), 100)

        yfit = slope * xfit + intercept

        plt.plot(

            xfit, yfit, color=group2color[group],

            linestyle='-', linewidth=2,

            alpha=0.75

        )

        # Optionally annotate correlation

        plt.text(

            0.98, 0.98-i*0.08,

            f"{group[0]}-{group[1]}: r={r:.2f}, p={p:.2g}",

            color=group2color[group],

            transform=plt.gca().transAxes, fontsize=10, ha='right', va='top'

        )



plt.xlabel('Median dFC Speed per animal')

plt.ylabel('Cognitive score (NOR index)')

plt.title('dFC speed vs. cognitive score, stratified by group')

plt.legend(title='Genotype-Treatment', fontsize=10, title_fontsize=12)

plt.tight_layout()

plt.show()



# %%
cog_data_filtered = data.cog_data_filtered
# %%
import numpy as np





import matplotlib.pyplot as plt

import pandas as pd

from scipy.stats import spearmanr, linregress



# 1. Compute median dFC speed per animal (as before)

n_animals = all_speed[0].shape[0]

per_animal_speeds = []

for animal_idx in range(n_animals):

    pooled = []

    for win_arr in all_speed:

        for tau in range(win_arr.shape[1]):

            arr = win_arr[animal_idx, tau, :].astype(float)

            arr = arr[~np.isnan(arr)]

            pooled.append(arr)

    if pooled:

        flat = np.concatenate(pooled)

        per_animal_speeds.append(np.median(flat))

    else:

        per_animal_speeds.append(np.nan)

per_animal_speeds = np.array(per_animal_speeds)



# 2. Cognitive scores and group labels

cog_df = cog_data_filtered.reset_index(drop=True)

cog_scores = cog_df['index_NOR'].values  # adjust if you want a different score

group_labels = list(zip(cog_df['genotype'], cog_df['treatment']))



# 3. Assign a unique color/marker to each group

import seaborn as sns

groups = sorted(set(group_labels))

palette = sns.color_palette('tab10', n_colors=len(groups))

group2color = {g: palette[i] for i, g in enumerate(groups)}

markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '+', 'x']

group2marker = {g: markers[i % len(markers)] for i, g in enumerate(groups)}



plt.figure(figsize=(8, 6))

for i, group in enumerate(groups):

    idxs = [j for j, g in enumerate(group_labels) if g == group]

    speeds = per_animal_speeds[idxs]

    scores = cog_scores[idxs]

    plt.scatter(

        speeds, scores,

        color=group2color[group], marker=group2marker[group],

        label=f"{group[0]}-{group[1]}", s=70, alpha=0.8

    )

    # Robust regression (within group), skip if too few points

    mask = ~np.isnan(speeds) & ~np.isnan(scores)

    if np.sum(mask) > 2:

        # Linear fit for simplicity (use Theil-Sen or quantile regression if desired)

        slope, intercept, r, p, _ = linregress(speeds[mask], scores[mask])

        xfit = np.linspace(np.nanmin(speeds[mask]), np.nanmax(speeds[mask]), 100)

        yfit = slope * xfit + intercept

        plt.plot(

            xfit, yfit, color=group2color[group],

            linestyle='-', linewidth=2,

            alpha=0.75

        )

        # Optionally annotate correlation

        plt.text(

            0.98, 0.98-i*0.08,

            f"{group[0]}-{group[1]}: r={r:.2f}, p={p:.2g}",

            color=group2color[group],

            transform=plt.gca().transAxes, fontsize=10, ha='right', va='top'

        )



plt.xlabel('Median dFC Speed per animal')

plt.ylabel('Cognitive score (NOR index)')

plt.title('dFC speed vs. cognitive score, stratified by group')

plt.legend(title='Genotype-Treatment', fontsize=10, title_fontsize=12)

plt.tight_layout()

plt.show()



# %%
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from scipy.stats import theilslopes, spearmanr

import seaborn as sns



# Median dFC speed per animal

n_animals = all_speed[0].shape[0]

per_animal_speeds = []

for animal_idx in range(n_animals):

    pooled = []

    for win_arr in all_speed:

        for tau in range(win_arr.shape[1]):

            arr = win_arr[animal_idx, tau, :].astype(float)

            arr = arr[~np.isnan(arr)]

            pooled.append(arr)

    if pooled:

        flat = np.concatenate(pooled)

        per_animal_speeds.append(np.median(flat))

    else:

        per_animal_speeds.append(np.nan)

per_animal_speeds = np.array(per_animal_speeds)



# Cognitive scores and group labels

cog_df = cog_data_filtered.reset_index(drop=True)

cog_scores = cog_df['index_NOR'].values

group_labels = list(zip(cog_df['genotype'], cog_df['treatment']))



# Assign color/marker per group

groups = sorted(set(group_labels))

palette = sns.color_palette('tab10', n_colors=len(groups))

group2color = {g: palette[i] for i, g in enumerate(groups)}

markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '+', 'x']

group2marker = {g: markers[i % len(markers)] for i, g in enumerate(groups)}



plt.figure(figsize=(8, 6))

for i, group in enumerate(groups):

    idxs = [j for j, g in enumerate(group_labels) if g == group]

    speeds = per_animal_speeds[idxs]

    scores = cog_scores[idxs]

    plt.scatter(

        speeds, scores,

        color=group2color[group], marker=group2marker[group],

        label=f"{group[0]}-{group[1]}", s=70, alpha=0.85

    )

    # Only fit if enough data

    mask = ~np.isnan(speeds) & ~np.isnan(scores)

    if np.sum(mask) > 2:

        # Theil-Sen regression (robust to outliers)

        ts_slope, ts_intercept, ts_low, ts_high = theilslopes(scores[mask], speeds[mask])

        xfit = np.linspace(np.nanmin(speeds[mask]), np.nanmax(speeds[mask]), 100)

        yfit = ts_slope * xfit + ts_intercept

        plt.plot(

            xfit, yfit, color=group2color[group],

            linestyle='-', linewidth=2,

            alpha=0.75

        )

        # Spearman correlation

        rho, pval = spearmanr(speeds[mask], scores[mask])

        plt.text(

            0.98, 0.98-i*0.09,

            f"{group[0]}-{group[1]}: ρ={rho:.2f}, p={pval:.2g}",

            color=group2color[group],

            transform=plt.gca().transAxes, fontsize=10, ha='right', va='top'

        )



plt.xlabel('Median dFC Speed per animal')

plt.ylabel('Cognitive score (NOR index)')

plt.title('dFC speed vs. cognitive score, stratified by group\n(Theil-Sen + Spearman)')

plt.legend(title='Genotype-Treatment', fontsize=10, title_fontsize=12)

plt.tight_layout()

plt.show()



# %%
#%%



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from scipy.stats import theilslopes, spearmanr

import seaborn as sns



# Median dFC speed per animal

n_animals = all_speed[0].shape[0]

per_animal_speeds = []

for animal_idx in range(n_animals):

    pooled = []

    for win_arr in all_speed:

        for tau in range(win_arr.shape[1]):

            arr = win_arr[animal_idx, tau, :].astype(float)

            arr = arr[~np.isnan(arr)]

            pooled.append(arr)

    if pooled:

        flat = np.concatenate(pooled)

        per_animal_speeds.append(np.median(flat))

    else:

        per_animal_speeds.append(np.nan)

per_animal_speeds = np.array(per_animal_speeds)



# Cognitive scores and group labels

cog_df = cog_data_filtered.reset_index(drop=True)

cog_scores = cog_df['index_NOR'].values

group_labels = list(zip(cog_df['genotype'], cog_df['treatment']))



# Assign color/marker per group

groups = sorted(set(group_labels))

palette = sns.color_palette('tab10', n_colors=len(groups))

group2color = {g: palette[i] for i, g in enumerate(groups)}

markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '+', 'x']

group2marker = {g: markers[i % len(markers)] for i, g in enumerate(groups)}



plt.figure(figsize=(8, 6))

for i, group in enumerate(groups):

    idxs = [j for j, g in enumerate(group_labels) if g == group]

    speeds = per_animal_speeds[idxs]

    scores = cog_scores[idxs]

    plt.scatter(

        speeds, scores,

        color=group2color[group], marker=group2marker[group],

        label=f"{group[0]}-{group[1]}", s=70, alpha=0.85

    )

    # Only fit if enough data

    mask = ~np.isnan(speeds) & ~np.isnan(scores)

    if np.sum(mask) > 2:

        # Theil-Sen regression (robust to outliers)

        ts_slope, ts_intercept, ts_low, ts_high = theilslopes(scores[mask], speeds[mask])

        xfit = np.linspace(np.nanmin(speeds[mask]), np.nanmax(speeds[mask]), 100)

        yfit = ts_slope * xfit + ts_intercept

        plt.plot(

            xfit, yfit, color=group2color[group],

            linestyle='-', linewidth=2,

            alpha=0.75

        )

        # Spearman correlation

        rho, pval = spearmanr(speeds[mask], scores[mask])

        plt.text(

            0.98, 0.98-i*0.09,

            f"{group[0]}-{group[1]}: ρ={rho:.2f}, p={pval:.2g}",

            color=group2color[group],

            transform=plt.gca().transAxes, fontsize=10, ha='right', va='top'

        )



plt.xlabel('Median dFC Speed per animal')

plt.ylabel('Cognitive score (NOR index)')

plt.title('dFC speed vs. cognitive score, stratified by group\n(Theil-Sen + Spearman)')

plt.legend(title='Genotype-Treatment', fontsize=10, title_fontsize=12)

plt.tight_layout()

plt.show()
# %%
# For IQR (interquartile range) per animal:

per_animal_IQR = []

for animal_idx in range(n_animals):

    pooled = []

    for win_arr in all_speed:

        for tau in range(win_arr.shape[1]):

            arr = win_arr[animal_idx, tau, :].astype(float)

            arr = arr[~np.isnan(arr)]

            pooled.append(arr)

    if pooled:

        flat = np.concatenate(pooled)

        q75 = np.percentile(flat, 75)

        q25 = np.percentile(flat, 25)

        per_animal_IQR.append(q75 - q25)

    else:

        per_animal_IQR.append(np.nan)

per_animal_IQR = np.array(per_animal_IQR)



# For a window-specific median (e.g., window 10):

per_animal_win10 = []

win10_idx = 10  # adjust as needed

win_arr = all_speed[win10_idx]

for animal_idx in range(n_animals):

    pooled = []

    for tau in range(win_arr.shape[1]):

        arr = win_arr[animal_idx, tau, :].astype(float)

        arr = arr[~np.isnan(arr)]

        pooled.append(arr)

    if pooled:

        flat = np.concatenate(pooled)

        per_animal_win10.append(np.median(flat))

    else:

        per_animal_win10.append(np.nan)

per_animal_win10 = np.array(per_animal_win10)



# Now use per_animal_IQR or per_animal_win10 in place of per_animal_speeds in previous plotting/statistics code.



# %%
import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import spearmanr



n_windows = len(all_speed)

n_animals = all_speed[0].shape[0]

cog_scores = cog_data_filtered.reset_index(drop=True)['index_NOR'].values



# Store correlation results

correlations = []

pvalues = []



for win_idx in range(n_windows):

    median_per_animal = []

    win_arr = all_speed[win_idx]  # shape: (n_animals, n_taus, n_timepoints)

    for animal_idx in range(n_animals):

        pooled = []

        for tau in range(win_arr.shape[1]):

            arr = win_arr[animal_idx, tau, :].astype(float)

            arr = arr[~np.isnan(arr)]

            pooled.append(arr)

        if pooled:

            flat = np.concatenate(pooled)

            median_per_animal.append(np.median(flat))

        else:

            median_per_animal.append(np.nan)

    median_per_animal = np.array(median_per_animal)

    mask = ~np.isnan(median_per_animal) & ~np.isnan(cog_scores)

    if np.sum(mask) > 2:

        rho, pval = spearmanr(median_per_animal[mask], cog_scores[mask])

    else:

        rho, pval = np.nan, np.nan

    correlations.append(rho)

    pvalues.append(pval)



window_sizes = time_window_range  # Or however you have it defined



plt.figure(figsize=(8,5))

plt.plot(window_sizes, correlations, '-o', color='navy', label="Spearman ρ")

plt.axhline(0, color='grey', linestyle='--', linewidth=1)

plt.xlabel('Window Size')

plt.ylabel("Spearman correlation (dFC speed, cognitive score)")

plt.title("Correlation vs. Window Size (per-animal, tau-pooled)")

plt.legend()

plt.tight_layout()

plt.show()



# Optional: plot p-values

plt.figure(figsize=(8,5))

plt.plot(window_sizes, pvalues, '-o', color='orangered')

plt.axhline(0.05, color='k', ls=':', lw=1, label='p=0.05')

plt.xlabel('Window Size')

plt.ylabel("p-value")

plt.title("Correlation p-value vs. Window Size")

plt.legend()

plt.tight_layout()

plt.show()



# %%
import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import spearmanr

import seaborn as sns



window_sizes = time_window_range

n_windows = len(window_sizes)

group_dict = data.groups  # {group_name: [animal_indices]}

cog_df = cog_data_filtered.reset_index(drop=True)



palette = sns.color_palette('tab10', n_colors=len(group_dict))

plt.figure(figsize=(9,6))



for idx, (group, animal_indices) in enumerate(group_dict.items()):

    correlations = []

    pvalues = []

    group_scores = cog_df.loc[animal_indices, 'index_NOR'].values



    for win_idx in range(n_windows):

        win_arr = all_speed[win_idx]  # shape: (n_animals, n_taus, n_timepoints)

        medians = []

        for animal_idx in animal_indices:

            pooled = []

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :].astype(float)

                arr = arr[~np.isnan(arr)]

                pooled.append(arr)

            if pooled:

                flat = np.concatenate(pooled)

                medians.append(np.median(flat))

            else:

                medians.append(np.nan)

        medians = np.array(medians)

        mask = ~np.isnan(medians) & ~np.isnan(group_scores)

        if np.sum(mask) > 2:

            rho, pval = spearmanr(medians[mask], group_scores[mask])

        else:

            rho, pval = np.nan, np.nan

        correlations.append(rho)

        pvalues.append(pval)



    label = f"{group[0]}-{group[1]}".lower()

    plt.plot(window_sizes, correlations, '-o', color=palette[idx], label=label)



plt.axhline(0, color='grey', linestyle='--', linewidth=1)

plt.xlabel('Window Size')

plt.ylabel("Spearman correlation (dFC speed, cognitive score)")

plt.title("Correlation vs. Window Size by Group\n(Per-animal, tau pooled)")

plt.legend(title='Group')

plt.tight_layout()

plt.show()



# %%
import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import spearmanr

import seaborn as sns



window_sizes = time_window_range

n_windows = len(window_sizes)

group_dict = data.groups

cog_df = cog_data_filtered.reset_index(drop=True)



palette = sns.color_palette('tab10', n_colors=len(group_dict))

plt.figure(figsize=(9,6))



for idx, (group, animal_indices) in enumerate(group_dict.items()):

    correlations = []

    pvalues = []

    group_scores = cog_df.loc[animal_indices, 'index_NOR'].values



    for win_idx in range(n_windows):

        win_arr = all_speed[win_idx]

        medians = []

        for animal_idx in animal_indices:

            pooled = []

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :].astype(float)

                arr = arr[~np.isnan(arr)]

                pooled.append(arr)

            if pooled:

                flat = np.concatenate(pooled)

                medians.append(np.median(flat))

            else:

                medians.append(np.nan)

        medians = np.array(medians)

        mask = ~np.isnan(medians) & ~np.isnan(group_scores)

        if np.sum(mask) > 2:

            rho, pval = spearmanr(medians[mask], group_scores[mask])

        else:

            rho, pval = np.nan, np.nan

        correlations.append(rho)

        pvalues.append(pval)



    label = f"{group[0]}-{group[1]}".lower()

    plt.plot(window_sizes, correlations, '-o', color=palette[idx], label=label, zorder=2)

    # Overlay significance marker for p < 0.05

    correlations = np.array(correlations)

    pvalues = np.array(pvalues)

    sig_idx = np.where((pvalues < 0.05) & ~np.isnan(pvalues) & ~np.isnan(correlations))[0]

    # Plot filled stars at significant points

    plt.scatter(np.array(window_sizes)[sig_idx], correlations[sig_idx],

                color=palette[idx], marker='*', s=110, edgecolor='k', linewidth=0.8, zorder=4, label=None)



plt.axhline(0, color='grey', linestyle='--', linewidth=1, zorder=1)

plt.xlabel('Window Size')

plt.ylabel("Spearman correlation (dFC speed, cognitive score)")

plt.title("Correlation vs. Window Size by Group\nStars = p < 0.05")

plt.legend(title='Group')

plt.tight_layout()

plt.show()



# %%
for idx, (group, animal_indices) in enumerate(group_dict.items()):

    # ... (existing code: correlations, pvalues, plot, scatter star markers)

    # ... after plotting significance stars



    for sig_win_idx in sig_idx:

        win = window_sizes[sig_win_idx]

        eff = correlations[sig_win_idx]

        plt.annotate(f"{eff:.2f}", 

                     (win, eff), 

                     textcoords="offset points", 

                     xytext=(0, 12),  # pixels above the star

                     ha='center', fontsize=10, color=palette[idx], weight='bold')



# %%
# %%



import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import spearmanr

import seaborn as sns



window_sizes = time_window_range

n_windows = len(window_sizes)

group_dict = data.groups

cog_df = cog_data_filtered.reset_index(drop=True)



palette = sns.color_palette('tab10', n_colors=len(group_dict))

plt.figure(figsize=(9,6))



for idx, (group, animal_indices) in enumerate(group_dict.items()):

    correlations = []

    pvalues = []

    group_scores = cog_df.loc[animal_indices, 'index_NOR'].values



    for win_idx in range(n_windows):

        win_arr = all_speed[win_idx]

        medians = []

        for animal_idx in animal_indices:

            pooled = []

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :].astype(float)

                arr = arr[~np.isnan(arr)]

                pooled.append(arr)

            if pooled:

                flat = np.concatenate(pooled)

                medians.append(np.median(flat))

            else:

                medians.append(np.nan)

        medians = np.array(medians)

        mask = ~np.isnan(medians) & ~np.isnan(group_scores)

        if np.sum(mask) > 2:

            rho, pval = spearmanr(medians[mask], group_scores[mask])

        else:

            rho, pval = np.nan, np.nan

        correlations.append(rho)

        pvalues.append(pval)



    label = f"{group[0]}-{group[1]}".lower()

    plt.plot(window_sizes, correlations, '-o', color=palette[idx], label=label, zorder=2)

    # Overlay significance marker for p < 0.05

    correlations = np.array(correlations)

    pvalues = np.array(pvalues)

    sig_idx = np.where((pvalues < 0.05) & ~np.isnan(pvalues) & ~np.isnan(correlations))[0]

    # Plot filled stars at significant points

    plt.scatter(np.array(window_sizes)[sig_idx], correlations[sig_idx],

                color=palette[idx], marker='*', s=110, edgecolor='k', linewidth=0.8, zorder=4, label=None)

    for idx, (group, animal_indices) in enumerate(group_dict.items()):

    # ... (existing code: correlations, pvalues, plot, scatter star markers)

    # ... after plotting significance stars



    for sig_win_idx in sig_idx:

        win = window_sizes[sig_win_idx]

        eff = correlations[sig_win_idx]

        plt.annotate(f"{eff:.2f}", 

                     (win, eff), 

                     textcoords="offset points", 

                     xytext=(0, 12),  # pixels above the star

                     ha='center', fontsize=10, color=palette[idx], weight='bold')





plt.axhline(0, color='grey', linestyle='--', linewidth=1, zorder=1)

plt.xlabel('Window Size')

plt.ylabel("Spearman correlation (dFC speed, cognitive score)")

plt.title("Correlation vs. Window Size by Group\nStars = p < 0.05")

plt.legend(title='Group')

plt.tight_layout()

plt.show()
# %%
# %%



import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import spearmanr

import seaborn as sns



window_sizes = time_window_range

n_windows = len(window_sizes)

group_dict = data.groups

cog_df = cog_data_filtered.reset_index(drop=True)



palette = sns.color_palette('tab10', n_colors=len(group_dict))

plt.figure(figsize=(9,6))



for idx, (group, animal_indices) in enumerate(group_dict.items()):

    correlations = []

    pvalues = []

    group_scores = cog_df.loc[animal_indices, 'index_NOR'].values



    for win_idx in range(n_windows):

        win_arr = all_speed[win_idx]

        medians = []

        for animal_idx in animal_indices:

            pooled = []

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :].astype(float)

                arr = arr[~np.isnan(arr)]

                pooled.append(arr)

            if pooled:

                flat = np.concatenate(pooled)

                medians.append(np.median(flat))

            else:

                medians.append(np.nan)

        medians = np.array(medians)

        mask = ~np.isnan(medians) & ~np.isnan(group_scores)

        if np.sum(mask) > 2:

            rho, pval = spearmanr(medians[mask], group_scores[mask])

        else:

            rho, pval = np.nan, np.nan

        correlations.append(rho)

        pvalues.append(pval)

        

        for idx, (group, animal_indices) in enumerate(group_dict.items()):

    # ... (existing code: correlations, pvalues, plot, scatter star markers)

    # ... after plotting significance stars



            for sig_win_idx in sig_idx:

                win = window_sizes[sig_win_idx]

                eff = correlations[sig_win_idx]

                plt.annotate(f"{eff:.2f}", 

                            (win, eff), 

                            textcoords="offset points", 

                            xytext=(0, 12),  # pixels above the star

                            ha='center', fontsize=10, color=palette[idx], weight='bold')







    label = f"{group[0]}-{group[1]}".lower()

    plt.plot(window_sizes, correlations, '-o', color=palette[idx], label=label, zorder=2)

    # Overlay significance marker for p < 0.05

    correlations = np.array(correlations)

    pvalues = np.array(pvalues)

    sig_idx = np.where((pvalues < 0.05) & ~np.isnan(pvalues) & ~np.isnan(correlations))[0]

    # Plot filled stars at significant points

    plt.scatter(np.array(window_sizes)[sig_idx], correlations[sig_idx],

                color=palette[idx], marker='*', s=110, edgecolor='k', linewidth=0.8, zorder=4, label=None)

    



plt.axhline(0, color='grey', linestyle='--', linewidth=1, zorder=1)

plt.xlabel('Window Size')

plt.ylabel("Spearman correlation (dFC speed, cognitive score)")

plt.title("Correlation vs. Window Size by Group\nStars = p < 0.05")

plt.legend(title='Group')

plt.tight_layout()

plt.show()
# %%
# %%



import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import spearmanr

import seaborn as sns



window_sizes = time_window_range

n_windows = len(window_sizes)

group_dict = data.groups

cog_df = cog_data_filtered.reset_index(drop=True)



palette = sns.color_palette('tab10', n_colors=len(group_dict))

plt.figure(figsize=(9,6))



for idx, (group, animal_indices) in enumerate(group_dict.items()):

    correlations = []

    pvalues = []

    group_scores = cog_df.loc[animal_indices, 'index_NOR'].values



    for win_idx in range(n_windows):

        win_arr = all_speed[win_idx]

        medians = []

        for animal_idx in animal_indices:

            pooled = []

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :].astype(float)

                arr = arr[~np.isnan(arr)]

                pooled.append(arr)

            if pooled:

                flat = np.concatenate(pooled)

                medians.append(np.median(flat))

            else:

                medians.append(np.nan)

        medians = np.array(medians)

        mask = ~np.isnan(medians) & ~np.isnan(group_scores)

        if np.sum(mask) > 2:

            rho, pval = spearmanr(medians[mask], group_scores[mask])

        else:

            rho, pval = np.nan, np.nan

        correlations.append(rho)

        pvalues.append(pval)

        

        for idx, (group, animal_indices) in enumerate(group_dict.items()):

    # ... (existing code: correlations, pvalues, plot, scatter star markers)

    # ... after plotting significance stars



            for sig_win_idx in sig_idx:

                win = window_sizes[sig_win_idx]

                eff = correlations[sig_win_idx]

                plt.annotate(f"{eff:.2f}", 

                            (win, eff), 

                            textcoords="offset points", 

                            xytext=(0, 12),  # pixels above the star

                            ha='center', fontsize=10, color=palette[idx], weight='bold')







    label = f"{group[0]}-{group[1]}".lower()

    plt.plot(window_sizes, correlations, '-o', color=palette[idx], label=label, zorder=2)

    # Overlay significance marker for p < 0.05

    correlations = np.array(correlations)

    pvalues = np.array(pvalues)

    sig_idx = np.where((pvalues < 0.05) & ~np.isnan(pvalues) & ~np.isnan(correlations))[0]

    # Plot filled stars at significant points

    plt.scatter(np.array(window_sizes)[sig_idx], correlations[sig_idx],

                color=palette[idx], marker='*', s=110, edgecolor='k', linewidth=0.8, zorder=4, label=None)

    



plt.axhline(0, color='grey', linestyle='--', linewidth=1, zorder=1)

plt.xlabel('Window Size')

plt.ylabel("Spearman correlation (dFC speed, cognitive score)")

plt.title("Correlation vs. Window Size by Group\nStars = p < 0.05")

plt.legend(title='Group')

plt.tight_layout()

plt.show()
# %%
# %%



import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import spearmanr

import seaborn as sns



window_sizes = time_window_range

n_windows = len(window_sizes)

group_dict = data.groups

cog_df = cog_data_filtered.reset_index(drop=True)



palette = sns.color_palette('tab10', n_colors=len(group_dict))

plt.figure(figsize=(9,6))



for idx, (group, animal_indices) in enumerate(group_dict.items()):

    correlations = []

    pvalues = []

    group_scores = cog_df.loc[animal_indices, 'index_NOR'].values



    for win_idx in range(n_windows):

        win_arr = all_speed[win_idx]

        medians = []

        for animal_idx in animal_indices:

            pooled = []

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :].astype(float)

                arr = arr[~np.isnan(arr)]

                pooled.append(arr)

            if pooled:

                flat = np.concatenate(pooled)

                medians.append(np.median(flat))

            else:

                medians.append(np.nan)

        medians = np.array(medians)

        mask = ~np.isnan(medians) & ~np.isnan(group_scores)

        if np.sum(mask) > 2:

            rho, pval = spearmanr(medians[mask], group_scores[mask])

        else:

            rho, pval = np.nan, np.nan

        correlations.append(rho)

        pvalues.append(pval)

        





    label = f"{group[0]}-{group[1]}".lower()

    plt.plot(window_sizes, correlations, '-o', color=palette[idx], label=label, zorder=2)

    # Overlay significance marker for p < 0.05

    correlations = np.array(correlations)

    pvalues = np.array(pvalues)

    sig_idx = np.where((pvalues < 0.05) & ~np.isnan(pvalues) & ~np.isnan(correlations))[0]

    # Plot filled stars at significant points

    plt.scatter(np.array(window_sizes)[sig_idx], correlations[sig_idx],

                color=palette[idx], marker='*', s=110, edgecolor='k', linewidth=0.8, zorder=4, label=None)

    

    for sig_win_idx in sig_idx:

        win = window_sizes[sig_win_idx]

        eff = correlations[sig_win_idx]

        plt.annotate(f"{eff:.2f}", 

                    (win, eff), 

                    textcoords="offset points", 

                    xytext=(0, 12),  # pixels above the star

                    ha='center', fontsize=10, color=palette[idx], weight='bold')







plt.axhline(0, color='grey', linestyle='--', linewidth=1, zorder=1)

plt.xlabel('Window Size')

plt.ylabel("Spearman correlation (dFC speed, cognitive score)")

plt.title("Correlation vs. Window Size by Group\nStars = p < 0.05")

plt.legend(title='Group')

plt.tight_layout()

plt.show()
# %%
import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import spearmanr

import seaborn as sns



window_sizes = time_window_range

n_windows = len(window_sizes)

quantile_levels = np.linspace(0.05, 0.95, 19)  # or np.arange(0,1.01,0.05)

n_quantiles = len(quantile_levels)

group_dict = data.groups

cog_df = cog_data_filtered.reset_index(drop=True)



for group_idx, (group, animal_indices) in enumerate(group_dict.items()):

    # Preallocate correlation matrix

    corr_matrix = np.full((n_quantiles, n_windows), np.nan)

    pval_matrix = np.full((n_quantiles, n_windows), np.nan)

    group_scores = cog_df.loc[animal_indices, 'index_NOR'].values



    for q_idx, quantile in enumerate(quantile_levels):

        for win_idx in range(n_windows):

            win_arr = all_speed[win_idx]  # (n_animals, n_taus, n_timepoints)

            values = []

            for animal_idx in animal_indices:

                pooled = []

                for tau in range(win_arr.shape[1]):

                    arr = win_arr[animal_idx, tau, :].astype(float)

                    arr = arr[~np.isnan(arr)]

                    pooled.append(arr)

                if pooled:

                    flat = np.concatenate(pooled)

                    values.append(np.quantile(flat, quantile))

                else:

                    values.append(np.nan)

            values = np.array(values)

            mask = ~np.isnan(values) & ~np.isnan(group_scores)

            if np.sum(mask) > 2:

                rho, pval = spearmanr(values[mask], group_scores[mask])

                corr_matrix[q_idx, win_idx] = rho

                pval_matrix[q_idx, win_idx] = pval



    # Plot heatmap of correlation across quantiles and window sizes

    plt.figure(figsize=(11, 5))

    im = plt.imshow(corr_matrix, aspect='auto', origin='lower',

                    extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

                    cmap='coolwarm', vmin=-1, vmax=1)

    plt.colorbar(im, label="Spearman ρ")

    plt.xlabel('Window Size')

    plt.ylabel('dFC Speed Quantile')

    plt.title(f'Brain-behavior correlation heatmap\nGroup: {group[0]}-{group[1]}'.lower())

    plt.tight_layout()

    plt.show()



    # (Optional) Overlay significance markers (e.g., dots where p<0.05)

    plt.figure(figsize=(11, 5))

    im = plt.imshow(corr_matrix, aspect='auto', origin='lower',

                    extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

                    cmap='coolwarm', vmin=-1, vmax=1)

    plt.colorbar(im, label="Spearman ρ")

    plt.xlabel('Window Size')

    plt.ylabel('dFC Speed Quantile')

    plt.title(f'Correlation (stars: p < 0.05) | Group: {group[0]}-{group[1]}'.lower())

    # Overlay markers where significant

    q_mesh, w_mesh = np.meshgrid(quantile_levels, window_sizes, indexing='ij')

    sig = (pval_matrix < 0.05) & ~np.isnan(pval_matrix)

    plt.scatter(w_mesh[sig], q_mesh[sig], marker='*', color='k', s=30)

    plt.tight_layout()

    plt.show()



# %%
import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import spearmanr



window_sizes = time_window_range

n_windows = len(window_sizes)

quantile_levels = np.linspace(0.05, 0.95, 19)

n_quantiles = len(quantile_levels)

group_dict = data.groups

cog_df = cog_data_filtered.reset_index(drop=True)



# Choose the two groups you want to compare

group_names = list(group_dict.keys())

gA, gB = group_names[0], group_names[1]  # Change as needed



def get_corr_matrix(animal_indices, scores):

    mat = np.full((n_quantiles, n_windows), np.nan)

    for q_idx, quantile in enumerate(quantile_levels):

        for win_idx in range(n_windows):

            win_arr = all_speed[win_idx]

            values = []

            for animal_idx in animal_indices:

                pooled = []

                for tau in range(win_arr.shape[1]):

                    arr = win_arr[animal_idx, tau, :].astype(float)

                    arr = arr[~np.isnan(arr)]

                    pooled.append(arr)

                if pooled:

                    flat = np.concatenate(pooled)

                    values.append(np.quantile(flat, quantile))

                else:

                    values.append(np.nan)

            values = np.array(values)

            mask = ~np.isnan(values) & ~np.isnan(scores)

            if np.sum(mask) > 2:

                rho, _ = spearmanr(values[mask], scores[mask])

                mat[q_idx, win_idx] = rho

    return mat



# Compute for each group

A_idxs = group_dict[gA]

B_idxs = group_dict[gB]

A_scores = cog_df.loc[A_idxs, 'index_NOR'].values

B_scores = cog_df.loc[B_idxs, 'index_NOR'].values



matA = get_corr_matrix(A_idxs, A_scores)

matB = get_corr_matrix(B_idxs, B_scores)



# Difference

diff_mat = matA - matB



plt.figure(figsize=(12, 5))

im = plt.imshow(diff_mat, aspect='auto', origin='lower',

                extent=[window_sizes[0], window_sizes[-1], quantile_levels[0], quantile_levels[-1]],

                cmap='bwr', vmin=-1, vmax=1)

plt.colorbar(im, label=f"ΔSpearman ρ ({gA} minus {gB})")

plt.xlabel('Window Size')

plt.ylabel('dFC Speed Quantile')

plt.title(f"Difference in dFC speed–cognition correlation\n{gA} minus {gB}".lower())

plt.tight_layout()

plt.show()



# %%
groups
# %%
data.groups
# %%
import pandas as pd



# For per-animal median speed as example

n_animals = sum(len(v) for v in data.groups.values())

df = pd.DataFrame(index=range(n_animals))

df['index_NOR'] = cog_data_filtered['index_NOR'].values



# Map each animal to genotype/treatment

geno = [''] * n_animals

treat = [''] * n_animals

for (g, t), idxs in data.groups.items():

    for i in idxs:

        geno[i] = g

        treat[i] = t

df['genotype'] = geno

df['treatment'] = treat



# dFC speed summary (replace with your per-animal metric)

df['dFC_speed'] = per_animal_speeds  # computed as in earlier steps



# %%
import pingouin as pg



# For partial Spearman correlation, must encode genotype/treatment as numeric or dummies:

df['geno_num'] = df['genotype'].map({'WT': 0, 'Dp1Yey': 1})

df['treat_num'] = df['treatment'].map({'VEH': 0, 'LCTB92': 1})



# Partial Spearman correlation between dFC speed and index_NOR, controlling for genotype and treatment

pcorr = pg.partial_corr(data=df, x='dFC_speed', y='index_NOR',

                       covar=['geno_num', 'treat_num'], method='spearman')

print(pcorr)



# %%
import pingouin as pg







# For partial Spearman correlation, must encode genotype/treatment as numeric or dummies:

df['geno_num'] = df['genotype'].map({'WT': 0, 'Dp1Yey': 1})

df['treat_num'] = df['treatment'].map({'VEH': 0, 'LCTB92': 1})



# Partial Spearman correlation between dFC speed and index_NOR, controlling for genotype and treatment

pcorr = pg.partial_corr(data=df, x='dFC_speed', y='index_NOR',

                       covar=['geno_num', 'treat_num'], method='spearman')

print(pcorr)



# %%
import statsmodels.api as sm



# Convert categorical to dummy/indicator variables

X = pd.get_dummies(df[['dFC_speed', 'genotype', 'treatment']], drop_first=True)

y = df['index_NOR']

X = sm.add_constant(X)

model = sm.OLS(y, X, missing='drop').fit()

print(model.summary())



# %%
import pingouin as pg











# For partial Spearman correlation, must encode genotype/treatment as numeric or dummies:

df['geno_num'] = df['genotype'].map({'WT': 0, 'Dp1Yey': 1})

df['treat_num'] = df['treatment'].map({'VEH': 0, 'LCTB92': 1})



# Partial Spearman correlation between dFC speed and index_NOR, controlling for genotype and treatment

pcorr = pg.partial_corr(data=df, x='dFC_speed', y='index_NOR',

                       covar=['geno_num', 'treat_num'], method='spearman')

print(pcorr)



# %%
import pingouin as pg











# For partial Spearman correlation, must encode genotype/treatment as numeric or dummies:

df['geno_num'] = df['genotype'].map({'WT': 0, 'Dp1Yey': 1})

df['treat_num'] = df['treatment'].map({'VEH': 0, 'LCTB92': 1})



# Partial Spearman correlation between dFC speed and index_NOR, controlling for genotype and treatment

pcorr = pg.partial_corr(data=df, x='dFC_speed', y='index_NOR',

                       covar=['geno_num'], method='spearman')





print(pcorr)



# %%
import pingouin as pg











# For partial Spearman correlation, must encode genotype/treatment as numeric or dummies:

df['geno_num'] = df['genotype'].map({'WT': 0, 'Dp1Yey': 1})

df['treat_num'] = df['treatment'].map({'VEH': 0, 'LCTB92': 1})



# Partial Spearman correlation between dFC speed and index_NOR, controlling for genotype and treatment

pcorr = pg.partial_corr(data=df, x='dFC_speed', y='index_NOR',

                       covar=['treat_num'], method='spearman')





print(pcorr)



# %%
import statsmodels.api as sm







# Convert categorical to dummy/indicator variables

X = pd.get_dummies(df[['dFC_speed', 'genotype', 'treatment']], drop_first=True)

y = df['index_NOR']

X = sm.add_constant(X)

model = sm.OLS(y, X, missing='drop').fit()

print(model.summary())



# %%
import statsmodels.api as sm







# Convert categorical to dummy/indicator variables

X = pd.get_dummies(df[['dFC_speed', 'genotype', 'treatment']], drop_first=True)



# %%


print(X.dtypes)



# %%
import statsmodels.api as sm







# Use get_dummies for all categorical variables and ensure numeric

X = pd.get_dummies(df[['dFC_speed', 'genotype', 'treatment']], drop_first=True)

X = X.apply(pd.to_numeric, errors='coerce')

X = sm.add_constant(X)

y = df['index_NOR']



# Drop any rows with NaN in either X or y

mask = (~X.isnull().any(axis=1)) & (~y.isnull())

X_clean = X.loc[mask]

y_clean = y.loc[mask]



model = sm.OLS(y_clean, X_clean).fit()

print(model.summary())



# %%
import statsmodels.api as sm





# Prepare predictors (ensure all columns are handled)

df['dFC_speed'] = per_animal_speeds  # or your metric



# This handles any dtype/NaN issues robustly

X = df[['dFC_speed', 'genotype', 'treatment']].copy()

X = pd.get_dummies(X, drop_first=True)  # 'genotype_WT', 'treatment_LCTB92', etc.

X = X.apply(pd.to_numeric, errors='coerce')

X = sm.add_constant(X)

y = df['index_NOR']



# Remove any row with NaN in X or y

mask = (~X.isnull().any(axis=1)) & (~y.isnull())

X = X.loc[mask]

y = y.loc[mask]



# Fit model

model = sm.OLS(y, X).fit()

print(model.summary())



# %%


print(X.dtypes)

print(X.head())

print(y.dtype)

print(y.head())



# %%
print(X_clean.shape, y_clean.shape)



# %%
X = X.astype({col: int for col in X.columns if X[col].dtype == bool})



# %%
print(X.dtypes)



# %%
model = sm.OLS(y, X).fit()

print(model.summary())



# %%
import numpy as np

import pandas as pd

import statsmodels.api as sm

import matplotlib.pyplot as plt



window_sizes = time_window_range

n_windows = len(window_sizes)

n_animals = cog_data_filtered.shape[0]



effect_sizes = []

p_values = []



for win_idx in range(n_windows):

    win_arr = all_speed[win_idx]  # shape: (n_animals, n_taus, n_timepoints)

    per_animal_speed = []

    for animal_idx in range(n_animals):

        pooled = []

        for tau in range(win_arr.shape[1]):

            arr = win_arr[animal_idx, tau, :].astype(float)

            arr = arr[~np.isnan(arr)]

            pooled.append(arr)

        if pooled:

            flat = np.concatenate(pooled)

            per_animal_speed.append(np.median(flat))  # Or np.mean, or np.quantile(flat, ...)

        else:

            per_animal_speed.append(np.nan)

    # Assemble DataFrame

    df = pd.DataFrame({

        'index_NOR': cog_data_filtered['index_NOR'].values,

        'genotype': cog_data_filtered['genotype'].values,

        'treatment': cog_data_filtered['treatment'].values,

        'dFC_speed': per_animal_speed

    })

    # Get dummies, ensure all numeric

    X = pd.get_dummies(df[['dFC_speed', 'genotype', 'treatment']], drop_first=True)

    X = X.astype({col: int for col in X.columns if X[col].dtype == bool})

    X = X.apply(pd.to_numeric, errors='coerce')

    X = sm.add_constant(X)

    y = pd.to_numeric(df['index_NOR'], errors='coerce')

    mask = (~X.isnull().any(axis=1)) & (~y.isnull())

    X_clean = X.loc[mask]

    y_clean = y.loc[mask]

    # Fit

    if X_clean.shape[0] > 10:  # Ensure enough data

        model = sm.OLS(y_clean, X_clean).fit()

        effect_sizes.append(model.params['dFC_speed'])

        p_values.append(model.pvalues['dFC_speed'])

    else:

        effect_sizes.append(np.nan)

        p_values.append(np.nan)



effect_sizes = np.array(effect_sizes)

p_values = np.array(p_values)



# Plot

plt.figure(figsize=(9,6))

plt.plot(window_sizes, effect_sizes, '-o', label='Effect size (coef)')

plt.axhline(0, color='grey', ls='--')

# Mark significant points

sig_idx = np.where(p_values < 0.05)[0]

plt.scatter(np.array(window_sizes)[sig_idx], effect_sizes[sig_idx], color='red', marker='*', s=100, label='p < 0.05')

for si in sig_idx:

    plt.annotate(f"{effect_sizes[si]:.2f}", (window_sizes[si], effect_sizes[si]), 

                 xytext=(0, 12), textcoords="offset points", ha='center', fontsize=10, color='red', weight='bold')

plt.xlabel('Window Size')

plt.ylabel('Effect of dFC Speed on Cognitive Score')

plt.title('Adjusted Effect of dFC Speed on Cognition\n(window-by-window, controlling for genotype & treatment)')

plt.legend()

plt.tight_layout()

plt.show()



# Optional: Plot p-values

plt.figure(figsize=(8,5))

plt.plot(window_sizes, p_values, '-o', color='darkgreen')

plt.axhline(0.05, color='k', ls=':', lw=1, label='p=0.05')

plt.xlabel('Window Size')

plt.ylabel('p-value for dFC speed')

plt.title('p-value for dFC speed (window-by-window regression)')

plt.legend()

plt.tight_layout()

plt.show()



# %%
import numpy as np

import pandas as pd

import statsmodels.api as sm

import matplotlib.pyplot as plt



window_sizes = time_window_range

n_windows = len(window_sizes)

n_animals = cog_data_filtered.shape[0]



effect_sizes = []

p_values = []



quantile_levels = np.linspace(0.05, 0.95, 19)

for win_idx in quantile_levels:





    # In per-animal_speed, use np.quantile(flat, q)

    # ...rest of code unchanged



    win_arr = all_speed[win_idx]  # shape: (n_animals, n_taus, n_timepoints)

    per_animal_speed = []

    for animal_idx in range(n_animals):

        pooled = []

        for tau in range(win_arr.shape[1]):

            arr = win_arr[animal_idx, tau, :].astype(float)

            arr = arr[~np.isnan(arr)]

            pooled.append(arr)

        if pooled:

            flat = np.concatenate(pooled)

            per_animal_speed.append(np.median(flat))  # Or np.mean, or np.quantile(flat, ...)

        else:

            per_animal_speed.append(np.nan)

    # Assemble DataFrame

    df = pd.DataFrame({

        'index_NOR': cog_data_filtered['index_NOR'].values,

        'genotype': cog_data_filtered['genotype'].values,

        'treatment': cog_data_filtered['treatment'].values,

        'dFC_speed': per_animal_speed

    })

    # Get dummies, ensure all numeric

    X = pd.get_dummies(df[['dFC_speed', 'genotype', 'treatment']], drop_first=True)

    X = X.astype({col: int for col in X.columns if X[col].dtype == bool})

    X = X.apply(pd.to_numeric, errors='coerce')

    X = sm.add_constant(X)

    y = pd.to_numeric(df['index_NOR'], errors='coerce')

    mask = (~X.isnull().any(axis=1)) & (~y.isnull())

    X_clean = X.loc[mask]

    y_clean = y.loc[mask]

    # Fit

    if X_clean.shape[0] > 10:  # Ensure enough data

        model = sm.OLS(y_clean, X_clean).fit()

        effect_sizes.append(model.params['dFC_speed'])

        p_values.append(model.pvalues['dFC_speed'])

    else:

        effect_sizes.append(np.nan)

        p_values.append(np.nan)



effect_sizes = np.array(effect_sizes)

p_values = np.array(p_values)



# Plot

plt.figure(figsize=(9,6))

plt.plot(window_sizes, effect_sizes, '-o', label='Effect size (coef)')

plt.axhline(0, color='grey', ls='--')

# Mark significant points

sig_idx = np.where(p_values < 0.05)[0]

plt.scatter(np.array(window_sizes)[sig_idx], effect_sizes[sig_idx], color='red', marker='*', s=100, label='p < 0.05')

for si in sig_idx:

    plt.annotate(f"{effect_sizes[si]:.2f}", (window_sizes[si], effect_sizes[si]), 

                 xytext=(0, 12), textcoords="offset points", ha='center', fontsize=10, color='red', weight='bold')

plt.xlabel('Window Size')

plt.ylabel('Effect of dFC Speed on Cognitive Score')

plt.title('Adjusted Effect of dFC Speed on Cognition\n(window-by-window, controlling for genotype & treatment)')

plt.legend()

plt.tight_layout()

plt.show()



# Optional: Plot p-values

plt.figure(figsize=(8,5))

plt.plot(window_sizes, p_values, '-o', color='darkgreen')

plt.axhline(0.05, color='k', ls=':', lw=1, label='p=0.05')

plt.xlabel('Window Size')

plt.ylabel('p-value for dFC speed')

plt.title('p-value for dFC speed (window-by-window regression)')

plt.legend()

plt.tight_layout()

plt.show()



# %%
import numpy as np

import pandas as pd

import statsmodels.api as sm

import matplotlib.pyplot as plt



quantile_levels = np.linspace(0.05, 0.95, 19)

n_quantiles = len(quantile_levels)

n_animals = cog_data_filtered.shape[0]



effect_sizes = []

p_values = []



for q in quantile_levels:

    per_animal_speed = []

    for animal_idx in range(n_animals):

        pooled = []

        # Pool all tau and timepoints for this animal

        for win_arr in all_speed:

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :].astype(float)

                arr = arr[~np.isnan(arr)]

                pooled.append(arr)

        if pooled:

            flat = np.concatenate(pooled)

            per_animal_speed.append(np.quantile(flat, q))

        else:

            per_animal_speed.append(np.nan)

    # Assemble DataFrame

    df = pd.DataFrame({

        'index_NOR': cog_data_filtered['index_NOR'].values,

        'genotype': cog_data_filtered['genotype'].values,

        'treatment': cog_data_filtered['treatment'].values,

        'dFC_speed': per_animal_speed

    })

    # Dummify, convert bools, ensure numeric

    X = pd.get_dummies(df[['dFC_speed', 'genotype', 'treatment']], drop_first=True)

    X = X.astype({col: int for col in X.columns if X[col].dtype == bool})

    X = X.apply(pd.to_numeric, errors='coerce')

    X = sm.add_constant(X)

    y = pd.to_numeric(df['index_NOR'], errors='coerce')

    mask = (~X.isnull().any(axis=1)) & (~y.isnull())

    X_clean = X.loc[mask]

    y_clean = y.loc[mask]

    if X_clean.shape[0] > 10:

        model = sm.OLS(y_clean, X_clean).fit()

        effect_sizes.append(model.params['dFC_speed'])

        p_values.append(model.pvalues['dFC_speed'])

    else:

        effect_sizes.append(np.nan)

        p_values.append(np.nan)



effect_sizes = np.array(effect_sizes)

p_values = np.array(p_values)



# Plot effect size

plt.figure(figsize=(8,5))

plt.plot(quantile_levels, effect_sizes, '-o', label='Effect size (coef)')

plt.axhline(0, color='grey', ls='--')

sig_idx = np.where(p_values < 0.05)[0]

plt.scatter(quantile_levels[sig_idx], effect_sizes[sig_idx], color='red', marker='*', s=100, label='p < 0.05')

for si in sig_idx:

    plt.annotate(f"{effect_sizes[si]:.2f}", (quantile_levels[si], effect_sizes[si]), 

                 xytext=(0, 12), textcoords="offset points", ha='center', fontsize=10, color='red', weight='bold')

plt.xlabel('dFC Speed Quantile')

plt.ylabel('Effect of dFC Speed on Cognitive Score')

plt.title('Adjusted Effect of dFC Speed on Cognition\n(quantile-by-quantile, controlling for genotype & treatment)')

plt.legend()

plt.tight_layout()

plt.show()



# Plot p-values

plt.figure(figsize=(8,4))

plt.plot(quantile_levels, p_values, '-o', color='darkgreen')

plt.axhline(0.05, color='k', ls=':', lw=1, label='p=0.05')

plt.xlabel('dFC Speed Quantile')

plt.ylabel('p-value for dFC speed')

plt.title('p-value for dFC speed (quantile-by-quantile regression)')

plt.legend()

plt.tight_layout()

plt.show()





# %%
# %%



#---------------------------- Two timescales --------------------------------



import numpy as np

import pandas as pd



# Split window indices into two pools (first half, second half)

n_windows = len(window_sizes)

first_half_idx = np.arange(n_windows // 2)

second_half_idx = np.arange(n_windows // 2, n_windows)



pools = [first_half_idx, second_half_idx]

pool_labels = ['Short', 'Long']



# Prepare

n_animals = cog_data_filtered.shape[0]

per_animal_summaries = {label: [] for label in pool_labels}



for pool_idx, idxs in enumerate(pools):

    for animal_idx in range(n_animals):

        pooled_speeds = []

        for win_idx in idxs:

            win_arr = all_speed[win_idx]  # shape: (n_animals, n_tau, n_timepoints)

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :].astype(float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled_speeds.append(arr)

        if pooled_speeds:

            all_pooled = np.concatenate(pooled_speeds)

            per_animal_summaries[pool_labels[pool_idx]].append(np.median(all_pooled))  # Use median, or mean, or quantile

        else:

            per_animal_summaries[pool_labels[pool_idx]].append(np.nan)



# Build a DataFrame for downstream analysis

df_summary = pd.DataFrame({

    'index_NOR': cog_data_filtered['index_NOR'].values,

    'genotype': cog_data_filtered['genotype'].values,

    'treatment': cog_data_filtered['treatment'].values,

    'dFC_speed_first': per_animal_summaries['First Half'],

    'dFC_speed_second': per_animal_summaries['Second Half']

})
# %%
# %%



#---------------------------- Two timescales --------------------------------



import numpy as np

import pandas as pd



# Split window indices into two pools (first half, second half)

n_windows = len(window_sizes)

first_half_idx = np.arange(n_windows // 2)

second_half_idx = np.arange(n_windows // 2, n_windows)



pools = [first_half_idx, second_half_idx]

pool_labels = ['short', 'long']



# Prepare

n_animals = cog_data_filtered.shape[0]

per_animal_summaries = {label: [] for label in pool_labels}



for pool_idx, idxs in enumerate(pools):

    for animal_idx in range(n_animals):

        pooled_speeds = []

        for win_idx in idxs:

            win_arr = all_speed[win_idx]  # shape: (n_animals, n_tau, n_timepoints)

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :].astype(float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled_speeds.append(arr)

        if pooled_speeds:

            all_pooled = np.concatenate(pooled_speeds)

            per_animal_summaries[pool_labels[pool_idx]].append(np.median(all_pooled))  # Use median, or mean, or quantile

        else:

            per_animal_summaries[pool_labels[pool_idx]].append(np.nan)



# Build a DataFrame for downstream analysis

df_summary = pd.DataFrame({

    'index_NOR': cog_data_filtered['index_NOR'].values,

    'genotype': cog_data_filtered['genotype'].values,

    'treatment': cog_data_filtered['treatment'].values,

    'dFC_speed_short': per_animal_summaries['short'],

    'dFC_speed_long': per_animal_summaries['long']

})
# %%
from scipy.stats import wilcoxon



x = df_summary['dFC_speed_short']

y = df_summary['dFC_speed_long']



# Drop animals with NaN in either

mask = (~x.isnull()) & (~y.isnull())

stat, p_value = wilcoxon(x[mask], y[mask])



print(f"Wilcoxon signed-rank test:\nStatistic={stat:.2f}, p={p_value:.3g}")



# %%
import statsmodels.api as sm



for pool in ['short', 'long']:

    X = pd.get_dummies(df_summary[['dFC_speed_' + pool, 'genotype', 'treatment']], drop_first=True)

    # Convert bools to int (critical!)

    X = X.astype({col: int for col in X.columns if X[col].dtype == bool})

    X = X.apply(pd.to_numeric, errors='coerce')

    X = sm.add_constant(X)

    y = pd.to_numeric(df_summary['index_NOR'], errors='coerce')

    mask = (~X.isnull().any(axis=1)) & (~y.isnull())

    X_clean = X.loc[mask]

    y_clean = y.loc[mask]

    model = sm.OLS(y_clean, X_clean).fit()

    print(f"\nRegression for {pool} windows:")

    print(model.summary())



# %%
import numpy as np

import pandas as pd

import statsmodels.api as sm



n_boot = 1000

coefs_short = []

coefs_long = []

np.random.seed(42)



for i in range(n_boot):

    # Sample with replacement, get indices

    idx = np.random.choice(df_summary.index, size=len(df_summary), replace=True)

    df_boot = df_summary.loc[idx]

    # Short

    Xs = pd.get_dummies(df_boot[['dFC_speed_short', 'genotype', 'treatment']], drop_first=True)

    Xs = Xs.astype({col: int for col in Xs.columns if Xs[col].dtype == bool})

    Xs = Xs.apply(pd.to_numeric, errors='coerce')

    Xs = sm.add_constant(Xs)

    ys = pd.to_numeric(df_boot['index_NOR'], errors='coerce')

    mask_s = (~Xs.isnull().any(axis=1)) & (~ys.isnull())

    Xs = Xs.loc[mask_s]

    ys = ys.loc[mask_s]

    if Xs.shape[0] > 10:

        coefs_short.append(sm.OLS(ys, Xs).fit().params['dFC_speed_short'])

    else:

        coefs_short.append(np.nan)

    # Long

    Xl = pd.get_dummies(df_boot[['dFC_speed_long', 'genotype', 'treatment']], drop_first=True)

    Xl = Xl.astype({col: int for col in Xl.columns if Xl[col].dtype == bool})

    Xl = Xl.apply(pd.to_numeric, errors='coerce')

    Xl = sm.add_constant(Xl)

    yl = pd.to_numeric(df_boot['index_NOR'], errors='coerce')

    mask_l = (~Xl.isnull().any(axis=1)) & (~yl.isnull())

    Xl = Xl.loc[mask_l]

    yl = yl.loc[mask_l]

    if Xl.shape[0] > 10:

        coefs_long.append(sm.OLS(yl, Xl).fit().params['dFC_speed_long'])

    else:

        coefs_long.append(np.nan)



coefs_short = np.array(coefs_short)

coefs_long = np.array(coefs_long)

delta = coefs_short - coefs_long



# Confidence interval

ci_lower = np.nanpercentile(delta, 2.5)

ci_upper = np.nanpercentile(delta, 97.5)

mean_diff = np.nanmean(delta)

pval = 2 * min(np.mean(delta > 0), np.mean(delta < 0))  # two-sided



print(f"Mean difference in effect size (short - long): {mean_diff:.3f}")

print(f"95% bootstrap CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

print(f"Two-sided bootstrap p-value: {pval:.3f}")



# Optional: plot the bootstrap distribution

import matplotlib.pyplot as plt

plt.figure(figsize=(7,4))

plt.hist(delta, bins=30, alpha=0.7, color='steelblue')

plt.axvline(0, color='k', ls='--', label='No difference')

plt.axvline(ci_lower, color='red', ls=':', label='95% CI')

plt.axvline(ci_upper, color='red', ls=':')

plt.title('Bootstrap Distribution of (Short - Long) dFC Speed Effect Size')

plt.xlabel('Δ Coefficient (short - long)')

plt.ylabel('Frequency')

plt.legend()

plt.tight_layout()

plt.show()



# %%
import pandas as pd

import statsmodels.api as sm



groups = df_summary.groupby(['genotype', 'treatment'])



results = []



for name, subdf in groups:

    for pool in ['short', 'long']:

        # Prepare

        X = subdf[['dFC_speed_' + pool]].copy()

        X = sm.add_constant(X)

        y = subdf['index_NOR']

        mask = (~X.isnull().any(axis=1)) & (~y.isnull())

        X_clean = X.loc[mask]

        y_clean = y.loc[mask]

        if X_clean.shape[0] > 3:  # Avoid crashing with tiny groups

            model = sm.OLS(y_clean, X_clean).fit()

            coef = model.params['dFC_speed_' + pool]

            pval = model.pvalues['dFC_speed_' + pool]

        else:

            coef = np.nan

            pval = np.nan

        results.append({'group': name, 'window': pool, 'coef': coef, 'pval': pval})



df_group_results = pd.DataFrame(results)



# %%
print(df_group_results)



# %%
# %%



short_idx = np.arange(n_windows // 2)

long_idx = np.arange(n_windows // 2, n_windows)



all_speeds_short = []

all_speeds_long = []



for idxs, pool in zip([short_idx, long_idx], ['short', 'long']):

    pool_speeds = []

    for win_idx in idxs:

        win_arr = all_speed[win_idx]  # (n_animals, n_tau, n_timepoints)

        for animal_idx in range(win_arr.shape[0]):

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :].astype(float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pool_speeds.append(arr)

    flat = np.concatenate(pool_speeds) if pool_speeds else np.array([])

    if pool == 'short':

        all_speeds_short = flat

    else:

        all_speeds_long = flat



import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize=(8,5))

sns.histplot(all_speeds_short, bins=75, color='royalblue', label='Short windows',

             stat='density', element='step', fill=False, linewidth=1.7)

sns.histplot(all_speeds_long, bins=75, color='firebrick', label='Long windows',

             stat='density', element='step', fill=False, linewidth=1.7)



# Optional: add median lines

plt.axvline(np.median(all_speeds_short), color='royalblue', linestyle='--', lw=1)

plt.axvline(np.median(all_speeds_long), color='firebrick', linestyle='--', lw=1)



plt.xlabel("dFC Speed")

plt.ylabel("Density")

plt.title("Distribution of dFC Speeds: Short vs. Long Window Pools")

plt.legend()

plt.tight_layout()

plt.show()
# %%
print(f"Short: n={len(all_speeds_short)}, median={np.median(all_speeds_short):.3f}")

print(f"Long: n={len(all_speeds_long)}, median={np.median(all_speeds_long):.3f}")





# %%
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



# Assume: groups = df_summary.groupby(['genotype', 'treatment']).groups

#         window_sizes, all_speed, etc. already defined



short_idx = np.arange(len(window_sizes) // 2)

long_idx = np.arange(len(window_sizes) // 2, len(window_sizes))

pool_defs = {'Short windows': short_idx, 'Long windows': long_idx}

palette = sns.color_palette('tab10', n_colors=len(groups))



plt.figure(figsize=(12, 6))



for pool_i, (pool_name, idxs) in enumerate(pool_defs.items()):

    plt.subplot(1, 2, pool_i+1)

    for g_idx, (group, animal_idxs) in enumerate(groups.items()):

        # Pool all speeds for this group and this pool

        group_speeds = []

        for win_idx in idxs:

            win_arr = all_speed[win_idx]  # (n_animals, n_tau, n_timepoints)

            for animal_idx in animal_idxs:

                for tau in range(win_arr.shape[1]):

                    arr = win_arr[animal_idx, tau, :].astype(float)

                    arr = arr[~np.isnan(arr)]

                    if arr.size > 0:

                        group_speeds.append(arr)

        group_speeds = np.concatenate(group_speeds) if group_speeds else np.array([])

        # Histogram (step)

        sns.histplot(group_speeds, bins=60, stat='density', element='step', fill=False,

                     color=palette[g_idx], linewidth=1.6, label=f'{group}', alpha=0.6)

        # KDE (over histogram)

        if group_speeds.size > 10:  # Avoid noise for tiny samples

            sns.kdeplot(group_speeds, color=palette[g_idx], lw=2.1, label=None)

        # Median

        plt.axvline(np.median(group_speeds), color=palette[g_idx], linestyle='--', lw=1)

    plt.xlabel("dFC Speed")

    plt.ylabel("Density")

    plt.title(f"{pool_name}")

    if pool_i == 0:

        plt.legend(title='Group', fontsize=10)

    else:

        plt.legend().set_visible(False)

    plt.tight_layout()



plt.suptitle("Distribution of dFC Speeds by Group\nShort vs. Long Window Pools", fontsize=15, y=1.02)

plt.tight_layout()

plt.show()



# %%
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



# Assume: groups = df_summary.groupby(['genotype', 'treatment']).groups

#         window_sizes, all_speed, etc. already defined



short_idx = np.arange(len(window_sizes) // 2)

long_idx = np.arange(len(window_sizes) // 2, len(window_sizes))

pool_defs = {'Short windows': short_idx, 'Long windows': long_idx}

palette = sns.color_palette('tab10', n_colors=len(groups))



plt.figure(figsize=(12, 6))



for pool_i, (pool_name, idxs) in enumerate(pool_defs.items()):

    plt.subplot(1, 2, pool_i+1)

    for g_idx, (group, animal_idxs) in enumerate(groups.items()):

        # Pool all speeds for this group and this pool

        group_speeds = []

        for win_idx in idxs:

            win_arr = all_speed[win_idx]  # (n_animals, n_tau, n_timepoints)

            for animal_idx in animal_idxs:

                for tau in range(win_arr.shape[1]):

                    arr = win_arr[animal_idx, tau, :].astype(float)

                    arr = arr[~np.isnan(arr)]

                    if arr.size > 0:

                        group_speeds.append(arr)

        group_speeds = np.concatenate(group_speeds) if group_speeds else np.array([])

        # Histogram (step)

        sns.histplot(group_speeds, bins=60, stat='density', element='step', fill=False,

                     color=palette[g_idx], linewidth=1.6, label=f'{group}', alpha=0.6)

        # KDE (over histogram)

        if group_speeds.size > 10:  # Avoid noise for tiny samples

            sns.kdeplot(group_speeds, color=palette[g_idx], lw=2.1, label=None)

        # Median

        plt.axvline(np.median(group_speeds), color=palette[g_idx], linestyle='--', lw=1)

    plt.xlabel("dFC Speed")

    plt.ylabel("Density")

    plt.title(f"{pool_name}")

    if pool_i == 0:

        plt.legend(title='Group', fontsize=10)

    else:

        plt.legend().set_visible(False)

    plt.tight_layout()



plt.suptitle("Distribution of dFC Speeds by Group\nShort vs. Long Window Pools", fontsize=15, y=1.02)

plt.tight_layout()

plt.show()



# %%
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



# Assume: groups = df_summary.groupby(['genotype', 'treatment']).groups

#         window_sizes, all_speed, etc. already defined



short_idx = np.arange(len(window_sizes) // 2)

long_idx = np.arange(len(window_sizes) // 2, len(window_sizes))

pool_defs = {'Short windows': short_idx, 'Long windows': long_idx}

palette = sns.color_palette('tab10', n_colors=len(groups))



plt.figure(figsize=(12, 6))



for pool_i, (pool_name, idxs) in enumerate(pool_defs.items()):

    plt.subplot(1, 2, pool_i+1)

    for g_idx, (group, animal_idxs) in enumerate(data.groups.items()):





        # Pool all speeds for this group and this pool

        group_speeds = []

        for win_idx in idxs:

            win_arr = all_speed[win_idx]  # (n_animals, n_tau, n_timepoints)

            for animal_idx in animal_idxs:

                for tau in range(win_arr.shape[1]):

                    arr = win_arr[animal_idx, tau, :].astype(float)

                    arr = arr[~np.isnan(arr)]

                    if arr.size > 0:

                        group_speeds.append(arr)

        group_speeds = np.concatenate(group_speeds) if group_speeds else np.array([])

        # Histogram (step)

        sns.histplot(group_speeds, bins=60, stat='density', element='step', fill=False,

                     color=palette[g_idx], linewidth=1.6, label=f'{group}', alpha=0.6)

        # KDE (over histogram)

        if group_speeds.size > 10:  # Avoid noise for tiny samples

            sns.kdeplot(group_speeds, color=palette[g_idx], lw=2.1, label=None)

        # Median

        plt.axvline(np.median(group_speeds), color=palette[g_idx], linestyle='--', lw=1)

    plt.xlabel("dFC Speed")

    plt.ylabel("Density")

    plt.title(f"{pool_name}")

    if pool_i == 0:

        plt.legend(title='Group', fontsize=10)

    else:

        plt.legend().set_visible(False)

    plt.tight_layout()



plt.suptitle("Distribution of dFC Speeds by Group\nShort vs. Long Window Pools", fontsize=15, y=1.02)

plt.tight_layout()

plt.show()



# %%
# %%



import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



# Assume: groups = df_summary.groupby(['genotype', 'treatment']).groups

#         window_sizes, all_speed, etc. already defined



short_idx = np.arange(len(window_sizes) // 2)

long_idx = np.arange(len(window_sizes) // 2, len(window_sizes))

pool_defs = {'Short windows': short_idx, 'Long windows': long_idx}

palette = sns.color_palette('tab10', n_colors=len(groups))



plt.figure(figsize=(12, 6))



for pool_i, (pool_name, idxs) in enumerate(pool_defs.items()):

    plt.subplot(1, 2, pool_i+1)

    for g_idx, (group, animal_idxs) in enumerate(data.groups.items()):





        # Pool all speeds for this group and this pool

        group_speeds = []

        for win_idx in idxs:

            win_arr = all_speed[win_idx]  # (n_animals, n_tau, n_timepoints)

            for animal_idx in animal_idxs:

                for tau in range(win_arr.shape[1]):

                    arr = win_arr[animal_idx, tau, :].astype(float)

                    arr = arr[~np.isnan(arr)]

                    if arr.size > 0:

                        group_speeds.append(arr)

        group_speeds = np.concatenate(group_speeds) if group_speeds else np.array([])

        # Histogram (step)

        sns.histplot(group_speeds, bins=60, stat='density', element='step', fill=False,

                     color=palette[g_idx], linewidth=1.6, label=f'{group}', alpha=0.6)

        # KDE (over histogram)

        if group_speeds.size > 10:  # Avoid noise for tiny samples

            sns.kdeplot(group_speeds, color=palette[g_idx], lw=2.1, label=None)

        # Median

        plt.axvline(np.median(group_speeds), color=palette[g_idx], linestyle='--', lw=1)

    plt.xlabel("dFC Speed")

    plt.ylabel("Density")

    plt.title(f"{pool_name}")

    plt.yscale('log')  # Log scale for better visibility

    if pool_i == 0:

        plt.legend(title='Group', fontsize=10)

    else:

        plt.legend().set_visible(False)

    plt.tight_layout()



plt.suptitle("Distribution of dFC Speeds by Group\nShort vs. Long Window Pools", fontsize=15, y=1.02)

plt.tight_layout()

plt.show()
# %%
# %%



import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



# Assume: groups = df_summary.groupby(['genotype', 'treatment']).groups

#         window_sizes, all_speed, etc. already defined



short_idx = np.arange(len(window_sizes) // 2)

long_idx = np.arange(len(window_sizes) // 2, len(window_sizes))

pool_defs = {'Short windows': short_idx, 'Long windows': long_idx}

palette = sns.color_palette('tab10', n_colors=len(groups))



plt.figure(figsize=(12, 6))



for pool_i, (pool_name, idxs) in enumerate(pool_defs.items()):

    plt.subplot(1, 2, pool_i+1)

    for g_idx, (group, animal_idxs) in enumerate(data.groups.items()):





        # Pool all speeds for this group and this pool

        group_speeds = []

        for win_idx in idxs:

            win_arr = all_speed[win_idx]  # (n_animals, n_tau, n_timepoints)

            for animal_idx in animal_idxs:

                for tau in range(win_arr.shape[1]):

                    arr = win_arr[animal_idx, tau, :].astype(float)

                    arr = arr[~np.isnan(arr)]

                    if arr.size > 0:

                        group_speeds.append(arr)

        group_speeds = np.concatenate(group_speeds) if group_speeds else np.array([])

        # Histogram (step)

        sns.histplot(group_speeds, bins=60, stat='density', element='step', fill=False,

                     color=palette[g_idx], linewidth=1.6, label=f'{group}', alpha=0.6)

        # KDE (over histogram)

        if group_speeds.size > 10:  # Avoid noise for tiny samples

            sns.kdeplot(group_speeds, color=palette[g_idx], lw=2.1, label=None)

        # Median

        plt.axvline(np.median(group_speeds), color=palette[g_idx], linestyle='--', lw=1)

    plt.xlabel("dFC Speed")

    plt.ylabel("Density")

    plt.title(f"{pool_name}")

    plt.yscale('log')  # Log scale for better visibility

    if pool_i == 0:

        plt.legend(title='Group', fontsize=10)

    else:

        plt.legend().set_visible(False)

    plt.tight_layout()



plt.suptitle("Distribution of dFC Speeds by Group\nShort vs. Long Window Pools", fontsize=15, y=1.02)

plt.tight_layout()

plt.show()
# %%
# Choose window pool: short_idx or long_idx

window_pool = short_idx  # or long_idx for the other pool



group_speeds_dict = {}



for group, animal_idxs in groups.items():

    pool_speeds = []

    for win_idx in window_pool:

        win_arr = all_speed[win_idx]  # (n_animals, n_tau, n_timepoints)

        for animal_idx in animal_idxs:

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :].astype(float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pool_speeds.append(arr)

    group_speeds = np.concatenate(pool_speeds) if pool_speeds else np.array([])

    group_speeds_dict[group] = group_speeds



# %%
# Choose window pool: short_idx or long_idx

window_pool = short_idx  # or long_idx for the other pool



group_speeds_dict = {}



for group, animal_idxs in data.groups.items():





    pool_speeds = []

    for win_idx in window_pool:

        win_arr = all_speed[win_idx]  # (n_animals, n_tau, n_timepoints)

        for animal_idx in animal_idxs:

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :].astype(float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pool_speeds.append(arr)

    group_speeds = np.concatenate(pool_speeds) if pool_speeds else np.array([])

    group_speeds_dict[group] = group_speeds



# %%
from scipy.stats import kruskal



# Prepare data for test (lists of arrays)

data_for_test = [arr for arr in group_speeds_dict.values()]

stat, pval = kruskal(*data_for_test)



print(f"Kruskal–Wallis H = {stat:.3f}, p = {pval:.3g}")



# %%
import matplotlib.pyplot as plt

import seaborn as sns



# For seaborn: melt into a dataframe

import pandas as pd

df_plot = pd.DataFrame({

    'dFC_speed': np.concatenate(list(group_speeds_dict.values())),

    'group': np.concatenate([

        np.repeat(str(g), len(arr)) for g, arr in group_speeds_dict.items()

    ])

})



plt.figure(figsize=(8,5))

sns.violinplot(x='group', y='dFC_speed', data=df_plot, inner='quartile', linewidth=1.4)

sns.stripplot(x='group', y='dFC_speed', data=df_plot, color='k', alpha=0.4, jitter=0.3, size=2)

plt.ylabel('dFC Speed')

plt.xlabel('Group')

plt.title('dFC Speed Distribution by Group\n(Short Window Pool)')

plt.tight_layout()

plt.show()



# %%
from scipy.stats import mannwhitneyu

from itertools import combinations



# Bonferroni correction for multiple comparisons

n_comps = len(group_speeds_dict) * (len(group_speeds_dict)-1) // 2

for g1, g2 in combinations(group_speeds_dict.keys(), 2):

    u, p = mannwhitneyu(group_speeds_dict[g1], group_speeds_dict[g2], alternative='two-sided')

    print(f"{g1} vs {g2}: U = {u:.2g}, uncorrected p = {p:.4f}, Bonferroni-corrected p = {min(p*n_comps,1):.4f}")



# %%
import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd



# Assuming df_plot from previous step

# Convert group labels to compact strings

df_plot['group_label'] = df_plot['group'].apply(lambda x: '-'.join(eval(x)) if isinstance(x, str) and x.startswith('(') else str(x))



plt.figure(figsize=(10, 6))

sns.violinplot(x='group_label', y='dFC_speed', data=df_plot, inner='quartile', linewidth=1.6, cut=0, scale='width', palette='tab10')

sns.stripplot(x='group_label', y='dFC_speed', data=df_plot, color='k', alpha=0.10, jitter=0.25, size=1.1)



plt.ylabel('dFC Speed', fontsize=15)

plt.xlabel('Group', fontsize=14)

plt.title('dFC Speed Distribution by Group\n(Short Window Pool)', fontsize=17, pad=18)

plt.xticks(fontsize=13, rotation=15)

plt.yticks(fontsize=13)

plt.ylim(0, 1.05)



sns.despine()

plt.tight_layout()

plt.show()



# %%
import numpy as np

import matplotlib.pyplot as plt



# Select your groups (replace as needed)

group1 = ('WT', 'VEH')

group2 = ('Dp1Yey', 'VEH')



# Get arrays of all dFC speed values for each group (use previous pooling code)

arr1 = group_speeds_dict[group1]

arr2 = group_speeds_dict[group2]



# Ensure arrays are of the same length for quantile alignment

n_points = min(len(arr1), len(arr2), 10_000)  # Limit to 10,000 for clarity if large

q = np.linspace(0, 1, n_points)

quant1 = np.quantile(arr1, q)

quant2 = np.quantile(arr2, q)



# Q–Q plot

plt.figure(figsize=(6,6))

plt.plot(quant1, quant2, 'o', alpha=0.2, markersize=2, label=f'{group1} vs {group2}')

plt.plot([quant1.min(), quant1.max()], [quant1.min(), quant1.max()], 'k--', label='y=x')

plt.xlabel(f'Quantiles: {group1}')

plt.ylabel(f'Quantiles: {group2}')

plt.title(f'Q–Q Plot of dFC Speeds\n({group1} vs {group2})')

plt.legend()

plt.tight_layout()

plt.show()



# %%
import numpy as np

import matplotlib.pyplot as plt

from itertools import combinations



# Use your pooled speeds per group for a given window pool

# Example: group_speeds_dict[group] gives you the dFC speed array for each group



groups_list = list(group_speeds_dict.keys())

n_points = 1000  # Number of quantile points for smooth curves



plt.figure(figsize=(8,8))



for (g1, g2) in combinations(groups_list, 2):

    arr1 = group_speeds_dict[g1]

    arr2 = group_speeds_dict[g2]

    # Avoid empty groups

    if len(arr1) == 0 or len(arr2) == 0:

        continue

    # Q-Q quantiles

    q = np.linspace(0, 1, n_points)

    quant1 = np.quantile(arr1, q)

    quant2 = np.quantile(arr2, q)

    # Compact label

    label = f"{'-'.join(g1)} vs {'-'.join(g2)}"

    plt.plot(quant1, quant2, lw=2, alpha=0.85, label=label)



# 1:1 line for reference

min_all = min([group_speeds_dict[g].min() for g in groups_list if len(group_speeds_dict[g]) > 0])

max_all = max([group_speeds_dict[g].max() for g in groups_list if len(group_speeds_dict[g]) > 0])

plt.plot([min_all, max_all], [min_all, max_all], 'k--', lw=1, label='y = x')



plt.xlabel('Quantiles, group 1')

plt.ylabel('Quantiles, group 2')

plt.title('Q–Q Plots: All Pairwise Group Comparisons\n(Short Window Pool)')

plt.legend(fontsize=9, frameon=True, loc='best', ncol=1)

plt.tight_layout()

plt.show()



# %%
import numpy as np

import matplotlib.pyplot as plt

from itertools import combinations



# Use your pooled speeds per group for a given window pool

# Example: group_speeds_dict[group] gives you the dFC speed array for each group



groups_list = list(group_speeds_dict.keys())

n_points = 1000  # Number of quantile points for smooth curves



plt.figure(figsize=(8,8))



for (g1, g2) in combinations(groups_list, 2):

    arr1 = group_speeds_dict[g1]

    arr2 = group_speeds_dict[g2]

    # Avoid empty groups

    if len(arr1) == 0 or len(arr2) == 0:

        continue

    # Q-Q quantiles

    q = np.linspace(0, 1, n_points)

    quant1 = np.quantile(arr1, q)

    quant2 = np.quantile(arr2, q)

    # Compact label

    label = f"{'-'.join(g1)} vs {'-'.join(g2)}"

    plt.plot(quant1, quant2, lw=2, alpha=0.85, label=label)



# 1:1 line for reference

min_all = min([group_speeds_dict[g].min() for g in groups_list if len(group_speeds_dict[g]) > 0])

max_all = max([group_speeds_dict[g].max() for g in groups_list if len(group_speeds_dict[g]) > 0])

plt.plot([min_all, max_all], [min_all, max_all], 'k--', lw=1, label='y = x')



plt.xlabel('Quantiles, group 1')

plt.ylabel('Quantiles, group 2')

plt.title('Q–Q Plots: All Pairwise Group Comparisons\n(Short Window Pool)')

plt.legend(fontsize=9, frameon=True, loc='best', ncol=1)

plt.tight_layout()

plt.show()



# %%
import numpy as np

import matplotlib.pyplot as plt

from itertools import combinations



def group_label_sort(g1, g2):

    """Return WT group first if present, else alphabetical."""

    if 'WT' in g1 and 'WT' not in g2:

        return (g1, g2)

    elif 'WT' not in g1 and 'WT' in g2:

        return (g2, g1)

    else:

        # Fallback: alphabetically

        return (g1, g2)



groups_list = list(group_speeds_dict.keys())

n_points = 1000



plt.figure(figsize=(8,8))



for pair in combinations(groups_list, 2):

    # Sort so WT is always first if present

    g1, g2 = group_label_sort(pair[0], pair[1])

    arr1 = group_speeds_dict[g1]

    arr2 = group_speeds_dict[g2]

    if len(arr1) == 0 or len(arr2) == 0:

        continue

    q = np.linspace(0, 1, n_points)

    quant1 = np.quantile(arr1, q)

    quant2 = np.quantile(arr2, q)

    above = quant2 > quant1

    below = ~above



    # Plot segments above the diagonal (blue), below (red)

    plt.plot(quant1[above], quant2[above], color='dodgerblue', alpha=0.8, lw=2)

    plt.plot(quant1[below], quant2[below], color='firebrick', alpha=0.8, lw=2)

    

    # Add a label only at the median quantile for clarity (reduce clutter)

    mid = n_points // 2

    label = f"{'-'.join(g1)} vs {'-'.join(g2)}"

    plt.text(quant1[mid], quant2[mid], label, fontsize=8, ha='left', va='bottom', alpha=0.75, rotation=25)



# Reference line

all_vals = np.concatenate([v for v in group_speeds_dict.values() if len(v) > 0])

plt.plot([all_vals.min(), all_vals.max()],

         [all_vals.min(), all_vals.max()],

         'k--', lw=1, label='y = x')



plt.xlabel('Quantile: WT or Group 1')

plt.ylabel('Quantile: Group 2')

plt.title('Q–Q Plots (WT First)\nBlue: Group2 > WT, Red: Group2 < WT')

plt.tight_layout()

plt.show()



# %%
import numpy as np

import matplotlib.pyplot as plt



# Pick your groups to compare (WT always first if present)

g1 = ('WT', 'VEH')

g2 = ('Dp1Yey', 'VEH')

arr1 = group_speeds_dict[g1]

arr2 = group_speeds_dict[g2]



n_points = 1000

q = np.linspace(0, 1, n_points)

quant1 = np.quantile(arr1, q)

quant2 = np.quantile(arr2, q)



plt.figure(figsize=(7,7))



# Fill between: blue if quant2 > quant1, red if quant2 < quant1

above = quant2 > quant1

below = quant2 < quant1



plt.fill_between(quant1, quant1, quant2, where=above, color='dodgerblue', alpha=0.4, label='Group2 > Group1')

plt.fill_between(quant1, quant1, quant2, where=below, color='firebrick', alpha=0.4, label='Group2 < Group1')



# Q-Q line

plt.plot(quant1, quant2, color='k', lw=2, label='Q–Q curve')

# Diagonal

plt.plot([quant1.min(), quant1.max()], [quant1.min(), quant1.max()], 'k--', lw=1, label='y = x')



plt.xlabel(f'Quantiles: {"-".join(g1)}')

plt.ylabel(f'Quantiles: {"-".join(g2)}')

plt.title(f'Q–Q Plot (Filled)\n{"-".join(g1)} vs {"-".join(g2)}')

plt.legend(loc='upper left', fontsize=11, frameon=True)

plt.tight_layout()

plt.show()



# %%
import numpy as np

import matplotlib.pyplot as plt

from itertools import combinations



groups_list = list(group_speeds_dict.keys())

n_pairs = len(groups_list) * (len(groups_list) - 1) // 2

n_cols = 3  # adjust if you have more/fewer groups

n_rows = int(np.ceil(n_pairs / n_cols))

n_points = 1000



fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), squeeze=False)



for ax, (g1, g2) in zip(axes.flat, combinations(groups_list, 2)):

    arr1 = group_speeds_dict[g1]

    arr2 = group_speeds_dict[g2]

    if len(arr1) == 0 or len(arr2) == 0:

        ax.axis('off')

        continue

    q = np.linspace(0, 1, n_points)

    quant1 = np.quantile(arr1, q)

    quant2 = np.quantile(arr2, q)

    above = quant2 > quant1

    below = quant2 < quant1

    # Fill between: RED if quant2 > quant1, BLUE if quant2 < quant1

    ax.fill_between(quant1, quant1, quant2, where=above, color='firebrick', alpha=0.45, label='Group2 > Group1')

    ax.fill_between(quant1, quant1, quant2, where=below, color='dodgerblue', alpha=0.45, label='Group2 < Group1')

    # Q-Q curve and diagonal

    ax.plot(quant1, quant2, color='k', lw=1.6)

    ax.plot([quant1.min(), quant1.max()], [quant1.min(), quant1.max()], 'k--', lw=1)

    # Labeling

    lab1 = '-'.join(g1) if isinstance(g1, tuple) else str(g1)

    lab2 = '-'.join(g2) if isinstance(g2, tuple) else str(g2)

    ax.set_xlabel(f'Quantiles: {lab1}', fontsize=12)

    ax.set_ylabel(f'Quantiles: {lab2}', fontsize=12)

    ax.set_title(f'{lab2} vs {lab1}', fontsize=14)

    ax.legend(fontsize=10, loc='upper left')

    ax.set_aspect('equal', adjustable='datalim')



# Turn off empty axes if any

for i in range(n_pairs, n_rows*n_cols):

    axes.flat[i].axis('off')



plt.suptitle('Q–Q Plots (Filled): All Group Pairwise Comparisons\nRed: Group2>Group1 | Blue: Group2<Group1', fontsize=16, y=1.02)

plt.tight_layout(rect=[0, 0, 1, 0.97])

plt.show()



# %%
import numpy as np

import matplotlib.pyplot as plt

from itertools import combinations



groups_list = list(group_speeds_dict.keys())

n_pairs = len(groups_list) * (len(groups_list) - 1) // 2

n_cols = 3  # adjust if you have more/fewer groups

n_rows = int(np.ceil(n_pairs / n_cols))

n_points = 1000



fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), squeeze=False)



for ax, (g1, g2) in zip(axes.flat, combinations(groups_list, 2)):

    arr1 = group_speeds_dict[g1]

    arr2 = group_speeds_dict[g2]

    if len(arr1) == 0 or len(arr2) == 0:

        ax.axis('off')

        continue

    q = np.linspace(0, 1, n_points)

    quant1 = np.quantile(arr1, q)

    quant2 = np.quantile(arr2, q)

    above = quant2 > quant1

    below = quant2 < quant1

    # Fill between: RED if quant2 > quant1, BLUE if quant2 < quant1

    ax.fill_between(quant1, quant1, quant2, where=above, color='firebrick', alpha=0.45, label='Group2 > Group1')

    ax.fill_between(quant1, quant1, quant2, where=below, color='dodgerblue', alpha=0.45, label='Group2 < Group1')

    # Q-Q curve and diagonal

    ax.plot(quant1, quant2, color='k', lw=1.6)

    ax.plot([quant1.min(), quant1.max()], [quant1.min(), quant1.max()], 'k--', lw=1)

    # Labeling

    def group_to_str(group):

        if isinstance(group, tuple):

            return f"{group[0]}-{group[1]}"

        else:

            return str(group)



    lab1 = group_to_str(g1)





    lab2 = group_to_str(g2)

    ax.set_xlabel(f'Quantiles: {lab1}', fontsize=13)

    ax.set_ylabel(f'Quantiles: {lab2}', fontsize=13)

    ax.set_title(f'Q–Q: {lab2} vs {lab1}', fontsize=15)



    ax.legend(fontsize=10, loc='upper left')

    ax.set_aspect('equal', adjustable='datalim')



# Turn off empty axes if any

for i in range(n_pairs, n_rows*n_cols):

    axes.flat[i].axis('off')



plt.suptitle('Q–Q Plots (Filled): All Group Pairwise Comparisons\nRed: Group2>Group1 | Blue: Group2<Group1', fontsize=16, y=1.02)

plt.tight_layout(rect=[0, 0, 1, 0.97])

plt.show()



# %%
import numpy as np

import matplotlib.pyplot as plt

from itertools import combinations



groups_list = list(group_speeds_dict.keys())

n_pairs = len(groups_list) * (len(groups_list) - 1) // 2

n_cols = 3  # adjust if you have more/fewer groups

n_rows = int(np.ceil(n_pairs / n_cols))

n_points = 1000



fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), squeeze=False)



for ax, (g1, g2) in zip(axes.flat, combinations(groups_list, 2)):

    arr1 = group_speeds_dict[g1]

    arr2 = group_speeds_dict[g2]

    if len(arr1) == 0 or len(arr2) == 0:

        ax.axis('off')

        continue

    q = np.linspace(0, 1, n_points)

    quant1 = np.quantile(arr1, q)

    quant2 = np.quantile(arr2, q)

    above = quant2 > quant1

    below = quant2 < quant1

    ax.fill_between(quant1, quant1, quant2, where=above, color='firebrick', alpha=0.45, label='Group2 > Group1')

    ax.fill_between(quant1, quant1, quant2, where=below, color='dodgerblue', alpha=0.45, label='Group2 < Group1')

    ax.plot(quant1, quant2, color='k', lw=1.6)

    ax.plot([quant1.min(), quant1.max()], [quant1.min(), quant1.max()], 'k--', lw=1)

    # --- Improved group labels:

    def group_to_str(group):

        if isinstance(group, tuple):

            return f"{group[0]}-{group[1]}"

        else:

            return str(group)

    lab1 = group_to_str(g1)

    lab2 = group_to_str(g2)

    ax.set_xlabel(f'Quantiles: {lab1}', fontsize=13)

    ax.set_ylabel(f'Quantiles: {lab2}', fontsize=13)

    ax.set_title(f'Q–Q: {lab2} vs {lab1}', fontsize=15)

    ax.legend(fontsize=10, loc='upper left')

    ax.set_aspect('equal', adjustable='datalim')







# Turn off empty axes if any

for i in range(n_pairs, n_rows*n_cols):

    axes.flat[i].axis('off')



plt.suptitle('Q–Q Plots (Filled): All Group Pairwise Comparisons\nRed: Group2>Group1 | Blue: Group2<Group1', fontsize=16, y=1.02)

plt.tight_layout(rect=[0, 0, 1, 0.97])

plt.show()



# %%
import numpy as np

import matplotlib.pyplot as plt

from itertools import combinations

import string



# Helper to convert group tuple to string

def group_to_str(group):

    if isinstance(group, tuple):

        return f"{group[0]}-{group[1]}"

    else:

        return str(group)



groups_list = list(group_speeds_dict.keys())

n_pairs = len(groups_list) * (len(groups_list) - 1) // 2

n_cols = 3

n_rows = int(np.ceil(n_pairs / n_cols))

n_points = 1000



# Get global min/max for all pooled arrays for consistent axis limits

all_vals = np.concatenate([v for v in group_speeds_dict.values() if len(v) > 0])

global_min = float(np.nanmin(all_vals))

global_max = float(np.nanmax(all_vals))



fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5*n_cols, 5.5*n_rows), squeeze=False)

legend_handles = []

labels_added = set()



for panel_idx, (ax, (g1, g2)) in enumerate(zip(axes.flat, combinations(groups_list, 2))):

    arr1 = group_speeds_dict[g1]

    arr2 = group_speeds_dict[g2]

    if len(arr1) == 0 or len(arr2) == 0:

        ax.axis('off')

        continue

    q = np.linspace(0, 1, n_points)

    quant1 = np.quantile(arr1, q)

    quant2 = np.quantile(arr2, q)

    above = quant2 > quant1

    below = quant2 < quant1



    # Fill areas: red for above, blue for below

    h1 = ax.fill_between(quant1, quant1, quant2, where=above, color='firebrick', alpha=0.40, label='Group2 > Group1')

    h2 = ax.fill_between(quant1, quant1, quant2, where=below, color='dodgerblue', alpha=0.40, label='Group2 < Group1')

    # Save one handle for the legend, if not already added

    if not legend_handles:

        legend_handles = [h1, h2]



    # Q–Q curve and diagonal

    ax.plot(quant1, quant2, color='k', lw=2)

    ax.plot([global_min, global_max], [global_min, global_max], 'k--', lw=1.3)



    # Group labels and title

    lab1 = group_to_str(g1)

    lab2 = group_to_str(g2)

    ax.set_xlabel(f'Quantiles: {lab1}', fontsize=15, fontweight='bold')

    ax.set_ylabel(f'Quantiles: {lab2}', fontsize=15, fontweight='bold')

    ax.set_title(f"Q–Q: {lab2} vs {lab1}", fontsize=16, fontweight='bold', pad=12)

    # Uniform axis scaling

    ax.set_xlim(global_min, global_max)

    ax.set_ylim(global_min, global_max)

    # Tick parameters

    ax.tick_params(axis='both', labelsize=13, width=1.2)

    # Panel label (optional, but strong for journals)

    ax.text(0.03, 0.93, string.ascii_lowercase[panel_idx],

            fontsize=18, fontweight='bold', transform=ax.transAxes, va='top', ha='left')



# Hide unused axes

for ax in axes.flat[n_pairs:]:

    ax.axis('off')



# Shared legend outside plot (right)

fig.legend(legend_handles, ['Group2 > Group1', 'Group2 < Group1'],

           loc='center right', fontsize=14, frameon=True, borderaxespad=1.0)



plt.subplots_adjust(left=0.07, right=0.87, bottom=0.09, top=0.92, wspace=0.25, hspace=0.28)

fig.suptitle('Q–Q Plots (Filled): All Group Pairwise Comparisons\nRed = Group2 > Group1 | Blue = Group2 < Group1',

             fontsize=19, fontweight='bold', y=0.98)



plt.show()



# %%
import numpy as np

import matplotlib.pyplot as plt

from itertools import combinations

import string



# Helper: tuple to label

def group_to_str(group):

    if isinstance(group, tuple):

        return f"{group[0]}-{group[1]}"

    else:

        return str(group)



groups_list = list(group_speeds_dict.keys())

n_pairs = len(groups_list) * (len(groups_list) - 1) // 2

n_cols = 3

n_rows = int(np.ceil(n_pairs / n_cols))

n_points = 1000



# All values for axis limits

all_vals = np.concatenate([v for v in group_speeds_dict.values() if len(v) > 0])

global_min = float(np.nanmin(all_vals))

global_max = float(np.nanmax(all_vals))



fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.7*n_cols, 5.7*n_rows), squeeze=False)

legend_handles = []



for panel_idx, (ax, (g1, g2)) in enumerate(zip(axes.flat, combinations(groups_list, 2))):

    arr1 = group_speeds_dict[g1]

    arr2 = group_speeds_dict[g2]

    if len(arr1) == 0 or len(arr2) == 0:

        ax.axis('off')

        continue

    q = np.linspace(0, 1, n_points)

    quant1 = np.quantile(arr1, q)

    quant2 = np.quantile(arr2, q)

    above = quant2 > quant1

    below = quant2 < quant1



    # Fill areas

    h1 = ax.fill_between(quant1, quant1, quant2, where=above, color='firebrick', alpha=0.40, label='Group2 > Group1')

    h2 = ax.fill_between(quant1, quant1, quant2, where=below, color='dodgerblue', alpha=0.40, label='Group2 < Group1')

    if not legend_handles:

        legend_handles = [h1, h2]

    # Q-Q and diagonal

    ax.plot(quant1, quant2, color='k', lw=2)

    ax.plot([global_min, global_max], [global_min, global_max], 'k--', lw=1.3)



    # Labels and title

    lab1 = group_to_str(g1)

    lab2 = group_to_str(g2)

    ax.set_xlabel(f'Quantiles: {lab1}', fontsize=15, fontweight='bold')

    ax.set_ylabel(f'Quantiles: {lab2}', fontsize=15, fontweight='bold')

    ax.set_title(f"Quantile–Quantile: {lab2} vs {lab1}", fontsize=15, fontweight='bold', pad=13)

    # Axis scale

    ax.set_xlim(global_min, global_max)

    ax.set_ylim(global_min, global_max)

    ax.tick_params(axis='both', labelsize=15, width=1.2)

    # Panel label (a, b, c, ...)

    ax.text(-0.10, 1.05, string.ascii_lowercase[panel_idx],

            fontsize=19, fontweight='bold', transform=ax.transAxes, va='top', ha='left')

    # Optional faint grid

    ax.grid(True, which='both', linestyle=':', linewidth=0.8, alpha=0.15)



# Hide unused axes

for ax in axes.flat[n_pairs:]:

    ax.axis('off')



# Shared legend below all panels

fig.legend(legend_handles, ['Group2 > Group1', 'Group2 < Group1'],

           loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=2,

           fontsize=14, frameon=True, borderaxespad=1.0)



# Interpretation hint below legend

fig.text(0.5, -0.13,

         "Red fill: Group2 > Group1 (Q–Q curve above diagonal). Blue fill: Group2 < Group1.",

         ha='center', va='center', fontsize=13, color='dimgray')



# Supertitle above panels

fig.suptitle('Q–Q Plots (Filled): All Group Pairwise Comparisons',

             fontsize=18, fontweight='semibold', y=1.03)



plt.subplots_adjust(left=0.09, right=0.96, bottom=0.13, top=0.94, wspace=0.25, hspace=0.28)

plt.show()



# %%
# %%



import numpy as np

import matplotlib.pyplot as plt

from itertools import combinations

import string



# Helper: tuple to label

def group_to_str(group):

    if isinstance(group, tuple):

        return f"{group[0]}-{group[1]}"

    else:

        return str(group)



groups_list = list(group_speeds_dict.keys())

n_pairs = len(groups_list) * (len(groups_list) - 1) // 2

n_cols = 3

n_rows = int(np.ceil(n_pairs / n_cols))

n_points = 1000



# All values for axis limits

all_vals = np.concatenate([v for v in group_speeds_dict.values() if len(v) > 0])

global_min = float(np.nanmin(all_vals))

global_max = float(np.nanmax(all_vals))



fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.7*n_cols, 5.7*n_rows), squeeze=False)

legend_handles = []



for panel_idx, (ax, (g1, g2)) in enumerate(zip(axes.flat, combinations(groups_list, 2))):

    arr1 = group_speeds_dict[g1]

    arr2 = group_speeds_dict[g2]

    if len(arr1) == 0 or len(arr2) == 0:

        ax.axis('off')

        continue

    q = np.linspace(0, 1, n_points)

    quant1 = np.quantile(arr1, q)

    quant2 = np.quantile(arr2, q)

    above = quant2 > quant1

    below = quant2 < quant1



    # Fill areas

    h1 = ax.fill_between(quant1, quant1, quant2, where=above, color='firebrick', alpha=0.40, label='Group2 > Group1')

    h2 = ax.fill_between(quant1, quant1, quant2, where=below, color='dodgerblue', alpha=0.40, label='Group2 < Group1')

    if not legend_handles:

        legend_handles = [h1, h2]

    # Q-Q and diagonal

    ax.plot(quant1, quant2, color='k', lw=2)

    ax.plot([global_min, global_max], [global_min, global_max], 'k--', lw=1.3)



    # Labels and title

    lab1 = group_to_str(g1)

    lab2 = group_to_str(g2)

    ax.set_xlabel(f'Quantiles: {lab1}', fontsize=15, fontweight='bold')

    ax.set_ylabel(f'Quantiles: {lab2}', fontsize=15, fontweight='bold')

    ax.set_title(f"Quantile–Quantile: {lab2} vs {lab1}", fontsize=15, fontweight='bold', pad=13)

    # Axis scale

    ax.set_xlim(global_min, global_max)

    ax.set_ylim(global_min, global_max)

    ax.tick_params(axis='both', labelsize=15, width=1.2)

    # Panel label (a, b, c, ...)

    ax.text(-0.10, 1.05, string.ascii_lowercase[panel_idx],

            fontsize=19, fontweight='bold', transform=ax.transAxes, va='top', ha='left')

    # Optional faint grid

    ax.grid(True, which='both', linestyle=':', linewidth=0.8, alpha=0.15)



# Hide unused axes

for ax in axes.flat[n_pairs:]:

    ax.axis('off')



# Shared legend below all panels

fig.legend(legend_handles, ['Group2 > Group1', 'Group2 < Group1'],

           loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=2,

           fontsize=14, frameon=True, borderaxespad=1.0)



# Interpretation hint below legend

fig.text(0.5, -0.13,

         "Red fill: Group2 > Group1 (Q–Q curve above diagonal). Blue fill: Group2 < Group1.",

         ha='center', va='center', fontsize=13, color='dimgray')



# Supertitle above panels

fig.suptitle('Q–Q Plots (Filled): All Group Pairwise Comparisons',

             fontsize=18, fontweight='semibold', y=1.03)



plt.subplots_adjust(left=0.09, right=0.96, bottom=0.13, top=0.94, wspace=0.25, hspace=0.28)

plt.show()
# %%
# %%



import numpy as np

import matplotlib.pyplot as plt

from itertools import combinations

import string



# Helper: tuple to label

def group_to_str(group):

    if isinstance(group, tuple):

        return f"{group[0]}-{group[1]}"

    else:

        return str(group)



groups_list = list(group_speeds_dict.keys())

n_pairs = len(groups_list) * (len(groups_list) - 1) // 2

n_cols = 3

n_rows = int(np.ceil(n_pairs / n_cols))

n_points = 1000



# All values for axis limits

all_vals = np.concatenate([v for v in group_speeds_dict.values() if len(v) > 0])

global_min = float(np.nanmin(all_vals))

global_max = float(np.nanmax(all_vals))



fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.7*n_cols, 5.7*n_rows), squeeze=False)

legend_handles = []



for panel_idx, (ax, (g1, g2)) in enumerate(zip(axes.flat, combinations(groups_list, 2))):

    arr1 = group_speeds_dict[g1]

    arr2 = group_speeds_dict[g2]

    if len(arr1) == 0 or len(arr2) == 0:

        ax.axis('off')

        continue

    q = np.linspace(0, 1, n_points)

    quant1 = np.quantile(arr1, q)

    quant2 = np.quantile(arr2, q)

    above = quant2 > quant1

    below = quant2 < quant1



    # Fill areas

    h1 = ax.fill_between(quant1, quant1, quant2, where=above, color='firebrick', alpha=0.40, label='Group2 > Group1')

    h2 = ax.fill_between(quant1, quant1, quant2, where=below, color='dodgerblue', alpha=0.40, label='Group2 < Group1')

    if not legend_handles:

        legend_handles = [h1, h2]

    # Q-Q and diagonal

    ax.plot(quant1, quant2, color='k', lw=2)

    ax.plot([global_min, global_max], [global_min, global_max], 'k--', lw=1.3)



    # Labels and title

    lab1 = group_to_str(g1)

    lab2 = group_to_str(g2)

    ax.set_xlabel(f'Quantiles: {lab1}', fontsize=15, fontweight='bold')

    ax.set_ylabel(f'Quantiles: {lab2}', fontsize=15, fontweight='bold')

    ax.set_title(f"Q-Q: {lab2} vs {lab1}", fontsize=15, fontweight='bold', pad=13)

    # Axis scale

    ax.set_xlim(global_min, global_max)

    ax.set_ylim(global_min, global_max)

    ax.tick_params(axis='both', labelsize=15, width=1.2)

    # Panel label (a, b, c, ...)

    ax.text(-0.10, 1.05, string.ascii_lowercase[panel_idx],

            fontsize=19, fontweight='bold', transform=ax.transAxes, va='top', ha='left')

    # Optional faint grid

    ax.grid(True, which='both', linestyle=':', linewidth=0.8, alpha=0.15)



# Hide unused axes

for ax in axes.flat[n_pairs:]:

    ax.axis('off')



# Shared legend below all panels

fig.legend(legend_handles, ['Group2 > Group1', 'Group2 < Group1'],

           loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=2,

           fontsize=14, frameon=True, borderaxespad=1.0)



# Interpretation hint below legend

fig.text(0.5, -0.13,

         "Red fill: Group2 > Group1 (Q–Q curve above diagonal). Blue fill: Group2 < Group1.",

         ha='center', va='center', fontsize=13, color='dimgray')



# Supertitle above panels

fig.suptitle('Q–Q Plots (Filled): All Group Pairwise Comparisons',

             fontsize=18, fontweight='semibold', y=1.03)



plt.subplots_adjust(left=0.09, right=0.96, bottom=0.13, top=0.94, wspace=0.25, hspace=0.28)

plt.show()
# %%
#%%



# Example: If your long pool indices are known

long_win_indices = np.arange(len(window_sizes)//2, len(window_sizes))



group_speeds_dict_long = {}



for group, animal_idxs in groups.items():

    pooled_speeds = []

    for animal_idx in animal_idxs:

        # Pool all long-window speeds for this animal (over all taus)

        for win_idx in long_win_indices:

            win_arr = all_speed[win_idx]  # (n_animals, n_tau, n_timepoints)

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :].astype(float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled_speeds.append(arr)

    if pooled_speeds:

        group_speeds_dict_long[group] = np.concatenate(pooled_speeds)

    else:

        group_speeds_dict_long[group] = np.array([])
# %%
#%%



# Build the per-group pooled speed dictionary for the long windows

long_win_indices = np.arange(len(window_sizes)//2, len(window_sizes))



group_speeds_dict_long = {}

for group, animal_idxs in groups.items():

    pooled_speeds = []

    for animal_idx in animal_idxs:

        for win_idx in long_win_indices:

            win_arr = all_speed[win_idx]

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :].astype(float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled_speeds.append(arr)

    if pooled_speeds:

        group_speeds_dict_long[group] = np.concatenate(pooled_speeds)

    else:

        group_speeds_dict_long[group] = np.array([])



# Now re-use the Q–Q grid code (with supertitle tweak)

import numpy as np

import matplotlib.pyplot as plt

from itertools import combinations

import string



def group_to_str(group):

    if isinstance(group, tuple):

        return f"{group[0]}-{group[1]}"

    else:

        return str(group)



groups_list = list(group_speeds_dict_long.keys())

n_pairs = len(groups_list) * (len(groups_list) - 1) // 2

n_cols = 3

n_rows = int(np.ceil(n_pairs / n_cols))

n_points = 1000



all_vals = np.concatenate([v for v in group_speeds_dict_long.values() if len(v) > 0])

global_min = float(np.nanmin(all_vals))

global_max = float(np.nanmax(all_vals))



fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.7*n_cols, 5.7*n_rows), squeeze=False)

legend_handles = []



for panel_idx, (ax, (g1, g2)) in enumerate(zip(axes.flat, combinations(groups_list, 2))):

    arr1 = group_speeds_dict_long[g1]

    arr2 = group_speeds_dict_long[g2]

    if len(arr1) == 0 or len(arr2) == 0:

        ax.axis('off')

        continue

    q = np.linspace(0, 1, n_points)

    quant1 = np.quantile(arr1, q)

    quant2 = np.quantile(arr2, q)

    above = quant2 > quant1

    below = quant2 < quant1



    h1 = ax.fill_between(quant1, quant1, quant2, where=above, color='firebrick', alpha=0.40, label='Group2 > Group1')

    h2 = ax.fill_between(quant1, quant1, quant2, where=below, color='dodgerblue', alpha=0.40, label='Group2 < Group1')

    if not legend_handles:

        legend_handles = [h1, h2]



    ax.plot(quant1, quant2, color='k', lw=2)

    ax.plot([global_min, global_max], [global_min, global_max], 'k--', lw=1.3)



    lab1 = group_to_str(g1)

    lab2 = group_to_str(g2)

    ax.set_xlabel(f'Quantiles: {lab1}', fontsize=15, fontweight='bold')

    ax.set_ylabel(f'Quantiles: {lab2}', fontsize=15, fontweight='bold')

    ax.set_title(f"Quantile–Quantile: {lab2} vs {lab1}", fontsize=15, fontweight='bold', pad=13)

    ax.set_xlim(global_min, global_max)

    ax.set_ylim(global_min, global_max)

    ax.tick_params(axis='both', labelsize=15, width=1.2)

    ax.text(-0.10, 1.05, string.ascii_lowercase[panel_idx],

            fontsize=19, fontweight='bold', transform=ax.transAxes, va='top', ha='left')

    ax.grid(True, which='both', linestyle=':', linewidth=0.8, alpha=0.15)



for ax in axes.flat[n_pairs:]:

    ax.axis('off')



fig.legend(legend_handles, ['Group2 > Group1', 'Group2 < Group1'],

           loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=2,

           fontsize=14, frameon=True, borderaxespad=1.0)



fig.text(0.5, -0.13,

         "Red fill: Group2 > Group1 (Q–Q curve above diagonal). Blue fill: Group2 < Group1.",

         ha='center', va='center', fontsize=13, color='dimgray')



fig.suptitle('Q–Q Plots (Filled): Long Window Pool, All Group Pairwise Comparisons',

             fontsize=18, fontweight='semibold', y=1.03)



plt.subplots_adjust(left=0.09, right=0.96, bottom=0.13, top=0.94, wspace=0.25, hspace=0.28)

plt.show()
# %%
#%%



# Build the per-group pooled speed dictionary for the long windows

long_win_indices = np.arange(len(window_sizes)//2, len(window_sizes))



group_speeds_dict_long = {}

for group, animal_idxs in groups.items():

    pooled_speeds = []

    for animal_idx in animal_idxs:

        for win_idx in long_win_indices:

            win_arr = all_speed[win_idx]

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :].astype(float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled_speeds.append(arr)

    if pooled_speeds:

        group_speeds_dict_long[group] = np.concatenate(pooled_speeds)

    else:

        group_speeds_dict_long[group] = np.array([])



# Now re-use the Q–Q grid code (with supertitle tweak)

import numpy as np

import matplotlib.pyplot as plt

from itertools import combinations

import string



def group_to_str(group):

    if isinstance(group, tuple):

        return f"{group[0]}-{group[1]}"

    else:

        return str(group)



groups_list = list(group_speeds_dict_long.keys())

n_pairs = len(groups_list) * (len(groups_list) - 1) // 2

n_cols = 3

n_rows = int(np.ceil(n_pairs / n_cols))

n_points = 1000



all_vals = np.concatenate([v for v in group_speeds_dict_long.values() if len(v) > 0])

global_min = float(np.nanmin(all_vals))

global_max = float(np.nanmax(all_vals))



fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.7*n_cols, 5.7*n_rows), squeeze=False)

legend_handles = []



for panel_idx, (ax, (g1, g2)) in enumerate(zip(axes.flat, combinations(groups_list, 2))):

    arr1 = group_speeds_dict_long[g1]

    arr2 = group_speeds_dict_long[g2]

    if len(arr1) == 0 or len(arr2) == 0:

        ax.axis('off')

        continue

    q = np.linspace(0, 1, n_points)

    quant1 = np.quantile(arr1, q)

    quant2 = np.quantile(arr2, q)

    above = quant2 > quant1

    below = quant2 < quant1



    h1 = ax.fill_between(quant1, quant1, quant2, where=above, color='firebrick', alpha=0.40, label='Group2 > Group1')

    h2 = ax.fill_between(quant1, quant1, quant2, where=below, color='dodgerblue', alpha=0.40, label='Group2 < Group1')

    if not legend_handles:

        legend_handles = [h1, h2]



    ax.plot(quant1, quant2, color='k', lw=2)

    ax.plot([global_min, global_max], [global_min, global_max], 'k--', lw=1.3)



    lab1 = group_to_str(g1)

    lab2 = group_to_str(g2)

    ax.set_xlabel(f'Quantiles: {lab1}', fontsize=15, fontweight='bold')

    ax.set_ylabel(f'Quantiles: {lab2}', fontsize=15, fontweight='bold')

    ax.set_title(f"Quantile–Quantile: {lab2} vs {lab1}", fontsize=15, fontweight='bold', pad=13)

    ax.set_xlim(global_min, global_max)

    ax.set_ylim(global_min, global_max)

    ax.tick_params(axis='both', labelsize=15, width=1.2)

    ax.text(-0.10, 1.05, string.ascii_lowercase[panel_idx],

            fontsize=19, fontweight='bold', transform=ax.transAxes, va='top', ha='left')

    ax.grid(True, which='both', linestyle=':', linewidth=0.8, alpha=0.15)



for ax in axes.flat[n_pairs:]:

    ax.axis('off')



fig.legend(legend_handles, ['Group2 > Group1', 'Group2 < Group1'],

           loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=2,

           fontsize=14, frameon=True, borderaxespad=1.0)



fig.text(0.5, -0.13,

         "Red fill: Group2 > Group1 (Q–Q curve above diagonal). Blue fill: Group2 < Group1.",

         ha='center', va='center', fontsize=13, color='dimgray')



fig.suptitle('Q–Q Plots (Filled): Long Window Pool, All Group Pairwise Comparisons',

             fontsize=18, fontweight='semibold', y=1.03)



plt.subplots_adjust(left=0.09, right=0.96, bottom=0.13, top=0.94, wspace=0.25, hspace=0.28)

plt.show()
# %%
#%%



# Build the per-group pooled speed dictionary for the long windows

long_win_indices = np.arange(len(window_sizes)//2, len(window_sizes))



group_speeds_dict_long = {}

for group, animal_idxs in data.groups.items():

    pooled_speeds = []

    for animal_idx in animal_idxs:

        for win_idx in long_win_indices:

            win_arr = all_speed[win_idx]

            for tau in range(win_arr.shape[1]):

                arr = win_arr[animal_idx, tau, :].astype(float)

                arr = arr[~np.isnan(arr)]

                if arr.size > 0:

                    pooled_speeds.append(arr)

    if pooled_speeds:

        group_speeds_dict_long[group] = np.concatenate(pooled_speeds)

    else:

        group_speeds_dict_long[group] = np.array([])



# Now re-use the Q–Q grid code (with supertitle tweak)

import numpy as np

import matplotlib.pyplot as plt

from itertools import combinations

import string



def group_to_str(group):

    if isinstance(group, tuple):

        return f"{group[0]}-{group[1]}"

    else:

        return str(group)



groups_list = list(group_speeds_dict_long.keys())

n_pairs = len(groups_list) * (len(groups_list) - 1) // 2

n_cols = 3

n_rows = int(np.ceil(n_pairs / n_cols))

n_points = 1000



all_vals = np.concatenate([v for v in group_speeds_dict_long.values() if len(v) > 0])

global_min = float(np.nanmin(all_vals))

global_max = float(np.nanmax(all_vals))



fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.7*n_cols, 5.7*n_rows), squeeze=False)

legend_handles = []



for panel_idx, (ax, (g1, g2)) in enumerate(zip(axes.flat, combinations(groups_list, 2))):

    arr1 = group_speeds_dict_long[g1]

    arr2 = group_speeds_dict_long[g2]

    if len(arr1) == 0 or len(arr2) == 0:

        ax.axis('off')

        continue

    q = np.linspace(0, 1, n_points)

    quant1 = np.quantile(arr1, q)

    quant2 = np.quantile(arr2, q)

    above = quant2 > quant1

    below = quant2 < quant1



    h1 = ax.fill_between(quant1, quant1, quant2, where=above, color='firebrick', alpha=0.40, label='Group2 > Group1')

    h2 = ax.fill_between(quant1, quant1, quant2, where=below, color='dodgerblue', alpha=0.40, label='Group2 < Group1')

    if not legend_handles:

        legend_handles = [h1, h2]



    ax.plot(quant1, quant2, color='k', lw=2)

    ax.plot([global_min, global_max], [global_min, global_max], 'k--', lw=1.3)



    lab1 = group_to_str(g1)

    lab2 = group_to_str(g2)

    ax.set_xlabel(f'Quantiles: {lab1}', fontsize=15, fontweight='bold')

    ax.set_ylabel(f'Quantiles: {lab2}', fontsize=15, fontweight='bold')

    ax.set_title(f"Quantile–Quantile: {lab2} vs {lab1}", fontsize=15, fontweight='bold', pad=13)

    ax.set_xlim(global_min, global_max)

    ax.set_ylim(global_min, global_max)

    ax.tick_params(axis='both', labelsize=15, width=1.2)

    ax.text(-0.10, 1.05, string.ascii_lowercase[panel_idx],

            fontsize=19, fontweight='bold', transform=ax.transAxes, va='top', ha='left')

    ax.grid(True, which='both', linestyle=':', linewidth=0.8, alpha=0.15)



for ax in axes.flat[n_pairs:]:

    ax.axis('off')



fig.legend(legend_handles, ['Group2 > Group1', 'Group2 < Group1'],

           loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=2,

           fontsize=14, frameon=True, borderaxespad=1.0)



fig.text(0.5, -0.13,

         "Red fill: Group2 > Group1 (Q–Q curve above diagonal). Blue fill: Group2 < Group1.",

         ha='center', va='center', fontsize=13, color='dimgray')



fig.suptitle('Q–Q Plots (Filled): Long Window Pool, All Group Pairwise Comparisons',

             fontsize=18, fontweight='semibold', y=1.03)



plt.subplots_adjust(left=0.09, right=0.96, bottom=0.13, top=0.94, wspace=0.25, hspace=0.28)

plt.show()
# %%
# %%



import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



# Assume: groups = df_summary.groupby(['genotype', 'treatment']).groups

#         window_sizes, all_speed, etc. already defined



short_idx = np.arange(len(window_sizes) // 2)

long_idx = np.arange(len(window_sizes) // 2, len(window_sizes))

pool_defs = {'Short windows': short_idx, 'Long windows': long_idx}

palette = sns.color_palette('tab10', n_colors=len(groups))



plt.figure(figsize=(12, 6))



for pool_i, (pool_name, idxs) in enumerate(pool_defs.items()):

    plt.subplot(1, 2, pool_i+1)

    for g_idx, (group, animal_idxs) in enumerate(data.groups.items()):





        # Pool all speeds for this group and this pool

        group_speeds = []

        for win_idx in idxs:

            win_arr = all_speed[win_idx]  # (n_animals, n_tau, n_timepoints)

            for animal_idx in animal_idxs:

                for tau in range(win_arr.shape[1]):

                    arr = win_arr[animal_idx, tau, :].astype(float)

                    arr = arr[~np.isnan(arr)]

                    if arr.size > 0:

                        group_speeds.append(arr)

        group_speeds = np.concatenate(group_speeds) if group_speeds else np.array([])

        # Histogram (step)

        sns.histplot(group_speeds, bins=60, stat='density', element='step', fill=False,

                     color=palette[g_idx], linewidth=1.6, label=f'{group}', alpha=0.6)

        # KDE (over histogram)

        if group_speeds.size > 10:  # Avoid noise for tiny samples

            sns.kdeplot(group_speeds, color=palette[g_idx], lw=2.1, label=None)

        # Median

        plt.axvline(np.median(group_speeds), color=palette[g_idx], linestyle='--', lw=1)

    plt.xlabel("dFC Speed")

    plt.ylabel("Density")

    plt.title(f"{pool_name}")

    if pool_i == 0:

        plt.legend(title='Group', fontsize=10)

    else:

        plt.legend().set_visible(False)

    plt.tight_layout()



plt.suptitle("Distribution of dFC Speeds by Group\nShort vs. Long Window Pools", fontsize=15, y=1.02)

plt.tight_layout()

plt.show()
