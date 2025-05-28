# merge_allegiance.py

#%%
import numpy as np
from pathlib import Path
from shared_code.fun_utils import get_paths
from shared_code.fun_metaconnectivity import load_merged_allegiance
from tqdm import tqdm
#%%
def merge_allegiance(window_size=9, lag=1, timecourse_folder='Timecourses_updated_03052024'):
    # Get paths
    paths = get_paths(timecourse_folder=timecourse_folder)
    ts = np.load(paths['sorted'] / 'ts_and_meta_2m4m.npz', allow_pickle=True)['ts']
    n_animals = len(ts)
    n_regions = ts[0].shape[1]

    # Get number of windows from a known DFC file
    filename_dfc = f'window_size={window_size}_lag={lag}_animals={n_animals}_regions={n_regions}'
    dfc_data = np.load(paths['mc'] / f'dfc_{filename_dfc}.npz')
    n_windows = np.transpose(dfc_data['dfc_stream'], (0, 3, 2, 1)).shape[1]

    arr_shape = (n_regions, n_regions)

    # Preallocate arrays with NaN
    dfc_communities = np.full((n_animals, n_windows, n_regions), np.nan)
    sort_allegiances = np.full((n_animals, n_windows, n_regions), np.nan)
    contingency_matrices = np.full((n_animals, n_windows, *arr_shape), np.nan)

    # Load data if present
    out_dir = paths['allegiance'] / 'temp'
    missing_count = 0

    for a in tqdm(range(n_animals), desc="Animals"):
        for w in range(n_windows):
            out_file = out_dir / f"{filename_dfc}_animal_{a:02d}_window_{w:04d}.npz"
            if out_file.exists():
                data = np.load(out_file)
                dfc_communities[a, w] = data["dfc_communities"]
                sort_allegiances[a, w] = data["sort_allegiance"]
                contingency_matrices[a, w] = data["contingency_matrix"]
            else:
                missing_count += 1
                print(f"[MISSING] Animal {a}, Window {w}")

    # Save merged result
    merged_out_file = paths['allegiance'] / f'merged_allegiance_{filename_dfc}.npz'
    np.savez_compressed(
        merged_out_file,
        dfc_communities=dfc_communities,
        sort_allegiances=sort_allegiances,
        contingency_matrices=contingency_matrices
    )

    print(f"[DONE] Merged data saved to: {merged_out_file}")
    print(f"[INFO] Missing entries: {missing_count} out of {n_animals * n_windows}")




if __name__ == "__main__":
    paths = get_paths(timecourse_folder='Timecourses_updated_03052024')
    merge_allegiance(window_size=9, lag=1)
    dfc_communities, sort_allegiances, contingency_matrices = load_merged_allegiance(paths, window_size=9, lag=1)

# %%
