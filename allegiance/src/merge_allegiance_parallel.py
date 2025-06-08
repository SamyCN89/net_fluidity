# merge_allegiance.py

#%%
import numpy as np
from pathlib import Path
from shared_code.fun_paths import get_paths
from shared_code.fun_metaconnectivity import load_merged_allegiance
from tqdm import tqdm
#%%
def merge_allegiance(window_size=9, lag=1, timecourse_folder='Timecourses_updated_03052024'):
    # Get paths
    paths = get_paths(dataset_name='ines_abdullah', 
                      timecourse_folder='Timecourses_updated_03052024',
                      cognitive_data_file='ROIs.xlsx')
    ts = np.load(paths['preprocessed'] / 'ts_and_meta_2m4m.npz', allow_pickle=True)['ts']
    n_animals = len(ts)
    n_regions = ts[0].shape[1]

    # Get number of windows from a known DFC file
    filename_dfc = f'window_size={window_size}_lag={lag}_animals={n_animals}_regions={n_regions}'
    print(f"[INFO] Merging allegiance data for {filename_dfc}...")

    # Load DFC data to determine number of windows
    dfc_data = np.load(paths['dfc'] / f'dfc_{filename_dfc}.npz')
    n_windows = np.transpose(dfc_data['dfc_stream'], (0, 3, 2, 1)).shape[-1]
    arr_shape = (n_regions, n_regions)

    # Preallocate arrays with NaN
    dfc_communities = np.full((n_animals, n_windows, n_regions), np.nan)
    sort_allegiances = np.full((n_animals, n_windows, n_regions), np.nan)
    contingency_matrices = np.full((n_animals, n_windows, *arr_shape), np.nan)

    # Load data if present
    out_dir = paths['allegiance'] / 'temp'
    missing_count = 0

    # Iterate through animals and windows to load allegiance data
    for ani in tqdm(range(n_animals), desc="Animals"):
        for ws in range(n_windows):
            out_file = out_dir / f"{filename_dfc}_animal_{ani:02d}_window_{ws:04d}.npz"
            print(f"[INFO] Processing Animal {ani}, Window {ws} - File: {out_file}")
            if out_file.exists():
                data = np.load(out_file)
                dfc_communities[ani, ws] = data["dfc_communities"]
                sort_allegiances[ani, ws] = data["sort_allegiance"]
                print(data['sort_allegiance'].shape)
                contingency_matrices[ani, ws] = data["contingency_matrix"]
            else:
                missing_count += 1
                print(f"[MISSING] Animal {ani}, Window {ws} - File not found: {out_file}")

    # Filepath for merged file
    merged_out_file = paths['allegiance'] / f'merged_allegiance_{filename_dfc}.npz'
    
    # Save merged result
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
