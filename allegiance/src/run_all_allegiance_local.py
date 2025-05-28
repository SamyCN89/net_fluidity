import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from shared_code.fun_utils import get_paths, load_timeseries_data
from shared_code.fun_metaconnectivity import fun_allegiance_communities
import os
import logging

import argparse

# ===================== CLI ARGUMENTS ==========================
parser = argparse.ArgumentParser()
parser.add_argument("--n_jobs", type=int, default=4,
                    help="Number of parallel jobs to run (default: 4)")
parser.add_argument("--data_root", type=str, default=None,
                    help="Override PROJECT_DATA_ROOT (default: env variable)")
parser.add_argument("--window_size", type=int, default=9, help="DFC window size (default: 9)")
parser.add_argument("--lag", type=int, default=1, help="Lag between windows (default: 1)")
parser.add_argument("--timecourse_folder", type=str, default="Timecourses_updated_03052024",
                    help="Folder with timecourses (default: Timecourses_updated_03052024)")
parser.add_argument("--n_runs", type=int, default=1000, help="Number of modularity runs (default: 1000)")
parser.add_argument("--gamma", type=float, default=100, help="Gamma parameter for modularity (default: 100)")
args = parser.parse_args()


# ===================== CONFIG ================================
# Set PROJECT_DATA_ROOT from argument if given
if args.data_root:
    os.environ["PROJECT_DATA_ROOT"] = args.data_root

processors = args.n_jobs

window_size = args.window_size
lag = args.lag
timecourse_folder = args.timecourse_folder
n_runs = args.n_runs
gamma = args.gamma


# --- Setup logging ---
log_file = Path("allegiance_errors.log")
logging.basicConfig(
    filename=log_file,
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Config ---
n_runs = 1000
gamma_pt = 100
window_size = 9
lag = 1
# processors = 8  # Adjust to your CPU
timecourse_folder = 'Timecourses_updated_03052024'

# --- Load data ---
paths = get_paths(timecourse_folder=timecourse_folder)
data_ts = load_timeseries_data(paths['sorted'] / 'ts_and_meta_2m4m.npz')
ts = data_ts['ts']
n_animals = len(ts)
n_regions = ts[0].shape[1]

filename_dfc = f'window_size={window_size}_lag={lag}_animals={n_animals}_regions={n_regions}'
dfc_data = np.load(paths['mc'] / f'dfc_{filename_dfc}.npz')
dfc_stream = np.transpose(dfc_data['dfc_stream'], (0, 3, 2, 1))  # (n_animals, n_windows, n_regions, n_regions)
n_windows = dfc_stream.shape[1]

out_dir = paths['allegiance'] / 'temp'
out_dir.mkdir(parents=True, exist_ok=True)

# --- Function to run one job ---
def run_one_job(animal_idx, window_idx):
    out_file = out_dir / f"{filename_dfc}_animal_{animal_idx:02d}_window_{window_idx:04d}.npz"
    if out_file.exists():
        print(f"[SKIP] Animal {animal_idx}, Window {window_idx}")
        return

    dfc = dfc_stream[animal_idx, window_idx]
    try:
        dfc_com, sort_all, cont_mat = fun_allegiance_communities(
            dfc,
            n_runs=n_runs,
            gamma_pt=gamma_pt,
            save_path=None,
            ref_name=f'animal_{animal_idx:02d}_window_{window_idx:04d}',
            n_jobs=1  # Only use 1 core per job to avoid CPU overload
        )

        np.savez_compressed(out_file,
                            dfc_communities=dfc_com,
                            sort_allegiance=sort_all,
                            contingency_matrix=cont_mat)
        print(f"[DONE] Animal {animal_idx}, Window {window_idx}")
    except Exception as e:
        # print(f"[ERROR] Animal {animal_idx}, Window {window_idx}: {e}")
        msg = f"[ERROR] Animal {animal_idx}, Window {window_idx}: {e}"
        print(msg)
        logging.error(msg, exc_info=True)

# --- Run in parallel ---
Parallel(n_jobs=processors, backend="loky")(
    delayed(run_one_job)(a, w)
    for a in range(n_animals)
    for w in range(n_windows)
)
print("All jobs completed.")
# --- Merge results ---
# This part is not included in the original code. You can implement it based on your needs.
# For example, you might want to merge all the .npz files into a single file or process them further.
# Note: The merging part is not implemented in this code snippet. You can add it as needed.
# --- End of script ---