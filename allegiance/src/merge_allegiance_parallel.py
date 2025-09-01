import os
import logging
import numpy as np
from pathlib import Path
from shared_code.fun_paths import get_paths
from shared_code.fun_metaconnectivity import load_merged_allegiance
from tqdm import tqdm


def setup_logging():
    """Configure logging from YAML if available; fallback to basicConfig."""
    cfg_path = os.getenv("NET_FLUIDITY_LOGGING", "config/logging.yaml")
    try:
        if os.path.exists(cfg_path):
            import yaml
            from logging.config import dictConfig

            with open(cfg_path, "r") as f:
                dictConfig(yaml.safe_load(f))
            return
    except Exception:
        # fall back to basic config
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


setup_logging()
logger = logging.getLogger(__name__)


# %%
def merge_allegiance(
    window_size=9, lag=1, timecourse_folder="Timecourses_updated_03052024"
):
    # Get paths
    paths = get_paths(
        dataset_name="ines_abdullah",
        timecourse_folder="Timecourses_updated_03052024",
        cognitive_data_file="ROIs.xlsx",
    )
    ts = np.load(paths["preprocessed"] / "ts_and_meta_2m4m.npz", allow_pickle=True)[
        "ts"
    ]
    n_animals = len(ts)
    n_regions = ts[0].shape[1]

    # Get number of windows from a known DFC file
    filename_dfc = (
        f"window_size={window_size}_lag={lag}_animals={n_animals}_regions={n_regions}"
    )
    logger.info("Merging allegiance data for %s", filename_dfc)

    # Load DFC data to determine number of windows
    dfc_data = np.load(paths["dfc"] / f"dfc_{filename_dfc}.npz")
    n_windows = np.transpose(dfc_data["dfc_stream"], (0, 3, 2, 1)).shape[-1]
    arr_shape = (n_regions, n_regions)

    # Preallocate arrays with NaN
    dfc_communities = np.full((n_animals, n_windows, n_regions), np.nan)
    sort_allegiances = np.full((n_animals, n_windows, n_regions), np.nan)
    contingency_matrices = np.full((n_animals, n_windows, *arr_shape), np.nan)

    # Load data if present
    out_dir = paths["allegiance"] / "temp"
    missing_count = 0

    # Iterate through animals and windows to load allegiance data
    for ani in tqdm(range(n_animals), desc="Animals"):
        for ws in range(n_windows):
            out_file = out_dir / f"{filename_dfc}_animal_{ani:02d}_window_{ws:04d}.npz"
            logger.debug(
                "Processing Animal %s, Window %s - File: %s", ani, ws, out_file
            )
            if out_file.exists():
                data = np.load(out_file)
                dfc_communities[ani, ws] = data["dfc_communities"]
                sort_allegiances[ani, ws] = data["sort_allegiance"]
                logger.debug("sort_allegiance shape: %s", data["sort_allegiance"].shape)
                contingency_matrices[ani, ws] = data["contingency_matrix"]
            else:
                missing_count += 1
                logger.warning(
                    "Missing allegiance file for animal %s, window %s: %s",
                    ani,
                    ws,
                    out_file,
                )

    # Filepath for merged file
    merged_out_file = paths["allegiance"] / f"merged_allegiance_{filename_dfc}.npz"

    # Save merged result
    np.savez_compressed(
        merged_out_file,
        dfc_communities=dfc_communities,
        sort_allegiances=sort_allegiances,
        contingency_matrices=contingency_matrices,
    )

    logger.info("Merged data saved to: %s", merged_out_file)
    logger.info("Missing entries: %s of %s", missing_count, n_animals * n_windows)


if __name__ == "__main__":
    paths = get_paths(timecourse_folder="Timecourses_updated_03052024")
    merge_allegiance(window_size=9, lag=1)
    dfc_communities, sort_allegiances, contingency_matrices = load_merged_allegiance(
        paths, window_size=9, lag=1
    )

# %%
