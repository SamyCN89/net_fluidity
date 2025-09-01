"""
Path helpers for datasets, results, and figures.

Environment variables (can be placed in a local .env):
- `PROJECT_ROOT_<ENV>`: absolute path to project root for a given environment (default ENV='LOCAL')
- `DATASET_NAME` (optional): dataset subfolder name used by `get_paths`

Example:
    PROJECT_ROOT_LOCAL=/abs/path/to/root
    DATASET_NAME=ines_abdullah
"""

from pathlib import Path
import os
from dotenv import load_dotenv
from typing import Dict, Optional

# Load environment variables from ../../.env if present
load_dotenv()


# =============================================================================
# Get Paths folder
# =============================================================================
def get_root_path(env="LOCAL") -> Path:
    """
    Resolve the project root for a given environment label.

    Parameters:
    - env: str (default 'LOCAL')
      Suffix used to read `PROJECT_ROOT_<env>` from the environment.

    Returns:
    - Path: absolute path to the configured project root.

    Raises:
    - EnvironmentError: if the corresponding environment variable is not set.
    """
    root = os.getenv(f"PROJECT_ROOT_{env}")
    if not root:
        raise EnvironmentError(f"Environment variable PROJECT_ROOT_{env} is not set.")
    return Path(root)


# =============================================================================
def build_paths(
    root: Path,
    dataset_name: str,
    timecourse_folder: str,
    cognitive_data_file: str,
    anat_labels_file: str,
) -> Dict[str, Path]:
    """
    Build canonical dataset/results/figures subpaths under a given root.

    Returns a dictionary with keys like `timeseries`, `results`, `dfc`, `speed`, and `figures`.
    """

    # Define paths based on dataset_name
    dataset = root / "dataset" / dataset_name
    results = root / "results" / dataset_name
    figures = root / "fig" / dataset_name

    return {
        "root": root,
        # Load raw dataset paths
        "timeseries": dataset / timecourse_folder,
        "cog_data": dataset / "cog_data" / cognitive_data_file,
        "labels": dataset / "cog_data" / anat_labels_file,
        # Results paths
        "results": results,
        "sorted": results / "sorted_data",
        "preprocessed": results / "preprocessed_data",
        "mc": results / "mc",
        "dfc": results / "dfc",
        "speed": results / "speed",
        "mc_mod": results / "mc_mod",
        "allegiance": results / "allegiance",
        "trimers": results / "trimers",
        # Figures paths
        "figures": figures,
        "fmodularity": figures / "modularity",
        "f_mod": figures / "modularity",
        "f_cog": figures / "cog",
        "f_speed": figures / "speed",
    }


# =============================================================================
def create_directories(paths: Dict[str, Path]) -> None:
    """
    Create all directories in the mapping that are not files (no suffix).
    """
    for path in paths.values():
        if not path.suffix and not path.exists():
            path.mkdir(parents=True, exist_ok=True)


# =============================================================================
def check_write_permissions(paths: Dict[str, Path]) -> None:
    """
    Check basic write permissions for directories in `paths`.

    Creates and removes a small test file in each directory. Raises a
    `PermissionError` with the list of failing entries.
    """
    unwritable = []
    for key, path in paths.items():
        if not path.suffix:  # Only check directories
            try:
                test_file = path / ".write_test"
                with open(test_file, "w") as f:
                    f.write("test")
                test_file.unlink()
            except Exception:
                unwritable.append((key, str(path)))
    if unwritable:
        raise PermissionError(f"Write permission denied for: {unwritable}")


# =============================================================================
def get_paths(
    dataset_name: Optional[str] = None,
    timecourse_folder: str = "Timecourses_updated_03052024",
    cognitive_data_file: str = "ROIs.xlsx",
    anat_labels_file: str = "all_ROI_coimagine.txt",
    create: bool = True,
    check_write: bool = False,
    env: str = "LOCAL",
) -> Dict[str, Path]:
    """
    Generate a dictionary of canonical paths for data, results, and figures.

    Parameters mirror `build_paths`. When `create=True`, missing folders are created.
    Use `env` to select which environment variable to read as the root.
    """
    # Load the root path from environment variable or default to LOCAL
    root = get_root_path(env)

    # Use dataset_name param or fallback to env
    dataset_name = dataset_name or os.getenv("DATASET_NAME", "ines_abdullah")

    # Define paths based on dataset_name
    if not dataset_name:
        raise ValueError(
            "dataset_name must be provided or set in environment variables."
        )

    # Build paths dictionary
    paths = build_paths(
        root, dataset_name, timecourse_folder, cognitive_data_file, anat_labels_file
    )

    # Create directories if they do not exist
    if create:
        create_directories(paths)

    # Check write permissions if requested
    if check_write:
        check_write_permissions(paths)
    return paths
