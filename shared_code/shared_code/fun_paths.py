
"""Samy Castro Novoa 04.06.2025"""

from pathlib import Path
import os
from dotenv import load_dotenv
from typing import Dict, Optional

# Load environment variables from ../../.env if present
load_dotenv()

# =============================================================================
# Get Paths folder
# =============================================================================
def get_root_path(env='LOCAL'):
    # root = os.environ.get("PROJECT_DATA_ROOT")
    root = os.getenv(f"PROJECT_ROOT_{env}")
    if not root:
        raise EnvironmentError("Environment variable PROJECT_ROOT_EXT is not set.")
    return Path(root)

# =============================================================================
def build_paths(
    root: Path,
    dataset_name: str,
    timecourse_folder: str,
    cognitive_data_file: str,
    anat_labels_file: str
) -> Dict[str, Path]:
    """Builds and returns all required paths as a dictionary."""
    
    
    # Define paths based on dataset_name
    dataset = root / 'dataset' / dataset_name
    results = root / 'results' / dataset_name
    figures = root / 'fig' / dataset_name

    return {
        'root': root,
        # Load raw dataset paths
        'timeseries': dataset / timecourse_folder,
        'cog_data': dataset / 'cog_data' / cognitive_data_file,
        'labels': dataset / 'cog_data' / anat_labels_file,
        
        # Results paths
        'results': results,
        'sorted': results / 'sorted_data',
        'preprocessed': results / 'preprocessed_data',
        'mc': results / 'mc',
        'dfc': results / 'dfc',
        'speed': results / 'speed',
        'mc_mod': results / 'mc_mod',
        'allegiance': results / 'allegiance',
        'trimers': results / 'trimers',

        # Figures paths
        'figures': figures,
        'fmodularity': figures / 'modularity',
        'f_mod': figures / 'modularity',
        'f_cog': figures / 'cog',
    }

# =============================================================================
def create_directories(paths: Dict[str, Path]) -> None:
    """Create all directories that are not files."""
    for path in paths.values():
        if not path.suffix and not path.exists():
            path.mkdir(parents=True, exist_ok=True)

# =============================================================================
def check_write_permissions(paths: Dict[str, Path]) -> None:
    """Check if directories are writable, raise error if not."""
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
    env: str = 'LOCAL'
) -> Dict[str, Path]:
    """
    Generate a dictionary of paths for various data and result directories.
    """
    # Load the root path from environment variable or default to LOCAL
    root = get_root_path(env)

    # Use dataset_name param or fallback to env
    dataset_name = dataset_name or os.getenv("DATASET_NAME", "ines_abdullah")

    # Define paths based on dataset_name
    if not dataset_name:
        raise ValueError("dataset_name must be provided or set in environment variables.")

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
