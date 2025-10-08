"""
Path helpers for datasets, results, and figures.

Environment variables (can be placed in a local .env):
- `PATHS_ROOT` (optional): absolute path to project root; overrides everything
- `PROJECT_ROOT_<ENV>`: absolute path to project root for a given environment (default ENV='LOCAL')
- `PATHS_ENV` (optional): which `<ENV>` label to use (e.g., `CLUSTER`)
- `DATASET_NAME` (optional): dataset subfolder name used by `get_paths`

Example:
    PROJECT_ROOT_LOCAL=/abs/path/to/root
    DATASET_NAME=ines_abdullah
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from current CWD .env and also try repository root .env
load_dotenv()  # CWD
try:
    here = Path(__file__).resolve()
    # Walk up a few levels to find repo-level .env
    for i in range(1, 6):
        env_file = here.parents[i] / ".env"
        if env_file.exists():
            load_dotenv(dotenv_path=env_file, override=False)
            break
except Exception:
    pass


# =============================================================================
# Get Paths folder
# =============================================================================
def get_root_path(env: str = "LOCAL") -> Path:
    """
    Resolve the project root for a given environment label.

    Parameters:
    - env: str (default 'LOCAL')
      Suffix used to read `PROJECT_ROOT_<env>` from the environment. If set to
      'AUTO', reads PATHS_ENV and falls back to 'LOCAL'. If PATHS_ROOT is set,
      it overrides all and is used directly.

    Returns:
    - Path: absolute path to the configured project root.

    Raises:
    - EnvironmentError: if the corresponding environment variable is not set.
    """
    # Hard override takes precedence
    override = os.getenv("PATHS_ROOT")
    if override:
        return Path(override)

    # Resolve environment label (supports AUTO via PATHS_ENV)
    label = env or "LOCAL"
    if str(label).upper() == "AUTO":
        label = os.getenv("PATHS_ENV", "LOCAL")

    root = os.getenv(f"PROJECT_ROOT_{label}")
    if not root:
        raise OSError(
            "Root path not configured. Set PATHS_ROOT, or define "
            f"PROJECT_ROOT_{label} (optionally set PATHS_ENV to pick label)."
        )
    return Path(root)


# =============================================================================
def build_paths(
    root: Path,
    dataset_name: str,
    timecourse_folder: str,
    cognitive_data_file: str,
    anat_labels_file: str,
) -> dict[str, Path]:
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
        "cohesion": results / "cohesion",

        # Figures paths
        "figures": figures,
        "fmodularity": figures / "modularity",
        "f_mod": figures / "modularity",
        "f_cog": figures / "cog",
        "f_speed": figures / "speed",
        "f_dfc": figures / "dfc",
        "f_cohesion": figures / "cohesion",
        "f_allegiance": figures / "allegiance",
        "f_trimers": figures / "trimers",
    }


# =============================================================================
def create_directories(paths: dict[str, Path]) -> None:
    """
    Create all directories in the mapping that are not files (no suffix).
    """
    for path in paths.values():
        if not path.suffix and not path.exists():
            path.mkdir(parents=True, exist_ok=True)


# =============================================================================
def check_write_permissions(paths: dict[str, Path]) -> None:
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
    dataset_name: str | None = None,
    timecourse_folder: str = "Timecourses_updated_03052024",
    cognitive_data_file: str = "ROIs.xlsx",
    anat_labels_file: str = "all_ROI_coimagine.txt",
    create: bool = True,
    check_write: bool = False,
    env: str = "LOCAL",
) -> dict[str, Path]:
    """
    Generate a dictionary of canonical paths for data, results, and figures.

    Parameters mirror `build_paths`. When `create=True`, missing folders are created.
    Use `env` to select which environment variable to read as the root.
    """
    # Load the root path from environment variable(s)
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
