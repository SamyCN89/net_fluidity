#!/usr/bin/env python3
"""
Created on Sat Apr  5 00:18:49 2025

@author: samy
"""
# %%
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat


def load_cognitive_data(path_to_csv: Path) -> pd.DataFrame:
    return pd.read_csv(path_to_csv)

def load_timeseries_data(path_to_npz: Path) -> dict:
    data = np.load(path_to_npz)
    return {
        "ts": data["ts"],
        "n_animals": int(data["n_animals"]),
        "total_tr": data["total_tr"],
        "regions": data["regions"],
        "anat_labels": data["anat_labels"],
        "is_2month_old": data["is_2month_old"],
    }


def load_timeseries(ts_file: Path) -> np.ndarray:
    """
    Load unstacked time series from a .npz file.

    Parameters
    ----------
    ts_file : Path
        Path to the .npz file containing 'ts'.

    Returns
    -------
    np.ndarray
        Array of time series data.
    """
    data = np.load(ts_file, allow_pickle=True)
    return data["ts"]


def validate_alignment(ts_data: np.ndarray, cog_data: pd.DataFrame):
    """
    Ensure time series and cognitive data are aligned.

    Raises
    ------
    AssertionError
        If the lengths do not match.
    """
    assert len(ts_data) == len(
        cog_data
    ), "Mismatch between time series and cognitive data entries."


# =============================================================================
# Preprocessing data
# =============================================================================


def filename_sort_mat(folder_path):
    """Read and sort MATLAB file names in a given folder path."""
    folder = Path(folder_path)
    files_name = sorted(f.name for f in folder.iterdir() if f.suffix == ".mat")
    # files_name = sorted(f for f in os.listdir(folder_path) if f.endswith('.mat'))

    return files_name


def load_matdata(folder_data, specific_folder, files_name):
    ts_list = []
    hash_dir = Path(folder_data) / specific_folder

    # Ensure the directory exists
    for file_name in files_name:
        file_path = hash_dir / file_name
        # Check if the file exists
        try:
            data = loadmat(file_path)["tc"]
            ts_list.append(data)
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")

    # Check if the first dimension is consistent
    first_dim_size = ts_list[0].shape[0]
    if all(data.shape[0] == first_dim_size for data in ts_list):
        # Convert the list to a NumPy array
        ts_array = np.array(ts_list)
        return ts_array
    else:
        print("Error: Inconsistent shapes along the first dimension.")


# %% functions to load grouping data
# ------------------------- Grouping data functions -------------------------
def classify_phenotypes(df, metric_prefix="OiP", threshold=0.2):
    """
    Classify cognitive phenotypes for a given metric, appending metric name to phenotype labels.
    Ines dataset funciton

    Parameters:
        df (pd.DataFrame): DataFrame containing the cognitive data.
        metric_prefix (str): Prefix for the metric (e.g., 'OiP', 'RO24H').
        threshold (float): Threshold to determine high vs. low performance.

    Returns:
        pd.DataFrame: DataFrame with a new column 'Phenotype_<metric>' with labels like 'good_OiP'.
    """
    # Build required column names (custom code)
    col_2m = f"{metric_prefix}_2M"
    col_4m = f"{metric_prefix}_4M"

    # Defensive: Check that required columns exist
    if col_2m not in df.columns or col_4m not in df.columns:
        raise ValueError(f"DataFrame must contain columns '{col_2m}' and '{col_4m}'.")

    # Define boolean masks for each phenotype category
    good = (df[col_2m] >= threshold) & (df[col_4m] >= threshold)
    learners = (df[col_2m] < threshold) & (df[col_4m] >= threshold)
    impaired = (df[col_2m] >= threshold) & (df[col_4m] < threshold)
    bad = (df[col_2m] < threshold) & (df[col_4m] < threshold)

    # Use numpy.select to assign phenotype labels; fallback is 'undefined'
    labels = np.select(
        [good, learners, impaired, bad],
        ["good", "learners", "impaired", "bad"],
        default="undefined",
    )

    # Create a new column for the phenotype labels
    phenotype_column = f"Phenotype_{metric_prefix}"

    # Create a copy of the DataFrame to avoid mutating the original
    df_out = df.copy()

    # Store results in a new column
    df_out[phenotype_column] = pd.Categorical(
        labels, categories=["good", "learners", "impaired", "bad"], ordered=False
    )

    return df_out


def load_grouping_data(path_to_pkl: Path):
    with open(path_to_pkl, "rb") as f:
        mask_groups, label_variables = pickle.load(f)
    return mask_groups, label_variables


def split_groups_by_age(
    group_masks, age_mask, group_labels=None, age_labels=("2m", "4m")
):
    """
    Splits groups into 2 age-based subgroups each.

    Parameters:
    - group_masks: list of boolean arrays (e.g., [wt_mask, dki_mask])
    - age_mask: boolean array (e.g., is_2month_old)
    - group_labels: optional list of group names
    - age_labels: tuple for age split names ('2m', '4m')

    Returns:
    - masks: list of boolean masks
    - labels: list of strings matching each mask
    """
    group_masks = [np.tile(np.asarray(mask), 2) for mask in group_masks]
    age_mask = np.asarray(age_mask)
    n_groups = len(group_masks)

    if group_labels is None:
        group_labels = [f"Group{i}" for i in range(n_groups)]

    masks = []
    labels = []

    for g_mask, g_label in zip(group_masks, group_labels, strict=False):
        for is_2m, age_label in zip([True, False], age_labels, strict=False):
            cond_mask = np.logical_and(g_mask, age_mask == is_2m)
            masks.append(cond_mask)
            labels.append(f"{g_label} {age_label}")

    return masks, labels


def make_masks(group_dict, is_2month_old):
    masks = []
    labels = []
    for group, label in group_dict:
        mask, lab = split_groups_by_age(group, is_2month_old, label)
        masks.append(mask)
        labels.append(lab)
    return tuple(masks), tuple(labels)


def make_combination_masks(
    df, primary_col, by_col, primary_levels, by_levels, is_2month_old
):
    labels = [f"{p}_{b}" for p in primary_levels for b in by_levels]
    conditions = [
        (df[primary_col] == p) & (df[by_col] == b)
        for p in primary_levels
        for b in by_levels
    ]
    return split_groups_by_age(tuple(conditions), is_2month_old, tuple(labels))


# %%

# =============================================================================
# Matrix manipulation functions
# =============================================================================


def matrix2vec(matrix3d):
    """
    Convert a 3D matrix into a 2D matrix by vectorizing each 2D matrix along the third dimension.

    Parameters:
    matrix3d (numpy.ndarray): 3D numpy array.

    Returns:
    numpy.ndarray: 2D numpy array where each column is the vectorized form of the 2D matrices from the 3D input.
    """
    # F: Frame, n: node
    F, n, _ = matrix3d.T.shape  # Assuming matrix3d shape is [F, n, n]
    return matrix3d.reshape((n * n, F))


def dfc_stream2fcd(dfc_stream):
    """
    Calculate the dynamic functional connectivity (dFC) matrix from a dfc_stream.

    Parameters:
    dfc_stream (numpy.ndarray): Input dynamic functional connectivity stream, can be 2D or 3D.

    Returns:
    numpy.ndarray: The dFC matrix computed as the correlation of the dfc_stream.
    """
    if dfc_stream.ndim < 2 or dfc_stream.ndim > 3:
        raise ValueError("Provide a valid size dfc_stream (2D or 3D)!")
    # Convert 3D dfc_stream to 2D if necessary

    if dfc_stream.ndim == 3:
        dfc_stream_2D = matrix2vec(dfc_stream)
    else:
        dfc_stream_2D = dfc_stream

    # Compute dFC
    dfc_stream_2D = dfc_stream_2D.T
    dfc = np.corrcoef(dfc_stream_2D)

    return dfc


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    """
    Check if the matrix a is symmetric
    """
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


# =============================================================================
# Set Figure Params
# =============================================================================


def set_figure_params(savefig=False):
    plt.rcParams.update(
        {
            "axes.labelsize": 15,
            "axes.titlesize": 13,
            "axes.spines.right": False,
            "axes.spines.top": False,
        }
    )
    if savefig == True:
        return savefig
