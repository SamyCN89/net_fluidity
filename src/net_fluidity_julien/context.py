#!/usr/bin/env python3
"""
Dataset analysis context for the Julien Caillette data.

Phase 3 migration target: this mirrors julien_data/class_dataanalysis_julien.py
with robust imports. Existing scripts may continue to import the legacy module
until the migration is complete.
"""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd


def _import_shared():
    # Prefer installed package
    try:
        from shared_code.fun_loaddata import (
            extract_mouse_ids,
            load_fc2_npz,
            load_mat_timeseries,
            load_npz_dict,
            load_pickle,
            make_file_path,
        )
        from shared_code.fun_paths import get_paths
        return {
            "extract_mouse_ids": extract_mouse_ids,
            "load_fc2_npz": load_fc2_npz,
            "load_mat_timeseries": load_mat_timeseries,
            "load_npz_dict": load_npz_dict,
            "load_pickle": load_pickle,
            "make_file_path": make_file_path,
            "get_paths": get_paths,
        }
    except ModuleNotFoundError:
        # Fallback to local package folder
        import sys

        here = Path(__file__).resolve().parent
        pkg_dir = here.parents[2] / "shared_code" / "shared_code"
        if pkg_dir.exists():
            sys.path.append(str(pkg_dir))
            from fun_loaddata import (
                extract_mouse_ids,
                load_fc2_npz,
                load_mat_timeseries,
                load_npz_dict,
                load_pickle,
                make_file_path,
            )
            from fun_paths import get_paths
            return {
                "extract_mouse_ids": extract_mouse_ids,
                "load_fc2_npz": load_fc2_npz,
                "load_mat_timeseries": load_mat_timeseries,
                "load_npz_dict": load_npz_dict,
                "load_pickle": load_pickle,
                "make_file_path": make_file_path,
                "get_paths": get_paths,
            }
        raise


_SH = _import_shared()


class DFCAnalysis:
    def __init__(self, dataset_name: str = "julien_caillette"):
        self.paths = _SH["get_paths"](
            dataset_name=dataset_name,
            timecourse_folder="time_courses_2",
            cognitive_data_file="mice_groups_comp_index_2.xlsx",
            anat_labels_file="all_ROI_coimagine_2.txt",
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
        self.ts_list, self.ts_shapes, loaded_files = _SH["load_mat_timeseries"](
            self.paths["timeseries"]
        )
        self.ts_ids = _SH["extract_mouse_ids"](loaded_files)

    # 1.2 Load raw cognitive data
    def load_raw_cognitive_data(self):
        self.cog_data = pd.read_excel(
            self.paths["cog_data"], sheet_name="mice_groups_comp_index"
        )

    # 1.3 Load raw region labels
    def load_raw_region_labels(self):
        self.region_labels = np.loadtxt(self.paths["labels"], dtype=str).tolist()

    # 2. Preprocessed data loading
    def get_metadata(self, meta_filename: str | None = None):
        preproc = Path(self.paths["preprocessed"])  # type: ignore[index]
        if meta_filename is None:
            files = list(preproc.glob("metadata_animals_*.pkl"))
            if not files:
                raise FileNotFoundError(
                    "No metadata pickle file found in preprocessed directory."
                )
            meta_file = files[0]
        else:
            meta_file = preproc / meta_filename
        with open(meta_file, "rb") as f:
            metadata_dict = pickle.load(f)

        self.metadata = metadata_dict
        self.cog_data_filtered = metadata_dict.get("mouse_metadata", None)
        self.region_labels_preprocessed = metadata_dict.get("region_labels", None)
        self.n_animals = metadata_dict.get("n_animals", None)
        self.regions = metadata_dict.get("regions", None)
        self.total_tr = metadata_dict.get("total_tr", None)
        self.filter_mode = metadata_dict.get("filter_mode", "unknown")

    def get_ts_preprocessed(self):
        data_ts_preprocessed = _SH["load_npz_dict"](
            self.paths["preprocessed"]
            / Path(
                f"ts_filtered_animals_{self.n_animals}_regions_{self.regions}_tr_{self.total_tr}.npz"
            )
        )
        self.ts = data_ts_preprocessed["ts"]

    def get_cogdata_preprocessed(self):
        self.cog_data_filtered = pd.read_csv(
            self.paths["preprocessed"]
            / Path(
                f"cog_data_filtered_animals_{self.n_animals}_regions_{self.regions}_tr_{self.total_tr}.csv"
            )
        )
        self.groups = self.cog_data_filtered.groupby(["genotype", "treatment"]).groups

    def load_preprocessed_data(self):
        self.get_metadata()
        self.get_ts_preprocessed()
        self.get_cogdata_preprocessed()

    # Analysis params
    def get_temporal_parameters(self):
        self.lag = self.metadata.get("lag", 1)  # type: ignore[union-attr]
        self.tau = self.metadata.get("tau", 3)  # type: ignore[union-attr]
        self.window_parameter = self.metadata.get("window_range", (5, 100, 1))  # type: ignore[union-attr]
        self.time_window_min, self.time_window_max, self.time_window_step = (
            self.window_parameter
        )
        self.time_window_range = np.arange(
            self.time_window_min, self.time_window_max + 1, self.time_window_step
        )

    # DFC IO helpers (compat)
    def load_dfc_1_window(self, lag=1, window=9, regions=48):
        prefix = "dfc"
        self.dfc_file_path = _SH["make_file_path"](
            self.paths["dfc"], prefix, window, lag, self.n_animals, regions
        )
        results = _SH["load_npz_dict"](self.dfc_file_path)
        self.dfc_stream = results[prefix]
        return self.dfc_stream

    def get_speed_analysis(
        self, tau_arange=np.arange(4), time_window_range=np.arange(5, 50 + 1, 1)
    ):
        prefix = "speed"
        file_path = (
            self.paths["speed"]
            / f"{prefix}_windows{len(time_window_range)}_tau{np.size(tau_arange)}_animals_{self.n_animals}.pkl"
        )
        self.speed = _SH["load_pickle"](file_path)

    def get_speed_fc_analysis(
        self, tau_arange=np.arange(4), time_window_range=np.arange(5, 50 + 1, 1)
    ):
        prefix = "speed"
        file_path = (
            self.paths["speed"]
            / f"{prefix}_windows{len(time_window_range)}_tau{np.size(tau_arange)}_animals_{self.n_animals}.npz"
        )
        self.speed_fc = _SH["load_fc2_npz"](file_path)

