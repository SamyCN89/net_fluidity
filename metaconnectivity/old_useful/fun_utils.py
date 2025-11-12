#!/usr/bin/env python3
"""
Legacy compatibility helpers for scripts under `metaconnectivity/old_useful`.

All functions proxy to the modern implementations inside `shared_code`.
This keeps the historical scripts runnable while we migrate them
incrementally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from shared_code.fun_loaddata import (
    extract_hash_numbers,
    load_matdata,
    load_timeseries_bundle,
)
from shared_code.fun_paths import get_paths as _get_paths
from shared_code.fun_utils import (
    classify_phenotypes,
    filename_sort_mat,
    load_cognitive_data,
    load_grouping_data,
    load_timeseries_data,
    set_figure_params,
    split_groups_by_age,
)

__all__ = [
    "get_paths",
    "set_figure_params",
    "load_cognitive_data",
    "load_timeseries_data",
    "load_grouping_data",
    "load_timeseries_bundle",
    "filename_sort_mat",
    "extract_hash_numbers",
    "load_matdata",
    "split_groups_by_age",
    "classify_phenotypes",
]


def get_paths(*args: Any, **kwargs: Any):
    """
    Thin wrapper around `shared_code.fun_paths.get_paths`.

    The legacy scripts passed `external_disk`, `external_path`, and similar
    arguments. We accept arbitrary positional/keyword inputs and forward them,
    relying on the shared helper to validate supported parameters.
    """
    return _get_paths(*args, **kwargs)
