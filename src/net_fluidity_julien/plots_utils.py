#!/usr/bin/env python3
from __future__ import annotations

from typing import Iterable

import numpy as np


def pool_window_speeds(win_array: np.ndarray, tau: int | None = None) -> np.ndarray:
    pooled: list[np.ndarray] = []
    for a in range(len(win_array)):
        arr = win_array[a]
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 2:
            if tau is None:
                pooled.append(arr[~np.isnan(arr)])
            else:
                if 0 <= int(tau) < arr.shape[0]:
                    pooled.append(arr[int(tau)][~np.isnan(arr[int(tau)])])
    return np.concatenate(pooled) if pooled else np.array([], float)


def pool_speeds_per_animal(
    win_array: np.ndarray, idxs: Iterable[int], tau: int | None = None
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for a in idxs:
        if a >= len(win_array):
            out.append(np.array([], float))
            continue
        arr = win_array[int(a)]
        if arr is None:
            out.append(np.array([], float))
            continue
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 2:
            if tau is None:
                vals = arr[~np.isnan(arr)]
            else:
                if 0 <= int(tau) < arr.shape[0]:
                    vals = arr[int(tau)][~np.isnan(arr[int(tau)])]
                else:
                    vals = np.array([], float)
        else:
            vals = np.array([], float)
        out.append(vals)
    return out


def subsample_equal_length(
    per_animal: list[np.ndarray],
    n_per_animal: int | None = None,
    replace: bool = False,
    random_state: int | None = 0,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    non_empty = [x for x in per_animal if x.size > 0]
    if not non_empty:
        return np.array([], float)
    if n_per_animal is None:
        n_per_animal = min(x.size for x in non_empty)
    pooled = []
    for arr in non_empty:
        idx = rng.choice(arr.size, size=n_per_animal, replace=(replace or arr.size < n_per_animal))
        pooled.append(arr[idx])
    return np.concatenate(pooled) if pooled else np.array([], float)


def split_window_indices(
    window_sizes: list[int], split_at: int | None = None
) -> tuple[list[int], list[int], str]:
    sizes = list(map(int, window_sizes))
    n = len(sizes)
    if split_at is not None:
        first = [i for i, w in enumerate(sizes) if w <= split_at]
        second = [i for i, w in enumerate(sizes) if w > split_at]
        info = f"split_at={split_at} (A: <= {split_at}, B: > {split_at})"
        return first, second, info

    if n % 2 == 0:
        mid = n // 2
        first = list(range(0, mid))
        second = list(range(mid, n))
        info = f"equal-count split between W={sizes[mid-1]} and W={sizes[mid]}"
        return first, second, info
    else:
        mid = n // 2
        dropped = sizes[mid]
        first = list(range(0, mid))
        second = list(range(mid + 1, n))
        info = (
            f"equal-count split by index; dropped middle W={dropped}; "
            f"A up to W={sizes[mid-1]}, B from W={sizes[mid+1]}"
        )
        return first, second, info


def per_animal_summary(
    all_speed: list[np.ndarray],
    reducer: str = "median",
    windows=None,
    taus=None,
    weighting: str = "sample",
    equalize_length: bool = False,
    replace: bool = False,
    random_state: int | None = 0,
) -> np.ndarray:
    n_animals = all_speed[0].shape[0]
    out = np.full(n_animals, np.nan)
    if windows is None:
        windows = range(len(all_speed))
    if taus is None:
        taus = range(all_speed[0].shape[1])
    rng = np.random.default_rng(random_state)

    min_len = None
    if equalize_length:
        lengths = []
        for a in range(n_animals):
            arrs = []
            for w in windows:
                arr3 = np.asarray(all_speed[w][a], float)
                for t in taus:
                    z = arr3[t]
                    z = z[~np.isnan(z)]
                    if z.size:
                        arrs.append(z)
            pooled_a = np.concatenate(arrs) if arrs else np.array([])
            lengths.append(len(pooled_a))
        valid = [l for l in lengths if l > 0]
        min_len = min(valid) if valid else None

    for a in range(n_animals):
        arrs = []
        for w in windows:
            arr3 = np.asarray(all_speed[w][a], float)
            for t in taus:
                z = arr3[t]
                z = z[~np.isnan(z)]
                if z.size:
                    arrs.append(z)
        pooled = np.concatenate(arrs) if arrs else np.array([])
        if pooled.size > 0:
            if equalize_length and min_len is not None and pooled.size >= min_len:
                pooled = rng.choice(pooled, size=min_len, replace=replace)
            if reducer == "median":
                out[a] = np.median(pooled)
            elif reducer == "mean":
                out[a] = np.mean(pooled)
            elif reducer.startswith("q"):
                q = float(reducer[1:]) / 100.0
                out[a] = np.quantile(pooled, q)
            else:
                raise ValueError("Unknown reducer.")
    return out

