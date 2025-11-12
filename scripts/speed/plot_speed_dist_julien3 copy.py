# %% ========================== IMPORTS & CONFIG ==========================

from collections.abc import Iterable, Sequence
from datetime import datetime
from itertools import combinations
import json
from pathlib import Path
import pickle
import time

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np

# from prometheus_client import g

# from src import preprocess

# Optional for Parquet saving (Option B)
try:
    import pandas as pd
except Exception:
    pd = None

from scripts.dfc.dfc_compute import DATASET_DEFAULTS, _canonical_dataset
from shared_code.fun_loaddata import load_timeseries_bundle
from shared_code.fun_paths import get_paths
from shared_code.fun_utils import load_cognitive_data, set_figure_params

# ----------------- User toggles -----------------
SAVE_MODE = {
    "npz_pack": True,  # Option A
    "parquet": False,
}  # Option B (requires pandas)
RNG_SEED = 123  # for future bootstrap reproducibility
# -----------------------------------------------

GROUP_RECIPES = {
    # single factors
    "sex": ["Sexe"],
    "age": ["Age"],
    "genotype": ["Genotype"],
    "phenotype_oip": ["Phenotype_OiP"],
    "phenotype_nor": ["Phenotype_RO24h"],
    # 2-way
    "age_sex": ["Age", "Sexe"],
    "age_genotype": ["Age", "Genotype"],
    "age_phenotype_oip": ["Age", "Phenotype_OiP"],
    "age_phenotype_nor": ["Age", "Phenotype_RO24h"],
    "sex_genotype": ["Sexe", "Genotype"],
    # 3-way
    "age_sex_genotype": ["Sexe", "Age", "Genotype"],
    "age_sex_phenotype_oip": ["Sexe", "Age", "Phenotype_OiP"],
    "age_sex_phenotype_nor": ["Sexe", "Age", "Phenotype_RO24h"],
    # default used in your “julien” branch originally
    "genotype_treatment": ["genotype", "treatment"],  # only if those cols exist
}


# # %% ========================== SMALL HELPERS ==========================
# def combo_color(genotype: str, treatment: str) -> str:
#     key = (genotype, treatment)
#     table = {
#         ("WT", "VEH"): "C0",
#         ("WT", "LCTB92"): "C1",
#         ("Dp1Yey", "VEH"): "C2",
#         ("Dp1Yey", "LCTB92"): "C3",
#     }
#     return table.get(key, "gray")


# def combo_label(genotype: str, treatment: str) -> str:
#     return f"{genotype}_{treatment}"


# # --- Load helpers ---
# def load_speed_stack(
#     paths_speed_root: Path, time_windows_range: Sequence[int]
# ) -> list[np.ndarray]:
#     """Return list S where S[j][i] is 1D np.array of samples for animal i at window j."""
#     speeds: list[np.ndarray] = []
#     for w in time_windows_range:
#         a = np.load(
#             paths_speed_root.format(w=w, n_animals=n_animals, regions=regions),
#             allow_pickle=True,
#         )
#         s = a["speeds"]
#         s_flat = [np.asarray(x, dtype=float).ravel() for x in s]
#         speeds.append(s_flat)
#     return speeds


# def load_speed_stack2(
#     paths_speed_root: Path, time_windows_range: Sequence[int], template: str
# ) -> list[np.ndarray]:
#     """Return list S where S[j][i] is 1D np.array of samples for animal i at window j."""
#     speeds: list[np.ndarray] = []
#     for w in time_windows_range:
#         a = np.load(paths_speed_root / template.format(w=w), allow_pickle=True)
#         s = a["speeds"]
#         s_flat = [np.asarray(x, dtype=float).ravel() for x in s]
#         speeds.append(s_flat)
#     return speeds


# # --- pooling helpers ---


# def count_samples_per_window(speeds: list[np.ndarray]) -> np.ndarray:
#     return np.array([sum(len(x) for x in speed) for speed in speeds], dtype=int)


# def cdf_split_indices(speeds: list[np.ndarray]) -> tuple[int, int, int]:
#     counts = count_samples_per_window(speeds)
#     cdf = (
#         np.cumsum(counts) / counts.sum()
#         if counts.sum() > 0
#         else np.zeros_like(counts, dtype=float)
#     )
#     i_third = int(np.searchsorted(cdf, 1.0 / 3.0))
#     i_half = int(np.searchsorted(cdf, 0.5))
#     i_two_third = int(np.searchsorted(cdf, 2.0 / 3.0))
#     i_third = max(1, i_third)
#     i_half = max(1, i_half)
#     i_two_third = max(i_third + 1, i_two_third)
#     return i_third, i_half, i_two_third


# def select_windows(
#     pool_split: str, n_windows: int, i_third: int, i_half: int, i_two_third: int
# ) -> dict[str, range]:
#     if pool_split == "all":
#         return {"all": range(0, n_windows)}
#     if pool_split == "half":
#         return {"short": range(0, i_half), "long": range(i_half, n_windows)}
#     return {
#         "short": range(0, i_third),
#         "mid": range(i_third, i_two_third),
#         "long": range(i_two_third, n_windows),
#     }


# def flatten_windows(speeds: list[np.ndarray], start: int, end: int) -> np.ndarray:
#     arrays = [
#         np.asarray(s, dtype=float).ravel() for speed in speeds[start:end] for s in speed
#     ]
#     return np.concatenate(arrays) if arrays else np.empty(0, dtype=float)


# def global_min_max(arrs: Iterable[np.ndarray]) -> tuple[float, float]:
#     vals_min = [np.nanmin(a) for a in arrs if a.size]
#     vals_max = [np.nanmax(a) for a in arrs if a.size]
#     vmin = min(vals_min) if vals_min else 0.0
#     vmax = max(vals_max) if vals_max else 1.0
#     if vmin == vmax:
#         vmax = vmin + 1.0
#     return float(vmin), float(vmax)


# # --- stats & histogram helpers ---
# def robust_percentiles(x: np.ndarray, qs=(1, 5, 95, 99)) -> dict[int, float]:
#     if x.size == 0 or not np.isfinite(x).any():
#         return {int(q): np.nan for q in qs}
#     x = x[np.isfinite(x)]
#     ps = np.percentile(x, qs)
#     return {int(q): float(p) for q, p in zip(qs, ps, strict=False)}


# def hist_prob(x, bins, rng):
#     h, e = np.histogram(x, bins=bins, range=rng, density=False)
#     s = h.sum()
#     return (h / s if s > 0 else np.zeros_like(h)), e


# def safe_hist(
#     x: np.ndarray, bins: int, rng: tuple[float, float]
# ) -> tuple[np.ndarray, np.ndarray]:
#     if x.size == 0:
#         edges = np.linspace(rng[0], rng[1], bins + 1)
#         return np.zeros(bins), edges
#     return np.histogram(x.T, bins=bins, range=rng, density=False)


# def normalize_counts_to_prob(h: np.ndarray) -> np.ndarray:
#     s = h.sum()
#     return (h / s) if s > 0 else np.zeros_like(h, dtype=float)


# def build_per_animal_normalized_hists(
#     speeds: list[np.ndarray],
#     selected_windows: range,
#     bins: int,
#     hist_range: tuple[float, float],
# ) -> np.ndarray:
#     """Build per-animal normalized histograms over selected windows.

#     speeds:
#         list S where S[j][i] is 1D np.array of samples for animal i at window j.

#     selected_windows: range of window indices to include.
#     Returns H where H[i] is normalized histogram for animal i over selected windows.
#     """
#     n_animals = len(speeds[0])
#     H = np.zeros((n_animals, bins), dtype=float)
#     for i in range(n_animals):
#         # Pool samples for animal i over selected windows
#         flat_i = (
#             np.concatenate([speeds[j][i].ravel() for j in selected_windows])
#             if selected_windows
#             else np.array([], dtype=float)
#         )
#         # Compute & normalize histogram
#         h_i, _ = safe_hist(flat_i, bins, hist_range)
#         H[i] = normalize_counts_to_prob(h_i)
#     return H


# # %%
# def flatten_group_animals_over_windows(
#     animal_speeds: list[list[np.ndarray]], indices: Sequence[int], w_range: range
# ) -> np.ndarray:
#     arrays = []
#     for i in indices:
#         print(f"[DEBUG] Processing animal index: {i}")
#         arrays.extend([animal_speeds[i][j] for j in w_range])
#         print(f"[DEBUG] Current number of arrays for animal {i}: {len(arrays)}")
#     print(f"[DEBUG] Flattened group animals over windows: n_arrays={len(arrays)}")
#     return (
#         np.concatenate([a.ravel() for a in arrays])
#         if arrays
#         else np.array([], dtype=float)
#     )


# def get_group_animals_over_windows(
#     animal_speeds: list[list[np.ndarray]], indices: Sequence[int], w_range: range
# ) -> np.ndarray:
#     """Return object array shape (len(indices), len(w_range)) of arrays."""
#     arr = np.array(animal_speeds, dtype=object)[indices]
#     arrays = np.transpose([arr[:, j] for j in w_range], (1, 0))
#     return (arrays,) if arrays.size else np.array([], dtype=float)


# # %%
# def plot_group_median_vs_window(
#     ax: plt.Axes,
#     time_windows_range: Sequence[int],
#     group_data: dict[tuple[str, str], Sequence[int]],
#     speeds: list[np.ndarray],
# ) -> None:
#     """Mean of per-animal medians vs window (group curve)."""
#     for (genotype, treatment), indices in group_data.items():
#         y = []
#         for j in range(len(time_windows_range)):
#             per_animal_medians = [float(np.median(speeds[j][i])) for i in indices]
#             y.append(
#                 float(np.mean(per_animal_medians)) if per_animal_medians else np.nan
#             )
#         ax.plot(time_windows_range, y, ".-", label=combo_label(genotype, treatment))
#     ax.set_xlabel("Time Window Size")
#     ax.set_ylabel("Mean of per-animal medians (dFC speed)")
#     ax.set_title("dFC Speed vs Window per Genotype–Treatment")
#     ax.legend()


# def plot_group_histograms(
#     ax: plt.Axes,
#     centers: np.ndarray,
#     group_hists: dict[tuple[str, str], np.ndarray],
#     title: str,
#     ylog: bool = False,
# ) -> None:
#     for (genotype, treatment), hist in group_hists.items():
#         ax.plot(centers, hist, lw=1, alpha=0.7, label=combo_label(genotype, treatment))
#     ax.set_title(title)
#     ax.set_xlabel("Speed")
#     ax.set_ylabel("Probability per bin")
#     if ylog:
#         ax.set_yscale("log")
#     # ax.legend()


# # ---------- BOOTSTRAP HELPERS ----------
# # %%


# def flatten_numeric(x) -> np.ndarray:
#     """Flatten lists/arrays (even nested) into a clean 1D float array; drop NaN/inf."""
#     if isinstance(x, (list, tuple)):
#         flat = []
#         for item in np.ravel(x, order="K"):
#             a = np.asarray(item)
#             if a.size:
#                 flat.append(a.ravel())
#         x = np.concatenate(flat) if flat else np.array([])
#     x = np.asarray(x, dtype=float).ravel()
#     return x[np.isfinite(x)]


# def _bootstrap_chunk_proc(
#     x: np.ndarray, q: np.ndarray, m: int, seed: int
# ) -> np.ndarray:
#     """
#     One parallel chunk.
#     x: (n,), q: (K,), m replicates → returns (m, K)
#     """
#     if m <= 0 or x.size == 0:
#         return np.full((0, q.size), np.nan, float)
#     rng = np.random.default_rng(seed)
#     n = x.size
#     # replicates as columns to avoid axis confusion
#     idx = rng.integers(0, n, size=(n, m))  # (n, m)
#     samples = x[idx]  # (n, m)
#     return np.percentile(samples, q, axis=0).T  # (m, K)


# def bootstrap_parallel(
#     data,
#     percentiles,
#     n_resamples: int = 2000,
#     chunk_size: int = 200,
#     n_jobs: int = -1,
#     random_state: int | None = 0,
#     downsample_n: int | None = None,
#     prefer: str = "processes",  # "threads" also allowed
# ):
#     """
#     Parallel 95% CIs for given percentiles of `data`.
#     Returns (low, high), each shape (len(percentiles),).
#     """
#     # print random state

#     x = flatten_numeric(data)
#     q = np.asarray(percentiles, dtype=float).ravel()

#     if x.size == 0:
#         nan = np.full(q.shape, np.nan, float)
#         return nan, nan

#     # optional downsampling
#     if downsample_n and x.size > downsample_n:
#         x = np.random.default_rng(random_state).choice(
#             x, size=downsample_n, replace=False
#         )

#     # plan chunks & deterministic seeds (safe for parallel)
#     offsets = list(range(0, n_resamples, chunk_size))
#     ss = (
#         np.random.SeedSequence(random_state)
#         if random_state is not None
#         else np.random.SeedSequence()
#     )
#     seeds = [int(s.entropy) for s in ss.spawn(len(offsets))]

#     # run chunks in parallel
#     chunks = Parallel(n_jobs=n_jobs, prefer=prefer)(
#         delayed(_bootstrap_chunk_proc)(
#             x, q, min(chunk_size, n_resamples - off), seeds[i]
#         )
#         for i, off in enumerate(offsets)
#     )

#     # combine results
#     boot_all = np.vstack(chunks)  # (n_resamples, len(q))
#     return boot_all


# def bootstrap_percentiles_parallel(
#     data,
#     percentiles,
#     n_resamples: int = 2000,
#     chunk_size: int = 200,
#     n_jobs: int = -1,
#     random_state: int | None = 0,
#     downsample_n: int | None = None,
#     prefer: str = "processes",  # "threads" also allowed
# ):
#     """
#     Parallel 95% CIs for given percentiles of `data`.
#     Returns (low, high), each shape (len(percentiles),).
#     """
#     boot_all = bootstrap_parallel(
#         data,
#         percentiles,
#         n_resamples=n_resamples,
#         chunk_size=chunk_size,
#         n_jobs=n_jobs,
#         random_state=random_state,
#         downsample_n=downsample_n,
#         prefer=prefer,
#     )
#     low, high = np.percentile(boot_all, [2.5, 97.5], axis=0)
#     return low, high


# import numpy as np


# # ---------- utility: efficient downsample without replacement ----------
# def _downsample_once(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
#     n = x.size
#     if k is None or k >= n:
#         return x
#     # choice is faster when k ≪ n; permutation is fine when k is larger
#     if k / n < 0.2:
#         idx = rng.choice(n, size=k, replace=False)
#     else:
#         idx = rng.permutation(n)[:k]
#     return x[idx]


# # ---------- one repeat = one downsample + percentiles ----------
# def _one_repeat_downsample_only(
#     data: np.ndarray,
#     q: np.ndarray,
#     downsample_n: int | None,
#     seed_seq: np.random.SeedSequence,
# ) -> np.ndarray:
#     rng = np.random.default_rng(seed_seq)
#     x = np.ravel(data)
#     if downsample_n and x.size > downsample_n:
#         x = _downsample_once(x, downsample_n, rng)
#     # directly compute percentiles of this downsampled draw
#     return np.percentile(x, q)


# def bootstrap_downsampling_repeat(
#     data: np.ndarray,
#     percentiles: np.ndarray,
#     repeat: int = 10_000,
#     downsample_n: int | None = None,
#     seed: int | None = 0,
#     n_jobs: int = 8,
# ):
#     """
#     Perform `repeat` independent downsampling draws (without replacement),
#     compute `percentiles` on each draw, and return the across-repeat envelope.

#     Returns:
#       ci_low_repeat, ci_high_repeat, ci_repeat
#         - ci_repeat: (repeat, K) matrix; row r = percentiles of the r-th downsample
#         - ci_low_repeat, ci_high_repeat: (K,), here [0th, 100th] across repeats to match your original
#     """
#     q = np.asarray(percentiles, dtype=float).ravel()
#     if data.size == 0:
#         nan = np.full(q.shape, np.nan, float)
#         return nan, nan, np.empty((0, q.size), float)

#     # robust, independent seeds per repeat
#     base_ss = np.random.SeedSequence(None if seed is None else int(seed))
#     child_ss = base_ss.spawn(repeat)

#     rows = Parallel(n_jobs=n_jobs, prefer="processes")(
#         delayed(_one_repeat_downsample_only)(
#             data=data,
#             q=q,
#             downsample_n=downsample_n,
#             seed_seq=child_ss[i],
#         )
#         for i in range(repeat)
#     )
#     ci_repeat = np.vstack(rows)  # (repeat, K)

#     # Use [0, 100] to keep compatibility with what you were doing;
#     # if you want classical CIs, use [2.5, 97.5] instead.
#     ci_low_repeat, ci_high_repeat = np.percentile(ci_repeat, [0, 100], axis=0)
#     return ci_low_repeat, ci_high_repeat, ci_repeat


# # %% ================ GROUP BOOTSTRAP RESAMPLING HELPERS ==========================


# def group_pool_bt_hierarchical_resampling(
#     ranges,
#     group_speed_by_segment,
#     group_data,
#     repeat=20,
#     downsample_n=None,
#     chunk_size=20,
#     n_resamples=10_000,
#     n_jobs=1,
# ):
#     """Hierarchical bootstrap resampling of group speeds."""
#     # Bootstrap per group speed histograms

#     ci_low_mean = {}
#     ci_high_mean = {}

#     # pooling loops segments
#     for seg_name, w_range in ranges.items():
#         start = time.time()
#         ci_low_mean_i = {}
#         ci_high_mean_i = {}
#         # Group loops
#         for gt, idxs in group_data.items():
#             print(
#                 "Speed shape:",
#                 np.shape(group_speed_by_segment[seg_name][gt][0]),
#                 seg_name,
#                 gt,
#             )
#             group_speed_i = group_speed_by_segment[seg_name][gt][0]
#             group_speed_n = len(group_speed_i)

#             # resampling indices
#             ci_low = np.empty((len(percentiles_), repeat), dtype=float)
#             ci_high = np.empty((len(percentiles_), repeat), dtype=float)
#             # print('Group speed i shape:', np.shape(group_speed_i))
#             for _ in range(repeat):  # repeat to see variability
#                 n_hierarchical_resampling = 8
#                 indices_resampling = np.random.choice(
#                     group_speed_n, size=n_hierarchical_resampling, replace=False
#                 )
#                 print(indices_resampling)

#                 # bootstrap samples
#                 flat_list = np.array(
#                     np.concatenate(group_speed_i[indices_resampling].ravel()).tolist()
#                 )

#                 # bootstrap percentiles
#                 ci_low_i, ci_high_i = bootstrap_percentiles_parallel(
#                     flat_list,
#                     percentiles_,
#                     n_resamples=n_resamples,
#                     chunk_size=chunk_size,
#                     random_state=np.random.randint(0, 1_000_000, n_resamples),
#                     n_jobs=n_jobs,
#                     downsample_n=downsample_n,
#                 )
#                 ci_low[:, _] = ci_low_i
#                 ci_high[:, _] = ci_high_i
#             ci_low_mean_i[gt] = ci_low.mean(axis=1)
#             ci_high_mean_i[gt] = ci_high.mean(axis=1)
#         end = time.time()
#         print(
#             f"Hierarchical BT resampling for segment {seg_name} took {end - start:.2f} seconds"
#         )
#         ci_low_mean[seg_name] = ci_low_mean_i
#         ci_high_mean[seg_name] = ci_high_mean_i
#     return ci_low_mean, ci_high_mean


# # Classic resampling of distribution: Assumes that the distributions of a group's speeds are identical across animals
# def group_pool_bt_classic_resampling(
#     ranges,
#     pooled_group_speed_by_segment,
#     group_data,
#     seed=np.random.default_rng(),
#     downsample_n=None,
# ):
#     """Classic bootstrap resampling of pooled group speeds."""
#     # Bootstrap per group speed histograms

#     # pooling loops segments
#     ci_low_mean = {}
#     ci_high_mean = {}

#     for seg_name, w_range in ranges.items():
#         # Group loops
#         ci_low_i = {}
#         ci_high_i = {}
#         for gt, idxs in group_data.items():
#             print(pooled_group_speed_by_segment[seg_name][gt].shape, seg_name, gt)
#             group_flat_speed = pooled_group_speed_by_segment[seg_name][gt]

#             data = np.ravel(group_flat_speed)
#             start = time.time()
#             ci_low, ci_high = bootstrap_percentiles_parallel(
#                 data,
#                 percentiles_,
#                 n_resamples=n_resamples,
#                 chunk_size=chunk_size,
#                 random_state=seed,
#                 n_jobs=8,
#                 downsample_n=downsample_n,
#             )
#             end = time.time()

#             # Print timing
#             if not downsample_n:
#                 print(
#                     f"Classic BT resampling took {end - start:.2f} seconds: {seg_name} {gt}"
#                 )
#             else:
#                 print(
#                     f"Downsampled Classic BT resampling took {end - start:.2f} seconds: {seg_name} {gt}"
#                 )
#             # Store results per group
#             ci_low_i[gt] = ci_low
#             ci_high_i[gt] = ci_high
#         # store results per segment
#         ci_low_mean[seg_name] = ci_low_i
#         ci_high_mean[seg_name] = ci_high_i
#     return ci_low_mean, ci_high_mean


# # Bootstrap downsampling with repeats
# def group_pool_bt_classic_resampling_downsampled(
#     ranges: dict,
#     pooled_group_speed_by_segment: dict,  # {seg: {group: array_like}}
#     group_data: dict,  # used for group iteration order
#     percentiles_: np.ndarray,
#     repeat: int = 10_000,
#     downsample_factor: int = 10,
#     seed: int | None = 0,
#     n_jobs: int = 8,
#     verbose: int = 0,
# ):
#     """
#     For each (segment, group):
#       - flatten its pooled speeds
#       - repeatedly downsample WITHOUT replacement to N/f
#       - compute given percentiles on each downsample
#       - aggregate percentiles across repeats to get an envelope
#     """
#     q = np.asarray(percentiles_, dtype=float).ravel()

#     ci_low_repeat: dict[str, dict[str, np.ndarray]] = {}
#     ci_high_repeat: dict[str, dict[str, np.ndarray]] = {}
#     vals_btr_downsample_repeat: dict[str, dict[str, np.ndarray]] = {}

#     # spawn independent seeds per (segment, group)
#     base_ss = np.random.SeedSequence(None if seed is None else int(seed))
#     pair_list = [
#         (seg_name, gt) for seg_name in ranges.keys() for gt in group_data.keys()
#     ]
#     child_ss = base_ss.spawn(len(pair_list))
#     ss_iter = iter(child_ss)

#     for seg_name in ranges.keys():
#         ci_low_seg: dict[str, np.ndarray] = {}
#         ci_high_seg: dict[str, np.ndarray] = {}
#         vals_seg: dict[str, np.ndarray] = {}

#         for gt in group_data.keys():
#             if (
#                 seg_name not in pooled_group_speed_by_segment
#                 or gt not in pooled_group_speed_by_segment[seg_name]
#             ):
#                 raise KeyError(f"Missing data for segment '{seg_name}', group '{gt}'")

#             data = np.ravel(pooled_group_speed_by_segment[seg_name][gt])
#             # guard against zero downsample size
#             if downsample_factor and downsample_factor > 1:
#                 ds_n = max(1, int(len(data) // downsample_factor))
#             else:
#                 ds_n = None

#             ss = next(ss_iter)
#             t0 = time.time()
#             lo, hi, vals = bootstrap_downsampling_repeat(
#                 data=data,
#                 percentiles=q,
#                 repeat=repeat,
#                 downsample_n=ds_n,
#                 seed=ss.entropy,  # okay to use an int; we could also pass ss itself and spawn again inside
#                 n_jobs=n_jobs,
#             )
#             t1 = time.time()

#             ci_low_seg[gt] = lo
#             ci_high_seg[gt] = hi
#             vals_seg[gt] = vals
#             if verbose:
#                 print(
#                     f"[{seg_name} | {gt}] n={len(data)}, downsample_n={ds_n}, repeats={repeat} in {t1-t0:.2f}s"
#                 )

#         ci_low_repeat[seg_name] = ci_low_seg
#         ci_high_repeat[seg_name] = ci_high_seg
#         vals_btr_downsample_repeat[seg_name] = vals_seg

#     return ci_low_repeat, ci_high_repeat, vals_btr_downsample_repeat


# # %%
# # ---------- CI SAVE HELPERS ----------
# def _gt_key(gt: tuple[str, str]) -> str:
#     return f"{gt[0]}__{gt[1]}"


# def save_bootstrap_cis_npz(
#     base_dir: Path,
#     dataset: str,
#     meta: dict,
#     percentiles: np.ndarray,
#     ci_low_mean: dict[str, dict[tuple[str, str], np.ndarray]],
#     ci_high_mean: dict[str, dict[tuple[str, str], np.ndarray]],
#     ci_low_btr: dict[str, dict[tuple[str, str], np.ndarray]],
#     ci_high_btr: dict[str, dict[tuple[str, str], np.ndarray]],
#     ci_low_btr_downsample: dict[str, dict[tuple[str, str], np.ndarray]],
#     ci_high_btr_downsample: dict[str, dict[tuple[str, str], np.ndarray]],
# ):
#     """
#     Writes one NPZ per segment with keys:
#       - 'percentiles'
#       - '{group}__hier_low', '{group}__hier_high'
#       - '{group}__classic_low', '{group}__classic_high'
#       - '{group}__classic_down_low', '{group}__classic_down_high'
#     """
#     out_dir = base_dir
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # store meta snapshot for the CI pack (won’t overwrite your other meta)
#     (out_dir / "meta_ci.json").write_text(json.dumps(meta, indent=2))

#     # write one file per segment
#     seg_names = sorted(
#         set(ci_low_mean.keys())
#         | set(ci_high_mean.keys())
#         | set(ci_low_btr.keys())
#         | set(ci_high_btr.keys())
#         | set(ci_low_btr_downsample.keys())
#         | set(ci_high_btr_downsample.keys())
#     )

#     for seg in seg_names:
#         pack = {"percentiles": np.asarray(percentiles, dtype=float)}

#         # helper to add a whole dict-of-groups with a prefix
#         def add_side(
#             side_dict: dict[str, dict[tuple[str, str], np.ndarray]], suffix: str
#         ):
#             if seg not in side_dict:
#                 return
#             for gt, arr in side_dict[seg].items():
#                 pack[f"{_gt_key(gt)}__{suffix}"] = np.asarray(arr, dtype=float)

#         add_side(ci_low_mean, "hier_low")
#         add_side(ci_high_mean, "hier_high")
#         add_side(ci_low_btr, "classic_low")
#         add_side(ci_high_btr, "classic_high")
#         add_side(ci_low_btr_downsample, "classic_down_low")
#         add_side(ci_high_btr_downsample, "classic_down_high")

#         np.savez_compressed(out_dir / f"cis__{seg}.npz", **pack)
#     print(f"[INFO] Saved bootstrap CI NPZ packs to: {out_dir} , cis__{seg}.npz")


# def save_bootstrap_cis_parquet(
#     base_dir: Path,
#     dataset: str,
#     meta: dict,
#     percentiles: np.ndarray,
#     ci_low_mean: dict[str, dict[tuple[str, str], np.ndarray]],
#     ci_high_mean: dict[str, dict[tuple[str, str], np.ndarray]],
#     ci_low_btr: dict[str, dict[tuple[str, str], np.ndarray]],
#     ci_high_btr: dict[str, dict[tuple[str, str], np.ndarray]],
#     ci_low_btr_downsample: dict[str, dict[tuple[str, str], np.ndarray]],
#     ci_high_btr_downsample: dict[str, dict[tuple[str, str], np.ndarray]],
# ):
#     """
#     Long-form Parquet with columns:
#       segment, group, percentile, method, bound, value
#     where method ∈ {'hier', 'classic', 'classic_down'}
#           bound  ∈ {'low', 'high'}
#     """
#     if pd is None:
#         print("[WARN] pandas not available; skipping CI parquet save.")
#         return

#     out_dir = base_dir
#     out_dir.mkdir(parents=True, exist_ok=True)

#     rows = []

#     def add_rows(
#         container: dict[str, dict[tuple[str, str], np.ndarray]], method: str, bound: str
#     ):
#         for seg, per_group in container.items():
#             for gt, arr in per_group.items():
#                 g = _gt_key(gt)
#                 for p, v in zip(
#                     percentiles, np.asarray(arr, dtype=float), strict=False
#                 ):
#                     rows.append(
#                         {
#                             "segment": seg,
#                             "group": g,
#                             "percentile": float(p),
#                             "method": method,
#                             "bound": bound,
#                             "value": float(v),
#                         }
#                     )

#     add_rows(ci_low_mean, "hier", "low")
#     add_rows(ci_high_mean, "hier", "high")
#     add_rows(ci_low_btr, "classic", "low")
#     add_rows(ci_high_btr, "classic", "high")
#     add_rows(ci_low_btr_downsample, "classic_down", "low")
#     add_rows(ci_high_btr_downsample, "classic_down", "high")

#     df = pd.DataFrame(rows)
#     df.to_parquet(out_dir / "cis_longform.parquet", index=False)
#     print(f"[INFO] Saved bootstrap CI Parquet to: {out_dir} , cis_longform.parquet")


# # %% ================ GROUPING HELPERS ==========================
# # Group indices from cognitive data


def make_long_cog(cog_data: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Return a long-form dataframe with columns:
    Name, Sexe, Genotype, Age, oip, ro24h, tc, Phenotype_OiP, Phenotype_RO24h
    """
    if dataset_name == "julien":
        # already a single age or different schema? adapt here if needed
        df = cog_data.copy()
        # ensure consistent columns exist; fill missing if required
        for c in ["oip", "ro24h", "tc", "Phenotype_OiP", "Phenotype_RO24h"]:
            if c not in df.columns:
                df[c] = np.nan
        if "Age" not in df.columns:
            df["Age"] = "NA"
        df = df[
            [
                "Name",
                "Sexe",
                "Genotype",
                "Age",
                "oip",
                "ro24h",
                "tc",
                "Phenotype_OiP",
                "Phenotype_RO24h",
            ]
        ]

    elif dataset_name == "ines":
        cols_common = ["Name", "Sexe", "Genotype", "Phenotype_OiP", "Phenotype_RO24h"]
        df2 = cog_data[cols_common + ["OiP_2M", "RO24h_2M", "TC_2M"]].copy()
        df4 = cog_data[cols_common + ["OiP_4M", "RO24h_4M", "TC_4M"]].copy()

        df2["Age"] = "2M"
        df4["Age"] = "4M"
        df2 = df2.rename(columns={"OiP_2M": "oip", "RO24h_2M": "ro24h", "TC_2M": "tc"})
        df4 = df4.rename(columns={"OiP_4M": "oip", "RO24h_4M": "ro24h", "TC_4M": "tc"})
        df = pd.concat([df2, df4], ignore_index=True)

    else:
        raise ValueError(f"Unknown dataset_name={dataset_name}")

    # Normalize values
    df["Sexe"] = df["Sexe"].map({"F": "female", "M": "male"}).fillna(df["Sexe"])
    # Optional: categorical types (faster grouping, stable order)
    for col in ["Sexe", "Age", "Genotype", "Phenotype_OiP", "Phenotype_RO24h"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def group_indices(df: pd.DataFrame, by: Sequence[str], index_col: str = "Name"):
    """
    Return a dict { group_key_tuple_or_scalar : np.ndarray[int] } of row indices.
    """
    if not by:
        # single group: "all"
        return {"all": np.arange(len(df), dtype=int)}
    gb = df.groupby(list(by), sort=False)
    return {k: v.values for k, v in gb.groups.items()}


def get_group_data(cog_data: pd.DataFrame, dataset_name: str, groups_selected: str):
    """Return group indices dict based on selected grouping recipe."""
    df_long = make_long_cog(cog_data, dataset_name)

    # If a recipe references missing columns, raise a helpful error.
    cols = GROUP_RECIPES.get(groups_selected)
    if cols is None:
        raise ValueError(
            f"Unknown groups_selected='{groups_selected}'. "
            f"Choose from: {sorted(GROUP_RECIPES.keys())}"
        )

    missing = [c for c in cols if c not in df_long.columns]
    if missing:
        raise ValueError(
            f"Grouping '{groups_selected}' needs columns {missing} "
            f"missing in df_long.columns={list(df_long.columns)}"
        )

    return group_indices(df_long, cols)


# # %%

# # =============================================================================
# # -----------------------------------------------------------------------------
# # ============================= Main Code ==============================
# # -----------------------------------------------------------------------------
# # =============================================================================


# # %% ========================== LOAD DATA ==========================
dataset_name = "ines"
# dataset_name = "julien"
save_fig = set_figure_params(True)
dataset = _canonical_dataset(dataset_name)
cfg = DATASET_DEFAULTS[dataset]

time_windows_range = np.arange(5, 100, 1)

# POOL_SPLIT = "third"  # 'half' | 'third' | 'all'
# BINS_HIST = 200

SAVE_GROUP_HISTS = True
# %% ============= Get paths & output locations ====================
paths = get_paths(
    dataset_name=dataset,
    timecourse_folder=cfg["timecourse_folder"],
    cognitive_data_file=cfg["cognitive_data_file"],
    anat_labels_file=cfg["anat_labels_file"],
)
# # Root locations
speed_root = Path(paths["speed"])
preprocessed_root = Path(paths["preprocessed"])

# # Load location
loaddir_ts_meta = preprocessed_root / "ts_and_meta_2m4m.npz"
loaddir_cog_data = str(
    preprocessed_root
    / "cog_data_filtered_animals_{n_animals}_regions_{regions}_tr_{total_tr}.csv"
)
# # loaddir_speed = str(speed_root / "all/all/speed_win{w}_lag1_tau4_animals_48_regions_37.npz")
# # loaddir_speed = str(speed_root / "dmn_within/nregs-6/speed_win{w}_lag1_tau4_animals_{n_animals}_regions_{regions}.npz")
# loaddir_speed = str(
#     speed_root / "all/speed_win{w}_lag1_tau2_animals_{n_animals}_regions_{regions}.npz"
# )

# # Output location for group histograms
# # outdir_save_group_hists = speed_root / f"{dataset}_pool_{POOL_SPLIT}_bins{BINS_HIST}" / f"pooled_group_hists__{seg_name}.npz"
# # bootstrap CI folder
bootstrap_folder = paths["speed"] / "bootstrap"
bootstrap_folder.mkdir(parents=True, exist_ok=True)


outdir_bootstrap_repeat = str(
    bootstrap_folder
    / "bootstrap_downsample_repeat_group_{groups_selected}_nresamples_{n_resamples}_downsample_factor_{downsample_factor}_seed_{seed}.pkl"
)

# outdir_speed_bootstrap_cis = str(
#     bootstrap_folder / f"bootstrap_cis_{POOL_SPLIT}_bins{BINS_HIST}.npz"
# )

# # Savedir location
# # time window plots
# time_window_folder = paths["f_speed"] / "time_windows"
# time_window_folder.mkdir(parents=True, exist_ok=True)

# savedir_dfc_speed_per_animal = str(
#     time_window_folder
#     / "dFC_speed_per_animal_{n_animals}_regions_{regions}_tr_{total_tr}.png"
# )
# savedir_dfc_speed_group_median_vs_window = str(
#     time_window_folder
#     / "dFC_speed_group_median_vs_window_{n_animals}_regions_{regions}_tr_{total_tr}.png"
# )
# savedir_dfc_speed_percentiles_vs_window = str(
#     time_window_folder
#     / "dFC_speed_percentiles_vs_window_{n_animals}_regions_{regions}_tr_{total_tr}.png"
# )

# # pooling plots
# pooling_folder = paths["f_speed"] / "pooling"
# pooling_folder.mkdir(parents=True, exist_ok=True)

# savedir_dfc_speed_cdf_windows = str(
#     pooling_folder
#     / "dFC_speed_cdf_windows_animals_{n_animals}_regions_{regions}_tr_{total_tr}.png"
# )
# savedir_pooled_speed_hist_bins = str(
#     pooling_folder
#     / "pooled_speed_hist_bins{BINS_HIST}_animals_{n_animals}_regions{regions}_tr{total_tr}.png"
# )
# savedir_pooled_group_hists = str(
#     speed_root
#     / "pooled_group_hists_{POOL_SPLIT}_bins{BINS_HIST}_animals_{n_animals}_regions{regions}_tr{total_tr}.png"
# )


# # %% ========================== LOAD DATA ==========================
# # Load timeseries bundle to get n_animals, n_regions, total_tr
bundle = load_timeseries_bundle(loaddir_ts_meta)
n_animals = bundle.n_animals
total_tr = bundle.total_tr
regions = bundle.n_regions

# Load cognitive data for grouping
cog_data = load_cognitive_data(
    loaddir_cog_data.format(n_animals=n_animals, regions=regions, total_tr=total_tr)
)
#%%
# Load speed data
# speeds = load_speed_stack(
#     loaddir_speed,
#     time_windows_range,
# )


# # %%
# # ========================== PRECOMPUTE DIMENSIONS ==========================
# # Basic dimensions
# n_windows = len(speeds)
# n_animals = len(speeds[0])


# # Precompute mean speed for each animal at each window (handy for mean-based bootstrap later)
# per_window_animal_means = [
#     np.array([float(np.mean(speeds[j][i])) for i in range(n_animals)], dtype=float)
#     for j in range(n_windows)
# ]

# print(
#     "[INFO] Data loaded. n_animals:",
#     n_animals,
#     "n_windows:",
#     n_windows,
#     "per_window_animal_means[0].shape:",
#     per_window_animal_means[0].shape,
# )
# # %% ========================== SPLITS IN POOLINGS & HIST SETUP ==========================
# # Get the split indices and ranges
# counts = count_samples_per_window(speeds)
# pooled_speeds_cdf = (
#     np.cumsum(counts) / np.sum(counts) if counts.sum() else np.zeros_like(counts)
# )
# i_third, i_half, i_two_third = cdf_split_indices(speeds)
# ranges = select_windows(
#     POOL_SPLIT, len(time_windows_range), i_third, i_half, i_two_third
# )
# # Get global min/max for histogram binning
# all_speeds_flat = flatten_windows(speeds, 0, len(speeds))
# all_speeds_min, all_speeds_max = global_min_max([all_speeds_flat])
# edges = np.linspace(all_speeds_min, all_speeds_max, BINS_HIST + 1)
# centers = 0.5 * (edges[:-1] + edges[1:])
# # %%


# # ================================================================================
# # --------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------
# # ================================================================================
# # ----------------------------------- group_data start ---------------------------
# # ================================================================================
# # --------------------------------------------------------------------------------
# # --------------------------------------------------------------------------------
# # ================================================================================

# percentiles_ = np.linspace(0, 100, 100)
n_resamples = 10_000
# # n_resamples = 10
downsample_factor = 10
seed = 42
# n_jobs = 60


# # %% ========================== GROUP INDICES ==========================

# # build once
df_long = make_long_cog(cog_data, dataset_name)

# # pick a grouping
# # groups_selected = "age_sex_genotype"  # or "sex", "age", "phenotype_oip", ...

groups_list = [
    "age_sex",
    "age_genotype",
    "age_sex_genotype",
    "age_sex_phenotype_oip",
    "age_sex_phenotype_nor",
]

# # groups_selected = "age_sex_phenotype_oip"  # or "sex", "age", "phenotype_oip", ...
# for groups_selected in groups_list:
#     print(f"Processing grouping: {groups_selected}")

#     group_data = get_group_data(cog_data, dataset_name, groups_selected)

#     #  ========================== PER-SEGMENT HISTOGRAMS ==========================

#     # Per-animal histograms & per-group mean histograms
#     H_per_segment: dict[str, np.ndarray] = {}
#     group_means_by_segment: dict[str, dict[tuple[str, str], np.ndarray]] = {}

#     # Build per-animal normalized histograms & per-group means
#     for seg_name, w_range in ranges.items():
#         # per-animal normalized histograms
#         H = build_per_animal_normalized_hists(
#             speeds, w_range, BINS_HIST, (all_speeds_min, all_speeds_max)
#         )
#         H_per_segment[seg_name] = H
#         # per-group average histogram over animals
#         group_means: dict[tuple[str, str], np.ndarray] = {}
#         for gt, idxs in group_data.items():
#             group_means[gt] = (
#                 np.mean(H[idxs], axis=0) if len(idxs) else np.zeros(BINS_HIST)
#             )
#         group_means_by_segment[seg_name] = group_means

#     #  ========================== POOLING ==========================

#     # Optional pooled hist per group (values aggregated across animals & windows)
#     animal_speeds = [[speeds[j][i] for j in range(n_windows)] for i in range(n_animals)]
#     pooled_group_hists_by_segment: dict[str, dict[tuple[str, str], np.ndarray]] = {}
#     pooled_group_speed_by_segment: dict[str, dict[tuple[str, str], np.ndarray]] = {}
#     group_speed_by_segment = {}

#     for seg_name, w_range in ranges.items():
#         pooled_group_hist_i = {}
#         pooled_group_speed_i = {}
#         group_speed = {}
#         for gt, idxs in group_data.items():
#             # Get group animal speeds over selected windows
#             group_speed_i = get_group_animals_over_windows(animal_speeds, idxs, w_range)
#             group_speed[gt] = group_speed_i
#             # Flatten values for group animals over selected windows
#             flat = flatten_group_animals_over_windows(animal_speeds, idxs, w_range)
#             pooled_group_speed_i[gt] = flat
#             # Compute & normalize histogram
#             h, _ = safe_hist(flat, BINS_HIST, (all_speeds_min, all_speeds_max))
#             pooled_group_hist_i[gt] = normalize_counts_to_prob(h) * 2
#         pooled_group_hists_by_segment[seg_name] = pooled_group_hist_i
#         pooled_group_speed_by_segment[seg_name] = pooled_group_speed_i
#         group_speed_by_segment[seg_name] = group_speed

#     #

#     # ========================== BOOTSTRAP RESAMPLING WITH DOWNSAMPLING & REPEATS ==========================
#     outdir_bootstrap_repeat_aux = Path(
#         outdir_bootstrap_repeat.format(
#             groups_selected=groups_selected,
#             n_resamples=n_resamples,
#             downsample_factor=downsample_factor,
#             seed=seed,
#         )
#     )
#     # Save ci_low_repeat , ci_high_repeat , ci_btr_downsample_repeat
#     if outdir_bootstrap_repeat_aux.exists():
#         print(f"Loading bootstrap: {outdir_bootstrap_repeat_aux}")
#         with open(
#             outdir_bootstrap_repeat_aux,
#             "rb",
#         ) as f:
#             data_loaded = pickle.load(f)
#             # metadata
#             groups_selected = data_loaded["groups_selected"]
#             group_data = data_loaded["group_data"]
#             ranges = data_loaded["ranges"]
#             percentiles_ = data_loaded["percentiles_"]
#             # bootstrap results
#             ci_low_repeat = data_loaded["ci_low_repeat"]
#             ci_high_repeat = data_loaded["ci_high_repeat"]
#             vals_btr_downsample_repeat = data_loaded["ci_btr_downsample_repeat"]
#             # speed data
#             group_means_by_segment = data_loaded.get(
#                 "group_means_by_segment", group_means_by_segment
#             )
#             pooled_group_hists_by_segment = data_loaded.get(
#                 "pooled_group_hists_by_segment", pooled_group_hists_by_segment
#             )
#             pooled_group_speed_by_segment = data_loaded.get(
#                 "pooled_group_speed_by_segment", pooled_group_speed_by_segment
#             )
#             group_speed_by_segment = data_loaded.get(
#                 "group_speed_by_segment", group_speed_by_segment
#             )

#     else:
#         # Downsampled classic bootstrap resampling with repeats
#         ci_low_repeat, ci_high_repeat, vals_btr_downsample_repeat = (
#             group_pool_bt_classic_resampling_downsampled(
#                 ranges,
#                 pooled_group_speed_by_segment,
#                 group_data,
#                 percentiles_,
#                 repeat=n_resamples,
#                 downsample_factor=downsample_factor,
#                 seed=seed,
#                 n_jobs=n_jobs,
#                 verbose=1,
#             )
#         )
#         with open(outdir_bootstrap_repeat_aux, "wb") as f:
#             pickle.dump(
#                 {
#                     "groups_selected": groups_selected,
#                     "group_data": group_data,
#                     "ranges": ranges,
#                     "percentiles_": percentiles_,
#                     "ci_low_repeat": ci_low_repeat,
#                     "ci_high_repeat": ci_high_repeat,
#                     "ci_btr_downsample_repeat": vals_btr_downsample_repeat,
#                     "group_means_by_segment": group_means_by_segment,
#                     "pooled_group_hists_by_segment": pooled_group_hists_by_segment,
#                     "pooled_group_speed_by_segment": pooled_group_speed_by_segment,
#                     "group_speed_by_segment": group_speed_by_segment,
#                 },
#                 f,
#             )
# %%

# % ========================== PLOTTING ==========================


for groups_selected in groups_list:
    print(f"Processing grouping: {groups_selected}")

    group_data = get_group_data(cog_data, dataset_name, groups_selected)
    print(f"Group data keys: {group_data.keys()}")

    outdir_bootstrap_repeat_aux = Path(
        outdir_bootstrap_repeat.format(
            groups_selected=groups_selected,
            n_resamples=n_resamples,
            downsample_factor=downsample_factor,
            seed=seed,
        )
    )
    print(f"Loading bootstrap: {outdir_bootstrap_repeat_aux}")
    if outdir_bootstrap_repeat_aux.exists():
        print(f"Loading bootstrap: {outdir_bootstrap_repeat_aux}")
        with open(
            outdir_bootstrap_repeat_aux,
            "rb",
        ) as f:
            data_loaded = pickle.load(f)
            # metadata
            groups_selected = data_loaded["groups_selected"]
            group_data = data_loaded["group_data"]
            ranges = data_loaded["ranges"]
            percentiles_ = data_loaded["percentiles_"]
            # bootstrap results
            ci_low_repeat = data_loaded["ci_low_repeat"]
            ci_high_repeat = data_loaded["ci_high_repeat"]
            vals_btr_downsample_repeat = data_loaded["ci_btr_downsample_repeat"]
            # speed data
            group_means_by_segment = data_loaded["group_means_by_segment"]
            pooled_group_hists_by_segment = data_loaded["pooled_group_hists_by_segment"]
            pooled_group_speed_by_segment = data_loaded["pooled_group_speed_by_segment"]
            group_speed_by_segment = data_loaded["group_speed_by_segment"]

    # # #plot group means_by_segment
    # plt.figure(figsize=(16, 8))
    # for seg_name, group_means in group_means_by_segment.items():
    #     print("Plotting Group Means for segment:", seg_name, group_means.keys())
    #     plt.subplot(1, len(ranges), list(ranges.keys()).index(seg_name) + 1)
    #     plt.title(f"Group Means - {seg_name}")
    #     for gt, mean_hist in group_means.items():
    #         plt.plot(centers, mean_hist, label=gt)
    #     plt.xlabel("Speed")
    #     plt.ylabel("Density")
    #     # plt.yscale("log")
    #     # plt.xscale("log")

    #     # plt.ylim(2e-1, 1.5e0)
    #     plt.legend()
    # plt.tight_layout()
    # plt.show()

    # plot pooled_group_hists_by_segment, top subplot female, bottom male
    # plt.figure(figsize=(16, 8))

    # for seg_name, pooled_group_hists in pooled_group_hists_by_segment.items():
    #     print(
    #         "Plotting Pooled Group Hists for segment:",
    #         seg_name,
    #         pooled_group_hists.keys(),
    #     )
    #     plt.subplot(1, len(ranges), list(ranges.keys()).index(seg_name) + 1)
    #     plt.title(f"Pooled Group Histograms - {seg_name}")
    #     for gt, pooled_hist in pooled_group_hists.items():
    #         plt.plot(centers, pooled_hist, label=gt, alpha=0.7)
    #     plt.xlabel("Speed")
    #     plt.ylabel("Density")
    #     plt.legend()
    # plt.tight_layout()
    # plt.show()

    # plot confidence intervals comparison
    plt.figure(figsize=(16, 8))
    for seg_name, w_range in ranges.items():
        plt.subplot(1, len(ranges), list(ranges.keys()).index(seg_name) + 1)
        plt.title(f"Confidence Intervals Comparison - {seg_name}")
        for gt in group_data.keys():
            # print("Plotting GT:", gt)
            plt.fill_between(
                percentiles_,
                ci_low_repeat[seg_name][gt],
                ci_high_repeat[seg_name][gt],
                alpha=0.5,
                label=gt,
            )

        # plt.plot(percentiles_, group_flat_speed_perc, label='Pooled Group Histogram')
        plt.xlabel("Percentiles")
        plt.ylabel("Speed")
        plt.yscale("log")
        plt.xscale("log")
        plt.ylim(2e-1, 1.5e0)
        # plt.xlim(0,3)

        # plt.xscale('log')
        plt.legend()
    plt.tight_layout()
    plt.show()
    # plt.savefig(speed_root / f'ci_comparison_downsampled_classic_bootstrap_{POOL_SPLIT}_bins{BINS_HIST}.png')

#%%
    for seg_name, w_range in ranges.items():
        # Group loops
        ci_low_i = {}
        ci_high_i = {}
        for gt, idxs in group_data.items():
            print(pooled_group_speed_by_segment[seg_name][gt].shape, seg_name, gt)
            group_flat_speed = pooled_group_speed_by_segment[seg_name][gt]

            # for classic resampling of distribution
            group_flat_speed_perc = np.percentile(group_flat_speed, percentiles_)

            # for each animal histogram
            animal_flat_speed_perc = np.empty(
                (len(group_speed_by_segment[seg_name][gt][0]), len(percentiles_)),
                dtype=float,
            )

            group_speed_by_segment_i = group_speed_by_segment[seg_name][gt][0]

            plt.figure(figsize=(12, 10))
            for i in range(len(group_speed_by_segment_i)):
                # print(f"Animal {i} speed shape: {np.shape(group_speed_by_segment_i[i])}")
                aux_animal_s = group_speed_by_segment_i[i]
                aux_animal_i_s_flat = [
                    aux_animal_s[j].tolist() for j in range(len(aux_animal_s))
                ]
                # flatten aux_animal_i_s_flat
                aux_animal_i_s_flat = np.array(
                    [item for sublist in aux_animal_i_s_flat for item in sublist]
                )
                # print(f"Animal {i} flat speed shape: {np.shape(aux_animal_i_s_flat)}")

                hist_aux = np.histogram(
                    np.ravel(aux_animal_i_s_flat),
                    bins=100,
                    range=(group_flat_speed.min(), group_flat_speed.max()),
                )

                plt.plot(
                    hist_aux[1][:-1], hist_aux[0], ".-", label=f"Animal {i}", alpha=0.5
                )  # nor {nor_index[idxs[i]]}')
                # plt.plot()

                plt.xlabel("Speed")
                plt.ylabel("Count")
                plt.title(f"Animal Speed Histogram - {seg_name} - {gt}")
                plt.xlim(0.1, 1.2)

                aux_animal_i_perc = np.percentile(aux_animal_i_s_flat, percentiles_)
                # print(f"Animal {i} percentiles: {aux_animal_i_perc}")
                animal_flat_speed_perc[i] = aux_animal_i_perc
            plt.legend()
            plt.tight_layout()
            plt.show()
            # plt.savefig(speed_root / f'animal_speed_histograms_{seg_name}_{_gt_key(gt)}.png')

# %%

# PArameters to play
groups_selected = groups_list[0]
group_data = get_group_data(cog_data, dataset_name, groups_selected)
print(f"Processing grouping: {groups_selected}")
print(f"Group data keys: {group_data.keys()}")

outdir_bootstrap_repeat_aux = Path(
    outdir_bootstrap_repeat.format(
        groups_selected=groups_selected,
        n_resamples=n_resamples,
        downsample_factor=downsample_factor,
        seed=seed,
    )
)
print(f"Loading bootstrap: {outdir_bootstrap_repeat_aux}")

if outdir_bootstrap_repeat_aux.exists():
    print(f"Loading bootstrap: {outdir_bootstrap_repeat_aux}")
    with open(
        outdir_bootstrap_repeat_aux,
        "rb",
    ) as f:
        data_loaded = pickle.load(f)
        # metadata
        groups_selected = data_loaded["groups_selected"]
        group_data = data_loaded["group_data"]
        ranges = data_loaded["ranges"]
        percentiles_ = data_loaded["percentiles_"]
        # bootstrap results
        ci_low_repeat = data_loaded["ci_low_repeat"]
        ci_high_repeat = data_loaded["ci_high_repeat"]
        vals_btr_downsample_repeat = data_loaded["ci_btr_downsample_repeat"]
        # speed data
        group_means_by_segment = data_loaded["group_means_by_segment"]
        pooled_group_hists_by_segment = data_loaded["pooled_group_hists_by_segment"]
        pooled_group_speed_by_segment = data_loaded["pooled_group_speed_by_segment"]
        group_speed_by_segment = data_loaded["group_speed_by_segment"]
        # --- unwrap legacy tuple structures ---
        for seg_name, seg_dict in group_speed_by_segment.items():
            for gt, val in seg_dict.items():
                if isinstance(val, tuple):
                    group_speed_by_segment[seg_name][gt] = val[0]



#%%
plt.figure(figsize=(16, 8))
for seg_name, w_range in ranges.items():
    plt.subplot(1, len(ranges), list(ranges.keys()).index(seg_name) + 1)
    for gt in group_data.keys():
        # group_speed_by_segment_i = group_speed_by_segment[seg_name][gt]
        arr = group_speed_by_segment[seg_name][gt]  # shape: (n_animals_in_group, n_windows), dtype=object
        n_animals_in_group = arr.shape[0]
        # per-animal loops of pooling windows
        for i in range(n_animals_in_group):
            per_animal_windows = arr[i, :]  # object array of length n_windows
            per_animal_flat = np.concatenate([w.ravel() for w in per_animal_windows if w is not None and np.size(w)])
            # e.g., plot a histogram
            h, edges = np.histogram(per_animal_flat, bins=100, range=(per_animal_flat.min(), per_animal_flat.max()))
            plt.plot(edges[:-1], h, ".-", alpha=0.5, label=f"Animal {i}")
#%%

# %%
# --- config: choose how to aggregate ---


WEIGH_EACH_ANIMAL_EQUALLY = True  # True = median-of-animal-medians; False = pool-all-samples



plt.figure(figsize=(14, 5*len(ranges)))

for si, (seg_name, w_range) in enumerate(ranges.items(), start=1):
    plt.subplot(len(ranges), 1, si)

    for gt in group_data.keys():
        # object array: (n_animals, n_windows), each cell is a 1D array of speeds (can be None/empty)
        arr = group_speed_by_segment[seg_name][gt]
        n_animals, n_windows = arr.shape

        # infer window sizes; replace with your own vector if you have it
        window_sizes = np.linspace(w_range[0], w_range[1], n_windows)

        if WEIGH_EACH_ANIMAL_EQUALLY:
            # 1) per-animal median per window
            per_animal = np.full((n_animals, n_windows), np.nan, float)
            for i in range(n_animals):
                for j in range(n_windows):
                    x = arr[i, j]
                    if x is not None and np.size(x):
                        per_animal[i, j] = np.median(np.ravel(x))
            # 2) group aggregation across animals (equal weight)
            med = np.nanmedian(per_animal, axis=0)
            q1  = np.nanpercentile(per_animal, 25, axis=0)
            q3  = np.nanpercentile(per_animal, 75, axis=0)
        else:
            # pool all samples per window (animals with more samples weigh more)
            med, q1, q3 = [], [], []
            for j in range(n_windows):
                pooled = np.concatenate(
                    [np.ravel(arr[i, j]) for i in range(n_animals)
                     if arr[i, j] is not None and np.size(arr[i, j])]
                ) if n_animals else np.array([])
                if pooled.size == 0:
                    med.append(np.nan); q1.append(np.nan); q3.append(np.nan)
                else:
                    med.append(np.median(pooled))
                    q1.append(np.percentile(pooled, 25))
                    q3.append(np.percentile(pooled, 75))
            med, q1, q3 = np.array(med), np.array(q1), np.array(q3)

        # plot: median with IQR band
        plt.plot(window_sizes, med, label=gt, linewidth=2)
        plt.fill_between(window_sizes, q1, q3, alpha=0.2)

    plt.title(f"{seg_name} — median speed vs window size")
    plt.xlabel("Window size")
    plt.ylabel("Speed (median)")
    plt.legend(frameon=False)
    plt.tight_layout()

#%%
# which quantiles to report (in percent)
QUANTILES = [1, 5, 50, 95, 99]

# choose pooling policy
WEIGH_EACH_ANIMAL_EQUALLY = False
# True  = per-animal median -> take percentiles across animals (equal animal weight)
# False = pool all samples per window across animals (animals with more samples weigh more)

plt.figure(figsize=(14, 5*len(ranges)))

for si, (seg_name, w_range) in enumerate(ranges.items(), start=1):
    plt.subplot(len(ranges), 1, si)

    for gt in group_data.keys():
        arr = group_speed_by_segment[seg_name][gt]  # (n_animals, n_windows) object array
        n_animals, n_windows = arr.shape

        # replace with your known vector if you have exact window sizes per segment
        window_sizes = np.linspace(w_range[0], w_range[1], n_windows)

        if WEIGH_EACH_ANIMAL_EQUALLY:
            # per-animal median per window (equal weight per animal),
            # then percentiles across animals of those medians
            per_animal = np.full((n_animals, n_windows), np.nan, float)
            for i in range(n_animals):
                for j in range(n_windows):
                    x = arr[i, j]
                    if x is not None and np.size(x):
                        per_animal[i, j] = np.median(np.ravel(x))

            lines = {}
            for q in QUANTILES:
                y = np.nanpercentile(per_animal, q, axis=0)
                lines[q], = plt.plot(window_sizes, y, linewidth=1.8 if q==50 else 1.0,
                                     linestyle='-' if q==50 else '--',
                                     label=f"{gt} q{q:02d}")
        else:
            # pool all raw samples per window (sample-weighted)
            qvals = {q: np.full(n_windows, np.nan, float) for q in QUANTILES}
            for j in range(n_windows):
                pooled = np.concatenate([
                    np.ravel(arr[i, j]) for i in range(n_animals)
                    if arr[i, j] is not None and np.size(arr[i, j])
                ]) if n_animals else np.array([])
                if pooled.size:
                    for q in QUANTILES:
                        qvals[q][j] = np.percentile(pooled, q)

            lines = {}
            for q in QUANTILES:
                lines[q], = plt.plot(window_sizes, qvals[q],
                                     linewidth=1.8 if q==50 else 1.0,
                                     linestyle='-' if q==50 else '--',
                                     label=f"{gt} q{q:02d}")

    plt.title(f"{seg_name} — speed quantiles vs window size")
    plt.xlabel("Window size")
    plt.ylabel("Speed")
    plt.legend(frameon=False, ncol=3)
    plt.tight_layout()



# %%
# ========================== PERCENTILE TRACKS ==========================


# Generate speed percentiles per segment
def build_per_animal_flat_speed(
    speeds: list[np.ndarray],
    selected_windows: range,
    bins: int,
    hist_range: tuple[float, float],
) -> np.ndarray:
    """Build per-animal normalized histograms over selected windows.

    speeds:
        list S where S[j][i] is 1D np.array of samples for animal i at window j.

    selected_windows: range of window indices to include.
    Returns H where H[i] is normalized histogram for animal i over selected windows.
    """
    n_animals = len(speeds[0])
    flat_speeds_per_segment = np.zeros((n_animals,), dtype=object)
    for i in range(n_animals):
        # Pool samples for animal i over selected windows
        flat_i = (
            np.concatenate([speeds[j][i].ravel() for j in selected_windows])
            if selected_windows
            else np.array([], dtype=float)
        )
        flat_speeds_per_segment[i] = flat_i
    flat_speeds_per_segment = np.vstack(flat_speeds_per_segment)
    return flat_speeds_per_segment


if dataset_name == "julien":
    nor_index = cog_data["index_NOR"].values
elif dataset_name == "ines":
    nor_index = df_long["ro24h"].values

percentiles_ = np.linspace(0, 100, 100)
n_windows = len(time_windows_range)
flat_speeds_per_segment_i = build_per_animal_flat_speed(
    speeds,
    range(0, n_windows - 1),
    BINS_HIST,
    hist_range=(all_speeds_min, all_speeds_max),
)
# speeds_percentile_per_segment = np.percentile(
speeds_percentile_per_segment = np.percentile(
    flat_speeds_per_segment_i, q=percentiles_, axis=1
)


speeds_percentile_per_segment = {}
for seg_name, w_range in ranges.items():
    flat_speeds_per_segment_i = build_per_animal_flat_speed(
        speeds, w_range, BINS_HIST, hist_range=(all_speeds_min, all_speeds_max)
    )
    print(
        f"[INFO] flat_speeds_per_segment_i shape for segment {seg_name}:",
        flat_speeds_per_segment_i.shape,
    )
    speeds_percentile_per_segment[seg_name] = np.percentile(
        flat_speeds_per_segment_i, q=percentiles_, axis=1
    )

    # %%


# %% ========================== PLOTS (unchanged style) ==========================
# 1) Per-animal curves
plt.figure(figsize=(8, 6))
for i in range(n_animals):
    mean_speeds = [
        per_window_animal_means[j][i] for j in range(len(time_windows_range))
    ]
    genotype = next(
        (g for g, idx_list in group_genotype.items() if i in idx_list), "Unknown"
    )
    treatment = next(
        (t for t, idx_list in group_treatment.items() if i in idx_list), "Unknown"
    )
    color = combo_color(genotype, treatment)
    plt.plot(time_windows_range, mean_speeds, color=color, alpha=0.3)
plt.xlabel("Time Window Size")
plt.ylabel("Mean dFC Speed")

from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color="C0", lw=2, label="WT_VEH"),
    Line2D([0], [0], color="C1", lw=2, label="WT_LCTB92"),
    Line2D([0], [0], color="C2", lw=2, label="Dp1Yey_VEH"),
    Line2D([0], [0], color="C3", lw=2, label="Dp1Yey_LCTB92"),
]
plt.legend(handles=legend_elements, loc="upper right")
plt.title("dFC Speed vs Time Window Size for Each Animal")
if save_fig:
    outpath_fig_aux = savedir_dfc_speed_per_animal.format(
        n_animals=n_animals, regions=regions, total_tr=total_tr
    )
    plt.savefig(outpath_fig_aux, dpi=300)
    print(f"[INFO] Figure saved to: {outpath_fig_aux}")
# %%
# 2) Group curve (mean of per-animal medians)
fig2, ax2 = plt.subplots(figsize=(8, 6))
plot_group_median_vs_window(ax2, time_windows_range, group_data, speeds)
if save_fig:
    outpath_fig2 = savedir_dfc_speed_group_median_vs_window.format(
        n_animals=n_animals, regions=regions, total_tr=total_tr
    )
    fig2.savefig(outpath_fig2, dpi=300)
    print(f"[INFO] Figure saved to: {outpath_fig2}")

# %%
# 3) Percentile tracks per group (1,5,median,95,99) on 5 subplots
fig, axs = plt.subplots(2, 3, figsize=(11, 8), sharex=True)
axes = axs.ravel()
titles = ["1st pct", "5th pct", "Median", "95th pct", "99th pct"]
for (genotype, treatment), indices in group_data.items():
    color = combo_color(genotype, treatment)
    s1, s5, sm, s95, s99 = [], [], [], [], []
    for j in range(len(time_windows_range)):
        gflat = (
            np.concatenate([speeds[j][i].ravel() for i in indices])
            if len(indices)
            else np.array([], dtype=float)
        )
        p = robust_percentiles(gflat, qs=(1, 5, 95, 99))
        s1.append(p[1])
        s5.append(p[5])
        s95.append(p[95])
        s99.append(p[99])
        sm.append(float(np.nanmedian(gflat)) if gflat.size else np.nan)
    axes[0].plot(
        time_windows_range,
        s1,
        ".-",
        alpha=0.6,
        color=color,
        label=combo_label(genotype, treatment),
    )
    axes[1].plot(time_windows_range, s5, ".-", alpha=0.6, color=color)
    axes[2].plot(time_windows_range, sm, ".-", alpha=0.6, color=color)
    axes[3].plot(time_windows_range, s95, ".-", alpha=0.6, color=color)
    axes[4].plot(time_windows_range, s99, ".-", alpha=0.6, color=color)
for ax, t in zip(axes[:5], titles, strict=False):
    ax.set_title(f"dFC speed {t}")
for ax in axes:
    ax.grid(alpha=0.2)
axes[3].set_xlabel("Time Window Size")
axes[4].set_xlabel("Time Window Size")
axes[2].set_ylabel("dFC Speed")
axes[0].legend(ncol=1, fontsize=10)
fig.delaxes(axes[5])
fig.tight_layout()
if save_fig:
    outpath_fig3 = savedir_dfc_speed_percentiles_vs_window.format(
        n_animals=n_animals, regions=regions, total_tr=total_tr
    )
    fig.savefig(outpath_fig3, dpi=300)
    print(f"[INFO] Figure saved to: {outpath_fig3}")

# %%
# 4) CDF across windows
plt.figure(figsize=(7, 5))
plt.title("Cumulative Distribution of dFC Speeds across Time Windows")
plt.plot(time_windows_range, pooled_speeds_cdf, color="orange", lw=2, alpha=0.8)
plt.axvline(
    x=time_windows_range[i_half],
    color="red",
    linestyle="--",
    label="Median Window Size",
    alpha=0.5,
)
plt.axvline(
    x=time_windows_range[i_third],
    color="green",
    linestyle="--",
    label="1/3 Window Size",
    alpha=0.5,
)
plt.axvline(
    x=time_windows_range[i_two_third],
    color="blue",
    linestyle="--",
    label="2/3 Window Size",
    alpha=0.5,
)
plt.axhline(y=0.5, color="red", linestyle="--", alpha=0.5)
plt.axhline(y=1 / 3, color="green", linestyle="--", alpha=0.5)
plt.axhline(y=2 / 3, color="blue", linestyle="--", alpha=0.5)
plt.xlabel("Time Window Size")
plt.ylabel("Cumulative Frequency")
step = max(1, len(time_windows_range) // 12)
plt.xticks(time_windows_range[::step])
plt.legend()
plt.tight_layout()

if save_fig:
    outpath_fig4 = savedir_dfc_speed_cdf_windows.format(
        n_animals=n_animals, regions=regions, total_tr=total_tr
    )
    plt.savefig(outpath_fig4, dpi=300)
    print(f"[INFO] Figure saved to: {outpath_fig4}")

# %%
# 5) Example: pooled histograms (all/short/mid/long)
all_speeds_hist, bin_edge = hist_prob(
    all_speeds_flat, BINS_HIST, (all_speeds_min, all_speeds_max)
)
if POOL_SPLIT == "half":
    short_speeds_flat = flatten_windows(speeds, 0, i_half)
    long_speeds_flat = flatten_windows(speeds, i_half, len(speeds))
elif POOL_SPLIT == "third":
    short_speeds_flat = flatten_windows(speeds, 0, i_third)
    mid_speeds_flat = flatten_windows(speeds, i_third, i_two_third)
    long_speeds_flat = flatten_windows(speeds, i_two_third, len(speeds))

plt.figure(figsize=(7, 5))
plt.title("Pooled Speed (all windows pooled)")
plt.plot(
    bin_edge[:-1],
    all_speeds_hist,
    color="dodgerblue",
    lw=2,
    alpha=0.8,
    label="all animals",
)
if POOL_SPLIT == "half":
    plt.plot(
        bin_edge[:-1],
        hist_prob(short_speeds_flat, BINS_HIST, (all_speeds_min, all_speeds_max))[0],
        color="orange",
        lw=2,
        alpha=0.8,
        label="short windows",
    )
    plt.plot(
        bin_edge[:-1],
        hist_prob(long_speeds_flat, BINS_HIST, (all_speeds_min, all_speeds_max))[0],
        color="green",
        lw=2,
        alpha=0.8,
        label="long windows",
    )
elif POOL_SPLIT == "third":
    plt.plot(
        bin_edge[:-1],
        hist_prob(short_speeds_flat, BINS_HIST, (all_speeds_min, all_speeds_max))[0],
        color="orange",
        lw=2,
        alpha=0.8,
        label="short windows",
    )
    plt.plot(
        bin_edge[:-1],
        hist_prob(mid_speeds_flat, BINS_HIST, (all_speeds_min, all_speeds_max))[0],
        color="purple",
        lw=2,
        alpha=0.8,
        label="mid windows",
    )
    plt.plot(
        bin_edge[:-1],
        hist_prob(long_speeds_flat, BINS_HIST, (all_speeds_min, all_speeds_max))[0],
        color="green",
        lw=2,
        alpha=0.8,
        label="long windows",
    )
plt.legend()
plt.xlabel("Speed")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

if save_fig:
    outpath_fig5 = savedir_pooled_speed_hist_bins.format(
        BINS_HIST=BINS_HIST, n_animals=n_animals, regions=regions, total_tr=total_tr
    )
    plt.savefig(outpath_fig5, dpi=300)
    print(f"[INFO] Figure saved to: {outpath_fig5}")
# %%
# 6) (Optional) Group histograms per segment (per-animal mean & pooled) — no extra saving
PLOT_GROUP_HISTS = True
if PLOT_GROUP_HISTS:
    for seg_name, seg_range in ranges.items():
        aux_range = (
            f"{time_windows_range[seg_range[0]]}-{time_windows_range[seg_range[-1]]} tr"
        )
        print(f"[INFO] Plotting group histograms for segment: {seg_name} {aux_range}")
        # pooled hist by group
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
        plot_group_histograms(
            ax[0],
            centers,
            pooled_group_hists_by_segment[seg_name],
            f"Pooled {seg_name}: {aux_range}",
            ylog=False,
        )
        plot_group_histograms(
            ax[1], centers, pooled_group_hists_by_segment[seg_name], "", ylog=True
        )
        ymin, ymax = ax[1].get_ylim()
        ax[1].set_ylim(bottom=max(1e-5, ymin), top=ymax)
        fig.tight_layout()
        plt.legend()

    if save_fig:
        outpath_fig6 = savedir_pooled_group_hists.format(
            POOL_SPLIT=POOL_SPLIT,
            BINS_HIST=BINS_HIST,
            n_animals=n_animals,
            regions=regions,
            total_tr=total_tr,
        )
        fig.savefig(outpath_fig6, dpi=300)
        print(f"[INFO] Figure saved to: {outpath_fig6}")

plt.show()
# %%
# ---- optional save of data (one NPZ per segment + a JSON meta)

if SAVE_GROUP_HISTS:
    np.savez_compressed(
        outdir_save_group_hists,
        centers=centers,
        **{f"{g}__{t}": v for (g, t), v in group_means.items()},
    )


# %% ========================== SAVE OPTIONS ==========================
def make_meta(
    dataset: str,
    pool_split: str,
    time_windows_range: Sequence[int],
    ranges: dict[str, range],
    groups: dict[tuple[str, str], Sequence[int]],
    bins_hist: int,
    seed: int | None = None,
) -> dict:
    return {
        "dataset": dataset,
        "pool_split": pool_split,
        "time_windows_range": list(map(int, time_windows_range)),
        "ranges": {k: [int(x) for x in v] for k, v in ranges.items()},
        "groups": {f"{k[0]}__{k[1]}": [int(i) for i in v] for k, v in groups.items()},
        "bins_hist": int(bins_hist),
        "rng_seed": int(seed) if seed is not None else None,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }


# ---------- Option A: NPZ Pack ----------
def save_npz_pack(
    base_dir: Path,
    dataset: str,
    meta: dict,
    per_window_animal_means: list[np.ndarray],
    H_per_segment: dict[str, np.ndarray],
    edges: np.ndarray,
    pooled_group_hists_by_segment: dict[str, dict[tuple[str, str], np.ndarray]],
):
    out_dir = (
        base_dir
        / "bootstrap_packs"
        / f"{dataset}_{meta['pool_split']}_bins{meta['bins_hist']}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # per-window per-animal means
    np.savez_compressed(
        out_dir / "per_window_animal_means.npz",
        **{
            f"w{j}": per_window_animal_means[j]
            for j in range(len(per_window_animal_means))
        },
    )
    # histograms per segment
    for seg, H in H_per_segment.items():
        np.savez_compressed(out_dir / f"per_animal_hists__{seg}.npz", H=H, edges=edges)
    # pooled group hists per segment
    for seg, d in pooled_group_hists_by_segment.items():
        np.savez_compressed(
            out_dir / f"pooled_group_hists__{seg}.npz",
            **{f"{g[0]}__{g[1]}": v for g, v in d.items()},
        )


# ---------- Option B: Parquet Long-Form ----------
def save_parquet_longform(
    base_dir: Path,
    dataset: str,
    meta: dict,
    centers: np.ndarray,
    H_per_segment: dict[str, np.ndarray],
    group_means_by_segment: dict[str, dict[tuple[str, str], np.ndarray]],
):
    if pd is None:
        print("[WARN] pandas not available; skipping parquet save.")
        return
    out_dir = (
        base_dir
        / "bootstrap_packs"
        / f"{dataset}_{meta['pool_split']}_bins{meta['bins_hist']}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Per-animal histograms (long form): one row per (segment, animal, bin)
    rows = []
    for seg, H in H_per_segment.items():
        n_animals, n_bins = H.shape
        for i_an in range(n_animals):
            rows.extend(
                [
                    {
                        "segment": seg,
                        "animal": int(i_an),
                        "bin": int(k),
                        "center": float(centers[k]),
                        "p": float(H[i_an, k]),
                    }
                    for k in range(n_bins)
                ]
            )
    df_anim = pd.DataFrame(rows)
    df_anim.to_parquet(out_dir / "per_animal_hists.parquet", index=False)

    # Group-average histograms (long form): one row per (segment, group, bin)
    rows = []
    for seg, d in group_means_by_segment.items():
        for (g, t), hist in d.items():
            for k, p in enumerate(hist):
                rows.append(
                    {
                        "segment": seg,
                        "group": f"{g}__{t}",
                        "bin": int(k),
                        "center": float(centers[k]),
                        "p": float(p),
                    }
                )
    df_group = pd.DataFrame(rows)
    df_group.to_parquet(out_dir / "group_mean_hists.parquet", index=False)


# ---------- Execute selected save modes ----------
meta = make_meta(
    dataset,
    POOL_SPLIT,
    time_windows_range,
    ranges,
    group_data,
    BINS_HIST,
    RNG_SEED,
)
base_dir = Path(paths["speed"])

if SAVE_MODE.get("npz_pack", False):
    save_npz_pack(
        base_dir,
        dataset,
        meta,
        per_window_animal_means,
        H_per_segment,
        edges,
        pooled_group_hists_by_segment,
    )

if SAVE_MODE.get("parquet", False):
    save_parquet_longform(
        base_dir, dataset, meta, centers, H_per_segment, group_means_by_segment
    )

print("[INFO] Save completed. Modes:", SAVE_MODE)


# # --- Drop-in replacement for the two plotting loops (with optional saving) ---
# out_dir = Path(paths["speed"]) / "bootstrap_packs" / f"{dataset}_{POOL_SPLIT}_bins{BINS_HIST}"
# out_dir.mkdir(parents=True, exist_ok=True)
# SAVE_GROUP_HISTS = True  # toggle saving of group histograms per segment

# # 1) Per-animal MEAN histograms by group, per segment
# for seg_name, w_range in ranges.items():
#     print(f"Building per-animal histograms for segment '{seg_name}' with windows {list(w_range)}")
#     H_per_animal = build_per_animal_normalized_hists(
#         speeds, w_range, BINS_HIST, (all_speeds_min, all_speeds_max)
#     )

#     # average over animals per group -> a histogram curve per group
#     group_means: dict[tuple[str, str], np.ndarray] = {}
#     for gt, idxs in group_data.items():
#         print(f"Processing group: {gt} with indices: {idxs}")
#         group_means[gt] = np.mean(H_per_animal[idxs], axis=0) if len(idxs) else np.zeros(BINS_HIST)

#     # ---- plot
#     fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
#     plot_group_histograms(ax[0], centers, group_means, f"Per-animal mean hists ({seg_name})", ylog=False)
#     plot_group_histograms(ax[1], centers, group_means, f"Per-animal mean hists (log, {seg_name})", ylog=True)
#     # keep log strictly positive
#     ymin, ymax = ax[1].get_ylim()
#     ax[1].set_ylim(bottom=max(1e-5, ymin), top=ymax)
#     fig.tight_layout()

#     # ---- optional save of data (one NPZ per segment + a JSON meta)
#     if SAVE_GROUP_HISTS:
#         np.savez_compressed(
#             out_dir / f"group_mean_hists__{seg_name}.npz",
#             centers=centers,
#             **{f"{g}__{t}": v for (g, t), v in group_means.items()},
#         )

# # 2) POOLED histograms by group, per segment (flatten all values over animals & windows)
# for seg_name, w_range in ranges.items():
#     pooled_group: dict[tuple[str, str], np.ndarray] = {}
#     for gt, idxs in group_data.items():
#         pooled_group[gt] = pooled_group_histogram(
#             animal_speeds, idxs, w_range, BINS_HIST, (all_speeds_min, all_speeds_max)
#         )

#     # ---- plot
#     fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
#     plot_group_histograms(ax[0], centers, pooled_group, f"Pooled hist ({seg_name})", ylog=False)
#     plot_group_histograms(ax[1], centers, pooled_group, f"Pooled hist (log, {seg_name})", ylog=True)
#     ymin, ymax = ax[1].get_ylim()
#     ax[1].set_ylim(bottom=max(1e-5, ymin), top=ymax)
#     fig.tight_layout()

#     # ---- optional save of data
#     if SAVE_GROUP_HISTS:
#         np.savez_compressed(
#             out_dir / f"pooled_group_hists__{seg_name}.npz",
#             centers=centers,
#             **{f"{g}__{t}": v for (g, t), v in pooled_group.items()},
#         )

# plt.show()

# %%

# %%

import os

from scipy.stats import spearmanr


# --------------------------
# Helper: bootstrap Spearman correlation
# --------------------------
def bootstrap_spearman(x, y, n_resamples=1000, random_state=0):
    """Return mean Spearman rho and 95% CI via bootstrapping."""
    rng = np.random.default_rng(random_state)
    n = len(x)
    r_boot = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        r_boot[i], _ = spearmanr(x[idx], y[idx])
    return np.mean(r_boot), np.percentile(r_boot, [2.5, 97.5])


# --------------------------
# Helper: bootstrap difference between two correlations
# --------------------------
def bootstrap_diff(x1, y1, x2, y2, n_resamples=1000, random_state=0):
    rng = np.random.default_rng(random_state)
    n1, n2 = len(x1), len(x2)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx1 = rng.integers(0, n1, n1)
        idx2 = rng.integers(0, n2, n2)
        r1, _ = spearmanr(x1[idx1], y1[idx1])
        r2, _ = spearmanr(x2[idx2], y2[idx2])
        diffs[i] = r1 - r2
    return np.mean(diffs), np.percentile(diffs, [2.5, 97.5])


# %%
# Generate Δρ plots with bootstrapped CIs for all group pairs & segments
# --------------------------
# Main: Iterate over all group pairs
# --------------------------
savedir_corr_plots = paths["f_speed"] / "correlation_plots"
savedir_corr_plots.mkdir(parents=True, exist_ok=True)
results_dir = str(savedir_corr_plots)
os.makedirs(results_dir, exist_ok=True)

group_keys = list(group_data.keys())
pairs = list(combinations(group_keys, 2))

# Bootstrap Δρ plots
for speed_seg_name, speeds_ppsegment in speeds_percentile_per_segment.items():
    print(
        f"[INFO] Processing segment: {speed_seg_name} with shape {speeds_ppsegment.shape}"
    )

    for group_a, group_b in pairs:
        print(f"\n=== Comparing {group_a} vs {group_b} in pool {speed_seg_name} ===")

        idx_a = group_data[group_a]
        idx_b = group_data[group_b]

        # Skip if any group has too few animals
        if len(idx_a) < 3 or len(idx_b) < 3:
            print(f"⚠️ Skipping {group_a} vs {group_b} (too few animals)")
            continue

        diff_means, diff_ci_low, diff_ci_high = [], [], []

        for i in range(speeds_ppsegment.shape[0]):
            print("Processing percentile index:", i)
            y1 = speeds_ppsegment[i, idx_a]
            y2 = speeds_ppsegment[i, idx_b]
            x1 = nor_index[idx_a]
            x2 = nor_index[idx_b]
            mean_diff, (ci_low, ci_high) = bootstrap_diff(
                x1, y1, x2, y2, n_resamples=1000
            )
            diff_means.append(mean_diff)
            diff_ci_low.append(ci_low)
            diff_ci_high.append(ci_high)

        diff_means, diff_ci_low, diff_ci_high = map(
            np.array, (diff_means, diff_ci_low, diff_ci_high)
        )

        # Plot Δρ curve
        plt.figure(figsize=(10, 5))
        plt.plot(
            percentiles_,
            diff_means,
            lw=2,
            color="purple",
            label=f"{group_a} − {group_b}",
        )
        plt.fill_between(
            percentiles_, diff_ci_low, diff_ci_high, color="purple", alpha=0.3
        )
        plt.axhline(0, color="black", lw=1)

        # Highlight significant regions (CI excludes 0)
        plt.fill_between(
            percentiles_,
            diff_ci_low,
            diff_ci_high,
            where=(diff_ci_low > 0) | (diff_ci_high < 0),
            color="purple",
            alpha=0.2,
            label="Significant Δρ (95% CI excludes 0)",
        )

        plt.xlabel("Percentiles")
        plt.ylabel("Δ Spearman ρ (Group A − Group B)")
        plt.title(
            f"Difference in Spearman Correlation between {group_a} and {group_b}\nwith Bootstrapped 95% CI"
        )
        plt.legend()
        plt.tight_layout()

        # Save each figure automatically
        fname = f"delta_rho_{group_a[0]}_{group_a[1]}__vs__{group_b[0]}_{group_b[1]}_{speed_seg_name}.png".replace(
            "'", ""
        )
        plt.savefig(os.path.join(results_dir, fname), dpi=300)
        plt.close()

        print(f"✅ Saved plot → {fname}")


# %%
# Save summary CSV of Δρ results for all group pairs & segments


# Ensure results folder exists
resultdir_speed_delta_rho = paths["speed"] / "delta_rho_results"
resultdir_speed_delta_rho.mkdir(parents=True, exist_ok=True)

summary_data = []
for speed_seg_name, speeds_ppsegment in speeds_percentile_per_segment.items():
    print(
        f"[INFO] Processing segment: {speed_seg_name} with shape {speeds_ppsegment.shape}"
    )

    for group_a, group_b in combinations(group_data.keys(), 2):
        fname_base = f"{group_a[0]}_{group_a[1]}__vs__{group_b[0]}_{group_b[1]}_{speed_seg_name}".replace(
            "'", ""
        )
        filepath_png = os.path.join(
            resultdir_speed_delta_rho, f"delta_rho_{fname_base}.png"
        )
        filepath_npz = os.path.join(
            resultdir_speed_delta_rho, f"delta_rho_{fname_base}.npz"
        )

        # Skip if one group too small or not computed
        idx_a = group_data[group_a]
        idx_b = group_data[group_b]
        if len(idx_a) < 3 or len(idx_b) < 3:
            continue

        print(f"Processing Δρ for {group_a} vs {group_b}...")

        diff_means, diff_ci_low, diff_ci_high = [], [], []

        for i in range(speeds_ppsegment.shape[0]):
            y1 = speeds_ppsegment[i, idx_a]
            y2 = speeds_ppsegment[i, idx_b]
            x1 = nor_index[idx_a]
            x2 = nor_index[idx_b]
            mean_diff, (ci_low, ci_high) = bootstrap_diff(
                x1, y1, x2, y2, n_resamples=1000
            )
            diff_means.append(mean_diff)
            diff_ci_low.append(ci_low)
            diff_ci_high.append(ci_high)

        diff_means, diff_ci_low, diff_ci_high = map(
            np.array, (diff_means, diff_ci_low, diff_ci_high)
        )

        # Compute significance mask (CI excludes 0)
        sig_mask = (diff_ci_low > 0) | (diff_ci_high < 0)

        # Save to NPZ (for future replotting or analysis)
        np.savez(
            filepath_npz,
            percentiles=percentiles_,
            delta_rho_mean=diff_means,
            delta_rho_ci_low=diff_ci_low,
            delta_rho_ci_high=diff_ci_high,
            significant_mask=sig_mask,
            groupA=group_a,
            groupB=group_b,
        )

        # Summaries for CSV
        mean_delta = np.nanmean(diff_means)
        ci_global = (np.nanmin(diff_ci_low), np.nanmax(diff_ci_high))

        # Identify contiguous significant percentile ranges
        sig_ranges = []
        in_block = False
        start = None
        for i, val in enumerate(sig_mask):
            if val and not in_block:
                start = percentiles_[i]
                in_block = True
            elif not val and in_block:
                end = percentiles_[i - 1]
                sig_ranges.append(f"{start:.1f}-{end:.1f}")
                in_block = False
        if in_block:
            sig_ranges.append(f"{start:.1f}-{percentiles_[-1]:.1f}")
        sig_range_str = ", ".join(sig_ranges) if sig_ranges else "None"

        summary_data.append(
            {
                "Group A": f"{group_a[0]} {group_a[1]}",
                "Group B": f"{group_b[0]} {group_b[1]}",
                "Mean Δρ": f"{mean_delta:.3f}",
                "95% CI (min,max)": f"[{ci_global[0]:.3f}, {ci_global[1]:.3f}]",
                "Significant percentile ranges": sig_range_str,
                "NPZ file": os.path.basename(filepath_npz),
            }
        )

    # Build summary table and export to CSV
    summary_df = pd.DataFrame(summary_data)
    csv_path = os.path.join(results_dir, "delta_rho_summary.csv")
    summary_df.to_csv(csv_path, index=False)

    print(f"\n✅ Summary CSV saved to: {csv_path}")
    display(summary_df)

# %%
# Generate group-wise Spearman correlation plots with bootstrapped CIs & significance shading
import matplotlib.pyplot as plt
import numpy as np


# --------------------------
# Helper: bootstrap Spearman correlation
# --------------------------
def bootstrap_spearman(x, y, n_resamples=1000, random_state=0):
    """Return mean Spearman rho and 95% CI via bootstrapping."""
    rng = np.random.default_rng(random_state)
    n = len(x)
    r_boot = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        r_boot[i], _ = spearmanr(x[idx], y[idx])
    return np.mean(r_boot), np.percentile(r_boot, [2.5, 97.5])


# --------------------------
# Group-wise Spearman correlations
# --------------------------
for speed_seg_name, speeds_ppsegment in speeds_percentile_per_segment.items():
    print(
        f"[INFO] Processing segment: {speed_seg_name} with shape {speeds_ppsegment.shape}"
    )

    group_results = {}

    for (
        gt,
        idxs,
    ) in group_data.items():  # e.g. {"WT": [0,1,...], "Mut": [24,...]}
        r_means, ci_lows, ci_highs, p_vals = [], [], [], []

        for i in range(speeds_ppsegment.shape[0]):
            y = speeds_ppsegment[i, idxs]
            x = nor_index[idxs]
            if np.std(y) == 0:
                r_means.append(np.nan)
                ci_lows.append(np.nan)
                ci_highs.append(np.nan)
                p_vals.append(np.nan)
                continue

            r, p = spearmanr(x, y)
            r_mean, (r_low, r_high) = bootstrap_spearman(x, y, n_resamples=1000)
            r_means.append(r_mean)
            ci_lows.append(r_low)
            ci_highs.append(r_high)
            p_vals.append(p)

        group_results[gt] = {
            "r_means": np.array(r_means),
            "ci_lows": np.array(ci_lows),
            "ci_highs": np.array(ci_highs),
            "p_vals": np.array(p_vals),
        }

    # --------------------------
    # Plot group-wise results with significance shading
    # --------------------------
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10.colors  # distinct colors for groups

    for idx, (gt, res) in enumerate(group_results.items()):
        color = colors[idx % len(colors)]
        plt.plot(
            percentiles_, res["r_means"], color=color, lw=2, label=f"{gt} (ρ mean)"
        )
        plt.fill_between(
            percentiles_, res["ci_lows"], res["ci_highs"], color=color, alpha=0.25
        )

        # Shade non-significant regions (p > 0.05)
        plt.fill_between(
            percentiles_,
            res["ci_lows"],
            res["ci_highs"],
            where=(res["p_vals"] > 0.05),
            color=color,
            alpha=0.1,
            label=f"{gt} p > 0.05",
        )

    plt.axhline(0, color="black", lw=1)
    plt.xlabel("Percentiles")
    plt.ylabel("Spearman Correlation (ρ)")
    plt.title(
        "Spearman Correlation - NOR vs Speed \nwith Bootstrapped 95% CI and p>0.05 Shading"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()
    savedir_spearman = savedir_corr_plots / f"spearman_correlation_{speed_seg_name}.png"
    plt.savefig(savedir_spearman, dpi=300)

# %%
