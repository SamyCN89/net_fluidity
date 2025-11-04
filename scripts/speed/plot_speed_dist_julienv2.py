# %% ========================== IMPORTS & CONFIG ==========================

import time
from collections.abc import Iterable, Sequence
from datetime import datetime
import json
from math import e
from pathlib import Path
from turtle import st

import jinja2
import matplotlib.pyplot as plt
import numpy as np

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


# %% ========================== SMALL HELPERS ==========================
def combo_color(genotype: str, treatment: str) -> str:
    key = (genotype, treatment)
    table = {
        ("WT", "VEH"): "C0",
        ("WT", "LCTB92"): "C1",
        ("Dp1Yey", "VEH"): "C2",
        ("Dp1Yey", "LCTB92"): "C3",
    }
    return table.get(key, "gray")


def combo_label(genotype: str, treatment: str) -> str:
    return f"{genotype}_{treatment}"


# --- Load helpers ---
def load_speed_stack(
    paths_speed_root: Path, time_windows_range: Sequence[int]
) -> list[np.ndarray]:
    """Return list S where S[j][i] is 1D np.array of samples for animal i at window j."""
    speeds: list[np.ndarray] = []
    for w in time_windows_range:
        a = np.load(paths_speed_root.format(w=w, n_animals=n_animals, regions=regions), allow_pickle=True)
        s = a["speeds"]
        s_flat = [np.asarray(x, dtype=float).ravel() for x in s]
        speeds.append(s_flat)
    return speeds


def load_speed_stack2(
    paths_speed_root: Path, time_windows_range: Sequence[int], template: str
) -> list[np.ndarray]:
    """Return list S where S[j][i] is 1D np.array of samples for animal i at window j."""
    speeds: list[np.ndarray] = []
    for w in time_windows_range:
        a = np.load(paths_speed_root / template.format(w=w), allow_pickle=True)
        s = a["speeds"]
        s_flat = [np.asarray(x, dtype=float).ravel() for x in s]
        speeds.append(s_flat)
    return speeds


# --- pooling helpers ---


def count_samples_per_window(speeds: list[np.ndarray]) -> np.ndarray:
    return np.array([sum(len(x) for x in speed) for speed in speeds], dtype=int)


def cdf_split_indices(speeds: list[np.ndarray]) -> tuple[int, int, int]:
    counts = count_samples_per_window(speeds)
    cdf = (
        np.cumsum(counts) / counts.sum()
        if counts.sum() > 0
        else np.zeros_like(counts, dtype=float)
    )
    i_third = int(np.searchsorted(cdf, 1.0 / 3.0))
    i_half = int(np.searchsorted(cdf, 0.5))
    i_two_third = int(np.searchsorted(cdf, 2.0 / 3.0))
    i_third = max(1, i_third)
    i_half = max(1, i_half)
    i_two_third = max(i_third + 1, i_two_third)
    return i_third, i_half, i_two_third


def select_windows(
    pool_split: str, n_windows: int, i_third: int, i_half: int, i_two_third: int
) -> dict[str, range]:
    if pool_split == "all":
        return {"all": range(0, n_windows)}
    if pool_split == "half":
        return {"short": range(0, i_half), "long": range(i_half, n_windows)}
    return {
        "short": range(0, i_third),
        "mid": range(i_third, i_two_third),
        "long": range(i_two_third, n_windows),
    }


def flatten_windows(speeds: list[np.ndarray], start: int, end: int) -> np.ndarray:
    arrays = [
        np.asarray(s, dtype=float).ravel() for speed in speeds[start:end] for s in speed
    ]
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=float)


def global_min_max(arrs: Iterable[np.ndarray]) -> tuple[float, float]:
    vals_min = [np.nanmin(a) for a in arrs if a.size]
    vals_max = [np.nanmax(a) for a in arrs if a.size]
    vmin = min(vals_min) if vals_min else 0.0
    vmax = max(vals_max) if vals_max else 1.0
    if vmin == vmax:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


# --- stats & histogram helpers ---
def robust_percentiles(x: np.ndarray, qs=(1, 5, 95, 99)) -> dict[int, float]:
    if x.size == 0 or not np.isfinite(x).any():
        return {int(q): np.nan for q in qs}
    x = x[np.isfinite(x)]
    ps = np.percentile(x, qs)
    return {int(q): float(p) for q, p in zip(qs, ps, strict=False)}


def hist_prob(x, bins, rng):
    h, e = np.histogram(x, bins=bins, range=rng, density=False)
    s = h.sum()
    return (h / s if s > 0 else np.zeros_like(h)), e


def safe_hist(
    x: np.ndarray, bins: int, rng: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        edges = np.linspace(rng[0], rng[1], bins + 1)
        return np.zeros(bins), edges
    return np.histogram(x.T, bins=bins, range=rng, density=False)


def normalize_counts_to_prob(h: np.ndarray) -> np.ndarray:
    s = h.sum()
    return (h / s) if s > 0 else np.zeros_like(h, dtype=float)


def build_per_animal_normalized_hists(
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
    H = np.zeros((n_animals, bins), dtype=float)
    for i in range(n_animals):
        # Pool samples for animal i over selected windows
        flat_i = (
            np.concatenate([speeds[j][i].ravel() for j in selected_windows])
            if selected_windows
            else np.array([], dtype=float)
        )
        # Compute & normalize histogram
        h_i, _ = safe_hist(flat_i, bins, hist_range)
        H[i] = normalize_counts_to_prob(h_i)
    return H

#%%
def flatten_group_animals_over_windows(
    animal_speeds: list[list[np.ndarray]], indices: Sequence[int], w_range: range
) -> np.ndarray:
    arrays = []
    for i in indices:
        print(f"[DEBUG] Processing animal index: {i}")
        arrays.extend([animal_speeds[i][j] for j in w_range])
        print(f"[DEBUG] Current number of arrays for animal {i}: {len(arrays)}")
    print(f"[DEBUG] Flattened group animals over windows: n_arrays={len(arrays)}")
    return (
        np.concatenate([a.ravel() for a in arrays])
        if arrays
        else np.array([], dtype=float)
    )

def get_group_animals_over_windows(
    animal_speeds: list[list[np.ndarray]],
    indices: Sequence[int],
    w_range: range
) -> np.ndarray:
    """Return object array shape (len(indices), len(w_range)) of arrays."""
    arr = np.array(animal_speeds, dtype=object)[indices]
    arrays = np.transpose([arr[:,j] for j in w_range], (1,0))
    return (arrays,) if arrays.size else np.array([], dtype=float)
#%%
def plot_group_median_vs_window(
    ax: plt.Axes,
    time_windows_range: Sequence[int],
    group_data: dict[tuple[str, str], Sequence[int]],
    speeds: list[np.ndarray],
) -> None:
    """Mean of per-animal medians vs window (group curve)."""
    for (genotype, treatment), indices in group_data.items():
        y = []
        for j in range(len(time_windows_range)):
            per_animal_medians = [float(np.median(speeds[j][i])) for i in indices]
            y.append(
                float(np.mean(per_animal_medians)) if per_animal_medians else np.nan
            )
        ax.plot(time_windows_range, y, ".-", label=combo_label(genotype, treatment))
    ax.set_xlabel("Time Window Size")
    ax.set_ylabel("Mean of per-animal medians (dFC speed)")
    ax.set_title("dFC Speed vs Window per Genotype–Treatment")
    ax.legend()


def plot_group_histograms(
    ax: plt.Axes,
    centers: np.ndarray,
    group_hists: dict[tuple[str, str], np.ndarray],
    title: str,
    ylog: bool = False,
) -> None:
    for (genotype, treatment), hist in group_hists.items():
        ax.plot(centers, hist, lw=1, alpha=0.7, label=combo_label(genotype, treatment))
    ax.set_title(title)
    ax.set_xlabel("Speed")
    ax.set_ylabel("Probability per bin")
    if ylog:
        ax.set_yscale("log")
    # ax.legend()


# %%

# =============================================================================
# -----------------------------------------------------------------------------
# ============================= Main Code ==============================
# -----------------------------------------------------------------------------
# =============================================================================


# %% ========================== LOAD DATA ==========================
dataset_name = "ines"
# dataset_name = "julien"
save_fig = set_figure_params(True)
dataset = _canonical_dataset(dataset_name)
cfg = DATASET_DEFAULTS[dataset]

time_windows_range = np.arange(5, 100, 1)

POOL_SPLIT = "third"  # 'half' | 'third' | 'all'
BINS_HIST = 200

SAVE_GROUP_HISTS = True
# %% ============= Get paths & output locations ====================
paths = get_paths(
    dataset_name=dataset,
    timecourse_folder=cfg["timecourse_folder"],
    cognitive_data_file=cfg["cognitive_data_file"],
    anat_labels_file=cfg["anat_labels_file"],
)
# Root locations
speed_root = Path(paths["speed"])
preprocessed_root = Path(paths["preprocessed"])

# Load location
loaddir_ts_meta = preprocessed_root / "ts_and_meta_2m4m.npz"
loaddir_cog_data = str(
    preprocessed_root
    / "cog_data_filtered_animals_{n_animals}_regions_{regions}_tr_{total_tr}.csv"
)
# loaddir_speed = str(speed_root / "all/all/speed_win{w}_lag1_tau4_animals_48_regions_37.npz")
# loaddir_speed = str(speed_root / "dmn_within/nregs-6/speed_win{w}_lag1_tau4_animals_{n_animals}_regions_{regions}.npz")
loaddir_speed = str(speed_root / "all/all/speed_win{w}_lag1_tau4_animals_{n_animals}_regions_{regions}.npz")

# Output location for group histograms
# outdir_save_group_hists = speed_root / f"{dataset}_pool_{POOL_SPLIT}_bins{BINS_HIST}" / f"pooled_group_hists__{seg_name}.npz"
# bootstrap CI folder
bootstrap_folder = paths["speed"] / "bootstrap"
bootstrap_folder.mkdir(parents=True, exist_ok=True)

outdir_speed_bootstrap_cis = str(bootstrap_folder / f"bootstrap_cis_{POOL_SPLIT}_bins{BINS_HIST}.npz")

# Savedir location
# time window plots
time_window_folder = paths["f_speed"] / "time_windows"
time_window_folder.mkdir(parents=True, exist_ok=True)


savedir_dfc_speed_per_animal = str(
    time_window_folder
    / "dFC_speed_per_animal_{n_animals}_regions_{regions}_tr_{total_tr}.png"
)
savedir_dfc_speed_group_median_vs_window = str(
    time_window_folder
    / "dFC_speed_group_median_vs_window_{n_animals}_regions_{regions}_tr_{total_tr}.png"
)
savedir_dfc_speed_percentiles_vs_window = str(
    time_window_folder
    / "dFC_speed_percentiles_vs_window_{n_animals}_regions_{regions}_tr_{total_tr}.png"
)

# pooling plots
pooling_folder = paths["f_speed"] / "pooling"
pooling_folder.mkdir(parents=True, exist_ok=True)

savedir_dfc_speed_cdf_windows = str(
    pooling_folder
    / "dFC_speed_cdf_windows_animals_{n_animals}_regions_{regions}_tr_{total_tr}.png"
)
savedir_pooled_speed_hist_bins = str(
    pooling_folder
    / "pooled_speed_hist_bins{BINS_HIST}_animals_{n_animals}_regions{regions}_tr{total_tr}.png"
)
savedir_pooled_group_hists = str(
    speed_root
    / "pooled_group_hists_{POOL_SPLIT}_bins{BINS_HIST}_animals_{n_animals}_regions{regions}_tr{total_tr}.png"
)
# %% ========================== LOAD DATA ==========================
# Load timeseries bundle to get n_animals, n_regions, total_tr
bundle = load_timeseries_bundle(loaddir_ts_meta)
n_animals = bundle.n_animals
total_tr = bundle.total_tr
regions = bundle.n_regions

# Load cognitive data for grouping
cog_data = load_cognitive_data(
    loaddir_cog_data.format(n_animals=n_animals, regions=regions, total_tr=total_tr)
)

# Load speed data
speeds = load_speed_stack(
    loaddir_speed,
    time_windows_range,
)

# %% ========================== GROUP INDICES ==========================
# Group indices from cognitive data
if dataset_name == "julien":
    group_treatment = cog_data.groupby("treatment").groups
    group_genotype = cog_data.groupby("genotype").groups
    group_genotype_treatment = cog_data.groupby(["genotype", "treatment"]).groups
    group_data = group_genotype_treatment

elif dataset_name == "ines":
    df_2m = cog_data[['Name','Sexe','Genotype','OiP_2M','RO24h_2M','TC_2M',
            'Phenotype_OiP','Phenotype_RO24h']].copy()
    df_2m['Age'] = '2M'
    df_2m = df_2m.rename(columns={'OiP_2M':'oip','RO24h_2M':'ro24h','TC_2M':'tc'})

    df_4m = cog_data[['Name','Sexe','Genotype','OiP_4M','RO24h_4M','TC_4M',
                'Phenotype_OiP','Phenotype_RO24h']].copy()
    df_4m['Age'] = '4M'
    df_4m = df_4m.rename(columns={'OiP_4M':'oip','RO24h_4M':'ro24h','TC_4M':'tc'})

    df_long = pd.concat([df_2m, df_4m], ignore_index=True)

    df_long['Sexe'] = df_long['Sexe'].map({'F':'female','M':'male'})

    group_sexe = df_long.groupby('Sexe').groups
    group_age = df_long.groupby('Age').groups
    group_genotype = df_long.groupby('Genotype').groups
    group_phenotype_oip = df_long.groupby('Phenotype_OiP').groups
    group_phenotype_nor = df_long.groupby('Phenotype_RO24h').groups

    group_sexe_age = df_long.groupby(['Sexe', 'Age']).groups
    group_sexe_genotype = df_long.groupby(['Sexe', 'Genotype']).groups
    group_sexe_phenotype_oip = df_long.groupby(['Sexe', 'Phenotype_OiP']).groups
    group_sexe_phenotype_nor = df_long.groupby(['Sexe', 'Phenotype_RO24h']).groups

    group_age_genotype = df_long.groupby(['Age', 'Genotype']).groups
    group_age_phenotype_oip = df_long.groupby(['Age', 'Phenotype_OiP']).groups
    group_age_phenotype_nor = df_long.groupby(['Age', 'Phenotype_RO24h']).groups

    group_sexe_age_genotype = df_long.groupby(['Sexe', 'Age', 'Genotype']).groups
    group_sexe_age_phenotype_oip = df_long.groupby(['Sexe', 'Age', 'Phenotype_OiP']).groups
    group_sexe_age_phenotype_nor = df_long.groupby(['Sexe', 'Age', 'Phenotype_RO24h']).groups
    group_data = group_sexe_age_genotype

#%%


#%%
# Basic dimensions
n_windows = len(speeds)
n_animals = len(speeds[0])

# Precompute mean speed for each animal at each window (handy for mean-based bootstrap later)
per_window_animal_means = [
    np.array([float(np.mean(speeds[j][i])) for i in range(n_animals)], dtype=float)
    for j in range(n_windows)
]

print(
    "[INFO] Data loaded. n_animals:",
    n_animals,
    "n_windows:",
    n_windows,
    "per_window_animal_means[0].shape:",
    per_window_animal_means[0].shape,
)
# %% ========================== SPLITS & HIST SETUP ==========================
# Get the split indices and ranges
counts = count_samples_per_window(speeds)
pooled_speeds_cdf = (
    np.cumsum(counts) / np.sum(counts) if counts.sum() else np.zeros_like(counts)
)
i_third, i_half, i_two_third = cdf_split_indices(speeds)
ranges = select_windows(
    POOL_SPLIT, len(time_windows_range), i_third, i_half, i_two_third
)
# Get global min/max for histogram binning
all_speeds_flat = flatten_windows(speeds, 0, len(speeds))
all_speeds_min, all_speeds_max = global_min_max([all_speeds_flat])
edges = np.linspace(all_speeds_min, all_speeds_max, BINS_HIST + 1)
centers = 0.5 * (edges[:-1] + edges[1:])

# %% ========================== PER-SEGMENT HISTOGRAMS ==========================
# Per-animal histograms & per-group mean histograms
H_per_segment: dict[str, np.ndarray] = {}
group_means_by_segment: dict[str, dict[tuple[str, str], np.ndarray]] = {}

# Build per-animal normalized histograms & per-group means
for seg_name, w_range in ranges.items():
    # per-animal normalized histograms
    H = build_per_animal_normalized_hists(
        speeds, w_range, BINS_HIST, (all_speeds_min, all_speeds_max)
    )
    H_per_segment[seg_name] = H * 2
    # per-group average histogram over animals
    group_means: dict[tuple[str, str], np.ndarray] = {}
    for gt, idxs in group_data.items():
        group_means[gt] = np.mean(H[idxs], axis=0) if len(idxs) else np.zeros(BINS_HIST)
    group_means_by_segment[seg_name] = group_means

#%% ========================== POOLING & BOOTSTRAP ==========================







# Optional pooled hist per group (values aggregated across animals & windows)
animal_speeds = [[speeds[j][i] for j in range(n_windows)] for i in range(n_animals)]
pooled_group_hists_by_segment: dict[str, dict[tuple[str, str], np.ndarray]] = {}
pooled_group_speed_by_segment: dict[str, dict[tuple[str, str], np.ndarray]] = {}
group_speed_by_segment = {}

for seg_name, w_range in ranges.items():
    pooled_group_hist_i = {}
    pooled_group_speed_i = {}
    group_speed = {}
    for gt, idxs in group_data.items():
        # Get group animal speeds over selected windows
        group_speed_i = get_group_animals_over_windows(animal_speeds, idxs, w_range)
        group_speed[gt] = group_speed_i
        # Flatten values for group animals over selected windows
        flat = flatten_group_animals_over_windows(animal_speeds, idxs, w_range)
        pooled_group_speed_i[gt] = flat
        # Compute & normalize histogram
        h, _ = safe_hist(flat, BINS_HIST, (all_speeds_min, all_speeds_max))
        pooled_group_hist_i[gt] = normalize_counts_to_prob(h) * 2
    pooled_group_hists_by_segment[seg_name] = pooled_group_hist_i
    pooled_group_speed_by_segment[seg_name] = pooled_group_speed_i
    group_speed_by_segment[seg_name] = group_speed

#%%

#%%



def bootstrap_chunk(data, percentiles_, n_resamples_chunk):
    idx = rng.integers(0, len(data), size=(n_resamples_chunk, len(data)))
    samples = data[idx]
    return np.percentile(samples, percentiles_, axis=0)

def bootstrap_percentiles(data, percentiles_, n_resamples=2000, chunk_size=200, jobs=-1):
    n_chunks = int(np.ceil(n_resamples / chunk_size))
    results = Parallel(n_jobs=jobs)(
        delayed(bootstrap_chunk)(data, percentiles_, chunk_size)
        for _ in range(n_chunks)
    )
    boot_all = np.vstack(results)
    low, high = np.percentile(boot_all, [2.5, 97.5], axis=0)
    return low, high
#%%
from joblib import Parallel, delayed

def flatten_numeric(x) -> np.ndarray:
    """Flatten lists/arrays (even nested) into a clean 1D float array; drop NaN/inf."""
    if isinstance(x, (list, tuple)):
        flat = []
        for item in np.ravel(x, order="K"):
            a = np.asarray(item)
            if a.size:
                flat.append(a.ravel())
        x = np.concatenate(flat) if flat else np.array([])
    x = np.asarray(x, dtype=float).ravel()
    return x[np.isfinite(x)]

def _bootstrap_chunk_proc(x: np.ndarray, q: np.ndarray, m: int, seed: int) -> np.ndarray:
    """
    One parallel chunk.
    x: (n,), q: (K,), m replicates → returns (m, K)
    """
    if m <= 0 or x.size == 0:
        return np.full((0, q.size), np.nan, float)
    rng = np.random.default_rng(seed)
    n = x.size
    # replicates as columns to avoid axis confusion
    idx = rng.integers(0, n, size=(n, m))   # (n, m)
    samples = x[idx]                         # (n, m)
    return np.percentile(samples, q, axis=0).T  # (m, K)



def bootstrap_parallel(
    data,
    percentiles,
    n_resamples: int = 2000,
    chunk_size:   int = 200,
    n_jobs:       int = -1,
    random_state: int | None = 0,
    downsample_n: int | None = None,
    prefer:       str = "processes",  # "threads" also allowed
):
    """
    Parallel 95% CIs for given percentiles of `data`.
    Returns (low, high), each shape (len(percentiles),).
    """
    # print random state

    x = flatten_numeric(data)
    q = np.asarray(percentiles, dtype=float).ravel()

    if x.size == 0:
        nan = np.full(q.shape, np.nan, float); return nan, nan

    # optional downsampling
    if downsample_n and x.size > downsample_n:
        x = np.random.default_rng(random_state).choice(x, size=downsample_n, replace=False)

    # plan chunks & deterministic seeds (safe for parallel)
    offsets = list(range(0, n_resamples, chunk_size))
    ss = np.random.SeedSequence(random_state) if random_state is not None else np.random.SeedSequence()
    seeds = [int(s.entropy) for s in ss.spawn(len(offsets))]


    # run chunks in parallel
    chunks = Parallel(n_jobs=n_jobs, prefer=prefer)(
        delayed(_bootstrap_chunk_proc)(
            x, q, min(chunk_size, n_resamples - off), seeds[i]
        )
        for i, off in enumerate(offsets)
    )

    # combine results
    boot_all = np.vstack(chunks)                         # (n_resamples, len(q))
    return boot_all

def bootstrap_percentiles_parallel(
    data,
    percentiles,
    n_resamples: int = 2000,
    chunk_size:   int = 200,
    n_jobs:       int = -1,
    random_state: int | None = 0,
    downsample_n: int | None = None,
    prefer:       str = "processes",  # "threads" also allowed
):
    """
    Parallel 95% CIs for given percentiles of `data`.
    Returns (low, high), each shape (len(percentiles),).
    """
    boot_all = bootstrap_parallel(
        data,
        percentiles,
        n_resamples=n_resamples,
        chunk_size=chunk_size,
        n_jobs=n_jobs,
        random_state=random_state,
        downsample_n=downsample_n,
        prefer=prefer,
    )
    low, high = np.percentile(boot_all, [2.5, 97.5], axis=0)
    return low, high
#%%
# ---------- CI SAVE HELPERS ----------
def _gt_key(gt: tuple[str, str]) -> str:
    return f"{gt[0]}__{gt[1]}"

def save_bootstrap_cis_npz(
    base_dir: Path,
    dataset: str,
    meta: dict,
    percentiles: np.ndarray,
    ci_low_mean: dict[str, dict[tuple[str, str], np.ndarray]],
    ci_high_mean: dict[str, dict[tuple[str, str], np.ndarray]],
    ci_low_btr: dict[str, dict[tuple[str, str], np.ndarray]],
    ci_high_btr: dict[str, dict[tuple[str, str], np.ndarray]],
    ci_low_btr_downsample: dict[str, dict[tuple[str, str], np.ndarray]],
    ci_high_btr_downsample: dict[str, dict[tuple[str, str], np.ndarray]],
):
    """
    Writes one NPZ per segment with keys:
      - 'percentiles'
      - '{group}__hier_low', '{group}__hier_high'
      - '{group}__classic_low', '{group}__classic_high'
      - '{group}__classic_down_low', '{group}__classic_down_high'
    """
    out_dir = base_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # store meta snapshot for the CI pack (won’t overwrite your other meta)
    (out_dir / "meta_ci.json").write_text(json.dumps(meta, indent=2))

    # write one file per segment
    seg_names = sorted(set(ci_low_mean.keys()) |
                       set(ci_high_mean.keys()) |
                       set(ci_low_btr.keys()) |
                       set(ci_high_btr.keys()) |
                       set(ci_low_btr_downsample.keys()) |
                       set(ci_high_btr_downsample.keys()))

    for seg in seg_names:
        pack = {"percentiles": np.asarray(percentiles, dtype=float)}

        # helper to add a whole dict-of-groups with a prefix
        def add_side(side_dict: dict[str, dict[tuple[str, str], np.ndarray]],
                     suffix: str):
            if seg not in side_dict:
                return
            for gt, arr in side_dict[seg].items():
                pack[f"{_gt_key(gt)}__{suffix}"] = np.asarray(arr, dtype=float)

        add_side(ci_low_mean, "hier_low")
        add_side(ci_high_mean, "hier_high")
        add_side(ci_low_btr, "classic_low")
        add_side(ci_high_btr, "classic_high")
        add_side(ci_low_btr_downsample, "classic_down_low")
        add_side(ci_high_btr_downsample, "classic_down_high")

        np.savez_compressed(out_dir / f"cis__{seg}.npz", **pack)
    print(f"[INFO] Saved bootstrap CI NPZ packs to: {out_dir} , cis__{seg}.npz")

def save_bootstrap_cis_parquet(
    base_dir: Path,
    dataset: str,
    meta: dict,
    percentiles: np.ndarray,
    ci_low_mean: dict[str, dict[tuple[str, str], np.ndarray]],
    ci_high_mean: dict[str, dict[tuple[str, str], np.ndarray]],
    ci_low_btr: dict[str, dict[tuple[str, str], np.ndarray]],
    ci_high_btr: dict[str, dict[tuple[str, str], np.ndarray]],
    ci_low_btr_downsample: dict[str, dict[tuple[str, str], np.ndarray]],
    ci_high_btr_downsample: dict[str, dict[tuple[str, str], np.ndarray]],
):
    """
    Long-form Parquet with columns:
      segment, group, percentile, method, bound, value
    where method ∈ {'hier', 'classic', 'classic_down'}
          bound  ∈ {'low', 'high'}
    """
    if pd is None:
        print("[WARN] pandas not available; skipping CI parquet save.")
        return

    out_dir = base_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    def add_rows(container: dict[str, dict[tuple[str, str], np.ndarray]],
                 method: str, bound: str):
        for seg, per_group in container.items():
            for gt, arr in per_group.items():
                g = _gt_key(gt)
                for p, v in zip(percentiles, np.asarray(arr, dtype=float), strict=False):
                    rows.append({
                        "segment": seg,
                        "group": g,
                        "percentile": float(p),
                        "method": method,
                        "bound": bound,
                        "value": float(v),
                    })

    add_rows(ci_low_mean, "hier", "low")
    add_rows(ci_high_mean, "hier", "high")
    add_rows(ci_low_btr, "classic", "low")
    add_rows(ci_high_btr, "classic", "high")
    add_rows(ci_low_btr_downsample, "classic_down", "low")
    add_rows(ci_high_btr_downsample, "classic_down", "high")

    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "cis_longform.parquet", index=False)
    print(f"[INFO] Saved bootstrap CI Parquet to: {out_dir} , cis_longform.parquet")

#%%

rng = np.random.default_rng(0)
n_resamples = 5_000
chunk_size = 5  # resamples per chunk to limit memory
downsample_n = 100_000  # cap sample size per group
ci_low_mean = {}
ci_high_mean = {}
for seg_name, w_range in ranges.items():
    start = time.time()
    pooled_group_hist_i = {}
    pooled_group_speed_i = {}
    ci_low_mean_i = {}
    ci_high_mean_i = {}
    for gt, idxs in group_data.items():
        idxs = group_data[gt]
        print('Speed shape:', np.shape(group_speed_by_segment[seg_name][gt][0]), seg_name, gt)
        group_speed_i = group_speed_by_segment[seg_name][gt][0]
        group_speed_n = len(group_speed_i)

        #resampling indices
        repeat = 20
        percentiles_ = np.linspace(0, 100, 100)
        ci_low = np.empty((len(percentiles_), repeat), dtype=float)
        ci_high = np.empty((len(percentiles_), repeat), dtype=float)
        for _ in range(repeat):  # repeat to see variability
            n_hierarchical_resampling = 8
            indices_resampling = np.random.choice(group_speed_n, size=n_hierarchical_resampling, replace=False)
            print(indices_resampling)

            # bootstrap samples
            flat_list = np.array(np.concatenate(group_speed_i[indices_resampling].ravel()).tolist())
            # np.array([group_speed_i[ii][j] for ii in indices_resampling for j in range(len(group_speed_i[0]))]).ravel()

            # bootstrap percentiles
            # percentiles_ = np.linspace(0, 100, 100)
            # ci_low, ci_high = bootstrap_percentiles(flat_list, percentiles_, n_resamples, chunk_size=chunk_size, jobs=8)
            ci_low_i, ci_high_i = bootstrap_percentiles_parallel(flat_list, percentiles_,
                                                            n_resamples=n_resamples,
                                                            chunk_size=chunk_size,
                                                            random_state=0,
                                                            n_jobs=8,
                                                            downsample_n=downsample_n)
            ci_low[:, _] = ci_low_i
            ci_high[:, _] = ci_high_i
        ci_low_mean_i[gt] = ci_low.mean(axis=1)
        ci_high_mean_i[gt] = ci_high.mean(axis=1)
    end = time.time()
    print(f"Hierarchical BT resampling for segment {seg_name} took {end - start:.2f} seconds")
    ci_low_mean[seg_name] = ci_low_mean_i
    ci_high_mean[seg_name] = ci_high_mean_i


#%%
# Classic resampling of distribution: Assumes that the distributions of a group's speeds are identical across animals

def group_pool_bt_classic_resampling(ranges,
                                     pooled_group_speed_by_segment,
                                     group_data,
                                     seed=np.random.default_rng(),
                                     downsample_n=None):
    """Classic bootstrap resampling of pooled group speeds."""
    # Bootstrap per group speed histograms

    # pooling loops segments
    ci_low_mean = {}
    ci_high_mean = {}

    for seg_name, w_range in ranges.items():
        # Group loops
        ci_low_i = {}
        ci_high_i = {}
        for gt, idxs in group_data.items():
            print(pooled_group_speed_by_segment[seg_name][gt].shape, seg_name, gt)
            group_flat_speed = pooled_group_speed_by_segment[seg_name][gt]

            data = np.ravel(group_flat_speed)
            start = time.time()
            ci_low, ci_high = bootstrap_percentiles_parallel(data,
                                                            percentiles_,
                                                            n_resamples=n_resamples,
                                                            chunk_size=chunk_size,
                                                            random_state=seed,
                                                            n_jobs=8,
                                                            downsample_n=downsample_n)
            end = time.time()

            # Print timing
            if not downsample_n:
                print(f"Classic BT resampling took {end - start:.2f} seconds: {seg_name} {gt}")
            else:
                print(f"Downsampled Classic BT resampling took {end - start:.2f} seconds: {seg_name} {gt}")
            # Store results per group
            ci_low_i[gt] = ci_low
            ci_high_i[gt] = ci_high
        #store results per segment
        ci_low_mean[seg_name] = ci_low_i
        ci_high_mean[seg_name] = ci_high_i
    return ci_low_mean, ci_high_mean

#%%
downsample_n = 150_000

# Classic bootstrap resampling
ci_low_btr, ci_high_btr = group_pool_bt_classic_resampling(ranges,
                                 pooled_group_speed_by_segment,
                                 group_data)
# Downsampled classic bootstrap resampling
ci_low_btr_downsample, ci_high_btr_downsample = group_pool_bt_classic_resampling(ranges,
                                 pooled_group_speed_by_segment,
                                 group_data,
                                 downsample_n=downsample_n)

#%%
# ----- SAVE CI ARTIFACTS -----
# base_dir = Path(paths["speed"])  # you already define this earlier
# enrich meta a tiny bit so the CI pack is reproducible
meta_ci = dict(meta)
meta_ci.update({
    "percentiles_len": int(len(percentiles_)),
})

# Always save NPZ; parquet depends on SAVE_MODE['parquet']
save_bootstrap_cis_npz(
    Path(outdir_speed_bootstrap_cis),
    dataset,
    meta_ci,
    percentiles_,
    ci_low_mean,
    ci_high_mean,
    ci_low_btr,
    ci_high_btr,
    ci_low_btr_downsample,
    ci_high_btr_downsample,
)

if SAVE_MODE.get("parquet", False):
    save_bootstrap_cis_parquet(
        Path(outdir_speed_bootstrap_cis),
        dataset,
        meta_ci,
        percentiles_,
        ci_low_mean,
        ci_high_mean,
        ci_low_btr,
        ci_high_btr,
        ci_low_btr_downsample,
        ci_high_btr_downsample,
    )

print("[INFO] CI save completed.")

#%%

# Playground to compare the different CIs





seg_name = 'short'
gt = ('female','2M','dKI')

# ci_low_mean_i = ci_low_mean[seg_name][gt]
# ci_high_mean_i = ci_high_mean[seg_name][gt]

# --- usage ---
group_flat_speed = pooled_group_speed_by_segment[seg_name][gt]
data = np.ravel(group_flat_speed)

start = time.time()
ci_low_btr2, ci_high_btr2 = bootstrap_percentiles_parallel(data, percentiles_,n_resamples=n_resamples,
                                                         chunk_size=chunk_size,
                                                         random_state=np.random.randint(0,1_000_000,n_resamples),
                                                         n_jobs=8)
end = time.time()
print(f"Classic BT resampling took {end - start:.2f} seconds")

start = time.time()
ci_low_btr2_downsample, ci_high_btr2_downsample = bootstrap_percentiles_parallel(data, percentiles_,n_resamples=n_resamples,
                                                         chunk_size=chunk_size,
                                                         random_state=np.random.randint(0,1_000_000,n_resamples),
                                                         n_jobs=8,
                                                         downsample_n= len(data)//5)
end = time.time()
print(f"Downsampled Classic BT resampling took {end - start:.2f} seconds")
#%%
#Repeat for downsampled

repeat = 60

ci_low_btr2_downsample_repeats = np.empty((len(percentiles_), repeat), dtype=float)
ci_high_btr2_downsample_repeats = np.empty((len(percentiles_), repeat), dtype=float)

for rp in range(repeat):
    print(f"Repeat {rp+1}/{repeat}")
    start = time.time()
    ci_low_btr2_downsample_rp, ci_high_btr2_downsample_rp = bootstrap_percentiles_parallel(data, percentiles_,n_resamples=n_resamples,
                                                            chunk_size=chunk_size,
                                                            random_state=rp,
                                                            n_jobs=8,
                                                            downsample_n= len(data)//5)
    ci_low_btr2_downsample_repeats[:, rp] = ci_low_btr2_downsample_rp
    ci_high_btr2_downsample_repeats[:, rp] = ci_high_btr2_downsample_rp
    end = time.time()
    print(f"Downsampled Classic BT resampling repeat {rp+1}/{repeat} took {end - start:.2f} seconds")


#%%
def bootstrap_single(
    data,
    percentiles,
    n_resamples: int = 2000,
    chunk_size:   int = 200,
    n_jobs:       int = -1,
    random_state: int | None = 0,
    downsample_n: int | None = None,
    prefer:       str = "processes",  # "threads" also allowed
):
    """
    Parallel 95% CIs for given percentiles of `data`.
    Returns (low, high), each shape (len(percentiles),).
    """
    # print random state

    x = flatten_numeric(data)
    q = np.asarray(percentiles, dtype=float).ravel()

    if x.size == 0:
        nan = np.full(q.shape, np.nan, float); return nan, nan

    # optional downsampling
    if downsample_n and x.size > downsample_n:
        x = np.random.default_rng(random_state).choice(x, size=downsample_n, replace=False)

    # plan chunks & deterministic seeds (safe for parallel)
    offsets = list(range(0, n_resamples, chunk_size))
    ss = np.random.SeedSequence(random_state) if random_state is not None else np.random.SeedSequence()
    seeds = [int(s.entropy) for s in ss.spawn(len(offsets))]


    # run chunks in parallel
    chunks = _bootstrap_chunk_proc(x, q, min(chunk_size, n_resamples), seeds[i])

    # combine results
    boot_all = np.vstack(chunks)                         # (n_resamples, len(q))
    return boot_all

#%%
# Repeat for downsampled



def bootstrap_repeat_downsampling(
    data: np.ndarray,
    percentiles: np.ndarray,
    repeat: int = 10_000,
    n_resamples: int = 2,
    chunk_size: int = 5,
    n_jobs: int = 8,
    downsample_n: int | None = None,
    seed: int = 0,):

    ci_btr_downsample_repeat = np.empty((repeat, len(percentiles_)), dtype=float)
    start = time.time()
    for rp in range(repeat):
        print(f"Repeat {rp+1}/{repeat}")
        ci_btr_downsample_repeat[rp], _ = bootstrap_parallel(data, percentiles_,n_resamples=2,
                                                                chunk_size=chunk_size,
                                                                random_state= np.random.randint(0,1_000_000),
                                                                n_jobs=8,
                                                                downsample_n= len(data)//10)
        # ci_low_btr2_downsample_repeats2[:, rp] = ci_low_btr2_downsample_rp
        # ci_high_btr2_downsample_repeats2[:, rp] = ci_high_btr2_downsample_rp
    end = time.time()
    print(f"Downsampled Classic BT resampling repeat {rp+1}/{repeat} took {end - start:.2f} seconds")

    ci_low_btr2_downsample_repeat, ci_high_btr2_downsample_repeat =np.percentile(ci_btr_downsample_repeat, [0,100], axis=0)
    return ci_low_btr2_downsample_repeat, ci_high_btr2_downsample_repeat, ci_btr_downsample_repeat

ci_low_btr_downsample_repeat  = {}
ci_high_btr_downsample_repeat = {}
ci_btr_downsample_repeat = {}
ci_low_btr_downsample_repeat: dict[str, dict[tuple[str, str], np.ndarray]] = {}
ci_high_btr_downsample_repeat: dict[str, dict[tuple[str, str], np.ndarray]] = {}
vals_btr_downsample_repeat: dict[str, dict[tuple[str, str], np.ndarray]] = {}

for seg_name, w_range in ranges.items():
    # Group loops
    ci_low_seg: dict[tuple[str, str], np.ndarray] = {}
    ci_high_seg: dict[tuple[str, str], np.ndarray] = {}
    vals_seg: dict[tuple[str, str], np.ndarray] = {}

    for gt, idxs in group_data.items():  # <-- use your correct grouping dict
        print('Speed shape:', np.shape(group_speed_by_segment[seg_name][gt][0]), seg_name, gt)
        group_flat_speed = pooled_group_speed_by_segment[seg_name][gt]
        print('Flat speed shape:', np.shape(group_flat_speed), seg_name, gt)

        data = np.ravel(group_flat_speed)
        lo, hi, vals = bootstrap_repeat_downsampling(
            data=data,
            percentiles=percentiles_,
            repeat=10_000,
            n_resamples=2,
            chunk_size=chunk_size,
            n_jobs=8,
            downsample_n=len(data)//10,
            seed=0,
        )
        ci_low_seg[gt] = lo
        ci_high_seg[gt] = hi
        vals_seg[gt] = vals

    # store dicts for the whole segment
    ci_low_btr_downsample_repeat[seg_name]  = ci_low_seg
    ci_high_btr_downsample_repeat[seg_name] = ci_high_seg
    vals_btr_downsample_repeat[seg_name]    = vals_seg

# ci_low_mean_i = ci_low_mean[seg_name][gt]
# ci_high_mean_i = ci_high_mean[seg_name][gt]

# --- usage ---



#%%
#plot to test the idea confidence intervals comparison
# plt.figure(figsize=(8, 5))
# plt.plot(ci_btr_downsample_repeat.T, percentiles_[:, np.newaxis], alpha=0.1)
# # plt.colorbar(label='Percentile Value')
# plt.ylabel('Percentiles Index')
# plt.xlabel('Speed')
# plt.yscale('log')
# low, high = np.percentile(boot_all, [2.5, 97.5], axis=0)
#%%

#plot confidence intervals comparison
plt.figure(figsize=(16, 8))

for seg_name, w_range in ranges.items():
    plt.subplot(1, len(ranges), list(ranges.keys()).index(seg_name)+1)
    plt.title(f'Confidence Intervals Comparison - {seg_name}')
    for gt in group_data.keys():
        print('Plotting GT:', gt)
        plt.fill_between(percentiles_,
                    ci_low_btr_downsample_repeat[seg_name][gt],
                    ci_high_btr_downsample_repeat[seg_name][gt],
                    alpha=0.5, label=gt)

    # plt.plot(percentiles_, group_flat_speed_perc, label='Pooled Group Histogram')
    plt.xlabel('Percentiles')
    plt.ylabel('Speed')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(2e-1,1.5e0)
    # plt.xlim(0,3)

    # plt.xscale('log')
    plt.legend()
plt.show()
#%%



for seg_name, w_range in ranges.items():
    # Group loops
    ci_low_i = {}
    ci_high_i = {}
    for gt, idxs in group_data.items():
        print(pooled_group_speed_by_segment[seg_name][gt].shape, seg_name, gt)
        group_flat_speed = pooled_group_speed_by_segment[seg_name][gt]


        #for classic resampling of distribution
        group_flat_speed_perc = np.percentile(group_flat_speed, percentiles_)


        #for each animal histogram
        animal_flat_speed_perc = np.empty((len(group_speed_by_segment[seg_name][gt][0]), len(percentiles_)), dtype=float)

        group_speed_by_segment_i = group_speed_by_segment[seg_name][gt][0]
        plt.figure(figsize=(12, 10))
        for i in range(len(group_speed_by_segment_i)):
            # print(f"Animal {i} speed shape: {np.shape(group_speed_by_segment_i[i])}")
            aux_animal_s = group_speed_by_segment_i[i]
            aux_animal_i_s_flat = [aux_animal_s[j].tolist() for j in range(len(aux_animal_s))]
            #flatten aux_animal_i_s_flat
            aux_animal_i_s_flat = np.array([item for sublist in aux_animal_i_s_flat for item in sublist])
            # print(f"Animal {i} flat speed shape: {np.shape(aux_animal_i_s_flat)}")

            hist_aux = np.histogram(np.ravel(aux_animal_i_s_flat), bins=100, range=(group_flat_speed.min(), group_flat_speed.max()))

            plt.plot(hist_aux[1][:-1], hist_aux[0], '.-', label=f'Animal {i} nor {nor_index[idxs[i]]}')
            plt.xlabel('Speed')
            plt.ylabel('Count')
            plt.title(f'Animal Speed Histogram - {seg_name} - {gt}')

            aux_animal_i_perc = np.percentile(aux_animal_i_s_flat, percentiles_)
            # print(f"Animal {i} percentiles: {aux_animal_i_perc}")
            animal_flat_speed_perc[i] = aux_animal_i_perc
        plt.legend()
        plt.show()
#%%
#plot aux_animal_i_s_flat histogram
plt.figure(figsize=(8, 5))
plt.hist(np.ravel(aux_animal_i_s_flat), bins=100)
plt.xlabel('Speed')
plt.ylabel('Count')
plt.title(f'Animal Speed Histogram - {seg_name} - {gt}')
plt.show()

#%%
# aux_hist_perc = pooled_group_hists_by_segment[seg_name][gt]
seg_name = 'short'
gt = ('WT','VEH')
group_flat_speed = pooled_group_speed_by_segment[seg_name][gt]
group_flat_speed_perc = np.percentile(group_flat_speed, percentiles_)

# Plot pooled group histogram
plt.figure(figsize=(8, 5))
plt.plot(percentiles_, group_flat_speed_perc, label='Pooled Group Histogram')
plt.fill_between(percentiles_, ci_low_btr2, ci_high_btr2, color='orange', alpha=0.5, label='Classic BT Resampling CI')
plt.fill_between(percentiles_, ci_low_btr2_downsample, ci_high_btr2_downsample, color='green', alpha=0.5, label='Downsampled Classic BT Resampling CI')
plt.fill_between(percentiles_, ci_low_mean_i, ci_high_mean_i, color='blue', alpha=0.5, label='Hierarchical BT Resampling CI')
plt.fill_between(percentiles_,
                 ci_low_btr2_downsample_repeats.mean(axis=1),
                 ci_high_btr2_downsample_repeats.mean(axis=1),
                 color='red', alpha=0.5, label='Downsampled Classic BT Resampling CI (mean over repeats)')

plt.xlabel('Percentiles')
plt.ylabel('Speed')
plt.title(f'Pooled Group Histogram for {seg_name} - {gt}')
plt.xlim(0,4)
# plt.ylim(0.07,0.15)
plt.yscale('log')
plt.legend()
plt.show()

# print(ci_low_btr2, group_flat_speed_perc, ci_high_btr2)

#%%
for i in percentiles_:
    # print(f"Percentile {np.round(i, 3)}: {np.round(group_flat_speed_perc[int(i)], 3)} (CI: {np.round(ci_low_btr2[int(i)], 3)}, {np.round(ci_high_btr2[int(i)], 3)})")
    if i==100.:
        continue
    else:
        # print(f"Percentile {np.round(i, 3)}: {np.round(group_flat_speed_perc[int(i)], 3)} (CI: {np.round(ci_low_btr2[int(i)], 3)}, {np.round(ci_high_btr2[int(i)], 3)})")
        print(ci_low_btr2[int(i)]<=group_flat_speed_perc[int(i)]<=ci_high_btr2[int(i)])
        aux_ci_test = (
            ci_low_btr2[int(i)]<=group_flat_speed_perc[int(i)]<=ci_high_btr2[int(i)]
        )
aux_ci_test = [(ci_low_btr2[int(i)]<=group_flat_speed_perc[int(i)]<=ci_high_btr2[int(i)]) for i in percentiles_ if i!=100.]
print('Classic BT resampling inside CI',np.sum(aux_ci_test)/(len(percentiles_)-1))

aux_ci_test = [(ci_low_btr2_downsample[int(i)]<=group_flat_speed_perc[int(i)]<=ci_high_btr2_downsample[int(i)]) for i in percentiles_ if i!=100.]
print('Downsampled Classic BT resampling inside CI',np.sum(aux_ci_test)/(len(percentiles_)-1))

aa=0  # test for animal 0
for aa in range(len(group_speed_by_segment[seg_name][gt][0])):
    aux_ci_test = [(ci_low_mean_i[int(i)]<=animal_flat_speed_perc[aa, int(i)]<=ci_high_mean_i[int(i)]) for i in percentiles_ if i!=100.]
    print(f'Animal {aa} classic BT resampling inside CI:',np.sum(aux_ci_test)/(len(percentiles_)-1))
#%%


























#%%
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


nor_index = cog_data["index_NOR"].values
percentiles_ = np.linspace(0, 100, 100)

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
from itertools import combinations
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
            print('Processing percentile index:', i)
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
from itertools import combinations

import pandas as pd

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
