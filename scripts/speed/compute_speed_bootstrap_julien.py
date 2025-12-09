# %% ========================== IMPORTS & CONFIG ==========================

from collections.abc import Iterable, Sequence
from pathlib import Path
import pickle
import time

from joblib import Parallel, delayed
import numpy as np
from glob import glob



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

GROUP_RECIPES = {
    # single factors
    "genotype": ["genotype"],
    "treatment": ["treatment"],
    # 2-way
    "genotype_treatment": ["genotype", "treatment"],  # only if those cols exist
}


# which speed subsets to bootstrap
# these must match the folder names under results/<dataset>/speed/
SPEED_SUBSETS = [
    "all",
    # "regions500",
    "per_region",
    "dmn_touching",
    "1st_touching",
    "2nd_touching",
    "3rd_touching",
    "4th_touching",
    "lat_touching",
    "mem_touching",
    "sal_touching",
    "dmn_within",
    "1st_within",
    "2nd_within",
    "3rd_within",
    "4th_within",
    "lat_within",
    "mem_within",
    "sal_within",
    # "gen1",
        # "gen2",
    # etc. – keep only what you want
]

groups_list = list(GROUP_RECIPES.keys())

# %% ========================== SMALL HELPERS ==========================


# --- Load helpers ---
def load_speed_stack(
    paths_speed_root: Path, time_windows_range: Sequence[int]
) -> list[np.ndarray]:
    """Return list S where S[j][i] is 1D np.array of samples for animal i at window j."""
    speeds: list[np.ndarray] = []
    for w in time_windows_range:
        a = np.load(
            paths_speed_root.format(w=w, n_animals=n_animals, regions=regions),
            allow_pickle=True,
        )
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


# %%
def flatten_group_animals_over_windows(
    animal_speeds: list[list[np.ndarray]], indices: Sequence[int], w_range: range
) -> np.ndarray:
    arrays = []
    for i in indices:
        # print(f"[DEBUG] Processing animal index: {i}")
        arrays.extend([animal_speeds[i][j] for j in w_range])
        # print(f"[DEBUG] Current number of arrays for animal {i}: {len(arrays)}")
    # print(f"[DEBUG] Flattened group animals over windows: n_arrays={len(arrays)}")
    return (
        np.concatenate([a.ravel() for a in arrays])
        if arrays
        else np.array([], dtype=float)
    )


def get_group_animals_over_windows(
    animal_speeds: list[list[np.ndarray]], indices: Sequence[int], w_range: range
) -> np.ndarray:
    """Return object array shape (len(indices), len(w_range)) of arrays."""
    arr = np.array(animal_speeds, dtype=object)[indices]
    arrays = np.transpose([arr[:, j] for j in w_range], (1, 0))
    return (arrays,) if arrays.size else np.array([], dtype=float)


# %%


# ---------- BOOTSTRAP HELPERS ----------
# %%

def discover_per_region_descriptors(
    subset_dir: Path,
    w0: int,
    n_animals: int,
    regions: int,
    lag: int = 1,
    tau_count: int = 2,
) -> list[str]:
    """
    Look at one window (w0) under per_region/ and get the list of region descriptors,
    e.g. ['region-AI', 'region-PL', ...].
    """
    pattern = (
        subset_dir
        / f"speed_win{w0}_lag{lag}_tau{tau_count}_animals_{n_animals}_regions_{regions}_region-*.npz"
    )
    files = sorted(glob(str(pattern)))
    if not files:
        raise FileNotFoundError(
            f"No per-region speed files found for window={w0} with pattern {pattern}"
        )

    descriptors: list[str] = []
    for fpath in files:
        name = Path(fpath).name
        # everything after 'regions_{regions}_'
        suffix = name.split(f"regions_{regions}_", 1)[1]  # region-XXX.npz
        descriptor = suffix[:-4]  # strip '.npz'
        descriptors.append(descriptor)

    return sorted(set(descriptors))


def load_speed_stack_single_region(
    subset_dir: Path,
    time_windows_range: Sequence[int],
    n_animals: int,
    regions: int,
    region_desc: str,
    lag: int = 1,
    tau_count: int = 2,
) -> list[list[np.ndarray]]:
    """
    Load speeds for ONE region (e.g. region-AI) across all windows.

    Returns speeds[w][i] where:
      - w indexes time_windows_range
      - i indexes animals
    """
    speeds_per_window: list[list[np.ndarray]] = []

    for w in time_windows_range:
        fpath = (
            subset_dir
            / f"speed_win{w}_lag{lag}_tau{tau_count}_animals_{n_animals}_regions_{regions}_{region_desc}.npz"
        )
        if not fpath.exists():
            raise FileNotFoundError(f"Missing per-region file: {fpath}")

        with np.load(fpath, allow_pickle=True) as z:
            if "speeds" not in z.files:
                raise KeyError(f"{fpath} missing 'speeds' array")
            s = z["speeds"]  # object array, len n_animals

        if len(s) != n_animals:
            raise ValueError(
                f"{fpath}: expected {n_animals} animals, got {len(s)}"
            )

        window_speeds: list[np.ndarray] = []
        for i in range(n_animals):
            arr = np.asarray(s[i], dtype=float)
            window_speeds.append(arr.ravel())
        speeds_per_window.append(window_speeds)

    return speeds_per_window
# %% ================ GROUP BOOTSTRAP RESAMPLING HELPERS ==========================


# ---------- utility: efficient downsample without replacement ----------
def _downsample_once(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = x.size
    if k is None or k >= n:
        return x
    # choice is faster when k ≪ n; permutation is fine when k is larger
    if k / n < 0.2:
        idx = rng.choice(n, size=k, replace=False)
    else:
        idx = rng.permutation(n)[:k]
    return x[idx]


# ---------- one repeat = one downsample + percentiles ----------
def _one_repeat_downsample_only(
    data: np.ndarray,
    q: np.ndarray,
    downsample_n: int | None,
    seed_seq: np.random.SeedSequence,
) -> np.ndarray:
    rng = np.random.default_rng(seed_seq)
    x = np.ravel(data)
    if downsample_n and x.size > downsample_n:
        x = _downsample_once(x, downsample_n, rng)
    # directly compute percentiles of this downsampled draw
    return np.percentile(x, q)


def bootstrap_downsampling_repeat(
    data: np.ndarray,
    percentiles: np.ndarray,
    repeat: int = 10_000,
    downsample_n: int | None = None,
    seed: int | None = 0,
    n_jobs: int = 8,
):
    """
    Perform `repeat` independent downsampling draws (without replacement),
    compute `percentiles` on each draw, and return the across-repeat envelope.

    Returns:
      ci_low_repeat, ci_high_repeat, ci_repeat
        - ci_repeat: (repeat, K) matrix; row r = percentiles of the r-th downsample
        - ci_low_repeat, ci_high_repeat: (K,), here [0th, 100th] across repeats to match your original
    """
    q = np.asarray(percentiles, dtype=float).ravel()
    if data.size == 0:
        nan = np.full(q.shape, np.nan, float)
        return nan, nan, np.empty((0, q.size), float)

    # robust, independent seeds per repeat
    base_ss = np.random.SeedSequence(None if seed is None else int(seed))
    child_ss = base_ss.spawn(repeat)

    rows = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_one_repeat_downsample_only)(
            data=data,
            q=q,
            downsample_n=downsample_n,
            seed_seq=child_ss[i],
        )
        for i in range(repeat)
    )
    ci_repeat = np.vstack(rows)  # (repeat, K)

    # Use [0, 100] to keep compatibility with what you were doing;
    # if you want classical CIs, use [2.5, 97.5] instead.
    ci_low_repeat, ci_high_repeat = np.percentile(ci_repeat, [0, 100], axis=0)
    return ci_low_repeat, ci_high_repeat, ci_repeat



# Bootstrap downsampling with repeats
def group_pool_bt_classic_resampling_downsampled(
    ranges: dict,
    pooled_group_speed_by_segment: dict,  # {seg: {group: array_like}}
    group_data: dict,  # used for group iteration order
    percentiles_: np.ndarray,
    repeat: int = 10_000,
    downsample_factor: int = 10,
    seed: int | None = 0,
    n_jobs: int = 8,
    verbose: int = 0,
):
    """
    For each (segment, group):
      - flatten its pooled speeds
      - repeatedly downsample WITHOUT replacement to N/f
      - compute given percentiles on each downsample
      - aggregate percentiles across repeats to get an envelope
    """
    q = np.asarray(percentiles_, dtype=float).ravel()

    ci_low_repeat: dict[str, dict[str, np.ndarray]] = {}
    ci_high_repeat: dict[str, dict[str, np.ndarray]] = {}
    vals_btr_downsample_repeat: dict[str, dict[str, np.ndarray]] = {}

    # spawn independent seeds per (segment, group)
    base_ss = np.random.SeedSequence(None if seed is None else int(seed))
    pair_list = [
        (seg_name, gt) for seg_name in ranges.keys() for gt in group_data.keys()
    ]
    child_ss = base_ss.spawn(len(pair_list))
    ss_iter = iter(child_ss)

    for seg_name in ranges.keys():
        ci_low_seg: dict[str, np.ndarray] = {}
        ci_high_seg: dict[str, np.ndarray] = {}
        vals_seg: dict[str, np.ndarray] = {}

        for gt in group_data.keys():
            if (
                seg_name not in pooled_group_speed_by_segment
                or gt not in pooled_group_speed_by_segment[seg_name]
            ):
                raise KeyError(f"Missing data for segment '{seg_name}', group '{gt}'")

            data = np.ravel(pooled_group_speed_by_segment[seg_name][gt])
            # guard against zero downsample size
            if downsample_factor and downsample_factor > 1:
                ds_n = max(1, int(len(data) // downsample_factor))
            else:
                ds_n = None

            ss = next(ss_iter)
            t0 = time.time()
            lo, hi, vals = bootstrap_downsampling_repeat(
                data=data,
                percentiles=q,
                repeat=repeat,
                downsample_n=ds_n,
                seed=ss.entropy,  # okay to use an int; we could also pass ss itself and spawn again inside
                n_jobs=n_jobs,
            )
            t1 = time.time()

            ci_low_seg[gt] = lo
            ci_high_seg[gt] = hi
            vals_seg[gt] = vals
            if verbose:
                print(
                    f"[{seg_name} | {gt}] n={len(data)}, downsample_n={ds_n}, repeats={repeat} in {t1-t0:.2f}s"
                )

        ci_low_repeat[seg_name] = ci_low_seg
        ci_high_repeat[seg_name] = ci_high_seg
        vals_btr_downsample_repeat[seg_name] = vals_seg

    return ci_low_repeat, ci_high_repeat, vals_btr_downsample_repeat


# %%

# %% ================ GROUPING HELPERS ==========================
# Group indices from cognitive data


def make_long_cog(cog_data: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Standardize cognitive / grouping data.

    For 'julien':
      - keep one row per animal
      - columns: name, genotype, treatment (+ grp if you want it)
    For 'ines':
      - keep the long 2M/4M format as before.
    """
    if dataset_name == "julien":
        df = cog_data.copy()

        # rename to a neutral "name" column if you ever need it
        if "mouse" in df.columns:
            df = df.rename(columns={"mouse": "name"})

        # keep only what we actually use here
        cols_keep = [c for c in ["name", "genotype", "treatment", "grp"] if c in df.columns]
        df = df[cols_keep]

        # categories for stable grouping
        for col in ["genotype", "treatment", "grp"]:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df

    elif dataset_name == "ines":
        # --- your previous INES code, unchanged ---
        cols_common = ["Name", "Sexe", "Genotype", "Phenotype_OiP", "Phenotype_RO24h"]
        df2 = cog_data[cols_common + ["OiP_2M", "RO24h_2M", "TC_2M"]].copy()
        df4 = cog_data[cols_common + ["OiP_4M", "RO24h_4M", "TC_4M"]].copy()

        df2["Age"] = "2M"
        df4["Age"] = "4M"
        df2 = df2.rename(columns={"OiP_2M": "oip", "RO24h_2M": "ro24h", "TC_2M": "tc"})
        df4 = df4.rename(columns={"OiP_4M": "oip", "RO24h_4M": "ro24h", "TC_4M": "tc"})
        df = pd.concat([df2, df4], ignore_index=True)

        df.rename(
            columns={
                "Name": "name",
                "Sexe": "sex",
                "Genotype": "genotype",
                "Age": "age",
                "Phenotype_OiP": "phenotype_oip",
                "Phenotype_RO24h": "phenotype_ro24h",
            },
            inplace=True,
        )

        if "sex" in df.columns:
            df["sex"] = df["sex"].map({"F": "female", "M": "male"}).fillna(df["sex"])

        for col in ["sex", "age", "genotype", "phenotype_oip", "phenotype_ro24h"]:
            if col in df.columns:
                df[col] = df[col].astype("category")

        return df

    else:
        raise ValueError(f"Unknown dataset_name={dataset_name}")



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


# %%

# =============================================================================
# -----------------------------------------------------------------------------
# ============================= Main Code ==============================
# -----------------------------------------------------------------------------
# =============================================================================
# %%


# ================================================================================
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# ================================================================================
# ----------------------------------- group_data start ---------------------------
# ================================================================================
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# ================================================================================

percentiles_ = np.linspace(0, 100, 100)
n_resamples = 10_000
# n_resamples = 10
downsample_factor = 10
seed = 42
n_jobs = 60


# %% ========================== GROUP INDICES ==========================


# %% ========================== LOAD DATA ==========================
# dataset_name = "ines"
dataset_name = "julien"
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
# NOTE: this still keeps {w}, {n_animals}, {regions} for later .format() in load_speed_stack
loaddir_speed_template = str(
    speed_root
    / "{subset}/speed_win{{w}}_lag1_tau2_animals_{{n_animals}}_regions_{{regions}}.npz"
)


# Output location for group histograms
# outdir_save_group_hists = speed_root / f"{dataset}_pool_{POOL_SPLIT}_bins{BINS_HIST}" / f"pooled_group_hists__{seg_name}.npz"
# bootstrap CI folder
bootstrap_folder = paths["speed"] / "bootstrap"
bootstrap_folder.mkdir(parents=True, exist_ok=True)

# outdir_bootstrap_repeat = str(
#     bootstrap_folder
#     / "bootstrap_downsample_repeat_group_{groups_selected}_nresamples_{n_resamples}_downsample_factor_{downsample_factor}_seed_{seed}.pkl"
# )

outdir_bootstrap_repeat = str(
    bootstrap_folder
    / "bootstrap_downsample_repeat_subset_{subset}_group_{groups_selected}"
      "_nresamples_{n_resamples}_downsample_factor_{downsample_factor}_seed_{seed}.pkl"
)



#%%

def run_bootstrap_for_subset_label(subset_label: str, speeds: list[list[np.ndarray]]):
    """Run full pooling + bootstrap pipeline for a given subset label."""
    global n_animals  # we re-assign locally anyway

    # 2) Precompute dimensions & pooling ranges for this subset
    n_windows = len(speeds)
    n_animals = len(speeds[0])

    per_window_animal_means = [
        np.array([float(np.mean(speeds[j][i])) for i in range(n_animals)], dtype=float)
        for j in range(n_windows)
    ]

    counts = count_samples_per_window(speeds)
    pooled_speeds_cdf = (
        np.cumsum(counts) / np.sum(counts) if counts.sum() else np.zeros_like(counts)
    )
    i_third, i_half, i_two_third = cdf_split_indices(speeds)
    ranges = select_windows(
        POOL_SPLIT, len(time_windows_range), i_third, i_half, i_two_third
    )

    all_speeds_flat = flatten_windows(speeds, 0, len(speeds))
    all_speeds_min, all_speeds_max = global_min_max([all_speeds_flat])
    edges = np.linspace(all_speeds_min, all_speeds_max, BINS_HIST + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # 3) Loop over groupings inside this subset
    for groups_selected in groups_list:
        print(f"[subset={subset_label}] Processing grouping: {groups_selected}")

        group_data = get_group_data(cog_data, dataset_name, groups_selected)

        # ----- PER-SEGMENT HISTOGRAMS -----

        group_means_by_segment: dict[str, dict[tuple[str, str], np.ndarray]] = {}

        for seg_name, w_range in ranges.items():
            H = build_per_animal_normalized_hists(
                speeds, w_range, BINS_HIST, (all_speeds_min, all_speeds_max)
            )
            group_means: dict[tuple[str, str], np.ndarray] = {}
            for gt, idxs in group_data.items():
                group_means[gt] = (
                    np.mean(H[idxs], axis=0) if len(idxs) else np.zeros(BINS_HIST)
                )
            group_means_by_segment[seg_name] = group_means

        # ----- POOLING -----
        animal_speeds = [
            [speeds[j][i] for j in range(n_windows)] for i in range(n_animals)
        ]
        pooled_group_hists_by_segment: dict[str, dict[tuple[str, str], np.ndarray]] = {}
        pooled_group_speed_by_segment: dict[str, dict[tuple[str, str], np.ndarray]] = {}
        group_speed_by_segment = {}

        for seg_name, w_range in ranges.items():
            pooled_group_hist_i = {}
            pooled_group_speed_i = {}
            group_speed = {}
            for gt, idxs in group_data.items():
                group_speed_i = get_group_animals_over_windows(
                    animal_speeds, idxs, w_range
                )
                group_speed[gt] = group_speed_i

                flat = flatten_group_animals_over_windows(
                    animal_speeds, idxs, w_range
                )
                pooled_group_speed_i[gt] = flat

                h, _ = safe_hist(flat, BINS_HIST, (all_speeds_min, all_speeds_max))
                pooled_group_hist_i[gt] = normalize_counts_to_prob(h) * 2

            pooled_group_hists_by_segment[seg_name] = pooled_group_hist_i
            pooled_group_speed_by_segment[seg_name] = pooled_group_speed_i
            group_speed_by_segment[seg_name] = group_speed

        # ----- BOOTSTRAP (per subset_label + grouping) -----
        outdir_bootstrap_repeat_aux = Path(
            outdir_bootstrap_repeat.format(
                subset=subset_label,
                groups_selected=groups_selected,
                n_resamples=n_resamples,
                downsample_factor=downsample_factor,
                seed=seed,
            )
        )

        if outdir_bootstrap_repeat_aux.exists():
            print(f"  Loading bootstrap: {outdir_bootstrap_repeat_aux}")
            with open(outdir_bootstrap_repeat_aux, "rb") as f:
                data_loaded = pickle.load(f)
                ci_low_repeat = data_loaded["ci_low_repeat"]
                ci_high_repeat = data_loaded["ci_high_repeat"]
                vals_btr_downsample_repeat = data_loaded["ci_btr_downsample_repeat"]
        else:
            ci_low_repeat, ci_high_repeat, vals_btr_downsample_repeat = (
                group_pool_bt_classic_resampling_downsampled(
                    ranges,
                    pooled_group_speed_by_segment,
                    group_data,
                    percentiles_,
                    repeat=n_resamples,
                    downsample_factor=downsample_factor,
                    seed=seed,
                    n_jobs=n_jobs,
                    verbose=1,
                )
            )
            with open(outdir_bootstrap_repeat_aux, "wb") as f:
                pickle.dump(
                    {
                        "subset": subset_label,
                        "groups_selected": groups_selected,
                        "group_data": group_data,
                        "ranges": ranges,
                        "percentiles_": percentiles_,
                        "centers": centers,
                        "ci_low_repeat": ci_low_repeat,
                        "ci_high_repeat": ci_high_repeat,
                        "ci_btr_downsample_repeat": vals_btr_downsample_repeat,
                        "group_means_by_segment": group_means_by_segment,
                        "pooled_group_hists_by_segment": pooled_group_hists_by_segment,
                        "pooled_group_speed_by_segment": pooled_group_speed_by_segment,
                        "group_speed_by_segment": group_speed_by_segment,
                    },
                    f,
                )
            print(f"  Saved bootstrap to: {outdir_bootstrap_repeat_aux}")

#%%
# ========================== MAIN LOOP OVER SUBSETS ==========================

# Load timeseries bundle once to know n_animals/regions
bundle = load_timeseries_bundle(loaddir_ts_meta)
n_animals = bundle.n_animals
total_tr = bundle.total_tr
regions = bundle.n_regions

# Load cognitive data once (same animals for all subsets)
cog_data = load_cognitive_data(
    loaddir_cog_data.format(n_animals=n_animals, regions=regions, total_tr=total_tr)
)

# ========================== MAIN LOOP OVER SUBSETS ==========================

for subset in SPEED_SUBSETS:
    print(f"\n=== Subset: {subset} ===")

    if subset == "per_region":
        # ---------- SPECIAL CASE: per_region → region-by-region bootstrap ----------
        subset_dir = speed_root / "per_region"

        # discover region descriptors (e.g. region-AI, region-PL, ...)
        region_descriptors = discover_per_region_descriptors(
            subset_dir=subset_dir,
            w0=int(time_windows_range[0]),
            n_animals=n_animals,
            regions=regions,
            lag=1,        # must match your dfc_speed_compute lag
            tau_count=2,  # TAU_RANGE=0,4 → 2 tau values
        )
        print(f"[per_region] Found {len(region_descriptors)} regions: {region_descriptors}")

        for region_desc in region_descriptors:
            print(f"\n[per_region] Region: {region_desc}")

            speeds_region = load_speed_stack_single_region(
                subset_dir=subset_dir,
                time_windows_range=time_windows_range,
                n_animals=n_animals,
                regions=regions,
                region_desc=region_desc,
                lag=1,
                tau_count=2,
            )

            subset_label = f"{subset}_{region_desc}"  # e.g. 'per_region_region-AI'
            run_bootstrap_for_subset_label(subset_label, speeds_region)

    else:
        # ---------- NORMAL SUBSETS (all, dmn_*, sal_*, etc.) ----------
        loaddir_speed = loaddir_speed_template.format(subset=subset)
        try:
            speeds = load_speed_stack(loaddir_speed, time_windows_range)
        except FileNotFoundError as e:
            print(f"[WARN] Skipping subset {subset}: {e}")
            continue

        run_bootstrap_for_subset_label(subset, speeds)



# %%

