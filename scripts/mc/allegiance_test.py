
# %%
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

from shared_code.fun_paths import get_paths
from shared_code.fun_metaconnectivity import fun_allegiance_communities

# -------------------
# CONFIG (edit)
# -------------------
DATASET = "ines_abdallah"
TIMECOURSE_FOLDER = "Timecourses_updated_03052024"
COGNITIVE_FILE = "ROIs.xlsx"
ANAT_LABELS_FILE = "41_Allen.txt"

REF_GENOTYPE = "wt"
REF_AGE = "2m"

GAMMAS = [50, 100, 150, 200]     # resolution sweep
N_RUNS_LIST = [100, 300, 1000]   # stability sweep
N_JOBS = -1

# -------------------
# Helpers
# -------------------
def find_latest(folder: Path, pattern: str) -> Path:
    hits = sorted(folder.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No files matching {pattern} in {folder}")
    return hits[-1]

def blockiness_score(M: np.ndarray, communities: np.ndarray) -> tuple[float, float]:
    """Mean |M| intra vs inter (diag ignored). Higher (intra-inter) is better."""
    M = np.asarray(M).copy()
    np.fill_diagonal(M, np.nan)
    c = np.asarray(communities)
    same = c[:, None] == c[None, :]
    eye = np.eye(M.shape[0], dtype=bool)
    intra = np.abs(M[same & ~eye])
    inter = np.abs(M[~same])
    return float(np.nanmean(intra)), float(np.nanmean(inter))

def basic_matrix_qc(C: np.ndarray) -> dict:
    C = np.asarray(C)
    diag = np.diag(C)
    sym = np.max(np.abs(C - C.T))
    return dict(
        shape=list(C.shape),
        diag_mean=float(np.mean(diag)),
        diag_std=float(np.std(diag)),
        sym_max=float(sym),
        min=float(np.min(C)),
        max=float(np.max(C)),
    )

# -------------------
# Load artifacts
# -------------------
paths = get_paths(
    dataset_name=DATASET,
    timecourse_folder=TIMECOURSE_FOLDER,
    cognitive_data_file=COGNITIVE_FILE,
    anat_labels_file=ANAT_LABELS_FILE,
)
results_dir = Path(paths["mc"])
preproc_dir = Path(paths["preprocessed"])
#%%
mc_raw_path = find_latest(results_dir / "mc_raw", "mc_raw_*.npz")
d = np.load(mc_raw_path, allow_pickle=True)
mc = d["mc"]
mouse_ids_ts = d["mouse_ids_ts"].astype(str)
age_ts = d["age_ts"].astype(str)

cog_csv_path = find_latest(preproc_dir, "cog_data_filtered_*.csv")
cog = pd.read_csv(cog_csv_path)
cog["Name"] = cog["Name"].astype(str)

ref_mice = cog.loc[cog["Genotype"].astype(str) == REF_GENOTYPE, "Name"].to_numpy(dtype=str)
ind_ref = np.isin(mouse_ids_ts, ref_mice) & (age_ts == REF_AGE)
if ind_ref.sum() == 0:
    raise RuntimeError("Empty reference group. Check Genotype labels and REF_AGE.")

mc_ref = np.nanmean(mc[ind_ref], axis=0)

print("Loaded mc_ref:", mc_ref.shape, "from", int(ind_ref.sum()), "sessions")
print("mc_raw:", mc_raw_path.name)

# -------------------
# Sweep + QC
# -------------------
rows = []
for gamma in GAMMAS:
    for n_runs in N_RUNS_LIST:
        t0 = time.time()
        comm, sort_idx, C = fun_allegiance_communities(
            mc_ref,
            n_runs=n_runs,
            gamma_pt=gamma,
            save_path=None,
            ref_name=f"{REF_GENOTYPE}_{REF_AGE}",
            n_jobs=N_JOBS,
        )
        dt = time.time() - t0

        comm = np.asarray(comm)
        sort_idx = np.asarray(sort_idx)

        # If C missing, we can still evaluate using mc_ref + comm
        if C is None:
            C_qc = {"shape": None}
        else:
            C_qc = basic_matrix_qc(C)

        intra0, inter0 = blockiness_score(mc_ref, comm)
        mc_ref_sorted = mc_ref[sort_idx][:, sort_idx]
        comm_sorted = comm[sort_idx]
        intra1, inter1 = blockiness_score(mc_ref_sorted, comm_sorted)

        n_comm = int(len(np.unique(comm)))
        rows.append(dict(
            gamma=gamma,
            n_runs=n_runs,
            seconds=dt,
            n_communities=n_comm,
            block_unsorted=intra0 - inter0,
            block_sorted=intra1 - inter1,
            delta_block=(intra1 - inter1) - (intra0 - inter0),
            C_diag_mean=C_qc.get("diag_mean", np.nan),
            C_sym_max=C_qc.get("sym_max", np.nan),
        ))

# Print nicely
df = pd.DataFrame(rows).sort_values(["gamma","n_runs"])
print(df.to_string(index=False))
# %%
