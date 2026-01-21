from __future__ import annotations

from pathlib import Path
import pickle
import joblib
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

import brainconn as bct


# ============================
# V2: matrix hygiene
# ============================
def v2_prep_undirected_matrix(W: np.ndarray) -> np.ndarray:
    """Make W safe for undirected Louvain: finite, symmetric, diag=0."""
    W = np.asarray(W, dtype=np.float32)
    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return W


def v2_run_louvain_und_sign(W: np.ndarray, gamma: float) -> tuple[np.ndarray, float]:
    W = v2_prep_undirected_matrix(W)
    Ci, Q = bct.modularity.modularity_louvain_und_sign(W, gamma=float(gamma))
    return np.asarray(Ci, dtype=np.int32), float(Q)


# ============================
# V2: agreement matrix
# ============================
def v2_build_agreement_matrix(communities_runs: np.ndarray) -> np.ndarray:
    """
    Memory-safe agreement matrix.
    communities_runs: (n_runs, n_nodes) int labels
    returns: (n_nodes, n_nodes) float32 counts (NOT normalized)
    """
    comm = np.asarray(communities_runs, dtype=np.int32)
    n_runs, n_nodes = comm.shape
    agree = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    for r in range(n_runs):
        labels = comm[r]
        for lab in np.unique(labels):
            idx = np.flatnonzero(labels == lab)
            if idx.size <= 1:
                continue
            agree[np.ix_(idx, idx)] += 1.0

    return agree


# ============================
# V2: contingency (stage 1)
# ============================
def v2_contingency_matrix(
    *,
    mc_data: np.ndarray,
    n_runs: int,
    n_gamma: int,
    gmin: float = 0.8,
    gmax: float = 1.3,
    n_jobs: int = -1,
    cache_path: str | Path | None = None,
    ref_name: str = "",
    return_runs: bool = False,
):
    """
    Build co-classification matrix by sampling partitions across gamma sweep.

    Returns:
      contingency: (N,N) in [0,1], symmetric, diag=0
      gamma_q: (n_gamma, n_runs)
      gamma_agree: (n_gamma, N, N) each in [0,1]
      optionally communities_mat: (n_gamma, n_runs, N)
    """
    W0 = v2_prep_undirected_matrix(mc_data)
    n_nodes = W0.shape[0]
    if W0.shape[0] != W0.shape[1]:
        raise ValueError(f"mc_data must be square; got {W0.shape}")

    gammas = np.linspace(float(gmin), float(gmax), int(n_gamma), dtype=np.float32)

    full_cache_path = None
    if cache_path:
        cache_dir = Path(cache_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe = ref_name.replace(" ", "_")
        full_cache_path = (
            cache_dir
            / f"v2_contingency_ref={safe}_N={n_nodes}_nruns={n_runs}_ngamma={n_gamma}_gmin={gmin}_gmax={gmax}.pkl"
        )
        if full_cache_path.exists():
            with full_cache_path.open("rb") as f:
                return pickle.load(f)

    # Louvain jobs
    job_list = [(float(g), r) for g in gammas for r in range(n_runs)]
    all_results = Parallel(n_jobs=n_jobs)(
        delayed(v2_run_louvain_und_sign)(W0, gamma)
        for (gamma, _) in tqdm(job_list, desc="V2 Louvain(gamma sweep)")
    )

    communities_mat = np.zeros((len(gammas), n_runs, n_nodes), dtype=np.int32)
    gamma_q = np.zeros((len(gammas), n_runs), dtype=np.float32)

    k = 0
    for gi in range(len(gammas)):
        for r in range(n_runs):
            Ci, Q = all_results[k]
            communities_mat[gi, r] = Ci
            gamma_q[gi, r] = Q
            k += 1

    gamma_agree = np.zeros((len(gammas), n_nodes, n_nodes), dtype=np.float32)
    contingency = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    for gi in tqdm(range(len(gammas)), desc="V2 building agreements"):
        agree_counts = v2_build_agreement_matrix(communities_mat[gi])  # counts
        agree = agree_counts / float(n_runs)                           # [0,1]
        gamma_agree[gi] = agree
        contingency += agree

    contingency /= float(len(gammas))
    contingency = v2_prep_undirected_matrix(contingency)  # sym + diag=0

    out = (contingency, gamma_q, gamma_agree)
    if return_runs:
        out = (contingency, gamma_q, gamma_agree, communities_mat)

    if full_cache_path is not None:
        with full_cache_path.open("wb") as f:
            pickle.dump(out, f)
        print(f"[cache] Saved {full_cache_path}")

    return out


# ============================
# V2: consensus (stage 2)
# ============================
def v2_consensus_from_contingency(
    contingency: np.ndarray,
    *,
    gamma_consensus: float = 1.2,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Cluster contingency to get consensus labels.
    Returns: labels (N,), sort_idx (N,), Q, contingency_prepped
    """
    C = v2_prep_undirected_matrix(contingency)
    Ci, Q = bct.modularity.modularity_louvain_und_sign(C, gamma=float(gamma_consensus))
    Ci = np.asarray(Ci, dtype=np.int32)
    sort_idx = np.argsort(Ci)
    return Ci, sort_idx, float(Q), C


# ============================
# V2: full wrapper
# ============================
def v2_allegiance_communities(
    mc_data: np.ndarray,
    *,
    n_runs: int = 300,
    n_gamma: int = 15,
    gmin: float = 0.8,
    gmax: float = 1.3,
    gamma_consensus: float = 1.2,
    ref_name: str | None = None,
    save_path: str | Path | None = None,
    n_jobs: int = -1,
):
    """
    Stage 1: build contingency by gamma sweep
    Stage 2: consensus Louvain on contingency
    Returns: communities, sort_idx, contingency
    """
    file_path = None
    if save_path and ref_name:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        safe = ref_name.replace(" ", "_")
        file_path = save_path / f"v2_allegiance_{safe}.joblib"
        if file_path.exists():
            print(f"[cache] Loading {file_path}")
            return joblib.load(file_path)

    contingency, gamma_q, gamma_agree = v2_contingency_matrix(
        mc_data=mc_data,
        n_runs=n_runs,
        n_gamma=n_gamma,
        gmin=gmin,
        gmax=gmax,
        n_jobs=n_jobs,
        cache_path=save_path,
        ref_name=ref_name or "",
        return_runs=False,
    )

    communities, sort_idx, Q, contingency2 = v2_consensus_from_contingency(
        contingency, gamma_consensus=gamma_consensus
    )

    if file_path is not None:
        joblib.dump((communities, sort_idx, contingency2), file_path)

    return communities, sort_idx, contingency2
