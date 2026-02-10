#!/usr/bin/env python3

#%%
import numpy as np
import matplotlib.pyplot as plt
from brainconn.modularity import modularity_louvain_und_sign

from shared_code.fun_allegiance_v2 import v2_prep_undirected_matrix


def gamma_sweep_protocol(
    mc_ref,
    gamma_vals=np.linspace(0.5, 1.5, 20),
    n_repeats=100,
    seed=0,
):
    """
    Empirical protocol to evaluate Louvain behavior vs gamma on MC_ref.
    """

    W = v2_prep_undirected_matrix(mc_ref)
    rng = np.random.default_rng(seed)

    results = []

    for gamma in gamma_vals:
        n_modules = []
        Q_vals = []

        for _ in range(n_repeats):
            Ci, Q = modularity_louvain_und_sign(W, gamma=gamma)

            n_modules.append(len(np.unique(Ci)))
            Q_vals.append(Q)

        results.append(
            dict(
                gamma=gamma,
                n_modules_mean=np.mean(n_modules),
                n_modules_std=np.std(n_modules),
                Q_mean=np.mean(Q_vals),
                Q_std=np.std(Q_vals),
            )
        )

    return results


def plot_gamma_protocol(results):
    gamma = [r["gamma"] for r in results]
    n_mod = [r["n_modules_mean"] for r in results]
    n_mod_std = [r["n_modules_std"] for r in results]
    Q = [r["Q_mean"] for r in results]
    Q_std = [r["Q_std"] for r in results]

    fig, ax1 = plt.subplots()

    ax1.errorbar(gamma, n_mod, yerr=n_mod_std, marker="o", label="#modules")
    ax1.set_xlabel("gamma")
    ax1.set_ylabel("#modules")

    ax2 = ax1.twinx()
    ax2.errorbar(gamma, Q, yerr=Q_std, marker="s", color="orange", label="Q")
    ax2.set_ylabel("modularity Q")

    fig.suptitle("Gamma sweep on MC_ref")
    fig.tight_layout()
    plt.show()


# -------------------
# Example usage
# -------------------
if __name__ == "__main__":
    from shared_code.fun_paths import get_paths
    from shared_code.fun_loaddata import load_timeseries_bundle
    from shared_code.fun_metaconnectivity import compute_metaconnectivity

    # Load mc_ref however you already do
    # Example:
    paths = get_paths(dataset_name="ines_abdallah",
                      timecourse_folder="Timecourses_updated_03052024",
                      cognitive_data_file="ROIs.xlsx",
                      anat_labels_file="41_Allen.txt")

    bundle = load_timeseries_bundle(
        paths["preprocessed"] / "ts_and_meta_2m4m.npz",
        paths["preprocessed"] / "grouping_data_oip.pkl",
    )

    ts = bundle.ts
    mask_groups = bundle.mask_groups
    ind_ref = mask_groups[2][0]  # WT 2m for example

    mc = compute_metaconnectivity(ts, window_size=7, lag=1, n_jobs=-1)
    mc_ref = np.mean(mc[ind_ref], axis=0)

    results = gamma_sweep_protocol(mc_ref)
    plot_gamma_protocol(results)
#%%


