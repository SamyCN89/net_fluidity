# %%
# =============================================================================
import numpy as np

try:
    import brainconn as bct
except Exception:  # pragma: no cover
    bct = None  # type: ignore
# Network analysis functions
# =============================================================================


def sort_modularity(fc):
    """
    Sort an FC matrix by community assignments inferred via Louvain modularity.

    Parameters:
    - fc: np.ndarray (N × N)
      Weighted (possibly signed) adjacency/FC matrix.

    Returns:
    - fc_sorted: np.ndarray (N × N)
      FC matrix with rows/columns permuted to group nodes by detected modules.

    Notes:
    - Uses `bct.modularity.modularity_louvain_und_sign` with `gamma=1.1`.
    - This helper is slated for replacement by an allegiance‑matrix‑based analysis in metaconnectivity.
    """
    # modules, louvain = bct.modularity.modularity_louvain_dir(fc)
    if bct is None:
        raise RuntimeError("brainconn is required for sort_modularity")
    modules, louvain = bct.modularity.modularity_louvain_und_sign(fc, gamma=1.1)

    # Sort FC according to module labels
    order = np.argsort(modules)
    fc_sorted = fc[:, order][order, :]
    return fc_sorted


# =============================================================================
