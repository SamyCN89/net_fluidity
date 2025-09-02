# %%
from pathlib import Path as _Path
import pickle
import time

from joblib import Parallel, delayed
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
import logging
import argparse
import sys

# Optional/extra dependencies
try:
    from mizani.palettes import brewer_pal
except Exception:  # pragma: no cover
    brewer_pal = None  # type: ignore

try:
    import brainconn as bct
except Exception:  # pragma: no cover
    bct = None  # type: ignore

try:
    from sklearn.manifold import TSNE
    from sklearn.metrics import mutual_info_score
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    TSNE = None  # type: ignore
    StandardScaler = None  # type: ignore
    mutual_info_score = None  # type: ignore
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr

from shared_code.fun_metaconnectivity import (
    build_agreement_matrix_vectorized,
    load_merged_allegiance,  # %%
    contingency_matrix_fun,
)
from shared_code.fun_paths import get_paths
#%%
