"""
FP1_compute_mc_raw.py

Finish point A:
- compute MC (animals, E, E)
- sanity checks (diag/off-diag)

Finish point B:
- load allegiance cache (preferred) OR compute if missing
- build module mask + trimer index
- save ONE frozen artifact to results/<dataset>/mc_raw/

"""
#%%
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from shared_code.fun_metaconnectivity import (
    compute_metaconnectivity,
)
from shared_code.fun_paths import get_paths

# %%
# =========================
# CONFIG
# =========================
DATASET = "ines_abdallah"
TIMECOURSE_FOLDER = "Timecourses_updated_03052024"
COGNITIVE_FILE = "ROIs.xlsx"
ANAT_LABELS_FILE = "41_Allen.txt"

WINDOW_SIZE = 15
LAG = 1
N_JOBS = -1

def _jsonable(x):
    import numpy as np
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    return x

# %%

def main():
    # Reference group selection
    paths = get_paths(
        dataset_name=DATASET,
        timecourse_folder=TIMECOURSE_FOLDER,
        cognitive_data_file=COGNITIVE_FILE,
        anat_labels_file=ANAT_LABELS_FILE,
    )

    # LOAD DATA
    npz_path = paths["preprocessed"] / "ts_and_meta_ines_abdallah.npz"
    d = np.load(npz_path, allow_pickle=True)

    ts = d["ts"]
    mouse_ids = d["mouse_ids"]          # (63,)
    mouse_ids_ts = d["mouse_ids_ts"]    # (126,)
    age_ts = d["age_ts"]                # (126,)
    regions = int(d["regions"])
    n_animals = ts.shape[0]

    # FINISH POINT A — Metaconnectivity computation
    E_expected = regions * (regions - 1) // 2
    print("E (FC edges) expected:", E_expected)

    t0 = time.time()
    mc = compute_metaconnectivity(
        ts,
        window_size=WINDOW_SIZE,
        lag=LAG,
        n_jobs=N_JOBS,
        save_path=None,     # IMPORTANT: no reports/ saving
    )
    mc = np.asarray(mc)
    dt = time.time() - t0
    print(f"MC shape: {mc.shape} (computed in {dt:.2f}s)")


    # --- Validate MC shape ---
    if mc.shape != (n_animals, E_expected, E_expected):
        raise ValueError(f"Unexpected MC shape {mc.shape}, expected {(n_animals, E_expected, E_expected)}")

    finite_mask = np.isfinite(mc)
    finite_frac = float(finite_mask.mean()) # overall fraction of finite values
    print("finite fraction:", float(finite_frac))

    diag = np.array([np.diag(mc[a]) for a in range(n_animals)])
    diag_mean, diag_std = float(np.nanmean(diag)), float(np.nanstd(diag))
    print("diag mean ± std:", float(np.nanmean(diag)), float(np.nanstd(diag)))

    rng = np.random.default_rng(0)
    a = int(rng.integers(0, n_animals))
    idx = rng.choice(E_expected, size=300, replace=False)
    sub = mc[a][np.ix_(idx, idx)]
    off = sub[~np.eye(sub.shape[0], dtype=bool)]
    print("off-diag mean ± std:", float(np.nanmean(off)), float(np.nanstd(off)))
    print("off-diag min/max:", float(np.nanmin(off)), float(np.nanmax(off)))

    off_stats = dict(
        mean=float(np.nanmean(off)),
        std=float(np.nanstd(off)),
        min=float(np.nanmin(off)),
        max=float(np.nanmax(off)),
    )

    params = dict(dataset=DATASET, window_size=WINDOW_SIZE, lag=LAG, n_jobs=N_JOBS, n_animals=n_animals, n_regions=regions)
    sanity = dict(
        seconds=dt,
        finite_fraction=finite_frac,
        diag_mean=diag_mean,
        diag_std=diag_std,
        off_diag=off_stats,
        mc_shape=list(mc.shape),
        regions=regions,
        E=E_expected,
    )
    out_dir = Path(paths["mc"]) / "mc_raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mc_raw_w={WINDOW_SIZE}_lag={LAG}_animals={n_animals}_regions={regions}.npz"

    np.savez_compressed(
        out_path,
        mc=mc,
        mouse_ids=mouse_ids,
        mouse_ids_ts=mouse_ids_ts,
        age_ts=age_ts,
        params_json=json.dumps(params, sort_keys=True),
        sanity_json=json.dumps(_jsonable(sanity), sort_keys=True),
    )
    print("[OK] Saved", out_path)
    # print(out_path)

if __name__ == "__main__":
    main()

# %%
