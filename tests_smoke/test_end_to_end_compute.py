import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import subprocess


def write_synthetic_context(root: Path) -> dict:
    ds = "testset"
    paths = {
        "dataset": root / "dataset" / ds,
        "preprocessed": root / "results" / ds / "preprocessed_data",
        "speed": root / "results" / ds / "speed",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    # Minimal metadata pickle matching DFCAnalysis expectations
    n_animals = 4
    regions = 48
    tr = 500
    meta = {
        "mouse_metadata": None,
        "region_labels": [f"R{i}" for i in range(regions)],
        "n_animals": n_animals,
        "regions": regions,
        "total_tr": tr,
        "lag": 1,
        "tau": 2,
        "window_range": (5, 20, 1),
    }
    meta_path = paths["preprocessed"] / f"metadata_animals_{n_animals}_regions_{regions}_tr_{tr}.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    # Minimal preprocessed ts npz
    ts = np.zeros((n_animals, regions, 10), dtype=float)
    np.savez(paths["preprocessed"] / f"ts_filtered_animals_{n_animals}_regions_{regions}_tr_{tr}.npz", ts=ts)

    # Cognitive CSV with grouping columns and a NOR column
    df = pd.DataFrame(
        {
            "genotype": ["WT", "WT", "Dp1Yey", "Dp1Yey"],
            "treatment": ["VEH", "LCTB92", "VEH", "LCTB92"],
            "index_NOR": [0.5, 0.6, 0.4, 0.45],
        }
    )
    df_path = paths["preprocessed"] / f"cog_data_filtered_animals_{n_animals}_regions_{regions}_tr_{tr}.csv"
    df.to_csv(df_path, index=False)

    # Synthetic per-window per-region speeds under subset
    subset = "regions500"
    region_dir = paths["speed"] / subset / "regions-ACC"
    region_dir.mkdir(parents=True, exist_ok=True)
    # Create a single window file with small arrays
    speeds = np.empty((n_animals,), dtype=object)
    for i in range(n_animals):
        arr = np.random.RandomState(i).randn(2, 20).astype(float)
        speeds[i] = arr
    np.savez(region_dir / "speed_win9_synth.npz", speeds=speeds)

    return {"root": root, "subset": subset}


def test_end_to_end_compute(tmp_path: Path):
    ctx = write_synthetic_context(tmp_path)
    env = os.environ.copy()
    env["PATHS_ROOT"] = str(ctx["root"])  # forces fun_paths to use this root
    env["DATASET_NAME"] = "testset"

    # Run compute with small n_boot and minimal q to be fast
    cmd = [
        "python",
        "scripts/compute_speed_bootstrap.py",
        "--tr",
        "500",
        "--subset",
        ctx["subset"],
        "--tau-index",
        "0",
        "--n-boot",
        "50",
        "--q",
        "50",
        "--jobs",
        "1",
        "--parallel-scope",
        "windows",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(Path.cwd()))
    if res.returncode != 0:
        raise AssertionError(f"compute CLI failed: {res.returncode}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")

    # Verify outputs exist and have expected columns
    outdir = tmp_path / "results" / "testset" / "speed" / ctx["subset"]
    q_csv = outdir / "speed_bootstrap_quantiles.csv"
    d_csv = outdir / "speed_bootstrap_diffs.csv"
    assert q_csv.exists(), f"Missing quantiles CSV at {q_csv}"
    assert d_csv.exists(), f"Missing diffs CSV at {d_csv}"

    qdf = pd.read_csv(q_csv)
    ddf = pd.read_csv(d_csv)
    assert {"region", "roi", "window", "group", "q", "point", "lo", "hi", "n"}.issubset(
        set(qdf.columns)
    )
    assert {"region", "roi", "window", "A", "B", "q", "diff", "lo", "hi"}.issubset(
        set(ddf.columns)
    )
    # At least one row per file in this tiny setup
    assert len(qdf) >= 1
    assert len(ddf) >= 1

