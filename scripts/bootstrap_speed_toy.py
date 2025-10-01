#!/usr/bin/env python3
"""
Toy script: bootstrap confidence intervals (CIs) for dFC speed.

Two modes:
  1) Synthetic toy data (default): generate per-animal speed arrays and bootstrap CIs
  2) From NPZ: load a `speed_win*.npz` file (key: 'speeds') and bootstrap CIs

Usage examples:
  # Synthetic 5 animals × 4 tau × 200 samples
  python scripts/bootstrap_speed_toy.py --n-animals 5 --n-tau 4 --tlen 200 --n-boot 2000

  # From a real NPZ speeds file (per-window output)
  python scripts/bootstrap_speed_toy.py --npz /path/to/speed_win9_..._animals_48_regions_37.npz --tau-index 0 --n-boot 5000

Notes:
- The NPZ should have key 'speeds' with an object array of length n_animals;
  each entry is a 2D array (n_tau, T_w). This matches the repo's speed outputs.
"""
#%%
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Callable

import numpy as np
import sys; sys.argv = [""]

#%%

def _bootstrap_ci_1d(x: np.ndarray, n_boot: int = 2000, stat: str = "median", ci: float = 95.0, random_state: int | None = 0) -> tuple[float, float, float]:
    """
    Basic bootstrap CI for a 1D array x (ignoring NaNs).
    Returns (estimate, lo, hi) where estimate = chosen statistic on original data.
    """
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    if stat == "median":
        stat_fn: Callable[[np.ndarray], float] = lambda a: float(np.median(a))
    elif stat == "mean":
        stat_fn = lambda a: float(np.mean(a))
    elif stat.startswith("q"):
        q = float(stat[1:]) / 100.0
        stat_fn = lambda a: float(np.quantile(a, q))
    else:
        raise ValueError("stat must be 'median', 'mean' or 'qXX'")

    est = stat_fn(x)
    rng = np.random.default_rng(random_state)
    boots = np.empty(n_boot, float)
    for i in range(n_boot):
        idx = rng.choice(x.size, size=x.size, replace=True)
        boots[i] = stat_fn(x[idx])
    alpha = (100.0 - float(ci)) / 2.0
    lo = float(np.percentile(boots, alpha))
    hi = float(np.percentile(boots, 100.0 - alpha))
    return est, lo, hi


def _load_npz_speeds(path: Path, tau_index: int | None = None) -> list[np.ndarray]:
    """
    Load speeds from an NPZ per-window file (key: 'speeds').
    Returns a list of per-animal arrays (pooled over taus if tau_index is None).
    """
    z = np.load(path, allow_pickle=True)
    if "speeds" not in z:
        raise KeyError(f"NPZ file missing 'speeds' key: {path}")
    speeds = z["speeds"]  # object array length n_animals; each entry 2D (n_tau, T_w)
    per_animal: list[np.ndarray] = []
    for a in range(len(speeds)):
        arr = np.asarray(speeds[a], float)
        if arr.ndim != 2:
            per_animal.append(np.array([], float))
            continue
        if tau_index is None:
            vals = arr[~np.isnan(arr)]
        else:
            if tau_index < 0 or tau_index >= arr.shape[0]:
                vals = np.array([], float)
            else:
                vals = arr[tau_index][~np.isnan(arr[tau_index])]
        per_animal.append(vals)
    return per_animal


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Bootstrap CIs for dFC speed (toy)", allow_abbrev=False)
    ap.add_argument("--npz", type=str, default=None, help="Path to a per-window speeds NPZ (key='speeds')")
    ap.add_argument("--tau-index", type=int, default=None, help="Tau index; if omitted, pools all taus")
    ap.add_argument("--n-boot", type=int, default=2000, help="Bootstrap resamples")
    ap.add_argument("--ci", type=float, default=95.0, help="Confidence interval percent")
    ap.add_argument("--stat", type=str, default="median", help="Statistic: median|mean|qXX")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    # Synthetic params
    ap.add_argument("--n-animals", type=int, default=5)
    ap.add_argument("--n-tau", type=int, default=4)
    ap.add_argument("--tlen", type=int, default=200)
    return ap.parse_args()


def run_bootstrap(
    *,
    npz: str | None = None,
    tau_index: int | None = None,
    n_boot: int = 2000,
    ci: float = 95.0,
    stat: str = "median",
    seed: int = 0,
    n_animals: int = 5,
    n_tau: int = 4,
    tlen: int = 200,
):
    """
    Notebook-friendly runner: returns results instead of exiting.

    Returns dict: {mode, per_animal: [(idx, est, lo, hi, n)], pooled: (est, lo, hi, n)}
    """
    rng = np.random.default_rng(seed)
    if npz:
        per_animal = _load_npz_speeds(Path(npz), tau_index=tau_index)
        mode = f"npz: {npz}"
    else:
        per_animal = []
        for _a in range(n_animals):
            base = rng.beta(2.0, 5.0, size=(n_tau, tlen)).astype(float) * 2.0
            mask = rng.random(size=base.shape) < 0.1
            base[mask] = np.nan
            if tau_index is None:
                vals = base[~np.isnan(base)]
            else:
                vals = base[tau_index][~np.isnan(base[tau_index])]
            per_animal.append(vals)
        mode = f"synthetic: n_animals={n_animals}, n_tau={n_tau}, tlen={tlen}"

    rows = []
    for i, arr in enumerate(per_animal):
        est, lo, hi = _bootstrap_ci_1d(arr, n_boot=n_boot, stat=stat, ci=ci, random_state=seed + i)
        rows.append((i, est, lo, hi, arr.size))
    pooled = np.concatenate([a for a in per_animal if a.size > 0]) if per_animal else np.array([])
    if pooled.size > 0:
        pest, plo, phi = _bootstrap_ci_1d(pooled, n_boot=n_boot, stat=stat, ci=ci, random_state=seed + 12345)
        overall = (pest, plo, phi, pooled.size)
    else:
        overall = (np.nan, np.nan, np.nan, 0)

    return {
        "mode": mode,
        "per_animal": rows,
        "pooled": overall,
        "stat": stat,
        "n_boot": n_boot,
        "ci": ci,
    }


def main() -> int:
    args = parse_args()
    res = run_bootstrap(
        npz=args.npz,
        tau_index=args.tau_index,
        n_boot=args.n_boot,
        ci=args.ci,
        stat=args.stat,
        seed=args.seed,
        n_animals=args.n_animals,
        n_tau=args.n_tau,
        tlen=args.tlen,
    )

    print(f"[toy] mode={res['mode']}")
    print(f"[toy] stat={res['stat']}, n_boot={res['n_boot']}, ci={res['ci']}")
    print("\nPer-animal CIs (index, est, lo, hi, n):")
    for i, (idx, est, lo, hi, n) in enumerate(res["per_animal"][:10]):
        print(f"  {idx:02d}: {est:.4f}  [{lo:.4f}, {hi:.4f}]  n={n}")
    if len(res["per_animal"]) > 10:
        print(f"  ... ({len(res['per_animal'])} total)")
    pest, plo, phi, pn = res["pooled"]
    if pn > 0:
        print(f"\nOverall pooled: est={pest:.4f}  [{plo:.4f}, {phi:.4f}]  n={pn}")
    else:
        print("\nOverall pooled: no data")
    
    return 0


if __name__ == "__main__":
    # Avoid SystemExit in IPython notebooks
    try:
        from IPython import get_ipython  # type: ignore
    except Exception:
        get_ipython = None  # type: ignore
    if get_ipython is not None and get_ipython():  # running inside IPython
        _ = main()
    else:
        raise SystemExit(main())


# %%
