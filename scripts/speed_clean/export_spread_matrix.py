#!/usr/bin/env python3
"""
export_spread_matrix.py
=======================
Exporta los resultados de spread de velocidad dFC en formato manipulable.

- Agrupamientos con 2 grupos  → compute_spread_matrix
- Agrupamientos con >2 grupos → compute_spread_matrix_all_pairs (todos los pares)

Caching
-------
  Level 1: animal_spread (percentiles por animal) se guarda en NPZ.
           Es lo más costoso (carga ~3900 NPZ + cómputo de percentiles).
           Controlado por FORCE_RECOMPUTE_SPREADS.

  Level 2: resultados bootstrap por grouping se guardan en PKL individuales.
           cache/{grouping}_N{n_boot}.pkl
           Controlado por FORCE_RECOMPUTE_BOOTSTRAP.
           Si FORCE_RECOMPUTE_SPREADS=True, se invalida también este cache.

Outputs
-------
  <f_speed>/analysis_ines/spread/exported/
    cache/animal_spread_lag{L}_tau{T}_nw{N}.npz  — cache Level 1
    cache/{grouping}_N{n_boot}.pkl               — cache Level 2 (uno por grouping)
    spread_matrix_results.pkl                     — dict completo con arrays numpy
    spread_matrix_long.csv                        — tabla long-format, una fila por celda

Columnas del CSV
----------------
  grouping   : ej. "age", "genotype", "age_genotype"
  group0     : primer grupo del par (Δ = group1 - group0)
  group1     : segundo grupo del par
  metric     : "q95q05" o "q99q01"
  segment    : "short", "mid", "long" (según POOL_SPLIT)
  location   : nombre de región o "all"
  delta      : diferencia de medias (group1 - group0)
  sig        : True/False — bootstrap CI excluye 0
  delta_sig  : delta si sig, NaN si ns

Uso
---
  PYTHONPATH=~/Bureau/vscode/net_fluidity \
  python scripts/speed_clean/export_spread_matrix.py
"""
#%%
from __future__ import annotations

import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent))

from dfc_speed_lib import (
    _bootstrap_diff_ci,
    cdf_split_indices,
    discover_per_region_descriptors,
    get_group_data,
    load_speed_stack,
    load_speed_stack_single_region,
    select_windows,
)
from scripts.dfc.dfc_compute import DATASET_DEFAULTS, _canonical_dataset
from shared_code.fun_loaddata import load_timeseries_bundle
from shared_code.fun_paths import get_paths
from shared_code.fun_utils import load_cognitive_data, set_figure_params

# =============================================================================
# CONFIG
# =============================================================================

DATASET_NAME       = "ines"
POOL_SPLIT         = "third"
TIME_WINDOWS_RANGE = np.arange(5, 100, 1)
SPEED_LAG          = 1
SPEED_TAU          = 2
METRICS            = ["q95q05", "q99q01"]
N_BOOT             = 5000

# Set True to recompute animal_spread from scratch (ignores Level 1 cache)
# Note: also invalidates Level 2 cache (bootstrap depends on spreads)
FORCE_RECOMPUTE_SPREADS   = False

# Set True to recompute bootstrap results (ignores Level 2 cache, reuses Level 1)
FORCE_RECOMPUTE_BOOTSTRAP = False

# Parallel workers for bootstrap loop (-1 = all cores, 1 = sequential)
N_JOBS = -1

GROUPINGS = [
    "age",
    "sex",
    "genotype",
    "age_sex",
    "age_genotype",
    # "age_phenotype_oip",
    # "age_phenotype_nor",
    "age_sex_genotype",
    # "age_sex_phenotype_oip",
    # "age_sex_phenotype_nor",
]

# Percentile pairs per metric
_PCT_MAP     = {"q95q05": [5, 95], "q99q01": [1, 99]}
_MIN_SAMPLES = {"q95q05": 10,      "q99q01": 20}


# =============================================================================
# PATH HELPERS
# =============================================================================

def _build_paths() -> dict:
    dataset = _canonical_dataset(DATASET_NAME)
    cfg = DATASET_DEFAULTS[dataset]
    return get_paths(
        dataset_name=dataset,
        timecourse_folder=cfg["timecourse_folder"],
        cognitive_data_file=cfg["cognitive_data_file"],
        anat_labels_file=cfg["anat_labels_file"],
    )


def _load_bundle_and_cog(paths: dict):
    preprocessed_root = Path(paths["preprocessed"])
    bundle = load_timeseries_bundle(preprocessed_root / "ts_and_meta_2m4m.npz")
    n  = int(bundle.n_animals)
    r  = int(bundle.n_regions)
    tr = int(bundle.total_tr)
    cog_path = (
        preprocessed_root
        / f"cog_data_filtered_animals_{n}_regions_{r}_tr_{tr}.csv"
    )
    cog_data = load_cognitive_data(str(cog_path))
    return bundle, cog_data


def _speed_template(paths: dict) -> str:
    return str(
        Path(paths["speed"])
        / f"all/speed_win{{w}}_lag{SPEED_LAG}_tau{SPEED_TAU}"
          "_animals_{n_animals}_regions_{regions}.npz"
    )


def _get_ranges(speeds: list) -> dict:
    i3, ih, i23 = cdf_split_indices(speeds)
    return select_windows(POOL_SPLIT, len(speeds), i3, ih, i23)


def _spread_cache_path(out_dir: Path) -> Path:
    return (
        out_dir / "cache"
        / f"animal_spread_lag{SPEED_LAG}_tau{SPEED_TAU}_nw{len(TIME_WINDOWS_RANGE)}.npz"
    )


def _bootstrap_cache_path(out_dir: Path, grouping: str) -> Path:
    return out_dir / "cache" / f"{grouping}_N{N_BOOT}.pkl"


def _save_bootstrap(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f)


def _load_bootstrap(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _bootstrap_cache_valid(out_dir: Path, grouping: str) -> bool:
    """Cache is valid only if Level 1 spreads were not force-recomputed."""
    if FORCE_RECOMPUTE_SPREADS or FORCE_RECOMPUTE_BOOTSTRAP:
        return False
    return _bootstrap_cache_path(out_dir, grouping).exists()


# =============================================================================
# LEVEL 1 CACHE — animal_spread
# =============================================================================

def _compute_animal_spreads(
    speeds_map: dict[str, list],
    ranges: dict,
    metrics: list[str],
) -> dict[tuple, np.ndarray]:
    """
    Compute per-animal spread for every (location, segment, metric).

    Returns
    -------
    spreads : dict[(location, segment, metric)] → ndarray shape (n_animals,)
    """
    spreads = {}
    n_loc = len(speeds_map)
    for li, (loc, speeds) in enumerate(speeds_map.items()):
        n_animals = len(speeds[0])
        print(f"  [{li+1}/{n_loc}] {loc}")
        for seg, w_range in ranges.items():
            for metric in metrics:
                pcts  = _PCT_MAP[metric]
                min_s = _MIN_SAMPLES[metric]
                animal_spread = np.full(n_animals, np.nan)
                for i in range(n_animals):
                    samples = (
                        np.concatenate([
                            np.ravel(speeds[w][i])
                            for w in w_range
                            if w < len(speeds) and np.ravel(speeds[w][i]).size > 0
                        ])
                        if len(w_range) > 0 else np.array([])
                    )
                    if samples.size < min_s:
                        continue
                    lo, hi = np.percentile(samples, pcts)
                    animal_spread[i] = hi - lo
                spreads[(loc, seg, metric)] = animal_spread
    return spreads


def _save_animal_spreads(spreads: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Encode tuple keys as "loc__seg__metric" (double underscore as separator)
    np.savez(path, **{"__".join(k): v for k, v in spreads.items()})


def _load_animal_spreads(path: Path) -> dict[tuple, np.ndarray]:
    data = np.load(path)
    return {tuple(k.split("__")): data[k] for k in data.files}


def _get_animal_spreads(
    speeds_map: dict[str, list],
    ranges: dict,
    metrics: list[str],
    cache_path: Path,
) -> dict[tuple, np.ndarray]:
    if not FORCE_RECOMPUTE_SPREADS and cache_path.exists():
        print(f"[CACHE] animal_spread cargado desde cache ({cache_path.name})")
        return _load_animal_spreads(cache_path)
    print("[COMPUTE] Calculando animal_spread para todas las locaciones …")
    spreads = _compute_animal_spreads(speeds_map, ranges, metrics)
    _save_animal_spreads(spreads, cache_path)
    print(f"[CACHE] animal_spread guardado → {cache_path}")
    return spreads


# =============================================================================
# BOOTSTRAP — usando animal_spread pre-computado
# =============================================================================

def _bootstrap_2group(
    spreads: dict,
    group_data: dict,
    locations: list[str],
    seg_names: list[str],
    metrics: list[str],
    n_boot: int,
) -> dict:
    """
    Calcula delta/sig para exactamente 2 grupos usando animal_spread pre-computado.
    Equivalente a compute_spread_matrix pero sin recargar speeds.
    """
    group_keys = list(group_data.keys())
    assert len(group_keys) == 2

    result = {}
    for m in metrics:
        n_seg = len(seg_names)
        n_loc = len(locations)
        delta    = np.full((n_seg, n_loc), np.nan)
        sig      = np.zeros((n_seg, n_loc), dtype=bool)

        for li, loc in enumerate(locations):
            for si, seg in enumerate(seg_names):
                v0 = spreads[(loc, seg, m)][group_data[group_keys[0]]]
                v1 = spreads[(loc, seg, m)][group_data[group_keys[1]]]
                v0 = v0[np.isfinite(v0)]
                v1 = v1[np.isfinite(v1)]
                if v0.size < 2 or v1.size < 2:
                    continue
                diff, lo_d, hi_d = _bootstrap_diff_ci(
                    v0, v1, n_boot=n_boot, seed=li * 100 + si
                )
                delta[si, li]  = diff
                sig[si, li]    = not (lo_d <= 0 <= hi_d)

        result[m] = {
            "locations":  locations,
            "segments":   seg_names,
            "delta":      delta,
            "sig":        sig,
            "delta_sig":  np.where(sig, delta, np.nan),
            "group_keys": group_keys,
        }
    return result


def _bootstrap_all_pairs(
    spreads: dict,
    group_data: dict,
    locations: list[str],
    seg_names: list[str],
    metric: str,
    n_boot: int,
) -> dict:
    """
    Calcula delta masked para todos los pares usando animal_spread pre-computado.
    Equivalente a compute_spread_matrix_all_pairs pero sin recargar speeds.
    """
    group_keys = list(group_data.keys())
    pairs      = list(combinations(group_keys, 2))
    n_pairs    = len(pairs)
    n_seg      = len(seg_names)
    n_loc      = len(locations)

    delta_masked = np.full((n_pairs, n_seg, n_loc), np.nan)
    sig          = np.zeros((n_pairs, n_seg, n_loc), dtype=bool)

    for li, loc in enumerate(locations):
        for si, seg in enumerate(seg_names):
            animal_spread = spreads[(loc, seg, metric)]
            for pi, (gk0, gk1) in enumerate(pairs):
                v0 = animal_spread[group_data[gk0]]
                v1 = animal_spread[group_data[gk1]]
                v0 = v0[np.isfinite(v0)]
                v1 = v1[np.isfinite(v1)]
                if v0.size < 2 or v1.size < 2:
                    continue
                diff, lo_d, hi_d = _bootstrap_diff_ci(
                    v0, v1, n_boot=n_boot, seed=li * 1000 + si * 100 + pi
                )
                is_sig = not (lo_d <= 0 <= hi_d)
                sig[pi, si, li]          = is_sig
                delta_masked[pi, si, li] = diff if is_sig else np.nan

    return {
        "pairs":        pairs,
        "locations":    locations,
        "segments":     seg_names,
        "delta_masked": delta_masked,
        "sig":          sig,
        "metric":       metric,
    }


# =============================================================================
# CSV CONVERSION
# =============================================================================

def _result_to_rows(spread_result: dict, grouping: str) -> list[dict]:
    """2-group result → filas CSV."""
    rows = []
    for metric, res in spread_result.items():
        gk = res["group_keys"]
        for si, seg in enumerate(res["segments"]):
            for li, loc in enumerate(res["locations"]):
                rows.append({
                    "grouping":  grouping,
                    "group0":    str(gk[0]),
                    "group1":    str(gk[1]),
                    "metric":    metric,
                    "segment":   seg,
                    "location":  loc,
                    "delta":     float(res["delta"][si, li]),
                    "sig":       bool(res["sig"][si, li]),
                    "delta_sig": float(res["delta_sig"][si, li]),
                })
    return rows


def _all_pairs_result_to_rows(result: dict, grouping: str) -> list[dict]:
    """All-pairs result → filas CSV."""
    rows = []
    metric = result["metric"]
    for pi, (gk0, gk1) in enumerate(result["pairs"]):
        for si, seg in enumerate(result["segments"]):
            for li, loc in enumerate(result["locations"]):
                d = result["delta_masked"][pi, si, li]
                s = bool(result["sig"][pi, si, li])
                rows.append({
                    "grouping":  grouping,
                    "group0":    str(gk0),
                    "group1":    str(gk1),
                    "metric":    metric,
                    "segment":   seg,
                    "location":  loc,
                    "delta":     float(d) if s else float("nan"),
                    "sig":       s,
                    "delta_sig": float(d),
                })
    return rows


# =============================================================================
# PER-GROUPING WORKER  (llamado en paralelo por joblib)
# =============================================================================

def _run_grouping(
    grouping: str,
    group_data: dict,
    spreads: dict,
    locations: list[str],
    seg_names: list[str],
    out_dir: Path,
) -> tuple[str, dict, list[dict]]:
    """
    Procesa un grouping completo: cache check → bootstrap → cache save.

    Returns
    -------
    grouping : str
    entry    : {"type": ..., "data": ...}
    rows     : list of CSV row dicts
    """
    n_groups   = len(group_data)
    boot_cache = _bootstrap_cache_path(out_dir, grouping)

    if _bootstrap_cache_valid(out_dir, grouping):
        print(f"[CACHE L2] {grouping} — cargando desde cache")
        cached = _load_bootstrap(boot_cache)
        if cached["type"] == "2group":
            rows = _result_to_rows(cached["data"], grouping)
        else:
            rows = []
            for metric, result in cached["data"].items():
                rows.extend(_all_pairs_result_to_rows(result, grouping))
        return grouping, cached, rows

    if n_groups == 2:
        print(f"[BOOTSTRAP 2G] {grouping}  groups={list(group_data.keys())}")
        spread_result = _bootstrap_2group(
            spreads, group_data, locations, seg_names, METRICS, N_BOOT
        )
        entry = {"type": "2group", "data": spread_result}
        rows  = _result_to_rows(spread_result, grouping)

    else:
        n_pairs = n_groups * (n_groups - 1) // 2
        print(f"[BOOTSTRAP NG] {grouping}  {n_groups} grupos → {n_pairs} pares")
        pairs_results = {}
        rows = []
        for metric in METRICS:
            result = _bootstrap_all_pairs(
                spreads, group_data, locations, seg_names, metric, N_BOOT
            )
            pairs_results[metric] = result
            rows.extend(_all_pairs_result_to_rows(result, grouping))
        entry = {"type": "multigroup", "data": pairs_results}

    _save_bootstrap(entry, boot_cache)
    print(f"[OK] {grouping} — cache guardado")
    return grouping, entry, rows


# =============================================================================
# MAIN
# =============================================================================

#%%
def main() -> None:
    set_figure_params(False)

    paths     = _build_paths()
    bundle, cog_data = _load_bundle_and_cog(paths)
    n_animals = int(bundle.n_animals)
    regions   = int(bundle.n_regions)

    speed_root = Path(paths["speed"])
    out_dir    = Path(paths["f_speed"]) / "analysis_ines" / "spread" / "exported"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Cargar speeds (subset all) ────────────────────────────────────────────
    print("[INFO] Cargando speed stack (subset=all) …")
    speeds_all = load_speed_stack(
        _speed_template(paths), TIME_WINDOWS_RANGE, n_animals, regions,
    )
    ranges = _get_ranges(speeds_all)
    print(f"  Segmentos: { {k: (list(v)[0], list(v)[-1]) for k, v in ranges.items()} }")

    # ── Cargar speeds per_region ──────────────────────────────────────────────
    speeds_per_region: dict[str, list] = {}
    per_region_dir = speed_root / "per_region"
    if per_region_dir.exists():
        try:
            region_descs = discover_per_region_descriptors(
                per_region_dir, int(TIME_WINDOWS_RANGE[0]),
                n_animals, regions, lag=SPEED_LAG, tau_count=SPEED_TAU,
            )
            print(f"  {len(region_descs)} regiones encontradas")
            for rd in region_descs:
                try:
                    speeds_per_region[rd] = load_speed_stack_single_region(
                        per_region_dir, TIME_WINDOWS_RANGE,
                        n_animals, regions, rd,
                        lag=SPEED_LAG, tau_count=SPEED_TAU,
                    )
                except FileNotFoundError:
                    print(f"    [WARN] {rd} — falta, omitiendo")
        except FileNotFoundError:
            print("  [WARN] No se encontraron archivos per_region")
    else:
        print("  [WARN] Directorio per_region no encontrado")

    # ── Level 1 cache: animal_spread ─────────────────────────────────────────
    speeds_map = {**speeds_per_region, "all": speeds_all}
    locations  = list(speeds_per_region.keys()) + ["all"]
    seg_names  = list(ranges.keys())

    cache_path = _spread_cache_path(out_dir)
    spreads    = _get_animal_spreads(speeds_map, ranges, METRICS, cache_path)

    # ── Computar bootstrap en paralelo ───────────────────────────────────────
    group_data_map = {g: get_group_data(cog_data, DATASET_NAME, g) for g in GROUPINGS}

    results_list = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_run_grouping)(
            grouping, group_data_map[grouping], spreads, locations, seg_names, out_dir
        )
        for grouping in GROUPINGS
    )

    # Reassemble in original order
    all_results: dict[str, dict] = {}
    all_rows: list[dict] = []
    for grouping, entry, rows in results_list:
        all_results[grouping] = entry
        all_rows.extend(rows)

    # ── Guardar pickle ────────────────────────────────────────────────────────
    pkl_path = out_dir / "spread_matrix_results.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\n[OK] Pickle guardado → {pkl_path}")

    # ── Guardar CSV long-format ───────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "spread_matrix_long.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] CSV guardado    → {csv_path}")
    print(f"\nDimensiones del DataFrame: {df.shape}")
    print(df.head(10).to_string(index=False))

    # ── Resumen: celdas significativas por grouping y par ─────────────────────
    print("\n── Resumen: celdas significativas ──")
    summary = (
        df[df["sig"]]
        .groupby(["grouping", "group0", "group1", "metric"])
        .size()
        .rename("n_sig")
        .reset_index()
    )
    total = (
        df.groupby(["grouping", "group0", "group1", "metric"])
        .size()
        .rename("n_total")
        .reset_index()
    )
    summary = summary.merge(total, on=["grouping", "group0", "group1", "metric"])
    summary["pct_sig"] = (100 * summary["n_sig"] / summary["n_total"]).round(1)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
