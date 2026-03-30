#!/usr/bin/env python3
"""
dev_group_dists.py
==================
Sandbox para ajustar plot_group_speed_distributions sin tocar el pipeline principal.

Carga un PKL de bootstrap y reconstruye el eje de centros igual que
plot_speed_distributions_ines.py, luego llama a una copia local de la función
de plotting que puedes modificar libremente.

Una vez satisfecho, copia el cuerpo de plot_group_speed_distributions_dev
sobre plot_group_speed_distributions en dfc_speed_lib.py (línea ~2001).

Uso
---
  PYTHONPATH=~/Bureau/vscode/net_fluidity \
  python scripts/speed_clean/dev_group_dists.py

  # Para probar un agrupamiento distinto:
  PYTHONPATH=... python scripts/speed_clean/dev_group_dists.py --grouping age_sex
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── importaciones del proyecto ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from dfc_speed_lib import (
    _TAB20,
    _pretty_label,
    flatten_windows,
    global_min_max,
    load_speed_stack,
)
from scripts.dfc.dfc_compute import DATASET_DEFAULTS, _canonical_dataset
from shared_code.fun_loaddata import load_timeseries_bundle
from shared_code.fun_paths import get_paths
from shared_code.fun_utils import load_cognitive_data, set_figure_params

# =============================================================================
# CONFIG  —  mismos valores que plot_speed_distributions_ines.py
# =============================================================================

DATASET_NAME       = "ines"
POOL_SPLIT         = "third"
BINS_HIST          = 200
TIME_WINDOWS_RANGE = np.arange(5, 100, 1)
SPEED_LAG          = 1
SPEED_TAU          = 2

PKL_PATTERN = (
    "bootstrap_downsample_repeat"
    "_subset_all"
    "_group_{grouping}"
    "_nresamples_10000_downsample_factor_10_seed_42.pkl"
)

# Directorio de salida para las pruebas (no sobreescribe los outputs reales)
OUT_DIR = Path("/tmp/dev_group_dists")


# =============================================================================
# HELPERS  (copiados de plot_speed_distributions_ines.py)
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


def _load_bundle(paths: dict):
    preprocessed_root = Path(paths["preprocessed"])
    return load_timeseries_bundle(preprocessed_root / "ts_and_meta_2m4m.npz")


def _speed_template(paths: dict) -> str:
    return str(
        Path(paths["speed"])
        / f"all/speed_win{{w}}_lag{SPEED_LAG}_tau{SPEED_TAU}"
          "_animals_{n_animals}_regions_{regions}.npz"
    )


# =============================================================================
# ── EDITA AQUÍ ────────────────────────────────────────────────────────────────
# Copia local de plot_group_speed_distributions — modifica a tu gusto.
# Cuando quedes conforme, pega el cuerpo en dfc_speed_lib.py ~línea 2001.
# =============================================================================

def plot_group_speed_distributions_dev(
    group_means: dict,
    centers: np.ndarray,
    seg_names: list[str],
    groups_selected: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Copia de desarrollo de plot_group_speed_distributions.

    Parámetros
    ----------
    group_means     : dict  seg_name → {group_key: mean_hist ndarray}
    centers         : array de bin-centers para el eje X de velocidad
    seg_names       : lista ordenada de segmentos (columnas de la figura)
    groups_selected : nombre del agrupamiento (título de la leyenda)
    save_path       : si se indica, guarda la figura aquí a 300 dpi
    """
    n_seg = len(seg_names)
    fig, axes = plt.subplots(2, n_seg, figsize=(6 * n_seg, 8), sharex=True)
    if n_seg == 1:
        axes = np.array([axes]).reshape(2, 1)

    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=_TAB20)

    for col, seg in enumerate(seg_names):
        means = group_means.get(seg, {})

        for scale_row, (ax, use_log) in enumerate(
            zip(axes[:, col], [False, True], strict=False)
        ):
            for gt, hist in means.items():
                ax.plot(
                    centers,
                    hist,
                    lw=1.2,
                    alpha=0.8,
                    label=_pretty_label(gt) if not scale_row else None,
                )

            ax.set_xlabel("Speed")
            ax.set_ylabel("Density" + (" (log)" if use_log else ""))
            ax.set_title(f"{seg} ({'log' if use_log else 'linear'} scale)")
            ax.grid(True, which="both", ls="--", lw=0.4)
            if use_log:
                ax.set_yscale("log")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        dict(zip(labels, handles, strict=False)).values(),
        dict(zip(labels, handles, strict=False)).keys(),
        title=groups_selected,
        loc="center left",
        bbox_to_anchor=(0.92, 0.5),
        fontsize=10,
        frameon=False,
    )
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[OK] Figura guardada → {save_path}")
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main(grouping: str) -> None:
    set_figure_params(True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── rutas y datos ─────────────────────────────────────────────────────────
    paths  = _build_paths()
    bundle = _load_bundle(paths)
    n_animals = int(bundle.n_animals)
    regions   = int(bundle.n_regions)

    # ── PKL ──────────────────────────────────────────────────────────────────
    pkl_path = Path(paths["speed"]) / "bootstrap" / PKL_PATTERN.format(grouping=grouping)
    if not pkl_path.exists():
        print(f"[ERROR] PKL no encontrado:\n  {pkl_path}")
        sys.exit(1)

    print(f"[INFO] Cargando PKL: {pkl_path.name}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    pool_group_hists = data.get("pooled_group_hists_by_segment", {})
    seg_names        = list(data["ranges"].keys())

    if not pool_group_hists:
        print("[ERROR] El PKL no contiene 'pooled_group_hists_by_segment'.")
        sys.exit(1)

    # ── reconstruir eje de centros (idéntico al script principal) ─────────────
    print("[INFO] Cargando speed stack para reconstruir bin centers …")
    speeds   = load_speed_stack(
        _speed_template(paths), TIME_WINDOWS_RANGE, n_animals, regions
    )
    all_flat = flatten_windows(speeds, 0, len(speeds))
    edges    = np.linspace(*global_min_max([all_flat]), BINS_HIST + 1)
    centers  = 0.5 * (edges[:-1] + edges[1:])

    # ── plot ──────────────────────────────────────────────────────────────────
    save_path = OUT_DIR / f"group_dists_dev_{grouping}.png"
    fig = plot_group_speed_distributions_dev(
        group_means=pool_group_hists,
        centers=centers,
        seg_names=seg_names,
        groups_selected=grouping,
        save_path=save_path,
    )
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sandbox para group dists plot")
    parser.add_argument(
        "--grouping",
        default="age",
        help="Agrupamiento a visualizar (default: age)",
    )
    args = parser.parse_args()
    main(args.grouping)
