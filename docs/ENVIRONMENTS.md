# Environments

## Dev Setup (TL;DR)

```bash
# 1) Create env (Conda recommended)
conda create -n funcog python=3.11 -y && conda activate funcog
conda install -y -c conda-forge numpy scipy numba numexpr joblib tqdm pandas matplotlib seaborn scikit-learn networkx statsmodels openpyxl python-dotenv
pip install duecredit webcolors statannotations "git+https://github.com/fiuneuro/brainconn.git"

# 2) Install local package (needed by allegiance/src)
pip install -e shared_code

# 3) Configure paths (env vars or .env)
export DATASET_NAME=ines_abdullah
# optional: PATHS_ROOT or PATHS_ENV/PROJECT_ROOT_*

# 4) Quick sanity
python - <<'PY'
from shared_code.fun_paths import get_paths
print('paths OK:', sorted(get_paths().keys())[:5])
PY
```


Guides for setting up a reliable Python environment for this repo, including a Conda workflow (recommended) and a pip/venv alternative. Targets Python 3.11.

## Overview

- Env name: `funcog`
- Python: 3.11
- Core libs: `numpy`, `scipy`, `numba`, `numexpr`, `joblib`, `tqdm`, `python-dotenv`, `brainconn`
- Analysis/plotting extras: `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `networkx`, `statsmodels`, `webcolors`, `statannotations`, `openpyxl`
- Local package: install `shared_code/` in editable mode

Note: Path resolution is profile‑driven via `shared_code/shared_code/fun_paths.py`.
You can either set a hard project root or select a profile label:

Environment variables (any shell or in a `.env`):
- `PATHS_ROOT` (optional): hard override for the project root.
- `PATHS_ENV` (optional): profile label used to pick a `PROJECT_ROOT_<ENV>`.
- `PROJECT_ROOT_<ENV>`: absolute path for a given profile (e.g., `PROJECT_ROOT_LOCAL`, `PROJECT_ROOT_CLUSTER_FS`).
- `DATASET_NAME` (optional): dataset subfolder name.

## Conda (recommended)

```bash
# Create and activate
conda create -n funcog python=3.11 -y
conda activate funcog

# Core scientific stack (conda-forge preferred for consistency)
conda install -y -c conda-forge \
  numpy scipy numba numexpr joblib tqdm python-dotenv \
  pandas matplotlib seaborn scikit-learn networkx statsmodels openpyxl

# Packages installed via pip (PyPI/GitHub)
pip install duecredit \
  "git+https://github.com/fiuneuro/brainconn.git" \
  webcolors statannotations

# Install local package in editable mode
pip install -e shared_code

# Optional: enable kernel in Jupyter
python -m ipykernel install --user --name funcog --display-name "Python (funcog)"
```

## pip + venv (alternative)

```bash
# Create venv (Python 3.11)
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip wheel setuptools

# Install dependencies
python -m pip install \
  numpy scipy numba numexpr joblib tqdm python-dotenv \
  pandas matplotlib seaborn scikit-learn networkx statsmodels openpyxl \
  duecredit \
  "git+https://github.com/fiuneuro/brainconn.git" \
  webcolors statannotations

# Install local package in editable mode
python -m pip install -e shared_code
```

## OS Dependencies

- Using Conda: none required (BLAS, compilers, and linked libs are provided by Conda packages).
- Using pip wheels: typically none on mainstream platforms. If a wheel is unavailable and a source build is attempted (rare), you may need:
  - Linux: `build-essential` (gcc/g++), `libopenblas-dev` (or system BLAS), and OpenMP runtime.
  - macOS: Xcode Command Line Tools.

Numba/NumPy/Scipy are widely distributed as wheels; prefer pinned Python (3.11) to match available wheels.

## Environment Variables (paths)

`shared_code.fun_paths` reads these variables to build canonical dataset/results/fig paths. Examples:

```bash
# Simple: hard override
export PATHS_ROOT=/abs/path/to/project/root
export DATASET_NAME=ines_abdullah

# Profile‑based: select a label and provide its root
export PATHS_ENV=CLUSTER_FS
export PROJECT_ROOT_CLUSTER_FS=/scratch/$USER/project_root
export DATASET_NAME=ines_abdullah

# Doctor: inspect + create + check write
python scripts/paths_doctor.py --show --check-write --create
```

## Basic Checks

After activation and installation, run:

```bash
# Confirm Python and packages
python -V
python -m pip list | rg -E "numpy|scipy|numba|numexpr|pandas|matplotlib|brainconn|shared_code"

# Import smoke tests
python - << 'PY'
import numpy as np
from shared_code.fun_dfcspeed import ts2dfc_stream, dfc_speed
print("shared_code import OK")
PY

# Validate path configuration (recommended)
python scripts/paths_doctor.py --show --check-write --create

# Run minimal tests (pytest if available)
pytest -q || true

# Or run the provided test script directly
python test_unified_dfc_speed.py
```

If `pytest` is not installed or tests are purely script-based, the last command suffices to validate core functionality.

## Cluster Notes (no Slurm)

- Long runs: use `tmux` or `nohup` to keep jobs alive after logout.
  - `tmux new -s boots && bash scripts/run_bootstrap_batches.sh both |& tee boots.log`
  - or `nohup bash scripts/run_bootstrap_batches.sh compute > boots.out 2>&1 & disown`
- Avoid thread oversubscription on shared nodes:
  - Set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` (the compute CLI auto‑caps when `--jobs > 1`).
