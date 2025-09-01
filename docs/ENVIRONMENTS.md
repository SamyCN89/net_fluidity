# Environments

Guides for setting up a reliable Python environment for this repo, including a Conda workflow (recommended) and a pip/venv alternative. Targets Python 3.11.

## Overview

- Env name: `funcog`
- Python: 3.11
- Core libs: `numpy`, `scipy`, `numba`, `numexpr`, `joblib`, `tqdm`, `python-dotenv`, `brainconn`
- Analysis/plotting extras: `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `networkx`, `statsmodels`, `webcolors`, `statannotations`, `openpyxl`
- Local package: install `shared_code/` in editable mode

Note: Some scripts read environment variables (see `shared_code/shared_code/fun_paths.py`), e.g. `PROJECT_ROOT_LOCAL` and optional `DATASET_NAME`. A `.env` at repo root can set these.

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

`shared_code.fun_paths` reads environment variables for data/results roots. You can set them in your shell or in a `.env` at the project root:

```bash
# .env example
PROJECT_ROOT_LOCAL=/absolute/path/to/project/root
DATASET_NAME=ines_abdullah
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

# Run minimal tests (pytest if available)
pytest -q || true

# Or run the provided test script directly
python test_unified_dfc_speed.py
```

If `pytest` is not installed or tests are purely script-based, the last command suffices to validate core functionality.
