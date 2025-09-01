# Environment Setup

This guide covers two supported setups for this project: a Conda environment named `funcog`, and a pure pip + venv setup. It also lists OS-level prerequisites and quickstart commands to verify your installation.

## Python Version

- Recommended: Python 3.11 (matches repository guidance and ensures broad wheel availability for scientific packages).
- Other versions: Python 3.10–3.12 typically work; Python 3.13 may work but some packages may still be rolling out wheels.

## OS Dependencies (if building wheels)

Most users on Linux/macOS/Windows will get prebuilt wheels via pip or conda and won’t need these. If you hit build errors (e.g., for SciPy/Numba), install:

- Ubuntu/Debian
  - `sudo apt-get update && sudo apt-get install -y build-essential python3-dev libopenblas-dev liblapack-dev gfortran`
- macOS
  - Xcode Command Line Tools: `xcode-select --install`
  - Optional BLAS: `brew install openblas`
- Windows
  - Install “Desktop development with C++” (MSVC) via Visual Studio Build Tools.

Conda users generally don’t need system compilers; conda-forge provides binary builds for the scientific stack.

## Conda Setup (env: funcog)

```bash
# Create and activate env (Python 3.11)
conda create -n funcog python=3.11 -y
conda activate funcog

# Install scientific stack from conda-forge
conda install -c conda-forge -y \
  numpy scipy numba numexpr joblib tqdm pandas matplotlib python-dotenv

# Install packages not always on conda (via pip)
pip install brainconn

# Install the local shared package in editable mode
pip install -e shared_code

# Optional: tools for notebooks/plots
conda install -c conda-forge -y jupyterlab seaborn
```

## Pure Pip + venv

```bash
# Create and activate a virtual environment (use your Python 3.11)
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip wheel

# Install runtime dependencies
pip install numpy scipy numba numexpr joblib tqdm pandas matplotlib python-dotenv brainconn

# Install the local shared package in editable mode
pip install -e shared_code
```

## Quickstart

```bash
# Verify import
python - <<'PY'
import shared_code
print('shared_code OK:', shared_code.__file__)
PY

# Run smoke tests (no external data required)
python test_unified_dfc_speed.py
python test_compatibility.py
```

## Optional: Environment Variables for Paths

Some helpers in `shared_code.fun_paths` read environment variables via `python-dotenv`. Create a `.env` file at repo root:

```bash
cat > .env <<'ENV'
# Used by shared_code/shared_code/fun_paths.py
PROJECT_ROOT_LOCAL=/absolute/path/to/project/root
DATASET_NAME=ines_abdullah
ENV
```

These variables are optional unless you use the path helpers. If set, the tool will construct standard subfolders under `dataset/`, `results/`, and `fig/` beneath the selected root.

## Notes

- Prefer the unified APIs in `shared_code/` for new work; `metaconnectivity/` and `julien_data/` contain research/legacy scripts.
- If joblib warns about permissions and disables parallelism, the computations still run in serial mode.
- If you need GPU acceleration, none of the current modules require CUDA; CPU with OpenBLAS/MKL is sufficient.

