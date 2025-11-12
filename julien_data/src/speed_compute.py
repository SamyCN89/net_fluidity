#!/usr/bin/env python3
"""
Thin, low-risk CLI for dFC speed computation that delegates to the
existing julien_data/3_dfc_speed_test_v6.py script without changing
algorithms, parameters, file naming, or outputs.

Usage:
  python julien_data/src/speed_compute.py --help
  python julien_data/src/speed_compute.py [same args as 3_dfc_speed_test_v6.py]

Notes:
- This wrapper only sets up logging and imports the original script
  in a way that preserves its behavior and CLI.
- All arguments are parsed by the original script; this wrapper
  forwards sys.argv unchanged.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path


def setup_logging() -> None:
    """Configure logging from YAML if available; fallback to basicConfig.

    Purely cosmetic; underlying compute runs in a subprocess.
    """
    cfg_path = os.getenv("NET_FLUIDITY_LOGGING", "config/logging.yaml")
    try:
        cfg_file = Path(cfg_path)
        if cfg_file.exists():
            from logging.config import dictConfig

            import yaml

            with cfg_file.open("r") as f:
                dictConfig(yaml.safe_load(f))
            return
    except Exception:
        pass
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def main() -> int:
    setup_logging()
    logger = logging.getLogger(__name__)

    here = Path(__file__).resolve()
    julien_dir = here.parent.parent  # .../julien_data
    v6_path = julien_dir / "3_dfc_speed_test_v6.py"
    if not v6_path.exists():
        logger.error("Underlying script not found: %s", v6_path)
        return 2

    # Run the original script as a subprocess to preserve joblib multiprocessing behavior.
    cmd = [sys.executable, str(v6_path), *sys.argv[1:]]
    logger.info("Delegating to original script: %s", v6_path.name)
    try:
        proc = subprocess.run(cmd, cwd=str(julien_dir), check=False)
        return int(proc.returncode)
    except KeyboardInterrupt:  # pragma: no cover
        return 130
    except Exception as e:  # pragma: no cover
        logger.exception("Failed to run delegated script: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
