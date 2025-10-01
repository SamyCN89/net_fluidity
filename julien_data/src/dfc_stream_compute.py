#!/usr/bin/env python3
"""
Thin wrapper CLI that runs julien_data/2_compute_dfc_stream.py as a subprocess.

This preserves the original behavior and parameters; any arguments passed
to this wrapper are forwarded but currently ignored by the underlying script.

Usage:
  python julien_data/src/dfc_stream_compute.py
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path


def setup_logging() -> None:
    cfg_path = os.getenv("NET_FLUIDITY_LOGGING", "config/logging.yaml")
    try:
        cfg = Path(cfg_path)
        if cfg.exists():
            from logging.config import dictConfig
            import yaml

            with cfg.open("r") as f:
                dictConfig(yaml.safe_load(f))
            return
    except Exception:
        pass
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def main() -> int:
    setup_logging()
    logger = logging.getLogger(__name__)

    here = Path(__file__).resolve()
    julien_dir = here.parent.parent
    script = julien_dir / "2_compute_dfc_stream.py"
    if not script.exists():
        logger.error("Underlying script not found: %s", script)
        return 2

    cmd = [sys.executable, str(script), *sys.argv[1:]]
    logger.info("Delegating to original script: %s", script.name)
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

