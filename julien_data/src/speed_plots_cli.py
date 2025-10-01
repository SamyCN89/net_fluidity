#!/usr/bin/env python3
"""
Thin CLI wrapper that delegates to julien_data/plot_merged_speed.py.

For help and options, run:
  python julien_data/src/speed_plots_cli.py --help

This preserves plotting behavior and accepts the same flags as the underlying script.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    here = Path(__file__).resolve()
    julien_dir = here.parent.parent
    script = julien_dir / "speed_plots.py"
    if not script.exists():
        print("Underlying plotting script not found:", script)
        return 2
    cmd = [sys.executable, str(script), *sys.argv[1:]]
    proc = subprocess.run(cmd, cwd=str(julien_dir), check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
