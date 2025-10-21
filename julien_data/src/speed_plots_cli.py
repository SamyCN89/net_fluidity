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

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    here = Path(__file__).resolve()
    root = here.parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    julien_dir = root / "julien_data"
    script = julien_dir / "speed_plots.py"
    if not script.exists():
        print("Underlying plotting script not found:", script)
        return 2
    cmd = [sys.executable, str(script), *sys.argv[1:]]
    proc = subprocess.run(cmd, cwd=str(julien_dir), check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
