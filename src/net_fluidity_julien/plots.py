#!/usr/bin/env python3
"""
Package entry adapter for Julien dFC speed plotting.

Delegates to the legacy script (julien_data/speed_plots.py) to avoid code
duplication during migration. This preserves CLI behavior while allowing
`import net_fluidity_julien.plots` usage.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_legacy() -> object:
    here = Path(__file__).resolve()
    # Project root → julien_data/speed_plots.py
    julien_dir = here.parents[2] / "julien_data"
    target = julien_dir / "speed_plots.py"
    if not target.exists():
        raise FileNotFoundError(f"Legacy plot module not found: {target}")
    # Ensure julien_data is importable for its local imports
    if str(julien_dir) not in sys.path:
        sys.path.insert(0, str(julien_dir))
    spec = importlib.util.spec_from_file_location("julien_speed_plots", target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import spec for {target}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    mod = _load_legacy()
    return int(mod.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())

