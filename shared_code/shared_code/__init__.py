"""Lightweight package init for shared_code.

Avoid importing heavy submodules at import time to keep CLI imports fast and
robust in minimal environments. Import submodules directly, e.g.::

    from shared_code.fun_paths import get_paths
    from shared_code.fun_plot import compute_pvalue

This file intentionally does not wildcard-import submodules.
"""

__all__ = []
