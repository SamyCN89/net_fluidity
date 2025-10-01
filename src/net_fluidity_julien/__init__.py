"""Net Fluidity — Julien dataset helpers (experimental package).

Phase 3 migration: centralizing context and plotting utilities under src/.
Existing scripts in julien_data/ continue to work via fallback imports.
"""

from .context import DFCAnalysis

__all__ = ["DFCAnalysis"]

