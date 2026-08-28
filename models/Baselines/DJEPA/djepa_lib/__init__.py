"""Vendored D-JEPA model code.

See ``models/Baselines/DJEPA/README.md`` for provenance, license, and what was
changed relative to upstream (only import paths — no logic was modified).
"""

from .djepa import DJEPA, djepa_base, djepa_huge, djepa_large

__all__ = ["DJEPA", "djepa_base", "djepa_large", "djepa_huge"]
