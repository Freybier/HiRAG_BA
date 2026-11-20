"""Pytest configuration for the adonis_converter test suite."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the adonis_converter package importable when pytest collects these
# tests from an arbitrary working directory.
PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))
