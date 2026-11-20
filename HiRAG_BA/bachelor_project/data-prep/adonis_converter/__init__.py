"""ADONIS JSONL-to-text converter for HiRAG (parent-only hierarchy mode).

This package converts the ADONIS 40% JSONL sample (models, hierarchy,
documents) into deterministic, LLM-friendly textual records for HiRAG's
entity and triplet extraction. See ``README.md`` for usage.
"""

from __future__ import annotations

from .config import ConverterConfig
from .converter import run_conversion
from .validator import ConversionError

__version__ = "1.0.0"

__all__ = ["ConverterConfig", "ConversionError", "run_conversion", "__version__"]
