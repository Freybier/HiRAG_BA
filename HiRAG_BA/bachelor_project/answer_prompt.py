#!/usr/bin/env python3
"""Shared answer-layout prompt for all three frontend RAG modes.

``ANSWER_LAYOUT_PROMPT`` is the single source of truth for the answer
structure (the 5-section Markdown layout + language rule). It is used by:

- ``streamlit_app._SYSTEM_PROMPT`` (naive_rag / hierarchical answer
  generation), where it is embedded verbatim between the role line and the
  JSON-specific citation rules, and
- ``HiRAG/hi_Search_openai._openai_request``, which appends it to the
  system prompt of every ``answer_generation`` stage call, so HiRAG's
  answers follow the same layout.

The sentence about ``entries with a 'node' key`` only matches the
frontend's JSON search-result context; it is kept verbatim so the layout
block is byte-identical across modes (HiRAG's graph context simply never
contains that shape).
"""

#: Layout instructions for the final answer of every RAG mode.
ANSWER_LAYOUT_PROMPT: str = """RESPONSE FORMAT:
- Use Markdown formatting throughout: **bold** for section headers, bullet lists for enumerations.
- Structure the answer with these sections where data is available:
  1. Überblick / Overview
  2. Prozessschritte / Process Steps
  3. Rollen & Verantwortlichkeiten / Roles & Responsibilities
  4. Risiken & Kontrollen / Risks & Controls
  5. Dokumente & Anhänge / Documents & Attachments
- When document text is present (entries with a 'node' key containing 'text'), use it to build detailed answers.
- Subprocess relationships are documented only in the parent process but apply from both perspectives: “A contains a subprocess B” also means “B is a subprocess of A.”
- Respond in the same language the user used (German or English)."""
