#!/usr/bin/env python3
"""Query-only HiRAG adapter for the Streamlit frontend.

Lets the Streamlit app (running in ``src/.venv``, Python 3.13) query existing
HiRAG indexes in-process. This module is intentionally query-only:

- It never calls ``HiRAG.insert()``. Index building stays in ``hirag_venv``
  via ``hi_insert_run.py`` (the insert-time clustering deps ``umap``,
  ``scikit-learn`` and ``graspologic`` are not installable in ``src/.venv``;
  the vendored hirag package imports them lazily, so a mistaken insert attempt
  fails loudly instead of silently corrupting anything).
- It reuses the gateway wiring (Azure OpenAI chat/embedding functions, LLM
  cache, stage-model routing) from ``hi_Search_openai.py`` via its
  ``build_hirag()`` factory, so model configuration stays centralized in
  ``bachelor_project/config.yaml``.

Usage from Streamlit::

    from hirag_query import list_available_indexes, query_hirag

    indexes = list_available_indexes()
    answer = query_hirag("Was ist ...?", index=indexes[0], mode="hi")
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_BACHELOR_DIR: Path = Path(__file__).resolve().parent
_HIRAG_DIR: Path = _BACHELOR_DIR / "HiRAG"

#: Root directory that holds the HiRAG index working dirs created by
#: ``hi_insert_run.py`` (one subdirectory per indexed model combination).
INDEX_ROOT: Path = _HIRAG_DIR / "hirag_index_samples"

# Make the vendored hirag package and the hi_Search_openai wiring importable.
for _path in (str(_HIRAG_DIR), str(_BACHELOR_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import hi_Search_openai as _wiring  # noqa: E402  (loads .env-dev on import)
from hirag import HiRAG, QueryParam  # noqa: E402
from source_id_mapping import load_source_id_mapping  # noqa: E402

#: One HiRAG instance per index directory (loading graph + vector stores from
#: disk takes a moment, so instances are reused across questions).
_HIRAG_CACHE: dict[str, HiRAG] = {}

#: Per-index source-resolution maps, see :func:`_load_source_maps`.
_SOURCE_MAPS_CACHE: dict[str, tuple[dict, dict, dict]] = {}


def list_available_indexes() -> list[str]:
    """List the available HiRAG index directories.

    A directory counts as an index when it contains HiRAG's entity vector
    store file (``vdb_entities.json``), which is written on the first
    successful insert.

    Returns:
        Sorted list of index directory names under
        ``HiRAG/hirag_index_samples/`` (empty when none exist).
    """
    if not INDEX_ROOT.is_dir():
        return []
    return sorted(
        entry.name
        for entry in INDEX_ROOT.iterdir()
        if entry.is_dir() and (entry / "vdb_entities.json").is_file()
    )


def _resolve_working_dir(index: str) -> Path:
    """Resolve an index name or path to an absolute working directory.

    Args:
        index: Either a plain directory name inside ``INDEX_ROOT`` or an
            (absolute or cwd-relative) path to a HiRAG working dir.

    Returns:
        The resolved absolute path.

    Raises:
        FileNotFoundError: If the resolved directory does not contain a
            ``vdb_entities.json`` (i.e. it is not an inserted index).
    """
    candidate = Path(index)
    if not candidate.is_absolute():
        in_root = INDEX_ROOT / index
        candidate = in_root if in_root.is_dir() else candidate.resolve()
    if not (candidate / "vdb_entities.json").is_file():
        raise FileNotFoundError(
            f"'{candidate}' is not a HiRAG index (vdb_entities.json missing). "
            f"Build an index first with hi_insert_run.py in hirag_venv."
        )
    return candidate


def _get_hirag(working_dir: Path) -> HiRAG:
    """Return the cached HiRAG instance for an index, building it if needed.

    Args:
        working_dir: Resolved absolute index directory.

    Returns:
        The ``HiRAG`` instance bound to that index.
    """
    key = str(working_dir)
    if key not in _HIRAG_CACHE:
        logger.info(f"Loading HiRAG index from {working_dir}")
        _HIRAG_CACHE[key] = _wiring.build_hirag(working_dir=str(working_dir))
    return _HIRAG_CACHE[key]


def _display_name(content: str) -> str:
    """Extract a readable record name from a stored full-document text.

    The textualized records start with a header line like
    ``PROCESS_NAME: <name>`` or ``DOCUMENT_TITLE: <title>``; the generic
    ``PREFIX: value`` prefix is stripped when present.

    Args:
        content: The full-document content from ``kv_store_full_docs.json``.

    Returns:
        The cleaned first line.
    """
    first_line = content.split("\n", 1)[0].strip()
    prefix, sep, value = first_line.partition(": ")
    if sep and prefix.replace("_", "").isalpha() and prefix.isupper():
        return value.strip() or first_line
    return first_line


def _load_source_maps(working_dir: Path) -> tuple[dict, dict, dict]:
    """Load (and cache) the source-resolution maps for an index directory.

    Args:
        working_dir: Resolved absolute HiRAG index directory.

    Returns:
        Tuple of ``chunk_to_doc`` (chunk id -> full doc id, from
        ``kv_store_text_chunks.json``), ``doc_to_sources`` (full doc id ->
        original ADONIS source IDs, from ``source_id_mapping.json``) and
        ``doc_names`` (full doc id -> readable record name, from
        ``kv_store_full_docs.json``). Missing files yield empty dicts.
    """
    key = str(working_dir)
    if key not in _SOURCE_MAPS_CACHE:
        chunks_file = working_dir / "kv_store_text_chunks.json"
        docs_file = working_dir / "kv_store_full_docs.json"
        chunk_to_doc = (
            {
                chunk_id: chunk.get("full_doc_id")
                for chunk_id, chunk in json.loads(chunks_file.read_text()).items()
            }
            if chunks_file.is_file()
            else {}
        )
        doc_to_sources = load_source_id_mapping(working_dir)
        doc_names = (
            {
                doc_id: _display_name(doc.get("content", ""))
                for doc_id, doc in json.loads(docs_file.read_text()).items()
            }
            if docs_file.is_file()
            else {}
        )
        _SOURCE_MAPS_CACHE[key] = (chunk_to_doc, doc_to_sources, doc_names)
    return _SOURCE_MAPS_CACHE[key]


def _sources_from_capture(capture: dict[str, Any], working_dir: Path) -> list[dict]:
    """Resolve the source documents a HiRAG query used, in context order.

    Walks the ``QueryParam.capture`` sections: retrieved source chunks carry a
    direct ``full_doc_id``/``chunk_id``, generated sections (entity/relation
    summaries, community reports) carry ``source_chunk_ids`` provenance. Doc
    IDs are mapped back to the original ADONIS source IDs via the index's
    ``source_id_mapping.json``.

    Args:
        capture: The filled capture dict after a query.
        working_dir: Resolved absolute HiRAG index directory.

    Returns:
        List of source entries with ``name``, ``doc_id``, ``source_ids`` and
        ``origin`` (``retrieved`` for direct chunks, ``provenance`` for docs
        that only contributed through generated summaries).
    """
    chunk_to_doc, doc_to_sources, doc_names = _load_source_maps(working_dir)
    ordered_doc_ids: list[str] = []
    retrieved_doc_ids: set[str] = set()
    provenance_doc_ids: set[str] = set()

    for section in capture.get("context_sections") or []:
        for item in section.get("items") or []:
            doc_id = item.get("full_doc_id")
            if doc_id is None and item.get("chunk_id"):
                doc_id = chunk_to_doc.get(item["chunk_id"])
            if doc_id:
                if doc_id not in ordered_doc_ids:
                    ordered_doc_ids.append(doc_id)
                retrieved_doc_ids.add(doc_id)
            for prov_chunk_id in item.get("source_chunk_ids") or []:
                prov_doc_id = chunk_to_doc.get(prov_chunk_id)
                if prov_doc_id:
                    if prov_doc_id not in ordered_doc_ids:
                        ordered_doc_ids.append(prov_doc_id)
                    provenance_doc_ids.add(prov_doc_id)

    # A doc that was directly retrieved stays "retrieved" even when it also
    # appears in the provenance closure of generated items.
    provenance_doc_ids -= retrieved_doc_ids

    return [
        {
            "name": doc_names.get(doc_id, ""),
            "doc_id": doc_id,
            "source_ids": list(doc_to_sources.get(doc_id, [])),
            "origin": "provenance" if doc_id in provenance_doc_ids else "retrieved",
        }
        for doc_id in ordered_doc_ids
    ]


def query_hirag(question: str, index: str, mode: str = "hi") -> dict[str, Any]:
    """Run a HiRAG query against an existing index.

    Args:
        question: The user's natural-language question.
        index: Index directory name under ``HiRAG/hirag_index_samples/``
            (see :func:`list_available_indexes`) or a path to a working dir.
        mode: HiRAG query mode (``hi``, ``hi_bridge``, ``hi_local``,
            ``hi_global``, ``hi_nobridge``, ``naive``).

    Returns:
        Dict with ``answer`` (HiRAG's synthesized answer string),
        ``working_dir`` and ``sources`` (the source documents the answer was
        built from, resolved via ``source_id_mapping.json``), or
        ``{"error": ...}`` when the query fails.
    """
    try:
        working_dir = _resolve_working_dir(index)
    except FileNotFoundError as exc:
        return {"error": str(exc)}
    capture: dict[str, Any] = {}
    try:
        answer = _get_hirag(working_dir).query(
            question, param=QueryParam(mode=mode, capture=capture)
        )
    except Exception as exc:
        logger.error(f"HiRAG query failed: {exc}")
        return {"error": str(exc)}
    return {
        "answer": answer,
        "working_dir": str(working_dir),
        "sources": _sources_from_capture(capture, working_dir),
    }
