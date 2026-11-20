"""Saved-run format shared by the collect and score workflows.

A run directory contains:

``run_metadata.json``
    How the answers were produced (dataset hash, system, models, retrieval
    settings, code versions, timestamps, counts). Required keys for scoring:
    ``schema_version``, ``system``, ``dataset``, ``id_granularity``.
``records.jsonl``
    One JSON object per question attempt series (see :data:`RECORD_FIELDS`).
    If a question was collected more than once (``--resume`` after errors),
    the LAST line for a ``question_id`` wins.

The format is system-agnostic: another RAG system can be scored by writing
the same two files. Each entry of ``retrieved_contexts`` must declare a
``kind``; use ``retrieved_source_chunk`` for original corpus text,
``injected_hierarchy_chunk`` for corpus text added by hierarchy enrichment
and a ``generated_*`` kind for anything an LLM wrote during indexing
(summaries, entity or relation descriptions, community reports).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "rag-eval-run/1"
RECORDS_FILE = "records.jsonl"
METADATA_FILE = "run_metadata.json"

SYSTEMS = ("hirag", "naive_rag", "test_data_rag")

# ``retrieved_context_ids`` hold original ADONIS source-record IDs (process
# model_id / document document_object_id). Runs with ``id_granularity: null``
# carry no comparable IDs and the ID-based metrics are skipped for them.
ADONIS_ID_GRANULARITY = "adonis_source_record"

SOURCE_CHUNK_KIND = "retrieved_source_chunk"
INJECTED_HIERARCHY_KIND = "injected_hierarchy_chunk"
GENERATED_KIND_PREFIX = "generated_"

RECORD_STATUSES = ("answered", "no_context", "context_only", "error")

RECORD_FIELDS = (
    "schema_version",
    "question_id",
    "user_input",
    "status",
    "response",
    "error",
    "attempts",
    "retrieved_contexts",
    "context_text",
    "context_builder",
    "system_prompt",
    "retrieved_context_ids",
    "provenance_context_ids",
    "unmapped_chunk_ids",
    "duration_s",
    "query_config",
    "usage",
)

CONTEXT_GRANULARITIES = ("items", "source_chunks_only", "full")


def _dedupe(values: Iterable[str]) -> list[str]:
    """Deduplicate while keeping first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def build_contexts_from_capture(
    capture: dict[str, Any],
    chunk_to_doc: dict[str, str],
    doc_to_sources: dict[str, list[str]],
) -> dict[str, Any]:
    """Turn a HiRAG ``QueryParam.capture`` dict into run-format fields.

    Order and boundaries are taken verbatim from the capture (the order in
    which sections and rows appear in the context string the LLM received).

    Args:
        capture: Filled capture dict from ``hirag._op``.
        chunk_to_doc: ``chunk_id -> full_doc_id`` from ``kv_store_text_chunks``.
        doc_to_sources: ``full_doc_id -> [original source IDs]`` from
            ``source_id_mapping.json``.

    Returns:
        Dict with ``retrieved_contexts``, ``retrieved_context_ids``,
        ``provenance_context_ids`` and ``unmapped_chunk_ids``.
    """
    contexts: list[dict[str, Any]] = []
    retrieved_ids: list[str] = []
    provenance_ids: list[str] = []
    unmapped: list[str] = []

    def resolve_chunk(
        chunk_id: str | None, full_doc_id: str | None
    ) -> tuple[str | None, list[str]]:
        doc_id = full_doc_id or (chunk_to_doc.get(chunk_id) if chunk_id else None)
        sources = list(doc_to_sources.get(doc_id, [])) if doc_id else []
        if chunk_id and not sources:
            unmapped.append(chunk_id)
        return doc_id, sources

    for section in capture.get("context_sections") or []:
        section_name = section["section"]
        kind = section["kind"]
        for item in section["items"]:
            attributes = {
                key: value
                for key, value in item.items()
                if key
                not in {
                    "row_id",
                    "content",
                    "chunk_id",
                    "full_doc_id",
                    "source_chunk_ids",
                }
            }
            entry: dict[str, Any] = {
                "context_id": f"{section_name}#{item['row_id']}",
                "kind": kind,
                "section": section_name,
                "content": item["content"],
                "chunk_id": None,
                "full_doc_id": None,
                "source_ids": [],
                "provenance": None,
                "attributes": attributes,
            }
            if kind == SOURCE_CHUNK_KIND:
                chunk_id = item.get("chunk_id")
                doc_id, sources = resolve_chunk(chunk_id, item.get("full_doc_id"))
                entry.update(chunk_id=chunk_id, full_doc_id=doc_id, source_ids=sources)
                retrieved_ids.extend(sources)
            else:
                chunk_ids = list(item.get("source_chunk_ids") or [])
                doc_ids: list[str] = []
                sources_all: list[str] = []
                for chunk_id in chunk_ids:
                    doc_id, sources = resolve_chunk(chunk_id, None)
                    if doc_id:
                        doc_ids.append(doc_id)
                    sources_all.extend(sources)
                entry["provenance"] = {
                    "source_chunk_ids": chunk_ids,
                    "full_doc_ids": _dedupe(doc_ids),
                    "source_ids": _dedupe(sources_all),
                }
                provenance_ids.extend(sources_all)
            contexts.append(entry)

    return {
        "retrieved_contexts": contexts,
        "retrieved_context_ids": _dedupe(retrieved_ids),
        "provenance_context_ids": _dedupe(provenance_ids),
        "unmapped_chunk_ids": _dedupe(unmapped),
    }


_INJECTED_PREFIXES = ("[Elternprozess-Kontext", "[Kindprozesse")


def build_contexts_from_search_response(response: dict[str, Any]) -> dict[str, Any]:
    """Turn an MCP-shaped ``search_adonis`` response into run-format fields.

    Used by the chunk-based systems (naive RAG and the parent-child test-data
    RAG). Each ``results`` item becomes one context entry, in response order.

    TODO: the chunk corpus has no IDs comparable to HiRAG's ADONIS source IDs
    (``model_{uuid}`` vs ``{uuid}``, no documents), so ``retrieved_context_ids``
    stays empty until all RAG systems share a common input-data structure.
    The raw chunk metadata IDs are kept under ``attributes``.

    Args:
        response: Dict with a ``results`` list of ``{score, node: {text,
            metadata}}`` items.

    Returns:
        Dict with ``retrieved_contexts``, ``retrieved_context_ids``,
        ``provenance_context_ids`` and ``unmapped_chunk_ids``.
    """
    contexts: list[dict[str, Any]] = []
    for index, item in enumerate(response.get("results") or []):
        node = item.get("node") or {}
        text = str(node.get("text") or "")
        metadata = node.get("metadata") or {}
        kind = (
            INJECTED_HIERARCHY_KIND
            if text.startswith(_INJECTED_PREFIXES)
            else SOURCE_CHUNK_KIND
        )
        contexts.append(
            {
                "context_id": f"results#{index}",
                "kind": kind,
                "section": "results",
                "content": text,
                "chunk_id": metadata.get("chunk_id") or metadata.get("node_id"),
                "full_doc_id": None,
                "source_ids": [],
                "provenance": None,
                "attributes": {
                    "score": item.get("score"),
                    "model_id": metadata.get("model_id"),
                    "model_name": metadata.get("model_name"),
                    "object_type": metadata.get("object_type"),
                    "source_url": metadata.get("source_url"),
                },
            }
        )
    return {
        "retrieved_contexts": contexts,
        "retrieved_context_ids": [],
        "provenance_context_ids": [],
        "unmapped_chunk_ids": [],
    }


def load_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load a saved run.

    Args:
        run_dir: Directory containing ``run_metadata.json`` and
            ``records.jsonl``.

    Returns:
        The metadata and ``question_id -> record`` (last line per ID wins).

    Raises:
        FileNotFoundError: If a required file is missing.
        ValueError: If the files are malformed.
    """
    metadata_path = run_dir / METADATA_FILE
    records_path = run_dir / RECORDS_FILE
    for path in (metadata_path, records_path):
        if not path.is_file():
            raise FileNotFoundError(f"Run file not found: {path}")
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_path}: expected a JSON object")
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{metadata_path}: unsupported schema_version "
            f"{metadata.get('schema_version')!r} (expected {SCHEMA_VERSION!r})"
        )
    if metadata.get("system") not in SYSTEMS:
        raise ValueError(f"{metadata_path}: 'system' must be one of {SYSTEMS}")

    records: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    with records_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON ({exc.msg})")
                continue
            problem = validate_record(record)
            if problem:
                errors.append(f"line {line_number}: {problem}")
                continue
            records[record["question_id"]] = record
    if errors:
        raise ValueError(
            f"{records_path} is malformed:\n  - " + "\n  - ".join(errors[:50])
        )
    return metadata, records


def validate_record(record: Any) -> str | None:
    """Return a problem description for a malformed record, else ``None``."""
    if not isinstance(record, dict):
        return "expected a JSON object"
    if record.get("schema_version") != SCHEMA_VERSION:
        return f"unsupported schema_version {record.get('schema_version')!r}"
    if not isinstance(record.get("question_id"), str):
        return "'question_id' must be a string"
    if record.get("status") not in RECORD_STATUSES:
        return f"'status' must be one of {RECORD_STATUSES}"
    contexts = record.get("retrieved_contexts")
    if not isinstance(contexts, list):
        return "'retrieved_contexts' must be a list"
    for index, context in enumerate(contexts):
        if not isinstance(context, dict) or not isinstance(context.get("content"), str):
            return f"retrieved_contexts[{index}] needs a string 'content'"
        if not isinstance(context.get("kind"), str):
            return f"retrieved_contexts[{index}] needs a 'kind'"
    return None


def select_contexts(record: dict[str, Any], granularity: str) -> list[str]:
    """Return the context strings handed to RAGAS for one record.

    Args:
        record: Run record.
        granularity: ``items`` (every rendered row, in order),
            ``source_chunks_only`` (only original corpus chunks) or ``full``
            (the exact context string as one element).

    Returns:
        Ordered list of context strings (may be empty).
    """
    if granularity == "full":
        text = record.get("context_text")
        return [text] if isinstance(text, str) and text.strip() else []
    contexts = record.get("retrieved_contexts") or []
    if granularity == "source_chunks_only":
        contexts = [c for c in contexts if c.get("kind") == SOURCE_CHUNK_KIND]
    elif granularity != "items":
        raise ValueError(f"Unknown context granularity {granularity!r}")
    return [c["content"] for c in contexts if c.get("content", "").strip()]
