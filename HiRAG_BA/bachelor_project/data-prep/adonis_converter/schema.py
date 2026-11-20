"""Input schema handling for the ADONIS JSONL exports.

The field defaults in :class:`~adonis_converter.config` reflect the actual
40% sample schemas discovered on 2026-10-08:

- ``models_40pct.jsonl``: ``model_id`` (braced UUID), ``name``,
  ``json_string`` (nested JSON carrying ``type``/``metaName``),
  ``description`` (constant metaName, no free text), ``purpose``,
  ``parent_model_id``/``hierarchy_path``/``hierarchy_depth`` (model-tree
  metadata, distinct from subprocess relations) and pipeline bookkeeping
  fields.
- ``hierarchy_40pct.jsonl``: ``parent_model_id`` (relation source),
  ``child_model_id`` (relation target), ``relation_type``
  (``C_SUB_PROCESS`` / ``C_CROSS_REFERENCE``), redundant name columns.
- ``documents_40pct.jsonl``: ``id`` (numeric database key), ``name``,
  ``content`` (extracted text), ``file_type``, ``document_object_id``.

The exports use the string ``"NULL"`` for missing values; ``clean``
normalizes those to ``None``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ConverterConfig

#: Sentinel strings the export uses for missing values.
NULL_SENTINELS: frozenset[str] = frozenset({"NULL", "null", "None", ""})


class SchemaError(Exception):
    """Raised when an input file cannot be parsed at all."""


# Normalize an export value to None when it represents "missing".
def clean(value: Any) -> str | None:
    """Normalize an export value to ``None`` when it represents "missing".

    Args:
        value: Raw value from a JSONL row.

    Returns:
        The stripped string, or ``None`` for NULL-like sentinels.
    """
    if not isinstance(value, str):
        return None if value is None else str(value)
    stripped = value.strip()
    return None if stripped in NULL_SENTINELS else stripped


@dataclass(frozen=True)
class RawRow:
    """A parsed JSONL row with provenance information."""

    data: dict[str, Any]
    line_number: int
    source_file: str


@dataclass(frozen=True)
class JsonlReadResult:
    """Result of reading one JSONL file."""

    rows: list[RawRow]
    blank_lines: int


# Read a JSONL file strictly, failing on malformed lines.
def read_jsonl(path: Path, limit: int | None = None) -> JsonlReadResult:
    """Read a JSONL file strictly, failing on malformed lines.

    Args:
        path: File to read (UTF-8).
        limit: Optional maximum number of data rows to read.

    Returns:
        Parsed rows with line numbers plus a blank-line count.

    Raises:
        SchemaError: If the file is missing or a line is not valid JSON.
    """
    if not path.is_file():
        raise SchemaError(f"Input file not found: {path}")
    rows: list[RawRow] = []
    blank_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit is not None and len(rows) >= limit:
                break
            stripped = line.strip()
            if not stripped:
                blank_lines += 1
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SchemaError(
                    f"Malformed JSON in {path.name} line {line_number}: {exc}"
                ) from exc
            if not isinstance(data, dict):
                raise SchemaError(
                    f"Line {line_number} in {path.name} is not a JSON object."
                )
            rows.append(
                RawRow(data=data, line_number=line_number, source_file=path.name)
            )
    return JsonlReadResult(rows=rows, blank_lines=blank_lines)


@dataclass(frozen=True)
class ProcessModel:
    """A normalized ADONIS process model record."""

    model_id: str
    name: str
    description: str | None
    type_name: str | None
    meta_name: str | None
    hierarchy_depth: str | None
    tree_parent_id: str | None
    raw: dict[str, Any]
    source_file: str
    line_number: int


@dataclass(frozen=True)
class DocumentRecord:
    """A normalized ADONIS document record."""

    doc_id: str
    title: str
    content: str
    file_type: str | None
    object_id: str | None
    raw: dict[str, Any]
    source_file: str
    line_number: int


@dataclass(frozen=True)
class RelationshipRow:
    """A normalized hierarchy row describing one relation occurrence."""

    source_id: str | None
    target_id: str | None
    relation_type: str | None
    source_name_exported: str | None
    target_name_exported: str | None
    raw: dict[str, Any]
    source_file: str
    line_number: int


# Clean exported model fields and retain their original rows and source locations in typed
# records.
def parse_models(
    rows: list[RawRow], config: ConverterConfig, warnings: list[str]
) -> tuple[list[ProcessModel], list[RawRow]]:
    """Parse raw rows into process models.

    Args:
        rows: Raw rows from the models file.
        config: Converter configuration with the field mapping.
        warnings: Mutable list collecting human-readable warnings.

    Returns:
        A tuple of valid process models and rejected raw rows. Rows with a
        missing/empty ID or name are rejected and reported, never invented.
    """
    fields = config.models_fields
    models: list[ProcessModel] = []
    rejected: list[RawRow] = []
    for row in rows:
        model_id = clean(row.data.get(fields.id))
        name = clean(row.data.get(fields.name))
        if not model_id or not name:
            warnings.append(
                f"{row.source_file}:{row.line_number}: rejected model row with "
                f"missing {fields.id!r} or {fields.name!r}."
            )
            rejected.append(row)
            continue
        type_name: str | None = None
        meta_name: str | None = None
        embedded = clean(row.data.get(fields.embedded_json))
        if embedded:
            try:
                payload = json.loads(embedded)
                if isinstance(payload, dict):
                    type_name = clean(payload.get("type"))
                    meta_name = clean(payload.get("metaName"))
            except json.JSONDecodeError:
                warnings.append(
                    f"{row.source_file}:{row.line_number}: could not parse "
                    f"{fields.embedded_json!r} for model {model_id}."
                )
        models.append(
            ProcessModel(
                model_id=model_id,
                name=name,
                description=clean(row.data.get(fields.description)),
                type_name=type_name,
                meta_name=meta_name,
                hierarchy_depth=clean(row.data.get(fields.hierarchy_depth)),
                tree_parent_id=clean(row.data.get(fields.tree_parent_id)),
                raw=row.data,
                source_file=row.source_file,
                line_number=row.line_number,
            )
        )
    return models, rejected


# Clean document fields and preserve source rows and locations in typed document records.
def parse_documents(
    rows: list[RawRow], config: ConverterConfig, warnings: list[str]
) -> tuple[list[DocumentRecord], list[RawRow]]:
    """Parse raw rows into document records.

    Args:
        rows: Raw rows from the documents file.
        config: Converter configuration with the field mapping.
        warnings: Mutable list collecting human-readable warnings.

    Returns:
        A tuple of valid document records and rejected raw rows. Rows with
        a missing/empty ID, title or content are rejected and reported.
    """
    fields = config.document_fields
    documents: list[DocumentRecord] = []
    rejected: list[RawRow] = []
    for row in rows:
        doc_id = clean(row.data.get(fields.id))
        title = clean(row.data.get(fields.name))
        content = clean(row.data.get(fields.content))
        if not doc_id or not title or not content:
            warnings.append(
                f"{row.source_file}:{row.line_number}: rejected document row "
                f"with missing {fields.id!r}, {fields.name!r} or "
                f"{fields.content!r}."
            )
            rejected.append(row)
            continue
        documents.append(
            DocumentRecord(
                doc_id=doc_id,
                title=title,
                content=content,
                file_type=clean(row.data.get(fields.file_type)),
                object_id=clean(row.data.get(fields.object_id)),
                raw=row.data,
                source_file=row.source_file,
                line_number=row.line_number,
            )
        )
    return documents, rejected


# Convert hierarchy rows into typed source-target relations while auditing rejected rows.
def parse_relationships(
    rows: list[RawRow], config: ConverterConfig, warnings: list[str]
) -> list[RelationshipRow]:
    """Parse raw rows into relationship rows.

    Rows with missing source/target IDs or a missing relation type are
    kept (with ``None`` values) so the relationship mapper can report them
    explicitly instead of silently dropping them.

    Args:
        rows: Raw rows from the hierarchy file.
        config: Converter configuration with the field mapping.
        warnings: Mutable list collecting human-readable warnings.

    Returns:
        Normalized relationship rows, one per input row.
    """
    fields = config.hierarchy_fields
    relationships: list[RelationshipRow] = []
    for row in rows:
        relationships.append(
            RelationshipRow(
                source_id=clean(row.data.get(fields.source_id)),
                target_id=clean(row.data.get(fields.target_id)),
                relation_type=clean(row.data.get(fields.relation_type)),
                source_name_exported=clean(row.data.get(fields.source_name)),
                target_name_exported=clean(row.data.get(fields.target_name)),
                raw=row.data,
                source_file=row.source_file,
                line_number=row.line_number,
            )
        )
    return relationships
