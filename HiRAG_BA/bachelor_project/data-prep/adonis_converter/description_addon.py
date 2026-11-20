"""Ingestion of the ``data_vector_store2`` chunk export (description addon).

The addon file (``description_hierarchy-addon.json``) is a filtered export
of the RAG chunk table. It contains one ``overview`` chunk per process
model whose ``text`` carries the real ``Beschreibung / Zweck`` description
that ``models_40pct.jsonl`` lacks (its ``description`` column only holds
the constant ADONIS metaName).

Join rule (confirmed by the data provider on 2026-10-12): the models
export stores braced UUIDs (``{uuid}``) while the addon stores bare
lowercase UUIDs. :func:`normalize_model_key` normalizes both sides. The
surrogate primary keys (``pkk`` in models, ``id`` in the chunk table) are
unrelated and must never be joined.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ConverterConfig
from .schema import SchemaError

#: Field label marking the description inside an overview chunk's text.
DESCRIPTION_LABEL: str = "Beschreibung / Zweck:"

#: Values that mean "no description" in the generated chunk text.
EMPTY_DESCRIPTION_SENTINELS: frozenset[str] = frozenset(
    {"Kein Eintrag", "NULL", "null", "None"}
)

#: First line of the *next* text field after the description. The chunk
#: text is generated from structured ADONIS attributes, so these labels
#: are fixed and deterministic.
_STOP_PATTERN: re.Pattern[str] = re.compile(
    r"(?m)^(?:Verantwortlich|Änderungskommentar|Anzahl Objekte im Modell|"
    r"Anzahl verknüpfte Dokumente|Quelle|Attribute):"
)


# Normalize an ADONIS model UUID for joining across exports.
def normalize_model_key(value: str | None) -> str:
    """Normalize an ADONIS model UUID for joining across exports.

    Args:
        value: Raw model ID from any export (braced or bare, any case).

    Returns:
        The bare lowercase UUID, or an empty string for missing values.
    """
    if not isinstance(value, str):
        return ""
    return value.strip().strip("{}").strip().lower()


# Extract the Beschreibung / Zweck text from an overview chunk.
def extract_overview_description(text: Any) -> str | None:
    """Extract the ``Beschreibung / Zweck`` text from an overview chunk.

    The description runs from its label to the first line of the next
    known field (e.g. ``Verantwortlich:``). Multi-paragraph descriptions
    are preserved; blank and non-breaking-space-only lines are dropped.

    Args:
        text: The chunk's ``text`` field.

    Returns:
        The cleaned description, or ``None`` when the chunk carries no
        usable description (missing label or an empty/sentinel value).
    """
    if not isinstance(text, str):
        return None
    start = text.find(DESCRIPTION_LABEL)
    if start < 0:
        return None
    remainder = text[start + len(DESCRIPTION_LABEL) :]
    stop = _STOP_PATTERN.search(remainder)
    if stop:
        remainder = remainder[: stop.start()]
    lines: list[str] = []
    for line in remainder.splitlines():
        # NBSP-only lines are paragraph separators in the generated text.
        cleaned = line.replace("\u00a0", " ").strip()
        if cleaned:
            lines.append(cleaned)
    description = "\n".join(lines).strip()
    if not description or description in EMPTY_DESCRIPTION_SENTINELS:
        return None
    return description


# Read the chunk export, accepting a JSON array or JSONL.
def read_description_chunks(path: Path) -> list[dict[str, Any]]:
    """Read the chunk export, accepting a JSON array or JSONL.

    Args:
        path: File to read (UTF-8). The real export is a single JSON
            array; newline-delimited JSON is accepted as well.

    Returns:
        The chunk rows as dictionaries, in file order.

    Raises:
        SchemaError: If the file is missing, malformed, or its rows are
            not JSON objects.
    """
    if not path.is_file():
        raise SchemaError(f"Descriptions file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    stripped = raw.lstrip()
    if not stripped:
        return []
    rows: list[Any]
    if stripped.startswith("["):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SchemaError(f"Malformed JSON array in {path.name}: {exc}") from exc
        if not isinstance(payload, list):
            raise SchemaError(f"{path.name} does not contain a JSON array.")
        rows = payload
    else:
        rows = []
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SchemaError(
                    f"Malformed JSON in {path.name} line {line_number}: {exc}"
                ) from exc
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SchemaError(f"Row {index + 1} in {path.name} is not a JSON object.")
    return rows


@dataclass(frozen=True)
class DescriptionIndex:
    """Extracted overview descriptions plus ingestion statistics.

    Attributes:
        descriptions: Normalized model key to description text.
        rows_read: Total chunk rows read from the addon file.
        overview_rows: Rows with the overview object type.
        empty_descriptions: Overview rows without a usable description.
        duplicate_overview_rows: Extra overview rows for a model that
            already had one (the first row wins).
    """

    descriptions: dict[str, str]
    rows_read: int
    overview_rows: int
    empty_descriptions: int
    duplicate_overview_rows: int


# Join overview descriptions to normalized model IDs and report unmatched or unusable chunks.
def build_description_index(
    rows: list[dict[str, Any]], config: ConverterConfig, warnings: list[str]
) -> DescriptionIndex:
    """Build the model-key-to-description index from chunk rows.

    Only ``overview`` rows are considered; lane/activity/compliance/other
    chunks are skipped. Rows with a missing model ID or an empty
    description are skipped and counted, never invented.

    Args:
        rows: Chunk rows from :func:`read_description_chunks`.
        config: Converter configuration with the field mapping.
        warnings: Mutable list collecting human-readable warnings.

    Returns:
        The description index with statistics.
    """
    fields = config.description_fields
    descriptions: dict[str, str] = {}
    overview_rows = 0
    empty_descriptions = 0
    duplicate_overview_rows = 0
    for row in rows:
        if row.get(fields.object_type) != fields.overview_type:
            continue
        overview_rows += 1
        key = normalize_model_key(row.get(fields.model_id))
        if not key:
            warnings.append(
                f"{config.descriptions_path}: skipped overview chunk with "
                f"missing {fields.model_id!r}."
            )
            continue
        description = extract_overview_description(row.get(fields.text))
        if description is None:
            empty_descriptions += 1
            continue
        if key in descriptions:
            duplicate_overview_rows += 1
            continue
        descriptions[key] = description
    return DescriptionIndex(
        descriptions=descriptions,
        rows_read=len(rows),
        overview_rows=overview_rows,
        empty_descriptions=empty_descriptions,
        duplicate_overview_rows=duplicate_overview_rows,
    )
