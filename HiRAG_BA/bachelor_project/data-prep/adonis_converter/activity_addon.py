"""Ingestion of per-process activity steps from a ``chunks.jsonl`` corpus.

``chunks.jsonl`` (built by ``convert_40pct_to_chunks.py`` from the
``data_vector_store3`` export) carries one ``activity`` chunk per
Prozessschritt of a process model (4,382 chunks across 560 models in the
40% sample). Every activity chunk's text starts with
``Prozessschritt / Aktivität: <name>`` and its metadata carries the
numeric ``step_number``. With ``--include-activities`` these steps are
appended to the process record text after the relationship sections.

Optionally, additional chunk object types from the same corpus can be
appended after the ``ACTIVITIES`` section (``ConverterConfig
.extra_chunk_types``, the "everything" texturization mode):

- ``lane`` chunks (``Swimlane / Pool im Prozessmodell: ...``) supply the
  process's swimlanes/pools; the lane name comes from the ``lane_name``
  metadata (falling back to the ``Lane-Name:`` text line), the
  swimlane/pool distinction from ``object_meta_name`` (falling back to
  the ``Typ:`` text line) and the lane number from the ``_lane_<n>``
  suffix of the chunk's ``node_id``.
- ``compliance`` chunks (``Compliance-Informationen für: ...``) supply
  the model status and risk level from the ``status`` / ``risk_level``
  metadata (falling back to the ``Status:`` / ``Risikostufe:`` text
  lines); empty values and the ``Kein Eintrag`` placeholder are treated
  as missing, mirroring the description addon.
- ``other`` chunks (``Weitere Objekte in: ...``) supply the verbatim
  related-objects listing lines (`` - <name> [<TYPE>]``); the trailing
  relation tag is parsed into the sentence rendering.

Header lines and the trailing ``Quelle:`` source-URL line are never
textualized (pure semantics, no per-chunk URL noise).

Join rule: the models export stores braced UUIDs (``{uuid}``) while the
chunk metadata stores ``model_{uuid}`` (or the mangled ``model_{uuid``
artifact without the closing brace). :func:`normalize_chunk_model_key`
normalizes both sides to the bare lowercase UUID.

Both chunk file layouts are accepted: the production-shaped
``chunks.jsonl`` (fields nested under ``metadata_``) and the raw chunk
export (fields at the top level, as in ``description_hierarchy-addon.json``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import ConverterConfig
from .schema import SchemaError

#: Field label marking the activity name in the first line of an activity
#: chunk's text.
ACTIVITY_LABEL: str = "Prozessschritt / Aktivität:"

#: Field label marking the lane name inside a lane chunk's text.
LANE_NAME_LABEL: str = "Lane-Name:"

#: Field label marking the lane/pool type inside a lane chunk's text.
LANE_TYPE_LABEL: str = "Typ:"

#: Field label marking the status inside a compliance chunk's text.
COMPLIANCE_STATUS_LABEL: str = "Status:"

#: Field label marking the risk level inside a compliance chunk's text.
COMPLIANCE_RISK_LABEL: str = "Risikostufe:"

#: First-line label of the related-objects (``other``) chunks.
RELATED_OBJECTS_HEADER_LABEL: str = "Weitere Objekte in"

#: Field label marking the source-URL line, which is never textualized.
SOURCE_LABEL: str = "Quelle:"

#: Placeholder value in the export meaning "no entry"; treated as
#: missing, mirroring the description addon's sentinel handling.
NO_ENTRY_PLACEHOLDER: str = "kein eintrag"

#: ``object_meta_name`` value marking pool chunks (vs. swimlanes).
POOL_META_NAME: str = "C_POOL"

#: Lane number suffix in a lane chunk's ``node_id`` (``..._lane_<n>``).
_LANE_NUMBER_PATTERN = re.compile(r"_lane_(\d+)\s*$")

#: Trailing relation tag of a related-objects listing line
#: (`` - <name> [<TYPE>]``).
_RELATED_TAG_PATTERN = re.compile(r"^(?P<name>.*?)\s*\[(?P<tag>[A-Z_]+)\]\s*$")


# Normalize a chunk-row model reference for joining to models.
def normalize_chunk_model_key(value: Any) -> str:
    """Normalize a chunk-row model reference for joining to models.

    Handles every observed form: braced UUIDs (``{uuid}``, models export),
    ``model_{uuid}`` (production chunk metadata), the mangled
    ``model_{uuid`` artifact (missing closing brace) and bare UUIDs.

    Args:
        value: Raw model reference from a chunk row.

    Returns:
        The bare lowercase UUID, or an empty string for missing values.
    """
    if not isinstance(value, str):
        return ""
    return value.strip().removeprefix("model_").strip("{}").strip().lower()


# Extract the activity name from an activity chunk's text.
def extract_activity_name(text: Any) -> str | None:
    """Extract the activity name from an activity chunk's text.

    The name is the remainder of the first line after the
    ``Prozessschritt / Aktivität:`` label; the following lines
    (``Reihenfolge``, ``Typ``, ``Prozessmodell``, ``Modell-Status``,
    ``Quelle``) are chunk metadata and never part of the name.

    Args:
        text: The chunk's ``text`` field.

    Returns:
        The activity name, or ``None`` when the first line does not carry
        the label or the name is empty.
    """
    if not isinstance(text, str):
        return None
    first_line = text.split("\n", 1)[0].strip()
    if not first_line.startswith(ACTIVITY_LABEL):
        return None
    name = first_line[len(ACTIVITY_LABEL) :].strip()
    return name or None


# Parse the step_number chunk metadata value.
def parse_step_number(value: Any) -> int | None:
    """Parse the ``step_number`` chunk metadata value.

    Args:
        value: Raw metadata value (a decimal string in the real export).

    Returns:
        The step number as an integer, or ``None`` when the value is
        missing or not numeric.
    """
    if value is None:
        return None
    digits = str(value).strip()
    return int(digits) if digits.isdigit() else None


# Read a chunk file, streaming JSONL and accepting a JSON array.
def read_activity_chunks(path: Path) -> list[dict[str, Any]]:
    """Read a chunk file, streaming JSONL and accepting a JSON array.

    Args:
        path: File to read (UTF-8). ``chunks.jsonl`` is newline-delimited
            JSON; a single JSON array is accepted as well.

    Returns:
        The chunk rows as dictionaries, in file order.

    Raises:
        SchemaError: If the file is missing, malformed, or its rows are
            not JSON objects.
    """
    if not path.is_file():
        raise SchemaError(f"Activities file not found: {path}")
    rows: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        # Peek at the first non-whitespace character to detect the layout.
        prefix = ""
        while char := handle.read(1):
            if not char.isspace():
                prefix = char
                break
        if not prefix:
            return []
        if prefix == "[":
            handle.seek(0)
            try:
                payload = json.load(handle)
            except json.JSONDecodeError as exc:
                raise SchemaError(
                    f"Malformed JSON array in {path.name}: {exc}"
                ) from exc
            if not isinstance(payload, list):
                raise SchemaError(f"{path.name} does not contain a JSON array.")
            rows = payload
        else:
            handle.seek(0)
            for line_number, line in enumerate(handle, start=1):
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
class Activity:
    """One process step extracted from an activity chunk.

    Attributes:
        name: Activity name from the chunk text.
        step_number: Position of the step inside the process, or ``None``
            when the chunk metadata carries no numeric step.
    """

    name: str
    step_number: int | None


@dataclass(frozen=True)
class ActivityIndex:
    """Extracted per-model activities plus ingestion statistics.

    Attributes:
        activities: Normalized model key to the model's activities,
            sorted by step number (steps without a number sort last,
            tie-broken by name).
        rows_read: Total chunk rows read from the file.
        activity_rows: Rows with the activity object type.
        missing_model_keys: Activity rows without a usable model ID.
        unnamed_activities: Activity rows whose text carried no name.
    """

    activities: dict[str, list[Activity]]
    rows_read: int
    activity_rows: int
    missing_model_keys: int
    unnamed_activities: int


# Read a chunk field from the top level or the nested metadata.
def _row_field(row: dict[str, Any], name: str, metadata_key: str) -> Any:
    """Read a chunk field from the top level or the nested metadata.

    Args:
        row: The chunk row.
        name: Field name to look up.
        metadata_key: Key of the nested metadata object (``metadata_`` in
            the production-shaped ``chunks.jsonl``).

    Returns:
        The field value, or ``None`` when absent in both places.
    """
    value = row.get(name)
    if value is None:
        metadata = row.get(metadata_key)
        if isinstance(metadata, dict):
            value = metadata.get(name)
    return value


# Match activity chunks to normalized model IDs, order their steps, and report unmatched or
# empty rows.
def build_activity_index(
    rows: list[dict[str, Any]], config: ConverterConfig, warnings: list[str]
) -> ActivityIndex:
    """Build the model-key-to-activities index from chunk rows.

    Only ``activity`` rows are considered; overview/lane/compliance/other
    and document chunks are skipped. Rows with a missing model ID or an
    unparseable activity name are skipped and counted, never invented.

    Args:
        rows: Chunk rows from :func:`read_activity_chunks`.
        config: Converter configuration with the field mapping.
        warnings: Mutable list collecting human-readable warnings.

    Returns:
        The activity index with statistics.
    """
    fields = config.activity_fields
    activities: dict[str, list[Activity]] = {}
    activity_rows = 0
    missing_model_keys = 0
    unnamed_activities = 0
    for row in rows:
        if _row_field(row, fields.object_type, fields.metadata_key) != (
            fields.activity_type
        ):
            continue
        activity_rows += 1
        key = normalize_chunk_model_key(
            _row_field(row, fields.model_id, fields.metadata_key)
        )
        if not key:
            missing_model_keys += 1
            continue
        name = extract_activity_name(_row_field(row, fields.text, fields.metadata_key))
        if name is None:
            unnamed_activities += 1
            continue
        step_number = parse_step_number(
            _row_field(row, fields.step_number, fields.metadata_key)
        )
        activities.setdefault(key, []).append(
            Activity(name=name, step_number=step_number)
        )
    for model_activities in activities.values():
        model_activities.sort(
            key=lambda activity: (
                activity.step_number is None,
                activity.step_number or 0,
                activity.name,
            )
        )
    if missing_model_keys:
        warnings.append(
            f"{config.activities_path}: skipped {missing_model_keys} activity "
            f"chunk(s) with missing {fields.model_id!r}."
        )
    if unnamed_activities:
        warnings.append(
            f"{config.activities_path}: skipped {unnamed_activities} activity "
            f"chunk(s) whose text has no {ACTIVITY_LABEL!r} first line."
        )
    return ActivityIndex(
        activities=activities,
        rows_read=len(rows),
        activity_rows=activity_rows,
        missing_model_keys=missing_model_keys,
        unnamed_activities=unnamed_activities,
    )


# ---------------------------------------------------------------------------
# Extra chunk types (lane / compliance / other) for the "everything" mode
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Lane:
    """One swimlane or pool extracted from a lane chunk.

    Attributes:
        name: Lane name from the ``lane_name`` metadata or the
            ``Lane-Name:`` text line.
        number: Lane number from the ``_lane_<n>`` suffix of the chunk's
            ``node_id``, or ``None`` when the suffix is missing.
        kind: ``"swimlane"`` or ``"pool"`` (from ``object_meta_name`` /
            the ``Typ:`` text line).
    """

    name: str
    number: int | None
    kind: str


@dataclass(frozen=True)
class RelatedObject:
    """One entry of a related-objects (``other``) chunk listing.

    Attributes:
        name: Object name (the listing line without bullet and tag).
        tag: The trailing relation tag (e.g. ``C_START_EVENT``), or
            ``None`` when the listing line carries no tag.
    """

    name: str
    tag: str | None


@dataclass(frozen=True)
class ExtraChunks:
    """Extra per-model chunk content for the "everything" mode.

    Attributes:
        lanes: The model's swimlanes/pools in chunk (file) order.
        compliance_status: Model status from the compliance chunk, or
            ``None`` when missing/``Kein Eintrag``.
        compliance_risk_level: Risk level from the compliance chunk, or
            ``None`` when missing/``Kein Eintrag``.
        related_objects: The related-objects listing in chunk order.
    """

    lanes: list[Lane] = field(default_factory=list)
    compliance_status: str | None = None
    compliance_risk_level: str | None = None
    related_objects: list[RelatedObject] = field(default_factory=list)


@dataclass(frozen=True)
class ExtraChunkIndex:
    """Extracted per-model extra chunks plus ingestion statistics.

    Attributes:
        chunks: Normalized model key to the model's extra chunk content.
        rows_by_type: Chunk rows read per requested object type.
        missing_model_keys: Extra rows without a usable model ID.
        empty_rows: Extra rows that contributed no content (no lane name,
            no compliance values, or an empty related-objects listing).
    """

    chunks: dict[str, ExtraChunks]
    rows_by_type: dict[str, int]
    missing_model_keys: int
    empty_rows: int


# Read a Label: value line from a chunk's text.
def _text_field_value(text: Any, label: str) -> str | None:
    """Read a ``Label: value`` line from a chunk's text.

    Args:
        text: The chunk's ``text`` field.
        label: The line label to look for (including the colon).

    Returns:
        The stripped value after the label, or ``None`` when no line
        carries the label or the value is empty.
    """
    if not isinstance(text, str):
        return None
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith(label):
            value = stripped[len(label) :].strip()
            return value or None
    return None


# Normalize a chunk metadata or text-field value.
def _clean_chunk_value(value: Any) -> str | None:
    """Normalize a chunk metadata or text-field value.

    Args:
        value: Raw value (e.g. ``status`` / ``risk_level`` metadata or a
            ``lane_name``).

    Returns:
        The stripped value, or ``None`` when it is empty or the
        ``Kein Eintrag`` placeholder.
    """
    cleaned = str(value or "").strip()
    if not cleaned or cleaned.lower() == NO_ENTRY_PLACEHOLDER:
        return None
    return cleaned


# Extract the lane/pool from one lane chunk row.
def _parse_lane(row: dict[str, Any], fields: Any) -> Lane | None:
    """Extract the lane/pool from one lane chunk row.

    Args:
        row: The chunk row.
        fields: The chunk field mapping (``ConverterConfig.activity_fields``).

    Returns:
        The lane, or ``None`` when the lane name cannot be determined.
    """
    name = _clean_chunk_value(_row_field(row, fields.lane_name, fields.metadata_key))
    text = _row_field(row, fields.text, fields.metadata_key)
    if name is None:
        name = _text_field_value(text, LANE_NAME_LABEL)
    if name is None:
        return None
    meta_name = (
        str(_row_field(row, fields.object_meta_name, fields.metadata_key) or "")
        .strip()
        .upper()
    )
    if not meta_name:
        meta_name = (_text_field_value(text, LANE_TYPE_LABEL) or "").upper()
    kind = "pool" if meta_name == POOL_META_NAME else "swimlane"
    node_id = str(_row_field(row, fields.node_id, fields.metadata_key) or "")
    number_match = _LANE_NUMBER_PATTERN.search(node_id)
    return Lane(
        name=name,
        number=int(number_match.group(1)) if number_match else None,
        kind=kind,
    )


# Parse the listing lines of a related-objects (other) chunk.
def _parse_related_objects(text: Any) -> list[RelatedObject]:
    """Parse the listing lines of a related-objects (``other``) chunk.

    The header line (``Weitere Objekte in: ...``) and the trailing
    ``Quelle:`` source-URL line are dropped; every other non-empty line
    becomes one entry, with the trailing ``[<TYPE>]`` tag parsed out.

    Args:
        text: The chunk's ``text`` field.

    Returns:
        The related objects in file order (possibly empty).
    """
    if not isinstance(text, str):
        return []
    objects: list[RelatedObject] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith(RELATED_OBJECTS_HEADER_LABEL)
            or stripped.startswith(SOURCE_LABEL)
        ):
            continue
        stripped = stripped.removeprefix("-").strip()
        if not stripped:
            continue
        tag_match = _RELATED_TAG_PATTERN.match(stripped)
        if tag_match:
            objects.append(
                RelatedObject(name=tag_match.group("name"), tag=tag_match.group("tag"))
            )
        else:
            objects.append(RelatedObject(name=stripped, tag=None))
    return objects


# Group lane, compliance, and related-object chunks by model while auditing unusable records.
def build_extra_chunk_index(
    rows: list[dict[str, Any]], config: ConverterConfig, warnings: list[str]
) -> ExtraChunkIndex:
    """Build the model-key-to-extra-chunks index from chunk rows.

    Only rows whose object type is in ``config.extra_chunk_types`` are
    considered; overview/activity/document chunks are skipped. Rows with
    a missing model ID or no extractable content are skipped and
    counted, never invented.

    Args:
        rows: Chunk rows from :func:`read_activity_chunks` (the same
            corpus the activity index is built from).
        config: Converter configuration with the field mapping and the
            requested extra chunk types.
        warnings: Mutable list collecting human-readable warnings.

    Returns:
        The extra chunk index with statistics.
    """
    fields = config.activity_fields
    extra_types = set(config.extra_chunk_types)
    lanes: dict[str, list[Lane]] = {}
    statuses: dict[str, str] = {}
    risk_levels: dict[str, str] = {}
    related: dict[str, list[RelatedObject]] = {}
    rows_by_type: dict[str, int] = {}
    missing_model_keys = 0
    empty_rows = 0
    for row in rows:
        object_type = _row_field(row, fields.object_type, fields.metadata_key)
        if object_type not in extra_types:
            continue
        rows_by_type[object_type] = rows_by_type.get(object_type, 0) + 1
        key = normalize_chunk_model_key(
            _row_field(row, fields.model_id, fields.metadata_key)
        )
        if not key:
            missing_model_keys += 1
            continue
        if object_type == "lane":
            lane = _parse_lane(row, fields)
            if lane is None:
                empty_rows += 1
                continue
            lanes.setdefault(key, []).append(lane)
        elif object_type == "compliance":
            text = _row_field(row, fields.text, fields.metadata_key)
            status = _clean_chunk_value(
                _row_field(row, fields.status, fields.metadata_key)
            ) or _clean_chunk_value(_text_field_value(text, COMPLIANCE_STATUS_LABEL))
            risk_level = _clean_chunk_value(
                _row_field(row, fields.risk_level, fields.metadata_key)
            ) or _clean_chunk_value(_text_field_value(text, COMPLIANCE_RISK_LABEL))
            if status is None and risk_level is None:
                empty_rows += 1
                continue
            # One compliance chunk per model; first non-empty value wins.
            if status is not None and key not in statuses:
                statuses[key] = status
            if risk_level is not None and key not in risk_levels:
                risk_levels[key] = risk_level
        else:  # "other"
            objects = _parse_related_objects(
                _row_field(row, fields.text, fields.metadata_key)
            )
            if not objects:
                empty_rows += 1
                continue
            related.setdefault(key, []).extend(objects)
    chunks = {
        key: ExtraChunks(
            lanes=lanes.get(key, []),
            compliance_status=statuses.get(key),
            compliance_risk_level=risk_levels.get(key),
            related_objects=related.get(key, []),
        )
        for key in (
            lanes.keys() | statuses.keys() | risk_levels.keys() | related.keys()
        )
    }
    if missing_model_keys:
        warnings.append(
            f"{config.activities_path}: skipped {missing_model_keys} extra "
            f"chunk(s) {sorted(extra_types)} with missing {fields.model_id!r}."
        )
    if empty_rows:
        warnings.append(
            f"{config.activities_path}: skipped {empty_rows} extra chunk(s) "
            f"{sorted(extra_types)} without extractable content."
        )
    return ExtraChunkIndex(
        chunks=chunks,
        rows_by_type=rows_by_type,
        missing_model_keys=missing_model_keys,
        empty_rows=empty_rows,
    )
