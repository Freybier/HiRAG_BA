"""Validation for the ADONIS JSONL-to-text converter.

Input validation rejects or reports unusable rows; output validation
verifies the integrity of the generated records. Errors that threaten
data integrity raise :class:`ConversionError` instead of silently
producing misleading output.
"""

from __future__ import annotations

from typing import Any

from .config import ConverterConfig
from .process_textualizer import relationship_sentence
from .relationship_mapper import RelationshipIndex
from .schema import DocumentRecord, ProcessModel


class ConversionError(Exception):
    """Raised when an integrity-threatening condition is detected."""


def validate_unique_ids(label: str, ids: list[str], source_file: str) -> None:
    """Fail when an export contains duplicate primary IDs.

    Args:
        label: Human-readable entity label for the error message.
        ids: All parsed IDs in file order.
        source_file: Name of the file the IDs came from.

    Raises:
        ConversionError: If any ID appears more than once.
    """
    seen: set[str] = set()
    duplicates: list[str] = []
    for identifier in ids:
        if identifier in seen and identifier not in duplicates:
            duplicates.append(identifier)
        seen.add(identifier)
    if duplicates:
        raise ConversionError(
            f"Duplicate {label} IDs in {source_file}: {sorted(duplicates)}. "
            "IDs are authoritative keys; conversion aborted."
        )


def validate_outputs(
    process_records: list[dict[str, Any]],
    document_records: list[dict[str, Any]],
    models: list[ProcessModel],
    documents: list[DocumentRecord],
    index: RelationshipIndex,
    config: ConverterConfig,
) -> list[str]:
    """Validate the generated records against the verified inputs.

    Checks (each raising :class:`ConversionError` on failure):

    - unique record identifiers across all outputs
    - one process record per valid unique input process
    - every canonical relationship textualized exactly once, only in the
      source/parent record (parent-only rule)
    - document sections tile the original content in order, accounting
      for configured overlap

    Args:
        process_records: Generated process record dictionaries.
        document_records: Generated document record dictionaries.
        models: Valid input process models.
        documents: Valid input documents.
        index: Canonical relationship index.
        config: Converter configuration.

    Returns:
        A list of non-fatal validation warnings.

    Raises:
        ConversionError: On any integrity-threatening violation.
    """
    warnings: list[str] = []
    record_ids = [rec["record_id"] for rec in process_records + document_records]
    if len(record_ids) != len(set(record_ids)):
        raise ConversionError("Generated record IDs are not unique.")

    if len(process_records) != len(models):
        raise ConversionError(
            f"Process record count mismatch: {len(process_records)} records "
            f"for {len(models)} valid input processes."
        )
    model_ids = {model.model_id for model in models}
    record_source_ids = {rec["source_id"] for rec in process_records}
    missing = model_ids - record_source_ids
    if missing:
        raise ConversionError(
            f"Missing process records for input IDs: {sorted(missing)}."
        )

    text_by_source: dict[str, str] = {
        rec["source_id"]: rec["text"] for rec in process_records
    }
    models_by_id = {model.model_id: model for model in models}
    for rel in index.ground_truth:
        source = models_by_id[rel.source_id]
        target = models_by_id[rel.target_id]
        # The full sentence contains both endpoint names, so it is unique
        # per canonical relationship (process names are unique in the
        # export) and can be located exactly.
        sentence = relationship_sentence(rel, source, target)
        source_text = text_by_source.get(rel.source_id, "")
        occurrences = source_text.count(sentence)
        if occurrences != 1:
            raise ConversionError(
                f"Relationship {rel.canonical_key} appears {occurrences} "
                f"times in the source record (expected exactly 1)."
            )
        target_text = text_by_source.get(rel.target_id, "")
        if sentence in target_text:
            raise ConversionError(
                f"Parent-only violation: {rel.canonical_key} is repeated in "
                f"the target/child record."
            )
        # A relationship must never be textualized in an unrelated record.
        for other_id, other_text in text_by_source.items():
            if other_id not in (rel.source_id, rel.target_id) and (
                sentence in other_text
            ):
                raise ConversionError(
                    f"Relationship {rel.canonical_key} leaked into the "
                    f"record of unrelated process {other_id}."
                )

    # Document section coverage: ordered, gapless, overlap-accounted.
    sections_by_doc: dict[str, list[dict[str, Any]]] = {}
    for rec in document_records:
        if rec["record_type"] == "DOCUMENT_SECTION":
            sections_by_doc.setdefault(str(rec["source_id"]), []).append(rec)
    docs_by_id = {doc.doc_id: doc for doc in documents}
    for doc_id, sections in sections_by_doc.items():
        document = docs_by_id[doc_id]
        sections.sort(key=lambda rec: int(rec["metadata"]["section_number"]))
        first = sections[0]["metadata"]
        last = sections[-1]["metadata"]
        if int(first["content_start"]) != 0:
            raise ConversionError(
                f"Document {doc_id}: first section does not start at 0."
            )
        if int(last["content_end"]) != len(document.content):
            raise ConversionError(
                f"Document {doc_id}: last section does not reach the end of "
                f"the original content."
            )
        for prev, nxt in zip(sections, sections[1:]):
            prev_meta = prev["metadata"]
            nxt_meta = nxt["metadata"]
            if int(nxt_meta["content_start"]) > int(prev_meta["content_end"]):
                raise ConversionError(
                    f"Document {doc_id}: gap between sections "
                    f"{prev_meta['section_number']} and "
                    f"{nxt_meta['section_number']} would lose content."
                )
            expected_overlap = int(prev_meta["content_end"]) - int(
                nxt_meta["content_start"]
            )
            if int(nxt_meta["overlap_with_previous_chars"]) != expected_overlap:
                raise ConversionError(
                    f"Document {doc_id}: recorded overlap of section "
                    f"{nxt_meta['section_number']} does not match the actual "
                    f"offsets."
                )
    return warnings
