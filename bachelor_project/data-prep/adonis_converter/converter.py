"""ADONIS JSONL-to-text converter for HiRAG (parent-only hierarchy mode).

Reads the three ADONIS exports (models, hierarchy, documents) and writes
deterministic, LLM-friendly textual records plus structured sidecar files:

- ``processes_textualized.jsonl``: one record per process, with verified
  outgoing relationships embedded in the source process's record only
  (parent-only representation).
- ``documents_textualized.jsonl``: one record per document or document
  section (deterministic chunking for large documents).
- ``relationship_ground_truth.jsonl``: all verified, deduplicated
  relationships for later evaluation of HiRAG's extraction performance.
- ``metadata_mapping.jsonl``: full original rows and section offsets for
  every generated record.
- ``entity_lookup.json``: authoritative ID-to-name lookup.
- ``conversion_report.json``: statistics, validation results, warnings.

The converter is deterministic and rule-based: no LLM calls, no
embeddings, no semantic similarity, no API access. The original JSONL
files are never modified.

Usage:
    python converter.py \\
        --models models_40pct.jsonl \\
        --hierarchy hierarchy_40pct.jsonl \\
        --documents documents_40pct.jsonl \\
        --output ./output \\
        --hierarchy-format parent-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    # Allow running as a plain script: python converter.py ...
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "adonis_converter"

from .activity_addon import (
    Activity,
    ExtraChunks,
    build_activity_index,
    build_extra_chunk_index,
    read_activity_chunks,
)
from .config import (
    SUPPORTED_EXTRA_CHUNK_TYPES,
    SUPPORTED_HIERARCHY_FORMATS,
    ConverterConfig,
)
from .description_addon import (
    build_description_index,
    normalize_model_key,
    read_description_chunks,
)
from .document_textualizer import render_document_record
from .process_textualizer import (
    description_source,
    render_process_text,
    sanitize_filename,
)
from .relationship_mapper import build_relationship_index, check_direction
from .schema import (
    SchemaError,
    parse_documents,
    parse_models,
    parse_relationships,
    read_jsonl,
)
from .validator import ConversionError, validate_outputs, validate_unique_ids

#: Deterministic delimiter separating records in the combined text files.
#: ``str.split()`` on this line yields one array element per record.
COMBINED_RECORD_DELIMITER: str = "=== RECORD SEPARATOR ==="


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records as JSONL with UTF-8 encoding and stable key order.

    Args:
        path: Destination file.
        records: Records to serialize, one JSON object per line.
    """
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    """Write a single JSON document with UTF-8 encoding.

    Args:
        path: Destination file.
        payload: JSON-serializable payload.
    """
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_combined_text(
    path: Path, records: list[dict[str, Any]], delimiter: str
) -> None:
    """Write all record texts into one file separated by a delimiter line.

    Splitting the file content on the delimiter line yields exactly one
    array element per record, suitable for interfaces that accept a list
    of strings (e.g. HiRAG's ``insert``).

    Args:
        path: Destination file.
        records: Records with a ``text`` field, in write order.
        delimiter: Delimiter line inserted between records.

    Raises:
        ConversionError: If any record text already contains the
            delimiter line, which would corrupt the array split.
    """
    delimiter_line = f"\n{delimiter}\n"
    for record in records:
        text = str(record["text"])
        if delimiter in text:
            raise ConversionError(
                f"Record {record['record_id']} contains the combined-text "
                f"delimiter {delimiter!r}; choose a different delimiter."
            )
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(delimiter_line.join(str(record["text"]) for record in records))


def run_conversion(config: ConverterConfig) -> dict[str, Any]:
    """Run the full deterministic conversion pipeline.

    Args:
        config: Converter configuration.

    Returns:
        The conversion report dictionary (also written to disk).

    Raises:
        SchemaError: On malformed or missing input files.
        ConversionError: On integrity-threatening validation failures.
    """
    warnings: list[str] = []

    models_read = read_jsonl(config.models_path, config.limit)
    hierarchy_read = read_jsonl(config.hierarchy_path, config.limit)
    documents_read = read_jsonl(config.documents_path, config.limit)

    models, rejected_models = parse_models(models_read.rows, config, warnings)
    documents, rejected_documents = parse_documents(
        documents_read.rows, config, warnings
    )
    valid_documents_before_exclusion = len(documents)
    excluded_documents: list[dict[str, Any]] = []
    if config.excluded_file_types:
        kept: list[Any] = []
        for document in documents:
            if (
                document.file_type
                and document.file_type.lower() in config.excluded_file_types
            ):
                excluded_documents.append(
                    {
                        "document_id": document.doc_id,
                        "title": document.title,
                        "file_type": document.file_type,
                        "source_file": document.source_file,
                        "source_line": document.line_number,
                    }
                )
            else:
                kept.append(document)
        documents = kept
        if excluded_documents:
            warnings.append(
                f"Excluded {len(excluded_documents)} document(s) by file type "
                f"{sorted(config.excluded_file_types)}; see "
                "documents.excluded_by_file_type in the report."
            )
    relationship_rows = parse_relationships(hierarchy_read.rows, config, warnings)

    # Optional data_vector_store2 chunk export supplying the real process
    # descriptions missing from the models export. The smoke-test limit is
    # intentionally NOT applied here: chunks are grouped per model, so a
    # row limit would skew coverage instead of sampling it.
    addon_descriptions: dict[str, str] = {}
    addon_stats: dict[str, Any] | None = None
    if config.descriptions_path is not None:
        addon_rows = read_description_chunks(config.descriptions_path)
        description_index = build_description_index(addon_rows, config, warnings)
        model_keys = {normalize_model_key(m.model_id) for m in models}
        unmatched_keys: list[str] = []
        for key, description in description_index.descriptions.items():
            if key in model_keys:
                addon_descriptions[key] = description
            else:
                unmatched_keys.append(key)
        if unmatched_keys:
            warnings.append(
                f"{len(unmatched_keys)} overview description(s) reference "
                "models that are not in the models export; they were "
                "ignored (see inputs.descriptions.unmatched_model_ids)."
            )
        addon_stats = {
            "rows_read": description_index.rows_read,
            "overview_rows": description_index.overview_rows,
            "empty_descriptions": description_index.empty_descriptions,
            "duplicate_overview_rows": description_index.duplicate_overview_rows,
            "matched_models": len(addon_descriptions),
            "unmatched_model_ids": sorted(unmatched_keys),
        }

    # Optional chunks corpus supplying each process's activity steps. The
    # smoke-test limit is intentionally NOT applied here either: chunks
    # are grouped per model, so a row limit would skew coverage instead
    # of sampling it.
    addon_activities: dict[str, list[Activity]] = {}
    addon_extras: dict[str, ExtraChunks] = {}
    activity_stats: dict[str, Any] | None = None
    if config.activities_path is not None:
        activity_rows = read_activity_chunks(config.activities_path)
        activity_index = build_activity_index(activity_rows, config, warnings)
        activity_model_keys = {normalize_model_key(m.model_id) for m in models}
        unmatched_activity_keys: list[str] = []
        for key, model_activities in activity_index.activities.items():
            if key in activity_model_keys:
                addon_activities[key] = model_activities
            else:
                unmatched_activity_keys.append(key)
        if unmatched_activity_keys:
            warnings.append(
                f"{len(unmatched_activity_keys)} model(s) with activity chunks "
                "are not in the models export; their activities were ignored "
                "(see inputs.activities.unmatched_model_ids)."
            )
        activity_stats = {
            "rows_read": activity_index.rows_read,
            "activity_rows": activity_index.activity_rows,
            "missing_model_keys": activity_index.missing_model_keys,
            "unnamed_activities": activity_index.unnamed_activities,
            "matched_models": len(addon_activities),
            "matched_activities": sum(
                len(model_activities) for model_activities in addon_activities.values()
            ),
            "unmatched_model_ids": sorted(unmatched_activity_keys),
        }
        # Extra chunk types (lane/compliance/other) are read from the
        # same already-loaded corpus rows — no second file pass.
        if config.extra_chunk_types:
            extra_index = build_extra_chunk_index(activity_rows, config, warnings)
            unmatched_extra_keys: list[str] = []
            for key, model_extras in extra_index.chunks.items():
                if key in activity_model_keys:
                    addon_extras[key] = model_extras
                else:
                    unmatched_extra_keys.append(key)
            if unmatched_extra_keys:
                warnings.append(
                    f"{len(unmatched_extra_keys)} model(s) with extra chunks "
                    f"{sorted(config.extra_chunk_types)} are not in the models "
                    "export; their extra chunks were ignored (see "
                    "inputs.activities.extras.unmatched_model_ids)."
                )
            activity_stats["extras"] = {
                "extra_chunk_types": list(config.extra_chunk_types),
                "rows_by_type": extra_index.rows_by_type,
                "missing_model_keys": extra_index.missing_model_keys,
                "empty_rows": extra_index.empty_rows,
                "matched_models": len(addon_extras),
                "matched_lanes": sum(
                    len(model_extras.lanes) for model_extras in addon_extras.values()
                ),
                "matched_compliance": sum(
                    1
                    for model_extras in addon_extras.values()
                    if model_extras.compliance_status
                    or model_extras.compliance_risk_level
                ),
                "matched_related_objects": sum(
                    len(model_extras.related_objects)
                    for model_extras in addon_extras.values()
                ),
                "unmatched_model_ids": sorted(unmatched_extra_keys),
            }

    validate_unique_ids(
        "process",
        [m.model_id for m in models],
        models_read.rows[0].source_file
        if models_read.rows
        else config.models_path.name,
    )
    validate_unique_ids(
        "document",
        [d.doc_id for d in documents],
        documents_read.rows[0].source_file
        if documents_read.rows
        else config.documents_path.name,
    )

    models_by_id = {model.model_id: model for model in models}
    index = build_relationship_index(relationship_rows, models_by_id, config)
    direction_report = check_direction(relationship_rows, models_by_id, config)
    if isinstance(direction_report.get("verdict"), str) and direction_report[
        "verdict"
    ].startswith("ambiguous"):
        warnings.append(f"Relationship direction check: {direction_report['verdict']}")

    # The real export stores the ADONIS metaName in the description column
    # instead of free text. Count how often that happens instead of
    # silently dropping or blindly trusting the column.
    description_equals_metaname = sum(
        1
        for model in models
        if model.description and model.description == model.meta_name
    )

    process_records: list[dict[str, Any]] = []
    oversized_processes: list[dict[str, Any]] = []
    description_source_counts = {"models_export": 0, "description_addon": 0, "none": 0}
    for model in models:
        outgoing = index.outgoing_for(model.model_id)
        model_key = normalize_model_key(model.model_id)
        addon_description = addon_descriptions.get(model_key)
        activities = addon_activities.get(model_key)
        extras = addon_extras.get(model_key)
        text = render_process_text(
            model, outgoing, models_by_id, config, addon_description, activities, extras
        )
        source = description_source(model, addon_description)
        description_source_counts[source] += 1
        if len(text) > config.process_size_warn_chars:
            oversized_processes.append(
                {
                    "model_id": model.model_id,
                    "name": model.name,
                    "text_chars": len(text),
                }
            )
        record_metadata: dict[str, Any] = {
            "source_file": model.source_file,
            "source_line": model.line_number,
            "description_source": source,
            "outgoing_relationship_count": len(outgoing),
            "subprocess_count": sum(
                1 for rel in outgoing if rel.category == "subprocess"
            ),
            "cross_reference_count": sum(
                1 for rel in outgoing if rel.category == "cross_reference"
            ),
        }
        if config.activities_path is not None:
            record_metadata["activity_count"] = len(activities or [])
        if config.extra_chunk_types:
            record_metadata["lane_count"] = len(extras.lanes) if extras else 0
            record_metadata["has_compliance"] = bool(
                extras and (extras.compliance_status or extras.compliance_risk_level)
            )
            record_metadata["related_object_count"] = (
                len(extras.related_objects) if extras else 0
            )
        process_records.append(
            {
                "record_id": f"process_{model.model_id}",
                "record_type": "PROCESS",
                "source_id": model.model_id,
                "text": text,
                "metadata": record_metadata,
            }
        )

    document_records: list[dict[str, Any]] = []
    chunked_documents = 0
    hard_splits = 0
    for document in documents:
        records, doc_hard_splits = render_document_record(document, config)
        hard_splits += doc_hard_splits
        if len(records) > 1:
            chunked_documents += 1
        document_records.extend(records)

    output_warnings = validate_outputs(
        process_records, document_records, models, documents, index, config
    )
    warnings.extend(output_warnings)
    if oversized_processes:
        warnings.append(
            f"{len(oversized_processes)} process record(s) exceed "
            f"{config.process_size_warn_chars} characters; they were kept "
            "whole (process records are never split)."
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    processes_path = config.output_dir / config.processes_file
    documents_path = config.output_dir / config.documents_file
    ground_truth_path = config.output_dir / config.ground_truth_file
    metadata_path = config.output_dir / config.metadata_file
    report_path = config.output_dir / config.report_file
    entity_lookup_path = config.output_dir / config.entity_lookup_file

    _write_jsonl(processes_path, process_records)
    _write_jsonl(documents_path, document_records)
    _write_jsonl(
        ground_truth_path,
        [
            {
                "source_id": rel.source_id,
                "source_name": models_by_id[rel.source_id].name,
                "relationship_type": rel.relation_type,
                "target_id": rel.target_id,
                "target_name": models_by_id[rel.target_id].name,
                "occurrences_in_source": rel.occurrences,
                "source_file": rel.source_file,
                "first_source_line": rel.first_line,
            }
            for rel in index.ground_truth
        ],
    )

    metadata_records: list[dict[str, Any]] = []
    for model, record in zip(models, process_records):
        metadata_records.append(
            {
                "record_id": record["record_id"],
                "record_type": record["record_type"],
                "source_id": model.model_id,
                "source_file": model.source_file,
                "source_line": model.line_number,
                "original_record": model.raw,
            }
        )
    documents_by_id = {doc.doc_id: doc for doc in documents}
    for record in document_records:
        document = documents_by_id[str(record["source_id"])]
        # Keep the full original row except the bulky extracted content:
        # repeating it for every section would blow up the sidecar file
        # (hundreds of MB). The content stays available in the untouched
        # source file and is reconstructable via the section offsets.
        original_record = {
            key: value
            for key, value in document.raw.items()
            if key != config.document_fields.content
        }
        metadata_records.append(
            {
                "record_id": record["record_id"],
                "record_type": record["record_type"],
                "source_id": document.doc_id,
                "source_file": document.source_file,
                "source_line": document.line_number,
                "content_chars": len(document.content),
                "content_omitted_from_sidecar": True,
                "section": (
                    {
                        "section_number": record["metadata"]["section_number"],
                        "total_sections": record["metadata"]["total_sections"],
                        "content_start": record["metadata"]["content_start"],
                        "content_end": record["metadata"]["content_end"],
                        "overlap_with_previous_chars": record["metadata"][
                            "overlap_with_previous_chars"
                        ],
                    }
                    if record["record_type"] == "DOCUMENT_SECTION"
                    else None
                ),
                "original_record": original_record,
            }
        )
    _write_jsonl(metadata_path, metadata_records)

    _write_json(
        entity_lookup_path,
        {
            "processes": {model.model_id: model.name for model in models},
            "documents": {doc.doc_id: doc.title for doc in documents},
        },
    )

    if config.export_txt:
        txt_dir = config.output_dir / "txt"
        txt_dir.mkdir(parents=True, exist_ok=True)
        for record in process_records + document_records:
            txt_path = txt_dir / f"{sanitize_filename(str(record['record_id']))}.txt"
            txt_path.write_text(str(record["text"]), encoding="utf-8")

    processes_combined_path: Path | None = None
    documents_combined_path: Path | None = None
    if config.export_combined_text:
        processes_combined_path = config.output_dir / config.processes_combined_file
        documents_combined_path = config.output_dir / config.documents_combined_file
        _write_combined_text(
            processes_combined_path, process_records, COMBINED_RECORD_DELIMITER
        )
        _write_combined_text(
            documents_combined_path, document_records, COMBINED_RECORD_DELIMITER
        )

    relation_type_counts: dict[str, int] = {}
    for rel in index.ground_truth:
        relation_type_counts[rel.relation_type] = (
            relation_type_counts.get(rel.relation_type, 0) + 1
        )

    report: dict[str, Any] = {
        "config": {
            "models_path": str(config.models_path),
            "hierarchy_path": str(config.hierarchy_path),
            "documents_path": str(config.documents_path),
            "output_dir": str(config.output_dir),
            "hierarchy_format": config.hierarchy_format,
            "chunk_documents": config.chunk_documents,
            "max_section_chars": config.max_section_chars,
            "section_overlap_chars": config.section_overlap_chars,
            "process_size_warn_chars": config.process_size_warn_chars,
            "export_txt": config.export_txt,
            "export_combined_text": config.export_combined_text,
            "excluded_file_types": list(config.excluded_file_types),
            "descriptions_path": (
                str(config.descriptions_path) if config.descriptions_path else None
            ),
            "activities_path": (
                str(config.activities_path) if config.activities_path else None
            ),
            "extra_chunk_types": list(config.extra_chunk_types),
            "limit": config.limit,
        },
        "schema_mapping": {
            "models": vars(config.models_fields),
            "hierarchy": vars(config.hierarchy_fields),
            "documents": vars(config.document_fields),
            "descriptions": vars(config.description_fields),
            "notes": [
                (
                    "models.description contains the ADONIS metaName constant, "
                    "not free text, in the inspected 40% sample; descriptions "
                    "equal to the metaName are not emitted as PROCESS_DESCRIPTION"
                ),
                (
                    "the optional descriptions addon is a data_vector_store2 "
                    "chunk export; overview chunks are joined to models via the "
                    "model_id column (braces stripped, lowercased) and supply "
                    "the real PROCESS_DESCRIPTION text"
                ),
                (
                    "the optional activities corpus is a chunks.jsonl-style "
                    "chunk file; activity chunks are joined to models via "
                    "model_id (model_{uuid} and braced forms normalized) and "
                    "appended as ACTIVITIES sentences after the relationship "
                    "sections, ordered by step_number; with "
                    "extra_chunk_types the lane/compliance/other chunks of "
                    "the same corpus are appended as LANES, COMPLIANCE and "
                    "RELATED_OBJECTS sentences after ACTIVITIES"
                ),
                (
                    "models.parent_model_id/hierarchy_path describe the ADONIS "
                    "model tree, which is a different structure than the "
                    "C_SUB_PROCESS relation; tree parents are preserved in the "
                    "sidecar metadata only, not textualized"
                ),
                (
                    "document ids are numeric database keys; document_object_id "
                    "carries the ADONIS object UUID"
                ),
            ],
        },
        "inputs": {
            "models": {
                "rows_read": len(models_read.rows),
                "blank_lines": models_read.blank_lines,
                "valid": len(models),
                "rejected": len(rejected_models),
            },
            "hierarchy": {
                "rows_read": len(hierarchy_read.rows),
                "blank_lines": hierarchy_read.blank_lines,
            },
            "documents": {
                "rows_read": len(documents_read.rows),
                "blank_lines": documents_read.blank_lines,
                "valid": valid_documents_before_exclusion,
                "rejected": len(rejected_documents),
            },
            "descriptions": addon_stats,
            "activities": activity_stats,
        },
        "direction_check": direction_report,
        "relationships": {
            "unique_canonical": len(index.ground_truth),
            "by_type": relation_type_counts,
            "duplicates_removed": len(index.duplicates),
            "duplicate_details": index.duplicates,
            "unknown_types": index.unknown_types,
            "broken_references": index.broken_references,
            "missing_endpoint_ids": index.missing_endpoint_ids,
            "name_mismatches": index.name_mismatches,
            "process_document_links_created": 0,
        },
        "processes": {
            "records": len(process_records),
            "with_subprocesses": sum(
                1 for rec in process_records if rec["metadata"]["subprocess_count"] > 0
            ),
            "with_cross_references": sum(
                1
                for rec in process_records
                if rec["metadata"]["cross_reference_count"] > 0
            ),
            "description_equals_metaname_count": description_equals_metaname,
            "description_source_counts": description_source_counts,
            "with_activities": sum(
                1 for rec in process_records if rec["metadata"].get("activity_count")
            ),
            "with_lanes": sum(
                1 for rec in process_records if rec["metadata"].get("lane_count")
            ),
            "with_compliance": sum(
                1 for rec in process_records if rec["metadata"].get("has_compliance")
            ),
            "with_related_objects": sum(
                1
                for rec in process_records
                if rec["metadata"].get("related_object_count")
            ),
            "oversized_records": oversized_processes,
        },
        "documents": {
            "source_documents": len(documents),
            "records": len(document_records),
            "chunked_documents": chunked_documents,
            "hard_splits": hard_splits,
            "excluded_by_file_type": excluded_documents,
        },
        "validation": {
            "errors": [],
            "warnings": warnings,
        },
        "outputs": {
            "processes": str(processes_path),
            "documents": str(documents_path),
            "relationship_ground_truth": str(ground_truth_path),
            "metadata_mapping": str(metadata_path),
            "entity_lookup": str(entity_lookup_path),
            "report": str(report_path),
            "processes_combined": (
                str(processes_combined_path) if processes_combined_path else None
            ),
            "documents_combined": (
                str(documents_combined_path) if documents_combined_path else None
            ),
        },
    }
    _write_json(report_path, report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert ADONIS JSONL exports into deterministic textual "
            "records for HiRAG (parent-only hierarchy representation)."
        )
    )
    parser.add_argument("--models", required=True, type=Path, help="Models JSONL file.")
    parser.add_argument(
        "--hierarchy", required=True, type=Path, help="Hierarchy JSONL file."
    )
    parser.add_argument(
        "--documents", required=True, type=Path, help="Documents JSONL file."
    )
    parser.add_argument("--output", required=True, type=Path, help="Output directory.")
    parser.add_argument(
        "--hierarchy-format",
        default="parent-only",
        choices=SUPPORTED_HIERARCHY_FORMATS,
        help="Hierarchy textualization mode (only parent-only is supported).",
    )
    parser.add_argument(
        "--chunk-documents",
        action="store_true",
        help=(
            "Split large documents into sections. Off by default: HiRAG "
            "chunks internally, so documents are emitted whole."
        ),
    )
    parser.add_argument(
        "--max-section-chars",
        type=int,
        default=4000,
        help=(
            "Maximum characters per document section, only with "
            "--chunk-documents (default: 4000)."
        ),
    )
    parser.add_argument(
        "--section-overlap-chars",
        type=int,
        default=200,
        help=(
            "Character overlap between document sections, only with "
            "--chunk-documents (default: 200)."
        ),
    )
    parser.add_argument(
        "--process-warn-chars",
        type=int,
        default=8000,
        help="Warn when a process record exceeds this size (default: 8000).",
    )
    parser.add_argument(
        "--export-txt",
        action="store_true",
        help="Also write one plain .txt file per record into output/txt/.",
    )
    parser.add_argument(
        "--combined-text",
        action="store_true",
        help=(
            "Also write processes_text.txt and documents_text.txt: one file "
            "each with records separated by a deterministic delimiter line, "
            "suitable for splitting into an array of strings."
        ),
    )
    parser.add_argument(
        "--exclude-file-type",
        action="append",
        default=[],
        metavar="EXT",
        dest="excluded_file_types",
        help=(
            "Skip documents with this file type (repeatable, "
            "case-insensitive). No file types are excluded by default; "
            "each use of this flag adds another type."
        ),
    )
    parser.add_argument(
        "--descriptions",
        type=Path,
        default=None,
        help=(
            "Optional data_vector_store2 chunk export (JSON array or JSONL) "
            "whose overview chunks supply the real process descriptions "
            "missing from the models export."
        ),
    )
    parser.add_argument(
        "--include-activities",
        nargs="?",
        const="",
        default=None,
        metavar="CHUNKS_JSONL",
        dest="include_activities",
        help=(
            "Append each process's activity steps (the activity chunks of "
            "the chunks corpus) as an ACTIVITIES section after the "
            "relationship sections. Optionally takes the chunks.jsonl "
            "path; without a value, chunks.jsonl next to the models "
            "export is used. Off by default."
        ),
    )
    parser.add_argument(
        "--include-everything",
        nargs="?",
        const="",
        default=None,
        metavar="CHUNKS_JSONL",
        dest="include_everything",
        help=(
            "Like --include-activities, but additionally appends the "
            "lane, compliance and other chunks of the chunks corpus as "
            "LANES, COMPLIANCE and RELATED_OBJECTS sections after "
            "ACTIVITIES. Optionally takes the chunks.jsonl path; without "
            "a value, chunks.jsonl next to the models export is used. "
            "Off by default."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Smoke-test limit: read at most N rows per input file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        Process exit code: 0 on success, 2 on integrity/parse failure.
    """
    args = build_arg_parser().parse_args(argv)
    chunks_arg = (
        args.include_everything
        if args.include_everything is not None
        else args.include_activities
    )
    activities_path: Path | None = None
    if chunks_arg is not None:
        activities_path = (
            Path(chunks_arg) if chunks_arg else args.models.parent / "chunks.jsonl"
        )
    extra_chunk_types: tuple[str, ...] = (
        SUPPORTED_EXTRA_CHUNK_TYPES if args.include_everything is not None else ()
    )
    config = ConverterConfig(
        models_path=args.models,
        hierarchy_path=args.hierarchy,
        documents_path=args.documents,
        output_dir=args.output,
        hierarchy_format=args.hierarchy_format,
        chunk_documents=args.chunk_documents,
        max_section_chars=args.max_section_chars,
        section_overlap_chars=args.section_overlap_chars,
        process_size_warn_chars=args.process_warn_chars,
        export_txt=args.export_txt,
        export_combined_text=args.combined_text,
        excluded_file_types=tuple(args.excluded_file_types),
        descriptions_path=args.descriptions,
        activities_path=activities_path,
        extra_chunk_types=extra_chunk_types,
        limit=args.limit,
    )
    try:
        report = run_conversion(config)
    except (SchemaError, ConversionError) as exc:
        print(f"Conversion failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"Conversion finished: {report['processes']['records']} process "
        f"records, {report['documents']['records']} document records, "
        f"{report['relationships']['unique_canonical']} verified "
        f"relationships. Report: {report['outputs']['report']}"
    )
    addon_counts = report["processes"]["description_source_counts"]
    if addon_counts["description_addon"]:
        print(
            f"Descriptions enriched from addon: "
            f"{addon_counts['description_addon']} process records."
        )
    activity_stats = report["inputs"].get("activities")
    if activity_stats:
        print(
            f"Activities included: {activity_stats['matched_activities']} steps "
            f"across {activity_stats['matched_models']} process records."
        )
        extras_stats = activity_stats.get("extras")
        if extras_stats:
            print(
                "Extra chunks included: "
                f"{extras_stats['matched_lanes']} lanes, "
                f"{extras_stats['matched_compliance']} compliance records, "
                f"{extras_stats['matched_related_objects']} related objects "
                f"across {extras_stats['matched_models']} process records."
            )
    if report["validation"]["warnings"]:
        print(f"Warnings: {len(report['validation']['warnings'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
