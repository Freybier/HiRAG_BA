"""Configuration for the ADONIS JSONL-to-text converter.

All field mappings are explicit and configurable so the converter can be
adapted if a future export renames columns. The defaults match the
discovered schemas of ``models_40pct.jsonl``, ``hierarchy_40pct.jsonl``,
``documents_40pct.jsonl`` and the ``data_vector_store2`` chunk export
(``description_hierarchy-addon.json``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: Roughly 1200 tokens (HiRAG's default chunk size) in characters.
#: This converter deliberately works with character counts, not token
#: counts, to stay dependency-free. Do not confuse the two.
DEFAULT_MAX_SECTION_CHARS: int = 4000

#: Default number of characters repeated between consecutive sections.
DEFAULT_SECTION_OVERLAP_CHARS: int = 200

#: Process records above this size trigger a warning (they are never split).
DEFAULT_PROCESS_WARN_CHARS: int = 8000

#: The only hierarchy textualization mode implemented by this converter.
SUPPORTED_HIERARCHY_FORMATS: tuple[str, ...] = ("parent-only",)

#: Extra chunk object types (beyond ``activity``) of the chunks corpus that
#: the converter can append to process records: ``lane`` (swimlane/pool
#: chunks), ``compliance`` (per-model status/risk-level chunks) and
#: ``other`` (per-model related-objects listings).
SUPPORTED_EXTRA_CHUNK_TYPES: tuple[str, ...] = ("lane", "compliance", "other")


@dataclass(frozen=True)
class ModelsFields:
    """Column mapping for the models JSONL export."""

    id: str = "model_id"
    name: str = "name"
    description: str = "description"
    embedded_json: str = "json_string"
    purpose: str = "purpose"
    tree_parent_id: str = "parent_model_id"
    hierarchy_path: str = "hierarchy_path"
    hierarchy_depth: str = "hierarchy_depth"


@dataclass(frozen=True)
class HierarchyFields:
    """Column mapping for the hierarchy JSONL export.

    Direction (verified against the 40% sample): ``parent_model_id`` is the
    source/parent process that owns the relation, ``child_model_id`` is the
    target process the relation points to.
    """

    source_id: str = "parent_model_id"
    target_id: str = "child_model_id"
    source_name: str = "parent_model_name"
    target_name: str = "child_model_name"
    relation_type: str = "relation_type"


@dataclass(frozen=True)
class DocumentFields:
    """Column mapping for the documents JSONL export."""

    id: str = "id"
    name: str = "name"
    content: str = "content"
    file_type: str = "file_type"
    object_id: str = "document_object_id"


@dataclass(frozen=True)
class DescriptionFields:
    """Column mapping for the ``data_vector_store2`` chunk export.

    The addon file is a JSON array (or JSONL) of chunk rows; only the
    ``overview`` chunks carry the real process descriptions.
    """

    model_id: str = "model_id"
    object_type: str = "object_type"
    text: str = "text"
    overview_type: str = "overview"


@dataclass(frozen=True)
class ActivityFields:
    """Field mapping for activity chunks in a ``chunks.jsonl`` corpus.

    Fields are read from the row's top level first and fall back to the
    nested metadata object (``metadata_`` in the production-shaped
    ``chunks.jsonl``), so both the raw chunk export and the converted
    corpus are accepted.
    """

    model_id: str = "model_id"
    object_type: str = "object_type"
    text: str = "text"
    step_number: str = "step_number"
    metadata_key: str = "metadata_"
    activity_type: str = "activity"
    # Fields only used by the extra chunk types (lane/compliance/other).
    node_id: str = "node_id"
    lane_name: str = "lane_name"
    object_meta_name: str = "object_meta_name"
    status: str = "status"
    risk_level: str = "risk_level"


@dataclass(frozen=True)
class ConverterConfig:
    """Top-level converter configuration.

    Attributes:
        models_path: Path to the models JSONL export.
        hierarchy_path: Path to the hierarchy JSONL export.
        documents_path: Path to the documents JSONL export.
        output_dir: Directory receiving all generated files.
        hierarchy_format: Hierarchy textualization mode. Only
            ``parent-only`` is implemented.
        chunk_documents: Split large documents into sections. Disabled by
            default: HiRAG applies its own chunking internally, so each
            document is emitted as one full record.
        max_section_chars: Maximum characters per document section (only
            applies when ``chunk_documents`` is enabled).
        section_overlap_chars: Character overlap between consecutive
            document sections (only applies when ``chunk_documents`` is
            enabled). Overlapped text is repeated context, not independent
            source information.
        process_size_warn_chars: Process records larger than this trigger
            a warning; process records are never split.
        export_txt: Also export one plain ``.txt`` file per record.
        export_combined_text: Also export two combined plain-text files
            (one for processes, one for documents) with a deterministic
            delimiter between records so they can be split into an array.
        excluded_file_types: Document file types (lower-case, e.g.
            ``("xlsx",)``) that are skipped entirely and reported.
        descriptions_path: Optional path to the ``data_vector_store2``
            chunk export whose overview chunks supply the real process
            descriptions missing from the models export.
        activities_path: Optional path to a ``chunks.jsonl``-style chunk
            corpus whose ``activity`` chunks supply the per-process
            activity steps appended after the relationship sections.
        extra_chunk_types: Optional chunk object types (subset of
            ``SUPPORTED_EXTRA_CHUNK_TYPES``) from the same chunks corpus
            that are additionally appended to the process records: lanes
            as ``LANES`` sentences, compliance status/risk level as
            ``COMPLIANCE`` sentences and the related-objects listing as
            ``RELATED_OBJECTS`` sentences, all after the ``ACTIVITIES``
            section. Requires ``activities_path`` (the same file is the
            source for all chunk-derived sections).
        limit: Optional smoke-test limit applied to input rows per file.
        models_fields: Field mapping for the models export.
        hierarchy_fields: Field mapping for the hierarchy export.
        document_fields: Field mapping for the documents export.
        subprocess_relation_types: Relation types textualized as
            subprocess relationships.
        cross_reference_relation_types: Relation types textualized as
            cross-references.
    """

    models_path: Path
    hierarchy_path: Path
    documents_path: Path
    output_dir: Path
    hierarchy_format: str = "parent-only"
    chunk_documents: bool = False
    max_section_chars: int = DEFAULT_MAX_SECTION_CHARS
    section_overlap_chars: int = DEFAULT_SECTION_OVERLAP_CHARS
    process_size_warn_chars: int = DEFAULT_PROCESS_WARN_CHARS
    export_txt: bool = False
    export_combined_text: bool = False
    excluded_file_types: tuple[str, ...] = ()
    descriptions_path: Path | None = None
    activities_path: Path | None = None
    extra_chunk_types: tuple[str, ...] = ()
    limit: int | None = None
    models_fields: ModelsFields = field(default_factory=ModelsFields)
    hierarchy_fields: HierarchyFields = field(default_factory=HierarchyFields)
    document_fields: DocumentFields = field(default_factory=DocumentFields)
    description_fields: DescriptionFields = field(default_factory=DescriptionFields)
    activity_fields: ActivityFields = field(default_factory=ActivityFields)
    subprocess_relation_types: tuple[str, ...] = ("C_SUB_PROCESS",)
    cross_reference_relation_types: tuple[str, ...] = ("C_CROSS_REFERENCE",)

    processes_file: str = "processes_textualized.jsonl"
    documents_file: str = "documents_textualized.jsonl"
    ground_truth_file: str = "relationship_ground_truth.jsonl"
    metadata_file: str = "metadata_mapping.jsonl"
    report_file: str = "conversion_report.json"
    entity_lookup_file: str = "entity_lookup.json"
    processes_combined_file: str = "processes_text.txt"
    documents_combined_file: str = "documents_text.txt"

    # Validate chunk limits and addon settings, normalizing file extensions and extra chunk types.
    def __post_init__(self) -> None:
        """Validate configuration consistency.

        Raises:
            ValueError: If the hierarchy format is unsupported or the
                chunking parameters are invalid.
        """
        if self.hierarchy_format not in SUPPORTED_HIERARCHY_FORMATS:
            raise ValueError(
                f"Unsupported hierarchy format {self.hierarchy_format!r}. "
                f"Supported: {SUPPORTED_HIERARCHY_FORMATS}."
            )
        if self.max_section_chars <= 0:
            raise ValueError("max_section_chars must be positive.")
        if not 0 <= self.section_overlap_chars < self.max_section_chars:
            raise ValueError(
                "section_overlap_chars must be >= 0 and < max_section_chars."
            )
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be a positive integer or None.")
        normalized = tuple(
            file_type.strip().lower()
            for file_type in self.excluded_file_types
            if file_type.strip()
        )
        if normalized != self.excluded_file_types:
            object.__setattr__(self, "excluded_file_types", normalized)
        normalized_types = tuple(
            chunk_type.strip().lower()
            for chunk_type in self.extra_chunk_types
            if chunk_type.strip()
        )
        if normalized_types != self.extra_chunk_types:
            object.__setattr__(self, "extra_chunk_types", normalized_types)
        unknown_types = set(self.extra_chunk_types) - set(SUPPORTED_EXTRA_CHUNK_TYPES)
        if unknown_types:
            raise ValueError(
                f"Unsupported extra chunk type(s) {sorted(unknown_types)}. "
                f"Supported: {SUPPORTED_EXTRA_CHUNK_TYPES}."
            )
        if self.extra_chunk_types and self.activities_path is None:
            raise ValueError(
                "extra_chunk_types requires activities_path: the chunks "
                "corpus is the shared source for activities and the extra "
                "chunk types."
            )
