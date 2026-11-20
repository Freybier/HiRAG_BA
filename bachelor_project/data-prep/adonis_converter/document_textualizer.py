"""Document textualization with optional deterministic section splitting.

Documents are processed independently from the process hierarchy. No
process-document relationships are created, attached or inferred: the
export does not expose machine-readable links, so every generated record
carries ``linked_process: unknown`` and
``relationship_provenance: unavailable`` in its structured metadata.

Chunking works on character offsets (never token counts), prefers
paragraph boundaries, falls back to sentence boundaries, and only then
hard-cuts. Section offsets into the original content are recorded so the
original document can be reconstructed exactly (accounting for overlap).
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass

from .config import ConverterConfig
from .schema import DocumentRecord

_PARAGRAPH_PATTERN = re.compile(r"\n\s*\n")
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class SectionSpan:
    """Character span of one document section in the original content."""

    number: int
    start: int
    end: int
    overlap_with_previous: int
    hard_split: bool


def _breakpoints(content: str) -> list[int]:
    """Collect sorted candidate split points (paragraphs, then sentences).

    Args:
        content: The full document content.

    Returns:
        Sorted character offsets at which a section may end cleanly.
    """
    points = {0, len(content)}
    for match in _PARAGRAPH_PATTERN.finditer(content):
        points.add(match.end())
    for match in _SENTENCE_PATTERN.finditer(content):
        points.add(match.end())
    return sorted(points)


def compute_section_spans(
    content: str, max_chars: int, overlap_chars: int
) -> tuple[list[SectionSpan], int]:
    """Split content into deterministic, ordered section spans.

    Args:
        content: The full document content.
        max_chars: Maximum characters per section.
        overlap_chars: Characters of trailing context repeated at the start
            of the next section. Repeated text is context, not independent
            source information.

    Returns:
        A tuple of the ordered spans and the number of hard splits
        (places where no paragraph/sentence boundary was available).
    """
    if not content:
        return [], 0
    if len(content) <= max_chars:
        return [SectionSpan(1, 0, len(content), 0, False)], 0
    points = _breakpoints(content)
    spans: list[SectionSpan] = []
    hard_splits = 0
    start = 0
    length = len(content)
    while start < length:
        limit = min(start + max_chars, length)
        if limit == length:
            end = length
            hard = False
        else:
            index = bisect.bisect_right(points, limit) - 1
            boundary = points[index] if index >= 0 else 0
            if boundary <= start:
                end = limit
                hard = True
                hard_splits += 1
            else:
                end = boundary
                hard = False
        if end >= length:
            spans.append(SectionSpan(len(spans) + 1, start, length, 0, hard))
            break
        spans.append(SectionSpan(len(spans) + 1, start, end, 0, hard))
        # Next start: the earliest clean breakpoint that still provides the
        # configured overlap. If no breakpoint lies inside the requested
        # overlap window, snap to the closest earlier boundary instead of
        # splitting mid-sentence; a zero overlap is always respected.
        target = max(end - overlap_chars, 0)
        window_start = bisect.bisect_left(points, target)
        window_end = bisect.bisect_left(points, end)
        next_start = end
        if overlap_chars > 0:
            if window_start < window_end:
                next_start = points[window_start]
            elif window_end > 0:
                candidate = points[window_end - 1]
                if start < candidate < end:
                    next_start = candidate
        if next_start <= start:
            next_start = end
        start = next_start
    # Record the actual overlap of each section with its predecessor.
    with_overlap: list[SectionSpan] = []
    previous_end: int | None = None
    for span in spans:
        overlap = 0
        if previous_end is not None and span.start < previous_end:
            overlap = previous_end - span.start
        with_overlap.append(
            SectionSpan(span.number, span.start, span.end, overlap, span.hard_split)
        )
        previous_end = span.end
    return with_overlap, hard_splits


def render_document_record(
    document: DocumentRecord, config: ConverterConfig
) -> tuple[list[dict[str, object]], int]:
    """Render one document as one or more textual records.

    Args:
        document: The document to textualize.
        config: Converter configuration with chunking parameters.

    Returns:
        A tuple of record dictionaries (with ``record_id``,
        ``record_type``, ``source_id``, ``text`` and ``metadata``) and the
        number of hard splits that were necessary.
    """
    if config.chunk_documents:
        spans, hard_splits = compute_section_spans(
            document.content,
            config.max_section_chars,
            config.section_overlap_chars,
        )
    elif document.content:
        # Default: no source-side chunking. HiRAG applies its own internal
        # chunking during insertion, so each document stays one record.
        spans = [SectionSpan(1, 0, len(document.content), 0, False)]
        hard_splits = 0
    else:
        spans = []
        hard_splits = 0
    if not spans:
        return [], 0
    total = len(spans)
    chunked = total > 1
    records: list[dict[str, object]] = []
    for span in spans:
        section_content = document.content[span.start : span.end]
        # The record text carries only the document name and its content;
        # IDs, file type and section bookkeeping live in the JSONL
        # metadata, not in the text.
        if chunked:
            section_id = f"doc_{document.doc_id}_section_{span.number:04d}"
            record_type = "DOCUMENT_SECTION"
            header = [
                f"DOCUMENT_TITLE: {document.title}",
                f"SECTION: {span.number}/{total}",
            ]
            record_id = f"document_{section_id}"
        else:
            record_type = "DOCUMENT"
            header = [f"DOCUMENT_TITLE: {document.title}"]
            record_id = f"document_doc_{document.doc_id}"
        text = "\n".join(header) + "\n\nDOCUMENT_CONTENT:\n" + section_content
        if not text.endswith("\n"):
            text += "\n"
        metadata: dict[str, object] = {
            "source_file": document.source_file,
            "source_line": document.line_number,
            "document_id": document.doc_id,
            "document_object_id": document.object_id,
            "linked_process": "unknown",
            "relationship_provenance": "unavailable",
        }
        if chunked:
            metadata["section_number"] = span.number
            metadata["total_sections"] = total
            metadata["section_id"] = f"doc_{document.doc_id}_section_{span.number:04d}"
            metadata["content_start"] = span.start
            metadata["content_end"] = span.end
            metadata["overlap_with_previous_chars"] = span.overlap_with_previous
        records.append(
            {
                "record_id": record_id,
                "record_type": record_type,
                "source_id": document.doc_id,
                "text": text,
                "metadata": metadata,
            }
        )
    return records, hard_splits
