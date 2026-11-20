"""Persistent mapping from HiRAG-internal document IDs to original source IDs.

HiRAG derives the internal ``full_doc_id`` of every inserted document from its
normalized content (``compute_mdhash_id(content.strip(), prefix="doc-")`` in
``hirag/hirag.py``). That hash says nothing about which input record the
content came from, and identical content from different input records shares
one internal ID. This module persists that relationship as a JSON file named
``source_id_mapping.json`` inside HiRAG's working directory::

    {
        "doc-<md5-of-stripped-content>": ["original-source-ID", ...]
    }

The mapping is additive across runs: existing entries are preserved, a source
ID is never duplicated under one internal ID, and multiple original IDs are
retained when identical content maps to one internal ID. Neither the internal
document IDs nor the chunk IDs (``chunk-`` hashes carrying the ``full_doc_id``
field) are affected; only this sidecar file is written.

The vendored ``hirag`` package must be importable when using this module
(same ``sys.path`` setup as ``hi_insert_run.py``), so the ID computation is
guaranteed to stay identical to the one HiRAG performs internally.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from hirag._utils import compute_mdhash_id

MAPPING_FILENAME = "source_id_mapping.json"


def compute_full_doc_id(content: str) -> str:
    """Compute HiRAG's internal ``full_doc_id`` for a document content.

    Replicates the exact normalization and ID generation of
    ``HiRAG.ainsert``: the content is stripped and then hashed with the
    ``doc-`` prefix.

    Args:
        content: Raw document text as passed to ``HiRAG.insert``.

    Returns:
        The internal document ID, e.g. ``doc-<md5 hex>``.
    """
    return compute_mdhash_id(content.strip(), prefix="doc-")


def mapping_path(working_dir: str | Path) -> Path:
    """Return the mapping file path inside a HiRAG working directory.

    Args:
        working_dir: HiRAG working directory (index directory).

    Returns:
        Path of the ``source_id_mapping.json`` file.
    """
    return Path(working_dir) / MAPPING_FILENAME


def load_source_id_mapping(working_dir: str | Path) -> dict[str, list[str]]:
    """Load the persisted mapping from a HiRAG working directory.

    Args:
        working_dir: HiRAG working directory (index directory).

    Returns:
        The mapping ``{full_doc_id: [source_id, ...]}``; an empty dict when
        no mapping file exists yet.

    Raises:
        ValueError: If the file exists but does not have the expected
            ``{str: list[str]}`` structure.
    """
    path = mapping_path(working_dir)
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as mapping_file:
        raw = json.load(mapping_file)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object, got {type(raw).__name__}")
    mapping: dict[str, list[str]] = {}
    for doc_id, source_ids in raw.items():
        if not isinstance(source_ids, list):
            raise ValueError(
                f"{path}: entry {doc_id!r} must be a list, "
                f"got {type(source_ids).__name__}"
            )
        mapping[str(doc_id)] = [str(source_id) for source_id in source_ids]
    return mapping


def save_source_id_mapping(
    working_dir: str | Path, mapping: Mapping[str, list[str]]
) -> Path:
    """Write the mapping atomically (temp file in place + rename).

    Args:
        working_dir: HiRAG working directory (index directory).
        mapping: The full mapping to persist.

    Returns:
        The path the mapping was written to.
    """
    path = mapping_path(working_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f"{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            json.dump(mapping, tmp_file, indent=2, ensure_ascii=False)
            tmp_file.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return path


def update_source_id_mapping(
    working_dir: str | Path,
    entries: Iterable[tuple[str, str]],
) -> tuple[dict[str, list[str]], int]:
    """Merge new ``(full_doc_id, source_id)`` pairs into the persisted mapping.

    Existing entries are preserved, a source ID already linked to an internal
    ID is not duplicated, and identical content coming from several sources
    accumulates several original IDs under the same internal ID.

    Args:
        working_dir: HiRAG working directory (index directory).
        entries: Pairs of internal document ID and original source ID.

    Returns:
        A tuple of the merged mapping and the number of newly added links.
    """
    mapping = load_source_id_mapping(working_dir)
    new_links = 0
    for doc_id, source_id in entries:
        known = mapping.setdefault(str(doc_id), [])
        source_id = str(source_id)
        if source_id not in known:
            known.append(source_id)
            new_links += 1
    save_source_id_mapping(working_dir, mapping)
    return mapping, new_links


def resolve_source_ids(
    reference: str | Mapping[str, Any], working_dir: str | Path
) -> list[str]:
    """Resolve an internal ``full_doc_id`` to its original source IDs.

    Args:
        reference: Either a ``full_doc_id`` string (``doc-...``) or a chunk
            mapping that carries the ``full_doc_id`` field (as stored in
            HiRAG's ``text_chunks`` storage).
        working_dir: HiRAG working directory (index directory).

    Returns:
        A copy of the original source IDs linked to the internal ID; an empty
        list when the ID is unknown (or the chunk has no ``full_doc_id``).

    Raises:
        TypeError: If ``reference`` is neither a string nor a chunk mapping.
    """
    if isinstance(reference, str):
        doc_id = reference
    elif isinstance(reference, Mapping):
        doc_id = str(reference.get("full_doc_id", ""))
    else:
        raise TypeError(
            "reference must be a full_doc_id string or a chunk mapping, "
            f"got {type(reference).__name__}"
        )
    return list(load_source_id_mapping(working_dir).get(doc_id, []))
