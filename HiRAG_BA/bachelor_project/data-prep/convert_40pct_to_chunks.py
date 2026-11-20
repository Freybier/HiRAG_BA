#!/usr/bin/env python3
"""Convert the ADONIS 40% sample exports into a ``chunks.jsonl`` corpus.

Rebuilds a retrieval corpus in the exact row shape used by
``src/test-data/small_anon_Dataset/chunks.jsonl`` (one JSON object per line
with ``id``, ``text``, ``metadata_``, ``node_id``) from the 40% sample
exports in ``src/test-data/test-data_40pct/``:

- ``models_40pct.jsonl``              — 567 model rows (name / id lookup)
- ``hierarchy_40pct.jsonl``           — 823 relations; ``C_SUB_PROCESS`` edges
  define the structural parent, depth and path of each child model
- ``description_hierarchy-addon_embedded.json`` — 7,819 pre-chunked rows
  converted by ``csv_to_json.py`` from
  ``data_vector_store3_40pct_modified.csv`` (id, node_id, model_id,
  object_type, text, metadata_ as a JSON *string*, embedding as a pgvector
  string)

Fixes applied during conversion
-------------------------------
1. ``metadata_`` is parsed from its JSON-string form into a nested object, so
   the output matches the ``small_anon_Dataset/chunks.jsonl`` layout exactly.
2. Hierarchy metadata (``hierarchy_depth``, ``hierarchy_path``,
   ``parent_model_id``, ``parent_model_name``, ``ancestor_summary``) is empty
   in the source export because those chunks were written before hierarchy
   metadata existed.  It is patched from ``hierarchy_40pct.jsonl``:
   ``C_SUB_PROCESS`` rows supply parent/depth/path; ``C_CROSS_REFERENCE``
   rows only fill depth/path for models that have no sub-process edge inside
   the sample (boundary relations were excluded during sampling, so some true
   sub-processes appear as roots — this mirrors the induced-subgraph rule in
   ``manifest.json``).
3. The mangled ``model_id`` artifact (``model_{uuid`` without the closing
   brace) inside ``metadata_`` and ``node_id`` is normalized to
   ``model_{uuid}``; joins always use the clean top-level ``model_id`` column.
4. ``embedding`` is parsed from the pgvector string into a JSON array of
   floats (``null`` when a row has no embedding).  No embedding calls are
   made.  When every row carries an embedding of one consistent dimension
   and NumPy is installed, the vectors are additionally written to
   ``chunk_embeddings.npy`` next to the output (float32 matrix aligned to
   row order) — the exact cache file ``app.test_data_search`` loads for
   offline vector search.

Known source-data caveats (passed through, not repaired)
--------------------------------------------------------
- ``[unrooted]/<name>`` path strings in ``hierarchy_40pct.jsonl`` mean the
  parent chain was unresolvable at export time.  The parent *link* is kept,
  but the path/ancestor fields are blanked.
- Some hierarchy rows are internally stale: their ``hierarchy_path`` root
  segment names a different model than their ``parent_model_*`` columns
  (e.g. after a model was re-parented).  The converter keeps both verbatim —
  retrieval-relevant linkage always uses the id columns, which are intact.

The result is a drop-in corpus for ``app.test_data_search``
(``ADONIS_TEST_DATA_SEARCH=true``); keyword mode and — via the generated
``chunk_embeddings.npy`` — vector mode both work fully offline.

Usage::

    cd /workspace/workdir/src/test-data
    ../.venv/bin/python data-prep/convert_40pct_to_chunks.py
    # or with explicit paths:
    ../.venv/bin/python data-prep/convert_40pct_to_chunks.py \
        --input-dir test-data_40pct --output test-data_40pct/chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

#: Default location of the 40% sample exports (one level up from data-prep/).
DEFAULT_INPUT_DIR = Path(__file__).resolve().parents[1] / "test-data_40pct"

#: Name of the addon chunk export including embeddings (produced by
#: ``csv_to_json.py`` from ``data_vector_store3_40pct_modified.csv``).
ADDON_FILENAME = "description_hierarchy-addon_embedded.json"

#: Relation type that defines the structural parent/child tree.
STRUCTURAL_RELATION_TYPE = "C_SUB_PROCESS"

#: Relation type for plain cross-references (used only as depth/path fallback).
CROSS_REFERENCE_RELATION_TYPE = "C_CROSS_REFERENCE"


# Normalize an ADONIS model identifier to a bare lowercase UUID.
def normalize_model_key(raw: Any) -> str:
    """Normalize an ADONIS model identifier to a bare lowercase UUID.

    Handles braced (``{uuid}``), bare and ``model_``-prefixed forms, including
    the mangled ``model_{uuid`` artifact from the addon export.

    Args:
        raw: Raw identifier value from any of the three export files.

    Returns:
        Bare lowercase UUID string without braces or prefix.
    """
    return str(raw or "").removeprefix("model_").strip("{}").lower()


# Load the models export into a bare-UUID -> model-name lookup.
def load_models(path: Path) -> dict[str, str]:
    """Load the models export into a bare-UUID -> model-name lookup.

    Args:
        path: Location of ``models_40pct.jsonl``.

    Returns:
        Mapping of normalized model key to the human-readable model name.
    """
    models: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            models[normalize_model_key(row.get("model_id"))] = str(row.get("name", ""))
    logger.info(f"Loaded {len(models)} models from {path}")
    return models


# Build per-model structural hierarchy info from the relations export.
def load_hierarchy(path: Path) -> dict[str, dict[str, Any]]:
    """Build per-model structural hierarchy info from the relations export.

    ``C_SUB_PROCESS`` rows define the canonical parent, depth and path of a
    child model.  If a model has no sub-process edge inside the sample (its
    parent was excluded as a boundary relation), the first
    ``C_CROSS_REFERENCE`` row supplies depth/path only — its
    ``parent_model_id`` is the *referencing* model, not a structural parent,
    and is therefore not adopted.

    Rows are processed in ascending ``id`` order so the result is
    deterministic; the first structural row per child model wins.

    Args:
        path: Location of ``hierarchy_40pct.jsonl``.

    Returns:
        Mapping of normalized child-model key to a dict with ``parent_id``,
        ``parent_name``, ``depth`` and ``path`` (parent fields empty when
        only cross-reference information exists).
    """
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows.sort(key=lambda row: int(row.get("id", 0)))

    hierarchy: dict[str, dict[str, Any]] = {}
    unrooted_rows = 0
    for row in rows:
        child_key = normalize_model_key(row.get("child_model_id"))
        if not child_key:
            continue
        # "[unrooted]" paths mean the parent chain was unresolvable when the
        # hierarchy row was recorded; the parent *link* is still real, but
        # the path string is unusable and would pollute breadcrumbs.
        relation_path = str(row.get("hierarchy_path", ""))
        if relation_path.startswith("[unrooted]"):
            relation_path = ""
            unrooted_rows += 1
        relation_type = str(row.get("relation_type", ""))
        if relation_type == STRUCTURAL_RELATION_TYPE:
            if child_key in hierarchy and hierarchy[child_key]["parent_id"]:
                continue  # first structural row wins
            hierarchy[child_key] = {
                "parent_id": normalize_model_key(row.get("parent_model_id")),
                "parent_name": str(row.get("parent_model_name", "")),
                "depth": str(row.get("depth", "0")),
                "path": relation_path,
            }
        elif relation_type == CROSS_REFERENCE_RELATION_TYPE:
            if child_key in hierarchy:
                continue  # structural info already present
            hierarchy[child_key] = {
                "parent_id": "",
                "parent_name": "",
                "depth": str(row.get("depth", "0")),
                "path": relation_path,
            }
    logger.info(
        f"Loaded hierarchy info for {len(hierarchy)} child models from {path} "
        f"({unrooted_rows} rows had an unusable '[unrooted]' path)"
    )
    return hierarchy


# Parse the addon metadata_ field into a mutable dict.
def parse_metadata(raw_metadata: Any) -> dict[str, Any]:
    """Parse the addon ``metadata_`` field into a mutable dict.

    Args:
        raw_metadata: ``metadata_`` value from an addon row (JSON string or
            already-parsed dict).

    Returns:
        Parsed metadata dict; empty dict on parse failure.
    """
    if isinstance(raw_metadata, dict):
        return dict(raw_metadata)
    if isinstance(raw_metadata, str):
        try:
            parsed = json.loads(raw_metadata)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            logger.warning("Skipping row with unparseable metadata_")
    return {}


# Re-insert the closing brace missing from mangled node_id values.
def fix_node_id(node_id: Any) -> str:
    """Re-insert the closing brace missing from mangled ``node_id`` values.

    The addon export carries ``node_id`` values such as
    ``model_{uuid_overview`` (closing brace lost).  This restores the
    ``model_{uuid}_overview`` form used by the production vector store.

    Args:
        node_id: Raw ``node_id`` value from an addon row.

    Returns:
        Normalized ``node_id`` string.
    """
    value = str(node_id or "")
    if value.startswith("model_{") and "}" not in value:
        # Insert the missing brace directly after the 36-char UUID.
        brace_pos = len("model_{") + 36
        if len(value) >= brace_pos:
            return f"{value[:brace_pos]}}}{value[brace_pos:]}"
    return value


# Parse a pgvector embedding string into a list of floats.
def parse_embedding(raw: Any) -> list[float] | None:
    """Parse a pgvector embedding string into a list of floats.

    Args:
        raw: Raw ``embedding`` value from an addon row (pgvector string such
            as ``[-0.01,0.02,...]``, an already-parsed list, or missing).

    Returns:
        List of floats, or ``None`` when the row carries no embedding.

    Raises:
        ValueError: If the string is not a valid bracketed float list.
    """
    if raw is None or raw == "":
        return None
    if isinstance(raw, list):
        return [float(component) for component in raw]
    value = str(raw).strip()
    if not (value.startswith("[") and value.endswith("]")):
        raise ValueError(f"Embedding is not a bracketed list: {value[:40]!r}...")
    return [float(component) for component in value[1:-1].split(",")]


# Join addon chunks with model names and hierarchy metadata, yielding rows with parsed
# embeddings.
def build_chunk_rows(
    addon_path: Path,
    models: dict[str, str],
    hierarchy: dict[str, dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    """Yield one chunks.jsonl-shaped row per addon chunk.

    Args:
        addon_path: Location of the embedded addon file (JSON array of chunk
            rows with an ``embedding`` pgvector string).
        models: Bare-UUID -> model-name lookup from the models export.
        hierarchy: Bare-UUID -> hierarchy info from the relations export.

    Yields:
        Dicts with keys ``id``, ``text``, ``metadata_`` (nested dict),
        ``node_id`` and ``embedding`` (list of floats, ``None`` when the
        source row carries no embedding).
    """
    with addon_path.open(encoding="utf-8") as handle:
        addon_rows: list[dict[str, Any]] = json.load(handle)

    for row in addon_rows:
        metadata = parse_metadata(row.get("metadata_"))
        # Always join on the clean top-level model_id column (the metadata_
        # copy is mangled in the export).
        model_key = normalize_model_key(row.get("model_id"))
        info = hierarchy.get(model_key)

        depth = info["depth"] if info else "0"
        path = info["path"] if info else ""
        parent_id = info["parent_id"] if info else ""
        parent_name = info["parent_name"] if info else ""

        metadata["model_id"] = f"model_{{{model_key}}}"
        metadata["hierarchy_depth"] = depth
        metadata["hierarchy_path"] = path
        metadata["parent_model_id"] = parent_id
        metadata["parent_model_name"] = parent_name
        # Observed production value: ancestor chain without the model itself
        # (for depth 1 this equals the parent name).  Model names may contain
        # "/" themselves, so strip the trailing "/<name>" instead of splitting
        # on "/".
        model_name = str(metadata.get("model_name", ""))
        if path and model_name and path.endswith(f"/{model_name}"):
            metadata["ancestor_summary"] = path[: -(len(model_name) + 1)]
        elif path:
            metadata["ancestor_summary"] = parent_name
        else:
            metadata["ancestor_summary"] = ""

        text = str(row.get("text", ""))
        embedding = parse_embedding(row.get("embedding"))

        yield {
            "id": row.get("id"),
            "text": text,
            "metadata_": metadata,
            "node_id": fix_node_id(row.get("node_id")),
            "embedding": embedding,
        }


@dataclass
class WriteStats:
    """Aggregate result of writing ``chunks.jsonl``.

    Attributes:
        rows_written: Number of chunk rows written.
        rows_with_parent: Rows carrying a structural parent link.
        covered_models: Distinct normalized model keys seen in the rows.
        embeddings: Per-row embedding vectors in write order (``None`` where
            the source row had no embedding).
    """

    rows_written: int = 0
    rows_with_parent: int = 0
    covered_models: set[str] = field(default_factory=set)
    embeddings: list[list[float] | None] = field(default_factory=list)


# Write chunk rows as JSON Lines and collect coverage stats in-pass.
def write_chunks_jsonl(rows: Iterator[dict[str, Any]], output_path: Path) -> WriteStats:
    """Write chunk rows as JSON Lines and collect coverage stats in-pass.

    Coverage is collected during the write pass so the output file (hundreds
    of MB with inline embeddings) never has to be re-parsed for the summary.

    Args:
        rows: Iterable of chunk row dicts.
        output_path: Destination file (overwritten).

    Returns:
        Write statistics including the per-row embedding vectors.
    """
    stats = WriteStats()
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats.embeddings.append(row["embedding"])
            metadata = row["metadata_"]
            stats.covered_models.add(normalize_model_key(metadata.get("model_id")))
            if metadata.get("parent_model_id"):
                stats.rows_with_parent += 1
            stats.rows_written += 1
    return stats


# Write the chunk embedding matrix cache for offline vector search.
def write_embedding_cache(embeddings: list[list[float] | None], npy_path: Path) -> bool:
    """Write the chunk embedding matrix cache for offline vector search.

    The file layout matches what
    ``app.test_data_search.build_chunk_embeddings`` produces: a NumPy float32
    array of shape ``(num_chunks, embed_dim)`` aligned to the
    ``chunks.jsonl`` row order.  The cache is only written when *every* row
    carries an embedding of one consistent dimension — a partial or ragged
    matrix would silently misalign similarity search.

    Args:
        embeddings: Per-row embedding vectors (``None`` where missing).
        npy_path: Destination ``chunk_embeddings.npy`` path.

    Returns:
        ``True`` when the cache was written, ``False`` when skipped.
    """
    total = len(embeddings)
    complete = [vector for vector in embeddings if vector is not None]
    if len(complete) != total:
        logger.warning(
            f"Skipping {npy_path.name}: {total - len(complete)}/{total} rows "
            "have no embedding; a partial cache would be misaligned."
        )
        # A stale cache from a previous run would silently misalign vector
        # search with the new chunks.jsonl — remove it.
        if npy_path.exists():
            npy_path.unlink()
            logger.warning(f"Removed stale embedding cache: {npy_path}")
        return False
    dimensions = {len(vector) for vector in complete}
    if len(dimensions) != 1:
        logger.warning(
            f"Skipping {npy_path.name}: inconsistent embedding dimensions "
            f"{sorted(dimensions)}."
        )
        return False
    try:
        import numpy as np
    except ImportError:
        logger.warning(
            f"Skipping {npy_path.name}: NumPy is not installed (embeddings "
            "are still stored inline in chunks.jsonl)."
        )
        return False
    matrix = np.asarray(complete, dtype=np.float32)
    np.save(npy_path, matrix)
    logger.info(
        f"Wrote embedding cache {npy_path} "
        f"({matrix.shape[0]}x{matrix.shape[1]} float32)"
    )
    return True


# Parse export paths, write the joined chunk corpus and optional embedding cache, and print
# coverage statistics.
def main() -> int:
    """Run the conversion and print a coverage summary.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert the ADONIS 40% sample (models + hierarchy + embedded "
            "description_hierarchy-addon) into a chunks.jsonl corpus."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing the three 40% export files (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output chunks.jsonl path (default: <input-dir>/chunks.jsonl)",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_path: Path = (
        args.output if args.output is not None else input_dir / "chunks.jsonl"
    )

    models_path = input_dir / "models_40pct.jsonl"
    hierarchy_path = input_dir / "hierarchy_40pct.jsonl"
    addon_path = input_dir / ADDON_FILENAME
    for required in (models_path, hierarchy_path, addon_path):
        if not required.exists():
            logger.error(f"Required input file missing: {required}")
            return 1

    models = load_models(models_path)
    hierarchy = load_hierarchy(hierarchy_path)
    stats = write_chunks_jsonl(
        build_chunk_rows(addon_path, models, hierarchy),
        output_path,
    )
    write_embedding_cache(stats.embeddings, output_path.parent / "chunk_embeddings.npy")

    orphan_models = {key for key in stats.covered_models if key not in models}
    rows_with_embedding = sum(1 for vector in stats.embeddings if vector is not None)

    logger.info(f"Wrote {stats.rows_written} chunk rows to {output_path}")
    logger.info(
        f"Rows with structural parent: "
        f"{stats.rows_with_parent}/{stats.rows_written}; "
        f"rows with embedding: {rows_with_embedding}/{stats.rows_written}; "
        f"distinct models covered: {len(stats.covered_models)}; "
        f"models not present in models_40pct.jsonl: {len(orphan_models)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
