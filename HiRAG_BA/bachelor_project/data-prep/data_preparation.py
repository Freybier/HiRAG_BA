#!/usr/bin/env python3
"""One-shot data preparation pipeline for the ADONIS 40% sample.

Runs the full offline preparation chain against the hardcoded data
directory ``src/test-data/test-data_40pct/``:

1. CSV -> JSON: ``data_vector_store3_40pct_modified.csv`` (the chunk export
   that additionally carries embeddings) is converted to
   ``description_hierarchy-addon_embedded.json`` via ``csv_to_json.py``.
2. Document chunks: ``documents_40pct.jsonl`` is split with the production
   4,000/200 character window and embedded via the gateway into
   ``document_chunks.jsonl`` via ``document_chunks.py``.  This is the ONLY
   non-offline step: it needs the ``src/.venv`` interpreter and a gateway
   API key (loaded from ``src/app/.env-dev``).
3. chunks.jsonl: the embedded addon JSON is joined with
   ``models_40pct.jsonl`` and ``hierarchy_40pct.jsonl`` via
   ``convert_40pct_to_chunks.py`` into ``chunks.jsonl`` (inline float
   embeddings), appended with the document chunk rows when
   ``document_chunks.jsonl`` exists.
4. Texturization: ``models_40pct.jsonl``, ``hierarchy_40pct.jsonl`` and
   ``documents_40pct.jsonl`` are converted by the ``adonis_converter``
   package into ``hirag_parent_only_output/`` (parent-only hierarchy mode,
   combined-text export, process descriptions taken from the embedded addon
   JSON).

Each step is guarded by an existence check on its output, so re-running the
script skips work that is already done:

- ``document_chunks.jsonl`` present   -> step 2 is skipped
- ``chunks.jsonl`` present            -> steps 1+3 are skipped
- ``processes_text.txt`` AND ``documents_text.txt`` present
                                      -> step 4 is skipped

Delete the corresponding output file(s) to force a rebuild of a step
(deleting ``chunks.jsonl`` rebuilds steps 1+3; a stale
``chunk_embeddings.npy`` is removed automatically when the cache cannot
cover the rebuilt corpus).
All paths are hardcoded; the only CLI argument is ``--mode``, selecting
the texturization variant of step 4:

- ``hierarchy_only`` (default): name + description + relationships only,
  written to ``hirag_parent_only_output/``.
- ``activities``: additionally appends each process's activity steps
  (the ``activity`` chunks from ``chunks.jsonl``, ordered by
  ``step_number``) as an ``ACTIVITIES`` section after the
  subprocess/cross-reference sentences, written to
  ``hirag_parent_only_output_with_activities/``.
- ``everything``: like ``activities``, plus the ``lane``, ``compliance``
  and ``other`` chunks from ``chunks.jsonl`` as ``LANES``,
  ``COMPLIANCE`` and ``RELATED_OBJECTS`` sections after ``ACTIVITIES``,
  written to ``hirag_parent_only_output_with_everything/``.

Each variant directory is guarded by its own combined text files, so all
three corpus variants coexist. NOTE: the non-default modes change the
record texts and therefore HiRAG's content hashes — re-index into a
FRESH working dir.

Usage::

    cd /workspace/workdir/src/test-data
    ../.venv/bin/python data-prep/data_preparation.py [--mode {hierarchy_only,activities,everything}]
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Make the sibling scripts and the adonis_converter package importable no
# matter which working directory the script is started from.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from adonis_converter.config import (  # noqa: E402
    SUPPORTED_EXTRA_CHUNK_TYPES,
    ConverterConfig,
)
from adonis_converter.converter import run_conversion  # noqa: E402
from convert_40pct_to_chunks import (  # noqa: E402
    build_chunk_rows,
    load_hierarchy,
    load_models,
    write_chunks_jsonl,
    write_embedding_cache,
)
from csv_to_json import convert_csv_to_json  # noqa: E402
from document_chunks import (  # noqa: E402
    read_document_chunk_rows,
    run_document_chunk_builder,
)

# ---------------------------------------------------------------------------
# Hardcoded paths (data directory layout of the ADONIS 40% sample)
# ---------------------------------------------------------------------------

#: Data directory holding the input exports and all generated artifacts
#: (_SCRIPT_DIR is already a directory, so .parent is src/test-data/).
DATA_DIR = _SCRIPT_DIR.parent / "test-data_40pct"

# --- Inputs ---
MODIFIED_CSV = DATA_DIR / "data_vector_store3_40pct_modified.csv"
MODELS_JSONL = DATA_DIR / "models_40pct.jsonl"
HIERARCHY_JSONL = DATA_DIR / "hierarchy_40pct.jsonl"
DOCUMENTS_JSONL = DATA_DIR / "documents_40pct.jsonl"

# --- Intermediate (regenerable from MODIFIED_CSV) ---
ADDON_EMBEDDED_JSON = DATA_DIR / "description_hierarchy-addon_embedded.json"

# --- Outputs checked by this pipeline ---
DOCUMENT_CHUNKS_JSONL = DATA_DIR / "document_chunks.jsonl"
CHUNKS_JSONL = DATA_DIR / "chunks.jsonl"
CHUNK_EMBEDDINGS_NPY = DATA_DIR / "chunk_embeddings.npy"
HIRAG_OUTPUT_DIR = DATA_DIR / "hirag_parent_only_output"
#: Separate output directories for the non-default texturization modes,
#: so the enriched corpora coexist with the default texturization
#: instead of overwriting it.
HIRAG_OUTPUT_DIR_EVERYTHING = DATA_DIR / "hirag_parent_only_output_with_everything"
HIRAG_OUTPUT_DIR_ACTIVITIES = DATA_DIR / "hirag_parent_only_output_with_activities"
PROCESSES_TEXT = HIRAG_OUTPUT_DIR / "processes_text.txt"
DOCUMENTS_TEXT = HIRAG_OUTPUT_DIR / "documents_text.txt"

#: Texturization mode: name + description + relationships only (default).
MODE_HIERARCHY_ONLY = "hierarchy_only"
#: Texturization mode: additionally append the ACTIVITIES section.
MODE_ACTIVITIES = "activities"
#: Texturization mode: additionally append ACTIVITIES plus LANES,
#: COMPLIANCE and RELATED_OBJECTS sections from the chunks corpus.
MODE_EVERYTHING = "everything"
#: All selectable ``--mode`` values.
TEXTURIZATION_MODES = (MODE_HIERARCHY_ONLY, MODE_ACTIVITIES, MODE_EVERYTHING)
#: Output directory per texturization mode.
MODE_OUTPUT_DIRS = {
    MODE_HIERARCHY_ONLY: HIRAG_OUTPUT_DIR,
    MODE_ACTIVITIES: HIRAG_OUTPUT_DIR_ACTIVITIES,
    MODE_EVERYTHING: HIRAG_OUTPUT_DIR_EVERYTHING,
}


# Convert the embedded chunk CSV to addon JSON when its cached output is missing.
def ensure_addon_json() -> None:
    """Generate the embedded addon JSON from the modified CSV if missing.

    Raises:
        FileNotFoundError: If the modified CSV input is missing.
    """
    if ADDON_EMBEDDED_JSON.exists():
        logger.info(f"Addon JSON already exists, keeping it: {ADDON_EMBEDDED_JSON}")
        return
    if not MODIFIED_CSV.exists():
        raise FileNotFoundError(f"Required input file missing: {MODIFIED_CSV}")
    logger.info(f"Converting {MODIFIED_CSV.name} -> {ADDON_EMBEDDED_JSON.name}")
    convert_csv_to_json(MODIFIED_CSV, ADDON_EMBEDDED_JSON)


# Split and embed exported documents through the gateway unless document chunks already exist.
def build_document_chunks() -> None:
    """Build embedded document chunks unless already present.

    Unlike every other step this one is NOT offline: it calls the Azure
    gateway to embed the chunks (key loaded from ``src/app/.env-dev``).

    Raises:
        FileNotFoundError: If the documents export is missing.
    """
    if DOCUMENT_CHUNKS_JSONL.exists():
        logger.info(
            f"document_chunks.jsonl already exists, skipping: {DOCUMENT_CHUNKS_JSONL}"
        )
        return
    if not DOCUMENTS_JSONL.exists():
        raise FileNotFoundError(f"Required input file missing: {DOCUMENTS_JSONL}")
    stats = run_document_chunk_builder(
        documents_path=DOCUMENTS_JSONL,
        output_path=DOCUMENT_CHUNKS_JSONL,
    )
    logger.info(
        f"Document chunks done: {stats['chunks_written']} chunks from "
        f"{stats['documents_kept']} documents "
        f"({stats['embedding_tokens']} embedding tokens, "
        f"{stats['documents_skipped_type']} excluded by file type)"
    )


# Join chunk exports with model and hierarchy metadata, then write the corpus and embedding
# cache.
def build_chunks() -> None:
    """Build chunks.jsonl (+ chunk_embeddings.npy) unless already present.

    Raises:
        FileNotFoundError: If a required input file is missing.
    """
    if CHUNKS_JSONL.exists():
        logger.info(f"chunks.jsonl already exists, skipping: {CHUNKS_JSONL}")
        return
    ensure_addon_json()
    for required in (MODELS_JSONL, HIERARCHY_JSONL):
        if not required.exists():
            raise FileNotFoundError(f"Required input file missing: {required}")
    logger.info(f"Building chunks corpus -> {CHUNKS_JSONL}")
    models = load_models(MODELS_JSONL)
    hierarchy = load_hierarchy(HIERARCHY_JSONL)
    rows: Iterator[dict[str, Any]] = build_chunk_rows(
        ADDON_EMBEDDED_JSON, models, hierarchy
    )
    document_chunk_count = 0
    if DOCUMENT_CHUNKS_JSONL.exists():
        # Document chunk rows carry pre-computed inline embeddings and are
        # appended after the model chunks; write_embedding_cache() then
        # covers them as well.
        document_rows = list(read_document_chunk_rows(DOCUMENT_CHUNKS_JSONL))
        document_chunk_count = len(document_rows)
        rows = itertools.chain(rows, iter(document_rows))
        logger.info(
            f"Appending {document_chunk_count} document chunks "
            f"from {DOCUMENT_CHUNKS_JSONL.name}"
        )
    else:
        logger.info(
            f"No {DOCUMENT_CHUNKS_JSONL.name} found — chunks.jsonl will "
            "contain model chunks only."
        )
    stats = write_chunks_jsonl(rows, CHUNKS_JSONL)
    write_embedding_cache(stats.embeddings, CHUNK_EMBEDDINGS_NPY)
    rows_with_embedding = sum(1 for vector in stats.embeddings if vector is not None)
    # Document chunks carry no model_id (empty normalized key) — exclude the
    # placeholder from the coverage/orphan stats.
    covered_models = {key for key in stats.covered_models if key}
    orphan_models = {key for key in covered_models if key not in models}
    logger.info(
        f"chunks.jsonl done: {stats.rows_written} rows "
        f"({stats.rows_with_parent} with structural parent, "
        f"{document_chunk_count} document chunks, "
        f"{rows_with_embedding} with embedding, "
        f"{len(covered_models)} distinct models, "
        f"{len(orphan_models)} not in the models export)"
    )


# Convert the selected corpus variant to parent-only process and document texts unless already
# built.
def build_texturization(mode: str = MODE_HIERARCHY_ONLY) -> None:
    """Build the HiRAG parent-only text corpus unless already present.

    The check covers both combined text files; if either one is missing the
    full texturization re-runs (it always writes both).

    Args:
        mode: Texturization variant (one of ``TEXTURIZATION_MODES``).
            ``activities`` appends each process's activity steps from
            ``chunks.jsonl`` as an ``ACTIVITIES`` section after the
            relationship sections; ``everything`` additionally appends
            the lane/compliance/other chunks as ``LANES``, ``COMPLIANCE``
            and ``RELATED_OBJECTS`` sections. Non-default modes write
            into their own output directories (see ``MODE_OUTPUT_DIRS``)
            and require ``chunks.jsonl`` (built by :func:`build_chunks`,
            which always runs before this step).

    Raises:
        FileNotFoundError: If a required input file is missing.
    """
    output_dir = MODE_OUTPUT_DIRS[mode]
    processes_text = output_dir / "processes_text.txt"
    documents_text = output_dir / "documents_text.txt"
    if processes_text.exists() and documents_text.exists():
        logger.info(
            "Texturization outputs already exist, skipping: "
            f"{processes_text} and {documents_text}"
        )
        return
    for required in (MODELS_JSONL, HIERARCHY_JSONL, DOCUMENTS_JSONL):
        if not required.exists():
            raise FileNotFoundError(f"Required input file missing: {required}")
    needs_chunks = mode in (MODE_ACTIVITIES, MODE_EVERYTHING)
    if needs_chunks and not CHUNKS_JSONL.exists():
        raise FileNotFoundError(f"Required input file missing: {CHUNKS_JSONL}")
    # The addon's overview chunks supply the real process descriptions that
    # the models export lacks (its description column is the metaName).
    ensure_addon_json()
    logger.info(
        f"Running adonis_converter texturization -> {output_dir} (mode: {mode})"
    )
    report = run_conversion(
        ConverterConfig(
            models_path=MODELS_JSONL,
            hierarchy_path=HIERARCHY_JSONL,
            documents_path=DOCUMENTS_JSONL,
            output_dir=output_dir,
            export_combined_text=True,
            descriptions_path=ADDON_EMBEDDED_JSON,
            activities_path=CHUNKS_JSONL if needs_chunks else None,
            extra_chunk_types=(
                SUPPORTED_EXTRA_CHUNK_TYPES if mode == MODE_EVERYTHING else ()
            ),
            # xlsx documents are spreadsheet dumps whose cell text harms
            # HiRAG extraction more than it helps; skip them at the source.
            excluded_file_types=("xlsx",),
        )
    )
    validation = report["validation"]
    logger.info(
        "Texturization done: "
        f"{report['processes']['records']} process records "
        f"({report['processes']['with_activities']} with activities, "
        f"{report['processes']['with_lanes']} with lanes, "
        f"{report['processes']['with_compliance']} with compliance, "
        f"{report['processes']['with_related_objects']} with related objects), "
        f"{report['documents']['records']} document records, "
        f"{len(validation['warnings'])} warning(s)"
    )
    for warning in validation["warnings"]:
        logger.warning(warning)


# Parse the corpus variant and run the guarded CSV, document, chunk, and text preparation steps.
def main() -> int:
    """Run the guarded preparation pipeline.

    Returns:
        Process exit code (0 on success, 1 on missing inputs or failures).
    """
    parser = argparse.ArgumentParser(
        description="One-shot data preparation pipeline for the ADONIS 40% "
        "sample (see the module docstring for the steps and guards)."
    )
    parser.add_argument(
        "--mode",
        choices=TEXTURIZATION_MODES,
        default=MODE_HIERARCHY_ONLY,
        help=(
            "Texturization variant: 'hierarchy_only' (default) writes "
            "name + description + relationships only; 'activities' also "
            "appends the ACTIVITIES section from chunks.jsonl; "
            "'everything' additionally appends the lane/compliance/other "
            "chunks as LANES, COMPLIANCE and RELATED_OBJECTS sections. "
            "Each mode writes into its own output directory so all "
            "corpus variants coexist; the non-default modes change HiRAG "
            "content hashes, so re-index HiRAG into a fresh working dir "
            "when using them."
        ),
    )
    args = parser.parse_args()
    logger.info(f"Data directory: {DATA_DIR}")
    try:
        build_document_chunks()
        build_chunks()
        build_texturization(mode=args.mode)
    except FileNotFoundError as error:
        logger.error(str(error))
        return 1
    logger.info("Data preparation finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
