#!/usr/bin/env python3
"""Build embedded document chunks for the ADONIS 40% sample corpus.

The chunk exports (``data_vector_store3_40pct_modified.csv``) only contain
process-model chunks, so the local corpora (``chunks.jsonl``) currently have
no document partition at all.  This script fills the gap from
``documents_40pct.jsonl`` so the file-backed hierarchical RAG
(``app.test_data_search``) and the naive RAG
(``bachelor_project/naive_RAG/openAI_Test_chunked.py``) can retrieve
documents exactly like the production Postgres path does.

Production parity (mirrors ``scripts/reingest_docvector_store.py`` and
``scripts/reingest_docvector_with_fixed_extractor.py``):

- documents with ``len(content) <= 4000`` chars stay a single chunk
- larger documents are split with a character sliding window of
  4,000 chars and 200 chars overlap (pure character slicing, not
  sentence/token-aware — same as production)
- chunk metadata mirrors
  ``app.postgres.structured_chunker.make_document_chunk``:
  ``object_type="document"``, ``title="DOCUMENT_<name>"`` (plus a
  ``[chunk n/m]`` suffix when split), plain ``model_name`` (used for source
  deduplication), ``source_url``/``user_facing_url``, ``chunk_index`` /
  ``total_chunks`` and empty hierarchy fields

Two deliberate consistency decisions:

- ``xlsx`` documents are excluded (same as the HiRAG texturization in
  ``data_preparation.py``: spreadsheet cell dumps harm retrieval more than
  they help)
- embeddings are produced by the same ``adapfkt_ada_002`` gateway deployment
  as the existing inline corpus embeddings, keeping the 1536-dimension
  vectors compatible with the model chunks in ``chunks.jsonl``

Unlike the other data-prep scripts this step is NOT offline: it needs the
Azure gateway for embeddings and the ``src/.venv`` interpreter (``openai``,
``python-dotenv``, PyYAML via ``llm_config``).  The API key is loaded from
``src/app/.env-dev`` (CRLF-sanitized, values win over shell exports, same as
``HiRAG/hi_insert_run.py``).

Output: ``document_chunks.jsonl`` — one chunks.jsonl-shaped row per chunk
(``id``, ``text``, ``metadata_``, ``node_id``, inline ``embedding``).
``data_preparation.py`` appends these rows to ``chunks.jsonl`` when the file
exists.

Usage::

    cd /workspace/workdir/src/test-data
    ../.venv/bin/python data-prep/document_chunks.py
    ../.venv/bin/python data-prep/document_chunks.py --limit 5  # smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

#: Path layout: data-prep/document_chunks.py -> data-prep is parents[0],
#: src/test-data is parents[1], src is parents[2].
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parents[1]

#: Default locations inside the 40% sample data directory.
DEFAULT_DOCUMENTS = _SCRIPT_DIR.parent / "test-data_40pct" / "documents_40pct.jsonl"
DEFAULT_OUTPUT = _SCRIPT_DIR.parent / "test-data_40pct" / "document_chunks.jsonl"

#: Production splitting constants (scripts/reingest_docvector_store.py).
CHUNK_CHARS: int = 4_000
CHUNK_OVERLAP: int = 200

#: Title prefix that marks document chunks (production retriever filters and
#: ``app.test_data_search._is_document_chunk`` both rely on it).
DOCUMENT_TITLE_PREFIX: str = "DOCUMENT_"

#: File types excluded by default (HiRAG texturization parity: xlsx cell
#: dumps are noise for extraction and retrieval alike).
DEFAULT_EXCLUDED_FILE_TYPES: tuple[str, ...] = ("xlsx",)

#: Embedding request batching.  The gateway handles batches of short texts
#: well; retries with exponential backoff cover transient 429/5xx responses.
EMBED_BATCH_SIZE: int = 32
EMBED_MAX_RETRIES: int = 6


# Load src/app/.env-dev into the process environment.
def load_env_dev() -> None:
    """Load ``src/app/.env-dev`` into the process environment.

    The file historically has CRLF line endings, which can leave ``\r``
    artifacts in header values; each value is therefore sanitized. Values
    from ``.env-dev`` win over the shell environment so stale placeholder
    exports cannot shadow the real keys (same approach as
    ``HiRAG/hi_insert_run.py``).
    """
    env_path = _SRC_DIR / "app" / ".env-dev"
    if not env_path.is_file():
        return
    from dotenv import load_dotenv

    load_dotenv(env_path, override=True)
    for key, value in list(os.environ.items()):
        cleaned = value.strip().strip('"').strip("'").replace("\r", "")
        if cleaned != value:
            os.environ[key] = cleaned


# Build the Azure gateway embedding client used for corpus embeddings.
def _get_embedding_client() -> tuple[Any, str]:
    """Build the Azure gateway embedding client used for corpus embeddings.

    Uses the central bachelor-project config (``openai.base_url`` /
    ``openai.api_version``) and the ``models.embedding_model`` deployment
    (``adapfkt_ada_002``), exactly like the naive-RAG corpus path.

    Returns:
        Tuple of the ``AzureOpenAI`` client and the embedding deployment
        name.
    """
    load_env_dev()
    sys.path.insert(0, str(_SRC_DIR / "app" / "bachelor_project"))
    from llm_config import get_api_key, get_embedding_model, load_config
    from openai import AzureOpenAI

    bp_config = load_config()
    api_key = get_api_key("openai")
    client = AzureOpenAI(
        azure_endpoint=bp_config["openai"]["base_url"],
        api_key=api_key,
        api_version=bp_config["openai"]["api_version"],
        default_headers={
            "Content-Type": "application/json",
            "X-Gateway-Api-Key": api_key,
        },
    )
    return client, get_embedding_model("openai")


# Load the documents export.
def load_documents(path: Path) -> list[dict[str, Any]]:
    """Load the documents export.

    Args:
        path: Location of ``documents_40pct.jsonl``.

    Returns:
        List of document rows as parsed dicts.
    """
    documents: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    logger.info(f"Loaded {len(documents)} documents from {path}")
    return documents


# Split document content with the production character sliding window.
def split_document_content(
    content: str,
    chunk_chars: int = CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split document content with the production character sliding window.

    Args:
        content: Full document text.
        chunk_chars: Maximum characters per chunk (production: 4,000).
        overlap: Character overlap between consecutive chunks
            (production: 200).

    Returns:
        List of chunk texts; a single-element list when the content fits
        into one chunk.
    """
    if len(content) <= chunk_chars:
        return [content]
    chunks: list[str] = []
    start = 0
    while start < len(content):
        end = min(start + chunk_chars, len(content))
        chunks.append(content[start:end])
        start += chunk_chars - overlap
    return chunks


# Convert document export rows into chunks.jsonl-shaped chunk rows.
def build_document_chunk_rows(
    documents: list[dict[str, Any]],
    excluded_file_types: tuple[str, ...] = DEFAULT_EXCLUDED_FILE_TYPES,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Convert document export rows into chunks.jsonl-shaped chunk rows.

    Args:
        documents: Parsed rows from ``documents_40pct.jsonl``.
        excluded_file_types: Lowercase file extensions to skip (xlsx is
            excluded by default, consistent with the HiRAG texturization).

    Returns:
        Tuple of the chunk rows (embeddings still ``None``) and a stats dict
        with ``documents_total``, ``documents_skipped_type``,
        ``documents_skipped_empty``, ``documents_kept``, ``chunks_single``
        and ``chunks_split``.
    """
    excluded = {ext.lower().lstrip(".") for ext in excluded_file_types}
    stats = {
        "documents_total": len(documents),
        "documents_skipped_type": 0,
        "documents_skipped_empty": 0,
        "documents_kept": 0,
        "chunks_single": 0,
        "chunks_split": 0,
    }
    rows: list[dict[str, Any]] = []
    for document in documents:
        file_type = str(document.get("file_type") or "").lower().lstrip(".")
        if file_type in excluded:
            stats["documents_skipped_type"] += 1
            continue
        content = str(document.get("content") or "")
        if not content.strip():
            stats["documents_skipped_empty"] += 1
            continue

        stats["documents_kept"] += 1
        name = str(document.get("name") or "")
        doc_key = str(document.get("document_object_id") or document.get("id") or "")
        user_facing_url = str(document.get("user_facing_url") or "")
        summary = str(document.get("summary") or "").strip().upper() == "TRUE"
        try:
            num_tokens: int | None = int(document.get("num_tokens"))
        except (TypeError, ValueError):
            num_tokens = None

        chunk_texts = split_document_content(content)
        total = len(chunk_texts)
        if total == 1:
            stats["chunks_single"] += 1
        else:
            stats["chunks_split"] += 1

        for index, chunk_text in enumerate(chunk_texts):
            title = f"{DOCUMENT_TITLE_PREFIX}{name}"
            if total > 1:
                title = f"{title} [chunk {index + 1}/{total}]"
            rows.append(
                {
                    "id": (
                        f"doc_{doc_key}"
                        if total == 1
                        else f"doc_{doc_key}_chunk{index}"
                    ),
                    "text": chunk_text,
                    # Mirrors
                    # ``app.postgres.structured_chunker.make_document_chunk``.
                    "metadata_": {
                        "title": title,
                        # Plain document name: ``test_data_search`` prefers
                        # ``model_name`` for source deduplication, so all
                        # chunks of one document collapse into one source.
                        "model_name": name,
                        "object_type": "document",
                        "file_type": file_type,
                        "num_tokens": num_tokens,
                        "source_url": user_facing_url,
                        "user_facing_url": user_facing_url,
                        # Model-only concepts stay empty (as in production).
                        "status": "",
                        "risk_level": "",
                        "bcbs_relevant": "",
                        "lane_name": "",
                        "step_number": "",
                        "object_meta_name": "",
                        # No process<->document links exist in the export.
                        "parent_model_id": "",
                        "parent_model_name": "",
                        "hierarchy_depth": "0",
                        "hierarchy_path": "",
                        "ancestor_summary": "",
                        "type": "file_document",
                        "summary": summary,
                        "chunk_index": index,
                        "total_chunks": total,
                    },
                    "node_id": (
                        f"document_{doc_key}"
                        if total == 1
                        else f"document_{doc_key}_chunk{index}"
                    ),
                    "embedding": None,
                }
            )
    logger.info(
        f"Built {len(rows)} document chunk rows "
        f"({stats['documents_kept']} documents kept, "
        f"{stats['documents_skipped_type']} excluded by file type, "
        f"{stats['documents_skipped_empty']} without content, "
        f"{stats['chunks_split']} documents split)"
    )
    return rows, stats


# Embed every chunk row in place via the Azure gateway.
def embed_chunk_rows(
    rows: list[dict[str, Any]],
    client: Any,
    model: str,
    batch_size: int = EMBED_BATCH_SIZE,
) -> int:
    """Embed every chunk row in place via the Azure gateway.

    Args:
        rows: Chunk rows whose ``embedding`` field is filled in place.
        client: ``AzureOpenAI`` client from :func:`_get_embedding_client`.
        model: Embedding deployment name (``adapfkt_ada_002``).
        batch_size: Number of texts per ``embeddings.create`` call.

    Returns:
        Total prompt tokens billed by the gateway (0 when the gateway did
        not report usage).

    Raises:
        Exception: Re-raised when a batch keeps failing after
            ``EMBED_MAX_RETRIES`` attempts.
    """
    total_tokens = 0
    total_batches = (len(rows) + batch_size - 1) // batch_size
    for batch_number, start in enumerate(range(0, len(rows), batch_size)):
        batch = rows[start : start + batch_size]
        texts = [row["text"] for row in batch]
        for attempt in range(EMBED_MAX_RETRIES):
            try:
                response = client.embeddings.create(input=texts, model=model)
                vectors = [item.embedding for item in response.data]
                if len(vectors) != len(batch):
                    raise RuntimeError(
                        f"Embedding count mismatch: sent {len(batch)} texts, "
                        f"got {len(vectors)} vectors."
                    )
                for row, vector in zip(batch, vectors):
                    row["embedding"] = vector
                usage = getattr(response, "usage", None)
                total_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                break
            except Exception:
                if attempt >= EMBED_MAX_RETRIES - 1:
                    raise
                sleep_s = min(5 * (2**attempt), 60)
                logger.warning(
                    f"Embedding batch {batch_number + 1}/{total_batches} failed "
                    f"(attempt {attempt + 1}); retrying in {sleep_s}s."
                )
                time.sleep(sleep_s)
        if (batch_number + 1) % 10 == 0 or batch_number + 1 == total_batches:
            logger.info(
                f"Embedded {start + len(batch)}/{len(rows)} document chunks "
                f"({total_tokens} tokens so far)"
            )
    return total_tokens


# Write document chunk rows as JSON Lines with inline embeddings.
def write_document_chunks_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write document chunk rows as JSON Lines with inline embeddings.

    Args:
        rows: Embedded chunk rows.
        output_path: Destination ``document_chunks.jsonl`` path
            (overwritten).

    Raises:
        ValueError: If any row is missing its embedding — a partially
            embedded file would silently break the inline-embedding fast
            path in ``app.test_data_search.build_chunk_embeddings``.
    """
    missing = sum(1 for row in rows if not isinstance(row.get("embedding"), list))
    if missing:
        raise ValueError(
            f"Refusing to write {output_path}: {missing}/{len(rows)} rows "
            "have no embedding."
        )
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(rows)} embedded document chunks to {output_path}")


# Yield chunk rows from an existing document_chunks.jsonl.
def read_document_chunk_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Yield chunk rows from an existing ``document_chunks.jsonl``.

    Used by ``data_preparation.py`` to append document chunks to the main
    ``chunks.jsonl`` corpus.

    Args:
        path: Location of ``document_chunks.jsonl``.

    Yields:
        Chunk row dicts with ``id``, ``text``, ``metadata_``, ``node_id``
        and ``embedding``.
    """
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


# Load documents, create character-window chunks, embed them, and write the completed corpus.
def run_document_chunk_builder(
    documents_path: Path = DEFAULT_DOCUMENTS,
    output_path: Path = DEFAULT_OUTPUT,
    excluded_file_types: tuple[str, ...] = DEFAULT_EXCLUDED_FILE_TYPES,
    limit: int | None = None,
) -> dict[str, int]:
    """Build, embed and write the document chunks for the corpus.

    Args:
        documents_path: Location of ``documents_40pct.jsonl``.
        output_path: Destination ``document_chunks.jsonl`` path.
        excluded_file_types: Lowercase file extensions to skip.
        limit: Optional cap on the number of documents (smoke tests).

    Returns:
        Stats dict from :func:`build_document_chunk_rows` plus
        ``chunks_written`` and ``embedding_tokens``.

    Raises:
        FileNotFoundError: If the documents export is missing.
    """
    if not documents_path.exists():
        raise FileNotFoundError(f"Required input file missing: {documents_path}")
    documents = load_documents(documents_path)
    if limit is not None:
        documents = documents[:limit]
        logger.info(f"Smoke-test limit active: first {len(documents)} documents")
    rows, stats = build_document_chunk_rows(documents, excluded_file_types)
    client, model = _get_embedding_client()
    logger.info(f"Embedding {len(rows)} document chunks via '{model}' ...")
    stats["embedding_tokens"] = embed_chunk_rows(rows, client, model)
    write_document_chunks_jsonl(rows, output_path)
    stats["chunks_written"] = len(rows)
    return stats


# Parse document chunking options and run the gateway-backed builder, reporting the resulting
# statistics.
def main() -> int:
    """Run the document chunk builder from the command line.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Split documents_40pct.jsonl into 4,000/200 character chunks, "
            "embed them via the gateway and write document_chunks.jsonl."
        )
    )
    parser.add_argument(
        "--documents",
        type=Path,
        default=DEFAULT_DOCUMENTS,
        help=f"Input documents JSONL (default: {DEFAULT_DOCUMENTS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output document_chunks.jsonl path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--include-xlsx",
        action="store_true",
        help="Also chunk xlsx spreadsheet dumps (excluded by default).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N documents (smoke test).",
    )
    args = parser.parse_args()

    excluded: tuple[str, ...] = () if args.include_xlsx else DEFAULT_EXCLUDED_FILE_TYPES
    stats = run_document_chunk_builder(
        documents_path=args.documents,
        output_path=args.output,
        excluded_file_types=excluded,
        limit=args.limit,
    )
    logger.info(f"Document chunk builder finished: {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
