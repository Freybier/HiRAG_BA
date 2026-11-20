"""Example: load generated textual records into an existing HiRAG instance.

HiRAG's supported insertion interface is ``HiRAG.insert(string_or_strings)``
(see ``hirag/hirag.py`` in the vendored HiRAG repository): it accepts a
single string or a list of strings, and each string is hashed and
differentiated individually by HiRAG.

This example reads the converter's JSONL outputs, extracts only the plain
``text`` field of each record (never the JSON serialization), and passes
the strings to an existing HiRAG instance as one list. It does not modify
HiRAG and does not require HiRAG for conversion or testing.

Important: calling ``insert`` once per record does not guarantee that
HiRAG keeps exactly one internal chunk per record — HiRAG applies its own
chunking on top. The converter's document sections are source-side
chunking only.

Usage (dry run, no HiRAG required):
    python hirag_loader_example.py --output ./output

Usage (real insertion, requires a configured HiRAG instance):
    python hirag_loader_example.py --output ./output --insert
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Protocol

if __package__ in (None, ""):
    # Allow running as a plain script: python hirag_loader_example.py ...
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "adonis_converter"


class _Insertable(Protocol):
    """Structural type for objects exposing HiRAG's insert interface."""

    # Declare the insertion interface required from a HiRAG instance; this protocol has no
    # implementation.
    def insert(self, string_or_strings: Any) -> Any:
        """Insert one string or a list of strings into the index."""
        ...


# Load the plain text of every generated record in deterministic order.
def load_record_texts(output_dir: Path) -> list[str]:
    """Load the plain text of every generated record in deterministic order.

    Process records come first (input order), then document records. Only
    the ``text`` field is returned, so no JSON serialization syntax is
    passed to HiRAG.

    Args:
        output_dir: The converter output directory containing
            ``processes_textualized.jsonl`` and
            ``documents_textualized.jsonl``.

    Returns:
        The list of record texts.

    Raises:
        FileNotFoundError: If an expected output file is missing.
        ValueError: If a record has no usable text field.
    """
    texts: list[str] = []
    for filename in ("processes_textualized.jsonl", "documents_textualized.jsonl"):
        path = output_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing converter output: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                text = record.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(
                        f"{filename}:{line_number}: record "
                        f"{record.get('record_id')!r} has no usable text."
                    )
                texts.append(text)
    return texts


# Yield deterministic batches of record texts.
def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    """Yield deterministic batches of record texts.

    Args:
        items: Record texts in load order.
        batch_size: Maximum number of texts per batch.

    Yields:
        Sublists of at most ``batch_size`` texts.
    """
    for offset in range(0, len(items), batch_size):
        yield items[offset : offset + batch_size]


# Send record texts to the existing HiRAG instance in batches and count the inserted records.
def insert_into_hirag(
    hirag: _Insertable, texts: list[str], batch_size: int = 100
) -> int:
    """Insert record texts into an existing HiRAG instance.

    Args:
        hirag: An initialized HiRAG instance (created and configured by
            the caller; this function never builds one implicitly).
        texts: Plain record texts from :func:`load_record_texts`.
        batch_size: Number of strings per ``insert`` call. Each string
            keeps its own HiRAG-internal hash regardless of batching.

    Returns:
        The number of inserted record texts.
    """
    inserted = 0
    for batch in batched(texts, batch_size):
        hirag.insert(batch)
        inserted += len(batch)
    return inserted


# Load generated record texts for a dry run or insert them into HiRAG when explicitly requested.
def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for the loader example.

    Args:
        argv: Optional argument list (defaults to ``sys.argv``).

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Demonstrate how converter output can be inserted into an "
            "existing HiRAG instance. Runs as a dry run unless --insert "
            "is given."
        )
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Converter output directory with the textualized JSONL files.",
    )
    parser.add_argument(
        "--insert",
        action="store_true",
        help=(
            "Actually insert into HiRAG. Without this flag the example "
            "only verifies that the texts can be loaded (dry run)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Strings per HiRAG insert call (default: 100).",
    )
    args = parser.parse_args(argv)

    texts = load_record_texts(args.output)
    print(f"Loaded {len(texts)} record texts from {args.output}.")

    if not args.insert:
        print("Dry run: no HiRAG indexing performed (use --insert to insert).")
        return 0

    # Lazy import: HiRAG (and its LLM configuration) is only required when
    # real insertion is requested, never for conversion or testing.
    hirag_package = Path(__file__).resolve().parent.parent / "HiRAG"
    sys.path.insert(0, str(hirag_package))
    from hirag import HiRAG  # type: ignore[import-not-found]

    hirag = HiRAG(
        working_dir=str(hirag_package / "hirag_index_samples" / "hirag_index_adonis")
    )
    inserted = insert_into_hirag(hirag, texts, batch_size=args.batch_size)
    print(f"Inserted {inserted} record texts into HiRAG.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
