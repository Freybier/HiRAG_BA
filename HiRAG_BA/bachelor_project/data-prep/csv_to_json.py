#!/usr/bin/env python3
"""Convert the modified ``data_vector_store3`` CSV export to addon JSON.

Reads ``data_vector_store3_40pct_modified.csv`` (the 40% sample chunk export
that additionally carries the ``embedding`` column) and writes a JSON array
in the same row shape as ``description_hierarchy-addon.json``, extended by
the embedding string:

- input columns: ``id``, ``node_id``, ``model_id``, ``object_type``,
  ``text``, ``metadata_``, ``embedding``
- output rows: same keys, ``id`` converted to ``int``, ``metadata_`` kept as
  its original JSON *string* (parsing happens in the chunks converter, which
  also applies the hierarchy metadata patch), ``embedding`` kept as the raw
  pgvector string (``[-0.01,0.02,...]``; parsed to floats in the chunks
  converter)

The script is stdlib-only and streams row by row, so the ~200 MB output is
never held in memory at once. No fields are modified; this is a pure format
conversion (CSV quoting -> JSON array).

Usage::

    cd /workspace/workdir/src/test-data
    ../.venv/bin/python data-prep/csv_to_json.py
    # or with explicit paths:
    ../.venv/bin/python data-prep/csv_to_json.py \
        --input test-data_40pct/data_vector_store3_40pct_modified.csv \
        --output test-data_40pct/description_hierarchy-addon_embedded.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

#: Default input CSV (the 40% sample chunk export that includes embeddings).
DEFAULT_INPUT_CSV = (
    Path(__file__).resolve().parents[1]
    / "test-data_40pct"
    / "data_vector_store3_40pct_modified.csv"
)

#: Default output JSON (original addon shape plus the ``embedding`` field).
DEFAULT_OUTPUT_JSON = (
    Path(__file__).resolve().parents[1]
    / "test-data_40pct"
    / "description_hierarchy-addon_embedded.json"
)

#: Columns the modified CSV export is expected to carry, in order.
EXPECTED_COLUMNS: tuple[str, ...] = (
    "id",
    "node_id",
    "model_id",
    "object_type",
    "text",
    "metadata_",
    "embedding",
)


# Check that an embedding value is a bracketed float list.
def validate_embedding_format(raw: str, row_id: str) -> int:
    """Check that an embedding value is a bracketed float list.

    Args:
        raw: Raw ``embedding`` cell from the CSV.
        row_id: Row ``id`` used in error messages.

    Returns:
        Number of float components in the embedding.

    Raises:
        ValueError: If the value is empty, not bracket-wrapped or contains
            non-numeric components.
    """
    value = raw.strip()
    if not (value.startswith("[") and value.endswith("]")):
        raise ValueError(
            f"Row {row_id}: embedding is not a bracketed list: {value[:40]!r}..."
        )
    parts = value[1:-1].split(",")
    for part in parts:
        float(part)  # raises ValueError on non-numeric components
    return len(parts)


# Convert the modified chunk CSV into the addon JSON array format.
def convert_csv_to_json(input_csv: Path, output_json: Path) -> int:
    """Convert the modified chunk CSV into the addon JSON array format.

    Rows are streamed one at a time; the output file is written
    incrementally so neither side is fully materialized in memory.

    Args:
        input_csv: Location of ``data_vector_store3_40pct_modified.csv``.
        output_json: Destination JSON file (overwritten).

    Returns:
        Number of rows written.

    Raises:
        ValueError: On unexpected columns or malformed embedding values.
    """
    rows_written = 0
    embedding_dim: int | None = None
    with (
        input_csv.open(newline="", encoding="utf-8") as csv_handle,
        output_json.open("w", encoding="utf-8", newline="\n") as json_handle,
    ):
        reader = csv.DictReader(csv_handle)
        if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
            raise ValueError(
                f"Unexpected CSV columns {reader.fieldnames}; "
                f"expected {list(EXPECTED_COLUMNS)}"
            )
        json_handle.write("[\n")
        for raw_row in reader:
            dim = validate_embedding_format(raw_row["embedding"], raw_row["id"])
            if embedding_dim is None:
                embedding_dim = dim
                logger.info(f"Embedding dimension: {embedding_dim}")
            elif dim != embedding_dim:
                raise ValueError(
                    f"Row {raw_row['id']}: embedding has {dim} components, "
                    f"expected {embedding_dim} (inconsistent vector size)"
                )
            row = {
                "id": int(raw_row["id"]),
                "node_id": raw_row["node_id"],
                "model_id": raw_row["model_id"],
                "object_type": raw_row["object_type"],
                "text": raw_row["text"],
                "metadata_": raw_row["metadata_"],
                "embedding": raw_row["embedding"],
            }
            prefix = ",\n" if rows_written else ""
            json_handle.write(prefix + json.dumps(row, ensure_ascii=False))
            rows_written += 1
        json_handle.write("\n]\n")
    logger.info(f"Wrote {rows_written} rows to {output_json}")
    return rows_written


# Parse source and destination paths, convert the chunk CSV to JSON, and report the resulting
# row count.
def main() -> int:
    """Run the CSV-to-JSON conversion.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert data_vector_store3_40pct_modified.csv into the "
            "description_hierarchy-addon JSON format (with embeddings)."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Input CSV path (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Output JSON path (default: %(default)s)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input CSV missing: {args.input}")
        return 1
    try:
        convert_csv_to_json(args.input, args.output)
    except ValueError as error:
        logger.error(str(error))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
