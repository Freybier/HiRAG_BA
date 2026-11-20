"""Console viewer for the anonymized Adonis test dataset SQL dump.

Parses the INSERT statements of ``anonymized_dataset.sql`` directly, so the
entries can be inspected without loading the dump into a database first.

Usage examples:
    python show_dataset.py                      # first 10 models
    python show_dataset.py --table chunks       # first 10 chunks
    python show_dataset.py --limit 3 --full     # 3 models, untruncated
    python show_dataset.py --search "Rückmeldung"
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DATASET_PATH = Path(__file__).with_name("anonymized_dataset.sql")
TABLE_ALIASES = {"models": "anon_models", "chunks": "anon_chunks"}
DEFAULT_LIMIT = 10
TRUNCATE_AT = 200

_INSERT_RE = re.compile(
    r"INSERT\s+INTO\s+(?P<table>\w+)\s*\((?P<cols>[^)]*)\)\s*VALUES\s*\(",
    re.IGNORECASE,
)


def _parse_values(sql_text: str, start: int) -> tuple[list[object], int]:
    """Parse one ``(...)`` VALUES tuple starting at ``start``.

    Handles single-quoted strings (with ``''`` escaping), numbers, and NULL.

    Args:
        sql_text: Full SQL dump text.
        start: Index of the opening parenthesis of the VALUES tuple.

    Returns:
        A tuple of the parsed values and the index just past the tuple.
    """
    values: list[object] = []
    index = start + 1
    length = len(sql_text)

    while index < length:
        char = sql_text[index]
        if char == "'":
            end = index + 1
            parts: list[str] = []
            while end < length:
                if sql_text[end] == "'":
                    if end + 1 < length and sql_text[end + 1] == "'":
                        parts.append("'")
                        end += 2
                        continue
                    end += 1
                    break
                parts.append(sql_text[end])
                end += 1
            values.append("".join(parts))
            index = end
        elif char == ")":
            return values, index + 1
        elif char in ", \t\r\n":
            index += 1
        else:
            end = index
            while end < length and sql_text[end] not in ",)":
                end += 1
            token = sql_text[index:end].strip()
            if token.upper() == "NULL":
                values.append(None)
            else:
                try:
                    values.append(int(token))
                except ValueError:
                    try:
                        values.append(float(token))
                    except ValueError:
                        values.append(token)
            index = end
    return values, index


def load_entries(dataset_path: Path) -> dict[str, list[dict[str, object]]]:
    """Parse all INSERT statements of the dump into row dictionaries.

    Args:
        dataset_path: Path to the anonymized dataset SQL file.

    Returns:
        Mapping of table name to a list of rows; each row maps column
        names to parsed values.
    """
    sql_text = dataset_path.read_text(encoding="utf-8")
    tables: dict[str, list[dict[str, object]]] = {}
    for match in _INSERT_RE.finditer(sql_text):
        table = match.group("table")
        columns = [col.strip() for col in match.group("cols").split(",")]
        values, _ = _parse_values(sql_text, match.end() - 1)
        row = dict(zip(columns, values))
        tables.setdefault(table, []).append(row)
    return tables


def _truncate(value: object, full: bool) -> str:
    """Render a value as a single-line string, truncated unless ``full``.

    Args:
        value: The parsed column value.
        full: If True, keep the full multi-line text.

    Returns:
        The formatted string for console output.
    """
    if value is None:
        return "NULL"
    text = str(value)
    if full:
        return text
    text = text.replace("\n", " ⏎ ")
    if len(text) > TRUNCATE_AT:
        return f"{text[:TRUNCATE_AT]}…"
    return text


def print_entries(
    rows: list[dict[str, object]],
    table: str,
    limit: int,
    full: bool,
    search: str | None,
) -> None:
    """Print dataset entries to the console.

    Args:
        rows: Parsed rows of one table.
        table: Table name used for the header output.
        limit: Maximum number of entries to print (0 means all).
        full: If True, print full untruncated field values.
        search: Optional case-insensitive substring filter over all fields.
    """
    if search:
        needle = search.casefold()
        rows = [
            row
            for row in rows
            if any(needle in str(value).casefold() for value in row.values())
        ]
    total = len(rows)
    shown = rows if limit <= 0 else rows[:limit]
    print(f"=== {table}: showing {len(shown)} of {total} entries ===")
    for position, row in enumerate(shown, start=1):
        print(f"\n--- entry {position} ---")
        for column, value in row.items():
            print(f"  {column}: {_truncate(value, full)}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the dataset console viewer.

    Args:
        argv: Optional argument list (defaults to sys.argv).

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        choices=["models", "chunks", "all"],
        default="models",
        help="Which table to show (default: models).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Max entries to print, 0 for all (default: {DEFAULT_LIMIT}).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print full field values without truncation.",
    )
    parser.add_argument(
        "--search",
        default=None,
        help="Case-insensitive substring filter applied to all fields.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=DATASET_PATH,
        help=f"Path to the SQL dump (default: {DATASET_PATH.name}).",
    )
    args = parser.parse_args(argv)

    if not args.file.is_file():
        print(f"Dataset file not found: {args.file}", file=sys.stderr)
        return 1

    tables = load_entries(args.file)
    selected = (
        list(TABLE_ALIASES.values())
        if args.table == "all"
        else [TABLE_ALIASES[args.table]]
    )
    for table in selected:
        print_entries(tables.get(table, []), table, args.limit, args.full, args.search)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
