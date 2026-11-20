"""Evaluation question dataset: schema, validation and loading.

Input format: JSON Lines (UTF-8), one JSON object per line. Blank lines are
ignored. Fields:

``question_id`` (required, string)
    Stable, unique identifier. Use a string even for numeric IDs (``"17"``)
    so IDs survive spreadsheets and joins unchanged.
``user_input`` (required, string)
    The question exactly as the RAG system should receive it.
``reference`` (optional, string or null)
    Human-written reference answer. Required by context precision (with
    reference), context recall and answer correctness. NEVER sent to the RAG
    system.
``reference_context_ids`` (optional, list of strings or null)
    ADONIS source-record IDs (process ``model_id`` or document
    ``document_object_id``) that a human verified as necessary to answer.
    Only used by the optional ID-based metrics. NEVER sent to the RAG system.
``metadata`` (optional, object)
    Free-form annotations (annotator, topic, difficulty, ...). Copied into the
    score output, never used for scoring or retrieval.

Run ``python evaluation/dataset.py <file>`` (from ``bachelor_project/``) to
validate a file.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from .common import sha256_file
except ImportError:  # executed as a script: python evaluation/dataset.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from evaluation.common import sha256_file

REQUIRED_FIELDS = ("question_id", "user_input")
OPTIONAL_FIELDS = ("reference", "reference_context_ids", "metadata")
ALLOWED_FIELDS = REQUIRED_FIELDS + OPTIONAL_FIELDS


class DatasetValidationError(ValueError):
    """Raised when a dataset file violates the schema.

    Attributes:
        errors: All problems found, each prefixed with the line number.
    """

    # Keep dataset validation errors and format a bounded list of them in the exception message.
    def __init__(self, path: Path, errors: list[str]) -> None:
        # Keep all dataset validation errors and format a bounded list for the exception message.
        self.errors = errors
        listing = "\n".join(f"  - {error}" for error in errors[:50])
        more = f"\n  ... and {len(errors) - 50} more" if len(errors) > 50 else ""
        super().__init__(
            f"Dataset {path} has {len(errors)} problem(s):\n{listing}{more}\n"
            f"Allowed fields: {', '.join(ALLOWED_FIELDS)} "
            f"(required: {', '.join(REQUIRED_FIELDS)})."
        )


@dataclass(frozen=True)
class EvalQuestion:
    """One validated evaluation question."""

    question_id: str
    user_input: str
    reference: str | None = None
    reference_context_ids: tuple[str, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    line_number: int = 0


# Validate one question record and return either a typed question or its field-level errors.
def _validate_record(
    line_number: int, record: Any
) -> tuple[EvalQuestion | None, list[str]]:
    """Validate one parsed JSON value.

    Args:
        line_number: 1-based line number for error messages.
        record: Parsed JSON value of the line.

    Returns:
        The question (``None`` on errors) and the list of problems.
    """
    prefix = f"line {line_number}"
    if not isinstance(record, dict):
        return None, [f"{prefix}: expected a JSON object, got {type(record).__name__}"]

    errors: list[str] = []
    unknown = sorted(set(record) - set(ALLOWED_FIELDS))
    if unknown:
        errors.append(f"{prefix}: unknown field(s) {unknown}")

    question_id = record.get("question_id")
    if not isinstance(question_id, str) or not question_id.strip():
        hint = (
            ' (write numeric IDs as strings, e.g. "17")'
            if isinstance(question_id, (int, float))
            else ""
        )
        errors.append(f"{prefix}: 'question_id' must be a non-empty string{hint}")
    elif question_id != question_id.strip():
        errors.append(f"{prefix}: 'question_id' has leading/trailing whitespace")

    user_input = record.get("user_input")
    if not isinstance(user_input, str) or not user_input.strip():
        errors.append(f"{prefix}: 'user_input' must be a non-empty string")

    reference = record.get("reference")
    if reference is not None and (
        not isinstance(reference, str) or not reference.strip()
    ):
        errors.append(
            f"{prefix}: 'reference' must be a non-empty string or null/omitted"
        )

    ref_ids = record.get("reference_context_ids")
    ref_ids_tuple: tuple[str, ...] | None = None
    if ref_ids is not None:
        if not isinstance(ref_ids, list):
            errors.append(
                f"{prefix}: 'reference_context_ids' must be a list of strings or null"
            )
        else:
            bad = [
                i
                for i, value in enumerate(ref_ids)
                if not isinstance(value, str) or not value.strip()
            ]
            if bad:
                errors.append(
                    f"{prefix}: 'reference_context_ids' entries at positions {bad} "
                    "must be non-empty strings"
                )
            elif len(set(ref_ids)) != len(ref_ids):
                errors.append(f"{prefix}: 'reference_context_ids' contains duplicates")
            elif not ref_ids:
                errors.append(
                    f"{prefix}: 'reference_context_ids' is an empty list; omit the "
                    "field or use null when no IDs were labeled"
                )
            else:
                ref_ids_tuple = tuple(ref_ids)

    metadata = record.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        errors.append(f"{prefix}: 'metadata' must be a JSON object or omitted")

    if errors:
        return None, errors
    return (
        EvalQuestion(
            question_id=question_id,
            user_input=user_input,
            reference=reference,
            reference_context_ids=ref_ids_tuple,
            metadata=dict(metadata or {}),
            line_number=line_number,
        ),
        [],
    )


# Parse question JSONL, collect schema and duplicate-ID errors, and return validated questions.
def load_dataset(path: Path) -> list[EvalQuestion]:
    """Load and validate a question dataset.

    All problems of the file are collected before raising, so one run of the
    validator reports everything that needs fixing.

    Args:
        path: JSONL dataset path.

    Returns:
        The questions in file order.

    Raises:
        FileNotFoundError: If the file does not exist.
        DatasetValidationError: If the file violates the schema or is empty.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    questions: list[EvalQuestion] = []
    errors: list[str] = []
    first_line_of: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(
                    f"line {line_number}: invalid JSON ({exc.msg} at column {exc.colno})"
                )
                continue
            question, record_errors = _validate_record(line_number, record)
            errors.extend(record_errors)
            if question is None:
                continue
            if question.question_id in first_line_of:
                errors.append(
                    f"line {line_number}: duplicate question_id "
                    f"{question.question_id!r} (first seen on line "
                    f"{first_line_of[question.question_id]})"
                )
                continue
            first_line_of[question.question_id] = line_number
            questions.append(question)

    if not questions and not errors:
        errors.append("file contains no questions")
    if errors:
        raise DatasetValidationError(path, errors)
    return questions


# Return the sha256 of the dataset file (identifies the exact version).
def dataset_sha256(path: Path) -> str:
    """Return the sha256 of the dataset file (identifies the exact version)."""
    return sha256_file(path)


# Count which optional label fields are present.
def summarize(questions: list[EvalQuestion]) -> dict[str, int]:
    """Count which optional label fields are present."""
    return {
        "questions": len(questions),
        "with_reference": sum(q.reference is not None for q in questions),
        "with_reference_context_ids": sum(
            q.reference_context_ids is not None for q in questions
        ),
    }


# Validate a dataset from the CLI and print its question and optional-label summary.
def main() -> int:
    """Validate a dataset file from the command line."""
    parser = argparse.ArgumentParser(
        description="Validate an evaluation question dataset (JSONL)."
    )
    parser.add_argument("dataset", type=Path, help="path to the JSONL dataset")
    args = parser.parse_args()
    try:
        questions = load_dataset(args.dataset)
    except (FileNotFoundError, DatasetValidationError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "valid": True,
                "sha256": dataset_sha256(args.dataset),
                **summarize(questions),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
