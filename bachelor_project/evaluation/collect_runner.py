"""Shared question loop for all collectors (stdlib only).

Every collector (``collect_hirag.py``, ``collect_naive.py``,
``collect_test_data.py``) only implements ``answer_fn(question) -> partial
record``; this module handles dataset loading, the run directory, resume,
error capture and the run metadata, so all systems produce the identical
run format (see ``run_format.py``).
"""

from __future__ import annotations

import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from evaluation.common import (
    EVAL_DIR,
    append_jsonl,
    compact_timestamp,
    read_json,
    utc_timestamp,
    write_json,
)
from evaluation.dataset import EvalQuestion, dataset_sha256, load_dataset, summarize
from evaluation.run_format import (
    METADATA_FILE,
    RECORD_FIELDS,
    RECORD_STATUSES,
    RECORDS_FILE,
    SCHEMA_VERSION,
    SYSTEMS,
)

DEFAULT_RUNS_DIR = EVAL_DIR / "runs"

AnswerFn = Callable[[EvalQuestion], dict[str, Any]]


def default_out_dir(system: str, label: str) -> Path:
    """Return ``evaluation/runs/<system>_<label>_<timestamp>``."""
    return DEFAULT_RUNS_DIR / f"{system}_{label}_{compact_timestamp()}"


def merge_usage(target: dict[str, Any], usage: dict[str, Any]) -> None:
    """Add per-model usage counters (``{kind: {model: {counter: int}}}``)."""
    for kind, models in (usage or {}).items():
        for model, counters in models.items():
            current = target.setdefault(kind, {}).setdefault(model, {})
            for name, value in counters.items():
                current[name] = current.get(name, 0) + int(value or 0)


def _last_records(records_path: Path) -> dict[str, dict[str, Any]]:
    """Return ``question_id -> last record`` of an existing records file."""
    records: dict[str, dict[str, Any]] = {}
    if not records_path.is_file():
        return records
    with records_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{records_path} line {line_number} is not valid JSON "
                    f"({exc.msg}); fix or remove the line before resuming."
                ) from exc
            records[record["question_id"]] = record
    return records


def _status_counts(records: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Count final records per status."""
    counts = {status: 0 for status in RECORD_STATUSES}
    for record in records.values():
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    return counts


def run_collection(
    *,
    system: str,
    dataset_path: Path,
    out_dir: Path,
    run_config: dict[str, Any],
    metadata_extra: dict[str, Any],
    id_granularity: str | None,
    answer_fn: AnswerFn,
    resume: bool = False,
    limit: int | None = None,
) -> int:
    """Run all (remaining) dataset questions through ``answer_fn``.

    ``answer_fn`` returns the system-specific record fields (``status``,
    ``response``, ``retrieved_contexts``, ``context_text``, ...). Exceptions
    are recorded as ``status: error`` and the loop continues; ``--resume``
    retries exactly those questions later.

    Args:
        system: One of :data:`run_format.SYSTEMS`.
        dataset_path: Validated JSONL question dataset.
        out_dir: Run directory (created if missing).
        run_config: Settings that must stay identical across resumed
            sessions (models, retrieval parameters, index, env flags).
        metadata_extra: Additional metadata (fingerprints, versions, ...).
        id_granularity: ``run_format.ADONIS_ID_GRANULARITY`` or ``None`` when
            the system produces no comparable source IDs.
        answer_fn: Callback producing the partial record for one question.
        resume: Continue an existing run directory.
        limit: Only process the first N questions of the dataset.

    Returns:
        Process exit code (0 = no errors, 1 = some questions errored).
    """
    if system not in SYSTEMS:
        raise ValueError(f"Unknown system {system!r}; expected one of {SYSTEMS}")
    all_questions = load_dataset(dataset_path)
    questions = all_questions[:limit] if limit is not None else all_questions
    dataset_info = {
        "path": str(dataset_path.resolve()),
        "sha256": dataset_sha256(dataset_path),
        **summarize(all_questions),
    }

    metadata_path = out_dir / METADATA_FILE
    records_path = out_dir / RECORDS_FILE
    if metadata_path.exists() or records_path.exists():
        if not resume:
            raise SystemExit(
                f"{out_dir} already contains a run; pass --resume to continue it "
                "or choose another --out-dir."
            )
        metadata = read_json(metadata_path)
        mismatches = []
        if metadata.get("system") != system:
            mismatches.append(f"system {metadata.get('system')!r} != {system!r}")
        if metadata.get("dataset", {}).get("sha256") != dataset_info["sha256"]:
            mismatches.append("dataset file content changed")
        if metadata.get("run_config") != json.loads(json.dumps(run_config)):
            mismatches.append("run_config differs (models / retrieval / env flags)")
        if mismatches:
            raise SystemExit(
                "Cannot resume, the run would mix incompatible settings:\n  - "
                + "\n  - ".join(mismatches)
            )
    else:
        if resume:
            print(f"--resume: no existing run in {out_dir}, starting a new one.")
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "system": system,
            "id_granularity": id_granularity,
            "created_at": utc_timestamp(),
            "dataset": dataset_info,
            "run_config": run_config,
            "sessions": [],
        }

    previous = _last_records(records_path)
    pending = [
        q
        for q in questions
        if q.question_id not in previous or previous[q.question_id]["status"] == "error"
    ]

    session: dict[str, Any] = {
        "started_at": utc_timestamp(),
        "finished_at": None,
        "argv": sys.argv[1:],
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "attempted": 0,
        "errors": 0,
        **metadata_extra,
    }
    metadata["sessions"].append(session)
    metadata["status"] = "running"
    metadata["updated_at"] = utc_timestamp()
    write_json(metadata_path, metadata)

    print(
        f"{system}: {len(pending)} question(s) to run "
        f"({len(questions) - len(pending)} already done) -> {out_dir}"
    )
    for index, question in enumerate(pending, start=1):
        attempts = int(previous.get(question.question_id, {}).get("attempts", 0)) + 1
        start = time.perf_counter()
        try:
            partial = answer_fn(question)
            error = None
        except Exception as exc:  # noqa: BLE001 - recorded, loop continues
            partial = {"status": "error"}
            error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        record: dict[str, Any] = {name: None for name in RECORD_FIELDS}
        record.update(
            {
                "retrieved_contexts": [],
                "retrieved_context_ids": [],
                "provenance_context_ids": [],
                "unmapped_chunk_ids": [],
                "usage": {},
            }
        )
        record.update(partial)
        record.update(
            schema_version=SCHEMA_VERSION,
            question_id=question.question_id,
            user_input=question.user_input,
            attempts=attempts,
            duration_s=round(time.perf_counter() - start, 3),
        )
        if error is not None:
            record["error"] = error
        append_jsonl(records_path, record)
        session["attempted"] += 1
        session["errors"] += record["status"] == "error"
        print(
            f"[{index}/{len(pending)}] {question.question_id}: {record['status']} "
            f"({record['duration_s']:.1f}s)"
        )

    final = _last_records(records_path)
    counts = _status_counts(final)
    session["finished_at"] = utc_timestamp()
    metadata["counts"] = counts
    metadata["status"] = "completed_with_errors" if counts["error"] else "completed"
    metadata["updated_at"] = utc_timestamp()
    write_json(metadata_path, metadata)
    print(f"Done: {json.dumps(counts)}")
    return 1 if counts["error"] else 0
