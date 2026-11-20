#!/usr/bin/env python3
"""Batch-run the evaluation questions through the frontend RAG modes.

Drives the exact functions the Streamlit frontend uses
(``streamlit_app.run_flat_search`` / ``run_adonis_search`` /
``run_hirag_search`` / ``generate_answer``) under a
``usage_logger.activate_tracker`` context per question, so every run lands
in ``usage_logs/frontend_usage.jsonl`` with per-call tokens and costs plus
one ``question_summary`` line — without needing a browser. The generated
answer text of each run is appended to ``usage_logs/batch_answers.jsonl``
(override with ``--answers-file``), linked to the usage-log summary via the
shared ``run_id``. Both files append, so every new query adds to them.

Modes:

- ``hirag``: one run per question per index (``--hirag-index``,
  repeatable). The index directory name and query mode are recorded in the
  summary's ``detail`` field; the answer-generation model is visible per
  call (``stage="answer_generation"``) and aggregated in the summary's
  ``answer_generation_models`` field.
- ``naive_rag``: flat top-k vector search + frontend answer generation.
- ``hierarchical``: MCP ``search_adonis`` (test-data parent-child
  pipeline) + frontend answer generation. Requires the MCP server on
  ``ADONIS_MCP_URL`` (default ``http://localhost:8001/mcp``) running with
  ``STAGE=test`` and the ``ADONIS_TEST_DATA_*`` flags; its retrieval-side
  calls are logged by the server process itself as standalone lines.

Usage (from the ``src`` directory, with the gateway key loaded)::

    cd src && set -a && source <(tr -d '\\r' < app/.env-dev) && set +a
    .venv/bin/python app/bachelor_project/run_frontend_usage_batch.py \
        --modes hirag,naive_rag,hierarchical
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_BP_DIR: Path = Path(__file__).resolve().parent
_SRC_DIR: Path = _BP_DIR.parents[1]

# Imports: ``app.settings`` needs src on sys.path; ``streamlit_app`` and
# ``usage_logger`` live in bachelor_project (streamlit_app also inserts it
# itself, but the import below needs it first).
for _path in (str(_SRC_DIR), str(_BP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import usage_logger  # noqa: E402

#: The three HiRAG gpt6 sample indexes (10 processes + 5 documents corpus)
#: the batch runs against by default, resolved under this directory.
_DEFAULT_INDEX_ROOT: Path = _BP_DIR / "HiRAG" / "hirag_index_samples_gpt6"
_DEFAULT_HIRAG_INDEXES: tuple[str, ...] = (
    "hirag_index_sample_10p_5d_questions_stages_pure_sol",
    "hirag_index_sample_10p_5d_stages_pure_luna_true",
    "hirag_index_sample_10p_5d_stages_luna_sol_mix2",
)

_DEFAULT_DATASET: Path = _BP_DIR / "evaluation" / "questions.jsonl"


def load_questions(dataset: Path) -> list[dict[str, str]]:
    """Load the RAGAS question dataset.

    Args:
        dataset: Path to the ``questions.jsonl`` file.

    Returns:
        One ``{"question_id": ..., "question": ...}`` dict per line.
    """
    questions: list[dict[str, str]] = []
    with open(dataset, encoding="utf-8") as dataset_file:
        for line in dataset_file:
            if line.strip():
                record = json.loads(line)
                questions.append(
                    {
                        "question_id": record["question_id"],
                        "question": record["user_input"],
                    }
                )
    return questions


def _resolve_hirag_index(index: str) -> str:
    """Resolve an index name under the gpt6 samples root to a full path.

    Args:
        index: Index directory name, or an already-resolvable path.

    Returns:
        The path string to pass to ``run_hirag_search``.
    """
    candidate = Path(index)
    if candidate.is_absolute() or candidate.is_dir():
        return str(candidate)
    return str(_DEFAULT_INDEX_ROOT / index)


def _run_one(
    frontend: Any,
    mode: str,
    question_id: str,
    question: str,
    *,
    hirag_index: str = "",
    hirag_mode: str = "hi",
    top_k: int = 5,
) -> dict[str, Any]:
    """Run one question through one RAG mode under a usage tracker.

    Mirrors the per-question flow of ``streamlit_app.main()``: retrieval via
    the mode's ``run_*`` function, then (except for HiRAG, which synthesizes
    its own answer) the frontend ``generate_answer`` call.

    Args:
        frontend: The imported ``streamlit_app`` module.
        mode: ``hirag``, ``naive_rag`` or ``hierarchical``.
        question_id: Dataset question ID (recorded in the summary detail).
        question: The question text.
        hirag_index: HiRAG index path (hirag mode only).
        hirag_mode: HiRAG query mode (hirag mode only).
        top_k: Number of retrieved chunks (naive_rag mode only).

    Returns:
        The tracker's summary dict, extended with ``ok``/``error`` and the
        ``question_id``.
    """
    if mode == "hirag":
        index_name = Path(hirag_index).name
        detail = (
            f"question_id={question_id}, index={index_name}, hirag_mode={hirag_mode}"
        )
    elif mode == "naive_rag":
        detail = f"question_id={question_id}, top_k={top_k}"
    else:
        detail = f"question_id={question_id}, mcp=search_adonis"

    started = time.monotonic()
    with usage_logger.activate_tracker(
        system=mode, question=question, detail=detail
    ) as tracker:
        error = ""
        answer = ""
        try:
            if mode == "hirag":
                response = frontend.run_hirag_search(question, hirag_index, hirag_mode)
            elif mode == "naive_rag":
                response = frontend.run_flat_search(question, top_k=top_k)
            else:
                response = frontend.run_adonis_search(question)

            if "error" in response:
                error = str(response["error"])
            elif mode == "hirag":
                # HiRAG synthesizes its answer inside the query.
                answer = str(response.get("answer", ""))
            else:
                # The other two modes generate the answer in this process.
                answer = frontend.generate_answer(question, response)
        except Exception as exc:  # noqa: BLE001 - one failure must not stop
            error = f"{type(exc).__name__}: {exc}"
        summary = tracker.summary()

    summary["question_id"] = question_id
    summary["ok"] = not error
    summary["error"] = error
    summary["answer"] = answer
    summary["wall_s"] = round(time.monotonic() - started, 1)
    return summary


def main() -> None:
    """Run the batch: all questions x all selected modes/indexes."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--modes",
        default="hirag,naive_rag,hierarchical",
        help="Comma-separated subset of hirag,naive_rag,hierarchical.",
    )
    parser.add_argument(
        "--dataset",
        default=str(_DEFAULT_DATASET),
        help="Path to the questions JSONL dataset.",
    )
    parser.add_argument(
        "--hirag-index",
        action="append",
        default=None,
        help="HiRAG index name (under hirag_index_samples_gpt6/) or path; "
        "repeatable. Defaults to the three 10p_5d gpt6 sample indexes.",
    )
    parser.add_argument("--hirag-mode", default="hi", help="HiRAG query mode.")
    parser.add_argument("--top-k", type=int, default=5, help="Flat-mode top_k.")
    parser.add_argument(
        "--limit", type=int, default=0, help="Only the first N questions."
    )
    parser.add_argument(
        "--answers-file",
        default=str(_BP_DIR / "usage_logs" / "batch_answers.jsonl"),
        help="JSONL file the per-run answer texts are appended to.",
    )
    args = parser.parse_args()

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    questions = load_questions(Path(args.dataset))
    if args.limit > 0:
        questions = questions[: args.limit]
    hirag_indexes = [
        _resolve_hirag_index(index)
        for index in (args.hirag_index or list(_DEFAULT_HIRAG_INDEXES))
    ]

    # Answer generation (naive_rag/hierarchical) uses the shared
    # ``answer_generation`` stage deployment from config.yaml via the
    # frontend's own gateway client; no Settings.llm setup needed here.
    import streamlit_app as frontend

    total_runs = len(questions) * (len(hirag_indexes) if "hirag" in modes else 0) + len(
        questions
    ) * len([m for m in modes if m != "hirag"])
    print(
        f"Batch: {len(questions)} questions x modes {modes}"
        + (f" x {len(hirag_indexes)} HiRAG indexes" if "hirag" in modes else "")
        + f" = {total_runs} runs -> {usage_logger._log_file()}"
    )

    answers_path = Path(args.answers_file)
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Answers -> {answers_path}")

    done = 0
    failed = 0
    for mode in modes:
        indexes = hirag_indexes if mode == "hirag" else [""]
        for index in indexes:
            for item in questions:
                done += 1
                summary = _run_one(
                    frontend,
                    mode,
                    item["question_id"],
                    item["question"],
                    hirag_index=index,
                    hirag_mode=args.hirag_mode,
                    top_k=args.top_k,
                )
                # One answers record per run, linked to the usage-log
                # question_summary via the shared run_id.
                answer_record = {
                    key: summary[key]
                    for key in (
                        "ts",
                        "run_id",
                        "system",
                        "question_id",
                        "question",
                        "detail",
                        "answer_generation_models",
                        "total_cost",
                        "currency",
                        "ok",
                        "error",
                        "answer",
                    )
                }
                with open(answers_path, "a", encoding="utf-8") as answers_file:
                    answers_file.write(
                        json.dumps(answer_record, ensure_ascii=False) + "\n"
                    )
                if not summary["ok"]:
                    failed += 1
                label = f"{mode}|{item['question_id']}"
                if mode == "hirag":
                    label += f"|{Path(index).name}"
                status = (
                    f"{summary['calls']} calls, "
                    f"{summary['total_input_tokens']} in / "
                    f"{summary['total_output_tokens']} out, "
                    f"{summary['total_cost']:.6f} {summary['currency']}, "
                    f"answer={','.join(summary['answer_generation_models']) or '-'}"
                    if summary["ok"]
                    else f"ERROR: {summary['error'][:200]}"
                )
                print(
                    f"[{done}/{total_runs}] {label}: {status} ({summary['wall_s']}s)",
                    flush=True,
                )

    print(f"Done: {done - failed}/{done} ok, {failed} failed.")


if __name__ == "__main__":
    main()
