"""Score saved RAG runs with the local RAGAS checkout.

Takes the answers/contexts a collector saved (``collect_*.py``) plus the
reference answers from the evaluation dataset and hands them to RAGAS.

Run it in the dedicated eval venv (never imports ``hirag``)::

    cd src/app/bachelor_project/evaluation
    .venv-eval/bin/python score.py --check-setup
    .venv-eval/bin/python score.py --run-dir runs/<run> --dataset <questions.jsonl>

Every (question, metric) pair ends in exactly one status:

``scored``   RAGAS returned a value (0.0 is a real score).
``skipped``  The metric was not applicable (reason given).
``failed``   RAGAS raised or returned NaN (error given).

Aggregates average ``scored`` values only and report all three counts.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

BACHELOR_DIR = Path(__file__).resolve().parent.parent
# Appended (not inserted) so the bachelor_project/ragas folder cannot shadow
# the installed ragas package.
if str(BACHELOR_DIR) not in sys.path:
    sys.path.append(str(BACHELOR_DIR))

from evaluation.common import (  # noqa: E402
    RAGAS_DIR,
    append_jsonl,
    compact_timestamp,
    get_evaluation_config,
    load_env_dev,
    normalize_id,
    package_versions,
    utc_timestamp,
    write_json,
)
from evaluation.dataset import EvalQuestion, dataset_sha256, load_dataset  # noqa: E402
from evaluation.run_format import (  # noqa: E402
    ADONIS_ID_GRANULARITY,
    CONTEXT_GRANULARITIES,
    load_run,
    select_contexts,
)

SCORES_FILE = "scores.jsonl"
SCORES_CSV = "scores.csv"
AGGREGATE_FILE = "aggregate.json"
SCORE_METADATA_FILE = "score_metadata.json"


# Build a factual-F1 metric that reports the claim counts already computed during verification.
def _factual_correctness_with_breakdown(llm: Any) -> Any:
    """Build a FactualCorrectness (f1) that also logs precision and recall.

    In f1 mode the metric already runs BOTH claim-verification directions
    (response claims vs. reference, reference claims vs. response), so TP/FP/FN
    exist anyway; this subclass only reuses them for the ``reason`` field —
    zero additional LLM calls.

    Args:
        llm: The judge LLM.

    Returns:
        Metric instance whose ``reason`` carries ``precision=…, recall=…
        (tp=…, fp=…, fn=…)`` per scored question.
    """
    import numpy as np  # noqa: PLC0415
    from ragas.metrics.collections import FactualCorrectness  # noqa: PLC0415
    from ragas.metrics.result import MetricResult  # noqa: PLC0415
    from ragas.metrics.utils import fbeta_score  # noqa: PLC0415

    class FactualCorrectnessWithBreakdown(FactualCorrectness):
        """FactualCorrectness f1 plus a free precision/recall breakdown."""

        # Verify claims in both directions and return factual F1 with precision, recall, and TP/FP/FN
        # counts.
        async def ascore(self, response: str, reference: str) -> Any:
            """Score like the parent, but report the decomposition counts."""
            if not response:
                raise ValueError("response is missing.")
            if not reference:
                raise ValueError("reference is missing.")
            reference_response = await self._decompose_and_verify_claims(
                response, reference
            )
            response_reference = await self._decompose_and_verify_claims(
                reference, response
            )
            tp = int(np.sum(reference_response))
            fp = int(np.sum(~reference_response))
            fn = int(np.sum(~response_reference))
            score = fbeta_score(tp, fp, fn, self.beta)
            reason = (
                f"precision={tp / (tp + fp + 1e-8):.2f}, "
                f"recall={tp / (tp + fn + 1e-8):.2f} "
                f"(tp={tp}, fp={fp}, fn={fn})"
            )
            return MetricResult(value=float(np.round(score, 2)), reason=reason)

    return FactualCorrectnessWithBreakdown(llm=llm, mode="f1")


@dataclass(frozen=True)
class MetricSpec:
    """Inputs a metric needs; used for the skip decision."""

    needs_response: bool
    needs_contexts: bool
    needs_reference: bool
    id_based: bool = False


METRIC_SPECS: dict[str, MetricSpec] = {
    "faithfulness": MetricSpec(True, True, False),
    "answer_relevancy": MetricSpec(True, False, False),
    "context_precision": MetricSpec(False, True, True),
    "context_utilization": MetricSpec(True, True, False),
    "context_recall": MetricSpec(False, True, True),
    "answer_correctness": MetricSpec(True, False, True),
    "factual_correctness": MetricSpec(True, False, True),
    "context_entity_recall": MetricSpec(False, True, True),
    "id_context_precision": MetricSpec(False, False, False, id_based=True),
    "id_context_recall": MetricSpec(False, False, False, id_based=True),
}


# Check metric prerequisites and return a skip reason when answers, labels, contexts, or usable
# IDs are missing.
def skip_reason(
    metric: str,
    record: dict[str, Any] | None,
    question: EvalQuestion,
    contexts: list[str],
    id_granularity: str | None,
) -> str | None:
    """Return why ``metric`` is not applicable to this question, else ``None``.

    Args:
        metric: Metric name from :data:`METRIC_SPECS`.
        record: Saved run record (``None`` if the question was not collected).
        question: Dataset entry with the reference data.
        contexts: Context strings at the chosen granularity.
        id_granularity: ``id_granularity`` of the run metadata.

    Returns:
        Human-readable skip reason or ``None`` when the metric can be scored.
    """
    spec = METRIC_SPECS[metric]
    if record is None:
        return "missing_record: question not present in the run"
    if record.get("status") != "answered":
        return f"record status is {record.get('status')!r}, not 'answered'"
    if spec.needs_response and not str(record.get("response") or "").strip():
        return "empty response"
    if spec.needs_reference and not question.reference:
        return "no reference answer in the dataset"
    if spec.needs_contexts and not contexts:
        return "no contexts at the selected granularity"
    if spec.id_based:
        if id_granularity != ADONIS_ID_GRANULARITY:
            return f"run has no comparable IDs (id_granularity={id_granularity!r})"
        if not question.reference_context_ids:
            return "no reference_context_ids in the dataset"
        if metric == "id_context_precision" and not record.get("retrieved_context_ids"):
            return "no retrieved_context_ids (precision undefined)"
    return None


# Create the requested RAGAS metrics with the configured judge and embedding models.
def build_metrics(eval_cfg: dict[str, Any], names: list[str]) -> dict[str, Any]:
    """Create the RAGAS metric objects for the evaluator models in ``config.yaml``.

    Args:
        eval_cfg: Resolved ``evaluation`` settings.
        names: Metric names to build.

    Returns:
        Mapping metric name -> metric instance.
    """
    import llm_config  # noqa: PLC0415 - bachelor_project dir is on sys.path
    from openai import AsyncAzureOpenAI  # noqa: PLC0415
    from ragas.embeddings.openai_provider import OpenAIEmbeddings  # noqa: PLC0415
    from ragas.llms import llm_factory  # noqa: PLC0415
    from ragas.metrics._context_precision import IDBasedContextPrecision  # noqa: PLC0415
    from ragas.metrics._context_recall import IDBasedContextRecall  # noqa: PLC0415
    from ragas.metrics.collections import (  # noqa: PLC0415
        AnswerCorrectness,
        AnswerRelevancy,
        ContextEntityRecall,
        ContextPrecision,
        ContextRecall,
        ContextUtilization,
        Faithfulness,
    )

    api_key = llm_config.get_api_key("openai")
    gateway = llm_config.OPENAI_CONFIG

    # Create an Azure gateway client for the requested evaluator deployment.
    def client(deployment: str) -> AsyncAzureOpenAI:
        # Create an Azure OpenAI gateway client for the requested deployment.
        return AsyncAzureOpenAI(
            azure_endpoint=gateway["base_url"],
            azure_deployment=deployment,
            api_key=api_key,
            api_version=gateway["api_version"],
            default_headers={"X-Gateway-Api-Key": api_key},
        )

    # The deployment comes from the client; ragas_model_name only steers
    # RAGAS' model-family handling (e.g. reasoning-model parameters).
    llm = llm_factory(
        eval_cfg["ragas_model_name"],
        provider="openai",
        client=client(eval_cfg["llm_model"]),
        max_tokens=eval_cfg["max_tokens"],
    )
    embeddings = OpenAIEmbeddings(
        client=client(eval_cfg["embedding_model"]), model=eval_cfg["embedding_model"]
    )
    factories = {
        "faithfulness": lambda: Faithfulness(llm=llm),
        "answer_relevancy": lambda: AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": lambda: ContextPrecision(llm=llm),
        "context_utilization": lambda: ContextUtilization(llm=llm),
        "context_recall": lambda: ContextRecall(llm=llm),
        "answer_correctness": lambda: AnswerCorrectness(llm=llm, embeddings=embeddings),
        "factual_correctness": lambda: _factual_correctness_with_breakdown(llm),
        "context_entity_recall": lambda: ContextEntityRecall(llm=llm),
        "id_context_precision": IDBasedContextPrecision,
        "id_context_recall": IDBasedContextRecall,
    }
    metrics = {name: factories[name]() for name in names}
    metrics["_llm"] = llm
    return metrics


# Translate the LLM metric prompts when prompt_language is not English.
async def adapt_prompts(metrics: dict[str, Any], language: str) -> None:
    """Translate the LLM metric prompts when ``prompt_language`` is not English."""
    from ragas.prompt.metrics.base_prompt import BasePrompt  # noqa: PLC0415

    llm = metrics["_llm"]
    for name, metric in metrics.items():
        if name.startswith("_"):
            continue
        for attr, value in list(vars(metric).items()):
            if isinstance(value, BasePrompt):
                setattr(metric, attr, await value.adapt(language, llm))


# Prepare the metric-specific inputs, await its score, and return the numeric value and
# explanation.
async def run_metric(
    name: str,
    metric: Any,
    record: dict[str, Any],
    question: EvalQuestion,
    contexts: list[str],
) -> tuple[float, str | None]:
    """Score one metric for one question.

    Returns:
        ``(value, reason)``; ``reason`` is the RAGAS explanation if any.
    """
    user_input = question.user_input
    response = record.get("response") or ""
    reference = question.reference or ""
    if name == "faithfulness":
        result = await metric.ascore(
            user_input=user_input, response=response, retrieved_contexts=contexts
        )
    elif name == "answer_relevancy":
        result = await metric.ascore(user_input=user_input, response=response)
    elif name == "context_precision":
        result = await metric.ascore(
            user_input=user_input, reference=reference, retrieved_contexts=contexts
        )
    elif name == "context_utilization":
        result = await metric.ascore(
            user_input=user_input, response=response, retrieved_contexts=contexts
        )
    elif name == "context_recall":
        result = await metric.ascore(
            user_input=user_input, retrieved_contexts=contexts, reference=reference
        )
    elif name == "answer_correctness":
        result = await metric.ascore(
            user_input=user_input, response=response, reference=reference
        )
    elif name == "factual_correctness":
        result = await metric.ascore(response=response, reference=reference)
    elif name == "context_entity_recall":
        result = await metric.ascore(reference=reference, retrieved_contexts=contexts)
    else:
        from ragas.dataset_schema import SingleTurnSample  # noqa: PLC0415

        sample = SingleTurnSample(
            retrieved_context_ids=[
                normalize_id(i) for i in record.get("retrieved_context_ids") or []
            ],
            reference_context_ids=[
                normalize_id(i) for i in question.reference_context_ids or ()
            ],
        )
        return float(await metric.single_turn_ascore(sample)), None
    return float(result.value), getattr(result, "reason", None)


# Score all requested metrics for one question concurrently while recording skips and failures
# separately.
async def score_question(
    question: EvalQuestion,
    record: dict[str, Any] | None,
    metrics: dict[str, Any],
    names: list[str],
    granularity: str,
    id_granularity: str | None,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Score all requested metrics for one question.

    Returns:
        Result line for ``scores.jsonl``.
    """
    contexts = select_contexts(record, granularity) if record else []
    results: dict[str, dict[str, Any]] = {}

    # Score one metric under the concurrency limit and record a scored, skipped, or failed result.
    async def one(name: str) -> None:
        # Score one metric with bounded concurrency and record scored, skipped, or failed status.
        reason = skip_reason(name, record, question, contexts, id_granularity)
        if reason:
            results[name] = {"status": "skipped", "reason": reason}
            return
        started = time.perf_counter()
        async with semaphore:
            try:
                value, explanation = await run_metric(
                    name, metrics[name], record, question, contexts
                )
            except Exception as exc:  # noqa: BLE001 - recorded per metric
                results[name] = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "duration_s": round(time.perf_counter() - started, 3),
                }
                return
        duration = round(time.perf_counter() - started, 3)
        if math.isnan(value):
            results[name] = {
                "status": "failed",
                "error": "metric returned NaN",
                "duration_s": duration,
            }
        else:
            results[name] = {
                "status": "scored",
                "value": value,
                "reason": explanation,
                "duration_s": duration,
            }

    await asyncio.gather(*(one(name) for name in names))
    return {
        "question_id": question.question_id,
        "record_status": record.get("status") if record else None,
        "n_contexts": len(contexts),
        "metrics": {name: results[name] for name in names},
    }


# Average successfully scored values and count scored, skipped, and failed results for each
# metric.
def aggregate(lines: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    """Mean over scored values plus scored/skipped/failed counts per metric."""
    summary: dict[str, Any] = {}
    for name in names:
        values = [
            line["metrics"][name]["value"]
            for line in lines
            if line["metrics"][name]["status"] == "scored"
        ]
        counts = {"scored": 0, "skipped": 0, "failed": 0}
        for line in lines:
            counts[line["metrics"][name]["status"]] += 1
        summary[name] = {
            "mean": sum(values) / len(values) if values else None,
            **counts,
        }
    record_statuses: dict[str, int] = {}
    for line in lines:
        key = str(line["record_status"])
        record_statuses[key] = record_statuses.get(key, 0) + 1
    return {
        "metrics": summary,
        "questions": len(lines),
        "record_statuses": record_statuses,
    }


# Write one row per question with <metric> value and <metric>_status.
def write_csv(path: Path, lines: list[dict[str, Any]], names: list[str]) -> None:
    """Write one row per question with ``<metric>`` value and ``<metric>_status``."""
    fields = ["question_id", "record_status", "n_contexts"]
    for name in names:
        fields += [name, f"{name}_status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for line in lines:
            row = {key: line[key] for key in fields[:3]}
            for name in names:
                result = line["metrics"][name]
                row[name] = result.get("value", "")
                row[f"{name}_status"] = result["status"]
            writer.writerow(row)


# Load a collected run, score its questions, and write per-question scores, aggregates, and
# metadata.
async def score_run(
    run_dir: Path,
    questions: list[EvalQuestion],
    dataset_path: Path,
    names: list[str],
    granularity: str,
    eval_cfg: dict[str, Any],
    out_dir: Path,
    concurrency: int,
    merge: bool = False,
) -> dict[str, Any]:
    """Score one saved run and write all output files to ``out_dir``.

    With ``merge=True`` the ``names`` metrics are scored and merged INTO an
    existing scores dir: per question, the new metric entries are added to (or
    replace) the ones already stored, and scores.jsonl / scores.csv /
    aggregate.json / score_metadata.json are rewritten over the union of old
    and new metrics. Nothing outside ``out_dir`` is touched.

    Returns:
        The aggregate dict (over all metrics present after merging).
    """
    run_metadata, records = load_run(run_dir)
    expected_sha = (run_metadata.get("dataset") or {}).get("sha256")
    actual_sha = dataset_sha256(dataset_path)
    if expected_sha and expected_sha != actual_sha:
        print(
            f"WARNING: {run_dir} was collected with a different dataset file "
            "(sha256 mismatch); scores use the current references.",
            file=sys.stderr,
        )

    metrics = build_metrics(eval_cfg, names)
    if eval_cfg["prompt_language"].lower() != "english":
        await adapt_prompts(metrics, eval_cfg["prompt_language"])

    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / SCORES_FILE
    existing_lines: list[dict[str, Any]] = []
    if scores_path.exists():
        if not merge:
            raise FileExistsError(
                f"{scores_path} already exists; choose a new --out-dir."
            )
        with scores_path.open(encoding="utf-8") as handle:
            existing_lines = [json.loads(line) for line in handle if line.strip()]
        print(f"Merging into {out_dir} ({len(existing_lines)} existing questions)")

    started_at = utc_timestamp()
    semaphore = asyncio.Semaphore(concurrency)
    id_granularity = run_metadata.get("id_granularity")
    lines: list[dict[str, Any]] = []
    for index, question in enumerate(questions, start=1):
        line = await score_question(
            question,
            records.get(question.question_id),
            metrics,
            names,
            granularity,
            id_granularity,
            semaphore,
        )
        if not merge:
            append_jsonl(scores_path, line)
        lines.append(line)
        statuses = ", ".join(
            f"{n}={r.get('value', r['status'])}" for n, r in line["metrics"].items()
        )
        print(f"[{index}/{len(questions)}] {question.question_id}: {statuses}")

    if merge and existing_lines:
        # Merge: new metric entries replace/add per question over the old file.
        new_by_qid = {line["question_id"]: line for line in lines}
        merged: list[dict[str, Any]] = []
        for old in existing_lines:
            new_line = new_by_qid.pop(old["question_id"], None)
            if new_line is None:
                merged.append(old)
                continue
            old["metrics"].update(new_line["metrics"])
            old["record_status"] = new_line["record_status"]
            old["n_contexts"] = new_line["n_contexts"]
            merged.append(old)
        merged.extend(new_by_qid.values())  # questions not in the old file
        old_names = list(existing_lines[0]["metrics"]) if existing_lines else []
        all_names = old_names + [n for n in names if n not in old_names]
        with scores_path.open("w", encoding="utf-8") as handle:
            for line in merged:
                handle.write(json.dumps(line, ensure_ascii=False) + "\n")
        lines = merged
        names = all_names

    write_csv(out_dir / SCORES_CSV, lines, names)
    summary = aggregate(lines, names)
    write_json(out_dir / AGGREGATE_FILE, summary)
    metadata = {
        "run_dir": str(run_dir),
        "system": run_metadata.get("system"),
        "run_id_granularity": id_granularity,
        "run_status": run_metadata.get("status"),
        "dataset": {"path": str(dataset_path), "sha256": actual_sha},
        "dataset_matches_run": expected_sha == actual_sha if expected_sha else None,
        "metrics": names,
        "context_granularity": granularity,
        "evaluator": eval_cfg,
        "concurrency": concurrency,
        "packages": package_versions(["ragas", "openai", "instructor"]),
        "started_at": started_at,
        "finished_at": utc_timestamp(),
    }
    if merge:
        metadata["merged_into_existing"] = True
    write_json(out_dir / SCORE_METADATA_FILE, metadata)
    return summary


# Offline setup check: ragas location, config section, API key presence.
def check_setup(eval_cfg_overrides: dict[str, Any]) -> int:
    """Offline setup check: ragas location, config section, API key presence."""
    import llm_config  # noqa: PLC0415
    import ragas  # noqa: PLC0415

    ok = True
    expected = (RAGAS_DIR / "src" / "ragas").resolve()
    actual = Path(ragas.__file__).resolve().parent
    print(f"ragas {package_versions(['ragas'])['ragas']} at {actual}")
    if actual != expected:
        print(f"ERROR: expected the local checkout at {expected}", file=sys.stderr)
        ok = False
    try:
        eval_cfg = get_evaluation_config(llm_config.config, eval_cfg_overrides)
        print(f"evaluator config: {eval_cfg}")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        ok = False
    try:
        llm_config.get_api_key("openai")
        print("API key: present")
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        ok = False
    try:
        from ragas.metrics.collections import Faithfulness  # noqa: F401, PLC0415
    except ImportError as exc:
        print(f"ERROR: ragas metrics import failed: {exc}", file=sys.stderr)
        ok = False
    print("setup OK" if ok else "setup FAILED")
    return 0 if ok else 1


# Parse the CLI arguments.
def parse_args() -> argparse.Namespace:
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=[],
        help="Saved run directory (repeatable).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Evaluation dataset JSONL with the reference answers.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=list(METRIC_SPECS),
        default=list(METRIC_SPECS),
        help="Metrics to compute (default: all).",
    )
    parser.add_argument(
        "--context-granularity",
        choices=CONTEXT_GRANULARITIES,
        default="items",
        help="Which saved contexts RAGAS sees.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output dir (only with a single --run-dir). "
        "Default: <run-dir>/scores_<timestamp>_<granularity>.",
    )
    parser.add_argument(
        "--merge-into",
        type=Path,
        help="Merge the --metrics results INTO this existing scores dir "
        "(per question, new entries add to or replace stored ones) instead of "
        "creating a new dir. Only with a single --run-dir.",
    )
    parser.add_argument("--limit", type=int, help="Score only the first N questions.")
    parser.add_argument(
        "--concurrency", type=int, default=4, help="Parallel metric calls (default: 4)."
    )
    parser.add_argument("--llm-model", help="Override evaluation.llm_model.")
    parser.add_argument(
        "--ragas-model-name", help="Override evaluation.ragas_model_name."
    )
    parser.add_argument(
        "--embedding-model", help="Override evaluation.embedding_model."
    )
    parser.add_argument(
        "--check-setup",
        action="store_true",
        help="Verify the environment offline and exit.",
    )
    return parser.parse_args()


# Parse evaluation options and run either an offline setup check or scoring for the selected
# saved runs.
def main() -> int:
    """CLI entry point."""
    args = parse_args()
    load_env_dev()
    overrides = {
        "llm_model": args.llm_model,
        "ragas_model_name": args.ragas_model_name,
        "embedding_model": args.embedding_model,
    }
    if args.check_setup:
        return check_setup(overrides)
    if not args.run_dir or not args.dataset:
        print("ERROR: --run-dir and --dataset are required.", file=sys.stderr)
        return 2
    if args.out_dir and len(args.run_dir) > 1:
        print("ERROR: --out-dir only works with a single --run-dir.", file=sys.stderr)
        return 2
    if args.merge_into and len(args.run_dir) > 1:
        print(
            "ERROR: --merge-into only works with a single --run-dir.", file=sys.stderr
        )
        return 2
    if args.merge_into and args.out_dir:
        print(
            "ERROR: --merge-into and --out-dir are mutually exclusive.", file=sys.stderr
        )
        return 2
    if args.merge_into and not (args.merge_into / SCORES_FILE).exists():
        print(
            f"ERROR: {args.merge_into / SCORES_FILE} does not exist.",
            file=sys.stderr,
        )
        return 2
    if args.concurrency < 1:
        print("ERROR: --concurrency must be >= 1.", file=sys.stderr)
        return 2

    import llm_config  # noqa: PLC0415

    eval_cfg = get_evaluation_config(llm_config.config, overrides)
    questions = load_dataset(args.dataset)
    if args.limit:
        questions = questions[: args.limit]

    for run_dir in args.run_dir:
        out_dir = (
            args.merge_into
            or args.out_dir
            or (run_dir / f"scores_{compact_timestamp()}_{args.context_granularity}")
        )
        summary = asyncio.run(
            score_run(
                run_dir,
                questions,
                args.dataset,
                args.metrics,
                args.context_granularity,
                eval_cfg,
                out_dir,
                args.concurrency,
                merge=args.merge_into is not None,
            )
        )
        print(f"\n{run_dir} -> {out_dir}")
        for name, entry in summary["metrics"].items():
            mean = "n/a" if entry["mean"] is None else f"{entry['mean']:.4f}"
            print(
                f"  {name:38s} mean={mean}  scored={entry['scored']} "
                f"skipped={entry['skipped']} failed={entry['failed']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
