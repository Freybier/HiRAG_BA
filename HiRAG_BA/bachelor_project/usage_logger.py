#!/usr/bin/env python3
"""Runtime token & cost logger for the bachelor_project RAG frontends.

Whenever a question is asked through the Streamlit frontend
(``streamlit_app.py``), every LLM chat and embedding call made by the
selected RAG system is recorded with its input/output tokens and cost, and
one summary line per question is appended to a JSONL log.

Covered systems (the three retrieval modes of the frontend):

- ``hirag``: HiRAG graph-RAG queries. Instrumented in
  ``HiRAG/hi_Search_openai.py`` (all chat calls incl. stage routing and
  LLM-cache hits, plus query embeddings); the answer is synthesized inside
  HiRAG, so it shows up as a chat call with ``stage="answer_generation"``.
- ``naive_rag``: flat top-k vector search. Instrumented in
  ``naive_RAG/openAI_Test_chunked.py`` (query embeddings + optional
  multi-query expansion); the answer-generation call runs in the frontend.
- ``hierarchical``: test-data parent-child pipeline via the MCP server. The
  answer-generation call runs in the frontend process and is tracked there;
  the server-side retrieval embedding (and optional multi-query expansion)
  is logged by ``app/test_data_search.py`` as standalone call lines in the
  same JSONL file (separate process, correlated by question text and
  timestamp; the container mounts ``./src`` so the log file is shared).

Token counting prefers the API ``usage`` fields of the gateway response
(exact, includes reasoning tokens of the gpt-5.x/6.x deployments) and falls
back to local tiktoken counting (``cl100k_base``) when the response carries
no usage — the same fallback strategy as ``HiRAG/hi_insert_run.py``.

Prices come from ``HiRAG/model_prices.yaml`` (single pricing source, shared
with the HiRAG insertion run logs; override with ``USAGE_PRICES_FILE``).
Models missing from that file are still token-tracked but reported under
``unpriced_models`` with cost ``None``; HiRAG LLM-cache hits cost 0.

Log file: ``usage_logs/frontend_usage.jsonl`` next to this module (override
with ``USAGE_LOG_FILE``). Set ``USAGE_LOG=0|false|no|off`` to disable file
writes; in-memory per-question summaries for the UI still work.

Log line shapes::

    {"type": "call", "run_id": ..., "system": "hirag|naive_rag|hierarchical",
     "question": ..., "phase": "retrieval|answer|query", "kind":
     "chat|embedding", "model": ..., "stage": ...|null, "cached": bool,
     "input_tokens": int, "output_tokens": int, "token_source":
     "api|tiktoken", "cost": float|null, "currency": "USD"}
    {"type": "question_summary", "run_id": ..., "calls": int,
     "total_input_tokens": int, "total_output_tokens": int,
     "per_model": {...}, "unpriced_models": [...], "total_cost": float,
     "cost_complete": bool, ...}
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_BP_DIR: Path = Path(__file__).resolve().parent
_HIRAG_DIR: Path = _BP_DIR / "HiRAG"

#: Default pricing file (shared with the HiRAG insertion run logs).
DEFAULT_PRICES_FILE: Path = _HIRAG_DIR / "model_prices.yaml"

#: Default JSONL log file (one ``call`` line per LLM/embedding call plus one
#: ``question_summary`` line per frontend question).
DEFAULT_LOG_FILE: Path = _BP_DIR / "usage_logs" / "frontend_usage.jsonl"

_TIKTOKEN_ENCODING: str = "cl100k_base"
_ENCODER: Any = None
_PRICES_CACHE: dict[str, Any] | None = None
_WRITE_LOCK: threading.Lock = threading.Lock()

#: The tracker of the currently answered frontend question (None outside a
#: question flow). ContextVar so async HiRAG sub-tasks attribute correctly.
_current_tracker: contextvars.ContextVar["UsageTracker | None"] = (
    contextvars.ContextVar("frontend_usage_tracker", default=None)
)


def _logging_enabled() -> bool:
    """Return whether JSONL file writes are enabled (``USAGE_LOG`` env)."""
    return os.getenv("USAGE_LOG", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _log_file() -> Path:
    """Return the active JSONL log file (``USAGE_LOG_FILE`` override)."""
    override = os.getenv("USAGE_LOG_FILE", "").strip()
    return Path(override) if override else DEFAULT_LOG_FILE


def _get_encoder() -> Any:
    """Return the shared tiktoken encoder, built lazily on first use."""
    global _ENCODER
    if _ENCODER is None:
        import tiktoken

        _ENCODER = tiktoken.get_encoding(_TIKTOKEN_ENCODING)
    return _ENCODER


def count_tokens(text: str) -> int:
    """Count the tokens of a text with the local tiktoken encoder.

    Args:
        text: The text to encode.

    Returns:
        The number of tokens.
    """
    return len(_get_encoder().encode(text))


def count_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Count the tokens of a chat message list (message contents only).

    Args:
        messages: Chat messages as ``{"role": ..., "content": ...}`` dicts.

    Returns:
        The summed token count of all string message contents.
    """
    total = 0
    for message in messages:
        content = (
            message.get("content")
            if isinstance(message, dict)
            else getattr(message, "content", None)
        )
        if isinstance(content, str):
            total += count_tokens(content)
    return total


def load_prices() -> dict[str, Any]:
    """Load per-model token prices from the YAML pricing file (cached).

    Returns:
        Mapping with a ``currency`` string and a ``models`` mapping of
        deployment name -> ``{"input": ..., "output": ...}`` prices per 1M
        tokens. Returns an empty models mapping (everything unpriced) when
        the file does not exist.
    """
    global _PRICES_CACHE
    if _PRICES_CACHE is None:
        import yaml

        override = os.getenv("USAGE_PRICES_FILE", "").strip()
        prices_path = Path(override) if override else DEFAULT_PRICES_FILE
        raw: dict[str, Any] = {}
        if prices_path.is_file():
            with open(prices_path, encoding="utf-8") as prices_file:
                raw = yaml.safe_load(prices_file) or {}
        else:
            logger.warning(f"Usage pricing file {prices_path} not found.")
        _PRICES_CACHE = {
            "currency": str(raw.get("currency") or "USD"),
            "models": raw.get("models") or {},
        }
    return _PRICES_CACHE


def compute_cost(
    model: str, input_tokens: int, output_tokens: int, cached: bool = False
) -> float | None:
    """Compute the cost of one call from the pricing file.

    Args:
        model: Gateway deployment name of the model.
        input_tokens: Billed input tokens of the call.
        output_tokens: Billed output tokens of the call.
        cached: Whether the call was an LLM-cache hit (always free).

    Returns:
        The cost in the pricing file's currency, ``0.0`` for cached calls,
        or ``None`` when the model is missing from the pricing file.
    """
    if cached:
        return 0.0
    entry = load_prices()["models"].get(model)
    if not isinstance(entry, dict):
        return None
    try:
        input_price = float(entry.get("input", 0.0))
        output_price = float(entry.get("output", 0.0))
    except (TypeError, ValueError):
        return None
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def _extract_api_usage(response_obj: Any) -> tuple[int, int] | None:
    """Extract (input, output) token counts from an API response object.

    Handles raw ``openai`` response objects (``response.usage``), their dict
    forms, and LlamaIndex responses wrapping the provider response in
    ``.raw``.

    Args:
        response_obj: The response object of a chat/embedding/completion
            call, or ``None``.

    Returns:
        ``(input_tokens, output_tokens)`` when the response carries usage
        information, otherwise ``None``.
    """
    if response_obj is None:
        return None
    raw = response_obj
    if not hasattr(raw, "usage") and hasattr(raw, "raw"):
        raw = raw.raw
    usage = raw.get("usage") if isinstance(raw, dict) else getattr(raw, "usage", None)
    if usage is None:
        return None
    if isinstance(usage, dict):
        prompt = usage.get("prompt_tokens") or usage.get("input_tokens")
        completion = usage.get("completion_tokens") or usage.get("output_tokens")
    else:
        prompt = getattr(usage, "prompt_tokens", None) or getattr(
            usage, "input_tokens", None
        )
        completion = getattr(usage, "completion_tokens", None) or getattr(
            usage, "output_tokens", None
        )
    if not prompt and not completion:
        return None
    return int(prompt or 0), int(completion or 0)


def _build_call_record(
    *,
    run_id: str | None,
    system: str,
    question: str,
    phase: str,
    kind: str,
    model: str,
    stage: str | None,
    cached: bool,
    input_tokens: int,
    output_tokens: int,
    token_source: str,
) -> dict[str, Any]:
    """Build one JSONL ``call`` record with its computed cost."""
    prices = load_prices()
    cost = compute_cost(model, input_tokens, output_tokens, cached=cached)
    return {
        "type": "call",
        "ts": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "system": system,
        "question": question,
        "phase": phase,
        "kind": kind,
        "model": model,
        "stage": stage,
        "cached": cached,
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "token_source": token_source,
        "cost": None if cost is None else round(cost, 9),
        "currency": prices["currency"],
    }


def _append_lines(records: list[dict[str, Any]]) -> None:
    """Append records to the JSONL log file (no-op when disabled).

    Write failures are logged and swallowed: usage logging must never break
    the RAG path it observes.
    """
    if not records or not _logging_enabled():
        return
    log_path = _log_file()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with _WRITE_LOCK:
            with open(log_path, "a", encoding="utf-8") as log_file:
                for record in records:
                    log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning(f"Usage log write failed ({log_path}): {exc}")


class UsageTracker:
    """Collects the LLM/embedding calls of one answered frontend question.

    Created per question by :func:`activate_tracker`; instrumented call
    sites record into the instance via the module-level ``track_*``
    helpers while the tracker is set in the context variable.
    """

    def __init__(self, system: str, question: str, detail: str = "") -> None:
        """Initialize a tracker for one frontend question.

        Args:
            system: RAG system key (``hirag``, ``naive_rag`` or
                ``hierarchical``).
            question: The user's question text.
            detail: Free-form mode details (index, mode, top_k, ...).
        """
        self.run_id: str = (
            f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
        )
        self.system = system
        self.question = question
        self.detail = detail
        self.started = datetime.now(timezone.utc)
        self.records: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def record(
        self,
        *,
        kind: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        phase: str,
        stage: str | None = None,
        cached: bool = False,
        token_source: str = "tiktoken",
    ) -> None:
        """Record one LLM/embedding call of this question.

        Args:
            kind: ``chat`` or ``embedding``.
            model: Gateway deployment name.
            input_tokens: Input (prompt) tokens of the call.
            output_tokens: Output (completion) tokens; 0 for embeddings.
            phase: Pipeline phase, e.g. ``retrieval``, ``answer``, ``query``.
            stage: HiRAG stage name when the call ran inside one.
            cached: Whether the call was an LLM-cache hit (costs 0).
            token_source: ``api`` (provider usage fields) or ``tiktoken``.
        """
        record = _build_call_record(
            run_id=self.run_id,
            system=self.system,
            question=self.question,
            phase=phase,
            kind=kind,
            model=model,
            stage=stage,
            cached=cached,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            token_source=token_source,
        )
        with self._lock:
            self.records.append(record)

    def summary(self) -> dict[str, Any]:
        """Aggregate the recorded calls into a per-question summary.

        Returns:
            Summary dict with totals, a per-model breakdown, the list of
            unpriced models (costs excluded from ``total_cost``) and the
            ``cost_complete`` flag (False when any billed call hit an
            unpriced model).
        """
        with self._lock:
            records = list(self.records)

        per_model: dict[str, dict[str, Any]] = {}
        unpriced: set[str] = set()
        total_input = 0
        total_output = 0
        for record in records:
            model = record["model"]
            entry = per_model.setdefault(
                model,
                {
                    "calls": 0,
                    "cached_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0.0,
                },
            )
            entry["calls"] += 1
            entry["cached_calls"] += 1 if record["cached"] else 0
            entry["input_tokens"] += record["input_tokens"]
            entry["output_tokens"] += record["output_tokens"]
            if record["cost"] is None:
                if not record["cached"]:
                    unpriced.add(model)
                    entry["cost"] = None
            elif entry["cost"] is not None:
                entry["cost"] += record["cost"]
            total_input += record["input_tokens"]
            total_output += record["output_tokens"]

        total_cost = 0.0
        for model, entry in per_model.items():
            if entry["cost"] is None:
                continue
            entry["cost"] = round(entry["cost"], 9)
            total_cost += entry["cost"]

        # Models that produced the final answer: HiRAG synthesizes it in its
        # answer_generation stage; naive_rag/hierarchical answers come from
        # the frontend's phase="answer" completion call.
        answer_generation_models = sorted(
            {
                record["model"]
                for record in records
                if record.get("stage") == "answer_generation"
                or record.get("phase") == "answer"
            }
        )

        return {
            "type": "question_summary",
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "system": self.system,
            "question": self.question,
            "detail": self.detail,
            "answer_generation_models": answer_generation_models,
            "duration_s": round(
                (datetime.now(timezone.utc) - self.started).total_seconds(), 3
            ),
            "calls": len(records),
            "cached_calls": sum(1 for record in records if record["cached"]),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "per_model": per_model,
            "unpriced_models": sorted(unpriced),
            "total_cost": round(total_cost, 9),
            "cost_complete": not unpriced,
            "currency": load_prices()["currency"],
            "log_file": str(_log_file()),
        }

    def finalize(self) -> None:
        """Append all call records plus the question summary to the log."""
        try:
            _append_lines([*self.records, self.summary()])
        except Exception as exc:  # noqa: BLE001 - logging must never break
            logger.warning(f"Usage log finalize failed: {exc}")


@contextlib.contextmanager
def activate_tracker(
    system: str, question: str, detail: str = ""
) -> Iterator[UsageTracker]:
    """Activate a usage tracker for the duration of one frontend question.

    Args:
        system: RAG system key (``hirag``, ``naive_rag``, ``hierarchical``).
        question: The user's question text.
        detail: Free-form mode details recorded in the summary.

    Yields:
        The active :class:`UsageTracker`. On exit (also on exceptions) the
        tracker is finalized: all call records plus the summary line are
        appended to the JSONL log.
    """
    tracker = UsageTracker(system=system, question=question, detail=detail)
    token = _current_tracker.set(tracker)
    try:
        yield tracker
    finally:
        _current_tracker.reset(token)
        tracker.finalize()


def current_tracker() -> "UsageTracker | None":
    """Return the tracker of the currently answered question, if any."""
    return _current_tracker.get()


def track_chat(
    *,
    model: str,
    messages: list[dict[str, Any]],
    response_text: str,
    response_obj: Any = None,
    phase: str = "",
    stage: str | None = None,
    cached: bool = False,
) -> None:
    """Record one chat call into the active tracker (no-op outside one).

    Token counts come from the API ``usage`` fields when available
    (``token_source="api"``), otherwise from local tiktoken counting of the
    message contents and the response text (``token_source="tiktoken"``).

    Args:
        model: Gateway deployment name.
        messages: The full message list sent to the model.
        response_text: The returned assistant content.
        response_obj: Optional raw API response (for usage extraction).
        phase: Pipeline phase label.
        stage: HiRAG stage name when applicable.
        cached: Whether the call was an LLM-cache hit (costs 0).
    """
    tracker = _current_tracker.get()
    if tracker is None:
        return
    try:
        usage = _extract_api_usage(response_obj)
        if usage is not None:
            input_tokens, output_tokens = usage
            token_source = "api"
        else:
            input_tokens = count_messages_tokens(messages)
            output_tokens = count_tokens(response_text or "")
            token_source = "tiktoken"
        tracker.record(
            kind="chat",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            phase=phase,
            stage=stage,
            cached=cached,
            token_source=token_source,
        )
    except Exception as exc:  # noqa: BLE001 - logging must never break
        logger.warning(f"Usage tracking of chat call failed: {exc}")


def track_embedding(
    *,
    model: str,
    texts: list[str],
    response_obj: Any = None,
    phase: str = "",
) -> None:
    """Record one embedding call into the active tracker.

    Embeddings have input tokens only; the output side is always 0.

    Args:
        model: Gateway embedding deployment name.
        texts: The embedded input texts.
        response_obj: Optional raw API response (for usage extraction).
        phase: Pipeline phase label.
    """
    tracker = _current_tracker.get()
    if tracker is None:
        return
    try:
        usage = _extract_api_usage(response_obj)
        if usage is not None:
            input_tokens, _ = usage
            token_source = "api"
        else:
            input_tokens = sum(count_tokens(text) for text in texts)
            token_source = "tiktoken"
        tracker.record(
            kind="embedding",
            model=model,
            input_tokens=input_tokens,
            output_tokens=0,
            phase=phase,
            token_source=token_source,
        )
    except Exception as exc:  # noqa: BLE001 - logging must never break
        logger.warning(f"Usage tracking of embedding call failed: {exc}")


def track_completion(
    *, model: str, prompt: str, completion: Any, phase: str = ""
) -> None:
    """Record one LlamaIndex ``llm.complete`` call into the active tracker.

    Args:
        model: Deployment name of the completion model.
        prompt: The full prompt text sent to the model.
        completion: The LlamaIndex ``CompletionResponse``.
        phase: Pipeline phase label (typically ``answer``).
    """
    tracker = _current_tracker.get()
    if tracker is None:
        return
    try:
        text = getattr(completion, "text", None)
        response_text = text if isinstance(text, str) else str(completion)
        usage = _extract_api_usage(completion)
        if usage is not None:
            input_tokens, output_tokens = usage
            token_source = "api"
        else:
            input_tokens = count_tokens(prompt)
            output_tokens = count_tokens(response_text)
            token_source = "tiktoken"
        tracker.record(
            kind="chat",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            phase=phase,
            token_source=token_source,
        )
    except Exception as exc:  # noqa: BLE001 - logging must never break
        logger.warning(f"Usage tracking of completion call failed: {exc}")


def standalone_chat(
    *,
    system: str,
    question: str,
    model: str,
    messages: list[dict[str, Any]],
    response_text: str,
    response_obj: Any = None,
    phase: str = "",
) -> None:
    """Write one standalone chat call line (no active tracker needed).

    Used by the MCP-server-side retrieval path (``test_data_search``), which
    runs in a different process than the frontend and therefore cannot join
    the frontend's question tracker; correlation happens via the question
    text and timestamp.

    Args:
        system: RAG system key (``hierarchical``).
        question: The user's question text.
        model: Gateway deployment name.
        messages: The full message list sent to the model.
        response_text: The returned assistant content.
        response_obj: Optional raw API response (for usage extraction).
        phase: Pipeline phase label.
    """
    try:
        usage = _extract_api_usage(response_obj)
        if usage is not None:
            input_tokens, output_tokens = usage
            token_source = "api"
        else:
            input_tokens = count_messages_tokens(messages)
            output_tokens = count_tokens(response_text or "")
            token_source = "tiktoken"
        record = _build_call_record(
            run_id=None,
            system=system,
            question=question,
            phase=phase,
            kind="chat",
            model=model,
            stage=None,
            cached=False,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            token_source=token_source,
        )
        _append_lines([record])
    except Exception as exc:  # noqa: BLE001 - logging must never break
        logger.warning(f"Standalone usage logging of chat call failed: {exc}")


def standalone_embedding(
    *,
    system: str,
    question: str,
    model: str,
    texts: list[str],
    response_obj: Any = None,
    phase: str = "",
) -> None:
    """Write one standalone embedding call line (no active tracker needed).

    Args:
        system: RAG system key (``hierarchical``).
        question: The user's question text.
        model: Gateway embedding deployment name.
        texts: The embedded input texts.
        response_obj: Optional raw API response (for usage extraction).
        phase: Pipeline phase label.
    """
    try:
        usage = _extract_api_usage(response_obj)
        if usage is not None:
            input_tokens, _ = usage
            token_source = "api"
        else:
            input_tokens = sum(count_tokens(text) for text in texts)
            token_source = "tiktoken"
        record = _build_call_record(
            run_id=None,
            system=system,
            question=question,
            phase=phase,
            kind="embedding",
            model=model,
            stage=None,
            cached=False,
            input_tokens=input_tokens,
            output_tokens=0,
            token_source=token_source,
        )
        _append_lines([record])
    except Exception as exc:  # noqa: BLE001 - logging must never break
        logger.warning(f"Standalone usage logging of embedding call failed: {exc}")
