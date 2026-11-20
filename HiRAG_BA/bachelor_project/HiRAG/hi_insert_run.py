# PROJECT ADDED relative to HiRAG 67b2dfa; this module is absent upstream.
"""Flexible HiRAG insertion runner over the converted ADONIS corpus.

Reads process and document records from the converter's textualized JSONL
output (``processes_textualized.jsonl`` / ``documents_textualized.jsonl``)
and inserts a chosen amount of them into HiRAG via
``HiRAG.insert(list[str])``. Only each record's ``text`` content enters
HiRAG's chunking and extraction pipeline; the original source ID
(``model_id`` for processes, ``document_object_id`` for documents) is kept
out of the pipeline and persisted separately as
``<working_dir>/source_id_mapping.json`` (internal ``doc-`` content hash ->
list of original source IDs), merged across runs.

Two modes:
    - ``sample``: insert only the first N records of each file
      (``--processes`` / ``--documents``).
    - ``all``: insert every record from both files.

Every LLM and embedding call is timed and its tokens are counted locally
with tiktoken (``cl100k_base``). All calls are appended as JSON lines to a
run log file, followed by one summary line with totals. A ``model_config``
line at the start of the log (and a ``model_config`` block inside the
summary) records which deployment served which HiRAG stage: whether
``stage_models`` routing from ``config.yaml`` was active, and the
effective stage -> deployment mapping (including the classic best/cheap
fallbacks for unconfigured stages).

Embedding calls are additionally cached per (model, text) pair in a
crash-safe append-only sidecar (``embedding_cache_index.jsonl`` +
``embedding_cache_vectors.f32`` inside the working dir, see
``embedding_cache.py``), so an interrupted run resumes without
re-embedding anything already computed; ``--no-embedding-cache``
disables this. The sidecar is purely additive and never touches the
HiRAG LLM cache, so enabling it before resuming an interrupted run is
safe.

Run from this folder, e.g.::

    cd src/app/bachelor_project/HiRAG
    hirag_venv/bin/python hi_insert_run.py --mode sample --processes 3 --documents 2
    hirag_venv/bin/python hi_insert_run.py --mode all

Model names come from ``../config.yaml`` (including the per-stage
``stage_models`` routing). The API key is loaded from ``src/app/.env-dev``
automatically (its values take precedence over stale shell exports).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken
import yaml
from dotenv import load_dotenv

# Make the vendored HiRAG package and the bachelor_project llm_config
# importable regardless of the current working directory.
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_SCRIPT_DIR.parent))


# PROJECT ADDED: Load src/app/.env-dev into the process environment.
def load_env_dev() -> None:
    """Load ``src/app/.env-dev`` into the process environment.

    The file historically has CRLF line endings, which can leave ``\r``
    artifacts in header values; each value is therefore sanitized. Values
    from ``.env-dev`` win over the shell environment so stale placeholder
    exports (e.g. a 5-char ``AZURE_OPENAI_API_KEY``) cannot shadow the
    real keys.
    """
    env_path = _SCRIPT_DIR.parents[2] / "app" / ".env-dev"
    if not env_path.is_file():
        return
    load_dotenv(env_path, override=True)
    for key, value in list(os.environ.items()):
        cleaned = value.strip().strip('"').strip("'").replace("\r", "")
        if cleaned != value:
            os.environ[key] = cleaned


load_env_dev()

import httpx  # noqa: E402
from embedding_cache import EmbeddingCache  # noqa: E402
from hirag import HiRAG, QueryParam  # noqa: E402
from hirag._storage import NetworkXStorage  # noqa: E402
from hirag._utils import compute_args_hash, hirag_current_stage  # noqa: E402
from hirag.base import BaseKVStorage  # noqa: E402
from llm_config import (  # noqa: E402
    CHAT_MODEL,
    EMBEDDING_MODEL,
    OPENAI_CONFIG,
    config,
    get_api_key,
    get_stage_model_config,
)
from openai import AsyncAzureOpenAI  # noqa: E402
from source_id_mapping import (  # noqa: E402
    compute_full_doc_id,
    mapping_path,
    update_source_id_mapping,
)

# Corpus produced by adonis_converter (one textualized JSON record per line).
# parents: [0]=bachelor_project, [1]=app, [2]=src
_CORPUS_DIR = (
    _SCRIPT_DIR.parents[2]
    / "test-data"
    / "test-data_40pct"
    / "hirag_parent_only_output"
)
PROCESSES_FILE = _CORPUS_DIR / "processes_textualized.jsonl"
DOCUMENTS_FILE = _CORPUS_DIR / "documents_textualized.jsonl"

TIKTOKEN_ENCODING = "cl100k_base"

_ENCODER = tiktoken.get_encoding(TIKTOKEN_ENCODING)

# Default per-model token pricing file (sits next to hirag_run_logs/).
DEFAULT_PRICES_FILE = Path("model_prices.yaml")

# Canonical HiRAG indexing stage keys (see hirag/_op.py +
# hirag/_cluster_utils.py get_stage_llm_func call sites). Order matches the
# pipeline order. answer_generation is a query-only stage and intentionally
# not listed here; its model is configured via models.answer_generation.
STAGE_KEYS = (
    "entity_extraction",
    "relation_extraction",
    "extraction_completeness_check",
    "extraction_gleaning",
    "description_summarization",
    "cluster_summary",
    "community_report",
)

# Stages whose prompts are built from clustering output (UMAP/GMM/Leiden),
# which is non-deterministic unless seeds are configured. Within a single
# run their prompts never repeat, so the cache never hits and the per-call
# whole-file cache rewrite is pure overhead there.
DEFAULT_NO_LLM_CACHE_STAGES = ("cluster_summary", "community_report")


# PROJECT ADDED: Count the tokens of a text with the local tiktoken encoder.
def count_tokens(text: str) -> int:
    """Count the tokens of a text with the local tiktoken encoder.

    Args:
        text: The text to encode.

    Returns:
        The number of tokens.
    """
    return len(_ENCODER.encode(text))


# PROJECT ADDED: Load the textualized records of a converter JSONL file.
def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Load the textualized records of a converter JSONL file.

    Args:
        path: Path to a ``*_textualized.jsonl`` file (one JSON object per
            line, as written by ``adonis_converter``).

    Returns:
        One record dict per line, in file order.
    """
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as records_file:
        for line in records_file:
            if line.strip():
                records.append(json.loads(line))
    return records


# PROJECT ADDED: Return the original source ID of a converter record, stored verbatim.
def extract_source_id(record: dict[str, Any]) -> str:
    """Return the original source ID of a converter record, stored verbatim.

    Processes are identified by their ADONIS ``model_id`` (which the
    converter stores in the top-level ``source_id`` field, e.g.
    ``{0026db9b-...}``); documents are identified by their ADONIS
    ``document_object_id`` from the record metadata. Both are treated as
    neutral *source* IDs of the same ingestion pipeline.

    Args:
        record: One record of a ``*_textualized.jsonl`` file.

    Returns:
        The original source ID, exactly as stored in the input file.

    Raises:
        ValueError: If the record carries no usable source ID at all.
    """
    if record.get("record_type") == "DOCUMENT":
        object_id = record.get("metadata", {}).get("document_object_id")
        if object_id:
            return str(object_id)
    source_id = record.get("source_id") or record.get("record_id")
    if not source_id:
        raise ValueError(
            f"Record {record.get('record_id', '<unknown>')} has no usable source ID"
        )
    return str(source_id)


# PROJECT ADDED: Load per-model token prices from the YAML pricing file.
def load_model_prices(prices_path: Path) -> dict[str, Any]:
    """Load per-model token prices from the YAML pricing file.

    Args:
        prices_path: Path to the pricing YAML file.

    Returns:
        Mapping with a ``currency`` string and a ``models`` mapping of
        model name -> ``{"input": ..., "output": ...}`` prices per 1M
        tokens. Returns an empty models mapping (everything unpriced)
        when the file does not exist.
    """
    if not prices_path.is_file():
        print(f"Pricing file {prices_path} not found; all costs unpriced.")
        return {"currency": "USD", "models": {}}
    with open(prices_path, encoding="utf-8") as prices_file:
        raw = yaml.safe_load(prices_file) or {}
    return {
        "currency": str(raw.get("currency") or "USD"),
        "models": raw.get("models") or {},
    }


# PROJECT ADDED: Compute run costs from an aggregated token summary and prices.
def compute_costs(summary: dict[str, Any], prices: dict[str, Any]) -> dict[str, Any]:
    """Compute run costs from an aggregated token summary and prices.

    Only billed tokens cost money: ``normal`` (embedding calls) and
    ``cache_write`` (chat cache misses); cached chat tokens are free.

    Args:
        summary: The token summary built by ``RunLogger.build_summary``.
        prices: Pricing mapping as returned by ``load_model_prices``.

    Returns:
        Cost breakdown with ``per_token_type`` (input/output/cached),
        ``per_model``, ``per_stage`` (priced from the billed per-model
        token split inside each stage entry of the summary),
        ``unpriced_models`` and ``total``. A stage's cost is ``None``
        when any model that served it is missing from the pricing file.
    """
    model_prices = prices["models"]
    per_model_costs: dict[str, Any] = {}
    unpriced: list[str] = []
    input_cost = 0.0
    output_cost = 0.0
    for model, entry in summary["per_model"].items():
        billed_input = entry["normal_input_tokens"] + entry["cache_write_input_tokens"]
        billed_output = (
            entry["normal_output_tokens"] + entry["cache_write_output_tokens"]
        )
        price = model_prices.get(model)
        if not isinstance(price, dict):
            unpriced.append(model)
            per_model_costs[model] = None
            continue
        model_input_cost = billed_input * float(price.get("input", 0.0)) / 1e6
        model_output_cost = billed_output * float(price.get("output", 0.0)) / 1e6
        input_cost += model_input_cost
        output_cost += model_output_cost
        per_model_costs[model] = {
            "input": round(model_input_cost, 6),
            "output": round(model_output_cost, 6),
            "total": round(model_input_cost + model_output_cost, 6),
        }
    per_stage_costs: dict[str, Any] = {}
    for stage, stage_entry in summary.get("per_stage", {}).items():
        stage_input_cost = 0.0
        stage_output_cost = 0.0
        stage_fully_priced = True
        for model, tokens in stage_entry["per_model"].items():
            price = model_prices.get(model)
            if not isinstance(price, dict):
                stage_fully_priced = False
                continue
            stage_input_cost += (
                tokens["billed_input_tokens"] * float(price.get("input", 0.0)) / 1e6
            )
            stage_output_cost += (
                tokens["billed_output_tokens"] * float(price.get("output", 0.0)) / 1e6
            )
        per_stage_costs[stage] = (
            {
                "input": round(stage_input_cost, 6),
                "output": round(stage_output_cost, 6),
                "total": round(stage_input_cost + stage_output_cost, 6),
            }
            if stage_fully_priced
            else None
        )
    return {
        "currency": prices["currency"],
        "per_token_type": {
            "input": round(input_cost, 6),
            "output": round(output_cost, 6),
            "cached_input": 0.0,
        },
        "per_model": per_model_costs,
        "per_stage": per_stage_costs,
        "unpriced_models": unpriced,
        "total": round(input_cost + output_cost, 6),
    }


# PROJECT ADDED: Describe which deployment serves which HiRAG stage for this run.
def build_model_config(
    use_stage_models: bool, stage_models: dict[str, str], cheap_model: str
) -> dict[str, Any]:
    """Describe which deployment serves which HiRAG stage for this run.

    In stage mode, unconfigured stages fall back to the classic model
    functions inside HiRAG: ``description_summarization`` falls back to
    ``cheap_model_func``, every other stage to ``best_model_func``
    (see ``hirag/_utils.get_stage_llm_func``). The returned mapping
    records those effective fallbacks explicitly so the log is
    self-contained.

    Args:
        use_stage_models: Whether per-stage routing from the
            ``stage_models`` section of ``config.yaml`` is active.
        stage_models: Resolved stage -> deployment mapping from
            ``get_stage_model_config()`` (may be empty or partial).
        cheap_model: Deployment configured for ``cheap_model_func``
            (empty string means it falls back to the chat model).

    Returns:
        A JSON-serializable mapping with the routing mode, the effective
        model per stage (stage mode) or per classic best/cheap role, and
        the embedding deployment.
    """
    resolved_cheap = cheap_model or CHAT_MODEL
    model_config: dict[str, Any] = {
        "use_stage_models": use_stage_models,
        "chat_model": CHAT_MODEL,
        "cheap_model": resolved_cheap,
        "embedding_model": EMBEDDING_MODEL,
    }
    if use_stage_models:
        model_config["stages"] = {
            stage: stage_models.get(
                stage,
                resolved_cheap if stage == "description_summarization" else CHAT_MODEL,
            )
            for stage in STAGE_KEYS
        }
        model_config["stages_from_config"] = sorted(stage_models)
        model_config["stages_on_fallback"] = [
            stage for stage in STAGE_KEYS if stage not in stage_models
        ]
    return model_config


@dataclass
class RunLogger:
    """Append-only JSONL run log for per-call timings and token counts.

    Attributes:
        log_path: File the JSON lines are written to.
        records: In-memory copy of all logged call entries.
        prices: Optional per-model pricing mapping (from
            ``load_model_prices``); when set, the summary gains a
            ``costs`` block.
        model_config: Optional model-routing description (from
            ``build_model_config``); when set via ``log_model_config``,
            it is written as a ``model_config`` line and repeated inside
            the summary.
    """

    log_path: Path
    records: list[dict[str, Any]] = field(default_factory=list)
    prices: dict[str, Any] | None = None
    model_config: dict[str, Any] | None = None

    # PROJECT ADDED: Create the log directory so the file can always be appended.
    def __post_init__(self) -> None:
        """Create the log directory so the file can always be appended."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # PROJECT ADDED: Log one LLM or embedding API call as a JSON line.
    def log_call(
        self,
        *,
        kind: str,
        model: str,
        input_tokens: int,
        output_tokens: int = 0,
        duration_s: float,
        cache_hit: bool = False,
        stage: str | None = None,
    ) -> None:
        """Log one LLM or embedding API call as a JSON line.

        Args:
            kind: Call category, e.g. ``chat`` or ``embedding``.
            model: Deployment/model name the call was sent to.
            input_tokens: tiktoken-counted input tokens (real count, also
                on cache hits; the cache only changes the bucket, not the
                size).
            output_tokens: tiktoken-counted output tokens (chat only; on
                cache hits the size of the cached response).
            duration_s: Wall-clock duration of the API call in seconds.
            cache_hit: True when the response came from a cache (the HiRAG
                LLM cache for chat calls, the embedding cache for
                embeddings).
            stage: HiRAG pipeline stage that issued the call (from
                ``hirag._utils.hirag_current_stage``), e.g.
                ``entity_extraction``; ``None`` for embeddings or calls
                outside a stage context.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "call",
            "kind": kind,
            "model": model,
            "stage": stage,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "duration_s": round(duration_s, 3),
            "cache_hit": cache_hit,
        }
        self._write(entry)

    # PROJECT ADDED: Log the wall-clock duration of a run phase (e.g.
    def log_phase(self, name: str, duration_s: float) -> None:
        """Log the wall-clock duration of a run phase (e.g. ``insert``).

        Args:
            name: Phase identifier.
            duration_s: Wall-clock duration of the phase in seconds.
        """
        self._write(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "phase",
                "name": name,
                "duration_s": round(duration_s, 3),
            }
        )

    # PROJECT ADDED: Log the model routing configuration as one JSON line.
    def log_model_config(self, model_config: dict[str, Any]) -> None:
        """Log the model routing configuration as one JSON line.

        The mapping is also stored on the logger so ``log_summary`` can
        embed it in the final summary line.

        Args:
            model_config: Mapping from ``build_model_config`` describing
                which deployment serves which HiRAG stage or classic
                best/cheap role.
        """
        self.model_config = model_config
        self._write(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "model_config",
                **model_config,
            }
        )

    # PROJECT ADDED: Aggregate all logged calls into exclusive token buckets.
    def build_summary(self) -> dict[str, Any]:
        """Aggregate all logged calls into exclusive token buckets.

                Every call's input tokens fall into exactly one bucket:
                ``normal`` (uncached calls, e.g. embedding API calls),
                ``cached`` (cache hits, zero billed cost) or ``cache_write``
                (chat misses, which write one entry into the HiRAG LLM
                cache). Output tokens follow the same partition; embeddings
                have no output. ``billed_*`` = normal + cache_write (cache
                hits cost nothing).

                Returns:
        A summary mapping with exclusive token-bucket totals, a
                per-model breakdown using the same buckets, and a ``per_stage``
                breakdown (calls, tokens, billed tokens, total duration) keyed
                by the HiRAG pipeline stage; calls outside a stage context
                (e.g. embeddings) are grouped under ``no_stage``.
        """
        buckets = ("normal", "cached", "cache_write")

        # PROJECT ADDED: Classify a call record into its exclusive token bucket.
        def bucket_of(record: dict[str, Any]) -> str:
            """Classify a call record into its exclusive token bucket.

            Args:
                record: One logged call entry.

            Returns:
                ``normal``, ``cached`` or ``cache_write``.
            """
            if record["cache_hit"]:
                return "cached"
            if record["kind"] != "chat":
                return "normal"
            return "cache_write"

        summary: dict[str, Any] = {
            "total_calls": len(self.records),
            "cache_hits": sum(1 for r in self.records if r["cache_hit"]),
            "cache_writes": sum(
                1 for r in self.records if r["kind"] == "chat" and not r["cache_hit"]
            ),
            "total_input_tokens": sum(r["input_tokens"] for r in self.records),
            "total_output_tokens": sum(r["output_tokens"] for r in self.records),
            "total_duration_s": round(sum(r["duration_s"] for r in self.records), 3),
            "per_model": {},
        }
        for bucket in buckets:
            in_bucket = [r for r in self.records if bucket_of(r) == bucket]
            summary[f"{bucket}_input_tokens"] = sum(
                r["input_tokens"] for r in in_bucket
            )
            summary[f"{bucket}_output_tokens"] = sum(
                r["output_tokens"] for r in in_bucket
            )
        summary["billed_input_tokens"] = (
            summary["normal_input_tokens"] + summary["cache_write_input_tokens"]
        )
        summary["billed_output_tokens"] = (
            summary["normal_output_tokens"] + summary["cache_write_output_tokens"]
        )
        summary["billed_tokens"] = (
            summary["billed_input_tokens"] + summary["billed_output_tokens"]
        )
        summary["total_tokens"] = (
            summary["total_input_tokens"] + summary["total_output_tokens"]
        )
        for record in self.records:
            bucket = bucket_of(record)
            model_entry = summary["per_model"].setdefault(
                record["model"],
                {
                    "calls": 0,
                    "cache_writes": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "normal_input_tokens": 0,
                    "cached_input_tokens": 0,
                    "cache_write_input_tokens": 0,
                    "normal_output_tokens": 0,
                    "cached_output_tokens": 0,
                    "cache_write_output_tokens": 0,
                    "billed_tokens": 0,
                    "duration_s": 0.0,
                },
            )
            model_entry["calls"] += 1
            if bucket == "cache_write":
                model_entry["cache_writes"] += 1
            model_entry["input_tokens"] += record["input_tokens"]
            model_entry["output_tokens"] += record["output_tokens"]
            model_entry[f"{bucket}_input_tokens"] += record["input_tokens"]
            model_entry[f"{bucket}_output_tokens"] += record["output_tokens"]
            if bucket != "cached":
                model_entry["billed_tokens"] += (
                    record["input_tokens"] + record["output_tokens"]
                )
            model_entry["duration_s"] = round(
                model_entry["duration_s"] + record["duration_s"], 3
            )
        summary["per_stage"] = {}
        for record in self.records:
            bucket = bucket_of(record)
            stage_name = record.get("stage") or "no_stage"
            stage_entry = summary["per_stage"].setdefault(
                stage_name,
                {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "billed_input_tokens": 0,
                    "billed_output_tokens": 0,
                    "billed_tokens": 0,
                    "cached_input_tokens": 0,
                    "cached_output_tokens": 0,
                    "duration_s": 0.0,
                    "per_model": {},
                },
            )
            stage_entry["calls"] += 1
            stage_entry["input_tokens"] += record["input_tokens"]
            stage_entry["output_tokens"] += record["output_tokens"]
            if bucket == "cached":
                stage_entry["cached_input_tokens"] += record["input_tokens"]
                stage_entry["cached_output_tokens"] += record["output_tokens"]
            else:
                stage_entry["billed_input_tokens"] += record["input_tokens"]
                stage_entry["billed_output_tokens"] += record["output_tokens"]
                stage_entry["billed_tokens"] += (
                    record["input_tokens"] + record["output_tokens"]
                )
                stage_model_entry = stage_entry["per_model"].setdefault(
                    record["model"],
                    {"billed_input_tokens": 0, "billed_output_tokens": 0},
                )
                stage_model_entry["billed_input_tokens"] += record["input_tokens"]
                stage_model_entry["billed_output_tokens"] += record["output_tokens"]
            stage_entry["duration_s"] = round(
                stage_entry["duration_s"] + record["duration_s"], 3
            )
        if self.prices is not None:
            summary["costs"] = compute_costs(summary, self.prices)
        return summary

    # PROJECT ADDED: Write the final summary line, extended with run metadata.
    def log_summary(self, extra: dict[str, Any]) -> dict[str, Any]:
        """Write the final summary line, extended with run metadata.

        Args:
            extra: Run metadata (mode, record counts, phase timings).

        Returns:
            The full summary that was written.
        """
        summary = {"type": "summary", **extra, **self.build_summary()}
        if self.model_config is not None:
            summary["model_config"] = self.model_config
        self._write(summary)
        return summary

    # PROJECT ADDED: Append one JSON line to the log file and keep it in memory.
    def _write(self, entry: dict[str, Any]) -> None:
        """Append one JSON line to the log file and keep it in memory.

        Args:
            entry: The JSON-serializable log entry.
        """
        if entry.get("type") == "call":
            self.records.append(entry)
        with open(self.log_path, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")


# PROJECT ADDED: Build a HiRAG chat model function with caching, timing and counting.
def make_tracked_chat_func(
    model_name: str,
    run_logger: RunLogger,
    client: AsyncAzureOpenAI,
    no_cache_stages: frozenset[str] | None = None,
) -> Any:
    """Build a HiRAG chat model function with caching, timing and counting.

    Mirrors the caching behavior of ``hi_Search_openai.py`` (cache key =
    model + messages) and additionally measures wall-clock time and counts
    input/output tokens locally with tiktoken.

    Args:
        model_name: Azure gateway deployment name for this function.
        run_logger: Logger receiving one entry per call.
        client: Shared Azure OpenAI async client.
        no_cache_stages: Stage names that skip the LLM response cache
            entirely (no lookup, no upsert, no per-call flush). Used for
            stages whose prompts depend on non-deterministic clustering
            output and therefore never hit the cache anyway.

    Returns:
        An async model function with the standard HiRAG signature.
    """
    skip_cache_stages = no_cache_stages or frozenset()

    # PROJECT ADDED: Reuse cached model replies or call the gateway, logging the pipeline stage,
    # token counts, and duration.
    async def chat_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list | None = None,
        **kwargs: Any,
    ) -> str:
        # Reuse cached chat replies or request new ones while logging stage usage and timing.
        hashing_kv: BaseKVStorage | None = kwargs.pop("hashing_kv", None)
        # HiRAG's stage resolver (get_stage_llm_func) wraps every stage call
        # so this names the pipeline stage that issued the call (None when
        # the func was invoked outside a stage context).
        stage = hirag_current_stage.get()
        # No-cache stages skip the lookup AND the upsert/flush below: their
        # prompts never repeat within a run, and skipping the per-call
        # whole-file cache rewrite measurably speeds up those stages.
        use_cache = hashing_kv is not None and stage not in skip_cache_stages
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages or [])
        messages.append({"role": "user", "content": prompt})

        if use_cache:
            args_hash = compute_args_hash(model_name, messages)
            cached = await hashing_kv.get_by_id(args_hash)
            if cached is not None:
                # Real token sizes are logged even on cache hits; the
                # summary partitions them into the ``cached`` bucket so
                # they count as saved, not billed.
                run_logger.log_call(
                    kind="chat",
                    model=model_name,
                    input_tokens=sum(
                        count_tokens(message["content"]) for message in messages
                    ),
                    output_tokens=count_tokens(cached["return"]),
                    duration_s=0.0,
                    cache_hit=True,
                    stage=stage,
                )
                return cached["return"]

        input_tokens = sum(count_tokens(message["content"]) for message in messages)

        # The GPT-5 gateway deployments require max_completion_tokens, not
        # the legacy max_tokens parameter HiRAG passes through **kwargs.
        if "max_tokens" in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")

        start = time.perf_counter()
        response = await client.chat.completions.create(
            model=model_name, messages=messages, **kwargs
        )
        duration_s = time.perf_counter() - start

        content = response.choices[0].message.content
        run_logger.log_call(
            kind="chat",
            model=model_name,
            input_tokens=input_tokens,
            output_tokens=count_tokens(content),
            duration_s=duration_s,
            stage=stage,
        )

        if use_cache:
            await hashing_kv.upsert(
                {args_hash: {"return": content, "model": model_name}}
            )
            # Flush the LLM cache after every call (mirrors the vendored
            # openai_complete_if_cache in hirag/_llm.py) so an interrupted
            # insertion run keeps all completed calls and a restart on the
            # same working dir resumes at LLM-call granularity instead of
            # redoing every call.
            await hashing_kv.index_done_callback()
        return content

    # Lets the triplet extraction logger (triplets.triplet_logger) record the
    # real deployment name for each stage's calls.
    chat_func.model_name = model_name
    return chat_func


# PROJECT ADDED: Build the HiRAG embedding function with timing and token counting.
def make_tracked_embedding_func(
    model_name: str,
    run_logger: RunLogger,
    client: AsyncAzureOpenAI,
    cache: EmbeddingCache | None = None,
) -> Any:
    """Build the HiRAG embedding function with timing and token counting.

    Args:
        model_name: Azure gateway embedding deployment name.
        run_logger: Logger receiving one entry per call.
        client: Shared Azure OpenAI async client.
        cache: Optional crash-safe embedding cache (see
            ``embedding_cache.py``). When given, only cache misses are sent
            to the gateway and every returned vector is flushed to disk
            immediately, so an interrupted run resumes without
            re-embedding; cache hits are logged with ``cache_hit=True`` and
            land in the zero-cost ``cached`` summary bucket.

    Returns:
        An async embedding function with HiRAG's ``EmbeddingFunc``
        attributes (``embedding_dim``, ``max_token_size``).
    """

    # PROJECT ADDED: Reuse cached vectors, embed missing texts, and record embedding tokens and
    # timing.
    async def embedding_func(texts: list[str]) -> np.ndarray:
        # Reuse cached vectors, embed cache misses, and log embedding usage.
        token_counts = [count_tokens(text) for text in texts]
        cached_vectors = (
            [cache.lookup(text) for text in texts]
            if cache is not None
            else [None] * len(texts)
        )
        # Only unique misses are sent to the gateway; duplicate texts inside
        # one batch reuse the freshly embedded vector.
        miss_texts: list[str] = []
        miss_seen: set[str] = set()
        for text, cached_vector in zip(texts, cached_vectors):
            if cached_vector is None and text not in miss_seen:
                miss_seen.add(text)
                miss_texts.append(text)
        new_vectors: dict[str, np.ndarray] = {}
        if miss_texts:
            start = time.perf_counter()
            response = await client.embeddings.create(
                model=model_name, input=miss_texts, encoding_format="float"
            )
            duration_s = time.perf_counter() - start
            run_logger.log_call(
                kind="embedding",
                model=model_name,
                input_tokens=sum(count_tokens(text) for text in miss_texts),
                duration_s=duration_s,
            )
            fresh = np.array([dp.embedding for dp in response.data])
            if cache is not None:
                cache.add(miss_texts, fresh)
            new_vectors = dict(zip(miss_texts, fresh))
        hit_tokens = sum(
            tokens
            for tokens, cached_vector in zip(token_counts, cached_vectors)
            if cached_vector is not None
        )
        if hit_tokens:
            run_logger.log_call(
                kind="embedding",
                model=model_name,
                input_tokens=hit_tokens,
                duration_s=0.0,
                cache_hit=True,
            )
        return np.array(
            [
                cached_vector if cached_vector is not None else new_vectors[text]
                for text, cached_vector in zip(texts, cached_vectors)
            ]
        )

    # HiRAG requires these attributes on the embedding function.
    embedding_func.embedding_dim = config["model_params"]["openai_embedding_dim"]
    embedding_func.max_token_size = config["model_params"]["max_token_size"]
    return embedding_func


# PROJECT ADDED: Parse the command-line arguments of the runner.
def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments of the runner.

    Returns:
        The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Insert a sample or the full converted ADONIS corpus into HiRAG "
            "and log per-call timings and tiktoken token counts."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["sample", "all"],
        default="sample",
        help="sample = first N records of each file; all = everything (default: sample)",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=5,
        help="number of process records in sample mode (default: 5)",
    )
    parser.add_argument(
        "--documents",
        type=int,
        default=5,
        help="number of document records in sample mode (default: 5)",
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=None,
        help=(
            "HiRAG index directory (default: "
            "./hirag_index_samples/hirag_index_sample for --mode sample, "
            "./hirag_index_samples/hirag_index_complete for --mode all)"
        ),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help=(
            "run log path (default: "
            "./hirag_run_logs/run_<timestamp>_<working-dir-name>.jsonl)"
        ),
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help=(
            "directory containing processes_textualized.jsonl / "
            "documents_textualized.jsonl (default: the data-prep "
            "hirag_parent_only_output directory; pass the "
            "hirag_parent_only_output_with_activities directory to index "
            "the activity-enriched corpus)"
        ),
    )
    parser.add_argument(
        "--prices-file",
        type=Path,
        default=DEFAULT_PRICES_FILE,
        help=f"per-model token pricing YAML (default: {DEFAULT_PRICES_FILE})",
    )
    parser.add_argument(
        "--no-embedding-cache",
        action="store_true",
        help=(
            "disable the crash-safe embedding cache sidecar in the working "
            "dir (enabled by default)"
        ),
    )
    parser.add_argument(
        "--no-cache-stages",
        type=str,
        default=None,
        help=(
            "comma-separated HiRAG stages that skip the LLM response cache "
            "(lookup AND write); pass an empty string to cache all stages. "
            "Default (flag omitted) is automatic, driven by "
            "hirag.static_random_seed in config.yaml: when true, ALL stages "
            "are cached (seeded clustering reproduces identical "
            "cluster_summary/community_report prompts on resume, so their "
            "cache can hit); when false, "
            f"{','.join(DEFAULT_NO_LLM_CACHE_STAGES)} are skipped because "
            "their prompts depend on non-deterministic clustering and never "
            "repeat, so caching them is pure overhead."
        ),
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="optional query to run after insertion (hi mode)",
    )
    return parser.parse_args()


# PROJECT ADDED: Run the insertion with the chosen sample size and write the log.
def main() -> None:
    """Run the insertion with the chosen sample size and write the log."""
    args = parse_args()

    corpus_dir = args.corpus_dir or _CORPUS_DIR
    process_records = load_jsonl_records(corpus_dir / PROCESSES_FILE.name)
    document_records = load_jsonl_records(corpus_dir / DOCUMENTS_FILE.name)

    if args.mode == "all":
        selected_records = process_records + document_records
    else:
        selected_records = (
            process_records[: args.processes] + document_records[: args.documents]
        )
    if not selected_records:
        raise ValueError("No records selected; check the corpus files and counts.")

    # Only the record content enters HiRAG; the original source IDs are
    # tracked separately in the mapping sidecar file, never in the text.
    selected = [record["text"] for record in selected_records]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    working_dir = args.working_dir or Path("./hirag_index_samples") / (
        f"hirag_index_{args.mode}" if args.mode == "sample" else "hirag_index_complete"
    )
    log_file = args.log_file or Path("hirag_run_logs") / (
        f"run_{timestamp}_{working_dir.name}.jsonl"
    )
    run_logger = RunLogger(
        log_path=log_file, prices=load_model_prices(args.prices_file)
    )

    api_key = get_api_key("openai")
    # Retry/timeout hardening: the SDK default (max_retries=2, httpx connect
    # timeout 5s) lets a single transient gateway connect blip abort a
    # multi-hour run; give connects 30s slack and retry more often.
    client = AsyncAzureOpenAI(
        azure_endpoint=OPENAI_CONFIG["base_url"],
        api_key=api_key,
        api_version=OPENAI_CONFIG.get("api_version", "2024-10-21"),
        max_retries=6,
        timeout=httpx.Timeout(600.0, connect=30.0),
        default_headers={
            "Content-Type": "application/json",
            "X-Gateway-Api-Key": api_key,
        },
    )

    embedding_cache: EmbeddingCache | None = None
    if not args.no_embedding_cache:
        embedding_cache = EmbeddingCache(
            working_dir,
            dim=int(config["model_params"]["openai_embedding_dim"]),
            model=EMBEDDING_MODEL,
        )
        print(
            f"Embedding cache enabled: {len(embedding_cache)} cached vectors "
            f"({embedding_cache.index_path})"
        )

    # --no-cache-stages omitted (None) = automatic mode driven by
    # hirag.static_random_seed: seeded clustering reproduces identical
    # cluster_summary/community_report prompts on resume, so caching those
    # stages is worthwhile only when the seed is on.
    static_random = bool(config.get("hirag", {}).get("static_random_seed", False))
    if args.no_cache_stages is None:
        no_cache_stages = (
            frozenset() if static_random else frozenset(DEFAULT_NO_LLM_CACHE_STAGES)
        )
        no_cache_source = f"auto (hirag.static_random_seed={static_random})"
    else:
        no_cache_stages = frozenset(
            stage.strip() for stage in args.no_cache_stages.split(",") if stage.strip()
        )
        no_cache_source = "--no-cache-stages"
    unknown_stages = no_cache_stages - frozenset(STAGE_KEYS)
    if unknown_stages:
        print(
            "Warning: unknown stage names in --no-cache-stages: "
            f"{sorted(unknown_stages)} (known: {list(STAGE_KEYS)})"
        )
    if no_cache_stages:
        print(
            "LLM cache disabled for stages: "
            f"{', '.join(sorted(no_cache_stages))} "
            f"(cache lookup and write skipped; {no_cache_source})"
        )
    else:
        print(f"LLM cache enabled for all stages ({no_cache_source})")

    chat_func = make_tracked_chat_func(
        CHAT_MODEL, run_logger, client, no_cache_stages=no_cache_stages
    )
    # Classic HiRAG split: cheap_model_func (used for description
    # summarization) can point at a cheaper deployment via
    # models.cheap_model in config.yaml; empty = same as the best model.
    cheap_model = str(config["models"].get("cheap_model") or "").strip()
    cheap_chat_func = (
        chat_func
        if not cheap_model or cheap_model == CHAT_MODEL
        else make_tracked_chat_func(
            cheap_model, run_logger, client, no_cache_stages=no_cache_stages
        )
    )
    use_stage_models, stage_models = get_stage_model_config()
    model_config = build_model_config(use_stage_models, stage_models, cheap_model)
    run_logger.log_model_config(model_config)
    if use_stage_models:
        print("Stage model routing enabled:")
        for stage in STAGE_KEYS:
            print(f"  {stage}: {model_config['stages'][stage]}")
    else:
        print(
            "Stage model routing disabled "
            f"(best={CHAT_MODEL}, cheap={model_config['cheap_model']}, "
            f"embedding={EMBEDDING_MODEL})"
        )

    # PROJECT ADDED: Create a tracked chat function pinned to one stage deployment.
    def stage_factory(model_name: str) -> Any:
        """Create a tracked chat function pinned to one stage deployment.

        Args:
            model_name: Resolved Azure gateway deployment name.

        Returns:
            The tracked async model function for the stage.
        """
        return make_tracked_chat_func(
            model_name, run_logger, client, no_cache_stages=no_cache_stages
        )

    graph_func = HiRAG(
        working_dir=str(working_dir),
        enable_llm_cache=config["hirag"]["enable_llm_cache"],
        embedding_func=make_tracked_embedding_func(
            EMBEDDING_MODEL, run_logger, client, cache=embedding_cache
        ),
        best_model_func=chat_func,
        cheap_model_func=cheap_chat_func,
        use_stage_models=use_stage_models,
        stage_models=stage_models,
        stage_model_func_factory=stage_factory,
        enable_hierachical_mode=config["hirag"]["enable_hierachical_mode"],
        embedding_batch_num=config["hirag"]["embedding_batch_num"],
        embedding_func_max_async=config["hirag"]["embedding_func_max_async"],
        enable_naive_rag=config["hirag"]["enable_naive_rag"],
        graph_storage_cls=NetworkXStorage,
    )

    print(f"Corpus directory: {corpus_dir}")
    print(f"Inserting {len(selected)} records (mode={args.mode}) ...")
    insert_start = time.perf_counter()
    graph_func.insert(selected)
    insert_duration = time.perf_counter() - insert_start
    run_logger.log_phase("insert", insert_duration)
    print(f"Insert finished in {insert_duration:.1f}s.")

    # Persist/merge the internal doc hash -> original source ID mapping. All
    # selected records contribute, even ones HiRAG skipped as already
    # inserted, so a new source with identical content still gets linked to
    # the existing internal doc ID.
    mapping_pairs = [
        (compute_full_doc_id(record["text"]), extract_source_id(record))
        for record in selected_records
    ]
    source_mapping, new_links = update_source_id_mapping(working_dir, mapping_pairs)
    mapping_file = mapping_path(working_dir)
    print(
        f"Source ID mapping: {len(source_mapping)} internal docs, "
        f"{new_links} new source links -> {mapping_file}"
    )

    query_duration = 0.0
    if args.query:
        query_start = time.perf_counter()
        answer = graph_func.query(args.query, param=QueryParam(mode="hi"))
        query_duration = time.perf_counter() - query_start
        run_logger.log_phase("query", query_duration)
        print(f"Answer: {answer}")

    summary = run_logger.log_summary(
        {
            "mode": args.mode,
            "process_records": len(process_records)
            if args.mode == "all"
            else min(args.processes, len(process_records)),
            "document_records": len(document_records)
            if args.mode == "all"
            else min(args.documents, len(document_records)),
            "insert_duration_s": round(insert_duration, 3),
            "query_duration_s": round(query_duration, 3),
            "working_dir": str(working_dir),
            "source_id_mapping": str(mapping_file),
            "source_id_mapping_docs": len(source_mapping),
            "source_id_mapping_new_links": new_links,
        }
    )
    print(f"Log written to {log_file}")
    print(json.dumps(summary, indent=2))
    costs = summary.get("costs")
    if costs is not None:
        currency = costs["currency"]
        token_type_costs = costs["per_token_type"]
        print(
            f"Cost ({currency}): "
            f"input={token_type_costs['input']:.6f}, "
            f"output={token_type_costs['output']:.6f}, "
            f"total={costs['total']:.6f}"
        )
        if costs["unpriced_models"]:
            print(
                "Unpriced models (missing in pricing file): "
                + ", ".join(costs["unpriced_models"])
            )
    per_stage = summary.get("per_stage")
    if per_stage:
        print("Per-stage totals:")
        for stage, stage_entry in per_stage.items():
            line = (
                f"  {stage}: calls={stage_entry['calls']}, "
                f"tokens={stage_entry['input_tokens'] + stage_entry['output_tokens']}, "
                f"billed={stage_entry['billed_tokens']}, "
                f"duration={stage_entry['duration_s']:.3f}s"
            )
            stage_cost = (costs or {}).get("per_stage", {}).get(stage)
            if isinstance(stage_cost, dict):
                line += f", cost={stage_cost['total']:.6f} {(costs or {})['currency']}"
            print(line)


if __name__ == "__main__":
    main()
