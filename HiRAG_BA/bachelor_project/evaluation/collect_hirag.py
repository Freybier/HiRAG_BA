"""Collect HiRAG answers and the exact answer-LLM context for RAGAS scoring.

Runs every question of a validated dataset against an EXISTING HiRAG index
(nothing is inserted) and writes a run directory in the shared run format
(``run_format.py``). Scoring happens separately with ``score.py`` in the
evaluation venv, so HiRAG is never re-run for scoring.

Run with the HiRAG venv::

    cd src/app/bachelor_project/HiRAG
    hirag_venv/bin/python ../evaluation/collect_hirag.py \\
        --dataset /path/to/questions.jsonl --mode hi

Contexts come from the opt-in ``QueryParam.capture`` hook in ``hirag/_op.py``:
the exact context string the answer LLM received plus its ordered sections.
Original ADONIS source IDs are resolved via ``kv_store_text_chunks.json`` and
``source_id_mapping.json`` of the index.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

_BACHELOR_DIR = Path(__file__).resolve().parent.parent
if str(_BACHELOR_DIR) not in sys.path:
    sys.path.insert(0, str(_BACHELOR_DIR))

from evaluation.common import (  # noqa: E402
    BACHELOR_DIR,
    HIRAG_DIR,
    code_fingerprint,
    ensure_import_paths,
    fingerprint_directory,
    git_info,
    load_env_dev,
    package_versions,
    read_json,
)

load_env_dev()
ensure_import_paths()

import numpy as np  # noqa: E402
from hirag import HiRAG, QueryParam  # noqa: E402
from hirag._storage import NetworkXStorage  # noqa: E402
from hirag._utils import always_get_an_event_loop, compute_args_hash  # noqa: E402
from llm_config import (  # noqa: E402
    CHAT_MODEL,
    EMBEDDING_MODEL,
    OPENAI_CONFIG,
    config,
    get_answer_generation_model,
    get_api_key,
    get_stage_model_config,
)
from openai import AsyncAzureOpenAI  # noqa: E402
from source_id_mapping import load_source_id_mapping  # noqa: E402

from evaluation.collect_runner import (  # noqa: E402
    default_out_dir,
    merge_usage,
    run_collection,
)
from evaluation.dataset import EvalQuestion  # noqa: E402
from evaluation.run_format import (  # noqa: E402
    ADONIS_ID_GRANULARITY,
    build_contexts_from_capture,
)

QUERY_MODES = ("hi", "hi_bridge", "hi_global", "hi_nobridge", "naive")
DEFAULT_WORKING_DIR = HIRAG_DIR / "hirag_index_samples" / "hirag_index_sample_20p_4d"
TEXT_CHUNKS_FILE = "kv_store_text_chunks.json"

# Per-question usage counters, reset by the answer function before each query.
_USAGE: dict[str, Any] = {}


# Add one call to the per-question usage counters.
def _count(
    kind: str, model: str, prompt: int, completion: int, cache_hit: bool
) -> None:
    """Add one call to the per-question usage counters."""
    merge_usage(
        _USAGE,
        {
            kind: {
                model: {
                    "calls": 1,
                    "cache_hits": int(cache_hit),
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                }
            }
        },
    )


# Build a HiRAG chat function (same caching as hi_insert_run.py).
def make_chat_func(model_name: str, client: AsyncAzureOpenAI) -> Any:
    """Build a HiRAG chat function (same caching as ``hi_insert_run.py``).

    Token counts come from the provider ``usage`` field; cache hits count as
    calls with zero billed tokens.

    Args:
        model_name: Azure gateway deployment name.
        client: Shared async Azure OpenAI client.

    Returns:
        Async model function with the standard HiRAG signature.
    """

    # Reuse cached gateway replies or request new ones while recording per-question token usage.
    async def chat_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list | None = None,
        **kwargs: Any,
    ) -> str:
        # Reuse cached gateway replies or request new ones while recording collection token usage.
        hashing_kv = kwargs.pop("hashing_kv", None)
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages or [])
        messages.append({"role": "user", "content": prompt})

        if hashing_kv is not None:
            args_hash = compute_args_hash(model_name, messages)
            cached = await hashing_kv.get_by_id(args_hash)
            if cached is not None:
                _count("chat", model_name, 0, 0, cache_hit=True)
                return cached["return"]

        # GPT-5 gateway deployments require max_completion_tokens.
        if "max_tokens" in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        response = await client.chat.completions.create(
            model=model_name, messages=messages, **kwargs
        )
        usage = response.usage
        _count(
            "chat",
            model_name,
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
            cache_hit=False,
        )
        content = response.choices[0].message.content
        if hashing_kv is not None:
            await hashing_kv.upsert(
                {args_hash: {"return": content, "model": model_name}}
            )
        return content

    chat_func.model_name = model_name
    return chat_func


# Build the HiRAG embedding function with provider usage counting.
def make_embedding_func(model_name: str, client: AsyncAzureOpenAI) -> Any:
    """Build the HiRAG embedding function with provider usage counting."""

    # Request gateway embeddings and record their token usage for the current evaluation question.
    async def embedding_func(texts: list[str]) -> np.ndarray:
        # Request gateway embeddings and record token usage for the collected query.
        response = await client.embeddings.create(
            model=model_name, input=texts, encoding_format="float"
        )
        usage = response.usage
        _count(
            "embedding",
            model_name,
            getattr(usage, "prompt_tokens", 0) or 0,
            0,
            cache_hit=False,
        )
        return np.array([dp.embedding for dp in response.data])

    embedding_func.embedding_dim = config["model_params"]["openai_embedding_dim"]
    embedding_func.max_token_size = config["model_params"]["max_token_size"]
    return embedding_func


# Construct HiRAG on an existing index with the central model config.
def build_hirag(working_dir: Path) -> tuple[HiRAG, dict[str, Any]]:
    """Construct HiRAG on an existing index with the central model config.

    Args:
        working_dir: Existing HiRAG index directory.

    Returns:
        The HiRAG instance and a description of the model wiring.
    """
    api_key = get_api_key("openai")
    client = AsyncAzureOpenAI(
        azure_endpoint=OPENAI_CONFIG["base_url"],
        api_key=api_key,
        api_version=OPENAI_CONFIG.get("api_version", "2024-10-21"),
        default_headers={
            "Content-Type": "application/json",
            "X-Gateway-Api-Key": api_key,
        },
    )
    chat_func = make_chat_func(CHAT_MODEL, client)
    cheap_model = str(config["models"].get("cheap_model") or "").strip()
    cheap_func = (
        chat_func
        if not cheap_model or cheap_model == CHAT_MODEL
        else make_chat_func(cheap_model, client)
    )
    use_stage_models, stage_models = get_stage_model_config()
    hirag_cfg = config["hirag"]
    graph = HiRAG(
        working_dir=str(working_dir),
        always_create_working_dir=False,
        enable_llm_cache=hirag_cfg["enable_llm_cache"],
        embedding_func=make_embedding_func(EMBEDDING_MODEL, client),
        best_model_func=chat_func,
        cheap_model_func=cheap_func,
        use_stage_models=use_stage_models,
        stage_models=stage_models,
        stage_model_func_factory=lambda name: make_chat_func(name, client),
        enable_hierachical_mode=hirag_cfg["enable_hierachical_mode"],
        embedding_batch_num=hirag_cfg["embedding_batch_num"],
        embedding_func_max_async=hirag_cfg["embedding_func_max_async"],
        enable_naive_rag=hirag_cfg["enable_naive_rag"],
        graph_storage_cls=NetworkXStorage,
    )
    models = {
        "chat_model": CHAT_MODEL,
        "cheap_model": cheap_model or None,
        "embedding_model": EMBEDDING_MODEL,
        "use_stage_models": use_stage_models,
        "stage_models": stage_models if use_stage_models else {},
        "answer_model": get_answer_generation_model(),
        "enable_llm_cache": hirag_cfg["enable_llm_cache"],
    }
    return graph, models


# Parse the command-line arguments.
def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True, help="question JSONL")
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=DEFAULT_WORKING_DIR,
        help=f"existing HiRAG index (default: {DEFAULT_WORKING_DIR.relative_to(BACHELOR_DIR)})",
    )
    parser.add_argument("--mode", choices=QUERY_MODES, default="hi")
    parser.add_argument("--top-k", type=int, default=None, help="QueryParam.top_k")
    parser.add_argument("--top-m", type=int, default=None, help="QueryParam.top_m")
    parser.add_argument("--level", type=int, default=None, help="QueryParam.level")
    parser.add_argument(
        "--response-type", default=None, help="QueryParam.response_type"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="seconds per question (default 600)",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="run directory")
    parser.add_argument("--resume", action="store_true", help="continue --out-dir")
    parser.add_argument(
        "--limit", type=int, default=None, help="first N questions only"
    )
    return parser.parse_args()


# Build the configured HiRAG instance and collect answers and exact contexts for the selected
# questions.
def main() -> int:
    """Collect a HiRAG run."""
    args = parse_args()
    working_dir = args.working_dir.resolve()
    if not working_dir.is_dir():
        raise SystemExit(f"HiRAG index not found: {working_dir}")
    chunks_path = working_dir / TEXT_CHUNKS_FILE
    if not chunks_path.is_file():
        raise SystemExit(f"Missing {chunks_path}; is this a HiRAG index?")

    chunk_to_doc = {
        chunk_id: chunk.get("full_doc_id")
        for chunk_id, chunk in read_json(chunks_path).items()
    }
    doc_to_sources = load_source_id_mapping(working_dir)
    if not doc_to_sources:
        print(
            "WARNING: no source_id_mapping.json in the index; retrieved_context_ids "
            "will be empty and ID-based metrics meaningless.",
            file=sys.stderr,
        )

    param_overrides = {
        key: value
        for key, value in {
            "top_k": args.top_k,
            "top_m": args.top_m,
            "level": args.level,
            "response_type": args.response_type,
        }.items()
        if value is not None
    }
    effective_param = vars(QueryParam(mode=args.mode, **param_overrides)).copy()
    effective_param.pop("capture", None)

    graph, models = build_hirag(working_dir)
    loop = always_get_an_event_loop()

    # Query HiRAG with context capture and return its answer, retrieved evidence, settings, and
    # token usage.
    def answer(question: EvalQuestion) -> dict[str, Any]:
        # Query HiRAG and return its answer, captured contexts, configuration, and usage.
        _USAGE.clear()
        capture: dict[str, Any] = {}
        param = QueryParam(mode=args.mode, capture=capture, **param_overrides)
        # Exceptions/timeouts propagate; the runner records status "error"
        # without any partially captured context.
        response = loop.run_until_complete(
            asyncio.wait_for(graph.aquery(question.user_input, param), args.timeout)
        )
        fields = build_contexts_from_capture(capture, chunk_to_doc, doc_to_sources)
        return {
            "status": capture.get("status") or "answered",
            "response": response,
            "context_text": capture.get("context_text"),
            "context_builder": capture.get("context_builder"),
            "system_prompt": capture.get("system_prompt"),
            "query_config": {"mode": capture.get("mode", args.mode)},
            "usage": {kind: dict(models_) for kind, models_ in _USAGE.items()},
            **fields,
        }

    run_config = {
        "working_dir": str(working_dir),
        # In run_config so --resume refuses a changed index.
        "index_fingerprint": fingerprint_directory(working_dir),
        "query_param": effective_param,
        "models": models,
        "timeout_s": args.timeout,
    }
    metadata_extra = {
        "code_fingerprint": code_fingerprint(
            [
                HIRAG_DIR / "hirag" / "_op.py",
                HIRAG_DIR / "hirag" / "hirag.py",
                HIRAG_DIR / "hirag" / "base.py",
                HIRAG_DIR / "hirag" / "prompt.py",
                BACHELOR_DIR / "config.yaml",
                Path(__file__),
            ]
        ),
        "git": git_info(BACHELOR_DIR),
        "packages": package_versions(["openai", "numpy", "networkx"]),
        # Clustering/graph iteration order can depend on hash randomization.
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }
    return run_collection(
        system="hirag",
        dataset_path=args.dataset,
        out_dir=args.out_dir
        or default_out_dir("hirag", f"{working_dir.name}_{args.mode}"),
        run_config=run_config,
        metadata_extra=metadata_extra,
        id_granularity=ADONIS_ID_GRANULARITY,
        answer_fn=answer,
        resume=args.resume,
        limit=args.limit,
    )


if __name__ == "__main__":
    sys.exit(main())
