"""Collect parent-child test-data RAG answers for RAGAS scoring.

Retrieval is ``app.test_data_search.search_test_data`` — the function the
MCP ``search_adonis`` tool returns directly when ``ADONIS_TEST_DATA_SEARCH``
is on, i.e. the Streamlit "Hierarchisch" mode — called in-process instead of
through the MCP server. The answer prompt is the byte-identical Streamlit
``generate_answer`` prompt (see ``streamlit_answer.py`` for the answer-model
deviation).

Run with the src venv::

    cd src
    .venv/bin/python app/bachelor_project/evaluation/collect_test_data.py \\
        --dataset /path/to/questions.jsonl

Env defaults set by this script when unset: ``ADONIS_TEST_DATA_DIR`` =
``src/test-data/test-data_40pct``, ``ADONIS_TEST_DATA_VECTOR_SEARCH=true``
(``poe ui`` default). All relevant flags are recorded in the run config.

``retrieved_context_ids`` stay empty (``id_granularity: null``), see the
FUTURE TODO on a common input-data structure in the README.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

_BACHELOR_DIR = Path(__file__).resolve().parent.parent
if str(_BACHELOR_DIR) not in sys.path:
    sys.path.insert(0, str(_BACHELOR_DIR))

from evaluation.common import (  # noqa: E402
    APP_DIR,
    BACHELOR_DIR,
    SRC_DIR,
    code_fingerprint,
    git_info,
    load_env_dev,
    package_versions,
)

load_env_dev()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_DATA_DIR = SRC_DIR / "test-data" / "test-data_40pct"
os.environ.setdefault("ADONIS_TEST_DATA_DIR", str(DEFAULT_DATA_DIR))
os.environ.setdefault("ADONIS_TEST_DATA_VECTOR_SEARCH", "true")

#: Env flags that change test-data retrieval; recorded in the run config.
ENV_FLAGS = (
    "ADONIS_TEST_DATA_DIR",
    "ADONIS_TEST_DATA_VECTOR_SEARCH",
    "ADONIS_TEST_DATA_NUM_QUERIES",
    "HIRAG_MULTI_QUERY_MODEL",
)

from evaluation.collect_runner import default_out_dir, run_collection  # noqa: E402
from evaluation.dataset import EvalQuestion  # noqa: E402
from evaluation.streamlit_answer import (  # noqa: E402
    STREAMLIT_APP_FILE,
    ConfigAnswerModel,
    answer_search_response,
)


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True, help="question JSONL")
    parser.add_argument(
        "--max-results",
        type=int,
        default=8,
        help="search_test_data max_results (MCP default 8)",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="run directory")
    parser.add_argument("--resume", action="store_true", help="continue --out-dir")
    parser.add_argument(
        "--limit", type=int, default=None, help="first N questions only"
    )
    return parser.parse_args()


def main() -> int:
    """Collect a parent-child test-data RAG run."""
    args = parse_args()

    from app import test_data_search

    if test_data_search.is_vector_search_enabled():
        # Vector mode embeds the query via LlamaIndex Settings.embed_model,
        # which the MCP server configures with set_llm_and_embed_model().
        from app.settings import set_llm_and_embed_model

        set_llm_and_embed_model()

    model = ConfigAnswerModel()

    def answer(question: EvalQuestion) -> dict[str, Any]:
        response = test_data_search.search_test_data(
            question.user_input, max_results=args.max_results
        )
        record = answer_search_response(question.user_input, response, model)
        record["query_config"] = {
            "max_results": args.max_results,
            "process_model_status_policy_notice": response.get(
                "process_model_status_policy_notice"
            ),
        }
        return record

    run_config = {
        "retrieval": "app.test_data_search.search_test_data (in-process)",
        "max_results": args.max_results,
        "vector_search": test_data_search.is_vector_search_enabled(),
        "num_queries": test_data_search.get_multi_query_count(),
        "answer_model": model.model,
        "env": {name: os.environ.get(name) for name in ENV_FLAGS},
    }
    metadata_extra = {
        "code_fingerprint": code_fingerprint(
            [
                APP_DIR / "test_data_search.py",
                STREAMLIT_APP_FILE,
                BACHELOR_DIR / "config.yaml",
                Path(__file__),
                Path(__file__).with_name("streamlit_answer.py"),
            ]
        ),
        "git": git_info(APP_DIR),
        "packages": package_versions(["openai", "numpy", "llama-index-core"]),
    }
    label = "vector" if run_config["vector_search"] else "keyword"
    return run_collection(
        system="test_data_rag",
        dataset_path=args.dataset,
        out_dir=args.out_dir
        or default_out_dir("test_data_rag", f"{label}_max{args.max_results}"),
        run_config=run_config,
        metadata_extra=metadata_extra,
        id_granularity=None,
        answer_fn=answer,
        resume=args.resume,
        limit=args.limit,
    )


if __name__ == "__main__":
    sys.exit(main())
