"""Collect naive-RAG answers (Streamlit "Flach" mode) for RAGAS scoring.

Retrieval is exactly ``streamlit_app.run_flat_search`` (flat top-k cosine
search via ``naive_RAG/openAI_Test_chunked.flat_test_data_search``); the
answer prompt is the byte-identical ``generate_answer`` prompt (see
``streamlit_answer.py`` for the answer-model deviation).

Run with the src venv::

    cd src
    .venv/bin/python app/bachelor_project/evaluation/collect_naive.py \\
        --dataset /path/to/questions.jsonl --top-k 5

``retrieved_context_ids`` stay empty (``id_granularity: null``): the chunk
corpus has no IDs comparable to HiRAG's ADONIS source IDs (FUTURE TODO:
common input-data structure for all RAG systems).
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
    BACHELOR_DIR,
    NAIVE_RAG_DIR,
    SRC_DIR,
    code_fingerprint,
    git_info,
    load_env_dev,
    package_versions,
)

load_env_dev()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.collect_runner import default_out_dir, run_collection  # noqa: E402
from evaluation.dataset import EvalQuestion  # noqa: E402
from evaluation.streamlit_answer import (  # noqa: E402
    STREAMLIT_APP_FILE,
    ConfigAnswerModel,
    answer_search_response,
    load_streamlit_module,
)

#: Env flags that change naive-RAG retrieval; recorded in the run config.
ENV_FLAGS = ("ADONIS_NAIVE_RAG_NUM_QUERIES", "ADONIS_TEST_DATA_DIR")


# Parse the command-line arguments.
def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True, help="question JSONL")
    parser.add_argument(
        "--top-k", type=int, default=5, help="chunks to retrieve (Streamlit default 5)"
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="run directory")
    parser.add_argument("--resume", action="store_true", help="continue --out-dir")
    parser.add_argument(
        "--limit", type=int, default=None, help="first N questions only"
    )
    return parser.parse_args()


# Configure flat vector retrieval and collect answers and contexts through the shared question
# runner.
def main() -> int:
    """Collect a naive-RAG run."""
    args = parse_args()
    streamlit_app = load_streamlit_module()
    model = ConfigAnswerModel()

    # Retrieve flat vector matches and generate an answer with the corresponding contexts and query
    # settings.
    def answer(question: EvalQuestion) -> dict[str, Any]:
        # Retrieve flat vector matches and generate an answer with contexts and query settings.
        response = streamlit_app.run_flat_search(question.user_input, top_k=args.top_k)
        record = answer_search_response(question.user_input, response, model)
        record["query_config"] = {"top_k": args.top_k}
        return record

    run_config = {
        "retrieval": "streamlit_app.run_flat_search",
        "top_k": args.top_k,
        "answer_model": model.model,
        "env": {name: os.environ.get(name) for name in ENV_FLAGS},
    }
    metadata_extra = {
        "code_fingerprint": code_fingerprint(
            [
                STREAMLIT_APP_FILE,
                NAIVE_RAG_DIR / "openAI_Test_chunked.py",
                BACHELOR_DIR / "config.yaml",
                Path(__file__),
                Path(__file__).with_name("streamlit_answer.py"),
            ]
        ),
        "git": git_info(BACHELOR_DIR),
        "packages": package_versions(["openai", "numpy", "llama-index-core"]),
    }
    return run_collection(
        system="naive_rag",
        dataset_path=args.dataset,
        out_dir=args.out_dir or default_out_dir("naive_rag", f"top{args.top_k}"),
        run_config=run_config,
        metadata_extra=metadata_extra,
        id_granularity=None,
        answer_fn=answer,
        resume=args.resume,
        limit=args.limit,
    )


if __name__ == "__main__":
    sys.exit(main())
