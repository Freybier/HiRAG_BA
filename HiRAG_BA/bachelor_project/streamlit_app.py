#!/usr/bin/env python3
"""Simple Streamlit frontend for the Adonis Agent.

A lightweight chat UI that calls the MCP server running on port 8001 via the
``search_adonis`` tool and uses the configured LLM to synthesize a
natural-language answer with a mandatory sources (``Quellen``) section.

Authorization is bypassed for local development: the frontend sends a dummy
bearer token, which the server accepts when it runs with ``STAGE=test``.

Usage:
    export STAGE=test
    export ADONIS_TEST_DATA_SEARCH=true   # optional: serve search from src/test-data
    export ADONIS_TEST_DATA_DIR=/app/test-data/test-data_40pct  # container path
    poe agent-up
    poe ui
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st
from fastmcp import Client

from app.settings import set_llm_and_embed_model

logger = logging.getLogger(__name__)

_MCP_URL: str = os.getenv("ADONIS_MCP_URL", "http://localhost:8001/mcp")
_DUMMY_TOKEN: str = os.getenv("ADONIS_MCP_DUMMY_TOKEN", "a.b.c")

_NAIVE_RAG_FILE: Path = (
    Path(__file__).resolve().parent / "naive_RAG" / "openAI_Test_chunked.py"
)

# Make the bachelor_project folder importable so the naive_RAG module can
# import the shared llm_config next to config.yaml.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import llm_config  # (bachelor_project is on sys.path) # noqa: E402
import usage_logger  # noqa: E402
from answer_prompt import ANSWER_LAYOUT_PROMPT  # noqa: E402

#: Retrieval modes selectable in the UI dropdown.
_MODE_HIERARCHICAL: str = "Hierarchisch (test-data Pipeline)"
_MODE_FLAT: str = "Flach (top-k Vektorsuche)"
_MODE_HIRAG: str = "HiRAG (Graph-RAG Index)"

#: Usage-logger system key per retrieval mode.
_SYSTEM_KEYS: dict[str, str] = {
    _MODE_HIERARCHICAL: "hierarchical",
    _MODE_FLAT: "naive_rag",
    _MODE_HIRAG: "hirag",
}


def _load_flat_search_module() -> Any:
    """Import the naive_RAG flat-search script as a module.

    The module is loaded directly from its file path, so the result does not
    depend on how ``streamlit`` sets up ``sys.path``.

    Returns:
        The loaded ``openAI_Test_chunked`` module.
    """
    spec = importlib.util.spec_from_file_location(
        "openAI_Test_chunked", _NAIVE_RAG_FILE
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load flat-search module from {_NAIVE_RAG_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("openAI_Test_chunked", module)
    spec.loader.exec_module(module)
    return module


def run_adonis_search(question: str) -> dict[str, Any]:
    """Call the ``search_adonis`` tool on the MCP server via HTTP.

    Args:
        question: The user's natural-language search question.

    Returns:
        The tool's structured response dict, or an ``{"error": ...}`` dict
        when the server call fails.
    """
    return asyncio.run(_call_search_tool(question))


def run_flat_search(question: str, top_k: int = 5) -> dict[str, Any]:
    """Run flat top-k vector retrieval over the pre-chunked test-data corpus.

    Uses the inline chunk embeddings of ``test-data_40pct/chunks.jsonl`` (with
    ``chunk_embeddings.npy`` as fallback) via
    :func:`openAI_Test_chunked.flat_test_data_search` — no MCP server, no
    hierarchy enrichment, pure cosine-similarity retrieval.

    Args:
        question: The user's natural-language search question.
        top_k: Number of most similar chunks to retrieve.

    Returns:
        Response dict shaped like the MCP ``search_adonis`` result
        (``results``, ``top_process_model_sources`` and
        ``mandatory_quellen_section``), or an ``{"error": ...}`` dict when
        the flat search fails.
    """
    try:
        module = _load_flat_search_module()
        hits = module.flat_test_data_search(question, top_k=top_k)
    except Exception as exc:
        logger.error(f"Flat test-data search failed: {exc}")
        return {"error": str(exc)}

    results = [
        {
            "score": hit["similarity"],
            "node": {
                "text": hit["passage"],
                "metadata": {
                    "model_name": hit.get("model_name", ""),
                    "source_url": hit.get("source_url", ""),
                },
            },
        }
        for hit in hits
    ]

    # Deduplicated model sources for the clickable Quellen chips (one entry
    # per model, even when several chunks of the same model are retrieved).
    model_sources: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for hit in hits:
        name = hit.get("model_name", "")
        if name and name not in seen_names:
            seen_names.add(name)
            model_sources.append({"name": name, "url": hit.get("source_url", "")})

    if model_sources:
        quellen_lines = [
            f"- [{src['name']}]({src['url']})" if src["url"] else f"- {src['name']}"
            for src in model_sources
        ]
    else:  # Fallback when the chunks carry no model metadata.
        quellen_lines = [
            f"- Chunk {hit['chunk_id']} (Ähnlichkeit: {hit['similarity']:.3f})"
            for hit in hits
        ]

    return {
        "results": results,
        "top_process_model_sources": model_sources,
        "top_document_sources": [],
        "process_model_status_policy_notice": None,
        "mandatory_quellen_section": "**Quellen**\n" + "\n".join(quellen_lines),
    }


def run_hirag_search(question: str, index: str, mode: str = "hi") -> dict[str, Any]:
    """Query an existing HiRAG index (retrieval + answer synthesis by HiRAG).

    Uses the query-only ``hirag_query`` adapter, which loads the vendored
    hirag package in-process. Indexes are built separately with
    ``hi_insert_run.py`` in ``hirag_venv`` — this path can only query them.

    Args:
        question: The user's natural-language question.
        index: Index directory name under ``HiRAG/hirag_index_samples/``.
        mode: HiRAG query mode (``hi``, ``hi_bridge``, ``hi_local``,
            ``hi_global``, ``hi_nobridge``, ``naive``).

    Returns:
        Dict with ``answer`` and ``working_dir``, or ``{"error": ...}``.
    """
    if not index:
        return {"error": "Kein HiRAG-Index ausgewählt oder vorhanden."}
    try:
        import hirag_query
    except Exception as exc:
        logger.error(f"HiRAG adapter import failed: {exc}")
        return {"error": str(exc)}
    return hirag_query.query_hirag(question, index=index, mode=mode)


async def _call_search_tool(question: str) -> dict[str, Any]:
    """Perform the async MCP ``tools/call`` round-trip.

    Args:
        question: The user's natural-language search question.

    Returns:
        The tool's structured response dict, or an ``{"error": ...}`` dict.
    """
    try:
        async with Client(_MCP_URL, auth=_DUMMY_TOKEN) as client:
            result = await client.call_tool("search_adonis", {"search_term": question})
    except Exception as exc:
        logger.error(f"MCP search_adonis call failed: {exc}")
        return {"error": str(exc)}

    data = getattr(result, "data", None)
    if isinstance(data, dict):
        return data
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        return structured
    return {"error": "MCP tool returned no structured result"}


# The response-format block is shared with HiRAG's answer_generation stage
# (see answer_prompt.py); the citation rules below stay frontend-specific
# (HiRAG sources are resolved via the capture hook, not copied by the LLM).
_SYSTEM_PROMPT: str = f"""You are an AI assistant specialized in Adonis business process management.
Answer the user's question using ONLY the search results provided below. Never answer from general knowledge alone.

{ANSWER_LAYOUT_PROMPT}

CITATION RULES:
- Cite sources ONLY as standard Markdown hyperlinks: [Name](url).
- NEVER invent URLs — only use URLs present in the search results.
- The search results include a 'mandatory_quellen_section' field. You MUST copy this field
  CHARACTER-FOR-CHARACTER as the ABSOLUTE LAST SECTION of your response.
- Every response MUST end with the **Quellen** section.

SEARCH RESULTS (JSON):
"""


@st.cache_resource(show_spinner="Initialisiere Adonis-Verbindung …")
def init_backend() -> None:
    """Initialize the shared LLM and embedding settings exactly once.

    Returns:
        None. The global ``Settings`` object is configured in-place.
    """
    set_llm_and_embed_model()
    logger.info("LLM and embedding model initialized for Streamlit app")


def _render_sources(response: dict[str, Any]) -> None:
    """Render clickable source chips for process models and documents.

    Args:
        response: The assembled search response dict containing
            ``top_process_model_sources`` and ``top_document_sources``.
    """
    sources = response.get("top_process_model_sources", []) + response.get(
        "top_document_sources", []
    )
    if not sources:
        return
    with st.expander(f"Quellen ({len(sources)})", expanded=False):
        for src in sources:
            name = src.get("name", "Quelle")
            url = src.get("url", "")
            if url:
                st.markdown(f"- [{name}]({url})")


def _render_hirag_sources(response: dict[str, Any]) -> None:
    """Render the source documents a HiRAG answer was built from.

    Args:
        response: The ``query_hirag`` result dict with a ``sources`` list
            (entries carry ``name``, ``source_ids`` and ``origin``).
    """
    sources = response.get("sources") or []
    if not sources:
        return
    retrieved = [s for s in sources if s.get("origin") == "retrieved"]
    provenance = [s for s in sources if s.get("origin") == "provenance"]

    def _render_source(src: dict[str, Any]) -> None:
        """Render one HiRAG source entry as a Markdown list item."""
        label = src.get("name") or src.get("doc_id", "Quelle")
        ids = ", ".join(src.get("source_ids") or [])
        line = f"- {label}"
        if ids:
            line += f" — `{ids}`"
        st.markdown(line)

    with st.expander(f"Quellen ({len(retrieved)})", expanded=False):
        for src in retrieved:
            _render_source(src)
    if provenance:
        with st.expander(
            f"Weitere beitragende Dokumente über Graph-Provenienz ({len(provenance)})",
            expanded=False,
        ):
            st.caption(
                "Diese Dokumente wurden nicht direkt abgerufen, sondern nur "
                "über generierte Entitäts-/Community-Zusammenfassungen einbezogen."
            )
            for src in provenance[:50]:
                _render_source(src)
            if len(provenance) > 50:
                st.caption(f"… und {len(provenance) - 50} weitere.")


#: Cached (client, deployment) for the shared answer-generation path.
_ANSWER_CLIENT: tuple[Any, str] | None = None


def _get_answer_client() -> tuple[Any, str]:
    """Return the gateway chat client and deployment for answer generation.

    The deployment comes from the same config HiRAG queries use
    (``stage_models.stages.answer_generation`` in
    ``bachelor_project/config.yaml``, via
    :func:`llm_config.get_answer_generation_model`), so all three RAG modes
    share one answer-model switch. Uses a raw ``AzureOpenAI`` client like
    the multi-query helpers (no ``temperature`` kwarg: the gpt-5.x/6.x
    reasoning deployments reject non-default values).

    Returns:
        Tuple of (``openai.AzureOpenAI`` client, deployment name).
    """
    global _ANSWER_CLIENT
    if _ANSWER_CLIENT is None:
        from openai import AzureOpenAI

        api_key = llm_config.get_api_key("openai")
        client = AzureOpenAI(
            azure_endpoint=llm_config.OPENAI_CONFIG["base_url"],
            api_key=api_key,
            api_version=llm_config.OPENAI_CONFIG["api_version"],
            default_headers={
                "Content-Type": "application/json",
                "X-Gateway-Api-Key": api_key,
            },
        )
        _ANSWER_CLIENT = (client, llm_config.get_answer_generation_model())
    return _ANSWER_CLIENT


def _mode_detail(
    retrieval_mode: str, hirag_index: str | None, hirag_mode: str, top_k: int
) -> str:
    """Build the free-form usage-log detail string for a retrieval mode.

    Args:
        retrieval_mode: The selected UI retrieval mode.
        hirag_index: Selected HiRAG index name (HiRAG mode only).
        hirag_mode: Selected HiRAG query mode (HiRAG mode only).
        top_k: Selected top-k (flat mode only).

    Returns:
        A compact ``key=value`` detail string.
    """
    if retrieval_mode == _MODE_HIRAG:
        return f"index={hirag_index}, hirag_mode={hirag_mode}"
    if retrieval_mode == _MODE_FLAT:
        return f"top_k={top_k}"
    return "mcp search_adonis"


def _render_usage(usage: dict[str, Any] | None) -> None:
    """Render the token/cost summary line of one answered question.

    Args:
        usage: The ``UsageTracker.summary()`` dict of the question, or
            ``None``/empty when nothing was tracked.
    """
    if not usage or not usage.get("calls"):
        return
    currency = usage.get("currency") or "USD"
    cost_text = f" · Kosten: {usage.get('total_cost', 0.0):.6f} {currency}"
    if not usage.get("cost_complete", True):
        cost_text += " (unvollständig)"
    unpriced = usage.get("unpriced_models") or []
    unpriced_text = f" · unbepreist: {', '.join(unpriced)}" if unpriced else ""
    st.caption(
        f"🔢 {usage['calls']} LLM-/Embedding-Calls "
        f"({usage.get('cached_calls', 0)} cached) · "
        f"Tokens: {usage['total_input_tokens']:,} in / "
        f"{usage['total_output_tokens']:,} out"
        f"{cost_text}{unpriced_text}"
    )


def generate_answer(question: str, response: dict[str, Any]) -> str:
    """Generate a natural-language answer from search results via the LLM.

    Uses the ``answer_generation`` stage deployment from
    ``bachelor_project/config.yaml`` — the same model config HiRAG queries
    use — so the naive_rag and hierarchical answers are directly comparable
    to HiRAG's. The call is recorded in the active usage tracker (phase
    ``answer``) when it happens inside a tracked frontend question.

    Args:
        question: The user's original question.
        response: The assembled search response dict from :func:`run_adonis_search`.

    Returns:
        The LLM-generated answer string, or a fallback message on error.
    """
    import json

    client, deployment = _get_answer_client()
    context = json.dumps(response, ensure_ascii=False, default=str)
    prompt = (
        _SYSTEM_PROMPT
        + context
        + f"\n\nUSER QUESTION:\n{question}\n\nANSWER (end with the **Quellen** section):"
    )
    # Same shape as the old Settings.llm.complete(prompt): one user message.
    messages = [{"role": "user", "content": prompt}]
    try:
        completion = client.chat.completions.create(model=deployment, messages=messages)
        answer = completion.choices[0].message.content or ""
        usage_logger.track_chat(
            model=deployment,
            messages=messages,
            response_text=answer,
            response_obj=completion,
            phase="answer",
        )
        return answer
    except Exception as exc:
        logger.error(f"LLM completion failed: {exc}")
        return f"Fehler bei der Antwortgenerierung: {exc}"


def main() -> None:
    """Run the Streamlit chat application."""
    st.set_page_config(page_title="Adonis Agent", page_icon="🏦", layout="wide")
    st.title("🏦 Adonis Agent")
    st.caption("Fragen zu Geschäftsprozessen, Dokumenten und Richtlinien")

    init_backend()

    retrieval_mode = st.sidebar.selectbox(
        "Suchmodus",
        [_MODE_HIERARCHICAL, _MODE_FLAT, _MODE_HIRAG],
        help=(
            "Hierarchisch: MCP-Server mit test-data Pipeline (Hierarchie, "
            "Status-Policy, Quellen). Flach: direkte top-k Vektorsuche über "
            "die Chunk-Embeddings (test-data_40pct) ohne Server. HiRAG: "
            "Graph-RAG über einen bestehenden HiRAG-Index (nur Abfrage, "
            "Indizes werden mit hi_insert_run.py gebaut)."
        ),
    )
    top_k: int = 5
    if retrieval_mode == _MODE_FLAT:
        top_k = st.sidebar.slider("top_k", min_value=1, max_value=20, value=5)

    hirag_index: str | None = None
    hirag_mode: str = "hi"
    if retrieval_mode == _MODE_HIRAG:
        try:
            import hirag_query

            hirag_indexes = hirag_query.list_available_indexes()
        except Exception as exc:
            hirag_indexes = []
            st.sidebar.error(f"HiRAG nicht verfügbar: {exc}")
        if hirag_indexes:
            hirag_index = st.sidebar.selectbox("HiRAG Index", hirag_indexes)
            hirag_mode = st.sidebar.selectbox(
                "HiRAG Modus",
                ["hi", "hi_bridge", "hi_local", "hi_global", "hi_nobridge", "naive"],
            )
        else:
            st.sidebar.warning(
                "Keine HiRAG-Indizes gefunden. Zuerst einen Index mit "
                "hi_insert_run.py (hirag_venv) bauen."
            )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("response"):
                _render_sources(message["response"])
                _render_hirag_sources(message["response"])
            _render_usage(message.get("usage"))

    if question := st.chat_input("Wie kann ich helfen?"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            # One usage tracker per question: every LLM/embedding call made
            # by the selected RAG system is recorded with tokens and costs
            # and written to usage_logs/frontend_usage.jsonl on exit (the
            # hierarchical mode's server-side retrieval is logged by the MCP
            # server process itself as standalone lines).
            with usage_logger.activate_tracker(
                system=_SYSTEM_KEYS[retrieval_mode],
                question=question,
                detail=_mode_detail(retrieval_mode, hirag_index, hirag_mode, top_k),
            ) as tracker:
                with st.spinner("Suche in Adonis …"):
                    if retrieval_mode == _MODE_HIRAG:
                        response = run_hirag_search(question, hirag_index, hirag_mode)
                    elif retrieval_mode == _MODE_FLAT:
                        response = run_flat_search(question, top_k=top_k)
                    else:
                        response = run_adonis_search(question)

                if "error" in response:
                    answer = f"Fehler bei der Suche: {response['error']}"
                    st.error(answer)
                elif retrieval_mode == _MODE_HIRAG:
                    # HiRAG synthesizes the final answer itself (no separate LLM
                    # step); its source documents are resolved via the index's
                    # source_id_mapping.json and rendered below the answer.
                    answer = response["answer"]
                    st.markdown(answer)
                    _render_hirag_sources(response)
                else:
                    with st.spinner("Generiere Antwort …"):
                        answer = generate_answer(question, response)
                    st.markdown(answer)
                    _render_sources(response)
                usage_summary = tracker.summary()

            _render_usage(usage_summary)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "response": response,
                "usage": usage_summary,
            }
        )


if __name__ == "__main__":
    main()
