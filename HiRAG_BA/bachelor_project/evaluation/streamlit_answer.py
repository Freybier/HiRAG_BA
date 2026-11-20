"""Reproduce the Streamlit answer step for the chunk-based collectors.

``streamlit_app.generate_answer`` builds one prompt from ``_SYSTEM_PROMPT`` +
the JSON-serialized search response + the question and sends it as a single
user message. This module rebuilds that prompt byte-identically (it imports
``_SYSTEM_PROMPT`` and ``run_flat_search`` from ``streamlit_app.py`` itself, so
prompt edits are picked up automatically) and answers it.

Known deviation (FUTURE TODO, see README): Streamlit answers with
``llama_index.core.Settings.llm`` (the production model from
``app/config/config.yml``); the collectors answer with ``models.chat_model``
from ``bachelor_project/config.yaml`` through a plain Azure gateway client so
that all three RAG systems are compared with the same answer model.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from typing import Any

from evaluation.common import BACHELOR_DIR, SRC_DIR

STREAMLIT_APP_FILE = BACHELOR_DIR / "streamlit_app.py"

_STREAMLIT_MODULE: Any = None


# Import streamlit_app.py from its file path (cached).
def load_streamlit_module() -> Any:
    """Import ``streamlit_app.py`` from its file path (cached).

    Importing does not start the UI (``main()`` only runs under
    ``__main__``); it needs ``src`` on ``sys.path`` for ``app.settings``.

    Returns:
        The loaded module.
    """
    global _STREAMLIT_MODULE
    if _STREAMLIT_MODULE is None:
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))
        spec = importlib.util.spec_from_file_location(
            "bachelor_streamlit_app", STREAMLIT_APP_FILE
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {STREAMLIT_APP_FILE}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _STREAMLIT_MODULE = module
    return _STREAMLIT_MODULE


# Reconstruct the frontend prompt from the question and search response for consistent
# evaluation answers.
def build_answer_prompt(question: str, response: dict[str, Any]) -> tuple[str, str]:
    """Rebuild the exact ``generate_answer`` prompt.

    Args:
        question: User question.
        response: Search response dict as passed to ``generate_answer``.

    Returns:
        ``(prompt, context_text)`` where ``context_text`` is the JSON blob the
        answer LLM saw inside the prompt.
    """
    module = load_streamlit_module()
    context = json.dumps(response, ensure_ascii=False, default=str)
    prompt = (
        module._SYSTEM_PROMPT
        + context
        + f"\n\nUSER QUESTION:\n{question}\n\nANSWER (end with the **Quellen** section):"
    )
    return prompt, context


class ConfigAnswerModel:
    """Answer model from ``bachelor_project/config.yaml`` via the gateway."""

    # Create the synchronous Azure gateway client and select the configured answer model.
    def __init__(self) -> None:
        """Create the sync Azure gateway client for ``models.chat_model``."""
        from llm_config import CHAT_MODEL, OPENAI_CONFIG, get_api_key
        from openai import AzureOpenAI

        api_key = get_api_key("openai")
        self.model = CHAT_MODEL
        self._client = AzureOpenAI(
            azure_endpoint=OPENAI_CONFIG["base_url"],
            api_key=api_key,
            api_version=OPENAI_CONFIG.get("api_version", "2024-10-21"),
            default_headers={
                "Content-Type": "application/json",
                "X-Gateway-Api-Key": api_key,
            },
        )

    # Send the prepared prompt as one user message and return the answer with provider-reported
    # usage.
    def complete(self, prompt: str) -> tuple[str, dict[str, Any]]:
        """Send the prompt as one user message (like ``Settings.llm.complete``).

        Args:
            prompt: Full prompt from :func:`build_answer_prompt`.

        Returns:
            ``(answer text, usage dict in the run-record format)``.
        """
        completion = self._client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}]
        )
        usage = completion.usage
        return completion.choices[0].message.content or "", {
            "chat": {
                self.model: {
                    "calls": 1,
                    "cache_hits": 0,
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                }
            }
        }


# Build evaluation contexts and the frontend answer prompt, then generate an answer when context
# exists.
def answer_search_response(
    question: str, response: dict[str, Any], model: ConfigAnswerModel
) -> dict[str, Any]:
    """Turn one search response into the shared record fields.

    Args:
        question: User question.
        response: Search response (``results`` ...) or ``{"error": ...}``.
        model: Answer model.

    Returns:
        Partial run record (status, response, contexts, context_text, ...).

    Raises:
        RuntimeError: If the search itself failed; the runner records the
            question as ``status: error``.
    """
    from evaluation.run_format import build_contexts_from_search_response

    if "error" in response:
        raise RuntimeError(f"search failed: {response['error']}")
    fields = build_contexts_from_search_response(response)
    prompt, context_text = build_answer_prompt(question, response)
    module = load_streamlit_module()
    base = {
        "context_text": context_text,
        "context_builder": "streamlit_generate_answer_json",
        "system_prompt": module._SYSTEM_PROMPT,
        **fields,
    }
    if not fields["retrieved_contexts"]:
        # Streamlit would still call the LLM; for evaluation an answer without
        # any retrieved context is recorded as no_context instead.
        return {**base, "status": "no_context", "response": None, "usage": {}}
    answer, usage = model.complete(prompt)
    return {**base, "status": "answered", "response": answer, "usage": usage}
