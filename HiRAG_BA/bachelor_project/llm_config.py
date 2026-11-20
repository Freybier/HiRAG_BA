"""Centralized model configuration for all bachelor_project entry scripts.

Loads ``config.yaml`` (the bachelor_project config next to this file) exactly
once, relative to this file, so the working directory no longer matters. The
old ``HiRAG/config.yaml`` is kept only as a backup and is no longer read.

To change the model used by ALL scripts, edit the global ``models`` section
in ``../config.yaml`` or set the ``HIRAG_CHAT_MODEL`` /
``HIRAG_EMBEDDING_MODEL`` / ``HIRAG_MULTI_QUERY_MODEL`` environment variables.

API keys are NOT stored in any YAML file. They come from the environment
(``src/app/.env-dev``): ``AZURE_OPENAI_API_KEY`` / ``X_GATEWAY_API_KEY`` /
``COHERE_API_KEY``.

Resolution order for model names (highest priority first):

1. Environment variable (``HIRAG_CHAT_MODEL`` / ``HIRAG_EMBEDDING_MODEL`` /
   ``HIRAG_MULTI_QUERY_MODEL``)
2. The global ``models`` section of ``../config.yaml``
3. The provider-specific section of ``../config.yaml``
4. The built-in default
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

DEFAULT_CHAT_MODEL = "adapfkt_gpt5"
DEFAULT_EMBEDDING_MODEL = "adapfkt_ada_002"

# Providers whose chat model config key is not "model".
_CHAT_MODEL_KEY_BY_PROVIDER: dict[str, str] = {
    "ollama": "chat_model",
}


def load_config(path: Path = _CONFIG_PATH) -> dict[str, Any]:
    """Load and parse the HiRAG ``config.yaml`` file.

    Args:
        path: Location of the YAML config file. Defaults to the
            ``config.yaml`` next to this module.

    Returns:
        The parsed YAML mapping.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file does not parse to a mapping.
    """
    try:
        with open(path, "r") as file:
            loaded = yaml.safe_load(file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"config.yaml not found at {path}. "
            "Create it before running any HiRAG script."
        ) from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"config.yaml at {path} did not parse to a mapping.")
    return loaded


config: dict[str, Any] = load_config()

OPENAI_CONFIG: dict[str, Any] = config.get("openai", {})
COHERE_CONFIG: dict[str, Any] = config.get("cohere", {})
DEEPSEEK_CONFIG: dict[str, Any] = config.get("deepseek", {})
GLM_CONFIG: dict[str, Any] = config.get("glm", {})
OLLAMA_CONFIG: dict[str, Any] = config.get("ollama", {})
MODELS_CONFIG: dict[str, Any] = config.get("models", {})
MODEL_PARAMS: dict[str, Any] = config.get("model_params", {})


def get_chat_model(provider: str = "openai") -> str:
    """Resolve the chat/LLM model name for a provider.

    Args:
        provider: Provider section name in ``config.yaml`` such as
            ``openai``, ``cohere``, ``deepseek``, ``glm`` or ``ollama``.

    Returns:
        The model name to use for chat completions.
    """
    env_value = os.environ.get("HIRAG_CHAT_MODEL", "").strip()
    if env_value:
        return env_value
    global_value = str(MODELS_CONFIG.get("chat_model", "")).strip()
    if global_value:
        return global_value
    key = _CHAT_MODEL_KEY_BY_PROVIDER.get(provider, "model")
    value = str(config.get(provider, {}).get(key, "")).strip()
    return value or DEFAULT_CHAT_MODEL


def get_embedding_model(provider: str = "openai") -> str:
    """Resolve the embedding model name for a provider.

    Args:
        provider: Provider section name in ``config.yaml`` such as
            ``openai``, ``cohere``, ``glm`` or ``ollama``.

    Returns:
        The model name to use for embeddings.
    """
    env_value = os.environ.get("HIRAG_EMBEDDING_MODEL", "").strip()
    if env_value:
        return env_value
    global_value = str(MODELS_CONFIG.get("embedding_model", "")).strip()
    if global_value:
        return global_value
    value = str(config.get(provider, {}).get("embedding_model", "")).strip()
    return value or DEFAULT_EMBEDDING_MODEL


def get_multi_query_model(provider: str = "openai") -> str:
    """Resolve the model used for multi-query query-variant generation.

    Resolution order: the ``HIRAG_MULTI_QUERY_MODEL`` environment variable,
    then ``models.multi_query_model`` in ``config.yaml``, then the provider's
    chat model as the fallback.

    Args:
        provider: Provider section name in ``config.yaml`` (used only for the
            chat-model fallback).

    Returns:
        The deployment/model name for query-expansion chat completions.
    """
    env_value = os.environ.get("HIRAG_MULTI_QUERY_MODEL", "").strip()
    if env_value:
        return env_value
    configured = str(MODELS_CONFIG.get("multi_query_model", "")).strip()
    if configured:
        return configured
    return get_chat_model(provider)


# Convenience constants for the default (Azure gateway / OpenAI) provider,
# which is what the currently used index (hirag_index_samples/hirag_index_cohere)
# was built with.
CHAT_MODEL: str = get_chat_model("openai")
EMBEDDING_MODEL: str = get_embedding_model("openai")
MULTI_QUERY_MODEL: str = get_multi_query_model("openai")


def get_api_key(provider: str = "openai") -> str:
    """Resolve an API key from the environment only (never from YAML).

    Args:
        provider: Provider name. ``openai`` maps to the api.central gateway
            key, ``cohere`` to ``COHERE_API_KEY``, etc.

    Returns:
        The API key value.

    Raises:
        RuntimeError: If the required environment variable is not set.
    """
    env_var_by_provider = {
        "openai": ("AZURE_OPENAI_API_KEY", "X_GATEWAY_API_KEY"),
        "cohere": ("COHERE_API_KEY",),
        "deepseek": ("DEEPSEEK_API_KEY",),
        "glm": ("GLM_API_KEY",),
    }
    for env_var in env_var_by_provider.get(provider, ()):
        value = os.environ.get(env_var, "").strip()
        if value:
            return value
    raise RuntimeError(
        f"No API key found for provider '{provider}'. "
        f"Set one of {env_var_by_provider.get(provider, ())} "
        "(see src/app/.env-dev)."
    )


def get_stage_model_config() -> tuple[bool, dict[str, str]]:
    """Load the HiRAG per-stage model routing config from ``config.yaml``.

    Reads the ``stage_models`` section, resolves every stage's alias through
    ``model_aliases`` and returns the switch plus the stage-to-deployment
    mapping. Stages without a configured (or empty) model are omitted, so the
    caller falls back to the original model function for them.

    Returns:
        A tuple ``(use_stage_models, stage_models)`` where
        ``use_stage_models`` is the boolean switch and ``stage_models`` maps
        stage names (e.g. ``entity_extraction``) to resolved Azure gateway
        deployment names.
    """
    stage_config = config.get("stage_models", {})
    if not isinstance(stage_config, dict):
        return False, {}
    use_stage_models = bool(stage_config.get("use_stage_models", False))
    aliases = stage_config.get("model_aliases", {}) or {}
    raw_stages = stage_config.get("stages", {}) or {}
    stage_models: dict[str, str] = {}
    for stage, alias in raw_stages.items():
        alias_str = str(alias).strip()
        if not alias_str:
            continue
        stage_models[str(stage)] = str(aliases.get(alias_str, alias_str)).strip()
    return use_stage_models, stage_models


def get_answer_generation_model() -> str:
    """Resolve the model used for final answer generation in ALL RAG modes.

    Reads ``models.answer_generation`` from ``config.yaml`` — a dedicated
    switch independent of the HiRAG indexing stage routing. Empty or
    missing falls back to ``models.chat_model``. The Streamlit frontend
    uses this for the naive_rag and hierarchical modes, and the HiRAG
    query wiring uses it for HiRAG's answer_generation stage, so all three
    RAG systems share one answer-model switch.

    Returns:
        The Azure gateway deployment name for answer generation.
    """
    answer_model = str(config.get("models", {}).get("answer_generation") or "").strip()
    if answer_model:
        return answer_model
    return get_chat_model("openai")
