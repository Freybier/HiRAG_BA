# PROJECT MODIFIED relative to HiRAG 67b2dfa; changed parts are marked below.
import os
# PROJECT ADDED: Import project dependencies or the shared configuration used by the marked
# sections.
import sys
from dataclasses import dataclass
# PROJECT ADDED: Import project dependencies or the shared configuration used by the marked
# sections.
from pathlib import Path

import numpy as np

# Make the vendored HiRAG package and the bachelor_project llm_config
# importable regardless of the current working directory.
# PROJECT ADDED: Anchor imports to this script directory so gateway wiring works from
# different working directories.
_SCRIPT_DIR = Path(__file__).resolve().parent
# PROJECT ADDED: Prepare project import paths or development environment values before
# creating model clients.
sys.path.insert(0, str(_SCRIPT_DIR))
# PROJECT ADDED: Prepare project import paths or development environment values before
# creating model clients.
sys.path.insert(0, str(_SCRIPT_DIR.parent))

# PROJECT ADDED: Import project dependencies or the shared configuration used by the marked
# sections.
from dotenv import load_dotenv  # noqa: E402


# PROJECT ADDED: Load and sanitize development environment values before the gateway clients
# are configured.
def _load_env_dev() -> None:
    """Load ``src/app/.env-dev`` into the process environment.

    The file historically has CRLF line endings, which can leave ``\r``
    artifacts in header values; each value is therefore sanitized. Values
    from ``.env-dev`` win over the shell environment so stale placeholder
    exports cannot shadow the real keys (mirrors ``hi_insert_run.py``).
    """
    env_path = _SCRIPT_DIR.parents[2] / "app" / ".env-dev"
    if not env_path.is_file():
        return
    load_dotenv(env_path, override=True)
    for key, value in list(os.environ.items()):
        cleaned = value.strip().strip('"').strip("'").replace("\r", "")
        if cleaned != value:
            os.environ[key] = cleaned


# PROJECT ADDED: Prepare project import paths or development environment values before
# creating model clients.
_load_env_dev()

from hirag import HiRAG, QueryParam  # noqa: E402
# PROJECT MODIFIED: Import project dependencies or the shared configuration used by the
# marked sections.
from hirag._storage import NetworkXStorage  # noqa: E402
# PROJECT MODIFIED: Import project dependencies or the shared configuration used by the
# marked sections.
from hirag._utils import compute_args_hash, hirag_current_stage  # noqa: E402
from hirag.base import BaseKVStorage  # noqa: E402
# PROJECT ADDED: Import project dependencies or the shared configuration used by the marked
# sections.
from llm_config import (  # noqa: E402
    CHAT_MODEL,
    EMBEDDING_MODEL,
    GLM_CONFIG,
    OPENAI_CONFIG,
    config,
    get_answer_generation_model,
    get_api_key,
    get_embedding_model,
    get_stage_model_config,
)
# PROJECT MODIFIED: Import project dependencies or the shared configuration used by the
# marked sections.
from openai import AsyncAzureOpenAI  # noqa: E402

# PROJECT ADDED: Load optional usage tracking or shared answer-layout support without
# blocking gateway queries.
try:  # (bachelor_project is already on sys.path via _SCRIPT_DIR.parent)
    import usage_logger as _usage_logger
except ImportError:  # pragma: no cover - usage logger is optional
    _usage_logger = None

# PROJECT ADDED: Load optional usage tracking or shared answer-layout support without
# blocking gateway queries.
try:  # shared answer-layout prompt (bachelor_project is on sys.path)
    from answer_prompt import ANSWER_LAYOUT_PROMPT as _ANSWER_LAYOUT_PROMPT
except ImportError:  # pragma: no cover - layout prompt is optional
    _ANSWER_LAYOUT_PROMPT = None


# PROJECT ADDED: Record chat tokens and cache hits in the active tracker using the task-
# local pipeline stage.
def _track_chat_call(
    model: str,
    messages: list,
    content: str,
    response_obj=None,
    cached: bool = False,
) -> None:
    """Record one gateway chat call in the frontend usage logger.

    No-op when no frontend question tracker is active (e.g. during index
    insertion runs or standalone script usage). The HiRAG stage that issued
    the call is taken from the task-local ``hirag_current_stage`` context
    variable (``None`` for calls outside a stage context).

    Args:
        model: Azure gateway deployment name used for the call.
        messages: The full message list sent to the model.
        content: The returned (or cached) assistant content.
        response_obj: The raw API response on cache misses (for usage
            extraction); ``None`` on cache hits.
        cached: Whether the call was served from the HiRAG LLM cache.
    """
    if _usage_logger is None:
        return
    try:
        stage = hirag_current_stage.get()
    except Exception:  # noqa: BLE001 - logging must never break the call
        stage = None
    _usage_logger.track_chat(
        model=model,
        messages=messages,
        response_text=content or "",
        response_obj=response_obj,
        phase="query",
        stage=stage,
        cached=cached,
    )


# Extract configurations (model names come from the bachelor_project
# ../config.yaml or HIRAG_CHAT_MODEL / HIRAG_EMBEDDING_MODEL env vars;
# the API key comes from AZURE_OPENAI_API_KEY / X_GATEWAY_API_KEY in .env-dev)
# PROJECT MODIFIED: Read the experiment setting from shared project configuration or its
# environment override.
OPENAI_EMBEDDING_MODEL = EMBEDDING_MODEL
# PROJECT MODIFIED: Read the experiment setting from shared project configuration or its
# environment override.
OPENAI_MODEL = CHAT_MODEL
# PROJECT MODIFIED: Read the experiment setting from shared project configuration or its
# environment override.
OPENAI_API_KEY = get_api_key("openai")
# PROJECT MODIFIED: Read the experiment setting from shared project configuration or its
# environment override.
OPENAI_URL = OPENAI_CONFIG["base_url"]
# PROJECT ADDED: Read the experiment setting from shared project configuration or its
# environment override.
OPENAI_API_VERSION = OPENAI_CONFIG.get("api_version", "2024-10-21")
# PROJECT MODIFIED: Read the experiment setting from shared project configuration or its
# environment override.
GLM_API_KEY = GLM_CONFIG.get("api_key", "")
# PROJECT MODIFIED: Read the experiment setting from shared project configuration or its
# environment override.
GLM_MODEL = GLM_CONFIG.get("model", "")
# PROJECT MODIFIED: Read the experiment setting from shared project configuration or its
# environment override.
GLM_URL = GLM_CONFIG.get("base_url", "")

# The api.central gateway is an Azure OpenAI endpoint that authenticates via the
# X-Gateway-Api-Key header (same pattern as openAI_Test.py and src/app).
# PROJECT ADDED: Authenticate Azure gateway requests with the project gateway header.
_AZURE_HEADERS = {
    "Content-Type": "application/json",
    "X-Gateway-Api-Key": OPENAI_API_KEY,
}


# PROJECT ADDED: Create an Azure gateway client with the configured endpoint, API version,
# and gateway header.
def _make_azure_client() -> AsyncAzureOpenAI:
    """Create an AsyncAzureOpenAI client for the api.central gateway."""
    return AsyncAzureOpenAI(
        azure_endpoint=OPENAI_URL,
        api_key=OPENAI_API_KEY,
        api_version=OPENAI_API_VERSION,
        default_headers=_AZURE_HEADERS,
    )


@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        # Await the wrapped embedding function and return its vectors.
        return await self.func(*args, **kwargs)


def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        # Attach embedding dimensions and token limits to the wrapped function.
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro


# PROJECT MODIFIED: Embed texts through the Azure gateway and report the response token
# usage to the active tracker.
@wrap_embedding_func_with_attrs(
    embedding_dim=config["model_params"]["openai_embedding_dim"],
    max_token_size=config["model_params"]["max_token_size"],
)
async def OPENAI_embedding(texts: list[str]) -> np.ndarray:
    # Request gateway embeddings and record their query-time token usage.
    openai_async_client = _make_azure_client()
    response = await openai_async_client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL, input=texts, encoding_format="float"
    )
    if _usage_logger is not None:
        _usage_logger.track_embedding(
            model=OPENAI_EMBEDDING_MODEL,
            texts=texts,
            response_obj=response,
            phase="query",
        )
    return np.array([dp.embedding for dp in response.data])


# PROJECT MODIFIED: Resolve the GLM embedding model through shared project configuration
# before requesting vectors.
@wrap_embedding_func_with_attrs(
    embedding_dim=config["model_params"]["glm_embedding_dim"],
    max_token_size=config["model_params"]["max_token_size"],
)
async def GLM_embedding(texts: list[str]) -> np.ndarray:
    # Request GLM embeddings for the input texts and return a NumPy array.
    model_name = get_embedding_model("glm")
    client = AsyncOpenAI(api_key=GLM_API_KEY, base_url=GLM_URL)  # noqa: F821
    embedding = await client.embeddings.create(
        input=texts,
        model=model_name,
    )
    final_embedding = [d.embedding for d in embedding.data]
    return np.array(final_embedding)


# PROJECT ADDED: Build gateway messages, apply the shared answer layout, and reuse model-
# specific cached replies.
async def _openai_request(
    model: str, prompt: str, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """Shared api.central gateway chat request with response caching.

    The cache key includes the model name together with the full message list
    (system prompt and history), so per-stage models never share cached
    responses.

    Args:
        model: Azure gateway deployment name to send as the model parameter.
        prompt: User prompt content.
        system_prompt: Optional system prompt prepended to the messages.
        history_messages: Optional prior messages for gleaning loops.
        **kwargs: Extra request parameters (e.g. max_tokens, hashing_kv).

    Returns:
        The assistant message content.
    """
    openai_async_client = _make_azure_client()

    # Give HiRAG's answer_generation stage the same answer-layout system
    # prompt the naive/hierarchical frontend answers use (streamlit_app),
    # appended after HiRAG's own local_rag_response instructions. The layout
    # text is part of the messages and therefore of the LLM-cache key, so
    # cached answers from before this change are simply missed once.
    if (
        _ANSWER_LAYOUT_PROMPT is not None
        and hirag_current_stage.get() == "answer_generation"
    ):
        system_prompt = (system_prompt or "") + "\n\n" + _ANSWER_LAYOUT_PROMPT

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            _track_chat_call(model, messages, if_cache_return["return"], cached=True)
            return if_cache_return["return"]
    # -----------------------------------------------------

    # The GPT-5 deployment on the gateway requires max_completion_tokens, not
    # the legacy max_tokens parameter that HiRAG passes through **kwargs.
    if "max_tokens" in kwargs:
        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )
    content = response.choices[0].message.content
    _track_chat_call(model, messages, content, response_obj=response)

    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {
                args_hash: {
                    "return": content,
                    "model": model,
                }
            }
        )
    # -----------------------------------------------------
    return content


# PROJECT MODIFIED: Delegate chat generation to the shared cached gateway request using the
# configured chat model.
async def OPENAI_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """Default model function using the global chat model from config.yaml."""
    return await _openai_request(
        OPENAI_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


# PROJECT ADDED: Create a chat callback pinned to one deployment so each HiRAG stage can use
# its own model.
def make_stage_model_func(model_name: str) -> callable:
    """Build a cached model function pinned to one stage's deployment name.

    Reuses the shared gateway request logic in ``_openai_request``; only the
    model parameter differs per stage.

    Args:
        model_name: Resolved Azure gateway deployment name for the stage.

    Returns:
        An async model function with the standard HiRAG signature.
    """

    # PROJECT ADDED: Forward this stage's prompt and history to the gateway using its pinned
    # deployment.
    async def stage_model_func(
        prompt, system_prompt=None, history_messages=[], **kwargs
    ) -> str:
        # Send this pipeline stage to its selected gateway chat model.
        return await _openai_request(
            model_name,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )

    # Lets the triplet extraction logger (triplets.triplet_logger) record the
    # real deployment name for each stage's calls.
    stage_model_func.model_name = model_name
    return stage_model_func


# PROJECT ADDED: Read the experiment setting from shared project configuration or its
# environment override.
USE_STAGE_MODELS, STAGE_MODELS = get_stage_model_config()
# The answer-generation model is configured separately from the indexing
# stages (models.answer_generation in config.yaml); it is only used at
# query time, so it is added to the stage routing here (ignored when
# use_stage_models is off).
# PROJECT ADDED: Read the experiment setting from shared project configuration or its
# environment override.
STAGE_MODELS["answer_generation"] = get_answer_generation_model()


# PROJECT ADDED: Construct a query-ready index with gateway models, stage routing, and
# NetworkX storage.
def build_hirag(working_dir: str | None = None) -> HiRAG:
    """Build a HiRAG instance wired to the api.central gateway.

    Extracted into a factory so other modules (e.g. the Streamlit frontend's
    ``hirag_query`` adapter) can construct HiRAG instances for arbitrary
    index directories without triggering the demo query below.

    Args:
        working_dir: HiRAG index directory. Defaults to
            ``config["hirag"]["working_dir"]`` from the shared
            ``bachelor_project/config.yaml``.

    Returns:
        A ready-to-query ``HiRAG`` instance (networkx graph storage).
    """
    return HiRAG(
        working_dir=working_dir or config["hirag"]["working_dir"],
        enable_llm_cache=config["hirag"]["enable_llm_cache"],
        embedding_func=OPENAI_embedding,
        best_model_func=OPENAI_model_if_cache,
        cheap_model_func=OPENAI_model_if_cache,
        use_stage_models=USE_STAGE_MODELS,
        stage_models=STAGE_MODELS,
        stage_model_func_factory=make_stage_model_func,
        enable_hierachical_mode=config["hirag"]["enable_hierachical_mode"],
        embedding_batch_num=config["hirag"]["embedding_batch_num"],
        embedding_func_max_async=config["hirag"]["embedding_func_max_async"],
        enable_naive_rag=config["hirag"]["enable_naive_rag"],
        graph_storage_cls=NetworkXStorage,
    )


# PROJECT ADDED: Run the standalone example through the project gateway factory when invoked
# as a script.
if __name__ == "__main__":
    graph_func = build_hirag()

    # comment this if the working directory has already been indexed
    # with open("./data/Germany.txt") as f:
    #    graph_func.insert(f.read())

    print("Perform hi search:")
    print(
        graph_func.query(
            "Auf welchem Dokument und auf welcher Seite befinden sich die unterschiedlichen KUST-Arten?",
            param=QueryParam(mode="hi"),
        )
    )
