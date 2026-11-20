from dataclasses import dataclass

import numpy as np
from hirag import HiRAG, QueryParam
from hirag._storage import Neo4jStorage
from hirag._utils import compute_args_hash
from hirag.base import BaseKVStorage
from llm_config import (
    DEEPSEEK_CONFIG,
    GLM_CONFIG,
    config,
    get_api_key,
    get_chat_model,
)
from openai import AsyncOpenAI

# Extract configurations (model names come from the bachelor_project
# ../config.yaml or HIRAG_CHAT_MODEL / HIRAG_EMBEDDING_MODEL env vars;
# API keys come from the environment, never from YAML)
GLM_API_KEY = GLM_CONFIG.get("api_key", "")
MODEL = get_chat_model("deepseek")
DEEPSEEK_API_KEY = get_api_key("deepseek")
DEEPSEEK_URL = DEEPSEEK_CONFIG["base_url"]
GLM_URL = GLM_CONFIG.get("base_url", "")


@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)


def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro


@wrap_embedding_func_with_attrs(
    embedding_dim=config["model_params"]["glm_embedding_dim"],
    max_token_size=config["model_params"]["max_token_size"],
)
async def GLM_embedding(texts: list[str]) -> np.ndarray:
    model_name = "embedding-3"
    client = AsyncOpenAI(api_key=GLM_API_KEY, base_url=GLM_URL)
    embedding = await client.embeddings.create(
        input=texts,
        model=model_name,
    )
    final_embedding = [d.embedding for d in embedding.data]
    return np.array(final_embedding)


async def deepseepk_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_URL)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------

    response = await openai_async_client.chat.completions.create(
        model=MODEL, messages=messages, **kwargs
    )

    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
        )
    # -----------------------------------------------------
    return response.choices[0].message.content


graph_func = HiRAG(
    working_dir=config["hirag"]["working_dir"],
    enable_llm_cache=config["hirag"]["enable_llm_cache"],
    embedding_func=GLM_embedding,
    best_model_func=deepseepk_model_if_cache,
    cheap_model_func=deepseepk_model_if_cache,
    enable_hierachical_mode=config["hirag"]["enable_hierachical_mode"],
    embedding_batch_num=config["hirag"]["embedding_batch_num"],
    embedding_func_max_async=config["hirag"]["embedding_func_max_async"],
    enable_naive_rag=config["hirag"]["enable_naive_rag"],
    graph_storage_cls=Neo4jStorage,
    addon_params={
        "neo4j_url": config["hirag"]["neo4j_url"],
        "neo4j_auth": config["hirag"]["neo4j_auth"],
    },
)

# comment this if the working directory has already been indexed
with open("your .txt file path") as f:
    graph_func.insert(f.read())

print("Perform hi search:")
print(
    graph_func.query(
        "What are the top themes in this story?", param=QueryParam(mode="hi")
    )
)
