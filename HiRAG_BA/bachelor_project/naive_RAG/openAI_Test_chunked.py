"""Minimal RAG experiment: embed a text corpus and run cosine-similarity search.

The corpus is selected via the ``SOURCE`` constant:

- ``"wiki"``: the Hugging Face ``rag-mini-wikipedia`` dataset, downloaded
  once to ``wiki_txt.csv`` and chunked with overlap.
- ``"test-data"``: the pre-chunked Adonis test corpus in
  ``src/test-data/test-data_40pct/chunks.jsonl`` (no chunking step needed;
  chunks carry their embeddings inline).

Multi-query expansion can be enabled via ``ADONIS_NAIVE_RAG_NUM_QUERIES``
(default ``1`` = off, matching the current production ``QueryFusionRetriever``
setting): values > 1 generate ``num_queries - 1`` LLM query variants with the
same ``QUERY_GEN_PROMPT`` used by the production ``QueryFusionRetriever`` and
fuse the per-variant rankings with reciprocal rank fusion (k=60).  The
variant-generation model is ``models.multi_query_model`` from the shared
``config.yaml`` (env override ``HIRAG_MULTI_QUERY_MODEL``, fallback
``models.chat_model``).

Retrieved chunks are additionally capped by a token budget mirroring HiRAG's
naive query mode (``naive_max_token_for_text_unit`` in
``HiRAG/hirag/base.py``): results are kept in score order until the
cumulative passage size exceeds ``ADONIS_NAIVE_RAG_MAX_CONTEXT_TOKENS``
(default 10000, same as HiRAG), counted with tiktoken ``cl100k_base``.

Each source uses its own embeddings file so the two corpora never overwrite
each other's cached embeddings.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from openai import AzureOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm_config import EMBEDDING_MODEL, MULTI_QUERY_MODEL, get_api_key, load_config

try:
    import usage_logger
except ImportError:  # pragma: no cover - usage logger is optional
    usage_logger = None

_BP_CONFIG: dict[str, Any] = load_config()
_API_KEY: str = get_api_key("openai")

headers = {
    "Content-Type": "application/json",
    "X-Gateway-Api-Key": _API_KEY,
}

client = AzureOpenAI(
    azure_endpoint=_BP_CONFIG["openai"]["base_url"],
    api_key=_API_KEY,
    api_version=_BP_CONFIG["openai"]["api_version"],
    default_headers=headers,
)

# ---------------------------------------------------------------------------
# Corpus source switch: "wiki" (Hugging Face dataset) or "test-data"
# (pre-chunked Adonis corpus from src/test-data/chunks.jsonl).
# ---------------------------------------------------------------------------
SOURCE: str = "test-data"

#: Embeddings cache file per source, so the corpora never clobber each other.
EMBEDDINGS_FILE: dict[str, str] = {
    "wiki": "chunked_embeddings.json",
    "test-data": "test_data_embeddings.json",
}

#: Pre-chunked Adonis test corpus directory. ``ADONIS_TEST_DATA_DIR`` is
#: honored when it points at a directory containing ``chunks.jsonl``; the
#: default is the 40% sample with inline embeddings, resolved relative to
#: this file so the script works regardless of the current working directory.
#: (parents: [0]=naive_RAG, [1]=bachelor_project, [2]=app, [3]=src)
_ENV_DATA_DIR: str = os.getenv("ADONIS_TEST_DATA_DIR", "")
if _ENV_DATA_DIR and (Path(_ENV_DATA_DIR) / "chunks.jsonl").exists():
    _TEST_DATA_DIR: Path = Path(_ENV_DATA_DIR)
else:
    _TEST_DATA_DIR = (
        Path(__file__).resolve().parents[3] / "test-data" / "test-data_40pct"
    )

#: Pre-chunked Adonis test corpus (rows carry inline embeddings).
TEST_DATA_CHUNKS_FILE: Path = _TEST_DATA_DIR / "chunks.jsonl"

#: Legacy precomputed embedding matrix, aligned with ``chunks.jsonl`` row
#: order. Only used as a fallback for corpora whose chunks do NOT carry
#: inline embeddings (e.g. ``small_anon_Dataset``).
TEST_DATA_EMBEDDINGS_NPY: Path = _TEST_DATA_DIR / "chunk_embeddings.npy"


def get_multi_query_count() -> int:
    """Return how many query variants flat retrieval runs per search.

    Mirrors the ``num_queries`` setting of the production
    ``QueryFusionRetriever`` wrappers in ``app.postgres.postgres_retriever``:
    ``1`` (the default, matching production) disables multi-query expansion;
    values > 1 generate ``num_queries - 1`` additional LLM-written query
    variants and fuse the per-variant rankings with reciprocal rank fusion.

    Returns:
        The configured ``ADONIS_NAIVE_RAG_NUM_QUERIES`` value, clamped to
        ``>= 1``; unparsable values fall back to ``1``.
    """
    raw = os.getenv("ADONIS_NAIVE_RAG_NUM_QUERIES", "1").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        print(f"Invalid ADONIS_NAIVE_RAG_NUM_QUERIES={raw!r}; falling back to 1")
        return 1


def _generate_query_variants(query: str, num_queries: int) -> list[str]:
    """Generate ``num_queries - 1`` LLM query variants like the production path.

    Uses the same prompt template, response parsing and trimming as the
    ``QueryFusionRetriever`` defaults used by production
    (``app.postgres.postgres_retriever``): the shared ``QUERY_GEN_PROMPT``,
    one variant per line, trimmed to the requested count.  The chat model is
    the shared ``MULTI_QUERY_MODEL`` (``models.multi_query_model`` in
    ``config.yaml``).  The original query always stays first; on any LLM
    failure the function degrades to single-query behavior.

    Args:
        query: The original user query.
        num_queries: Total number of queries (original + generated variants).

    Returns:
        ``[query, *variants]``; just ``[query]`` when generation is disabled,
        fails, or yields nothing.
    """
    if num_queries <= 1:
        return [query]

    from llama_index.core.retrievers.fusion_retriever import QUERY_GEN_PROMPT

    prompt = QUERY_GEN_PROMPT.format(num_queries=num_queries - 1, query=query)
    try:
        # No temperature kwarg: the gpt-5.x reasoning deployments reject
        # non-default temperature values (production sends none either).
        response = client.chat.completions.create(
            model=MULTI_QUERY_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=256,
        )
        text = response.choices[0].message.content or ""
        if usage_logger is not None:
            usage_logger.track_chat(
                model=MULTI_QUERY_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_text=text,
                response_obj=response,
                phase="retrieval",
            )
    except Exception as exc:  # noqa: BLE001 - degrade to single-query mode
        print(
            f"Multi-query expansion failed ({exc!r}); "
            "falling back to the original query only"
        )
        return [query]

    # Production parsing (QueryFusionRetriever._generate_queries): strip code
    # fences, one query per line, drop blanks, trim to the requested count.
    # Numbering such as "1. " is passed through verbatim, as in production.
    variants = [line.strip() for line in text.strip("`").split("\n") if line.strip()][
        : num_queries - 1
    ]

    # Drop duplicates of the original query or of earlier variants.
    seen = {query.strip().casefold()}
    unique_variants: list[str] = []
    for variant in variants:
        if variant.casefold() in seen:
            continue
        seen.add(variant.casefold())
        unique_variants.append(variant)

    if unique_variants:
        # Production runs the fusion retrievers with verbose=True, which
        # prints the generated queries; mirror that observability here.
        print("Generated queries:\n" + "\n".join(unique_variants))
    return [query, *unique_variants]


def _reciprocal_rank_fusion(
    per_query_rankings: list[list[tuple[float, int]]],
    top_k: int,
) -> list[tuple[float, int]]:
    """Fuse per-query rankings with reciprocal rank fusion (RRF, k=60).

    In-memory port of ``QueryFusionRetriever._reciprocal_rerank_fusion``:
    each chunk accumulates ``1 / (rank + k)`` across all query rankings and
    the fused RRF score replaces the raw similarity score (production also
    reports the fused score as the node score).

    Args:
        per_query_rankings: One ranked ``(similarity, chunk_index)`` list per
            query, each already sorted by similarity (highest first).
        top_k: Maximum number of fused chunks to return.

    Returns:
        Fused ``(rrf_score, chunk_index)`` tuples, highest fused score first.
    """
    rrf_constant = 60.0  # k from the RRF paper; same value as production.
    fused_scores: dict[int, float] = {}

    for ranking in per_query_rankings:
        for rank, (_similarity, chunk_index) in enumerate(ranking):
            fused_scores[chunk_index] = fused_scores.get(chunk_index, 0.0) + (
                1.0 / (rank + rrf_constant)
            )

    reranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return [(score, chunk_index) for chunk_index, score in reranked[:top_k]]


#: Default context token budget, mirroring HiRAG's naive query mode
#: (``QueryParam.naive_max_token_for_text_unit = 10000`` in
#: ``HiRAG/hirag/base.py``).
DEFAULT_MAX_CONTEXT_TOKENS: int = 10_000

#: Lazily created tiktoken encoder (``cl100k_base``, same encoding
#: ``hi_insert_run.py`` uses for local token counting).
_TIKTOKEN_ENCODER: Any = None


def get_max_context_tokens() -> int:
    """Return the token budget applied to retrieved chunks per search.

    Mirrors HiRAG's naive query mode, which truncates the retrieved chunk
    list to ``naive_max_token_for_text_unit`` tokens before building the
    prompt.  The budget only binds when ``top_k`` is set high enough that
    the retrieved passages exceed it.

    Returns:
        The configured ``ADONIS_NAIVE_RAG_MAX_CONTEXT_TOKENS`` value;
        unparsable or non-positive values fall back to
        ``DEFAULT_MAX_CONTEXT_TOKENS`` (10000, same as HiRAG).
    """
    raw = os.getenv("ADONIS_NAIVE_RAG_MAX_CONTEXT_TOKENS", "").strip()
    if not raw:
        return DEFAULT_MAX_CONTEXT_TOKENS
    try:
        value = int(raw)
    except ValueError:
        print(
            f"Invalid ADONIS_NAIVE_RAG_MAX_CONTEXT_TOKENS={raw!r}; "
            f"falling back to {DEFAULT_MAX_CONTEXT_TOKENS}"
        )
        return DEFAULT_MAX_CONTEXT_TOKENS
    if value <= 0:
        return DEFAULT_MAX_CONTEXT_TOKENS
    return value


def _count_tokens(text: str) -> int:
    """Count the tokens of a text with tiktoken ``cl100k_base``.

    Same local counting strategy as ``hi_insert_run.py`` (no dependency on
    gateway usage fields).  Special-token-like strings in corpus texts are
    encoded as plain text (``disallowed_special=()``) so chunk content can
    never raise.  Falls back to a chars/4 heuristic when tiktoken is not
    installed.

    Args:
        text: Text to tokenize.

    Returns:
        Number of tokens (or the heuristic estimate without tiktoken).
    """
    global _TIKTOKEN_ENCODER
    try:
        import tiktoken
    except ImportError:  # pragma: no cover - tiktoken is in src/.venv
        return max(1, len(text) // 4)
    if _TIKTOKEN_ENCODER is None:
        _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
    return len(_TIKTOKEN_ENCODER.encode(text, disallowed_special=()))


def _truncate_results_by_token_budget(
    results: list[dict], max_tokens: int
) -> list[dict]:
    """Keep the highest-scored results that fit into the token budget.

    Mirrors ``hirag._utils.truncate_list_by_token_size`` as used by HiRAG's
    naive query mode: results are kept in (score-sorted) order while the
    cumulative passage tokens stay within ``max_tokens``.  Unlike the HiRAG
    helper, at least the single best result is always kept so a degenerate
    oversized first chunk cannot produce an empty context.

    Args:
        results: Score-sorted retrieval results with a ``passage`` field.
        max_tokens: Cumulative passage token budget.

    Returns:
        The kept prefix of ``results``.
    """
    if not results or max_tokens <= 0:
        return results[:1]
    kept: list[dict] = []
    tokens = 0
    for result in results:
        tokens += _count_tokens(result["passage"])
        if tokens > max_tokens and kept:
            break
        kept.append(result)
    return kept


def loadFiles() -> None:
    """Download the rag-mini-wikipedia passages and save them to wiki_txt.csv."""
    df = pd.read_parquet(
        "hf://datasets/rag-datasets/rag-mini-wikipedia/data/passages.parquet/part.0.parquet"
    )
    df.to_csv("wiki_txt.csv")


def load_test_data_chunks() -> pd.DataFrame:
    """Load the pre-chunked Adonis test corpus from chunks.jsonl.

    The corpus is already chunked by the project's hierarchical chunker, so
    this replaces the download + chunking steps of the wiki path.

    Returns:
        DataFrame with ``chunk_id`` and ``passage`` columns, matching the
        shape produced by :func:`chunk_with_overlap`.
    """
    records = []
    with open(TEST_DATA_CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            records.append({"chunk_id": row["id"], "passage": row["text"]})
    print(f"Loaded {len(records)} chunks from {TEST_DATA_CHUNKS_FILE}")
    return pd.DataFrame(records)


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute the cosine similarity between two embedding vectors.

    Args:
        vec1: First embedding vector.
        vec2: Second embedding vector.

    Returns:
        Cosine similarity in [-1, 1], or 0.0 if either vector is zero.
    """
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_vec1 = sum(a * a for a in vec1) ** 0.5
    norm_vec2 = sum(b * b for b in vec2) ** 0.5
    return dot_product / (norm_vec1 * norm_vec2) if norm_vec1 and norm_vec2 else 0.0


def next_neighbor_search(
    embeddings_file: str, query_embedding: list[float], top_k: int = 5
) -> list[dict]:
    """Return the ``top_k`` most similar chunks for a query embedding.

    Args:
        embeddings_file: Path to the JSON embeddings cache.
        query_embedding: Embedding vector of the search query.
        top_k: Number of results to return.

    Returns:
        Result dicts with ``chunk_id``, ``passage`` and ``similarity``,
        sorted by descending similarity.
    """
    with open(embeddings_file, "r", encoding="utf-8") as f:
        embeddings = json.load(f)

    # Calculate cosine similarity between query embedding and document embeddings
    similarities = []
    for idx, record in enumerate(embeddings):
        if isinstance(record, dict) and "embedding" in record:
            doc_embedding = record["embedding"]
            chunk_id = record.get("chunk_id", idx)
            passage = record.get("passage", "")
        else:
            doc_embedding = record
            chunk_id = idx
            passage = ""

        similarity = cosine_similarity(query_embedding, doc_embedding)
        similarities.append(
            {
                "chunk_id": chunk_id,
                "passage": passage,
                "similarity": similarity,
            }
        )

    # Sort by similarity and return top_k results
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return similarities[:top_k]


def _parse_chunk_metadata(raw: Any) -> dict:
    """Parse the ``metadata_`` column of a chunks.jsonl record.

    The export stores metadata either as a JSON object or as a stringified
    Python dict (Postgres pass-through); both forms are accepted.

    Args:
        raw: Raw ``metadata_`` value from a corpus record.

    Returns:
        Parsed metadata dict, or an empty dict when unparseable.
    """
    import ast

    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _load_embedding_matrix(records: list[dict]) -> Any:
    """Return the corpus embedding matrix, preferring inline chunk embeddings.

    Chunks exported by ``data-prep`` (``test-data_40pct``) carry their
    embedding inline as a JSON float array; the ``chunk_embeddings.npy`` file
    is only a legacy fallback for corpora without inline embeddings.

    Args:
        records: Corpus records in ``chunks.jsonl`` row order.

    Returns:
        NumPy float32 matrix of shape ``(num_chunks, embed_dim)``.

    Raises:
        FileNotFoundError: If the chunks carry no inline embeddings and the
            ``.npy`` fallback is missing.
        ValueError: If the fallback matrix does not match the chunk count.
    """
    import numpy as np

    if records and all(
        isinstance(record.get("embedding"), list) and record["embedding"]
        for record in records
    ):
        dimensions = {len(record["embedding"]) for record in records}
        if len(dimensions) == 1:
            print("using inline chunk embeddings (no .npy file needed)")
            return np.asarray(
                [record["embedding"] for record in records], dtype=np.float32
            )

    if not TEST_DATA_EMBEDDINGS_NPY.exists():
        raise FileNotFoundError(
            "Neither inline chunk embeddings nor a precomputed matrix found: "
            f"{TEST_DATA_EMBEDDINGS_NPY}"
        )
    matrix = np.load(TEST_DATA_EMBEDDINGS_NPY)
    if matrix.shape[0] != len(records):
        raise ValueError(
            f"Embedding matrix has {matrix.shape[0]} rows but the corpus has "
            f"{len(records)} chunks — the cache is stale."
        )
    return matrix


def flat_test_data_search(
    query: str, top_k: int = 5, max_context_tokens: int | None = None
) -> list[dict]:
    """Run flat top-k vector retrieval over the pre-chunked test-data corpus.

    Pure vector similarity search without any hierarchy enrichment: the query
    is embedded once with ``adapfkt_ada_002`` and compared against the inline
    chunk embeddings (or the ``chunk_embeddings.npy`` fallback) via cosine
    similarity.  When ``ADONIS_NAIVE_RAG_NUM_QUERIES`` > 1, the query is first
    expanded into LLM variants (production ``QUERY_GEN_PROMPT``), each variant
    is ranked independently, and the rankings are merged with reciprocal rank
    fusion (k=60) — mirroring the production ``QueryFusionRetriever``.
    The ranked results are then truncated to a cumulative passage token
    budget, mirroring HiRAG's naive query mode
    (``naive_max_token_for_text_unit``).

    Args:
        query: Natural-language search query (German or English).
        top_k: Number of most similar chunks to return (also the per-variant
            ranking depth when multi-query is active).
        max_context_tokens: Cumulative passage token budget for the returned
            results; ``None`` (default) uses
            ``ADONIS_NAIVE_RAG_MAX_CONTEXT_TOKENS`` or
            ``DEFAULT_MAX_CONTEXT_TOKENS`` (10000, same as HiRAG naive).

    Returns:
        Result dicts with ``chunk_id``, ``passage``, ``similarity``,
        ``model_name`` and ``source_url``, sorted by descending score (raw
        cosine similarity in single-query mode, RRF score when multi-query
        fusion is active, as in production), truncated to the token budget.

    Raises:
        FileNotFoundError: If no inline embeddings and no ``.npy`` fallback.
        ValueError: If the fallback matrix does not match the chunk count.
    """
    import numpy as np

    records = []
    with open(TEST_DATA_CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            records.append(
                {
                    "chunk_id": row["id"],
                    "passage": row["text"],
                    "metadata": _parse_chunk_metadata(row.get("metadata_")),
                    "embedding": row.get("embedding"),
                }
            )

    matrix = _load_embedding_matrix(records)

    queries = _generate_query_variants(query, get_multi_query_count())

    # Cosine similarity: (A · q) / (|A| * |q|), row-wise over the matrix.
    chunk_norms = np.linalg.norm(matrix, axis=1)

    def _rank_one(query_vector: Any) -> list[tuple[float, int]]:
        """Rank all chunks by cosine similarity to one query vector.

        Args:
            query_vector: Embedded query (1-D NumPy-compatible array).

        Returns:
            ``(similarity, chunk_index)`` tuples, highest similarity first.
        """
        query_norm = np.linalg.norm(query_vector)
        denom = chunk_norms * query_norm
        denom[denom == 0] = 1e-10
        similarities = (matrix @ query_vector) / denom
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(float(similarities[int(i)]), int(i)) for i in top_indices]

    query_vectors = []
    for q in queries:
        embed_response = client.embeddings.create(input=q, model="adapfkt_ada_002")
        if usage_logger is not None:
            usage_logger.track_embedding(
                model="adapfkt_ada_002",
                texts=[q],
                response_obj=embed_response,
                phase="retrieval",
            )
        query_vectors.append(
            np.asarray(embed_response.data[0].embedding, dtype=np.float32)
        )
    per_query_rankings = [_rank_one(vector) for vector in query_vectors]

    if len(per_query_rankings) > 1:
        top_ranked = _reciprocal_rank_fusion(per_query_rankings, top_k)
    else:
        top_ranked = per_query_rankings[0]

    results = []
    for score, chunk_index in top_ranked:
        record = records[chunk_index]
        metadata = record["metadata"]
        results.append(
            {
                "chunk_id": record["chunk_id"],
                "passage": record["passage"],
                "similarity": score,
                "model_name": str(
                    metadata.get("model_name") or metadata.get("title") or ""
                ),
                "source_url": str(metadata.get("source_url") or ""),
            }
        )

    budget = (
        get_max_context_tokens() if max_context_tokens is None else max_context_tokens
    )
    truncated = _truncate_results_by_token_budget(results, budget)
    if len(truncated) < len(results):
        # Same log line as HiRAG's naive query mode.
        print(f"Truncate {len(results)} to {len(truncated)} chunks")
    return truncated


def chunk_with_overlap(df: pd.DataFrame, chunk_size: int, overlap: int) -> pd.DataFrame:
    """Chunk the wiki passages into overlapping windows of DataFrame rows.

    Args:
        df: Wiki passages DataFrame with a ``passage`` column.
        chunk_size: Number of rows per chunk.
        overlap: Number of rows shared between consecutive chunks.

    Returns:
        DataFrame with ``chunk_id`` and ``passage`` columns; also saved to
        ``chunked_wiki_txt.parquet``.
    """
    step = chunk_size - overlap

    chunks = []

    for start in range(0, len(df), step):
        part = df.iloc[start : start + chunk_size]

        if len(part) == 0:
            continue

        chunks.append(
            {
                "chunk_id": len(chunks),
                "passage": "\n".join(part["passage"]),
            }
        )

    chunk_df = pd.DataFrame(chunks)
    chunk_df.to_parquet("chunked_wiki_txt.parquet", index=False)
    return chunk_df


def create_embeddings_for_chunks(chunked_df: pd.DataFrame, output_file: str) -> None:
    """Embed every chunk with the Azure embedding model and cache to JSON.

    Args:
        chunked_df: DataFrame with ``chunk_id`` and ``passage`` columns.
        output_file: Path of the JSON embeddings cache to write.
    """
    print(f"Creating embeddings for {len(chunked_df)} chunks...")
    records = []
    for _, row in chunked_df.iterrows():
        response = client.embeddings.create(input=row["passage"], model=EMBEDDING_MODEL)
        records.append(
            {
                "chunk_id": row["chunk_id"],
                "passage": row["passage"],
                "embedding": response.data[0].embedding,
            }
        )
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f)


def main() -> None:
    """Run the similarity-search experiment for the corpus chosen by SOURCE."""
    embeddings_file = EMBEDDINGS_FILE[SOURCE]

    # --- Corpus preparation (uncomment to regenerate the embeddings) --------
    if SOURCE == "wiki":
        # loadFiles()
        # df_loaded = pd.read_csv("wiki_txt.csv")
        # chunked_df = chunk_with_overlap(df_loaded, chunk_size=5, overlap=2)
        # chunked_df = pd.read_parquet("chunked_wiki_txt.parquet")
        pass
    else:  # "test-data": already chunked, no download or chunking needed.
        # chunked_df = load_test_data_chunks()
        pass
    # create_embeddings_for_chunks(chunked_df, output_file=embeddings_file)

    query = "What is the tallest building in Romania?"
    response = client.embeddings.create(input=query, model="adapfkt_ada_002")
    query_embedding = response.data[0].embedding
    results = next_neighbor_search(embeddings_file, query_embedding)

    print(
        f"Top {len(results)} similar chunks from source '{SOURCE}' "
        f"for the query: '{query}'"
    )
    for result in results:
        print(
            f"chunk_id: {result['chunk_id']}, Similarity: {result['similarity']}\n"
            f"passage: {result['passage']}\n"
        )


if __name__ == "__main__":
    main()
