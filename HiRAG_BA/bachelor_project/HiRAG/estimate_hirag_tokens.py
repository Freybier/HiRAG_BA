# PROJECT ADDED relative to HiRAG 67b2dfa; this module is absent upstream.
"""Offline token-cost estimate for building the HiRAG knowledge graph.

This script estimates the LLM and embedding token consumption of a full
HiRAG indexing run over the ADONIS ``test-data_40pct`` corpus WITHOUT
calling any LLM. It reproduces the exact call structure of
``hirag._op.extract_hierarchical_entities`` and
``hirag._op.generate_community_report``:

- per chunk: 1 entity-extraction call + 1 if-loop check + up to
  ``entity_extract_max_gleaning`` gleaning calls (best model)
- per chunk: same structure again for relation extraction (best model)
- per over-long merged entity/relation description: 1 summarization call
  (cheap model, only when the merged description exceeds
  ``entity_summary_to_max_tokens``)
- per community per hierarchy level: 1 community-report call (best model)
- embeddings: entity descriptions, all chunks (naive RAG), community
  reports (cheap ada-002)

The fixed prompt overhead is measured from the real templates in
``hirag/prompt.py`` rendered with the real delimiter/entity-type values.
Data-driven quantities (entity density, dedup factor, gleaning rate,
community count) cannot be known before the run, so the script reports
three scenarios: optimistic, expected and pessimistic.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import tiktoken

# ---------------------------------------------------------------------------
# Paths and real HiRAG defaults (from hirag/hirag.py and hirag/config.yaml)
# ---------------------------------------------------------------------------

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "test-data"
    / "test-data_40pct"
    / "hirag_records.jsonl"
)
PROMPT_PATH = Path(__file__).resolve().parent / "hirag" / "prompt.py"

CHUNK_TOKEN_SIZE = 1200  # HiRAG default, not overridden in config.yaml
CHUNK_OVERLAP_TOKEN_SIZE = 100  # HiRAG default
TIKTOKEN_MODEL = "cl100k_base"
ENTITY_EXTRACT_MAX_GLEANING = 1  # HiRAG default
ENTITY_SUMMARY_TO_MAX_TOKENS = 500  # HiRAG default
MAX_GRAPH_CLUSTER_SIZE = 10  # HiRAG default (leiden community target)
# config.yaml: enable_naive_rag = true -> chunks are embedded as well
ENABLE_NAIVE_RAG = True


# ---------------------------------------------------------------------------
# Scenario parameters (unknowable before the real run -> scenario-based)
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """Assumptions that can only be verified by a real run.

    Attributes:
        entity_out_tokens: output tokens of one entity-extraction call.
        relation_out_tokens: output tokens of one relation-extraction call.
        gleaning_rate: fraction of chunks where the if-loop judge answers
            "yes" (triggers one extra gleaning call with full history).
        entities_per_chunk: unique entities produced per chunk (before dedup).
        dedup_factor: duplicated entities / unique entities across corpus.
        summary_share: fraction of unique entities+edges whose merged
            description exceeds 500 tokens and needs a summarization call.
        community_input_tokens: average packed community describe size.
        community_report_out_tokens: average community report output size.
        hierarchy_levels: number of community hierarchy levels.
    """

    name: str
    entity_out_tokens: int
    relation_out_tokens: int
    gleaning_rate: float
    entities_per_chunk: float
    dedup_factor: float
    summary_share: float
    community_input_tokens: int
    community_report_out_tokens: int
    hierarchy_levels: int


SCENARIOS = [
    Scenario(
        name="optimistic",
        entity_out_tokens=150,
        relation_out_tokens=120,
        gleaning_rate=0.25,
        entities_per_chunk=6,
        dedup_factor=2.0,
        summary_share=0.05,
        community_input_tokens=4000,
        community_report_out_tokens=500,
        hierarchy_levels=2,
    ),
    Scenario(
        name="expected",
        entity_out_tokens=300,
        relation_out_tokens=250,
        gleaning_rate=0.50,
        entities_per_chunk=10,
        dedup_factor=2.5,
        summary_share=0.15,
        community_input_tokens=8000,
        community_report_out_tokens=800,
        hierarchy_levels=3,
    ),
    Scenario(
        name="pessimistic",
        entity_out_tokens=500,
        relation_out_tokens=400,
        gleaning_rate=0.80,
        entities_per_chunk=15,
        dedup_factor=3.0,
        summary_share=0.30,
        community_input_tokens=12000,
        community_report_out_tokens=1200,
        hierarchy_levels=3,
    ),
]


# PROJECT ADDED: Load hirag/prompt.py standalone so no HiRAG package import is needed.
def load_prompts() -> dict:
    """Load hirag/prompt.py standalone so no HiRAG package import is needed.

    Returns:
        The raw PROMPTS dictionary from the HiRAG package.
    """
    spec = importlib.util.spec_from_file_location("hirag_prompt", PROMPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PROMPTS


# PROJECT ADDED: Count cl100k_base tokens of a string.
def count_tokens(text: str, enc: tiktoken.Encoding) -> int:
    """Count cl100k_base tokens of a string."""
    return len(enc.encode(text))


# PROJECT ADDED: Reproduce chunking_by_token_size sizes for one record.
def chunk_record(token_count: int) -> list[int]:
    """Reproduce chunking_by_token_size sizes for one record.

    Returns:
        List of chunk sizes in tokens (1200 max, 100 overlap).
    """
    if token_count <= CHUNK_TOKEN_SIZE:
        return [token_count]
    sizes: list[int] = []
    start = 0
    while start < token_count:
        end = min(start + CHUNK_TOKEN_SIZE, token_count)
        sizes.append(end - start)
        if end == token_count:
            break
        start = end - CHUNK_OVERLAP_TOKEN_SIZE
    return sizes


# PROJECT ADDED: Compute and print the token estimate for all scenarios.
def main() -> None:
    """Compute and print the token estimate for all scenarios."""
    enc = tiktoken.get_encoding(TIKTOKEN_MODEL)
    prompts = load_prompts()

    # --- 1. Corpus statistics with the real chunking ---------------------
    chunk_sizes: list[int] = []
    total_record_tokens = 0
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            n = count_tokens(record["text"], enc)
            total_record_tokens += n
            chunk_sizes.extend(chunk_record(n))
    num_chunks = len(chunk_sizes)

    # --- 2. Fixed prompt overhead from the real templates ----------------
    context_base = dict(
        tuple_delimiter=prompts["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=prompts["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=prompts["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(prompts["META_ENTITY_TYPES"]),
    )
    entity_tpl_tokens = count_tokens(
        prompts["hi_entity_extraction"].format(**context_base, input_text=""), enc
    )
    # relation template additionally carries the chunk's entity list;
    # approximated as ~10 tokens per entity name, sized per scenario later
    relation_tpl_tokens = count_tokens(
        prompts["hi_relation_extraction"].format(
            tuple_delimiter=prompts["DEFAULT_TUPLE_DELIMITER"],
            record_delimiter=prompts["DEFAULT_RECORD_DELIMITER"],
            completion_delimiter=prompts["DEFAULT_COMPLETION_DELIMITER"],
            entities="",
            input_text="",
        ),
        enc,
    )
    community_tpl_tokens = count_tokens(
        prompts["community_report"].format(input_text=""), enc
    )
    continue_tokens = count_tokens(prompts["entiti_continue_extraction"], enc)
    if_loop_tokens = count_tokens(prompts["entiti_if_loop_extraction"], enc)

    print(f"Corpus: {DATA_PATH.name}")
    print(f"  total record tokens: {total_record_tokens:,}")
    print(
        f"  chunks ({CHUNK_TOKEN_SIZE} tok / {CHUNK_OVERLAP_TOKEN_SIZE} overlap): "
        f"{num_chunks:,}"
    )
    avg_chunk = sum(chunk_sizes) / num_chunks
    print(f"  average chunk size: {avg_chunk:.0f} tokens")
    print("\nPrompt template overhead (measured):")
    print(f"  entity extraction:  {entity_tpl_tokens:,} tokens")
    print(f"  relation extraction: {relation_tpl_tokens:,} tokens (+ entity list)")
    print(f"  community report:   {community_tpl_tokens:,} tokens")
    print(f"  continue / if-loop: {continue_tokens} / {if_loop_tokens} tokens")

    # --- 3. Scenario estimates -------------------------------------------
    for s in SCENARIOS:
        entity_list_tokens = int(s.entities_per_chunk * 10)

        # Entity pass per chunk: extract + if-loop + gleaning with history
        ent_in = ent_out = 0
        for size in chunk_sizes:
            base = entity_tpl_tokens + size
            ent_in += base
            ent_out += s.entity_out_tokens
            # if-loop check sees full history (prompt + first answer)
            ent_in += base + s.entity_out_tokens + if_loop_tokens
            ent_out += 5
            # fractional gleaning: gleaning_rate is the expected share of
            # chunks where the judge answers "yes" and one gleaning call runs
            ent_in += s.gleaning_rate * (base + s.entity_out_tokens + continue_tokens)
            ent_out += s.gleaning_rate * (s.entity_out_tokens // 2)

        # Relation pass per chunk: same shape, plus entity list in prompt
        rel_in = rel_out = 0
        for size in chunk_sizes:
            base = relation_tpl_tokens + entity_list_tokens + size
            rel_in += base
            rel_out += s.relation_out_tokens
            rel_in += base + s.relation_out_tokens + if_loop_tokens
            rel_out += 5
            rel_in += s.gleaning_rate * (base + s.relation_out_tokens + continue_tokens)
            rel_out += s.gleaning_rate * (s.relation_out_tokens // 2)

        # Summarization of over-long merged descriptions (cheap model)
        total_extracted = s.entities_per_chunk * num_chunks
        unique_entities = total_extracted / s.dedup_factor
        unique_edges = unique_entities * 0.8  # edges typically < nodes
        summary_calls = int((unique_entities + unique_edges) * s.summary_share)
        sum_in = summary_calls * (ENTITY_SUMMARY_TO_MAX_TOKENS + 150)
        sum_out = summary_calls * ENTITY_SUMMARY_TO_MAX_TOKENS

        # Community reports: level-0 communities = unique entities /
        # cluster size; higher levels shrink geometrically by the same
        # factor (each parent community groups ~cluster_size children).
        base_communities = unique_entities / MAX_GRAPH_CLUSTER_SIZE
        ratio = 1.0 / MAX_GRAPH_CLUSTER_SIZE
        levels = min(s.hierarchy_levels, 5)
        communities = int(base_communities * sum(ratio**lvl for lvl in range(levels)))
        com_in = communities * (community_tpl_tokens + s.community_input_tokens)
        com_out = communities * s.community_report_out_tokens

        # Embeddings (ada-002): entity descriptions, chunk texts, reports
        emb_entity = int(unique_entities * 120)
        emb_chunks = sum(chunk_sizes) if ENABLE_NAIVE_RAG else 0
        emb_reports = communities * s.community_report_out_tokens
        emb_total = emb_entity + emb_chunks + emb_reports

        llm_in = ent_in + rel_in + sum_in + com_in
        llm_out = ent_out + rel_out + sum_out + com_out
        total_calls = (
            num_chunks * 2  # extraction calls
            + num_chunks * 2  # if-loop checks
            + int(num_chunks * 2 * s.gleaning_rate)  # gleaning calls
            + summary_calls
            + communities
        )

        print(f"\n=== Scenario: {s.name} ===")
        print(f"  LLM calls (chat): {total_calls:,}")
        print(f"  entity pass:   in {ent_in:>12,}  out {ent_out:>10,}")
        print(f"  relation pass: in {rel_in:>12,}  out {rel_out:>10,}")
        print(
            f"  summaries ({summary_calls:,} calls): "
            f"in {sum_in:>10,}  out {sum_out:>10,}"
        )
        print(
            f"  communities ({communities:,} reports): "
            f"in {com_in:>10,}  out {com_out:>10,}"
        )
        print(f"  LLM total:     in {llm_in:>12,}  out {llm_out:>10,}")
        print(f"  embeddings:    {emb_total:>12,} tokens")


if __name__ == "__main__":
    main()
