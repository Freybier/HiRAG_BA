# RAGAS evaluation (HiRAG / naive-RAG / test-data RAG)

Two-step workflow:

1. **collect**: run one RAG system over a question dataset and save answers plus
   the exact retrieved contexts into a run directory;
2. **score**: compute RAGAS metrics for one or more saved runs.

Collect and score are separated on purpose: collection uses each system's own
venv, scoring uses a dedicated eval venv with the local, unmodified RAGAS
checkout (`../ragas`). No dataset, questions or reference answers are shipped
here; the dataset has to be written by hand from real ADONIS content.

## Dataset (`questions.jsonl`)

One JSON object per line:

| field                   | required | meaning                                                        |
| ----------------------- | -------- | -------------------------------------------------------------- |
| `question_id`           | yes      | unique string                                                  |
| `user_input`            | yes      | the question                                                   |
| `reference`             | no       | reference answer (enables reference-based metrics)             |
| `reference_context_ids` | no       | list of ADONIS source IDs (process `model_id` / document uuid) |
| `metadata`              | no       | free-form object, passed through                               |

Unknown fields, duplicate IDs, invalid JSON and empty files are rejected with
all problems listed at once (`dataset.py`).

## Collect

Every collector writes `evaluation/runs/<system>_<label>_<timestamp>/` with
`run_metadata.json` + `records.jsonl` (format: `run_format.py`).
`--resume --out-dir <dir>` retries only missing/errored questions, `--limit N`
takes the first N questions.

```bash
# HiRAG (hirag_venv, run from HiRAG/)
cd src/app/bachelor_project/HiRAG
hirag_venv/bin/python ../evaluation/collect_hirag.py --dataset /path/questions.jsonl \
    [--working-dir ./hirag_index_samples/hirag_index_sample_20p_4d] [--mode hi]

# naive-RAG / Streamlit "Flach" (src venv, run from src/)
cd src
.venv/bin/python app/bachelor_project/evaluation/collect_naive.py --dataset /path/questions.jsonl --top-k 5

# parent-child test-data RAG (src venv, run from src/; in-process test_data_search)
.venv/bin/python app/bachelor_project/evaluation/collect_test_data.py --dataset /path/questions.jsonl
```

HiRAG contexts are captured through the opt-in `QueryParam.capture` hook in the
vendored `hirag` (default `None` = unchanged behavior). Context capture only
records what was already sent to the answer LLM.

## Score

```bash
cd src/app/bachelor_project/evaluation
uv venv --python 3.12 .venv-eval
uv pip install --python .venv-eval/bin/python -r requirements-eval.txt
.venv-eval/bin/python score.py --check-setup            # no API calls
.venv-eval/bin/python score.py --run-dir runs/<run> --dataset /path/questions.jsonl \
    [--metrics faithfulness,answer_relevancy] [--context-granularity items|source_chunks_only|full] \
    [--concurrency 4] [--limit N]
```

`requirements-eval.txt` pins langchain < 1.0 / langchain-community < 0.4: the
RAGAS checkout leaves langchain unpinned but still imports modules removed in
the 1.x line (`langchain_community.chat_models.vertexai`).

Metrics: `faithfulness`, `answer_relevancy`, `context_precision_with_reference`,
`context_precision_without_reference`, `context_recall`, `answer_correctness`,
`id_context_precision`, `id_context_recall` (ID metrics only for runs with
`id_granularity: adonis_source_record`, currently HiRAG only).

Output: `<run-dir>/scores_<timestamp>_<granularity>/` with `scores.jsonl` (per
question and metric), `scores.csv`, `aggregate.json`, `score_metadata.json`. An
existing `scores.jsonl` is never overwritten.

Per-metric status:

- `scored`: a real value; `0.0` is a valid score and counts in the mean;
- `skipped`: preconditions missing (no reference, no contexts, no comparable
  IDs, question missing in the run, run record errored) with a reason;
- `failed`: the metric raised or returned NaN; error recorded, excluded from the
  mean.

`aggregate.json` reports the mean over scored values plus scored/skipped/failed
counts per metric, so missing data is never silently turned into zeros.

## Configuration

Evaluator models come from `bachelor_project/config.yaml`, section `evaluation:`
(`llm_model` = gateway deployment, `ragas_model_name` = model name RAGAS uses
for its own reasoning/adapter logic, `embedding_model`, `max_tokens`,
`prompt_language`); CLI flags override them. API key via
`llm_config.get_api_key` (`src/app/.env-dev`). Token usage of RAGAS calls is not
tracked.

**Bias warning:** the judge LLM is from the same model family as the answer
models, so self-preference bias is possible. Compare systems relative to each
other, not as absolute quality.

## FUTURE TODO: common structure for all RAG systems

The three systems are not yet directly comparable:

1. Corpus mismatch: the HiRAG indexes contain only sample records (e.g. 24), the
   chunk systems use the full 40pct corpus (567 models / 7819 chunks).
2. `chunks.jsonl` contains no document chunks.
3. ID formats differ (`{uuid}` / bare uuid / `model_{uuid}`); the chunk systems
   therefore get no ID-based metrics.
4. naive-RAG/test-data answers use `config.yaml` `chat_model`, not production
   `Settings.llm`.
5. Context formats differ (HiRAG JSON/CSV sections vs chunk text).
6. Quellen/source lists are not context items.
7. Env flags (e.g. `ADONIS_TEST_DATA_*`) affect results; they are recorded per
   run.
8. `collect_test_data` runs `test_data_search` in-process, not through the
   Docker MCP server.

Goal: one shared input-data and ID structure for all systems.
