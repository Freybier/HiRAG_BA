# Bachelor-Project Experiments

Everything in this folder is **standalone research/experiment code** and is
deliberately decoupled from the production Adonis MCP path
(`app/mcp_standalone.py`, `app/postgres/`, `app/nl_to_query_converters/`).

## Running the Streamlit frontend (all three RAG modes)

The Streamlit app ([streamlit_app.py](streamlit_app.py)) is a single UI with
three retrieval modes selected in the sidebar:

1. **Flach (top-k Vektorsuche)** — naive flat RAG over the local test corpus
   ([naive_RAG/openAI_Test_chunked.py](naive_RAG/openAI_Test_chunked.py)).
2. **Hierarchisch (test-data Pipeline)** — hierarchical retrieval through the
   MCP server pipeline ([src/app/test_data_search.py](../test_data_search.py)).
3. **HiRAG (Graph-RAG Index)** — queries a pre-built HiRAG graph index
   ([hirag_query.py](hirag_query.py)); HiRAG synthesizes the answer itself and
   the UI lists its source documents.

### 1. Prerequisites (once)

- The `src/.venv` virtual environment must exist and be healthy (see
  [coder.yaml](../../../coder.yaml) `verifyPythonVenv` job; interpreter must be
  on the persistent `/workspace` volume).
- Fill in the API key: copy `src/app/.env-dev.example` to `src/app/.env-dev` (or
  edit the existing `.env-dev`) and set `AZURE_OPENAI_API_KEY` (the api.central
  gateway key). All modes need it for the query embedding and answer generation.
  Get the real value from the workspace owner.
- The local corpus must exist at `src/test-data/test-data_40pct/`
  (`chunks.jsonl` with inline embeddings — it does; `chunk_embeddings.npy` is
  optional legacy). Regenerate it only if needed:
  `cd src/test-data && ../.venv/bin/python data-prep/data_preparation.py`.

### 2. Start the frontend

```bash
cd src && poe ui
```

The `ui` task automatically sets `STAGE=test`, `ADONIS_TEST_DATA_SEARCH=true`,
`ADONIS_TEST_DATA_VECTOR_SEARCH=true`, `AI_PROVIDER=openai_azure` and unsets any
stale shell `AZURE_OPENAI_API_KEY`, then starts Streamlit. Open the printed
local URL.

### 3. Start the MCP server (only for the "Hierarchisch" mode)

The hierarchical mode talks to the MCP server in Docker. The server must have
the test-data flags and the **container path** of the corpus:

```bash
cd src
export STAGE=test ADONIS_TEST_DATA_SEARCH=true ADONIS_TEST_DATA_DIR=/app/test-data/test-data_40pct
poe agent-up      # also starts local postgres; waits for health
```

(`/app` is the compose mount of `./src`.) Logs: `poe agent-logs`. Stop
everything: `poe agent-down`. The Naive and HiRAG modes do **not** need the MCP
server.

### 4. HiRAG indexes

The HiRAG mode can only **query** existing indexes under
[HiRAG/hirag_index_samples/](HiRAG/hirag_index_samples/) (currently
`hirag_index_complete`). Querying works directly in `src/.venv` — nothing else
to start. Building a **new** index requires the separate `hirag_venv`:

```bash
cd src/app/bachelor_project/HiRAG
hirag_venv/bin/python -u hi_insert_run.py --mode sample   # or: --mode all
```

### 5. Choosing the models used by the frontend

All models are switched centrally in [config.yaml](config.yaml) — **no code
changes or CLI flags**:

- `models.answer_generation` → the answer model for **all three modes**
  (currently `adapfkt_gpt6_1_sol`).
- `stage_models.stages.*` → per-stage HiRAG **indexing** models (active when
  `use_stage_models: true`; answer generation is not an indexing stage).
- `models.chat_model` / `models.cheap_model` → fallback best/cheap split when
  stage routing is off.
- `models.embedding_model` → embedding deployment (`adapfkt_ada_002`).
- `models.multi_query_model` → model for multi-query expansion (enabled via
  `ADONIS_NAIVE_RAG_NUM_QUERIES` / `ADONIS_TEST_DATA_NUM_QUERIES`, default `1` =
  off).
- `stage_models.model_aliases` maps the short alias names to real gateway
  deployments.

Env overrides per shell: `HIRAG_CHAT_MODEL` / `HIRAG_EMBEDDING_MODEL` /
`HIRAG_MULTI_QUERY_MODEL`.

### 6. Per-question usage logging

Every frontend question is logged (tokens, cost, models) to
[usage_logs/frontend_usage.jsonl](usage_logs/frontend_usage.jsonl) (gitignored)
and shown as a caption under each answer. Disable with `USAGE_LOG=0`. The driver
[run_frontend_usage_batch.py](run_frontend_usage_batch.py) replays the full
question dataset through all modes headlessly.

Contents:

- `HiRAG/` — vendored fork of the HiRAG hierarchical-RAG package plus the
  `hi_Search_*.py` entry scripts. Run scripts **from inside the HiRAG folder**
  so the vendored relative paths (`./data`, `./datasets`,
  `./hirag_index_samples/hirag_index_cohere`) keep resolving:

  ```bash
  cd src/app/bachelor_project/HiRAG
  ../../../.venv/bin/python hi_Search_openai.py
  ```

  `HiRAG/config.yaml` is kept only as a **backup**; the scripts actually load
  `../config.yaml` (the bachelor_project config) via the shared
  `../llm_config.py` next to it.

- `llm_config.py` — shared config loader used by ALL experiments here (`HiRAG`,
  `basic-RAG`, and any future RAG). Exposes `get_chat_model`,
  `get_embedding_model`, `get_api_key`, and the convenience constants
  `CHAT_MODEL` / `EMBEDDING_MODEL`.

- `basic-RAG/` — flat top-k cosine-similarity RAG scripts. Configuration comes
  from `../config.yaml`; the API key comes from `AZURE_OPENAI_API_KEY` (see
  `src/app/.env-dev`).

- `streamlit_app.py` — Streamlit frontend. Start it with the poe task:

  ```bashsn
  cd src && poe ui
  ```

## Central configuration

`config.yaml` in this folder is the single place to switch the chat model and
the embedding model for every experiment above:

```yaml
models:
  chat_model: "adapfkt_gpt5"
  embedding_model: "adapfkt_ada_002"
```

Per-environment overrides: `HIRAG_CHAT_MODEL` / `HIRAG_EMBEDDING_MODEL`.

**API keys live only in `src/app/.env-dev`** (`AZURE_OPENAI_API_KEY` /
`X_GATEWAY_API_KEY`). If a key is missing, the scripts fail with a clear error
telling you to set the environment variable.
