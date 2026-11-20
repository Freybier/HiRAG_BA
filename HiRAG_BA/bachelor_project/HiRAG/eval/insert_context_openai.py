# PROJECT MODIFIED relative to HiRAG 67b2dfa; changed parts are marked below.
import os
import json
# PROJECT MODIFIED: Import project dependencies or the shared configuration used by the
# marked sections.
from hirag import HiRAG

os.environ["OPENAI_API_KEY"] = "***"

DATASET = "mix"
file_path = f"./datasets/{DATASET}/{DATASET}_unique_contexts.json"

graph_func = HiRAG(
    working_dir=f"./datasets/{DATASET}/work_dir_hi",
    enable_hierachical_mode=True,
    embedding_func_max_async=4,
    enable_naive_rag=True,
)

with open(file_path, mode="r") as f:
    unique_contexts = json.load(f)
    graph_func.insert(unique_contexts)
