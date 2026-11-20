# PROJECT MODIFIED relative to HiRAG 67b2dfa; changed parts are marked below.
from dataclasses import dataclass, field
# PROJECT MODIFIED: Import project dependencies or the shared configuration used by the
# marked sections.
from typing import Generic, Literal, Optional, TypedDict, TypeVar, Union

import numpy as np

from ._utils import EmbeddingFunc


@dataclass
class QueryParam:
    mode: Literal[
        "hi_global", "hi_local", "hi_bridge", "hi_nobridge", "naive", "hi"
    ] = "hi"
    only_need_context: bool = False
    response_type: str = "Multiple Paragraphs"
    level: int = 2
    top_k: int = 20  # retrieve top-k entities
    top_m: int = 10  # retrieve top-m entities in each retrieved community
    # naive search
    naive_max_token_for_text_unit = 10000
    # hi search
    max_token_for_text_unit: int = 20000
    max_token_for_local_context: int = 20000
    max_token_for_bridge_knowledge: int = 12500
    max_token_for_community_report: int = 12500
    community_single_one: bool = False
    # Opt-in evaluation capture: when a dict is passed, the query functions
    # fill it with the exact context, its ordered sections and the system
    # prompt of THIS query execution (see ``_op._capture_*``). ``None`` (the
    # default) leaves query behavior and return type completely unchanged.
    # PROJECT ADDED: Allow callers to capture exact retrieval context and generation metadata
    # for evaluation.
    capture: Optional[dict] = None


TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int},
)

SingleCommunitySchema = TypedDict(
    "SingleCommunitySchema",
    {
        "level": int,
        "title": str,
        "edges": list[list[str, str]],
        "nodes": list[str],
        "chunk_ids": list[str],
        "occurrence": float,
        "sub_communities": list[str],
    },
)


class CommunitySchema(SingleCommunitySchema):
    report_string: str
    report_json: dict


T = TypeVar("T")


@dataclass
class StorageNameSpace:
    namespace: str
    global_config: dict

    async def index_start_callback(self):
        """commit the storage operations after indexing"""
        pass

    async def index_done_callback(self):
        """commit the storage operations after indexing"""
        pass

    async def query_done_callback(self):
        """commit the storage operations after querying"""
        pass


@dataclass
class BaseVectorStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc
    meta_fields: set = field(default_factory=set)

    async def query(self, query: str, top_k: int) -> list[dict]:
        # Require storage backends to retrieve the most similar vector records; this base method raises.
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict]):
        """Use 'content' field from value for embedding, use key as id.
        If embedding_func is None, use 'embedding' field from value
        """
        raise NotImplementedError


@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    async def all_keys(self) -> list[str]:
        # Require storage backends to list all stored record IDs; this base method raises.
        raise NotImplementedError

    async def get_by_id(self, id: str) -> Union[T, None]:
        # Require storage backends to look up one record by ID; this base method raises.
        raise NotImplementedError

    async def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[T, None]]:
        # Require storage backends to look up records with optional field filtering; this base method raises.
        raise NotImplementedError

    async def filter_keys(self, data: list[str]) -> set[str]:
        """return un-exist keys"""
        raise NotImplementedError

    async def upsert(self, data: dict[str, T]):
        # Require storage backends to insert or replace records by ID; this base method raises.
        raise NotImplementedError

    async def drop(self):
        # Require storage backends to clear all stored records; this base method raises.
        raise NotImplementedError


@dataclass
class BaseGraphStorage(StorageNameSpace):
    async def has_node(self, node_id: str) -> bool:
        # Require storage backends to check for a graph node; this base method raises.
        raise NotImplementedError

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        # Require storage backends to check for a graph relationship; this base method raises.
        raise NotImplementedError

    async def node_degree(self, node_id: str) -> int:
        # Require storage backends to count relationships incident to a node; this base method raises.
        raise NotImplementedError

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        # Require storage backends to return the combined endpoint degrees; this base method raises.
        raise NotImplementedError

    async def get_node(self, node_id: str) -> Union[dict, None]:
        # Require storage backends to retrieve node attributes; this base method raises.
        raise NotImplementedError

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        # Require storage backends to retrieve relationship attributes; this base method raises.
        raise NotImplementedError

    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        # Require storage backends to list the edges associated with a node; this base method raises.
        raise NotImplementedError

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        # Require storage backends to create or update a graph node; this base method raises.
        raise NotImplementedError

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        # Require storage backends to create or update a graph relationship; this base method raises.
        raise NotImplementedError

    async def clustering(self, algorithm: str):
        # Require storage backends to cluster graph nodes into communities; this base method raises.
        raise NotImplementedError

    async def community_schema(self) -> dict[str, SingleCommunitySchema]:
        """Return the community representation with report and nodes"""
        raise NotImplementedError

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        # Require storage backends to generate node embeddings with their IDs; this base method raises.
        raise NotImplementedError("Node embedding is not used in HiRAG.")
