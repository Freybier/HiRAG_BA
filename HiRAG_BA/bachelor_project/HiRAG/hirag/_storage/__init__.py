# PROJECT MODIFIED relative to HiRAG 67b2dfa; changed parts are marked below.
from .gdb_networkx import NetworkXStorage  # noqa: F401

# PROJECT ADDED: Keep NetworkX-only usage importable when the optional Neo4j driver is
# absent.
try:
    from .gdb_neo4j import Neo4jStorage
except ImportError:  # neo4j driver is optional; only needed for Neo4jStorage
    Neo4jStorage = None

from .kv_json import JsonKVStorage  # noqa: F401
from .vdb_nanovectordb import NanoVectorDBStorage  # noqa: F401
