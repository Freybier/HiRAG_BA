"""Vector storage adapter for HiRAG backed by nano-vectordb.

Uses :class:`LeanNanoVectorDB`, a drop-in subclass that avoids the
multi-hundred-MB transient memory spikes of the stock implementation while
keeping the on-disk JSON format byte-identical.
"""

import asyncio
import base64
import json
import mmap
import os
from dataclasses import dataclass

import numpy as np
from nano_vectordb import NanoVectorDB

from .._utils import logger
from ..base import BaseVectorStorage

_FLOAT = np.float32

#: Raw bytes per chunk when streaming the base64 matrix on save. Must be a
#: multiple of 3 so the concatenated base64 chunks stay valid.
_SAVE_CHUNK_BYTES = 3 * 1024 * 1024

#: Base64 characters per chunk when streaming the matrix on load. Must be a
#: multiple of 4 so each chunk decodes cleanly.
_LOAD_CHUNK_CHARS = 4 * 1024 * 1024


def _load_storage_stock(raw: mmap.mmap) -> dict:
    """Whole-file JSON parse identical to nano-vectordb's own loader.

    Args:
        raw: The mmap'd storage file content.

    Returns:
        The parsed storage dict with the matrix as an ndarray.
    """
    data = json.loads(raw[:])
    data["matrix"] = np.frombuffer(
        base64.b64decode(data["matrix"]), dtype=_FLOAT
    ).reshape(-1, data["embedding_dim"])
    logger.info(f"Load {data['matrix'].shape} data (stock parser)")
    return data


def _load_storage_lean(file_name: str) -> dict | None:
    """Load a nano-vectordb JSON store without big transient allocations.

    The file is memory-mapped (page cache, not anonymous RAM), the base64
    matrix — always the last JSON key — is located by byte markers and
    decoded in chunks straight into a writable buffer, and only the small
    head (``embedding_dim`` + ``data`` records) goes through the JSON parser.
    Falls back to the stock whole-file parser when the expected layout is
    not found.

    Args:
        file_name: Path to the ``vdb_*.json`` storage file.

    Returns:
        The storage dict (``embedding_dim``/``data``/``matrix``), or None
        when the file does not exist.
    """
    if not os.path.exists(file_name):
        return None
    if os.path.getsize(file_name) == 0:
        raise ValueError(f"{file_name}: empty vector store file")
    with open(file_name, "rb") as f:
        raw = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            marker = b', "matrix": "'
            idx = raw.rfind(marker)
            if idx == -1 or raw[-2:] != b'"}':
                return _load_storage_stock(raw)
            head = json.loads(raw[:idx] + b"}")
            start = idx + len(marker)
            b64_len = len(raw) - start - 2
            rows = len(head["data"])
            dim = head["embedding_dim"]
            expected = rows * dim * np.dtype(_FLOAT).itemsize
            buf = bytearray(expected)
            view = memoryview(buf)
            out_off = 0
            for off in range(0, b64_len, _LOAD_CHUNK_CHARS):
                decoded = base64.b64decode(
                    raw[start + off : start + off + _LOAD_CHUNK_CHARS]
                )
                view[out_off : out_off + len(decoded)] = decoded
                out_off += len(decoded)
            if out_off != expected:
                raise ValueError(
                    f"{file_name}: matrix size mismatch (decoded {out_off} "
                    f"bytes, expected {expected} for {rows} rows)"
                )
            head["matrix"] = np.frombuffer(buf, dtype=_FLOAT).reshape(rows, dim)
            logger.info(f"Load {head['matrix'].shape} data (streaming)")
            return head
        finally:
            raw.close()


class LeanNanoVectorDB(NanoVectorDB):
    """``NanoVectorDB`` without the giant transient memory allocations.

    The storage file format is 100% identical to nano-vectordb 0.0.2, so
    existing indexes load unchanged and files written here stay readable by
    the stock library. Only the in-memory behavior differs:

    * ``__post_init__`` loads the store via a streaming chunked base64
      decoder over a memory-mapped file, so the 662 MB entities store no
      longer materializes as a JSON string + parsed base64 string + decoded
      bytes + matrix copy (~2 GB transient) — the matrix lands directly in
      one writable 513 MB buffer.
    * ``save`` streams the base64 matrix to disk in small chunks instead of
      materializing the whole base64 string (~684 MB for the entities store)
      plus a full ``tobytes`` copy, and writes atomically (tmp file + fsync
      + ``os.replace``), so a kill mid-flush can no longer tear the file.
    * ``pre_process`` normalizes the matrix in place instead of allocating
      a second full-size copy (~513 MB for the entities store).
    """

    def __post_init__(self) -> None:
        """Mirror the parent init, but load via the streaming low-RAM path."""
        default_storage = {
            "embedding_dim": self.embedding_dim,
            "data": [],
            "matrix": np.empty((0, self.embedding_dim), dtype=_FLOAT),
        }
        storage = _load_storage_lean(self.storage_file) or default_storage
        assert storage["embedding_dim"] == self.embedding_dim, (
            f"Embedding dim mismatch, expected: {self.embedding_dim}, "
            f"but loaded: {storage['embedding_dim']}"
        )
        self._NanoVectorDB__storage = storage
        self.usable_metrics = {"cosine": self._cosine_query}
        assert self.metric in self.usable_metrics, f"Metric {self.metric} not supported"
        self.pre_process()
        logger.info(
            f"Init LeanNanoVectorDB {self.storage_file} {len(storage['data'])} data"
        )

    @property
    def _storage(self) -> dict:
        """Return the name-mangled private storage dict of the parent.

        Returns:
            The parent's ``_NanoVectorDB__storage`` dict with keys
            ``embedding_dim``, ``data`` and ``matrix``.
        """
        return self._NanoVectorDB__storage

    def pre_process(self) -> None:
        """Normalize the loaded matrix in place (no full-size copy)."""
        if self.metric != "cosine":
            return
        matrix = self._storage["matrix"]
        if matrix.size == 0:
            return
        if not matrix.flags.writeable:
            # The legacy JSON load produces a read-only np.frombuffer view;
            # one copy is unavoidable before in-place updates/normalization.
            matrix = matrix.copy()
            self._storage["matrix"] = matrix
        norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
        np.divide(matrix, norms, out=matrix)

    def save(self) -> None:
        """Write the store atomically without a giant base64 string in RAM.

        The output is byte-compatible with nano-vectordb's own ``save``:
        same key order, same default JSON separators, same base64 payload.
        """
        storage = self._storage
        matrix = np.ascontiguousarray(storage["matrix"], dtype=_FLOAT)
        tmp_file = f"{self.storage_file}.tmp"
        with open(tmp_file, "w") as f:
            f.write('{"embedding_dim": ')
            json.dump(storage["embedding_dim"], f)
            f.write(', "data": ')
            json.dump(storage["data"], f)
            f.write(', "matrix": "')
            view = memoryview(matrix)
            for offset in range(0, matrix.nbytes, _SAVE_CHUNK_BYTES):
                chunk = view[offset : offset + _SAVE_CHUNK_BYTES]
                f.write(base64.b64encode(chunk).decode("ascii"))
            f.write('"}')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, self.storage_file)


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._client = LeanNanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )
        self.cosine_better_than_threshold = self.global_config.get(
            "query_better_than_threshold", self.cosine_better_than_threshold
        )

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]
        results = self._client.upsert(datas=list_data)
        return results

    async def query(self, query: str, top_k=5):
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        return results

    async def index_done_callback(self):
        self._client.save()
