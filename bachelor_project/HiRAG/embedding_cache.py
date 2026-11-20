"""Append-only, crash-safe embedding vector cache for ``hi_insert_run.py``.

Motivation: the HiRAG LLM cache (``kv_store_llm_response_cache.json``) makes
interrupted insertion runs resumable at LLM-call granularity, but embedding
calls were not cached at all - the nano-vectordb stores are only committed at
the end of ``HiRAG.insert()``, so a killed run re-embedded everything on
resume. This module adds the same per-call durability for embeddings.

Storage layout (both files live in the HiRAG working dir, next to the LLM
cache and the other HiRAG stores):

- ``embedding_cache_vectors.f32``: raw float32 matrix, one ``dim``-wide row
  per cached text; row ``i`` sits at byte offset ``i * dim * 4``. Append-only.
- ``embedding_cache_index.jsonl``: one JSON line per cached text
  (``{"key", "row", "model", "dim"}``), appended in the same flush *after*
  the vector bytes.

Flush order matters: vector bytes are written and fsynced first, then the
index lines. A crash therefore leaves at worst a torn index tail or a
partial trailing vector row; both are detected on load (index rows must
point inside the durable part of the vector file, and the file size must be
a whole number of rows) and silently truncated away.

The cache is purely additive: it never touches the LLM cache, the
nano-vectordb stores or any other HiRAG state, so enabling it before
resuming an interrupted insertion run is safe - it simply starts empty and
caches every embedding computed from then on.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

INDEX_FILENAME = "embedding_cache_index.jsonl"
VECTORS_FILENAME = "embedding_cache_vectors.f32"

_ROW_DTYPE = np.float32


def embedding_cache_key(model: str, text: str) -> str:
    """Compute the stable cache key for one (model, text) pair.

    Args:
        model: Embedding deployment name; part of the key so switching the
            embedding model can never silently return stale vectors.
        text: Embedded input text.

    Returns:
        Hex-digest key namespaced with the ``emb-`` prefix.
    """
    digest = hashlib.md5(f"{model}\n{text}".encode("utf-8")).hexdigest()
    return f"emb-{digest}"


class EmbeddingCache:
    """Append-only cache mapping ``(model, text)`` to a vector row.

    All mutating methods are synchronous (no ``await`` between the in-memory
    update and the disk flush), so concurrent asyncio tasks from HiRAG's
    ``limit_async_func_call`` wrapper cannot interleave a half-written entry.
    """

    def __init__(self, working_dir: str | Path, dim: int, model: str) -> None:
        """Open (or lazily create) the cache stored inside ``working_dir``.

        Args:
            working_dir: HiRAG working dir; created when missing.
            dim: Expected embedding dimension. A mismatch with an existing
                cache raises ``ValueError`` (delete the cache files or use a
                fresh working dir in that case).
            model: Embedding deployment name, stored per entry for audit.
        """
        self.working_dir = Path(working_dir)
        self.dim = int(dim)
        self.model = model
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.working_dir / INDEX_FILENAME
        self.vectors_path = self.working_dir / VECTORS_FILENAME
        self._rows_by_key: dict[str, int] = {}
        self._vectors = np.empty((0, self.dim), dtype=_ROW_DTYPE)
        self._next_row = 0
        self._load()

    def __len__(self) -> int:
        """Return the number of usable cached vectors."""
        return len(self._rows_by_key)

    def _load(self) -> None:
        """Load index and vectors, truncating torn writes from a crash.

        Index entries whose row points beyond the durable vector rows (or
        whose JSON line is torn) are dropped; a partial trailing vector row
        is ignored because only whole ``dim``-sized rows count.
        """
        row_bytes = self.dim * np.dtype(_ROW_DTYPE).itemsize
        valid_rows = 0
        if self.vectors_path.is_file():
            size = self.vectors_path.stat().st_size
            valid_rows = size // row_bytes
            if valid_rows:
                self._vectors = np.fromfile(
                    self.vectors_path,
                    dtype=_ROW_DTYPE,
                    count=valid_rows * self.dim,
                ).reshape(valid_rows, self.dim)
        self._next_row = valid_rows
        if not self.index_path.is_file():
            return
        with self.index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn final line after a crash
                if entry.get("dim") != self.dim:
                    raise ValueError(
                        f"Embedding cache dimension mismatch: cache was built "
                        f"with dim={entry.get('dim')}, current dim={self.dim}. "
                        f"Delete {self.index_path} and {self.vectors_path} or "
                        f"use a fresh working dir."
                    )
                row = entry.get("row")
                key = entry.get("key")
                if not isinstance(row, int) or not isinstance(key, str):
                    continue
                if row >= valid_rows:
                    continue  # index line without durable vector bytes
                self._rows_by_key.setdefault(key, row)

    def lookup(self, text: str) -> np.ndarray | None:
        """Return the cached vector for ``text``, or ``None`` on a miss.

        Args:
            text: Embedded input text (keyed together with ``self.model``).

        Returns:
            The ``(dim,)`` float32 vector, or ``None`` when not cached.
        """
        row = self._rows_by_key.get(embedding_cache_key(self.model, text))
        if row is None:
            return None
        return self._vectors[row]

    def add(self, texts: list[str], vectors: np.ndarray) -> int:
        """Append new text -> vector entries and flush them to disk.

        Already-cached texts are skipped, making the call idempotent when
        two concurrent batches embed the same text. The flush writes the
        vector bytes first (fsync), then the index lines (fsync), so a crash
        can never leave an index entry without durable vector data.

        Args:
            texts: Embedded texts, aligned with the rows of ``vectors``.
            vectors: ``(len(texts), dim)`` matrix; cast to float32.

        Returns:
            Number of entries actually appended (0 when all were cached).

        Raises:
            ValueError: When ``vectors`` does not match
                ``(len(texts), self.dim)``.
        """
        vectors = np.asarray(vectors, dtype=_ROW_DTYPE)
        expected = (len(texts), self.dim)
        if vectors.shape != expected:
            raise ValueError(
                f"Embedding cache add shape mismatch: got {vectors.shape}, "
                f"expected {expected}."
            )
        pending: list[tuple[str, int]] = []
        new_rows: list[np.ndarray] = []
        for i, text in enumerate(texts):
            key = embedding_cache_key(self.model, text)
            if key in self._rows_by_key:
                continue
            self._rows_by_key[key] = self._next_row
            pending.append((key, self._next_row))
            self._next_row += 1
            new_rows.append(vectors[i])
        if not pending:
            return 0
        new_matrix = np.stack(new_rows).astype(_ROW_DTYPE, copy=False)
        # 1) vector bytes first ...
        with self.vectors_path.open("ab") as handle:
            new_matrix.tofile(handle)
            handle.flush()
            os.fsync(handle.fileno())
        # 2) ... then the index lines referencing them.
        with self.index_path.open("a", encoding="utf-8") as handle:
            for key, row in pending:
                handle.write(
                    json.dumps(
                        {
                            "key": key,
                            "row": row,
                            "model": self.model,
                            "dim": self.dim,
                        }
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        self._vectors = np.concatenate([self._vectors, new_matrix], axis=0)
        return len(pending)
