import os
from dataclasses import dataclass

from .._utils import load_json, logger, write_json
from ..base import (
    BaseKVStorage,
)


@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        # Load namespace records from JSON, starting with an empty store if no file exists.
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._data = load_json(self._file_name) or {}
        logger.info(f"Load KV {self.namespace} with {len(self._data)} data")

    async def all_keys(self) -> list[str]:
        # Return all record IDs currently held by the key-value store.
        return list(self._data.keys())

    async def index_done_callback(self):
        # Persist the current key-value records to the namespace JSON file.
        write_json(self._data, self._file_name)

    async def get_by_id(self, id):
        # Look up one record by ID, returning None when it is missing.
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        # Look up records in ID order, optionally retaining only the requested fields.
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        # Return only the supplied IDs that are not already stored.
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        # Merge new records into the in-memory key-value store, replacing matching IDs.
        self._data.update(data)

    async def drop(self):
        # Clear all records from the in-memory key-value store.
        self._data = {}
