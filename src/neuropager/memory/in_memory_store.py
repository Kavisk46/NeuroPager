"""In-memory page store: a scientifically-equivalent, zero-filesystem-I/O backend.

Implements the exact same :class:`~neuropager.memory.page_store.PageStore`
contract as :class:`~neuropager.memory.disk_store.DiskPageStore`, including
round-tripping every page through the identical JSON-serialization shape
(so any type coercion JSON would impose on read-back, e.g. tuples becoming
lists, happens here too) -- the only difference is that records live in an
in-process ``dict`` instead of the filesystem.

This exists for experiment orchestration, where evicted pages only ever
need to survive within one process run (a live simulation reading back its
own writes), never across process boundaries or after the process exits.
:class:`~neuropager.memory.disk_store.DiskPageStore` remains the backend
for any use case needing real durability; this class changes nothing about
page-fault handling, eviction decisions, or any policy's observable
behavior -- see ``tests/unit/test_page_store_equivalence.py`` for the
exact-equivalence check between the two backends on a fixed episode.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from neuropager.utils.types import MemoryKey, MemoryPage, MemoryTier


class InMemoryPageStore:
    """Durable-contract, in-memory key-value store for cold memory pages."""

    def __init__(self) -> None:
        """Initialize an empty in-memory store."""
        self._records: dict[str, str] = {}

    def write(self, key: MemoryKey, page: MemoryPage) -> None:
        """Persist a memory page in memory, JSON-round-tripped like DiskPageStore.

        Args:
            key: Logical identifier of the memory page.
            page: The page content and metadata to persist.

        Raises:
            TypeError: If ``page.content`` or ``page.metadata`` is not
                JSON-serializable.
        """
        record: dict[str, Any] = {
            "key": str(page.key),
            "content": page.content,
            "tier": page.tier.value,
            "created_at": page.created_at.isoformat(),
            "last_accessed_at": page.last_accessed_at.isoformat(),
            "access_count": page.access_count,
            "metadata": page.metadata,
        }
        self._records[str(key)] = json.dumps(record)

    def read(self, key: MemoryKey) -> MemoryPage:
        """Read a previously persisted memory page.

        Args:
            key: Logical identifier of the memory page to read.

        Returns:
            The persisted :class:`~neuropager.utils.types.MemoryPage`.

        Raises:
            KeyError: If ``key`` has no persisted page.
        """
        raw = self._records.get(str(key))
        if raw is None:
            raise KeyError(key)
        record: dict[str, Any] = json.loads(raw)
        return MemoryPage(
            key=MemoryKey(record["key"]),
            content=record["content"],
            tier=MemoryTier(record["tier"]),
            created_at=datetime.fromisoformat(record["created_at"]),
            last_accessed_at=datetime.fromisoformat(record["last_accessed_at"]),
            access_count=record["access_count"],
            metadata=record["metadata"],
        )

    def delete(self, key: MemoryKey) -> None:
        """Delete a persisted memory page.

        Args:
            key: Logical identifier of the memory page to delete.

        Raises:
            KeyError: If ``key`` has no persisted page.
        """
        if str(key) not in self._records:
            raise KeyError(key)
        del self._records[str(key)]

    def contains(self, key: MemoryKey) -> bool:
        """Return whether a page for ``key`` is currently persisted.

        Args:
            key: Logical memory identifier to check.
        """
        return str(key) in self._records
