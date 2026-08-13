"""Disk page store: durable, cold storage for evicted memory pages.

The disk store is the "swap space" of NeuroPager's memory hierarchy —
where pages evicted from working memory are archived when they are not
otherwise served by the vector store or knowledge graph indexes.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from neuropager.utils.types import MemoryKey, MemoryPage, MemoryTier


class DiskPageStore:
    """Durable key-value store for cold memory pages, backed by JSON files.

    Each page is serialized to its own file, named by the SHA-256 digest of
    its logical key (rather than the raw key) so that keys containing path
    separators or other filesystem-sensitive characters cannot escape the
    store directory.

    Note:
        Page ``content`` and ``metadata`` must be JSON-serializable for this
        backend. Non-serializable payloads will raise ``TypeError`` on
        :meth:`write`.
    """

    def __init__(self, path: Path) -> None:
        """Initialize the disk page store, creating its directory if needed.

        Args:
            path: Filesystem directory used for durable storage.
        """
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def _file_path(self, key: MemoryKey) -> Path:
        """Return the on-disk file path for ``key``.

        Args:
            key: Logical memory identifier.

        Returns:
            A path derived from the SHA-256 digest of ``key``.
        """
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.path / f"{digest}.json"

    def write(self, key: MemoryKey, page: MemoryPage) -> None:
        """Persist a memory page to durable storage.

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
        self._file_path(key).write_text(json.dumps(record), encoding="utf-8")

    def read(self, key: MemoryKey) -> MemoryPage:
        """Read a previously persisted memory page.

        Args:
            key: Logical identifier of the memory page to read.

        Returns:
            The persisted :class:`~neuropager.utils.types.MemoryPage`.

        Raises:
            KeyError: If ``key`` has no persisted page.
        """
        file_path = self._file_path(key)
        if not file_path.exists():
            raise KeyError(key)
        record: dict[str, Any] = json.loads(file_path.read_text(encoding="utf-8"))
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
        file_path = self._file_path(key)
        if not file_path.exists():
            raise KeyError(key)
        file_path.unlink()

    def contains(self, key: MemoryKey) -> bool:
        """Return whether a page for ``key`` is currently persisted.

        Args:
            key: Logical memory identifier to check.
        """
        return self._file_path(key).exists()
