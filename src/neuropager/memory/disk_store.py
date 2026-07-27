"""Disk page store: durable, cold storage for evicted memory pages.

The disk store is the "swap space" of NeuroPager's memory hierarchy —
where pages evicted from working memory are archived when they are not
otherwise served by the vector store or knowledge graph indexes.
"""

from __future__ import annotations

from pathlib import Path

from neuropager.utils.types import MemoryKey, MemoryPage


class DiskPageStore:
    """Durable key-value store for cold memory pages."""

    def __init__(self, path: Path) -> None:
        """Initialize the disk page store.

        Args:
            path: Filesystem directory (or backend-specific location) used
                for durable storage.
        """
        self.path = path
        # TODO(neuropager): back this with actual filesystem (or object
        # storage) persistence.

    def write(self, key: MemoryKey, page: MemoryPage) -> None:
        """Persist a memory page to durable storage.

        Args:
            key: Logical identifier of the memory page.
            page: The page content and metadata to persist.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError

    def read(self, key: MemoryKey) -> MemoryPage:
        """Read a previously persisted memory page.

        Args:
            key: Logical identifier of the memory page to read.

        Returns:
            The persisted :class:`~neuropager.utils.types.MemoryPage`.

        Raises:
            KeyError: If ``key`` has no persisted page.
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError

    def delete(self, key: MemoryKey) -> None:
        """Delete a persisted memory page.

        Args:
            key: Logical identifier of the memory page to delete.

        Raises:
            KeyError: If ``key`` has no persisted page.
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError
