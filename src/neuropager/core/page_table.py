"""Page table: tracks residency and access metadata for every memory unit.

The page table is the source of truth for *where* a given logical memory
key currently lives (working memory, vector store, knowledge graph, or
disk) and the access-pattern metadata (recency, frequency) that page
replacement policies use to make eviction decisions.
"""

from __future__ import annotations

from neuropager.utils.types import MemoryKey, MemoryTier


class PageTable:
    """Maps logical memory keys to their current physical tier and metadata."""

    def __init__(self) -> None:
        """Initialize an empty page table."""
        # TODO(neuropager): back this with a dict[MemoryKey, PageTableEntry]
        # once the entry schema is finalized.

    def lookup(self, key: MemoryKey) -> MemoryTier:
        """Return the tier currently holding ``key``.

        Args:
            key: Logical memory identifier to look up.

        Raises:
            KeyError: If ``key`` is not tracked by the page table.
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError

    def is_resident(self, key: MemoryKey) -> bool:
        """Return whether ``key`` currently resides in working memory.

        Args:
            key: Logical memory identifier to check.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError

    def update_tier(self, key: MemoryKey, tier: MemoryTier) -> None:
        """Record that ``key`` now resides in ``tier``.

        Args:
            key: Logical memory identifier being updated.
            tier: The new tier holding this key.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError

    def record_access(self, key: MemoryKey) -> None:
        """Update access metadata (recency, frequency) for ``key``.

        Invoked on every read/write so that replacement policies can make
        informed eviction decisions.

        Args:
            key: Logical memory identifier that was accessed.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError

    def evict(self, key: MemoryKey) -> None:
        """Remove ``key`` from the page table's tracked entries.

        Args:
            key: Logical memory identifier being evicted entirely.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError
