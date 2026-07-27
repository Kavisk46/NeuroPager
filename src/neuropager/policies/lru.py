"""Least-Recently-Used (LRU) page replacement policy.

Evicts the resident page whose most recent access is furthest in the past.
Classical baseline replacement policy; see ``docs/research.md`` for
discussion of when LRU is and isn't a good fit for agent memory access
patterns.
"""

from __future__ import annotations

from neuropager.core.page_table import PageTable
from neuropager.policies.base import PageReplacementPolicy
from neuropager.utils.types import MemoryKey


class LRUPolicy(PageReplacementPolicy):
    """Evicts the least-recently-accessed resident page."""

    def __init__(self) -> None:
        """Initialize the LRU policy with empty recency tracking."""
        # TODO(neuropager): back this with an ordered structure (e.g.
        # OrderedDict) tracking access recency in O(1).

    def select_victim(self, page_table: PageTable) -> MemoryKey:
        """Select the least-recently-used resident key for eviction.

        Args:
            page_table: The page table to consult for residency and access
                metadata.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError

    def on_access(self, key: MemoryKey) -> None:
        """Mark ``key`` as most-recently-used.

        Args:
            key: The logical memory identifier that was accessed.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError

    def on_insert(self, key: MemoryKey) -> None:
        """Register a newly resident page as most-recently-used.

        Args:
            key: The logical memory identifier that was inserted.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError

    def on_evict(self, key: MemoryKey) -> None:
        """Remove ``key`` from recency tracking.

        Args:
            key: The logical memory identifier that was evicted.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError
