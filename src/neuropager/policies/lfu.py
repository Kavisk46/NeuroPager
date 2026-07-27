"""Least-Frequently-Used (LFU) page replacement policy.

Evicts the resident page with the lowest cumulative access count. Useful
when access frequency is a more reliable eviction signal than recency
(e.g., frequently-referenced facts vs. one-off episodic detail).
"""

from __future__ import annotations

from neuropager.core.page_table import PageTable
from neuropager.policies.base import PageReplacementPolicy
from neuropager.utils.types import MemoryKey


class LFUPolicy(PageReplacementPolicy):
    """Evicts the least-frequently-accessed resident page."""

    def __init__(self) -> None:
        """Initialize the LFU policy with empty frequency tracking."""
        # TODO(neuropager): back this with a frequency-count structure
        # (e.g. min-heap or frequency buckets) for efficient victim
        # selection.

    def select_victim(self, page_table: PageTable) -> MemoryKey:
        """Select the least-frequently-used resident key for eviction.

        Args:
            page_table: The page table to consult for residency and access
                metadata.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError

    def on_access(self, key: MemoryKey) -> None:
        """Increment the access frequency counter for ``key``.

        Args:
            key: The logical memory identifier that was accessed.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError

    def on_insert(self, key: MemoryKey) -> None:
        """Register a newly resident page with an initial frequency count.

        Args:
            key: The logical memory identifier that was inserted.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError

    def on_evict(self, key: MemoryKey) -> None:
        """Remove ``key`` from frequency tracking.

        Args:
            key: The logical memory identifier that was evicted.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError
