"""First-In-First-Out (FIFO) page replacement policy.

Evicts the resident page that has been in working memory the longest,
regardless of access recency or frequency. Simplest possible baseline
policy, useful as a floor for benchmarking smarter policies.
"""

from __future__ import annotations

from neuropager.core.page_table import PageTable
from neuropager.policies.base import PageReplacementPolicy
from neuropager.utils.types import MemoryKey


class FIFOPolicy(PageReplacementPolicy):
    """Evicts the resident page that was inserted longest ago."""

    def __init__(self) -> None:
        """Initialize the FIFO policy with an empty insertion queue."""
        # TODO(neuropager): back this with a simple queue (e.g. deque)
        # tracking insertion order.

    def select_victim(self, page_table: PageTable) -> MemoryKey:
        """Select the oldest-inserted resident key for eviction.

        Args:
            page_table: The page table to consult for residency and access
                metadata.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError

    def on_access(self, key: MemoryKey) -> None:
        """No-op: FIFO does not consider access recency or frequency.

        Args:
            key: The logical memory identifier that was accessed.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError

    def on_insert(self, key: MemoryKey) -> None:
        """Append ``key`` to the insertion queue.

        Args:
            key: The logical memory identifier that was inserted.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError

    def on_evict(self, key: MemoryKey) -> None:
        """Remove ``key`` from the insertion queue.

        Args:
            key: The logical memory identifier that was evicted.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError
