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
    """Evicts the resident page that was inserted longest ago.

    Insertion order is not duplicated inside this policy: it is read
    directly from :class:`~neuropager.core.page_table.PageTable`, which
    already tracks a per-key ``resident_since`` logical tick for exactly
    this purpose.
    """

    def select_victim(self, page_table: PageTable) -> MemoryKey:
        """Select the oldest-inserted resident key for eviction.

        Args:
            page_table: The page table to consult for residency and access
                metadata.

        Returns:
            The resident key with the smallest (oldest) ``resident_since``
            tick.

        Raises:
            ValueError: If ``page_table`` has no resident pages.
        """
        resident = page_table.resident_keys()
        if not resident:
            raise ValueError("cannot select a victim: no resident pages in working memory")
        return min(resident, key=lambda key: page_table.get_entry(key).resident_since)

    def on_access(self, key: MemoryKey) -> None:
        """No-op: FIFO does not consider access recency or frequency.

        Args:
            key: The logical memory identifier that was accessed.
        """

    def on_insert(self, key: MemoryKey) -> None:
        """No-op: insertion order is tracked by the page table.

        Args:
            key: The logical memory identifier that was inserted.
        """

    def on_evict(self, key: MemoryKey) -> None:
        """No-op: this policy holds no per-key state to clean up.

        Args:
            key: The logical memory identifier that was evicted.
        """
