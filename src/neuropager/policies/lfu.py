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
    """Evicts the least-frequently-accessed resident page.

    Frequency state is not duplicated inside this policy: it is read
    directly from :class:`~neuropager.core.page_table.PageTable`, which
    already tracks a per-key cumulative access count for exactly this
    purpose. Ties (equal access counts) are broken by the older
    ``last_accessed_tick``, so victim selection stays fully deterministic.
    """

    def select_victim(self, page_table: PageTable) -> MemoryKey:
        """Select the least-frequently-used resident key for eviction.

        Args:
            page_table: The page table to consult for residency and access
                metadata.

        Returns:
            The resident key with the smallest access count, breaking ties
            by the smallest (oldest) last-accessed tick.

        Raises:
            ValueError: If ``page_table`` has no resident pages.
        """
        resident = page_table.resident_keys()
        if not resident:
            raise ValueError("cannot select a victim: no resident pages in working memory")

        def sort_key(key: MemoryKey) -> tuple[int, int]:
            entry = page_table.get_entry(key)
            return (entry.access_count, entry.last_accessed_tick)

        return min(resident, key=sort_key)

    def on_access(self, key: MemoryKey) -> None:
        """No-op: access frequency is tracked by the page table.

        Preserved for API stability and for future policies (e.g. the
        planned learned Memory Utility Model) that require incremental
        per-access state updates.

        Args:
            key: The logical memory identifier that was accessed.
        """

    def on_insert(self, key: MemoryKey) -> None:
        """No-op: initial frequency is tracked by the page table.

        Args:
            key: The logical memory identifier that was inserted.
        """

    def on_evict(self, key: MemoryKey) -> None:
        """No-op: this policy holds no per-key state to clean up.

        Args:
            key: The logical memory identifier that was evicted.
        """
