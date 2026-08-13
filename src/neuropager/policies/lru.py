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
    """Evicts the least-recently-accessed resident page.

    Recency state is not duplicated inside this policy: it is read directly
    from :class:`~neuropager.core.page_table.PageTable`, which already
    tracks a per-key last-accessed logical tick for exactly this purpose.
    This keeps the policy stateless and avoids any risk of the policy's
    bookkeeping drifting out of sync with the page table.
    """

    def select_victim(self, page_table: PageTable) -> MemoryKey:
        """Select the least-recently-used resident key for eviction.

        Args:
            page_table: The page table to consult for residency and access
                metadata.

        Returns:
            The resident key with the smallest (oldest) last-accessed tick.

        Raises:
            ValueError: If ``page_table`` has no resident pages.
        """
        resident = page_table.resident_keys()
        if not resident:
            raise ValueError("cannot select a victim: no resident pages in working memory")
        return min(resident, key=lambda key: page_table.get_entry(key).last_accessed_tick)

    def on_access(self, key: MemoryKey) -> None:
        """No-op: recency is tracked by the page table, not this policy.

        Preserved for API stability and for future policies (e.g. the
        planned learned Memory Utility Model) that require incremental
        per-access state updates.

        Args:
            key: The logical memory identifier that was accessed.
        """

    def on_insert(self, key: MemoryKey) -> None:
        """No-op: insertion recency is tracked by the page table.

        Args:
            key: The logical memory identifier that was inserted.
        """

    def on_evict(self, key: MemoryKey) -> None:
        """No-op: this policy holds no per-key state to clean up.

        Args:
            key: The logical memory identifier that was evicted.
        """
