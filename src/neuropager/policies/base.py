"""Abstract base interface for page replacement policies.

A page replacement policy decides which resident memory page to evict when
working memory is full and a new page must be installed. This mirrors
classical OS cache/page replacement algorithms (LRU, LFU, FIFO, Clock,
ARC, ...).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from neuropager.core.page_table import PageTable
from neuropager.utils.types import MemoryKey


class PageReplacementPolicy(ABC):
    """Abstract interface all page replacement policies must implement."""

    @abstractmethod
    def select_victim(self, page_table: PageTable) -> MemoryKey:
        """Select a resident memory key to evict.

        Args:
            page_table: The page table to consult for residency and access
                metadata.

        Returns:
            The logical key of the page selected for eviction.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def on_access(self, key: MemoryKey) -> None:
        """Update internal policy state in response to a page access.

        Called on every read/write of ``key`` so the policy can track
        whatever signal it uses (recency, frequency, etc.).

        Args:
            key: The logical memory identifier that was accessed.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def on_insert(self, key: MemoryKey) -> None:
        """Register a newly resident page with the policy.

        Args:
            key: The logical memory identifier that was inserted.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def on_evict(self, key: MemoryKey) -> None:
        """Notify the policy that a page has been evicted.

        Args:
            key: The logical memory identifier that was evicted.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError
