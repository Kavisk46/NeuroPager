"""Working memory: the agent's bounded, resident "hot" context.

Working memory is the analogue of physical RAM in an OS virtual memory
system — a small, fast-access set of memory pages directly available to the
LLM agent, as opposed to memory paged out to the vector store, knowledge
graph, or disk (see ``neuropager.memory``).
"""

from __future__ import annotations

from neuropager.utils.types import MemoryKey, MemoryPage


class WorkingMemoryFullError(RuntimeError):
    """Raised when inserting a new key into a :class:`WorkingMemory` at capacity."""


class WorkingMemory:
    """Bounded container holding the agent's currently resident memory pages.

    Attributes:
        capacity: Maximum number of pages that may be resident at once.
    """

    def __init__(self, capacity: int) -> None:
        """Initialize working memory with a fixed capacity.

        Args:
            capacity: Maximum number of resident memory pages. Must be a
                positive integer.

        Raises:
            ValueError: If ``capacity`` is not a positive integer.
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be a positive integer, got {capacity}")
        self.capacity = capacity
        self._pages: dict[MemoryKey, MemoryPage] = {}

    def __len__(self) -> int:
        """Return the number of pages currently resident."""
        return len(self._pages)

    def is_full(self) -> bool:
        """Return whether working memory is at capacity."""
        return len(self._pages) >= self.capacity

    def contains(self, key: MemoryKey) -> bool:
        """Return whether ``key`` is currently resident.

        Args:
            key: Logical memory identifier to check.
        """
        return key in self._pages

    def get(self, key: MemoryKey) -> MemoryPage:
        """Return the resident page for ``key``.

        Args:
            key: Logical memory identifier to fetch.

        Raises:
            KeyError: If ``key`` is not resident.
        """
        try:
            return self._pages[key]
        except KeyError:
            raise KeyError(key) from None

    def insert(self, page: MemoryPage) -> None:
        """Insert a page into working memory.

        Callers are responsible for ensuring capacity is available (e.g. by
        evicting via a :class:`~neuropager.policies.base.PageReplacementPolicy`
        beforehand); this method does not evict on its own. Re-inserting an
        already-resident key (an update) is always permitted, even at
        capacity.

        Args:
            page: The memory page to insert.

        Raises:
            WorkingMemoryFullError: If working memory is at capacity and
                ``page.key`` is not already resident.
        """
        if self.is_full() and page.key not in self._pages:
            raise WorkingMemoryFullError(
                f"cannot insert key {page.key!r}: working memory is at capacity "
                f"({self.capacity})"
            )
        self._pages[page.key] = page

    def remove(self, key: MemoryKey) -> MemoryPage:
        """Remove and return the resident page for ``key``.

        Args:
            key: Logical memory identifier to remove.

        Raises:
            KeyError: If ``key`` is not resident.
        """
        try:
            return self._pages.pop(key)
        except KeyError:
            raise KeyError(key) from None
