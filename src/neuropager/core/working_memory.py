"""Working memory: the agent's bounded, resident "hot" context.

Working memory is the analogue of physical RAM in an OS virtual memory
system — a small, fast-access set of memory pages directly available to the
LLM agent, as opposed to memory paged out to the vector store, knowledge
graph, or disk (see ``neuropager.memory``).
"""

from __future__ import annotations

from neuropager.utils.types import MemoryKey, MemoryPage


class WorkingMemory:
    """Bounded container holding the agent's currently resident memory pages.

    Attributes:
        capacity: Maximum number of pages that may be resident at once.
    """

    def __init__(self, capacity: int) -> None:
        """Initialize working memory with a fixed capacity.

        Args:
            capacity: Maximum number of resident memory pages.
        """
        self.capacity = capacity
        # TODO(neuropager): back this with an actual ordered store once
        # implemented (e.g. dict preserving insertion/access order).

    def __len__(self) -> int:
        """Return the number of pages currently resident.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError

    def is_full(self) -> bool:
        """Return whether working memory is at capacity.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError

    def contains(self, key: MemoryKey) -> bool:
        """Return whether ``key`` is currently resident.

        Args:
            key: Logical memory identifier to check.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError

    def get(self, key: MemoryKey) -> MemoryPage:
        """Return the resident page for ``key``.

        Args:
            key: Logical memory identifier to fetch.

        Raises:
            KeyError: If ``key`` is not resident.
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError

    def insert(self, page: MemoryPage) -> None:
        """Insert a page into working memory.

        Callers are responsible for ensuring capacity is available (e.g. by
        evicting via a :class:`~neuropager.policies.base.PageReplacementPolicy`
        beforehand); this method does not evict on its own.

        Args:
            page: The memory page to insert.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError

    def remove(self, key: MemoryKey) -> MemoryPage:
        """Remove and return the resident page for ``key``.

        Args:
            key: Logical memory identifier to remove.

        Raises:
            KeyError: If ``key`` is not resident.
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError
