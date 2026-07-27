"""Memory manager: the top-level, agent-facing memory orchestrator.

:class:`MemoryManager` is the single entry point an LLM agent uses to read
and write memory. It delegates to :class:`~neuropager.core.working_memory.WorkingMemory`,
:class:`~neuropager.core.page_table.PageTable`, and
:class:`~neuropager.core.page_fault.PageFaultHandler` — mirroring how an OS
kernel's virtual memory subsystem orchestrates RAM, page tables, and fault
handling, without implementing any of that logic itself.
"""

from __future__ import annotations

from typing import Any

from neuropager.core.page_fault import PageFaultHandler
from neuropager.core.page_table import PageTable
from neuropager.core.working_memory import WorkingMemory
from neuropager.utils.types import MemoryKey, MemoryPage


class MemoryManager:
    """Facade orchestrating working memory, the page table, and page faults.

    Attributes:
        working_memory: The agent's resident memory store.
        page_table: Tracks residency and access metadata for all memory.
        fault_handler: Services misses (page faults) via retrieval.
    """

    def __init__(
        self,
        working_memory: WorkingMemory,
        page_table: PageTable,
        fault_handler: PageFaultHandler,
    ) -> None:
        """Initialize the memory manager with its collaborators.

        Args:
            working_memory: The agent's resident memory store.
            page_table: Tracks residency and access metadata.
            fault_handler: Services page faults via retrieval.
        """
        self.working_memory = working_memory
        self.page_table = page_table
        self.fault_handler = fault_handler

    def get(self, key: MemoryKey) -> MemoryPage:
        """Retrieve a memory unit by key, resolving a page fault if needed.

        Args:
            key: Logical identifier of the memory unit to retrieve.

        Returns:
            The resolved :class:`~neuropager.utils.types.MemoryPage`.

        Raises:
            KeyError: If ``key`` cannot be resolved by any backend.
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError

    def put(self, key: MemoryKey, value: Any) -> None:
        """Write a new or updated memory unit.

        Args:
            key: Logical identifier of the memory unit to write.
            value: The content to store.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError

    def evict(self, key: MemoryKey) -> None:
        """Explicitly evict a memory unit from working memory.

        Args:
            key: Logical identifier of the memory unit to evict.

        Raises:
            KeyError: If ``key`` is not currently resident.
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError
