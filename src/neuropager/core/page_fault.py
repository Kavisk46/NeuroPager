"""Page fault handling: servicing memory requests not currently resident.

When the :class:`~neuropager.core.memory_manager.MemoryManager` requests a
memory key not present in working memory, a page fault occurs. The
:class:`PageFaultHandler` is responsible for retrieving the content (via
the hybrid retriever), installing it into working memory, and triggering
eviction through the active replacement policy if working memory is full.
"""

from __future__ import annotations

from neuropager.core.page_table import PageTable
from neuropager.core.working_memory import WorkingMemory
from neuropager.policies.base import PageReplacementPolicy
from neuropager.retrieval.hybrid_retriever import HybridRetriever
from neuropager.utils.types import MemoryKey, MemoryPage


class PageFaultHandler:
    """Services page faults by coordinating retrieval and eviction.

    Attributes:
        working_memory: The working memory instance to install pages into.
        page_table: The page table to update on residency changes.
        retriever: The hybrid retriever used to fetch missing content.
        policy: The active page replacement policy, used to select an
            eviction victim when working memory is full.
    """

    def __init__(
        self,
        working_memory: WorkingMemory,
        page_table: PageTable,
        retriever: HybridRetriever,
        policy: PageReplacementPolicy,
    ) -> None:
        """Initialize the page fault handler with its collaborators.

        Args:
            working_memory: The working memory instance to install pages
                into.
            page_table: The page table to update on residency changes.
            retriever: The hybrid retriever used to fetch missing content.
            policy: The active page replacement policy.
        """
        self.working_memory = working_memory
        self.page_table = page_table
        self.retriever = retriever
        self.policy = policy

    def handle_fault(self, key: MemoryKey) -> MemoryPage:
        """Resolve a page fault for ``key``, returning the retrieved page.

        Retrieves the memory content via :attr:`retriever`, evicting a
        victim page via :attr:`policy` if :attr:`working_memory` is full,
        then installs the retrieved page and updates :attr:`page_table`.

        Args:
            key: The logical memory identifier that was not resident.

        Returns:
            The now-resident :class:`~neuropager.utils.types.MemoryPage`.

        Raises:
            KeyError: If ``key`` cannot be resolved by any backend.
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError
