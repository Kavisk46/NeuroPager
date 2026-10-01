"""Page fault handling: servicing memory requests not currently resident.

When the :class:`~neuropager.core.memory_manager.MemoryManager` requests a
memory key not present in working memory, a page fault occurs. The
:class:`PageFaultHandler` is responsible for retrieving the content,
installing it into working memory, and triggering eviction through the
active replacement policy if working memory is full.

For this milestone, faults are resolved against the
:class:`~neuropager.memory.disk_store.DiskPageStore` only: a page fault
here is a direct key lookup (the OS analogue of resolving a known virtual
address), not an open-ended semantic search, so it does not require the
still-unimplemented :class:`~neuropager.retrieval.hybrid_retriever.HybridRetriever`.
The retriever collaborator is accepted optionally so that future,
semantically-triggered fault resolution can be layered in without another
breaking constructor change.

If a :class:`~neuropager.trace.logger.TraceLogger` is attached, this class
also emits ``PAGE_FAULT`` and ``EVICTION`` trace events, always snapshotting
:attr:`page_table`'s resident keys *before* the corresponding state change
is applied, so no event ever reflects information from after its own
occurrence.
"""

from __future__ import annotations

from neuropager.core.page_table import PageTable
from neuropager.core.working_memory import WorkingMemory
from neuropager.memory.disk_store import DiskPageStore
from neuropager.policies.base import PageReplacementPolicy
from neuropager.retrieval.hybrid_retriever import HybridRetriever
from neuropager.trace.logger import TraceLogger
from neuropager.utils.types import MemoryKey, MemoryPage, MemoryTier


class PageFaultHandler:
    """Services page faults and administers working-memory admission/eviction.

    This class is the single chokepoint through which pages enter and leave
    working memory, so that eviction bookkeeping (page table updates,
    policy notifications, disk spill, trace events) never has to be
    duplicated by callers such as
    :class:`~neuropager.core.memory_manager.MemoryManager`.

    Attributes:
        working_memory: The working memory instance to install pages into.
        page_table: The page table to update on residency changes.
        disk_store: Durable store consulted to resolve faults and to
            receive spilled pages on eviction.
        policy: The active page replacement policy, used to select an
            eviction victim when working memory is full.
        retriever: Optional hybrid retriever for future semantic fault
            resolution. Unused by this milestone's direct key-based faults.
        trace_logger: Optional trace logger. When set, every fault and
            eviction this handler performs is recorded.
    """

    def __init__(
        self,
        working_memory: WorkingMemory,
        page_table: PageTable,
        disk_store: DiskPageStore,
        policy: PageReplacementPolicy,
        retriever: HybridRetriever | None = None,
        trace_logger: TraceLogger | None = None,
    ) -> None:
        """Initialize the page fault handler with its collaborators.

        Args:
            working_memory: The working memory instance to install pages
                into.
            page_table: The page table to update on residency changes.
            disk_store: Durable store used to resolve faults and receive
                evicted pages.
            policy: The active page replacement policy.
            retriever: Optional hybrid retriever, reserved for future
                semantic fault resolution.
            trace_logger: Optional trace logger for recording faults and
                evictions.
        """
        self.working_memory = working_memory
        self.page_table = page_table
        self.disk_store = disk_store
        self.policy = policy
        self.retriever = retriever
        self.trace_logger = trace_logger

    def handle_fault(self, key: MemoryKey) -> MemoryPage:
        """Resolve a page fault for ``key``, returning the retrieved page.

        Expects ``key`` to not already be resident in working memory;
        callers (e.g. :class:`~neuropager.core.memory_manager.MemoryManager`)
        are responsible for checking residency first.

        Args:
            key: The logical memory identifier that was not resident.

        Returns:
            The now-resident :class:`~neuropager.utils.types.MemoryPage`.

        Raises:
            KeyError: If ``key`` cannot be resolved by any backend.
        """
        if self.page_table.is_tracked(key) and self.page_table.lookup(key) == MemoryTier.DISK:
            page = self.disk_store.read(key)
        else:
            raise KeyError(f"page fault unresolved for key {key!r}: no backend holds this page")

        resident_before = self.page_table.resident_keys()
        self.page_table.record_page_fault(key)
        if self.trace_logger is not None:
            entry = self.page_table.get_entry(key)
            self.trace_logger.record_page_fault(
                page_id=key,
                resident_pages=resident_before,
                working_memory_capacity=self.working_memory.capacity,
                policy=type(self.policy).__name__,
                faulted_from_tier=MemoryTier.DISK.value,
                page_fault_count=entry.page_fault_count,
            )

        self.install(page)
        return page

    def install(self, page: MemoryPage) -> None:
        """Admit ``page`` into working memory, evicting a victim if full.

        Expects ``page.key`` to not already be resident; updates in place
        should go through the caller rather than a re-install.

        Args:
            page: The page to admit into working memory.
        """
        if self.working_memory.is_full():
            self._evict_victim()
        self.working_memory.insert(page)
        self.page_table.update_tier(page.key, MemoryTier.WORKING)
        self.page_table.record_access(page.key)
        self.policy.on_insert(page.key)

    def evict(self, key: MemoryKey, *, reason: str = "explicit") -> MemoryPage:
        """Evict a specific resident key, spilling it to the disk store.

        Args:
            key: Logical memory identifier of the resident page to evict.
            reason: Why the eviction is happening (``"capacity"`` when
                triggered by :meth:`install`, ``"explicit"`` for a direct
                request), recorded on the trace event if tracing is
                enabled.

        Returns:
            The evicted :class:`~neuropager.utils.types.MemoryPage`.

        Raises:
            KeyError: If ``key`` is not currently resident.
        """
        resident_before = self.page_table.resident_keys()
        page = self.working_memory.remove(key)
        self.disk_store.write(key, page)
        self.page_table.update_tier(key, MemoryTier.DISK)
        self.policy.on_evict(key)
        if self.trace_logger is not None:
            entry = self.page_table.get_entry(key)
            self.trace_logger.record_eviction(
                page_id=key,
                resident_pages=resident_before,
                working_memory_capacity=self.working_memory.capacity,
                policy=type(self.policy).__name__,
                reason=reason,
                page_access_count=entry.access_count,
                page_fault_count=entry.page_fault_count,
            )
        return page

    def _evict_victim(self) -> MemoryKey:
        """Select and evict a victim page chosen by :attr:`policy`.

        Returns:
            The key of the evicted victim page.
        """
        victim_key = self.policy.select_victim(self.page_table)
        self.evict(victim_key, reason="capacity")
        return victim_key
