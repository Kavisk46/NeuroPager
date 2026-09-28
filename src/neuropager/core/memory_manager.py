"""Memory manager: the top-level, agent-facing memory orchestrator.

:class:`MemoryManager` is the single entry point an LLM agent uses to read
and write memory. It delegates to :class:`~neuropager.core.working_memory.WorkingMemory`,
:class:`~neuropager.core.page_table.PageTable`, and
:class:`~neuropager.core.page_fault.PageFaultHandler` — mirroring how an OS
kernel's virtual memory subsystem orchestrates RAM, page tables, and fault
handling, without implementing any of that logic itself.

If :attr:`~neuropager.core.page_fault.PageFaultHandler.trace_logger` is
set, every ``get``/``put`` call emits one ACCESS trace event, sharing its
logical tick with any PAGE_FAULT/EVICTION events the same call triggers
inside :class:`~neuropager.core.page_fault.PageFaultHandler`. Residency
snapshots used in trace events are always captured before this call's own
effect is applied.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from neuropager.core.page_fault import PageFaultHandler
from neuropager.core.page_table import PageTable
from neuropager.core.working_memory import WorkingMemory
from neuropager.trace.events import AccessOperation, AccessOutcome
from neuropager.utils.types import MemoryKey, MemoryPage, create_page


class MemoryManager:
    """Facade orchestrating working memory, the page table, and page faults.

    Attributes:
        working_memory: The agent's resident memory store.
        page_table: Tracks residency and access metadata for all memory.
        fault_handler: Services misses (page faults) via retrieval, and
            administers all working-memory admission/eviction. Also holds
            the optional shared :class:`~neuropager.trace.logger.TraceLogger`.
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
        """
        trace_logger = self.fault_handler.trace_logger
        resident_before: list[MemoryKey] = []
        resident_snapshot: list[MemoryKey] | None = None
        if trace_logger is not None:
            trace_logger.begin_step()
            resident_before = self.page_table.resident_keys()
            resident_snapshot = resident_before

        if self.page_table.is_resident(key):
            page = self.working_memory.get(key)
            self._record_hit(page)
            outcome = AccessOutcome.HIT
        else:
            # Reusing this snapshot (rather than letting handle_fault()
            # recompute it) is safe: nothing between this line and
            # handle_fault()'s own use of it changes any key's tier -- see
            # PageFaultHandler.handle_fault()'s docstring for the full
            # argument.
            page = self.fault_handler.handle_fault(key, resident_snapshot=resident_snapshot)
            outcome = AccessOutcome.FAULT

        if trace_logger is not None:
            entry = self.page_table.get_entry(key)
            trace_logger.record_access(
                page_id=key,
                operation=AccessOperation.GET,
                outcome=outcome,
                resident_pages=resident_before,
                working_memory_capacity=self.working_memory.capacity,
                policy=type(self.fault_handler.policy).__name__,
                page_access_count=entry.access_count,
                page_fault_count=entry.page_fault_count,
            )
        return page

    def put(self, key: MemoryKey, value: Any) -> None:
        """Write a new or updated memory unit.

        If ``key`` is already resident, its content is updated in place.
        If ``key`` is tracked but not resident (e.g. previously evicted),
        it is re-admitted with the newly written content. Otherwise a new
        page is created and admitted. In both non-resident cases,
        admission goes through the fault handler, evicting a victim if
        necessary.

        Args:
            key: Logical identifier of the memory unit to write.
            value: The content to store.
        """
        trace_logger = self.fault_handler.trace_logger
        resident_before: list[MemoryKey] = []
        resident_snapshot: list[MemoryKey] | None = None
        if trace_logger is not None:
            trace_logger.begin_step()
            resident_before = self.page_table.resident_keys()
            resident_snapshot = resident_before

        if self.page_table.is_resident(key):
            page = self.working_memory.get(key)
            page.content = value
            self._record_hit(page)
            outcome = AccessOutcome.HIT
        elif self.page_table.is_tracked(key):
            # Snapshot reuse is safe here too: is_resident()/is_tracked()
            # above are pure reads, so nothing has changed any key's tier
            # since resident_before was captured -- see
            # PageFaultHandler.handle_fault()'s docstring.
            self.fault_handler.install(create_page(key, value), resident_snapshot=resident_snapshot)
            outcome = AccessOutcome.REWRITE
        else:
            self.fault_handler.install(create_page(key, value), resident_snapshot=resident_snapshot)
            outcome = AccessOutcome.NEW

        if trace_logger is not None:
            entry = self.page_table.get_entry(key)
            trace_logger.record_access(
                page_id=key,
                operation=AccessOperation.PUT,
                outcome=outcome,
                resident_pages=resident_before,
                working_memory_capacity=self.working_memory.capacity,
                policy=type(self.fault_handler.policy).__name__,
                page_access_count=entry.access_count,
                page_fault_count=entry.page_fault_count,
            )

    def evict(self, key: MemoryKey) -> None:
        """Explicitly evict a memory unit from working memory.

        Args:
            key: Logical identifier of the memory unit to evict.

        Raises:
            KeyError: If ``key`` is not currently resident.
        """
        trace_logger = self.fault_handler.trace_logger
        if trace_logger is not None:
            trace_logger.begin_step()
        self.fault_handler.evict(key, reason="explicit")

    def _record_hit(self, page: MemoryPage) -> None:
        """Update page table, policy, and page metadata for a resident hit.

        :class:`~neuropager.core.page_table.PageTable` is the authoritative
        source used for eviction decisions; the counters mirrored onto
        ``page`` itself are purely informational for callers inspecting a
        :class:`~neuropager.utils.types.MemoryPage` directly.

        Args:
            page: The resident page that was just accessed.
        """
        self.page_table.record_access(page.key)
        self.fault_handler.policy.on_access(page.key)
        entry = self.page_table.get_entry(page.key)
        page.access_count = entry.access_count
        page.last_accessed_at = datetime.now(UTC)
