"""The trace event schema: one structured record per memory-management occurrence.

A trace is a sequence of :class:`TraceEvent` records. Three kinds of
occurrence are recorded, sharing one schema so a single JSONL file can hold
a full episode:

* ``ACCESS`` — a :class:`~neuropager.core.memory_manager.MemoryManager`
  ``get``/``put`` call resolved, either as a hit or via a fault.
* ``PAGE_FAULT`` — a requested key had to be resolved from a non-working
  tier (this milestone: disk).
* ``EVICTION`` — a resident page was removed from working memory, either
  because capacity was exceeded or by an explicit request.

Every field is derived exclusively from state that already existed *before*
the event's own effect was applied — see the callers in
:mod:`neuropager.core.page_fault` and :mod:`neuropager.core.memory_manager`
for where each snapshot is taken. No field is ever populated from
information that only becomes available after this event.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class TraceEventType(StrEnum):
    """The kind of occurrence a :class:`TraceEvent` records."""

    ACCESS = "access"
    PAGE_FAULT = "page_fault"
    EVICTION = "eviction"


class AccessOperation(StrEnum):
    """Which manager call an ACCESS event represents (``get`` or ``put``)."""

    GET = "get"
    PUT = "put"


class AccessOutcome(StrEnum):
    """How an ACCESS event was resolved.

    Attributes:
        HIT: The key was already resident; no fault or admission needed.
        FAULT: A ``get`` on a non-resident, disk-tracked key required a
            page fault to resolve.
        NEW: A ``put`` on a key never tracked before (fresh creation).
        REWRITE: A ``put`` on a key that is tracked but currently
            non-resident (e.g. previously evicted); it is re-admitted with
            the newly written content rather than the archived content.
    """

    HIT = "hit"
    FAULT = "fault"
    NEW = "new"
    REWRITE = "rewrite"


@dataclass(frozen=True)
class TraceEvent:
    """A single, immutable record in a NeuroPager memory-access trace.

    Attributes:
        episode_id: Identifier for the episode (run) this event belongs to.
        tick: Logical step number in the reference string, shared by every
            event caused by the same top-level
            :class:`~neuropager.core.memory_manager.MemoryManager` call
            (e.g. a fault and the eviction it triggers share one tick).
        event_type: Which kind of occurrence this record represents.
        page_id: The logical key this event concerns (the accessed key, the
            faulted key, or the evicted key).
        working_memory_capacity: Working memory's configured capacity at
            the time of this event.
        resident_pages: The keys resident in working memory immediately
            before this event's own effect was applied.
        policy: Class name of the page replacement policy active at the
            time of this event.
        operation: For ``ACCESS`` events, which manager call this was.
        outcome: For ``ACCESS`` events, how it was resolved.
        faulted_from_tier: For ``PAGE_FAULT`` events, the tier the page was
            resolved from.
        eviction_reason: For ``EVICTION`` events, why the eviction
            happened (``"capacity"`` or ``"explicit"``).
        page_access_count: The subject page's cumulative access count (per
            :class:`~neuropager.core.page_table.PageTableEntry`) as of this
            event.
        page_fault_count: The subject page's cumulative page-fault count as
            of this event.
    """

    episode_id: str
    tick: int
    event_type: TraceEventType
    page_id: str
    working_memory_capacity: int
    resident_pages: tuple[str, ...]
    policy: str
    operation: AccessOperation | None = None
    outcome: AccessOutcome | None = None
    faulted_from_tier: str | None = None
    eviction_reason: str | None = None
    page_access_count: int | None = None
    page_fault_count: int | None = None

    def to_json_dict(self) -> dict[str, Any]:
        """Convert this event to a plain, JSON-serializable dict.

        Returns:
            A dict using only ``str``/``int``/``None``/``list`` values, so
            it can be consumed without importing NeuroPager's internal
            types.
        """
        return {
            "episode_id": self.episode_id,
            "tick": self.tick,
            "event_type": self.event_type.value,
            "page_id": self.page_id,
            "working_memory_capacity": self.working_memory_capacity,
            "resident_pages": list(self.resident_pages),
            "policy": self.policy,
            "operation": self.operation.value if self.operation is not None else None,
            "outcome": self.outcome.value if self.outcome is not None else None,
            "faulted_from_tier": self.faulted_from_tier,
            "eviction_reason": self.eviction_reason,
            "page_access_count": self.page_access_count,
            "page_fault_count": self.page_fault_count,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> TraceEvent:
        """Reconstruct a :class:`TraceEvent` from :meth:`to_json_dict` output.

        Args:
            data: A dict previously produced by :meth:`to_json_dict` (e.g.
                parsed back from a JSONL line).

        Returns:
            The equivalent :class:`TraceEvent`.
        """
        operation = data["operation"]
        outcome = data["outcome"]
        return cls(
            episode_id=data["episode_id"],
            tick=data["tick"],
            event_type=TraceEventType(data["event_type"]),
            page_id=data["page_id"],
            working_memory_capacity=data["working_memory_capacity"],
            resident_pages=tuple(data["resident_pages"]),
            policy=data["policy"],
            operation=AccessOperation(operation) if operation is not None else None,
            outcome=AccessOutcome(outcome) if outcome is not None else None,
            faulted_from_tier=data["faulted_from_tier"],
            eviction_reason=data["eviction_reason"],
            page_access_count=data["page_access_count"],
            page_fault_count=data["page_fault_count"],
        )
