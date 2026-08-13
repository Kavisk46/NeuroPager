"""Deterministic, append-only trace logging for NeuroPager memory episodes.

:class:`TraceLogger` is the sole place events are constructed and written.
Its ``record_*`` methods only accept parameters describing state that has
already been observed by the caller (a residency snapshot taken before the
current step's effect, counts read from
:class:`~neuropager.core.page_table.PageTable` at call time); there is no
parameter through which a caller could inject information about what
happens later in the episode. The logical tick is likewise entirely
internal: it advances only via :meth:`TraceLogger.begin_step`, never by
looking at wall-clock time or any other external, non-reproducible source.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import TextIO

from neuropager.trace.events import (
    AccessOperation,
    AccessOutcome,
    TraceEvent,
    TraceEventType,
)
from neuropager.utils.types import MemoryKey


class TraceLogger:
    """Records a deterministic, replayable trace of one memory-management episode.

    Every event is appended to :attr:`events` in memory, and — if ``path``
    was given — also written as one JSON line per event to that file,
    flushed immediately. Re-running the identical sequence of manager
    operations against a fresh :class:`TraceLogger` with the same
    ``episode_id`` reproduces byte-identical JSONL output, since nothing in
    this class reads wall-clock time or any other non-deterministic source.

    Attributes:
        episode_id: Identifier stamped onto every event this logger emits.
        path: Optional JSONL output file; ``None`` means in-memory only.
        events: All events recorded so far, in emission order.
    """

    def __init__(self, episode_id: str, path: Path | None = None) -> None:
        """Initialize the trace logger.

        Args:
            episode_id: Identifier for this episode. Callers should supply
                a stable, meaningful id (e.g. ``"lru-cap8-run1"``) rather
                than a randomly generated one, so that trace output stays
                reproducible across runs.
            path: If given, every recorded event is also appended to this
                file as one JSON line. The file and its parent directory
                are created if they do not exist.
        """
        self.episode_id = episode_id
        self.path = path
        self.events: list[TraceEvent] = []
        self._tick = 0
        self._file: TextIO | None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("a", encoding="utf-8")
        else:
            self._file = None

    @property
    def current_tick(self) -> int:
        """The current logical tick (the step in progress, or 0 before the first)."""
        return self._tick

    def begin_step(self) -> int:
        """Advance to the next logical trace tick.

        Called once per top-level
        :class:`~neuropager.core.memory_manager.MemoryManager` operation;
        every event recorded for that operation's consequences (a fault,
        an eviction) shares the resulting tick.

        Returns:
            The new, strictly-incremented tick value.
        """
        self._tick += 1
        return self._tick

    def record_access(
        self,
        *,
        page_id: MemoryKey,
        operation: AccessOperation,
        outcome: AccessOutcome,
        resident_pages: list[MemoryKey],
        working_memory_capacity: int,
        policy: str,
        page_access_count: int,
        page_fault_count: int,
    ) -> TraceEvent:
        """Record an ACCESS event for a resolved ``get``/``put`` call.

        Args:
            page_id: The key that was accessed.
            operation: Whether this was a ``get`` or a ``put``.
            outcome: How the access was resolved.
            resident_pages: Working memory's resident keys immediately
                before this access's own effect was applied.
            working_memory_capacity: Working memory's configured capacity.
            policy: Class name of the active replacement policy.
            page_access_count: The page's cumulative access count as of
                this event.
            page_fault_count: The page's cumulative page-fault count as of
                this event.

        Returns:
            The recorded :class:`~neuropager.trace.events.TraceEvent`.
        """
        return self._emit(
            TraceEvent(
                episode_id=self.episode_id,
                tick=self._tick,
                event_type=TraceEventType.ACCESS,
                page_id=str(page_id),
                working_memory_capacity=working_memory_capacity,
                resident_pages=tuple(str(k) for k in resident_pages),
                policy=policy,
                operation=operation,
                outcome=outcome,
                page_access_count=page_access_count,
                page_fault_count=page_fault_count,
            )
        )

    def record_page_fault(
        self,
        *,
        page_id: MemoryKey,
        resident_pages: list[MemoryKey],
        working_memory_capacity: int,
        policy: str,
        faulted_from_tier: str,
        page_fault_count: int,
    ) -> TraceEvent:
        """Record a PAGE_FAULT event for a key resolved from a non-working tier.

        Args:
            page_id: The key that faulted.
            resident_pages: Working memory's resident keys immediately
                before this fault's resolution begins.
            working_memory_capacity: Working memory's configured capacity.
            policy: Class name of the active replacement policy.
            faulted_from_tier: The tier the page was resolved from.
            page_fault_count: The page's cumulative page-fault count as of
                this event (after this fault).

        Returns:
            The recorded :class:`~neuropager.trace.events.TraceEvent`.
        """
        return self._emit(
            TraceEvent(
                episode_id=self.episode_id,
                tick=self._tick,
                event_type=TraceEventType.PAGE_FAULT,
                page_id=str(page_id),
                working_memory_capacity=working_memory_capacity,
                resident_pages=tuple(str(k) for k in resident_pages),
                policy=policy,
                faulted_from_tier=faulted_from_tier,
                page_fault_count=page_fault_count,
            )
        )

    def record_eviction(
        self,
        *,
        page_id: MemoryKey,
        resident_pages: list[MemoryKey],
        working_memory_capacity: int,
        policy: str,
        reason: str,
        page_access_count: int,
        page_fault_count: int,
    ) -> TraceEvent:
        """Record an EVICTION event for a page removed from working memory.

        Args:
            page_id: The evicted key.
            resident_pages: Working memory's resident keys immediately
                before this eviction (the candidate pool the policy chose
                from, including the victim itself).
            working_memory_capacity: Working memory's configured capacity.
            policy: Class name of the active replacement policy.
            reason: Why the eviction happened (``"capacity"`` or
                ``"explicit"``).
            page_access_count: The evicted page's cumulative access count
                as of this event.
            page_fault_count: The evicted page's cumulative page-fault
                count as of this event.

        Returns:
            The recorded :class:`~neuropager.trace.events.TraceEvent`.
        """
        return self._emit(
            TraceEvent(
                episode_id=self.episode_id,
                tick=self._tick,
                event_type=TraceEventType.EVICTION,
                page_id=str(page_id),
                working_memory_capacity=working_memory_capacity,
                resident_pages=tuple(str(k) for k in resident_pages),
                policy=policy,
                eviction_reason=reason,
                page_access_count=page_access_count,
                page_fault_count=page_fault_count,
            )
        )

    def _emit(self, event: TraceEvent) -> TraceEvent:
        """Append ``event`` to :attr:`events` and, if configured, to disk.

        Args:
            event: The event to record.

        Returns:
            The same event, for convenience at call sites.
        """
        self.events.append(event)
        if self._file is not None:
            self._file.write(json.dumps(event.to_json_dict(), sort_keys=True))
            self._file.write("\n")
            self._file.flush()
        return event

    def close(self) -> None:
        """Close the underlying JSONL file, if one was opened."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> TraceLogger:
        """Enter the context manager, returning this logger."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the underlying file on context exit, if one was opened."""
        self.close()
