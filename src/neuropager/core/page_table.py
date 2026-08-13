"""Page table: tracks residency and access metadata for every memory unit.

The page table is the source of truth for *where* a given logical memory
key currently lives (working memory, vector store, knowledge graph, or
disk) and the access-pattern metadata (recency, frequency) that page
replacement policies use to make eviction decisions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from neuropager.utils.types import MemoryKey, MemoryTier

RECENT_ACCESS_WINDOW = 8
"""Maximum number of recent access ticks retained per key.

A small bounded window is enough to derive near-term recency-pattern
features (e.g. inter-access intervals) later, without letting a single
long-lived key accumulate unbounded history over a long episode.
"""


@dataclass
class PageTableEntry:
    """Tracked residency and access metadata for a single memory key.

    Recency and insertion order are tracked with a monotonically increasing
    logical clock rather than wall-clock timestamps, so that replacement
    policy decisions — and any trace derived from them — are fully
    deterministic and reproducible regardless of host timer resolution.

    Attributes:
        tier: The physical tier currently holding this key.
        created_tick: Logical tick at which this key was first tracked
            (i.e. first created), never updated again afterward.
        resident_since: Logical tick at which this key most recently
            transitioned into :attr:`~neuropager.utils.types.MemoryTier.WORKING`.
        last_accessed_tick: Logical tick of the most recent access.
        access_count: Total number of recorded accesses.
        page_fault_count: Total number of times this key has been resolved
            via a page fault (i.e. re-installed after being non-resident).
        recent_access_ticks: The last :data:`RECENT_ACCESS_WINDOW` access
            ticks for this key, oldest first.
    """

    tier: MemoryTier
    created_tick: int
    resident_since: int
    last_accessed_tick: int
    access_count: int = 0
    page_fault_count: int = 0
    recent_access_ticks: deque[int] = field(
        default_factory=lambda: deque[int](maxlen=RECENT_ACCESS_WINDOW)
    )


class PageTable:
    """Maps logical memory keys to their current physical tier and metadata."""

    def __init__(self) -> None:
        """Initialize an empty page table."""
        self._entries: dict[MemoryKey, PageTableEntry] = {}
        self._clock = 0

    def _tick(self) -> int:
        """Advance and return the page table's logical clock.

        Returns:
            The new (strictly increasing) clock value.
        """
        self._clock += 1
        return self._clock

    def lookup(self, key: MemoryKey) -> MemoryTier:
        """Return the tier currently holding ``key``.

        Args:
            key: Logical memory identifier to look up.

        Returns:
            The :class:`~neuropager.utils.types.MemoryTier` holding ``key``.

        Raises:
            KeyError: If ``key`` is not tracked by the page table.
        """
        try:
            return self._entries[key].tier
        except KeyError:
            raise KeyError(key) from None

    def is_tracked(self, key: MemoryKey) -> bool:
        """Return whether ``key`` has any entry in the page table.

        Args:
            key: Logical memory identifier to check.

        Returns:
            True if the page table has an entry for ``key``, in any tier.
        """
        return key in self._entries

    def is_resident(self, key: MemoryKey) -> bool:
        """Return whether ``key`` currently resides in working memory.

        Args:
            key: Logical memory identifier to check.

        Returns:
            False if ``key`` is untracked or held in a non-working tier.
        """
        entry = self._entries.get(key)
        return entry is not None and entry.tier == MemoryTier.WORKING

    def update_tier(self, key: MemoryKey, tier: MemoryTier) -> None:
        """Record that ``key`` now resides in ``tier``.

        If ``key`` has no existing entry, one is created. Transitioning
        into :attr:`~neuropager.utils.types.MemoryTier.WORKING` refreshes
        :attr:`PageTableEntry.resident_since`.

        Args:
            key: Logical memory identifier being updated.
            tier: The new tier holding this key.
        """
        tick = self._tick()
        entry = self._entries.get(key)
        if entry is None:
            self._entries[key] = PageTableEntry(
                tier=tier,
                created_tick=tick,
                resident_since=tick if tier == MemoryTier.WORKING else 0,
                last_accessed_tick=tick,
                access_count=0,
            )
            return
        entry.tier = tier
        if tier == MemoryTier.WORKING:
            entry.resident_since = tick

    def record_access(self, key: MemoryKey) -> None:
        """Update access metadata (recency, frequency) for ``key``.

        Invoked on every read/write so that replacement policies can make
        informed eviction decisions.

        Args:
            key: Logical memory identifier that was accessed.

        Raises:
            KeyError: If ``key`` is not tracked by the page table.
        """
        entry = self._entries.get(key)
        if entry is None:
            raise KeyError(key)
        tick = self._tick()
        entry.last_accessed_tick = tick
        entry.access_count += 1
        entry.recent_access_ticks.append(tick)

    def record_page_fault(self, key: MemoryKey) -> None:
        """Increment the page-fault counter for ``key``.

        Called exactly once per resolved fault, by
        :meth:`~neuropager.core.page_fault.PageFaultHandler.handle_fault`
        (the component whose responsibility it is to decide *when* a fault
        has genuinely occurred). ``PageTable`` itself only records facts it
        is explicitly told; it does not infer faults from tier transitions.

        Args:
            key: Logical memory identifier that was faulted in.

        Raises:
            KeyError: If ``key`` is not tracked by the page table.
        """
        entry = self._entries.get(key)
        if entry is None:
            raise KeyError(key)
        entry.page_fault_count += 1

    def get_entry(self, key: MemoryKey) -> PageTableEntry:
        """Return the tracked :class:`PageTableEntry` for ``key``.

        Args:
            key: Logical memory identifier to look up.

        Returns:
            The tracked :class:`PageTableEntry`.

        Raises:
            KeyError: If ``key`` is not tracked by the page table.
        """
        try:
            return self._entries[key]
        except KeyError:
            raise KeyError(key) from None

    def resident_keys(self) -> list[MemoryKey]:
        """Return all keys currently resident in working memory.

        Returns:
            Keys whose tier is
            :attr:`~neuropager.utils.types.MemoryTier.WORKING`, in the
            order they were first tracked.
        """
        return [key for key, entry in self._entries.items() if entry.tier == MemoryTier.WORKING]

    def evict(self, key: MemoryKey) -> None:
        """Remove ``key`` from the page table's tracked entries entirely.

        Args:
            key: Logical memory identifier being evicted entirely.

        Raises:
            KeyError: If ``key`` is not tracked by the page table.
        """
        try:
            del self._entries[key]
        except KeyError:
            raise KeyError(key) from None
