"""Unit tests for :mod:`neuropager.core.page_table`."""

from __future__ import annotations

import pytest

from neuropager.core.page_table import RECENT_ACCESS_WINDOW, PageTable
from neuropager.utils.types import MemoryKey, MemoryTier

KEY_A = MemoryKey("a")
KEY_B = MemoryKey("b")


def test_lookup_raises_for_untracked_key() -> None:
    """Lookup raises KeyError for a key with no entry."""
    table = PageTable()

    with pytest.raises(KeyError):
        table.lookup(KEY_A)


def test_is_resident_false_for_untracked_key() -> None:
    """is_resident returns False (not an exception) for an unknown key."""
    table = PageTable()

    assert table.is_resident(KEY_A) is False


def test_update_tier_creates_entry_and_tracks_residency() -> None:
    """update_tier creates a new entry and correctly reports residency."""
    table = PageTable()

    table.update_tier(KEY_A, MemoryTier.WORKING)

    assert table.is_tracked(KEY_A) is True
    assert table.is_resident(KEY_A) is True
    assert table.lookup(KEY_A) == MemoryTier.WORKING


def test_update_tier_to_disk_clears_residency() -> None:
    """Moving a key to DISK tier makes is_resident False but keeps it tracked."""
    table = PageTable()
    table.update_tier(KEY_A, MemoryTier.WORKING)

    table.update_tier(KEY_A, MemoryTier.DISK)

    assert table.is_resident(KEY_A) is False
    assert table.is_tracked(KEY_A) is True
    assert table.lookup(KEY_A) == MemoryTier.DISK


def test_record_access_raises_for_untracked_key() -> None:
    """record_access raises KeyError if the key has no entry yet."""
    table = PageTable()

    with pytest.raises(KeyError):
        table.record_access(KEY_A)


def test_record_access_increments_count_and_tick() -> None:
    """record_access increments access_count and advances last_accessed_tick.

    ``get_entry`` returns a live reference into the page table, so the
    "before" snapshot is captured by value rather than by holding the
    entry object itself (which would alias the post-mutation state).
    """
    table = PageTable()
    table.update_tier(KEY_A, MemoryTier.WORKING)
    count_before = table.get_entry(KEY_A).access_count
    tick_before = table.get_entry(KEY_A).last_accessed_tick

    table.record_access(KEY_A)
    entry_after = table.get_entry(KEY_A)

    assert entry_after.access_count == count_before + 1
    assert entry_after.last_accessed_tick > tick_before


def test_resident_keys_excludes_disk_tier() -> None:
    """resident_keys only returns keys currently in the WORKING tier."""
    table = PageTable()
    table.update_tier(KEY_A, MemoryTier.WORKING)
    table.update_tier(KEY_B, MemoryTier.DISK)

    assert table.resident_keys() == [KEY_A]


def test_recency_ordering_is_deterministic_via_logical_clock() -> None:
    """Access order is tracked deterministically, without relying on wall time."""
    table = PageTable()
    table.update_tier(KEY_A, MemoryTier.WORKING)
    table.update_tier(KEY_B, MemoryTier.WORKING)

    table.record_access(KEY_A)

    entry_a = table.get_entry(KEY_A)
    entry_b = table.get_entry(KEY_B)
    assert entry_a.last_accessed_tick > entry_b.last_accessed_tick


def test_evict_removes_entry_entirely() -> None:
    """Evict removes the key from tracking, unlike a tier transition to DISK."""
    table = PageTable()
    table.update_tier(KEY_A, MemoryTier.WORKING)

    table.evict(KEY_A)

    assert table.is_tracked(KEY_A) is False
    with pytest.raises(KeyError):
        table.lookup(KEY_A)


def test_evict_raises_for_untracked_key() -> None:
    """Evict raises KeyError if the key has no entry."""
    table = PageTable()

    with pytest.raises(KeyError):
        table.evict(KEY_A)


def test_get_entry_raises_for_untracked_key() -> None:
    """get_entry raises KeyError if the key has no entry."""
    table = PageTable()

    with pytest.raises(KeyError):
        table.get_entry(KEY_A)


def test_created_tick_set_once_and_never_changes() -> None:
    """created_tick is stamped on first tracking and stays fixed thereafter."""
    table = PageTable()
    table.update_tier(KEY_A, MemoryTier.WORKING)
    created_tick = table.get_entry(KEY_A).created_tick

    table.update_tier(KEY_A, MemoryTier.DISK)
    table.update_tier(KEY_A, MemoryTier.WORKING)

    assert table.get_entry(KEY_A).created_tick == created_tick


def test_record_page_fault_raises_for_untracked_key() -> None:
    """record_page_fault raises KeyError if the key has no entry."""
    table = PageTable()

    with pytest.raises(KeyError):
        table.record_page_fault(KEY_A)


def test_record_page_fault_increments_counter() -> None:
    """record_page_fault increments page_fault_count and nothing else."""
    table = PageTable()
    table.update_tier(KEY_A, MemoryTier.WORKING)
    access_count_before = table.get_entry(KEY_A).access_count

    table.record_page_fault(KEY_A)
    table.record_page_fault(KEY_A)

    entry = table.get_entry(KEY_A)
    assert entry.page_fault_count == 2
    assert entry.access_count == access_count_before


def test_recent_access_ticks_records_history_in_order() -> None:
    """recent_access_ticks accumulates ticks in the order accesses occurred."""
    table = PageTable()
    table.update_tier(KEY_A, MemoryTier.WORKING)

    table.record_access(KEY_A)
    table.record_access(KEY_A)
    table.record_access(KEY_A)

    ticks = list(table.get_entry(KEY_A).recent_access_ticks)
    assert ticks == sorted(ticks)
    assert len(ticks) == 3


def test_recent_access_ticks_is_bounded() -> None:
    """recent_access_ticks never grows past RECENT_ACCESS_WINDOW entries."""
    table = PageTable()
    table.update_tier(KEY_A, MemoryTier.WORKING)

    for _ in range(RECENT_ACCESS_WINDOW + 5):
        table.record_access(KEY_A)

    ticks = list(table.get_entry(KEY_A).recent_access_ticks)
    assert len(ticks) == RECENT_ACCESS_WINDOW
    # The window should hold the most recent ticks, not the oldest.
    assert ticks == sorted(ticks)
    assert ticks[-1] == table.get_entry(KEY_A).last_accessed_tick
