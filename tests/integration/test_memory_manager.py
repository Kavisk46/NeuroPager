"""End-to-end integration tests for the MemoryManager vertical slice.

Exercises the full collaborator chain — MemoryManager, PageTable,
WorkingMemory, PageFaultHandler, and DiskPageStore — under both LRU and
FIFO replacement policies, confirming the milestone's ``get``/``put``/
``evict`` flows actually work together, not just in isolation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neuropager.core.memory_manager import MemoryManager
from neuropager.core.page_fault import PageFaultHandler
from neuropager.core.page_table import PageTable
from neuropager.core.working_memory import WorkingMemory
from neuropager.memory.disk_store import DiskPageStore
from neuropager.policies.base import PageReplacementPolicy
from neuropager.policies.fifo import FIFOPolicy
from neuropager.policies.lru import LRUPolicy
from neuropager.utils.types import MemoryKey, MemoryTier

KEY_A = MemoryKey("a")
KEY_B = MemoryKey("b")
KEY_C = MemoryKey("c")


def _build_manager(tmp_path: Path, capacity: int, policy: PageReplacementPolicy) -> MemoryManager:
    working_memory = WorkingMemory(capacity=capacity)
    page_table = PageTable()
    fault_handler = PageFaultHandler(
        working_memory=working_memory,
        page_table=page_table,
        disk_store=DiskPageStore(tmp_path),
        policy=policy,
    )
    return MemoryManager(working_memory, page_table, fault_handler)


def test_put_then_get_within_capacity(tmp_path: Path) -> None:
    """Writing and reading keys within capacity never touches disk."""
    manager = _build_manager(tmp_path, capacity=2, policy=LRUPolicy())

    manager.put(KEY_A, "value-a")
    manager.put(KEY_B, "value-b")

    assert manager.get(KEY_A).content == "value-a"
    assert manager.get(KEY_B).content == "value-b"
    assert manager.fault_handler.disk_store.contains(KEY_A) is False


def test_put_beyond_capacity_evicts_to_disk(tmp_path: Path) -> None:
    """Exceeding capacity spills the LRU victim (A: oldest, never re-touched) to disk."""
    manager = _build_manager(tmp_path, capacity=2, policy=LRUPolicy())
    manager.put(KEY_A, "value-a")
    manager.put(KEY_B, "value-b")

    manager.put(KEY_C, "value-c")

    assert manager.page_table.is_resident(KEY_C) is True
    assert manager.working_memory.contains(KEY_A) is False
    assert manager.working_memory.contains(KEY_B) is True
    assert manager.fault_handler.disk_store.contains(KEY_A) is True


def test_get_on_evicted_key_triggers_page_fault_and_resolves(tmp_path: Path) -> None:
    """Fetching an evicted key transparently resolves it from disk."""
    manager = _build_manager(tmp_path, capacity=1, policy=LRUPolicy())
    manager.put(KEY_A, "value-a")
    manager.put(KEY_B, "value-b")  # evicts A to disk

    assert manager.page_table.is_resident(KEY_A) is False

    page = manager.get(KEY_A)

    assert page.content == "value-a"
    assert manager.page_table.is_resident(KEY_A) is True
    # B is now the victim, since it was resident and A was just re-installed.
    assert manager.working_memory.contains(KEY_B) is False


def test_get_unknown_key_raises_key_error(tmp_path: Path) -> None:
    """A key never written raises KeyError rather than faulting silently."""
    manager = _build_manager(tmp_path, capacity=2, policy=LRUPolicy())

    with pytest.raises(KeyError):
        manager.get(KEY_A)


def test_put_updates_resident_key_in_place(tmp_path: Path) -> None:
    """Writing to an already-resident key updates content without eviction."""
    manager = _build_manager(tmp_path, capacity=1, policy=LRUPolicy())
    manager.put(KEY_A, "original")

    manager.put(KEY_A, "updated")

    assert manager.get(KEY_A).content == "updated"
    assert manager.fault_handler.disk_store.contains(KEY_A) is False


def test_explicit_evict_spills_to_disk(tmp_path: Path) -> None:
    """MemoryManager.evict moves a resident key to the disk tier on demand."""
    manager = _build_manager(tmp_path, capacity=2, policy=LRUPolicy())
    manager.put(KEY_A, "value-a")

    manager.evict(KEY_A)

    assert manager.page_table.is_resident(KEY_A) is False
    assert manager.fault_handler.disk_store.contains(KEY_A) is True


def test_evict_unresident_key_raises_key_error(tmp_path: Path) -> None:
    """Explicitly evicting a key that isn't resident raises KeyError."""
    manager = _build_manager(tmp_path, capacity=2, policy=LRUPolicy())

    with pytest.raises(KeyError):
        manager.evict(KEY_A)


def test_lru_policy_evicts_least_recently_used(tmp_path: Path) -> None:
    """Under LRU, re-accessing A before inserting C protects A from eviction."""
    manager = _build_manager(tmp_path, capacity=2, policy=LRUPolicy())
    manager.put(KEY_A, "value-a")
    manager.put(KEY_B, "value-b")
    manager.get(KEY_A)  # A is now most-recently-used; B becomes the LRU victim.

    manager.put(KEY_C, "value-c")

    assert manager.working_memory.contains(KEY_A) is True
    assert manager.working_memory.contains(KEY_B) is False
    assert manager.working_memory.contains(KEY_C) is True


def test_fifo_policy_evicts_oldest_inserted_regardless_of_access(tmp_path: Path) -> None:
    """Under FIFO, re-accessing A does not protect it: B was never inserted first.

    Same put/get sequence as the LRU test above, but with FIFOPolicy the
    outcome differs — this demonstrates that swapping the pluggable policy
    genuinely changes MemoryManager's end-to-end eviction behavior.
    """
    manager = _build_manager(tmp_path, capacity=2, policy=FIFOPolicy())
    manager.put(KEY_A, "value-a")
    manager.put(KEY_B, "value-b")
    manager.get(KEY_A)  # Access does not matter to FIFO.

    manager.put(KEY_C, "value-c")

    assert manager.working_memory.contains(KEY_A) is False
    assert manager.working_memory.contains(KEY_B) is True
    assert manager.working_memory.contains(KEY_C) is True


def test_faulted_page_round_trips_through_disk_tier(tmp_path: Path) -> None:
    """A page's tier correctly reflects WORKING -> DISK -> WORKING transitions."""
    manager = _build_manager(tmp_path, capacity=1, policy=LRUPolicy())
    manager.put(KEY_A, "value-a")
    assert manager.page_table.lookup(KEY_A) == MemoryTier.WORKING

    manager.put(KEY_B, "value-b")  # evicts A
    assert manager.page_table.lookup(KEY_A) == MemoryTier.DISK

    manager.get(KEY_A)  # faults A back in
    assert manager.page_table.lookup(KEY_A) == MemoryTier.WORKING
