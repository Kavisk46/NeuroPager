"""Unit tests for :mod:`neuropager.core.page_fault`."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuropager.core.page_fault import PageFaultHandler
from neuropager.core.page_table import PageTable
from neuropager.core.working_memory import WorkingMemory
from neuropager.memory.disk_store import DiskPageStore
from neuropager.policies.lru import LRUPolicy
from neuropager.utils.types import MemoryKey, MemoryTier, create_page

KEY_A = MemoryKey("a")
KEY_B = MemoryKey("b")
KEY_C = MemoryKey("c")


def _make_handler(tmp_path: Path, capacity: int) -> PageFaultHandler:
    return PageFaultHandler(
        working_memory=WorkingMemory(capacity=capacity),
        page_table=PageTable(),
        disk_store=DiskPageStore(tmp_path),
        policy=LRUPolicy(),
    )


def test_handle_fault_raises_for_completely_unknown_key(tmp_path: Path) -> None:
    """A key never seen by any backend raises KeyError."""
    handler = _make_handler(tmp_path, capacity=2)

    with pytest.raises(KeyError):
        handler.handle_fault(KEY_A)


def test_handle_fault_resolves_from_disk_and_installs(tmp_path: Path) -> None:
    """A key tracked as DISK-tier is read back and installed into working memory."""
    handler = _make_handler(tmp_path, capacity=2)
    page = create_page(KEY_A, "archived", tier=MemoryTier.DISK)
    handler.disk_store.write(KEY_A, page)
    handler.page_table.update_tier(KEY_A, MemoryTier.DISK)

    resolved = handler.handle_fault(KEY_A)

    assert resolved.content == "archived"
    assert handler.working_memory.contains(KEY_A) is True
    assert handler.page_table.is_resident(KEY_A) is True


def test_install_evicts_via_policy_when_full(tmp_path: Path) -> None:
    """install() spills the policy-selected victim to disk when at capacity."""
    handler = _make_handler(tmp_path, capacity=1)
    handler.install(create_page(KEY_A, "first"))

    handler.install(create_page(KEY_B, "second"))

    assert handler.working_memory.contains(KEY_A) is False
    assert handler.working_memory.contains(KEY_B) is True
    assert handler.page_table.lookup(KEY_A) == MemoryTier.DISK
    assert handler.disk_store.contains(KEY_A) is True


def test_evict_moves_page_to_disk_and_updates_page_table(tmp_path: Path) -> None:
    """evict() removes a specific key from working memory and persists it."""
    handler = _make_handler(tmp_path, capacity=2)
    handler.install(create_page(KEY_A, "value"))

    evicted = handler.evict(KEY_A)

    assert evicted.content == "value"
    assert handler.working_memory.contains(KEY_A) is False
    assert handler.page_table.lookup(KEY_A) == MemoryTier.DISK
    assert handler.disk_store.contains(KEY_A) is True


def test_evict_raises_for_non_resident_key(tmp_path: Path) -> None:
    """evict() raises KeyError for a key that is not currently resident."""
    handler = _make_handler(tmp_path, capacity=2)

    with pytest.raises(KeyError):
        handler.evict(KEY_C)


def test_faulted_page_can_be_evicted_again(tmp_path: Path) -> None:
    """A page faulted back in from disk can be evicted again (full round trip)."""
    handler = _make_handler(tmp_path, capacity=1)
    handler.install(create_page(KEY_A, "value"))
    handler.evict(KEY_A)

    handler.handle_fault(KEY_A)

    assert handler.working_memory.contains(KEY_A) is True
    assert handler.page_table.is_resident(KEY_A) is True
