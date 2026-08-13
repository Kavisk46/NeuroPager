"""Unit tests for :mod:`neuropager.core.working_memory`."""

from __future__ import annotations

import pytest

from neuropager.core.working_memory import WorkingMemory, WorkingMemoryFullError
from neuropager.utils.types import MemoryKey, create_page

KEY_A = MemoryKey("a")
KEY_B = MemoryKey("b")
KEY_C = MemoryKey("c")


def test_invalid_capacity_raises_value_error() -> None:
    """Non-positive capacity is rejected at construction time."""
    with pytest.raises(ValueError):
        WorkingMemory(capacity=0)

    with pytest.raises(ValueError):
        WorkingMemory(capacity=-1)


def test_empty_working_memory_state() -> None:
    """A freshly constructed WorkingMemory is empty and not full."""
    memory = WorkingMemory(capacity=2)

    assert len(memory) == 0
    assert memory.is_full() is False
    assert memory.contains(KEY_A) is False


def test_insert_and_get_round_trip() -> None:
    """A page inserted can be retrieved by key."""
    memory = WorkingMemory(capacity=2)
    page = create_page(KEY_A, "hello")

    memory.insert(page)

    assert memory.contains(KEY_A) is True
    assert memory.get(KEY_A) is page
    assert len(memory) == 1


def test_get_raises_for_missing_key() -> None:
    """Get raises KeyError for a non-resident key."""
    memory = WorkingMemory(capacity=2)

    with pytest.raises(KeyError):
        memory.get(KEY_A)


def test_insert_beyond_capacity_raises() -> None:
    """Inserting a new key while at capacity raises WorkingMemoryFullError."""
    memory = WorkingMemory(capacity=1)
    memory.insert(create_page(KEY_A, "first"))

    with pytest.raises(WorkingMemoryFullError):
        memory.insert(create_page(KEY_B, "second"))


def test_insert_update_of_existing_key_allowed_at_capacity() -> None:
    """Re-inserting an already-resident key is allowed even at capacity."""
    memory = WorkingMemory(capacity=1)
    memory.insert(create_page(KEY_A, "first"))

    memory.insert(create_page(KEY_A, "updated"))

    assert memory.get(KEY_A).content == "updated"
    assert len(memory) == 1


def test_is_full_reflects_capacity() -> None:
    """is_full becomes True once the number of resident pages hits capacity."""
    memory = WorkingMemory(capacity=2)
    memory.insert(create_page(KEY_A, "a"))
    assert memory.is_full() is False

    memory.insert(create_page(KEY_B, "b"))
    assert memory.is_full() is True


def test_remove_returns_page_and_frees_capacity() -> None:
    """Remove pops the page and reduces occupancy."""
    memory = WorkingMemory(capacity=1)
    page = create_page(KEY_A, "a")
    memory.insert(page)

    removed = memory.remove(KEY_A)

    assert removed is page
    assert len(memory) == 0
    assert memory.contains(KEY_A) is False


def test_remove_raises_for_missing_key() -> None:
    """Remove raises KeyError for a non-resident key."""
    memory = WorkingMemory(capacity=1)

    with pytest.raises(KeyError):
        memory.remove(KEY_C)
