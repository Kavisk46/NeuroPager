"""Unit tests for :mod:`neuropager.memory.in_memory_store`."""

from __future__ import annotations

import pytest

from neuropager.memory.in_memory_store import InMemoryPageStore
from neuropager.utils.types import MemoryKey, MemoryTier, create_page

KEY_A = MemoryKey("a")


def test_write_then_read_round_trip() -> None:
    """A written page can be read back with equivalent content and metadata."""
    store = InMemoryPageStore()
    page = create_page(KEY_A, {"note": "hello"}, metadata={"source": "test"})

    store.write(KEY_A, page)
    loaded = store.read(KEY_A)

    assert loaded.key == page.key
    assert loaded.content == page.content
    assert loaded.tier == page.tier
    assert loaded.metadata == page.metadata
    assert loaded.access_count == page.access_count
    assert loaded.created_at == page.created_at
    assert loaded.last_accessed_at == page.last_accessed_at


def test_read_raises_for_missing_key() -> None:
    """Read raises KeyError if no page was ever written for the key."""
    store = InMemoryPageStore()

    with pytest.raises(KeyError):
        store.read(KEY_A)


def test_contains_reflects_written_state() -> None:
    """Contains is False before write and True after."""
    store = InMemoryPageStore()
    assert store.contains(KEY_A) is False

    store.write(KEY_A, create_page(KEY_A, "x"))

    assert store.contains(KEY_A) is True


def test_delete_removes_page() -> None:
    """Delete removes a persisted page so contains/read reflect its absence."""
    store = InMemoryPageStore()
    store.write(KEY_A, create_page(KEY_A, "x"))

    store.delete(KEY_A)

    assert store.contains(KEY_A) is False
    with pytest.raises(KeyError):
        store.read(KEY_A)


def test_delete_raises_for_missing_key() -> None:
    """Delete raises KeyError if no page was ever written for the key."""
    store = InMemoryPageStore()

    with pytest.raises(KeyError):
        store.delete(KEY_A)


def test_write_rejects_non_json_serializable_content() -> None:
    """Write surfaces a TypeError for content that cannot be JSON-encoded, like DiskPageStore."""
    store = InMemoryPageStore()
    page = create_page(KEY_A, object())

    with pytest.raises(TypeError):
        store.write(KEY_A, page)


def test_round_trip_preserves_disk_tier() -> None:
    """A page written with DISK tier reads back with DISK tier."""
    store = InMemoryPageStore()
    page = create_page(KEY_A, "x", tier=MemoryTier.DISK)

    store.write(KEY_A, page)
    loaded = store.read(KEY_A)

    assert loaded.tier == MemoryTier.DISK


def test_round_trip_coerces_content_like_json() -> None:
    """Content is JSON-round-tripped, so e.g. a tuple becomes a list, matching DiskPageStore."""
    store = InMemoryPageStore()
    page = create_page(KEY_A, (1, 2, 3))

    store.write(KEY_A, page)
    loaded = store.read(KEY_A)

    assert loaded.content == [1, 2, 3]
