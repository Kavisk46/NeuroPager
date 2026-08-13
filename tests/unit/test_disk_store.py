"""Unit tests for :mod:`neuropager.memory.disk_store`."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuropager.memory.disk_store import DiskPageStore
from neuropager.utils.types import MemoryKey, MemoryTier, create_page

KEY_A = MemoryKey("a")
KEY_UNSAFE = MemoryKey("../../etc/passwd")


def test_write_creates_store_directory(tmp_path: Path) -> None:
    """Constructing a DiskPageStore creates its backing directory."""
    store_dir = tmp_path / "nested" / "disk_store"

    DiskPageStore(store_dir)

    assert store_dir.exists()


def test_write_then_read_round_trip(tmp_path: Path) -> None:
    """A written page can be read back with equivalent content and metadata."""
    store = DiskPageStore(tmp_path)
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


def test_read_raises_for_missing_key(tmp_path: Path) -> None:
    """Read raises KeyError if no page was ever written for the key."""
    store = DiskPageStore(tmp_path)

    with pytest.raises(KeyError):
        store.read(KEY_A)


def test_contains_reflects_written_state(tmp_path: Path) -> None:
    """Contains is False before write and True after."""
    store = DiskPageStore(tmp_path)
    assert store.contains(KEY_A) is False

    store.write(KEY_A, create_page(KEY_A, "x"))

    assert store.contains(KEY_A) is True


def test_delete_removes_page(tmp_path: Path) -> None:
    """Delete removes a persisted page so contains/read reflect its absence."""
    store = DiskPageStore(tmp_path)
    store.write(KEY_A, create_page(KEY_A, "x"))

    store.delete(KEY_A)

    assert store.contains(KEY_A) is False
    with pytest.raises(KeyError):
        store.read(KEY_A)


def test_delete_raises_for_missing_key(tmp_path: Path) -> None:
    """Delete raises KeyError if no page was ever written for the key."""
    store = DiskPageStore(tmp_path)

    with pytest.raises(KeyError):
        store.delete(KEY_A)


def test_keys_are_hashed_and_cannot_escape_store_directory(tmp_path: Path) -> None:
    """A key containing path-traversal characters stays confined to the store dir."""
    store = DiskPageStore(tmp_path)

    store.write(KEY_UNSAFE, create_page(KEY_UNSAFE, "x"))

    written_files = list(tmp_path.glob("*.json"))
    assert len(written_files) == 1
    assert written_files[0].parent == tmp_path


def test_write_rejects_non_json_serializable_content(tmp_path: Path) -> None:
    """Write surfaces a TypeError for content that cannot be JSON-encoded."""
    store = DiskPageStore(tmp_path)
    page = create_page(KEY_A, object())

    with pytest.raises(TypeError):
        store.write(KEY_A, page)


def test_round_trip_preserves_disk_tier(tmp_path: Path) -> None:
    """A page written with DISK tier reads back with DISK tier."""
    store = DiskPageStore(tmp_path)
    page = create_page(KEY_A, "x", tier=MemoryTier.DISK)

    store.write(KEY_A, page)
    loaded = store.read(KEY_A)

    assert loaded.tier == MemoryTier.DISK
