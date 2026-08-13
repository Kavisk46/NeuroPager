"""Unit tests for :mod:`neuropager.utils.types`."""

from __future__ import annotations

from neuropager.utils.types import MemoryKey, MemoryTier, create_page


def test_create_page_sets_defaults() -> None:
    """create_page stamps creation/access time and defaults tier/metadata."""
    page = create_page(MemoryKey("k1"), {"note": "hello"})

    assert page.key == MemoryKey("k1")
    assert page.content == {"note": "hello"}
    assert page.tier == MemoryTier.WORKING
    assert page.access_count == 0
    assert page.metadata == {}
    assert page.created_at == page.last_accessed_at


def test_create_page_accepts_explicit_tier_and_metadata() -> None:
    """create_page honors explicit tier and metadata overrides."""
    page = create_page(
        MemoryKey("k2"),
        "content",
        tier=MemoryTier.DISK,
        metadata={"source": "test"},
    )

    assert page.tier == MemoryTier.DISK
    assert page.metadata == {"source": "test"}


def test_create_page_metadata_is_independent_per_call() -> None:
    """Each create_page call gets its own metadata dict, not a shared default."""
    page_a = create_page(MemoryKey("a"), "x")
    page_b = create_page(MemoryKey("b"), "y")

    page_a.metadata["mutated"] = True

    assert "mutated" not in page_b.metadata
