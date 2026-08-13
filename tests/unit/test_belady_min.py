"""Unit tests for :mod:`neuropager.policies.belady_min`."""

from __future__ import annotations

import pytest

from neuropager.core.page_table import PageTable
from neuropager.policies.belady_min import BeladyMinPolicy
from neuropager.utils.types import MemoryKey, MemoryTier

KEY_A = MemoryKey("a")
KEY_B = MemoryKey("b")
KEY_C = MemoryKey("c")


def test_select_victim_raises_when_empty() -> None:
    """select_victim raises ValueError with no resident pages."""
    policy = BeladyMinPolicy(future=[])
    table = PageTable()

    with pytest.raises(ValueError):
        policy.select_victim(table)


def test_never_reused_key_beats_a_soon_reused_key() -> None:
    """A key absent from the future entirely is evicted before any reused key."""
    table = PageTable()
    table.update_tier(KEY_A, MemoryTier.WORKING)
    table.update_tier(KEY_B, MemoryTier.WORKING)
    table.update_tier(KEY_C, MemoryTier.WORKING)
    # A is used immediately next, B shortly after; C never appears again.
    policy = BeladyMinPolicy(future=[KEY_A, KEY_B])

    victim = policy.select_victim(table)

    assert victim == KEY_C


def test_evicts_the_key_used_furthest_in_the_future() -> None:
    """Among keys that are all reused, the one reused latest is evicted."""
    table = PageTable()
    policy = BeladyMinPolicy(future=[KEY_A, KEY_B, KEY_C, KEY_A, KEY_B])
    # Replay the first three references so the cursor sits at position 3,
    # exactly as PageFaultHandler.install() would drive it.
    policy.on_insert(KEY_A)
    policy.on_insert(KEY_B)
    policy.on_insert(KEY_C)
    table.update_tier(KEY_A, MemoryTier.WORKING)
    table.update_tier(KEY_B, MemoryTier.WORKING)
    table.update_tier(KEY_C, MemoryTier.WORKING)

    # From position 3: A is used again at index 3, B at index 4, C never again.
    victim = policy.select_victim(table)

    assert victim == KEY_C


def test_ties_among_never_reused_keys_favor_the_earliest_tracked() -> None:
    """When multiple resident keys are never reused, tie-break is deterministic."""
    table = PageTable()
    table.update_tier(KEY_A, MemoryTier.WORKING)
    table.update_tier(KEY_B, MemoryTier.WORKING)
    policy = BeladyMinPolicy(future=[])

    victim = policy.select_victim(table)

    assert victim == KEY_A


def test_on_access_and_on_insert_advance_the_cursor() -> None:
    """Replaying the exact future sequence via on_access/on_insert succeeds."""
    policy = BeladyMinPolicy(future=[KEY_A, KEY_B, KEY_A])

    policy.on_insert(KEY_A)  # position 0 -> 1
    policy.on_access(KEY_B)  # position 1 -> 2
    policy.on_access(KEY_A)  # position 2 -> 3

    assert policy._position == 3  # noqa: SLF001 -- internal cursor is the thing under test


def test_mismatched_key_raises_value_error() -> None:
    """Replaying a key that diverges from the promised future raises immediately."""
    policy = BeladyMinPolicy(future=[KEY_A, KEY_B])

    with pytest.raises(ValueError):
        policy.on_insert(KEY_B)  # future[0] is A, not B


def test_overrunning_the_future_raises_value_error() -> None:
    """More references than the future reference string covers raises."""
    policy = BeladyMinPolicy(future=[KEY_A])
    policy.on_insert(KEY_A)

    with pytest.raises(ValueError):
        policy.on_access(KEY_A)


def test_on_evict_is_a_safe_no_op() -> None:
    """on_evict never raises and does not affect the cursor position."""
    policy = BeladyMinPolicy(future=[KEY_A])

    policy.on_evict(KEY_B)

    assert policy._position == 0  # noqa: SLF001 -- internal cursor is the thing under test
