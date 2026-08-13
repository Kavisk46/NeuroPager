"""Unit tests for the stateless classical policies: LRU, FIFO, LFU, Random."""

from __future__ import annotations

import pytest

from neuropager.core.page_table import PageTable
from neuropager.policies.fifo import FIFOPolicy
from neuropager.policies.lfu import LFUPolicy
from neuropager.policies.lru import LRUPolicy
from neuropager.policies.random_policy import RandomPolicy
from neuropager.utils.types import MemoryKey, MemoryTier

KEY_A = MemoryKey("a")
KEY_B = MemoryKey("b")
KEY_C = MemoryKey("c")


def test_lru_select_victim_raises_when_empty() -> None:
    """LRUPolicy.select_victim raises ValueError with no resident pages."""
    table = PageTable()

    with pytest.raises(ValueError):
        LRUPolicy().select_victim(table)


def test_fifo_select_victim_raises_when_empty() -> None:
    """FIFOPolicy.select_victim raises ValueError with no resident pages."""
    table = PageTable()

    with pytest.raises(ValueError):
        FIFOPolicy().select_victim(table)


def test_lru_evicts_least_recently_accessed() -> None:
    """LRU picks the resident key whose last access is furthest in the past."""
    table = PageTable()
    table.update_tier(KEY_A, MemoryTier.WORKING)
    table.update_tier(KEY_B, MemoryTier.WORKING)
    table.update_tier(KEY_C, MemoryTier.WORKING)

    # Touch A and C after insertion; B is now the least recently used.
    table.record_access(KEY_A)
    table.record_access(KEY_C)

    victim = LRUPolicy().select_victim(table)

    assert victim == KEY_B


def test_lru_victim_changes_after_access() -> None:
    """Accessing the current LRU victim promotes a different key to victim."""
    table = PageTable()
    table.update_tier(KEY_A, MemoryTier.WORKING)
    table.update_tier(KEY_B, MemoryTier.WORKING)
    policy = LRUPolicy()

    assert policy.select_victim(table) == KEY_A

    table.record_access(KEY_A)

    assert policy.select_victim(table) == KEY_B


def test_fifo_evicts_oldest_inserted_regardless_of_access() -> None:
    """FIFO picks the first-inserted key even if it was accessed most recently."""
    table = PageTable()
    table.update_tier(KEY_A, MemoryTier.WORKING)
    table.update_tier(KEY_B, MemoryTier.WORKING)
    table.update_tier(KEY_C, MemoryTier.WORKING)

    # Access A repeatedly; FIFO must still pick A since it was inserted first.
    table.record_access(KEY_A)
    table.record_access(KEY_A)

    victim = FIFOPolicy().select_victim(table)

    assert victim == KEY_A


def test_fifo_and_lru_disagree_after_reaccessing_oldest_key() -> None:
    """FIFO and LRU choose different victims once the oldest key is re-accessed.

    This directly demonstrates the two policies are behaviorally distinct,
    not just differently named wrappers around the same logic.
    """
    table = PageTable()
    table.update_tier(KEY_A, MemoryTier.WORKING)
    table.update_tier(KEY_B, MemoryTier.WORKING)

    table.record_access(KEY_A)  # A is now most-recently-used, but oldest-inserted.

    assert LRUPolicy().select_victim(table) == KEY_B
    assert FIFOPolicy().select_victim(table) == KEY_A


def test_policy_hooks_are_callable_no_ops() -> None:
    """on_access/on_insert/on_evict are safely callable (no-ops) on both policies."""
    for policy in (LRUPolicy(), FIFOPolicy()):
        policy.on_access(KEY_A)
        policy.on_insert(KEY_A)
        policy.on_evict(KEY_A)


def test_lfu_select_victim_raises_when_empty() -> None:
    """LFUPolicy.select_victim raises ValueError with no resident pages."""
    table = PageTable()

    with pytest.raises(ValueError):
        LFUPolicy().select_victim(table)


def test_lfu_evicts_least_frequently_accessed() -> None:
    """LFU picks the resident key with the lowest cumulative access count."""
    table = PageTable()
    table.update_tier(KEY_A, MemoryTier.WORKING)
    table.update_tier(KEY_B, MemoryTier.WORKING)
    table.update_tier(KEY_C, MemoryTier.WORKING)

    table.record_access(KEY_A)
    table.record_access(KEY_A)
    table.record_access(KEY_C)
    # B is never re-accessed after insertion; it has the lowest count.

    victim = LFUPolicy().select_victim(table)

    assert victim == KEY_B


def test_lfu_ties_broken_by_oldest_last_access() -> None:
    """When access counts tie, LFU evicts the one accessed longest ago."""
    table = PageTable()
    table.update_tier(KEY_A, MemoryTier.WORKING)
    table.update_tier(KEY_B, MemoryTier.WORKING)
    # Both A and B are freshly inserted with access_count 0, but A was
    # tracked (and thus last-accessed) at an earlier tick than B.

    victim = LFUPolicy().select_victim(table)

    assert victim == KEY_A


def test_lfu_and_lru_disagree_on_a_frequently_reused_old_key() -> None:
    """LFU and LRU can pick different victims for the same table state.

    A is accessed many times but not recently; B was just inserted once.
    LRU evicts A (oldest last access); LFU evicts B (lowest frequency).
    """
    table = PageTable()
    table.update_tier(KEY_A, MemoryTier.WORKING)
    table.record_access(KEY_A)
    table.record_access(KEY_A)
    table.record_access(KEY_A)
    table.update_tier(KEY_B, MemoryTier.WORKING)

    assert LRUPolicy().select_victim(table) == KEY_A
    assert LFUPolicy().select_victim(table) == KEY_B


def test_random_select_victim_raises_when_empty() -> None:
    """RandomPolicy.select_victim raises ValueError with no resident pages."""
    table = PageTable()

    with pytest.raises(ValueError):
        RandomPolicy(seed=0).select_victim(table)


def test_random_always_picks_a_resident_key() -> None:
    """RandomPolicy never picks a key outside the resident set."""
    table = PageTable()
    table.update_tier(KEY_A, MemoryTier.WORKING)
    table.update_tier(KEY_B, MemoryTier.WORKING)
    table.update_tier(KEY_C, MemoryTier.WORKING)
    policy = RandomPolicy(seed=1)

    for _ in range(20):
        assert policy.select_victim(table) in (KEY_A, KEY_B, KEY_C)


def test_random_is_deterministic_given_a_fixed_seed() -> None:
    """Two RandomPolicy instances with the same seed make identical choices."""
    table = PageTable()
    table.update_tier(KEY_A, MemoryTier.WORKING)
    table.update_tier(KEY_B, MemoryTier.WORKING)
    table.update_tier(KEY_C, MemoryTier.WORKING)

    policy_1 = RandomPolicy(seed=42)
    policy_2 = RandomPolicy(seed=42)

    choices_1 = [policy_1.select_victim(table) for _ in range(10)]
    choices_2 = [policy_2.select_victim(table) for _ in range(10)]

    assert choices_1 == choices_2


def test_random_hooks_are_callable_no_ops() -> None:
    """on_access/on_insert/on_evict are safely callable (no-ops) on RandomPolicy."""
    policy = RandomPolicy(seed=0)
    policy.on_access(KEY_A)
    policy.on_insert(KEY_A)
    policy.on_evict(KEY_A)
