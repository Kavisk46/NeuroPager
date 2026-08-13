"""Unit tests for :mod:`neuropager.experiment.policy`."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from neuropager.core.page_table import PageTable
from neuropager.dataset.features import FEATURE_NAMES, MISSING_HISTORY_SENTINEL
from neuropager.experiment.policy import LearnedUtilityPolicy
from neuropager.utils.types import MemoryKey, MemoryTier

KEY_A = MemoryKey("a")
KEY_B = MemoryKey("b")
KEY_C = MemoryKey("c")


class _FakeModel:
    """A stub model whose P(reused) is a simple, known function of `recency`.

    Higher recency (staler page) -> lower predicted reuse probability, so
    tests can predict exactly which candidate should be evicted without
    needing a real fitted classifier.
    """

    def __init__(self) -> None:
        self.last_x: np.ndarray | None = None

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self.last_x = x
        recency_column = x[:, FEATURE_NAMES.index("recency")]
        p1 = 1.0 / (1.0 + recency_column)
        return np.column_stack([1.0 - p1, p1])


def test_select_victim_raises_when_no_resident_pages() -> None:
    """select_victim raises ValueError with an empty page table."""
    policy = LearnedUtilityPolicy(model=_FakeModel(), horizon=25)
    table = PageTable()

    with pytest.raises(ValueError):
        policy.select_victim(table)


def test_select_victim_picks_the_stalest_candidate() -> None:
    """Given the fake model's recency-based scoring, the stalest page is evicted."""
    table = PageTable()
    table.update_tier(KEY_A, MemoryTier.WORKING)
    table.update_tier(KEY_B, MemoryTier.WORKING)
    table.update_tier(KEY_C, MemoryTier.WORKING)
    policy = LearnedUtilityPolicy(model=_FakeModel(), horizon=25)
    # Drive the policy's own tick/history tracking the same way
    # PageFaultHandler would, via on_insert for each installed page.
    policy.on_insert(KEY_A)
    policy.on_insert(KEY_B)
    policy.on_insert(KEY_C)
    # Touch B and C again so A becomes the stalest (largest recency).
    policy.on_access(KEY_B)
    policy.on_access(KEY_C)

    victim = policy.select_victim(table)

    assert victim == KEY_A


def test_fault_count_and_ratio_come_from_page_table_exactly() -> None:
    """fault_count/fault_ratio use PageTable's exact counts, not self-tracked history."""
    table = PageTable()
    table.update_tier(KEY_A, MemoryTier.WORKING)
    table.update_tier(KEY_B, MemoryTier.WORKING)
    table.record_access(KEY_A)  # access_count(A) = 1
    table.record_page_fault(KEY_A)  # fault_count(A) = 1
    model = _FakeModel()
    policy = LearnedUtilityPolicy(model=model, horizon=25)
    policy.on_insert(KEY_A)
    policy.on_insert(KEY_B)

    policy.select_victim(table)

    assert model.last_x is not None
    fault_count_col = FEATURE_NAMES.index("fault_count")
    fault_ratio_col = FEATURE_NAMES.index("fault_ratio")
    resident_order = table.resident_keys()
    a_row = resident_order.index(KEY_A)
    expected_ratio = 1.0 / table.get_entry(KEY_A).access_count
    assert model.last_x[a_row, fault_count_col] == 1.0
    assert model.last_x[a_row, fault_ratio_col] == pytest.approx(expected_ratio)


def test_elapsed_since_fault_is_always_the_missing_history_sentinel() -> None:
    """elapsed_since_fault is unavailable online and always reports the sentinel.

    This is a documented, deliberate gap (see docs/model.md): the policy
    hook interface cannot distinguish a fault-driven install from a
    fresh-put install, so it never learns *when* a fault happened, only
    (via PageTable) how many there were in total.
    """
    table = PageTable()
    table.update_tier(KEY_A, MemoryTier.WORKING)
    table.record_page_fault(KEY_A)
    table.record_page_fault(KEY_A)
    table.update_tier(KEY_B, MemoryTier.WORKING)
    model = _FakeModel()
    policy = LearnedUtilityPolicy(model=model, horizon=25)
    policy.on_insert(KEY_A)
    policy.on_insert(KEY_B)

    policy.select_victim(table)

    assert model.last_x is not None
    elapsed_col = FEATURE_NAMES.index("elapsed_since_fault")
    assert (model.last_x[:, elapsed_col] == MISSING_HISTORY_SENTINEL).all()


def test_hooks_are_callable_and_advance_tick() -> None:
    """on_access/on_insert each advance the internal tick by exactly one; on_evict does not."""
    policy = LearnedUtilityPolicy(model=_FakeModel(), horizon=25)

    assert policy._tick == 0  # noqa: SLF001 -- internal state is what's under test
    policy.on_insert(KEY_A)
    assert policy._tick == 1  # noqa: SLF001
    policy.on_access(KEY_A)
    assert policy._tick == 2  # noqa: SLF001
    policy.on_evict(KEY_A)
    assert policy._tick == 2  # noqa: SLF001


def test_select_victim_signature_matches_the_shared_policy_interface() -> None:
    """LearnedUtilityPolicy.select_victim has the same (self, page_table) signature.

    Locks in the same structural guarantee proven for every other policy:
    no parameter exists through which future information (or a Belady
    reference string) could reach this method.
    """
    signature = inspect.signature(LearnedUtilityPolicy.select_victim)

    assert list(signature.parameters) == ["self", "page_table"]


def test_constructor_has_no_future_or_belady_parameter() -> None:
    """LearnedUtilityPolicy's constructor only accepts a model and a horizon.

    Unlike BeladyMinPolicy (which legitimately takes a future reference
    string at construction time), the learned policy must never be
    constructible with anything resembling future/oracle information.
    """
    signature = inspect.signature(LearnedUtilityPolicy.__init__)

    assert list(signature.parameters) == ["self", "model", "horizon"]
