"""Unit tests for :mod:`neuropager.workloads.statistics`."""

from __future__ import annotations

import math

from neuropager.utils.types import MemoryKey
from neuropager.workloads.statistics import (
    access_entropy,
    burst_run_lengths,
    mean_stack_distance,
    stack_distances,
    unique_page_count,
)

A = MemoryKey("a")
B = MemoryKey("b")
C = MemoryKey("c")


def test_unique_page_count_empty() -> None:
    """unique_page_count is 0 for an empty workload."""
    assert unique_page_count([]) == 0


def test_unique_page_count_counts_distinct_keys() -> None:
    """unique_page_count counts each distinct key once, regardless of repeats."""
    assert unique_page_count([A, B, A, A, C, B]) == 3


def test_access_entropy_empty_is_zero() -> None:
    """access_entropy is 0.0 for an empty workload."""
    assert access_entropy([]) == 0.0


def test_access_entropy_single_key_is_zero() -> None:
    """A workload of one repeated key has zero entropy (no uncertainty)."""
    assert access_entropy([A, A, A, A]) == 0.0


def test_access_entropy_uniform_two_keys_is_one_bit() -> None:
    """Two equally-likely keys give exactly 1 bit of entropy."""
    workload = [A, B, A, B, A, B]
    assert math.isclose(access_entropy(workload), 1.0, abs_tol=1e-9)


def test_stack_distances_first_occurrence_is_none() -> None:
    """A key's first occurrence has no previous reference, so its distance is None."""
    distances = stack_distances([A, B, C])
    assert distances == [None, None, None]


def test_stack_distances_immediate_repeat_is_zero() -> None:
    """Repeating the same key with nothing in between gives stack distance 0."""
    distances = stack_distances([A, A])
    assert distances == [None, 0]


def test_stack_distances_counts_distinct_intervening_keys() -> None:
    """Distance counts distinct keys strictly between the two occurrences."""
    # A, B, C, B, A -> second A's distance = |{B, C, B}| = 2 distinct keys.
    distances = stack_distances([A, B, C, B, A])
    assert distances[-1] == 2


def test_mean_stack_distance_none_when_no_repeats() -> None:
    """mean_stack_distance is None when every key is a cold, first-time reference."""
    assert mean_stack_distance([A, B, C]) is None


def test_mean_stack_distance_averages_non_cold_distances() -> None:
    """mean_stack_distance averages only the non-None (repeat) distances."""
    # A, A -> distance 0; B, C, B -> distance 1. Mean of [0, 1] = 0.5.
    workload = [A, A, B, C, B]
    result = mean_stack_distance(workload)
    assert result == 0.5


def test_burst_run_lengths_empty() -> None:
    """burst_run_lengths is empty for an empty workload."""
    assert burst_run_lengths([]) == []


def test_burst_run_lengths_single_key() -> None:
    """A single key repeated N times is one run of length N."""
    assert burst_run_lengths([A, A, A]) == [3]


def test_burst_run_lengths_no_repeats() -> None:
    """A workload with no consecutive repeats is all runs of length 1."""
    assert burst_run_lengths([A, B, C]) == [1, 1, 1]


def test_burst_run_lengths_mixed() -> None:
    """Runs are correctly segmented at every key change."""
    # A A B C C C -> runs [2, 1, 3]
    assert burst_run_lengths([A, A, B, C, C, C]) == [2, 1, 3]
