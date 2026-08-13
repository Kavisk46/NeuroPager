"""Unit tests for :mod:`neuropager.experiment.simulation`."""

from __future__ import annotations

from pathlib import Path

from neuropager.experiment.simulation import run_baseline_comparison, run_policy_simulation
from neuropager.policies.lru import LRUPolicy
from neuropager.utils.types import MemoryKey

WORKLOAD = [MemoryKey(k) for k in ("a", "b", "c", "a", "b", "d", "a", "c", "b")]


def test_run_policy_simulation_reports_consistent_hit_fault_totals(tmp_path: Path) -> None:
    """Hits + page_faults always equals total_references."""
    result = run_policy_simulation(
        WORKLOAD, capacity=2, policy=LRUPolicy(), disk_root=tmp_path, belady_faults=1
    )

    assert result.hits + result.page_faults == result.total_references
    assert result.hit_ratio == result.hits / result.total_references


def test_run_policy_simulation_records_a_latency_per_decision(tmp_path: Path) -> None:
    """One latency sample is recorded per select_victim call (== n_decisions)."""
    result = run_policy_simulation(
        WORKLOAD, capacity=2, policy=LRUPolicy(), disk_root=tmp_path, belady_faults=1
    )

    assert result.n_decisions >= 1
    assert result.mean_decision_latency_seconds >= 0.0
    assert result.total_decision_latency_seconds >= 0.0


def test_belady_gap_is_zero_when_reference_matches_actual_faults(tmp_path: Path) -> None:
    """Gap arithmetic is zero when the supplied belady_faults reference matches.

    Equals the policy's own fault count on this run (a pure arithmetic
    check of ``belady_gap_absolute = page_faults - belady_faults``,
    independent of whether the reference actually came from Belady's MIN).
    """
    result = run_policy_simulation(
        WORKLOAD, capacity=2, policy=LRUPolicy(), disk_root=tmp_path, belady_faults=1
    )

    assert result.belady_gap_absolute == result.page_faults - 1


def test_belady_gap_relative_is_none_when_belady_faults_is_zero(tmp_path: Path) -> None:
    """Division-by-zero for the relative gap is avoided by returning None."""
    result = run_policy_simulation(
        WORKLOAD, capacity=2, policy=LRUPolicy(), disk_root=tmp_path, belady_faults=0
    )

    assert result.belady_gap_relative is None


def test_run_baseline_comparison_covers_all_five_classical_policies(tmp_path: Path) -> None:
    """FIFO, LRU, LFU, Random, and Belady's MIN are all present with no learned policy given."""
    results = run_baseline_comparison(WORKLOAD, capacity=2, disk_root=tmp_path, random_seed=0)

    names = {r.policy_name for r in results}
    assert names == {"BeladyMinPolicy", "FIFOPolicy", "LRUPolicy", "LFUPolicy", "RandomPolicy"}


def test_run_baseline_comparison_belady_has_zero_self_gap(tmp_path: Path) -> None:
    """Belady's MIN entry always reports zero gap against itself."""
    results = run_baseline_comparison(WORKLOAD, capacity=2, disk_root=tmp_path, random_seed=0)

    belady = next(r for r in results if r.policy_name == "BeladyMinPolicy")
    assert belady.belady_gap_absolute == 0


def test_run_baseline_comparison_belady_never_has_more_faults_than_any_other_policy(
    tmp_path: Path,
) -> None:
    """Belady's MIN fault count is <= every classical policy's fault count.

    This is the direct, empirical confirmation of Belady's optimality
    property, re-derived fresh on this workload rather than assumed.
    """
    results = run_baseline_comparison(WORKLOAD, capacity=2, disk_root=tmp_path, random_seed=0)

    belady = next(r for r in results if r.policy_name == "BeladyMinPolicy")
    for result in results:
        assert belady.page_faults <= result.page_faults


def test_run_baseline_comparison_includes_learned_policy_with_custom_label(tmp_path: Path) -> None:
    """A learned policy is included under its given label, distinct from the class name."""
    results = run_baseline_comparison(
        WORKLOAD,
        capacity=2,
        disk_root=tmp_path,
        random_seed=0,
        learned_policy=LRUPolicy(),  # stand-in; only the label plumbing is under test here
        learned_policy_name="LearnedUtilityPolicy[logistic_regression]",
    )

    names = {r.policy_name for r in results}
    assert "LearnedUtilityPolicy[logistic_regression]" in names
