"""Unit tests for :mod:`neuropager.dataset.features`.

Every expected value below is hand-computed in the test itself (see
comments), matching the exact mathematical definitions in
``docs/dataset.md``.
"""

from __future__ import annotations

from neuropager.dataset.features import (
    FEATURE_NAMES,
    MISSING_HISTORY_SENTINEL,
    Features,
    compute_features,
)


def test_feature_names_matches_dataclass_fields() -> None:
    """FEATURE_NAMES enumerates exactly the Features dataclass fields, in order."""
    assert FEATURE_NAMES == (
        "recency",
        "frequency",
        "avg_inter_access_interval",
        "inter_access_interval_variance",
        "fault_count",
        "fault_ratio",
        "page_age",
        "recent_burst",
        "long_burst",
        "recent_interval",
        "normalized_recency",
        "elapsed_since_fault",
        "max_historical_interval",
    )


def test_no_history_at_all_is_handled_safely() -> None:
    """A page with zero access ticks never crashes; every field is sentinel or 0."""
    features = compute_features(access_ticks=[], fault_ticks=[], decision_tick=100)

    assert features.frequency == 0.0
    assert features.fault_count == 0.0
    assert features.recency == MISSING_HISTORY_SENTINEL
    assert features.page_age == MISSING_HISTORY_SENTINEL
    assert features.avg_inter_access_interval == MISSING_HISTORY_SENTINEL
    assert features.inter_access_interval_variance == MISSING_HISTORY_SENTINEL
    assert features.recent_interval == MISSING_HISTORY_SENTINEL
    assert features.max_historical_interval == MISSING_HISTORY_SENTINEL
    assert features.normalized_recency == MISSING_HISTORY_SENTINEL
    assert features.elapsed_since_fault == MISSING_HISTORY_SENTINEL
    assert features.fault_ratio == MISSING_HISTORY_SENTINEL
    assert features.recent_burst == 0.0
    assert features.long_burst == 0.0


def test_single_access_cold_start_is_handled_safely() -> None:
    """A page accessed exactly once: recency/age well-defined, intervals are not.

    Accessed once at tick 95, decision at tick 100.
    recency = 100 - 95 = 5; page_age = 100 - 95 = 5 (only access is the first).
    normalized_recency = recency / page_age = 5 / 5 = 1.0.
    Interval-based features need >= 2 accesses -> sentinel.
    """
    features = compute_features(access_ticks=[95], fault_ticks=[], decision_tick=100)

    assert features.frequency == 1.0
    assert features.recency == 5.0
    assert features.page_age == 5.0
    assert features.normalized_recency == 1.0
    assert features.avg_inter_access_interval == MISSING_HISTORY_SENTINEL
    assert features.inter_access_interval_variance == MISSING_HISTORY_SENTINEL
    assert features.recent_interval == MISSING_HISTORY_SENTINEL
    assert features.max_historical_interval == MISSING_HISTORY_SENTINEL


def test_recency_is_ticks_since_last_access() -> None:
    """Recency = decision_tick - last access tick."""
    features = compute_features(access_ticks=[10, 30, 42], fault_ticks=[], decision_tick=50)

    assert features.recency == 8.0  # 50 - 42


def test_frequency_is_total_access_count() -> None:
    """Frequency is a plain count of accesses up to the decision tick."""
    features = compute_features(access_ticks=[1, 2, 3, 4, 5], fault_ticks=[], decision_tick=10)

    assert features.frequency == 5.0


def test_page_age_is_ticks_since_first_access() -> None:
    """page_age = decision_tick - first access tick."""
    features = compute_features(access_ticks=[10, 30, 42], fault_ticks=[], decision_tick=50)

    assert features.page_age == 40.0  # 50 - 10


def test_interval_statistics_with_irregular_gaps() -> None:
    """avg/variance/recent/max interval, hand-computed for irregular gaps.

    access_ticks = [1, 5, 6, 20]; decision_tick = 25.
    gaps = [5-1, 6-5, 20-6] = [4, 1, 14].
    avg = (4 + 1 + 14) / 3 = 19/3 = 6.3333...
    variance = mean((g - avg)^2 for g in gaps)
             = ((4-6.333)^2 + (1-6.333)^2 + (14-6.333)^2) / 3
             = (5.4444 + 28.4444 + 58.7778) / 3 = 92.6667 / 3 = 30.8889
    recent_interval = gaps[-1] = 14 (gap between the last two accesses: 20-6).
    max_historical_interval = max(gaps) = 14.
    """
    features = compute_features(access_ticks=[1, 5, 6, 20], fault_ticks=[], decision_tick=25)

    avg = (4 + 1 + 14) / 3
    variance = ((4 - avg) ** 2 + (1 - avg) ** 2 + (14 - avg) ** 2) / 3

    assert features.avg_inter_access_interval == avg
    assert features.inter_access_interval_variance == variance
    assert features.recent_interval == 14.0
    assert features.max_historical_interval == 14.0


def test_fault_statistics_are_correct() -> None:
    """Fault stats: fault_count is a count, fault_ratio = fault_count / frequency.

    elapsed_since_fault = decision_tick - last fault tick.
    """
    features = compute_features(
        access_ticks=[10, 20, 30, 40],
        fault_ticks=[10, 30],
        decision_tick=50,
    )

    assert features.fault_count == 2.0
    assert features.fault_ratio == 0.5  # 2 / 4
    assert features.elapsed_since_fault == 20.0  # 50 - 30


def test_fault_ratio_is_zero_faults_when_no_faults_recorded() -> None:
    """fault_ratio is a real 0.0 (not a sentinel) when accesses exist but no faults do."""
    features = compute_features(access_ticks=[1, 2, 3], fault_ticks=[], decision_tick=10)

    assert features.fault_count == 0.0
    assert features.fault_ratio == 0.0
    assert features.elapsed_since_fault == MISSING_HISTORY_SENTINEL


def test_recent_and_long_burst_windows() -> None:
    """recent_burst counts accesses within the last 10 ticks; long_burst within 50.

    decision_tick = 100. Accesses at 40, 60, 92, 95, 99.
    recent_burst window is (90, 100]: catches 92, 95, 99 -> 3.
    long_burst window is (50, 100]: catches 60, 92, 95, 99 -> 4.
    """
    features = compute_features(
        access_ticks=[40, 60, 92, 95, 99],
        fault_ticks=[],
        decision_tick=100,
    )

    assert features.recent_burst == 3.0
    assert features.long_burst == 4.0


def test_normalized_recency_is_deterministic() -> None:
    """normalized_recency = recency / page_age, and is identical across repeated calls."""
    args = dict(access_ticks=[5, 10, 12], fault_ticks=[], decision_tick=20)

    first = compute_features(**args)  # type: ignore[arg-type]
    second = compute_features(**args)  # type: ignore[arg-type]

    expected = (20 - 12) / (20 - 5)  # recency / page_age
    assert first.normalized_recency == expected
    assert first.normalized_recency == second.normalized_recency


def test_division_by_zero_guards_never_produce_nan_or_inf() -> None:
    """Every degenerate (empty-history) case yields the finite sentinel, not NaN/inf."""
    features = compute_features(access_ticks=[], fault_ticks=[], decision_tick=0)

    for name, value in features.to_dict().items():
        assert value == value, f"{name} is NaN"  # NaN != NaN
        assert value not in (float("inf"), float("-inf")), f"{name} is infinite"


def test_to_dict_round_trips_into_features() -> None:
    """Features.to_dict() output can reconstruct an identical Features instance."""
    features = compute_features(access_ticks=[1, 5, 9], fault_ticks=[5], decision_tick=15)

    reconstructed = Features(**features.to_dict())

    assert reconstructed == features
