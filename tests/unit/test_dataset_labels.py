"""Unit tests for :mod:`neuropager.dataset.labels`.

Covers hand-crafted scenarios 1-4 and 6 from the milestone spec directly
against the pure `compute_label` function (scenarios that require a full
trace -- multiple episodes, cold start, etc. -- are covered in
``test_dataset_generation.py`` instead).
"""

from __future__ import annotations

import pytest

from neuropager.dataset.labels import compute_label


def test_reused_within_horizon() -> None:
    """Scenario 1: an access strictly inside (t, t+H] labels as reused."""
    # decision_tick=10, horizon=10 -> window is (10, 20]. Access at 15 is inside.
    assert compute_label(full_access_ticks=[3, 15], decision_tick=10, horizon=10) == 1


def test_never_reused() -> None:
    """Scenario 2: no future access at all labels as not reused."""
    assert compute_label(full_access_ticks=[3, 5, 10], decision_tick=10, horizon=10) == 0


def test_reused_exactly_at_horizon_boundary() -> None:
    """Scenario 3: an access at exactly t+H counts as reused (closed upper bound)."""
    assert compute_label(full_access_ticks=[20], decision_tick=10, horizon=10) == 1


def test_reused_after_horizon_does_not_count() -> None:
    """Scenario 4: an access at t+H+1 is outside the window."""
    assert compute_label(full_access_ticks=[21], decision_tick=10, horizon=10) == 0


def test_access_at_decision_tick_itself_does_not_count() -> None:
    """Scenario 5: an access exactly at t is not future reuse (open lower bound)."""
    assert compute_label(full_access_ticks=[10], decision_tick=10, horizon=10) == 0


def test_multiple_accesses_within_horizon_still_labels_as_one() -> None:
    """Scenario 6: several qualifying accesses still just yield label 1."""
    assert compute_label(full_access_ticks=[12, 14, 18], decision_tick=10, horizon=10) == 1


def test_boundary_is_exclusive_below_and_inclusive_above_together() -> None:
    """A single call exercising both boundaries: t itself and t+H both present."""
    # t=10, H=10: tick 10 must not count, tick 20 must count.
    assert compute_label(full_access_ticks=[10, 20], decision_tick=10, horizon=10) == 1
    # Remove the qualifying tick 20 and only t remains -> not reused.
    assert compute_label(full_access_ticks=[10], decision_tick=10, horizon=10) == 0


def test_empty_access_history_labels_as_not_reused() -> None:
    """A page with no recorded accesses at all is trivially not reused."""
    assert compute_label(full_access_ticks=[], decision_tick=10, horizon=10) == 0


@pytest.mark.parametrize("horizon", [0, -1, -100])
def test_non_positive_horizon_raises(horizon: int) -> None:
    """Horizon must be a positive integer."""
    with pytest.raises(ValueError):
        compute_label(full_access_ticks=[1, 2, 3], decision_tick=10, horizon=horizon)


def test_supported_default_horizons_are_exactly_as_specified() -> None:
    """DEFAULT_HORIZONS matches the research plan's five named horizons."""
    from neuropager.dataset.labels import DEFAULT_HORIZONS

    assert DEFAULT_HORIZONS == (25, 50, 100, 200, 500)
