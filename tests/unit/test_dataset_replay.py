"""Unit tests for :mod:`neuropager.dataset.replay`."""

from __future__ import annotations

import pytest

from neuropager.dataset.replay import TraceReplay, group_by_episode
from neuropager.trace.events import AccessOperation, AccessOutcome, TraceEvent, TraceEventType


def _access(episode_id: str, tick: int, page_id: str, resident: tuple[str, ...] = ()) -> TraceEvent:
    return TraceEvent(
        episode_id=episode_id,
        tick=tick,
        event_type=TraceEventType.ACCESS,
        page_id=page_id,
        working_memory_capacity=2,
        resident_pages=resident,
        policy="LRUPolicy",
        operation=AccessOperation.GET,
        outcome=AccessOutcome.HIT,
        page_access_count=1,
        page_fault_count=0,
    )


def _fault(episode_id: str, tick: int, page_id: str, resident: tuple[str, ...] = ()) -> TraceEvent:
    return TraceEvent(
        episode_id=episode_id,
        tick=tick,
        event_type=TraceEventType.PAGE_FAULT,
        page_id=page_id,
        working_memory_capacity=2,
        resident_pages=resident,
        policy="LRUPolicy",
        faulted_from_tier="disk",
        page_fault_count=1,
    )


def _eviction(
    episode_id: str,
    tick: int,
    victim: str,
    resident: tuple[str, ...],
    reason: str = "capacity",
    capacity: int = 2,
) -> TraceEvent:
    return TraceEvent(
        episode_id=episode_id,
        tick=tick,
        event_type=TraceEventType.EVICTION,
        page_id=victim,
        working_memory_capacity=capacity,
        resident_pages=resident,
        policy="LRUPolicy",
        eviction_reason=reason,
        page_access_count=1,
        page_fault_count=0,
    )


def test_replay_rejects_empty_event_list() -> None:
    """TraceReplay refuses to replay an empty sequence."""
    with pytest.raises(ValueError):
        TraceReplay([])


def test_replay_rejects_multiple_episodes() -> None:
    """TraceReplay requires all events to share one episode_id."""
    events = [_access("ep-1", 1, "a"), _access("ep-2", 1, "b")]

    with pytest.raises(ValueError):
        TraceReplay(events)


def test_replay_rejects_non_monotonic_ticks() -> None:
    """A decreasing tick within one episode indicates a corrupted trace."""
    events = [_access("ep-1", 5, "a"), _access("ep-1", 3, "b")]

    with pytest.raises(ValueError):
        TraceReplay(events)


def test_eviction_missing_reason_raises() -> None:
    """An EVICTION event without eviction_reason is treated as malformed."""
    bad_event = TraceEvent(
        episode_id="ep-1",
        tick=1,
        event_type=TraceEventType.EVICTION,
        page_id="a",
        working_memory_capacity=2,
        resident_pages=("a",),
        policy="LRUPolicy",
        eviction_reason=None,
    )

    with pytest.raises(ValueError):
        TraceReplay([bad_event])


def test_access_ticks_up_to_filters_correctly() -> None:
    """access_ticks_up_to only returns ticks at or before the given bound."""
    events = [_access("ep-1", 1, "a"), _access("ep-1", 5, "a"), _access("ep-1", 9, "a")]
    replay = TraceReplay(events)

    assert replay.access_ticks_up_to("a", 5) == [1, 5]
    assert replay.access_ticks_up_to("a", 4) == [1]
    assert replay.access_ticks_up_to("a", 100) == [1, 5, 9]
    assert replay.access_ticks_up_to("a", 0) == []


def test_access_ticks_up_to_unknown_page_is_empty() -> None:
    """A page never seen in the trace has no history, not an error."""
    replay = TraceReplay([_access("ep-1", 1, "a")])

    assert replay.access_ticks_up_to("never-seen", 100) == []


def test_fault_ticks_up_to_filters_correctly() -> None:
    """fault_ticks_up_to only returns PAGE_FAULT ticks at or before the bound."""
    events = [_access("ep-1", 1, "a"), _fault("ep-1", 5, "a"), _fault("ep-1", 9, "a")]
    replay = TraceReplay(events)

    assert replay.fault_ticks_up_to("a", 5) == [5]
    assert replay.fault_ticks_up_to("a", 8) == [5]
    assert replay.fault_ticks_up_to("a", 9) == [5, 9]


def test_full_access_ticks_includes_the_future() -> None:
    """full_access_ticks returns every access tick, unlike access_ticks_up_to."""
    events = [_access("ep-1", 1, "a"), _access("ep-1", 50, "a")]
    replay = TraceReplay(events)

    assert replay.access_ticks_up_to("a", 10) == [1]
    assert replay.full_access_ticks("a") == [1, 50]


def test_eviction_decisions_filters_by_reason() -> None:
    """eviction_decisions() only returns decisions matching the requested reason."""
    events = [
        _access("ep-1", 1, "a", resident=()),
        _eviction("ep-1", 2, "a", resident=("a",), reason="capacity"),
        _eviction("ep-1", 3, "b", resident=("b",), reason="explicit"),
    ]
    replay = TraceReplay(events)

    capacity_decisions = replay.eviction_decisions(reason="capacity")
    explicit_decisions = replay.eviction_decisions(reason="explicit")

    assert len(capacity_decisions) == 1
    assert capacity_decisions[0].decision_tick == 2
    assert capacity_decisions[0].chosen_victim == "a"
    assert capacity_decisions[0].candidate_pages == ("a",)
    assert len(explicit_decisions) == 1
    assert explicit_decisions[0].decision_tick == 3


def test_group_by_episode_partitions_preserving_order() -> None:
    """group_by_episode splits a mixed event list without reordering within episodes."""
    events = [
        _access("ep-1", 1, "a"),
        _access("ep-2", 1, "x"),
        _access("ep-1", 2, "b"),
        _access("ep-2", 2, "y"),
    ]

    grouped = group_by_episode(events)

    assert set(grouped) == {"ep-1", "ep-2"}
    assert [e.page_id for e in grouped["ep-1"]] == ["a", "b"]
    assert [e.page_id for e in grouped["ep-2"]] == ["x", "y"]
