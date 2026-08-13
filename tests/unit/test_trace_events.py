"""Unit tests for :mod:`neuropager.trace.events`."""

from __future__ import annotations

import dataclasses

import pytest

from neuropager.trace.events import (
    AccessOperation,
    AccessOutcome,
    TraceEvent,
    TraceEventType,
)


def _make_event(**overrides: object) -> TraceEvent:
    defaults: dict[str, object] = {
        "episode_id": "ep-1",
        "tick": 3,
        "event_type": TraceEventType.ACCESS,
        "page_id": "a",
        "working_memory_capacity": 2,
        "resident_pages": ("a", "b"),
        "policy": "LRUPolicy",
        "operation": AccessOperation.GET,
        "outcome": AccessOutcome.HIT,
        "page_access_count": 1,
        "page_fault_count": 0,
    }
    defaults.update(overrides)
    return TraceEvent(**defaults)  # type: ignore[arg-type]


def test_trace_event_is_frozen() -> None:
    """TraceEvent cannot be mutated after construction."""
    event = _make_event()

    with pytest.raises(dataclasses.FrozenInstanceError):
        event.tick = 999  # type: ignore[misc]


def test_to_json_dict_uses_plain_values() -> None:
    """to_json_dict emits only str/int/list/None, with enums unwrapped to .value."""
    event = _make_event()

    data = event.to_json_dict()

    assert data["event_type"] == "access"
    assert data["operation"] == "get"
    assert data["outcome"] == "hit"
    assert data["resident_pages"] == ["a", "b"]
    assert isinstance(data["resident_pages"], list)
    assert data["faulted_from_tier"] is None
    assert data["eviction_reason"] is None


def test_to_json_dict_handles_none_operation_and_outcome() -> None:
    """A non-ACCESS event's operation/outcome serialize as None, not an error."""
    event = _make_event(
        event_type=TraceEventType.EVICTION,
        operation=None,
        outcome=None,
        eviction_reason="capacity",
    )

    data = event.to_json_dict()

    assert data["operation"] is None
    assert data["outcome"] is None
    assert data["eviction_reason"] == "capacity"


def test_from_json_dict_round_trips_to_json_dict() -> None:
    """from_json_dict(event.to_json_dict()) reconstructs an equal event."""
    original = _make_event()

    restored = TraceEvent.from_json_dict(original.to_json_dict())

    assert restored == original


def test_from_json_dict_round_trips_eviction_event() -> None:
    """Round-trip also holds for events with operation/outcome left as None."""
    original = _make_event(
        event_type=TraceEventType.EVICTION,
        operation=None,
        outcome=None,
        page_id="b",
        eviction_reason="explicit",
    )

    restored = TraceEvent.from_json_dict(original.to_json_dict())

    assert restored == original
