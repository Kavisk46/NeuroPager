"""Unit tests for :mod:`neuropager.trace.logger`."""

from __future__ import annotations

import json
from pathlib import Path

from neuropager.trace.events import AccessOperation, AccessOutcome, TraceEvent, TraceEventType
from neuropager.trace.logger import TraceLogger
from neuropager.utils.types import MemoryKey

KEY_A = MemoryKey("a")
KEY_B = MemoryKey("b")


def test_current_tick_starts_at_zero() -> None:
    """A fresh logger's current_tick is 0 before any step begins."""
    logger = TraceLogger("ep-1")

    assert logger.current_tick == 0


def test_begin_step_is_monotonically_increasing() -> None:
    """Repeated begin_step calls return a strictly increasing sequence."""
    logger = TraceLogger("ep-1")

    ticks = [logger.begin_step() for _ in range(5)]

    assert ticks == [1, 2, 3, 4, 5]
    assert logger.current_tick == 5


def test_record_access_stamps_current_tick_and_episode() -> None:
    """record_access uses the logger's episode_id and current tick."""
    logger = TraceLogger("ep-42")
    logger.begin_step()
    logger.begin_step()

    event = logger.record_access(
        page_id=KEY_A,
        operation=AccessOperation.GET,
        outcome=AccessOutcome.HIT,
        resident_pages=[KEY_A],
        working_memory_capacity=4,
        policy="LRUPolicy",
        page_access_count=1,
        page_fault_count=0,
    )

    assert event.episode_id == "ep-42"
    assert event.tick == 2
    assert event.event_type == TraceEventType.ACCESS
    assert event.page_id == "a"
    assert event.resident_pages == ("a",)
    assert event in logger.events


def test_multiple_events_within_one_step_share_the_same_tick() -> None:
    """A fault and its eviction, recorded within one step, share one tick."""
    logger = TraceLogger("ep-1")
    logger.begin_step()

    fault_event = logger.record_page_fault(
        page_id=KEY_A,
        resident_pages=[KEY_B],
        working_memory_capacity=1,
        policy="LRUPolicy",
        faulted_from_tier="disk",
        page_fault_count=1,
    )
    eviction_event = logger.record_eviction(
        page_id=KEY_B,
        resident_pages=[KEY_B],
        working_memory_capacity=1,
        policy="LRUPolicy",
        reason="capacity",
        page_access_count=1,
        page_fault_count=0,
    )

    assert fault_event.tick == eviction_event.tick == 1


def test_record_eviction_sets_reason_and_no_operation_outcome() -> None:
    """An eviction event carries a reason and leaves operation/outcome unset."""
    logger = TraceLogger("ep-1")
    logger.begin_step()

    event = logger.record_eviction(
        page_id=KEY_A,
        resident_pages=[KEY_A],
        working_memory_capacity=1,
        policy="FIFOPolicy",
        reason="capacity",
        page_access_count=3,
        page_fault_count=1,
    )

    assert event.event_type == TraceEventType.EVICTION
    assert event.eviction_reason == "capacity"
    assert event.operation is None
    assert event.outcome is None


def test_writes_jsonl_file_when_path_given(tmp_path: Path) -> None:
    """Each recorded event is appended as one JSON line to the given path."""
    trace_path = tmp_path / "trace.jsonl"
    logger = TraceLogger("ep-1", path=trace_path)
    logger.begin_step()
    logger.record_access(
        page_id=KEY_A,
        operation=AccessOperation.PUT,
        outcome=AccessOutcome.NEW,
        resident_pages=[],
        working_memory_capacity=2,
        policy="LRUPolicy",
        page_access_count=1,
        page_fault_count=0,
    )
    logger.begin_step()
    logger.record_access(
        page_id=KEY_B,
        operation=AccessOperation.PUT,
        outcome=AccessOutcome.NEW,
        resident_pages=[KEY_A],
        working_memory_capacity=2,
        policy="LRUPolicy",
        page_access_count=1,
        page_fault_count=0,
    )
    logger.close()

    lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["page_id"] == "a"
    assert first["tick"] == 1
    second = json.loads(lines[1])
    assert second["page_id"] == "b"
    assert second["tick"] == 2


def test_creates_missing_parent_directories(tmp_path: Path) -> None:
    """Constructing a TraceLogger creates the path's parent directory if needed.

    Mirrors DiskPageStore's behavior of creating its own directory, so
    callers don't need to pre-create the output location by hand.
    """
    trace_path = tmp_path / "nested" / "run1" / "trace.jsonl"

    logger = TraceLogger("ep-1", path=trace_path)
    logger.begin_step()
    logger.record_access(
        page_id=KEY_A,
        operation=AccessOperation.GET,
        outcome=AccessOutcome.HIT,
        resident_pages=[KEY_A],
        working_memory_capacity=1,
        policy="LRUPolicy",
        page_access_count=1,
        page_fault_count=0,
    )
    logger.close()

    assert trace_path.exists()


def test_no_file_written_without_path() -> None:
    """Without a path, events are recorded in memory only (no file created)."""
    logger = TraceLogger("ep-1")
    logger.begin_step()

    logger.record_access(
        page_id=KEY_A,
        operation=AccessOperation.GET,
        outcome=AccessOutcome.HIT,
        resident_pages=[KEY_A],
        working_memory_capacity=1,
        policy="LRUPolicy",
        page_access_count=1,
        page_fault_count=0,
    )

    assert logger.path is None
    assert len(logger.events) == 1


def test_context_manager_closes_file(tmp_path: Path) -> None:
    """Using TraceLogger as a context manager flushes and closes the file on exit.

    On Windows an unclosed handle can make the file unreadable by another
    open() call, so successfully reading it back here is itself evidence
    that __exit__ actually closed the underlying file.
    """
    trace_path = tmp_path / "trace.jsonl"

    with TraceLogger("ep-1", path=trace_path) as logger:
        logger.begin_step()
        logger.record_access(
            page_id=KEY_A,
            operation=AccessOperation.GET,
            outcome=AccessOutcome.HIT,
            resident_pages=[KEY_A],
            working_memory_capacity=1,
            policy="LRUPolicy",
            page_access_count=1,
            page_fault_count=0,
        )

    assert trace_path.read_text(encoding="utf-8").strip() != ""
    logger.close()  # closing again after __exit__ must not raise


def test_jsonl_lines_round_trip_via_trace_event(tmp_path: Path) -> None:
    """Lines written to disk can be parsed back into equivalent TraceEvent objects."""
    trace_path = tmp_path / "trace.jsonl"
    logger = TraceLogger("ep-1", path=trace_path)
    logger.begin_step()
    original = logger.record_access(
        page_id=KEY_A,
        operation=AccessOperation.GET,
        outcome=AccessOutcome.FAULT,
        resident_pages=[KEY_B],
        working_memory_capacity=1,
        policy="LRUPolicy",
        page_access_count=1,
        page_fault_count=1,
    )
    logger.close()

    line = trace_path.read_text(encoding="utf-8").strip()
    restored = TraceEvent.from_json_dict(json.loads(line))

    assert restored == original
