"""Integration tests for the offline dataset-generation pipeline.

Covers the hand-crafted scenarios from the milestone spec that require a
full trace rather than a single pure-function call (multiple episodes,
cold-start, prior faults, irregular intervals), a direct proof that
features never see the future, and an end-to-end run against a trace
produced by the real MemoryManager/TraceLogger/policy stack.
"""

from __future__ import annotations

from pathlib import Path

from neuropager.core.memory_manager import MemoryManager
from neuropager.core.page_fault import PageFaultHandler
from neuropager.core.page_table import PageTable
from neuropager.core.working_memory import WorkingMemory
from neuropager.dataset.features import compute_features
from neuropager.dataset.generator import generate_dataset
from neuropager.dataset.replay import TraceReplay
from neuropager.memory.disk_store import DiskPageStore
from neuropager.policies.lru import LRUPolicy
from neuropager.trace.events import AccessOperation, AccessOutcome, TraceEvent, TraceEventType
from neuropager.trace.logger import TraceLogger
from neuropager.utils.types import MemoryKey


def _access(episode_id: str, tick: int, page_id: str, **overrides: object) -> TraceEvent:
    defaults: dict[str, object] = {
        "episode_id": episode_id,
        "tick": tick,
        "event_type": TraceEventType.ACCESS,
        "page_id": page_id,
        "working_memory_capacity": 2,
        "resident_pages": (),
        "policy": "LRUPolicy",
        "operation": AccessOperation.GET,
        "outcome": AccessOutcome.HIT,
        "page_access_count": 1,
        "page_fault_count": 0,
    }
    defaults.update(overrides)
    return TraceEvent(**defaults)  # type: ignore[arg-type]


def _fault(episode_id: str, tick: int, page_id: str, **overrides: object) -> TraceEvent:
    defaults: dict[str, object] = {
        "episode_id": episode_id,
        "tick": tick,
        "event_type": TraceEventType.PAGE_FAULT,
        "page_id": page_id,
        "working_memory_capacity": 2,
        "resident_pages": (),
        "policy": "LRUPolicy",
        "faulted_from_tier": "disk",
        "page_fault_count": 1,
    }
    defaults.update(overrides)
    return TraceEvent(**defaults)  # type: ignore[arg-type]


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


def test_multiple_episodes_are_fully_isolated() -> None:
    """Scenario 7: two episodes reusing identical page_ids and tick numbers.

    Episode A: page "shared" accessed at 1, evicted at tick 5 with a
    future access at tick 8 -> reused within H=10.
    Episode B: page "shared" accessed at 1, evicted at tick 5, with NO
    further access ever -> not reused, even though the tick numbers and
    page_id are identical to episode A.
    """
    events = [
        _access("ep-A", 1, "shared", resident_pages=()),
        _access("ep-A", 4, "other", resident_pages=("shared",)),
        _eviction("ep-A", 5, "shared", resident=("shared", "other"), capacity=2),
        _access("ep-A", 8, "shared", resident_pages=("other",)),
        _access("ep-B", 1, "shared", resident_pages=()),
        _access("ep-B", 4, "other", resident_pages=("shared",)),
        _eviction("ep-B", 5, "shared", resident=("shared", "other"), capacity=2),
    ]

    examples = generate_dataset(events, horizons=[10])

    by_episode_and_page = {(e.episode_id, e.page_id): e for e in examples if e.page_id == "shared"}
    assert by_episode_and_page[("ep-A", "shared")].label == 1
    assert by_episode_and_page[("ep-B", "shared")].label == 0


def test_cold_start_candidate_via_full_pipeline() -> None:
    """Scenario 8: a candidate with exactly one access before its decision point."""
    events = [
        _access("ep-1", 9, "cold", resident_pages=()),
        _access("ep-1", 10, "warm", resident_pages=("cold",)),
        _eviction("ep-1", 11, "cold", resident=("cold", "warm"), capacity=2),
    ]

    examples = generate_dataset(events, horizons=[5])
    cold = next(e for e in examples if e.page_id == "cold")

    assert cold.features.frequency == 1.0
    assert cold.features.recency == 2.0  # 11 - 9
    assert cold.features.page_age == 2.0
    assert cold.features.avg_inter_access_interval == -1.0  # sentinel: only 1 access


def test_candidate_with_previous_faults_via_full_pipeline() -> None:
    """Scenario 9: a candidate that faulted before the decision point."""
    events = [
        _access("ep-1", 1, "p", resident_pages=()),
        _fault("ep-1", 1, "never-used"),  # unrelated noise event, ignored for page "p"
        _access("ep-1", 5, "q", resident_pages=("p",)),
        _fault("ep-1", 20, "p"),
        _access("ep-1", 20, "p", resident_pages=("q",)),
        _eviction("ep-1", 25, "p", resident=("p", "q"), capacity=2),
    ]

    examples = generate_dataset(events, horizons=[5])
    p_example = next(e for e in examples if e.page_id == "p")

    assert p_example.features.fault_count == 1.0
    assert p_example.features.fault_ratio == 0.5  # 1 fault / 2 accesses
    assert p_example.features.elapsed_since_fault == 5.0  # 25 - 20


def test_irregular_access_intervals_via_full_pipeline() -> None:
    """Scenario 10: a candidate with irregular access gaps before the decision."""
    events = [
        _access("ep-1", 1, "p", resident_pages=()),
        _access("ep-1", 3, "p", resident_pages=()),
        _access("ep-1", 4, "p", resident_pages=()),
        _access("ep-1", 18, "p", resident_pages=()),
        _access("ep-1", 20, "other", resident_pages=("p",)),
        _eviction("ep-1", 25, "p", resident=("p", "other"), capacity=2),
    ]

    examples = generate_dataset(events, horizons=[5])
    p_example = next(e for e in examples if e.page_id == "p")

    # gaps = [3-1, 4-3, 18-4] = [2, 1, 14]
    assert p_example.features.max_historical_interval == 14.0
    assert p_example.features.recent_interval == 14.0
    assert p_example.features.avg_inter_access_interval == (2 + 1 + 14) / 3


def test_one_decision_point_yields_one_example_per_candidate_per_horizon() -> None:
    """Three candidates times two horizons at one decision point yields six examples."""
    events = [
        _access("ep-1", 1, "a", resident_pages=()),
        _access("ep-1", 2, "b", resident_pages=("a",)),
        _access("ep-1", 3, "c", resident_pages=("a", "b")),
        _access("ep-1", 4, "d", resident_pages=("a", "b", "c")),
        _eviction("ep-1", 5, "a", resident=("a", "b", "c"), capacity=3),
    ]

    examples = generate_dataset(events, horizons=[10, 20])

    assert len(examples) == 6  # 3 candidates * 2 horizons
    assert {e.page_id for e in examples} == {"a", "b", "c"}
    assert {e.horizon for e in examples} == {10, 20}


def test_explicit_evictions_are_excluded_from_examples() -> None:
    """Only capacity-triggered decisions generate examples in this milestone."""
    events = [
        _access("ep-1", 1, "a", resident_pages=()),
        _eviction("ep-1", 2, "a", resident=("a",), reason="explicit", capacity=2),
    ]

    examples = generate_dataset(events, horizons=[10])

    assert examples == []


def test_no_duplicate_examples_at_scale() -> None:
    """A larger synthetic trace with many decisions/candidates/horizons has no duplicates."""
    events: list[TraceEvent] = []
    keys = [f"k{i}" for i in range(5)]
    tick = 0
    for key in keys:
        tick += 1
        events.append(_access("ep-1", tick, key, resident_pages=tuple(keys[: keys.index(key)])))
    for round_ in range(5):
        tick += 1
        victim = keys[round_ % len(keys)]
        events.append(_eviction("ep-1", tick, victim, resident=tuple(keys), capacity=5))

    examples = generate_dataset(events, horizons=[5, 10, 25])

    seen = {(e.episode_id, e.decision_tick, e.page_id, e.horizon) for e in examples}
    assert len(seen) == len(examples)


def test_features_never_see_the_future() -> None:
    """Truncating the trace at the decision tick must not change computed features.

    Builds a trace where a candidate page has accesses both before AND
    after the decision tick, computes its features via the real replay
    path, then recomputes them from a version of the trace with every
    post-decision event physically removed. If features ever leaked
    future information, removing the future events would change the
    result; it must not.
    """
    events = [
        _access("ep-1", 1, "p", resident_pages=()),
        _access("ep-1", 3, "p", resident_pages=()),
        _access("ep-1", 5, "other", resident_pages=("p",)),
        _eviction("ep-1", 10, "p", resident=("p", "other"), capacity=2),
        # Future events for "p", after the decision tick:
        _access("ep-1", 15, "p", resident_pages=("other",)),
        _access("ep-1", 40, "p", resident_pages=("other",)),
    ]

    full_replay = TraceReplay(events)
    full_history = full_replay.access_ticks_up_to("p", 10)
    full_faults = full_replay.fault_ticks_up_to("p", 10)
    features_with_future_present = compute_features(full_history, full_faults, 10)

    truncated_events = [e for e in events if e.tick <= 10]
    truncated_replay = TraceReplay(truncated_events)
    truncated_history = truncated_replay.access_ticks_up_to("p", 10)
    truncated_faults = truncated_replay.fault_ticks_up_to("p", 10)
    features_without_future = compute_features(truncated_history, truncated_faults, 10)

    assert features_with_future_present == features_without_future

    # And the label generated from the full trace DOES reflect the future,
    # confirming the two code paths are genuinely different, not just both
    # accidentally ignoring the future.
    examples = generate_dataset(events, horizons=[10])
    p_example = next(e for e in examples if e.page_id == "p")
    assert p_example.label == 1  # access at tick 15 is within (10, 20]
    assert p_example.features == features_with_future_present


def _run_real_workload(tmp_path: Path) -> list[TraceEvent]:
    """Drive a real MemoryManager/TraceLogger/LRUPolicy stack and return its trace."""
    trace_logger = TraceLogger("real-ep-1")
    working_memory = WorkingMemory(capacity=2)
    page_table = PageTable()
    fault_handler = PageFaultHandler(
        working_memory=working_memory,
        page_table=page_table,
        disk_store=DiskPageStore(tmp_path / "disk"),
        policy=LRUPolicy(),
        trace_logger=trace_logger,
    )
    manager = MemoryManager(working_memory, page_table, fault_handler)

    workload = [
        MemoryKey("a"),
        MemoryKey("b"),
        MemoryKey("c"),
        MemoryKey("a"),
        MemoryKey("b"),
        MemoryKey("d"),
        MemoryKey("a"),
        MemoryKey("c"),
        MemoryKey("b"),
    ]
    seen: set[MemoryKey] = set()
    for key in workload:
        if key not in seen:
            manager.put(key, str(key))
            seen.add(key)
        else:
            manager.get(key)

    return trace_logger.events


def test_end_to_end_against_a_real_memory_manager_trace(tmp_path: Path) -> None:
    """The pipeline runs cleanly end-to-end on a trace from the real system.

    This is the strongest possible integration proof: no hand-crafted
    TraceEvent objects, just the real MemoryManager + PageFaultHandler +
    TraceLogger + LRUPolicy stack driving a workload, with the dataset
    pipeline consuming its actual output.
    """
    events = _run_real_workload(tmp_path)

    examples = generate_dataset(events, horizons=[5, 10])

    assert len(examples) > 0
    for example in examples:
        assert example.episode_id == "real-ep-1"
        assert example.eviction_reason == "capacity"
        assert example.label in (0, 1)
        assert example.features.frequency >= 1.0
