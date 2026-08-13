"""End-to-end integration tests for the research trace infrastructure.

Exercises MemoryManager + PageFaultHandler + TraceLogger together and
proves the specific properties the trace is meant to guarantee: monotonic
ticks, correct access/fault/eviction bookkeeping, and — most importantly —
that no event ever reflects information from later in the episode, and
that replacement policies never receive trace data at all.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from neuropager.core.memory_manager import MemoryManager
from neuropager.core.page_fault import PageFaultHandler
from neuropager.core.page_table import PageTable
from neuropager.core.working_memory import WorkingMemory
from neuropager.memory.disk_store import DiskPageStore
from neuropager.policies.base import PageReplacementPolicy
from neuropager.policies.lru import LRUPolicy
from neuropager.trace.events import AccessOutcome, TraceEventType
from neuropager.trace.logger import TraceLogger
from neuropager.utils.types import MemoryKey

KEY_A = MemoryKey("a")
KEY_B = MemoryKey("b")
KEY_C = MemoryKey("c")


class _SpyPolicy(PageReplacementPolicy):
    """Wraps a real policy and records exactly what select_victim received."""

    def __init__(self, wrapped: PageReplacementPolicy) -> None:
        self.wrapped = wrapped
        self.select_victim_calls: list[tuple[object, ...]] = []

    def select_victim(self, page_table: PageTable) -> MemoryKey:
        self.select_victim_calls.append((page_table,))
        return self.wrapped.select_victim(page_table)

    def on_access(self, key: MemoryKey) -> None:
        self.wrapped.on_access(key)

    def on_insert(self, key: MemoryKey) -> None:
        self.wrapped.on_insert(key)

    def on_evict(self, key: MemoryKey) -> None:
        self.wrapped.on_evict(key)


def _build_manager(
    tmp_path: Path,
    capacity: int,
    policy: PageReplacementPolicy,
    trace_logger: TraceLogger | None,
) -> MemoryManager:
    working_memory = WorkingMemory(capacity=capacity)
    page_table = PageTable()
    fault_handler = PageFaultHandler(
        working_memory=working_memory,
        page_table=page_table,
        disk_store=DiskPageStore(tmp_path),
        policy=policy,
        trace_logger=trace_logger,
    )
    return MemoryManager(working_memory, page_table, fault_handler)


def test_ticks_are_monotonic_across_operations(tmp_path: Path) -> None:
    """Each top-level manager call advances the trace tick by exactly one."""
    trace_logger = TraceLogger("ep-tick")
    manager = _build_manager(tmp_path, capacity=2, policy=LRUPolicy(), trace_logger=trace_logger)

    manager.put(KEY_A, "1")
    manager.put(KEY_B, "2")
    manager.get(KEY_A)
    manager.evict(KEY_B)

    events = trace_logger.events
    access_ticks = [e.tick for e in events if e.event_type == TraceEventType.ACCESS]
    assert access_ticks == [1, 2, 3]
    eviction_ticks = [e.tick for e in events if e.event_type == TraceEventType.EVICTION]
    assert eviction_ticks == [4]


def test_access_statistics_update_correctly(tmp_path: Path) -> None:
    """page_access_count on recorded events matches PageTable ground truth."""
    trace_logger = TraceLogger("ep-stats")
    manager = _build_manager(tmp_path, capacity=2, policy=LRUPolicy(), trace_logger=trace_logger)

    manager.put(KEY_A, "1")
    manager.get(KEY_A)
    manager.get(KEY_A)

    access_events = [e for e in trace_logger.events if e.event_type == TraceEventType.ACCESS]
    assert [e.page_access_count for e in access_events] == [1, 2, 3]
    assert access_events[0].outcome == AccessOutcome.NEW
    assert access_events[1].outcome == AccessOutcome.HIT
    assert access_events[2].outcome == AccessOutcome.HIT


def test_page_faults_are_recorded_correctly(tmp_path: Path) -> None:
    """A get() on an evicted key emits a PAGE_FAULT event with correct fields."""
    trace_logger = TraceLogger("ep-fault")
    manager = _build_manager(tmp_path, capacity=1, policy=LRUPolicy(), trace_logger=trace_logger)

    manager.put(KEY_A, "1")
    manager.put(KEY_B, "2")  # evicts A to disk
    manager.get(KEY_A)  # faults A back in

    fault_events = [e for e in trace_logger.events if e.event_type == TraceEventType.PAGE_FAULT]
    assert len(fault_events) == 1
    fault = fault_events[0]
    assert fault.page_id == "a"
    assert fault.faulted_from_tier == "disk"
    assert fault.page_fault_count == 1
    assert fault.policy == "LRUPolicy"

    access_events = [e for e in trace_logger.events if e.event_type == TraceEventType.ACCESS]
    assert access_events[-1].outcome == AccessOutcome.FAULT
    assert access_events[-1].tick == fault.tick


def test_repeated_faults_increment_page_fault_count(tmp_path: Path) -> None:
    """Faulting the same key in multiple times increments its fault counter."""
    trace_logger = TraceLogger("ep-repeat-fault")
    manager = _build_manager(tmp_path, capacity=1, policy=LRUPolicy(), trace_logger=trace_logger)

    manager.put(KEY_A, "1")
    manager.put(KEY_B, "2")  # capacity 1: evicts A to disk
    manager.get(KEY_A)  # A's fault #1; evicts B to disk to make room
    manager.get(KEY_B)  # B's fault #1; evicts A to disk again
    manager.get(KEY_A)  # A's fault #2

    fault_events = [e for e in trace_logger.events if e.event_type == TraceEventType.PAGE_FAULT]
    a_faults = [e for e in fault_events if e.page_id == "a"]
    assert [e.page_fault_count for e in a_faults] == [1, 2]


def test_eviction_events_are_recorded_correctly(tmp_path: Path) -> None:
    """Capacity- and explicitly-triggered evictions both emit correct EVICTION events."""
    trace_logger = TraceLogger("ep-evict")
    manager = _build_manager(tmp_path, capacity=1, policy=LRUPolicy(), trace_logger=trace_logger)

    manager.put(KEY_A, "1")
    manager.put(KEY_B, "2")  # capacity eviction of A
    manager.evict(KEY_B)  # explicit eviction of B

    eviction_events = [e for e in trace_logger.events if e.event_type == TraceEventType.EVICTION]
    assert len(eviction_events) == 2
    assert eviction_events[0].page_id == "a"
    assert eviction_events[0].eviction_reason == "capacity"
    assert eviction_events[0].resident_pages == ("a",)
    assert eviction_events[1].page_id == "b"
    assert eviction_events[1].eviction_reason == "explicit"
    assert eviction_events[1].resident_pages == ("b",)


def test_resident_pages_never_contains_a_not_yet_created_key(tmp_path: Path) -> None:
    """No event's resident_pages snapshot ever names a key before it was first put().

    This is an end-to-end causality check: the "ground truth" first-seen
    tick for each key is computed independently from the driving script
    below (not from any internal trace/page-table state), then every
    recorded event is cross-checked against it. If any event's
    resident_pages listed a key ahead of its actual creation, this test
    would catch it directly.
    """
    trace_logger = TraceLogger("ep-causality")
    manager = _build_manager(tmp_path, capacity=2, policy=LRUPolicy(), trace_logger=trace_logger)

    first_seen_tick: dict[str, int] = {}
    script = [
        ("put", KEY_A, "1"),
        ("put", KEY_B, "2"),
        ("get", KEY_A, None),
        ("put", KEY_C, "3"),  # forces an eviction
        ("get", KEY_B, None),  # forces a fault + eviction
    ]
    for step, (op, key, value) in enumerate(script, start=1):
        if op == "put":
            if str(key) not in first_seen_tick:
                first_seen_tick[str(key)] = step
            manager.put(key, value)
        else:
            manager.get(key)

    for event in trace_logger.events:
        for page in event.resident_pages:
            assert page in first_seen_tick, f"{page} appears before it was ever created"
            assert first_seen_tick[page] <= event.tick, (
                f"event at tick {event.tick} lists {page!r}, "
                f"but it was not created until tick {first_seen_tick[page]}"
            )


def test_policy_interface_has_no_trace_or_future_parameter() -> None:
    """PageReplacementPolicy.select_victim only ever accepts a PageTable.

    This structurally guarantees no policy -- including a future learned
    Memory Utility Model -- can be handed the trace log or any other
    forward-looking state through this interface.
    """
    signature = inspect.signature(PageReplacementPolicy.select_victim)
    param_names = list(signature.parameters)

    assert param_names == ["self", "page_table"]


def test_select_victim_is_only_ever_called_with_the_page_table(tmp_path: Path) -> None:
    """At runtime, select_victim is invoked with nothing but the PageTable.

    A spy policy wraps LRUPolicy and records the exact arguments each
    select_victim call received; this proves the runtime call sites (not
    just the abstract signature) never pass trace data to a policy.
    """
    spy = _SpyPolicy(LRUPolicy())
    trace_logger = TraceLogger("ep-spy")
    manager = _build_manager(tmp_path, capacity=1, policy=spy, trace_logger=trace_logger)

    manager.put(KEY_A, "1")
    manager.put(KEY_B, "2")  # forces select_victim to run

    assert len(spy.select_victim_calls) == 1
    (call_args,) = spy.select_victim_calls
    assert len(call_args) == 1
    assert isinstance(call_args[0], PageTable)


def test_trace_output_is_byte_identical_across_repeated_runs(tmp_path: Path) -> None:
    """Replaying the same operation sequence produces byte-identical JSONL.

    Nothing in TraceLogger reads wall-clock time or randomness, so this is
    the direct, strong form of the "reproducible" requirement: two
    independent runs of the same script are not just equivalent, they are
    identical on disk.
    """

    def run_script(store_dir: Path, trace_path: Path) -> None:
        trace_logger = TraceLogger("ep-repro", path=trace_path)
        policy = LRUPolicy()
        manager = _build_manager(store_dir, capacity=2, policy=policy, trace_logger=trace_logger)
        manager.put(KEY_A, "1")
        manager.put(KEY_B, "2")
        manager.get(KEY_A)
        manager.put(KEY_C, "3")
        manager.get(KEY_B)
        trace_logger.close()

    trace_path_1 = tmp_path / "run1.jsonl"
    trace_path_2 = tmp_path / "run2.jsonl"
    run_script(tmp_path / "disk1", trace_path_1)
    run_script(tmp_path / "disk2", trace_path_2)

    assert trace_path_1.read_bytes() == trace_path_2.read_bytes()
    assert trace_path_1.read_text(encoding="utf-8").strip() != ""
