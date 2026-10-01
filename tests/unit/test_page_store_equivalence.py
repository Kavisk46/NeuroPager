"""Equivalence test: DiskPageStore vs. InMemoryPageStore on real episodes.

Proves the in-memory backend introduced to eliminate DiskPageStore's
filesystem I/O bottleneck (see scripts/microbench_disk_io.py) produces
IDENTICAL observable behavior -- reference sequence, policy decisions,
page faults, hit rate, and (for Belady) the offline-optimal result -- to
the original filesystem-backed store. Only the storage medium differs;
every eviction decision is made from :class:`~neuropager.core.page_table.PageTable`
state, which neither backend can see or influence.
"""

from __future__ import annotations

from pathlib import Path

from neuropager.core.memory_manager import MemoryManager
from neuropager.core.page_fault import PageFaultHandler
from neuropager.core.page_table import PageTable
from neuropager.core.working_memory import WorkingMemory
from neuropager.memory.disk_store import DiskPageStore
from neuropager.memory.in_memory_store import InMemoryPageStore
from neuropager.memory.page_store import PageStore
from neuropager.policies.base import PageReplacementPolicy
from neuropager.policies.belady_min import BeladyMinPolicy
from neuropager.policies.fifo import FIFOPolicy
from neuropager.policies.lru import LRUPolicy
from neuropager.policies.random_policy import RandomPolicy
from neuropager.trace.events import TraceEvent, TraceEventType
from neuropager.trace.logger import TraceLogger
from neuropager.utils.types import MemoryKey
from neuropager.workloads.config import TemporalLocalityConfig
from neuropager.workloads.generator import generate_workload

CAPACITY = 4


def replay(
    workload: list[MemoryKey], policy: PageReplacementPolicy, disk_store: PageStore
) -> list[TraceEvent]:
    """Replay ``workload`` through a real MemoryManager, returning its trace events."""
    trace_logger = TraceLogger("equivalence-episode")
    working_memory = WorkingMemory(capacity=CAPACITY)
    page_table = PageTable()
    fault_handler = PageFaultHandler(
        working_memory=working_memory,
        page_table=page_table,
        disk_store=disk_store,
        policy=policy,
        trace_logger=trace_logger,
    )
    manager = MemoryManager(working_memory, page_table, fault_handler)

    seen: set[MemoryKey] = set()
    for key in workload:
        if key not in seen:
            manager.put(key, str(key))
            seen.add(key)
        else:
            manager.get(key)
    return trace_logger.events


def summarize(events: list[TraceEvent]) -> tuple[int, int, int, tuple[str, ...]]:
    """Return (n_events, n_faults, n_evictions, event-type-sequence) for one trace."""
    faults = sum(1 for e in events if e.event_type == TraceEventType.PAGE_FAULT)
    evictions = sum(1 for e in events if e.event_type == TraceEventType.EVICTION)
    sequence = tuple(e.event_type.value for e in events)
    return len(events), faults, evictions, sequence


def _workload() -> list[MemoryKey]:
    config = TemporalLocalityConfig(length=40, key_space=10, seed=7, locality=0.3)
    return generate_workload(config)


def _assert_equivalent(disk_events: list[TraceEvent], memory_events: list[TraceEvent]) -> None:
    """Assert two traces are equivalent in every observable respect."""
    disk_summary = summarize(disk_events)
    memory_summary = summarize(memory_events)
    assert disk_summary == memory_summary

    assert len(disk_events) == len(memory_events)
    for disk_event, memory_event in zip(disk_events, memory_events, strict=True):
        assert disk_event.tick == memory_event.tick
        assert disk_event.event_type == memory_event.event_type
        assert disk_event.page_id == memory_event.page_id
        assert disk_event.resident_pages == memory_event.resident_pages
        assert disk_event.working_memory_capacity == memory_event.working_memory_capacity
        assert disk_event.policy == memory_event.policy
        assert disk_event.eviction_reason == memory_event.eviction_reason
        assert disk_event.page_access_count == memory_event.page_access_count
        assert disk_event.page_fault_count == memory_event.page_fault_count
        assert disk_event.faulted_from_tier == memory_event.faulted_from_tier


def test_lru_equivalent_across_backends(tmp_path: Path) -> None:
    """LRU produces identical traces under DiskPageStore and InMemoryPageStore."""
    workload = _workload()

    disk_events = replay(workload, LRUPolicy(), DiskPageStore(tmp_path / "disk"))
    memory_events = replay(workload, LRUPolicy(), InMemoryPageStore())

    _assert_equivalent(disk_events, memory_events)


def test_fifo_equivalent_across_backends(tmp_path: Path) -> None:
    """FIFO produces identical traces under DiskPageStore and InMemoryPageStore."""
    workload = _workload()

    disk_events = replay(workload, FIFOPolicy(), DiskPageStore(tmp_path / "disk"))
    memory_events = replay(workload, FIFOPolicy(), InMemoryPageStore())

    _assert_equivalent(disk_events, memory_events)


def test_belady_min_equivalent_across_backends(tmp_path: Path) -> None:
    """Belady MIN's offline-optimal decisions are identical across backends."""
    workload = _workload()

    disk_events = replay(
        workload, BeladyMinPolicy(future=workload), DiskPageStore(tmp_path / "disk")
    )
    memory_events = replay(workload, BeladyMinPolicy(future=workload), InMemoryPageStore())

    _assert_equivalent(disk_events, memory_events)


def test_random_seeded_equivalent_across_backends(tmp_path: Path) -> None:
    """A seeded Random policy is identical (same seed => same decisions) across backends."""
    workload = _workload()

    disk_events = replay(workload, RandomPolicy(seed=3), DiskPageStore(tmp_path / "disk"))
    memory_events = replay(workload, RandomPolicy(seed=3), InMemoryPageStore())

    _assert_equivalent(disk_events, memory_events)


def test_page_content_round_trips_identically(tmp_path: Path) -> None:
    """Reading back a resident page's content is identical across backends."""
    workload = _workload()
    disk_store = DiskPageStore(tmp_path / "disk")
    memory_store = InMemoryPageStore()

    replay(workload, LRUPolicy(), disk_store)
    replay(workload, LRUPolicy(), memory_store)

    sample_key = workload[0]
    if disk_store.contains(sample_key):
        assert memory_store.contains(sample_key)
        assert disk_store.read(sample_key).content == memory_store.read(sample_key).content
