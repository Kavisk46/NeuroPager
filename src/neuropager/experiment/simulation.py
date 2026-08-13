"""Run a workload through each candidate policy and compare outcomes.

:func:`run_baseline_comparison` drives FIFO, LRU, LFU, Random, Belady's
MIN, and (optionally) the learned-utility policy over the *identical*
workload, through a fresh :class:`~neuropager.core.memory_manager.MemoryManager`
each time, and reports page faults, hit ratio, the gap to Belady's
provably-optimal fault count, and per-decision latency.

Belady's MIN is used *only* to compute the reference fault count for the
gap metric — its future-reference-string is never passed to, or otherwise
made available to, any other policy in this module.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from neuropager.core.memory_manager import MemoryManager
from neuropager.core.page_fault import PageFaultHandler
from neuropager.core.page_table import PageTable
from neuropager.core.working_memory import WorkingMemory
from neuropager.memory.disk_store import DiskPageStore
from neuropager.policies.base import PageReplacementPolicy
from neuropager.policies.belady_min import BeladyMinPolicy
from neuropager.policies.fifo import FIFOPolicy
from neuropager.policies.lfu import LFUPolicy
from neuropager.policies.lru import LRUPolicy
from neuropager.policies.random_policy import RandomPolicy
from neuropager.trace.events import TraceEventType
from neuropager.trace.logger import TraceLogger
from neuropager.utils.types import MemoryKey

BELADY_MIN = "BeladyMinPolicy"


class _TimingPolicy(PageReplacementPolicy):
    """Wraps a policy to time every select_victim call, without changing its behavior."""

    def __init__(self, wrapped: PageReplacementPolicy) -> None:
        self.wrapped = wrapped
        self.latencies_seconds: list[float] = []

    def select_victim(self, page_table: PageTable) -> MemoryKey:
        start = time.perf_counter()
        victim = self.wrapped.select_victim(page_table)
        self.latencies_seconds.append(time.perf_counter() - start)
        return victim

    def on_access(self, key: MemoryKey) -> None:
        self.wrapped.on_access(key)

    def on_insert(self, key: MemoryKey) -> None:
        self.wrapped.on_insert(key)

    def on_evict(self, key: MemoryKey) -> None:
        self.wrapped.on_evict(key)


@dataclass(frozen=True)
class SimulationResult:
    """Outcome of running one policy over one workload.

    Attributes:
        policy_name: Class name of the policy that was run.
        working_memory_capacity: Working memory capacity used.
        total_references: Total number of ``get``/``put`` calls replayed.
        hits: Number of accesses resolved without a page fault.
        page_faults: Number of accesses that required a page fault.
        hit_ratio: ``hits / total_references``.
        belady_faults: Belady's MIN fault count on the same workload,
            provided for the gap calculation below.
        belady_gap_absolute: ``page_faults - belady_faults``.
        belady_gap_relative: ``belady_gap_absolute / belady_faults``, or
            ``None`` if ``belady_faults == 0`` (division undefined).
        n_decisions: Number of capacity-triggered eviction decisions made.
        mean_decision_latency_seconds: Mean wall-clock time per
            ``select_victim`` call.
        total_decision_latency_seconds: Summed wall-clock time across all
            ``select_victim`` calls.
    """

    policy_name: str
    working_memory_capacity: int
    total_references: int
    hits: int
    page_faults: int
    hit_ratio: float
    belady_faults: int
    belady_gap_absolute: int
    belady_gap_relative: float | None
    n_decisions: int
    mean_decision_latency_seconds: float
    total_decision_latency_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return this result as a plain, JSON/CSV-serializable dict.

        Returns:
            A flat dict of all fields.
        """
        return {
            "policy_name": self.policy_name,
            "working_memory_capacity": self.working_memory_capacity,
            "total_references": self.total_references,
            "hits": self.hits,
            "page_faults": self.page_faults,
            "hit_ratio": self.hit_ratio,
            "belady_faults": self.belady_faults,
            "belady_gap_absolute": self.belady_gap_absolute,
            "belady_gap_relative": self.belady_gap_relative,
            "n_decisions": self.n_decisions,
            "mean_decision_latency_seconds": self.mean_decision_latency_seconds,
            "total_decision_latency_seconds": self.total_decision_latency_seconds,
        }


def run_policy_simulation(
    workload: Sequence[MemoryKey],
    capacity: int,
    policy: PageReplacementPolicy,
    disk_root: Path,
    belady_faults: int,
    policy_name: str | None = None,
) -> SimulationResult:
    """Replay ``workload`` through a fresh MemoryManager under one policy.

    The first occurrence of each key is a ``put`` (fresh write); every
    later occurrence is a ``get`` (a genuine re-reference).

    Args:
        workload: The sequence of keys to replay, in order.
        capacity: Working memory capacity to configure.
        policy: The (unwrapped) policy to evaluate. Internally wrapped in
            a timing shim; the caller's instance is otherwise untouched.
        disk_root: Directory for this run's isolated disk store and trace.
        belady_faults: Belady's MIN fault count on this same workload,
            used only to compute the gap metrics.
        policy_name: Label recorded on the result and used for the trace's
            episode_id. Defaults to ``type(policy).__name__``; pass an
            explicit label when comparing several policies that share a
            class (e.g. two :class:`~neuropager.experiment.policy.LearnedUtilityPolicy`
            instances backed by different models).

    Returns:
        The resulting :class:`SimulationResult`.
    """
    label = policy_name if policy_name is not None else type(policy).__name__
    timing_policy = _TimingPolicy(policy)
    trace_logger = TraceLogger(f"sim-{label}", path=disk_root / "trace.jsonl")
    working_memory = WorkingMemory(capacity=capacity)
    page_table = PageTable()
    fault_handler = PageFaultHandler(
        working_memory=working_memory,
        page_table=page_table,
        disk_store=DiskPageStore(disk_root / "disk"),
        policy=timing_policy,
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
    trace_logger.close()

    access_events = [e for e in trace_logger.events if e.event_type == TraceEventType.ACCESS]
    fault_events = [e for e in trace_logger.events if e.event_type == TraceEventType.PAGE_FAULT]
    eviction_events = [e for e in trace_logger.events if e.event_type == TraceEventType.EVICTION]

    page_faults = len(fault_events)
    total_references = len(access_events)
    hits = total_references - page_faults
    hit_ratio = hits / total_references if total_references > 0 else 0.0

    gap_absolute = page_faults - belady_faults
    gap_relative = gap_absolute / belady_faults if belady_faults > 0 else None

    latencies = timing_policy.latencies_seconds
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return SimulationResult(
        policy_name=label,
        working_memory_capacity=capacity,
        total_references=total_references,
        hits=hits,
        page_faults=page_faults,
        hit_ratio=hit_ratio,
        belady_faults=belady_faults,
        belady_gap_absolute=gap_absolute,
        belady_gap_relative=gap_relative,
        n_decisions=len(eviction_events),
        mean_decision_latency_seconds=mean_latency,
        total_decision_latency_seconds=sum(latencies),
    )


def run_baseline_comparison(
    workload: Sequence[MemoryKey],
    capacity: int,
    disk_root: Path,
    random_seed: int,
    learned_policy: PageReplacementPolicy | None = None,
    learned_policy_name: str | None = None,
) -> list[SimulationResult]:
    """Run every classical baseline plus an optional learned policy.

    Runs FIFO, LRU, LFU, Random, Belady's MIN, and (optionally) a learned
    policy over the identical workload, and returns one result per policy.

    Args:
        workload: The sequence of keys to replay, identically, for every
            policy.
        capacity: Working memory capacity to configure.
        disk_root: Root directory; each policy gets its own subdirectory
            for an isolated disk store and trace.
        random_seed: Seed for :class:`~neuropager.policies.random_policy.RandomPolicy`.
        learned_policy: An optional, already-constructed
            :class:`~neuropager.experiment.policy.LearnedUtilityPolicy` (or
            any other policy) to include in the comparison.
        learned_policy_name: Label for ``learned_policy`` in the results
            (e.g. ``"LearnedUtilityPolicy[hist_gradient_boosting]"``),
            since multiple learned-policy variants share one class name.
            Defaults to the class name if not given.

    Returns:
        One :class:`SimulationResult` per policy, in the order: Belady's
        MIN, FIFO, LRU, LFU, Random, then the learned policy if given.
    """
    # Belady's own gap against itself is zero by construction; run it once
    # with a placeholder reference, then patch the gap fields rather than
    # replaying the whole workload a second time just to fix them up.
    belady_result = run_policy_simulation(
        workload,
        capacity,
        BeladyMinPolicy(future=workload),
        disk_root / "belady_min",
        belady_faults=0,
    )
    belady_faults = belady_result.page_faults
    belady_result = replace(
        belady_result,
        belady_faults=belady_faults,
        belady_gap_absolute=0,
        belady_gap_relative=(0.0 if belady_faults > 0 else None),
    )

    results = [
        belady_result,
        run_policy_simulation(workload, capacity, FIFOPolicy(), disk_root / "fifo", belady_faults),
        run_policy_simulation(workload, capacity, LRUPolicy(), disk_root / "lru", belady_faults),
        run_policy_simulation(workload, capacity, LFUPolicy(), disk_root / "lfu", belady_faults),
        run_policy_simulation(
            workload,
            capacity,
            RandomPolicy(seed=random_seed),
            disk_root / "random",
            belady_faults,
        ),
    ]

    if learned_policy is not None:
        results.append(
            run_policy_simulation(
                workload,
                capacity,
                learned_policy,
                disk_root / "learned",
                belady_faults,
                policy_name=learned_policy_name,
            )
        )

    return results
