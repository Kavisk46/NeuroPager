"""Equivalence proof for the resident-keys snapshot-reuse optimization.

PageFaultHandler.handle_fault()/install()/evict() used to each call
PageTable.resident_keys() (an O(tracked keys) scan) independently, even
though nothing mutates any key's tier between MemoryManager's own snapshot
and any of these three calls on the fault/eviction path -- see each
method's docstring in neuropager/core/page_fault.py for the exact
argument. The optimized code threads MemoryManager's already-computed
snapshot through instead of recomputing it up to three times per fault.

This test proves the optimization changed nothing observable: it replays
identical workloads under both the pre-optimization implementation
(frozen in tests/unit/_fixtures/pre_resident_snapshot_*.py, copied
verbatim from git history immediately before the optimization was made)
and the current implementation, and asserts every recorded TraceEvent --
including every `resident_pages` field the optimization touches -- is
byte-for-byte identical.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from neuropager.core.page_table import PageTable
from neuropager.core.working_memory import WorkingMemory
from neuropager.memory.in_memory_store import InMemoryPageStore
from neuropager.policies.belady_min import BeladyMinPolicy
from neuropager.policies.fifo import FIFOPolicy
from neuropager.policies.lfu import LFUPolicy
from neuropager.policies.lru import LRUPolicy
from neuropager.policies.random_policy import RandomPolicy
from neuropager.trace.logger import TraceLogger
from neuropager.utils.types import MemoryKey
from neuropager.workloads.config import BurstyConfig, TemporalLocalityConfig, UniformRandomConfig
from neuropager.workloads.generator import generate_workload

_FIXTURES = Path(__file__).resolve().parent / "_fixtures"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load the pre-optimization implementation frozen at fixture-capture time.
# old_memory_manager.py's own `from neuropager.core.page_fault import
# PageFaultHandler` line resolves against the CURRENT (optimized) package,
# but that import is used only as a type hint -- MemoryManager never
# constructs a PageFaultHandler itself, only calls methods on whatever
# instance it is given -- so it is safe to hand it an
# _old_page_fault.PageFaultHandler instance at runtime regardless.
_old_page_fault = _load_module(
    "_old_page_fault_fixture", _FIXTURES / "pre_resident_snapshot_page_fault.py"
)
_old_memory_manager = _load_module(
    "_old_memory_manager_fixture", _FIXTURES / "pre_resident_snapshot_memory_manager.py"
)

from neuropager.core.memory_manager import MemoryManager  # noqa: E402
from neuropager.core.page_fault import PageFaultHandler  # noqa: E402

CAPACITY = 8


def _replay(
    manager_cls: type[Any],
    fault_handler_cls: type[Any],
    workload: list[MemoryKey],
    policy_factory: Any,
    episode_id: str,
) -> list[dict[str, Any]]:
    """Replay ``workload`` through a fresh manager built from the given classes."""
    trace_logger = TraceLogger(episode_id)
    working_memory = WorkingMemory(capacity=CAPACITY)
    page_table = PageTable()
    fault_handler = fault_handler_cls(
        working_memory=working_memory,
        page_table=page_table,
        disk_store=InMemoryPageStore(),
        policy=policy_factory(),
        trace_logger=trace_logger,
    )
    manager = manager_cls(working_memory, page_table, fault_handler)

    seen: set[MemoryKey] = set()
    for key in workload:
        if key not in seen:
            manager.put(key, str(key))
            seen.add(key)
        else:
            manager.get(key)
    return [event.to_json_dict() for event in trace_logger.events]


WORKLOADS = {
    "uniform_random": generate_workload(UniformRandomConfig(length=400, key_space=20, seed=0)),
    "temporal_locality": generate_workload(
        TemporalLocalityConfig(length=400, key_space=20, locality=0.3, seed=1)
    ),
    "bursty": generate_workload(
        BurstyConfig(length=400, key_space=20, mean_burst_length=10.0, seed=2)
    ),
}

POLICY_FACTORIES: dict[str, Any] = {
    "FIFO": lambda: FIFOPolicy(),
    "LRU": lambda: LRUPolicy(),
    "LFU": lambda: LFUPolicy(),
    "Random_seed0": lambda: RandomPolicy(seed=0),
}


@pytest.mark.parametrize("workload_name", sorted(WORKLOADS))
@pytest.mark.parametrize("policy_name", sorted(POLICY_FACTORIES))
def test_resident_snapshot_reuse_matches_pre_optimization_trace(
    workload_name: str, policy_name: str
) -> None:
    """New (snapshot-reusing) and old (always-recompute) traces are byte-identical."""
    workload = WORKLOADS[workload_name]
    policy_factory = POLICY_FACTORIES[policy_name]

    old_events = _replay(
        _old_memory_manager.MemoryManager,
        _old_page_fault.PageFaultHandler,
        workload,
        policy_factory,
        f"old-{policy_name}-{workload_name}",
    )
    new_events = _replay(
        MemoryManager,
        PageFaultHandler,
        workload,
        policy_factory,
        f"old-{policy_name}-{workload_name}",  # same episode_id so events compare equal
    )

    assert new_events == old_events


def test_resident_snapshot_reuse_matches_pre_optimization_trace_belady() -> None:
    """Same equivalence proof for Belady MIN, which needs the full workload at construction."""
    workload = WORKLOADS["uniform_random"]

    old_events = _replay(
        _old_memory_manager.MemoryManager,
        _old_page_fault.PageFaultHandler,
        workload,
        lambda: BeladyMinPolicy(future=workload),
        "old-belady-uniform_random",
    )
    new_events = _replay(
        MemoryManager,
        PageFaultHandler,
        workload,
        lambda: BeladyMinPolicy(future=workload),
        "old-belady-uniform_random",
    )

    assert new_events == old_events


def test_resident_snapshot_reuse_matches_pre_optimization_explicit_evict() -> None:
    """MemoryManager.evict() (the explicit-eviction path, not fault-triggered) is unaffected too."""

    def _replay_with_explicit_evict(manager_cls: type[Any], fault_handler_cls: type[Any]) -> Any:
        trace_logger = TraceLogger("explicit-evict")
        working_memory = WorkingMemory(capacity=CAPACITY)
        page_table = PageTable()
        fault_handler = fault_handler_cls(
            working_memory=working_memory,
            page_table=page_table,
            disk_store=InMemoryPageStore(),
            policy=LRUPolicy(),
            trace_logger=trace_logger,
        )
        manager = manager_cls(working_memory, page_table, fault_handler)
        for i in range(5):
            manager.put(f"k{i}", str(i))
        manager.evict("k2")
        manager.get("k0")
        manager.put("k5", "5")
        return [event.to_json_dict() for event in trace_logger.events]

    old_events = _replay_with_explicit_evict(
        _old_memory_manager.MemoryManager, _old_page_fault.PageFaultHandler
    )
    new_events = _replay_with_explicit_evict(MemoryManager, PageFaultHandler)
    assert new_events == old_events
