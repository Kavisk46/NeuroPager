#!/usr/bin/env python3
"""Sanity-check that the six benchmark workloads actually differ.

Generates ONE 2,000-tick episode per workload kind (not the large-scale,
2,000+ episode experiment), computes access-pattern statistics that don't
require the memory system (unique pages, entropy, mean stack distance,
mean burst-run length), and separately drives each workload through a
real MemoryManager (LRU policy, capacity=16) to count capacity-triggered
evictions -- so the report includes a memory-system-relevant statistic,
not just abstract sequence statistics.

This does NOT claim any workload is "realistic" -- only that the six
kinds are statistically distinguishable from one another, as designed.

Usage:
    python scripts/sanity_workloads.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from neuropager.core.memory_manager import MemoryManager
from neuropager.core.page_fault import PageFaultHandler
from neuropager.core.page_table import PageTable
from neuropager.core.working_memory import WorkingMemory
from neuropager.memory.disk_store import DiskPageStore
from neuropager.policies.lru import LRUPolicy
from neuropager.trace.events import TraceEventType
from neuropager.trace.logger import TraceLogger
from neuropager.utils.types import MemoryKey
from neuropager.workloads.config import (
    BurstyConfig,
    LongRangeReuseConfig,
    NonStationaryConfig,
    PhaseChangingConfig,
    TemporalLocalityConfig,
    UniformRandomConfig,
    WorkloadConfig,
)
from neuropager.workloads.generator import generate_workload
from neuropager.workloads.statistics import (
    access_entropy,
    burst_run_lengths,
    mean_stack_distance,
    unique_page_count,
)

LENGTH = 2000
CAPACITY = 16
SEED = 0

CONFIGS: dict[str, WorkloadConfig] = {
    "A_uniform_random": UniformRandomConfig(length=LENGTH, key_space=64, seed=SEED),
    "B_temporal_locality": TemporalLocalityConfig(
        length=LENGTH, key_space=64, seed=SEED, locality=0.3
    ),
    "C_bursty": BurstyConfig(length=LENGTH, key_space=64, seed=SEED, mean_burst_length=15.0),
    "D_long_range_reuse": LongRangeReuseConfig(length=LENGTH, key_space=200, seed=SEED),
    "E_phase_changing": PhaseChangingConfig(
        length=LENGTH, seed=SEED, phase_length=250, keys_per_phase=32
    ),
    "F_non_stationary": NonStationaryConfig(
        length=LENGTH, key_space=64, seed=SEED, drift_rate=0.01
    ),
}


def _run_through_memory_manager(workload: list[MemoryKey], disk_root: Path) -> tuple[int, int]:
    """Replay workload through a real MemoryManager and count faults/capacity evictions.

    Args:
        workload: The key sequence to replay.
        disk_root: Isolated directory for this run's disk store and trace.

    Returns:
        A tuple ``(page_faults, capacity_evictions)``.
    """
    trace_logger = TraceLogger("sanity", path=disk_root / "trace.jsonl")
    working_memory = WorkingMemory(capacity=CAPACITY)
    page_table = PageTable()
    fault_handler = PageFaultHandler(
        working_memory=working_memory,
        page_table=page_table,
        disk_store=DiskPageStore(disk_root / "disk"),
        policy=LRUPolicy(),
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

    faults = sum(1 for e in trace_logger.events if e.event_type == TraceEventType.PAGE_FAULT)
    evictions = sum(
        1
        for e in trace_logger.events
        if e.event_type == TraceEventType.EVICTION and e.eviction_reason == "capacity"
    )
    return faults, evictions


def main() -> None:
    """Generate each workload, compute statistics, and print a comparison table."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        rows = []
        for name, config in CONFIGS.items():
            workload = generate_workload(config)
            mean_distance = mean_stack_distance(workload)
            runs = burst_run_lengths(workload)
            mean_run = sum(runs) / len(runs) if runs else 0.0

            disk_root = tmp_path / name
            disk_root.mkdir(parents=True, exist_ok=True)
            faults, evictions = _run_through_memory_manager(workload, disk_root)

            rows.append(
                {
                    "workload": name,
                    "unique_pages": unique_page_count(workload),
                    "entropy_bits": access_entropy(workload),
                    "mean_stack_distance": mean_distance,
                    "mean_burst_run": mean_run,
                    "page_faults": faults,
                    "capacity_evictions": evictions,
                }
            )

        header = (
            f"{'workload':22s} {'unique':>7s} {'entropy':>8s} "
            f"{'mean_dist':>10s} {'mean_run':>9s} {'faults':>7s} {'evictions':>10s}"
        )
        print(header)
        print("-" * len(header))
        for row in rows:
            mean_dist_str = (
                f"{row['mean_stack_distance']:.2f}"
                if row["mean_stack_distance"] is not None
                else "n/a"
            )
            print(
                f"{row['workload']:22s} {row['unique_pages']:7d} "
                f"{row['entropy_bits']:8.3f} {mean_dist_str:>10s} "
                f"{row['mean_burst_run']:9.2f} {row['page_faults']:7d} "
                f"{row['capacity_evictions']:10d}"
            )


if __name__ == "__main__":
    main()
