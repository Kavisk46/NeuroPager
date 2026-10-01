#!/usr/bin/env python3
"""Re-time HistGradientBoosting.fit on the REAL episode's features, in isolation.

The full profiler (scripts/profile_experiment.py) measured 1253s for this
exact fit, while a synthetic array of the identical shape took 14.7s in a
separate, isolated run -- an 85x gap. This script rebuilds the same
real feature matrix and re-times the fit alone, with nothing else running
concurrently, to check whether the original number reflected resource
contention (other background processes sharing this machine's 2 CPU cores)
rather than something pathological about the real data itself.

Usage:
    PYTHONPATH=src python -u scripts/profile_hgb_isolated.py
"""

from __future__ import annotations

import time

from neuropager.core.memory_manager import MemoryManager
from neuropager.core.page_fault import PageFaultHandler
from neuropager.core.page_table import PageTable
from neuropager.core.working_memory import WorkingMemory
from neuropager.dataset.generator import generate_dataset
from neuropager.experiment.models import HIST_GRADIENT_BOOSTING, make_model
from neuropager.experiment.preprocessing import build_feature_matrix
from neuropager.memory.in_memory_store import InMemoryPageStore
from neuropager.policies.lru import LRUPolicy
from neuropager.trace.logger import TraceLogger
from neuropager.workloads.config import UniformRandomConfig
from neuropager.workloads.generator import generate_workload

LENGTH = 2000
CAPACITY = 16


def main() -> None:
    """Rebuild the real episode's h=50 features and time HGB.fit alone."""
    config = UniformRandomConfig(length=LENGTH, key_space=64, seed=0)
    workload = generate_workload(config)

    trace_logger = TraceLogger("isolated-episode")
    working_memory = WorkingMemory(capacity=CAPACITY)
    page_table = PageTable()
    fault_handler = PageFaultHandler(
        working_memory=working_memory,
        page_table=page_table,
        disk_store=InMemoryPageStore(),
        policy=LRUPolicy(),
        trace_logger=trace_logger,
    )
    manager = MemoryManager(working_memory, page_table, fault_handler)
    seen = set()
    for key in workload:
        if key not in seen:
            manager.put(key, str(key))
            seen.add(key)
        else:
            manager.get(key)

    examples = generate_dataset(trace_logger.events, horizons=[50])
    print(f"examples: {len(examples)}")

    x, y, _ = build_feature_matrix(examples)
    print(f"matrix shape: {x.shape}, dtype: {x.dtype}")
    print(f"label balance: positive={int(y.sum())}, negative={int(len(y) - y.sum())}")
    print(f"unique values per column: {[len(set(x[:, i].tolist())) for i in range(x.shape[1])]}")

    model = make_model(HIST_GRADIENT_BOOSTING, seed=0)
    start = time.perf_counter()
    model.fit(x, y)
    elapsed = time.perf_counter() - start
    print(f"HistGradientBoosting.fit (real data, isolated): {elapsed:.4f}s")


if __name__ == "__main__":
    main()
