#!/usr/bin/env python3
"""Profile one representative episode through the full experiment pipeline.

Measures, with real timers (not assumptions), where time goes across:
    workload generation, trace replay (per policy), dataset generation,
    feature-matrix construction, model training, model inference (via
    LearnedUtilityPolicy replay), and JSON serialization.

Uses the UNCHANGED neuropager package and the same episode configuration
as scripts/generalization_experiment_2.py (A_uniform_random, seed=0,
length=2000, capacity=16). Uses InMemoryPageStore (see
scripts/microbench_disk_io.py and tests/unit/test_page_store_equivalence.py
for why this is behaviorally identical to DiskPageStore but not
I/O-bound), so this profile measures every OTHER phase's real cost without
the filesystem bottleneck dominating and hiding everything else.

Usage:
    PYTHONPATH=src python scripts/profile_experiment.py
"""

from __future__ import annotations

import cProfile
import json
import pstats
import time
from collections.abc import Callable
from io import StringIO
from typing import Any

from neuropager.core.memory_manager import MemoryManager
from neuropager.core.page_fault import PageFaultHandler
from neuropager.core.page_table import PageTable
from neuropager.core.working_memory import WorkingMemory
from neuropager.dataset.generator import generate_dataset
from neuropager.experiment.models import HIST_GRADIENT_BOOSTING, LOGISTIC_REGRESSION, make_model
from neuropager.experiment.policy import LearnedUtilityPolicy
from neuropager.experiment.preprocessing import build_feature_matrix
from neuropager.memory.in_memory_store import InMemoryPageStore
from neuropager.policies.belady_min import BeladyMinPolicy
from neuropager.policies.fifo import FIFOPolicy
from neuropager.policies.lru import LRUPolicy
from neuropager.trace.logger import TraceLogger
from neuropager.workloads.config import UniformRandomConfig
from neuropager.workloads.generator import generate_workload

LENGTH = 2000
CAPACITY = 16
HORIZONS = [25, 50, 100, 200, 500]


def timed(label: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Run fn, print its wall-clock time, and return its result."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    print(f"{label:45s} {elapsed:10.4f}s")
    return result, elapsed


def replay(workload: list[Any], policy: Any) -> list[Any]:
    """Replay workload through a real MemoryManager (mirrors the experiment script)."""
    trace_logger = TraceLogger("profile-episode")
    working_memory = WorkingMemory(capacity=CAPACITY)
    page_table = PageTable()
    fault_handler = PageFaultHandler(
        working_memory=working_memory,
        page_table=page_table,
        disk_store=InMemoryPageStore(),
        policy=policy,
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
    return trace_logger.events


def main() -> None:
    """Run the full profiling pass and print a breakdown."""
    print("=" * 70)
    print("PHASE 1: workload generation")
    print("=" * 70)
    config = UniformRandomConfig(length=LENGTH, key_space=64, seed=0)
    workload, _ = timed("generate_workload", generate_workload, config)

    print()
    print("=" * 70)
    print("PHASE 2: trace replay (wall clock per policy)")
    print("=" * 70)
    events_lru, t_lru = timed("replay: LRU", replay, workload, LRUPolicy())
    events_fifo, t_fifo = timed("replay: FIFO", replay, workload, FIFOPolicy())
    events_belady, t_belady = timed(
        "replay: Belady_MIN", replay, workload, BeladyMinPolicy(future=workload)
    )
    n_faults = sum(1 for e in events_lru if e.event_type.name == "PAGE_FAULT")
    n_evictions = sum(1 for e in events_lru if e.event_type.name == "EVICTION")
    print(f"  LRU trace: {len(events_lru)} events, {n_faults} faults, {n_evictions} evictions")

    print()
    print("=" * 70)
    print("PHASE 2b: cProfile breakdown of ONE LRU replay (top 15 by cumulative time)")
    print("=" * 70)
    profiler = cProfile.Profile()
    profiler.enable()
    replay(workload, LRUPolicy())
    profiler.disable()
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(15)
    print(stream.getvalue())

    print("=" * 70)
    print("PHASE 3: dataset generation (one episode, all 5 horizons)")
    print("=" * 70)
    examples, t_dataset = timed(
        "generate_dataset (1 episode, 5 horizons)", generate_dataset, events_lru, HORIZONS
    )
    print(f"  examples generated: {len(examples)}")
    per_horizon = len(examples) / len(HORIZONS)
    print(f"  examples per horizon: {per_horizon:.0f}")

    print()
    print("=" * 70)
    print("PHASE 4: feature matrix + model train (horizon=50 only, this episode's examples)")
    print("=" * 70)
    h50_examples = [e for e in examples if e.horizon == 50]
    (x, y, _), t_matrix = timed("build_feature_matrix", build_feature_matrix, h50_examples)
    print(f"  matrix shape: {x.shape}")

    if len(set(y.tolist())) >= 2:
        lr_model = make_model(LOGISTIC_REGRESSION, seed=0)
        _, t_lr_fit = timed("LogisticRegression.fit", lr_model.fit, x, y)
        _, t_lr_predict = timed("LogisticRegression.predict_proba", lr_model.predict_proba, x)

        hgb_model = make_model(HIST_GRADIENT_BOOSTING, seed=0)
        _, t_hgb_fit = timed("HistGradientBoosting.fit", hgb_model.fit, x, y)
        _, t_hgb_predict = timed("HistGradientBoosting.predict_proba", hgb_model.predict_proba, x)

        print()
        print("=" * 70)
        print("PHASE 5: LearnedUtilityPolicy replay (model inference inside a live episode)")
        print("=" * 70)
        learned_policy = LearnedUtilityPolicy(model=lr_model, horizon=50)
        events_learned, t_learned = timed(
            "replay: LearnedUtilityPolicy", replay, workload, learned_policy
        )
    else:
        print("  (single-class labels at this horizon; skipping model train/inference phase)")
        t_lr_fit = t_lr_predict = t_hgb_fit = t_hgb_predict = t_learned = None

    print()
    print("=" * 70)
    print("PHASE 6: JSON serialization (mimics write_json on classical_results-sized payload)")
    print("=" * 70)
    payload = {
        f"key{i}": {"page_faults": i, "total_references": 2000, "hit_ratio": 0.5}
        for i in range(200 * 8)  # one family's worth of classical-result entries
    }
    _, t_serialize = timed(
        "json.dumps (1 family's classical_results)",
        json.dumps,
        payload,
        default=str,
    )

    print()
    print("=" * 70)
    print("SUMMARY (seconds, one episode)")
    print("=" * 70)
    print("  workload generation:        negligible (see PHASE 1 above)")
    print(f"  one replay (LRU):            {t_lru:.4f}s")
    print(f"  one replay (FIFO):           {t_fifo:.4f}s")
    print(f"  one replay (Belady_MIN):     {t_belady:.4f}s")
    print(f"  dataset generation (5 horizons): {t_dataset:.4f}s -> {len(examples)} examples")
    print(f"  feature matrix (1 horizon):  {t_matrix:.4f}s")
    if t_learned is not None:
        print(f"  LR fit (1 horizon):          {t_lr_fit:.4f}s")
        print(f"  LR predict:                  {t_lr_predict:.4f}s")
        print(f"  HGB fit (1 horizon):         {t_hgb_fit:.4f}s")
        print(f"  HGB predict:                 {t_hgb_predict:.4f}s")
        print(f"  learned-policy replay:       {t_learned:.4f}s")
    print(f"  JSON serialize (1600 entries): {t_serialize:.4f}s")


if __name__ == "__main__":
    main()
