#!/usr/bin/env python3
"""Measure how HistGradientBoosting.fit time scales with real training-set size.

Generates real dataset examples from several A_uniform_random episodes
(using the now-optimized InMemoryPageStore + bisect-based TraceReplay, both
already verified equivalent to the originals), pools them, and times
HGB.fit at increasing subset sizes -- to empirically extrapolate to the
locked experiment's real per-family (~140 episodes) and per-fold (~820-1000
episodes) training-set sizes, rather than guessing from a single data
point.

Usage:
    PYTHONPATH=src python -u scripts/profile_hgb_scaling.py
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
from neuropager.trace.events import TraceEvent
from neuropager.trace.logger import TraceLogger
from neuropager.workloads.config import UniformRandomConfig
from neuropager.workloads.generator import generate_workload

LENGTH = 2000
CAPACITY = 16
N_EPISODES = 3  # generate this many episodes' worth of real examples to draw subsets from


def build_episode_events(seed: int) -> list[TraceEvent]:
    """Replay one A_uniform_random episode under LRU, return its trace events."""
    config = UniformRandomConfig(length=LENGTH, key_space=64, seed=seed)
    workload = generate_workload(config)
    trace_logger = TraceLogger(f"scaling-ep-{seed}")
    wm = WorkingMemory(capacity=CAPACITY)
    pt = PageTable()
    fh = PageFaultHandler(
        working_memory=wm,
        page_table=pt,
        disk_store=InMemoryPageStore(),
        policy=LRUPolicy(),
        trace_logger=trace_logger,
    )
    manager = MemoryManager(wm, pt, fh)
    seen = set()
    for key in workload:
        if key not in seen:
            manager.put(key, str(key))
            seen.add(key)
        else:
            manager.get(key)
    return trace_logger.events


def main() -> None:
    """Time HGB.fit at increasing real-data subset sizes and print a scaling table."""
    print(f"Generating {N_EPISODES} episodes and their h=50 examples...")
    start = time.perf_counter()
    all_events: list[TraceEvent] = []
    for seed in range(N_EPISODES):
        all_events.extend(build_episode_events(seed))
    examples = generate_dataset(all_events, horizons=[50])
    elapsed = time.perf_counter() - start
    print(f"  {len(examples)} examples from {N_EPISODES} episodes, {elapsed:.2f}s")

    x_full, y_full, _ = build_feature_matrix(examples)
    per_episode = len(examples) / N_EPISODES
    print(f"  ~{per_episode:.0f} examples/episode\n")

    print(f"{'episodes':>10s} {'n_rows':>10s} {'fit_seconds':>14s} {'s/1000rows':>12s}")
    for n_ep in [1, 2, 3]:
        n_rows = round(per_episode * n_ep)
        n_rows = min(n_rows, len(x_full))
        x_sub, y_sub = x_full[:n_rows], y_full[:n_rows]
        model = make_model(HIST_GRADIENT_BOOSTING, seed=0)
        t0 = time.perf_counter()
        model.fit(x_sub, y_sub)
        elapsed = time.perf_counter() - t0
        print(f"{n_ep:10d} {n_rows:10d} {elapsed:14.4f} {elapsed / n_rows * 1000:12.4f}")


if __name__ == "__main__":
    main()
