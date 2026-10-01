#!/usr/bin/env python3
"""Measure real peak memory of build_family_cache() at representative production scale.

Not an estimate: builds 200 real episodes of the densest known family
(A_uniform_random -- measured at ~23,488 examples/episode/horizon, the
worst case observed among the six locked families) and calls
build_family_cache() across all 5 locked horizons at once (the single most
memory-expensive call the production run makes, since it is the case that
holds every horizon's labels simultaneously for one family). Reports:

    - tracemalloc peak (Python-level allocations, includes numpy buffers)
    - process RSS before/after/delta, via psutil if available
    - the resulting FamilyCache's actual numpy array byte sizes (.nbytes)

Usage:
    PYTHONPATH=src python -u scripts/measure_family_cache_memory.py
"""

from __future__ import annotations

import gc
import importlib.util
import sys
import tracemalloc
from pathlib import Path
from types import ModuleType

_SCRIPT_PATH = Path(__file__).resolve().parent / "generalization_experiment_2.py"
_spec = importlib.util.spec_from_file_location("generalization_experiment_2", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
ge2: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["generalization_experiment_2"] = ge2
_spec.loader.exec_module(ge2)

FAMILY = "A_uniform_random"
N_EPISODES = 200


def _rss_mb() -> float | None:
    try:
        import psutil  # type: ignore[import-untyped]

        return float(psutil.Process().memory_info().rss) / (1024 * 1024)
    except Exception:
        return None


def main() -> None:
    """Build a real 200-episode family cache and report its measured peak memory."""
    print(f"Building {N_EPISODES} real episodes for {FAMILY} (unchanged build_episode())...")
    episodes = [ge2.build_episode(FAMILY, seed)[0] for seed in range(N_EPISODES)]
    n_events = sum(len(e.events) for e in episodes)
    print(f"  {N_EPISODES} episodes, {n_events} trace events total")

    gc.collect()
    rss_before = _rss_mb()
    tracemalloc.start()

    print(f"Building FamilyCache across all 5 horizons {ge2.HORIZONS} (the worst-case call)...")
    cache = ge2.build_family_cache(episodes, list(ge2.HORIZONS))

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()
    rss_after = _rss_mb()

    n_rows = len(cache.episode_idx)
    x_bytes = cache.x.nbytes
    idx_bytes = cache.episode_idx.nbytes
    y_bytes = sum(arr.nbytes for arr in cache.y_by_horizon.values())
    total_array_bytes = x_bytes + idx_bytes + y_bytes

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  rows (examples/horizon):        {n_rows:,}")
    print(f"  cache.x.nbytes:                 {x_bytes / 1e6:,.1f} MB")
    print(f"  cache.episode_idx.nbytes:       {idx_bytes / 1e6:,.1f} MB")
    print(f"  sum(y_by_horizon[*].nbytes):    {y_bytes / 1e6:,.1f} MB")
    print(f"  total FamilyCache array bytes:  {total_array_bytes / 1e6:,.1f} MB")
    print(f"  tracemalloc current (Python allocations, post-call): {current / 1e6:,.1f} MB")
    print(f"  tracemalloc PEAK during build_family_cache():        {peak / 1e6:,.1f} MB")
    if rss_before is not None and rss_after is not None:
        print(f"  process RSS before build_family_cache(): {rss_before:,.1f} MB")
        print(f"  process RSS after build_family_cache():  {rss_after:,.1f} MB")
        print(f"  process RSS delta:                       {rss_after - rss_before:,.1f} MB")
    else:
        print("  process RSS: psutil not available, skipped")
    print()
    print("  Extrapolated worst case, all 6 families this dense, held simultaneously:")
    print(f"    ~{6 * total_array_bytes / 1e6:,.0f} MB steady-state array memory")


if __name__ == "__main__":
    main()
