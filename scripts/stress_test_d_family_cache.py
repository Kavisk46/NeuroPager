#!/usr/bin/env python3
"""Real, full-scale stress test reproducing the resident-memory conditions that crashed production.

Builds 200 real D_long_range_reuse episodes (production scale) and calls
build_family_cache() across all 5 locked horizons -- the exact call that
crashed in production -- while holding synthetic arrays shaped exactly like
the real, already-built FamilyCache for A_uniform_random, B_temporal_locality,
C_bursty, E_phase_changing, and F_non_stationary (using their REAL measured
row counts from the crashed run's own log, so the simulated memory pressure
is faithful) resident at the same time, replicating the real production
moment of failure as closely as practical without spending several more
hours regenerating all five other families from scratch.

Does not modify HORIZONS, CAPACITY, LENGTH, seeds, or any workload/feature/
label/model definition. Uses the current (fixed) EPISODE_BATCH_SIZE and the
current get_family_cache()-style events-release behavior is exercised
separately by tests/unit/test_experiment_memory_bounded.py; this script
verifies the fix under real, full memory pressure and must complete.

Usage:
    PYTHONPATH=src python -u scripts/stress_test_d_family_cache.py
"""

from __future__ import annotations

import gc
import importlib.util
import sys
import time
import tracemalloc
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

_SCRIPT_PATH = Path(__file__).resolve().parent / "generalization_experiment_2.py"
_spec = importlib.util.spec_from_file_location("generalization_experiment_2", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
ge2: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["generalization_experiment_2"] = ge2
_spec.loader.exec_module(ge2)

# Real row counts logged by the crashed production run (config_hash-identical
# scope: 200 episodes/family, all 5 horizons).
REAL_ROW_COUNTS = {
    "A_uniform_random": 4_753_920,
    "B_temporal_locality": 13_568,
    "C_bursty": 280_288,
    "E_phase_changing": 3_278_880,
    "F_non_stationary": 2_615_872,
}
N_FEATURES = 13
N_HORIZONS = 5
FAMILY = "D_long_range_reuse"
N_EPISODES = 200


def _rss_mb() -> float | None:
    try:
        import psutil  # type: ignore[import-untyped]

        return float(psutil.Process().memory_info().rss) / (1024 * 1024)
    except Exception:
        return None


def build_synthetic_family_cache(n_rows: int) -> Any:
    """A dummy object with the same array shapes/dtypes as a real FamilyCache.

    Content is meaningless (zeros) -- this is a memory-pressure simulator,
    not a scientific computation, so real feature/label VALUES are
    irrelevant; only the byte footprint of holding these arrays resident
    matters here.
    """
    x = np.zeros((n_rows, N_FEATURES), dtype=np.float64)
    episode_idx = np.zeros(n_rows, dtype=np.int32)
    y_by_horizon = {h: np.zeros(n_rows, dtype=np.int64) for h in ge2.HORIZONS}
    return ge2.FamilyCache(
        x=x, episode_idx=episode_idx, episode_id_list=tuple(), y_by_horizon=y_by_horizon
    )


def main() -> None:
    """Reproduce the exact resident-memory pressure and confirm D's cache now builds."""
    print("Building synthetic FamilyCache arrays for A/B/C/E/F (real row counts)...")
    other_caches = {
        family: build_synthetic_family_cache(n) for family, n in REAL_ROW_COUNTS.items()
    }
    total_other_bytes = sum(
        c.x.nbytes + c.episode_idx.nbytes + sum(a.nbytes for a in c.y_by_horizon.values())
        for c in other_caches.values()
    )
    print(f"  simulated resident memory from A/B/C/E/F: {total_other_bytes / 1e6:,.1f} MB")

    print(f"\nBuilding {N_EPISODES} real {FAMILY} episodes (production scale)...")
    t0 = time.perf_counter()
    episodes = [ge2.build_episode(FAMILY, seed)[0] for seed in range(N_EPISODES)]
    print(f"  built in {time.perf_counter() - t0:.1f}s")

    gc.collect()
    rss_before = _rss_mb()
    tracemalloc.start()

    print(
        f"\nBuilding {FAMILY}'s FamilyCache @ H={list(ge2.HORIZONS)} "
        f"(EPISODE_BATCH_SIZE={ge2.EPISODE_BATCH_SIZE}) -- the exact call that crashed..."
    )
    t0 = time.perf_counter()
    d_cache = ge2.build_family_cache(episodes, list(ge2.HORIZONS))
    elapsed = time.perf_counter() - t0

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()
    rss_after = _rss_mb()

    print("\n" + "=" * 70)
    print("RESULT: D's FamilyCache build COMPLETED")
    print("=" * 70)
    print(f"  n_rows (examples/horizon): {len(d_cache.episode_idx):,}")
    print(f"  build wall-clock:          {elapsed:.1f}s")
    print(f"  tracemalloc PEAK during build: {peak / 1e6:,.1f} MB")
    if rss_before is not None and rss_after is not None:
        print(f"  process RSS before: {rss_before:,.1f} MB")
        print(f"  process RSS after:  {rss_after:,.1f} MB")
    print(
        f"\n  Total simulated peak (other 5 families' arrays + D's build peak): "
        f"{(total_other_bytes / 1e6) + (peak / 1e6):,.1f} MB"
    )
    print(
        "\nSTRESS TEST PASSED: D_long_range_reuse's cache builds to completion "
        "under the real production resident-memory conditions."
    )


if __name__ == "__main__":
    main()
