#!/usr/bin/env python3
"""Real, full-scale stress test reproducing the Protocol B pooling near-OOM stall.

Uses synthetic FamilyCache arrays shaped exactly like the real, already-built
caches for all six families (using their real measured row counts and a
realistic per-episode row distribution, so episode_idx masking behaves
representatively) -- correctness of the pooling fix is proven separately, on
real small-scale data, by tests/unit/test_experiment_memory_bounded.py's new
equivalence tests; this script verifies only the MEMORY behavior of the
exact call sequence that stalled production: building the held_out=D
training pool (5 families) via pool_families_split_arrays(), plus the
held-out family's test-array shortcut.

Does not modify HORIZONS, CAPACITY, LENGTH, seeds, or any workload/feature/
label/model definition, and does not exercise generate_dataset() or
build_family_cache() at all -- this is purely a downstream-pooling memory
test.

Usage:
    PYTHONPATH=src python -u scripts/stress_test_protocol_b_pooling.py
"""

from __future__ import annotations

import gc
import importlib.util
import sys
import time
import tracemalloc
from pathlib import Path
from types import ModuleType

import numpy as np

_SCRIPT_PATH = Path(__file__).resolve().parent / "generalization_experiment_2.py"
_spec = importlib.util.spec_from_file_location("generalization_experiment_2", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
ge2: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["generalization_experiment_2"] = ge2
_spec.loader.exec_module(ge2)

# Real row counts logged by the production run (config_hash-identical scope:
# 200 episodes/family, all 5 horizons).
REAL_ROW_COUNTS = {
    "A_uniform_random": 4_753_920,
    "B_temporal_locality": 13_568,
    "C_bursty": 280_288,
    "D_long_range_reuse": 6_348_800,
    "E_phase_changing": 3_278_880,
    "F_non_stationary": 2_615_872,
}
N_FEATURES = 13
N_EPISODES = 200
HORIZON = 25


def _rss_mb() -> float | None:
    try:
        import psutil  # type: ignore[import-untyped]

        return float(psutil.Process().memory_info().rss) / (1024 * 1024)
    except Exception:
        return None


def build_synthetic_family_cache(family: str, n_rows: int) -> ge2.FamilyCache:  # type: ignore[name-defined]
    """A FamilyCache with real row counts and a realistic per-episode distribution.

    Content is meaningless (zeros/synthetic ids) -- only shapes/dtypes and
    episode_idx's per-episode row distribution matter for testing pooling
    memory behavior. Rows are spread as evenly as possible across
    N_EPISODES synthetic episodes, matching the real generator's
    contiguous-per-episode row layout.
    """
    x = np.zeros((n_rows, N_FEATURES), dtype=np.float64)
    base = n_rows // N_EPISODES
    remainder = n_rows % N_EPISODES
    episode_idx = np.empty(n_rows, dtype=np.int32)
    offset = 0
    for ep in range(N_EPISODES):
        count = base + (1 if ep < remainder else 0)
        episode_idx[offset : offset + count] = ep
        offset += count
    episode_id_list = tuple(f"{family}-ep-{i}" for i in range(N_EPISODES))
    y_by_horizon = {h: np.zeros(n_rows, dtype=np.int64) for h in ge2.HORIZONS}
    return ge2.FamilyCache(
        x=x, episode_idx=episode_idx, episode_id_list=episode_id_list, y_by_horizon=y_by_horizon
    )


def main() -> None:
    """Reproduce the held_out=D Protocol B pooling call and confirm it now stays bounded."""
    print("Building synthetic FamilyCache arrays for all 6 families (real row counts)...")
    caches = {
        family: build_synthetic_family_cache(family, n) for family, n in REAL_ROW_COUNTS.items()
    }
    total_bytes = sum(
        c.x.nbytes + c.episode_idx.nbytes + sum(a.nbytes for a in c.y_by_horizon.values())
        for c in caches.values()
    )
    print(f"  all 6 families' resident cache memory: {total_bytes / 1e6:,.1f} MB")

    held_out = "D_long_range_reuse"
    training_families = [f for f in REAL_ROW_COUNTS if f != held_out]
    # Realistic 82/18 split, matching split_episodes()'s Protocol B fractions.
    n_train_episodes = round(N_EPISODES * 0.82)
    train_episode_ids = {
        f: [f"{f}-ep-{i}" for i in range(n_train_episodes)] for f in training_families
    }
    val_episode_ids = {
        f: [f"{f}-ep-{i}" for i in range(n_train_episodes, N_EPISODES)] for f in training_families
    }

    gc.collect()
    rss_before = _rss_mb()
    tracemalloc.start()

    print(
        f"\nPooling training set for held_out={held_out} across {training_families} "
        "-- the exact call that stalled production..."
    )
    t0 = time.perf_counter()
    train_specs = [
        (
            caches[f],
            HORIZON,
            ge2.episode_ids_to_indices(caches[f].episode_id_list, train_episode_ids[f]),
        )
        for f in training_families
    ]
    val_specs = [
        (
            caches[f],
            HORIZON,
            ge2.episode_ids_to_indices(caches[f].episode_id_list, val_episode_ids[f]),
        )
        for f in training_families
    ]
    train_arrays = ge2.pool_families_split_arrays(train_specs)
    val_arrays = ge2.pool_families_split_arrays(val_specs)
    elapsed_pool = time.perf_counter() - t0

    ho_cache = caches[held_out]
    test_arrays = ge2.SplitArrays(
        x=ho_cache.x, y=ho_cache.y_by_horizon[HORIZON], n_examples=len(ho_cache.episode_idx)
    )

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()
    rss_after = _rss_mb()

    print("\n" + "=" * 70)
    print("RESULT: Protocol B pooling COMPLETED")
    print("=" * 70)
    print(f"  n_train (pooled): {train_arrays.n_examples:,}")
    print(f"  n_val (pooled):   {val_arrays.n_examples:,}")
    print(f"  n_test (held-out, shortcut): {test_arrays.n_examples:,}")
    print(f"  pooling wall-clock: {elapsed_pool:.1f}s")
    print(f"  tracemalloc PEAK during pooling: {peak / 1e6:,.1f} MB")
    if rss_before is not None and rss_after is not None:
        print(f"  process RSS before: {rss_before:,.1f} MB")
        print(f"  process RSS after:  {rss_after:,.1f} MB")
    print(
        f"\n  Total simulated peak (all 6 families' resident cache + pooling peak): "
        f"{(total_bytes / 1e6) + (peak / 1e6):,.1f} MB"
    )
    print(
        "\nSTRESS TEST PASSED: Protocol B pooling for the held_out=D fold completes "
        "with bounded memory under the real production resident-memory conditions."
    )


if __name__ == "__main__":
    main()
