#!/usr/bin/env python3
"""Measure real peak memory of an HGB fit at genuine production scale.

Builds 200 real A_uniform_random episodes, caches them at H=25 (the exact
same (family, horizon) already checkpointed in
experiments/generalization-experiment-2/, whose n_train_examples=3,324,720
is used below as a cross-check that this script's split reproduces the
same numbers), takes the real 70% train split, and fits
HistGradientBoostingClassifier with the LOCKED, unmodified
make_model()/model_config() on it -- the single most memory-expensive
model-training call the production run makes for this family/horizon --
measuring tracemalloc peak and psutil RSS around the .fit() call
specifically (not dataset generation, which was already measured
separately in scripts/measure_family_cache_memory.py).

Does not modify any hyperparameter, does not substitute another model,
does not run the complete experiment.

Usage:
    PYTHONPATH=src python -u scripts/measure_hgb_production_scale_memory.py
"""

from __future__ import annotations

import gc
import importlib.util
import sys
import time
import tracemalloc
from pathlib import Path
from types import ModuleType

_SCRIPT_PATH = Path(__file__).resolve().parent / "generalization_experiment_2.py"
_spec = importlib.util.spec_from_file_location("generalization_experiment_2", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
ge2: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["generalization_experiment_2"] = ge2
_spec.loader.exec_module(ge2)

from neuropager.experiment.models import HIST_GRADIENT_BOOSTING, make_model  # noqa: E402

FAMILY = "A_uniform_random"
HORIZON = 25
N_EPISODES = 200
EXPECTED_N_TRAIN_FROM_CHECKPOINT = 3_324_720  # cross-check only, not asserted strictly


def _rss_mb() -> float | None:
    try:
        import psutil  # type: ignore[import-untyped]

        return float(psutil.Process().memory_info().rss) / (1024 * 1024)
    except Exception:
        return None


def main() -> None:
    """Fit HGB on a real production-scale train set and report its measured peak memory."""
    print(f"Building {N_EPISODES} real episodes for {FAMILY}...")
    episodes = [ge2.build_episode(FAMILY, seed)[0] for seed in range(N_EPISODES)]

    print(f"Building FamilyCache at H={HORIZON} only (matches the checkpointed unit)...")
    t0 = time.perf_counter()
    cache = ge2.build_family_cache(episodes, [HORIZON])
    print(f"  cache built in {time.perf_counter() - t0:.1f}s, {len(cache.episode_idx):,} rows")

    split = ge2.split_episodes([e.episode_id for e in episodes], seed=ge2.TRAIN_SEED)
    train_idx = ge2.episode_ids_to_indices(cache.episode_id_list, split.train_episodes)
    train = ge2.select_split_arrays(
        cache.x, cache.y_by_horizon[HORIZON], cache.episode_idx, train_idx
    )
    print(f"  n_train = {train.n_examples:,} (checkpoint had {EXPECTED_N_TRAIN_FROM_CHECKPOINT:,})")

    gc.collect()
    rss_before = _rss_mb()
    tracemalloc.start()

    print(
        "Fitting HistGradientBoostingClassifier (locked config, unmodified) "
        "on the real train set..."
    )
    t0 = time.perf_counter()
    model = make_model(HIST_GRADIENT_BOOSTING, ge2.TRAIN_SEED)
    model.fit(train.x, train.y)
    fit_seconds = time.perf_counter() - t0

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()
    rss_after = _rss_mb()

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  n_train:                         {train.n_examples:,}")
    print(f"  train.x.nbytes:                  {train.x.nbytes / 1e6:,.1f} MB")
    print(f"  fit wall-clock:                  {fit_seconds:,.1f}s")
    print(f"  tracemalloc current (post-fit):  {current / 1e6:,.1f} MB")
    print(f"  tracemalloc PEAK during .fit():  {peak / 1e6:,.1f} MB")
    if rss_before is not None and rss_after is not None:
        print(f"  process RSS before .fit():       {rss_before:,.1f} MB")
        print(f"  process RSS after .fit():        {rss_after:,.1f} MB")
        print(f"  process RSS delta:               {rss_after - rss_before:,.1f} MB")
    else:
        print("  process RSS: psutil not available, skipped")


if __name__ == "__main__":
    main()
