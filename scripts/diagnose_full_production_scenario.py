#!/usr/bin/env python3
"""Most faithful reproduction of the held_out=D production failure scenario.

Combines everything the two prior diagnostics measured in isolation:
all 6 families' FamilyCache arrays resident (real measured row counts) +
pool_families_split_arrays() pooling (the LOCKED, already-fixed orchestration
code) + the real, LOCKED LogisticRegression and HistGradientBoostingClassifier
training pipeline (via make_model(), unmodified) -- run sequentially in one
process, exactly matching build_horizon_result()'s real call order. This is
the closest a synthetic-data script can get to the actual production
scenario that has failed twice.

Uses synthetic (random, not zero) array content at real measured shapes --
correctness of every piece here (pooling, model training) is already proven
separately on real data by the existing equivalence tests; this script
measures only peak memory behavior at full production scale.

Does NOT change any model class, hyperparameter, feature set, label
definition, or scientific methodology.

Usage:
    PYTHONPATH=src python -u scripts/diagnose_full_production_scenario.py
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

from neuropager.experiment.models import (  # noqa: E402
    HIST_GRADIENT_BOOSTING,
    LOGISTIC_REGRESSION,
)

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


def report(label: str) -> None:
    """Print RSS/tracemalloc memory readings tagged with a stage label."""
    rss = _rss_mb()
    current, peak = tracemalloc.get_traced_memory()
    print(
        f"  [{label}] RSS={rss:.1f}MB "
        f"tracemalloc_current={current / 1e6:.1f}MB tracemalloc_peak={peak / 1e6:.1f}MB"
    )


def build_synthetic_family_cache(family: str, n_rows: int, seed: int) -> ge2.FamilyCache:  # type: ignore[name-defined]
    """Build a synthetic FamilyCache with real row counts and per-episode row distribution."""
    rng = np.random.default_rng(seed)
    x = rng.random((n_rows, N_FEATURES)).astype(np.float64)
    base = n_rows // N_EPISODES
    remainder = n_rows % N_EPISODES
    episode_idx = np.empty(n_rows, dtype=np.int32)
    offset = 0
    for ep in range(N_EPISODES):
        count = base + (1 if ep < remainder else 0)
        episode_idx[offset : offset + count] = ep
        offset += count
    episode_id_list = tuple(f"{family}-ep-{i}" for i in range(N_EPISODES))
    y_by_horizon = {h: rng.integers(0, 2, size=n_rows).astype(np.int64) for h in ge2.HORIZONS}
    return ge2.FamilyCache(
        x=x, episode_idx=episode_idx, episode_id_list=episode_id_list, y_by_horizon=y_by_horizon
    )


def main() -> None:
    """Reproduce the full held_out=D unit: 6 caches resident + pooling + LR + HGB."""
    tracemalloc.start()
    gc.collect()
    report("start")

    print("Building synthetic FamilyCache arrays for all 6 families (real row counts)...")
    caches = {
        family: build_synthetic_family_cache(family, n, seed=i)
        for i, (family, n) in enumerate(REAL_ROW_COUNTS.items())
    }
    gc.collect()
    report("after all 6 families cached (matches real production state entering held_out=D)")

    held_out = "D_long_range_reuse"
    training_families = [f for f in REAL_ROW_COUNTS if f != held_out]
    n_train_episodes = round(N_EPISODES * 0.82)
    train_episode_ids = {
        f: [f"{f}-ep-{i}" for i in range(n_train_episodes)] for f in training_families
    }
    val_episode_ids = {
        f: [f"{f}-ep-{i}" for i in range(n_train_episodes, N_EPISODES)] for f in training_families
    }

    print("Pooling training set (pool_families_split_arrays(), the real fixed code)...")
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
    gc.collect()
    report(f"after pooling (n_train={train_arrays.n_examples:,}, n_val={val_arrays.n_examples:,})")

    print("\n--- LogisticRegression via real train_and_get_model() (includes new gc.collect()) ---")
    t0 = time.perf_counter()
    lr_result = ge2.train_and_get_model(train_arrays, LOGISTIC_REGRESSION)
    report("after LR.fit (via train_and_get_model)")
    assert lr_result is not None
    lr_model, _lr_train_metrics = lr_result
    lr_val_metrics = ge2.eval_model(lr_model, val_arrays)
    report("after LR eval_model on val")
    print(f"  LR wall-clock: {time.perf_counter() - t0:.1f}s")

    print("\n--- HistGradientBoostingClassifier via real train_and_get_model() ---")
    t0 = time.perf_counter()
    hgb_result = ge2.train_and_get_model(train_arrays, HIST_GRADIENT_BOOSTING)
    report("after HGB.fit (via train_and_get_model)")
    assert hgb_result is not None
    hgb_model, _hgb_train_metrics = hgb_result
    hgb_val_metrics = ge2.eval_model(hgb_model, val_arrays)
    report("after HGB eval_model on val")
    print(f"  HGB wall-clock: {time.perf_counter() - t0:.1f}s")

    del lr_model, lr_val_metrics, hgb_model, hgb_val_metrics
    gc.collect()
    report("after model cleanup")

    tracemalloc.stop()
    report("end")
    print(
        "\nDIAGNOSTIC COMPLETE: this is the closest synthetic reproduction of the real "
        "held_out=D Protocol B unit (6 caches resident + pooling + LR + HGB, in order)."
    )


if __name__ == "__main__":
    main()
