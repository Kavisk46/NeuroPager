#!/usr/bin/env python3
"""Focused, full-scale memory diagnostic for the held_out=D training pipeline.

Measures peak memory for each stage of the LOCKED, UNCHANGED model-training
pipeline (StandardScaler->LogisticRegression, and
HistGradientBoostingClassifier, both via the real make_model()) at the
exact real production data scale that has failed twice in live production
during the held_out=D Protocol B pooling+training phase -- to locate where
memory usage exceeds the ~3.9GB the pooling-only stress test measured.

Uses synthetic data at the REAL measured row counts (pooled train=8,972,964,
val=1,969,564, 13 features) with RANDOM (not zero) values, so
HistGradientBoostingClassifier's histogram binning behaves representatively
-- this measures ARRAY-SIZE-DRIVEN memory behavior of the locked training
code, not scientific correctness (already proven separately on real data by
the equivalence tests). Does NOT change any model class, hyperparameter,
feature set, label definition, or scientific methodology.

Usage:
    PYTHONPATH=src python -u scripts/diagnose_training_memory.py lr
    PYTHONPATH=src python -u scripts/diagnose_training_memory.py hgb
    PYTHONPATH=src python -u scripts/diagnose_training_memory.py all
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
    make_model,
)

N_TRAIN = 8_972_964  # real measured pooled-train row count for held_out=D
N_VAL = 1_969_564  # real measured pooled-val row count for held_out=D
N_FEATURES = 13
SEED = 0


def _rss_mb() -> float | None:
    try:
        import psutil  # type: ignore[import-untyped]

        return float(psutil.Process().memory_info().rss) / (1024 * 1024)
    except Exception:
        return None


def _ru_maxrss_mb() -> float | None:
    try:
        import resource  # type: ignore[import-not-found]

        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024
    except Exception:
        return None  # not available on Windows


def report(label: str) -> None:
    """Print RSS/tracemalloc memory readings tagged with a stage label."""
    rss = _rss_mb()
    maxrss = _ru_maxrss_mb()
    current, peak = tracemalloc.get_traced_memory()
    parts = [f"RSS={rss:.1f}MB" if rss is not None else "RSS=?"]
    if maxrss is not None:
        parts.append(f"ru_maxrss={maxrss:.1f}MB")
    parts.append(f"tracemalloc_current={current / 1e6:.1f}MB")
    parts.append(f"tracemalloc_peak={peak / 1e6:.1f}MB")
    print(f"  [{label}] " + " ".join(parts))


def build_data(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Build synthetic (X, y) arrays of the given row count for memory-shape testing."""
    rng = np.random.default_rng(seed)
    x = rng.random((n, N_FEATURES)).astype(np.float64)
    y = rng.integers(0, 2, size=n).astype(np.int64)
    return x, y


def main() -> None:
    """Run the requested training stage(s) and report memory at each step."""
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(f"=== diagnose_training_memory.py stage={stage} ===")

    tracemalloc.start()
    gc.collect()
    report("start")

    print(
        f"Building synthetic train ({N_TRAIN:,} rows) and val ({N_VAL:,} rows), "
        f"{N_FEATURES} features, real production row counts..."
    )
    train_x, train_y = build_data(N_TRAIN, 1)
    val_x, val_y = build_data(N_VAL, 2)
    gc.collect()
    report("after data construction (pool equivalent)")

    if stage in ("lr", "all"):
        print("\n--- LogisticRegression pipeline (StandardScaler + LR) ---")
        gc.collect()
        report("before LR.fit")
        t0 = time.perf_counter()
        lr_model = make_model(LOGISTIC_REGRESSION, SEED)
        lr_model.fit(train_x, train_y)
        report("after LR.fit")
        lr_proba = lr_model.predict_proba(val_x)[:, 1]
        report("after LR.predict_proba")
        print(f"  LR total wall-clock: {time.perf_counter() - t0:.1f}s")
        del lr_model, lr_proba
        gc.collect()
        report("after LR cleanup (del + gc.collect)")

    if stage in ("hgb", "all"):
        print("\n--- HistGradientBoostingClassifier ---")
        gc.collect()
        report("before HGB.fit")
        t0 = time.perf_counter()
        hgb_model = make_model(HIST_GRADIENT_BOOSTING, SEED)
        hgb_model.fit(train_x, train_y)
        report("after HGB.fit")
        hgb_proba = hgb_model.predict_proba(val_x)[:, 1]
        report("after HGB.predict_proba")
        print(f"  HGB total wall-clock: {time.perf_counter() - t0:.1f}s")
        del hgb_model, hgb_proba
        gc.collect()
        report("after HGB cleanup (del + gc.collect)")

    tracemalloc.stop()
    report("end")


if __name__ == "__main__":
    main()
