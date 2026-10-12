#!/usr/bin/env python3
"""Predictive-only train/serve sensitivity analysis for elapsed_since_fault,
scoped to all 30 Protocol B (leave-one-family-out) units.

Read-only with respect to the production campaign: this script imports
scripts/generalization_experiment_2.py's own functions unmodified (dynamic
import, same pattern as the repo's other diagnostic scripts) and reuses
them exactly -- build_episode(), build_family_cache(), split_episodes(),
pool_families_split_arrays(), train_and_get_model(), eval_model(),
compute_metrics() -- so every number produced here is computed by the same
code paths the original campaign used, not reimplemented.

No existing checkpoint, CSV, or report is read for writing, and none is
written to. No operational (page-fault) replay is performed anywhere in
this script -- MemoryManager/PageFaultHandler/LearnedUtilityPolicy are
never instantiated. Only two model classes are ever fit: HistGradientBoosting,
twice per unit (CONDITION A: original features; CONDITION B: the same
feature matrices with column `elapsed_since_fault` forced to -1.0).

Workloads are regenerated deterministically from the same seeds the
production campaign used (FAMILY_BUILDERS[family](seed)) -- verified in a
prior feasibility check to reproduce the stored classical-baseline
checkpoint for E_phase_changing-ep-0 exactly. No workload generator, seed,
capacity, episode length, or feature definition is modified; the only
engineered difference between conditions is the single feature column
named above.

Usage:
    PYTHONPATH=src python -u scripts/train_serve_sensitivity_protocol_b.py
"""

from __future__ import annotations

import gc
import importlib.util
import json
import sys
import time
import tracemalloc
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO / "scripts" / "generalization_experiment_2.py"
OUT_DIR = REPO / "experiments" / "generalization-experiment-2" / "analysis" / "train_serve_sensitivity"

spec = importlib.util.spec_from_file_location("generalization_experiment_2", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
ge2: ModuleType = importlib.util.module_from_spec(spec)
sys.modules["generalization_experiment_2"] = ge2
spec.loader.exec_module(ge2)

from neuropager.experiment.models import HIST_GRADIENT_BOOSTING  # noqa: E402

ELAPSED_SINCE_FAULT_INDEX = list(ge2.FEATURE_NAMES).index("elapsed_since_fault")
assert ELAPSED_SINCE_FAULT_INDEX == 11, "feature order changed -- stop, do not guess"
RUNTIME_SENTINEL = -1.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rss_mb() -> float | None:
    try:
        import psutil  # type: ignore[import-untyped]

        return float(psutil.Process().memory_info().rss) / (1024 * 1024)
    except Exception:
        return None


def runtime_faithful(split: "ge2.SplitArrays") -> "ge2.SplitArrays":
    """Return a new SplitArrays with elapsed_since_fault forced to -1.0.

    Copies .x (never mutates the original array in place -- CONDITION A's
    arrays must remain untouched for CONDITION A's own fit). .y and
    n_examples are passed through unchanged. No other column is modified.
    """
    if split.x is None:
        return split
    x2 = split.x.copy()
    x2[:, ELAPSED_SINCE_FAULT_INDEX] = RUNTIME_SENTINEL
    return ge2.SplitArrays(x=x2, y=split.y, n_examples=split.n_examples)


def metrics_to_row(m: "ge2.ClassificationMetrics | None") -> dict[str, Any]:
    if m is None:
        return {
            "n_examples": None, "roc_auc": None, "pr_auc": None, "brier_score": None,
            "f1": None, "precision": None, "recall": None, "accuracy": None,
        }
    return {
        "n_examples": m.n_examples, "roc_auc": m.roc_auc, "pr_auc": m.pr_auc,
        "brier_score": m.brier_score, "f1": m.f1, "precision": m.precision,
        "recall": m.recall, "accuracy": m.accuracy,
    }


CACHE_DIR = OUT_DIR / "_family_cache_scratch"


def cache_path(family: str) -> Path:
    return CACHE_DIR / f"{family}.npz"


def save_family_cache(family: str, cache: "ge2.FamilyCache") -> None:
    """Persist one family's cache to disk so a restart can skip rebuilding it.

    Scratch-only: lives under the new analysis output directory, never
    touches any existing production artifact, and holds nothing the
    production pipeline doesn't already compute -- just a disk copy of the
    in-memory FamilyCache this script already builds via build_family_cache().
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {"x": cache.x, "episode_idx": cache.episode_idx}
    for h, y in cache.y_by_horizon.items():
        arrays[f"y_{h}"] = y
    # np.savez() silently appends ".npz" to any path not already ending in
    # ".npz" -- naming this "<family>.tmp.npz" (not "<family>.npz.tmp")
    # keeps the on-disk name identical to what we pass in, so the rename
    # below renames the file that was actually written.
    tmp_path = CACHE_DIR / f"{family}.tmp.npz"
    np.savez(tmp_path, **arrays)
    tmp_path.rename(cache_path(family))
    (CACHE_DIR / f"{family}.episode_ids.json").write_text(
        json.dumps(list(cache.episode_id_list)), encoding="utf-8"
    )


def batch_dir(family: str) -> Path:
    return CACHE_DIR / f"{family}_batches"


def batch_file(family: str, batch_idx: int) -> Path:
    return batch_dir(family) / f"batch_{batch_idx:04d}.npz"


def build_family_cache_resumable(family: str) -> "ge2.FamilyCache":
    """Same inputs/outputs as ge2.build_family_cache(), checkpointed per batch.

    Mirrors build_family_cache()'s own loop body exactly (same
    EPISODE_BATCH_SIZE, same generate_dataset()/build_feature_matrix()/
    _dedup_horizon_stride() calls, same per-horizon slicing) -- the only
    difference is that each batch's arrays are saved to scratch disk right
    after being computed, and a batch already saved by a prior (possibly
    interrupted) run is loaded instead of recomputed. A batch's episodes
    are only regenerated via build_episode() when that batch isn't already
    on disk, so a restart re-pays at most one batch's replay cost, not a
    whole family's.
    """
    sorted_horizons = sorted(ge2.HORIZONS)
    n_horizons = len(sorted_horizons)
    n_episodes = ge2.FULL_SPEC_EPISODES_PER_FAMILY
    batch_size = ge2.EPISODE_BATCH_SIZE
    episode_id_list = tuple(f"{family}-ep-{seed}" for seed in range(n_episodes))

    bdir = batch_dir(family)
    bdir.mkdir(parents=True, exist_ok=True)

    x_chunks: list[Any] = []
    idx_chunks: list[Any] = []
    y_chunks: dict[int, list[Any]] = {h: [] for h in sorted_horizons}

    for batch_start in range(0, n_episodes, batch_size):
        batch_idx = batch_start // batch_size
        bfile = batch_file(family, batch_idx)
        if bfile.exists():
            with np.load(bfile) as data:
                x_b = np.array(data["x"])
                idx_b = np.array(data["episode_idx"])
                y_b = {h: np.array(data[f"y_{h}"]) for h in sorted_horizons}
        else:
            batch_episodes = [
                ge2.build_episode(family, seed)[0]
                for seed in range(batch_start, min(batch_start + batch_size, n_episodes))
            ]
            batch_events = [event for ep in batch_episodes for event in ep.events]
            examples = ge2.generate_dataset(batch_events, horizons=sorted_horizons)
            if not examples:
                x_b = np.empty((0, len(ge2.FEATURE_NAMES)))
                idx_b = np.empty((0,), dtype=np.int32)
                y_b = {h: np.empty((0,), dtype=np.int64) for h in sorted_horizons}
            else:
                dedup = ge2._dedup_horizon_stride(examples, n_horizons)
                x_b, _, _ = ge2.build_feature_matrix(dedup)
                # episode_id is f"{family}-ep-{seed}" and episode_id_list is
                # seed-ordered (index == seed), so the seed IS the global
                # index -- identical to build_family_cache()'s own
                # episode_id_to_idx[e.episode_id] lookup, just without
                # needing the full family's id-to-index dict in scope.
                idx_b = np.array(
                    [int(e.episode_id.rsplit("-", 1)[-1]) for e in dedup],
                    dtype=np.int32,
                )
                y_b = {}
                for offset, horizon in enumerate(sorted_horizons):
                    horizon_examples = examples[offset::n_horizons]
                    y_b[horizon] = np.array([e.label for e in horizon_examples], dtype=np.int64)
            tmp = bdir / f"batch_{batch_idx:04d}.tmp.npz"
            save_arrays = {"x": x_b, "episode_idx": idx_b}
            for h, y in y_b.items():
                save_arrays[f"y_{h}"] = y
            np.savez(tmp, **save_arrays)
            tmp.rename(bfile)
            del batch_episodes, batch_events, examples
            gc.collect()
            log(f"    batch {batch_idx + 1}/{(n_episodes + batch_size - 1) // batch_size} done, rss={rss_mb()}")

        x_chunks.append(x_b)
        idx_chunks.append(idx_b)
        for h in sorted_horizons:
            y_chunks[h].append(y_b[h])

    x = np.concatenate(x_chunks) if x_chunks else np.empty((0, len(ge2.FEATURE_NAMES)))
    episode_idx = np.concatenate(idx_chunks) if idx_chunks else np.empty((0,), dtype=np.int32)
    y_by_horizon = {
        h: (np.concatenate(chunks) if chunks else np.empty((0,), dtype=np.int64))
        for h, chunks in y_chunks.items()
    }
    return ge2.FamilyCache(
        x=x, episode_idx=episode_idx, episode_id_list=episode_id_list, y_by_horizon=y_by_horizon
    )


def load_family_cache(family: str) -> "ge2.FamilyCache | None":
    p = cache_path(family)
    ids_path = CACHE_DIR / f"{family}.episode_ids.json"
    if not p.exists() or not ids_path.exists():
        return None
    with np.load(p) as data:
        x = np.array(data["x"])
        episode_idx = np.array(data["episode_idx"])
        y_by_horizon = {h: np.array(data[f"y_{h}"]) for h in ge2.HORIZONS}
    episode_id_list = tuple(json.loads(ids_path.read_text(encoding="utf-8")))
    return ge2.FamilyCache(
        x=x, episode_idx=episode_idx,
        episode_id_list=episode_id_list, y_by_horizon=y_by_horizon,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tracemalloc.start()
    t_start = time.perf_counter()

    log("=== Protocol B train/serve sensitivity analysis: elapsed_since_fault ===")
    log(f"Feature index under test: {ELAPSED_SINCE_FAULT_INDEX} (name={ge2.FEATURE_NAMES[ELAPSED_SINCE_FAULT_INDEX]!r})")
    log(f"Output directory (new, separate from production artifacts): {OUT_DIR}")

    # --- Steps 1+2, per family, resumable at the batch level: a family ---
    #     whose final cache is already saved is loaded instantly; otherwise
    #     build_family_cache_resumable() regenerates and processes only the
    #     episode batches not already checkpointed on scratch disk, so a
    #     restart mid-family re-pays at most one batch's cost, not the
    #     whole family's (previously ~3h of regeneration+caching for the
    #     largest families was lost whole on every interruption).
    family_cache: dict[str, Any] = {}
    for family in ge2.FAMILIES:
        cached = load_family_cache(family)
        if cached is not None:
            family_cache[family] = cached
            log(f"  loaded cached family cache from disk: {family} ({len(cached.episode_idx)} examples/horizon)")
            continue
        log(f"  building family cache (resumable): {family}")
        cache = build_family_cache_resumable(family)
        save_family_cache(family, cache)
        family_cache[family] = cache
        import shutil
        shutil.rmtree(batch_dir(family), ignore_errors=True)
        gc.collect()
        log(f"    cached {len(cache.episode_idx)} examples/horizon, saved to disk, rss={rss_mb()}")

    log(f"All 6 family caches ready. rss={rss_mb()}")

    # episode_id_list is still needed (not full Episode objects) to build
    # Protocol B's fold splits below -- reuse the cached id lists, no need
    # to keep the regenerated Episode objects themselves alive.
    episode_ids_by_family = {f: family_cache[f].episode_id_list for f in ge2.FAMILIES}

    # --- Step 3: for each of the 30 Protocol B units, fit HGB under both ---
    #     conditions and record predictive metrics only. Resumable: a unit
    #     already fully written to the detail CSV (all 6 rows: 2 conditions
    #     x 3 splits) from a prior interrupted run is skipped, and every
    #     completed unit's rows are flushed to disk immediately rather than
    #     held in memory until the end.
    detail_path = OUT_DIR / "protocol_b_train_serve_sensitivity.csv"
    fieldnames = [
        "held_out_family", "horizon", "n_train", "n_val", "n_test",
        "condition", "split", "n_examples", "roc_auc", "pr_auc",
        "brier_score", "f1", "precision", "recall", "accuracy",
    ]
    done_units: set[tuple[str, int]] = set()
    if detail_path.exists():
        import csv as csv_mod
        with detail_path.open("r", newline="", encoding="utf-8") as f:
            existing_rows = list(csv_mod.DictReader(f))
        counts: dict[tuple[str, int], int] = {}
        for r in existing_rows:
            key = (r["held_out_family"], int(r["horizon"]))
            counts[key] = counts.get(key, 0) + 1
        done_units = {k for k, n in counts.items() if n == 6}
        log(f"resuming: {len(done_units)}/30 units already complete in {detail_path.name}")
        write_header = False
    else:
        write_header = True

    csv_file = detail_path.open("a", newline="", encoding="utf-8")
    import csv as csv_mod
    csv_writer = csv_mod.DictWriter(csv_file, fieldnames=fieldnames)
    if write_header:
        csv_writer.writeheader()
        csv_file.flush()

    def write_rows(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            csv_writer.writerow(row)
        csv_file.flush()

    unit_index = 0
    for held_out in ge2.FAMILIES:
        training_families = [f for f in ge2.FAMILIES if f != held_out]
        training_episode_ids = [
            eid for f in training_families for eid in episode_ids_by_family[f]
        ]
        fold_split = ge2.split_episodes(
            training_episode_ids, seed=ge2.TRAIN_SEED,
            train_fraction=0.82, val_fraction=0.18, test_fraction=0.0,
        )
        for horizon in ge2.HORIZONS:
            unit_index += 1
            if (held_out, horizon) in done_units:
                log(f"[{unit_index}/30] held_out={held_out} H={horizon} -- already done, skipping")
                continue
            log(f"[{unit_index}/30] held_out={held_out} H={horizon}")

            train_specs = []
            val_specs = []
            for f in training_families:
                cache = family_cache[f]
                train_idx = ge2.episode_ids_to_indices(cache.episode_id_list, fold_split.train_episodes)
                val_idx = ge2.episode_ids_to_indices(cache.episode_id_list, fold_split.val_episodes)
                train_specs.append((cache, horizon, train_idx))
                val_specs.append((cache, horizon, val_idx))
            train_arrays = ge2.pool_families_split_arrays(train_specs)
            val_arrays = ge2.pool_families_split_arrays(val_specs)

            ho_cache = family_cache[held_out]
            test_arrays = ge2.SplitArrays(
                x=ho_cache.x, y=ho_cache.y_by_horizon[horizon],
                n_examples=len(ho_cache.episode_idx),
            )
            gc.collect()

            # CONDITION A -- original, unmodified features
            result_a = ge2.train_and_get_model(train_arrays, HIST_GRADIENT_BOOSTING)
            row_common = {
                "held_out_family": held_out, "horizon": horizon,
                "n_train": train_arrays.n_examples, "n_val": val_arrays.n_examples,
                "n_test": test_arrays.n_examples,
            }
            if result_a is None:
                log(f"    CONDITION A: no model trained (unexpected for Protocol B) -- recording null row")
                rows = []
                for split_name in ["train", "val", "test"]:
                    rows.append({**row_common, "condition": "ORIGINAL", "split": split_name,
                                 **metrics_to_row(None)})
                    rows.append({**row_common, "condition": "RUNTIME_FAITHFUL", "split": split_name,
                                 **metrics_to_row(None)})
                write_rows(rows)
                continue
            model_a, train_metrics_a = result_a
            val_metrics_a = ge2.eval_model(model_a, val_arrays)
            test_metrics_a = ge2.eval_model(model_a, test_arrays)
            rows = [
                {**row_common, "condition": "ORIGINAL", "split": "train", **metrics_to_row(train_metrics_a)},
                {**row_common, "condition": "ORIGINAL", "split": "val", **metrics_to_row(val_metrics_a)},
                {**row_common, "condition": "ORIGINAL", "split": "test", **metrics_to_row(test_metrics_a)},
            ]
            del model_a
            gc.collect()

            # CONDITION B -- elapsed_since_fault forced to -1.0, nothing else changed
            train_b = runtime_faithful(train_arrays)
            val_b = runtime_faithful(val_arrays)
            test_b = runtime_faithful(test_arrays)
            result_b = ge2.train_and_get_model(train_b, HIST_GRADIENT_BOOSTING)
            if result_b is None:
                for split_name in ["train", "val", "test"]:
                    rows.append({**row_common, "condition": "RUNTIME_FAITHFUL", "split": split_name,
                                 **metrics_to_row(None)})
            else:
                model_b, train_metrics_b = result_b
                val_metrics_b = ge2.eval_model(model_b, val_b)
                test_metrics_b = ge2.eval_model(model_b, test_b)
                rows.append({**row_common, "condition": "RUNTIME_FAITHFUL", "split": "train", **metrics_to_row(train_metrics_b)})
                rows.append({**row_common, "condition": "RUNTIME_FAITHFUL", "split": "val", **metrics_to_row(val_metrics_b)})
                rows.append({**row_common, "condition": "RUNTIME_FAITHFUL", "split": "test", **metrics_to_row(test_metrics_b)})
                del model_b
            write_rows(rows)
            del train_b, val_b, test_b, train_arrays, val_arrays, test_arrays
            gc.collect()
            log(f"    done. rss={rss_mb()}")

    csv_file.close()
    elapsed = time.perf_counter() - t_start
    current, peak = tracemalloc.get_traced_memory()
    log(f"All 30 units complete in {elapsed:.1f}s. tracemalloc peak={peak/1e6:.1f}MB")
    log(f"detail CSV: {detail_path}")

    with detail_path.open("r", newline="", encoding="utf-8") as f:
        import csv as csv_mod2
        n_detail_rows = sum(1 for _ in csv_mod2.reader(f)) - 1

    meta_path = OUT_DIR / "run_metadata.json"
    meta_path.write_text(json.dumps({
        "elapsed_seconds": elapsed,
        "tracemalloc_peak_mb": peak / 1e6,
        "n_detail_rows": n_detail_rows,
        "feature_index_tested": ELAPSED_SINCE_FAULT_INDEX,
        "feature_name_tested": ge2.FEATURE_NAMES[ELAPSED_SINCE_FAULT_INDEX],
        "sentinel_value": RUNTIME_SENTINEL,
        "config_hash_of_production_script": ge2.config_hash(
            ge2.build_run_config(ge2.FAMILIES, ge2.HORIZONS, ge2.FULL_SPEC_EPISODES_PER_FAMILY, ge2.RANDOM_SEEDS)
        ),
    }, indent=2), encoding="utf-8")
    log(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
