#!/usr/bin/env python3
"""The generalization research experiment: Protocols A and B.

Protocol A (within-distribution) and Protocol B (leave-one-family-out)
over the six LOCKED workload families.

This script performs pure orchestration of existing, UNCHANGED
NeuroPager components:
    - neuropager.workloads: generator/config (LOCKED, untouched)
    - neuropager.core / neuropager.trace: memory system (untouched)
    - neuropager.policies: FIFO/LRU/LFU/Random/Belady (untouched)
    - neuropager.dataset: feature/label extraction (untouched)
    - neuropager.experiment: split/models/metrics/policy/runner (untouched)

No source module is modified by this script. If a correctness bug is
found while running it, this script will stop and print a clear error
rather than silently working around it.

Scale note: due to a measured ~170s/replay throughput bottleneck in
DiskPageStore's per-eviction filesystem writes in this environment (see
the reproducibility metadata for the calibration), this run uses
N_EPISODES_PER_FAMILY=15 rather than the originally-specified 200 --
flagged explicitly in the output as a reduced-scale run, not the full
specification.

This script collects raw, per-episode results only -- deliberately
decoupled from statistical aggregation (mean/std/95% bootstrap CI, paired
comparisons), which is computed afterward by the fast, separate
scripts/generalization_analysis.py from the saved JSON. Given the
multi-day runtime of the data-collection phase, any bug found in the
aggregation logic should never require re-running the replays.

Usage:
    python scripts/generalization_experiment.py
"""

from __future__ import annotations

import json
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sklearn

import neuropager
from neuropager.core.memory_manager import MemoryManager
from neuropager.core.page_fault import PageFaultHandler
from neuropager.core.page_table import PageTable
from neuropager.core.working_memory import WorkingMemory
from neuropager.dataset.generator import generate_dataset
from neuropager.dataset.schema import DatasetExample
from neuropager.experiment.metrics import ClassificationMetrics, compute_metrics
from neuropager.experiment.models import HIST_GRADIENT_BOOSTING, LOGISTIC_REGRESSION, make_model
from neuropager.experiment.policy import LearnedUtilityPolicy
from neuropager.experiment.preprocessing import build_feature_matrix
from neuropager.experiment.split import filter_examples_by_episodes, split_episodes
from neuropager.memory.disk_store import DiskPageStore
from neuropager.policies.base import PageReplacementPolicy
from neuropager.policies.belady_min import BeladyMinPolicy
from neuropager.policies.fifo import FIFOPolicy
from neuropager.policies.lfu import LFUPolicy
from neuropager.policies.lru import LRUPolicy
from neuropager.policies.random_policy import RandomPolicy
from neuropager.trace.events import TraceEvent, TraceEventType
from neuropager.trace.logger import TraceLogger
from neuropager.utils.types import MemoryKey
from neuropager.workloads.config import (
    BurstyConfig,
    LongRangeReuseConfig,
    NonStationaryConfig,
    PhaseChangingConfig,
    TemporalLocalityConfig,
    UniformRandomConfig,
    WorkloadConfig,
)
from neuropager.workloads.generator import generate_workload

# =========================================================================
# LOCKED configuration (do not modify -- matches the approved pre-flight
# review exactly).
# =========================================================================
LENGTH = 2000
CAPACITY = 16
HORIZONS = [25, 50, 100, 200, 500]
MODEL_NAMES = [LOGISTIC_REGRESSION, HIST_GRADIENT_BOOSTING]
RANDOM_SEEDS = [0, 1, 2, 3, 4]
TRAIN_SEED = 0
N_BOOTSTRAP = 2000

# Reduced-scale run: 15 episodes/family instead of the specified 200, per
# the throughput calibration documented above and in the run metadata.
N_EPISODES_PER_FAMILY = 15
FULL_SPEC_EPISODES_PER_FAMILY = 200

FAMILY_BUILDERS: dict[str, Callable[[int], WorkloadConfig]] = {
    "A_uniform_random": lambda seed: UniformRandomConfig(length=LENGTH, key_space=64, seed=seed),
    "B_temporal_locality": lambda seed: TemporalLocalityConfig(
        length=LENGTH, key_space=64, seed=seed, locality=0.3
    ),
    "C_bursty": lambda seed: BurstyConfig(
        length=LENGTH, key_space=64, seed=seed, mean_burst_length=15.0
    ),
    "D_long_range_reuse": lambda seed: LongRangeReuseConfig(
        length=LENGTH, key_space=200, seed=seed
    ),
    "E_phase_changing": lambda seed: PhaseChangingConfig(
        length=LENGTH, seed=seed, phase_length=250, keys_per_phase=32
    ),
    "F_non_stationary": lambda seed: NonStationaryConfig(
        length=LENGTH, key_space=64, seed=seed, drift_rate=0.01
    ),
}
FAMILIES = list(FAMILY_BUILDERS)

EXPERIMENT_ROOT = Path("experiments/generalization-experiment-1")


def log(message: str) -> None:
    """Print a timestamped progress line (flushed immediately)."""
    print(f"[{datetime.now(UTC).isoformat()}] {message}", flush=True)


# =========================================================================
# Step 1: episode generation (LRU-driven base trace; dual purpose: training
# data AND the LRU baseline policy result for that episode, computed once).
# =========================================================================


@dataclass
class Episode:
    """One generated episode: its identity, workload, and base (LRU) trace."""

    episode_id: str
    family: str
    seed: int
    workload: list[MemoryKey]
    events: list[TraceEvent]


def build_episode(family: str, seed: int, disk_root: Path) -> Episode:
    """Generate one episode's workload and replay it once under LRU.

    Args:
        family: Workload family name (key into FAMILY_BUILDERS).
        seed: Episode seed (also used as the workload generator's seed).
        disk_root: Isolated directory for this episode's disk store.

    Returns:
        The generated Episode, including its LRU-driven trace events.
    """
    config: WorkloadConfig = FAMILY_BUILDERS[family](seed)
    workload = generate_workload(config)
    episode_id = f"{family}-ep-{seed}"

    events = replay_policy(workload, LRUPolicy(), disk_root, episode_id)
    return Episode(
        episode_id=episode_id, family=family, seed=seed, workload=workload, events=events
    )


def replay_policy(
    workload: list[MemoryKey],
    policy: PageReplacementPolicy,
    disk_root: Path,
    episode_id: str,
) -> list[TraceEvent]:
    """Replay a workload through a real MemoryManager under one policy.

    Args:
        workload: The key sequence to replay.
        policy: The (unwrapped) policy under evaluation.
        disk_root: Isolated directory for this replay's disk store.
        episode_id: Trace episode identifier.

    Returns:
        The resulting trace events (in-memory only; no JSONL written).
    """
    trace_logger = TraceLogger(episode_id)  # no path: in-memory only
    working_memory = WorkingMemory(capacity=CAPACITY)
    page_table = PageTable()
    fault_handler = PageFaultHandler(
        working_memory=working_memory,
        page_table=page_table,
        disk_store=DiskPageStore(disk_root),
        policy=policy,
        trace_logger=trace_logger,
    )
    manager = MemoryManager(working_memory, page_table, fault_handler)

    seen: set[MemoryKey] = set()
    for key in workload:
        if key not in seen:
            manager.put(key, str(key))
            seen.add(key)
        else:
            manager.get(key)
    return trace_logger.events


@dataclass(frozen=True)
class PolicyResult:
    """Fault/hit/latency summary for one policy on one episode."""

    page_faults: int
    total_references: int
    hits: int
    hit_ratio: float

    def to_dict(self) -> dict[str, Any]:
        """Return this result as a plain, JSON-serializable dict."""
        return {
            "page_faults": self.page_faults,
            "total_references": self.total_references,
            "hits": self.hits,
            "hit_ratio": self.hit_ratio,
        }


def summarize_events(events: list[TraceEvent]) -> PolicyResult:
    """Summarize a trace's fault/hit counts.

    Args:
        events: Trace events from one replay.

    Returns:
        The resulting PolicyResult.
    """
    access_events = [e for e in events if e.event_type == TraceEventType.ACCESS]
    faults = sum(1 for e in events if e.event_type == TraceEventType.PAGE_FAULT)
    total = len(access_events)
    hits = total - faults
    return PolicyResult(
        page_faults=faults,
        total_references=total,
        hits=hits,
        hit_ratio=hits / total if total else float("nan"),
    )


# =========================================================================
# Step 2: shared classical-baseline pool, computed once per episode, reused
# by lookup in both protocols.
# =========================================================================


def compute_classical_baselines(episode: Episode, disk_root: Path) -> dict[str, PolicyResult]:
    """Compute FIFO/LRU/LFU/Belady/Random(x5) results for one episode.

    LRU is taken directly from the episode's own base trace (no
    re-simulation needed).

    Args:
        episode: The episode to evaluate (already carries its LRU trace).
        disk_root: Root directory for this episode's isolated replays.

    Returns:
        A dict keyed by policy label (e.g. "LRU", "FIFO", "Random_seed0").
    """
    results: dict[str, PolicyResult] = {"LRU": summarize_events(episode.events)}

    fifo_events = replay_policy(
        episode.workload, FIFOPolicy(), disk_root / "fifo", episode.episode_id
    )
    results["FIFO"] = summarize_events(fifo_events)

    lfu_events = replay_policy(episode.workload, LFUPolicy(), disk_root / "lfu", episode.episode_id)
    results["LFU"] = summarize_events(lfu_events)

    belady_events = replay_policy(
        episode.workload,
        BeladyMinPolicy(future=episode.workload),
        disk_root / "belady",
        episode.episode_id,
    )
    results["Belady_MIN"] = summarize_events(belady_events)

    for seed in RANDOM_SEEDS:
        random_events = replay_policy(
            episode.workload,
            RandomPolicy(seed=seed),
            disk_root / f"random{seed}",
            episode.episode_id,
        )
        results[f"Random_seed{seed}"] = summarize_events(random_events)

    return results


# =========================================================================
# Interpretation-rule metadata (documentation only -- does not suppress or
# alter any computed number).
# =========================================================================

STRUCTURAL_NULL_CONDITIONS = {("D_long_range_reuse", h) for h in (25, 50, 100)}
REDUNDANT_HORIZON_PAIRS = {
    "D_long_range_reuse": (200, 500),
    "E_phase_changing": (200, 500),
}
IMBALANCED_FAMILIES = {"B_temporal_locality", "C_bursty"}


def interpretation_flags(family: str, horizon: int) -> list[str]:
    """Return the interpretation-rule flags applicable to one (family, horizon).

    Args:
        family: Workload family name.
        horizon: Reuse horizon.

    Returns:
        A list of short flag strings (empty if none apply).
    """
    flags = []
    if (family, horizon) in STRUCTURAL_NULL_CONDITIONS:
        flags.append("STRUCTURAL_NULL: labels are one-class; ROC-AUC/PR-AUC not meaningful")
    redundant = REDUNDANT_HORIZON_PAIRS.get(family)
    if redundant and horizon in redundant:
        other_horizon = redundant[0] if horizon == redundant[1] else redundant[1]
        flags.append(f"REDUNDANT_HORIZON: near-duplicate of H={other_horizon}")
    if family in IMBALANCED_FAMILIES:
        flags.append(
            "CLASS_IMBALANCE: strong positive-class skew; do not rely on accuracy/F1 alone"
        )
    return flags


def main() -> None:
    """Run the full generalization experiment (Protocols A and B)."""
    start_time = time.perf_counter()
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)

    integrity_issues: list[str] = []

    log(
        f"=== Starting generalization experiment: {N_EPISODES_PER_FAMILY} episodes/family "
        f"(reduced from spec of {FULL_SPEC_EPISODES_PER_FAMILY}) ==="
    )

    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # --- Step 1+2: generate episodes and shared classical baselines ---
        episodes: dict[str, list[Episode]] = {f: [] for f in FAMILIES}
        classical_results: dict[str, dict[str, PolicyResult]] = {}
        all_events: list[TraceEvent] = []

        for family in FAMILIES:
            for seed in range(N_EPISODES_PER_FAMILY):
                episode_disk_root = tmp_path / family / f"ep{seed}"
                episode_disk_root.mkdir(parents=True, exist_ok=True)
                episode = build_episode(family, seed, episode_disk_root / "base")
                episodes[family].append(episode)
                all_events.extend(episode.events)

                classical_results[episode.episode_id] = compute_classical_baselines(
                    episode, episode_disk_root / "classical"
                )
            log(f"  family {family}: {N_EPISODES_PER_FAMILY} episodes + classical baselines done")
            # Checkpoint: if this process is interrupted, this family's
            # (expensive, replay-heavy) classical-baseline work is not lost.
            write_json(
                {
                    eid: {name: r.to_dict() for name, r in res.items()}
                    for eid, res in classical_results.items()
                },
                EXPERIMENT_ROOT / "_checkpoint_classical_results.json",
            )

        log(f"Total episodes: {sum(len(v) for v in episodes.values())}")

        # --- Step 3: dataset generation (unchanged neuropager.dataset) ---
        examples = generate_dataset(all_events, horizons=HORIZONS)
        log(f"Dataset examples generated: {len(examples)}")

        # --- Protocol A: within-distribution ---
        protocol_a_results = run_protocol_a(episodes, examples, classical_results, tmp_path)
        log("Protocol A complete")
        write_json(protocol_a_results, EXPERIMENT_ROOT / "_checkpoint_protocol_a_results.json")

        # --- Protocol B: leave-one-family-out ---
        protocol_b_results = run_protocol_b(episodes, examples, classical_results, tmp_path)
        log("Protocol B complete")

    elapsed = time.perf_counter() - start_time

    # --- Reproducibility metadata ---
    metadata = {
        "experiment_id": "generalization-experiment-1",
        "generated_at": datetime.now(UTC).isoformat(),
        "neuropager_version": neuropager.__version__,
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "locked_config": {
            "episode_length": LENGTH,
            "working_memory_capacity": CAPACITY,
            "horizons": HORIZONS,
            "families": {
                "A_uniform_random": {"key_space": 64},
                "B_temporal_locality": {"key_space": 64, "locality": 0.3},
                "C_bursty": {"key_space": 64, "mean_burst_length": 15.0},
                "D_long_range_reuse": {"key_space": 200},
                "E_phase_changing": {"phase_length": 250, "keys_per_phase": 32},
                "F_non_stationary": {"key_space": 64, "drift_rate": 0.01},
            },
        },
        "scale": {
            "episodes_per_family_used": N_EPISODES_PER_FAMILY,
            "episodes_per_family_specified": FULL_SPEC_EPISODES_PER_FAMILY,
            "reduced_scale_reason": (
                "Measured throughput calibration: ~170s/replay for a 2000-tick episode "
                "at capacity=16 (DiskPageStore per-eviction filesystem I/O dominates; "
                "core MemoryManager logic alone takes <0.5s/episode with no eviction "
                "pressure). Full spec (200 episodes/family, both protocols) would require "
                "~24,600 replays (~45+ days). Reduced to 15/family (~1,890 replays, "
                "~89 hours) with explicit user authorization."
            ),
        },
        "random_seeds": RANDOM_SEEDS,
        "train_seed": TRAIN_SEED,
        "n_bootstrap_resamples": N_BOOTSTRAP,
        "total_wall_clock_seconds": elapsed,
        "integrity_issues": integrity_issues,
    }
    write_json(metadata, EXPERIMENT_ROOT / "metadata.json")
    write_json(protocol_a_results, EXPERIMENT_ROOT / "protocol_a_results.json")
    write_json(protocol_b_results, EXPERIMENT_ROOT / "protocol_b_results.json")

    log(f"=== Experiment complete in {elapsed / 3600:.2f} hours. Results in {EXPERIMENT_ROOT}/ ===")


def write_json(data: Any, path: Path) -> None:
    """Write ``data`` to ``path`` as pretty-printed, deterministic-key JSON.

    Args:
        data: JSON-serializable data.
        path: Destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")


def train_and_get_model(
    train_examples: list[DatasetExample], horizon: int, model_name: str
) -> tuple[Any, ClassificationMetrics] | None:
    """Train one model on one horizon's training examples.

    Args:
        train_examples: Training examples for this horizon.
        horizon: The horizon being trained for.
        model_name: One of neuropager.experiment.models.MODEL_NAMES.

    Returns:
        A tuple of (fitted model, training-set metrics), or None if
        ``train_examples`` is single-class (e.g. D_long_range_reuse at
        H=25/50/100, a known, documented structural null -- scikit-learn's
        classifiers cannot fit on single-class data, and this is expected,
        not an error condition).
    """
    x_train, y_train, _ = build_feature_matrix(train_examples)
    if len(set(y_train.tolist())) < 2:
        return None
    model = make_model(model_name, TRAIN_SEED)
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_train)[:, 1]
    return model, compute_metrics(y_train, proba)


def eval_model(model: Any, split_examples: list[DatasetExample]) -> ClassificationMetrics | None:
    """Evaluate a fitted model on a set of examples.

    Args:
        model: A fitted scikit-learn pipeline.
        split_examples: Examples to evaluate on.

    Returns:
        The resulting ClassificationMetrics, or None if there are no
        examples to evaluate.
    """
    if not split_examples:
        return None
    x, y, _ = build_feature_matrix(split_examples)
    proba = model.predict_proba(x)[:, 1]
    return compute_metrics(y, proba)


def evaluate_learned_policy_on_episodes(
    model: Any,
    horizon: int,
    episodes: list[Episode],
    disk_root: Path,
) -> dict[str, PolicyResult]:
    """Replay LearnedUtilityPolicy(model, horizon) on a set of episodes.

    Args:
        model: A fitted scikit-learn pipeline.
        horizon: The horizon this model was trained for.
        episodes: Episodes to replay.
        disk_root: Root directory for isolated replays.

    Returns:
        A dict from episode_id to PolicyResult.
    """
    results = {}
    for episode in episodes:
        policy = LearnedUtilityPolicy(model=model, horizon=horizon)
        events = replay_policy(
            episode.workload, policy, disk_root / episode.episode_id, episode.episode_id
        )
        results[episode.episode_id] = summarize_events(events)
    return results


def run_protocol_a(
    episodes: dict[str, list[Episode]],
    examples: list[DatasetExample],
    classical_results: dict[str, dict[str, PolicyResult]],
    tmp_path: Path,
) -> dict[str, Any]:
    """Run Protocol A (within-distribution) for every family and horizon.

    Args:
        episodes: All generated episodes, keyed by family.
        examples: The full dataset (all families, all horizons).
        classical_results: Pre-computed classical policy results, keyed by
            episode_id.
        tmp_path: Root scratch directory.

    Returns:
        A nested dict of results, keyed by family then horizon.
    """
    results: dict[str, Any] = {}
    for family in FAMILIES:
        family_episode_ids = [e.episode_id for e in episodes[family]]
        split = split_episodes(family_episode_ids, seed=TRAIN_SEED)
        episode_by_id = {e.episode_id: e for e in episodes[family]}
        test_episodes = [episode_by_id[eid] for eid in split.test_episodes]

        family_results: dict[str, Any] = {
            "split": split.to_dict(),
            "n_train": len(split.train_episodes),
            "n_val": len(split.val_episodes),
            "n_test": len(split.test_episodes),
            "horizons": {},
        }

        for horizon in HORIZONS:
            horizon_examples = [
                e for e in examples if e.horizon == horizon and e.episode_id in family_episode_ids
            ]
            train_examples = filter_examples_by_episodes(horizon_examples, split.train_episodes)
            val_examples = filter_examples_by_episodes(horizon_examples, split.val_episodes)
            test_examples = filter_examples_by_episodes(horizon_examples, split.test_episodes)

            horizon_result: dict[str, Any] = {
                "flags": interpretation_flags(family, horizon),
                "n_train_examples": len(train_examples),
                "models": {},
            }

            for model_name in MODEL_NAMES:
                if not train_examples:
                    horizon_result["models"][model_name] = {"error": "no training examples"}
                    continue
                trained = train_and_get_model(train_examples, horizon, model_name)
                if trained is None:
                    horizon_result["models"][model_name] = {
                        "skipped": "structural_null_single_class_training_data"
                    }
                    continue
                model, train_metrics = trained
                val_metrics = eval_model(model, val_examples)
                test_metrics = eval_model(model, test_examples)

                policy_disk = tmp_path / "protoA" / family / str(horizon) / model_name
                learned_results = evaluate_learned_policy_on_episodes(
                    model, horizon, test_episodes, policy_disk
                )

                horizon_result["models"][model_name] = {
                    "train_metrics": train_metrics.to_dict(),
                    "val_metrics": val_metrics.to_dict() if val_metrics else None,
                    "test_metrics": test_metrics.to_dict() if test_metrics else None,
                    "policy_faults": {eid: r.to_dict() for eid, r in learned_results.items()},
                }

            # Attach classical baseline results for these same test episodes.
            horizon_result["classical_policy_faults"] = {
                eid: {name: r.to_dict() for name, r in classical_results[eid].items()}
                for eid in split.test_episodes
            }
            family_results["horizons"][str(horizon)] = horizon_result

        results[family] = family_results
        log(f"  Protocol A: {family} done")
        write_json(results, EXPERIMENT_ROOT / "_checkpoint_protocol_a_results.json")

    return results


def run_protocol_b(
    episodes: dict[str, list[Episode]],
    examples: list[DatasetExample],
    classical_results: dict[str, dict[str, PolicyResult]],
    tmp_path: Path,
) -> dict[str, Any]:
    """Run Protocol B (leave-one-family-out) for every held-out family and horizon.

    Args:
        episodes: All generated episodes, keyed by family.
        examples: The full dataset (all families, all horizons).
        classical_results: Pre-computed classical policy results, keyed by
            episode_id.
        tmp_path: Root scratch directory.

    Returns:
        A nested dict of results, keyed by held-out family then horizon.
    """
    results: dict[str, Any] = {}
    for held_out in FAMILIES:
        training_families = [f for f in FAMILIES if f != held_out]
        training_episode_ids = [e.episode_id for f in training_families for e in episodes[f]]
        test_episode_ids = [e.episode_id for e in episodes[held_out]]
        episode_by_id = {e.episode_id: e for f in FAMILIES for e in episodes[f]}
        test_episodes = [episode_by_id[eid] for eid in test_episode_ids]

        train_val_split = split_episodes(
            training_episode_ids,
            seed=TRAIN_SEED,
            train_fraction=0.82,
            val_fraction=0.18,
            test_fraction=0.0,
        )

        held_out_results: dict[str, Any] = {
            "held_out_family": held_out,
            "n_train_episodes": len(train_val_split.train_episodes),
            "n_val_episodes": len(train_val_split.val_episodes),
            "n_test_episodes": len(test_episode_ids),
            "horizons": {},
        }

        for horizon in HORIZONS:
            horizon_examples = [e for e in examples if e.horizon == horizon]
            train_examples = filter_examples_by_episodes(
                horizon_examples, train_val_split.train_episodes
            )
            val_examples = filter_examples_by_episodes(
                horizon_examples, train_val_split.val_episodes
            )
            test_examples = filter_examples_by_episodes(horizon_examples, test_episode_ids)

            horizon_result: dict[str, Any] = {
                "flags": interpretation_flags(held_out, horizon),
                "n_train_examples": len(train_examples),
                "models": {},
            }

            for model_name in MODEL_NAMES:
                if not train_examples:
                    horizon_result["models"][model_name] = {"error": "no training examples"}
                    continue
                trained = train_and_get_model(train_examples, horizon, model_name)
                if trained is None:
                    horizon_result["models"][model_name] = {
                        "skipped": "structural_null_single_class_training_data"
                    }
                    continue
                model, train_metrics = trained
                val_metrics = eval_model(model, val_examples)
                test_metrics = eval_model(model, test_examples)

                policy_disk = tmp_path / "protoB" / held_out / str(horizon) / model_name
                learned_results = evaluate_learned_policy_on_episodes(
                    model, horizon, test_episodes, policy_disk
                )

                horizon_result["models"][model_name] = {
                    "train_metrics": train_metrics.to_dict(),
                    "val_metrics": val_metrics.to_dict() if val_metrics else None,
                    "test_metrics": test_metrics.to_dict() if test_metrics else None,
                    "policy_faults": {eid: r.to_dict() for eid, r in learned_results.items()},
                }

            horizon_result["classical_policy_faults"] = {
                eid: {name: r.to_dict() for name, r in classical_results[eid].items()}
                for eid in test_episode_ids
            }
            held_out_results["horizons"][str(horizon)] = horizon_result

        results[held_out] = held_out_results
        log(f"  Protocol B: held_out={held_out} done")
        write_json(results, EXPERIMENT_ROOT / "_checkpoint_protocol_b_results.json")

    return results


if __name__ == "__main__":
    main()
