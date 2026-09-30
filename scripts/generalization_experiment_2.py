#!/usr/bin/env python3
"""The full-specification generalization research experiment (Protocols A/B).

Extends scripts/generalization_experiment.py (which ran a reduced N=15/family
scale due to a measured ~170s/replay throughput bottleneck) to the literal
locked specification: N=200 episodes/family, both protocols, all five
horizons, paired statistics, decision latency, and Belady gap.

This script performs pure orchestration of existing, UNCHANGED NeuroPager
components -- no source module is modified. Two engineering additions exist
ONLY in this orchestration layer, not in any frozen baseline:
    - TimingPolicy: a pure delegating wrapper that times select_victim()
      calls without altering any eviction decision, used to measure mean
      decision latency per policy.
    - Post-hoc model selection: for each (family, horizon) both
      logistic_regression and hist_gradient_boosting are trained and
      evaluated on train/val/test (cheap), but only the model with the
      higher validation ROC-AUC (PR-AUC fallback when ROC-AUC is undefined)
      is replayed as "the" Learned Utility Policy inside a real
      MemoryManager (expensive). This is a validation-only decision -- the
      test set is never used to choose between models -- and roughly halves
      the number of expensive policy replays relative to running both.

Every replay uses neuropager.memory.in_memory_store.InMemoryPageStore
rather than DiskPageStore. scripts/microbench_disk_io.py measured
DiskPageStore.write/read at ~15-120ms/op (environment-dependent, at times
far worse) versus ~0.003ms/op for an in-memory dict -- filesystem I/O was
the dominant bottleneck driving the originally-measured ~165.5s/replay.
InMemoryPageStore implements the identical PageStore contract (including
JSON-round-tripping page content exactly like DiskPageStore does), proven
behaviorally identical in tests/unit/test_page_store_equivalence.py (same
trace events, faults, evictions, hit rate, Belady result) -- swapping it in
changes nothing about any policy's decisions or the resulting dataset, only
how evicted pages are stored between a write and a later read within the
same process run.

Supports --trial for the mandated fail-safe smoke test: one family, one
horizon, a small episode count, train/val/test, all policies, one Random
seed. Run this and inspect its output before launching the full run.

Usage:
    python scripts/generalization_experiment_2.py --trial
    python scripts/generalization_experiment_2.py
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from numpy.typing import NDArray
from scipy import stats as scipy_stats

import neuropager
from neuropager.core.memory_manager import MemoryManager
from neuropager.core.page_fault import PageFaultHandler
from neuropager.core.page_table import PageTable
from neuropager.core.working_memory import WorkingMemory
from neuropager.dataset.features import FEATURE_NAMES
from neuropager.dataset.generator import generate_dataset
from neuropager.dataset.schema import DatasetExample
from neuropager.experiment.metrics import ClassificationMetrics, compute_metrics
from neuropager.experiment.models import (
    HIST_GRADIENT_BOOSTING,
    LOGISTIC_REGRESSION,
    make_model,
    model_config,
)
from neuropager.experiment.policy import LearnedUtilityPolicy
from neuropager.experiment.preprocessing import build_feature_matrix
from neuropager.experiment.split import split_episodes
from neuropager.memory.in_memory_store import InMemoryPageStore
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
# LOCKED configuration (matches the approved pre-flight review exactly).
# =========================================================================
LENGTH = 2000
CAPACITY = 16
HORIZONS = [25, 50, 100, 200, 500]
MODEL_NAME_LIST = [LOGISTIC_REGRESSION, HIST_GRADIENT_BOOSTING]
RANDOM_SEEDS = [0, 1, 2, 3, 4]
TRAIN_SEED = 0
FULL_SPEC_EPISODES_PER_FAMILY = 200
CONFIDENCE_LEVEL = 0.95

CHECKPOINT_SCHEMA_VERSION = "2"
"""Bump this whenever the checkpoint file structure changes.

A checkpoint written under a different schema version is refused rather
than silently reinterpreted -- see load_checkpoint().
"""

FAMILY_PARAMS: dict[str, dict[str, Any]] = {
    "A_uniform_random": {"key_space": 64},
    "B_temporal_locality": {"key_space": 64, "locality": 0.3},
    "C_bursty": {"key_space": 64, "mean_burst_length": 15.0},
    "D_long_range_reuse": {"key_space": 200},
    "E_phase_changing": {"phase_length": 250, "keys_per_phase": 32},
    "F_non_stationary": {"key_space": 64, "drift_rate": 0.01},
}
"""Single source of truth for every family's non-length, non-seed parameters.

Consumed both by FAMILY_BUILDERS (to actually construct workload configs)
and by build_run_config() (to hash into the checkpoint config fingerprint),
so the two can never silently drift apart.
"""

FAMILY_BUILDERS: dict[str, Callable[[int], WorkloadConfig]] = {
    "A_uniform_random": lambda seed: UniformRandomConfig(
        length=LENGTH, seed=seed, **FAMILY_PARAMS["A_uniform_random"]
    ),
    "B_temporal_locality": lambda seed: TemporalLocalityConfig(
        length=LENGTH, seed=seed, **FAMILY_PARAMS["B_temporal_locality"]
    ),
    "C_bursty": lambda seed: BurstyConfig(length=LENGTH, seed=seed, **FAMILY_PARAMS["C_bursty"]),
    "D_long_range_reuse": lambda seed: LongRangeReuseConfig(
        length=LENGTH, seed=seed, **FAMILY_PARAMS["D_long_range_reuse"]
    ),
    "E_phase_changing": lambda seed: PhaseChangingConfig(
        length=LENGTH, seed=seed, **FAMILY_PARAMS["E_phase_changing"]
    ),
    "F_non_stationary": lambda seed: NonStationaryConfig(
        length=LENGTH, seed=seed, **FAMILY_PARAMS["F_non_stationary"]
    ),
}
FAMILIES = list(FAMILY_BUILDERS)

STRUCTURAL_NULL_CONDITIONS = {("D_long_range_reuse", h) for h in (25, 50, 100)}
REDUNDANT_HORIZON_PAIRS = {
    "D_long_range_reuse": (200, 500),
    "E_phase_changing": (200, 500),
}
IMBALANCED_FAMILIES = {"B_temporal_locality", "C_bursty"}


def interpretation_flags(family: str, horizon: int) -> list[str]:
    """Return the pre-registered interpretation-rule flags for one (family, horizon)."""
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


def log(message: str) -> None:
    """Print a timestamped progress line (flushed immediately)."""
    print(f"[{datetime.now(UTC).isoformat()}] {message}", flush=True)


# =========================================================================
# Interruption-safe checkpointing: atomic writes, and validated,
# fail-closed loading that refuses to silently resume from a mismatched or
# corrupt checkpoint. None of this changes any scientific computation --
# it only changes how already-computed results are persisted and reused.
# =========================================================================


class ConfigMismatchError(RuntimeError):
    """Raised when a checkpoint's config_hash or identifying fields don't match the current run."""


class CorruptCheckpointError(RuntimeError):
    """Raised when a checkpoint file exists but is not valid JSON or has an invalid schema."""


def build_run_config(
    families: list[str],
    horizons: list[int],
    n_episodes_per_family: int,
    random_seeds: list[int],
) -> dict[str, Any]:
    """Build the canonical, hashable descriptor of every locked scientific parameter.

    Every value here is a scientific-methodology parameter (workload
    definitions/parameters, episode count, horizons, seeds, model names,
    interpretation rules, feature order) -- nothing about execution
    mechanics (checkpoint paths, timing, hardware). Two runs with the same
    build_run_config() output are, by construction, scientifically
    identical runs.
    """
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "episode_length": LENGTH,
        "working_memory_capacity": CAPACITY,
        "horizons": sorted(horizons),
        "random_seeds": sorted(random_seeds),
        "train_seed": TRAIN_SEED,
        "episodes_per_family": n_episodes_per_family,
        "families": sorted(families),
        "family_params": {name: FAMILY_PARAMS[name] for name in sorted(families)},
        "model_names": sorted(MODEL_NAME_LIST),
        "structural_null_conditions": sorted([list(pair) for pair in STRUCTURAL_NULL_CONDITIONS]),
        "redundant_horizon_pairs": {k: list(v) for k, v in sorted(REDUNDANT_HORIZON_PAIRS.items())},
        "imbalanced_families": sorted(IMBALANCED_FAMILIES),
        "feature_order": list(FEATURE_NAMES),
    }


def config_hash(run_config: dict[str, Any]) -> str:
    """Deterministic sha256 hash of a run config dict (stable key order)."""
    canonical = json.dumps(run_config, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_json(data: Any, path: Path) -> None:
    """Write ``data`` to ``path`` atomically: write to a temp file, then rename.

    A process death mid-write can never leave ``path`` truncated or
    corrupt -- ``os.replace`` is an atomic rename on both POSIX and
    Windows when the temp file is on the same filesystem (guaranteed here
    since it is written into ``path``'s own parent directory), so ``path``
    always contains either the previous complete content or the new
    complete content, never a partial write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp_path, path)


def load_checkpoint(
    path: Path, expected_config_hash: str, **expected_fields: Any
) -> dict[str, Any] | None:
    """Load and validate a checkpoint file, or return None if it does not exist.

    Never silently accepts a checkpoint that doesn't match the current
    configuration or identifying fields -- raises instead, so a stale or
    mismatched checkpoint can never be silently resumed from or silently
    merged with fresh results.

    Args:
        path: Checkpoint file to load.
        expected_config_hash: The current run's config hash; must match
            the checkpoint's stored hash exactly.
        **expected_fields: Additional identifying fields (e.g.
            family="A_uniform_random", horizon=50) that must match the
            checkpoint's stored values.

    Returns:
        The loaded checkpoint dict, or None if the file does not exist
        (the normal, expected case for a unit not yet computed).

    Raises:
        CorruptCheckpointError: If the file exists but is not valid JSON,
            is missing required schema fields, or its schema_version does
            not match CHECKPOINT_SCHEMA_VERSION.
        ConfigMismatchError: If the checkpoint's config_hash or any
            expected_fields value does not match.
    """
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CorruptCheckpointError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(data, dict) or "schema_version" not in data or "config_hash" not in data:
        raise CorruptCheckpointError(f"{path} is missing required checkpoint schema fields")
    if data["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise CorruptCheckpointError(
            f"{path} has schema_version {data['schema_version']!r}, expected "
            f"{CHECKPOINT_SCHEMA_VERSION!r} -- refusing to resume from an incompatible "
            "checkpoint format"
        )
    if data["config_hash"] != expected_config_hash:
        raise ConfigMismatchError(
            f"{path} was written under a different experiment configuration "
            f"(config_hash {data['config_hash']!r} != current {expected_config_hash!r}) -- "
            "refusing to silently resume; delete stale checkpoints or restore the matching "
            "configuration"
        )
    for key, expected in expected_fields.items():
        if data.get(key) != expected:
            raise ConfigMismatchError(
                f"{path} field {key!r}={data.get(key)!r} does not match expected {expected!r}"
            )
    return data


# =========================================================================
# Timing-instrumented policy wrapper (orchestration-layer only; delegates
# every decision unchanged, so it cannot alter any eviction outcome).
# =========================================================================


class TimingPolicy(PageReplacementPolicy):
    """Wraps a policy, timing every select_victim() call in wall-clock seconds."""

    def __init__(self, inner: PageReplacementPolicy) -> None:
        """Wrap ``inner``, initializing an empty decision-latency log."""
        self._inner = inner
        self.decision_times_s: list[float] = []

    def select_victim(self, page_table: PageTable) -> MemoryKey:
        """Time and delegate victim selection to the wrapped policy."""
        start = time.perf_counter()
        victim = self._inner.select_victim(page_table)
        self.decision_times_s.append(time.perf_counter() - start)
        return victim

    def on_access(self, key: MemoryKey) -> None:
        """Delegate access notification unchanged."""
        self._inner.on_access(key)

    def on_insert(self, key: MemoryKey) -> None:
        """Delegate insert notification unchanged."""
        self._inner.on_insert(key)

    def on_evict(self, key: MemoryKey) -> None:
        """Delegate eviction notification unchanged."""
        self._inner.on_evict(key)


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


def build_episode(family: str, seed: int) -> tuple[Episode, list[float]]:
    """Generate one episode's workload and replay it once under (timed) LRU.

    Returns:
        A tuple of (the generated Episode, LRU decision latencies in seconds).
    """
    config: WorkloadConfig = FAMILY_BUILDERS[family](seed)
    workload = generate_workload(config)
    episode_id = f"{family}-ep-{seed}"

    events, latencies = replay_policy(workload, LRUPolicy(), episode_id)
    return (
        Episode(
            episode_id=episode_id,
            family=family,
            seed=seed,
            workload=workload,
            events=events,
        ),
        latencies,
    )


def replay_policy(
    workload: list[MemoryKey],
    policy: PageReplacementPolicy,
    episode_id: str,
) -> tuple[list[TraceEvent], list[float]]:
    """Replay a workload through a real MemoryManager under one (timed) policy.

    Uses :class:`~neuropager.memory.in_memory_store.InMemoryPageStore` rather
    than :class:`~neuropager.memory.disk_store.DiskPageStore`: measured at
    ~165s/replay, filesystem I/O was the dominant bottleneck (see
    scripts/microbench_disk_io.py), and the two backends are proven
    behaviorally identical (see tests/unit/test_page_store_equivalence.py) --
    swapping them changes nothing PageFaultHandler, any policy, or the
    resulting trace can observe.

    Returns:
        A tuple of (trace events, per-decision latencies in seconds).
    """
    timed = TimingPolicy(policy)
    trace_logger = TraceLogger(episode_id)  # no path: in-memory only
    working_memory = WorkingMemory(capacity=CAPACITY)
    page_table = PageTable()
    fault_handler = PageFaultHandler(
        working_memory=working_memory,
        page_table=page_table,
        disk_store=InMemoryPageStore(),
        policy=timed,
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
    return trace_logger.events, timed.decision_times_s


@dataclass(frozen=True)
class PolicyResult:
    """Fault/hit/latency summary for one policy on one episode."""

    page_faults: int
    total_references: int
    hits: int
    hit_ratio: float
    decision_latency_mean_s: float | None
    decision_latency_std_s: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return this result as a plain, JSON-serializable dict."""
        return {
            "page_faults": self.page_faults,
            "total_references": self.total_references,
            "hits": self.hits,
            "hit_ratio": self.hit_ratio,
            "decision_latency_mean_s": self.decision_latency_mean_s,
            "decision_latency_std_s": self.decision_latency_std_s,
        }


def summarize_events(events: list[TraceEvent], latencies: list[float]) -> PolicyResult:
    """Summarize a trace's fault/hit counts plus decision-latency statistics."""
    access_events = [e for e in events if e.event_type == TraceEventType.ACCESS]
    faults = sum(1 for e in events if e.event_type == TraceEventType.PAGE_FAULT)
    total = len(access_events)
    hits = total - faults
    latency_mean = statistics.fmean(latencies) if latencies else None
    latency_std = statistics.pstdev(latencies) if len(latencies) > 1 else None
    return PolicyResult(
        page_faults=faults,
        total_references=total,
        hits=hits,
        hit_ratio=hits / total if total else float("nan"),
        decision_latency_mean_s=latency_mean,
        decision_latency_std_s=latency_std,
    )


def belady_gap(faults_policy: float, faults_belady: int) -> float:
    """(Faults_policy - Faults_MIN) / max(Faults_MIN, 1), never using Belady online."""
    return (faults_policy - faults_belady) / max(faults_belady, 1)


# =========================================================================
# Step 2: shared classical-baseline pool, computed once per episode, reused
# by lookup in both protocols.
# =========================================================================


def compute_classical_baselines(
    episode: Episode, lru_latencies: list[float], random_seeds: list[int]
) -> dict[str, PolicyResult]:
    """Compute FIFO/LRU/LFU/Belady/Random(xN) results for one episode.

    LRU is taken directly from the episode's own base trace (no
    re-simulation needed).
    """
    results: dict[str, PolicyResult] = {"LRU": summarize_events(episode.events, lru_latencies)}

    fifo_events, fifo_lat = replay_policy(episode.workload, FIFOPolicy(), episode.episode_id)
    results["FIFO"] = summarize_events(fifo_events, fifo_lat)

    lfu_events, lfu_lat = replay_policy(episode.workload, LFUPolicy(), episode.episode_id)
    results["LFU"] = summarize_events(lfu_events, lfu_lat)

    belady_events, belady_lat = replay_policy(
        episode.workload,
        BeladyMinPolicy(future=episode.workload),
        episode.episode_id,
    )
    results["Belady_MIN"] = summarize_events(belady_events, belady_lat)

    for seed in random_seeds:
        random_events, random_lat = replay_policy(
            episode.workload,
            RandomPolicy(seed=seed),
            episode.episode_id,
        )
        results[f"Random_seed{seed}"] = summarize_events(random_events, random_lat)

    return results


def run_classical_baselines_for_family(
    family: str,
    n_episodes: int,
    random_seeds: list[int],
    cfg_hash: str,
    checkpoint_dir: Path,
) -> tuple[list[Episode], dict[str, dict[str, PolicyResult]], int]:
    """Generate (or resume) one family's episodes and classical-baseline results.

    Every episode's workload + LRU replay is regenerated unconditionally --
    this is cheap (a single deterministic, seeded replay) and its output
    (Episode.workload/events) is needed in memory regardless, for dataset
    generation and later protocol replay, and is never itself persisted to
    the checkpoint. Only the expensive FIFO/LFU/Belady/Random(xN) replays
    are skipped when a valid, config-matching checkpoint entry already
    exists for that exact episode_id -- checkpointed atomically after every
    newly-computed episode, so an interruption loses at most one
    in-progress episode's classical-baseline work, never a whole family's.

    Args:
        family: Workload family name.
        n_episodes: Episodes to generate for this family (seeds 0..n-1).
        random_seeds: Random policy seeds to evaluate.
        cfg_hash: The current run's config hash (see build_run_config()).
        checkpoint_dir: Directory holding per-family checkpoint files.

    Returns:
        A tuple of (episodes, classical_results, n_freshly_computed) where
        n_freshly_computed counts episodes whose classical baselines were
        actually replayed this call (not loaded from checkpoint) -- used
        for accurate replay-count provenance in run metadata.

    Raises:
        CorruptCheckpointError: If a checkpoint exists but is invalid.
        ConfigMismatchError: If a checkpoint exists but was written under a
            different configuration.
    """
    checkpoint_path = checkpoint_dir / f"classical_{family}.json"
    existing = load_checkpoint(checkpoint_path, cfg_hash, family=family)
    completed: dict[str, dict[str, Any]] = dict(existing["episodes"]) if existing else {}

    episodes: list[Episode] = []
    classical_results: dict[str, dict[str, PolicyResult]] = {}
    n_freshly_computed = 0

    for seed in range(n_episodes):
        episode, lru_latencies = build_episode(family, seed)
        episodes.append(episode)

        if episode.episode_id in completed:
            classical_results[episode.episode_id] = {
                name: PolicyResult(**info) for name, info in completed[episode.episode_id].items()
            }
            continue

        classical_results[episode.episode_id] = compute_classical_baselines(
            episode, lru_latencies, random_seeds
        )
        n_freshly_computed += 1
        completed[episode.episode_id] = {
            name: r.to_dict() for name, r in classical_results[episode.episode_id].items()
        }
        write_json(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "config_hash": cfg_hash,
                "family": family,
                "n_episodes_target": n_episodes,
                "episodes": completed,
            },
            checkpoint_path,
        )

    if n_freshly_computed:
        log(
            f"  family {family}: {n_episodes} episodes + classical baselines done "
            f"({n_freshly_computed} newly computed, {n_episodes - n_freshly_computed} resumed)"
        )
    else:
        log(f"  family {family}: {n_episodes} episodes RESUMED from checkpoint (all complete)")

    return episodes, classical_results, n_freshly_computed


def migrate_legacy_classical_checkpoint(
    experiment_root: Path, checkpoint_dir: Path, cfg_hash: str, n_episodes: int
) -> None:
    """One-time, best-effort migration of the pre-resume single-file checkpoint.

    Earlier versions of this script wrote one cumulative
    _checkpoint_classical_results.json for all families combined, with no
    resume support. If that file exists and a family's episodes are fully
    present in it, migrate that family into the new per-family, resumable
    checkpoint format (stamped with the current config_hash) -- preserving
    already-valid work from a prior run rather than silently discarding or
    recomputing it. Never overwrites an existing new-format checkpoint,
    and never migrates a family that is not fully complete in the legacy
    file (a partial legacy family is left alone; that family will simply
    be computed fresh, safely, under the new per-episode checkpointing).
    """
    legacy_path = experiment_root / "_checkpoint_classical_results.json"
    if not legacy_path.exists():
        return
    try:
        legacy: dict[str, Any] = json.loads(legacy_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log(f"  legacy checkpoint {legacy_path} is not valid JSON; skipping migration")
        return

    by_family: dict[str, dict[str, Any]] = {}
    for episode_id, policies in legacy.items():
        family = episode_id.rsplit("-ep-", 1)[0]
        by_family.setdefault(family, {})[episode_id] = policies

    for family, episode_map in by_family.items():
        expected_ids = {f"{family}-ep-{s}" for s in range(n_episodes)}
        if set(episode_map) != expected_ids:
            log(
                f"  legacy checkpoint for {family} is incomplete "
                f"({len(episode_map)}/{n_episodes}); not migrated"
            )
            continue
        new_path = checkpoint_dir / f"classical_{family}.json"
        if new_path.exists():
            continue  # a new-format checkpoint already exists; never overwrite it
        write_json(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "config_hash": cfg_hash,
                "family": family,
                "n_episodes_target": n_episodes,
                "episodes": episode_map,
            },
            new_path,
        )
        log(f"  migrated legacy classical checkpoint for {family} ({n_episodes} episodes)")


# =========================================================================
# Step 3: model training/evaluation and validation-only model selection.
# =========================================================================


@dataclass(frozen=True)
class SplitArrays:
    """A compact (X, y, count) view of one train/val/test split.

    Replaces holding a list[DatasetExample] for a split: everywhere this
    experiment used to pass around DatasetExample objects for a split, it
    now passes this instead -- the same feature values and labels
    (build_feature_matrix() is unchanged and still the only place that
    reads a DatasetExample's fields), just already reduced to numpy
    arrays so the DatasetExample objects themselves can be released.
    """

    x: NDArray[np.float64] | None
    y: NDArray[np.int64] | None
    n_examples: int


EPISODE_BATCH_SIZE = 2
"""Episodes processed per generate_dataset() call when building a family's cache.

Purely a memory/execution-mechanics parameter (like checkpoint_dir) -- NOT
a scientific parameter, so it is deliberately not part of
build_run_config()/config_hash: changing it changes nothing about any
generated feature, label, split, or model outcome, only how much
transient memory one call to build_family_cache() uses at a time. Bounds
the transient DatasetExample list generate_dataset() builds to at most
this many episodes' worth (across every horizon requested in that call)
rather than a whole family's 200 episodes at once.

Lowered from 10 to 2 after a real production MemoryError while caching
D_long_range_reuse: measured tracemalloc peak for one 10-episode/5-horizon
batch of that family was 585MB (see the incident report); at batch_size=2
the same measurement showed 127MB -- a ~4.6x reduction in the largest
single transient allocation this script makes, at negligible extra
call-count overhead (TraceReplay/group_by_episode already process one
episode at a time internally regardless of batch size -- see
build_family_cache()'s docstring). This alone would not have prevented
the crash (the dominant cause was retained Episode.events -- see
get_family_cache()); it is retained as additional safety margin against
memory pressure from concurrent processes on this shared machine.
"""


@dataclass(frozen=True)
class FamilyCache:
    """One family's compact, horizon-independent feature cache plus per-horizon labels.

    compute_features() (see neuropager.dataset.features) is a pure
    function of (access_ticks, fault_ticks, decision_tick) alone -- it
    never depends on horizon. Only compute_label() does. build_family_cache()
    exploits this: x/episode_idx are computed exactly ONCE per candidate
    regardless of how many horizons are cached here, instead of once per
    (family, horizon) as the original design did -- see that function's
    docstring for the measured cost of the redundancy this removes.

    Attributes:
        x: Compact horizon-independent feature matrix, shape (N, len(FEATURE_NAMES)).
        episode_idx: For each row, an index into episode_id_list identifying
            its source episode -- a compact int32 replacement for carrying
            full episode_id strings alongside every row.
        episode_id_list: This family's episode IDs, in the order build_episode()
            generated them (seed-ascending); episode_idx values index into this.
        y_by_horizon: horizon -> label vector, shape (N,), aligned row-for-row with x.
    """

    x: NDArray[np.float64]
    episode_idx: NDArray[np.int32]
    episode_id_list: tuple[str, ...]
    y_by_horizon: dict[int, NDArray[np.int64]]


def _dedup_horizon_stride(examples: list[DatasetExample], n_horizons: int) -> list[DatasetExample]:
    """Return one example per (episode, decision, candidate), dropping horizon-duplicates.

    generate_dataset() emits examples with horizon as the fastest-varying
    index (for each candidate, it loops over every horizon before moving
    to the next candidate -- see its source), and every example in one
    such block shares the exact same `features` object (compute_features()
    runs once per candidate, before the horizon loop, and every
    per-horizon DatasetExample in that block is built from that one
    object). Taking every n_horizons-th example, starting at 0, therefore
    recovers exactly one feature row per candidate: no loss, no
    duplication, same relative order as the candidates were produced in.
    """
    return examples[0::n_horizons]


def build_family_cache(
    family_episodes: list[Episode],
    horizons: list[int],
    batch_size: int = EPISODE_BATCH_SIZE,
) -> FamilyCache:
    """Build one family's compact cache covering every horizon in ``horizons``, at once.

    This replaces calling generate_dataset() once per (family, horizon):
    profiling one representative episode (A_uniform_random, seed=0)
    measured generate_dataset() over all 5 locked horizons at 60.91s
    total (23,280 examples/horizon), dominated by per-candidate
    TraceReplay/compute_features() work that is IDENTICAL regardless of
    how many horizons are requested in that same call -- calling it once
    per horizon instead (the original memory-bounded design) redoes that
    dominant cost up to 5x for no benefit, since only the cheap
    compute_label() step actually varies by horizon. Requesting every
    still-needed horizon in one call removes that redundancy entirely.

    ``family_episodes`` is processed in contiguous batches of
    ``batch_size`` (not all 200 at once): a single generate_dataset() call
    across a whole family and every horizon would transiently materialize
    ~200 x 23,280 x len(horizons) DatasetExample objects -- for all 5
    horizons at once, tens of millions of objects, too large for this
    machine's ~8.4GB RAM. Batching by episode count does not change any
    output (TraceReplay/group_by_episode already process one episode at a
    time internally regardless of how many episodes are handed to one
    generate_dataset() call -- see neuropager.dataset.replay), only how
    many episodes' worth of DatasetExample objects are alive in Python
    memory at once; each batch's objects are converted to compact arrays
    and discarded before the next batch is generated. Concatenating
    batches in order reproduces exactly the row order a single whole-family
    call would have produced, since batches are contiguous, non-overlapping
    ranges of the same seed-ordered episode list and no candidate ever
    spans two episodes -- proven in tests/unit/test_experiment_memory_bounded.py.
    """
    sorted_horizons = sorted(horizons)
    n_horizons = len(sorted_horizons)
    episode_id_list = tuple(episode.episode_id for episode in family_episodes)
    episode_id_to_idx = {eid: i for i, eid in enumerate(episode_id_list)}

    x_chunks: list[NDArray[np.float64]] = []
    idx_chunks: list[NDArray[np.int32]] = []
    y_chunks: dict[int, list[NDArray[np.int64]]] = {h: [] for h in sorted_horizons}

    for batch_start in range(0, len(family_episodes), batch_size):
        batch = family_episodes[batch_start : batch_start + batch_size]
        batch_events = [event for episode in batch for event in episode.events]
        examples = generate_dataset(batch_events, horizons=sorted_horizons)
        if not examples:
            continue

        dedup = _dedup_horizon_stride(examples, n_horizons)
        x_batch, _, _ = build_feature_matrix(dedup)
        idx_batch = np.array([episode_id_to_idx[e.episode_id] for e in dedup], dtype=np.int32)
        x_chunks.append(x_batch)
        idx_chunks.append(idx_batch)

        for offset, horizon in enumerate(sorted_horizons):
            horizon_examples = examples[offset::n_horizons]
            y_chunks[horizon].append(np.array([e.label for e in horizon_examples], dtype=np.int64))

    x = np.concatenate(x_chunks) if x_chunks else np.empty((0, len(FEATURE_NAMES)))
    episode_idx = np.concatenate(idx_chunks) if idx_chunks else np.empty((0,), dtype=np.int32)
    y_by_horizon = {
        h: (np.concatenate(chunks) if chunks else np.empty((0,), dtype=np.int64))
        for h, chunks in y_chunks.items()
    }
    return FamilyCache(
        x=x, episode_idx=episode_idx, episode_id_list=episode_id_list, y_by_horizon=y_by_horizon
    )


def episode_ids_to_indices(
    episode_id_list: tuple[str, ...], wanted_episode_ids: Sequence[str]
) -> list[int]:
    """Translate a set of wanted episode IDs into local indices into ``episode_id_list``."""
    wanted = set(wanted_episode_ids)
    return [i for i, eid in enumerate(episode_id_list) if eid in wanted]


def select_split_arrays(
    x: NDArray[np.float64],
    y: NDArray[np.int64],
    episode_idx: NDArray[np.int32],
    wanted_indices: Sequence[int],
) -> SplitArrays:
    """Filter cached (X, y, episode_idx) arrays down to one split by episode-index membership.

    Equivalent to filter_examples_by_episodes() followed by
    build_feature_matrix(), but operating on already-compact arrays
    instead of DatasetExample objects, and on compact integer episode
    indices (see episode_ids_to_indices()) rather than full episode_id
    strings.
    """
    if not wanted_indices:
        return SplitArrays(x=None, y=None, n_examples=0)
    mask = np.isin(episode_idx, list(wanted_indices))
    n = int(mask.sum())
    if n == 0:
        return SplitArrays(x=None, y=None, n_examples=0)
    return SplitArrays(x=x[mask], y=y[mask], n_examples=n)


def pool_split_arrays(parts: list[SplitArrays]) -> SplitArrays:
    """Concatenate several families' SplitArrays for the same split into one.

    Only ever called on already-compact numpy arrays (never on
    DatasetExample lists), and only over the small number of parts being
    pooled (at most 5 families for Protocol B) -- a single bounded
    np.concatenate, not a repeated/quadratic copy.
    """
    non_empty = [p for p in parts if p.x is not None and p.y is not None]
    if not non_empty:
        return SplitArrays(x=None, y=None, n_examples=0)
    return SplitArrays(
        x=np.concatenate([p.x for p in non_empty]),
        y=np.concatenate([p.y for p in non_empty]),
        n_examples=sum(p.n_examples for p in non_empty),
    )


def pool_families_split_arrays(
    caches_and_indices: Sequence[tuple[FamilyCache, int, Sequence[int]]],
) -> SplitArrays:
    """Pool several families' selected rows into one array, one family's copy at a time.

    Scientifically equivalent to
    ``pool_split_arrays([select_split_arrays(cache.x, cache.y_by_horizon[horizon],
    cache.episode_idx, indices) for cache, horizon, indices in caches_and_indices])``
    -- same row values, same row order (families processed in the given
    order, exactly as select_split_arrays()+pool_split_arrays() would) --
    but avoids that combination's peak memory: select_split_arrays() copies
    each family's masked rows immediately, so pool_split_arrays() previously
    received up to 5 already-materialized full-size copies simultaneously
    (one per Protocol B training-pool family), then concatenated them into a
    6th, equally large array while all 5 inputs were still alive. Measured in
    production: this contributed to a genuine near-OOM stall (~7GB committed,
    system down to ~180MB free) during a held-out fold's training-pool
    construction -- a peak build_family_cache()'s own memory-bounded design
    never touched, since it only bounds dataset *generation*, not this
    downstream *pooling* step.

    This instead computes every family's boolean mask and row count first
    (cheap -- a mask is the same size as episode_idx, not the much larger
    feature matrix), allocates the pooled output array exactly once from the
    total count, then fills it family by family: each family's masked copy
    exists only transiently, inside the assignment expression that writes it
    into the pre-allocated output slice, and is released before the next
    family's mask is applied.
    """
    mask_and_count: list[tuple[NDArray[np.bool_], int] | None] = []
    total_n = 0
    n_features: int | None = None
    x_dtype: np.dtype[Any] | None = None
    y_dtype: np.dtype[Any] | None = None
    for cache, horizon, wanted_indices in caches_and_indices:
        if not wanted_indices:
            mask_and_count.append(None)
            continue
        mask = np.isin(cache.episode_idx, list(wanted_indices))
        n = int(mask.sum())
        if n == 0:
            mask_and_count.append(None)
            continue
        mask_and_count.append((mask, n))
        total_n += n
        if n_features is None:
            n_features = cache.x.shape[1]
            x_dtype = cache.x.dtype
            y_dtype = cache.y_by_horizon[horizon].dtype

    if total_n == 0 or n_features is None:
        return SplitArrays(x=None, y=None, n_examples=0)

    x = np.empty((total_n, n_features), dtype=x_dtype)
    y = np.empty(total_n, dtype=y_dtype)
    offset = 0
    for (cache, horizon, _), entry in zip(caches_and_indices, mask_and_count, strict=True):
        if entry is None:
            continue
        mask, n = entry
        x[offset : offset + n] = cache.x[mask]
        y[offset : offset + n] = cache.y_by_horizon[horizon][mask]
        offset += n
    return SplitArrays(x=x, y=y, n_examples=total_n)


def train_and_get_model(
    train: SplitArrays, model_name: str
) -> tuple[Any, ClassificationMetrics] | None:
    """Train one model on one horizon's training arrays.

    Returns:
        A tuple of (fitted model, training-set metrics), or None if
        ``train`` is single-class (a known, documented structural null --
        scikit-learn cannot fit on single-class data).
    """
    assert train.x is not None and train.y is not None
    if len(set(train.y.tolist())) < 2:
        return None
    # Reclaim memory left over from whatever ran immediately before this
    # (pooling's transient per-family masked copies, or a previous model's
    # fit()) before starting the next memory-intensive fit -- a pure
    # garbage-collection timing change, identical training data and model
    # either way.
    gc.collect()
    model = make_model(model_name, TRAIN_SEED)
    model.fit(train.x, train.y)
    proba = model.predict_proba(train.x)[:, 1]
    return model, compute_metrics(train.y, proba)


def eval_model(model: Any, split: SplitArrays) -> ClassificationMetrics | None:
    """Evaluate a fitted model on a split's arrays, or None if there are none."""
    if split.x is None or split.y is None:
        return None
    proba = model.predict_proba(split.x)[:, 1]
    return compute_metrics(split.y, proba)


def select_best_model(
    model_metrics: dict[str, dict[str, Any]],
) -> str | None:
    """Pick the model with the higher validation ROC-AUC (PR-AUC fallback).

    Uses ONLY validation metrics -- never the test set -- so this selection
    cannot leak test-set information into the choice of "the" Learned
    Utility Policy. Returns None if no model trained successfully (e.g. a
    structural-null horizon where training data is single-class).

    Args:
        model_metrics: model_name -> {"model": fitted model or None,
            "val_metrics": ClassificationMetrics or None, ...}.

    Returns:
        The name of the selected model, or None.
    """
    candidates = [
        name
        for name, info in model_metrics.items()
        if info.get("model") is not None and info.get("val_metrics") is not None
    ]
    if not candidates:
        return None

    def score(name: str) -> float:
        val: ClassificationMetrics = model_metrics[name]["val_metrics"]
        if val.roc_auc is not None:
            return float(val.roc_auc)
        if val.pr_auc is not None:
            return float(val.pr_auc)
        return -1.0

    return max(candidates, key=score)


def evaluate_learned_policy_on_episodes(
    model: Any,
    horizon: int,
    episodes: list[Episode],
) -> dict[str, PolicyResult]:
    """Replay LearnedUtilityPolicy(model, horizon) on a set of episodes."""
    results = {}
    for episode in episodes:
        policy = LearnedUtilityPolicy(model=model, horizon=horizon)
        events, latencies = replay_policy(episode.workload, policy, episode.episode_id)
        results[episode.episode_id] = summarize_events(events, latencies)
    return results


# =========================================================================
# Paired episode-level statistics (Learned vs. one comparison policy).
# =========================================================================


def paired_comparison(
    learned_faults: dict[str, int], other_faults: dict[str, int]
) -> dict[str, Any]:
    """Paired episode-level Learned-vs-other fault-count comparison.

    Computes the mean difference, standard deviation, a 95% confidence
    interval (Student-t), a paired t-test, and a Wilcoxon signed-rank test
    (as a non-parametric robustness check) over the episodes both dicts
    share.

    Args:
        learned_faults: episode_id -> page_faults for the learned policy.
        other_faults: episode_id -> page_faults for the comparison policy.

    Returns:
        A dict of the statistics above, or an "insufficient_data" flag if
        fewer than 2 shared episodes exist.
    """
    shared = sorted(set(learned_faults) & set(other_faults))
    if len(shared) < 2:
        return {"n_episodes": len(shared), "insufficient_data": True}

    diffs = np.array([learned_faults[e] - other_faults[e] for e in shared], dtype=np.float64)
    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    sem = std_diff / np.sqrt(n)
    t_crit = float(scipy_stats.t.ppf(1 - (1 - CONFIDENCE_LEVEL) / 2, df=n - 1))
    ci_low = mean_diff - t_crit * sem
    ci_high = mean_diff + t_crit * sem

    ttest_result = scipy_stats.ttest_rel(
        [learned_faults[e] for e in shared], [other_faults[e] for e in shared]
    )
    all_zero = bool(np.all(diffs == 0))
    wilcoxon_p: float | None
    if all_zero:
        wilcoxon_p = None
    else:
        wilcoxon_p = float(scipy_stats.wilcoxon(diffs).pvalue)

    return {
        "n_episodes": n,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "paired_ttest_statistic": float(ttest_result.statistic),
        "paired_ttest_pvalue": float(ttest_result.pvalue),
        "wilcoxon_pvalue": wilcoxon_p,
        "interpretation": (
            "negative mean_diff means the learned policy had FEWER faults "
            "(better) than the comparison policy"
        ),
    }


def faults_by_episode(results: dict[str, PolicyResult]) -> dict[str, int]:
    """Extract {episode_id: page_faults} from a policy-result dict."""
    return {eid: r.page_faults for eid, r in results.items()}


def random_mean_faults_by_episode(
    classical_results: dict[str, dict[str, PolicyResult]],
    episode_ids: list[str],
    random_seeds: list[int],
) -> dict[str, float]:
    """Mean page-fault count across the Random seeds, per episode."""
    out: dict[str, float] = {}
    for eid in episode_ids:
        seed_faults = [
            classical_results[eid][f"Random_seed{seed}"].page_faults for seed in random_seeds
        ]
        out[eid] = statistics.fmean(seed_faults)
    return out


def build_horizon_result(
    family_or_heldout: str,
    horizon: int,
    train: SplitArrays,
    val: SplitArrays,
    test: SplitArrays,
    test_episodes: list[Episode],
    classical_results: dict[str, dict[str, PolicyResult]],
    random_seeds: list[int],
) -> dict[str, Any]:
    """Train both models, select one by validation, replay it, and assemble one horizon's result."""
    model_info: dict[str, dict[str, Any]] = {}
    for model_name in MODEL_NAME_LIST:
        if train.x is None:
            model_info[model_name] = {
                "model": None,
                "train_metrics": None,
                "val_metrics": None,
                "test_metrics": None,
                "status": "no_training_examples",
            }
            continue
        trained = train_and_get_model(train, model_name)
        if trained is None:
            model_info[model_name] = {
                "model": None,
                "train_metrics": None,
                "val_metrics": None,
                "test_metrics": None,
                "status": "structural_null_single_class_training_data",
            }
            continue
        model, train_metrics = trained
        model_info[model_name] = {
            "model": model,
            "train_metrics": train_metrics,
            "val_metrics": eval_model(model, val),
            "test_metrics": eval_model(model, test),
            "status": "trained",
        }

    selected_name = select_best_model(model_info)

    test_episode_ids = [e.episode_id for e in test_episodes]
    classical_for_test = {eid: classical_results[eid] for eid in test_episode_ids}

    if selected_name is not None:
        selected_model = model_info[selected_name]["model"]
        learned_results = evaluate_learned_policy_on_episodes(
            selected_model, horizon, test_episodes
        )
    else:
        learned_results = {}

    learned_faults = faults_by_episode(learned_results)
    lru_faults = {eid: classical_for_test[eid]["LRU"].page_faults for eid in test_episode_ids}
    fifo_faults = {eid: classical_for_test[eid]["FIFO"].page_faults for eid in test_episode_ids}
    lfu_faults = {eid: classical_for_test[eid]["LFU"].page_faults for eid in test_episode_ids}
    belady_faults = {
        eid: classical_for_test[eid]["Belady_MIN"].page_faults for eid in test_episode_ids
    }
    random_mean_faults = random_mean_faults_by_episode(
        classical_results, test_episode_ids, random_seeds
    )

    fault_maps: dict[str, dict[str, float]] = {
        "Learned": {eid: float(v) for eid, v in learned_faults.items()},
        "LRU": {eid: float(v) for eid, v in lru_faults.items()},
        "FIFO": {eid: float(v) for eid, v in fifo_faults.items()},
        "LFU": {eid: float(v) for eid, v in lfu_faults.items()},
        "Random_mean": random_mean_faults,
    }
    belady_gaps = {
        policy_label: {eid: belady_gap(faults[eid], belady_faults[eid]) for eid in faults}
        for policy_label, faults in fault_maps.items()
        if faults
    }

    paired_comparisons = {
        "Learned_vs_LRU": paired_comparison(learned_faults, lru_faults),
        "Learned_vs_FIFO": paired_comparison(learned_faults, fifo_faults),
        "Learned_vs_LFU": paired_comparison(learned_faults, lfu_faults),
        "Learned_vs_Random": paired_comparison(
            learned_faults, {k: round(v) for k, v in random_mean_faults.items()}
        ),
        "Learned_vs_Belady_MIN": paired_comparison(learned_faults, belady_faults),
    }

    return {
        "flags": interpretation_flags(family_or_heldout, horizon),
        "n_train_examples": train.n_examples,
        "n_val_examples": val.n_examples,
        "n_test_examples": test.n_examples,
        "selected_model": selected_name,
        "models": {
            name: {
                "status": info["status"],
                "train_metrics": info["train_metrics"].to_dict() if info["train_metrics"] else None,
                "val_metrics": info["val_metrics"].to_dict() if info["val_metrics"] else None,
                "test_metrics": info["test_metrics"].to_dict() if info["test_metrics"] else None,
            }
            for name, info in model_info.items()
        },
        "learned_policy_faults": {eid: r.to_dict() for eid, r in learned_results.items()},
        "classical_policy_faults": {
            eid: {name: r.to_dict() for name, r in res.items()}
            for eid, res in classical_for_test.items()
        },
        "belady_gaps": belady_gaps,
        "paired_comparisons_vs_learned": paired_comparisons,
    }


def run_protocols_memory_bounded(
    episodes: dict[str, list[Episode]],
    classical_results: dict[str, dict[str, PolicyResult]],
    experiment_root: Path,
    horizons: list[int],
    random_seeds: list[int],
    cfg_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run Protocol A (within-distribution) and Protocol B (leave-one-family-out) together.

    Protocol A: one independent result per family (never pools the six
    families into one headline result -- each family's 70/15/15 split,
    training, and test evaluation are entirely independent).

    Protocol B: for each held-out family F, train/validate only on the
    other five families' episodes; test on ALL of F's episodes, never
    seen in training or validation.

    Both are resumable at (family-or-fold, horizon) granularity: each
    unit's result is checkpointed to its own file immediately after
    computing it, and reused (never recomputed) if a valid,
    config-matching checkpoint is found.

    Memory-bounded design: each family's FamilyCache (see build_family_cache())
    is built at most ONCE per run, covering every horizon that family is
    ever needed at (not once per horizon as an earlier version of this
    function did -- see build_family_cache()'s docstring for the ~5x
    redundant-computation cost that removed). Which horizons a family is
    "needed" at is determined with exactly the same per-horizon
    families_needed logic this function always used (a family is needed
    at horizon H if it has its own pending Protocol A unit at H, or is
    part of any pending Protocol B fold at H, as either the held-out
    family or a training-pool member) -- just unioned across every
    horizon up front, before any cache is built, rather than recomputed
    fresh inside a per-horizon loop. Caches are built lazily (only for
    families some pending unit actually needs) and released as soon as
    every horizon that needed them has been processed, so peak memory
    still never holds more than len(FAMILIES) families' compact arrays at
    once -- now amortized across every horizon a family is needed at,
    instead of rebuilt from scratch at every one of them.
    """
    checkpoint_dir = experiment_root / "checkpoints"
    families = list(episodes)
    run_protocol_b = len(families) >= 2

    family_splits = {
        family: split_episodes([e.episode_id for e in episodes[family]], seed=TRAIN_SEED)
        for family in families
    }
    fold_splits = {}
    if run_protocol_b:
        for held_out in families:
            training_families = [f for f in families if f != held_out]
            training_episode_ids = [e.episode_id for f in training_families for e in episodes[f]]
            fold_splits[held_out] = split_episodes(
                training_episode_ids,
                seed=TRAIN_SEED,
                train_fraction=0.82,
                val_fraction=0.18,
                test_fraction=0.0,
            )

    protocol_a_results: dict[str, Any] = {
        family: {
            "split": family_splits[family].to_dict(),
            "n_train": len(family_splits[family].train_episodes),
            "n_val": len(family_splits[family].val_episodes),
            "n_test": len(family_splits[family].test_episodes),
            "horizons": {},
        }
        for family in families
    }
    protocol_b_results: dict[str, Any]
    if run_protocol_b:
        protocol_b_results = {
            held_out: {
                "held_out_family": held_out,
                "n_train_episodes": len(fold_splits[held_out].train_episodes),
                "n_val_episodes": len(fold_splits[held_out].val_episodes),
                "n_test_episodes": len(episodes[held_out]),
                "horizons": {},
            }
            for held_out in families
        }
    else:
        protocol_b_results = {"skipped": "fewer than 2 families in scope"}

    # First pass: determine, per horizon, exactly which Protocol A/B units
    # are still pending (identical logic/order to the original per-horizon
    # design), loading already-checkpointed results immediately. This pass
    # touches only small JSON checkpoint files -- no dataset generation.
    pending_by_horizon: dict[int, tuple[list[str], list[str]]] = {}
    for horizon in horizons:
        a_pending = []
        for family in families:
            unit_path = checkpoint_dir / f"protocol_a_{family}_{horizon}.json"
            existing = load_checkpoint(
                unit_path, cfg_hash, protocol="A", family=family, horizon=horizon
            )
            if existing is not None:
                protocol_a_results[family]["horizons"][str(horizon)] = existing["result"]
            else:
                a_pending.append(family)

        b_pending = []
        if run_protocol_b:
            for held_out in families:
                unit_path = checkpoint_dir / f"protocol_b_{held_out}_{horizon}.json"
                existing = load_checkpoint(
                    unit_path, cfg_hash, protocol="B", held_out_family=held_out, horizon=horizon
                )
                if existing is not None:
                    protocol_b_results[held_out]["horizons"][str(horizon)] = existing["result"]
                else:
                    b_pending.append(held_out)

        pending_by_horizon[horizon] = (a_pending, b_pending)

    # Union, per family, every horizon at which ANY pending unit needs it
    # -- the same families_needed set the original per-horizon loop
    # computed, just accumulated across all horizons before building
    # anything, so build_family_cache() can be called once per family
    # with its full needed-horizon set instead of once per horizon.
    family_needed_horizons: dict[str, set[int]] = {family: set() for family in families}
    for horizon, (a_pending, b_pending) in pending_by_horizon.items():
        families_needed: set[str] = set(a_pending)
        for held_out in b_pending:
            families_needed.update(f for f in families if f != held_out)
            families_needed.add(held_out)
        for family in families_needed:
            family_needed_horizons[family].add(horizon)

    family_cache: dict[str, FamilyCache] = {}

    def get_family_cache(family: str) -> FamilyCache:
        if family not in family_cache:
            needed = sorted(family_needed_horizons[family])
            log(f"    building family cache: {family} @ H={needed}")
            family_cache[family] = build_family_cache(episodes[family], needed)
            log(f"    cached {family}: {len(family_cache[family].episode_idx)} examples/horizon")
            # episode.events (the raw TraceEvent list) has now served its only two
            # purposes -- compute_classical_baselines() (already done earlier, in
            # run_classical_baselines_for_family(), before this function is ever
            # called) and this build_family_cache() call, which just finished
            # reading every one of this family's episodes' .events exactly once.
            # Nothing downstream (classical/learned-policy replay, model
            # training, statistics) ever reads .events again for this family --
            # everywhere else uses episode.workload instead (see the grep audit
            # in the accompanying incident report). Releasing it here, rather
            # than holding all 1200 episodes' full trace-event lists resident
            # for the entire run, was the dominant, previously-unquantified
            # memory cost behind the D_long_range_reuse MemoryError: measured at
            # roughly 2-2.5GB across all six families' episodes, on top of the
            # already-built FamilyCache arrays for whichever families were
            # cached before this one.
            for episode in episodes[family]:
                episode.events = []
            # Force an immediate collection rather than relying on the
            # generational GC to get around to it: thousands of now-
            # unreferenced TraceEvent objects (and the batch-building
            # scaffolding build_family_cache() used internally) are exactly
            # the kind of memory a long-running process can leave
            # uncollected for a while, and the next step (another family's
            # cache, or model training) is memory-sensitive enough that
            # freeing this promptly matters. Purely a garbage-collection
            # timing change -- touches no scientific computation.
            gc.collect()
        return family_cache[family]

    processed_horizons: set[int] = set()
    for horizon in horizons:
        a_pending, b_pending = pending_by_horizon[horizon]
        if not a_pending and not b_pending:
            log(f"  H={horizon}: all Protocol A/B units already checkpointed, skipping")
            processed_horizons.add(horizon)
            continue

        log(
            f"  H={horizon}: {len(a_pending)} Protocol A unit(s) and {len(b_pending)} "
            f"Protocol B unit(s) pending"
        )

        for family in a_pending:
            split = family_splits[family]
            cache = get_family_cache(family)
            y = cache.y_by_horizon[horizon]
            train_idx = episode_ids_to_indices(cache.episode_id_list, split.train_episodes)
            val_idx = episode_ids_to_indices(cache.episode_id_list, split.val_episodes)
            test_idx = episode_ids_to_indices(cache.episode_id_list, split.test_episodes)
            train_arrays = select_split_arrays(cache.x, y, cache.episode_idx, train_idx)
            val_arrays = select_split_arrays(cache.x, y, cache.episode_idx, val_idx)
            test_arrays = select_split_arrays(cache.x, y, cache.episode_idx, test_idx)
            episode_by_id = {e.episode_id: e for e in episodes[family]}
            test_episode_objs = [episode_by_id[eid] for eid in split.test_episodes]

            horizon_result = build_horizon_result(
                family,
                horizon,
                train_arrays,
                val_arrays,
                test_arrays,
                test_episode_objs,
                classical_results,
                random_seeds,
            )
            protocol_a_results[family]["horizons"][str(horizon)] = horizon_result
            write_json(
                {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "config_hash": cfg_hash,
                    "protocol": "A",
                    "family": family,
                    "horizon": horizon,
                    "result": horizon_result,
                },
                checkpoint_dir / f"protocol_a_{family}_{horizon}.json",
            )
            log(f"    Protocol A: {family} @ H={horizon} done")

        for held_out in b_pending:
            fold_split = fold_splits[held_out]
            training_families = [f for f in families if f != held_out]

            train_specs = []
            val_specs = []
            for f in training_families:
                cache = get_family_cache(f)
                train_idx = episode_ids_to_indices(cache.episode_id_list, fold_split.train_episodes)
                val_idx = episode_ids_to_indices(cache.episode_id_list, fold_split.val_episodes)
                train_specs.append((cache, horizon, train_idx))
                val_specs.append((cache, horizon, val_idx))
            train_arrays = pool_families_split_arrays(train_specs)
            val_arrays = pool_families_split_arrays(val_specs)
            # Reclaim pooling's transient per-family masked-copy memory
            # (see pool_families_split_arrays()'s docstring) before the
            # memory-intensive model-training phase below -- pure
            # garbage-collection timing, no effect on the pooled arrays
            # themselves or any downstream computation.
            gc.collect()

            # Protocol B's test set is, by definition, ALL of the held-out
            # family's episodes (see this function's own docstring) -- every
            # row in ho_cache already belongs to the test set, so unlike the
            # pooling above there is no filtering to do, and no need to pay
            # for select_split_arrays()'s boolean-mask copy (which would
            # duplicate this family's entire, already-resident cache just to
            # select 100% of its own rows).
            ho_cache = get_family_cache(held_out)
            test_arrays = SplitArrays(
                x=ho_cache.x,
                y=ho_cache.y_by_horizon[horizon],
                n_examples=len(ho_cache.episode_idx),
            )
            test_episode_objs = list(episodes[held_out])

            horizon_result = build_horizon_result(
                held_out,
                horizon,
                train_arrays,
                val_arrays,
                test_arrays,
                test_episode_objs,
                classical_results,
                random_seeds,
            )
            protocol_b_results[held_out]["horizons"][str(horizon)] = horizon_result
            write_json(
                {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "config_hash": cfg_hash,
                    "protocol": "B",
                    "held_out_family": held_out,
                    "horizon": horizon,
                    "result": horizon_result,
                },
                checkpoint_dir / f"protocol_b_{held_out}_{horizon}.json",
            )
            log(f"    Protocol B: held_out={held_out} @ H={horizon} done")

        write_json(protocol_a_results, experiment_root / "_checkpoint_protocol_a_results.json")
        write_json(protocol_b_results, experiment_root / "_checkpoint_protocol_b_results.json")

        processed_horizons.add(horizon)
        for family in list(family_cache):
            if family_needed_horizons[family] <= processed_horizons:
                del family_cache[family]  # every horizon that needed this family is now done

    return protocol_a_results, protocol_b_results


def run_experiment(
    families: list[str],
    horizons: list[int],
    n_episodes_per_family: int,
    random_seeds: list[int],
    experiment_root: Path,
    experiment_id: str,
) -> dict[str, Any]:
    """Run the full generation + Protocol A + Protocol B pipeline for a given scope.

    Parameterized so the same code path serves both --trial (small scope,
    schema verification) and the full run (all families/horizons, N=200).

    Interruption-safe and resumable: every unit of work (a classical
    baseline episode, a Protocol A/B (family-or-fold, horizon) combo) is
    checkpointed atomically as soon as it completes, tagged with a hash of
    every locked scientific parameter (build_run_config()). Re-running this
    function with the SAME experiment_root and the SAME scientific
    configuration skips every already-completed unit and picks up exactly
    where a prior run left off. Re-running with a DIFFERENT configuration
    against an experiment_root that already has checkpoints raises
    ConfigMismatchError rather than silently mixing results from two
    different configurations.
    """
    start_time = time.perf_counter()
    experiment_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = experiment_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    replay_count = 0

    run_cfg = build_run_config(families, horizons, n_episodes_per_family, random_seeds)
    cfg_hash = config_hash(run_cfg)

    config_path = checkpoint_dir / "config.json"
    existing_config = load_checkpoint(config_path, cfg_hash)
    if existing_config is None:
        write_json(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "config_hash": cfg_hash,
                "config": run_cfg,
                "written_at": datetime.now(UTC).isoformat(),
            },
            config_path,
        )
        log(f"  wrote new config checkpoint (config_hash={cfg_hash[:12]}...)")
    else:
        log(f"  resuming existing run (config_hash={cfg_hash[:12]}... matches checkpoint)")

    migrate_legacy_classical_checkpoint(
        experiment_root, checkpoint_dir, cfg_hash, n_episodes_per_family
    )

    log(
        f"=== Starting {experiment_id}: {len(families)} families, "
        f"{n_episodes_per_family} episodes/family, horizons={horizons}, "
        f"random_seeds={random_seeds} ==="
    )

    episodes: dict[str, list[Episode]] = {}
    classical_results: dict[str, dict[str, PolicyResult]] = {}

    for family in families:
        family_episodes, family_classical, n_freshly_computed = run_classical_baselines_for_family(
            family, n_episodes_per_family, random_seeds, cfg_hash, checkpoint_dir
        )
        episodes[family] = family_episodes
        classical_results.update(family_classical)
        replay_count += n_episodes_per_family  # LRU replay always happens, resumed or not
        replay_count += n_freshly_computed * (3 + len(random_seeds))

    log(f"Total episodes: {sum(len(v) for v in episodes.values())}")

    # NOTE: dataset examples are intentionally NOT generated here for all
    # families/horizons at once (that materialized ~140M DatasetExample
    # objects for the full 1200-episode x 5-horizon production scope --
    # tens of GB, infeasible on this machine's 8.4GB RAM). See
    # run_protocols_memory_bounded()'s docstring for the memory-bounded,
    # horizon-outer replacement, which generates and immediately reduces
    # to compact numpy arrays one family at a time.
    protocol_a_results, protocol_b_results = run_protocols_memory_bounded(
        episodes, classical_results, experiment_root, horizons, random_seeds, cfg_hash
    )
    log("Protocol A complete")
    write_json(protocol_a_results, experiment_root / "protocol_a_results.json")
    if len(families) >= 2:
        log("Protocol B complete")
        write_json(protocol_b_results, experiment_root / "protocol_b_results.json")
    else:
        log("Protocol B skipped: fewer than 2 families in scope")

    elapsed = time.perf_counter() - start_time

    metadata = {
        "experiment_id": experiment_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "neuropager_version": neuropager.__version__,
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "feature_order": list(FEATURE_NAMES),
        "model_configs": {name: model_config(name, TRAIN_SEED) for name in MODEL_NAME_LIST},
        "config_hash": cfg_hash,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "locked_config": {
            "episode_length": LENGTH,
            "working_memory_capacity": CAPACITY,
            "horizons": horizons,
            "families": families,
        },
        "scale": {
            "episodes_per_family": n_episodes_per_family,
            "random_seeds": random_seeds,
            "measured_replay_count": replay_count,
            "measured_total_wall_clock_seconds": elapsed,
            "measured_seconds_per_replay": elapsed / replay_count if replay_count else None,
            "note": (
                "measured_replay_count/measured_seconds_per_replay reflect only THIS "
                "process invocation's own work -- replays skipped via resume from a prior "
                "invocation's checkpoints are not counted here."
            ),
        },
        "confidence_level": CONFIDENCE_LEVEL,
        "total_wall_clock_seconds": elapsed,
    }
    write_json(metadata, experiment_root / "metadata.json")

    log(
        f"=== {experiment_id} complete in {elapsed / 3600:.3f} hours. "
        f"Results in {experiment_root}/ ==="
    )
    return {
        "metadata": metadata,
        "protocol_a_results": protocol_a_results,
        "protocol_b_results": protocol_b_results,
    }


def main() -> None:
    """Parse CLI args and run either the fail-safe trial or the full experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trial",
        action="store_true",
        help="Run the mandated small fail-safe trial instead of the full experiment.",
    )
    args = parser.parse_args()

    if args.trial:
        run_experiment(
            families=["A_uniform_random", "B_temporal_locality"],
            horizons=[50],
            n_episodes_per_family=10,
            random_seeds=[0],
            experiment_root=Path("experiments/generalization-experiment-2-trial"),
            experiment_id="generalization-experiment-2-trial",
        )
    else:
        run_experiment(
            families=FAMILIES,
            horizons=HORIZONS,
            n_episodes_per_family=FULL_SPEC_EPISODES_PER_FAMILY,
            random_seeds=RANDOM_SEEDS,
            experiment_root=Path("experiments/generalization-experiment-2"),
            experiment_id="generalization-experiment-2",
        )


if __name__ == "__main__":
    main()
