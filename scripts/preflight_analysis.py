#!/usr/bin/env python3
"""Pre-flight statistical characterization of the six benchmark workload families.

Runs the EXISTING, unmodified `neuropager.workloads` generator (multiple
episodes per family, deterministic seeds, length >= 2,000 ticks) and
computes the statistics the DeepSeek review requested: reuse-distance
distribution (mean/median/p90/p95/p99/histogram), access entropy, stack
distance, burstiness, unique-page ratio, sliding-window working-set size,
phase/non-stationarity drift, per-horizon label-positivity fraction, and
Belady MIN fault/hit rate.

This script adds no new statistics to `neuropager.workloads.statistics` --
every metric it needs beyond what that module already provides (reuse
percentiles, working-set size, coefficient of variation, drift, per-horizon
label fractions, Belady stats) is computed locally, so the benchmark
generator itself is untouched, per the review's explicit instruction.

Usage:
    python scripts/preflight_analysis.py
"""

from __future__ import annotations

import json
import math
from collections import Counter, deque
from pathlib import Path
from typing import Any

from neuropager.core.memory_manager import MemoryManager
from neuropager.core.page_fault import PageFaultHandler
from neuropager.core.page_table import PageTable
from neuropager.core.working_memory import WorkingMemory
from neuropager.memory.disk_store import DiskPageStore
from neuropager.policies.belady_min import BeladyMinPolicy
from neuropager.trace.events import TraceEventType
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
from neuropager.workloads.statistics import (
    access_entropy,
    burst_run_lengths,
    mean_stack_distance,
    unique_page_count,
)

LENGTH = 2000
CAPACITY = 16
HORIZONS = [25, 50, 100, 200, 500]
N_STAT_EPISODES = 10  # episodes for pure-sequence statistics (cheap)
N_BELADY_EPISODES = 4  # episodes for MemoryManager/Belady replay (I/O-bound)
WORKING_SET_WINDOWS = [16, 50, 200]

FAMILIES: dict[str, list[WorkloadConfig]] = {
    "A_uniform_random": [
        UniformRandomConfig(length=LENGTH, key_space=64, seed=seed)
        for seed in range(N_STAT_EPISODES)
    ],
    "B_temporal_locality": [
        TemporalLocalityConfig(length=LENGTH, key_space=64, seed=seed, locality=0.3)
        for seed in range(N_STAT_EPISODES)
    ],
    "C_bursty": [
        BurstyConfig(length=LENGTH, key_space=64, seed=seed, mean_burst_length=15.0)
        for seed in range(N_STAT_EPISODES)
    ],
    "D_long_range_reuse": [
        LongRangeReuseConfig(length=LENGTH, key_space=200, seed=seed)
        for seed in range(N_STAT_EPISODES)
    ],
    "E_phase_changing": [
        PhaseChangingConfig(length=LENGTH, seed=seed, phase_length=250, keys_per_phase=32)
        for seed in range(N_STAT_EPISODES)
    ],
    "F_non_stationary": [
        NonStationaryConfig(length=LENGTH, key_space=64, seed=seed, drift_rate=0.01)
        for seed in range(N_STAT_EPISODES)
    ],
}

HISTOGRAM_BINS = [0, 10, 25, 50, 100, 200, 500, 1000, 2000]


# --- local statistics helpers (do not touch neuropager.workloads) -----------


def tick_reuse_distances(workload: list[MemoryKey]) -> list[int | None]:
    """Ticks until each access's key next recurs; None if it never recurs again."""
    distances: list[int | None] = [None] * len(workload)
    last_seen: dict[MemoryKey, int] = {}
    for i in range(len(workload) - 1, -1, -1):
        key = workload[i]
        if key in last_seen:
            distances[i] = last_seen[key] - i
        last_seen[key] = i
    return distances


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile, matching numpy's default method."""
    if not values:
        return float("nan")
    data = sorted(values)
    k = (len(data) - 1) * (p / 100.0)
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return data[int(k)]
    return data[f] + (data[c] - data[f]) * (k - f)


def histogram(values: list[float], bins: list[int]) -> dict[str, int]:
    """Binned counts using the given bin edges (last bin catches the overflow)."""
    counts = {f"[{bins[i]},{bins[i + 1]})": 0 for i in range(len(bins) - 1)}
    counts[f">={bins[-1]}"] = 0
    for v in values:
        placed = False
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i + 1]:
                counts[f"[{bins[i]},{bins[i + 1]})"] += 1
                placed = True
                break
        if not placed:
            counts[f">={bins[-1]}"] += 1
    return counts


def coefficient_of_variation(values: list[float]) -> float:
    """Std / mean; nan if empty or mean is zero."""
    if not values:
        return float("nan")
    mean = sum(values) / len(values)
    if mean == 0:
        return float("nan")
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance) / mean


def sliding_window_working_set(workload: list[MemoryKey], window: int) -> list[int]:
    """Distinct-key count in the trailing `window` accesses, once the window has filled."""
    sizes: list[int] = []
    counts: Counter[MemoryKey] = Counter()
    dq: deque[MemoryKey] = deque()
    for key in workload:
        dq.append(key)
        counts[key] += 1
        if len(dq) > window:
            old = dq.popleft()
            counts[old] -= 1
            if counts[old] == 0:
                del counts[old]
        if len(dq) == window:
            sizes.append(len(counts))
    return sizes


def early_late_tv_distance(workload: list[MemoryKey]) -> float:
    """Total variation distance between the first-1/5 and last-1/5 empirical key distributions.

    Near 0 for a stationary workload; higher for genuine popularity drift.
    """
    n = len(workload)
    chunk = max(1, n // 5)
    early, late = Counter(workload[:chunk]), Counter(workload[-chunk:])
    n_early, n_late = sum(early.values()), sum(late.values())
    keys = set(early) | set(late)
    return 0.5 * sum(abs(early.get(k, 0) / n_early - late.get(k, 0) / n_late) for k in keys)


def label_positive_fraction(distances: list[int | None], horizon: int) -> float:
    """Fraction of accesses with a repeat within (t, t+horizon].

    Matches the real dataset pipeline's label positivity rate exactly
    (episode-end truncation counts as negative, as neuropager.dataset.labels does).
    """
    if not distances:
        return float("nan")
    positive = sum(1 for d in distances if d is not None and d <= horizon)
    return positive / len(distances)


def conditional_fraction_within_horizon(distances: list[int | None], horizon: int) -> float:
    """Of accesses that DO recur before episode end, fraction recurring within horizon."""
    reused = [d for d in distances if d is not None]
    if not reused:
        return float("nan")
    return sum(1 for d in reused if d <= horizon) / len(reused)


def belady_stats(workload: list[MemoryKey], disk_root: Path) -> dict[str, float]:
    """Run `workload` through a real MemoryManager under BeladyMinPolicy (in-memory trace)."""
    policy = BeladyMinPolicy(future=workload)
    trace_logger = TraceLogger("belady-preflight")  # no path: in-memory only, no disk I/O
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

    faults = sum(1 for e in trace_logger.events if e.event_type == TraceEventType.PAGE_FAULT)
    total = sum(1 for e in trace_logger.events if e.event_type == TraceEventType.ACCESS)
    hit_rate = 1.0 - faults / total if total else float("nan")
    return {"belady_faults": faults, "belady_total": total, "belady_hit_rate": hit_rate}


def mean(values: list[float]) -> float:
    """Arithmetic mean, or nan for an empty list."""
    return sum(values) / len(values) if values else float("nan")


def analyze_family(name: str, configs: list[WorkloadConfig], disk_root: Path) -> dict[str, Any]:
    """Compute every requested statistic for one workload family, averaged across episodes."""
    per_episode: list[dict[str, Any]] = []

    for episode_index, config in enumerate(configs):
        workload = generate_workload(config)
        distances_raw = tick_reuse_distances(workload)
        distances = [float(d) for d in distances_raw if d is not None]

        stats: dict[str, Any] = {
            "reuse_mean": mean(distances),
            "reuse_median": percentile(distances, 50),
            "reuse_p90": percentile(distances, 90),
            "reuse_p95": percentile(distances, 95),
            "reuse_p99": percentile(distances, 99),
            "reuse_has_repeats": bool(distances),
            "entropy_bits": access_entropy(workload),
            "mean_stack_distance": mean_stack_distance(workload),
            "burst_cv": coefficient_of_variation([float(r) for r in burst_run_lengths(workload)]),
            "unique_pages": unique_page_count(workload),
            "unique_page_ratio": unique_page_count(workload) / len(workload),
            "early_late_tv_distance": early_late_tv_distance(workload),
        }
        for window in WORKING_SET_WINDOWS:
            sizes = sliding_window_working_set(workload, window)
            stats[f"working_set_mean_w{window}"] = mean([float(s) for s in sizes])
            stats[f"working_set_max_w{window}"] = max(sizes) if sizes else float("nan")

        for horizon in HORIZONS:
            stats[f"label_positive_frac_H{horizon}"] = label_positive_fraction(
                distances_raw, horizon
            )
            stats[f"conditional_frac_H{horizon}"] = conditional_fraction_within_horizon(
                distances_raw, horizon
            )

        if episode_index < N_BELADY_EPISODES:
            episode_disk_root = disk_root / name / f"ep{episode_index}"
            episode_disk_root.mkdir(parents=True, exist_ok=True)
            stats.update(belady_stats(workload, episode_disk_root))

        if episode_index == 0:
            stats["reuse_histogram_sample"] = histogram(distances, HISTOGRAM_BINS)

        per_episode.append(stats)

    aggregated: dict[str, Any] = {"n_episodes": len(configs), "length": LENGTH}
    numeric_keys = [
        k for k in per_episode[0] if k not in ("reuse_has_repeats", "reuse_histogram_sample")
    ]
    for key in numeric_keys:
        values = [ep[key] for ep in per_episode if not math.isnan(float(ep.get(key, float("nan"))))]
        aggregated[f"{key}_mean"] = mean(values)
        aggregated[f"{key}_std"] = (
            math.sqrt(sum((v - mean(values)) ** 2 for v in values) / len(values))
            if len(values) > 1
            else 0.0
        )
    aggregated["reuse_histogram_sample"] = per_episode[0]["reuse_histogram_sample"]
    aggregated["per_episode"] = per_episode
    return aggregated


def main() -> None:
    """Run the pre-flight analysis for all six families and print/write the report."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        disk_root = Path(tmp_dir)
        results: dict[str, Any] = {}
        for name, configs in FAMILIES.items():
            print(f"=== {name} ===")
            results[name] = analyze_family(name, configs, disk_root)
            r = results[name]
            print(
                f"  reuse: mean={r['reuse_mean_mean']:.1f} median={r['reuse_median_mean']:.1f} "
                f"p90={r['reuse_p90_mean']:.1f} p95={r['reuse_p95_mean']:.1f} "
                f"p99={r['reuse_p99_mean']:.1f}"
            )
            print(
                f"  entropy={r['entropy_bits_mean']:.3f} "
                f"stack_dist={r['mean_stack_distance_mean']:.2f} "
                f"burst_cv={r['burst_cv_mean']:.3f} "
                f"unique_ratio={r['unique_page_ratio_mean']:.3f} "
                f"tv_drift={r['early_late_tv_distance_mean']:.3f}"
            )
            for h in HORIZONS:
                print(
                    f"  H={h:>4}: label_pos={r[f'label_positive_frac_H{h}_mean']:.3f} "
                    f"cond_pos={r[f'conditional_frac_H{h}_mean']:.3f}"
                )
            if "belady_hit_rate_mean" in r:
                print(
                    f"  belady: faults={r['belady_faults_mean']:.1f} "
                    f"hit_rate={r['belady_hit_rate_mean']:.3f}"
                )
            print()

        out_path = Path("experiments/preflight-analysis-1/statistics.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote full results to {out_path}")


if __name__ == "__main__":
    main()
