"""Configurable benchmark workload generators for NeuroPager experiments.

Provides six deterministic, seeded access-pattern generators (uniform
random, temporal locality, bursty, long-range reuse, phase-changing,
non-stationary) so that policy/model comparisons can be run against
workloads with known, mathematically-defined statistical properties
instead of only uniform-random access. See ``docs/workloads.md`` for the
precise definition of each kind, and ``neuropager.workloads.statistics``
for diagnostics to verify a generated instance actually exhibits the
intended properties.

This package produces plain ``list[MemoryKey]`` sequences only -- it has
no dependency on, and makes no change to, ``neuropager.trace`` or
``neuropager.dataset``.
"""

from __future__ import annotations

from neuropager.workloads.config import (
    BurstyConfig,
    LongRangeReuseConfig,
    NonStationaryConfig,
    PhaseChangingConfig,
    TemporalLocalityConfig,
    UniformRandomConfig,
    WorkloadConfig,
)
from neuropager.workloads.generator import (
    generate_bursty,
    generate_long_range_reuse,
    generate_non_stationary,
    generate_phase_changing,
    generate_temporal_locality,
    generate_uniform_random,
    generate_workload,
)
from neuropager.workloads.statistics import (
    access_entropy,
    burst_run_lengths,
    mean_stack_distance,
    stack_distances,
    unique_page_count,
)

__all__ = [
    "BurstyConfig",
    "LongRangeReuseConfig",
    "NonStationaryConfig",
    "PhaseChangingConfig",
    "TemporalLocalityConfig",
    "UniformRandomConfig",
    "WorkloadConfig",
    "access_entropy",
    "burst_run_lengths",
    "generate_bursty",
    "generate_long_range_reuse",
    "generate_non_stationary",
    "generate_phase_changing",
    "generate_temporal_locality",
    "generate_uniform_random",
    "generate_workload",
    "mean_stack_distance",
    "stack_distances",
    "unique_page_count",
]
