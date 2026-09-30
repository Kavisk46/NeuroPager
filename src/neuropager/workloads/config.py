"""Configuration dataclasses for the benchmark workload generator.

Each workload kind has its own frozen, validated config dataclass rather
than one kitchen-sink config, so an invalid combination of parameters
(e.g. a locality bias supplied to the uniform-random generator) is a type
error, not a silently-ignored field. Every config carries its own
``seed``, so a given config is a complete, deterministic recipe for one
workload instance -- the same config always produces the same sequence of
keys (see ``docs/workloads.md`` for the exact mathematical definition of
each kind).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UniformRandomConfig:
    """Independent, identically distributed uniform access over a key space.

    Attributes:
        length: Number of accesses (ticks) to generate.
        key_space: Number of distinct keys, labeled ``k0..k{key_space-1}``.
        seed: Seed for the deterministic random generator.
    """

    length: int
    key_space: int
    seed: int

    def __post_init__(self) -> None:
        """Validate parameter ranges.

        Raises:
            ValueError: If ``length`` or ``key_space`` is not positive.
        """
        if self.length <= 0:
            raise ValueError(f"length must be positive, got {self.length}")
        if self.key_space <= 0:
            raise ValueError(f"key_space must be positive, got {self.key_space}")


@dataclass(frozen=True)
class TemporalLocalityConfig:
    """LRU-stack-distance access: recency-biased via a fixed rank distribution.

    Attributes:
        length: Number of accesses (ticks) to generate.
        key_space: Number of distinct keys.
        seed: Seed for the deterministic random generator.
        locality: Bias strength in ``(0, 1)``. At each tick, a recency
            rank ``d`` in ``{0, ..., key_space - 1}`` (0 = most recently
            used) is drawn with ``P(d = i) proportional to (1 - locality) ** i``.
            Larger values concentrate access more strongly on
            recently-used keys.
    """

    length: int
    key_space: int
    seed: int
    locality: float

    def __post_init__(self) -> None:
        """Validate parameter ranges.

        Raises:
            ValueError: If ``length``/``key_space`` is not positive, or
                ``locality`` is not in ``(0, 1)``.
        """
        if self.length <= 0:
            raise ValueError(f"length must be positive, got {self.length}")
        if self.key_space <= 0:
            raise ValueError(f"key_space must be positive, got {self.key_space}")
        if not 0.0 < self.locality < 1.0:
            raise ValueError(f"locality must be in (0, 1), got {self.locality}")


@dataclass(frozen=True)
class BurstyConfig:
    """Runs of the same key, with geometrically-distributed burst lengths.

    Attributes:
        length: Number of accesses (ticks) to generate.
        key_space: Number of distinct keys.
        seed: Seed for the deterministic random generator.
        mean_burst_length: Expected number of consecutive accesses to the
            same key before switching to a fresh, uniformly-random key.
            Each burst's length is geometrically distributed with this
            mean (continuation probability ``1 - 1/mean_burst_length`` at
            every tick).
    """

    length: int
    key_space: int
    seed: int
    mean_burst_length: float

    def __post_init__(self) -> None:
        """Validate parameter ranges.

        Raises:
            ValueError: If ``length``/``key_space`` is not positive, or
                ``mean_burst_length`` is less than 1.
        """
        if self.length <= 0:
            raise ValueError(f"length must be positive, got {self.length}")
        if self.key_space <= 0:
            raise ValueError(f"key_space must be positive, got {self.key_space}")
        if self.mean_burst_length < 1.0:
            raise ValueError(f"mean_burst_length must be >= 1, got {self.mean_burst_length}")


@dataclass(frozen=True)
class LongRangeReuseConfig:
    """Full cyclic passes over a (shuffled-per-cycle) key universe.

    Every key is accessed exactly once per cycle, so the stack distance
    (distinct keys seen since a key's previous occurrence) between any two
    consecutive occurrences of the same key is always exactly
    ``key_space - 1`` -- deterministically large, regardless of shuffling.

    Attributes:
        length: Number of accesses (ticks) to generate.
        key_space: Number of distinct keys per cycle. Should be set well
            above the working-memory capacity under test for the reuse
            gap to actually exceed the cache's reach.
        seed: Seed for the deterministic random generator (controls the
            per-cycle shuffle order only; reuse distance is unaffected).
    """

    length: int
    key_space: int
    seed: int

    def __post_init__(self) -> None:
        """Validate parameter ranges.

        Raises:
            ValueError: If ``length`` or ``key_space`` is not positive.
        """
        if self.length <= 0:
            raise ValueError(f"length must be positive, got {self.length}")
        if self.key_space <= 0:
            raise ValueError(f"key_space must be positive, got {self.key_space}")


@dataclass(frozen=True)
class PhaseChangingConfig:
    """Discrete working-set shifts: disjoint key subsets in sequential phases.

    The episode is divided into consecutive phases of ``phase_length``
    ticks each. Phase ``i`` draws uniformly from a disjoint block of
    ``keys_per_phase`` keys (``k{i*keys_per_phase}..k{(i+1)*keys_per_phase-1}``)
    not used by any other phase. The number of phases is derived from
    ``length`` and ``phase_length`` (the final phase may be shorter).

    Attributes:
        length: Number of accesses (ticks) to generate.
        seed: Seed for the deterministic random generator.
        phase_length: Number of ticks per phase.
        keys_per_phase: Number of distinct keys in each phase's subset.
    """

    length: int
    seed: int
    phase_length: int
    keys_per_phase: int

    def __post_init__(self) -> None:
        """Validate parameter ranges.

        Raises:
            ValueError: If any parameter is not positive.
        """
        if self.length <= 0:
            raise ValueError(f"length must be positive, got {self.length}")
        if self.phase_length <= 0:
            raise ValueError(f"phase_length must be positive, got {self.phase_length}")
        if self.keys_per_phase <= 0:
            raise ValueError(f"keys_per_phase must be positive, got {self.keys_per_phase}")


@dataclass(frozen=True)
class NonStationaryConfig:
    """Gradually drifting popularity ranking over a fixed key universe.

    A fixed Zipf-like weight profile (``w_i = 1 / (i + 1)`` for rank
    ``i``) is assigned to a *rotating* subset of key positions: at tick
    ``t`` the ranking-to-key assignment is cyclically shifted by
    ``floor(t * drift_rate)`` positions before sampling. Unlike
    :class:`PhaseChangingConfig`'s discrete jumps, popularity here changes
    continuously.

    Attributes:
        length: Number of accesses (ticks) to generate.
        key_space: Number of distinct keys.
        seed: Seed for the deterministic random generator.
        drift_rate: Ranking-rotation speed, in positions per tick (e.g.
            ``0.01`` rotates the ranking by one position every 100 ticks).
    """

    length: int
    key_space: int
    seed: int
    drift_rate: float

    def __post_init__(self) -> None:
        """Validate parameter ranges.

        Raises:
            ValueError: If ``length``/``key_space`` is not positive, or
                ``drift_rate`` is negative.
        """
        if self.length <= 0:
            raise ValueError(f"length must be positive, got {self.length}")
        if self.key_space <= 0:
            raise ValueError(f"key_space must be positive, got {self.key_space}")
        if self.drift_rate < 0.0:
            raise ValueError(f"drift_rate must be non-negative, got {self.drift_rate}")


WorkloadConfig = (
    UniformRandomConfig
    | TemporalLocalityConfig
    | BurstyConfig
    | LongRangeReuseConfig
    | PhaseChangingConfig
    | NonStationaryConfig
)
"""Union of every workload config type accepted by
:func:`~neuropager.workloads.generator.generate_workload`.
"""
