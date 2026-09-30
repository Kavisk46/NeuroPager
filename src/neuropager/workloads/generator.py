"""Deterministic generators for each benchmark workload kind.

Every generator is a pure function of its config: the same config object
always produces the same sequence of :class:`~neuropager.utils.types.MemoryKey`
values, since all randomness is drawn from a :class:`random.Random`
seeded from ``config.seed``. See ``docs/workloads.md`` for the precise
mathematical definition each function implements.
"""

from __future__ import annotations

import random

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


def _key_universe(key_space: int) -> list[MemoryKey]:
    """Return the canonical, ordered list of keys ``k0..k{key_space-1}``.

    Args:
        key_space: Number of distinct keys.

    Returns:
        The key universe, in a fixed, deterministic order.
    """
    return [MemoryKey(f"k{i}") for i in range(key_space)]


def generate_uniform_random(config: UniformRandomConfig) -> list[MemoryKey]:
    """Generate i.i.d. uniform access: ``key_t ~ Uniform{k0, ..., k(K-1)}``.

    Args:
        config: The workload configuration.

    Returns:
        A list of ``config.length`` keys.
    """
    rng = random.Random(config.seed)
    keys = _key_universe(config.key_space)
    return [rng.choice(keys) for _ in range(config.length)]


def generate_temporal_locality(config: TemporalLocalityConfig) -> list[MemoryKey]:
    """Generate recency-biased access via an LRU-stack-distance model.

    Maintains a recency stack (most-recently-used key at position 0). At
    each tick, a rank ``d`` is drawn with
    ``P(d = i) proportional to (1 - locality) ** i`` for ``i`` in
    ``0..key_space-1``, the key at that rank is accessed, and it is then
    moved to the front of the stack.

    Args:
        config: The workload configuration.

    Returns:
        A list of ``config.length`` keys.
    """
    rng = random.Random(config.seed)
    stack = _key_universe(config.key_space)
    weights = [(1.0 - config.locality) ** i for i in range(config.key_space)]
    ranks = list(range(config.key_space))
    output: list[MemoryKey] = []
    for _ in range(config.length):
        depth = rng.choices(ranks, weights=weights, k=1)[0]
        key = stack[depth]
        output.append(key)
        stack.pop(depth)
        stack.insert(0, key)
    return output


def generate_bursty(config: BurstyConfig) -> list[MemoryKey]:
    """Generate bursty access: geometric-length runs of a single key.

    Each burst continues with probability ``1 - 1 / mean_burst_length``
    at every tick, giving burst lengths ``L`` with
    ``P(L = l) = (1 - q) ** (l - 1) * q``, ``q = 1 / mean_burst_length``,
    so ``E[L] = mean_burst_length``. When a burst ends, the next key is
    drawn uniformly at random (independent of history).

    Args:
        config: The workload configuration.

    Returns:
        A list of ``config.length`` keys.
    """
    rng = random.Random(config.seed)
    keys = _key_universe(config.key_space)
    continue_probability = 1.0 - 1.0 / config.mean_burst_length
    output: list[MemoryKey] = []
    current_key = rng.choice(keys)
    for _ in range(config.length):
        output.append(current_key)
        if rng.random() >= continue_probability:
            current_key = rng.choice(keys)
    return output


def generate_long_range_reuse(config: LongRangeReuseConfig) -> list[MemoryKey]:
    """Generate long-range reuse: repeated passes over one fixed key permutation.

    A single random permutation of all ``key_space`` keys is drawn once
    (from ``config.seed``) and then repeated ("tiled") as many times as
    needed. Because the *same* cycle repeats exactly, every key sits at
    the same position in every pass, so the stack distance between any
    two consecutive occurrences of a key is always exactly
    ``key_space - 1`` (precisely the other ``key_space - 1`` keys of one
    full period, with no possibility of double-counting).

    Args:
        config: The workload configuration.

    Returns:
        A list of ``config.length`` keys.
    """
    rng = random.Random(config.seed)
    cycle = _key_universe(config.key_space)
    rng.shuffle(cycle)
    repeats = -(-config.length // config.key_space)  # ceil division
    return (cycle * repeats)[: config.length]


def generate_phase_changing(config: PhaseChangingConfig) -> list[MemoryKey]:
    """Generate discrete working-set shifts across sequential phases.

    Phase ``i`` (ticks ``[i * phase_length, (i + 1) * phase_length)``)
    draws uniformly at random from the disjoint key block
    ``k{i*keys_per_phase}..k{(i+1)*keys_per_phase-1}``. The final phase is
    truncated to fit ``config.length`` exactly.

    Args:
        config: The workload configuration.

    Returns:
        A list of ``config.length`` keys.
    """
    rng = random.Random(config.seed)
    output: list[MemoryKey] = []
    phase_index = 0
    while len(output) < config.length:
        start = phase_index * config.keys_per_phase
        phase_keys = [MemoryKey(f"k{start + j}") for j in range(config.keys_per_phase)]
        remaining = config.length - len(output)
        this_phase_length = min(config.phase_length, remaining)
        output.extend(rng.choice(phase_keys) for _ in range(this_phase_length))
        phase_index += 1
    return output


def generate_non_stationary(config: NonStationaryConfig) -> list[MemoryKey]:
    """Generate access under a continuously drifting popularity ranking.

    A fixed Zipf-like weight profile (``w_i = 1 / (i + 1)`` for rank
    ``i``) is applied to a key ordering that rotates by
    ``floor(t * drift_rate) mod key_space`` positions at tick ``t``, so
    which key currently holds each popularity rank changes gradually
    (rather than jumping discretely, as in :func:`generate_phase_changing`).

    Args:
        config: The workload configuration.

    Returns:
        A list of ``config.length`` keys.
    """
    rng = random.Random(config.seed)
    keys = _key_universe(config.key_space)
    weights = [1.0 / (i + 1) for i in range(config.key_space)]
    output: list[MemoryKey] = []
    for t in range(config.length):
        shift = int(t * config.drift_rate) % config.key_space
        rotated = keys[shift:] + keys[:shift]
        key = rng.choices(rotated, weights=weights, k=1)[0]
        output.append(key)
    return output


def generate_workload(config: WorkloadConfig) -> list[MemoryKey]:
    """Dispatch to the generator matching ``config``'s type.

    Args:
        config: Any of the six workload config types.

    Returns:
        The generated key sequence.

    Raises:
        TypeError: If ``config`` is not one of the known config types.
    """
    match config:
        case UniformRandomConfig():
            return generate_uniform_random(config)
        case TemporalLocalityConfig():
            return generate_temporal_locality(config)
        case BurstyConfig():
            return generate_bursty(config)
        case LongRangeReuseConfig():
            return generate_long_range_reuse(config)
        case PhaseChangingConfig():
            return generate_phase_changing(config)
        case NonStationaryConfig():
            return generate_non_stationary(config)
        case _:
            raise TypeError(f"unrecognized workload config type: {type(config)!r}")
