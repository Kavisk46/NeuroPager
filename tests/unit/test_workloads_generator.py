"""Unit tests validating each workload generator's intended mathematical properties."""

from __future__ import annotations

from collections import Counter

from neuropager.utils.types import MemoryKey
from neuropager.workloads.config import (
    BurstyConfig,
    LongRangeReuseConfig,
    NonStationaryConfig,
    PhaseChangingConfig,
    TemporalLocalityConfig,
    UniformRandomConfig,
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

# --- determinism (applies to every kind) ------------------------------------


def test_uniform_random_is_deterministic_given_the_same_seed() -> None:
    """Same config -> byte-identical output; a different seed -> (almost certainly) different."""
    config = UniformRandomConfig(length=500, key_space=10, seed=42)

    run_1 = generate_uniform_random(config)
    run_2 = generate_uniform_random(config)
    run_3 = generate_uniform_random(UniformRandomConfig(length=500, key_space=10, seed=43))

    assert run_1 == run_2
    assert run_1 != run_3


def test_all_generators_produce_the_requested_length() -> None:
    """Every generator returns exactly `length` keys."""
    length = 337
    assert len(generate_uniform_random(UniformRandomConfig(length, 8, 0))) == length
    assert len(generate_temporal_locality(TemporalLocalityConfig(length, 8, 0, 0.5))) == length
    assert len(generate_bursty(BurstyConfig(length, 8, 0, 5.0))) == length
    assert len(generate_long_range_reuse(LongRangeReuseConfig(length, 8, 0))) == length
    assert len(generate_phase_changing(PhaseChangingConfig(length, 0, 20, 4))) == length
    assert len(generate_non_stationary(NonStationaryConfig(length, 8, 0, 0.01))) == length


def test_generate_workload_dispatches_correctly() -> None:
    """generate_workload(config) matches the type-specific function's output."""
    config = UniformRandomConfig(length=100, key_space=8, seed=1)

    assert generate_workload(config) == generate_uniform_random(config)


# --- A. uniform random -------------------------------------------------------


def test_uniform_random_entropy_is_near_maximal() -> None:
    """Uniform access over K keys has entropy close to log2(K)."""
    import math

    config = UniformRandomConfig(length=5000, key_space=10, seed=0)
    workload = generate_uniform_random(config)

    entropy = access_entropy(workload)

    assert entropy > math.log2(10) - 0.05


def test_uniform_random_key_frequencies_are_roughly_balanced() -> None:
    """Each key's observed frequency is close to the expected 1/K share."""
    config = UniformRandomConfig(length=5000, key_space=10, seed=0)
    workload = generate_uniform_random(config)

    counts = Counter(workload)
    expected = config.length / config.key_space
    for count in counts.values():
        assert abs(count - expected) < expected * 0.3


# --- B. temporal locality -----------------------------------------------------


def test_temporal_locality_has_lower_mean_stack_distance_than_uniform_random() -> None:
    """Recency-biased access has a smaller mean stack distance than i.i.d. uniform."""
    key_space, length, seed = 20, 3000, 0
    locality_workload = generate_temporal_locality(
        TemporalLocalityConfig(length, key_space, seed, locality=0.5)
    )
    uniform_workload = generate_uniform_random(UniformRandomConfig(length, key_space, seed))

    locality_mean = mean_stack_distance(locality_workload)
    uniform_mean = mean_stack_distance(uniform_workload)

    assert locality_mean is not None
    assert uniform_mean is not None
    assert locality_mean < uniform_mean


def test_higher_locality_parameter_gives_lower_mean_stack_distance() -> None:
    """Increasing the locality parameter strengthens the recency bias monotonically."""
    key_space, length, seed = 20, 3000, 0
    weak = generate_temporal_locality(TemporalLocalityConfig(length, key_space, seed, 0.2))
    strong = generate_temporal_locality(TemporalLocalityConfig(length, key_space, seed, 0.8))

    weak_mean = mean_stack_distance(weak)
    strong_mean = mean_stack_distance(strong)

    assert weak_mean is not None
    assert strong_mean is not None
    assert strong_mean < weak_mean


# --- C. bursty ----------------------------------------------------------------


def test_bursty_mean_run_length_is_close_to_configured_mean() -> None:
    """Observed mean burst-run length is close to mean_burst_length."""
    config = BurstyConfig(length=5000, key_space=20, seed=0, mean_burst_length=8.0)
    workload = generate_bursty(config)

    runs = burst_run_lengths(workload)
    observed_mean = sum(runs) / len(runs)

    assert abs(observed_mean - config.mean_burst_length) < config.mean_burst_length * 0.25


def test_bursty_has_longer_runs_than_uniform_random() -> None:
    """Bursty access shows much longer consecutive-repeat runs than i.i.d. uniform."""
    key_space, length, seed = 20, 3000, 0
    bursty_workload = generate_bursty(BurstyConfig(length, key_space, seed, mean_burst_length=10.0))
    uniform_workload = generate_uniform_random(UniformRandomConfig(length, key_space, seed))

    bursty_runs = burst_run_lengths(bursty_workload)
    uniform_runs = burst_run_lengths(uniform_workload)

    bursty_mean = sum(bursty_runs) / len(bursty_runs)
    uniform_mean = sum(uniform_runs) / len(uniform_runs)

    assert bursty_mean > uniform_mean * 3


# --- D. long-range reuse --------------------------------------------------------


def test_long_range_reuse_stack_distance_is_always_exactly_key_space_minus_one() -> None:
    """Every repeat occurrence has stack distance exactly key_space - 1, by construction."""
    config = LongRangeReuseConfig(length=1000, key_space=25, seed=0)
    workload = generate_long_range_reuse(config)

    distances = [d for d in stack_distances(workload) if d is not None]

    assert distances  # some repeats must have occurred
    assert all(d == config.key_space - 1 for d in distances)


def test_long_range_reuse_visits_every_key_each_full_cycle() -> None:
    """A full-length multiple of key_space visits every key the same number of times."""
    config = LongRangeReuseConfig(length=250, key_space=25, seed=0)  # 10 full cycles
    workload = generate_long_range_reuse(config)

    counts = Counter(workload)
    assert len(counts) == config.key_space
    assert all(count == 10 for count in counts.values())


# --- E. phase-changing -----------------------------------------------------------


def test_phase_changing_uses_disjoint_key_blocks_per_phase() -> None:
    """No key from one phase's block appears during a different phase's segment."""
    config = PhaseChangingConfig(length=1000, seed=0, phase_length=100, keys_per_phase=5)
    workload = generate_phase_changing(config)

    for phase_index in range(10):
        start_tick = phase_index * config.phase_length
        end_tick = start_tick + config.phase_length
        segment = set(workload[start_tick:end_tick])
        expected_block = {
            MemoryKey(f"k{phase_index * config.keys_per_phase + j}")
            for j in range(config.keys_per_phase)
        }
        assert segment <= expected_block


def test_phase_changing_has_lower_unique_page_count_than_uniform_random_with_same_key_budget() -> (
    None
):
    """Confining each phase to a small key block reduces total unique pages touched.

    Compares against a uniform-random workload whose key_space equals the
    *total* number of keys phase-changing could reach (n_phases *
    keys_per_phase), to show phase confinement does not trivially reduce
    unique pages just because fewer keys exist overall.
    """
    config = PhaseChangingConfig(length=1000, seed=0, phase_length=100, keys_per_phase=5)
    phase_workload = generate_phase_changing(config)
    n_phases = 10
    total_key_budget = n_phases * config.keys_per_phase

    uniform_workload = generate_uniform_random(
        UniformRandomConfig(length=1000, key_space=total_key_budget, seed=0)
    )

    assert unique_page_count(phase_workload) <= total_key_budget
    assert unique_page_count(uniform_workload) > unique_page_count(phase_workload) * 0.8


# --- F. non-stationary -------------------------------------------------------------


def test_non_stationary_early_and_late_distributions_differ() -> None:
    """The most-frequent key in the first 20% differs from the most-frequent key in the last 20%.

    Demonstrates genuine drift, as opposed to a static distribution.
    """
    config = NonStationaryConfig(length=5000, key_space=20, seed=0, drift_rate=0.02)
    workload = generate_non_stationary(config)

    early = workload[: config.length // 5]
    late = workload[-(config.length // 5) :]

    early_top = Counter(early).most_common(1)[0][0]
    late_top = Counter(late).most_common(1)[0][0]

    assert early_top != late_top


def test_zero_drift_rate_gives_a_static_distribution() -> None:
    """drift_rate == 0 means the ranking never rotates (the degenerate, non-drifting case)."""
    config = NonStationaryConfig(length=5000, key_space=20, seed=0, drift_rate=0.0)
    workload = generate_non_stationary(config)

    early = workload[: config.length // 5]
    late = workload[-(config.length // 5) :]

    early_top = Counter(early).most_common(1)[0][0]
    late_top = Counter(late).most_common(1)[0][0]

    assert early_top == late_top
