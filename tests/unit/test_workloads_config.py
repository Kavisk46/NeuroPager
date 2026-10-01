"""Unit tests for :mod:`neuropager.workloads.config`."""

from __future__ import annotations

import pytest

from neuropager.workloads.config import (
    BurstyConfig,
    LongRangeReuseConfig,
    NonStationaryConfig,
    PhaseChangingConfig,
    TemporalLocalityConfig,
    UniformRandomConfig,
)


def test_uniform_random_rejects_non_positive_length() -> None:
    """UniformRandomConfig raises for length <= 0."""
    with pytest.raises(ValueError):
        UniformRandomConfig(length=0, key_space=8, seed=0)


def test_uniform_random_rejects_non_positive_key_space() -> None:
    """UniformRandomConfig raises for key_space <= 0."""
    with pytest.raises(ValueError):
        UniformRandomConfig(length=10, key_space=0, seed=0)


def test_temporal_locality_rejects_locality_out_of_range() -> None:
    """TemporalLocalityConfig raises for locality outside (0, 1)."""
    with pytest.raises(ValueError):
        TemporalLocalityConfig(length=10, key_space=8, seed=0, locality=0.0)
    with pytest.raises(ValueError):
        TemporalLocalityConfig(length=10, key_space=8, seed=0, locality=1.0)
    with pytest.raises(ValueError):
        TemporalLocalityConfig(length=10, key_space=8, seed=0, locality=-0.1)


def test_temporal_locality_accepts_valid_locality() -> None:
    """TemporalLocalityConfig accepts a locality strictly between 0 and 1."""
    config = TemporalLocalityConfig(length=10, key_space=8, seed=0, locality=0.5)
    assert config.locality == 0.5


def test_bursty_rejects_mean_burst_length_below_one() -> None:
    """BurstyConfig raises for mean_burst_length < 1."""
    with pytest.raises(ValueError):
        BurstyConfig(length=10, key_space=8, seed=0, mean_burst_length=0.5)


def test_bursty_accepts_mean_burst_length_of_exactly_one() -> None:
    """BurstyConfig accepts the boundary value mean_burst_length == 1."""
    config = BurstyConfig(length=10, key_space=8, seed=0, mean_burst_length=1.0)
    assert config.mean_burst_length == 1.0


def test_long_range_reuse_rejects_non_positive_key_space() -> None:
    """LongRangeReuseConfig raises for key_space <= 0."""
    with pytest.raises(ValueError):
        LongRangeReuseConfig(length=10, key_space=0, seed=0)


def test_phase_changing_rejects_non_positive_phase_length() -> None:
    """PhaseChangingConfig raises for phase_length <= 0."""
    with pytest.raises(ValueError):
        PhaseChangingConfig(length=10, seed=0, phase_length=0, keys_per_phase=4)


def test_phase_changing_rejects_non_positive_keys_per_phase() -> None:
    """PhaseChangingConfig raises for keys_per_phase <= 0."""
    with pytest.raises(ValueError):
        PhaseChangingConfig(length=10, seed=0, phase_length=5, keys_per_phase=0)


def test_non_stationary_rejects_negative_drift_rate() -> None:
    """NonStationaryConfig raises for a negative drift_rate."""
    with pytest.raises(ValueError):
        NonStationaryConfig(length=10, key_space=8, seed=0, drift_rate=-0.01)


def test_non_stationary_accepts_zero_drift_rate() -> None:
    """NonStationaryConfig accepts drift_rate == 0 (a static, non-drifting case)."""
    config = NonStationaryConfig(length=10, key_space=8, seed=0, drift_rate=0.0)
    assert config.drift_rate == 0.0


def test_configs_are_frozen() -> None:
    """Config dataclasses cannot be mutated after construction."""
    import dataclasses

    config = UniformRandomConfig(length=10, key_space=8, seed=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.length = 20  # type: ignore[misc]
