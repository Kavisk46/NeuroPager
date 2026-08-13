"""Unit tests for :mod:`neuropager.dataset.validation`."""

from __future__ import annotations

import dataclasses

import pytest

from neuropager.dataset.features import Features
from neuropager.dataset.schema import DatasetExample
from neuropager.dataset.validation import DatasetValidationError, validate_dataset

_VALID_FEATURES = Features(
    recency=1.0,
    frequency=2.0,
    avg_inter_access_interval=3.0,
    inter_access_interval_variance=4.0,
    fault_count=0.0,
    fault_ratio=0.0,
    page_age=5.0,
    recent_burst=1.0,
    long_burst=2.0,
    recent_interval=3.0,
    normalized_recency=0.2,
    elapsed_since_fault=-1.0,
    max_historical_interval=3.0,
)


def _valid_example(**overrides: object) -> DatasetExample:
    defaults: dict[str, object] = {
        "episode_id": "ep-1",
        "decision_tick": 10,
        "page_id": "a",
        "policy": "LRUPolicy",
        "working_memory_capacity": 4,
        "eviction_reason": "capacity",
        "horizon": 25,
        "features": _VALID_FEATURES,
        "label": 1,
    }
    defaults.update(overrides)
    return DatasetExample(**defaults)  # type: ignore[arg-type]


def test_valid_dataset_passes_silently() -> None:
    """A well-formed set of examples raises nothing."""
    validate_dataset([_valid_example()])


def test_empty_episode_id_raises() -> None:
    """An empty episode_id is rejected."""
    with pytest.raises(DatasetValidationError):
        validate_dataset([_valid_example(episode_id="")])


def test_negative_decision_tick_raises() -> None:
    """A negative decision_tick is rejected."""
    with pytest.raises(DatasetValidationError):
        validate_dataset([_valid_example(decision_tick=-1)])


def test_non_positive_horizon_raises() -> None:
    """A zero or negative horizon is rejected."""
    with pytest.raises(DatasetValidationError):
        validate_dataset([_valid_example(horizon=0)])


def test_non_positive_capacity_raises() -> None:
    """A zero or negative working_memory_capacity is rejected."""
    with pytest.raises(DatasetValidationError):
        validate_dataset([_valid_example(working_memory_capacity=0)])


@pytest.mark.parametrize("label", [-1, 2, 5])
def test_label_outside_zero_one_raises(label: int) -> None:
    """A label outside {0, 1} is rejected."""
    with pytest.raises(DatasetValidationError):
        validate_dataset([_valid_example(label=label)])


def test_missing_page_history_raises() -> None:
    """A candidate with zero recorded accesses (frequency 0) is rejected."""
    zero_history_features = dataclasses.replace(_VALID_FEATURES, frequency=0.0)

    with pytest.raises(DatasetValidationError):
        validate_dataset([_valid_example(features=zero_history_features)])


def test_non_finite_feature_raises() -> None:
    """A NaN or infinite feature value is rejected."""
    nan_features = dataclasses.replace(_VALID_FEATURES, recency=float("nan"))
    inf_features = dataclasses.replace(_VALID_FEATURES, recency=float("inf"))

    with pytest.raises(DatasetValidationError):
        validate_dataset([_valid_example(features=nan_features)])
    with pytest.raises(DatasetValidationError):
        validate_dataset([_valid_example(features=inf_features)])


def test_duplicate_examples_raise() -> None:
    """Two examples with the same (episode_id, decision_tick, page_id, horizon) raise."""
    example = _valid_example()

    with pytest.raises(DatasetValidationError):
        validate_dataset([example, example])


def test_duplicate_check_ignores_unrelated_differences() -> None:
    """Two examples differing only in horizon are not duplicates."""
    validate_dataset([_valid_example(horizon=25), _valid_example(horizon=50)])


def test_duplicate_check_ignores_different_pages() -> None:
    """Two examples differing only in page_id are not duplicates."""
    validate_dataset([_valid_example(page_id="a"), _valid_example(page_id="b")])
