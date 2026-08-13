"""Unit tests for :mod:`neuropager.experiment.preprocessing`."""

from __future__ import annotations

import pytest

from neuropager.dataset.features import FEATURE_NAMES, Features
from neuropager.dataset.schema import DatasetExample
from neuropager.experiment.preprocessing import build_feature_matrix

_FEATURES_A = Features(
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

_FEATURES_B = Features(
    recency=10.0,
    frequency=20.0,
    avg_inter_access_interval=30.0,
    inter_access_interval_variance=40.0,
    fault_count=5.0,
    fault_ratio=0.25,
    page_age=50.0,
    recent_burst=6.0,
    long_burst=7.0,
    recent_interval=8.0,
    normalized_recency=0.5,
    elapsed_since_fault=9.0,
    max_historical_interval=11.0,
)


def _example(page_id: str, features: Features, label: int = 0) -> DatasetExample:
    return DatasetExample(
        episode_id="ep-1",
        decision_tick=10,
        page_id=page_id,
        policy="LRUPolicy",
        working_memory_capacity=4,
        eviction_reason="capacity",
        horizon=25,
        features=features,
        label=label,
    )


def test_build_feature_matrix_rejects_empty_input() -> None:
    """build_feature_matrix raises for zero examples."""
    with pytest.raises(ValueError):
        build_feature_matrix([])


def test_build_feature_matrix_shape_and_columns() -> None:
    """X has one row per example and one column per approved feature."""
    examples = [_example("a", _FEATURES_A), _example("b", _FEATURES_B)]

    x, y, feature_order = build_feature_matrix(examples)

    assert x.shape == (2, 13)
    assert feature_order == FEATURE_NAMES
    assert y.tolist() == [0, 0]


def test_build_feature_matrix_column_order_matches_feature_names() -> None:
    """Each column matches the corresponding named field, in FEATURE_NAMES order."""
    examples = [_example("a", _FEATURES_A)]

    x, _, feature_order = build_feature_matrix(examples)

    for column_index, name in enumerate(feature_order):
        assert x[0, column_index] == getattr(_FEATURES_A, name)


def test_build_feature_matrix_labels_match_examples() -> None:
    """Y contains each example's label, in the same order as the examples."""
    examples = [
        _example("a", _FEATURES_A, label=1),
        _example("b", _FEATURES_B, label=0),
    ]

    _, y, _ = build_feature_matrix(examples)

    assert y.tolist() == [1, 0]


def test_build_feature_matrix_is_deterministic() -> None:
    """Calling build_feature_matrix twice on the same input gives identical output."""
    examples = [_example("a", _FEATURES_A), _example("b", _FEATURES_B)]

    x1, y1, _ = build_feature_matrix(examples)
    x2, y2, _ = build_feature_matrix(examples)

    assert (x1 == x2).all()
    assert (y1 == y2).all()


def test_build_feature_matrix_ignores_page_id() -> None:
    """Two examples with identical features but different page_ids produce identical rows.

    This is the structural proof that page identity never enters the
    feature matrix: only `example.features` is ever read.
    """
    examples = [_example("totally-different-id-1", _FEATURES_A), _example("id-2", _FEATURES_A)]

    x, _, _ = build_feature_matrix(examples)

    assert (x[0] == x[1]).all()
