"""Unit tests for :mod:`neuropager.dataset.schema`."""

from __future__ import annotations

from neuropager.dataset.features import Features
from neuropager.dataset.schema import DatasetExample

_FEATURES = Features(
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


def _make_example() -> DatasetExample:
    return DatasetExample(
        episode_id="ep-1",
        decision_tick=10,
        page_id="a",
        policy="LRUPolicy",
        working_memory_capacity=4,
        eviction_reason="capacity",
        horizon=25,
        features=_FEATURES,
        label=1,
    )


def test_to_flat_dict_prefixes_feature_fields() -> None:
    """Feature fields are flattened with a feature_ prefix; metadata is not."""
    flat = _make_example().to_flat_dict()

    assert flat["episode_id"] == "ep-1"
    assert flat["decision_tick"] == 10
    assert flat["page_id"] == "a"
    assert flat["policy"] == "LRUPolicy"
    assert flat["working_memory_capacity"] == 4
    assert flat["eviction_reason"] == "capacity"
    assert flat["horizon"] == 25
    assert flat["label"] == 1
    assert flat["feature_recency"] == 1.0
    assert flat["feature_normalized_recency"] == 0.2
    assert "recency" not in flat  # only the prefixed form is present


def test_flat_dict_round_trips_to_an_equal_example() -> None:
    """to_flat_dict() -> from_flat_dict() reconstructs an identical example."""
    original = _make_example()

    restored = DatasetExample.from_flat_dict(original.to_flat_dict())

    assert restored == original


def test_flat_dict_is_json_serializable() -> None:
    """The flat dict uses only JSON-safe primitive types."""
    import json

    flat = _make_example().to_flat_dict()

    # Should not raise.
    encoded = json.dumps(flat, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded == flat
