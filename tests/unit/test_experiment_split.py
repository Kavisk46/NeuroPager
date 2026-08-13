"""Unit tests for :mod:`neuropager.experiment.split`."""

from __future__ import annotations

import pytest

from neuropager.dataset.features import Features
from neuropager.dataset.schema import DatasetExample
from neuropager.experiment.split import filter_examples_by_episodes, split_episodes

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


def _example(episode_id: str, page_id: str = "a") -> DatasetExample:
    return DatasetExample(
        episode_id=episode_id,
        decision_tick=10,
        page_id=page_id,
        policy="LRUPolicy",
        working_memory_capacity=4,
        eviction_reason="capacity",
        horizon=25,
        features=_FEATURES,
        label=1,
    )


def test_split_rejects_empty_episode_list() -> None:
    """split_episodes raises for an empty input."""
    with pytest.raises(ValueError):
        split_episodes([], seed=0)


def test_split_rejects_fractions_not_summing_to_one() -> None:
    """split_episodes raises if the fractions do not sum to 1.0."""
    with pytest.raises(ValueError):
        split_episodes(["a", "b"], seed=0, train_fraction=0.5, val_fraction=0.3, test_fraction=0.3)


def test_split_rejects_negative_fraction() -> None:
    """split_episodes raises for a negative fraction."""
    with pytest.raises(ValueError):
        split_episodes(["a"], seed=0, train_fraction=1.1, val_fraction=-0.1, test_fraction=0.0)


def test_split_covers_every_episode_exactly_once() -> None:
    """Every episode appears in exactly one of train/val/test."""
    episodes = [f"ep-{i}" for i in range(20)]

    split = split_episodes(episodes, seed=0)

    all_assigned = list(split.train_episodes) + list(split.val_episodes) + list(split.test_episodes)
    assert sorted(all_assigned) == sorted(episodes)
    assert len(set(all_assigned)) == len(all_assigned)


def test_split_has_no_overlap_between_partitions() -> None:
    """No episode ID appears in more than one partition."""
    episodes = [f"ep-{i}" for i in range(20)]

    split = split_episodes(episodes, seed=0)

    train_set = set(split.train_episodes)
    val_set = set(split.val_episodes)
    test_set = set(split.test_episodes)
    assert train_set & val_set == set()
    assert train_set & test_set == set()
    assert val_set & test_set == set()


def test_split_is_deterministic_given_the_same_seed() -> None:
    """The same episodes and seed always produce the same split."""
    episodes = [f"ep-{i}" for i in range(20)]

    split_1 = split_episodes(episodes, seed=42)
    split_2 = split_episodes(episodes, seed=42)

    assert split_1 == split_2


def test_split_is_independent_of_input_order() -> None:
    """Shuffling the input episode order does not change the resulting split."""
    episodes = [f"ep-{i}" for i in range(20)]
    reversed_episodes = list(reversed(episodes))

    split_1 = split_episodes(episodes, seed=7)
    split_2 = split_episodes(reversed_episodes, seed=7)

    assert split_1 == split_2


def test_different_seeds_usually_produce_different_splits() -> None:
    """Different seeds produce a different assignment (sanity, not a hard guarantee)."""
    episodes = [f"ep-{i}" for i in range(20)]

    split_1 = split_episodes(episodes, seed=1)
    split_2 = split_episodes(episodes, seed=2)

    assert split_1.train_episodes != split_2.train_episodes


def test_split_respects_requested_fractions_approximately() -> None:
    """With enough episodes, partition sizes are close to the requested fractions."""
    episodes = [f"ep-{i}" for i in range(100)]

    split = split_episodes(
        episodes, seed=0, train_fraction=0.7, val_fraction=0.15, test_fraction=0.15
    )

    assert 65 <= len(split.train_episodes) <= 75
    assert 10 <= len(split.val_episodes) <= 20
    assert 10 <= len(split.test_episodes) <= 20


def test_filter_examples_by_episodes_keeps_only_matching_episodes() -> None:
    """filter_examples_by_episodes drops examples from episodes not in the list."""
    examples = [_example("a"), _example("b"), _example("c")]

    filtered = filter_examples_by_episodes(examples, ["a", "c"])

    assert {e.episode_id for e in filtered} == {"a", "c"}


def test_filter_examples_by_episodes_preserves_order() -> None:
    """filter_examples_by_episodes does not reorder the surviving examples."""
    examples = [_example("a"), _example("b"), _example("a"), _example("c")]

    filtered = filter_examples_by_episodes(examples, ["a", "b"])

    assert [e.episode_id for e in filtered] == ["a", "b", "a"]
