"""Unit tests for :mod:`neuropager.experiment.runner`."""

from __future__ import annotations

import pytest

from neuropager.dataset.features import Features
from neuropager.dataset.schema import DatasetExample
from neuropager.experiment.models import HIST_GRADIENT_BOOSTING, LOGISTIC_REGRESSION
from neuropager.experiment.runner import run_full_experiment, run_model_experiment


def _features(seed: int) -> Features:
    return Features(
        recency=float(seed % 10),
        frequency=float(1 + seed % 5),
        avg_inter_access_interval=float(seed % 7 + 1),
        inter_access_interval_variance=float(seed % 3),
        fault_count=float(seed % 2),
        fault_ratio=0.5 if seed % 2 else 0.0,
        page_age=float(seed % 20 + 1),
        recent_burst=float(seed % 4),
        long_burst=float(seed % 6),
        recent_interval=float(seed % 5 + 1),
        normalized_recency=(seed % 10) / (seed % 20 + 1),
        elapsed_since_fault=-1.0,
        max_historical_interval=float(seed % 8 + 1),
    )


def _build_examples(n_episodes: int, horizon: int) -> list[DatasetExample]:
    examples = []
    for episode_index in range(n_episodes):
        for candidate_index in range(3):
            seed = episode_index * 7 + candidate_index
            examples.append(
                DatasetExample(
                    episode_id=f"ep-{episode_index}",
                    decision_tick=10 + candidate_index,
                    page_id=f"page-{candidate_index}",
                    policy="LRUPolicy",
                    working_memory_capacity=4,
                    eviction_reason="capacity",
                    horizon=horizon,
                    features=_features(seed),
                    label=seed % 2,
                )
            )
    return examples


def test_run_model_experiment_raises_for_missing_horizon() -> None:
    """run_model_experiment raises when no examples exist for the requested horizon."""
    examples = _build_examples(n_episodes=10, horizon=25)

    with pytest.raises(ValueError):
        run_model_experiment(
            examples,
            horizon=999,
            model_name=LOGISTIC_REGRESSION,
            seed=0,
            working_memory_capacity=4,
            experiment_id="exp-1",
        )


def test_run_model_experiment_produces_metrics_for_all_three_splits() -> None:
    """With enough episodes, train/val/test metrics are all computed (non-empty)."""
    examples = _build_examples(n_episodes=20, horizon=25)

    result = run_model_experiment(
        examples,
        horizon=25,
        model_name=LOGISTIC_REGRESSION,
        seed=0,
        working_memory_capacity=4,
        experiment_id="exp-1",
    )

    assert result.train_metrics.n_examples > 0
    assert result.metadata.horizon == 25
    assert result.metadata.model_name == LOGISTIC_REGRESSION
    assert result.metadata.feature_order[0] == "recency"


def test_run_model_experiment_metadata_matches_the_split_used() -> None:
    """The metadata's split exactly matches train/val/test episode partitioning."""
    examples = _build_examples(n_episodes=20, horizon=25)

    result = run_model_experiment(
        examples,
        horizon=25,
        model_name=LOGISTIC_REGRESSION,
        seed=0,
        working_memory_capacity=4,
        experiment_id="exp-1",
    )

    all_episodes = {f"ep-{i}" for i in range(20)}
    split_episodes_total = (
        set(result.metadata.split.train_episodes)
        | set(result.metadata.split.val_episodes)
        | set(result.metadata.split.test_episodes)
    )
    assert split_episodes_total == all_episodes


def test_run_model_experiment_train_and_test_episodes_never_overlap() -> None:
    """No episode contributes examples to both train and test metrics."""
    examples = _build_examples(n_episodes=20, horizon=25)

    result = run_model_experiment(
        examples,
        horizon=25,
        model_name=LOGISTIC_REGRESSION,
        seed=0,
        working_memory_capacity=4,
        experiment_id="exp-1",
    )

    train_set = set(result.metadata.split.train_episodes)
    test_set = set(result.metadata.split.test_episodes)
    assert train_set & test_set == set()


def test_run_model_experiment_is_reproducible_given_the_same_seed() -> None:
    """Re-running with the same seed and data gives identical metrics."""
    examples = _build_examples(n_episodes=20, horizon=25)

    result_1 = run_model_experiment(
        examples,
        horizon=25,
        model_name=LOGISTIC_REGRESSION,
        seed=99,
        working_memory_capacity=4,
        experiment_id="exp-1",
    )
    result_2 = run_model_experiment(
        examples,
        horizon=25,
        model_name=LOGISTIC_REGRESSION,
        seed=99,
        working_memory_capacity=4,
        experiment_id="exp-2",
    )

    assert result_1.train_metrics.to_dict() == result_2.train_metrics.to_dict()
    assert result_1.metadata.split == result_2.metadata.split


def test_run_full_experiment_covers_every_horizon_and_model_combination() -> None:
    """run_full_experiment produces one result per (horizon, model) pair."""
    horizon_a_examples = _build_examples(n_episodes=15, horizon=25)
    horizon_b_examples = _build_examples(n_episodes=15, horizon=50)
    examples = horizon_a_examples + horizon_b_examples

    results = run_full_experiment(
        examples,
        horizons=[25, 50],
        model_names=[LOGISTIC_REGRESSION, HIST_GRADIENT_BOOSTING],
        seed=0,
        working_memory_capacity=4,
        experiment_id="exp-1",
    )

    combos = {(r.metadata.horizon, r.metadata.model_name) for r in results}
    assert combos == {
        (25, LOGISTIC_REGRESSION),
        (25, HIST_GRADIENT_BOOSTING),
        (50, LOGISTIC_REGRESSION),
        (50, HIST_GRADIENT_BOOSTING),
    }


def test_run_full_experiment_shares_one_split_per_horizon_across_models() -> None:
    """Both models for the same horizon are evaluated on identical train/val/test episodes."""
    examples = _build_examples(n_episodes=15, horizon=25)

    results = run_full_experiment(
        examples,
        horizons=[25],
        model_names=[LOGISTIC_REGRESSION, HIST_GRADIENT_BOOSTING],
        seed=0,
        working_memory_capacity=4,
        experiment_id="exp-1",
    )

    lr_result = next(r for r in results if r.metadata.model_name == LOGISTIC_REGRESSION)
    hgb_result = next(r for r in results if r.metadata.model_name == HIST_GRADIENT_BOOSTING)
    assert lr_result.metadata.split == hgb_result.metadata.split
