"""Top-level orchestration for one Memory Utility Model experiment.

Ties together: episode split -> feature matrix -> model training -> metric
evaluation -> (optionally) online policy simulation, once per (horizon,
model) combination. This module contains no new modeling logic of its
own; it only sequences the other modules in this package.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sklearn.pipeline import Pipeline

from neuropager.dataset.schema import DatasetExample
from neuropager.experiment.metrics import ClassificationMetrics, compute_metrics
from neuropager.experiment.models import make_model, model_config
from neuropager.experiment.preprocessing import build_feature_matrix
from neuropager.experiment.reproducibility import ExperimentMetadata, build_experiment_metadata
from neuropager.experiment.split import EpisodeSplit, filter_examples_by_episodes, split_episodes

_EMPTY_METRICS = ClassificationMetrics(
    n_examples=0,
    n_positive=0,
    n_negative=0,
    accuracy=0.0,
    precision=0.0,
    recall=0.0,
    f1=0.0,
    roc_auc=None,
    pr_auc=None,
    brier_score=0.0,
    calibration_curve=(),
)


@dataclass(frozen=True)
class ModelHorizonResult:
    """The full result of training and evaluating one model for one horizon.

    Attributes:
        metadata: Reproducibility metadata for this run.
        train_metrics: Metrics on the training split.
        val_metrics: Metrics on the validation split (may be the empty
            placeholder if the split has zero examples).
        test_metrics: Metrics on the test split (same caveat).
        model: The fitted pipeline, kept for optional downstream use (e.g.
            building a :class:`~neuropager.experiment.policy.LearnedUtilityPolicy`).
    """

    metadata: ExperimentMetadata
    train_metrics: ClassificationMetrics
    val_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics
    model: Pipeline

    def to_dict(self) -> dict[str, Any]:
        """Return this result as a plain, JSON-serializable dict (excluding the model).

        Returns:
            A dict with ``metadata``/``train_metrics``/``val_metrics``/
            ``test_metrics`` each flattened via their own ``to_dict()``.
        """
        return {
            "metadata": self.metadata.to_dict(),
            "train_metrics": self.train_metrics.to_dict(),
            "val_metrics": self.val_metrics.to_dict(),
            "test_metrics": self.test_metrics.to_dict(),
        }


def run_model_experiment(
    examples: Sequence[DatasetExample],
    horizon: int,
    model_name: str,
    seed: int,
    working_memory_capacity: int,
    experiment_id: str,
    split: EpisodeSplit | None = None,
) -> ModelHorizonResult:
    """Train and evaluate one model for one horizon.

    Args:
        examples: The full example set (any horizon; this function filters
            to ``horizon`` itself).
        horizon: Which horizon's examples to use.
        model_name: One of :data:`~neuropager.experiment.models.MODEL_NAMES`.
        seed: Random seed for the episode split and the model.
        working_memory_capacity: Recorded in metadata for provenance.
        experiment_id: A human-chosen identifier for this run.
        split: An existing :class:`~neuropager.experiment.split.EpisodeSplit`
            to reuse (so multiple models/horizons in one experiment can
            share a split); if omitted, one is computed from this
            horizon's own episode IDs.

    Returns:
        The resulting :class:`ModelHorizonResult`.

    Raises:
        ValueError: If no examples exist for ``horizon``, or the training
            split is empty.
    """
    horizon_examples = [example for example in examples if example.horizon == horizon]
    if not horizon_examples:
        raise ValueError(f"no examples found for horizon={horizon}")

    if split is None:
        episode_ids = sorted({example.episode_id for example in horizon_examples})
        split = split_episodes(episode_ids, seed=seed)

    train_examples = filter_examples_by_episodes(horizon_examples, split.train_episodes)
    val_examples = filter_examples_by_episodes(horizon_examples, split.val_episodes)
    test_examples = filter_examples_by_episodes(horizon_examples, split.test_episodes)

    if not train_examples:
        raise ValueError(
            f"training split has zero examples for horizon={horizon}; "
            "provide more episodes or adjust split fractions"
        )

    x_train, y_train, feature_order = build_feature_matrix(train_examples)
    model = make_model(model_name, seed)
    model.fit(x_train, y_train)

    def evaluate(split_examples: list[DatasetExample]) -> ClassificationMetrics:
        if not split_examples:
            return _EMPTY_METRICS
        x, y, _ = build_feature_matrix(split_examples)
        proba = model.predict_proba(x)[:, 1]
        return compute_metrics(y, proba)

    metadata = build_experiment_metadata(
        experiment_id=experiment_id,
        seed=seed,
        model_name=model_name,
        model_config=model_config(model_name, seed),
        feature_order=feature_order,
        horizon=horizon,
        split=split,
        working_memory_capacity=working_memory_capacity,
    )

    return ModelHorizonResult(
        metadata=metadata,
        train_metrics=evaluate(train_examples),
        val_metrics=evaluate(val_examples),
        test_metrics=evaluate(test_examples),
        model=model,
    )


def run_full_experiment(
    examples: Sequence[DatasetExample],
    horizons: Sequence[int],
    model_names: Sequence[str],
    seed: int,
    working_memory_capacity: int,
    experiment_id: str,
) -> list[ModelHorizonResult]:
    """Train and evaluate every (horizon, model) combination.

    Each horizon gets its own episode split (computed once, from that
    horizon's own examples) shared across all models evaluated for that
    horizon, so different models for the same horizon are compared on
    identical train/val/test episodes.

    Args:
        examples: The full example set across all horizons.
        horizons: Which horizons to run.
        model_names: Which models to run for each horizon.
        seed: Random seed for splitting and model training.
        working_memory_capacity: Recorded in metadata for provenance.
        experiment_id: A human-chosen identifier for this run.

    Returns:
        One :class:`ModelHorizonResult` per (horizon, model) combination,
        ordered by horizon then model.
    """
    results: list[ModelHorizonResult] = []
    for horizon in horizons:
        horizon_examples = [example for example in examples if example.horizon == horizon]
        if not horizon_examples:
            raise ValueError(f"no examples found for horizon={horizon}")
        episode_ids = sorted({example.episode_id for example in horizon_examples})
        split = split_episodes(episode_ids, seed=seed)

        for model_name in model_names:
            results.append(
                run_model_experiment(
                    examples,
                    horizon,
                    model_name,
                    seed,
                    working_memory_capacity,
                    experiment_id,
                    split=split,
                )
            )
    return results
