"""Unit tests for :mod:`neuropager.experiment.metrics`."""

from __future__ import annotations

import math

import numpy as np

from neuropager.experiment.metrics import compute_metrics


def test_perfect_predictions_score_perfectly() -> None:
    """Perfectly separated, confident predictions yield accuracy/F1/AUC of 1.0."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred_proba = np.array([0.01, 0.02, 0.03, 0.98, 0.97, 0.99])

    metrics = compute_metrics(y_true, y_pred_proba)

    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.roc_auc == 1.0
    assert metrics.pr_auc == 1.0
    assert metrics.brier_score < 0.01


def test_worst_case_predictions_score_at_the_floor() -> None:
    """Confidently wrong predictions yield accuracy/F1/AUC of 0.0."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred_proba = np.array([0.99, 0.98, 0.97, 0.02, 0.01, 0.03])

    metrics = compute_metrics(y_true, y_pred_proba)

    assert metrics.accuracy == 0.0
    assert metrics.roc_auc == 0.0


def test_metrics_report_class_counts() -> None:
    """n_examples/n_positive/n_negative match the input exactly."""
    y_true = np.array([0, 0, 1])
    y_pred_proba = np.array([0.1, 0.2, 0.9])

    metrics = compute_metrics(y_true, y_pred_proba)

    assert metrics.n_examples == 3
    assert metrics.n_positive == 1
    assert metrics.n_negative == 2


def test_single_class_ground_truth_disables_auc_metrics_gracefully() -> None:
    """ROC-AUC and PR-AUC are None (not a crash) when only one class is present."""
    y_true = np.array([0, 0, 0, 0])
    y_pred_proba = np.array([0.1, 0.2, 0.3, 0.4])

    metrics = compute_metrics(y_true, y_pred_proba)

    assert metrics.roc_auc is None
    assert metrics.pr_auc is None
    # Non-ranking metrics must still be computed, not skipped.
    assert metrics.accuracy == 1.0  # all predicted 0 at threshold 0.5, all true 0


def test_precision_recall_are_zero_not_a_crash_with_no_positive_predictions() -> None:
    """zero_division=0 avoids sklearn warnings/crashes when precision is undefined."""
    y_true = np.array([0, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.2, 0.3, 0.4])  # nothing crosses the 0.5 threshold

    metrics = compute_metrics(y_true, y_pred_proba)

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


def test_brier_score_is_never_nan_or_infinite() -> None:
    """Brier score is always a finite number for well-formed inputs."""
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred_proba = np.array([0.4, 0.6, 0.5, 0.5, 0.5])

    metrics = compute_metrics(y_true, y_pred_proba)

    assert math.isfinite(metrics.brier_score)


def test_calibration_curve_is_empty_when_too_few_examples() -> None:
    """calibration_curve degrades to empty rather than raising for tiny inputs."""
    y_true = np.array([0, 1])
    y_pred_proba = np.array([0.2, 0.8])

    metrics = compute_metrics(y_true, y_pred_proba, calibration_bins=5)

    assert metrics.calibration_curve == ()


def test_to_dict_is_json_serializable() -> None:
    """ClassificationMetrics.to_dict() round-trips through json.dumps."""
    import json

    y_true = np.array([0, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.4, 0.6, 0.9])

    metrics = compute_metrics(y_true, y_pred_proba)

    json.dumps(metrics.to_dict())  # must not raise


def test_custom_threshold_changes_hard_predictions() -> None:
    """A stricter decision threshold can change accuracy for the same probabilities."""
    y_true = np.array([0, 0, 1, 1])
    y_pred_proba = np.array([0.3, 0.3, 0.55, 0.55])

    default = compute_metrics(y_true, y_pred_proba, threshold=0.5)
    strict = compute_metrics(y_true, y_pred_proba, threshold=0.6)

    assert default.accuracy == 1.0
    assert strict.accuracy < 1.0
