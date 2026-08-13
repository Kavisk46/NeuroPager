"""Classification metrics for evaluating the Memory Utility Model.

Labels are imbalanced by construction (most resident pages are *not*
reused within a short horizon), so accuracy alone is not trusted here —
every report includes precision/recall/F1, ranking metrics (ROC-AUC,
PR-AUC), and a probability-calibration signal (Brier score plus a small
binned reliability table).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_DECISION_THRESHOLD = 0.5
DEFAULT_CALIBRATION_BINS = 5


@dataclass(frozen=True)
class ClassificationMetrics:
    """Evaluation metrics for one (model, horizon, split) combination.

    Attributes:
        n_examples: Number of examples evaluated.
        n_positive: Number of positive (``label == 1``) examples.
        n_negative: Number of negative (``label == 0``) examples.
        accuracy: Fraction of correct predictions at
            :data:`DEFAULT_DECISION_THRESHOLD`.
        precision: Precision at the same threshold (``0`` if no positive
            predictions were made, rather than raising).
        recall: Recall at the same threshold.
        f1: F1 score at the same threshold.
        roc_auc: Area under the ROC curve, or ``None`` if undefined (only
            one class present in ``y_true``).
        pr_auc: Area under the precision-recall curve (average precision),
            or ``None`` under the same condition as ``roc_auc``.
        brier_score: Mean squared error between predicted probabilities
            and true labels — a proper scoring rule that rewards
            calibration, not just ranking.
        calibration_curve: ``(mean_predicted, mean_observed)`` pairs from
            :func:`sklearn.calibration.calibration_curve`, or an empty
            tuple if it could not be computed (e.g. too few examples).
    """

    n_examples: int
    n_positive: int
    n_negative: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    pr_auc: float | None
    brier_score: float
    calibration_curve: tuple[tuple[float, float], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return these metrics as a plain, JSON-serializable dict.

        Returns:
            A dict with ``calibration_curve`` flattened to a list of
            ``[predicted, observed]`` pairs.
        """
        return {
            "n_examples": self.n_examples,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "brier_score": self.brier_score,
            "calibration_curve": [list(point) for point in self.calibration_curve],
        }


def compute_metrics(
    y_true: NDArray[np.int64],
    y_pred_proba: NDArray[np.float64],
    threshold: float = DEFAULT_DECISION_THRESHOLD,
    calibration_bins: int = DEFAULT_CALIBRATION_BINS,
) -> ClassificationMetrics:
    """Compute the full metric suite for one set of predictions.

    Args:
        y_true: Ground-truth labels (``0``/``1``).
        y_pred_proba: Predicted ``P(label == 1)`` for each example.
        threshold: Decision threshold used for accuracy/precision/recall/F1.
        calibration_bins: Number of bins for the calibration curve.

    Returns:
        The computed :class:`ClassificationMetrics`.
    """
    y_pred = (y_pred_proba >= threshold).astype(np.int64)
    n_positive = int(np.sum(y_true == 1))
    n_negative = int(np.sum(y_true == 0))

    if n_positive > 0 and n_negative > 0:
        roc_auc: float | None = float(roc_auc_score(y_true, y_pred_proba))
        pr_auc: float | None = float(average_precision_score(y_true, y_pred_proba))
    else:
        roc_auc = None
        pr_auc = None

    curve: tuple[tuple[float, float], ...] = ()
    if n_positive > 0 and n_negative > 0 and len(y_true) >= calibration_bins:
        observed, predicted = calibration_curve(
            y_true, y_pred_proba, n_bins=calibration_bins, strategy="quantile"
        )
        curve = tuple(
            (float(pred), float(obs)) for pred, obs in zip(predicted, observed, strict=True)
        )

    return ClassificationMetrics(
        n_examples=len(y_true),
        n_positive=n_positive,
        n_negative=n_negative,
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        brier_score=float(brier_score_loss(y_true, y_pred_proba)),
        calibration_curve=curve,
    )
