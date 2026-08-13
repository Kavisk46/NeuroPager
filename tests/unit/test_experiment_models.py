"""Unit tests for :mod:`neuropager.experiment.models`."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from neuropager.experiment.models import (
    HIST_GRADIENT_BOOSTING,
    LOGISTIC_REGRESSION,
    MODEL_NAMES,
    make_hist_gradient_boosting,
    make_logistic_regression,
    make_model,
    model_config,
)


def _toy_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 13))
    y = (x[:, 0] > 0).astype(int)
    return x, y


def test_make_logistic_regression_returns_scaler_plus_classifier() -> None:
    """The logistic regression pipeline has a scaler step and a classifier step."""
    pipeline = make_logistic_regression(seed=0)

    assert [name for name, _ in pipeline.steps] == ["scaler", "classifier"]
    assert isinstance(pipeline.named_steps["classifier"], LogisticRegression)


def test_make_hist_gradient_boosting_returns_classifier_only() -> None:
    """The HGB pipeline has just a classifier step (no scaling needed)."""
    pipeline = make_hist_gradient_boosting(seed=0)

    assert [name for name, _ in pipeline.steps] == ["classifier"]
    assert isinstance(pipeline.named_steps["classifier"], HistGradientBoostingClassifier)


def test_make_model_dispatches_by_name() -> None:
    """make_model(name, seed) builds the same kind of pipeline as the direct factories."""
    lr = make_model(LOGISTIC_REGRESSION, seed=0)
    hgb = make_model(HIST_GRADIENT_BOOSTING, seed=0)

    assert isinstance(lr.named_steps["classifier"], LogisticRegression)
    assert isinstance(hgb.named_steps["classifier"], HistGradientBoostingClassifier)


def test_make_model_rejects_unknown_name() -> None:
    """make_model raises for an unrecognized model name."""
    with pytest.raises(ValueError):
        make_model("not_a_real_model", seed=0)


def test_model_config_rejects_unknown_name() -> None:
    """model_config raises for an unrecognized model name."""
    with pytest.raises(ValueError):
        model_config("not_a_real_model", seed=0)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_config_is_json_serializable(name: str) -> None:
    """Every model's config dict round-trips through json.dumps."""
    import json

    config = model_config(name, seed=0)

    json.dumps(config)  # must not raise


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_fits_and_predicts_probabilities(name: str) -> None:
    """Both models can fit on toy data and produce valid probability output."""
    x, y = _toy_data()
    model = make_model(name, seed=0)

    model.fit(x, y)
    proba = model.predict_proba(x)

    assert proba.shape == (40, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all()
    assert (proba <= 1).all()


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_training_is_reproducible_given_the_same_seed(name: str) -> None:
    """Training the same model twice with the same seed and data gives the same predictions."""
    x, y = _toy_data()

    model_1 = make_model(name, seed=123)
    model_1.fit(x, y)
    proba_1 = model_1.predict_proba(x)

    model_2 = make_model(name, seed=123)
    model_2.fit(x, y)
    proba_2 = model_2.predict_proba(x)

    np.testing.assert_allclose(proba_1, proba_2, rtol=1e-10, atol=1e-12)
