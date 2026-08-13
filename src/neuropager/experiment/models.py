"""The two baseline classifiers for the first Memory Utility Model experiment.

Both models expose the same interface (``fit(X, y)`` /
``predict_proba(X)``) via :class:`sklearn.pipeline.Pipeline`, so the rest
of the experiment pipeline (training, evaluation, the online policy
adapter) never needs to know which one it is holding.

- **Logistic regression**: a linear baseline. Feature scaling matters for
  linear models, so it is wrapped with a :class:`~sklearn.preprocessing.StandardScaler`
  fit only on the training split.
- **Histogram gradient boosting**: the one nonlinear, tree-based model
  requested, using scikit-learn's built-in
  :class:`~sklearn.ensemble.HistGradientBoostingClassifier` — no scaling
  needed (tree splits are scale-invariant), and no extra dependency beyond
  scikit-learn itself (no XGBoost/LightGBM).

Missing-history values are represented uniformly as
:data:`~neuropager.dataset.features.MISSING_HISTORY_SENTINEL` (``-1.0``)
for *both* models, deliberately not converted to ``NaN`` for one and not
the other — see ``docs/model.md`` for why: it keeps the two models' inputs
directly comparable (same raw numbers), and keeps this first experiment's
preprocessing minimal.
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LOGISTIC_REGRESSION = "logistic_regression"
HIST_GRADIENT_BOOSTING = "hist_gradient_boosting"

MODEL_NAMES: tuple[str, ...] = (LOGISTIC_REGRESSION, HIST_GRADIENT_BOOSTING)


def make_logistic_regression(seed: int) -> Pipeline:
    """Build the logistic regression baseline pipeline.

    Args:
        seed: Random seed for the underlying solver.

    Returns:
        An unfitted ``StandardScaler -> LogisticRegression`` pipeline.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(random_state=seed, max_iter=1000),
            ),
        ]
    )


def make_hist_gradient_boosting(seed: int) -> Pipeline:
    """Build the histogram gradient boosting baseline pipeline.

    Args:
        seed: Random seed for the underlying estimator.

    Returns:
        An unfitted single-step pipeline wrapping
        :class:`~sklearn.ensemble.HistGradientBoostingClassifier`.
    """
    return Pipeline(
        steps=[
            (
                "classifier",
                HistGradientBoostingClassifier(random_state=seed, max_iter=100),
            ),
        ]
    )


def make_model(name: str, seed: int) -> Pipeline:
    """Build a model pipeline by name.

    Args:
        name: One of :data:`MODEL_NAMES`.
        seed: Random seed for the underlying estimator.

    Returns:
        The corresponding unfitted pipeline.

    Raises:
        ValueError: If ``name`` is not a recognized model name.
    """
    if name == LOGISTIC_REGRESSION:
        return make_logistic_regression(seed)
    if name == HIST_GRADIENT_BOOSTING:
        return make_hist_gradient_boosting(seed)
    raise ValueError(f"unknown model name {name!r}; expected one of {MODEL_NAMES}")


def model_config(name: str, seed: int) -> dict[str, object]:
    """Return the hyperparameter configuration for a named model.

    Used to record experiment metadata (see
    :mod:`neuropager.experiment.reproducibility`) independently of any
    live, unfitted estimator object.

    Args:
        name: One of :data:`MODEL_NAMES`.
        seed: Random seed that would be used to construct the model.

    Returns:
        A plain, JSON-serializable dict of the model's configuration.

    Raises:
        ValueError: If ``name`` is not a recognized model name.
    """
    if name == LOGISTIC_REGRESSION:
        return {"model": name, "max_iter": 1000, "random_state": seed, "scaled": True}
    if name == HIST_GRADIENT_BOOSTING:
        return {"model": name, "max_iter": 100, "random_state": seed, "scaled": False}
    raise ValueError(f"unknown model name {name!r}; expected one of {MODEL_NAMES}")
