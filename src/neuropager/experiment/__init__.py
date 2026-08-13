"""The first Memory Utility Model experiment: the smallest rigorous pipeline.

Trains two baseline classifiers (logistic regression, histogram gradient
boosting) on `neuropager.dataset` output, evaluates them with an
imbalance-aware metric suite, wraps a fitted model as an online
:class:`~neuropager.policies.base.PageReplacementPolicy`, and compares it
against FIFO/LRU/LFU/Random/Belady's-MIN on identical workloads.

This is a scientific instrument, not a finished result — see
``docs/model.md`` for what it does and does not yet establish, and for the
specific, named gaps between offline training features and online serving
features.
"""

from __future__ import annotations

from neuropager.experiment.metrics import ClassificationMetrics, compute_metrics
from neuropager.experiment.models import (
    HIST_GRADIENT_BOOSTING,
    LOGISTIC_REGRESSION,
    MODEL_NAMES,
    make_model,
    model_config,
)
from neuropager.experiment.policy import LearnedUtilityPolicy
from neuropager.experiment.preprocessing import build_feature_matrix
from neuropager.experiment.reproducibility import (
    ExperimentMetadata,
    build_experiment_metadata,
    write_csv,
    write_json,
)
from neuropager.experiment.runner import (
    ModelHorizonResult,
    run_full_experiment,
    run_model_experiment,
)
from neuropager.experiment.simulation import (
    SimulationResult,
    run_baseline_comparison,
    run_policy_simulation,
)
from neuropager.experiment.split import (
    DEFAULT_TEST_FRACTION,
    DEFAULT_TRAIN_FRACTION,
    DEFAULT_VAL_FRACTION,
    EpisodeSplit,
    filter_examples_by_episodes,
    split_episodes,
)

__all__ = [
    "DEFAULT_TEST_FRACTION",
    "DEFAULT_TRAIN_FRACTION",
    "DEFAULT_VAL_FRACTION",
    "HIST_GRADIENT_BOOSTING",
    "LOGISTIC_REGRESSION",
    "MODEL_NAMES",
    "ClassificationMetrics",
    "EpisodeSplit",
    "ExperimentMetadata",
    "LearnedUtilityPolicy",
    "ModelHorizonResult",
    "SimulationResult",
    "build_experiment_metadata",
    "build_feature_matrix",
    "compute_metrics",
    "filter_examples_by_episodes",
    "make_model",
    "model_config",
    "run_baseline_comparison",
    "run_full_experiment",
    "run_model_experiment",
    "run_policy_simulation",
    "split_episodes",
    "write_csv",
    "write_json",
]
