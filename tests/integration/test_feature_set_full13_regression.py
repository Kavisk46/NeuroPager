"""Regression test: feature-set FULL13 must reproduce an original,
pre-feature-set production checkpoint's numbers exactly.

Proves the elapsed_since_fault skew fix's new --feature-set plumbing is a
true no-op for the default case, by rerunning the smallest real
production unit (Protocol A, B_temporal_locality, horizon=25) from
scratch under FULL13 and comparing against the actual, already-committed
checkpoint from the original full campaign
(experiments/generalization-experiment-2/checkpoints/). Skips cleanly if
that checkpoint isn't present (e.g. a fresh clone before the campaign's
checkpoints are restored) rather than failing on a missing fixture.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "generalization_experiment_2.py"
_spec = importlib.util.spec_from_file_location("generalization_experiment_2", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
ge2: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["generalization_experiment_2"] = ge2
_spec.loader.exec_module(ge2)

REPO_ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_CHECKPOINT = (
    REPO_ROOT
    / "experiments"
    / "generalization-experiment-2"
    / "checkpoints"
    / "protocol_a_B_temporal_locality_25.json"
)

METRIC_KEYS = ("roc_auc", "pr_auc", "brier_score", "f1", "precision", "recall", "accuracy")


def test_full13_reproduces_original_b_temporal_locality_horizon25_checkpoint(
    tmp_path: Path,
) -> None:
    if not ORIGINAL_CHECKPOINT.exists():
        pytest.skip(f"original checkpoint not present: {ORIGINAL_CHECKPOINT}")
    original = json.loads(ORIGINAL_CHECKPOINT.read_text(encoding="utf-8"))["result"]

    ge2.run_experiment(
        families=["B_temporal_locality"],
        horizons=[25],
        n_episodes_per_family=ge2.FULL_SPEC_EPISODES_PER_FAMILY,
        random_seeds=ge2.RANDOM_SEEDS,
        experiment_root=tmp_path / "regression",
        experiment_id="regression-full13",
        feature_set="FULL13",
    )

    rerun_path = (
        tmp_path / "regression" / "checkpoints" / "protocol_a_B_temporal_locality_25.json"
    )
    rerun = json.loads(rerun_path.read_text(encoding="utf-8"))["result"]

    assert rerun["selected_model"] == original["selected_model"]
    assert rerun["n_train_examples"] == original["n_train_examples"]
    assert rerun["n_val_examples"] == original["n_val_examples"]
    assert rerun["n_test_examples"] == original["n_test_examples"]
    assert rerun["feature_set_columns"] == list(ge2.FEATURE_NAMES)

    for model_name in ("hist_gradient_boosting", "logistic_regression"):
        orig_info = original["models"][model_name]
        new_info = rerun["models"][model_name]
        assert new_info["status"] == orig_info["status"]
        if orig_info["test_metrics"] is None:
            assert new_info["test_metrics"] is None
            continue
        for key in METRIC_KEYS:
            assert new_info["test_metrics"][key] == pytest.approx(
                orig_info["test_metrics"][key]
            ), f"{model_name}.test_metrics.{key} regressed"
