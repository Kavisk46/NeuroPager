"""Experiment metadata: everything needed to reproduce one experiment run.

Every field here is either a deterministic input to the experiment (seed,
model config, feature list, horizon, split) or plain provenance (versions,
timestamp) — never a result. Results live in
:class:`~neuropager.experiment.metrics.ClassificationMetrics` and
:class:`~neuropager.experiment.simulation.SimulationResult`, recorded
alongside this metadata by :mod:`neuropager.experiment.runner`.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sklearn

import neuropager
from neuropager.experiment.split import EpisodeSplit


@dataclass(frozen=True)
class ExperimentMetadata:
    """Everything needed to reproduce one experiment run.

    Attributes:
        experiment_id: A human-chosen identifier for this run.
        seed: The random seed used throughout (split, model, Random
            policy).
        model_name: Which model this metadata describes (see
            :data:`~neuropager.experiment.models.MODEL_NAMES`).
        model_config: The model's hyperparameter configuration (see
            :func:`~neuropager.experiment.models.model_config`).
        feature_order: The exact, ordered feature list the model was
            trained on.
        horizon: The reuse horizon this run used.
        split: The episode-level train/validation/test split.
        working_memory_capacity: Working memory capacity used for both
            dataset generation and policy simulation.
        neuropager_version: :data:`neuropager.__version__` at run time.
        sklearn_version: :data:`sklearn.__version__` at run time.
        generated_at: ISO-8601 UTC timestamp of when this metadata was
            created. Provenance only — never used by any feature, label,
            or metric computation, which remain purely tick-based.
    """

    experiment_id: str
    seed: int
    model_name: str
    model_config: dict[str, Any]
    feature_order: tuple[str, ...]
    horizon: int
    split: EpisodeSplit
    working_memory_capacity: int
    neuropager_version: str
    sklearn_version: str
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        """Return this metadata as a plain, JSON-serializable dict.

        Returns:
            A dict including the flattened split metadata.
        """
        return {
            "experiment_id": self.experiment_id,
            "seed": self.seed,
            "model_name": self.model_name,
            "model_config": self.model_config,
            "feature_order": list(self.feature_order),
            "horizon": self.horizon,
            "split": self.split.to_dict(),
            "working_memory_capacity": self.working_memory_capacity,
            "neuropager_version": self.neuropager_version,
            "sklearn_version": self.sklearn_version,
            "generated_at": self.generated_at,
        }


def build_experiment_metadata(
    experiment_id: str,
    seed: int,
    model_name: str,
    model_config: dict[str, Any],
    feature_order: tuple[str, ...],
    horizon: int,
    split: EpisodeSplit,
    working_memory_capacity: int,
) -> ExperimentMetadata:
    """Construct :class:`ExperimentMetadata`, filling in version/timestamp provenance.

    Args:
        experiment_id: A human-chosen identifier for this run.
        seed: The random seed used throughout.
        model_name: Which model this metadata describes.
        model_config: The model's hyperparameter configuration.
        feature_order: The exact, ordered feature list used.
        horizon: The reuse horizon this run used.
        split: The episode-level train/validation/test split.
        working_memory_capacity: Working memory capacity used.

    Returns:
        The populated :class:`ExperimentMetadata`.
    """
    return ExperimentMetadata(
        experiment_id=experiment_id,
        seed=seed,
        model_name=model_name,
        model_config=model_config,
        feature_order=feature_order,
        horizon=horizon,
        split=split,
        working_memory_capacity=working_memory_capacity,
        neuropager_version=neuropager.__version__,
        sklearn_version=sklearn.__version__,
        generated_at=datetime.now(UTC).isoformat(),
    )


def write_json(data: dict[str, Any], path: Path) -> None:
    """Write a plain dict to disk as pretty-printed, deterministic-key-order JSON.

    Args:
        data: The dict to write.
        path: Destination file. Parent directories are created if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write a list of flat dicts to disk as a CSV file.

    Args:
        rows: The rows to write; all rows must share the same keys.
        path: Destination file. Parent directories are created if needed.

    Raises:
        ValueError: If ``rows`` is empty.
    """
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
