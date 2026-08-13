"""Unit tests for :mod:`neuropager.experiment.reproducibility`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuropager.experiment.reproducibility import build_experiment_metadata, write_csv, write_json
from neuropager.experiment.split import split_episodes


def _sample_split() -> object:
    return split_episodes(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"], seed=0)


def test_build_experiment_metadata_includes_versions_and_timestamp() -> None:
    """Metadata records neuropager/sklearn versions and a timestamp."""
    metadata = build_experiment_metadata(
        experiment_id="exp-1",
        seed=0,
        model_name="logistic_regression",
        model_config={"model": "logistic_regression"},
        feature_order=("recency", "frequency"),
        horizon=25,
        split=_sample_split(),
        working_memory_capacity=4,
    )

    assert metadata.neuropager_version
    assert metadata.sklearn_version
    assert metadata.generated_at


def test_metadata_to_dict_is_json_serializable() -> None:
    """to_dict() output round-trips through json.dumps/json.loads."""
    metadata = build_experiment_metadata(
        experiment_id="exp-1",
        seed=0,
        model_name="logistic_regression",
        model_config={"model": "logistic_regression"},
        feature_order=("recency", "frequency"),
        horizon=25,
        split=_sample_split(),
        working_memory_capacity=4,
    )

    encoded = json.dumps(metadata.to_dict())
    decoded = json.loads(encoded)

    assert decoded["experiment_id"] == "exp-1"
    assert decoded["horizon"] == 25
    assert decoded["split"]["seed"] == 0


def test_write_json_creates_parent_directories(tmp_path: Path) -> None:
    """write_json creates missing parent directories."""
    path = tmp_path / "nested" / "results.json"

    write_json({"a": 1}, path)

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}


def test_write_json_uses_deterministic_key_order(tmp_path: Path) -> None:
    """write_json sorts keys, so identical data produces byte-identical output."""
    path_1 = tmp_path / "run1.json"
    path_2 = tmp_path / "run2.json"

    write_json({"z": 1, "a": 2}, path_1)
    write_json({"a": 2, "z": 1}, path_2)

    assert path_1.read_bytes() == path_2.read_bytes()


def test_write_csv_creates_parent_directories_and_a_header(tmp_path: Path) -> None:
    """write_csv creates missing parent directories and writes a header row."""
    path = tmp_path / "nested" / "results.csv"
    rows = [{"policy": "LRU", "faults": 3}, {"policy": "FIFO", "faults": 5}]

    write_csv(rows, path)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "policy,faults"
    assert lines[1] == "LRU,3"
    assert lines[2] == "FIFO,5"


def test_write_csv_rejects_empty_rows(tmp_path: Path) -> None:
    """write_csv raises rather than silently writing an empty/headerless file."""
    with pytest.raises(ValueError):
        write_csv([], tmp_path / "empty.csv")
