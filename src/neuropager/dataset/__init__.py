"""Offline dataset-generation pipeline for the Memory Utility Model.

Realizes: JSONL trace -> trace replay -> resident-page state
reconstruction -> feature extraction -> future-reuse labels -> training
dataset. See ``docs/dataset.md`` for the full research documentation
(feature definitions, label semantics, and leakage-prevention argument).

This package does not implement any learning — it only turns a completed
:mod:`neuropager.trace` episode into labeled, tabular training examples
for a model to be trained elsewhere.
"""

from __future__ import annotations

from neuropager.dataset.features import (
    FEATURE_NAMES,
    LONG_BURST_WINDOW,
    MISSING_HISTORY_SENTINEL,
    RECENT_BURST_WINDOW,
    Features,
    compute_features,
)
from neuropager.dataset.generator import (
    generate_dataset,
    generate_dataset_from_jsonl,
    read_dataset_jsonl,
    read_trace_jsonl,
    write_dataset_jsonl,
)
from neuropager.dataset.labels import DEFAULT_HORIZONS, compute_label
from neuropager.dataset.replay import EvictionDecision, TraceReplay, group_by_episode
from neuropager.dataset.schema import DatasetExample
from neuropager.dataset.validation import DatasetValidationError, validate_dataset

__all__ = [
    "DEFAULT_HORIZONS",
    "FEATURE_NAMES",
    "LONG_BURST_WINDOW",
    "MISSING_HISTORY_SENTINEL",
    "RECENT_BURST_WINDOW",
    "DatasetExample",
    "DatasetValidationError",
    "EvictionDecision",
    "Features",
    "TraceReplay",
    "compute_features",
    "compute_label",
    "generate_dataset",
    "generate_dataset_from_jsonl",
    "group_by_episode",
    "read_dataset_jsonl",
    "read_trace_jsonl",
    "validate_dataset",
    "write_dataset_jsonl",
]
