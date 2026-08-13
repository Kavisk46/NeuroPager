"""Top-level orchestration: JSONL trace in, validated dataset examples out.

Realizes the full pipeline: JSONL trace -> trace replay -> resident-page
state reconstruction -> feature extraction -> future-reuse labels ->
training dataset.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from neuropager.dataset.features import compute_features
from neuropager.dataset.labels import DEFAULT_HORIZONS, compute_label
from neuropager.dataset.replay import TraceReplay, group_by_episode
from neuropager.dataset.schema import DatasetExample
from neuropager.dataset.validation import validate_dataset
from neuropager.trace.events import TraceEvent


def generate_dataset(
    events: Sequence[TraceEvent],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> list[DatasetExample]:
    """Generate labeled dataset examples from one or more episodes of trace events.

    Examples are generated only at capacity-triggered eviction decisions
    (explicit evictions are not included in this milestone). For every
    such decision, every resident candidate produces one example per
    horizon. The resulting set is validated before being returned (see
    :func:`~neuropager.dataset.validation.validate_dataset`).

    Args:
        events: Trace events, from one or more episodes, in their original
            (possibly interleaved) order.
        horizons: The reuse-window sizes to label each candidate for.

    Returns:
        Validated :class:`~neuropager.dataset.schema.DatasetExample` rows.

    Raises:
        ValueError: If ``horizons`` is empty or contains a non-positive
            value, or if the underlying trace is malformed (see
            :class:`~neuropager.dataset.replay.TraceReplay`).
        neuropager.dataset.validation.DatasetValidationError: If the
            generated examples fail validation.
    """
    if not horizons:
        raise ValueError("at least one horizon must be provided")
    for horizon in horizons:
        if horizon <= 0:
            raise ValueError(f"horizon must be a positive integer, got {horizon}")

    examples: list[DatasetExample] = []

    for episode_id, episode_events in group_by_episode(events).items():
        replay = TraceReplay(episode_events)
        for decision in replay.eviction_decisions(reason="capacity"):
            for page_id in decision.candidate_pages:
                history = replay.access_ticks_up_to(page_id, decision.decision_tick)
                fault_history = replay.fault_ticks_up_to(page_id, decision.decision_tick)
                features = compute_features(history, fault_history, decision.decision_tick)

                # Labels alone may see the future: full_access_ticks is
                # NOT tick-filtered, unlike `history` above.
                full_history = replay.full_access_ticks(page_id)

                for horizon in horizons:
                    label = compute_label(full_history, decision.decision_tick, horizon)
                    examples.append(
                        DatasetExample(
                            episode_id=episode_id,
                            decision_tick=decision.decision_tick,
                            page_id=page_id,
                            policy=decision.policy,
                            working_memory_capacity=decision.working_memory_capacity,
                            eviction_reason=decision.reason,
                            horizon=horizon,
                            features=features,
                            label=label,
                        )
                    )

    validate_dataset(examples)
    return examples


def read_trace_jsonl(path: Path) -> list[TraceEvent]:
    """Read a trace file written by :class:`~neuropager.trace.logger.TraceLogger`.

    Args:
        path: Path to a JSONL file of trace events.

    Returns:
        The parsed events, in file order.
    """
    events: list[TraceEvent] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            events.append(TraceEvent.from_json_dict(json.loads(line)))
    return events


def generate_dataset_from_jsonl(
    path: Path,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> list[DatasetExample]:
    """Convenience entry point: read a trace file and generate its dataset.

    Args:
        path: Path to a JSONL trace file (see :func:`read_trace_jsonl`).
        horizons: The reuse-window sizes to label each candidate for.

    Returns:
        Validated :class:`~neuropager.dataset.schema.DatasetExample` rows.
    """
    return generate_dataset(read_trace_jsonl(path), horizons=horizons)


def write_dataset_jsonl(examples: Sequence[DatasetExample], path: Path) -> None:
    """Write dataset examples as one flat JSON object per line.

    Args:
        examples: The examples to write.
        path: Destination file. Parent directories are created if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_flat_dict(), sort_keys=True))
            handle.write("\n")


def read_dataset_jsonl(path: Path) -> list[DatasetExample]:
    """Read dataset examples previously written by :func:`write_dataset_jsonl`.

    Args:
        path: Path to a JSONL dataset file.

    Returns:
        The parsed examples, in file order.
    """
    examples: list[DatasetExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            examples.append(DatasetExample.from_flat_dict(json.loads(line)))
    return examples
