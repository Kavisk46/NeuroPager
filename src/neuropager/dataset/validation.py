"""Fail-loudly validation for generated dataset examples.

The pipeline should never silently emit malformed research data. Every
check here raises :class:`DatasetValidationError` on the *first* violation
found, with a message specific enough to locate the offending example.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from neuropager.dataset.schema import DatasetExample


class DatasetValidationError(ValueError):
    """Raised when one or more generated dataset examples are invalid."""


def validate_dataset(examples: Sequence[DatasetExample]) -> None:
    """Validate a full set of generated dataset examples.

    Checks, per example: non-empty episode_id, non-negative decision_tick,
    positive horizon, positive working_memory_capacity, label in
    ``{0, 1}``, at least one recorded access (a candidate must have been
    installed via some prior access — zero history indicates a generation
    bug, not a legitimate research example), and every feature value
    finite. Across the whole set: no duplicate
    ``(episode_id, decision_tick, page_id, horizon)`` combinations.

    Args:
        examples: The examples to validate.

    Raises:
        DatasetValidationError: On the first invalid example or duplicate
            found.
    """
    seen: set[tuple[str, int, str, int]] = set()

    for example in examples:
        if not example.episode_id:
            raise DatasetValidationError(f"example has an empty episode_id: {example!r}")

        if example.decision_tick < 0:
            raise DatasetValidationError(
                f"invalid decision_tick {example.decision_tick} in episode "
                f"{example.episode_id!r}"
            )

        if example.horizon <= 0:
            raise DatasetValidationError(
                f"invalid horizon {example.horizon} in episode {example.episode_id!r} "
                f"at tick {example.decision_tick}"
            )

        if example.working_memory_capacity <= 0:
            raise DatasetValidationError(
                f"invalid working_memory_capacity {example.working_memory_capacity} "
                f"in episode {example.episode_id!r}"
            )

        if example.label not in (0, 1):
            raise DatasetValidationError(
                f"label {example.label!r} outside {{0, 1}} for page "
                f"{example.page_id!r} in episode {example.episode_id!r} "
                f"at tick {example.decision_tick}"
            )

        if example.features.frequency < 1.0:
            raise DatasetValidationError(
                f"missing page history: page {example.page_id!r} in episode "
                f"{example.episode_id!r} was a candidate at tick "
                f"{example.decision_tick} with zero recorded accesses"
            )

        for name, value in example.features.to_dict().items():
            if not math.isfinite(value):
                raise DatasetValidationError(
                    f"non-finite feature {name}={value} for page {example.page_id!r} "
                    f"in episode {example.episode_id!r} at tick {example.decision_tick}"
                )

        key = (example.episode_id, example.decision_tick, example.page_id, example.horizon)
        if key in seen:
            raise DatasetValidationError(f"duplicate example: {key}")
        seen.add(key)
