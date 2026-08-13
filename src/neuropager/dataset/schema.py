"""The dataset row schema: one training example per (page, decision, horizon).

A :class:`DatasetExample` represents a single candidate for eviction at a
single capacity-triggered decision point in a single episode, labeled for
one horizon. If a decision point had 3 resident candidates and the
pipeline is run for 5 horizons, that one decision point produces 15
examples (one per candidate per horizon).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neuropager.dataset.features import Features

_FEATURE_PREFIX = "feature_"


@dataclass(frozen=True)
class DatasetExample:
    """One training example: a candidate page at an eviction decision point.

    Attributes:
        episode_id: The episode this example was drawn from.
        decision_tick: The tick of the eviction decision this candidate
            was considered at.
        page_id: The candidate page.
        policy: Class name of the policy active at this decision (metadata
            only — the label does not depend on it).
        working_memory_capacity: Working memory's configured capacity at
            this decision.
        eviction_reason: Why this decision point occurred. Always
            ``"capacity"`` in this milestone (see
            :mod:`neuropager.dataset.generator`), preserved explicitly so
            the field is not silently assumed elsewhere.
        horizon: The reuse-window size, in ticks, this example is labeled
            for.
        features: The candidate's feature vector as of ``decision_tick``.
        label: ``1`` if the candidate is reused within ``horizon`` ticks
            after ``decision_tick``, else ``0``.
    """

    episode_id: str
    decision_tick: int
    page_id: str
    policy: str
    working_memory_capacity: int
    eviction_reason: str
    horizon: int
    features: Features
    label: int

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten this example into a single-level dict.

        Feature fields are prefixed with ``feature_`` so a dataset loaded
        into pandas can select them with a simple ``filter(like=...)``
        without any other preprocessing.

        Returns:
            A JSON-serializable, pandas-ready dict.
        """
        flat: dict[str, Any] = {
            "episode_id": self.episode_id,
            "decision_tick": self.decision_tick,
            "page_id": self.page_id,
            "policy": self.policy,
            "working_memory_capacity": self.working_memory_capacity,
            "eviction_reason": self.eviction_reason,
            "horizon": self.horizon,
            "label": self.label,
        }
        for name, value in self.features.to_dict().items():
            flat[f"{_FEATURE_PREFIX}{name}"] = value
        return flat

    @classmethod
    def from_flat_dict(cls, data: dict[str, Any]) -> DatasetExample:
        """Reconstruct a :class:`DatasetExample` from :meth:`to_flat_dict` output.

        Args:
            data: A dict previously produced by :meth:`to_flat_dict`.

        Returns:
            The equivalent :class:`DatasetExample`.
        """
        feature_kwargs = {
            key[len(_FEATURE_PREFIX) :]: value
            for key, value in data.items()
            if key.startswith(_FEATURE_PREFIX)
        }
        return cls(
            episode_id=data["episode_id"],
            decision_tick=data["decision_tick"],
            page_id=data["page_id"],
            policy=data["policy"],
            working_memory_capacity=data["working_memory_capacity"],
            eviction_reason=data["eviction_reason"],
            horizon=data["horizon"],
            features=Features(**feature_kwargs),
            label=data["label"],
        )
