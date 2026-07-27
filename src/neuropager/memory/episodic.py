"""Episodic memory: a time-ordered log of agent experience.

Episodic memory stores raw, timestamped records of what the agent observed,
did, or was told. It is the substrate from which derived representations
(vector embeddings in :mod:`neuropager.memory.vector_store`, symbolic
triples in :mod:`neuropager.memory.knowledge_graph`) are typically
constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from neuropager.utils.types import MemoryKey


@dataclass
class Event:
    """A single episodic record.

    Attributes:
        key: Logical identifier for this event.
        timestamp: When the event occurred.
        content: The event payload (observation, action, outcome, etc.).
        metadata: Free-form metadata (source, tags, etc.).
    """

    key: MemoryKey
    timestamp: datetime
    content: Any
    metadata: dict[str, Any]


class EpisodicMemory:
    """Append-oriented store of timestamped agent experience."""

    def __init__(self) -> None:
        """Initialize an empty episodic memory store."""
        # TODO(neuropager): back this with an ordered store supporting
        # efficient time-range queries.

    def record(self, event: Event) -> None:
        """Append a new event to episodic memory.

        Args:
            event: The event to record.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError

    def query(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> list[Event]:
        """Query events within an optional time range.

        Args:
            start: Inclusive lower bound on event timestamp, if any.
            end: Inclusive upper bound on event timestamp, if any.
            limit: Maximum number of events to return, most recent first.

        Returns:
            A list of matching :class:`Event` records.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError
