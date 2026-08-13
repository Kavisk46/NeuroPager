"""Sequential trace replay: reconstructs per-page history from TraceEvents.

:class:`TraceReplay` is the only place in the dataset pipeline that reads
raw :class:`~neuropager.trace.events.TraceEvent` objects. It reconstructs,
purely from the sequence of events, each page's access and fault history
and the pool of resident candidates at every capacity-triggered eviction.

The research clock is :attr:`~neuropager.trace.events.TraceEvent.tick`
exclusively. Nothing here imports or touches
:class:`~neuropager.core.page_table.PageTable`; the dataset pipeline is
designed to run entirely offline, from a serialized trace file, with no
dependency on the live memory-management objects that produced it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from neuropager.trace.events import TraceEvent, TraceEventType


@dataclass(frozen=True)
class EvictionDecision:
    """One eviction decision point reconstructed from the trace.

    Attributes:
        episode_id: The episode this decision belongs to.
        decision_tick: The tick at which the eviction occurred.
        policy: Class name of the policy active at this decision.
        working_memory_capacity: Working memory's configured capacity.
        candidate_pages: Every page resident immediately before the
            eviction (the full pool the policy chose a victim from).
        chosen_victim: The page the classical policy actually evicted.
        reason: Why the eviction happened (``"capacity"`` or ``"explicit"``).
    """

    episode_id: str
    decision_tick: int
    policy: str
    working_memory_capacity: int
    candidate_pages: tuple[str, ...]
    chosen_victim: str
    reason: str


def group_by_episode(events: Sequence[TraceEvent]) -> dict[str, list[TraceEvent]]:
    """Partition a (possibly multi-episode) event sequence by episode_id.

    Args:
        events: Events in original order, from one or more episodes,
            possibly interleaved.

    Returns:
        A dict mapping each episode_id to its events, in their original
        relative order. Iteration order follows each episode's first
        appearance in ``events``.
    """
    grouped: dict[str, list[TraceEvent]] = defaultdict(list)
    for event in events:
        grouped[event.episode_id].append(event)
    return dict(grouped)


class TraceReplay:
    """Replays a single episode's events to reconstruct per-page history.

    Attributes:
        episode_id: The episode this replay was built from.
    """

    def __init__(self, events: Sequence[TraceEvent]) -> None:
        """Replay ``events`` in order, building per-page history.

        Args:
            events: All events for exactly one episode, in emission order
                (as produced by
                :class:`~neuropager.trace.logger.TraceLogger`, or read back
                via :meth:`~neuropager.trace.events.TraceEvent.from_json_dict`).

        Raises:
            ValueError: If ``events`` is empty, spans more than one
                episode, ticks are not non-decreasing, or an EVICTION event
                is missing its ``eviction_reason``.
        """
        if not events:
            raise ValueError("cannot replay an empty event sequence")

        episode_ids = {event.episode_id for event in events}
        if len(episode_ids) > 1:
            raise ValueError(
                "TraceReplay requires events from a single episode; got "
                f"{sorted(episode_ids)}. Use group_by_episode() first."
            )
        self.episode_id = events[0].episode_id

        self._access_ticks: dict[str, list[int]] = defaultdict(list)
        self._fault_ticks: dict[str, list[int]] = defaultdict(list)
        self._decisions: list[EvictionDecision] = []

        last_tick = 0
        for event in events:
            if event.tick < last_tick:
                raise ValueError(
                    f"non-monotonic tick in episode {self.episode_id!r}: "
                    f"tick {event.tick} follows tick {last_tick}"
                )
            last_tick = event.tick

            if event.event_type == TraceEventType.ACCESS:
                self._access_ticks[event.page_id].append(event.tick)
            elif event.event_type == TraceEventType.PAGE_FAULT:
                self._fault_ticks[event.page_id].append(event.tick)
            elif event.event_type == TraceEventType.EVICTION:
                if event.eviction_reason is None:
                    raise ValueError(
                        f"EVICTION event at tick {event.tick} in episode "
                        f"{self.episode_id!r} is missing eviction_reason"
                    )
                self._decisions.append(
                    EvictionDecision(
                        episode_id=event.episode_id,
                        decision_tick=event.tick,
                        policy=event.policy,
                        working_memory_capacity=event.working_memory_capacity,
                        candidate_pages=event.resident_pages,
                        chosen_victim=event.page_id,
                        reason=event.eviction_reason,
                    )
                )

    def access_ticks_up_to(self, page_id: str, tick: int) -> list[int]:
        """Return ``page_id``'s ACCESS ticks at or before ``tick``.

        Args:
            page_id: The page to look up.
            tick: The inclusive upper bound.

        Returns:
            Ascending ticks; empty if the page has no such history.
        """
        return [t for t in self._access_ticks.get(page_id, []) if t <= tick]

    def fault_ticks_up_to(self, page_id: str, tick: int) -> list[int]:
        """Return ``page_id``'s PAGE_FAULT ticks at or before ``tick``.

        Args:
            page_id: The page to look up.
            tick: The inclusive upper bound.

        Returns:
            Ascending ticks; empty if the page has no such history.
        """
        return [t for t in self._fault_ticks.get(page_id, []) if t <= tick]

    def full_access_ticks(self, page_id: str) -> list[int]:
        """Return every ACCESS tick ever recorded for ``page_id`` in this episode.

        Unlike :meth:`access_ticks_up_to`, this includes ticks *after* any
        given decision point. It exists only for label generation (see
        :mod:`neuropager.dataset.labels`), which is explicitly allowed to
        inspect the future because it runs offline, after the episode is
        complete — never for feature computation.

        Args:
            page_id: The page to look up.

        Returns:
            Ascending ticks; empty if the page was never accessed.
        """
        return list(self._access_ticks.get(page_id, []))

    def eviction_decisions(self, reason: str = "capacity") -> list[EvictionDecision]:
        """Return every eviction decision matching ``reason``, in order.

        Args:
            reason: Which eviction reason to include (``"capacity"`` or
                ``"explicit"``).

        Returns:
            Matching :class:`EvictionDecision` records, in trace order.
        """
        return [d for d in self._decisions if d.reason == reason]
