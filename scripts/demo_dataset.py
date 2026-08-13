#!/usr/bin/env python3
"""Generate a tiny demonstration dataset from hand-crafted trace fixtures.

This is a demonstration only -- it does not produce the large training
dataset. It exists to show, concretely, what the offline dataset pipeline
produces: episode/example counts, label balance, feature names, and a
couple of complete example rows.

Usage:
    python scripts/demo_dataset.py
"""

from __future__ import annotations

import json

from neuropager.dataset import FEATURE_NAMES, generate_dataset
from neuropager.trace.events import AccessOperation, AccessOutcome, TraceEvent, TraceEventType


def _access(episode_id: str, tick: int, page_id: str, resident: tuple[str, ...]) -> TraceEvent:
    return TraceEvent(
        episode_id=episode_id,
        tick=tick,
        event_type=TraceEventType.ACCESS,
        page_id=page_id,
        working_memory_capacity=3,
        resident_pages=resident,
        policy="LRUPolicy",
        operation=AccessOperation.GET,
        outcome=AccessOutcome.HIT,
        page_access_count=1,
        page_fault_count=0,
    )


def _eviction(episode_id: str, tick: int, victim: str, resident: tuple[str, ...]) -> TraceEvent:
    return TraceEvent(
        episode_id=episode_id,
        tick=tick,
        event_type=TraceEventType.EVICTION,
        page_id=victim,
        working_memory_capacity=3,
        resident_pages=resident,
        policy="LRUPolicy",
        eviction_reason="capacity",
        page_access_count=1,
        page_fault_count=0,
    )


def build_demo_trace() -> list[TraceEvent]:
    """Build two tiny, hand-crafted episodes with a mix of reuse patterns."""
    episode_1 = [
        _access("demo-ep-1", 1, "a", ()),
        _access("demo-ep-1", 2, "b", ("a",)),
        _access("demo-ep-1", 3, "c", ("a", "b")),
        _eviction("demo-ep-1", 4, "a", ("a", "b", "c")),
        _access("demo-ep-1", 6, "a", ("b", "c")),  # "a" reused shortly after eviction
        _access("demo-ep-1", 30, "d", ("a", "b", "c")),
        _eviction("demo-ep-1", 31, "b", ("a", "b", "c")),  # "b" never reused again
    ]
    episode_2 = [
        _access("demo-ep-2", 1, "x", ()),
        _access("demo-ep-2", 2, "y", ("x",)),
        _access("demo-ep-2", 3, "z", ("x", "y")),
        _eviction("demo-ep-2", 4, "x", ("x", "y", "z")),
        _access("demo-ep-2", 5, "x", ("y", "z")),  # reused immediately
    ]
    return episode_1 + episode_2


def main() -> None:
    """Generate the demo dataset and print a summary."""
    events = build_demo_trace()
    examples = generate_dataset(events, horizons=[5, 25])

    episode_ids = sorted({e.episode_id for e in examples})
    positives = sum(1 for e in examples if e.label == 1)
    negatives = sum(1 for e in examples if e.label == 0)

    print("=== NeuroPager offline dataset demo ===")
    print(f"episodes:            {len(episode_ids)} {episode_ids}")
    print(f"examples:            {len(examples)}")
    print(f"positive labels:     {positives}")
    print(f"negative labels:     {negatives}")
    print(f"feature names ({len(FEATURE_NAMES)}): {list(FEATURE_NAMES)}")
    print()
    print("=== Two complete example rows ===")
    for example in examples[:2]:
        print(json.dumps(example.to_flat_dict(), indent=2, sort_keys=True))
        print()


if __name__ == "__main__":
    main()
