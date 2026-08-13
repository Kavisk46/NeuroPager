"""Research trace infrastructure for NeuroPager memory-management episodes.

This package records a deterministic, replayable log of everything a
:class:`~neuropager.core.memory_manager.MemoryManager` did during an
episode — accesses, page faults, and evictions — so the resulting traces
can later be used as supervised training data for a learned Memory Utility
Model (see ``docs/research.md``). It does not implement any learning: it
only produces the ground-truth event log that a future model would be
trained and evaluated against.

Every event is stamped only with information available at the moment it
occurred; no component in this package ever looks ahead at what happens
next.
"""

from __future__ import annotations

from neuropager.trace.events import (
    AccessOperation,
    AccessOutcome,
    TraceEvent,
    TraceEventType,
)
from neuropager.trace.logger import TraceLogger

__all__ = [
    "AccessOperation",
    "AccessOutcome",
    "TraceEvent",
    "TraceEventType",
    "TraceLogger",
]
