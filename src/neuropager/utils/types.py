"""Shared type aliases and data structures used across NeuroPager.

These types define the common vocabulary for memory addressing and page
content, so that ``core``, ``memory``, ``retrieval``, and ``policies``
modules can interoperate without depending on each other's internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, NewType

MemoryKey = NewType("MemoryKey", str)
"""Logical identifier for a single unit of memory, stable across tiers."""

Embedding = list[float]
"""Dense vector representation of a memory unit's content."""


class MemoryTier(StrEnum):
    """The physical tier currently holding a given memory page."""

    WORKING = "working"
    VECTOR_STORE = "vector_store"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    DISK = "disk"


@dataclass
class MemoryPage:
    """A single unit of memory tracked by the page table.

    Attributes:
        key: Logical identifier for this memory unit.
        content: The raw content payload (text, structured data, etc.).
        tier: The physical tier currently holding this page.
        created_at: Timestamp the page was first created.
        last_accessed_at: Timestamp of the most recent access, used by
            recency-based replacement policies.
        access_count: Number of times this page has been accessed, used by
            frequency-based replacement policies.
        metadata: Free-form metadata (source, provenance, tags, etc.).
    """

    key: MemoryKey
    content: Any
    tier: MemoryTier
    created_at: datetime
    last_accessed_at: datetime
    access_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """A single ranked result returned by the hybrid retriever.

    Attributes:
        key: Logical identifier of the retrieved memory unit.
        score: Fused relevance score (higher is more relevant).
        source: Which retrieval backend(s) contributed this result.
    """

    key: MemoryKey
    score: float
    source: str
