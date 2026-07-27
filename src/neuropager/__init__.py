"""NeuroPager: a neuro-symbolic virtual memory manager for lifelong LLM agents.

NeuroPager applies an operating-system-inspired virtual memory model
(working memory, page tables, page faults, pluggable replacement policies)
to LLM agent memory, layered with a neuro-symbolic hybrid retrieval system
(dense vector search + symbolic knowledge graph).

This package currently exposes typed interfaces and stubs only. See
``docs/roadmap.md`` in the repository root for the implementation plan.

Example:
    >>> import neuropager
    >>> neuropager.__version__
    '0.1.0'
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
