"""Core virtual memory substrate: the OS-inspired heart of NeuroPager.

Contains the :class:`~neuropager.core.memory_manager.MemoryManager`
orchestrator and its collaborators: working memory, the page table, and the
page fault handler.
"""

from __future__ import annotations
