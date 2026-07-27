"""Sanity checks that the NeuroPager package structure is importable.

These tests validate the scaffold itself (imports resolve, public classes
exist) rather than behavior, since no algorithmic implementation exists
yet. As modules gain real implementations, replace/extend these with
behavioral tests colocated by module.
"""

from __future__ import annotations

import neuropager


def test_package_has_version() -> None:
    """The top-level package exposes a semantic version string."""
    assert isinstance(neuropager.__version__, str)
    assert neuropager.__version__


def test_core_symbols_importable() -> None:
    """Core virtual-memory substrate classes are importable."""
    from neuropager.core.memory_manager import MemoryManager
    from neuropager.core.page_fault import PageFaultHandler
    from neuropager.core.page_table import PageTable
    from neuropager.core.working_memory import WorkingMemory

    assert MemoryManager and PageFaultHandler and PageTable and WorkingMemory


def test_memory_symbols_importable() -> None:
    """Memory subsystem classes are importable."""
    from neuropager.memory.disk_store import DiskPageStore
    from neuropager.memory.episodic import EpisodicMemory
    from neuropager.memory.knowledge_graph import KnowledgeGraph
    from neuropager.memory.vector_store import VectorStore

    assert DiskPageStore and EpisodicMemory and KnowledgeGraph and VectorStore


def test_policy_symbols_importable() -> None:
    """Page replacement policy classes are importable."""
    from neuropager.policies.base import PageReplacementPolicy
    from neuropager.policies.fifo import FIFOPolicy
    from neuropager.policies.lfu import LFUPolicy
    from neuropager.policies.lru import LRUPolicy

    assert PageReplacementPolicy and FIFOPolicy and LFUPolicy and LRUPolicy


def test_retrieval_and_agent_symbols_importable() -> None:
    """Retrieval and agent-facing classes are importable."""
    from neuropager.agents.base_agent import BaseAgent
    from neuropager.retrieval.hybrid_retriever import HybridRetriever

    assert BaseAgent and HybridRetriever
