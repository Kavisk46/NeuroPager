"""Typed application settings for NeuroPager.

Defines the settings schema loaded from ``configs/config.yaml`` (and
overridable via environment variables). This module intentionally contains
no I/O logic yet beyond a TODO-marked loader stub — see
``docs/design.md`` for the intended configuration surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")
DEFAULT_LOGGING_CONFIG_PATH = Path("configs/logging.yaml")


@dataclass(frozen=True)
class WorkingMemorySettings:
    """Settings governing the working memory (resident context) subsystem.

    Attributes:
        capacity: Maximum number of memory pages resident at once.
        eviction_policy: Name of the registered page replacement policy to
            use (e.g. ``"lru"``, ``"lfu"``, ``"fifo"``).
    """

    capacity: int = 128
    eviction_policy: str = "lru"


@dataclass(frozen=True)
class VectorStoreSettings:
    """Settings for the dense vector store backend.

    Attributes:
        backend: Backend identifier (e.g. ``"in_memory"``, ``"faiss"``).
        embedding_dim: Dimensionality of stored embeddings.
        distance_metric: Similarity/distance metric used for search.
    """

    backend: str = "in_memory"
    embedding_dim: int = 768
    distance_metric: str = "cosine"


@dataclass(frozen=True)
class KnowledgeGraphSettings:
    """Settings for the symbolic knowledge graph backend.

    Attributes:
        backend: Backend identifier (e.g. ``"in_memory"``, ``"external"``).
    """

    backend: str = "in_memory"


@dataclass(frozen=True)
class DiskStoreSettings:
    """Settings for the cold-storage disk page store.

    Attributes:
        backend: Backend identifier (e.g. ``"filesystem"``).
        path: Filesystem path (or backend-specific location string) used
            for durable storage.
    """

    backend: str = "filesystem"
    path: Path = Path("./data/disk_store")


@dataclass(frozen=True)
class RetrievalSettings:
    """Settings for hybrid retrieval fusion.

    Attributes:
        top_k: Number of results to return per retrieval call.
        vector_weight: Relative weight given to vector-store results.
        graph_weight: Relative weight given to knowledge-graph results.
    """

    top_k: int = 10
    vector_weight: float = 0.6
    graph_weight: float = 0.4


@dataclass(frozen=True)
class MemorySettings:
    """Aggregate settings for all memory subsystems.

    Attributes:
        vector_store: Vector store configuration.
        knowledge_graph: Knowledge graph configuration.
        disk_store: Disk page store configuration.
    """

    vector_store: VectorStoreSettings = field(default_factory=VectorStoreSettings)
    knowledge_graph: KnowledgeGraphSettings = field(default_factory=KnowledgeGraphSettings)
    disk_store: DiskStoreSettings = field(default_factory=DiskStoreSettings)


@dataclass(frozen=True)
class Settings:
    """Root NeuroPager configuration object.

    Attributes:
        working_memory: Working memory subsystem settings.
        memory: Memory backend settings (vector store, knowledge graph,
            disk store).
        retrieval: Hybrid retrieval settings.
        seed: Global random seed for reproducibility.
        logging_config_path: Path to the logging configuration file.
    """

    working_memory: WorkingMemorySettings = field(default_factory=WorkingMemorySettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    seed: int = 42
    logging_config_path: Path = DEFAULT_LOGGING_CONFIG_PATH


def load_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> Settings:
    """Load :class:`Settings` from a YAML configuration file.

    Args:
        config_path: Path to a ``config.yaml`` file matching the schema
            documented in ``configs/config.yaml``.

    Returns:
        A populated :class:`Settings` instance.

    Raises:
        NotImplementedError: Always — configuration loading is not yet
            implemented. See ``docs/roadmap.md`` (Phase 1).
    """
    # TODO(neuropager): parse YAML at `config_path` and construct a
    # `Settings` instance, applying environment-variable overrides.
    raise NotImplementedError("Settings loading is not yet implemented.")
