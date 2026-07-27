"""Dense vector store: semantic similarity search over memory content.

The vector store is the "neuro" half of NeuroPager's neuro-symbolic
retrieval system, complementing the symbolic knowledge graph with
approximate nearest-neighbor search over dense embeddings. See
``docs/memory.md`` for the motivation behind hybrid retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neuropager.utils.types import Embedding, MemoryKey


@dataclass(frozen=True)
class VectorMatch:
    """A single nearest-neighbor search result.

    Attributes:
        key: Logical identifier of the matched memory unit.
        score: Similarity (or inverse distance) score.
    """

    key: MemoryKey
    score: float


class VectorStore:
    """Dense embedding index supporting approximate nearest-neighbor search."""

    def __init__(self, embedding_dim: int, distance_metric: str = "cosine") -> None:
        """Initialize the vector store.

        Args:
            embedding_dim: Dimensionality of stored embeddings.
            distance_metric: Similarity/distance metric to use for search
                (e.g. ``"cosine"``, ``"euclidean"``, ``"dot"``).
        """
        self.embedding_dim = embedding_dim
        self.distance_metric = distance_metric
        # TODO(neuropager): back this with an in-memory index or a
        # pluggable ANN backend (e.g. FAISS, HNSW).

    def add(
        self,
        key: MemoryKey,
        embedding: Embedding,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add or update an embedding in the store.

        Args:
            key: Logical identifier of the memory unit being indexed.
            embedding: The dense vector representation.
            metadata: Optional free-form metadata to associate with the
                entry.

        Raises:
            ValueError: If ``embedding`` does not match :attr:`embedding_dim`.
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError

    def search(self, query_embedding: Embedding, top_k: int = 10) -> list[VectorMatch]:
        """Return the ``top_k`` nearest neighbors to ``query_embedding``.

        Args:
            query_embedding: The query vector.
            top_k: Maximum number of results to return.

        Returns:
            A list of :class:`VectorMatch` results, ranked by similarity.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError

    def remove(self, key: MemoryKey) -> None:
        """Remove an embedding from the store.

        Args:
            key: Logical identifier of the memory unit to remove.

        Raises:
            KeyError: If ``key`` is not present in the store.
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError
