"""Hybrid retriever: fuses vector store and knowledge graph results.

Services page faults by querying both the dense
:class:`~neuropager.memory.vector_store.VectorStore` and the symbolic
:class:`~neuropager.memory.knowledge_graph.KnowledgeGraph`, then merging
their results into a single ranked list. See ``docs/memory.md`` for the
rationale behind fusing both signals.
"""

from __future__ import annotations

from neuropager.memory.knowledge_graph import KnowledgeGraph
from neuropager.memory.vector_store import VectorStore
from neuropager.utils.types import RetrievalResult


class HybridRetriever:
    """Fuses vector store and knowledge graph retrieval into ranked results.

    Attributes:
        vector_store: The dense embedding index to query.
        knowledge_graph: The symbolic graph store to query.
        vector_weight: Relative weight given to vector-store results during
            fusion.
        graph_weight: Relative weight given to knowledge-graph results
            during fusion.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        knowledge_graph: KnowledgeGraph,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4,
    ) -> None:
        """Initialize the hybrid retriever with its backends and fusion weights.

        Args:
            vector_store: The dense embedding index to query.
            knowledge_graph: The symbolic graph store to query.
            vector_weight: Relative weight given to vector-store results.
            graph_weight: Relative weight given to knowledge-graph results.
        """
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Retrieve and fuse results for ``query`` from both backends.

        Args:
            query: The natural-language (or structured) query string.
            top_k: Maximum number of fused results to return.

        Returns:
            A list of :class:`~neuropager.utils.types.RetrievalResult`,
            ranked by fused relevance score.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError
