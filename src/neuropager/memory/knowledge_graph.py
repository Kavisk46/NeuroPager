"""Symbolic knowledge graph: entities, relations, and structured facts.

The knowledge graph is the "symbolic" half of NeuroPager's neuro-symbolic
retrieval system, complementing the dense vector store with explicit,
explainable relational structure. See ``docs/memory.md`` for the
motivation behind hybrid retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Triple:
    """A single subject-predicate-object fact.

    Attributes:
        subject: The subject entity identifier.
        predicate: The relation type.
        obj: The object entity or literal value.
    """

    subject: str
    predicate: str
    obj: str


class KnowledgeGraph:
    """Symbolic store of entities and typed relations."""

    def __init__(self) -> None:
        """Initialize an empty knowledge graph."""
        # TODO(neuropager): back this with an in-memory graph structure
        # (e.g. adjacency list) or an external graph database client.

    def add_triple(self, triple: Triple) -> None:
        """Add a fact to the knowledge graph.

        Args:
            triple: The subject-predicate-object fact to add.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError

    def query(self, subject: str | None = None, predicate: str | None = None) -> list[Triple]:
        """Query triples matching an optional subject and/or predicate.

        Args:
            subject: If provided, restrict results to this subject.
            predicate: If provided, restrict results to this predicate.

        Returns:
            A list of matching :class:`Triple` facts.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError

    def neighbors(self, entity: str, depth: int = 1) -> list[Triple]:
        """Return facts reachable from ``entity`` within ``depth`` hops.

        Args:
            entity: The starting entity identifier.
            depth: Maximum number of relation hops to traverse.

        Returns:
            A list of :class:`Triple` facts within the traversal depth.

        Raises:
            NotImplementedError: Not yet implemented.
        """
        raise NotImplementedError
