"""Memory subsystems: episodic memory, knowledge graph, vector store, disk store.

These modules define the *content* backends that memory pages ultimately
resolve to when not resident in working memory. See
``neuropager.core`` for the paging substrate that sits above them.
"""

from __future__ import annotations
