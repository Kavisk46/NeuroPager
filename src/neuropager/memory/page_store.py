"""Structural interface satisfied by every page-store backend.

:class:`~neuropager.core.page_fault.PageFaultHandler` only ever calls
``write``/``read``/``delete``/``contains`` on its ``disk_store``
collaborator, so it can be typed against this structural
:class:`~typing.Protocol` instead of the concrete
:class:`~neuropager.memory.disk_store.DiskPageStore` class. This lets any
backend satisfying the same contract (e.g.
:class:`~neuropager.memory.in_memory_store.InMemoryPageStore`) be used
interchangeably, with no change to fault-handling behavior.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from neuropager.utils.types import MemoryKey, MemoryPage


@runtime_checkable
class PageStore(Protocol):
    """The read/write/delete/contains contract every page-store backend implements."""

    def write(self, key: MemoryKey, page: MemoryPage) -> None:
        """Persist ``page`` under ``key``."""
        ...

    def read(self, key: MemoryKey) -> MemoryPage:
        """Return the page persisted under ``key``.

        Raises:
            KeyError: If ``key`` has no persisted page.
        """
        ...

    def delete(self, key: MemoryKey) -> None:
        """Remove the page persisted under ``key``.

        Raises:
            KeyError: If ``key`` has no persisted page.
        """
        ...

    def contains(self, key: MemoryKey) -> bool:
        """Return whether a page is currently persisted under ``key``."""
        ...
