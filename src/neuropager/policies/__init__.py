"""Pluggable page replacement policies for working memory eviction.

All policies implement :class:`~neuropager.policies.base.PageReplacementPolicy`,
so the :class:`~neuropager.core.page_fault.PageFaultHandler` can select and
swap policies without change.
"""

from __future__ import annotations
