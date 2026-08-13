"""Random page replacement policy.

Evicts a uniformly random resident page, ignoring recency, frequency, and
insertion order entirely. Serves as a naive baseline: any replacement
policy worth using should consistently beat random eviction.

The module is named ``random_policy`` rather than ``random`` so that it
never shadows the standard library :mod:`random` module it imports from.
"""

from __future__ import annotations

import random

from neuropager.core.page_table import PageTable
from neuropager.policies.base import PageReplacementPolicy
from neuropager.utils.types import MemoryKey


class RandomPolicy(PageReplacementPolicy):
    """Evicts a uniformly random resident page.

    Attributes:
        seed: The seed this policy's random generator was constructed
            with, if any.
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the random policy.

        Args:
            seed: Seed for the internal random generator. Pass a fixed
                seed for reproducible experiment runs; omit it only for
                genuinely non-deterministic use, since NeuroPager's traces
                are otherwise designed to be byte-for-byte reproducible
                (see :mod:`neuropager.trace.logger`).
        """
        self.seed = seed
        self._rng = random.Random(seed)

    def select_victim(self, page_table: PageTable) -> MemoryKey:
        """Select a uniformly random resident key for eviction.

        Args:
            page_table: The page table to consult for the set of resident
                keys. Access metadata is not consulted, since this policy
                ignores it by design.

        Returns:
            A resident key chosen uniformly at random.

        Raises:
            ValueError: If ``page_table`` has no resident pages.
        """
        resident = page_table.resident_keys()
        if not resident:
            raise ValueError("cannot select a victim: no resident pages in working memory")
        return self._rng.choice(resident)

    def on_access(self, key: MemoryKey) -> None:
        """No-op: this policy ignores access history by design.

        Args:
            key: The logical memory identifier that was accessed.
        """

    def on_insert(self, key: MemoryKey) -> None:
        """No-op: this policy ignores insertion history by design.

        Args:
            key: The logical memory identifier that was inserted.
        """

    def on_evict(self, key: MemoryKey) -> None:
        """No-op: this policy holds no per-key state to clean up.

        Args:
            key: The logical memory identifier that was evicted.
        """
