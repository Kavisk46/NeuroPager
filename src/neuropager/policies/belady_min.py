"""Belady's MIN: the offline-optimal page replacement algorithm.

Belady's MIN always evicts the resident page whose next use lies furthest
in the future (or is never used again). It is provably optimal — no online
policy (one that decides using only past/present state) can ever cause
fewer faults on a given reference string. It exists here purely as an
*offline* upper-bound baseline for evaluating other policies; it is not,
and cannot be, an online policy.

Unlike every other policy in this package, :class:`BeladyMinPolicy`
requires knowing the future: the complete sequence of keys that will be
referenced over the episode, supplied at construction time. This is the
one deliberate, explicit exception to the rule enforced everywhere else in
NeuroPager that a replacement policy only ever sees past/present state
(see :meth:`~neuropager.policies.base.PageReplacementPolicy.select_victim`,
whose signature — ``(self, page_table)`` — still does not grant access to
the future; the future only ever enters through this class's constructor).
Use it only to compute an offline-optimal baseline for comparison, never
as a policy driving a real online agent.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from collections.abc import Sequence

from neuropager.core.page_table import PageTable
from neuropager.policies.base import PageReplacementPolicy
from neuropager.utils.types import MemoryKey


class BeladyMinPolicy(PageReplacementPolicy):
    """Evicts the resident page whose next reference is furthest in the future.

    The ``future`` sequence passed at construction must be exactly the
    sequence of keys passed to every subsequent
    :class:`~neuropager.core.memory_manager.MemoryManager` ``get``/``put``
    call, in order (explicit ``evict`` calls are not references and are
    not included). Any divergence between the promised sequence and what
    is actually replayed is detected and raised immediately, rather than
    silently producing an incorrect "optimal" baseline.

    Attributes:
        future: The complete reference string this policy was constructed
            with.
    """

    def __init__(self, future: Sequence[MemoryKey]) -> None:
        """Initialize the policy with the complete future reference string.

        Args:
            future: The full, ordered sequence of keys that will be passed
                to ``get``/``put`` over the episode.
        """
        self.future = list(future)
        self._position = 0
        self._positions_by_key: dict[MemoryKey, list[int]] = defaultdict(list)
        for index, key in enumerate(self.future):
            self._positions_by_key[key].append(index)

    def _next_use(self, key: MemoryKey) -> int:
        """Return the index of ``key``'s next occurrence at or after now.

        Args:
            key: The resident key to look up.

        Returns:
            The smallest index ``>= self._position`` at which ``key``
            occurs in :attr:`future`, or ``len(self.future)`` (treated as
            "infinitely far away") if it is never referenced again.
        """
        positions = self._positions_by_key.get(key)
        if not positions:
            return len(self.future)
        index = bisect_left(positions, self._position)
        if index == len(positions):
            return len(self.future)
        return positions[index]

    def select_victim(self, page_table: PageTable) -> MemoryKey:
        """Select the resident key whose next use is furthest in the future.

        Args:
            page_table: The page table to consult for the set of resident
                keys. Access metadata is not consulted, since this policy
                looks forward, not backward.

        Returns:
            The resident key with the largest "next use" index (ties
            broken by :meth:`~neuropager.core.page_table.PageTable.resident_keys`
            order, i.e. the earliest-tracked key wins).

        Raises:
            ValueError: If ``page_table`` has no resident pages.
        """
        resident = page_table.resident_keys()
        if not resident:
            raise ValueError("cannot select a victim: no resident pages in working memory")
        return max(resident, key=self._next_use)

    def on_access(self, key: MemoryKey) -> None:
        """Advance the reference-string cursor past a resolved hit.

        Args:
            key: The logical memory identifier that was accessed.

        Raises:
            ValueError: If ``key`` does not match the next key in
                :attr:`future`, meaning the actual replay has diverged
                from the sequence this policy was constructed with.
        """
        self._advance(key)

    def on_insert(self, key: MemoryKey) -> None:
        """Advance the reference-string cursor past a resolved fault or new page.

        Args:
            key: The logical memory identifier that was inserted.

        Raises:
            ValueError: If ``key`` does not match the next key in
                :attr:`future`, meaning the actual replay has diverged
                from the sequence this policy was constructed with.
        """
        self._advance(key)

    def on_evict(self, key: MemoryKey) -> None:
        """No-op: the future reference string does not depend on eviction choices.

        Args:
            key: The logical memory identifier that was evicted.
        """

    def _advance(self, key: MemoryKey) -> None:
        """Consume one step of the reference string, validating it matches.

        Args:
            key: The key actually observed at the current position.

        Raises:
            ValueError: If the episode has run longer than :attr:`future`
                covers, or if ``key`` does not match the expected key at
                the current position.
        """
        if self._position >= len(self.future):
            raise ValueError(
                "BeladyMinPolicy received more references than its future "
                f"reference string covers ({len(self.future)})"
            )
        expected = self.future[self._position]
        if expected != key:
            raise ValueError(
                f"reference-string mismatch at position {self._position}: "
                f"expected {expected!r}, observed {key!r}. BeladyMinPolicy requires "
                "the actual get()/put() key sequence to exactly match the future "
                "reference string it was constructed with."
            )
        self._position += 1
