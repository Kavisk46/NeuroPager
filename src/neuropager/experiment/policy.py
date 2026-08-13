"""The learned-utility page replacement policy: an experimental adapter only.

:class:`LearnedUtilityPolicy` implements
:class:`~neuropager.policies.base.PageReplacementPolicy` exactly like
FIFO/LRU/LFU/Random — ``select_victim(self, page_table)`` is its only
external input, so it is structurally bound by the same "no future
information" guarantee every other policy already has (see
``docs/dataset.md``). It does not replace any production policy; it exists
so the offline-trained model can be evaluated inside a real, running
:class:`~neuropager.core.memory_manager.MemoryManager`, not just against a
held-out static dataset.

This module also documents, precisely, where the online feature
computation cannot achieve exact parity with the offline training
pipeline — see the module-level "Known parity gaps" section in
``docs/model.md``. In short: everything except ``elapsed_since_fault`` is
either exact or self-consistent; that one feature is structurally
unavailable online given the current
:class:`~neuropager.policies.base.PageReplacementPolicy` hook interface,
and is always reported as the missing-history sentinel.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.pipeline import Pipeline

from neuropager.core.page_table import PageTable
from neuropager.dataset.features import FEATURE_NAMES, MISSING_HISTORY_SENTINEL, compute_features
from neuropager.policies.base import PageReplacementPolicy
from neuropager.utils.types import MemoryKey


class LearnedUtilityPolicy(PageReplacementPolicy):
    """Evicts the resident page the model predicts is least likely to be reused.

    Maintains its own internal logical tick, advanced once per
    ``on_access``/``on_insert`` call — exactly matching the cadence of
    :meth:`~neuropager.trace.logger.TraceLogger.begin_step`, since exactly
    one of those two hooks fires per top-level
    :class:`~neuropager.core.memory_manager.MemoryManager` ``get``/``put``
    call (this mirrors :class:`~neuropager.policies.belady_min.BeladyMinPolicy`'s
    reference-string cursor). This keeps online feature computation on the
    same logical clock the offline model was trained on, rather than
    :class:`~neuropager.core.page_table.PageTable`'s own internal clock
    (which advances a different number of times per operation depending on
    whether it was a hit, a fault, or an eviction, and so cannot be
    rescaled to match).

    Attributes:
        model: A fitted scikit-learn pipeline exposing ``predict_proba``.
        horizon: The reuse horizon this model was trained for (metadata
            only; it does not affect victim selection directly).
    """

    def __init__(self, model: Pipeline, horizon: int) -> None:
        """Initialize the learned-utility policy.

        Args:
            model: A fitted scikit-learn pipeline (see
                :mod:`neuropager.experiment.models`) exposing
                ``predict_proba``.
            horizon: The reuse horizon this model was trained for.
        """
        self.model = model
        self.horizon = horizon
        self._tick = 0
        self._access_ticks: dict[MemoryKey, list[int]] = defaultdict(list)

    def select_victim(self, page_table: PageTable) -> MemoryKey:
        """Select the resident page with the lowest predicted reuse probability.

        Args:
            page_table: The page table to consult for the set of resident
                keys and their exact, clock-independent fault counts.

        Returns:
            The resident key with the lowest ``P(reused within horizon)``.

        Raises:
            ValueError: If ``page_table`` has no resident pages.
        """
        resident = page_table.resident_keys()
        if not resident:
            raise ValueError("cannot select a victim: no resident pages in working memory")

        # select_victim always runs before on_access/on_insert fires for the
        # reference currently being processed (see PageFaultHandler.install),
        # so self._tick is exactly one behind "now" at this point. Since
        # ticks advance by exactly one per top-level operation and only one
        # operation is ever in flight, self._tick + 1 is not a heuristic —
        # it is the exact tick this decision is being made at.
        decision_tick = self._tick + 1

        rows = []
        for key in resident:
            entry = page_table.get_entry(key)
            access_ticks = self._access_ticks.get(key, [])
            features = compute_features(
                access_ticks=access_ticks,
                # fault_ticks is unavailable online: PageReplacementPolicy's
                # hooks cannot distinguish a fault-driven on_insert from a
                # fresh-put on_insert, and PageTableEntry retains only a
                # cumulative fault *count*, not fault *ticks*. elapsed_since_fault
                # is therefore always the missing-history sentinel here; see
                # docs/model.md.
                fault_ticks=[],
                decision_tick=decision_tick,
            )
            # fault_count/fault_ratio ARE exactly available (clock-independent
            # counts straight from PageTable), so use the real values instead
            # of what the fault_ticks=[] call above computed for them.
            fault_count = float(entry.page_fault_count)
            fault_ratio = (
                fault_count / entry.access_count
                if entry.access_count > 0
                else MISSING_HISTORY_SENTINEL
            )
            row = [getattr(features, name) for name in FEATURE_NAMES]
            row[FEATURE_NAMES.index("fault_count")] = fault_count
            row[FEATURE_NAMES.index("fault_ratio")] = fault_ratio
            rows.append((key, row))

        x = np.asarray([row for _, row in rows], dtype=np.float64)
        probabilities = self.model.predict_proba(x)[:, 1]
        victim_index = int(np.argmin(probabilities))
        return rows[victim_index][0]

    def on_access(self, key: MemoryKey) -> None:
        """Advance the internal tick and record this access for ``key``.

        Args:
            key: The logical memory identifier that was accessed.
        """
        self._tick += 1
        self._access_ticks[key].append(self._tick)

    def on_insert(self, key: MemoryKey) -> None:
        """Advance the internal tick and record this access for ``key``.

        Args:
            key: The logical memory identifier that was inserted.
        """
        self._tick += 1
        self._access_ticks[key].append(self._tick)

    def on_evict(self, key: MemoryKey) -> None:
        """No-op: access history is retained in case the page is faulted back in.

        Args:
            key: The logical memory identifier that was evicted.
        """
