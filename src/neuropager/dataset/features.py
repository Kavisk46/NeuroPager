"""Per-page temporal feature extraction for the offline dataset pipeline.

Every function here is a pure function of a page's access/fault *history
up to and including the decision tick* — never anything after it. This is
the load-bearing leakage boundary of the whole pipeline: features are
computed from history alone, and callers (see
:mod:`neuropager.dataset.generator`) are responsible for only ever passing
in tick-filtered history. See ``docs/dataset.md`` for the full leakage
argument.

Some features are undefined when a page has too little history (e.g. an
average interval needs at least two accesses). Rather than raising or
producing ``NaN``/``inf`` — which would make downstream dataset validation
(:mod:`neuropager.dataset.validation`) fail unpredictably — every such case
resolves to :data:`MISSING_HISTORY_SENTINEL`, a fixed, documented, finite
value that can never be produced by a real computation (all genuine feature
values are ``>= 0``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields

MISSING_HISTORY_SENTINEL = -1.0
"""Returned wherever a feature is mathematically undefined given the
available history (e.g. a variance with fewer than two intervals). Always
finite and always negative, so it can never collide with a genuine value.
"""

RECENT_BURST_WINDOW = 10
"""Width, in ticks, of the short lookback window for :attr:`Features.recent_burst`."""

LONG_BURST_WINDOW = 50
"""Width, in ticks, of the longer lookback window for :attr:`Features.long_burst`."""


@dataclass(frozen=True)
class Features:
    """The feature vector computed for one (page, decision_tick) pair.

    All fields are ``float`` for uniform, pandas/sklearn-friendly output,
    even where the underlying quantity (e.g. :attr:`frequency`) is a count.
    See module docstring for the missing-history sentinel convention.

    Attributes:
        recency: Ticks elapsed since the page's most recent access,
            ``decision_tick - last_access_tick``.
        frequency: Total number of accesses observed up to the decision
            tick (a plain count, never a sentinel; ``0`` is a real value).
        avg_inter_access_interval: Mean gap between consecutive accesses.
        inter_access_interval_variance: Population variance of those gaps.
        fault_count: Total number of page faults observed up to the
            decision tick (a plain count, never a sentinel).
        fault_ratio: ``fault_count / frequency``.
        page_age: Ticks elapsed since the page's first observed access,
            ``decision_tick - first_access_tick``.
        recent_burst: Number of accesses within the last
            :data:`RECENT_BURST_WINDOW` ticks.
        long_burst: Number of accesses within the last
            :data:`LONG_BURST_WINDOW` ticks.
        recent_interval: Gap between the two most recent accesses (the
            single most recent inter-access interval).
        normalized_recency: ``recency / page_age``, i.e. what fraction of
            the page's observed lifetime has elapsed since its last use.
        elapsed_since_fault: Ticks elapsed since the page's most recent
            page fault.
        max_historical_interval: The largest gap ever observed between two
            consecutive accesses of this page.
    """

    recency: float
    frequency: float
    avg_inter_access_interval: float
    inter_access_interval_variance: float
    fault_count: float
    fault_ratio: float
    page_age: float
    recent_burst: float
    long_burst: float
    recent_interval: float
    normalized_recency: float
    elapsed_since_fault: float
    max_historical_interval: float

    def to_dict(self) -> dict[str, float]:
        """Return this feature vector as a plain ``{name: value}`` dict.

        Returns:
            A dict in field-declaration order, suitable for flattening
            into a dataset row.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}


FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in fields(Features))
"""The canonical, ordered list of feature names this module produces."""


def compute_features(
    access_ticks: Sequence[int],
    fault_ticks: Sequence[int],
    decision_tick: int,
) -> Features:
    """Compute the feature vector for one page as of ``decision_tick``.

    Args:
        access_ticks: This page's ACCESS-event ticks, already filtered to
            ``<= decision_tick`` and sorted ascending. Callers (see
            :mod:`neuropager.dataset.replay`) are responsible for the
            filtering; this function trusts its inputs and does not
            itself look at anything beyond what it is given.
        fault_ticks: This page's PAGE_FAULT-event ticks, likewise already
            filtered to ``<= decision_tick`` and sorted ascending.
        decision_tick: The tick at which the eviction decision is being
            made.

    Returns:
        The computed :class:`Features`.
    """
    access_count = len(access_ticks)
    fault_count = len(fault_ticks)
    frequency = float(access_count)

    fault_ratio = fault_count / access_count if access_count > 0 else MISSING_HISTORY_SENTINEL

    if access_count >= 1:
        recency = float(decision_tick - access_ticks[-1])
        page_age = float(decision_tick - access_ticks[0])
    else:
        recency = MISSING_HISTORY_SENTINEL
        page_age = MISSING_HISTORY_SENTINEL

    gaps = [access_ticks[i] - access_ticks[i - 1] for i in range(1, access_count)]
    if gaps:
        avg_interval = sum(gaps) / len(gaps)
        interval_variance = sum((gap - avg_interval) ** 2 for gap in gaps) / len(gaps)
        recent_interval = float(gaps[-1])
        max_interval = float(max(gaps))
    else:
        avg_interval = MISSING_HISTORY_SENTINEL
        interval_variance = MISSING_HISTORY_SENTINEL
        recent_interval = MISSING_HISTORY_SENTINEL
        max_interval = MISSING_HISTORY_SENTINEL

    recent_burst = float(sum(1 for t in access_ticks if t > decision_tick - RECENT_BURST_WINDOW))
    long_burst = float(sum(1 for t in access_ticks if t > decision_tick - LONG_BURST_WINDOW))

    if page_age > 0:
        normalized_recency = recency / page_age
    else:
        normalized_recency = MISSING_HISTORY_SENTINEL

    if fault_count >= 1:
        elapsed_since_fault = float(decision_tick - fault_ticks[-1])
    else:
        elapsed_since_fault = MISSING_HISTORY_SENTINEL

    return Features(
        recency=recency,
        frequency=frequency,
        avg_inter_access_interval=avg_interval,
        inter_access_interval_variance=interval_variance,
        fault_count=float(fault_count),
        fault_ratio=fault_ratio,
        page_age=page_age,
        recent_burst=recent_burst,
        long_burst=long_burst,
        recent_interval=recent_interval,
        normalized_recency=normalized_recency,
        elapsed_since_fault=elapsed_since_fault,
        max_historical_interval=max_interval,
    )
