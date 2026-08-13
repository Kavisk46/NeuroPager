"""Future-reuse label generation for the offline dataset pipeline.

Labels are the one deliberate exception to this codebase's "no future
information" rule: they are computed *offline*, after an episode has fully
completed, by looking forward in the already-recorded trace. This is safe
precisely because label generation never feeds back into any online
decision — the runtime policies in :mod:`neuropager.policies` never see a
label, and :mod:`neuropager.dataset.features` never sees anything beyond
the decision tick. See ``docs/dataset.md`` for the full argument.
"""

from __future__ import annotations

from collections.abc import Sequence

DEFAULT_HORIZONS: tuple[int, ...] = (25, 50, 100, 200, 500)
"""The horizons the research plan names for comparison."""


def compute_label(full_access_ticks: Sequence[int], decision_tick: int, horizon: int) -> int:
    """Compute ``y(p, t, H)``: whether a page is reused within a horizon.

    ``y = 1`` if the page has an access tick in ``(decision_tick,
    decision_tick + horizon]``, else ``0``. The window is open at
    ``decision_tick`` (an access exactly at the decision tick is not
    "future reuse") and closed at ``decision_tick + horizon`` (an access
    exactly at the horizon boundary does count).

    Args:
        full_access_ticks: *All* of the page's access ticks for the
            episode, including ticks after ``decision_tick`` — this is the
            one place in the pipeline that is allowed to see the future.
        decision_tick: The tick the eviction decision was made at.
        horizon: The size of the future window to check, in ticks.

    Returns:
        ``1`` if the page is reused within the horizon, else ``0``.

    Raises:
        ValueError: If ``horizon`` is not a positive integer.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be a positive integer, got {horizon}")

    window_end = decision_tick + horizon
    for tick in full_access_ticks:
        if decision_tick < tick <= window_end:
            return 1
    return 0
