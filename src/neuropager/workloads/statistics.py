"""Pure, workload-agnostic diagnostic statistics for a generated access sequence.

These functions operate on a plain ``list[MemoryKey]`` only -- no
dependency on :mod:`neuropager.core` or :mod:`neuropager.trace` -- so they
can be used to sanity-check a workload's access-pattern properties
*before* ever running it through the memory system. Eviction/fault counts
(which do require the real memory system) are computed separately, in
``scripts/sanity_workloads.py``.

Complexity note: :func:`stack_distances` is O(n * average reuse distance)
in the worst case (it materializes the set of distinct keys between each
pair of repeat occurrences). This is intentional and adequate for the
small sanity checks and unit-test fixtures this module is scoped to; it
is not intended for the eventual large-scale (2,000+ episode) experiment.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from neuropager.utils.types import MemoryKey


def unique_page_count(workload: Sequence[MemoryKey]) -> int:
    """Return the number of distinct keys referenced in ``workload``.

    Args:
        workload: The access sequence to inspect.

    Returns:
        The count of distinct keys.
    """
    return len(set(workload))


def access_entropy(workload: Sequence[MemoryKey]) -> float:
    """Return the Shannon entropy, in bits, of the empirical key distribution.

    ``H = -sum(p_k * log2(p_k))`` over each distinct key ``k`` with
    empirical probability ``p_k = count(k) / len(workload)``. Maximal
    (``log2(unique_page_count)``) for a workload that visits every key
    with equal frequency; lower for skewed or repetitive access.

    Args:
        workload: The access sequence to inspect.

    Returns:
        The entropy in bits, or ``0.0`` for an empty workload.
    """
    if not workload:
        return 0.0
    counts = Counter(workload)
    n = len(workload)
    return -sum((count / n) * math.log2(count / n) for count in counts.values())


def stack_distances(workload: Sequence[MemoryKey]) -> list[int | None]:
    """Return the LRU stack distance for each access in ``workload``.

    The stack distance of an access is the number of *distinct* keys
    referenced strictly between it and the previous access to the same
    key -- i.e. the working-set size an LRU cache would need to guarantee
    a hit on this access. The first occurrence of a key has no previous
    occurrence and is reported as ``None`` (a cold reference).

    Args:
        workload: The access sequence to inspect.

    Returns:
        One entry per access, in order, aligned with ``workload``.
    """
    last_seen_index: dict[MemoryKey, int] = {}
    distances: list[int | None] = []
    for index, key in enumerate(workload):
        previous = last_seen_index.get(key)
        if previous is None:
            distances.append(None)
        else:
            distances.append(len(set(workload[previous + 1 : index])))
        last_seen_index[key] = index
    return distances


def mean_stack_distance(workload: Sequence[MemoryKey]) -> float | None:
    """Return the mean stack distance over all non-cold accesses.

    Args:
        workload: The access sequence to inspect.

    Returns:
        The mean of the non-``None`` values from :func:`stack_distances`,
        or ``None`` if every access was a cold reference (no repeats).
    """
    observed = [d for d in stack_distances(workload) if d is not None]
    if not observed:
        return None
    return sum(observed) / len(observed)


def burst_run_lengths(workload: Sequence[MemoryKey]) -> list[int]:
    """Return the lengths of maximal runs of consecutive identical keys.

    A run of length 1 means the key was not immediately repeated. A
    bursty workload is expected to show a longer mean run length than a
    memoryless (e.g. uniform-random) one.

    Args:
        workload: The access sequence to inspect.

    Returns:
        One run length per maximal run, in order. Empty for an empty
        workload.
    """
    if not workload:
        return []
    runs: list[int] = []
    current_length = 1
    # workload and workload[1:] are deliberately different lengths (pairing
    # each element with its successor), so strict=False is correct here,
    # not an oversight.
    for previous, current in zip(workload, workload[1:], strict=False):
        if current == previous:
            current_length += 1
        else:
            runs.append(current_length)
            current_length = 1
    runs.append(current_length)
    return runs
