"""Build model-ready feature matrices from dataset examples.

The model receives *only* the 13 approved features, in the exact,
canonical order declared by
:data:`~neuropager.dataset.features.FEATURE_NAMES`. No metadata field
(``episode_id``, ``page_id``, ``policy``, ...) is ever read here, which is
what structurally guarantees no identity or context leakage into the
feature matrix (see ``docs/model.md``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from neuropager.dataset.features import FEATURE_NAMES
from neuropager.dataset.schema import DatasetExample


def build_feature_matrix(
    examples: Sequence[DatasetExample],
) -> tuple[NDArray[np.float64], NDArray[np.int64], tuple[str, ...]]:
    """Convert dataset examples into a model-ready ``(X, y)`` pair.

    Args:
        examples: The dataset examples to convert.

    Returns:
        A tuple ``(X, y, feature_order)`` where ``X`` has shape
        ``(len(examples), len(FEATURE_NAMES))`` with columns in
        :data:`~neuropager.dataset.features.FEATURE_NAMES` order, ``y`` is
        the corresponding label vector, and ``feature_order`` is that same
        column ordering, returned alongside the matrix so callers never
        have to assume it.

    Raises:
        ValueError: If ``examples`` is empty.
    """
    if not examples:
        raise ValueError("cannot build a feature matrix from zero examples")

    rows = [[getattr(example.features, name) for name in FEATURE_NAMES] for example in examples]
    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray([example.label for example in examples], dtype=np.int64)
    return x, y, FEATURE_NAMES
