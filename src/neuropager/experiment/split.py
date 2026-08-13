"""Deterministic, episode-level train/validation/test splitting.

Splitting happens on *episode IDs*, never on individual
:class:`~neuropager.dataset.schema.DatasetExample` rows. Examples from the
same episode always land in the same split — otherwise a model could see
near-duplicate context from an episode's training portion while being
"tested" on the rest of the same episode, silently inflating every metric.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from neuropager.dataset.schema import DatasetExample

DEFAULT_TRAIN_FRACTION = 0.7
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_TEST_FRACTION = 0.15


@dataclass(frozen=True)
class EpisodeSplit:
    """A deterministic partition of episode IDs into train/validation/test.

    Attributes:
        train_episodes: Episode IDs assigned to training.
        val_episodes: Episode IDs assigned to validation.
        test_episodes: Episode IDs assigned to testing.
        seed: The seed used to produce this split.
        train_fraction: The requested training fraction.
        val_fraction: The requested validation fraction.
        test_fraction: The requested test fraction.
    """

    train_episodes: tuple[str, ...]
    val_episodes: tuple[str, ...]
    test_episodes: tuple[str, ...]
    seed: int
    train_fraction: float
    val_fraction: float
    test_fraction: float

    def to_dict(self) -> dict[str, object]:
        """Return this split as a plain, JSON-serializable dict.

        Returns:
            A dict suitable for inclusion in experiment metadata.
        """
        return {
            "train_episodes": list(self.train_episodes),
            "val_episodes": list(self.val_episodes),
            "test_episodes": list(self.test_episodes),
            "seed": self.seed,
            "train_fraction": self.train_fraction,
            "val_fraction": self.val_fraction,
            "test_fraction": self.test_fraction,
        }


def split_episodes(
    episode_ids: Sequence[str],
    seed: int,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> EpisodeSplit:
    """Deterministically partition episode IDs into train/validation/test.

    Unique episode IDs are sorted (so input order never affects the
    result), then shuffled with a seeded :class:`random.Random` instance
    and cut at the requested fractions. The same ``episode_ids`` and
    ``seed`` always produce the same split.

    Args:
        episode_ids: Episode IDs to split (duplicates are ignored).
        seed: Seed for the deterministic shuffle.
        train_fraction: Fraction of episodes assigned to training.
        val_fraction: Fraction of episodes assigned to validation.
        test_fraction: Fraction of episodes assigned to testing.

    Returns:
        The resulting :class:`EpisodeSplit`.

    Raises:
        ValueError: If ``episode_ids`` is empty, if any fraction is
            negative, or if the fractions do not sum to ``1.0``.
    """
    if not episode_ids:
        raise ValueError("cannot split an empty sequence of episode_ids")
    if train_fraction < 0 or val_fraction < 0 or test_fraction < 0:
        raise ValueError("split fractions must be non-negative")
    if not math.isclose(train_fraction + val_fraction + test_fraction, 1.0, abs_tol=1e-9):
        raise ValueError(
            "split fractions must sum to 1.0, got "
            f"{train_fraction} + {val_fraction} + {test_fraction} = "
            f"{train_fraction + val_fraction + test_fraction}"
        )

    unique_episodes = sorted(set(episode_ids))
    rng = random.Random(seed)
    shuffled = list(unique_episodes)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * train_fraction)
    n_val = round(n * val_fraction)
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)

    train = tuple(shuffled[:n_train])
    val = tuple(shuffled[n_train : n_train + n_val])
    test = tuple(shuffled[n_train + n_val :])

    return EpisodeSplit(
        train_episodes=train,
        val_episodes=val,
        test_episodes=test,
        seed=seed,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )


def filter_examples_by_episodes(
    examples: Sequence[DatasetExample],
    episode_ids: Sequence[str],
) -> list[DatasetExample]:
    """Return only the examples whose episode_id is in ``episode_ids``.

    Args:
        examples: The full example set to filter.
        episode_ids: Episode IDs to keep (e.g. one split's episodes).

    Returns:
        The matching examples, preserving their original relative order.
    """
    allowed = set(episode_ids)
    return [example for example in examples if example.episode_id in allowed]
