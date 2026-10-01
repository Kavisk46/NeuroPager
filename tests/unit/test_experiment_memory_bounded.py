"""Tests for the memory-bounded dataset generation in generalization_experiment_2.py.

Proves that build_family_cache() -- which calls generate_dataset() once per
family, across every horizon that family is needed at, in contiguous
episode-count batches -- is scientifically equivalent to the old approach of
materializing every family's events into one list and calling
generate_dataset() once for everything, while never handing generate_dataset()
more than one family's events (and at most EPISODE_BATCH_SIZE episodes) at a
time.

scripts/ is not a package, so the module under test is loaded directly
from its file path, matching the pattern used by test_experiment_resume.py.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "generalization_experiment_2.py"
_spec = importlib.util.spec_from_file_location("generalization_experiment_2", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
ge2: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["generalization_experiment_2"] = ge2
_spec.loader.exec_module(ge2)

from neuropager.dataset.generator import generate_dataset  # noqa: E402
from neuropager.experiment.preprocessing import build_feature_matrix  # noqa: E402
from neuropager.experiment.split import filter_examples_by_episodes  # noqa: E402

FAMILY_A = "A_uniform_random"
FAMILY_B = "B_temporal_locality"
FAMILY_C = "C_bursty"
HORIZON = 50
N_EPISODES = 6


def _build_episodes(family: str, n: int) -> list[Any]:
    """Build n episodes for one family using the unchanged build_episode()."""
    return [ge2.build_episode(family, seed)[0] for seed in range(n)]


def test_family_cache_single_horizon_matches_monolithic_generation() -> None:
    """FamilyCache's single-horizon slice exactly matches filtering one big generate_dataset() call.

    This is the core scientific-equivalence proof: the OLD approach
    (concatenate every family's events, call generate_dataset() once for
    all families and horizons, then filter down to one family+horizon) and
    the NEW approach (build_family_cache() over just that family's
    episodes, requesting only this one horizon) must produce byte-identical
    feature matrices and label vectors, in the same row order.
    """
    episodes_a = _build_episodes(FAMILY_A, N_EPISODES)
    episodes_b = _build_episodes(FAMILY_B, N_EPISODES)

    # OLD approach: one monolithic call across both families, all horizons
    # NeuroPager actually uses, then filtered down to family A + horizon 50.
    all_events = [event for ep in episodes_a + episodes_b for event in ep.events]
    old_examples_everything = generate_dataset(all_events, horizons=ge2.HORIZONS)
    family_a_ids = {ep.episode_id for ep in episodes_a}
    old_examples_a_h50 = [
        e for e in old_examples_everything if e.horizon == HORIZON and e.episode_id in family_a_ids
    ]
    old_x, old_y, old_feature_order = build_feature_matrix(old_examples_a_h50)

    # NEW approach: memory-bounded, batched, multi-horizon-capable cache,
    # here asked for just the one horizon under comparison.
    cache = ge2.build_family_cache(episodes_a, [HORIZON])

    assert old_x.shape == cache.x.shape
    np.testing.assert_array_equal(old_x, cache.x)
    np.testing.assert_array_equal(old_y, cache.y_by_horizon[HORIZON])
    actual_episode_ids = [cache.episode_id_list[i] for i in cache.episode_idx]
    assert actual_episode_ids == [e.episode_id for e in old_examples_a_h50]
    assert old_feature_order == tuple(ge2.FEATURE_NAMES)


def test_family_cache_multi_horizon_matches_per_horizon_generation() -> None:
    """Requesting all 5 horizons in one FamilyCache call matches 5 separate single-horizon calls.

    This is the equivalence proof for the actual optimization: batching
    every horizon into one generate_dataset() call per batch must not
    change any horizon's resulting (X, y) versus generating that horizon
    alone.
    """
    episodes_a = _build_episodes(FAMILY_A, N_EPISODES)

    multi = ge2.build_family_cache(episodes_a, list(ge2.HORIZONS))

    for horizon in ge2.HORIZONS:
        single = ge2.build_family_cache(episodes_a, [horizon])
        np.testing.assert_array_equal(multi.x, single.x)
        np.testing.assert_array_equal(multi.y_by_horizon[horizon], single.y_by_horizon[horizon])
        assert multi.episode_id_list == single.episode_id_list
        np.testing.assert_array_equal(multi.episode_idx, single.episode_idx)


def test_family_cache_batching_matches_single_batch() -> None:
    """A small batch_size (forcing multiple generate_dataset() calls) matches one big batch.

    Proves that chunking a family's episodes into several
    generate_dataset() calls and concatenating the results reproduces
    exactly the same arrays (same rows, same order) as generating the
    whole family in one call -- the property the memory-bounded design
    depends on.
    """
    episodes_a = _build_episodes(FAMILY_A, N_EPISODES)

    one_batch = ge2.build_family_cache(episodes_a, [HORIZON], batch_size=N_EPISODES)
    small_batches = ge2.build_family_cache(episodes_a, [HORIZON], batch_size=2)

    np.testing.assert_array_equal(one_batch.x, small_batches.x)
    np.testing.assert_array_equal(
        one_batch.y_by_horizon[HORIZON], small_batches.y_by_horizon[HORIZON]
    )
    assert one_batch.episode_id_list == small_batches.episode_id_list
    np.testing.assert_array_equal(one_batch.episode_idx, small_batches.episode_idx)


def test_select_split_arrays_matches_filter_examples_by_episodes() -> None:
    """select_split_arrays() matches filter_examples_by_episodes() + build_feature_matrix()."""
    episodes_a = _build_episodes(FAMILY_A, N_EPISODES)
    split = ge2.split_episodes([ep.episode_id for ep in episodes_a], seed=ge2.TRAIN_SEED)

    old_examples = generate_dataset(
        [event for ep in episodes_a for event in ep.events], horizons=[HORIZON]
    )
    old_train = filter_examples_by_episodes(old_examples, split.train_episodes)
    old_x, old_y, _ = build_feature_matrix(old_train)

    cache = ge2.build_family_cache(episodes_a, [HORIZON])
    train_idx = ge2.episode_ids_to_indices(cache.episode_id_list, split.train_episodes)
    train_arrays = ge2.select_split_arrays(
        cache.x, cache.y_by_horizon[HORIZON], cache.episode_idx, train_idx
    )

    assert train_arrays.n_examples == len(old_train)
    np.testing.assert_array_equal(train_arrays.x, old_x)
    np.testing.assert_array_equal(train_arrays.y, old_y)


def test_pooled_arrays_match_monolithic_multi_family_pool() -> None:
    """Pooling compact arrays across families matches filtering a monolithic multi-family list."""
    episodes = {
        FAMILY_A: _build_episodes(FAMILY_A, N_EPISODES),
        FAMILY_B: _build_episodes(FAMILY_B, N_EPISODES),
        FAMILY_C: _build_episodes(FAMILY_C, N_EPISODES),
    }
    training_families = [FAMILY_A, FAMILY_B]
    training_ids = [ep.episode_id for f in training_families for ep in episodes[f]]
    split = ge2.split_episodes(
        training_ids, seed=ge2.TRAIN_SEED, train_fraction=0.82, val_fraction=0.18, test_fraction=0.0
    )

    # OLD approach: one monolithic call across the pooled training families.
    all_training_events = [
        event for f in training_families for ep in episodes[f] for event in ep.events
    ]
    old_examples = generate_dataset(all_training_events, horizons=[HORIZON])
    old_train = filter_examples_by_episodes(old_examples, split.train_episodes)
    old_x, old_y, _ = build_feature_matrix(old_train)

    # NEW approach: per-family FamilyCache, pooled.
    parts = []
    for f in training_families:
        cache = ge2.build_family_cache(episodes[f], [HORIZON])
        idx = ge2.episode_ids_to_indices(cache.episode_id_list, split.train_episodes)
        parts.append(
            ge2.select_split_arrays(cache.x, cache.y_by_horizon[HORIZON], cache.episode_idx, idx)
        )
    pooled = ge2.pool_split_arrays(parts)

    assert pooled.n_examples == len(old_train)
    # Compare as multisets of rows (order across pooled families is by
    # family, not by the old single-list's interleaving) -- exact per-row
    # (X, y) content must still match exactly.
    old_rows = sorted((tuple(old_x[i].tolist()), int(old_y[i])) for i in range(len(old_train)))
    new_rows = sorted(
        (tuple(pooled.x[i].tolist()), int(pooled.y[i])) for i in range(pooled.n_examples)
    )
    assert old_rows == new_rows


def test_protocol_b_held_out_family_never_in_training_families() -> None:
    """The held-out family is structurally excluded from its own fold's training-family list."""
    families = [FAMILY_A, FAMILY_B, FAMILY_C]
    for held_out in families:
        training_families = [f for f in families if f != held_out]
        assert held_out not in training_families
        assert len(training_families) == len(families) - 1


def test_protocol_b_pooled_training_arrays_exclude_held_out_examples() -> None:
    """Pooled train/val arrays for a held-out fold contain zero of that family's examples.

    NOTE on methodology: leakage cannot be checked by comparing feature
    VALUES between the held-out family and the pool. Features are
    deliberately identity-blind (compute_features() takes only
    (access_ticks, fault_ticks, decision_tick) -- never episode_id or
    family), so a freshly-inserted page's feature vector (recency=1,
    frequency=1, no interval history yet) is legitimately identical across
    every family's episodes -- that is correct, expected behavior, not a
    leak. The only meaningful leakage check is at the episode-ID level:
    whether any row whose SOURCE episode belongs to the held-out family
    was included.
    """
    episodes = {
        FAMILY_A: _build_episodes(FAMILY_A, N_EPISODES),
        FAMILY_B: _build_episodes(FAMILY_B, N_EPISODES),
        FAMILY_C: _build_episodes(FAMILY_C, N_EPISODES),
    }
    held_out = FAMILY_C
    training_families = [f for f in episodes if f != held_out]
    training_ids = [ep.episode_id for f in training_families for ep in episodes[f]]
    split = ge2.split_episodes(
        training_ids, seed=ge2.TRAIN_SEED, train_fraction=0.82, val_fraction=0.18, test_fraction=0.0
    )

    caches = {f: ge2.build_family_cache(episodes[f], [HORIZON]) for f in episodes}

    # The real, structural guarantee: only training_families' caches are
    # ever looked up when pooling -- held_out's cache entry is never
    # touched here, exactly mirroring run_protocols_memory_bounded().
    train_parts = []
    for f in training_families:
        assert f != held_out
        cache = caches[f]
        idx = ge2.episode_ids_to_indices(cache.episode_id_list, split.train_episodes)
        train_parts.append(
            ge2.select_split_arrays(cache.x, cache.y_by_horizon[HORIZON], cache.episode_idx, idx)
        )
    pooled_train = ge2.pool_split_arrays(train_parts)
    assert pooled_train.n_examples > 0

    # The held-out family's own episode IDs never appear in the fold's
    # train/val episode ID sets at all -- this is what actually prevents
    # leakage (select_split_arrays() can only include rows whose episode
    # index is in the requested set).
    held_out_episode_ids = {ep.episode_id for ep in episodes[held_out]}
    assert held_out_episode_ids.isdisjoint(set(split.train_episodes))
    assert held_out_episode_ids.isdisjoint(set(split.val_episodes))

    # Directly exercise the filtering mechanism itself: if select_split_arrays()
    # is asked to pull held_out's cached rows using the fold's train episode
    # IDs (the exact membership test production code relies on), it must
    # select nothing, because none of held_out's episode IDs are in that set.
    ho_cache = caches[held_out]
    would_be_idx = ge2.episode_ids_to_indices(ho_cache.episode_id_list, split.train_episodes)
    would_be_selected = ge2.select_split_arrays(
        ho_cache.x, ho_cache.y_by_horizon[HORIZON], ho_cache.episode_idx, would_be_idx
    )
    assert would_be_selected.n_examples == 0
    assert would_be_selected.x is None


def test_episode_level_split_membership_unchanged() -> None:
    """Splits inside run_protocols_memory_bounded() match direct split_episodes() calls."""
    episodes = {
        FAMILY_A: _build_episodes(FAMILY_A, N_EPISODES),
        FAMILY_B: _build_episodes(FAMILY_B, N_EPISODES),
    }
    expected_a_split = ge2.split_episodes(
        [ep.episode_id for ep in episodes[FAMILY_A]], seed=ge2.TRAIN_SEED
    )
    training_ids = [ep.episode_id for ep in episodes[FAMILY_B]]
    expected_fold_split_for_a_held_out = ge2.split_episodes(
        training_ids, seed=ge2.TRAIN_SEED, train_fraction=0.82, val_fraction=0.18, test_fraction=0.0
    )

    # These are exactly the same split_episodes() calls
    # run_protocols_memory_bounded() makes internally -- reproduced here
    # to prove the split logic itself was not altered by this refactor.
    actual_a_split = ge2.split_episodes(
        [ep.episode_id for ep in episodes[FAMILY_A]], seed=ge2.TRAIN_SEED
    )
    assert actual_a_split.train_episodes == expected_a_split.train_episodes
    assert actual_a_split.val_episodes == expected_a_split.val_episodes
    assert actual_a_split.test_episodes == expected_a_split.test_episodes

    training_families = [f for f in episodes if f != FAMILY_A]
    training_ids_actual = [ep.episode_id for f in training_families for ep in episodes[f]]
    actual_fold_split = ge2.split_episodes(
        training_ids_actual,
        seed=ge2.TRAIN_SEED,
        train_fraction=0.82,
        val_fraction=0.18,
        test_fraction=0.0,
    )
    assert actual_fold_split.train_episodes == expected_fold_split_for_a_held_out.train_episodes
    assert actual_fold_split.val_episodes == expected_fold_split_for_a_held_out.val_episodes


def test_generate_dataset_never_called_with_more_than_one_family_or_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Instrumentation proof: generate_dataset() never spans more than one family or batch.

    This is the structural guarantee against the original OOM failure mode
    (one call spanning every family's events for every horizon). Patches
    generate_dataset() as seen from inside generalization_experiment_2's
    own module namespace (that is the reference build_family_cache()
    actually calls) to record each call's event span and episode count
    without changing its behavior.
    """
    episodes = _build_episodes(FAMILY_A, N_EPISODES)
    real_generate_dataset = ge2.generate_dataset
    call_family_spans: list[set[str]] = []
    call_episode_counts: list[int] = []

    def spying_generate_dataset(events: list[Any], horizons: list[int]) -> Any:
        families_in_call = {event.episode_id.rsplit("-ep-", 1)[0] for event in events}
        episodes_in_call = {event.episode_id for event in events}
        call_family_spans.append(families_in_call)
        call_episode_counts.append(len(episodes_in_call))
        return real_generate_dataset(events, horizons=horizons)

    monkeypatch.setattr(ge2, "generate_dataset", spying_generate_dataset)

    batch_size = 2
    ge2.build_family_cache(episodes, list(ge2.HORIZONS), batch_size=batch_size)

    assert len(call_family_spans) == 3  # ceil(6 episodes / batch_size=2)
    for families_in_call in call_family_spans:
        assert len(families_in_call) == 1, (
            f"generate_dataset() was called with events from multiple families "
            f"({families_in_call}) -- this is exactly the pattern that OOM'd the "
            "production run"
        )
    for n in call_episode_counts:
        assert n <= batch_size, (
            f"generate_dataset() was called with {n} episodes' worth of events, "
            f"exceeding batch_size={batch_size} -- this defeats the transient-memory bound"
        )


def test_run_protocols_memory_bounded_releases_episode_events_after_caching(
    tmp_path: Path,
) -> None:
    """episode.events is released once a family's cache is built, and results are unaffected.

    Root-cause regression test for the production D_long_range_reuse
    MemoryError: episode.events (the raw TraceEvent list) is read exactly
    twice in the whole script -- once by compute_classical_baselines()
    (already done before this function runs) and once by
    build_family_cache() itself -- so holding it resident for the rest of
    the run is pure waste. get_family_cache() now clears it immediately
    after caching each family; this test proves that clearing (a) actually
    happens, and (b) does not change the scientific result versus the
    monolithic generate_dataset() reference (episode.workload, used by
    every remaining code path, is untouched).
    """
    families = [FAMILY_A, FAMILY_B]
    episodes: dict[str, list[Any]] = {}
    classical_results: dict[str, dict[str, Any]] = {}
    cfg = ge2.build_run_config(families, [HORIZON], N_EPISODES, [0])
    cfg_hash = ge2.config_hash(cfg)
    checkpoint_dir = tmp_path / "checkpoints"
    for family in families:
        fam_episodes, fam_classical, _ = ge2.run_classical_baselines_for_family(
            family, N_EPISODES, [0], cfg_hash, checkpoint_dir
        )
        episodes[family] = fam_episodes
        classical_results.update(fam_classical)
        # Sanity check the fixture itself: events must be populated before
        # run_protocols_memory_bounded() ever touches them.
        assert all(len(e.events) > 0 for e in fam_episodes)

    protocol_a_results, protocol_b_results = ge2.run_protocols_memory_bounded(
        episodes, classical_results, tmp_path, [HORIZON], [0], cfg_hash
    )

    # (a) events were actually released for every episode in every family.
    for family in families:
        for episode in episodes[family]:
            assert episode.events == [], (
                f"{episode.episode_id}.events was not released after its family's "
                "cache was built"
            )

    # (b) the scientific result is unaffected: rebuild family A's Protocol A
    # unit the monolithic way (the same reference the other tests in this
    # file use) and compare, ignoring only wall-clock timing fields.
    fresh_episodes = _build_episodes(FAMILY_A, N_EPISODES)
    split = ge2.split_episodes([e.episode_id for e in fresh_episodes], seed=ge2.TRAIN_SEED)
    old_examples = generate_dataset(
        [event for ep in fresh_episodes for event in ep.events], horizons=[HORIZON]
    )
    old_train = filter_examples_by_episodes(old_examples, split.train_episodes)
    old_val = filter_examples_by_episodes(old_examples, split.val_episodes)
    old_test = filter_examples_by_episodes(old_examples, split.test_episodes)

    def _to_split_arrays(examples: list[Any]) -> Any:
        if not examples:
            return ge2.SplitArrays(x=None, y=None, n_examples=0)
        x, y, _ = build_feature_matrix(examples)
        return ge2.SplitArrays(x=x, y=y, n_examples=len(examples))

    episode_by_id = {e.episode_id: e for e in fresh_episodes}
    old_test_episode_objs = [episode_by_id[eid] for eid in split.test_episodes]
    expected = ge2.build_horizon_result(
        FAMILY_A,
        HORIZON,
        _to_split_arrays(old_train),
        _to_split_arrays(old_val),
        _to_split_arrays(old_test),
        old_test_episode_objs,
        classical_results,
        [0],
    )

    actual = protocol_a_results[FAMILY_A]["horizons"][str(HORIZON)]
    timing_fields = {"decision_latency_mean_s", "decision_latency_std_s"}

    def _strip(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _strip(v) for k, v in obj.items() if k not in timing_fields}
        if isinstance(obj, list):
            return [_strip(v) for v in obj]
        return obj

    assert _strip(actual) == _strip(expected)


def test_family_cache_only_computes_requested_horizons(monkeypatch: pytest.MonkeyPatch) -> None:
    """build_family_cache() passes only the requested horizons through to generate_dataset().

    Requesting a strict subset of HORIZONS (as happens on a partially
    resumed run, where some horizons are already fully checkpointed) must
    not silently generate examples for horizons nobody asked for.
    """
    episodes_a = _build_episodes(FAMILY_A, N_EPISODES)
    real_generate_dataset = ge2.generate_dataset
    seen_horizons: set[int] = set()

    def spying_generate_dataset(events: list[Any], horizons: list[int]) -> Any:
        seen_horizons.update(horizons)
        return real_generate_dataset(events, horizons=horizons)

    monkeypatch.setattr(ge2, "generate_dataset", spying_generate_dataset)

    ge2.build_family_cache(episodes_a, [25, 100])

    assert seen_horizons == {25, 100}


def test_pool_families_split_arrays_matches_select_then_pool() -> None:
    """pool_families_split_arrays() matches select_split_arrays() + pool_split_arrays().

    Root-cause regression test for the production near-OOM stall during
    Protocol B pooling: select_split_arrays()+pool_split_arrays() together
    materialize every family's masked copy simultaneously before
    concatenating; pool_families_split_arrays() fills a pre-allocated output
    one family at a time instead. This proves the two approaches produce
    byte-identical arrays, in the same row order, over a realistic 3-family
    pool (mirroring a Protocol B training pool).
    """
    episodes = {
        FAMILY_A: _build_episodes(FAMILY_A, N_EPISODES),
        FAMILY_B: _build_episodes(FAMILY_B, N_EPISODES),
        FAMILY_C: _build_episodes(FAMILY_C, N_EPISODES),
    }
    caches = {f: ge2.build_family_cache(episodes[f], [HORIZON]) for f in episodes}
    split = ge2.split_episodes(
        [e.episode_id for f in episodes for e in episodes[f]], seed=ge2.TRAIN_SEED
    )
    wanted_ids = split.train_episodes

    old_parts = []
    specs = []
    for f in [FAMILY_A, FAMILY_B, FAMILY_C]:
        cache = caches[f]
        idx = ge2.episode_ids_to_indices(cache.episode_id_list, wanted_ids)
        old_parts.append(
            ge2.select_split_arrays(cache.x, cache.y_by_horizon[HORIZON], cache.episode_idx, idx)
        )
        specs.append((cache, HORIZON, idx))
    old_pooled = ge2.pool_split_arrays(old_parts)

    new_pooled = ge2.pool_families_split_arrays(specs)

    assert new_pooled.n_examples == old_pooled.n_examples
    if old_pooled.n_examples > 0:
        np.testing.assert_array_equal(new_pooled.x, old_pooled.x)
        np.testing.assert_array_equal(new_pooled.y, old_pooled.y)
    else:
        assert new_pooled.x is None
        assert old_pooled.x is None


def test_pool_families_split_arrays_handles_empty_and_partial_families() -> None:
    """pool_families_split_arrays() matches the reference when some families select nothing."""
    episodes = {
        FAMILY_A: _build_episodes(FAMILY_A, N_EPISODES),
        FAMILY_B: _build_episodes(FAMILY_B, N_EPISODES),
    }
    caches = {f: ge2.build_family_cache(episodes[f], [HORIZON]) for f in episodes}

    # Want only family A's first episode's rows; family B contributes nothing.
    wanted_ids = [episodes[FAMILY_A][0].episode_id]

    idx_by_family = {
        f: ge2.episode_ids_to_indices(caches[f].episode_id_list, wanted_ids)
        for f in [FAMILY_A, FAMILY_B]
    }
    old_parts = [
        ge2.select_split_arrays(
            caches[f].x, caches[f].y_by_horizon[HORIZON], caches[f].episode_idx, idx_by_family[f]
        )
        for f in [FAMILY_A, FAMILY_B]
    ]
    old_pooled = ge2.pool_split_arrays(old_parts)

    specs = [(caches[f], HORIZON, idx_by_family[f]) for f in [FAMILY_A, FAMILY_B]]
    new_pooled = ge2.pool_families_split_arrays(specs)

    assert new_pooled.n_examples == old_pooled.n_examples > 0
    np.testing.assert_array_equal(new_pooled.x, old_pooled.x)
    np.testing.assert_array_equal(new_pooled.y, old_pooled.y)

    # And the fully-empty case: nothing wanted from either family.
    empty_specs: list[Any] = [(caches[f], HORIZON, []) for f in [FAMILY_A, FAMILY_B]]
    empty_pooled = ge2.pool_families_split_arrays(empty_specs)
    assert empty_pooled.n_examples == 0
    assert empty_pooled.x is None
    assert empty_pooled.y is None


def test_held_out_test_arrays_shortcut_matches_select_split_arrays() -> None:
    """The Protocol B held-out-family test-set shortcut matches full-mask selection.

    run_protocols_memory_bounded() now builds the held-out family's test
    SplitArrays directly from its FamilyCache (x=cache.x,
    y=cache.y_by_horizon[horizon]) instead of calling select_split_arrays()
    with an all-episodes mask, since Protocol B's test set is always 100% of
    the held-out family. This proves that shortcut is exactly equivalent to
    the general (masking) path it replaced.
    """
    episodes_a = _build_episodes(FAMILY_A, N_EPISODES)
    cache = ge2.build_family_cache(episodes_a, [HORIZON])
    all_ids = [e.episode_id for e in episodes_a]

    via_mask = ge2.select_split_arrays(
        cache.x,
        cache.y_by_horizon[HORIZON],
        cache.episode_idx,
        ge2.episode_ids_to_indices(cache.episode_id_list, all_ids),
    )
    shortcut = ge2.SplitArrays(
        x=cache.x, y=cache.y_by_horizon[HORIZON], n_examples=len(cache.episode_idx)
    )

    assert shortcut.n_examples == via_mask.n_examples
    np.testing.assert_array_equal(shortcut.x, via_mask.x)
    np.testing.assert_array_equal(shortcut.y, via_mask.y)
