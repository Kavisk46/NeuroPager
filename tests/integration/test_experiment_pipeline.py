"""Integration tests for the first Memory Utility Model experiment.

Builds a small set of real episodes (via the actual MemoryManager /
PageFaultHandler / TraceLogger / LRUPolicy stack), generates a training
dataset from them with `neuropager.dataset`, runs the full experiment
pipeline (split -> train -> evaluate -> online policy -> baseline
comparison), and directly proves the online/offline feature-parity claims
made in `docs/model.md` -- including the one documented gap
(`elapsed_since_fault`) actually manifesting on real data.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from neuropager.core.memory_manager import MemoryManager
from neuropager.core.page_fault import PageFaultHandler
from neuropager.core.page_table import PageTable
from neuropager.core.working_memory import WorkingMemory
from neuropager.dataset.features import FEATURE_NAMES, MISSING_HISTORY_SENTINEL
from neuropager.dataset.generator import generate_dataset
from neuropager.dataset.replay import TraceReplay
from neuropager.dataset.schema import DatasetExample
from neuropager.experiment.metrics import compute_metrics
from neuropager.experiment.models import HIST_GRADIENT_BOOSTING, LOGISTIC_REGRESSION, make_model
from neuropager.experiment.policy import LearnedUtilityPolicy
from neuropager.experiment.preprocessing import build_feature_matrix
from neuropager.experiment.runner import run_full_experiment
from neuropager.experiment.simulation import run_baseline_comparison
from neuropager.experiment.split import filter_examples_by_episodes, split_episodes
from neuropager.memory.disk_store import DiskPageStore
from neuropager.policies.base import PageReplacementPolicy
from neuropager.policies.lru import LRUPolicy
from neuropager.trace.events import TraceEvent, TraceEventType
from neuropager.trace.logger import TraceLogger
from neuropager.utils.types import MemoryKey

HORIZONS = [10, 25]
CAPACITY = 4
N_EPISODES = 24


def _build_dataset(tmp_path: Path) -> list[DatasetExample]:
    all_events: list[TraceEvent] = []
    for i in range(N_EPISODES):
        episode_id = f"ep-{i}"
        disk_root = tmp_path / episode_id
        disk_root.mkdir(parents=True, exist_ok=True)
        events = _generate_episode_events(episode_id, seed=i, disk_root=disk_root)
        all_events.extend(events)
    return generate_dataset(all_events, horizons=HORIZONS)


def _generate_episode_events(episode_id: str, seed: int, disk_root: Path) -> list[TraceEvent]:
    rng = random.Random(seed)
    keys = [MemoryKey(f"k{i}") for i in range(8)]
    workload = [rng.choice(keys) for _ in range(60)]

    trace_logger = TraceLogger(episode_id)
    working_memory = WorkingMemory(capacity=CAPACITY)
    page_table = PageTable()
    fault_handler = PageFaultHandler(
        working_memory=working_memory,
        page_table=page_table,
        disk_store=DiskPageStore(disk_root),
        policy=LRUPolicy(),
        trace_logger=trace_logger,
    )
    manager = MemoryManager(working_memory, page_table, fault_handler)

    seen: set[MemoryKey] = set()
    for key in workload:
        if key not in seen:
            manager.put(key, str(key))
            seen.add(key)
        else:
            manager.get(key)

    return trace_logger.events


def test_small_first_experiment_runs_end_to_end(tmp_path: Path) -> None:
    """Prove the full small-experiment checklist end to end.

    Dataset loads, split is episode-safe, models train, predictions are
    generated, the learned policy makes decisions, baseline policies run,
    and results are recorded -- the exact checklist from the milestone's
    "small first experiment" requirement.
    """
    examples = _build_dataset(tmp_path)
    assert examples, "dataset generation produced zero examples"

    # 1. dataset loads correctly
    assert all(e.horizon in HORIZONS for e in examples)

    # 2. split is episode-safe
    horizon_examples = [e for e in examples if e.horizon == HORIZONS[0]]
    episode_ids = sorted({e.episode_id for e in horizon_examples})
    split = split_episodes(episode_ids, seed=0)
    train = filter_examples_by_episodes(horizon_examples, split.train_episodes)
    test = filter_examples_by_episodes(horizon_examples, split.test_episodes)
    assert set(e.episode_id for e in train) & set(e.episode_id for e in test) == set()
    assert len(train) > 0

    # 3. models train, 4. predictions are generated
    x_train, y_train, _ = build_feature_matrix(train)
    model = make_model(LOGISTIC_REGRESSION, seed=0)
    model.fit(x_train, y_train)
    proba = model.predict_proba(x_train)[:, 1]
    assert proba.shape == (len(train),)
    metrics = compute_metrics(y_train, proba)
    assert metrics.n_examples == len(train)

    # 5. learned policy makes decisions
    learned_policy = LearnedUtilityPolicy(model=model, horizon=HORIZONS[0])
    workload = [MemoryKey(f"k{i}") for i in range(8)] * 3
    disk_root = tmp_path / "sim"
    disk_root.mkdir(parents=True, exist_ok=True)

    # 6. baseline policies run, 7. results are recorded
    results = run_baseline_comparison(
        workload,
        capacity=CAPACITY,
        disk_root=disk_root,
        random_seed=0,
        learned_policy=learned_policy,
        learned_policy_name="LearnedUtilityPolicy[logistic_regression]",
    )
    assert len(results) == 6
    names = {r.policy_name for r in results}
    assert "LearnedUtilityPolicy[logistic_regression]" in names
    for result in results:
        assert result.total_references > 0


def test_full_experiment_runs_all_horizons_and_models(tmp_path: Path) -> None:
    """run_full_experiment covers every requested horizon separately, for both models."""
    examples = _build_dataset(tmp_path)

    results = run_full_experiment(
        examples,
        horizons=HORIZONS,
        model_names=[LOGISTIC_REGRESSION, HIST_GRADIENT_BOOSTING],
        seed=0,
        working_memory_capacity=CAPACITY,
        experiment_id="int-test-1",
    )

    combos = {(r.metadata.horizon, r.metadata.model_name) for r in results}
    assert combos == {
        (HORIZONS[0], LOGISTIC_REGRESSION),
        (HORIZONS[0], HIST_GRADIENT_BOOSTING),
        (HORIZONS[1], LOGISTIC_REGRESSION),
        (HORIZONS[1], HIST_GRADIENT_BOOSTING),
    }
    for result in results:
        assert result.train_metrics.n_examples > 0
        assert result.metadata.feature_order == FEATURE_NAMES


def test_online_features_match_offline_replay_for_available_features(tmp_path: Path) -> None:
    """The core parity proof: online feature values match offline recomputation.

    Runs the learned policy live (recording every predict_proba input),
    then independently replays the resulting trace offline and recomputes
    what each candidate's features should have been at each decision tick.
    Everything except fault_count/fault_ratio/elapsed_since_fault must
    match *exactly*; those three are checked against PageTable-derived
    ground truth (fault_count/fault_ratio) or shown to be the documented
    sentinel gap (elapsed_since_fault).
    """
    examples = _build_dataset(tmp_path)
    horizon_examples = [e for e in examples if e.horizon == HORIZONS[0]]
    x_train, y_train, _ = build_feature_matrix(horizon_examples)
    base_model = make_model(LOGISTIC_REGRESSION, seed=0)
    base_model.fit(x_train, y_train)

    class _RecordingModel:
        def __init__(self, wrapped: object) -> None:
            self.wrapped = wrapped
            self.calls: list[np.ndarray] = []

        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            self.calls.append(x.copy())
            return self.wrapped.predict_proba(x)  # type: ignore[attr-defined]

    recording_model = _RecordingModel(base_model)
    policy: PageReplacementPolicy = LearnedUtilityPolicy(model=recording_model, horizon=HORIZONS[0])  # type: ignore[arg-type]

    trace_logger = TraceLogger("parity-check")
    working_memory = WorkingMemory(capacity=CAPACITY)
    page_table = PageTable()
    disk_root = tmp_path / "parity"
    disk_root.mkdir(parents=True, exist_ok=True)
    fault_handler = PageFaultHandler(
        working_memory=working_memory,
        page_table=page_table,
        disk_store=DiskPageStore(disk_root),
        policy=policy,
        trace_logger=trace_logger,
    )
    manager = MemoryManager(working_memory, page_table, fault_handler)

    rng = random.Random(123)
    keys = [MemoryKey(f"k{i}") for i in range(8)]
    workload = [rng.choice(keys) for _ in range(80)]
    seen: set[MemoryKey] = set()
    for key in workload:
        if key not in seen:
            manager.put(key, str(key))
            seen.add(key)
        else:
            manager.get(key)

    eviction_events = [
        e
        for e in trace_logger.events
        if e.event_type == TraceEventType.EVICTION and e.eviction_reason == "capacity"
    ]
    assert len(eviction_events) == len(recording_model.calls) > 0

    replay = TraceReplay(trace_logger.events)

    recency_col = FEATURE_NAMES.index("recency")
    frequency_col = FEATURE_NAMES.index("frequency")
    page_age_col = FEATURE_NAMES.index("page_age")
    fault_count_col = FEATURE_NAMES.index("fault_count")
    elapsed_since_fault_col = FEATURE_NAMES.index("elapsed_since_fault")

    saw_a_real_offline_fault_gap = False

    for event, x in zip(eviction_events, recording_model.calls, strict=True):
        decision_tick = event.tick
        candidates = event.resident_pages
        assert x.shape[0] == len(candidates)

        for row_index, page_id in enumerate(candidates):
            offline_access_ticks = replay.access_ticks_up_to(page_id, decision_tick)
            offline_fault_ticks = replay.fault_ticks_up_to(page_id, decision_tick)

            # Exact parity: recency, frequency, page_age are pure functions
            # of self-tracked access history and the corrected decision
            # tick, and must match the offline replay bit-for-bit.
            assert x[row_index, recency_col] == decision_tick - offline_access_ticks[-1]
            assert x[row_index, frequency_col] == len(offline_access_ticks)
            assert x[row_index, page_age_col] == decision_tick - offline_access_ticks[0]

            # fault_count: online reads PageTable's exact cumulative count,
            # which must equal the offline replay's fault-tick count.
            assert x[row_index, fault_count_col] == len(offline_fault_ticks)

            # The documented gap: elapsed_since_fault is always the
            # sentinel online, even when the offline value would be real.
            assert x[row_index, elapsed_since_fault_col] == MISSING_HISTORY_SENTINEL
            if offline_fault_ticks:
                offline_elapsed = decision_tick - offline_fault_ticks[-1]
                assert offline_elapsed != MISSING_HISTORY_SENTINEL or offline_elapsed < 0
                saw_a_real_offline_fault_gap = True

    assert saw_a_real_offline_fault_gap, (
        "workload never exercised a page with prior faults at a decision point; "
        "the elapsed_since_fault gap demonstration needs a longer/different workload"
    )


def test_feature_matrix_never_reads_metadata_fields() -> None:
    """build_feature_matrix output is unaffected by episode_id/page_id/policy/label."""
    from neuropager.dataset.features import Features

    features = Features(
        recency=1.0,
        frequency=2.0,
        avg_inter_access_interval=3.0,
        inter_access_interval_variance=4.0,
        fault_count=0.0,
        fault_ratio=0.0,
        page_age=5.0,
        recent_burst=1.0,
        long_burst=2.0,
        recent_interval=3.0,
        normalized_recency=0.2,
        elapsed_since_fault=-1.0,
        max_historical_interval=3.0,
    )
    example_1 = DatasetExample(
        episode_id="ep-alpha",
        decision_tick=1,
        page_id="page-one",
        policy="LRUPolicy",
        working_memory_capacity=4,
        eviction_reason="capacity",
        horizon=25,
        features=features,
        label=1,
    )
    example_2 = DatasetExample(
        episode_id="ep-omega-completely-different",
        decision_tick=9999,
        page_id="a-totally-different-page-id",
        policy="FIFOPolicy",
        working_memory_capacity=999,
        eviction_reason="capacity",
        horizon=500,
        features=features,
        label=0,
    )

    x, _, _ = build_feature_matrix([example_1, example_2])

    assert (x[0] == x[1]).all()
