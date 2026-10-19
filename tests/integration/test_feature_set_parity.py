"""Train/serve column-parity tests for the elapsed_since_fault skew fix's
new feature-set variants (NO_ESF12, NO_RECENCY7).

For the same trace and decision tick, LearnedUtilityPolicy's runtime
feature row must equal TraceReplay's offline recomputation for every
column the active feature set actually kept -- proving
select_feature_columns() (production script) and
LearnedUtilityPolicy.feature_names (policy.py) together produce a model
input that never silently drifts from what training saw, for any of the
three defined feature sets. elapsed_since_fault itself is NOT re-tested
here -- see tests/integration/test_experiment_pipeline.py's
test_online_features_match_offline_replay_for_available_features for the
original documented-gap proof under FULL13; this file only proves that
NO_ESF12/NO_RECENCY7 drop it entirely rather than silently keeping the
sentinel.

scripts/ is not a package, so the module under test is loaded directly
from its file path, matching the pattern used by
tests/unit/test_experiment_memory_bounded.py.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "generalization_experiment_2.py"
_spec = importlib.util.spec_from_file_location("generalization_experiment_2", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
ge2: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["generalization_experiment_2"] = ge2
_spec.loader.exec_module(ge2)

from neuropager.core.memory_manager import MemoryManager  # noqa: E402
from neuropager.core.page_fault import PageFaultHandler  # noqa: E402
from neuropager.core.page_table import PageTable  # noqa: E402
from neuropager.core.working_memory import WorkingMemory  # noqa: E402
from neuropager.dataset.features import FEATURE_NAMES  # noqa: E402
from neuropager.dataset.replay import TraceReplay  # noqa: E402
from neuropager.experiment.policy import LearnedUtilityPolicy  # noqa: E402
from neuropager.memory.disk_store import DiskPageStore  # noqa: E402
from neuropager.trace.events import TraceEventType  # noqa: E402
from neuropager.trace.logger import TraceLogger  # noqa: E402
from neuropager.utils.types import MemoryKey  # noqa: E402

CAPACITY = 4


class _RecordingModel:
    """Records every predict_proba() input; returns fixed, arbitrary scores.

    This test only inspects the INPUT rows LearnedUtilityPolicy builds,
    never the resulting eviction decision, so the returned scores'
    specific values don't matter -- they just need to be valid
    probabilities of the right shape.
    """

    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self.calls.append(x.copy())
        return np.tile([0.5, 0.5], (x.shape[0], 1))


@pytest.mark.parametrize("feature_set_name", ["NO_ESF12", "NO_RECENCY7"])
def test_runtime_row_matches_offline_for_kept_columns(
    tmp_path: Path, feature_set_name: str
) -> None:
    feature_names = ge2.FEATURE_SETS[feature_set_name]
    assert "elapsed_since_fault" not in feature_names, (
        f"{feature_set_name} is supposed to drop elapsed_since_fault"
    )

    recording_model = _RecordingModel()
    policy = LearnedUtilityPolicy(model=recording_model, horizon=25, feature_names=feature_names)

    trace_logger = TraceLogger("parity-" + feature_set_name)
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
    kept_cols = {name: feature_names.index(name) for name in feature_names}

    for event, x in zip(eviction_events, recording_model.calls, strict=True):
        decision_tick = event.tick
        candidates = event.resident_pages
        assert x.shape[1] == len(feature_names)
        assert x.shape[0] == len(candidates)

        for row_index, page_id in enumerate(candidates):
            offline_access_ticks = replay.access_ticks_up_to(page_id, decision_tick)
            offline_fault_ticks = replay.fault_ticks_up_to(page_id, decision_tick)

            # Exact parity, for whichever of these columns this feature
            # set actually kept -- a dropped column was never computed
            # into the row at all, so there's nothing to compare for it.
            if "recency" in kept_cols:
                col = kept_cols["recency"]
                assert x[row_index, col] == decision_tick - offline_access_ticks[-1]
            if "frequency" in kept_cols:
                col = kept_cols["frequency"]
                assert x[row_index, col] == len(offline_access_ticks)
            if "page_age" in kept_cols:
                col = kept_cols["page_age"]
                assert x[row_index, col] == decision_tick - offline_access_ticks[0]
            if "fault_count" in kept_cols:
                col = kept_cols["fault_count"]
                assert x[row_index, col] == len(offline_fault_ticks)


def test_feature_sets_drop_exactly_the_documented_columns() -> None:
    """Pin the three FEATURE_SETS to the exact, user-confirmed column lists."""
    assert ge2.FEATURE_SETS["FULL13"] == FEATURE_NAMES
    assert ge2.FEATURE_SETS["NO_ESF12"] == tuple(
        name for name in FEATURE_NAMES if name != "elapsed_since_fault"
    )
    dropped_by_no_recency7 = set(FEATURE_NAMES) - set(ge2.FEATURE_SETS["NO_RECENCY7"])
    assert dropped_by_no_recency7 == {
        "elapsed_since_fault",
        "recency",
        "recent_burst",
        "long_burst",
        "recent_interval",
        "normalized_recency",
    }


def test_full13_is_a_no_op_column_selection() -> None:
    """select_feature_columns() under FULL13 must return every column, unchanged."""
    x = np.arange(13 * 5, dtype=np.float64).reshape(5, 13)
    split = ge2.SplitArrays(x=x, y=np.zeros(5, dtype=np.int64), n_examples=5)
    result = ge2.select_feature_columns(split, ge2.FEATURE_SETS["FULL13"])
    assert np.array_equal(result.x, x)
