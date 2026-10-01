#!/usr/bin/env python3
"""One-time empirical proof that old and new orchestration produce identical results.

This is deliberately a standalone script, not a pytest-collected test: it
trains real models (LogisticRegression + HistGradientBoosting) and replays
LearnedUtilityPolicy on real episodes, which is too slow to run on every
`pytest` invocation. The unit tests in tests/unit/test_experiment_memory_bounded.py
already prove, piece by piece, that build_family_cache() produces identical
SplitArrays to the old approach; this script closes the loop by running the
*complete* pipeline (classical baselines -> dataset generation -> split ->
train -> select -> replay -> paired statistics -> Belady gap -> interpretation
flags) both ways on the same fixture and diffing the final result dicts
directly -- exactly the comparison requested for the production sign-off
report.

"Old" here means: generate_dataset() called once per (family, horizon) on
that family's full event list (the monolithic-per-unit pattern every version
of this script's orchestration used before build_family_cache() -- including
the very first, truly monolithic all-families-all-horizons call the original
OOM report described, generalized down to a family-sized fixture where it is
safe to actually run) -- built from the UNCHANGED primitives
(generate_dataset, filter_examples_by_episodes, build_feature_matrix,
split_episodes, build_horizon_result) so the comparison is against real
production logic, not a hand-rolled reference.

"New" means: the actual production entry point, run_experiment(), using
build_family_cache() and run_protocols_memory_bounded() exactly as a real
launch would.

Usage:
    PYTHONPATH=src python -u scripts/verify_old_vs_new_orchestration_equivalence.py
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

_SCRIPT_PATH = Path(__file__).resolve().parent / "generalization_experiment_2.py"
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
HORIZON = 50
N_EPISODES = 4
RANDOM_SEEDS = [0]

TIMING_FIELDS = {"decision_latency_mean_s", "decision_latency_std_s"}


def strip_timing(obj: Any) -> Any:
    """Recursively drop nondeterministic wall-clock timing fields before comparing."""
    if isinstance(obj, dict):
        return {k: strip_timing(v) for k, v in obj.items() if k not in TIMING_FIELDS}
    if isinstance(obj, list):
        return [strip_timing(v) for v in obj]
    return obj


def build_old_split_arrays(family_episodes: list[Any], wanted_episode_ids: list[str]) -> Any:
    """Old-style: generate_dataset() on a family's full event list, then filter+matrix."""
    events = [event for episode in family_episodes for event in episode.events]
    examples = generate_dataset(events, horizons=[HORIZON])
    filtered = filter_examples_by_episodes(examples, wanted_episode_ids)
    if not filtered:
        return ge2.SplitArrays(x=None, y=None, n_examples=0)
    x, y, _ = build_feature_matrix(filtered)
    return ge2.SplitArrays(x=x, y=y, n_examples=len(filtered))


def run_old_path(
    episodes: dict[str, list[Any]], classical_results: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reproduce the pre-FamilyCache orchestration's Protocol A/B results.

    Covers Protocol A (family A) and Protocol B (held_out=A), using only
    unchanged primitives.
    """
    families = list(episodes)

    # Protocol A for family A.
    split_a = ge2.split_episodes([e.episode_id for e in episodes[FAMILY_A]], seed=ge2.TRAIN_SEED)
    train = build_old_split_arrays(episodes[FAMILY_A], list(split_a.train_episodes))
    val = build_old_split_arrays(episodes[FAMILY_A], list(split_a.val_episodes))
    test = build_old_split_arrays(episodes[FAMILY_A], list(split_a.test_episodes))
    episode_by_id = {e.episode_id: e for e in episodes[FAMILY_A]}
    test_episode_objs = [episode_by_id[eid] for eid in split_a.test_episodes]
    protocol_a_result = ge2.build_horizon_result(
        FAMILY_A, HORIZON, train, val, test, test_episode_objs, classical_results, RANDOM_SEEDS
    )

    # Protocol B, held_out=A: train/val pooled from every OTHER family (just B here).
    training_families = [f for f in families if f != FAMILY_A]
    training_ids = [e.episode_id for f in training_families for e in episodes[f]]
    fold_split = ge2.split_episodes(
        training_ids, seed=ge2.TRAIN_SEED, train_fraction=0.82, val_fraction=0.18, test_fraction=0.0
    )
    train_parts = [
        build_old_split_arrays(episodes[f], list(fold_split.train_episodes))
        for f in training_families
    ]
    val_parts = [
        build_old_split_arrays(episodes[f], list(fold_split.val_episodes))
        for f in training_families
    ]
    pooled_train = ge2.pool_split_arrays(train_parts)
    pooled_val = ge2.pool_split_arrays(val_parts)
    test_b = build_old_split_arrays(episodes[FAMILY_A], [e.episode_id for e in episodes[FAMILY_A]])
    protocol_b_result = ge2.build_horizon_result(
        FAMILY_A,
        HORIZON,
        pooled_train,
        pooled_val,
        test_b,
        list(episodes[FAMILY_A]),
        classical_results,
        RANDOM_SEEDS,
    )
    return protocol_a_result, protocol_b_result


def main() -> None:
    """Run both paths on the same fixture and diff the results."""
    print(f"Building episodes + classical baselines for {[FAMILY_A, FAMILY_B]} ...")
    episodes: dict[str, list[Any]] = {}
    classical_results: dict[str, dict[str, Any]] = {}
    tmp_checkpoint_dir = Path(tempfile.mkdtemp(prefix="verify_old_ckpt_"))
    cfg = ge2.build_run_config([FAMILY_A, FAMILY_B], [HORIZON], N_EPISODES, RANDOM_SEEDS)
    cfg_hash = ge2.config_hash(cfg)
    try:
        for family in [FAMILY_A, FAMILY_B]:
            fam_episodes, fam_classical, _ = ge2.run_classical_baselines_for_family(
                family, N_EPISODES, RANDOM_SEEDS, cfg_hash, tmp_checkpoint_dir
            )
            episodes[family] = fam_episodes
            classical_results.update(fam_classical)
    finally:
        shutil.rmtree(tmp_checkpoint_dir, ignore_errors=True)

    print("Running OLD-style orchestration (monolithic generate_dataset() per family)...")
    old_a, old_b = run_old_path(episodes, classical_results)

    print("Running NEW orchestration (run_experiment(), the real production entry point)...")
    new_root = Path(tempfile.mkdtemp(prefix="verify_new_root_"))
    try:
        ge2.run_experiment(
            families=[FAMILY_A, FAMILY_B],
            horizons=[HORIZON],
            n_episodes_per_family=N_EPISODES,
            random_seeds=RANDOM_SEEDS,
            experiment_root=new_root,
            experiment_id="verify-old-vs-new",
        )
        new_protocol_a = json.loads((new_root / "protocol_a_results.json").read_text())
        new_protocol_b = json.loads((new_root / "protocol_b_results.json").read_text())
    finally:
        shutil.rmtree(new_root, ignore_errors=True)

    new_a = new_protocol_a[FAMILY_A]["horizons"][str(HORIZON)]
    new_b = new_protocol_b[FAMILY_A]["horizons"][str(HORIZON)]

    ok_a = strip_timing(old_a) == strip_timing(new_a)
    ok_b = strip_timing(old_b) == strip_timing(new_b)

    print()
    print(f"Protocol A (family={FAMILY_A}, H={HORIZON}): {'MATCH' if ok_a else 'MISMATCH'}")
    print(f"Protocol B (held_out={FAMILY_A}, H={HORIZON}): {'MATCH' if ok_b else 'MISMATCH'}")

    if not (ok_a and ok_b):
        print("\nFAIL: old and new orchestration produced different scientific results.")
        if not ok_a:
            print("--- Protocol A diff ---")
            print(json.dumps(strip_timing(old_a), indent=2, sort_keys=True)[:3000])
            print("--- vs new ---")
            print(json.dumps(strip_timing(new_a), indent=2, sort_keys=True)[:3000])
        sys.exit(1)

    print("\nPASS: old and new orchestration produced identical scientific results.")


if __name__ == "__main__":
    main()
