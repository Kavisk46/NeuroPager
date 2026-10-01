"""Tests for scripts/generalization_experiment_2.py's checkpoint/resume mechanism.

scripts/ is not a package, so the module under test is loaded directly from
its file path (matching the pattern already used to smoke-test this script
earlier in the project). Every test here exercises only the orchestration
layer (checkpointing, resume-skip, config-hash validation) -- never the
scientific functions (build_episode, compute_classical_baselines,
build_horizon_result, etc.), which are exercised by their own existing
tests and are not touched by this module.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "generalization_experiment_2.py"
_spec = importlib.util.spec_from_file_location("generalization_experiment_2", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
ge2: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["generalization_experiment_2"] = ge2
_spec.loader.exec_module(ge2)


FAMILY = "A_uniform_random"
SEEDS = [0]


def test_atomic_write_leaves_no_temp_files(tmp_path: Path) -> None:
    """write_json leaves only the target file behind, never a stray temp file."""
    target = tmp_path / "out.json"
    ge2.write_json({"a": 1}, target)
    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    assert list(tmp_path.glob("*.tmp*")) == []


def test_atomic_write_never_touches_target_before_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the atomic rename never happens, the original target is left untouched."""
    target = tmp_path / "out.json"
    ge2.write_json({"version": 1}, target)

    def _boom(*args: object, **kwargs: object) -> None:
        raise OSError("simulated crash between write and rename")

    monkeypatch.setattr(ge2.os, "replace", _boom)
    with pytest.raises(OSError):
        ge2.write_json({"version": 2}, target)

    # The original, complete content must still be there -- never truncated
    # or partially overwritten.
    assert json.loads(target.read_text(encoding="utf-8")) == {"version": 1}


def test_load_checkpoint_missing_file_returns_none(tmp_path: Path) -> None:
    """A checkpoint that has never been written is a normal, expected None."""
    assert ge2.load_checkpoint(tmp_path / "nope.json", "somehash") is None


def test_load_checkpoint_rejects_corrupt_json(tmp_path: Path) -> None:
    """Invalid JSON is never silently treated as 'not done yet' -- it raises."""
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ge2.CorruptCheckpointError):
        ge2.load_checkpoint(path, "somehash")


def test_load_checkpoint_rejects_missing_schema_fields(tmp_path: Path) -> None:
    """Valid JSON missing required checkpoint fields is rejected, not silently accepted."""
    path = tmp_path / "bad_schema.json"
    path.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
    with pytest.raises(ge2.CorruptCheckpointError):
        ge2.load_checkpoint(path, "somehash")


def test_load_checkpoint_rejects_wrong_schema_version(tmp_path: Path) -> None:
    """A checkpoint from an incompatible schema version is refused, not reinterpreted."""
    path = tmp_path / "old_schema.json"
    ge2.write_json({"schema_version": "0", "config_hash": "somehash", "payload": 1}, path)
    with pytest.raises(ge2.CorruptCheckpointError):
        ge2.load_checkpoint(path, "somehash")


def test_load_checkpoint_rejects_config_hash_mismatch(tmp_path: Path) -> None:
    """A checkpoint written under a different configuration is refused, never silently resumed."""
    path = tmp_path / "checkpoint.json"
    ge2.write_json(
        {
            "schema_version": ge2.CHECKPOINT_SCHEMA_VERSION,
            "config_hash": "hash-from-a-different-run",
            "payload": 1,
        },
        path,
    )
    with pytest.raises(ge2.ConfigMismatchError):
        ge2.load_checkpoint(path, "current-hash")


def test_load_checkpoint_rejects_identifying_field_mismatch(tmp_path: Path) -> None:
    """Extra identifying fields (family, horizon, ...) are validated, not just config_hash."""
    path = tmp_path / "checkpoint.json"
    cfg_hash = "matching-hash"
    ge2.write_json(
        {
            "schema_version": ge2.CHECKPOINT_SCHEMA_VERSION,
            "config_hash": cfg_hash,
            "family": "B_temporal_locality",
            "payload": 1,
        },
        path,
    )
    with pytest.raises(ge2.ConfigMismatchError):
        ge2.load_checkpoint(path, cfg_hash, family="A_uniform_random")


def test_load_checkpoint_accepts_matching_checkpoint(tmp_path: Path) -> None:
    """A checkpoint matching both config_hash and identifying fields loads cleanly."""
    path = tmp_path / "checkpoint.json"
    cfg_hash = "matching-hash"
    ge2.write_json(
        {
            "schema_version": ge2.CHECKPOINT_SCHEMA_VERSION,
            "config_hash": cfg_hash,
            "family": "A_uniform_random",
            "payload": 42,
        },
        path,
    )
    result = ge2.load_checkpoint(path, cfg_hash, family="A_uniform_random")
    assert result is not None
    assert result["payload"] == 42


def test_build_run_config_deterministic_and_seed_sensitive() -> None:
    """The same inputs hash identically; changing a locked seed changes the hash."""
    cfg1 = ge2.build_run_config([FAMILY], [50], 4, [0])
    cfg2 = ge2.build_run_config([FAMILY], [50], 4, [0])
    assert ge2.config_hash(cfg1) == ge2.config_hash(cfg2)

    cfg3 = ge2.build_run_config([FAMILY], [50], 4, [0, 1])
    assert ge2.config_hash(cfg1) != ge2.config_hash(cfg3)


def test_classical_baselines_fresh_run_computes_all_episodes(tmp_path: Path) -> None:
    """A first run with no checkpoint computes every episode fresh."""
    cfg_hash = ge2.config_hash(ge2.build_run_config([FAMILY], [50], 3, SEEDS))
    episodes, results, n_fresh = ge2.run_classical_baselines_for_family(
        FAMILY, 3, SEEDS, cfg_hash, tmp_path
    )
    assert len(episodes) == 3
    assert n_fresh == 3
    assert {e.episode_id for e in episodes} == set(results)
    checkpoint_path = tmp_path / f"classical_{FAMILY}.json"
    assert checkpoint_path.exists()
    saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert len(saved["episodes"]) == 3


def test_already_completed_work_is_skipped(tmp_path: Path) -> None:
    """Calling again with identical config recomputes nothing (n_freshly_computed == 0)."""
    cfg_hash = ge2.config_hash(ge2.build_run_config([FAMILY], [50], 3, SEEDS))
    _, first_results, _ = ge2.run_classical_baselines_for_family(
        FAMILY, 3, SEEDS, cfg_hash, tmp_path
    )
    episodes2, second_results, n_fresh2 = ge2.run_classical_baselines_for_family(
        FAMILY, 3, SEEDS, cfg_hash, tmp_path
    )
    assert n_fresh2 == 0
    assert len(episodes2) == 3
    # Same episode IDs, same policy result values -- proves resumed results
    # are identical to the originally-computed ones, not placeholders.
    for episode_id in first_results:
        for policy_name in first_results[episode_id]:
            assert (
                first_results[episode_id][policy_name].page_faults
                == second_results[episode_id][policy_name].page_faults
            )


def test_partial_work_is_resumed(tmp_path: Path) -> None:
    """Only the episodes missing from an existing partial checkpoint are computed."""
    cfg_hash = ge2.config_hash(ge2.build_run_config([FAMILY], [50], 4, SEEDS))
    # Simulate a death partway through a 4-episode family: only 2 episodes done.
    _, partial_results, n_fresh_partial = ge2.run_classical_baselines_for_family(
        FAMILY, 2, SEEDS, cfg_hash, tmp_path
    )
    assert n_fresh_partial == 2
    # Manually splice the 2-episode checkpoint into place as a "4-episode run
    # that died after 2" would leave it -- write_json already did this via
    # the n_episodes=2 call above, at path classical_<FAMILY>.json. Resuming
    # with n_episodes=4 must compute exactly the 2 missing episodes.
    episodes_full, full_results, n_fresh_full = ge2.run_classical_baselines_for_family(
        FAMILY, 4, SEEDS, cfg_hash, tmp_path
    )
    assert n_fresh_full == 2
    assert len(episodes_full) == 4
    assert len(full_results) == 4
    for episode_id in partial_results:
        assert episode_id in full_results


def test_no_duplicate_episode_entries_after_resume(tmp_path: Path) -> None:
    """Resuming twice in a row never produces duplicate or extra episode entries."""
    cfg_hash = ge2.config_hash(ge2.build_run_config([FAMILY], [50], 5, SEEDS))
    ge2.run_classical_baselines_for_family(FAMILY, 5, SEEDS, cfg_hash, tmp_path)
    ge2.run_classical_baselines_for_family(FAMILY, 5, SEEDS, cfg_hash, tmp_path)
    episodes3, results3, _ = ge2.run_classical_baselines_for_family(
        FAMILY, 5, SEEDS, cfg_hash, tmp_path
    )
    expected_ids = {f"{FAMILY}-ep-{s}" for s in range(5)}
    assert {e.episode_id for e in episodes3} == expected_ids
    assert set(results3) == expected_ids
    checkpoint_path = tmp_path / f"classical_{FAMILY}.json"
    saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert set(saved["episodes"]) == expected_ids
    assert len(saved["episodes"]) == 5


def test_classical_baselines_rejects_config_mismatch(tmp_path: Path) -> None:
    """A family checkpoint from a different configuration is never silently resumed from."""
    cfg_hash_a = ge2.config_hash(ge2.build_run_config([FAMILY], [50], 3, [0]))
    cfg_hash_b = ge2.config_hash(ge2.build_run_config([FAMILY], [50], 3, [0, 1]))
    assert cfg_hash_a != cfg_hash_b

    ge2.run_classical_baselines_for_family(FAMILY, 3, [0], cfg_hash_a, tmp_path)
    with pytest.raises(ge2.ConfigMismatchError):
        ge2.run_classical_baselines_for_family(FAMILY, 3, [0, 1], cfg_hash_b, tmp_path)


def test_migrate_legacy_checkpoint_only_migrates_complete_families(tmp_path: Path) -> None:
    """Legacy migration preserves complete families and leaves incomplete ones alone."""
    experiment_root = tmp_path
    checkpoint_dir = experiment_root / "checkpoints"
    checkpoint_dir.mkdir()
    cfg_hash = ge2.config_hash(
        ge2.build_run_config(["A_uniform_random", "B_temporal_locality"], [50], 2, SEEDS)
    )

    policy_stub = {
        "page_faults": 1,
        "total_references": 2,
        "hits": 1,
        "hit_ratio": 0.5,
        "decision_latency_mean_s": 0.0,
        "decision_latency_std_s": 0.0,
    }
    legacy = {
        "A_uniform_random-ep-0": {"LRU": policy_stub},
        "A_uniform_random-ep-1": {"LRU": policy_stub},
        # B is incomplete: only 1 of 2 expected episodes present.
        "B_temporal_locality-ep-0": {"LRU": policy_stub},
    }
    (experiment_root / "_checkpoint_classical_results.json").write_text(
        json.dumps(legacy), encoding="utf-8"
    )

    ge2.migrate_legacy_classical_checkpoint(experiment_root, checkpoint_dir, cfg_hash, 2)

    assert (checkpoint_dir / "classical_A_uniform_random.json").exists()
    assert not (checkpoint_dir / "classical_B_temporal_locality.json").exists()

    migrated = json.loads((checkpoint_dir / "classical_A_uniform_random.json").read_text())
    assert migrated["config_hash"] == cfg_hash
    assert set(migrated["episodes"]) == {"A_uniform_random-ep-0", "A_uniform_random-ep-1"}


def test_migrate_legacy_checkpoint_never_overwrites_new_format(tmp_path: Path) -> None:
    """Migration never clobbers a checkpoint already written under the new format."""
    experiment_root = tmp_path
    checkpoint_dir = experiment_root / "checkpoints"
    checkpoint_dir.mkdir()
    cfg_hash = ge2.config_hash(ge2.build_run_config([FAMILY], [50], 1, SEEDS))

    new_format_path = checkpoint_dir / f"classical_{FAMILY}.json"
    ge2.write_json(
        {
            "schema_version": ge2.CHECKPOINT_SCHEMA_VERSION,
            "config_hash": cfg_hash,
            "family": FAMILY,
            "n_episodes_target": 1,
            "episodes": {"sentinel": "do-not-overwrite"},
        },
        new_format_path,
    )

    policy_stub = {
        "page_faults": 1,
        "total_references": 2,
        "hits": 1,
        "hit_ratio": 0.5,
        "decision_latency_mean_s": 0.0,
        "decision_latency_std_s": 0.0,
    }
    legacy = {f"{FAMILY}-ep-0": {"LRU": policy_stub}}
    (experiment_root / "_checkpoint_classical_results.json").write_text(
        json.dumps(legacy), encoding="utf-8"
    )

    ge2.migrate_legacy_classical_checkpoint(experiment_root, checkpoint_dir, cfg_hash, 1)

    untouched = json.loads(new_format_path.read_text(encoding="utf-8"))
    assert untouched["episodes"] == {"sentinel": "do-not-overwrite"}
