#!/usr/bin/env python3
"""Run the smallest rigorous Memory Utility Model experiment, end to end.

This is a demonstration and smoke test, not a real research result: a
handful of small, synthetic episodes, both baseline models, two horizons,
and one baseline-vs-learned-policy comparison. It exists to show,
concretely, what the experiment pipeline produces.

Do NOT read the printed metrics as evidence the research hypothesis is
supported -- the workload is tiny and synthetic. See docs/model.md.

Usage:
    python scripts/demo_experiment.py
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

from neuropager.core.memory_manager import MemoryManager
from neuropager.core.page_fault import PageFaultHandler
from neuropager.core.page_table import PageTable
from neuropager.core.working_memory import WorkingMemory
from neuropager.dataset.generator import generate_dataset
from neuropager.experiment.models import HIST_GRADIENT_BOOSTING, LOGISTIC_REGRESSION
from neuropager.experiment.policy import LearnedUtilityPolicy
from neuropager.experiment.reproducibility import write_csv, write_json
from neuropager.experiment.runner import ModelHorizonResult, run_full_experiment
from neuropager.experiment.simulation import run_baseline_comparison
from neuropager.memory.disk_store import DiskPageStore
from neuropager.policies.lru import LRUPolicy
from neuropager.trace.events import TraceEvent
from neuropager.trace.logger import TraceLogger
from neuropager.utils.types import MemoryKey

HORIZONS = [10, 25]
CAPACITY = 4
N_EPISODES = 24
SEED = 0


def _generate_episode(episode_id: str, seed: int, disk_root: Path) -> list[TraceEvent]:
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


def _print_result(result: ModelHorizonResult) -> None:
    m = result.metadata
    print(f"\n--- horizon={m.horizon}  model={m.model_name} ---")
    print(
        f"  split: train={len(m.split.train_episodes)} val={len(m.split.val_episodes)} "
        f"test={len(m.split.test_episodes)} episodes"
    )
    for split_name, metrics in (
        ("train", result.train_metrics),
        ("val", result.val_metrics),
        ("test", result.test_metrics),
    ):
        auc = f"{metrics.roc_auc:.3f}" if metrics.roc_auc is not None else "n/a"
        pr_auc = f"{metrics.pr_auc:.3f}" if metrics.pr_auc is not None else "n/a"
        print(
            f"  {split_name:5s} n={metrics.n_examples:4d} "
            f"pos={metrics.n_positive:4d} "
            f"acc={metrics.accuracy:.3f} f1={metrics.f1:.3f} "
            f"roc_auc={auc} pr_auc={pr_auc} brier={metrics.brier_score:.3f}"
        )


def main() -> None:
    """Generate episodes, run the full experiment, and print/write results."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        print(f"=== Generating {N_EPISODES} synthetic episodes ===")
        all_events: list[TraceEvent] = []
        for i in range(N_EPISODES):
            episode_id = f"demo-ep-{i}"
            disk_root = tmp_path / episode_id
            disk_root.mkdir(parents=True, exist_ok=True)
            all_events.extend(_generate_episode(episode_id, seed=i, disk_root=disk_root))

        examples = generate_dataset(all_events, horizons=HORIZONS)
        print(f"generated {len(examples)} dataset examples across {N_EPISODES} episodes")

        print("\n=== Training and evaluating both models, per horizon ===")
        results = run_full_experiment(
            examples,
            horizons=HORIZONS,
            model_names=[LOGISTIC_REGRESSION, HIST_GRADIENT_BOOSTING],
            seed=SEED,
            working_memory_capacity=CAPACITY,
            experiment_id="demo-experiment-1",
        )
        for result in results:
            _print_result(result)

        # Use the first horizon's logistic regression model to drive the
        # online learned-utility policy in a baseline comparison.
        chosen = next(
            r
            for r in results
            if r.metadata.horizon == HORIZONS[0] and r.metadata.model_name == LOGISTIC_REGRESSION
        )
        learned_policy = LearnedUtilityPolicy(model=chosen.model, horizon=HORIZONS[0])

        print(
            f"\n=== Baseline comparison (horizon={HORIZONS[0]}, " "learned=logistic_regression) ==="
        )
        workload = [MemoryKey(f"k{i}") for i in range(8)] * 4
        sim_root = tmp_path / "simulation"
        sim_root.mkdir(parents=True, exist_ok=True)
        sim_results = run_baseline_comparison(
            workload,
            capacity=CAPACITY,
            disk_root=sim_root,
            random_seed=SEED,
            learned_policy=learned_policy,
            learned_policy_name="LearnedUtilityPolicy[logistic_regression]",
        )
        header = (
            f"{'policy':38s} {'faults':>7s} {'hit_ratio':>10s} "
            f"{'belady_gap':>11s} {'mean_lat_us':>12s}"
        )
        print(header)
        for r in sim_results:
            gap = "n/a" if r.belady_gap_relative is None else f"{r.belady_gap_relative:.2f}"
            print(
                f"{r.policy_name:38s} {r.page_faults:7d} {r.hit_ratio:10.3f} "
                f"{gap:>11s} {r.mean_decision_latency_seconds * 1e6:12.2f}"
            )

        out_dir = Path("experiments/demo-experiment-1")
        write_json(
            {"model_results": [r.to_dict() for r in results]},
            out_dir / "model_results.json",
        )
        write_csv([r.to_dict() for r in sim_results], out_dir / "simulation_results.csv")
        print(f"\nWrote results to {out_dir}/")


if __name__ == "__main__":
    main()
