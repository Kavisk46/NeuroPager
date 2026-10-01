#!/usr/bin/env python3
"""Scale up the first Memory Utility Model experiment to check robustness.

Same pipeline as scripts/demo_experiment.py -- no changes to
neuropager.dataset or neuropager.experiment -- run at a larger scale:

- ~5x the episodes (80 vs. 24) and ~3x the episode length (200 vs. 60
  ticks), so the full horizon set (25/50/100/200/500) is meaningfully
  covered instead of being dominated by end-of-episode truncation.
- All 5 horizons from neuropager.dataset.labels.DEFAULT_HORIZONS, not
  just two.
- The baseline-vs-learned-policy comparison is repeated across 5
  independent simulation workloads (different seeds, disjoint from the
  episode-generation seeds) so "did the learned policy actually win" can
  be answered with a mean +/- spread instead of a single N=1 run.

This still is not a rigorous, publication-grade evaluation (see
docs/model.md and the final report) -- it is one step up from the tiny
smoke-test demo, meant to check whether that demo's favorable result was
a small-sample fluke or a consistent pattern.

Usage:
    python scripts/scale_experiment.py
"""

from __future__ import annotations

import random
import statistics
import tempfile
import time
from pathlib import Path

from neuropager.core.memory_manager import MemoryManager
from neuropager.core.page_fault import PageFaultHandler
from neuropager.core.page_table import PageTable
from neuropager.core.working_memory import WorkingMemory
from neuropager.dataset.generator import generate_dataset
from neuropager.dataset.labels import DEFAULT_HORIZONS
from neuropager.experiment.models import HIST_GRADIENT_BOOSTING, LOGISTIC_REGRESSION
from neuropager.experiment.policy import LearnedUtilityPolicy
from neuropager.experiment.reproducibility import write_csv, write_json
from neuropager.experiment.runner import ModelHorizonResult, run_full_experiment
from neuropager.experiment.simulation import SimulationResult, run_baseline_comparison
from neuropager.memory.disk_store import DiskPageStore
from neuropager.policies.lru import LRUPolicy
from neuropager.trace.events import TraceEvent
from neuropager.trace.logger import TraceLogger
from neuropager.utils.types import MemoryKey

N_EPISODES = 80
EPISODE_LENGTH = 200
CAPACITY = 4
KEY_SPACE = 8
TRAIN_SEED = 0
SIM_SEEDS = [1000, 1001, 1002, 1003, 1004]
SIM_WORKLOAD_LENGTH = 200


def _generate_episode(episode_id: str, seed: int, disk_root: Path) -> list[TraceEvent]:
    rng = random.Random(seed)
    keys = [MemoryKey(f"k{i}") for i in range(KEY_SPACE)]
    workload = [rng.choice(keys) for _ in range(EPISODE_LENGTH)]

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
            f"  {split_name:5s} n={metrics.n_examples:5d} "
            f"pos={metrics.n_positive:5d} "
            f"acc={metrics.accuracy:.3f} f1={metrics.f1:.3f} "
            f"roc_auc={auc} pr_auc={pr_auc} brier={metrics.brier_score:.3f}"
        )


def _run_simulation_replicates(
    learned_model: object,
    horizon: int,
    model_label: str,
    tmp_path: Path,
) -> dict[str, list[SimulationResult]]:
    by_policy: dict[str, list[SimulationResult]] = {}
    for seed in SIM_SEEDS:
        rng = random.Random(seed)
        keys = [MemoryKey(f"k{i}") for i in range(KEY_SPACE)]
        workload = [rng.choice(keys) for _ in range(SIM_WORKLOAD_LENGTH)]
        sim_root = tmp_path / f"sim-{seed}"
        sim_root.mkdir(parents=True, exist_ok=True)

        learned_policy = LearnedUtilityPolicy(model=learned_model, horizon=horizon)  # type: ignore[arg-type]
        results = run_baseline_comparison(
            workload,
            capacity=CAPACITY,
            disk_root=sim_root,
            random_seed=seed,
            learned_policy=learned_policy,
            learned_policy_name=model_label,
        )
        for result in results:
            by_policy.setdefault(result.policy_name, []).append(result)
    return by_policy


def main() -> None:
    """Generate a larger dataset, run the full horizon/model sweep, and repeat.

    Repeats the baseline comparison across several simulation seeds.
    """
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        print(f"=== Generating {N_EPISODES} episodes x {EPISODE_LENGTH} ticks ===")
        all_events: list[TraceEvent] = []
        for i in range(N_EPISODES):
            episode_id = f"scale-ep-{i}"
            disk_root = tmp_path / episode_id
            disk_root.mkdir(parents=True, exist_ok=True)
            all_events.extend(_generate_episode(episode_id, seed=i, disk_root=disk_root))
        print(f"  generated in {time.perf_counter() - start:.1f}s")

        examples = generate_dataset(all_events, horizons=list(DEFAULT_HORIZONS))
        print(f"generated {len(examples)} dataset examples across {N_EPISODES} episodes")

        n_horizons = len(DEFAULT_HORIZONS)
        print(f"\n=== Training and evaluating both models, all {n_horizons} horizons ===")
        results = run_full_experiment(
            examples,
            horizons=list(DEFAULT_HORIZONS),
            model_names=[LOGISTIC_REGRESSION, HIST_GRADIENT_BOOSTING],
            seed=TRAIN_SEED,
            working_memory_capacity=CAPACITY,
            experiment_id="scale-experiment-1",
        )
        for result in results:
            _print_result(result)

        write_json(
            {"model_results": [r.to_dict() for r in results]},
            Path("experiments/scale-experiment-1/model_results.json"),
        )

        print(f"\n=== Baseline comparison across {len(SIM_SEEDS)} simulation seeds ===")
        all_sim_rows = []
        for horizon in DEFAULT_HORIZONS:
            for model_name in (LOGISTIC_REGRESSION, HIST_GRADIENT_BOOSTING):
                chosen = next(
                    r
                    for r in results
                    if r.metadata.horizon == horizon and r.metadata.model_name == model_name
                )
                label = f"LearnedUtilityPolicy[{model_name}]"
                by_policy = _run_simulation_replicates(chosen.model, horizon, label, tmp_path)

                print(f"\n--- horizon={horizon} model={model_name} ---")
                print(f"{'policy':38s} {'mean_faults':>12s} {'std_faults':>11s} {'mean_gap':>10s}")
                for policy_name, runs in by_policy.items():
                    faults = [r.page_faults for r in runs]
                    gaps = [
                        r.belady_gap_relative for r in runs if r.belady_gap_relative is not None
                    ]
                    mean_faults = statistics.mean(faults)
                    std_faults = statistics.pstdev(faults) if len(faults) > 1 else 0.0
                    mean_gap = statistics.mean(gaps) if gaps else float("nan")
                    print(
                        f"{policy_name:38s} {mean_faults:12.2f} {std_faults:11.2f} {mean_gap:10.2f}"
                    )
                    for run in runs:
                        row = run.to_dict()
                        row["horizon"] = horizon
                        row["backing_model"] = model_name
                        all_sim_rows.append(row)

        write_csv(all_sim_rows, Path("experiments/scale-experiment-1/simulation_results.csv"))
        print(f"\nTotal wall time: {time.perf_counter() - start:.1f}s")
        print("Wrote results to experiments/scale-experiment-1/")


if __name__ == "__main__":
    main()
