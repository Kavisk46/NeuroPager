#!/usr/bin/env python3
"""Generate the 5 workload-characterization figures/tables listed in Appendix D.

Covers exactly the items in paper/neuropager-workload-generalization-draft.md
that do NOT depend on the still-running generalization experiment.
"""

from __future__ import annotations

import json
from collections import Counter, deque
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neuropager.utils.types import MemoryKey
from neuropager.workloads.config import (
    BurstyConfig,
    LongRangeReuseConfig,
    NonStationaryConfig,
    PhaseChangingConfig,
    TemporalLocalityConfig,
    UniformRandomConfig,
    WorkloadConfig,
)
from neuropager.workloads.generator import generate_workload

STATS_PATH = Path("experiments/preflight-analysis-1/statistics.json")
FIG_DIR = Path("paper/figures")


def sliding_window_working_set(workload: list[MemoryKey], window: int) -> list[int]:
    """Distinct-key count in the trailing window accesses, once the window has filled.

    Copied verbatim from scripts/preflight_analysis.py (a script-local
    helper there, not part of neuropager.workloads.statistics) so this
    script regenerates the identical series without modifying that file.
    """
    sizes: list[int] = []
    counts: Counter[MemoryKey] = Counter()
    dq: deque[MemoryKey] = deque()
    for key in workload:
        dq.append(key)
        counts[key] += 1
        if len(dq) > window:
            old = dq.popleft()
            counts[old] -= 1
            if counts[old] == 0:
                del counts[old]
        if len(dq) == window:
            sizes.append(len(counts))
    return sizes


LENGTH = 2000
WORKING_SET_WINDOWS = [16, 50, 200]
FAMILY_SEED0_CONFIGS: dict[str, WorkloadConfig] = {
    "A_uniform_random": UniformRandomConfig(length=LENGTH, key_space=64, seed=0),
    "B_temporal_locality": TemporalLocalityConfig(
        length=LENGTH, key_space=64, seed=0, locality=0.3
    ),
    "C_bursty": BurstyConfig(length=LENGTH, key_space=64, seed=0, mean_burst_length=15.0),
    "D_long_range_reuse": LongRangeReuseConfig(length=LENGTH, key_space=200, seed=0),
    "E_phase_changing": PhaseChangingConfig(
        length=LENGTH, seed=0, phase_length=250, keys_per_phase=32
    ),
    "F_non_stationary": NonStationaryConfig(length=LENGTH, key_space=64, seed=0, drift_rate=0.01),
}
STRUCTURAL_NULL = {("D_long_range_reuse", h) for h in (25, 50, 100)}
REDUNDANT_HORIZON_PAIRS = {"D_long_range_reuse": (200, 500), "E_phase_changing": (200, 500)}

HORIZONS = [25, 50, 100, 200, 500]
FAMILY_ORDER = [
    "A_uniform_random",
    "B_temporal_locality",
    "C_bursty",
    "D_long_range_reuse",
    "E_phase_changing",
    "F_non_stationary",
]
HISTOGRAM_BIN_ORDER = [
    "[0,10)",
    "[10,25)",
    "[25,50)",
    "[50,100)",
    "[100,200)",
    "[200,500)",
    "[500,1000)",
    "[1000,2000)",
    ">=2000",
]


def load_stats() -> dict[str, Any]:
    """Load experiments/preflight-analysis-1/statistics.json, raising if absent."""
    if not STATS_PATH.exists():
        raise FileNotFoundError(
            f"{STATS_PATH} not found -- run scripts/preflight_analysis.py first. "
            "Refusing to fabricate summary statistics."
        )
    data: dict[str, Any] = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    return data


def write_markdown_table(
    rows: list[list[str]], header: list[str], path: Path, caption: str
) -> None:
    """Write a captioned Markdown table to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        caption,
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def figure_1_summary_statistics(stats: dict[str, Any]) -> None:
    """Item 1: per-family entropy, stack distance, burst CV, unique-page ratio, drift."""
    header = [
        "Family",
        "Entropy (bits)",
        "Mean stack distance",
        "Burst CV",
        "Unique-page ratio",
        "Early/late TV drift",
        "n episodes",
    ]
    rows = []
    for family in FAMILY_ORDER:
        f = stats[family]
        rows.append(
            [
                family,
                f"{f['entropy_bits_mean']:.3f} +/- {f['entropy_bits_std']:.3f}",
                f"{f['mean_stack_distance_mean']:.2f} +/- {f['mean_stack_distance_std']:.2f}",
                f"{f['burst_cv_mean']:.3f} +/- {f['burst_cv_std']:.3f}",
                f"{f['unique_page_ratio_mean']:.3f} +/- {f['unique_page_ratio_std']:.3f}",
                f"{f['early_late_tv_distance_mean']:.3f} +/- {f['early_late_tv_distance_std']:.3f}",
                str(f["n_episodes"]),
            ]
        )
    caption = (
        "Table 1 -- Per-family summary statistics. Source: "
        f"{STATS_PATH} (produced by scripts/preflight_analysis.py). "
        "Each value is the mean +/- population std across the 10 generated "
        "episodes for that family (length=2000 ticks each, seeds 0-9). "
        "No smoothing or extrapolation applied."
    )
    write_markdown_table(
        rows, header, FIG_DIR / "figure_1_per_family_summary_statistics.md", caption
    )


def figure_2_reuse_distance_distribution(stats: dict[str, Any]) -> None:
    """Item 2: per-family reuse-distance mean/median/p90/p95/p99."""
    header = ["Family", "Mean", "Median", "p90", "p95", "p99"]
    rows = []
    for family in FAMILY_ORDER:
        f = stats[family]
        rows.append(
            [
                family,
                f"{f['reuse_mean_mean']:.1f}",
                f"{f['reuse_median_mean']:.1f}",
                f"{f['reuse_p90_mean']:.1f}",
                f"{f['reuse_p95_mean']:.1f}",
                f"{f['reuse_p99_mean']:.1f}",
            ]
        )
    caption = (
        "Table 2 -- Per-family reuse-distance (tick-gap) distribution. "
        f"Source: {STATS_PATH}. Each column is the across-episode mean of "
        "that statistic computed independently per episode."
    )
    write_markdown_table(rows, header, FIG_DIR / "figure_2_reuse_distance_distribution.md", caption)


def figure_3_reuse_distance_histograms(stats: dict[str, Any]) -> None:
    """Item 3: six small-multiple reuse-distance histograms, seed=0 episode per family."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, family in zip(axes.flat, FAMILY_ORDER, strict=True):
        histogram = stats[family]["reuse_histogram_sample"]
        counts = [histogram[b] for b in HISTOGRAM_BIN_ORDER]
        ax.bar(
            range(len(HISTOGRAM_BIN_ORDER)),
            counts,
            color="#3b6ea5",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_xticks(range(len(HISTOGRAM_BIN_ORDER)))
        ax.set_xticklabels(HISTOGRAM_BIN_ORDER, rotation=45, ha="right", fontsize=7)
        ax.set_title(family, fontsize=10)
        ax.set_ylabel("count", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle(
        "Reuse-distance (tick-gap) histograms, single representative episode "
        "(seed=0) per family -- not aggregated across episodes",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = FIG_DIR / "figure_3_reuse_distance_histograms.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def figure_4_horizon_coverage(stats: dict[str, Any]) -> None:
    """Item 4: per-family, per-horizon label-positive fraction, with interpretation flags."""
    header = ["Family"] + [f"H={h}" for h in HORIZONS]
    rows = []
    for family in FAMILY_ORDER:
        f = stats[family]
        row = [family]
        for h in HORIZONS:
            value = f[f"label_positive_frac_H{h}_mean"]
            cell = f"{value:.3f}"
            if (family, h) in STRUCTURAL_NULL:
                cell += " [structural-null]"
            redundant = REDUNDANT_HORIZON_PAIRS.get(family)
            if redundant and h in redundant:
                other = redundant[0] if h == redundant[1] else redundant[1]
                cell += f" [redundant w/ H={other}]"
            row.append(cell)
        rows.append(row)
    caption = (
        "Table 4 -- Horizon coverage: fraction of accesses with a positive "
        f"reuse label at each horizon. Source: {STATS_PATH}. Values are "
        "the across-episode mean of label_positive_frac_H{h}, which "
        "counts an access as positive iff its key recurs within (t, t+H] "
        "before episode end (episode-end truncation counts as negative), "
        "exactly matching neuropager.dataset.labels. [structural-null] "
        "and [redundant] annotations are carried over verbatim from "
        "scripts/generalization_experiment.py pre-registered "
        "interpretation rules, not recomputed here."
    )
    write_markdown_table(rows, header, FIG_DIR / "figure_4_horizon_coverage.md", caption)


def figure_5_working_set_over_time() -> None:
    """Item 5: sliding-window working-set size over time, regenerated from seed=0 workloads."""
    fig, axes = plt.subplots(len(WORKING_SET_WINDOWS), 1, figsize=(11, 10), sharex=True)
    colors = {
        "A_uniform_random": "#3b6ea5",
        "B_temporal_locality": "#a53b3b",
        "C_bursty": "#3ba55c",
        "D_long_range_reuse": "#a5883b",
        "E_phase_changing": "#7a3ba5",
        "F_non_stationary": "#3ba5a0",
    }
    for window, ax in zip(WORKING_SET_WINDOWS, axes, strict=True):
        for family in FAMILY_ORDER:
            workload = generate_workload(FAMILY_SEED0_CONFIGS[family])
            series = sliding_window_working_set(workload, window)
            x = list(range(window - 1, window - 1 + len(series)))
            ax.plot(x, series, label=family, color=colors[family], linewidth=1.1)
        ax.set_ylabel(f"distinct keys\n(window={window})", fontsize=9)
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel("tick", fontsize=9)
    axes[0].legend(fontsize=7, ncol=3, loc="upper right")
    fig.suptitle(
        "Sliding-window working-set size over episode time, seed=0 per family "
        "(regenerated; not stored in statistics.json)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = FIG_DIR / "figure_5_working_set_size_over_time.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    """Generate all 5 workload-characterization figures/tables."""
    stats = load_stats()
    figure_1_summary_statistics(stats)
    figure_2_reuse_distance_distribution(stats)
    figure_3_reuse_distance_histograms(stats)
    figure_4_horizon_coverage(stats)
    figure_5_working_set_over_time()


if __name__ == "__main__":
    main()
