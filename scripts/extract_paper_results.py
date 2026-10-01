#!/usr/bin/env python3
"""Extract every Appendix-D results table/figure (items 6-14) from the real,
completed generalization-experiment-2 checkpoints.

Reads directly from experiments/generalization-experiment-2/checkpoints/
(the 67 per-unit JSON checkpoints, never the large aggregate
_checkpoint_*.json/protocol_*_results.json files, so this works identically
whether run against the full run or a partial one) and writes:

  paper/figures/table_6_protocol_a_predictive_metrics.md
  paper/figures/table_9_protocol_a_operational_results.md
  paper/figures/table_10_protocol_b_operational_results.md
  paper/figures/table_11_protocol_a_vs_b_generalization.md
  paper/figures/table_14_rq6_model_comparison.md
  paper/figures/figure_12_policy_comparison_bars.png
  paper/figures/figure_13_decision_latency.png
  paper/figures/headline_summary.json  (compact numbers for prose, not fabricated)

Every number here is read directly from a checkpoint file's own stored
fields (paired_comparisons_vs_learned, train/val/test_metrics,
classical_policy_faults, learned_policy_faults) -- nothing is
recomputed, approximated, or invented. Structural-null and
redundant-horizon conditions (per generalization_experiment_2.py's own
STRUCTURAL_NULL_CONDITIONS/REDUNDANT_HORIZON_PAIRS) are annotated, never
silently dropped or silently treated as ordinary results.

Usage:
    python scripts/extract_paper_results.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHECKPOINT_DIR = Path("experiments/generalization-experiment-2/checkpoints")
FIG_DIR = Path("paper/figures")

HORIZONS = [25, 50, 100, 200, 500]
FAMILY_ORDER = [
    "A_uniform_random",
    "B_temporal_locality",
    "C_bursty",
    "D_long_range_reuse",
    "E_phase_changing",
    "F_non_stationary",
]
STRUCTURAL_NULL = {("D_long_range_reuse", h) for h in (25, 50, 100)}
REDUNDANT_HORIZON_PAIRS = {"D_long_range_reuse": (200, 500), "E_phase_changing": (200, 500)}
POLICY_ORDER = ["Belady_MIN", "FIFO", "LRU", "LFU", "Random", "Learned"]
MODEL_NAMES = ["logistic_regression", "hist_gradient_boosting"]


def flags_suffix(family: str, horizon: int) -> str:
    """Return the same bracketed annotation style used elsewhere in this paper's tables."""
    parts = []
    if (family, horizon) in STRUCTURAL_NULL:
        parts.append("[structural-null]")
    redundant = REDUNDANT_HORIZON_PAIRS.get(family)
    if redundant and horizon in redundant:
        other = redundant[0] if horizon == redundant[1] else redundant[1]
        parts.append(f"[redundant w/ H={other}]")
    return " " + " ".join(parts) if parts else ""


def load(family: str, horizon: int, protocol: str, held_out: bool = False) -> dict[str, Any]:
    """Load one checkpoint's 'result' dict by (family, horizon, protocol)."""
    prefix = "protocol_b" if protocol == "B" else "protocol_a"
    path = CHECKPOINT_DIR / f"{prefix}_{family}_{horizon}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["result"]  # type: ignore[no-any-return]


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


def fmt(x: float | None, digits: int = 3) -> str:
    """Format a metric value, or 'n/a' for null (structural-null) metrics."""
    return "n/a" if x is None else f"{x:.{digits}f}"


def table_6_predictive_metrics() -> None:
    """Item 6: ROC-AUC/PR-AUC/Brier per family, horizon, model, split."""
    header = ["Family", "H", "Model", "Split", "ROC-AUC", "PR-AUC", "Brier", "F1", "n"]
    rows = []
    for family in FAMILY_ORDER:
        for horizon in HORIZONS:
            r = load(family, horizon, "A")
            for model in MODEL_NAMES:
                m = r["models"][model]
                if m["status"] != "trained":
                    continue
                for split in ["train", "val", "test"]:
                    metrics = m[f"{split}_metrics"]
                    rows.append(
                        [
                            family + flags_suffix(family, horizon),
                            str(horizon),
                            model,
                            split,
                            fmt(metrics["roc_auc"]),
                            fmt(metrics["pr_auc"]),
                            fmt(metrics["brier_score"]),
                            fmt(metrics["f1"]),
                            str(metrics["n_examples"]),
                        ]
                    )
    caption = (
        "Table 6 -- Protocol A (within-distribution) predictive metrics, per "
        "family, horizon, model, and split. Source: "
        "experiments/generalization-experiment-2/checkpoints/protocol_a_*.json "
        "(the `models.<name>.{train,val,test}_metrics` fields, read verbatim). "
        "n/a = ROC-AUC/PR-AUC undefined on a structurally one-class split "
        "(see flags)."
    )
    write_markdown_table(rows, header, FIG_DIR / "table_6_protocol_a_predictive_metrics.md", caption)


def paired_episode_stats(result: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Per-policy mean/std page faults and hit ratio, computed from stored per-episode data.

    classical_policy_faults/learned_policy_faults already store every
    episode's exact {page_faults, hit_ratio, ...} -- this only takes the
    mean/std across episodes, it does not recompute faults or hit ratio
    themselves.

    On a structural-null unit (e.g. D_long_range_reuse @ H in {25,50,100}),
    both models' status is "structural_null_single_class_training_data" --
    scikit-learn cannot fit on single-class training data, so no Learned
    policy was ever trained or replayed, and learned_policy_faults is an
    empty dict for that unit (confirmed by direct inspection, not assumed).
    This is omitted from the returned dict entirely (the caller must check
    for its absence), rather than fabricating a placeholder value.
    """
    out: dict[str, dict[str, float]] = {}
    classical = result["classical_policy_faults"]
    for policy in ["Belady_MIN", "FIFO", "LRU", "LFU"]:
        faults = [ep[policy]["page_faults"] for ep in classical.values()]
        hits = [ep[policy]["hit_ratio"] for ep in classical.values()]
        out[policy] = {
            "mean_faults": statistics.mean(faults),
            "std_faults": statistics.pstdev(faults) if len(faults) > 1 else 0.0,
            "mean_hit_ratio": statistics.mean(hits),
            "n": len(faults),
        }
    # Random: average across its 5 seeded sub-policies' per-episode results first
    random_keys = [k for k in next(iter(classical.values())) if k.startswith("Random_seed")]
    if random_keys:
        faults = [
            statistics.mean(ep[k]["page_faults"] for k in random_keys) for ep in classical.values()
        ]
        hits = [
            statistics.mean(ep[k]["hit_ratio"] for k in random_keys) for ep in classical.values()
        ]
        out["Random"] = {
            "mean_faults": statistics.mean(faults),
            "std_faults": statistics.pstdev(faults) if len(faults) > 1 else 0.0,
            "mean_hit_ratio": statistics.mean(hits),
            "n": len(faults),
        }
    learned = result["learned_policy_faults"]
    if learned:
        faults = [ep["page_faults"] for ep in learned.values()]
        hits = [ep["hit_ratio"] for ep in learned.values()]
        out["Learned"] = {
            "mean_faults": statistics.mean(faults),
            "std_faults": statistics.pstdev(faults) if len(faults) > 1 else 0.0,
            "mean_hit_ratio": statistics.mean(hits),
            "n": len(faults),
        }
    return out


def belady_gap_stats(result: dict[str, Any]) -> dict[str, float]:
    """Mean Belady gap per policy, from the stored belady_gaps field (already computed).

    Key naming is not perfectly uniform in the stored data: the Random
    baseline's key is "Random_mean" (not "Random"), "Belady_MIN" has no
    entry (a policy's gap to itself is trivially zero and not stored), and
    "Learned" has no entry at all on a structural-null unit (no model was
    trained -- see paired_episode_stats()). get_gap() below is the single
    place that resolves these naming differences for every caller.
    """
    return {
        policy: statistics.mean(values.values())
        for policy, values in result["belady_gaps"].items()
    }


def get_gap(gaps: dict[str, float], policy: str) -> float | None:
    """Look up one policy's mean Belady gap, resolving the Random/Random_mean naming difference."""
    if policy == "Random":
        return gaps.get("Random_mean", gaps.get("Random"))
    return gaps.get(policy)


def table_9_protocol_a_operational() -> None:
    """Item 9: Protocol A operational results -- mean faults, hit ratio, Belady gap, per family/policy."""
    header = ["Family", "H", "Policy", "Mean faults", "Std", "Hit ratio", "Belady gap"]
    rows = []
    for family in FAMILY_ORDER:
        for horizon in HORIZONS:
            r = load(family, horizon, "A")
            stats = paired_episode_stats(r)
            gaps = belady_gap_stats(r)
            for policy in POLICY_ORDER:
                if policy not in stats:
                    if policy == "Learned":
                        rows.append(
                            [
                                family + flags_suffix(family, horizon),
                                str(horizon),
                                policy,
                                "n/a",
                                "n/a",
                                "n/a",
                                "n/a (model not trained -- one-class training data)",
                            ]
                        )
                    continue
                s = stats[policy]
                gap = get_gap(gaps, policy)
                rows.append(
                    [
                        family + flags_suffix(family, horizon),
                        str(horizon),
                        policy,
                        f"{s['mean_faults']:.1f}",
                        f"{s['std_faults']:.1f}",
                        f"{s['mean_hit_ratio']:.3f}",
                        fmt(gap, 2),
                    ]
                )
    caption = (
        "Table 9 -- Protocol A (within-distribution) operational results: "
        "mean page faults (+/- population std across test episodes), mean "
        "hit ratio, and mean Belady gap, per family/horizon/policy. "
        "Source: classical_policy_faults, learned_policy_faults, and "
        "belady_gaps fields of each protocol_a_*.json checkpoint; means "
        "computed here, no underlying value recomputed. D_long_range_reuse "
        "@ H in {25,50,100} has no Learned row: both models' status is "
        "structural_null_single_class_training_data (scikit-learn cannot "
        "fit on single-class data), so no model was trained or replayed."
    )
    write_markdown_table(rows, header, FIG_DIR / "table_9_protocol_a_operational_results.md", caption)


def table_10_protocol_b_operational() -> None:
    """Item 10: Protocol B operational results, one block per held-out family."""
    header = ["Held-out family", "H", "Policy", "Mean faults", "Std", "Hit ratio", "Belady gap"]
    rows = []
    for family in FAMILY_ORDER:
        for horizon in HORIZONS:
            r = load(family, horizon, "B")
            stats = paired_episode_stats(r)
            gaps = belady_gap_stats(r)
            for policy in POLICY_ORDER:
                if policy not in stats:
                    if policy == "Learned":
                        rows.append(
                            [
                                family + flags_suffix(family, horizon),
                                str(horizon),
                                policy,
                                "n/a",
                                "n/a",
                                "n/a",
                                "n/a (model not trained -- one-class training data)",
                            ]
                        )
                    continue
                s = stats[policy]
                gap = get_gap(gaps, policy)
                rows.append(
                    [
                        family + flags_suffix(family, horizon),
                        str(horizon),
                        policy,
                        f"{s['mean_faults']:.1f}",
                        f"{s['std_faults']:.1f}",
                        f"{s['mean_hit_ratio']:.3f}",
                        fmt(gap, 2),
                    ]
                )
    caption = (
        "Table 10 -- Protocol B (leave-one-family-out) operational results, "
        "same structure as Table 9, one block per held-out family. All "
        "test episodes are 100% of the held-out family's episodes, never "
        "seen in training or validation. Source: protocol_b_*.json "
        "checkpoints."
    )
    write_markdown_table(rows, header, FIG_DIR / "table_10_protocol_b_operational_results.md", caption)


def table_11_generalization_comparison() -> None:
    """Item 11: paired learned-vs-LRU advantage, Protocol A vs Protocol B, per family/horizon."""
    header = [
        "Family",
        "H",
        "A: Learned-LRU mean diff",
        "A: 95% CI",
        "A: p (paired t)",
        "B: Learned-LRU mean diff",
        "B: 95% CI",
        "B: p (paired t)",
    ]
    def fmt_cmp(c: dict[str, Any] | None) -> tuple[str, str, str]:
        """Format one paired-comparison dict's (mean_diff, CI, p-value), or 'n/a' if unavailable."""
        if c is None or c.get("insufficient_data"):
            return ("n/a", "n/a", "n/a")
        return (
            f"{c['mean_diff']:.2f}",
            f"[{c['ci_95_low']:.2f}, {c['ci_95_high']:.2f}]",
            f"{c['paired_ttest_pvalue']:.3g}",
        )

    rows = []
    for family in FAMILY_ORDER:
        for horizon in HORIZONS:
            ra = load(family, horizon, "A")
            rb = load(family, horizon, "B")
            ca = ra["paired_comparisons_vs_learned"].get("Learned_vs_LRU")
            cb = rb["paired_comparisons_vs_learned"].get("Learned_vs_LRU")
            a_diff, a_ci, a_p = fmt_cmp(ca)
            b_diff, b_ci, b_p = fmt_cmp(cb)
            rows.append(
                [
                    family + flags_suffix(family, horizon),
                    str(horizon),
                    a_diff,
                    a_ci,
                    a_p,
                    b_diff,
                    b_ci,
                    b_p,
                ]
            )
    caption = (
        "Table 11 -- The central generalization comparison (Section 18): "
        "learned policy's paired advantage over LRU in page faults "
        "(negative mean_diff = learned had FEWER faults, i.e. better), "
        "Protocol A (within-distribution) vs Protocol B (leave-one-"
        "family-out), per family and horizon. Source: "
        "paired_comparisons_vs_learned['Learned_vs_LRU'] in each "
        "checkpoint, read verbatim (mean_diff, ci_95_low/high, "
        "paired_ttest_pvalue) -- these are the exact bootstrap/paired-"
        "test outputs scripts/generalization_experiment_2.py already "
        "computed, not recomputed here."
    )
    write_markdown_table(
        rows, header, FIG_DIR / "table_11_protocol_a_vs_b_generalization.md", caption
    )


def table_14_rq6_model_comparison() -> None:
    """Item 14: RQ6 -- logistic regression vs HGB, train/test gap, per horizon."""
    header = ["Family", "H", "Model", "Train ROC-AUC", "Test ROC-AUC", "Gap", "Selected?"]
    rows = []
    for family in FAMILY_ORDER:
        for horizon in HORIZONS:
            r = load(family, horizon, "A")
            selected = r["selected_model"]
            for model in MODEL_NAMES:
                m = r["models"][model]
                if m["status"] != "trained":
                    continue
                train_auc = m["train_metrics"]["roc_auc"]
                test_auc = m["test_metrics"]["roc_auc"]
                gap = (
                    None
                    if train_auc is None or test_auc is None
                    else train_auc - test_auc
                )
                rows.append(
                    [
                        family + flags_suffix(family, horizon),
                        str(horizon),
                        model,
                        fmt(train_auc),
                        fmt(test_auc),
                        fmt(gap),
                        "yes" if model == selected else "no",
                    ]
                )
    caption = (
        "Table 14 -- RQ6: logistic regression vs. histogram gradient "
        "boosting, train-vs-test ROC-AUC and the gap between them (a "
        "larger gap indicates more overfitting), per family/horizon. "
        "'Selected?' marks which model Section 13.3's post-hoc "
        "validation-ROC-AUC selection chose as the Learned Utility Policy "
        "for that (family, horizon) -- the Protocol A/B operational "
        "tables (9, 10) always report the SELECTED model's results, not "
        "necessarily HGB. Source: models.<name>.{train,test}_metrics.roc_auc "
        "in each protocol_a_*.json checkpoint."
    )
    write_markdown_table(rows, header, FIG_DIR / "table_14_rq6_model_comparison.md", caption)


def figure_12_policy_comparison_bars() -> None:
    """Item 12: page-faults bar chart, all policies, Protocol A at H=25, one subplot per family."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, family in zip(axes.flat, FAMILY_ORDER, strict=True):
        r = load(family, 25, "A")
        stats = paired_episode_stats(r)
        policies = [p for p in POLICY_ORDER if p in stats]
        means = [stats[p]["mean_faults"] for p in policies]
        stds = [stats[p]["std_faults"] for p in policies]
        colors = ["#555555" if p != "Learned" else "#a53b3b" for p in policies]
        ax.bar(policies, means, yerr=stds, color=colors, edgecolor="black", linewidth=0.5, capsize=3)
        ax.set_title(family, fontsize=10)
        ax.set_ylabel("mean page faults", fontsize=8)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle(
        "Protocol A (within-distribution), H=25: mean page faults per policy, "
        "error bars = population std across test episodes (Random: std across "
        "the 5 seeds' per-episode means)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = FIG_DIR / "figure_12_policy_comparison_bars.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def figure_13_decision_latency() -> None:
    """Item 13: decision latency comparison, log scale, Protocol A at H=25."""
    fig, ax = plt.subplots(figsize=(9, 5))
    all_policies = ["Belady_MIN", "FIFO", "LFU", "Learned"]
    data_by_policy: dict[str, list[float]] = {p: [] for p in all_policies}
    for family in FAMILY_ORDER:
        r = load(family, 25, "A")
        classical = r["classical_policy_faults"]
        for policy in ["Belady_MIN", "FIFO", "LFU"]:
            data_by_policy[policy].extend(
                ep[policy]["decision_latency_mean_s"]
                for ep in classical.values()
                if ep[policy]["decision_latency_mean_s"] is not None
            )
        learned = r["learned_policy_faults"]
        data_by_policy["Learned"].extend(
            ep["decision_latency_mean_s"]
            for ep in learned.values()
            if ep["decision_latency_mean_s"] is not None
        )
    # An episode with zero capacity-triggered eviction decisions has no
    # latency samples to average, so its decision_latency_mean_s is None
    # (a handful of such episodes exist, e.g. short bursts that never
    # filled working memory) -- filtered above rather than passed to
    # np.mean as a mixed float/None array.
    ax.boxplot(
        [data_by_policy[p] for p in all_policies],
        tick_labels=all_policies,
        showfliers=False,
    )
    ax.set_yscale("log")
    ax.set_ylabel("mean decision latency per eviction (s, log scale)", fontsize=9)
    ax.set_title(
        "Decision latency across all families, Protocol A, H=25 "
        "(LRU/Random excluded from timing by construction)",
        fontsize=10,
    )
    fig.tight_layout()
    out_path = FIG_DIR / "figure_13_decision_latency.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def headline_summary() -> None:
    """Compact JSON of real headline numbers for the Results/Discussion prose -- nothing invented."""
    summary: dict[str, Any] = {"protocol_a_vs_b_learned_vs_lru": {}, "rq6_mean_gap": {}}
    for family in FAMILY_ORDER:
        for horizon in HORIZONS:
            ra = load(family, horizon, "A")
            rb = load(family, horizon, "B")
            ca = ra["paired_comparisons_vs_learned"].get("Learned_vs_LRU")
            cb = rb["paired_comparisons_vs_learned"].get("Learned_vs_LRU")
            if ca and cb and not ca.get("insufficient_data") and not cb.get("insufficient_data"):
                summary["protocol_a_vs_b_learned_vs_lru"][f"{family}_H{horizon}"] = {
                    "protocol_a_mean_diff": ca["mean_diff"],
                    "protocol_a_pvalue": ca["paired_ttest_pvalue"],
                    "protocol_b_mean_diff": cb["mean_diff"],
                    "protocol_b_pvalue": cb["paired_ttest_pvalue"],
                    "flags": [
                        f
                        for f in (
                            "structural_null" if (family, horizon) in STRUCTURAL_NULL else None,
                        )
                        if f
                    ],
                }
    gaps_by_model: dict[str, list[float]] = {m: [] for m in MODEL_NAMES}
    selected_counts: dict[str, int] = {m: 0 for m in MODEL_NAMES}
    selected_counts["none_structural_null"] = 0
    for family in FAMILY_ORDER:
        for horizon in HORIZONS:
            r = load(family, horizon, "A")
            selected = r["selected_model"]
            selected_counts[selected if selected is not None else "none_structural_null"] += 1
            for model in MODEL_NAMES:
                m = r["models"][model]
                if m["status"] != "trained":
                    continue
                tr = m["train_metrics"]["roc_auc"]
                te = m["test_metrics"]["roc_auc"]
                if tr is not None and te is not None:
                    gaps_by_model[model].append(tr - te)
    for model in MODEL_NAMES:
        vals = gaps_by_model[model]
        summary["rq6_mean_gap"][model] = {
            "mean_train_minus_test_roc_auc": statistics.mean(vals) if vals else None,
            "n_comparable_units": len(vals),
        }
    summary["selected_model_counts"] = selected_counts
    out_path = FIG_DIR / "headline_summary.json"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


def main() -> None:
    """Generate all Appendix-D results tables/figures (items 6, 9, 10, 11, 12, 13, 14)."""
    table_6_predictive_metrics()
    table_9_protocol_a_operational()
    table_10_protocol_b_operational()
    table_11_generalization_comparison()
    table_14_rq6_model_comparison()
    figure_12_policy_comparison_bars()
    figure_13_decision_latency()
    headline_summary()


if __name__ == "__main__":
    main()
