#!/usr/bin/env python3
"""Compute the full descriptive/statistical analysis requested against the
committed analysis CSVs (operational_results.csv, predictive_metrics.csv)
plus the 67 checkpoints directly for paired-comparison p-values that are
not present in the CSVs (never fabricated -- read verbatim from
paired_comparisons_vs_learned, exactly as scripts/generalization_experiment_2.py
computed them).

Prints everything needed to write the analysis report; does not write the
report itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ANALYSIS_DIR = Path("experiments/generalization-experiment-2/analysis")
CHECKPOINT_DIR = Path("experiments/generalization-experiment-2/checkpoints")

op = pd.read_csv(ANALYSIS_DIR / "operational_results.csv")
pred = pd.read_csv(ANALYSIS_DIR / "predictive_metrics.csv")

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)

FAMILIES = [
    "A_uniform_random",
    "B_temporal_locality",
    "C_bursty",
    "D_long_range_reuse",
    "E_phase_changing",
    "F_non_stationary",
]
HORIZONS = [25, 50, 100, 200, 500]
POLICIES = ["Belady_MIN", "FIFO", "LRU", "LFU", "Random_mean_of_5_seeds"]


def section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


# ---------------------------------------------------------------------
section("1. OVERALL OPERATIONAL PERFORMANCE BY POLICY (both protocols pooled, n=60 units/policy)")
# ---------------------------------------------------------------------
by_policy = op.groupby("policy", sort=False).agg(
    n_units=("mean_page_faults", "count"),
    mean_of_mean_faults=("mean_page_faults", "mean"),
    std_of_mean_faults=("mean_page_faults", "std"),
    mean_hit_ratio=("mean_hit_ratio", "mean"),
)
print(by_policy.round(4))

print("\nRank-based (lower fault count = better rank, computed within each of the 60 "
      "protocol/family/horizon units, mean rank across units -- comparable across "
      "families despite very different absolute fault scales):")
ranked = op.copy()
ranked["rank"] = ranked.groupby(["protocol", "family_or_held_out", "horizon"])[
    "mean_page_faults"
].rank(method="average")
print(ranked.groupby("policy", sort=False)["rank"].mean().round(3).sort_values())


# ---------------------------------------------------------------------
section("2. PERFORMANCE BY WORKLOAD FAMILY (both protocols pooled, per policy)")
# ---------------------------------------------------------------------
for family in FAMILIES:
    sub = op[op["family_or_held_out"] == family]
    print(f"\n--- {family} ---")
    print(
        sub.groupby("policy", sort=False)[["mean_page_faults", "mean_hit_ratio"]]
        .mean()
        .round(4)
    )


# ---------------------------------------------------------------------
section("3. PERFORMANCE BY HORIZON (both protocols pooled, per policy)")
# ---------------------------------------------------------------------
for h in HORIZONS:
    sub = op[op["horizon"] == h]
    print(f"\n--- H={h} ---")
    print(
        sub.groupby("policy", sort=False)[["mean_page_faults", "mean_hit_ratio"]]
        .mean()
        .round(4)
    )


# ---------------------------------------------------------------------
section("4. PROTOCOL A ONLY -- by policy")
# ---------------------------------------------------------------------
a = op[op["protocol"] == "A"]
print(a.groupby("policy", sort=False)[["mean_page_faults", "mean_hit_ratio"]].mean().round(4))


# ---------------------------------------------------------------------
section("5. PROTOCOL B ONLY -- by policy")
# ---------------------------------------------------------------------
b = op[op["protocol"] == "B"]
print(b.groupby("policy", sort=False)[["mean_page_faults", "mean_hit_ratio"]].mean().round(4))


# ---------------------------------------------------------------------
section("5b. PROTOCOL A vs B: Learned policy's mean_page_faults side by side, per family/horizon")
# ---------------------------------------------------------------------
learned = op[op["policy"].str.startswith("Learned")].copy()
pivot = learned.pivot_table(
    index=["family_or_held_out", "horizon", "flags"],
    columns="protocol",
    values="mean_page_faults",
    aggfunc="first",
)
pivot["A_minus_B"] = pivot.get("A") - pivot.get("B") if "B" in pivot else None
print(pivot)


# ---------------------------------------------------------------------
section("5c. PAIRED Learned-vs-LRU comparison, read verbatim from checkpoints (p-values, CIs)")
# ---------------------------------------------------------------------
rows = []
for family in FAMILIES:
    for h in HORIZONS:
        for protocol, prefix in [("A", "protocol_a"), ("B", "protocol_b")]:
            path = CHECKPOINT_DIR / f"{prefix}_{family}_{h}.json"
            d = json.loads(path.read_text(encoding="utf-8"))
            r = d["result"]
            cmp = r["paired_comparisons_vs_learned"].get("Learned_vs_LRU")
            if cmp is None or cmp.get("insufficient_data"):
                rows.append(
                    {
                        "family": family,
                        "H": h,
                        "protocol": protocol,
                        "mean_diff": None,
                        "ci_low": None,
                        "ci_high": None,
                        "p_ttest": None,
                        "p_wilcoxon": None,
                        "n_episodes": cmp["n_episodes"] if cmp else None,
                    }
                )
            else:
                rows.append(
                    {
                        "family": family,
                        "H": h,
                        "protocol": protocol,
                        "mean_diff": cmp["mean_diff"],
                        "ci_low": cmp["ci_95_low"],
                        "ci_high": cmp["ci_95_high"],
                        "p_ttest": cmp["paired_ttest_pvalue"],
                        "p_wilcoxon": cmp["wilcoxon_pvalue"],
                        "n_episodes": cmp["n_episodes"],
                    }
                )
cmp_df = pd.DataFrame(rows)
pd.set_option("display.max_colwidth", None)
print(cmp_df.to_string(index=False))


# ---------------------------------------------------------------------
section("6. LEARNED-MODEL PREDICTIVE METRICS -- summary by split (both protocols, both models, trained only)")
# ---------------------------------------------------------------------
trained = pred[pred["status"] == "trained"]
print(
    trained.groupby(["model", "split"])[["roc_auc", "pr_auc", "brier_score", "accuracy", "f1"]]
    .agg(["mean", "min", "max"])
    .round(4)
)


# ---------------------------------------------------------------------
section("7. LR vs HGB -- train/test ROC-AUC gap (both protocols, per model)")
# ---------------------------------------------------------------------
pivot2 = trained.pivot_table(
    index=["protocol", "family_or_held_out", "horizon", "model"],
    columns="split",
    values="roc_auc",
)
pivot2["train_minus_test"] = pivot2["train"] - pivot2["test"]
print(
    pivot2.reset_index()
    .groupby(["protocol", "model"])["train_minus_test"]
    .agg(["mean", "std", "count"])
    .round(4)
)

print("\nSelection counts (which model was selected per unit, both protocols):")
sel = pred.drop_duplicates(subset=["protocol", "family_or_held_out", "horizon", "model"])
sel_counts = sel[sel["selected"] == True].groupby(["protocol", "model"]).size()  # noqa: E712
print(sel_counts)
null_counts = (
    pred[pred["status"] != "trained"]
    .drop_duplicates(subset=["protocol", "family_or_held_out", "horizon"])
    .groupby("protocol")
    .size()
)
print("\nStructural-null units (no model trained) per protocol:")
print(null_counts)


# ---------------------------------------------------------------------
section("8. STRUCTURAL-NULL D CASES (H in {25,50,100}) -- full operational rows")
# ---------------------------------------------------------------------
null_rows = op[op["flags"].astype(str).str.contains("structural_null", na=False)]
print(null_rows.to_string(index=False))


# ---------------------------------------------------------------------
section("9. FLAGGED (unreliable-interpretation) ROWS -- structural_null or redundant horizons")
# ---------------------------------------------------------------------
flagged = op[op["flags"].astype(str) != ""]["family_or_held_out horizon flags".split()].drop_duplicates()
print(flagged.to_string(index=False))


# ---------------------------------------------------------------------
section("11. CASES WHERE LEARNED DOES NOT BEAT LRU (mean_page_faults Learned >= LRU)")
# ---------------------------------------------------------------------
lru = op[op["policy"] == "LRU"][["protocol", "family_or_held_out", "horizon", "mean_page_faults"]].rename(
    columns={"mean_page_faults": "lru_faults"}
)
lrn = op[op["policy"].str.startswith("Learned")][
    ["protocol", "family_or_held_out", "horizon", "policy", "mean_page_faults", "n_episodes"]
].rename(columns={"mean_page_faults": "learned_faults"})
merged = lrn.merge(lru, on=["protocol", "family_or_held_out", "horizon"])
merged = merged[merged["n_episodes"] > 0]
worse_or_equal = merged[merged["learned_faults"] >= merged["lru_faults"]].sort_values(
    ["protocol", "family_or_held_out", "horizon"]
)
print(worse_or_equal.to_string(index=False))
print(f"\n{len(worse_or_equal)} / {len(merged)} non-null units where Learned >= LRU on mean faults")


# ---------------------------------------------------------------------
section("12. BELADY GAP -- LRU_faults - Belady_faults and Learned_faults - Belady_faults, per unit")
# ---------------------------------------------------------------------
belady = op[op["policy"] == "Belady_MIN"][
    ["protocol", "family_or_held_out", "horizon", "mean_page_faults"]
].rename(columns={"mean_page_faults": "belady_faults"})
gap_df = merged.merge(belady, on=["protocol", "family_or_held_out", "horizon"])
gap_df["learned_minus_belady"] = gap_df["learned_faults"] - gap_df["belady_faults"]
gap_df["lru_minus_belady"] = gap_df["lru_faults"] - gap_df["belady_faults"]
gap_df["pct_gap_closed_by_learned"] = 100 * (
    1 - gap_df["learned_minus_belady"] / gap_df["lru_minus_belady"]
)
print(
    gap_df[
        [
            "protocol",
            "family_or_held_out",
            "horizon",
            "belady_faults",
            "lru_faults",
            "learned_faults",
            "lru_minus_belady",
            "learned_minus_belady",
            "pct_gap_closed_by_learned",
        ]
    ]
    .round(2)
    .to_string(index=False)
)
