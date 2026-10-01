#!/usr/bin/env python3
"""Aggregate all 67 checkpoints into two clean, analysis-ready CSV tables.

No interpretation, no derived claims, no fabricated or estimated values:
every cell is read directly from a stored field in a checkpoint file (or
is a mean/std of already-computed per-episode values -- the same
aggregation `scripts/extract_paper_results.py` performs, re-run here as
an independent verification pass with a different, flatter output shape
meant for direct loading into a spreadsheet/dataframe rather than for
prose).

Writes:
  experiments/generalization-experiment-2/analysis/operational_results.csv
  experiments/generalization-experiment-2/analysis/predictive_metrics.csv
  experiments/generalization-experiment-2/analysis/verification_report.json

Usage:
    python scripts/aggregate_results_table.py
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

CHECKPOINT_DIR = Path("experiments/generalization-experiment-2/checkpoints")
OUT_DIR = Path("experiments/generalization-experiment-2/analysis")
EXPECTED_HASH = "2d4e3f4f40f07b8a2b584d33b9496addb563addd8a99d66dd1d77ee25ba079f3"

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
MODEL_NAMES = ["logistic_regression", "hist_gradient_boosting"]


def load(family: str, horizon: int, protocol: str) -> dict[str, Any]:
    """Load one checkpoint file by (family, horizon, protocol); return the full stored dict."""
    prefix = "protocol_b" if protocol == "B" else "protocol_a"
    path = CHECKPOINT_DIR / f"{prefix}_{family}_{horizon}.json"
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def flags(family: str, horizon: int) -> str:
    """Semicolon-joined structural annotations for a (family, horizon) pair."""
    parts = []
    if (family, horizon) in STRUCTURAL_NULL:
        parts.append("structural_null")
    redundant = REDUNDANT_HORIZON_PAIRS.get(family)
    if redundant and horizon in redundant:
        other = redundant[0] if horizon == redundant[1] else redundant[1]
        parts.append(f"redundant_with_H{other}")
    return ";".join(parts)


def operational_rows(protocol: str) -> list[dict[str, Any]]:
    """One row per (family-or-held-out, horizon, policy) operational result."""
    rows: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        for horizon in HORIZONS:
            data = load(family, horizon, protocol)
            r = data["result"]
            classical = r["classical_policy_faults"]
            common = {
                "protocol": protocol,
                "family_or_held_out": family,
                "horizon": horizon,
                "flags": flags(family, horizon),
            }
            for policy in ["Belady_MIN", "FIFO", "LRU", "LFU"]:
                faults = [ep[policy]["page_faults"] for ep in classical.values()]
                hits = [ep[policy]["hit_ratio"] for ep in classical.values()]
                rows.append(
                    {
                        **common,
                        "policy": policy,
                        "n_episodes": len(faults),
                        "mean_page_faults": round(statistics.mean(faults), 4),
                        "std_page_faults": round(
                            statistics.pstdev(faults) if len(faults) > 1 else 0.0, 4
                        ),
                        "mean_hit_ratio": round(statistics.mean(hits), 6),
                    }
                )
            random_keys = [
                k for k in next(iter(classical.values())) if k.startswith("Random_seed")
            ]
            if random_keys:
                per_ep_faults = [
                    statistics.mean(ep[k]["page_faults"] for k in random_keys)
                    for ep in classical.values()
                ]
                per_ep_hits = [
                    statistics.mean(ep[k]["hit_ratio"] for k in random_keys)
                    for ep in classical.values()
                ]
                rows.append(
                    {
                        **common,
                        "policy": "Random_mean_of_5_seeds",
                        "n_episodes": len(per_ep_faults),
                        "mean_page_faults": round(statistics.mean(per_ep_faults), 4),
                        "std_page_faults": round(
                            statistics.pstdev(per_ep_faults) if len(per_ep_faults) > 1 else 0.0, 4
                        ),
                        "mean_hit_ratio": round(statistics.mean(per_ep_hits), 6),
                    }
                )
            learned = r["learned_policy_faults"]
            if learned:
                faults = [ep["page_faults"] for ep in learned.values()]
                hits = [ep["hit_ratio"] for ep in learned.values()]
                rows.append(
                    {
                        **common,
                        "policy": f"Learned({r['selected_model']})",
                        "n_episodes": len(faults),
                        "mean_page_faults": round(statistics.mean(faults), 4),
                        "std_page_faults": round(
                            statistics.pstdev(faults) if len(faults) > 1 else 0.0, 4
                        ),
                        "mean_hit_ratio": round(statistics.mean(hits), 6),
                    }
                )
            else:
                rows.append(
                    {
                        **common,
                        "policy": "Learned",
                        "n_episodes": 0,
                        "mean_page_faults": "",
                        "std_page_faults": "",
                        "mean_hit_ratio": "",
                    }
                )
    return rows


def predictive_rows(protocol: str) -> list[dict[str, Any]]:
    """One row per (family-or-held-out, horizon, model, split) predictive-metric result."""
    rows: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        for horizon in HORIZONS:
            data = load(family, horizon, protocol)
            r = data["result"]
            selected = r["selected_model"]
            for model in MODEL_NAMES:
                m = r["models"][model]
                common = {
                    "protocol": protocol,
                    "family_or_held_out": family,
                    "horizon": horizon,
                    "flags": flags(family, horizon),
                    "model": model,
                    "status": m["status"],
                    "selected": model == selected,
                }
                if m["status"] != "trained":
                    rows.append(
                        {
                            **common,
                            "split": "",
                            "n_examples": "",
                            "roc_auc": "",
                            "pr_auc": "",
                            "brier_score": "",
                            "accuracy": "",
                            "f1": "",
                        }
                    )
                    continue
                for split in ["train", "val", "test"]:
                    metrics = m[f"{split}_metrics"]
                    rows.append(
                        {
                            **common,
                            "split": split,
                            "n_examples": metrics["n_examples"],
                            "roc_auc": metrics["roc_auc"],
                            "pr_auc": metrics["pr_auc"],
                            "brier_score": metrics["brier_score"],
                            "accuracy": metrics["accuracy"],
                            "f1": metrics["f1"],
                        }
                    )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write rows to path as CSV, column order = first row's key order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def verify() -> dict[str, Any]:
    """Independent verification pass: file count, hash consistency, schema sanity."""
    files = sorted(CHECKPOINT_DIR.glob("*.json"))
    hash_mismatches = []
    schema_errors = []
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            schema_errors.append(f"{f.name}: {exc!r}")
            continue
        if d.get("config_hash") != EXPECTED_HASH:
            hash_mismatches.append(f.name)
    report = {
        "total_files": len(files),
        "expected_hash": EXPECTED_HASH,
        "hash_mismatches": hash_mismatches,
        "schema_errors": schema_errors,
        "all_valid": len(files) == 67 and not hash_mismatches and not schema_errors,
    }
    out_path = OUT_DIR / "verification_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return report


def main() -> None:
    """Verify all checkpoints, then aggregate operational and predictive results to CSV."""
    report = verify()
    print(json.dumps(report, indent=2))
    if not report["all_valid"]:
        raise SystemExit("Verification failed -- refusing to aggregate. See report above.")

    op_rows = operational_rows("A") + operational_rows("B")
    write_csv(op_rows, OUT_DIR / "operational_results.csv")

    pred_rows = predictive_rows("A") + predictive_rows("B")
    write_csv(pred_rows, OUT_DIR / "predictive_metrics.csv")


if __name__ == "__main__":
    main()
