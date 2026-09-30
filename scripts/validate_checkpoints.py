#!/usr/bin/env python3
"""Read-only validator for the generalization-experiment-2 checkpoint directory.

Never opens any checkpoint for writing. Checks every *.json file's
config_hash against the known-good production hash, reports which
Protocol A/B units are complete, and exits 1 if anything fails.

config.json and the 6 classical_<family>.json files are not Protocol
A/B "results" (they have no protocol/horizon fields), so they are still
hash-checked -- config.json in particular is the very first checkpoint
generalization_experiment_2.py itself reads, and a mismatch there aborts
the whole run before touching anything else, so skipping its check here
would leave the single most consequential file unverified -- but they
are reported in their own category rather than folded into the
protocol_a/protocol_b completion counts.

Usage:
    python scripts/validate_checkpoints.py
    python scripts/validate_checkpoints.py --checkpoint-dir <path>
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

EXPECTED_HASH = "2d4e3f4f40f07b8a2b584d33b9496addb563addd8a99d66dd1d77ee25ba079f3"
DEFAULT_CHECKPOINT_DIR = "experiments/generalization-experiment-2/checkpoints"


def classify(data: dict[str, Any]) -> str:
    """Return which bucket a checkpoint belongs to, using .get() throughout.

    Never raises: a checkpoint missing every classifying field (or not
    even a dict) falls into "unknown" rather than crashing the validator.
    """
    if not isinstance(data, dict):
        return "unknown"
    protocol = data.get("protocol")
    if protocol == "A":
        return "protocol_a"
    if protocol == "B":
        return "protocol_b"
    if "config" in data and "config_hash" in data:
        return "config"
    if "family" in data and "protocol" not in data:
        return "classical"
    return "unknown"


def main() -> None:
    """Validate every checkpoint file's schema and config_hash; exit 1 on any failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    files = sorted(glob.glob(str(checkpoint_dir / "*.json")))

    print(f"checkpoint dir: {checkpoint_dir}")
    print(f"found {len(files)} *.json file(s)\n")

    n_passed = 0
    n_failed = 0
    buckets: dict[str, list[str]] = {
        "config": [],
        "classical": [],
        "protocol_a": [],
        "protocol_b": [],
        "unknown": [],
    }

    for f in files:
        name = Path(f).name
        try:
            text = Path(f).read_text(encoding="utf-8")
            data = json.loads(text)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: any parse/read
            # failure must be reported as a FAIL row, never crash the validator.
            print(f"FAIL  {name}: could not read/parse ({exc!r})")
            n_failed += 1
            continue

        if not isinstance(data, dict):
            print(f"FAIL  {name}: top-level JSON is not an object")
            n_failed += 1
            continue

        computed_hash = data.get("config_hash")
        if computed_hash != EXPECTED_HASH:
            print(f"FAIL  {name}: config_hash={computed_hash!r} != expected")
            n_failed += 1
            continue

        bucket = classify(data)
        buckets[bucket].append(name)

        family = data.get("family")
        held_out = data.get("held_out_family")
        horizon = data.get("horizon")
        protocol = data.get("protocol")
        detail = ""
        if protocol == "A":
            detail = f"  (family={family}, horizon={horizon})"
        elif protocol == "B":
            detail = f"  (held_out={held_out}, horizon={horizon})"
        elif bucket == "classical":
            detail = f"  (family={family})"

        print(f"OK    {name}{detail}")
        n_passed += 1

    n_excluded = len(buckets["config"]) + len(buckets["classical"])

    print("\n--- summary ---")
    print(f"total *.json files:        {len(files)}")
    print(f"passed hash check:         {n_passed}")
    print(f"failed:                    {n_failed}")
    print(f"excluded from unit counts: {n_excluded}  (config.json + classical_*.json)")
    print(f"  config:      {buckets['config']}")
    print(f"  classical:   {sorted(buckets['classical'])}")
    print(f"  protocol_a:  {sorted(buckets['protocol_a'])}")
    print(f"  protocol_b:  {sorted(buckets['protocol_b'])}")
    if buckets["unknown"]:
        print(f"  unknown:     {sorted(buckets['unknown'])}")

    if n_failed > 0:
        print("\nRESULT: FAIL")
        sys.exit(1)

    print("\nRESULT: PASS")


if __name__ == "__main__":
    main()
