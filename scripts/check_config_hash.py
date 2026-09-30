#!/usr/bin/env python3
"""Recompute the production config_hash using the script's own functions.

Imports build_run_config() and config_hash() directly from
generalization_experiment_2.py (no duplicated logic) and calls them with
the exact production parameters main() uses for the full run, then
compares the result against the known-good hash recorded in every one of
the 16 existing checkpoints. This is a portability check: if this prints
MATCH on a new machine, every locked scientific parameter that feeds the
hash (workload params, episode count, horizons, seeds, model names,
feature order) round-trips identically there, independent of hostname,
OS, paths, or library versions -- see the accompanying migration audit
for why none of those can leak into the hash by construction.

Usage:
    PYTHONPATH=src python -u scripts/check_config_hash.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

EXPECTED_HASH = "2d4e3f4f40f07b8a2b584d33b9496addb563addd8a99d66dd1d77ee25ba079f3"

_SCRIPT_PATH = Path(__file__).resolve().parent / "generalization_experiment_2.py"
_spec = importlib.util.spec_from_file_location("generalization_experiment_2", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
ge2: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["generalization_experiment_2"] = ge2
_spec.loader.exec_module(ge2)


def main() -> None:
    """Recompute config_hash for the production run and compare to EXPECTED_HASH."""
    run_cfg = ge2.build_run_config(
        families=ge2.FAMILIES,
        horizons=ge2.HORIZONS,
        n_episodes_per_family=ge2.FULL_SPEC_EPISODES_PER_FAMILY,
        random_seeds=ge2.RANDOM_SEEDS,
    )
    computed = ge2.config_hash(run_cfg)

    print(f"expected:  {EXPECTED_HASH}")
    print(f"computed:  {computed}")

    if computed != EXPECTED_HASH:
        print("MISMATCH -- this environment computes a different config_hash for the "
              "same locked parameters. Do NOT resume from the existing checkpoints "
              "until this is understood; see build_run_config()'s inputs.")
        sys.exit(1)

    print("MATCH -- config_hash is portable to this environment.")


if __name__ == "__main__":
    main()
