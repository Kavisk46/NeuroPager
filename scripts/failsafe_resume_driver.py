#!/usr/bin/env python3
"""Driver for the fail-safe interrupt/resume proof: runs a tiny, real experiment.

Not part of the production experiment -- this exists only so
scripts/generalization_experiment_2.py can be launched as a real subprocess
(and killed mid-run, then relaunched) to prove interruption-safety with
actual process semantics, rather than only in-process unit tests.

Usage:
    PYTHONPATH=src python -u scripts/failsafe_resume_driver.py <experiment_root>
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parent / "generalization_experiment_2.py"
_spec = importlib.util.spec_from_file_location("generalization_experiment_2", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
ge2 = importlib.util.module_from_spec(_spec)
sys.modules["generalization_experiment_2"] = ge2
_spec.loader.exec_module(ge2)


def main() -> None:
    """Run the tiny fail-safe scope against the experiment_root given on argv."""
    experiment_root = Path(sys.argv[1])
    ge2.run_experiment(
        families=["A_uniform_random", "B_temporal_locality"],
        horizons=[50],
        n_episodes_per_family=4,
        random_seeds=[0],
        experiment_root=experiment_root,
        experiment_id="failsafe-resume-check",
    )


if __name__ == "__main__":
    main()
