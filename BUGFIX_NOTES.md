# `elapsed_since_fault` train/serve skew — notes

## The bug

`elapsed_since_fault` (one of 13 model features) is computed from real
fault-tick history offline (`src/neuropager/dataset/features.py:154-157`),
but the runtime policy (`src/neuropager/experiment/policy.py`, before this
fix) always called `compute_features(fault_ticks=[], ...)`, forcing this
one feature to the missing-history sentinel (`-1.0`) on every single
decision, 100% of the time. Verified empirically: on one regenerated
episode, 96.3% of offline training rows had a real value for this feature,
while the runtime value was the sentinel on every row with no exceptions.

**This is not a newly-discovered defect.** `docs/model.md`'s "Scientific
concerns and known parity gaps" section already documents it precisely,
and `tests/unit/test_experiment_policy.py::test_elapsed_since_fault_is_always_the_missing_history_sentinel`
and `tests/integration/test_experiment_pipeline.py::test_online_features_match_offline_replay_for_available_features`
already prove it on real data. The gap exists because `PageReplacementPolicy`'s
hook interface (`on_insert(key)`) fires identically for a fault-driven
reinstall and a fresh `put()` — the policy structurally cannot tell which
happened, and `PageTable` tracks only a cumulative fault *count*, never
fault *ticks*.

What **was** missing: a worked-through accounting of which existing
results this affects, and a fix.

## Impact on existing artifacts

- **`predictive_metrics.csv`, `checkpoints/protocol_a_*.json`,
  `checkpoints/protocol_b_*.json`** — affected. Every trained model used
  13 features including real offline `elapsed_since_fault`; test-set
  predictive metrics (ROC-AUC, PR-AUC, F1, Brier, accuracy) were computed
  on offline-reconstructed features that include this real value, which
  the deployed policy never has access to. These numbers measure a model
  evaluated on inputs it will never actually see in production.
- **`operational_results.csv`'s Learned-policy rows** — a more subtle
  case. These came from literally replaying the policy inside a real
  `MemoryManager`, so the *measurement* itself is honest: it genuinely
  reflects what the deployed policy does. The concern is that the
  *policy being measured* was trained assuming it would sometimes see a
  real `elapsed_since_fault`, when it never does at serving time — so the
  deployed policy is likely weaker than a policy trained with this
  mismatch removed. We do not call these operational numbers
  "confounded"; we call the training that produced that policy
  suboptimal.
- **Classical baselines (`classical_*.json`, and the Belady/FIFO/LRU/LFU/Random
  rows everywhere)** — confirmed unaffected. None of
  `fifo.py`/`lru.py`/`lfu.py`/`belady_min.py`/`random_policy.py` import
  anything from `features`/`model`/`predict` — they never use a trained
  model at all.

## The fix

Added two new feature-set variants (`NO_ESF12`, `NO_RECENCY7`) alongside
the original, unchanged default (`FULL13`). We chose to **drop the
feature** rather than compute it causally at runtime, because computing
it causally would require changing `PageReplacementPolicy`'s hook
interface itself (to distinguish a fault-driven install from a fresh
one) and `PageTable`'s storage (ticks, not just counts) — a cross-cutting
change touching every policy class, versus dropping a column, which
touches nothing in the dataset/feature pipeline at all.

### What changed

- `scripts/generalization_experiment_2.py`:
  - `FEATURE_SETS` — `FULL13` (all 13, unchanged order — a no-op),
    `NO_ESF12` (drops `elapsed_since_fault`), `NO_RECENCY7` (drops
    `elapsed_since_fault` plus `recency`, `recent_burst`, `long_burst`,
    `recent_interval`, `normalized_recency` — every feature encoding time
    since the page's most recent access(es), per the "this is just
    learned LRU" ablation).
  - `select_feature_columns()` — the one place a feature set actually
    takes effect: it slices `SplitArrays.x` to the active feature set's
    columns, in order, right before `train_and_get_model()`/`eval_model()`
    ever see them. `FamilyCache`/`build_family_cache()` (dataset
    generation itself) are **never told which feature set is active** and
    are completely unchanged — identical workloads, episodes, features,
    labels, splits, and baselines regardless of feature set.
  - `build_run_config()`, `build_horizon_result()`,
    `evaluate_learned_policy_on_episodes()`, `run_protocols_memory_bounded()`,
    `run_experiment()` — all gained a `feature_names`/`feature_set`
    parameter, threaded through to the one place that matters
    (`select_feature_columns()` and `LearnedUtilityPolicy`), defaulting to
    `FULL13`/`FEATURE_NAMES` everywhere so **omitting the new CLI flags
    reproduces the original script's behavior exactly** (verified by
    `tests/integration/test_feature_set_full13_regression.py`).
  - `main()` — new `--feature-set {FULL13,NO_ESF12,NO_RECENCY7}` (default
    `FULL13`) and `--experiment-root` (override) CLI flags. Default
    output directories: `experiments/generalization-experiment-2/` for
    FULL13 (unchanged), `experiments/v2_no_esf12/` for NO_ESF12,
    `experiments/v2_no_recency7/` for NO_RECENCY7. No existing results
    directory is ever written to under a different feature set.
- `src/neuropager/experiment/policy.py`:
  - `LearnedUtilityPolicy.__init__` gained `feature_names: tuple[str, ...] = FEATURE_NAMES`.
    `select_victim()` now builds its model-input row from
    `self.feature_names` instead of the module-level `FEATURE_NAMES`
    constant, so a model trained on a reduced feature set is served the
    same reduced, correctly-ordered row at decision time. Default
    argument preserves exact original behavior for any caller that
    doesn't pass it (including existing tests).

### What did NOT change

Workload generators, seeds, episode length, working-memory capacity,
labels, splits, classical baselines, horizons, evaluation/metrics code,
or `FamilyCache`/dataset generation. The config hash now includes
`feature_order`, which already existed in `build_run_config()` before
this fix — passing `FULL13` (the original `FEATURE_NAMES`, in original
order) produces the identical `feature_order` value as before, so
nothing about the default run's fingerprint changes in substance; passing
a reduced feature set correctly produces a different hash, so a
mismatched checkpoint can never be silently reused across feature sets.

## Which paper sections need to change

- **Results (predictive metrics subsections)**: must note that FULL13
  predictive metrics (ROC-AUC/PR-AUC/etc.) reflect training/serving on 13
  features including one with a documented train/serve mismatch, and
  should present the NO_ESF12 numbers as the metric that actually reflects
  deployed-model quality once available.
- **Discussion / Limitations**: the existing framing already correctly
  separated "operational results are measured via the real runtime path"
  from "predictive metrics are offline-only" — this fix gives that
  separation a concrete resolution (NO_ESF12) rather than leaving it as an
  open caveat. The no-recency ablation (NO_RECENCY7), once run, directly
  answers the "is this just learned LRU" reviewer objection this session
  flagged.
- **Methodology / Experimental Setup**: should describe the three feature
  sets and reference this file.

## What this fix does NOT claim

It does not claim to make `elapsed_since_fault` computable online — that
would require changing `PageReplacementPolicy`'s hook interface and
`PageTable`'s storage, which we did not do (see "The fix" above for why).
It only removes the mismatched feature (and, for the ablation, the
recency-family features) from training so train and serve agree on every
remaining column.
