# The First Memory Utility Model Experiment

This document describes `neuropager.experiment`: the smallest rigorous
pipeline for testing whether a supervised model can predict page reuse
well enough to make better eviction decisions than FIFO/LRU/LFU/Random.
It builds directly on `neuropager.dataset` (see [dataset.md](dataset.md))
and does not modify it.

> **Status:** First experimental instrument, not a result. Nothing in this
> document or its accompanying code claims the research hypothesis is
> supported. See [Scientific concerns](#scientific-concerns-and-known-parity-gaps)
> below before trusting any number this pipeline produces.

## What an experiment run represents

One experiment run trains **one model** (logistic regression or histogram
gradient boosting) on **one horizon**'s worth of dataset examples, using an
**episode-level** train/validation/test split, and reports metrics on all
three splits. `run_full_experiment` repeats this for every requested
(horizon, model) combination, sharing one split per horizon across both
models so they are compared on identical data.

Separately, `run_baseline_comparison` takes a *fitted* model, wraps it as
an online `LearnedUtilityPolicy`, and runs it — alongside FIFO, LRU, LFU,
Random, and Belady's MIN — over an actual workload through a real
`MemoryManager`, reporting page faults, hit ratio, gap to Belady's optimal,
and per-decision latency.

## Models

- **Logistic regression** (`neuropager.experiment.models.make_logistic_regression`):
  `StandardScaler -> LogisticRegression`. Scaling is fit on the training
  split only.
- **Histogram gradient boosting** (`make_hist_gradient_boosting`):
  scikit-learn's `HistGradientBoostingClassifier`, no scaling (tree splits
  are scale-invariant). This is the one nonlinear model requested; no
  additional dependency (XGBoost/LightGBM) was added.

Both receive the **same 13 approved features**, in the same raw form,
including the `-1.0` missing-history sentinel — deliberately *not*
converted to `NaN` for one model and not the other, so the two models are
compared on identical inputs (see `neuropager.dataset.features` for the
sentinel convention).

## Feature list

Exactly, and only, the 13 features from `neuropager.dataset.features.FEATURE_NAMES`:
`recency`, `frequency`, `avg_inter_access_interval`,
`inter_access_interval_variance`, `fault_count`, `fault_ratio`, `page_age`,
`recent_burst`, `long_burst`, `recent_interval`, `normalized_recency`,
`elapsed_since_fault`, `max_historical_interval`. `build_feature_matrix`
only ever reads `example.features` — never `episode_id`, `page_id`,
`policy`, or `label` — which is what structurally guarantees no identity
leakage into the model (see `test_build_feature_matrix_ignores_page_id`).
The exact feature order is recorded in every run's
`ExperimentMetadata.feature_order`.

## Split strategy

`neuropager.experiment.split.split_episodes` partitions **unique episode
IDs** — never individual examples — deterministically: sort, shuffle with
a seeded `random.Random`, cut at 70/15/15 by default. The same episodes
and seed always produce the same split, and no episode ever contributes
examples to more than one partition (`test_split_has_no_overlap_between_partitions`).
The split is recorded in full in `ExperimentMetadata.split`.

## Horizon handling

Each horizon (`25, 50, 100, 200, 500` by default, matching
`neuropager.dataset.labels.DEFAULT_HORIZONS`) gets its **own** model, its
**own** split (computed from that horizon's own episode set), and its
**own** metrics — never averaged or combined across horizons.
`run_full_experiment` makes no assumption about which horizon is "best";
it simply runs and reports all of them.

## Metrics

Because reuse labels are imbalanced (most resident pages are *not* reused
soon), every run reports accuracy, precision, recall, F1, ROC-AUC, PR-AUC,
Brier score, and a small binned calibration curve — never accuracy alone.
ROC-AUC/PR-AUC/calibration degrade gracefully to `None`/empty (not a
crash) when a split has only one class present, which is realistic for a
small first experiment's validation/test splits.

## Policy simulation design

`LearnedUtilityPolicy` (`neuropager.experiment.policy`) implements
`PageReplacementPolicy` with the same `select_victim(self, page_table)`
signature as every other policy — the same structural guarantee proven for
FIFO/LRU/LFU/Random/Belady's-MIN in earlier milestones applies here
unchanged (`test_select_victim_signature_matches_the_shared_policy_interface`).
For every resident candidate, it computes features from its own
self-tracked access history (via the existing `on_access`/`on_insert`
hooks — the same mechanism `BeladyMinPolicy` already uses to track its
reference-string cursor), asks the model for `P(reused)`, and evicts the
candidate with the **lowest** predicted probability.

Belady's MIN is used **only** to compute the reference fault count for the
gap metric in `run_baseline_comparison`; its future reference string is
never constructed from, or passed to, `LearnedUtilityPolicy` or any other
online policy (`test_constructor_has_no_future_or_belady_parameter`).

## Scientific concerns and known parity gaps

The online policy cannot achieve *perfect* parity with the offline
training pipeline, because `PageReplacementPolicy`'s hook interface (last
touched to add `BeladyMinPolicy`, and left unchanged here) does not carry
enough information for a policy to reconstruct everything
`neuropager.dataset.features` can see from a complete trace file. These
gaps are real, specific, and testable — not hand-waved:

1. **`elapsed_since_fault` is always the missing-history sentinel online**,
   even when the offline value would be a real number. `on_insert(key)`
   fires identically for a fault-driven reinstall and a fresh `put()`, so
   the policy cannot tell *which* happened, and `PageTableEntry` tracks
   only a cumulative fault *count*, not fault *ticks*. `fault_count` and
   `fault_ratio` **are** exact online (read straight from
   `PageTable.get_entry(key).page_fault_count`, which is clock-independent
   arithmetic, not a tick delta). `test_online_features_match_offline_replay_for_available_features`
   demonstrates this gap concretely on real data, not just in theory.
2. **Two different logical clocks exist, and the policy must not confuse
   them.** `PageTable`'s own internal clock (`PageTable._clock`) advances a
   different number of times per operation depending on whether it was a
   hit, a fault, or an eviction — it cannot be rescaled to match
   `TraceLogger.tick`, which the offline model was trained on. The policy
   therefore tracks its **own** tick, advanced once per
   `on_access`/`on_insert` call, matching `TraceLogger.begin_step()`'s
   cadence exactly under normal `get`/`put` workloads.
3. **`select_victim` runs one step "behind" its own tick counter**, because
   it is called from inside `PageFaultHandler.install()` *before*
   `on_insert` fires for the reference currently being processed. The
   policy corrects for this by using `self._tick + 1` as the decision
   tick — exact, not a heuristic, given the single-operation-in-flight
   invariant of this system — but this is a subtle, easy-to-regress detail
   if `PageFaultHandler`'s call order ever changes.
4. **Explicit `MemoryManager.evict()` calls silently desynchronize the two
   clocks.** They advance `TraceLogger.tick` (via `begin_step()`) but not
   the learned policy's internal tick (no `on_access`/`on_insert` fires).
   Workloads that mix explicit evictions with normal `get`/`put` traffic
   will see the online policy's notion of "now" drift behind the trace's.
   This milestone's workloads do not exercise explicit eviction, so the
   drift is untested in practice — flagged here rather than silently
   ignored.

None of these gaps are hidden from the model by rounding or approximation
— they are honest sentinels or self-consistent alternate values, and the
whole point of `test_online_features_match_offline_replay_for_available_features`
is to make the *size and shape* of the remaining gap visible, not to paper
over it.

## Not yet done

Per the milestone scope: no hyperparameter sweeps, no large-scale runs, no
reinforcement learning, no neural networks, no embeddings/graph features,
no online learning. See `docs/roadmap.md`.
