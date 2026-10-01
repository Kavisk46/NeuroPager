# Benchmark Workload Generators

This document describes `neuropager.workloads`: a set of six deterministic,
seeded access-pattern generators used to drive `MemoryManager` for policy
and Memory Utility Model evaluation. It exists because Experiment 1 (see
[research.md](research.md) and the pilot results referenced there) used
only uniform-random access over a tiny key space — too narrow a benchmark
to support a policy-level research claim, and with an episode length too
short for the longer reuse horizons (H=200, H=500) to be distinguishable
from H=100.

> **Status:** Generator, configuration, and statistics only. No large-scale
> (2,000+ episode) experiment has been run against these workloads yet —
> see `scripts/sanity_workloads.py` for the small, single-episode-per-kind
> sanity check this milestone produced instead.

This module has no dependency on, and makes no change to,
`neuropager.trace`, `neuropager.dataset`, or the five replacement policies
— it only produces plain `list[MemoryKey]` sequences that any of those
existing components can already consume unchanged.

## Common conventions

Every workload config carries `length` (number of ticks) and `seed`
(deterministic RNG seed); the same config always produces the same
sequence. Keys are drawn from a canonical universe `k0, k1, ..., k(K-1)`
for a given `key_space` `K`. For real horizon experiments, use
`length >= 2000` so H=25/50/100/200/500 are not all truncation-dominated
(see the Experiment 1 postmortem in `research.md` for why this matters).

## A. Uniform random (`UniformRandomConfig`)

$$\text{key}_t \sim \text{Uniform}\{k_0, \dots, k_{K-1}\}, \quad \text{i.i.d. for } t = 1, \dots, T$$

The baseline, memoryless workload — every key equally likely at every
tick, independent of history. Maximizes entropy (`log2(K)` bits) and
minimizes any generator-induced locality; used in Experiment 1's pilot run.

## B. Temporal locality (`TemporalLocalityConfig`)

An LRU-stack-distance / independent-reference model. A recency stack
(most-recently-used key at rank 0) is maintained. At each tick, a rank
$d \in \{0, \dots, K-1\}$ is drawn with

$$P(d = i) \propto (1 - \ell)^i, \qquad i = 0, \dots, K-1$$

where $\ell \in (0, 1)$ is the `locality` parameter, then the key at that
rank is accessed and moved to rank 0. Larger $\ell$ concentrates access
more strongly on recently-used keys, producing lower stack distances —
this is exactly the pattern LRU is designed to exploit.

## C. Bursty (`BurstyConfig`)

A run of the same key continues with probability
$1 - 1/\bar{L}$ at every tick, where $\bar{L}$ is `mean_burst_length`, giving
geometrically-distributed burst lengths:

$$P(L = l) = (1 - q)^{l-1} q, \qquad q = 1/\bar{L}, \qquad \mathbb{E}[L] = \bar{L}$$

When a burst ends, the next key is drawn uniformly at random, independent
of history. Produces long same-key runs (high `burst_run_lengths`)
interleaved with switches — distinct from B in that within a burst there
is *zero* variation (not just a recency bias), and switches are memoryless
rather than stack-distance-biased.

## D. Long-range reuse (`LongRangeReuseConfig`)

One random permutation ("cycle") of all `key_space` keys is drawn **once**
from `seed`, then tiled (repeated) to fill `length`. Because the identical
cycle repeats, every key sits at the same offset in every pass, so the
stack distance between any two consecutive occurrences of a key is always
**exactly** `key_space - 1` — deterministically large reuse, in sharp
contrast to A–C. `key_space` should be set well above the working-memory
capacity under test, or the "long-range" property is moot.

## E. Phase-changing (`PhaseChangingConfig`)

The episode is split into consecutive phases of `phase_length` ticks.
Phase $i$ draws uniformly from a **disjoint** key block
$\{k_{i \cdot p}, \dots, k_{(i+1) \cdot p - 1}\}$ where $p$ = `keys_per_phase`
— no key is shared between phases. This models a discrete working-set
shift (e.g. a program moving from one code region to another). Number of
phases is derived from `length // phase_length` (the last phase is
truncated to fit).

## F. Non-stationary (`NonStationaryConfig`)

A fixed Zipf-like weight profile $w_i = 1/(i+1)$ over ranks
$i = 0, \dots, K-1$ is applied to a key ordering that rotates by
$\lfloor t \cdot r \rfloor \bmod K$ positions at tick $t$, where $r$ =
`drift_rate` (positions per tick):

$$\text{key}_t \sim \text{Categorical}\big(w; \text{rotate}(k_0,\dots,k_{K-1},\, \lfloor t r \rfloor)\big)$$

Unlike E's discrete jumps, popularity here drifts **continuously** — no
single tick marks a regime change, but the identity of the "hot" key
changes gradually over the episode. `drift_rate = 0` degenerates to a
static (non-drifting) Zipf distribution.

## Diagnostics (`neuropager.workloads.statistics`)

Pure functions over a generated `list[MemoryKey]`, with no dependency on
the memory system:

- `unique_page_count` — distinct keys touched.
- `access_entropy` — Shannon entropy (bits) of the empirical key
  distribution; maximal for A, typically lower for the others.
- `stack_distances` / `mean_stack_distance` — per-access and mean LRU
  stack distance (distinct keys since the same key's previous
  occurrence); `None` for a key's first (cold) occurrence.
- `burst_run_lengths` — lengths of maximal consecutive-identical-key runs;
  the direct signature of burstiness.

`stack_distances` is O(n · average reuse distance) — adequate for the
small sanity checks and test fixtures this milestone is scoped to, not
intended for the eventual large-scale experiment (see module docstring).

## What this milestone does not claim

No workload here is asserted to be "realistic" for any particular
real-world LLM agent access pattern — that would require evidence this
milestone doesn't produce. These are six workloads with *known,
mathematically-defined* statistical properties, verified against those
properties by `scripts/sanity_workloads.py` and the test suite. Whether
any of them resembles a real deployment's access pattern is a separate,
open question.
