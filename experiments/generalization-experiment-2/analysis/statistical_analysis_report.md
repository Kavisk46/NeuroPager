# Statistical/Descriptive Analysis — generalization-experiment-2

**Source data**: `operational_results.csv` (360 rows), `predictive_metrics.csv` (348
rows), `verification_report.json` (67/67 checkpoints valid, config hash
`2d4e3f4f40f07b8a2b584d33b9496addb563addd8a99d66dd1d77ee25ba079f3`), plus direct
reads of `paired_comparisons_vs_learned` from the 67 checkpoints for p-values/CIs
not present in the CSVs (never recomputed — read verbatim).

**Scope note**: this is a descriptive/statistical inventory of what the data show.
It is explicitly *not* the paper's Results/Discussion section — no causal claims,
no narrative framing, no selection of which findings are "important" for a story.
Every number below is traced to its source; every pattern states the exact rows
that support it.

---

## 0. One correction to the existing paper draft, surfaced by this analysis

The paper draft (`paper/neuropager-workload-generalization-draft.md`) currently
states that `D_long_range_reuse` at H∈{25,50,100} has **no Learned policy in
either protocol**. This is **only true for Protocol A**. Verified directly
against `protocol_b_D_long_range_reuse_25.json`:

```
lr status:  trained
hgb status: trained
selected_model: hist_gradient_boosting
learned_policy_faults: 200 episodes (non-empty)
```

**Why the asymmetry is real, not a bug**: in Protocol A, the model trains on
D's *own* episodes, whose labels are one-class at H≤100 (training fails:
`structural_null_single_class_training_data`). In Protocol B, held-out=D's
model trains on the pooled *other five* families (A,B,C,E,F), whose labels are
not one-class — training succeeds. The model is then replayed on D's 200 test
episodes, where **it produces results numerically identical to every classical
baseline** (1800 faults, 0.100 hit ratio, exactly — Row set in §8). ROC-AUC on
the test split is still undefined (D's test labels are one-class), which is
why the `structural_null` flag still applies, but for a different reason
(test-label degeneracy, not training failure). This needs to be fixed in the
paper's Section 15/19 text; flagged here, not fixed (out of scope this turn).

---

## 1. Overall operational performance by policy (both protocols, 60 units/policy pooled)

Descriptive only — raw fault counts are pooled across families with very
different absolute scales (A ≈1440 faults vs. B ≈2–8 faults), so this table is
of limited standalone meaning; §2 (by family) is the more valid comparison.
Included because requested.

| Policy | n units | Mean of per-unit mean faults | Mean hit ratio |
|---|---|---|---|
| Belady_MIN | 60 | — (oracle, see §12) | 0.7340 |
| FIFO | 60 | pooled, not meaningful alone | 0.5871 |
| LRU | 60 | pooled, not meaningful alone | 0.5965 |
| LFU | 60 | pooled, not meaningful alone | 0.5225 |
| Random (mean of 5 seeds) | 60 | pooled, not meaningful alone | 0.5833 |
| Learned (logistic_regression) | 13 | pooled, not meaningful alone | 0.6319 |
| Learned (hist_gradient_boosting) | 44 | pooled, not meaningful alone | 0.6161 |
| Learned (no model — structural null) | 3 | n/a | n/a |

**Rank-based comparison** (computed within each of the 60 protocol/family/horizon
units — 1=best, averaged across units; valid across heterogeneous scales
because it's a within-unit rank, not a pooled raw value):

| Policy | Mean rank (lower = better) |
|---|---|
| Belady_MIN | 1.000 |
| LRU | 3.292 |
| Learned (logistic_regression) | 3.462 |
| Learned (hist_gradient_boosting) | 3.636 |
| FIFO | 3.758 |
| Random (mean of 5 seeds) | 4.108 |
| LFU | 5.125 |

**Descriptive observation**: by mean rank across all 60 units (both protocols
pooled), LRU ranks best among non-oracle policies, narrowly ahead of both
Learned variants; LFU ranks worst. This pools Protocol A and Protocol B
together, which §4–5 show behave very differently — the pooled rank should not
be read as "the" answer without that context.

---

## 2. Performance by workload family (both protocols pooled)

Exact means (page faults / hit ratio), per policy, per family:

**A_uniform_random**: Belady_MIN 813.37/0.593, FIFO 1438.80/0.281, LRU
1438.43/0.281, LFU 1438.00/0.281, Random 1436.61/0.282, Learned(LR)
1439.77/0.280, Learned(HGB) 1439.06/0.281. All six non-oracle policies are
within ~3 faults of each other (range 1436.6–1439.8) — effectively
indistinguishable on this family.

**B_temporal_locality**: Belady_MIN 0.188/0.9999, FIFO 6.301/0.9968, LRU
1.644/0.9992, LFU 17.991/0.9910, Random 8.720/0.9956, Learned(HGB)
5.935/0.9970, Learned(LR) 1.700/0.9992. LRU and Learned(LR) are nearly tied and
both far ahead of FIFO/LFU/Random/Learned(HGB) — but see §7 for the
Protocol A/B split, which is dramatically different for this family.

**C_bursty**: Belady_MIN 14.23/0.993, FIFO 46.56/0.977, LRU 46.42/0.977, LFU
46.56/0.977, Random 46.52/0.977, Learned(LR) 46.08/0.977, Learned(HGB)
46.89/0.977. All five non-oracle policies within <1 fault of each other.

**D_long_range_reuse**: Belady_MIN 1656.0/0.172, FIFO/LRU/LFU/Random all
*exactly* 1800.0/0.100 (see §8 — this is deterministic, not a coincidence:
with a 200-key periodic reuse gap against a 16-slot capacity, every
non-oracle, non-learned policy produces an identical trace). Learned(LR)
1770.0/0.115 (Protocol A only, H=200/500 — the only D rows where LR was
trained and selected), Learned(HGB) 1798.4/0.101 (pooled across Protocol A's
two trained rows and Protocol B's five rows, where it's mostly identical to
baseline — see §8).

**E_phase_changing**: Belady_MIN 306.74/0.847, FIFO 782.47/0.609, LRU
786.14/0.607, LFU 1519.46/0.240, Random 823.05/0.589, Learned(HGB)
823.30/0.588, Learned(LR) 784.35/0.608. LFU is a dramatic outlier here (nearly
2× every other policy's fault count).

**F_non_stationary**: Belady_MIN 401.75/0.799, FIFO 881.07/0.560, LRU
769.91/0.615, LFU 907.66/0.546, Random 885.38/0.557, Learned(HGB)
775.98/0.612 (pooled across both protocols — §5b/§5c show this masks a sharp
protocol split). No logistic-regression row for F in the pooled table because
LR was never the *selected* model for F in Protocol A (HGB was selected at
every F horizon — §7 selection counts) and Protocol B uses HGB exclusively
(§7).

---

## 3. Performance by horizon (both protocols pooled)

Belady_MIN, FIFO, LRU, LFU, Random's means are **identical across all five
horizons** (532.04 / 825.87 / 807.09 / 954.95 / 833.38 faults respectively) —
expected and mechanically necessary: these five policies don't depend on the
reuse-prediction horizon $H$ at all, so pooling them by horizon is really just
re-reporting the same family-level numbers from §2, partitioned by horizon
label rather than by anything horizon-dependent in these policies.

The only horizon-varying rows are the Learned policy's:

| H | Learned(LR) mean faults | Learned(HGB) mean faults |
|---|---|---|
| 25 | 744.02 | 713.45 |
| 50 | 722.45 | 720.83 |
| 100 | 45.60 *(see caveat below)* | 785.70 |
| 200 | 852.11 | 788.08 |
| 500 | 807.29 | 846.16 |

**Caveat on the H=100 LR value (45.60)**: this is not a general H=100 effect —
it is dominated by C_bursty's H=100 row where LR was selected and C's whole
family operates in the 14–48 fault range (two orders of magnitude below most
other families). Pooling raw fault counts across families by horizon conflates
family identity with horizon; this table is included because requested, but
§4/§5 (protocol-separated, still family-mixed) and §2 (family-separated) are
more interpretable.

---

## 4. Protocol A (within-distribution) — by policy, pooled across all 6 families/5 horizons

| Policy | Mean faults | Mean hit ratio |
|---|---|---|
| Belady_MIN | 531.99 | 0.7340 |
| FIFO | 825.94 | 0.5870 |
| LRU | 807.35 | 0.5963 |
| LFU | 954.73 | 0.5226 |
| Random | 833.12 | 0.5834 |
| Learned(LR) | 736.26 | 0.6319 |
| Learned(HGB) | 613.93 | 0.6930 |

**Descriptive observation**: pooled across all Protocol A units, Learned(HGB)'s
mean fault count (613.93) is lower than every classical baseline's, including
LRU (807.35) and even approaches Belady_MIN (531.99). This pooled number is
heavily influenced by the two families with large within-distribution
effects (D at H≥200, F at all horizons — §5c); it is not evidence that HGB
beats LRU in every family (§2 shows near-ties on A, B, C).

---

## 5. Protocol B (leave-one-family-out) — by policy, pooled across all 6 held-out folds/5 horizons

| Policy | Mean faults | Mean hit ratio |
|---|---|---|
| Belady_MIN | 532.09 | 0.7340 |
| FIFO | 825.79 | 0.5871 |
| LRU | 806.83 | 0.5966 |
| LFU | 955.16 | 0.5224 |
| Random | 833.64 | 0.5832 |
| Learned(HGB) *(exclusively — see §7)* | 839.55 | 0.5802 |

**Descriptive observation**: pooled across all Protocol B units, the Learned
policy's mean fault count (839.55) is *higher* than every classical baseline
except LFU, and higher than LRU specifically by 32.7 faults on average. This is
the mirror image of §4's Protocol A pooled result.

---

## 6. Learned-model predictive metrics — summary by model/split (both protocols, trained units only, n=27 units/protocol)

| Model | Split | Mean ROC-AUC | Min | Max |
|---|---|---|---|---|
| hist_gradient_boosting | train | 0.8559 | 0.5714 | 1.0 |
| hist_gradient_boosting | val | 0.8189 | 0.4673 | 1.0 |
| hist_gradient_boosting | test | 0.6652 | 0.2462 | 1.0 |
| logistic_regression | train | 0.7763 | 0.5050 | 1.0 |
| logistic_regression | val | 0.7728 | 0.4804 | 1.0 |
| logistic_regression | test | 0.6397 | 0.3380 | 1.0 |

(F1/precision/recall/Brier/accuracy are in `predictive_metrics.csv` in full;
omitted here for space — every value is there, not summarized away.)

**Descriptive observation**: HGB's mean test ROC-AUC (0.6652) is numerically
higher than LR's (0.6397), but HGB's train→test drop (0.8559→0.6652, Δ=0.1907)
is much larger than LR's (0.7763→0.6397, Δ=0.1366) in this pooled view — this
pooled train/val/test comparison mixes both protocols; §7 separates them and
shows the gap is far more extreme in Protocol B specifically.

**Note on test ROC-AUC minimum = 0.2462**: a test ROC-AUC below 0.5 means the
model's ranking was *anti-correlated* with the true labels on that specific
unit's test split — worse than random guessing, not merely "no better than
random." This is a real, verifiable value in the data (not a typo); which
unit(s) produce this is not isolated here but is identifiable in
`predictive_metrics.csv` by filtering `roc_auc < 0.5`.

---

## 7. Comparison between the two learned models (LR vs. HGB), separated by protocol

**Train-minus-test ROC-AUC gap** (mean across the 27 non-structural-null units
each protocol has — recall Protocol A has 27 trained units + 3 structural-null;
Protocol B has all 30 trained, but only 27 are compared here against the same
family/horizon set for consistency with Protocol A's 27):

| Protocol | Model | Mean gap | Std | n |
|---|---|---|---|---|
| A | hist_gradient_boosting | 0.0743 | 0.0832 | 27 |
| A | logistic_regression | 0.0065 | 0.0100 | 27 |
| B | hist_gradient_boosting | 0.3235 | 0.1958 | 27 |
| B | logistic_regression | 0.2826 | 0.1843 | 27 |

**Descriptive observation**: the train/test gap is roughly **4× larger in
Protocol B than Protocol A for both models** (HGB: 0.0743→0.3235; LR:
0.0065→0.2826), and in Protocol B the two models' gaps are much closer to each
other (0.3235 vs 0.2826, a 0.0409 difference) than in Protocol A (0.0743 vs
0.0065, a 0.0678 difference, nearly a 10× ratio). In Protocol A, HGB's
overfitting signature is distinctly worse than LR's; in Protocol B, both
models show a large gap of similar magnitude. This is a descriptive pattern
in the gap statistic itself — no causal claim about why is made here.

**Selection counts**:

| Protocol | Model | Count |
|---|---|---|
| A | hist_gradient_boosting | 14 |
| A | logistic_regression | 13 |
| A | *(no model — structural null)* | 3 |
| B | hist_gradient_boosting | 30 |
| B | logistic_regression | 0 |

**Descriptive observation**: in Protocol B, HGB was selected in **all 30**
units — logistic regression was never once the validation-ROC-AUC winner when
trained on a pooled five-family set. In Protocol A, selection is close to even
(14 HGB / 13 LR / 3 no-model).

---

## 8. Structural-null D cases (H ∈ {25, 50, 100}) — full operational rows

All six rows below are read directly from `operational_results.csv`; note the
Protocol A/B asymmetry documented in §0.

**Protocol A** (no Learned row exists — model never trained):
```
Belady_MIN: 1656.0 faults, 0.172 hit ratio (n=30)
FIFO/LRU/LFU/Random: 1800.0 faults, 0.100 hit ratio, IDENTICAL across all four (n=30 each)
Learned: n=0, no data
```
This exact pattern (1656.0 / 1800.0, identical across FIFO/LRU/LFU/Random) repeats
unchanged at all three horizons (25, 50, 100) — expected, since none of these five
policies/metrics depend on the reuse-prediction horizon.

**Protocol B** (model trained on pooled A,B,C,E,F; selected model = hist_gradient_boosting
at all three horizons):
```
Belady_MIN: 1656.0 / 0.172 (n=200)
FIFO/LRU/LFU/Random: 1800.0 / 0.100, identical (n=200 each)
Learned(hist_gradient_boosting): 1800.0 / 0.100 (n=200) — IDENTICAL to FIFO/LRU/LFU/Random
```
at H=25, H=50, and H=100 — all three values match to the stored precision
(0.0 std, exact 1800.0 mean). The paired `Learned_vs_LRU` comparison read from
the checkpoints confirms `mean_diff=0.0`, `ci=[0.0, 0.0]` at all three horizons
(p-values are `NaN` — scipy's paired test is undefined for a zero-variance
difference vector, not a sign of statistical insignificance, a sign that every
one of the 200 paired differences was exactly zero).

**Descriptive observation**: in Protocol B, the trained Learned policy for D at
H∈{25,50,100} produces fault/hit-ratio outcomes bitwise identical to the
classical non-oracle baselines on every one of 200 held-out episodes. This is
a real, verified, degenerate result — not missing data and not a tie by
approximation.

---

## 9. Workload/horizon combinations where interpretation is unreliable

1. **`D_long_range_reuse` @ H∈{25,50,100}** — `structural_null` flag (one-class
   test labels; Protocol A also has one-class *training* labels, no model
   trained at all — §0, §8). ROC-AUC/PR-AUC undefined for Protocol A (no
   model) and undefined on the test split for Protocol B (model trained but
   test ROC-AUC is `null` in the checkpoint, since test labels are one-class).
2. **`D_long_range_reuse` @ H=200 and H=500** — flagged `redundant_with_H{other}`
   in the production script's own pre-registered rules; these are near-duplicate
   conditions, not independent evidence. Protocol A values are literally
   identical at H=200 and H=500 (Learned(LR) 1770.0 faults both horizons; §2).
3. **`E_phase_changing` @ H=200 and H=500** — same `redundant_with_H{other}` flag.
   Unlike D, these two horizons are *not* numerically identical in Protocol B
   (808.16 vs 859.17 faults) despite being flagged redundant — worth noting as
   a descriptive fact (the redundancy flag reflects the horizon's structural
   role in the family definition, not an empirical claim that results are
   identical).
4. **`B_temporal_locality`'s percentage-based Belady-gap metric** (§12) — the
   LRU-to-Belady gap on this family is only 1.44–1.66 faults (near-floor), so
   any ratio computed against it (e.g. "percent of gap closed") is numerically
   unstable: Protocol B's `pct_gap_closed_by_learned` for this family ranges
   from **−337.72% to −458.13%** — a mathematically correct computation
   ((learned−belady)/(lru−belady)) but not a meaningful percentage given the
   tiny denominator. Reported as a caveat, not a suppressed value (§12 has the
   full numbers).
5. **Imbalanced families (B, C)** — per the production config
   (`config.json`'s `imbalanced_families` field), B and C are flagged as
   class-imbalanced; accuracy/F1 should not be read as primary metrics there
   (this is a design-time flag carried in the checkpoint schema, not a new
   finding from this analysis).
6. **Protocol A's n=30 test episodes vs. Protocol B's n=200** — every Protocol
   A comparison in this report is computed over 30 test episodes (the 15%
   test split of 200), while every Protocol B comparison uses all 200 held-out
   episodes. Protocol A's wider confidence intervals (§5c — e.g.
   `A_uniform_random` H=25: CI `[−4.20, 7.60]` on n=30 vs. Protocol B's same
   cell CI `[−2.44, 2.95]` on n=200) partly reflect this sample-size
   difference, not only a weaker effect.

---

## 10. Surprising or contradictory findings

1. **The Protocol A/B asymmetry for D's structural-null units (§0, §8)** is
   the most structurally surprising finding in this dataset — it directly
   contradicts a claim already written into the paper draft, and was only
   caught by this independent re-verification pass.
2. **F_non_stationary's sign flip** (§5c): Protocol A mean_diff ranges from
   −153.3 (H=25, $p=5.6\times10^{-30}$) to −21.1 (H=500, $p=2.0\times10^{-4}$) —
   Learned beats LRU by a large, significant margin at every horizon. Protocol
   B mean_diff ranges from +79.75 (H=100) to +255.81 (H=500,
   $p=4.5\times10^{-186}$) — Learned loses to LRU by a large, significant
   margin at every horizon. The sign of the effect is opposite in every one of
   the 5 horizons, and the *largest* Protocol A advantage (H=25: −153.3) and
   *largest* Protocol B disadvantage (H=500: +255.81) occur at opposite ends
   of the horizon range — contradictory not just in sign but in where along
   the horizon axis the effect is strongest.
3. **E_phase_changing's within-distribution effect is never significant, but
   its generalization effect always is** (§5c): all 5 Protocol A p-values for
   `Learned_vs_LRU` are >0.08 (range 0.082–0.419); all 5 Protocol B p-values
   are <$10^{-59}$. A result that is statistically indistinguishable from zero
   within-distribution becomes one of the largest, most significant effects in
   the entire dataset once the family is held out.
4. **HGB is selected in 100% of Protocol B units (30/30) despite — on the
   evidence in §7 — showing an overfitting gap statistically similar in
   magnitude to LR's in that same protocol** (0.3235 vs 0.2826 mean gap,
   not the ~10× difference seen in Protocol A). The selection procedure
   (validation ROC-AUC) consistently favors HGB in Protocol B even though its
   generalization-gap *advantage* over LR has nearly vanished there, unlike in
   Protocol A where the selection is close to even (14 vs 13) despite HGB's
   gap being 10× larger.
5. **A_uniform_random's Belady gap is essentially never closed by any
   policy** (§12): the oracle-achievable improvement (625.67 faults in
   Protocol A, 624.47 in Protocol B) is matched by a `pct_gap_closed_by_learned`
   that is *negative* in nearly every A_uniform_random row (both protocols) —
   meaning the Learned policy is typically slightly *worse* than LRU specifically
   on this family, never mind closing the oracle gap.

---

## 11. Cases where the Learned policy does NOT outperform classical baselines

Comparing each unit's `Learned` mean faults against `LRU` mean faults
(lower faults = better): **36 of 57 non-null units (63.2%) have
`Learned ≥ LRU`.** Full list (protocol, family, horizon):

- **Protocol A** (14 of 27): A_uniform_random {25,50,100,200}; B_temporal_locality
  {25,50,100,200,500}; C_bursty {25,50,100,200,500}.
- **Protocol B** (22 of 30): A_uniform_random {25,50,100,200}; B_temporal_locality
  {25,50,100,200,500}; D_long_range_reuse {25,50,100} (exactly tied, not worse —
  see §8); E_phase_changing {25,50,100,200,500} *(all 5)*; F_non_stationary
  {25,50,100,200,500} *(all 5)*.

Note Protocol A's C_bursty rows are *all 5* included above (every C_bursty
horizon in Protocol A has Learned ≥ LRU, by margins of 0.27–1.77 faults) while
Protocol B's C_bursty rows are *not* in this list (Learned < LRU at all 5 C
horizons in Protocol B — the one family where Protocol B's Learned policy is
more often ahead of LRU than Protocol A's is, albeit by small margins of
0.2–0.6 faults — §5c shows these C_bursty Protocol B differences are mostly
not statistically significant except H=25, $p=0.0081$).

---

## 12. Cases where Belady MIN provides an important gap/reference

Largest absolute LRU-to-Belady gaps (`lru_minus_belady`, i.e. oracle-achievable
improvement over LRU), both protocols:

| Family | Horizon | Protocol | LRU−Belady gap (faults) |
|---|---|---|---|
| A_uniform_random | all | A | 625.67 |
| A_uniform_random | all | B | 624.47 |
| D_long_range_reuse | 25,50,100 | A & B | 144.00 |
| E_phase_changing | all | A | 480.70 |
| E_phase_changing | all | B | 478.10 |
| F_non_stationary | all | A | 368.67 |
| F_non_stationary | all | B | 367.67 |
| C_bursty | all | A | 31.63 |
| C_bursty | all | B | 32.75 |
| B_temporal_locality | all | A & B | ~1.44–1.47 |

**Descriptive observation**: A_uniform_random has both the largest absolute
Belady gap (625.67/624.47 faults) and the result (§10.5) that essentially none
of it is captured by any online policy, learned or classical — the oracle
reference here establishes a large theoretical headroom that this benchmark's
online policies do not approach at all, which is itself informative about the
family's fundamental unpredictability under a per-page feature representation.

F_non_stationary's gap (368.67/367.67) is the family where the Learned policy
captures the largest *fraction* of it in Protocol A (§12 raw table: up to
41.58% at H=25) — consistent with, but not the same statistic as, the large
Protocol A mean_diff in §10.2.

---

## Summary classification

**Strongest positive findings** (large magnitude, statistically significant per
the production script's own paired test, within-distribution):
- F_non_stationary, Protocol A, all 5 horizons: Learned beats LRU by
  21.1–153.3 faults, $p \le 2.0\times10^{-4}$ at every horizon.
- D_long_range_reuse, Protocol A, H∈{200,500}: Learned beats LRU by exactly
  30.0 faults, $p\approx0$ (both t-test and Wilcoxon).

**Positive but much smaller / non-significant findings**:
- A_uniform_random, Protocol A, H=500: mean_diff −4.17, $p=0.416$ — numerically
  favorable, not significant.
- E_phase_changing, Protocol A, all 5 horizons: mean_diff consistently
  negative (−2.9 to −8.2) but never significant ($p \ge 0.082$).

**Neutral findings** (no detectable difference either direction):
- A_uniform_random, both protocols, all horizons: mean_diff small
  (−4.17 to +3.97), never significant ($p \ge 0.10$).
- C_bursty, Protocol A, all horizons: mean_diff small (+0.27 to +1.77),
  never significant ($p \ge 0.11$).

**Negative findings** (Learned significantly worse than LRU):
- F_non_stationary, Protocol B, all 5 horizons: Learned worse by
  79.75–255.81 faults, $p \le 4.6\times10^{-138}$ at every horizon.
- E_phase_changing, Protocol B, all 5 horizons: Learned worse by
  23.41–76.68 faults, $p \le 1.2\times10^{-60}$ at every horizon.
- B_temporal_locality, Protocol B, all 5 horizons: Learned worse by
  4.88–6.62 faults, $p \le 1.0\times10^{-36}$ at every horizon.
- D_long_range_reuse, Protocol B, H∈{200,500}: Learned worse by 3.0–5.0
  faults (smaller magnitude but still $p\approx0$) — the *attenuated but
  still present* within-family generalization case.

**Limitations revealed by the data itself** (not asserted in advance, observed
in these tables):
- The structural-null/redundant-horizon annotations cover 5 of 30 (16.7%) of
  each protocol's family×horizon grid — a non-trivial fraction of the
  benchmark where standard predictive metrics don't apply.
- Protocol A (n=30 test episodes) and Protocol B (n=200) have substantially
  different statistical power per comparison; several Protocol A effects with
  the same sign/direction as a significant Protocol B effect fail to reach
  significance at n=30 (E_phase_changing is the clearest case, §10.3).
- The percentage-based Belady-gap-closed metric is unstable/uninformative on
  any family where the LRU-Belady gap itself is near zero (B_temporal_locality,
  §9.4), which this dataset's own numbers expose rather than a separate
  methodological critique.
- HGB's selection rate (30/30 in Protocol B) does not track its relative
  overfitting advantage over LR, which nearly disappears in Protocol B
  (§7) — the model-selection criterion and the overfitting-gap metric appear
  decoupled in Protocol B specifically, a pattern visible only by comparing
  §7's two sub-tables side by side.
