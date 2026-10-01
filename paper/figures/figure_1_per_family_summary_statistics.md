Table 1 -- Per-family summary statistics. Source: experiments\preflight-analysis-1\statistics.json (produced by scripts/preflight_analysis.py). Each value is the mean +/- population std across the 10 generated episodes for that family (length=2000 ticks each, seeds 0-9). No smoothing or extrapolation applied.

| Family | Entropy (bits) | Mean stack distance | Burst CV | Unique-page ratio | Early/late TV drift | n episodes |
| --- | --- | --- | --- | --- | --- | --- |
| A_uniform_random | 5.978 +/- 0.004 | 30.96 +/- 0.46 | 0.123 +/- 0.007 | 0.032 +/- 0.000 | 0.223 +/- 0.024 | 10 |
| B_temporal_locality | 4.002 +/- 0.072 | 2.24 +/- 0.04 | 0.556 +/- 0.013 | 0.009 +/- 0.001 | 0.463 +/- 0.060 | 10 |
| C_bursty | 5.418 +/- 0.084 | 0.91 +/- 0.11 | 0.941 +/- 0.060 | 0.028 +/- 0.001 | 0.806 +/- 0.072 | 10 |
| D_long_range_reuse | 7.644 +/- 0.000 | 199.00 +/- 0.00 | 0.000 +/- 0.000 | 0.100 +/- 0.000 | 0.000 +/- 0.000 | 10 |
| E_phase_changing | 7.906 +/- 0.008 | 14.35 +/- 0.22 | 0.181 +/- 0.009 | 0.128 +/- 0.000 | 1.000 +/- 0.000 | 10 |
| F_non_stationary | 5.580 +/- 0.025 | 16.59 +/- 0.41 | 0.306 +/- 0.017 | 0.032 +/- 0.000 | 0.674 +/- 0.018 | 10 |
