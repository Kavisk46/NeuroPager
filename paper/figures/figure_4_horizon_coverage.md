Table 4 -- Horizon coverage: fraction of accesses with a positive reuse label at each horizon. Source: experiments\preflight-analysis-1\statistics.json. Values are the across-episode mean of label_positive_frac_H{h}, which counts an access as positive iff its key recurs within (t, t+H] before episode end (episode-end truncation counts as negative), exactly matching neuropager.dataset.labels. [structural-null] and [redundant] annotations are carried over verbatim from scripts/generalization_experiment.py pre-registered interpretation rules, not recomputed here.

| Family | H=25 | H=50 | H=100 | H=200 | H=500 |
| --- | --- | --- | --- | --- | --- |
| A_uniform_random | 0.323 | 0.539 | 0.778 | 0.930 | 0.968 |
| B_temporal_locality | 0.899 | 0.944 | 0.969 | 0.982 | 0.989 |
| C_bursty | 0.934 | 0.936 | 0.939 | 0.944 | 0.957 |
| D_long_range_reuse | 0.000 [structural-null] | 0.000 [structural-null] | 0.000 [structural-null] | 0.900 [redundant w/ H=500] | 0.900 [redundant w/ H=200] |
| E_phase_changing | 0.519 | 0.735 | 0.852 | 0.872 [redundant w/ H=500] | 0.872 [redundant w/ H=200] |
| F_non_stationary | 0.577 | 0.701 | 0.810 | 0.897 | 0.957 |
